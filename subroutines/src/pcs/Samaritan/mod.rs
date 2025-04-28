pub(crate) mod srs;

pub(crate) mod util;

use crate::pcs::structs::Commitment;
use crate::pcs::Samaritan::util::*;
use crate::pcs::{PCSError, PolynomialCommitmentScheme, StructuredReferenceString};
use crate::{BatchProof, PolyIOP, SumCheck};
use arithmetic::{build_eq_x_r_vec, VPAuxInfo, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ec::VariableBaseMSM;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseMultilinearExtension, DenseUVPolynomial,
    MultilinearExtension, Polynomial,
};
use ark_std::time::Instant;
use ark_poly_commit::kzg10::Proof;
use ark_poly_commit::kzg10::VerifierKey;
use ark_poly_commit::kzg10::KZG10;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::collections::BTreeMap;
use ark_std::{
    borrow::Borrow,
    format, log2,
    marker::PhantomData,
    rand::{Rng, SeedableRng},
    string::ToString,
    sync::Arc,
    vec,
    vec::Vec,
};
use rand_chacha::ChaCha8Rng;
use srs::{SamaritanProverParam, SamaritanUniversalParams, SamaritanVerifierParam};
use std::iter;
use std::ops::Deref;
use transcript::IOPTranscript;

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct SamaritanProof<E: Pairing> {
    commitments: Vec<Commitment<E>>,
    evaluations: Vec<E::ScalarField>,
    batch_proof: Option<Vec<u8>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
pub struct SamaritanPCS<E: Pairing> {
    phantom: PhantomData<E>,
}

impl<E: Pairing> PolynomialCommitmentScheme<E> for SamaritanPCS<E>
where
    E::ScalarField: PrimeField,
    E::G1Affine: CanonicalSerialize + CanonicalDeserialize,
    E::G2Affine: CanonicalSerialize + CanonicalDeserialize,
{
    type ProverParam = SamaritanProverParam<E::G1Affine>;
    type VerifierParam = SamaritanVerifierParam<E>;
    type SRS = SamaritanUniversalParams<E>;
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    type Commitment = Commitment<E>;
    type Proof = SamaritanProof<E>;
    type BatchProof = BatchProof<E, Self>;
    

    fn gen_srs_for_testing<R: Rng>(rng: &mut R, log_size: usize) -> Result<Self::SRS, PCSError> {
        let start = Instant::now();
        let srs = Self::SRS::gen_srs_for_testing(rng, log_size)?;
        let duration = start.elapsed();
        println!("-----------------Setup Samaritan Duration{:?}",duration);
        Ok(srs)
    }

    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_degree: Option<usize>,
        supported_num_vars: Option<usize>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        assert!(supported_degree.is_none());

        let supported_num_vars = match supported_num_vars {
            Some(p) => p,
            None => {
                return Err(PCSError::InvalidParameters(
                    "multilinear should receive a num_var param".to_string(),
                ))
            },
        };

        // multilinear 对应的 univariate 的 max_degree
        let uni_degree: usize = 1 << supported_num_vars;
        srs.borrow().trim(uni_degree)
    }

    fn commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let prover_param = prover_param.borrow();
        let f_hat_coeffs = poly.evaluations.to_vec();
        let n = f_hat_coeffs.len();
        let miu = log2(n);
        let f_hat_coeffs = compute_f_hat(&f_hat_coeffs, miu as usize);
        let f_hat = DensePolynomial::from_coefficients_vec(trim_trailing_zeros(f_hat_coeffs));
        let comm = commit(prover_param, &f_hat)?;
        Ok(comm)
    }
    fn open(
        pp: impl Borrow<Self::ProverParam>,
        f_poly: &Self::Polynomial,
        z: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let f_hat_coeffs = f_poly.evaluations.to_vec();
        let n = z.len();
        let f_hat_coeffs = compute_f_hat(&f_hat_coeffs, n);
        let miu = z.len() as u32;
        let (proof, evaluation) =
            open_internal::<E>(pp.borrow(), &f_hat_coeffs, f_poly, miu, z.to_vec())?;
        Ok((proof, evaluation))
    }

    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        z: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        verify_internal::<E>(commitment, verifier_param, z.to_vec(), proof)
    }

    /// Input a list of multilinear extensions, and a same number of points, and
    /// a transcript, compute a multi-opening for all the polynomials.
    fn multi_open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
        evals: &[Self::Evaluation],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<BatchProof<E, Self>, PCSError> {
        for eval_point in points.iter() {
            transcript.append_serializable_element(b"eval_point", eval_point)?;
        }
        for eval in evals.iter() {
            transcript.append_field_element(b"eval", eval)?;
        }

        // TODO: sanity checks
        let num_var = polynomials[0].num_vars;
        let k = polynomials.len();
        let ell = log2(k) as usize;

        // challenge point t
        let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

        // eq(t, i) for i in [0..k]
        let eq_t_i_list = build_eq_x_r_vec(t.as_ref())?;

        // \tilde g_i(b) = eq(t, i) * f_i(b)
        // combine the polynomials that have same opening point first to reduce the
        // cost of sum check later.
        let point_indices = points
            .iter()
            .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
                let idx = indices.len();
                indices.entry(point).or_insert(idx);
                indices
            });
        let deduped_points =
            BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
                .into_values()
                .collect::<Vec<_>>();
        let merged_tilde_gs = polynomials
            .iter()
            .zip(points.iter())
            .zip(eq_t_i_list.iter())
            .fold(
                iter::repeat_with(DenseMultilinearExtension::zero)
                    .map(Arc::new)
                    .take(point_indices.len())
                    .collect::<Vec<_>>(),
                |mut merged_tilde_gs, ((poly, point), coeff)| {
                    *Arc::make_mut(&mut merged_tilde_gs[point_indices[point]]) +=
                        (*coeff, poly.deref());
                    merged_tilde_gs
                },
            );

        let tilde_eqs: Vec<_> = deduped_points
            .iter()
            .map(|point| {
                let eq_b_zi = build_eq_x_r_vec(point).unwrap();
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_var, eq_b_zi,
                ))
            })
            .collect();

        // built the virtual polynomial for SumCheck
        let mut sum_check_vp = VirtualPolynomial::new(num_var);
        for (merged_tilde_g, tilde_eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
            sum_check_vp.add_mle_list([merged_tilde_g.clone(), tilde_eq], E::ScalarField::one())?;
        }

        let proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
            &sum_check_vp,
            transcript,
        ) {
            Ok(p) => p,
            Err(_e) => {
                // cannot wrap IOPError with PCSError due to cyclic dependency
                return Err(PCSError::InvalidProver(
                    "Sumcheck in batch proving Failed".to_string(),
                ));
            },
        };

        // a2 := sumcheck's point
        let a2 = &proof.point[..num_var];

        // build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is the
        // sumcheck's point \tilde eq_i(a2) = eq(a2, point_i)
        let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
        for (merged_tilde_g, point) in merged_tilde_gs.iter().zip(deduped_points.iter()) {
            let eq_i_a2 = eq_eval(a2, point)?;
            *Arc::make_mut(&mut g_prime) += (eq_i_a2, merged_tilde_g.deref());
        }

        let (g_prime_proof, _g_prime_eval) =
            Self::open(prover_param, &g_prime, a2.to_vec().as_ref())?;
        // assert_eq!(g_prime_eval, tilde_g_eval);

        Ok(BatchProof {
            sum_check_proof: proof,
            f_i_eval_at_point_i: evals.to_vec(),
            g_prime_proof,
        })
    }

    /// Verifies that `value` is the evaluation at `x` of the polynomial
    /// committed inside `comm`.
    ///
    /// This function takes
    /// - num_var number of pairing product.
    /// - num_var number of MSM

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        for eval_point in points.iter() {
            transcript.append_serializable_element(b"eval_point", eval_point)?;
        }
        for eval in batch_proof.f_i_eval_at_point_i.iter() {
            transcript.append_field_element(b"eval", eval)?;
        }

        // TODO: sanity checks

        let k = commitments.len();
        let ell = log2(k) as usize;
        let num_var = batch_proof.sum_check_proof.point.len();

        // challenge point t
        let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

        // sum check point (a2)
        let a2 = &batch_proof.sum_check_proof.point[..num_var];

        // build g' commitment

        let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

        let mut scalars = vec![];
        let mut bases = vec![];

        for (i, point) in points.iter().enumerate() {
            let eq_i_a2 = eq_eval(a2, point)?;
            scalars.push(eq_i_a2 * eq_t_list[i]);
            bases.push(commitments[i].0);
        }
        let g_prime_commit = E::G1::msm_unchecked(&bases, &scalars);

        // ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
        let mut sum = E::ScalarField::zero();
        for (i, &e) in eq_t_list.iter().enumerate().take(k) {
            sum += e * batch_proof.f_i_eval_at_point_i[i];
        }
        let aux_info = VPAuxInfo {
            max_degree: 2,
            num_variables: num_var,
            phantom: PhantomData,
        };
        let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
            sum,
            &batch_proof.sum_check_proof,
            &aux_info,
            transcript,
        ) {
            Ok(p) => p,
            Err(_e) => {
                // cannot wrap IOPError with PCSError due to cyclic dependency
                return Err(PCSError::InvalidProver(
                    "Sumcheck in batch verification failed".to_string(),
                ));
            },
        };
        let tilde_g_eval = subclaim.expected_evaluation;

        // verify commitment
        let res = Self::verify(
            verifier_param,
            &Commitment(g_prime_commit.into_affine()),
            a2.to_vec().as_ref(),
            &tilde_g_eval,
            &batch_proof.g_prime_proof,
        )?;

        Ok(res)
    }
}

fn open_internal<E: Pairing>(
    pp: &SamaritanProverParam<E::G1Affine>,
    f_hat_coeffs: &[E::ScalarField],
    f_poly: &DenseMultilinearExtension<E::ScalarField>,
    miu: u32,
    z: Vec<E::ScalarField>,
) -> Result<(SamaritanProof<E>, E::ScalarField), PCSError>
where
    E::ScalarField: PrimeField,
    E::G1Affine: CanonicalSerialize + CanonicalDeserialize,
    E::G2Affine: CanonicalSerialize + CanonicalDeserialize,
{
    let v = f_poly
        .evaluate(&z)
        .ok_or_else(|| PCSError::InvalidParameters("Failed to evaluate f_poly at z".to_string()))?;
    let n = 1 << miu;
    let l = 1 << miu.ilog2();
    let m = n / l;
    let k = log2(m) as usize;

    let mut commitments = Vec::new();
    let mut evaluations = Vec::new();
    let mut proofs: Vec<ark_poly_commit::kzg10::Proof<E>> = Vec::new();

    let g_hat = generate_ghat(&f_hat_coeffs, miu);

    let q_poly = compute_q(&g_hat, l);
    let z_x = &z[z.len() - k..];
    let z_y = &z[..miu as usize - k];

    let mut v_vec: Vec<E::ScalarField> = Vec::with_capacity(l);
    for i in 1..=l {
        let g_tilde = compute_gtilde(f_poly, miu, i);
        let v_i = g_tilde.evaluate(&z_x).unwrap();
        v_vec.push(v_i);
    }

    let v_hat = DensePolynomial::from_coefficients_vec(v_vec.clone());
    let psi_hat = compute_psihat(&z_y);

    let v_psi = polynomial_mul(&v_hat, &psi_hat);

    let mut a_coeffs = vec![E::ScalarField::zero(); v_psi.degree() - l + 1];
    let mut b_coeffs = vec![E::ScalarField::zero(); l - 1];

    for i in l..=v_psi.degree() {
        a_coeffs[i - l] = v_psi.coeffs[i];
    }

    for i in 0..l - 1 {
        b_coeffs[i] = v_psi.coeffs[i];
    }

    let a_hat = DensePolynomial::from_coefficients_vec(a_coeffs);
    let b_hat = DensePolynomial::from_coefficients_vec(b_coeffs);

    let cm_v = commit(pp, &v_hat)?;
    let cm_a = commit(pp, &a_hat)?;
    commitments.push(cm_v);
    commitments.push(cm_a);

    let mut transcript = IOPTranscript::<E::ScalarField>::new(b"MyKZGPCS");

    let mut buf_v = Vec::new();
    cm_v.serialize_compressed(&mut buf_v)?;
    transcript.append_message(b"commitment_v", &buf_v)?;

    let mut buf_a = Vec::new();
    cm_a.serialize_compressed(&mut buf_a)?;
    transcript.append_message(b"commitment_a", &buf_a)?;

    let gamma: E::ScalarField = transcript.get_and_append_challenge(b"challenge_gamma")?;

    let (v_gamma, proof_vgamma) = kzg_prove(pp, &v_hat, gamma)?;
    proofs.push(proof_vgamma);

    let p_hat = compute_phat(q_poly.clone(), gamma);

    let r_poly = compute_r(q_poly, gamma);
    let r_hat = compute_r_hat(r_poly, m);

    let psi_hat_y = compute_psihat(z_x);
    let p_psi = &p_hat * &psi_hat_y;

    let mut h_coeffs = vec![E::ScalarField::zero(); p_psi.degree() - m + 1];
    let mut u_coeffs = vec![E::ScalarField::zero(); m - 1];

    for i in m..=p_psi.degree() {
        h_coeffs[i - m] = p_psi.coeffs[i];
    }

    for i in 0..m - 1 {
        u_coeffs[i] = p_psi.coeffs[i];
    }

    let h_hat = DensePolynomial::from_coefficients_vec(h_coeffs);
    let u_hat = DensePolynomial::from_coefficients_vec(u_coeffs);

    let cm_p = commit(pp, &p_hat)?;
    let cm_r = commit(pp, &r_hat)?;
    let cm_h = commit(pp, &h_hat)?;
    commitments.push(cm_p);
    commitments.push(cm_r);
    commitments.push(cm_h);

    let mut buf_p = Vec::new();
    cm_p.serialize_compressed(&mut buf_p)?;
    transcript.append_message(b"commitment_p", &buf_p)?;
    let mut buf_r = Vec::new();
    cm_r.serialize_compressed(&mut buf_r)?;
    transcript.append_message(b"commitment_r", &buf_r)?;
    let mut buf_h = Vec::new();
    cm_h.serialize_compressed(&mut buf_h)?;
    transcript.append_message(b"commitment_h", &buf_h)?;

    let beta: E::ScalarField = transcript.get_and_append_challenge(b"challenge_beta")?;

    let t_poly = compute_t(&p_hat, &u_hat, &b_hat, beta, m as u64, l as u64);
    let cm_t = commit(pp, &t_poly)?;
    commitments.push(cm_t);
    let mut buf_t = Vec::new();
    cm_t.serialize_compressed(&mut buf_t)?;
    transcript.append_message(b"commitment_t", &buf_t)?;

    let delta: E::ScalarField = transcript.get_and_append_challenge(b"challenge_delta")?;
    let delta_inv = delta.inverse().unwrap();

    let f_hat = DensePolynomial::from_coefficients_vec(f_hat_coeffs.to_vec());
    let (f_delta, proof_fdelta) = kzg_prove(&pp, &f_hat, delta)?;
    proofs.push(proof_fdelta);

    let (p_delta, proof_pdelta) = kzg_prove(&pp, &p_hat, delta)?;
    proofs.push(proof_pdelta);

    let (h_delta, proof_hdelta) = kzg_prove(&pp, &h_hat, delta)?;
    proofs.push(proof_hdelta);

    let (v_delta, proof_vdelta) = kzg_prove(&pp, &v_hat, delta)?;
    proofs.push(proof_vdelta);

    let (a_delta, proof_adelta) = kzg_prove(&pp, &a_hat, delta)?;
    proofs.push(proof_adelta);

    let (t_delta_inv, proof_tdelta_inv) = kzg_prove(pp, &t_poly, delta_inv)?;
    proofs.push(proof_tdelta_inv);

    let eval_slice: Vec<E::ScalarField> = vec![
        delta,
        t_delta_inv,
        f_delta,
        p_delta,
        h_delta,
        v_delta,
        a_delta,
        v,
        v_gamma,
        beta,
        E::ScalarField::from(m as u64),
        E::ScalarField::from(l as u64),
    ];
    evaluations.extend_from_slice(&eval_slice);
    let evaluation = evaluations[7];

    let mut proofs_byte = Vec::new();
    proofs.serialize_compressed(&mut proofs_byte)?;

    Ok((
        SamaritanProof {
            commitments,
            evaluations,
            batch_proof: Some(proofs_byte),
        },
        evaluation,
    ))
}

fn verify_internal<E: Pairing>(
    commitment: &Commitment<E>,
    verifier_param: &SamaritanVerifierParam<E>,
    z: Vec<E::ScalarField>,
    proof: &SamaritanProof<E>,
) -> Result<bool, PCSError>
where
    E::ScalarField: PrimeField,
    E::G1Affine: CanonicalSerialize + CanonicalDeserialize,
    E::G2Affine: CanonicalSerialize + CanonicalDeserialize,
{
    let cm_v = proof.commitments[0];
    let cm_a = proof.commitments[1];
    let cm_p = proof.commitments[2];
    let cm_r = proof.commitments[3];
    let cm_h = proof.commitments[4];
    let cm_t = proof.commitments[5];

    //恢复gamma
    let mut transcript = IOPTranscript::<E::ScalarField>::new(b"MyKZGPCS");
    let mut buf_v = Vec::new();
    cm_v.serialize_compressed(&mut buf_v)?; // 序列化为字节
    transcript.append_message(b"commitment_v", &buf_v)?;

    let mut buf_a = Vec::new();
    cm_a.serialize_compressed(&mut buf_a)?; // 序列化为字节
    transcript.append_message(b"commitment_a", &buf_a)?;

    let gamma: E::ScalarField = transcript.get_and_append_challenge(b"challenge_gamma")?;

    let mut buf_p = Vec::new();
    cm_p.serialize_compressed(&mut buf_p)?; // 序列化为字节
    transcript.append_message(b"commitment_p", &buf_p)?;
    let mut buf_r = Vec::new();
    cm_r.serialize_compressed(&mut buf_r)?; // 序列化为字节
    transcript.append_message(b"commitment_r", &buf_r)?;
    let mut buf_h = Vec::new();
    cm_h.serialize_compressed(&mut buf_h)?; // 序列化为字节
    transcript.append_message(b"commitment_h", &buf_h)?;
    let beta: E::ScalarField = transcript.get_and_append_challenge(b"challenge_beta")?;

    let mut buf_t = Vec::new();
    cm_t.serialize_compressed(&mut buf_t)?; // 序列化为字节
    transcript.append_message(b"commitment_t", &buf_t)?;

    let delta: E::ScalarField = transcript.get_and_append_challenge(b"challenge_delta")?;

    let t_delta_inv = proof.evaluations[1];
    let f_delta = proof.evaluations[2];
    let p_delta = proof.evaluations[3];
    let h_delta = proof.evaluations[4];
    let v_delta = proof.evaluations[5];
    let a_delta = proof.evaluations[6];
    let v = proof.evaluations[7];
    let v_gamma = proof.evaluations[8];

    let m_f = proof.evaluations[10];
    let l_f = proof.evaluations[11];

    let m: usize = m_f.into_bigint().as_ref()[0] as usize;
    let l: usize = l_f.into_bigint().as_ref()[0] as usize;

    let delta_inv = delta
        .inverse()
        .ok_or_else(|| PCSError::InvalidParameters("Delta inverse failed".to_string()))?;
    let delta_m1 = delta_inv.pow(&[(m - 1) as u64]);
    let delta_m2 = delta_inv.pow(&[(m - 2) as u64]);
    let delta_l2 = delta_inv.pow(&[(l - 2) as u64]);

    let batch_commitments: Vec<ark_poly_commit::kzg10::Commitment<E>> = vec![
        ark_poly_commit::kzg10::Commitment(cm_v.0),
        ark_poly_commit::kzg10::Commitment(commitment.0),
        ark_poly_commit::kzg10::Commitment(cm_p.0),
        ark_poly_commit::kzg10::Commitment(cm_h.0),
        ark_poly_commit::kzg10::Commitment(cm_v.0),
        ark_poly_commit::kzg10::Commitment(cm_a.0),
        ark_poly_commit::kzg10::Commitment(cm_t.0),
    ];

    let batch_points = vec![gamma, delta, delta, delta, delta, delta, delta_inv];

    let batch_values = vec![
        v_gamma,
        f_delta,
        p_delta,
        h_delta,
        v_delta,
        a_delta,
        t_delta_inv,
    ];

    let batch_proof_bytes = proof
        .batch_proof
        .as_ref()
        .ok_or_else(|| PCSError::InvalidProof("Missing batch proof".to_string()))?;
    let kzg_proofs: Vec<Proof<E>> =
        CanonicalDeserialize::deserialize_compressed(&mut &batch_proof_bytes[..]).map_err(|e| {
            PCSError::InvalidProof(format!("Failed to deserialize batch proof: {:?}", e))
        })?;

    let vk = VerifierKey {
        g: verifier_param.g.clone(),
        gamma_g: verifier_param.g.clone(),
        h: verifier_param.h.clone(),
        beta_h: verifier_param.beta_h.clone(),
        prepared_h: E::G2Prepared::from(verifier_param.h.clone()),
        prepared_beta_h: E::G2Prepared::from(verifier_param.beta_h.clone()),
    };

    let mut rng = ChaCha8Rng::from_seed([0u8; 32]);

    let batch_result = KZG10::<E, DensePolynomial<E::ScalarField>>::batch_check(
        &vk,
        &batch_commitments,
        &batch_points,
        &batch_values,
        &kzg_proofs,
        &mut rng,
    )
    .map_err(|e| PCSError::InvalidProof(format!("KZG batch check failed: {:?}", e)))?;

    let result = batch_result;


    let mu = z.len();
    let k = log2(m) as usize;
    let z_y = &z[..mu - k];
    let z_x = &z[mu - k..];
    let psi_hat_y = compute_psihat(z_x);
    let psi_delta_y = psi_hat_y.evaluate(&delta);
    let delta_m = delta.pow(&[m as u64]);
    let delta_m_1 = delta.pow(&[(m - 1) as u64]);
    let u_delta = p_delta * psi_delta_y - delta_m * h_delta - v_gamma * delta_m_1;

    let psi_hat_x = compute_psihat(z_y);
    let psi_delta_x = psi_hat_x.evaluate(&delta);
    let delta_l = delta.pow(&[l as u64]);
    let delta_l_1 = delta.pow(&[(l - 1) as u64]);
    let b_delta = v_delta * psi_delta_x - delta_l * a_delta - v * delta_l_1;

    // println!("b_verify{:?}", b_delta);
    let rhs = delta_m1 * p_delta + beta * delta_m2 * u_delta + beta * beta * delta_l2 * b_delta;
    let mut res = false;

    if t_delta_inv == rhs {
        res = true;
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{sync::Arc, test_rng, vec::Vec, UniformRand};
    use env_logger;

    #[test]
    fn test_method() -> Result<(), PCSError> {
        env_logger::init();
        let mut rng = test_rng();

        // 测试不同变量数量（2 和 4）
        for num_vars in [2, 8].iter() {
            let miu = *num_vars as usize;
            println!("Testing SamaritanPCS with {} variables", miu);

            // 生成 SRS
            let pp = SamaritanPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, miu)
                .expect("生成 SRS 失败");

            // 构造多线性多项式
            let evaluations: Vec<Fr> = (1..=1 << miu).map(|i| Fr::from(i as u64)).collect();
            let poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                miu,
                evaluations.clone(),
            ));

            let nv = poly.num_vars();
            assert_ne!(nv, 0);
            let (ck, vk) = SamaritanPCS::trim(pp, None, Some(nv))?;

            // 生成承诺
            let commitment = SamaritanPCS::<Bls12_381>::commit(&ck, &poly).expect("承诺失败");

            // 随机生成评估点
            let z: Vec<Fr> = (0..miu).map(|_| Fr::rand(&mut rng)).collect();
            println!("z{:?}", z);
            let expected_eval = poly.evaluate(&z).expect("多项式评估失败");
            println!("Expected evaluation at z = {:?}", expected_eval);

            // 打开多项式
            let (proof, evaluation) =
                SamaritanPCS::<Bls12_381>::open(&ck, &poly, &z).expect("打开承诺失败");

            // 验证评估值
            assert_eq!(evaluation, expected_eval, "评估值不匹配");

            // 验证证明
            let is_valid =
                SamaritanPCS::<Bls12_381>::verify(&vk, &commitment, &z, &evaluation, &proof)
                    .expect("验证失败");

            assert!(is_valid, "证明验证未通过 for num_vars = {}", miu);
        }
        Ok(())
    }
}