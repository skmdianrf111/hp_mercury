#![feature(isqrt)]
pub(crate) mod srs;

pub(crate) mod util;

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::iter;
use std::marker::PhantomData;
use std::ops::{Deref, Mul, Neg};
use std::sync::Arc;
use crate::pcs::{
    prelude::Commitment, PCSError, PolynomialCommitmentScheme, StructuredReferenceString
};
use ark_ec::{
    pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, AffineRepr, CurveGroup,
};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{DenseMultilinearExtension, univariate::DensePolynomial, DenseUVPolynomial, Polynomial, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, log2, start_timer, test_rng,  rand::Rng};
use transcript::IOPTranscript;
use crate::{BatchProof, PolyIOP, SumCheck};
use util::{split_u, partial_sum, compute_folded_g_and_q};

// use ark_crypto_primitives::sponge::poseidon::{PoseidonSponge, PoseidonConfig, find_poseidon_ark_and_mds};
// use ark_crypto_primitives::sponge::{CryptographicSponge, FieldBasedCryptographicSponge};
use srs::{MercuryProverParam, MercuryUniversalParams, MercuryVerifierParam};
use crate::pcs::mercury::util::{compute_batch_f, compute_batch_l, compute_batch_w, compute_batch_w_hat, compute_big_h, compute_d, compute_s, compute_zs, generate_r_i, pu_evaluate};
use arithmetic::{build_eq_x_r_vec, evaluate_opt, VPAuxInfo, VirtualPolynomial};
/// KZG Polynomial Commitment Scheme on multilinear polynomials.
pub struct MercuryPCS<E: Pairing> {
    #[doc(hidden)]
    phantom: PhantomData<E>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug, PartialEq, Eq)]
/// proof of opening
pub struct MercuryProof<E: Pairing> {
    /// Evaluation of quotients
    pub proofs: Vec<E::G1Affine>,
    pub g_value: Vec<E::ScalarField>,
    pub h_value: Vec<E::ScalarField>,
    pub s_value: Vec<E::ScalarField>,
    pub d_value: Vec<E::ScalarField>,
    pub pi_z: E::G1Affine,
    pub b: usize,
}


impl<E: Pairing> PolynomialCommitmentScheme<E> for MercuryPCS<E> {
    // Parameters
    type ProverParam = MercuryProverParam<E::G1Affine>;
    type VerifierParam = MercuryVerifierParam<E>;
    type SRS = MercuryUniversalParams<E>;
    // Polynomial and its associated types
    type Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>;
    // type ProverCommitmentAdvice = ();
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // Commitments and proofs
    type Commitment = Commitment<E>;
    type Proof = MercuryProof<E>;
    type BatchProof = BatchProof<E, Self>;


    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `log_size` is the log of maximum degree.
    /// - For multilinear polynomials, `log_size` is the number of variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, log_size: usize) -> Result<Self::SRS, PCSError> {
        Self::SRS::gen_srs_for_testing(rng, log_size)
    }

    /// Trim the universal parameters to specialize the public parameters.
    /// Input both `supported_log_degree` for univariate and
    /// `supported_num_vars` for multilinear.
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
        let uni_degree:usize = 1 << supported_num_vars;
        srs.borrow().trim(uni_degree)
    }

    /// Generate a commitment for a polynomial.
    ///
    /// This function takes `2^num_vars` number of scalar multiplications over
    /// G1.
    /// Note that the scheme is not hidding.
    fn commit(
        prover_param: impl Borrow<Self::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let prover_param = prover_param.borrow();
        let commit_timer = start_timer!(|| "commit");

        let f_poly = DensePolynomial::from_coefficients_vec(poly.evaluations.to_vec());

        if f_poly.degree() >= prover_param.powers_of_g.len() {
            return Err(PCSError::InvalidParameters(format!(
                "poly degree {} is larger than allowed {}",
                f_poly.degree(),
                prover_param.powers_of_g.len()
            )));
        }

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&f_poly);

        let msm_time = start_timer!(|| "MSM to compute commitment to plaintext poly");
        let commitment =
            E::G1::msm_unchecked(&prover_param.powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);

        end_timer!(commit_timer);
        Ok(Commitment(commitment))
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the
    /// same. This function does not need to take the evaluation value as an
    /// input.
    ///
    /// This function takes 2^{num_var +1} number of scalar multiplications over
    /// G1:
    /// - it prodceeds with `num_var` number of rounds,
    /// - at round i, we compute an MSM for `2^{num_var - i + 1}` number of G2
    ///   elements.
    fn open(
        prover_param: impl Borrow<Self::ProverParam>,
        polynomial: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let open_timer = start_timer!(|| format!("open mle with {} variable", polynomial.num_vars));
        let f_poly = DensePolynomial::from_coefficients_vec(polynomial.evaluations.to_vec());

        if polynomial.num_vars() != point.len() {
            return Err(PCSError::InvalidParameters(format!(
                "Polynomial num_vars {} does not match point len {}",
                polynomial.num_vars,
                point.len()
            )));
        }

        if f_poly.degree() >= prover_param.borrow().powers_of_g.len() {
            return Err(PCSError::InvalidParameters(format!(
                "poly degree {} is larger than allowed {}",
                f_poly.degree(),
                prover_param.borrow().powers_of_g.len()
            )));
        }

        let n = f_poly.degree() + 1;
        let b = n.isqrt();
        let (u2, u1) = split_u(&point);


        // 1. h(X)
        let h_poly = partial_sum(&f_poly, &u1);
        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&h_poly);
        let msm_time = start_timer!(|| "MSM to compute commitment to h(X)");
        let com_h =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs).into();
        end_timer!(msm_time);

        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"HyperPlonkProtocol");
        let mut buf = Vec::new();
        com_h.serialize_compressed(&mut buf)?; // 序列化为字节
        transcript.append_message(b"commitment_h", &buf)?;


        // 2. g(X), q(X)
        let alpha: E::ScalarField = transcript.get_and_append_challenge(b"challenge_alpha")?;
        let (g_poly, q_poly) = compute_folded_g_and_q(&f_poly, alpha);


        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&g_poly);
        let msm_time = start_timer!(|| "MSM to compute commitment to g(X)");
        let com_g =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&q_poly);
        let msm_time = start_timer!(|| "MSM to compute commitment to q(X)");
        let com_q =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);

        let mut buf_g = Vec::new();
        com_g.serialize_compressed(&mut buf_g).unwrap(); // 序列化为字节
        let mut buf_q = Vec::new();
        com_q.serialize_compressed(&mut buf_q).unwrap(); // 序列化为字节

        transcript.append_message(b"commitment_g", &buf_g)?;
        transcript.append_message(b"commitment_q", &buf_q)?;

        // 3. S(X), D(X)
        let gamma: E::ScalarField = transcript.get_and_append_challenge(b"challenge_gamma")?;
        let d_poly = compute_d(&g_poly, b);
        let s_poly = compute_s(&g_poly, &h_poly, &u1, &u2, gamma, b);

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&s_poly);
        let msm_time = start_timer!(|| "MSM to compute commitment to S(X)");
        let com_s =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&d_poly);
        let msm_time = start_timer!(|| "MSM to compute commitment to D(X)");
        let com_d =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);

        let mut buf_s = Vec::new();
        com_s.serialize_compressed(&mut buf_s).unwrap(); // 序列化为字节
        let mut buf_d = Vec::new();
        com_d.serialize_compressed(&mut buf_d).unwrap(); // 序列化为字节

        transcript.append_message(b"commitment_s", &buf_s)?;
        transcript.append_message(b"commitment_d", &buf_d)?;

        // 4. KZG evaluation ---- H(X) and poly evaluation of z
        let z: E::ScalarField = transcript.get_and_append_challenge(b"challenge_z")?;
        let z_inverse = z.inverse().expect("z must have an inverse");
        let g_z = g_poly.evaluate(&z);
        let g_hat_z = g_poly.evaluate(&z_inverse);
        let h_z = h_poly.evaluate(&z);
        let h_hat_z = h_poly.evaluate(&z_inverse);
        let s_z = s_poly.evaluate(&z);
        let s_hat_z = s_poly.evaluate(&z_inverse);

        let h_alpha = h_poly.evaluate(&alpha);
        let d_z = d_poly.evaluate(&z);

        let big_h_poly = compute_big_h(&f_poly, &q_poly, z, b, alpha, g_z);

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&big_h_poly);
        let msm_time = start_timer!(|| "MSM to compute commitment to H(X)");
        let com_big_h =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);

        let mut buf_big_h = Vec::new();
        com_big_h.serialize_compressed(&mut buf_big_h).unwrap(); // 序列化为字节

        transcript.append_message(b"commitment_big_h", &buf_big_h)?;
        transcript.append_field_element(b"g_z", &g_z)?;
        transcript.append_field_element(b"g_hat_z", &g_hat_z)?;
        transcript.append_field_element(b"h_z", &h_z)?;
        transcript.append_field_element(b"h_hat_z", &h_hat_z)?;
        transcript.append_field_element(b"h_a", &h_alpha)?;
        transcript.append_field_element(b"s_z", &s_z)?;
        transcript.append_field_element(b"s_hat_z", &s_hat_z)?;
        transcript.append_field_element(b"d_z", &d_z)?;

        // 5. BDFG20
        let batch_gamma: E::ScalarField = transcript.get_and_append_challenge(b"batch_gamma")?;

        let vec_f = vec![&g_poly, &h_poly, &s_poly, &d_poly];
        let g_points = vec![z, z_inverse];
        let g_values = vec![g_z, g_hat_z];
        let rr_g = generate_r_i(&g_points, &g_values);

        let h_points = vec![z, z_inverse, alpha];
        let h_values = vec![h_z, h_hat_z, h_alpha];
        let rr_h = generate_r_i(&h_points, &h_values);

        let s_points = vec![z, z_inverse];
        let s_values = vec![s_z, s_hat_z];
        let rr_s = generate_r_i(&s_points, &s_values);

        let d_points = vec![z];
        let d_values = vec![d_z];
        let rr_d = generate_r_i(&d_points, &d_values);

        let rr = vec![&rr_g, &rr_h, &rr_s, &rr_d];
        let t = vec![z, z_inverse, alpha];
        let all_s = vec![g_points.clone(), h_points.clone(), s_points.clone(), d_points.clone()];

        let batch_f = compute_batch_f(vec_f.clone(), rr.clone(), t.clone(), all_s.clone(), batch_gamma);
        let batch_w = compute_batch_w(batch_f, t.clone());

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&batch_w);
        let msm_time = start_timer!(|| "MSM to compute commitment to batch_w(X)");
        let com_batch_w =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);

        let mut buf_batch_w = Vec::new();
        com_batch_w.serialize_compressed(&mut buf_batch_w).unwrap(); // 序列化为字节
        transcript.append_message(b"commitment_batch_w", &buf_batch_w)?;

        let batch_z: E::ScalarField = transcript.get_and_append_challenge(b"batch_z")?;
        let batch_l = compute_batch_l(vec_f, rr, &batch_w, t, all_s, batch_gamma, batch_z);
        let batch_w_hat = compute_batch_w_hat(batch_l, batch_z);

        let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(&batch_w_hat);
        let msm_time = start_timer!(|| "MSM to compute commitment to batch_w_hat(X)");
        let com_batch_w_hat =
            E::G1::msm_unchecked(&prover_param.borrow().powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
        end_timer!(msm_time);


        let proofs = vec![com_h, com_g, com_q, com_s, com_d, com_big_h, com_batch_w, com_batch_w_hat];

        let proof:MercuryProof<E> = MercuryProof{
            proofs,
            g_value: g_values,
            h_value: h_values,
            s_value: s_values,
            d_value: d_values,
            pi_z: com_big_h,
            b,
        };

        let eval = evaluate_opt(polynomial, point);

        end_timer!(open_timer);
        Ok((proof, eval))
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
        let open_timer = start_timer!(|| format!("multi open {} points", points.len()));
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
        let timer = start_timer!(|| format!("compute tilde g for {} points", points.len()));
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
        end_timer!(timer);

        let timer = start_timer!(|| format!("compute tilde eq for {} points", points.len()));
        let tilde_eqs: Vec<_> = deduped_points
            .iter()
            .map(|point| {
                let eq_b_zi = build_eq_x_r_vec(point).unwrap();
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_var, eq_b_zi,
                ))
            })
            .collect();
        end_timer!(timer);

        // built the virtual polynomial for SumCheck
        let timer = start_timer!(|| format!("sum check prove of {} variables", num_var));

        let step = start_timer!(|| "add mle");
        let mut sum_check_vp = VirtualPolynomial::new(num_var);
        for (merged_tilde_g, tilde_eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
            sum_check_vp.add_mle_list([merged_tilde_g.clone(), tilde_eq], E::ScalarField::one())?;
        }
        end_timer!(step);

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

        end_timer!(timer);

        // a2 := sumcheck's point
        let a2 = &proof.point[..num_var];

        // build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is the
        // sumcheck's point \tilde eq_i(a2) = eq(a2, point_i)
        let step = start_timer!(|| "evaluate at a2");
        let mut g_prime = Arc::new(DenseMultilinearExtension::zero());
        for (merged_tilde_g, point) in merged_tilde_gs.iter().zip(deduped_points.iter()) {
            let eq_i_a2 = eq_eval(a2, point)?;
            *Arc::make_mut(&mut g_prime) += (eq_i_a2, merged_tilde_g.deref());
        }
        end_timer!(step);

        let step = start_timer!(|| "pcs open");
        let (g_prime_proof, _g_prime_eval) = Self::open(prover_param, &g_prime, a2.to_vec().as_ref())?;
        // assert_eq!(g_prime_eval, tilde_g_eval);
        end_timer!(step);

        let step = start_timer!(|| "evaluate fi(pi)");
        end_timer!(step);
        end_timer!(open_timer);

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
    fn verify(
        verifier_param: &Self::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &E::ScalarField,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let check_time = start_timer!(|| "verify");
        let cha_time = start_timer!(|| "generate challenge");

        let mut transcript = IOPTranscript::<E::ScalarField>::new(b"HyperPlonkProtocol");
        let com_h = proof.proofs[0];
        let com_g = proof.proofs[1];
        let com_q = proof.proofs[2];
        let com_s = proof.proofs[3];
        let com_d = proof.proofs[4];
        let com_big_h = proof.proofs[5];
        let com_batch_w = proof.proofs[6];
        let com_batch_w_hat = proof.proofs[7];
        let b = proof.b;

        let g_z = proof.g_value[0];
        let g_hat_z = proof.g_value[1];
        let h_z = proof.h_value[0];
        let h_hat_z = proof.h_value[1];
        let h_a = proof.h_value[2];
        let s_z = proof.s_value[0];
        let s_hat_z = proof.s_value[1];
        let d_z = proof.d_value[0];


        let (u2, u1) = split_u(&point);

        // recreate alpha
        let mut buf = Vec::new();
        com_h.serialize_compressed(&mut buf)?; // 序列化为字节
        transcript.append_message(b"commitment_h", &buf)?;
        let alpha: E::ScalarField = transcript.get_and_append_challenge(b"challenge_alpha")?;

        // recreate gamma
        let mut buf_g = Vec::new();
        com_g.serialize_compressed(&mut buf_g).unwrap(); // 序列化为字节
        let mut buf_q = Vec::new();
        com_q.serialize_compressed(&mut buf_q).unwrap(); // 序列化为字节
        transcript.append_message(b"commitment_g", &buf_g)?;
        transcript.append_message(b"commitment_q", &buf_q)?;
        let gamma: E::ScalarField = transcript.get_and_append_challenge(b"challenge_gamma")?;

        // recreate z
        let mut buf_s = Vec::new();
        com_s.serialize_compressed(&mut buf_s).unwrap(); // 序列化为字节
        let mut buf_d = Vec::new();
        com_d.serialize_compressed(&mut buf_d).unwrap(); // 序列化为字节
        transcript.append_message(b"commitment_s", &buf_s)?;
        transcript.append_message(b"commitment_d", &buf_d)?;
        let z: E::ScalarField = transcript.get_and_append_challenge(b"challenge_z")?;
        let z_inverse = z.inverse().expect("z must have an inverse");

        // recreate batch_gamma
        let mut buf_big_h = Vec::new();
        com_big_h.serialize_compressed(&mut buf_big_h).unwrap(); // 序列化为字节
        transcript.append_message(b"commitment_big_h", &buf_big_h)?;
        transcript.append_field_element(b"g_z", &g_z)?;
        transcript.append_field_element(b"g_hat_z", &g_hat_z)?;
        transcript.append_field_element(b"h_z", &h_z)?;
        transcript.append_field_element(b"h_hat_z", &h_hat_z)?;
        transcript.append_field_element(b"h_a", &h_a)?;
        transcript.append_field_element(b"s_z", &s_z)?;
        transcript.append_field_element(b"s_hat_z", &s_hat_z)?;
        transcript.append_field_element(b"d_z", &d_z)?;
        let batch_gamma: E::ScalarField = transcript.get_and_append_challenge(b"batch_gamma")?;

        // recreate batch_z
        let mut buf_batch_w = Vec::new();
        com_batch_w.serialize_compressed(&mut buf_batch_w).unwrap(); // 序列化为字节
        transcript.append_message(b"commitment_batch_w", &buf_batch_w)?;

        let batch_z: E::ScalarField = transcript.get_and_append_challenge(b"batch_z")?;

        end_timer!(cha_time);

        // 0. step c
        let cha_time = start_timer!(|| "step c");
        let exponent = (b - 1) as u64;
        let d_z_v = z.pow([exponent]) * g_hat_z;

        let two = E::ScalarField::from(2u64);
        let h_alpha = ( g_z * pu_evaluate(&u1 , z_inverse) + g_hat_z * pu_evaluate(&u1 , z)
            + gamma * ( h_z * pu_evaluate(&u2 , z_inverse) + h_hat_z * pu_evaluate(&u2 , z) - *value * two)
            - z * s_z - z_inverse * s_hat_z) / two ;
        let d_acc = d_z_v == proof.d_value[0];
        let h_acc = h_alpha == h_a ;

        end_timer!(cha_time);

        // 1. step f
        let cha_time = start_timer!(|| "step f");
        let z_b = z.pow([b as u64]);
        let z_b_alpha = z_b - alpha;
        let comm = commitment.0;
        let comm_q = com_q.into_group();
        let g =  verifier_param.g.into_group();
        let beta_h = verifier_param.beta_h.into_group();
        let h = verifier_param.h.into_group();

        let inner_1 = comm.into_group() - comm_q.mul(z_b_alpha) - g.mul(g_z);
        let inner_2 = beta_h -  h.mul(z);
        // let lhs = E::pairing(inner_1, h);
        // let rhs = E::pairing(proof.pi_z.into_group(), inner_2);
        // let step_f = {
        //     lhs == rhs
        // };

        let step_f = E::multi_pairing(
            [
                E::G1Prepared::from(inner_1),
                E::G1Prepared::from(proof.pi_z.into_group().neg()), // 取负来等价地变为 = 1 检查
            ],
            [
                E::G2Prepared::from(h),
                E::G2Prepared::from(inner_2),
            ],
        ).0.is_one();

        end_timer!(cha_time);


        // 2. step g
        let cha_time = start_timer!(|| "step g");
        let g_points = vec![z, z_inverse];
        let g_values = &proof.g_value;
        let rr_g = generate_r_i(&g_points, &g_values);

        let h_points = vec![z, z_inverse, alpha];
        let h_values = &proof.h_value;
        let rr_h = generate_r_i(&h_points, &h_values);

        let s_points = vec![z, z_inverse];
        let s_values = &proof.s_value;
        let rr_s = generate_r_i(&s_points, &s_values);

        let d_points = vec![z];
        let d_values = &proof.d_value;
        let rr_d = generate_r_i(&d_points, &d_values);

        let t = vec![z, z_inverse, alpha];
        let rr = vec![&rr_g, &rr_h, &rr_s, &rr_d];
        let vec_com = vec![com_g, com_h, com_s, com_d];
        // let all_s = vec![g_points, h_points, s_points, d_points];
        let t_s = vec![vec![alpha], vec![],vec![alpha],vec![alpha,z_inverse]];
        let mut re = com_batch_w_hat.into_group().mul(batch_z);
        let mut scaler_2_sum = E::ScalarField::zero();
        for i in 0..vec_com.len() {
            let gamma_pow_i = batch_gamma.pow([i as u64]);
            let z_s_i = compute_zs(&t_s[i]);
            let scaler = gamma_pow_i * z_s_i.evaluate(&batch_z);
            let scaler_2 = rr[i].evaluate(&batch_z) * scaler;

            re += vec_com[i].into_group().mul(scaler);
            scaler_2_sum += scaler_2;
        }

        // 将最终 g.mul(scaler_2_sum) 一次性减掉
        re -= g.mul(scaler_2_sum);
        let z_t = compute_zs(&t);
        re = re - com_batch_w.into_group().mul(z_t.evaluate(&batch_z));   // batch_z

        // let lhs_2 = E::pairing(re, h);
        // let rhs_2 = E::pairing(com_batch_w_hat.into_group(), beta_h);
        // let step_g = {
        //     lhs_2 == rhs_2
        // };

        let step_g = E::multi_pairing(
            [
                E::G1Prepared::from(re),
                E::G1Prepared::from(com_batch_w_hat.into_group().neg()), // 取负来等价地变为 = 1 检查
            ],
            [
                E::G2Prepared::from(h),
                E::G2Prepared::from(beta_h),
            ],
        ).0.is_one();

        end_timer!(cha_time);


        let all_pass = step_f && step_g && d_acc && h_acc;

        end_timer!(check_time, || format!("Result: {}", all_pass));
        Ok(all_pass)
    }

    /// Verifies that `value_i` is the evaluation at `x_i` of the polynomial
    /// `poly_i` committed inside `comm`.
    fn batch_verify(
        verifier_param: &Self::VerifierParam,
        commitments: &[Self::Commitment],
        points: &[Self::Point],
        batch_proof: &Self::BatchProof,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, PCSError> {
        let open_timer = start_timer!(|| "batch verification");
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
        let step = start_timer!(|| "build homomorphic commitment");
        let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

        let mut scalars = vec![];
        let mut bases = vec![];

        for (i, point) in points.iter().enumerate() {
            let eq_i_a2 = eq_eval(a2, point)?;
            scalars.push(eq_i_a2 * eq_t_list[i]);
            bases.push(commitments[i].0);
        }
        let g_prime_commit = E::G1::msm_unchecked(&bases, &scalars);
        end_timer!(step);

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

        end_timer!(open_timer);
        Ok(res)
    }
}

fn skip_leading_zeros<F: PrimeField, P: DenseUVPolynomial<F>>(p: &P) -> (usize, &[F]) {
    let mut num_leading_zeros = 0;
    while num_leading_zeros < p.coeffs().len() && p.coeffs()[num_leading_zeros].is_zero() {
        num_leading_zeros += 1;
    }
    (num_leading_zeros, &p.coeffs()[num_leading_zeros..])
}



#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{test_rng, vec::Vec, UniformRand};
    use ark_std::rand::Rng;
    use crate::{MultilinearKzgPCS, MultilinearUniversalParams};


    type E = Bls12_381;
    type Fr = <E as Pairing>::ScalarField;

    fn test_single_helper<R: Rng>(
        params: &MercuryUniversalParams<E>,
        poly: &Arc<DenseMultilinearExtension<Fr>>,
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let nv = poly.num_vars();
        assert_ne!(nv, 0);
        let (ck, vk) = MercuryPCS::trim(params, None, Some(nv))?;
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
        let com= MercuryPCS::<E>::commit(&ck, poly)?;
        let (proof,value) = MercuryPCS::<E>::open(&ck, poly, &point)?;
        assert!(MercuryPCS::verify(&vk, &com, &point, &value, &proof)?);

        // let value = Fr::rand(rng);
        // assert!(!MercuryPCS::verify(&vk, &com, &point, &value, &proof)?);

        Ok(())
    }

    fn test_single_helper_2<R: Rng>(
        params: &MultilinearUniversalParams<E>,
        poly: &Arc<DenseMultilinearExtension<Fr>>,
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let nv = poly.num_vars();
        assert_ne!(nv, 0);
        let (ck, vk) = MultilinearKzgPCS::trim(params, None, Some(nv))?;
        let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
        let com = MultilinearKzgPCS::commit(&ck, poly)?;
        let (proof,value) = MultilinearKzgPCS::open(&ck, poly, &point)?;

        assert!(MultilinearKzgPCS::verify(
            &vk, &com, &point, &value, &proof
        )?);

        // let value = Fr::rand(rng);
        // assert!(!MultilinearKzgPCS::verify(
        //     &vk, &com, &point, &value, &proof
        // )?);

        Ok(())
    }

    #[test]
    fn test_mercury() -> Result<(), PCSError> {
        let mut rng = test_rng();
        let mut mer_setup = Duration::new(0, 0);
        let mut mer_com = Duration::new(0, 0);
        let mut mer_open = Duration::new(0, 0);
        let mut mer_ver = Duration::new(0, 0);

        let mut kzg_setup = Duration::new(0, 0);
        let mut kzg_com = Duration::new(0, 0);
        let mut kzg_open = Duration::new(0, 0);
        let mut kzg_ver = Duration::new(0, 0);

        let mut sam_setup = Duration::new(0, 0);
        let mut sam_com = Duration::new(0, 0);
        let mut sam_open = Duration::new(0, 0);
        let mut sam_ver = Duration::new(0, 0);

        for _ in 0..10 {
            let poly1 = Arc::new(DenseMultilinearExtension::rand(8, &mut rng));

            let start_mer_setup = Instant::now();
            let params = MercuryPCS::<E>::gen_srs_for_testing(&mut rng, 8)?;
            mer_setup += start_mer_setup.elapsed();

            let nv = poly1.num_vars();
            assert_ne!(nv, 0);
            let (ck, vk) = MercuryPCS::trim(params, None, Some(nv))?;
            let point: Vec<_> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

            let start_mer_com = Instant::now();
            let com= MercuryPCS::<E>::commit(&ck, &poly1)?;
            mer_com += start_mer_com.elapsed();

            let start_mer_open = Instant::now();
            let (proof,value) = MercuryPCS::<E>::open(&ck, &poly1,  &point)?;
            mer_open += start_mer_open.elapsed();

            let start_mer_ver = Instant::now();
            // let value = evaluate_opt(&poly1, &point);
            assert!(MercuryPCS::verify(&vk, &com, &point, &value, &proof)?);
            mer_ver += start_mer_ver.elapsed();



            let start_kzg_setup = Instant::now();
            let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 8)?;
            kzg_setup += start_kzg_setup.elapsed();

            let (ck, vk) = MultilinearKzgPCS::trim(params, None, Some(nv))?;

            let start_kzg_com = Instant::now();
            let com= MultilinearKzgPCS::commit(&ck, &poly1)?;
            kzg_com += start_kzg_com.elapsed();

            let start_kzg_open = Instant::now();
            let (proof,value) = MultilinearKzgPCS::open(&ck, &poly1, &point)?;
            kzg_open += start_kzg_open.elapsed();

            let start_kzg_ver = Instant::now();
            // let value = evaluate_opt(&poly1, &point);
            assert!(MultilinearKzgPCS::verify(
                &vk, &com, &point, &value, &proof
            )?);
            kzg_ver += start_kzg_ver.elapsed();



            // let mut lambda = 128;
            // let miu = nv;
            // let start_sam_setup = Instant::now();
            // let pp = SamaritanPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, miu).expect("生成 SRS 失败");
            // sam_setup += start_sam_setup.elapsed();
            //
            // let evaluations: Vec<ark_bls12_381::Fr> = (1..=1 << miu).map(|i| ark_bls12_381::Fr::from(i as u64)).collect();
            // let f_hat = DensePolynomial::from_coefficients_vec(poly1.evaluations.to_vec());
            //
            // let start_sam_com = Instant::now();
            // let (commitment, _) = SamaritanPCS::<Bls12_381>::commit(&pp, &f_hat).expect("承诺失败");
            // sam_com += start_sam_com.elapsed();
            //
            // let start_sam_open = Instant::now();
            // let proof = SamaritanPCS::<Bls12_381>::open(&pp, &f_hat, &(), &point).expect("打开承诺失败");
            // sam_open += start_sam_open.elapsed();
            //
            // let start_sam_ver = Instant::now();
            // let is_valid =
            //     SamaritanPCS::<Bls12_381>::verify(&pp, &commitment, &point, &proof.evaluations[0], &proof)
            //         .expect("验证失败");
            // sam_ver += start_sam_ver.elapsed();
            // assert!(is_valid, "证明验证未通过");
            // println!("承诺、打开和验证测试成功！");
        }

        let rounds = 10;

        let mer_total = mer_setup + mer_com + mer_open + mer_ver;
        let mer_total_wo_setup = mer_com + mer_open + mer_ver;

        let kzg_total = kzg_setup + kzg_com + kzg_open + kzg_ver;
        let kzg_total_wo_setup = kzg_com + kzg_open + kzg_ver;

        let sam_total = sam_setup + sam_com + sam_open + sam_ver;
        let sam_total_wo_setup = sam_com + sam_open + sam_ver;

        println!("\n--- MercuryPCS 平均耗时 ---");
        println!("Setup:         {:?} μs", mer_setup.as_micros() / rounds);
        println!("Commit:        {:?} μs", mer_com.as_micros() / rounds);
        println!("Open:          {:?} μs", mer_open.as_micros() / rounds);
        println!("Verify:        {:?} μs", mer_ver.as_micros() / rounds);
        println!("Total:         {:?} μs", mer_total.as_micros() / rounds);
        println!("Total w/o Setup: {:?} μs", mer_total_wo_setup.as_micros() / rounds);

        println!("\n--- KZGPCS 平均耗时 ---");
        println!("Setup:         {:?} μs", kzg_setup.as_micros() / rounds);
        println!("Commit:        {:?} μs", kzg_com.as_micros() / rounds);
        println!("Open:          {:?} μs", kzg_open.as_micros() / rounds);
        println!("Verify:        {:?} μs", kzg_ver.as_micros() / rounds);
        println!("Total:         {:?} μs", kzg_total.as_micros() / rounds);
        println!("Total w/o Setup: {:?} μs", kzg_total_wo_setup.as_micros() / rounds);

        println!("\n--- SamaritanPCS 平均耗时 ---");
        println!("Setup:         {:?} μs", sam_setup.as_micros() / rounds);
        println!("Commit:        {:?} μs", sam_com.as_micros() / rounds);
        println!("Open:          {:?} μs", sam_open.as_micros() / rounds);
        println!("Verify:        {:?} μs", sam_ver.as_micros() / rounds);
        println!("Total:         {:?} μs", sam_total.as_micros() / rounds);
        println!("Total w/o Setup: {:?} μs", sam_total_wo_setup.as_micros() / rounds);


        // let params = MercuryPCS::<E>::gen_srs_for_testing(&mut rng, 8)?;
        // let poly1 = Arc::new(DenseMultilinearExtension::rand(8, &mut rng));
        // test_single_helper(&params, &poly1, &mut rng)?;
        // let params = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 8)?;
        // test_single_helper_2(&params, &poly1, &mut rng)?;
        //

        // num_vars = 1 时 有问题
        // let poly2 = Arc::new(DenseMultilinearExtension::rand(2, &mut rng));
        // test_single_helper(&params, &poly2, &mut rng)?;

        Ok(())
    }
}

// pub fn print_vec_field<F: PrimeField>(v: &[F]) {
//     let formatted: Vec<String> = v.iter().map(|x| x.to_string()).collect();
//     println!("[{}]", formatted.join(", "));
// }


use crate::pcs::multilinear_kzg::util::eq_eval;
