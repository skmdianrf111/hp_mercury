// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Useful utilities for KZG PCS
use ark_ec::pairing::Pairing;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseMultilinearExtension, DenseUVPolynomial,
    MultilinearExtension, Polynomial,
};
use ark_ec::{
    scalar_mul::variable_base::VariableBaseMSM, AffineRepr, CurveGroup,
};
use crate::pcs::Samaritan::srs::{SamaritanProverParam, SamaritanUniversalParams, SamaritanVerifierParam};
use ark_poly_commit::kzg10::{Powers, Randomness, UniversalParams, KZG10};
use ark_std::{end_timer, start_timer, vec::Vec};
use ark_std::{format, log2, string::ToString, test_rng, vec};
use std::{borrow::Cow, collections::HashMap};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use crate::PCSError;
use crate::pcs::structs::Commitment;
use ark_poly_commit::kzg10::Proof;

type MultiPoly<F> = HashMap<(usize, usize), F>;


/// Generate eq(t,x), a product of multilinear polynomials with fixed t.
/// eq(a,b) is takes extensions of a,b in {0,1}^num_vars such that if a and b in
/// {0,1}^num_vars are equal then this polynomial evaluates to 1.
pub(crate) fn eq_extension<F: PrimeField>(t: &[F]) -> Vec<DenseMultilinearExtension<F>> {
    let start = start_timer!(|| "eq extension");

    let dim = t.len();
    let mut result = Vec::new();
    for (i, &ti) in t.iter().enumerate().take(dim) {
        let mut poly = Vec::with_capacity(1 << dim);
        for x in 0..(1 << dim) {
            let xi = if x >> i & 1 == 1 { F::one() } else { F::zero() };
            let ti_xi = ti * xi;
            poly.push(ti_xi + ti_xi - xi - ti + F::one());
        }
        result.push(DenseMultilinearExtension::from_evaluations_vec(dim, poly));
    }

    end_timer!(start);
    result
}

/// Evaluate eq polynomial. use the public one later
pub(crate) fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> Result<F, PCSError> {
    if x.len() != y.len() {
        return Err(PCSError::InvalidParameters(
            "x and y have different length".to_string(),
        ));
    }
    let start = start_timer!(|| "eq_eval");
    let mut res = F::one();
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xi_yi = xi * yi;
        res *= xi_yi + xi_yi - xi - yi + F::one();
    }
    end_timer!(start);
    Ok(res)
}
fn skip_leading_zeros<F: PrimeField, P: DenseUVPolynomial<F>>(p: &P) -> (usize, &[F]) {
    let mut num_leading_zeros = 0;
    while num_leading_zeros < p.coeffs().len() && p.coeffs()[num_leading_zeros].is_zero() {
        num_leading_zeros += 1;
    }
    (num_leading_zeros, &p.coeffs()[num_leading_zeros..])
}
pub(crate) fn commit<E: Pairing>(
    pp: &SamaritanProverParam<E::G1Affine>,
    f: &DensePolynomial<E::ScalarField>,
) -> Result<Commitment<E>, PCSError>
where
    E::ScalarField: PrimeField,
    E::G1Affine: CanonicalSerialize + CanonicalDeserialize,
    E::G2Affine: CanonicalSerialize + CanonicalDeserialize,
{
    let mut rng = test_rng();

    let degree = f.degree();
    let (num_leading_zeros, plain_coeffs) = skip_leading_zeros(f);
    let commitment =
            E::G1::msm_unchecked(&pp.powers_of_g[num_leading_zeros..], plain_coeffs)
                .into();
    Ok(Commitment(commitment))
}
pub(crate) fn trim_trailing_zeros<F: Field>(mut coeffs: Vec<F>) -> Vec<F> {
    while coeffs.len() > 1 && coeffs.last().map_or(false, |c| c.is_zero()) {
        coeffs.pop();
    }
    coeffs
}
pub(crate) fn compute_f_hat<F: Field>(evaluations: &[F], num_vars: usize) -> Vec<F> {
    let n = num_vars;
    let mut f_hat_coeffs = vec![F::zero(); 1 << n];

    for i in 0..(1 << n) {
        let index = (0..n)
            .map(|k| (i >> k) & 1)
            .fold(0, |acc, bit| (acc << 1) | bit);
        f_hat_coeffs[i] = evaluations[index];
    }

    f_hat_coeffs
}

pub(crate) fn generate_ghat<F: Field>(f_coeffs: &[F], miu: u32) -> Vec<DensePolynomial<F>> {
    let n = 1 << miu;
    let l = 1 << miu.ilog2();
    let m = n / l;

    let mut f_matrix: Vec<Vec<F>> = vec![vec![F::zero(); m]; l];
    for i in 0..l {
        for j in 0..m {
            let index = i * m + j;
            f_matrix[i][j] = f_coeffs[index];
        }
    }

    let g_hat: Vec<DensePolynomial<F>> = f_matrix
        .iter()
        .map(|row| DensePolynomial::from_coefficients_vec(row.clone()))
        .collect();

    g_hat
}

pub(crate) fn compute_q<F: Field>(g_vec: &[DensePolynomial<F>], l: usize) -> MultiPoly<F> {
    let mut q_poly: MultiPoly<F> = HashMap::new();

    for (i, g_i) in g_vec.iter().enumerate().take(l) {
        let x_deg = i;
        for (y_deg, coeff) in g_i.coeffs.iter().enumerate() {
            if *coeff == F::zero() {
                continue;
            }
            let key = (x_deg, y_deg);
            *q_poly.entry(key).or_insert(F::zero()) += coeff;
        }
    }
    q_poly
}

pub(crate) fn compute_gtilde<F: Field>(
    f_poly: &DenseMultilinearExtension<F>,
    miu: u32,
    i: usize,
) -> DenseMultilinearExtension<F> {
    let n = 1 << miu;
    let l = 1 << miu.ilog2();
    let m = n / l;
    let nu = log2(l) as usize;

    let i = i - 1;
    let i_binary: Vec<F> = (0..nu)
        .map(|j| {
            if (i >> (nu - 1 - j)) & 1 == 1 {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect();
    let g_tilde = f_poly.fix_variables(&i_binary);
    g_tilde
}

pub(crate) fn kron<F: Field>(v1: &Vec<F>, v2: &Vec<F>) -> Vec<F> {
    let mut result = Vec::with_capacity(v1.len() * v2.len());
    for x in v1 {
        for y in v2 {
            result.push(*x * *y);
        }
    }
    result
}

pub(crate) fn compute_psihat<F: PrimeField>(z: &[F]) -> DensePolynomial<F> {
    let mut psi_hat = DensePolynomial::from_coefficients_vec(vec![F::one()]);
    let mu = z.len();
    let mut phi = vec![F::one() - z[0], z[0]];
    for i in 1..mu {
        let next_vector = vec![F::one() - z[i], z[i]];
        phi = kron(&phi, &next_vector);
    }
    phi.reverse();
    psi_hat = DensePolynomial::from_coefficients_vec(phi);
    psi_hat
}

pub(crate) fn compute_phat<F: PrimeField>(q_poly: MultiPoly<F>, gamma: F) -> DensePolynomial<F> {
    let max_y = q_poly.keys().map(|&(_, y_deg)| y_deg).max().unwrap_or(0);
    let mut p_hat_coeffs = vec![F::zero(); max_y + 1];

    for (&(x_deg, y_deg), &coeff) in &q_poly {
        let mut term = coeff;
        for _ in 0..x_deg {
            term *= gamma;
        }
        p_hat_coeffs[y_deg] += term;
    }
    DensePolynomial {
        coeffs: p_hat_coeffs,
    }
}

pub(crate) fn compute_r<F: Field>(q_poly: MultiPoly<F>, gamma: F) -> MultiPoly<F>
where
    F: std::ops::MulAssign + std::ops::AddAssign + Copy + PartialEq,
{
    let mut r_poly: MultiPoly<F> = HashMap::new();
    for (&(x_deg, y_deg), &coeff) in &q_poly {
        if x_deg == 0 {
            continue;
        }
        let mut gamma_power = F::one();
        for _ in 0..(x_deg - 1) {
            gamma_power *= gamma;
        }
        for k in 0..x_deg {
            let r_coeff = coeff * gamma_power;
            *r_poly.entry((k, y_deg)).or_insert(F::zero()) += r_coeff;
            gamma_power *= gamma;
        }
    }
    r_poly
}

pub(crate) fn compute_r_hat<F: Field>(r_poly: MultiPoly<F>, m: usize) -> DensePolynomial<F>
where
    F: std::ops::MulAssign + std::ops::AddAssign + Copy + PartialEq,
{
    let max_deg = r_poly
        .keys()
        .map(|&(x_deg, y_deg)| m * x_deg + y_deg)
        .max()
        .unwrap_or(0);
    let mut r_hat_coeffs = vec![F::zero(); max_deg + 1];
    for (&(x_deg, y_deg), &coeff) in &r_poly {
        let degree = m * x_deg + y_deg;
        r_hat_coeffs[degree] += coeff;
    }
    DensePolynomial {
        coeffs: r_hat_coeffs,
    }
}
pub(crate) fn polynomial_mul<F: Field>(
    poly1: &DensePolynomial<F>,
    poly2: &DensePolynomial<F>,
) -> DensePolynomial<F> {
    let mut result_coeffs = vec![F::zero(); poly1.degree() + poly2.degree() + 1];

    for (i, &coeff1) in poly1.coeffs.iter().enumerate() {
        for (j, &coeff2) in poly2.coeffs.iter().enumerate() {
            result_coeffs[i + j] += coeff1 * coeff2;
        }
    }

    DensePolynomial::from_coefficients_vec(result_coeffs)
}

pub(crate) fn compute_t<F: PrimeField>(
    p_hat: &DensePolynomial<F>,
    u_hat: &DensePolynomial<F>,
    b_hat: &DensePolynomial<F>,
    beta: F,
    m: u64,
    ell: u64,
) -> DensePolynomial<F> {
    let compute_term = |poly: &DensePolynomial<F>, k: u64| -> DensePolynomial<F> {
        let d = poly.degree() as u64;
        let mut coeffs = vec![F::zero(); (k + 1) as usize];
        for i in 0..=d {
            let power = k as i64 - i as i64;
            if power < 0 {
                panic!(
                    "Resulting polynomial has negative powers: k = {}, i = {}",
                    k, i
                );
            }
            let idx = power as usize;
            if idx < coeffs.len() {
                coeffs[idx] = poly.coeffs()[i as usize];
            }
        }
        DensePolynomial::from_coefficients_vec(coeffs)
    };

    let add_polynomials =
        |p1: &DensePolynomial<F>, p2: &DensePolynomial<F>| -> DensePolynomial<F> {
            let max_len = p1.coeffs().len().max(p2.coeffs().len());
            let mut result = vec![F::zero(); max_len];
            for i in 0..max_len {
                let coeff1 = if i < p1.coeffs().len() {
                    p1.coeffs()[i]
                } else {
                    F::zero()
                };
                let coeff2 = if i < p2.coeffs().len() {
                    p2.coeffs()[i]
                } else {
                    F::zero()
                };
                result[i] = coeff1 + coeff2;
            }
            while result.len() > 1 && result.last() == Some(&F::zero()) {
                result.pop();
            }
            DensePolynomial::from_coefficients_vec(result)
        };

    let multiply_by_scalar = |poly: &DensePolynomial<F>, beta: F| -> DensePolynomial<F> {
        let new_coeffs = poly.coeffs().iter().map(|&coeff| coeff * beta).collect();
        DensePolynomial::from_coefficients_vec(new_coeffs)
    };

    let term1 = compute_term(p_hat, m.checked_sub(1).expect("m must be >= 1"));

    let term2_base = compute_term(u_hat, m.checked_sub(2).expect("m must be >= 2"));
    let term2 = multiply_by_scalar(&term2_base, beta);

    let term3_base = compute_term(b_hat, ell.checked_sub(2).expect("ell must be >= 2"));
    let beta_squared = beta * beta;
    let term3 = multiply_by_scalar(&term3_base, beta_squared);

    let sum1 = add_polynomials(&term1, &term2);
    let result = add_polynomials(&sum1, &term3);

    result
}

pub(crate) fn kzg_prove<E: Pairing>(
    pp: &SamaritanProverParam<E::G1Affine>,
    poly: &DensePolynomial<E::ScalarField>,
    alpha: E::ScalarField,
) -> Result<(E::ScalarField, Proof<E>), PCSError>
where
    E::ScalarField: PrimeField,
    E::G1Affine: CanonicalSerialize + CanonicalDeserialize,
    E::G2Affine: CanonicalSerialize + CanonicalDeserialize,
{
    let v = poly.evaluate(&alpha);
    let mut q_coeffs = poly.coeffs.clone();

    // 检查空多项式或零多项式
    if q_coeffs.is_empty() || q_coeffs.iter().all(|c| c.is_zero()) {
        return Ok((
            E::ScalarField::zero(),
            Proof {
                w: E::G1Affine::zero(),
                random_v: None,
            },
        ));
    }

    q_coeffs[0] -= v;

    let mut quotient = Vec::with_capacity(q_coeffs.len().saturating_sub(1));
    let mut remainder = E::ScalarField::zero();
    for i in (1..q_coeffs.len()).rev() {
        let leading_coeff = q_coeffs[i];
        quotient.push(leading_coeff);
        q_coeffs[i - 1] += leading_coeff * alpha;
    }
    remainder = q_coeffs[0];

    if !remainder.is_zero() {
        return Err(PCSError::InvalidParameters(
            "Polynomial division remainder is not zero".to_string(),
        ));
    }
    quotient.reverse();
    let q_poly = DensePolynomial::from_coefficients_vec(quotient);
    let commitment: crate::pcs::structs::Commitment<E> = commit(pp, &q_poly)?;
    let proof = Proof {
        w: commitment.0,
        random_v: None,
    };
    Ok((v, proof))
}