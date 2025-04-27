#![feature(isqrt)]
use ark_ec::pairing::Pairing;
use ark_ff::{ One, Zero};

use ark_std::ops::{Mul, Sub};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, domain::{EvaluationDomain, Radix2EvaluationDomain}};
use ark_poly::{Polynomial};
use ark_ff::PrimeField;
use ark_ff::Field;
use std::iter::Iterator;
use std::cmp;


/// 将 u 拆分成两半
pub fn split_u<F: Field>(u: &[F]) -> (Vec<F>, Vec<F>) {
    let mid = u.len() / 2;
    let u2 = u[..mid].to_vec();
    let u1 = u[mid..].to_vec();
    (u1, u2)
}

/// eq 的计算
fn eq<F: Field>(i: usize, u: &[F], s: usize) -> F {
    let mut result = F::one();
    for j in 0..s {
        let i_j = if (i >> j) & 1 == 1 { F::one() } else { F::zero() };
        result *= i_j * u[j] + (F::one() - i_j) * (F::one() - u[j]);
    }
    result
}

/// h(X)
/// input: poly: f(X) , u1: u = ( u1 , u2 )
pub fn partial_sum<F: Field>(
    poly: &DensePolynomial<F>,
    u1: &Vec<F>,
) -> DensePolynomial<F> {
    let n = poly.degree() + 1;
    let mut b = n.isqrt(); // 计算 b = sqrt(n)
    let t = b.ilog2() as usize;
    // println!("n:{:?}", n);
    // println!("n:{:?}", b);
    // println!("n:{:?}", t);
    let mut h_coff: Vec<F> = vec![F::zero(); b]; // 初始化系数向量
    for (i, coeff) in poly.coeffs.iter().enumerate() {
        h_coff[i / b] += *coeff * eq(i % b, u1, t);
    }
    DensePolynomial::from_coefficients_vec(h_coff)
}

pub fn compute_folded_g_and_q<F: Field>(
    f: &DensePolynomial<F>,
    alpha: F,
) -> (DensePolynomial<F>, DensePolynomial<F>) {
    let n = f.degree() + 1;
    let b = n.isqrt();
    let mut g_coeffs = Vec::with_capacity(b);
    let mut q_coeffs = vec![F::zero(); n];
    let mut tmp_coeffs = Vec::with_capacity(b);

    for i in 0..b {
        tmp_coeffs.extend((0..b).map(|k| f.coeffs[i + k * b]));
        let f_poly = DensePolynomial::from_coefficients_vec(std::mem::take(&mut tmp_coeffs));

        let g_i = f_poly.evaluate(&alpha);
        g_coeffs.push(g_i);

        let q_i = divide_polynomial(&f_poly, alpha);
        for j in 0..b {
            let coeff = q_i.coeffs.get(j).copied().unwrap_or(F::zero());
            q_coeffs[i + j * b] = coeff;
        }
    }

    (
        DensePolynomial::from_coefficients_vec(g_coeffs),
        DensePolynomial::from_coefficients_vec(q_coeffs),
    )
}


/// 计算 q_i
pub fn divide_polynomial<F: Field>(
    f_i: &DensePolynomial<F>,
    alpha: F,
) -> DensePolynomial<F> {
    let coeffs = &f_i.coeffs; // 获取多项式系数
    let mut quotient_coeffs = vec![F::zero(); coeffs.len() - 1]; // 商多项式系数
    let mut remainder = coeffs[coeffs.len() - 1]; // 最高次项初始化

    // Horner 方法计算商和余数
    for i in (0..coeffs.len() - 1).rev() {
        quotient_coeffs[i] = remainder; // 记录当前商系数
        remainder = coeffs[i] + alpha * remainder;
    }

    DensePolynomial::from_coefficients_vec(quotient_coeffs) // 返回商多项式
}

/// D(X)
// input: gr: g(x) , b: g(x).degree() + 1
pub fn compute_d<F: Field>(
    g: &DensePolynomial<F>,
    b: usize,
) -> DensePolynomial<F> {
    let coeffs = g.coeffs.iter().rev().cloned().collect::<Vec<_>>(); // 反转系数
    let mut new_coeffs = vec![F::zero(); b - g.degree() - 1]; // 填充零系数
    new_coeffs.extend(coeffs); // 追加 G(x) 的系数

    DensePolynomial::from_coefficients_vec(new_coeffs)
}

/// P_u(X)
pub fn pu_poly<F: Field>(u: &[F]) -> DensePolynomial<F> {
    let one = DensePolynomial::from_coefficients_vec(vec![F::one()]); // 常数 1
    let mut poly = one.clone(); // 初始值设为 1

    for (i, &ui) in u.iter().enumerate() {
        let mut coeffs = vec![F::zero(); 1 << i];
        coeffs.push(ui); // 直接 push ui
        let x_pow = DensePolynomial::from_coefficients_vec(coeffs);

        // 计算 (u_i X^(2^i) + 1 - u_i)
        let term = x_pow + DensePolynomial::from_coefficients_vec(vec![F::one() - ui]);

        // 多项式相乘
        poly = polynomial_mul(&poly, &term);
    }

    poly
}

/// 计算 P_u(x) 在某个特定点的值
pub fn pu_evaluate<F: PrimeField>(u: &[F], x: F) -> F {
    let mut result = F::one();

    // 提前计算 x^{2^i} 的值
    let mut x_pow = x;
    for &u_i in u {
        // u_i * x^{2^i} + (1 - u_i)
        let term = u_i * x_pow + (F::one() - u_i);
        result *= term;
        // 准备下一个 2^i 次幂
        x_pow = x_pow.square();
    }

    result
}

fn polynomial_mul<F: Field>(poly1: &DensePolynomial<F>, poly2: &DensePolynomial<F>) -> DensePolynomial<F> {
    let mut result_coeffs = vec![F::zero(); poly1.degree() + poly2.degree() + 1]; // 结果多项式的系数

    // 对每个多项式的系数进行相乘并累加到结果
    for (i, &coeff1) in poly1.coeffs.iter().enumerate() {
        for (j, &coeff2) in poly2.coeffs.iter().enumerate() {
            result_coeffs[i + j] += coeff1 * coeff2; // 累加到对应指数的系数上
        }
    }

    DensePolynomial::from_coefficients_vec(result_coeffs)
}

/// S(X) 分为两部分计算
// input: gr: g(X) , hr: h(X) , u = ( u1 , u2 ) , f(u) = v , alpha and gamma are random
pub fn compute_s<F: PrimeField>(
    g: &DensePolynomial<F>,
    h: &DensePolynomial<F>,
    u1: &[F],
    u2: &[F],
    gamma: F,
    b: usize,
) -> DensePolynomial<F> {
    let pu1 = pu_poly(&u1);
    let pu2 = pu_poly(&u2);
    let s1 = compute_partial_s(&g,&pu1,b);
    let s2 = compute_partial_s(&h,&pu2,b);
    let s_poly = s1 + &s2 * gamma;
    s_poly
}

/// 各部分 S(X) 的计算
pub fn compute_partial_s<F: ark_ff::PrimeField>(
    g1: &DensePolynomial<F>,
    g2: &DensePolynomial<F>,
    b: usize
) -> DensePolynomial<F> {
    // 1. 确保 g1 和 g2 的长度一致，并填充 0 直到 b
    let mut g1_coeffs = g1.coeffs.clone();
    let mut g2_coeffs = g2.coeffs.clone();

    g1_coeffs.resize(b, F::zero()); // 填充到长度 b
    g2_coeffs.resize(b, F::zero());

    let domain = Radix2EvaluationDomain::<F>::new(2 * b).unwrap(); // 选择 2b 个点的 FFT 域

    // 2. 计算 g1(X) 和 g2(1/X) 在 2b 个点上的值
    let g1_evals = domain.fft(&g1_coeffs);
    let g2_inv_evals = domain.fft(&g2_coeffs.iter().rev().cloned().collect::<Vec<_>>()); // g2(1/X)

    // 3. 计算 g1(1/X) 和 g2(X) 在 2b 个点上的值
    let g1_inv_evals = domain.fft(&g1_coeffs.iter().rev().cloned().collect::<Vec<_>>()); // g1(1/X)
    let g2_evals = domain.fft(&g2_coeffs);

    // 4. 计算 LHS = g1(X)g2(1/X) + g1(1/X)g2(X)
    let lhs_evals: Vec<F> = g1_evals.iter().zip(&g2_inv_evals)
        .map(|(a, b)| *a * b) // g1(X) * g2(1/X)
        .zip(g1_inv_evals.iter().zip(&g2_evals))
        .map(|(p1, (a, b))| p1 + *a * b) // + g1(1/X) * g2(X)
        .collect();

    // 5. 逆 FFT 计算系数
    let lhs_poly = DensePolynomial::from_coefficients_vec(domain.ifft(&lhs_evals));

    // 6. 提取 S(X) 的系数（cb, ..., c_{2b-2}）
    let s_coeffs = lhs_poly.coeffs[b..].to_vec();

    DensePolynomial::from_coefficients_vec(s_coeffs) // 返回 S(X)
}

/// H(X)
pub fn compute_big_h<F: PrimeField>(
    f: &DensePolynomial<F>, // 多项式 f(X)
    q: &DensePolynomial<F>, // 多项式 q(X)
    z: F,                   // 常量 z
    b: usize,               // 常量 b
    alpha: F,               // 常量 alpha
    g_z: F                  // 常量 g(z)
) -> DensePolynomial<F>{
    // 计算 z^b - alpha
    let zb_minus_alpha = z.pow([b as u64]) - alpha;

    // 计算 (z^b - alpha) * q(X)
    let term2 = q * &DensePolynomial::from_coefficients_vec(vec![zb_minus_alpha]);

    // 计算 f(X) - (z^b - alpha) * q(X) - g(z)
    let const_poly = DensePolynomial::from_coefficients_vec(vec![g_z]);
    let numerator = f.clone().sub(&term2).sub(&const_poly);
    // 除以 X - z：多项式除法
    let denominator = DensePolynomial::from_coefficients_vec(vec![-z, F::one()]); // X - z

    let quotient = &numerator / &denominator; // 使用多项式除法
    quotient
}


/// 多项式相乘（朴素实现），返回新多项式的系数向量
fn poly_mul<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let mut result = vec![F::zero(); a.len() + b.len() - 1];
    for (i, coeff_a) in a.iter().enumerate() {
        for (j, coeff_b) in b.iter().enumerate() {
            result[i+j] += *coeff_a * *coeff_b;
        }
    }
    result
}

/// 多项式加法，假设两个多项式用 Vec<F> 表示
fn poly_add<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let n = cmp::max(a.len(), b.len());
    // let n = a.len().max();
    let mut result = vec![F::zero(); n];
    for i in 0..n {
        let ai = if i < a.len() { a[i] } else { F::zero() };
        let bi = if i < b.len() { b[i] } else { F::zero() };
        result[i] = ai + bi;
    }
    result
}

/// 利用拉格朗日插值构造多项式 r，输入点 points 与对应的 values
fn lagrange_interpolation<F: Field>(points: &[F], values: &[F]) -> Vec<F> {
    assert_eq!(points.len(), values.len());
    let k = points.len();
    // 初始化结果多项式为零
    let mut result = vec![F::zero(); k];

    for i in 0..k {
        // 分母 d_i = ∏_{j ≠ i} (z_i - z_j)
        let mut d_i = F::one();
        // 分子多项式 N_i(X)，初始化为 [1]
        let mut numerator_poly = vec![F::one()];
        for j in 0..k {
            if i == j { continue; }
            d_i *= points[i] - points[j];
            // 当前因子为 (X - points[j])，表示为 [-points[j], 1]
            let factor = vec![-points[j], F::one()];
            numerator_poly = poly_mul(&numerator_poly, &factor);
        }
        // 计算 d_i 的逆元
        let d_inv = d_i.inverse().expect("Denominator should be nonzero");
        // Lagrange 基多项式 L_i(X) = d_inv * N_i(X)
        let L_i: Vec<F> = numerator_poly.iter().map(|c| *c * d_inv).collect();
        // 将 v_i * L_i(X) 累加到结果中
        let mut term = L_i.iter().map(|c| *c * values[i]).collect::<Vec<F>>();
        result = poly_add(&result, &term);
    }
    result
}

/// r_i(X) ---- BDFG20
pub fn generate_r_i<F: Field>(
    points: &[F],
    values: &[F]
) -> DensePolynomial<F> {
    let coffs = lagrange_interpolation(points, values);
    DensePolynomial::from_coefficients_vec(coffs)

}

pub fn difference<T: Eq + Clone>(t: &[T], s: &[T]) -> Vec<T> {
    t.iter().filter(|x| !s.contains(*x)).cloned().collect()
}

pub fn compute_batch_f<F: PrimeField>(
    fr: Vec<&DensePolynomial<F>>,
    rr: Vec<&DensePolynomial<F>>,
    t: Vec<F>,
    s: Vec<Vec<F>>,
    gamma: F,
) -> DensePolynomial<F> {
    let mut poly = DensePolynomial::from_coefficients_vec(vec![F::zero()]);

    for i in 0..fr.len() {
        // f_i - r_i
        let f_r = fr[i] - rr[i];

        // γ^i · (f_i - r_i)
        let scaled = f_r.mul(gamma.pow([i as u64]));

        // Z_{T \ S_i}(X)
        let t_s = difference(&t, &s[i]);
        let z_s_i = compute_zs(&t_s);

        // 累加多项式
        let term = scaled.mul(&z_s_i);
        poly = poly + term;
    }

    poly
}

pub fn compute_zs<F: Field>(points: &[F]) -> DensePolynomial<F> {
    if points.is_empty() {
        return DensePolynomial::from_coefficients_slice(&[F::one()]);
    }
    let coff = compute_zs_coeffs(points);
    DensePolynomial::from_coefficients_vec(coff)
}

fn compute_zs_coeffs<F: Field>(points: &[F]) -> Vec<F> {
    // 从常数多项式 [1] 开始
    points.iter().fold(vec![F::one()], |poly, z| {
        let mut new_poly = vec![F::zero(); poly.len() + 1];
        for (i, &coeff) in poly.iter().enumerate() {
            // 乘以 X 得到 coeff * X^(i+1)
            new_poly[i + 1] += coeff;
            // 乘以 (-z) 得到 coeff * (-z) * X^i
            new_poly[i] += coeff * (-*z);
        }
        new_poly
    })
}

pub fn compute_batch_w<F: PrimeField>(
    f: DensePolynomial<F>,
    t: Vec<F>,
) -> DensePolynomial<F>{
    let z_t = compute_zs(&t);
    let poly = &f / &z_t;
    poly
}

pub fn compute_batch_l<F: Field>(
    fr: Vec<&DensePolynomial<F>>,
    rr: Vec<&DensePolynomial<F>>,
    w: &DensePolynomial<F>,
    t: Vec<F>,
    s: Vec<Vec<F>>,
    gamma: F,
    z: F,
) -> DensePolynomial<F> {
    let mut poly = DensePolynomial::from_coefficients_vec(vec![F::zero()]);
    for i in 0..fr.len() {
        let eval_r = rr[i].evaluate(&z);
        let f_r = fr[i] - &DensePolynomial::from_coefficients_vec(vec![eval_r]);
        let scaled = f_r.mul(gamma.pow([i as u64]));
        let s_i = &s[i];
        let t_s = difference(&t, s_i);
        let z_s_i = compute_zs(&t_s);
        let eval_z_s_i = z_s_i.evaluate(&z);
        let term = &scaled * eval_z_s_i;
        poly = &poly + &term;
    }
    let z_t = compute_zs(&t);
    let eval_z_t = z_t.evaluate(&z);
    let correction = w.mul(eval_z_t);

    &poly - &correction
}

pub fn compute_batch_w_hat<F: Field>(
    l: DensePolynomial<F>,
    z: F,
) -> DensePolynomial<F> {
    let poly = DensePolynomial::from_coefficients_vec(vec![-z, F::one()]);
    let w_poly = &l / &poly;
    w_poly
}

pub fn multilinear_eval<F: Field>(f: &[F], u: &[F], s: usize) -> F {
    let mut result = F::zero();
    let n = 1 << s; // n = 2^s
    for i in 0..n {
        let eq_i_u = eq(i, u, s);
        result += eq_i_u * f[i];
    }
    result
}