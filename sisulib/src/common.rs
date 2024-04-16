use std::ops::{AddAssign, Mul, MulAssign};

use ark_ff::{Field, Fp, FpConfig, PrimeField};
use ark_poly::univariate::{self, DensePolynomial};
use ark_poly::DenseUVPolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};

use crate::field::FpFRI;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IOError: {0}")]
    ChannelIO(String),

    #[error("IOError: {0}")]
    SendChannelIO(std::sync::mpsc::SendError<Vec<u8>>),

    #[error("IOError: {0}")]
    RecvChannelIO(std::sync::mpsc::RecvError),

    #[error("FRIError: {0}")]
    FRI(String),

    #[error("FunctionEvaluationError: {0}")]
    FunctionEvaluation(String),

    #[error("SumcheckError: {0}")]
    Sumcheck(String),

    #[error("GKRError: {0}")]
    GRK(String),

    #[error("VPD: {0}")]
    VPD(String),

    #[error("Sisu: {0}")]
    Sisu(String),

    #[error("KZG: {0}")]
    KZG(String),

    #[error("MerkleTree: {0}")]
    MerkleTree(String),
}

pub fn bin2dec<T: Zero + One + PartialEq, Output: From<u8> + MulAssign + AddAssign>(
    b: &[T],
) -> Output {
    let mut result = Output::from(0);

    for x in b {
        assert!(x.is_zero() || x.is_one());
        result *= Output::from(2);
        if x.is_one() {
            result += Output::from(1);
        }
    }

    result
}

pub fn to_bit_string<T: Zero + One + PartialEq>(b: &[T]) -> String {
    let mut result = String::new();

    for x in b {
        assert!(x.is_zero() || x.is_one());
        if x.is_zero() {
            result += "0";
        } else {
            result += "1";
        }
    }

    result
}

pub fn dec2bin<Input: Into<u128>, Output: From<u8>>(d: Input, size: usize) -> Vec<Output> {
    let mut result = vec![];

    let mut d: u128 = d.into();

    while d > 0 {
        result.push(Output::from((d & 1) as u8));
        d = d >> 1;
    }

    for _ in result.len()..size {
        result.push(Output::from(0));
    }

    result.reverse();
    result
}

pub fn dec2bin_limit<Output: From<u8>>(mut d: u64, limit: usize) -> Vec<Output> {
    let mut result = vec![];

    while d > 0 && result.len() < limit {
        result.push(Output::from((d & 1) as u8));
        d = d >> 1;
    }

    for _ in result.len()..limit {
        result.push(Output::from(0));
    }

    result.reverse();
    result
}

pub fn le2be(le: usize, num_bit: usize) -> usize {
    let mut be = 0;
    for i in 0..num_bit {
        be = (be << 1) + ((le >> i) & 1);
    }

    be
}

pub fn padding_pow_of_two_size<F: From<u8> + Clone>(v: &mut Vec<F>) {
    if v.len() > 0 && !v.len().is_power_of_two() {
        let padding_size = 2usize.pow(v.len().ilog2() + 1) - v.len();
        v.extend(vec![F::from(0u8); padding_size]);
    }
}

pub fn round_to_pow_of_two(mut v: usize) -> usize {
    if !v.is_power_of_two() {
        v = 2usize.pow(v.ilog2() + 1);
    }

    v
}

pub fn combine_point<F: Clone>(v: Vec<&[F]>) -> Vec<F> {
    let mut p = vec![];
    for vi in v {
        p.extend_from_slice(&vi);
    }
    p
}

pub fn combine_integer(v: Vec<(usize, usize)>) -> usize {
    let mut result = 0;
    for (value, size) in v {
        result = (result << size) | value;
    }

    result
}

#[inline]
pub fn split_number(a: &usize, size: usize) -> (usize, usize) {
    let b = a & ((1 << size) - 1);
    let a = a >> size;

    (a, b)
}

/// ilog2_ceil(0)     == 0
/// ilog2_ceil(1)     == 0
/// ilog2_ceil(2^k)   == k   (where k >= 1)
/// ilog2_ceil(2^k+x) == k+1 (where x < 2^k)
pub fn ilog2_ceil(n: usize) -> usize {
    if n == 0 {
        return n;
    }

    if n.is_power_of_two() {
        n.ilog2() as usize
    } else {
        n.ilog2() as usize + 1
    }
}

pub fn divide_dense_by_sparse<F: Field>(
    a: &univariate::DensePolynomial<F>,
    b: &univariate::SparsePolynomial<F>,
) -> (
    univariate::DensePolynomial<F>,
    univariate::DensePolynomial<F>,
) {
    let a = univariate::DenseOrSparsePolynomial::from(a);
    let b = univariate::DenseOrSparsePolynomial::from(b);
    a.divide_with_q_and_r(&b).unwrap()
}

pub fn divide_dense_by_dense<F: Field>(
    a: &univariate::DensePolynomial<F>,
    b: &univariate::DensePolynomial<F>,
) -> (
    univariate::DensePolynomial<F>,
    univariate::DensePolynomial<F>,
) {
    let a = univariate::DenseOrSparsePolynomial::from(a);
    let b = univariate::DenseOrSparsePolynomial::from(b);
    a.divide_with_q_and_r(&b).unwrap()
}

/// Compute Z(x) = (x-x0) * (x-x1) * ... * (x - xk).
/// This function automatically ignores the overlapping points.
pub fn compute_zero_polynomial<F: Field>(points: &[F]) -> DensePolynomial<F> {
    let mut filtered_points = vec![];
    for i in 0..points.len() {
        let mut is_overlap = false;
        for j in i + 1..points.len() {
            if points[i] == points[j] {
                is_overlap = true;
                break;
            }
        }

        if !is_overlap {
            filtered_points.push(points[i].clone());
        }
    }

    let mut zero_polynomial_coeffs = vec![F::ONE];

    for i in 0..filtered_points.len() {
        // tmp = zero_polynomial * (X - xi).
        let mut tmp = vec![F::ZERO];
        tmp.extend_from_slice(&zero_polynomial_coeffs);

        for j in 0..tmp.len() - 1 {
            tmp[j] -= filtered_points[i] * zero_polynomial_coeffs[j];
        }

        zero_polynomial_coeffs = tmp;
    }

    DensePolynomial::from_coefficients_vec(zero_polynomial_coeffs)
}

/// This function automatically ignores the overlapping points.
pub fn lagrange_interpolate<F: Field>(evaluations: &[(F, F)]) -> DensePolynomial<F> {
    let mut filtered_evaluations = vec![];
    for i in 0..evaluations.len() {
        let mut is_overlap = false;
        for j in i + 1..evaluations.len() {
            if evaluations[i].0 == evaluations[j].0 {
                is_overlap = true;
                break;
            }
        }

        if !is_overlap {
            filtered_evaluations.push(evaluations[i].clone());
        }
    }

    let mut result = DensePolynomial::zero();

    for i in 0..filtered_evaluations.len() {
        let mut zero_polynomial_points = vec![];
        let mut divisor = F::ONE;
        for j in 0..filtered_evaluations.len() {
            if filtered_evaluations[i].0 != filtered_evaluations[j].0 {
                zero_polynomial_points.push(filtered_evaluations[j].0);

                // divisor = divisor * (xi - xj).
                divisor *= filtered_evaluations[i].0 - filtered_evaluations[j].0;
            }
        }

        let zero_polynomial = compute_zero_polynomial(&zero_polynomial_points);

        // zero_polynomial = yi * zero_polynomial / divisor.
        let yi_div_divisor = filtered_evaluations[i].1 / divisor;
        let tmp = zero_polynomial.mul(yi_div_divisor);

        result += &tmp;
    }

    result
}

pub fn div_round(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

pub fn convert_field_to_usize(v: &FpFRI) -> usize {
    v.into_bigint().0[0] as usize
}

pub fn convert_vec_fp_to_raw_bigint<P: FpConfig<N>, const N: usize>(v: &[Fp<P, N>]) -> String {
    let mut result = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            result += ", ";
        }
        result += &format!("Fp({:?}, PhantomData)", x.0);
    }

    result += "]";

    result
}

pub fn convert_vec_field_to_string<F: Field>(v: &[F]) -> Vec<String> {
    let mut result = vec![];
    for x in v {
        result.push(convert_field_to_string(x));
    }

    result
}

pub fn convert_field_to_string<F: Field>(v: &F) -> String {
    if v == &F::ZERO {
        String::from("0")
    } else {
        v.to_string()
    }
}

pub fn serialize<T: CanonicalSerialize>(t: &T) -> Vec<u8> {
    let mut s = vec![];
    t.serialize_uncompressed(&mut s).unwrap();

    s
}

pub fn deserialize<T: CanonicalDeserialize>(msg: &[u8]) -> T {
    T::deserialize_uncompressed(msg).unwrap()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::field::{Fp97, Fr255};
    use ark_poly::Polynomial;

    #[test]
    fn test_le2be() {
        assert_eq!(le2be(4, 3), 1);
        assert_eq!(le2be(7, 4), 14);
    }

    #[test]
    fn test_combine_integer() {
        assert_eq!(
            combine_integer(vec![(0b100, 3), (0b01, 2), (0b11, 2)]),
            0b1000111
        );
    }

    #[test]
    fn test_x() {
        let x = Fr255::from(12333333);
        println!("{:?}", x);
    }

    #[test]
    fn test_split_number() {
        for _ in 0..10000000 {
            split_number(&20, 10);
        }
    }

    type FpTest = Fp97;

    #[test]
    fn test_lagrange() {
        let evaluations = vec![
            (FpTest::from(0u8), FpTest::from(4u8)),
            (FpTest::from(1u8), FpTest::from(6u8)),
            (FpTest::from(2u8), FpTest::from(8u8)),
            (FpTest::from(3u8), FpTest::from(10u8)),
            (FpTest::from(3u8), FpTest::from(10u8)),
        ];

        let poly = lagrange_interpolate(&evaluations);
        println!("{:?}", poly);

        for i in 0..evaluations.len() {
            assert_eq!(poly.evaluate(&evaluations[i].0), evaluations[i].1);
        }
    }
}
