use circuitlib::constant::WORD_COUNT;
use std::collections::HashMap;

use crate::circuit_util::CircuitUtil;
use crate::constant::NUM_LEN;
use crate::conversion::bigint_to_field_binary;
use crate::conversion::bigint_to_u8s_be;
use crate::conversion::bigint_to_words;
use crate::conversion::carry_extra_bigint_to_field;
use crate::conversion::modulo_bigint;
use crate::conversion::mul_carry_bigint_to_field;
use crate::conversion::u8_arr_to_field_arr;
use crate::conversion::u8_to_words;
use ark_bls12_381::G1Affine;
use ark_ff::Field;
use ark_ff::One;
use ark_ff::PrimeField;
use ark_ff::Zero;
use num_bigint::BigInt;
use num_bigint::BigUint;
use sisulib::field::FpBLS12_381;
use std::str::FromStr;

// Formula for adding 2 points (P, Q) to a new point R.
// x_R = ((y_Q - y_P) / (x_Q - x_P)) ^ 2 - x_P - x_Q
// y_R = (y_Q - y_P) / (x_Q - x_P) * (x_P - x_R) - y_P
//
// In the circuit we will calculate:
// diff_y_QP = y_Q - y_P
// diff_x_QP = x_Q - x_P
// div_y_x = diff_y_QP / diff_x_QP
// div_y_x_2 = div_y_x * div_y_x
// sum_x_PQ = x_P + x_Q
// x_R = div_y_x_2 - sum_x_PQ
//
// diff_x_PR = x_P - x_R
// div_y_x_PR = div_y_x * diff_x_PR
// y_R = div_y_x_PR - y_P
//
// We will need the following vars to be included in the input.
// x_P, y_P, x_Q, y_Q, x_R, y_R, diff_y_QP, diff_x_QP, div_y_x, div_y_x_2, sum_x_PQ, diff_x_PR,
// div_y_x_PR

fn p381() -> BigInt {
    BigInt::from_str("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787").unwrap()
}

fn bigint_from_flag(x1: &BigInt, x2: &BigInt, flag: u8) -> BigInt {
    if flag == 1 {
        x2.clone()
    } else {
        x1.clone()
    }
}

fn find_inverse(x: &BigInt) -> BigInt {
    let x_field = FpBLS12_381::from_str(&x.to_string()).unwrap();
    let inv_field = FpBLS12_381::ONE / x_field;
    BigInt::from_str(&inv_field.to_string()).unwrap()
}

fn assign_lt<F: PrimeField>(
    result: &mut HashMap<String, Vec<F>>,
    a_label: &str,
    b_label: &str,
    a: &BigInt,
    b: &BigInt,
) {
    let a_bin = bigint_to_u8s_be(a, NUM_LEN);
    let b_bin = bigint_to_u8s_be(b, NUM_LEN);

    let n = NUM_LEN;
    assert_eq!(NUM_LEN, a_bin.len());
    assert_eq!(NUM_LEN, b_bin.len());

    // lt
    let mut lt: Vec<F> = vec![F::ZERO; n];
    for j in 0..n {
        if a_bin[j] < b_bin[j] {
            lt[j] = F::ONE;
        }
    }

    // eq
    let mut eq: Vec<F> = vec![F::ZERO; n];
    for j in 0..n {
        if a_bin[j] == b_bin[j] {
            eq[j] = F::ONE;
        }
    }

    // ors[i] holds (lt[0] || (eq[0] && lt[1]) .. || (eq[i - 1] && .. && lt[i]))
    // ands[i] holds (eq[0] && .. && lt[i])
    // eq_ands[i] holds (eq[0] && .. && eq[i])
    let mut ands: Vec<F> = vec![F::ZERO; n];
    let mut eq_ands: Vec<F> = vec![F::ZERO; n];
    let mut ors: Vec<F> = vec![F::ZERO; n];
    for j in 1..n {
        if j == 1 {
            ands[j] = eq[j - 1] * lt[j];
            eq_ands[j] = eq[j - 1] * eq[j];
            ors[j] = lt[j - 1] + ands[j] - lt[j - 1] * ands[j];
        } else {
            ands[j] = eq_ands[j - 1] * lt[j];
            eq_ands[j] = eq_ands[j - 1] * eq[j];
            ors[j] = ors[j - 1] + ands[j] - ors[j - 1] * ands[j];
        }
    }

    let cmp_label = CircuitUtil::cmp_lt_label(a_label, b_label);
    result.insert(format!("{cmp_label}_ands"), ands);
    result.insert(format!("{cmp_label}_eq_ands"), eq_ands);
    result.insert(format!("{cmp_label}_ors"), ors);
    result.insert(format!("{cmp_label}_lt0"), vec![lt[0]]);
    result.insert(format!("{cmp_label}_eq0"), vec![eq[0]]);
}

fn compress_form_to_bigint(xs: &Vec<u8>) -> String {
    assert_eq!(384, xs.len());
    let mut x: Vec<u8> = vec![0, 0, 0];
    x.extend_from_slice(&xs[3..]);
    assert_eq!(384, x.len());

    let mut num: Vec<u8> = vec![];
    for i in 0..(384 / 8) {
        let mut sum = 0u8;
        let mut coeff = 1u8;
        for j in (0..8).rev() {
            sum += x[i * 8 + j] * coeff;
            if j > 0 {
                coeff *= 2;
            }
        }
        // println!("sum = {}", sum);
        num.push(sum);
    }

    BigUint::from_bytes_be(&num).to_string()
}

fn assign_eq<F: PrimeField>(
    result: &mut HashMap<String, Vec<F>>,
    a_label: &str,
    b_label: &str,
    a: &BigInt,
    b: &BigInt,
) {
    let a_bin = bigint_to_u8s_be(a, NUM_LEN);
    let b_bin = bigint_to_u8s_be(b, NUM_LEN);

    let a_words_field = u8_to_words::<F>(&a_bin);
    let b_words_field = u8_to_words::<F>(&b_bin);

    // Add the inverse
    let mut inv_values: Vec<F> = vec![];
    for i in 0..WORD_COUNT {
        let diff = a_words_field[i] - b_words_field[i];
        if diff == F::ZERO {
            inv_values.push(F::ZERO);
        } else {
            inv_values.push(F::ONE / diff);
        }
        assert_eq!(F::ZERO, -diff * inv_values[inv_values.len() - 1] + F::ONE);
    }

    let eq_label = CircuitUtil::eq_label(a_label, b_label);
    result.insert(format!("{eq_label}_inv"), inv_values);
}

pub fn assign_mul<F: PrimeField>(
    result: &mut HashMap<String, Vec<F>>,
    label_c: &str,
    a: &BigInt,
    b: &BigInt,
) {
    let p381 = &p381();
    let ab = &(a * b);
    let d = &(ab / p381);
    let c = &(ab - d * p381);
    let modulo = &modulo_bigint();

    // Calculate the carries.
    let words_a = bigint_to_words(a, WORD_COUNT);
    let words_b = bigint_to_words(b, WORD_COUNT);
    let words_d = bigint_to_words(d, WORD_COUNT);
    let words_c = bigint_to_words(c, WORD_COUNT);
    let words_p381 = bigint_to_words(&p381, WORD_COUNT);
    let carry_limit = modulo * WORD_COUNT;

    let mut left_carries: Vec<BigInt> = vec![BigInt::from(0); WORD_COUNT * 2];
    let mut right_carries: Vec<BigInt> = vec![BigInt::from(0); WORD_COUNT * 2];
    for k in 0..2 * WORD_COUNT - 1 {
        let mut left_sum: BigInt = BigInt::from(0);
        let mut right_sum: BigInt = BigInt::from(0);
        for i in 0..k + 1 {
            let j = k - i;
            if i > WORD_COUNT - 1 || j > WORD_COUNT - 1 {
                continue;
            }

            left_sum += &words_a[WORD_COUNT - 1 - i] * &words_b[WORD_COUNT - 1 - j];
            right_sum += &words_d[WORD_COUNT - 1 - i] * &words_p381[WORD_COUNT - 1 - j];
        }

        let carry_index = 2 * WORD_COUNT - 1 - k;

        if k > 0 {
            left_sum += &left_carries[carry_index + 1];
            right_sum += &right_carries[carry_index + 1];
        }

        if k < WORD_COUNT {
            right_sum += &words_c[WORD_COUNT - 1 - k];
        }

        assert_eq!(&left_sum % modulo, &right_sum % modulo);

        left_carries[carry_index] = &left_sum / modulo;
        right_carries[carry_index] = &right_sum / modulo;
        assert!(left_carries[carry_index] < carry_limit);
        assert!(right_carries[carry_index] < carry_limit);
    }
    assert_eq!(left_carries[1], right_carries[1]);
    assert!(&left_carries[0] < modulo);
    assert!(&right_carries[0] < modulo);

    let left_carries_field = mul_carry_bigint_to_field(&left_carries, modulo);
    let right_carries_field = mul_carry_bigint_to_field(&right_carries, modulo);
    let left_carries_extra = carry_extra_bigint_to_field(&left_carries, modulo);
    let right_carries_extra = carry_extra_bigint_to_field(&right_carries, modulo);

    result.insert(
        format!("{label_c}_mul_d"),
        bigint_to_field_binary(&d, NUM_LEN),
    );
    result.insert(format!("{label_c}_mul_carry_left"), left_carries_field);
    result.insert(format!("{label_c}_mul_carry_right"), right_carries_field);
    result.insert(
        format!("{label_c}_mul_carry_left_extra"),
        left_carries_extra,
    );
    result.insert(
        format!("{label_c}_mul_carry_right_extra"),
        right_carries_extra,
    );
}

pub fn assign_add<F: PrimeField>(
    result: &mut HashMap<String, Vec<F>>,
    label_c: &str,
    a: &BigInt,
    b: &BigInt,
) {
    let p381 = &p381();

    let ab = a + b;
    let d = &(&ab / p381);
    assert!(*d <= BigInt::from(1));
    let c = &ab - d * p381;

    let mut left_carries: Vec<BigInt> = Vec::with_capacity(4);
    let mut right_carries: Vec<BigInt> = Vec::with_capacity(4);
    let words_a = bigint_to_words(&a, 4);
    let words_b = bigint_to_words(&b, 4);
    let words_c = bigint_to_words(&c, 4);
    let words_p381 = bigint_to_words(&p381, 4);
    let zero = BigInt::from(0);
    let modulo = &modulo_bigint();

    for k in 0..4 {
        let mut left: BigInt;
        let mut right: BigInt;
        left = &words_a[3 - k] + &words_b[3 - k];
        right = &words_p381[3 - k] * d + &words_c[3 - k];

        if k > 0 {
            left += &left_carries[k - 1];
            right += &right_carries[k - 1];
        }

        if left == right {
            left_carries.push(BigInt::zero());
            right_carries.push(BigInt::zero());
        } else if left < right {
            left_carries.push(BigInt::zero());
            right_carries.push(BigInt::one());
        } else {
            // left > right
            left_carries.push(BigInt::one());
            right_carries.push(BigInt::zero());
        }

        // double check again
        let mut sum: BigInt;
        sum = &words_a[3 - k] + &words_b[3 - k] - &left_carries[k] * modulo
            + &right_carries[k] * modulo
            - d * &words_p381[3 - k]
            - &words_c[3 - k];

        if k > 0 {
            sum += &left_carries[k - 1];
            sum -= &right_carries[k - 1];
        }

        assert_eq!(zero, sum);
    }
    left_carries.reverse();
    right_carries.reverse();
    for i in 0..WORD_COUNT {
        assert!(left_carries[i] <= BigInt::one());
        assert!(right_carries[i] <= BigInt::one());
    }

    result.insert(
        format!("{label_c}_add_d"),
        vec![if *d == BigInt::one() { F::ONE } else { F::ZERO }],
    );

    let left_carries_field = left_carries
        .iter()
        .map(|x| {
            assert!(*x == BigInt::zero() || *x == BigInt::one());
            if *x == BigInt::zero() {
                F::ZERO
            } else {
                F::ONE
            }
        })
        .collect::<Vec<F>>();
    let right_carries_field = right_carries
        .iter()
        .map(|x| {
            assert!(*x == BigInt::zero() || *x == BigInt::one());
            if *x == BigInt::zero() {
                F::ZERO
            } else {
                F::ONE
            }
        })
        .collect::<Vec<F>>();

    result.insert(format!("{label_c}_add_carry_left"), left_carries_field);
    result.insert(format!("{label_c}_add_carry_right"), right_carries_field);
}

pub fn assign_input<F: PrimeField>(
    p_x_u8: &Vec<u8>,
    q_x_u8: &Vec<u8>,
    r_x_u8: &Vec<u8>,
) -> HashMap<String, Vec<F>> {
    let mut result: HashMap<String, Vec<F>> = HashMap::default();

    // px
    let p_x_field = FpBLS12_381::from_str(&compress_form_to_bigint(p_x_u8)).unwrap();
    let p_x = &BigInt::from_str(&p_x_field.to_string()).unwrap();

    let p_ys_field = G1Affine::get_ys_from_x_unchecked(p_x_field).unwrap();
    let p_y1 = &BigInt::from_str(&p_ys_field.0.to_string()).unwrap();
    let p_y2 = &BigInt::from_str(&p_ys_field.1.to_string()).unwrap();
    let p_y = &bigint_from_flag(p_y1, p_y2, p_x_u8[2]);

    // qx
    let q_x_field = FpBLS12_381::from_str(&compress_form_to_bigint(q_x_u8)).unwrap();
    let q_x = &BigInt::from_str(&q_x_field.to_string()).unwrap();

    let q_ys_field = G1Affine::get_ys_from_x_unchecked(q_x_field).unwrap();
    let q_y1 = &BigInt::from_str(&q_ys_field.0.to_string()).unwrap();
    let q_y2 = &BigInt::from_str(&q_ys_field.1.to_string()).unwrap();
    let q_y = &bigint_from_flag(&q_y1, &q_y2, q_x_u8[2]);

    // rx
    let r_x_field = FpBLS12_381::from_str(&compress_form_to_bigint(r_x_u8)).unwrap();
    let r_x = &BigInt::from_str(&r_x_field.to_string()).unwrap();

    let r_ys_field = G1Affine::get_ys_from_x_unchecked(r_x_field).unwrap();
    let r_y1 = &BigInt::from_str(&r_ys_field.0.to_string()).unwrap();
    let r_y2 = &BigInt::from_str(&r_ys_field.1.to_string()).unwrap();
    let r_y = &bigint_from_flag(r_y1, r_y2, r_x_u8[2]);

    // Add in out
    let labels = [
        "p_x", "p_y1", "p_y2", "p_y", "q_x", "q_y1", "q_y2", "q_y", "r_x", "r_y1", "r_y2", "r_y",
    ];
    let values = [
        p_x, p_y1, p_y2, p_y, q_x, q_y1, q_y2, q_y, r_x, r_y1, r_y2, r_y,
    ];
    assert_eq!(labels.len(), values.len());
    for i in 0..labels.len() {
        result.insert(
            labels[i].to_string(),
            bigint_to_field_binary(&values[i], NUM_LEN),
        );
    }

    let p381 = &p381();
    let diff_y_qp = &((q_y - p_y + p381) % p381);
    let diff_x_qp = &((q_x - p_x + p381) % p381);
    // let div_y_x = &(diff_y_qp / diff_x_qp % p381);
    let div_y_x = &(diff_y_qp * find_inverse(diff_x_qp) % p381);
    let div_y_x_2 = &(div_y_x * div_y_x % p381);
    let div_y_x_2_minus_p_x = &((div_y_x_2 - p_x + p381) % p381);
    assert_eq!(r_x.clone(), (div_y_x_2_minus_p_x - q_x + p381) % p381);
    let diff_x_pr = &((p_x - r_x + p381) % p381);
    let mul_y_x_pr = &(div_y_x * diff_x_pr % p381);
    assert_eq!(r_y.clone(), (mul_y_x_pr - p_y + p381) % p381);

    // assign other words
    result.insert(
        "diff_y_qp".to_string(),
        bigint_to_field_binary(&diff_y_qp, NUM_LEN),
    );
    result.insert(
        "diff_x_qp".to_string(),
        bigint_to_field_binary(&diff_x_qp, NUM_LEN),
    );
    result.insert(
        "div_y_x".to_string(),
        bigint_to_field_binary(&div_y_x, NUM_LEN),
    );
    result.insert(
        "div_y_x_2".to_string(),
        bigint_to_field_binary(&div_y_x_2, NUM_LEN),
    );
    result.insert(
        "diff_x_pr".to_string(),
        bigint_to_field_binary(&diff_x_pr, NUM_LEN),
    );
    result.insert(
        "mul_y_x_pr".to_string(),
        bigint_to_field_binary(&mul_y_x_pr, NUM_LEN),
    );
    result.insert(
        "div_y_x_2_minus_p_x".to_string(),
        bigint_to_field_binary(&div_y_x_2_minus_p_x, NUM_LEN),
    );

    // multiplication
    assign_mul::<F>(&mut result, "mul_y_x_pr", &div_y_x, &diff_x_pr);
    assign_mul::<F>(&mut result, "diff_y_qp", &div_y_x, &diff_x_qp);
    assign_mul::<F>(&mut result, "div_y_x_2", &div_y_x, &div_y_x);

    // add
    assign_add::<F>(&mut result, "q_x", &p_x, &diff_x_qp);
    assign_add::<F>(&mut result, "p_x", &r_x, &diff_x_pr);
    assign_add::<F>(&mut result, "mul_y_x_pr", &p_y, &r_y);
    assign_add::<F>(&mut result, "q_y", &p_y, &diff_y_qp);
    assign_add::<F>(&mut result, "div_y_x_2_minus_p_x", &r_x, &q_x);
    assign_add::<F>(&mut result, "div_y_x_2", &p_x, &div_y_x_2_minus_p_x);

    // input1
    result.insert("input1".to_string(), vec![F::ONE]);

    // flags
    result.insert(
        "p_x_flags".to_string(),
        u8_arr_to_field_arr(p_x_u8[..3].to_vec()),
    );
    result.insert(
        "q_x_flags".to_string(),
        u8_arr_to_field_arr(q_x_u8[..3].to_vec()),
    );
    result.insert(
        "r_x_flags".to_string(),
        u8_arr_to_field_arr(r_x_u8[..3].to_vec()),
    );

    // fp
    result.insert(
        "fp".to_string(),
        u8_arr_to_field_arr(bigint_to_u8s_be(p381, NUM_LEN)),
    );

    // assign less than values
    assign_lt::<F>(&mut result, "r_x", "fp", &r_x, &p381);
    assign_lt::<F>(&mut result, "p_y1", "p_y2", &p_y1, &p_y2);
    assign_lt::<F>(&mut result, "q_y1", "q_y2", &q_y1, &q_y2);
    assign_lt::<F>(&mut result, "r_y1", "r_y2", &r_y1, &r_y2);

    assign_eq::<F>(&mut result, "p_x", "q_x", &p_x, &q_x);

    // input0 & input1
    result.insert("input0".to_string(), vec![F::ZERO]);
    result.insert("input1".to_string(), vec![F::ONE]);

    result
}
