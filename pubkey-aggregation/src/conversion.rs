use ark_ff::{Field, PrimeField};
use ark_ff::{One, Zero};
use circuitlib::constant::{COEFFS_96, WORD_COUNT, WORD_SIZE};
use num_bigint::BigInt;

use crate::constant::NUM_LEN;

pub fn modulo_bigint() -> BigInt {
    BigInt::from(79228162514264337593543950336u128) // 2^96
}

pub fn bigint_to_field_binary<F: Field>(x: &BigInt, len: usize) -> Vec<F> {
    let mut u8s: Vec<u8> = vec![];
    let mut x = x.clone();
    let two = &BigInt::from(2);
    for _ in 0..len {
        let m = &x % two;
        u8s.push(if m == BigInt::one() { 1 } else { 0 });
        x = x / two;
    }
    assert_eq!(BigInt::zero(), x);

    u8s.reverse();

    u8s.iter().map(|x| F::from(*x as u128)).collect::<Vec<F>>()
}

pub fn bigint_to_u8s_be(x: &BigInt, len: usize) -> Vec<u8> {
    let mut ret: Vec<u8> = Vec::with_capacity(len);

    let mut tmp = x.clone();
    let two = BigInt::from(2);
    for _ in 0..len {
        let a = &tmp % &two;
        ret.push(if a == BigInt::zero() { 0 } else { 1 });
        tmp = tmp / &two;
    }
    assert_eq!(BigInt::zero(), tmp);

    ret.reverse();

    return ret;
}

pub fn u8_to_words<F: Field>(u8s: &Vec<u8>) -> Vec<F> {
    assert_eq!(NUM_LEN, u8s.len());

    let mut arr = vec![0, 0, 0];
    arr.extend(u8s);

    let coeffs96 = COEFFS_96;
    let mut words = vec![];

    for i in 0..WORD_COUNT {
        let mut sum = 0u128;
        let start = i * WORD_SIZE;
        for j in 0..WORD_SIZE {
            sum += (arr[start + j] as u128) * coeffs96[j];
        }
        words.push(F::from(sum));
    }

    words
}

pub fn u8_arr_to_field_arr<F: PrimeField>(u8s: Vec<u8>) -> Vec<F> {
    u8s.iter().map(|x| F::from(*x as u128)).collect::<Vec<F>>()
}

pub fn bigint_to_words(x: &BigInt, min_size: usize) -> Vec<BigInt> {
    let mut a = x;
    let mut res: Vec<BigInt> = Vec::with_capacity(min_size);

    let modulo = modulo_bigint();

    let mut tmp: BigInt;
    while a > &BigInt::from(0) {
        let m = a % &modulo;
        tmp = a.clone() / &modulo;
        a = &tmp;
        res.push(m);
    }
    if res.len() < min_size {
        let extra = min_size - res.len();
        for _ in 0..extra {
            res.push(BigInt::from(0));
        }
    }

    res.reverse();
    res
}

pub fn mul_carry_bigint_to_field<F: PrimeField>(carries: &Vec<BigInt>, modulo: &BigInt) -> Vec<F> {
    let mut carries_u8: Vec<F> = Vec::with_capacity(WORD_SIZE * 8);

    for k in 0..carries.len() {
        let x = &carries[k] % modulo;
        let bz = bigint_to_field_binary::<F>(&x, WORD_SIZE);
        assert_eq!(WORD_SIZE, bz.len());
        carries_u8.extend(bz);
    }
    assert_eq!(WORD_SIZE * 8, carries_u8.len());

    carries_u8
}

pub fn carry_extra_bigint_to_field<F: PrimeField>(
    carries: &Vec<BigInt>,
    modulo: &BigInt,
) -> Vec<F> {
    let mut carries_extra: Vec<F> = Vec::with_capacity(2 * carries.len());

    for k in 0..carries.len() {
        let two = BigInt::from(2);
        let mut extra = carries[k].clone() / modulo;
        let mut tmp: Vec<F> = Vec::with_capacity(2);
        for _ in 0..2 {
            let x = (&extra % &two).to_u64_digits().1;
            assert!(1 >= x.len());
            if x.len() == 0 {
                tmp.push(F::ZERO);
            } else {
                tmp.push(F::from(x[0]));
            }
            extra /= &two;
        }
        assert_eq!(BigInt::zero(), extra);
        tmp.reverse();

        carries_extra.extend(tmp);
    }

    carries_extra
}
