use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::ToPrimitive;
use sha2::Sha256;
use sisulib::{
    common::{deserialize, serialize},
    field::{FpFRI, FpSisu},
};
use std::fmt::Debug;

use crate::hash::{SisuHasher, SisuMimc};

pub trait FiatShamirEngine<F: Field>: Send + Sync {
    fn freeze(&self) -> Self;
    fn set_seed(&mut self, seed: F);
    fn inherit_seed(&mut self);
    fn ignore_seed(&mut self);
    fn begin_protocol(&mut self);

    fn reduce_g(&mut self, v: &[F]) -> F;
    fn reduce_and_hash_to_field(&mut self, current_g: &[F]) -> F {
        let g = self.reduce_g(current_g);
        self.hash_to_field(&g)
    }
    fn reduce_and_hash_to_fields(&mut self, current_g: &[F], count: usize) -> Vec<F> {
        let g = self.reduce_g(current_g);
        self.hash_to_fields(&g, count)
    }
    fn reduce_and_hash_to_u64(&mut self, current_g: &[F], modulus: u64) -> u64 {
        let g = self.reduce_g(current_g);
        self.hash_to_u64(&g, modulus)
    }
    fn reduce_and_set_seed(&mut self, seed: &[F]) {
        let g = self.reduce_g(seed);
        self.set_seed(g)
    }

    fn hash_to_field(&mut self, current_g: &F) -> F;
    fn hash_to_fields(&mut self, current_g: &F, count: usize) -> Vec<F> {
        let mut result = vec![F::ZERO; count];
        for i in 0..count {
            if i == 0 {
                result[i] = self.hash_to_field(current_g);
            } else {
                result[i] = self.hash_to_field(&result[i - 1]);
            }
        }

        result
    }

    fn hash_to_u64(&mut self, current_g: &F, modulus: u64) -> u64 {
        hash_to_u64(&self.hash_to_field(current_g)) % modulus
    }
}

#[derive(Default)]
pub struct DummyFiatShamirEngine;

impl<F: Field> FiatShamirEngine<F> for DummyFiatShamirEngine {
    fn freeze(&self) -> Self {
        Self {}
    }

    fn inherit_seed(&mut self) {
        panic!("dummy fiat shamir")
    }

    fn set_seed(&mut self, _: F) {
        panic!("dummy fiat shamir")
    }

    fn ignore_seed(&mut self) {
        panic!("dummy fiat shamir")
    }

    fn reduce_g(&mut self, _: &[F]) -> F {
        panic!("dummy fiat shamir")
    }

    fn begin_protocol(&mut self) {
        panic!("dummy fiat shamir")
    }

    fn hash_to_field(&mut self, _: &F) -> F {
        panic!("dummy fiat shamir")
    }
}

#[derive(Default)]
pub struct DefaultFiatShamirEngine<F: Field, H: SisuHasher<F>> {
    pending_seed: Option<F>,
    next_seed: Option<F>,
    round: u64,
    previous_r: F,
    hasher: H,
}

impl<F: Field> DefaultFiatShamirEngine<F, Sha256> {
    pub fn default_sha256() -> Self {
        Self {
            pending_seed: None,
            next_seed: None,
            round: 0,
            previous_r: F::ZERO,
            hasher: Sha256::default(),
        }
    }
}

impl DefaultFiatShamirEngine<FpSisu, SisuMimc<FpSisu>> {
    pub fn default_fpsisu() -> Self {
        Self {
            pending_seed: None,
            next_seed: None,
            round: 0,
            previous_r: FpSisu::ZERO,
            hasher: SisuMimc::default(),
        }
    }
}

impl DefaultFiatShamirEngine<FpFRI, SisuMimc<FpFRI>> {
    pub fn default_fpfri() -> Self {
        Self {
            pending_seed: None,
            next_seed: None,
            round: 0,
            previous_r: FpFRI::ZERO,
            hasher: SisuMimc::default(),
        }
    }
}

impl<F: Field, H: SisuHasher<F>> Debug for DefaultFiatShamirEngine<F, H> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Result::Ok(())
    }
}

unsafe impl<F: Field, H: SisuHasher<F>> Send for DefaultFiatShamirEngine<F, H> {}

unsafe impl<F: Field, H: SisuHasher<F>> Sync for DefaultFiatShamirEngine<F, H> {}

impl<F: Field, H: SisuHasher<F>> FiatShamirEngine<F> for DefaultFiatShamirEngine<F, H> {
    fn freeze(&self) -> Self {
        Self {
            pending_seed: self.pending_seed,
            next_seed: self.next_seed,
            round: self.round,
            previous_r: self.previous_r,
            hasher: self.hasher.clone(),
        }
    }

    fn inherit_seed(&mut self) {
        assert!(self.next_seed.is_some(), "No seed found");
        self.pending_seed = self.next_seed;
        self.next_seed = None;
    }

    fn set_seed(&mut self, seed: F) {
        self.pending_seed = Some(seed);
    }

    fn ignore_seed(&mut self) {
        assert!(
            self.pending_seed.is_none(),
            "There is a seed not yet comsumed"
        );
        assert!(
            self.next_seed.is_none(),
            "There is a seed is preparing for consuming, do not set again"
        );

        self.pending_seed = None;
    }

    fn begin_protocol(&mut self) {
        assert!(
            self.pending_seed.is_some(),
            "Must provide seed value before beginning protocol, please call set_seed()"
        );

        self.next_seed = self.pending_seed;
        self.pending_seed = None;
    }

    fn reduce_g(&mut self, v: &[F]) -> F {
        self.hasher.hash_slice(v)
    }

    fn hash_to_field(&mut self, current_g: &F) -> F {
        assert!(
            self.pending_seed.is_none(),
            "a seed is waiting for consume, please call begin_protocol()"
        );

        let data = match self.next_seed {
            Some(seed) => self.hasher.hash_slice(&[seed, current_g.clone()]),
            None => current_g.clone(),
        };

        let r = self
            .hasher
            .hash_slice(&[F::from(self.round), data, self.previous_r]);

        self.previous_r = r;
        self.round += 1;
        self.next_seed = None;

        r
    }
}

#[derive(Debug, Default, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Transcript {
    inner: Vec<Vec<u8>>,
}

impl Transcript {
    pub fn new() -> Self {
        Self { inner: vec![] }
    }
    pub fn from_vec<T: CanonicalSerialize>(v: Vec<&T>) -> Self {
        let mut transcript = Self::default();
        for t in v {
            transcript.serialize_and_push(t);
        }

        transcript
    }

    pub fn push(&mut self, data: Vec<u8>) {
        self.inner.push(data);
    }

    pub fn serialize_and_push<T: CanonicalSerialize>(&mut self, t: &T) {
        self.push(serialize(t));
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn into_iter<'a>(&'a self) -> TranscriptIter<'a> {
        TranscriptIter {
            transcript: self,
            current: 0,
        }
    }

    pub fn into_vec_iter<'a>(v: &'a [Self]) -> Vec<TranscriptIter<'a>> {
        let mut result = vec![];
        for i in 0..v.len() {
            result.push(TranscriptIter {
                transcript: &v[i],
                current: 0,
            })
        }

        result
    }
}

#[derive(Clone)]
pub struct TranscriptIter<'a> {
    transcript: &'a Transcript,
    current: usize,
}

impl<'a> TranscriptIter<'a> {
    pub fn reset(&mut self) {
        self.current = 0;
    }

    pub fn pop(&mut self) -> &[u8] {
        let result = &self.transcript.inner[self.current];
        self.current += 1;
        result
    }

    pub fn pop_and_deserialize<T: CanonicalDeserialize>(&mut self) -> T {
        deserialize::<T>(self.pop())
    }

    pub fn pop_and_deserialize_vec<T: CanonicalDeserialize>(v: &mut [Self]) -> Vec<T> {
        let mut result = vec![];
        for i in 0..v.len() {
            result.push(deserialize::<T>(v[i].pop()));
        }

        result
    }

    pub fn len(&self) -> usize {
        self.transcript.len()
    }

    pub fn remaining_len(&self) -> usize {
        self.len() - self.current
    }
}

fn hash_to_u64<F: Field>(v: &F) -> u64 {
    let x: BigUint = v
        .to_base_prime_field_elements()
        .collect::<Vec<F::BasePrimeField>>()[0]
        .into();

    x.mod_floor(&BigUint::from_bytes_be(&[1, 0, 0, 0, 0, 0, 0, 0, 0]))
        .to_u64()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use sisulib::field::FpFRI;

    use crate::fiat_shamir::FiatShamirEngine;
    use crate::hash::SisuMimc;

    use super::{serialize, DefaultFiatShamirEngine};

    #[test]
    fn test_serialize() {
        let t = FpFRI::from(300u128);
        let bz = serialize(&t);
        assert_eq!(vec![44, 1, 0, 0, 0, 0, 0, 0], bz);
    }

    #[test]
    fn test_fiat_shamir() {
        let mut fiat_shamir = DefaultFiatShamirEngine::<FpFRI, SisuMimc<FpFRI>>::default();

        // One to One
        let input0 = FpFRI::from(3u64);
        let output0 = fiat_shamir.hash_to_field(&input0);
        assert_eq!(output0, FpFRI::from(19683u64));

        // One to Many
        let input1 = FpFRI::from(3u64);
        let output1 = fiat_shamir.hash_to_fields(&input1, 3);
        assert_eq!(output1[0], FpFRI::from(7700224345723u64));
        assert_eq!(output1[1], FpFRI::from(14386208105691471149u64));
        assert_eq!(output1[2], FpFRI::from(5883528248509324381u64));

        // Many to One
        let input2 = vec![FpFRI::from(3u64), FpFRI::from(4u64), FpFRI::from(5u64)];
        let output2 = fiat_shamir.reduce_and_hash_to_field(&input2);
        assert_eq!(output2, FpFRI::from(7985082835725426031u64));

        // Many to Many
        let input3 = vec![FpFRI::from(3u64), FpFRI::from(4u64), FpFRI::from(5u64)];
        let output3 = fiat_shamir.reduce_and_hash_to_fields(&input3, 2);
        assert_eq!(output3[0], FpFRI::from(6678922713716911253u64));
        assert_eq!(output3[1], FpFRI::from(6470906768176837196u64));

        // One to One with Seed
        let seed = FpFRI::from(7u64);
        let input4 = FpFRI::from(3u64);
        fiat_shamir.set_seed(seed);
        let output4 = fiat_shamir.hash_to_field(&input4);
        assert_eq!(output4, FpFRI::from(10364119421809708578u64));
    }
}
