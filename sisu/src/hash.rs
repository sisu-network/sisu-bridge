use crate::mimc_k::{self, K_BN254, K_FRI};
use ark_crypto_primitives::{
    crh::{poseidon, CRHScheme, TwoToOneCRHScheme},
    sponge::{self, Absorb},
};
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;
use sha2::Digest;
use sisulib::{
    common::serialize,
    field::{FpFRI, FrBN254},
};
use std::ops::MulAssign;

pub trait Hasher<Leaf>: Default {
    type InnerLeaf: Clone + PartialEq;

    fn hash_slice(&mut self, input: &[Leaf]) -> Self::InnerLeaf;
    fn hash_two(&mut self, a: &Self::InnerLeaf, b: &Self::InnerLeaf) -> Self::InnerLeaf;
}

impl<Leaf: CanonicalSerialize> Hasher<Leaf> for sha2::Sha256 {
    type InnerLeaf = Vec<u8>;

    fn hash_slice(&mut self, input: &[Leaf]) -> Vec<u8> {
        for inp in input {
            self.update(serialize(inp));
        }

        self.finalize_reset().to_vec()
    }

    fn hash_two(&mut self, a: &Self::InnerLeaf, b: &Self::InnerLeaf) -> Self::InnerLeaf {
        self.update(a);
        self.update(b);
        self.finalize_reset().to_vec()
    }
}

pub trait SisuHasher<F: Field>: Default + Clone + Send + Sync {
    fn hash_slice(&mut self, input: &[F]) -> F;
    fn hash_two(&mut self, a: &F, b: &F) -> F;
}

#[derive(Default, Clone)]
pub struct DummyHash;

impl<F: Field> SisuHasher<F> for DummyHash {
    fn hash_slice(&mut self, _: &[F]) -> F {
        panic!("not implemented");
    }

    fn hash_two(&mut self, _: &F, _: &F) -> F {
        panic!("not implemented");
    }
}

impl<F: Field> SisuHasher<F> for sha2::Sha256 {
    fn hash_slice(&mut self, input: &[F]) -> F {
        if input.len() == 1 {
            return input[0].clone();
        }

        for inp in input {
            self.update(&serialize(inp));
        }

        let result = self.finalize_reset();
        F::from_base_prime_field(<F as Field>::BasePrimeField::from_be_bytes_mod_order(
            &result,
        ))
    }

    fn hash_two(&mut self, a: &F, b: &F) -> F {
        self.update(serialize(a));
        self.update(serialize(b));
        let result = self.finalize_reset();
        F::from_base_prime_field(<F as Field>::BasePrimeField::from_be_bytes_mod_order(
            &result,
        ))
    }
}

impl<F: Field> SisuHasher<F> for sha2::Sha224 {
    fn hash_slice(&mut self, input: &[F]) -> F {
        if input.len() == 1 {
            return input[0].clone();
        }

        for inp in input {
            self.update(&serialize(inp));
        }

        let result = self.finalize_reset();
        F::from_base_prime_field(<F as Field>::BasePrimeField::from_be_bytes_mod_order(
            &result,
        ))
    }

    fn hash_two(&mut self, a: &F, b: &F) -> F {
        self.update(serialize(a));
        self.update(serialize(b));
        let result = self.finalize_reset();
        F::from_base_prime_field(<F as Field>::BasePrimeField::from_be_bytes_mod_order(
            &result,
        ))
    }
}

#[derive(Clone)]
pub struct Poseidon<F: PrimeField> {
    parameters: sponge::poseidon::PoseidonConfig<F>,
}

impl<F: PrimeField> Default for Poseidon<F> {
    fn default() -> Self {
        let mut mds = vec![vec![]; 3];
        for i in 0..3 {
            for j in 0..3 {
                mds[i].push(F::from((i * j + 1) as u64));
            }
        }

        let mut ark = vec![vec![]; 8 + 24];
        for i in 0..8 + 24 {
            for j in 0..3 {
                ark[i].push(F::from((i * j + 1) as u64));
            }
        }

        let mut test_a = Vec::new();
        let mut test_b = Vec::new();
        for i in 0..3 {
            test_a.push(F::from((i + 1) as u64));
            test_b.push(F::from((i + 1) * 2 as u64));
        }

        let parameters = sponge::poseidon::PoseidonConfig::new(8, 24, 31, mds, ark, 2, 1);
        Self { parameters }
    }
}

impl<F: Field + PrimeField + Absorb> SisuHasher<F> for Poseidon<F> {
    fn hash_slice(&mut self, input: &[F]) -> F {
        assert!(input.len() == 1, "Poseidon only handles a single input");
        poseidon::CRH::evaluate(&self.parameters, vec![input[0].clone()]).unwrap()
    }

    fn hash_two(&mut self, a: &F, b: &F) -> F {
        poseidon::TwoToOneCRH::compress(&self.parameters, a.clone(), b.clone()).unwrap()
    }
}

#[derive(Clone)]
pub struct SisuMimc<F: Field> {
    parameters: &'static [F; MAX_MIMC_K],
}

impl Default for SisuMimc<FrBN254> {
    fn default() -> Self {
        Self {
            parameters: &K_BN254,
        }
    }
}

impl Default for SisuMimc<FpFRI> {
    fn default() -> Self {
        Self { parameters: &K_FRI }
    }
}

impl SisuMimc<FpFRI> {
    pub fn hash_one_field(x: &FpFRI) -> FpFRI {
        // FpFRI.MODULUS % 3 == 2 --> x^3
        let mut result = x.square();
        result.mul_assign(x);
        result
    }

    pub fn hash_two_fields(&self, a: &FpFRI, b: &FpFRI) -> FpFRI {
        Self::hash_one_field(&(self.parameters[0] * a + self.parameters[1] * b))
    }

    pub fn hash_array(&self, arr: &[FpFRI]) -> FpFRI {
        let mut r = FpFRI::ZERO;
        for i in 0..arr.len() {
            r += self.parameters[i] * arr[i];
        }
        Self::hash_one_field(&r)
    }
}

impl SisuMimc<FrBN254> {
    pub fn hash_one_field(x: &FrBN254) -> FrBN254 {
        // Permuation polynomial: FrBN254.MODULUS % 5 == 2 --> x^5
        //
        // But we can use a lower power because we do this operation many times
        // (128 times).
        //
        // Currently, x^3 is enough.
        let mut result = x.square(); // x^2
        result.mul_assign(x); // x^3
        result
    }

    pub fn hash_array(&self, arr: &[FrBN254]) -> FrBN254 {
        // The total repetitions at least is two.
        let mut num_repetitions = MAX_MIMC_K / arr.len();
        if num_repetitions < 2 {
            num_repetitions = 2;
        }

        let mut r = FrBN254::ZERO;
        for repetition_index in 0..num_repetitions {
            let d = mimc_k::D[repetition_index % 8];
            let start = repetition_index % arr.len();
            for i in 0..arr.len() {
                let k_index = (repetition_index * arr.len() + i) % MAX_MIMC_K;
                let v_index = (start + d * i) % arr.len();

                r = Self::hash_one_field(&(r + arr[v_index] + self.parameters[k_index]));
            }
        }

        r
    }
}

impl SisuHasher<FpFRI> for SisuMimc<FpFRI> {
    fn hash_slice(&mut self, input: &[FpFRI]) -> FpFRI {
        self.hash_array(input)
    }

    fn hash_two(&mut self, a: &FpFRI, b: &FpFRI) -> FpFRI {
        self.hash_two_fields(a, b)
    }
}

impl SisuHasher<FrBN254> for SisuMimc<FrBN254> {
    fn hash_slice(&mut self, input: &[FrBN254]) -> FrBN254 {
        self.hash_array(input)
    }

    fn hash_two(&mut self, a: &FrBN254, b: &FrBN254) -> FrBN254 {
        self.hash_array(&[a.clone(), b.clone()])
    }
}

pub const MAX_MIMC_K: usize = 128;

pub fn generate_mimc_k<F: Field + PrimeField>(n: usize) -> Vec<F> {
    let mut rng = test_rng();
    let mut result = vec![];
    for _ in 0..n {
        result.push(F::rand(&mut rng));
    }

    result
}

#[cfg(test)]
mod test {

    use ark_crypto_primitives::crh::poseidon;
    use ark_crypto_primitives::crh::{pedersen, CRHScheme, TwoToOneCRHScheme};
    use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
    use ark_ec::CurveGroup;
    use ark_ed_on_bls12_381::EdwardsProjective as JubJub;
    use ark_ff::{BigInteger, PrimeField};
    use ark_std::rand::Rng;
    use ark_std::UniformRand;
    use sisulib::common::{convert_field_to_string, convert_vec_fp_to_raw_bigint};
    use sisulib::field::{Fp97, FpFRI, FpSisu, FrBN254};

    use crate::hash::{generate_mimc_k, SisuMimc};

    type TestCRH = pedersen::CRH<JubJub, Window>;
    type TestTwoToOneCRH = pedersen::TwoToOneCRH<JubJub, Window>;

    #[derive(Clone, PartialEq, Eq, Hash)]
    pub(super) struct Window;

    impl pedersen::Window for Window {
        const WINDOW_SIZE: usize = 127;
        const NUM_WINDOWS: usize = 9;
    }

    fn generate_affine<R: Rng>(rng: &mut R) -> <JubJub as CurveGroup>::Affine {
        let val = <JubJub as CurveGroup>::Affine::rand(rng);
        val
    }

    #[test]
    fn test_native_equality() {
        let rng = &mut rand::thread_rng();
        let input = vec![1, 2, 3, 4, 5, 6];

        let parameters = TestCRH::setup(rng).unwrap();
        let primitive_result = TestCRH::evaluate(&parameters, input.as_slice()).unwrap();

        println!("{:?}", primitive_result)
    }

    #[test]
    fn test_naive_two_to_one_equality() {
        let rng = &mut rand::thread_rng();

        let left_input = generate_affine(rng);
        let right_input = generate_affine(rng);
        let parameters = TestTwoToOneCRH::setup(rng).unwrap();
        let primitive_result =
            TestTwoToOneCRH::compress(&parameters, left_input, right_input).unwrap();

        println!("{:?}", primitive_result)
    }

    #[test]
    fn test_consistency() {
        // The following way of generating the MDS and ARK matrix is incorrect
        // and is only for test purposes.

        let mut mds = vec![vec![]; 3];
        for i in 0..3 {
            for j in 0..3 {
                mds[i].push(Fp97::from((i * j + 1) as u64));
            }
        }

        let mut ark = vec![vec![]; 8 + 24];
        for i in 0..8 + 24 {
            for j in 0..3 {
                ark[i].push(Fp97::from((i * j + 1) as u64));
            }
        }

        let mut test_a = Vec::new();
        let mut test_b = Vec::new();
        for i in 0..3 {
            test_a.push(Fp97::from((i + 1) as u64));
            test_b.push(Fp97::from((i + 1) * 2 as u64));
        }

        let params = PoseidonConfig::<Fp97>::new(8, 24, 31, mds, ark, 2, 1);
        let crh_b = poseidon::CRH::<Fp97>::evaluate(&params, test_b.clone()).unwrap();
        let crh_a = poseidon::CRH::<Fp97>::evaluate(&params, test_a.clone()).unwrap();
        let crh = poseidon::TwoToOneCRH::<Fp97>::compress(&params, crh_a, crh_b).unwrap();

        println!("{:?} {:?} {:?}", crh, crh_a, crh_b);
    }

    #[test]
    fn test_mimc_fpfri() {
        let hash: FpFRI = SisuMimc::<FpFRI>::hash_one_field(&FpFRI::from(8u8));
        assert_eq!(FpFRI::from(512u128), hash);

        let mimc = SisuMimc::<FpFRI>::default();
        let hash: FpFRI =
            mimc.hash_array(&[FpFRI::from(123u8), FpFRI::from(96u8), FpFRI::from(200u8)]);
        assert_eq!(FpFRI::from(15856628003397761452u128), hash);
    }

    #[test]
    fn gen_mimc_k() {
        let n = 128;
        println!(
            "{:?}",
            convert_vec_fp_to_raw_bigint(&generate_mimc_k::<FrBN254>(n))
        );
        println!("================================");
        println!(
            "{:?}",
            convert_vec_fp_to_raw_bigint(&generate_mimc_k::<FpFRI>(n))
        );

        let f = FrBN254::from(1u64);
        println!("{:?}", f.into_bigint().to_bits_be());
    }

    #[test]
    fn print_f() {
        let mimc = SisuMimc::<FpSisu>::default();
        let hash: FpSisu =
            mimc.hash_array(&[FpSisu::from(123u8), FpSisu::from(96u8), FpSisu::from(200u8)]);

        println!("{:?}", convert_field_to_string(&hash));
    }
}
