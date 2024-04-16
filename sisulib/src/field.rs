use ark_ff::{Fp256, Fp64, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "389"]
#[generator = "2"]
pub struct Fr389Config;
pub type Fp389 = Fp64<MontBackend<Fr389Config, 1>>;

#[derive(MontConfig)]
#[modulus = "12037"]
#[generator = "2"]
pub struct Fr12037Config;
pub type Fp12037 = Fp64<MontBackend<Fr12037Config, 1>>;

#[derive(MontConfig)]
#[modulus = "11111119"]
#[generator = "2"]
pub struct Fr11111119Config;
pub type Fp11111119 = Fp64<MontBackend<Fr11111119Config, 1>>;

#[derive(MontConfig)]
#[modulus = "20011"]
#[generator = "2"]
pub struct Fr20011Config;
pub type Fp20011 = Fp64<MontBackend<Fr20011Config, 1>>;

#[derive(MontConfig)]
#[modulus = "337"]
#[generator = "5"]
pub struct Fr337Config;
pub type Fp337 = Fp64<MontBackend<Fr337Config, 1>>;

#[derive(MontConfig)]
#[modulus = "97"]
#[generator = "5"]
pub struct Fr97Config;
pub type Fp97 = Fp64<MontBackend<Fr97Config, 1>>;

#[derive(MontConfig)]
#[modulus = "41"]
#[generator = "2"]
pub struct Fr41Config;
pub type Fp41 = Fp64<MontBackend<Fr41Config, 1>>;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct FrFRIConfig;
pub type FpFRI = Fp64<MontBackend<FrFRIConfig, 1>>;

#[derive(MontConfig)]
#[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
#[generator = "7"]
#[small_subgroup_base = "3"]
#[small_subgroup_power = "1"]
pub struct Fr255Config;
pub type Fr255 = Fp256<MontBackend<Fr255Config, 4>>;

pub type FrBN254 = ark_bn254::Fr;
pub type FrBLS12_377 = ark_bls12_377::Fr;
pub type FrBLS12_381 = ark_bls12_381::Fr;
pub type FpSisu = FrBN254;
pub type FpBLS12_381 = ark_bls12_381::Fq;
