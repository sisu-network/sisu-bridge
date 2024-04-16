use std::marker::PhantomData;

use ark_ff::{BigInt, Field, Fp};
use ark_std::cfg_iter;
use icicle_core::traits::FieldImpl;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use sisulib::{common::convert_vec_field_to_string, field::FrBN254};

pub trait IcicleConvertibleField: Field {
    type IcicleField: FieldImpl;

    fn from_icicle(f: &Self::IcicleField) -> Self;
    fn to_icicle(&self) -> Self::IcicleField;
}

impl IcicleConvertibleField for FrBN254 {
    type IcicleField = icicle_bn254::curve::ScalarField;

    fn from_icicle(f: &Self::IcicleField) -> Self {
        Fp(BigInt(f.limbs), PhantomData)
    }

    fn to_icicle(&self) -> Self::IcicleField {
        Self::IcicleField::from(self.0 .0)
    }
}

pub fn device_to_string<F: IcicleConvertibleField>(
    device: &HostOrDeviceSlice<F::IcicleField>,
) -> Vec<String> {
    let mut v = vec![F::IcicleField::zero(); device.len()];
    device.copy_to_host(&mut v).unwrap();

    let v: Vec<_> = cfg_iter!(v).map(|x| F::from_icicle(x)).collect();
    convert_vec_field_to_string(&v)
}
