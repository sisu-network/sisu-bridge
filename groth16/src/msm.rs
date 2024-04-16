use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, AffineRepr, VariableBaseMSM};
use ark_ff::{Field, PrimeField, Zero};
use ark_std::{cfg_into_iter, rand::Rng, UniformRand};
use icicle_bn254::curve::{BaseField, CurveCfg, G2BaseField, G2CurveCfg};
use icicle_core::{
    msm::{get_default_msm_config, msm, MSMConfig},
    traits::{ArkConvertible, FieldImpl},
};
use icicle_cuda_runtime::{memory::HostOrDeviceSlice, stream::CudaStream};
use std::marker::PhantomData;

pub trait MSMEngine<'a, E: Pairing>: Clone {
    type HostScalarField: Clone + Send + Sync;
    type HostG1Affine: Clone;
    type HostG2Affine: Clone;

    type DeviceScalarFields;
    type DeviceG1Affines: Send + Sync;
    type DeviceG2Affines: Send + Sync;

    type MSMStream;
    type G1MSMConfig: Send + Sync;
    type G2MSMConfig: Send + Sync;

    // UTILITIES
    fn create_stream() -> Self::MSMStream;
    fn create_g1_msm_config(stream: &'a Self::MSMStream) -> Self::G1MSMConfig;
    fn create_g2_msm_config(stream: &'a Self::MSMStream) -> Self::G2MSMConfig;
    fn host_scalar_is_zero(scalar: &Self::HostScalarField) -> bool;
    fn host_g1_is_zero(base: &Self::HostG1Affine) -> bool;
    fn host_g2_is_zero(base: &Self::HostG2Affine) -> bool;
    fn host_scalar_random<R: Rng>(rng: &mut R) -> Self::HostScalarField;
    fn host_scalar_zero() -> Self::HostScalarField;
    fn host_g1_zero() -> Self::HostG1Affine;

    // ARK to HOST
    fn ark_to_host_scalar(scalar: E::ScalarField) -> Self::HostScalarField;
    fn ark_to_host_g1_affine(base: E::G1Affine) -> Self::HostG1Affine;
    fn ark_to_host_g2_affine(base: E::G2Affine) -> Self::HostG2Affine;

    fn ark_to_host_scalars(scalars: Vec<E::ScalarField>) -> Vec<Self::HostScalarField>;
    fn ark_to_host_g1_affines(bases: Vec<E::G1Affine>) -> Vec<Self::HostG1Affine>;
    fn ark_to_host_g2_affines(bases: Vec<E::G2Affine>) -> Vec<Self::HostG2Affine>;

    // HOST to ARK
    fn host_to_ark_scalar(scalar: Self::HostScalarField) -> E::ScalarField;
    fn host_to_ark_g1_affine(base: Self::HostG1Affine) -> E::G1Affine;
    fn host_to_ark_g2_affine(base: Self::HostG2Affine) -> E::G2Affine;

    // HOST to DEVICE
    fn host_to_device_g1_affines(bases: &[Self::HostG1Affine]) -> Self::DeviceG1Affines;
    fn host_to_device_g2_affines(bases: &[Self::HostG2Affine]) -> Self::DeviceG2Affines;

    // MSM
    fn msm_g1(
        cfg: &Self::G1MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG1Affines,
    ) -> E::G1;
    fn msm_g2(
        cfg: &Self::G2MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG2Affines,
    ) -> E::G2;
}

#[derive(Clone)]
pub struct ArkMSMEngine<E: Pairing> {
    __phantom: PhantomData<E>,
}

impl<'a, E: Pairing> MSMEngine<'a, E> for ArkMSMEngine<E> {
    type HostScalarField = E::ScalarField;
    type HostG1Affine = E::G1Affine;
    type HostG2Affine = E::G2Affine;

    type DeviceScalarFields = Vec<E::ScalarField>;
    type DeviceG1Affines = Vec<E::G1Affine>;
    type DeviceG2Affines = Vec<E::G2Affine>;

    type MSMStream = bool;
    type G1MSMConfig = bool; // No Config for arkworks MSM
    type G2MSMConfig = bool; // No Config for arkworks MSM

    // UTILITIES
    fn create_stream() -> Self::MSMStream {
        true
    }

    fn create_g1_msm_config(_: &Self::MSMStream) -> Self::G1MSMConfig {
        true
    }

    fn create_g2_msm_config(_: &Self::MSMStream) -> Self::G2MSMConfig {
        true
    }

    fn host_scalar_is_zero(scalar: &Self::HostScalarField) -> bool {
        scalar.is_zero()
    }

    fn host_g1_is_zero(base: &Self::HostG1Affine) -> bool {
        base.is_zero()
    }

    fn host_g2_is_zero(base: &Self::HostG2Affine) -> bool {
        base.is_zero()
    }

    fn host_scalar_random<R: Rng>(rng: &mut R) -> Self::HostScalarField {
        E::ScalarField::rand(rng)
    }

    fn host_scalar_zero() -> Self::HostScalarField {
        Self::HostScalarField::ZERO
    }

    fn host_g1_zero() -> Self::HostG1Affine {
        Self::HostG1Affine::zero()
    }

    // ARK to HOST
    fn ark_to_host_scalar(scalar: <E as Pairing>::ScalarField) -> Self::HostScalarField {
        scalar
    }

    fn ark_to_host_g1_affine(base: <E as Pairing>::G1Affine) -> Self::HostG1Affine {
        base
    }

    fn ark_to_host_g2_affine(base: <E as Pairing>::G2Affine) -> Self::HostG2Affine {
        base
    }

    fn ark_to_host_scalars(scalars: Vec<E::ScalarField>) -> Vec<Self::HostScalarField> {
        scalars
    }

    fn ark_to_host_g1_affines(bases: Vec<E::G1Affine>) -> Vec<Self::HostG1Affine> {
        bases
    }

    fn ark_to_host_g2_affines(bases: Vec<E::G2Affine>) -> Vec<Self::HostG2Affine> {
        bases
    }

    // HOST to ARK
    fn host_to_ark_scalar(scalar: Self::HostScalarField) -> <E as Pairing>::ScalarField {
        scalar
    }

    fn host_to_ark_g1_affine(base: Self::HostG1Affine) -> <E as Pairing>::G1Affine {
        base
    }

    fn host_to_ark_g2_affine(base: Self::HostG2Affine) -> <E as Pairing>::G2Affine {
        base
    }

    // HOST to DEVICE
    fn host_to_device_g1_affines(bases: &[Self::HostG1Affine]) -> Self::DeviceG1Affines {
        bases.to_vec()
    }

    fn host_to_device_g2_affines(bases: &[Self::HostG2Affine]) -> Self::DeviceG2Affines {
        bases.to_vec()
    }

    fn msm_g1(
        _: &Self::G1MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG1Affines,
    ) -> E::G1 {
        let bigints = cfg_into_iter!(scalars)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();

        E::G1::msm_bigint(&bases, &bigints)
    }

    fn msm_g2(
        _: &Self::G2MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG2Affines,
    ) -> E::G2 {
        let bigints = cfg_into_iter!(scalars)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        E::G2::msm_bigint(bases, &bigints)
    }
}

#[derive(Clone)]
pub struct IcicleMSMEngine<'a, E: Pairing> {
    __phantom: &'a PhantomData<E>,
}

impl<'a> MSMEngine<'a, Bn254> for IcicleMSMEngine<'a, Bn254> {
    type HostScalarField = icicle_bn254::curve::ScalarField;
    type HostG1Affine = icicle_bn254::curve::G1Affine;
    type HostG2Affine = icicle_bn254::curve::G2Affine;

    type DeviceScalarFields = HostOrDeviceSlice<'a, Self::HostScalarField>;
    type DeviceG1Affines = HostOrDeviceSlice<'a, Self::HostG1Affine>;
    type DeviceG2Affines = HostOrDeviceSlice<'a, Self::HostG2Affine>;

    type MSMStream = CudaStream;
    type G1MSMConfig = MSMConfig<'a>;
    type G2MSMConfig = MSMConfig<'a>;

    // UTILITIES
    fn create_stream() -> Self::MSMStream {
        CudaStream::create().unwrap()
    }

    fn create_g1_msm_config(stream: &'a Self::MSMStream) -> Self::G1MSMConfig {
        let mut config = get_default_msm_config::<CurveCfg>();
        config.ctx.stream = stream;
        config.is_async = true;
        config
    }

    fn create_g2_msm_config(stream: &'a Self::MSMStream) -> Self::G2MSMConfig {
        let mut config = get_default_msm_config::<G2CurveCfg>();
        config.ctx.stream = stream;
        config.is_async = true;
        config
    }

    fn host_scalar_is_zero(scalar: &Self::HostScalarField) -> bool {
        scalar.eq(&Self::HostScalarField::zero())
    }

    fn host_g1_is_zero(base: &Self::HostG1Affine) -> bool {
        base.x.eq(&BaseField::zero()) && base.y.eq(&BaseField::zero())
    }

    fn host_g2_is_zero(base: &Self::HostG2Affine) -> bool {
        base.x.eq(&G2BaseField::zero()) && base.y.eq(&G2BaseField::zero())
    }

    fn host_scalar_random<R: Rng>(rng: &mut R) -> Self::HostScalarField {
        Self::HostScalarField::from_ark(<Bn254 as Pairing>::ScalarField::rand(rng))
    }

    fn host_scalar_zero() -> Self::HostScalarField {
        Self::HostScalarField::zero()
    }

    fn host_g1_zero() -> Self::HostG1Affine {
        Self::HostG1Affine::zero()
    }

    // ARK to HOST
    fn ark_to_host_scalar(scalar: <Bn254 as Pairing>::ScalarField) -> Self::HostScalarField {
        Self::HostScalarField::from_ark(scalar)
    }

    fn ark_to_host_g1_affine(base: <Bn254 as Pairing>::G1Affine) -> Self::HostG1Affine {
        Self::HostG1Affine::from_ark(base)
    }

    fn ark_to_host_g2_affine(base: <Bn254 as Pairing>::G2Affine) -> Self::HostG2Affine {
        Self::HostG2Affine::from_ark(base)
    }

    fn ark_to_host_scalars(
        scalars: Vec<<Bn254 as Pairing>::ScalarField>,
    ) -> Vec<Self::HostScalarField> {
        scalars
            .into_iter()
            .map(|x| Self::HostScalarField::from_ark(x))
            .collect()
    }

    fn ark_to_host_g1_affines(bases: Vec<<Bn254 as Pairing>::G1Affine>) -> Vec<Self::HostG1Affine> {
        bases
            .into_iter()
            .map(|x| Self::HostG1Affine::from_ark(x))
            .collect()
    }

    fn ark_to_host_g2_affines(bases: Vec<<Bn254 as Pairing>::G2Affine>) -> Vec<Self::HostG2Affine> {
        bases
            .into_iter()
            .map(|x| Self::HostG2Affine::from_ark(x))
            .collect()
    }

    // HOST to ARK
    fn host_to_ark_scalar(scalar: Self::HostScalarField) -> <Bn254 as Pairing>::ScalarField {
        scalar.to_ark()
    }

    fn host_to_ark_g1_affine(base: Self::HostG1Affine) -> <Bn254 as Pairing>::G1Affine {
        base.to_ark()
    }

    fn host_to_ark_g2_affine(base: Self::HostG2Affine) -> <Bn254 as Pairing>::G2Affine {
        base.to_ark()
    }

    // HOST to DEVICE
    fn host_to_device_g1_affines(bases: &[Self::HostG1Affine]) -> Self::DeviceG1Affines {
        let mut device_bases = HostOrDeviceSlice::cuda_malloc(bases.len()).unwrap();
        device_bases.copy_from_host(bases).unwrap();
        device_bases
    }

    fn host_to_device_g2_affines(bases: &[Self::HostG2Affine]) -> Self::DeviceG2Affines {
        let mut device_bases = HostOrDeviceSlice::cuda_malloc(bases.len()).unwrap();
        device_bases.copy_from_host(bases).unwrap();
        device_bases
    }

    fn msm_g1(
        config: &Self::G1MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG1Affines,
    ) -> <Bn254 as Pairing>::G1 {
        assert!(scalars.len() == bases.len());

        let mut device_scalars = HostOrDeviceSlice::cuda_malloc(scalars.len()).unwrap();
        device_scalars.copy_from_host(&scalars).unwrap();

        let mut device_result = HostOrDeviceSlice::cuda_malloc(1).unwrap();
        msm(&device_scalars, bases, &config, &mut device_result).unwrap();

        let mut result = vec![icicle_bn254::curve::G1Projective::zero()];
        device_result.copy_to_host(&mut result).unwrap();

        config.ctx.stream.synchronize().unwrap();

        result[0].to_ark()
    }

    fn msm_g2(
        config: &Self::G2MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG2Affines,
    ) -> <Bn254 as Pairing>::G2 {
        assert!(scalars.len() == bases.len());

        let mut device_scalars = HostOrDeviceSlice::cuda_malloc(scalars.len()).unwrap();
        device_scalars.copy_from_host(&scalars).unwrap();

        let mut device_result = HostOrDeviceSlice::cuda_malloc(1).unwrap();
        msm(&device_scalars, bases, &config, &mut device_result).unwrap();

        let mut result = vec![icicle_bn254::curve::G2Projective::zero()];
        device_result.copy_to_host(&mut result).unwrap();

        config.ctx.stream.synchronize().unwrap();

        result[0].to_ark()
    }
}

#[derive(Clone)]
pub struct FilterZeroIcicleMSMEngine<E: Pairing> {
    __phantom: PhantomData<E>,
}

impl<'a> MSMEngine<'a, Bn254> for FilterZeroIcicleMSMEngine<Bn254> {
    type HostScalarField = icicle_bn254::curve::ScalarField;
    type HostG1Affine = icicle_bn254::curve::G1Affine;
    type HostG2Affine = icicle_bn254::curve::G2Affine;

    type DeviceScalarFields = Vec<Self::HostScalarField>;
    type DeviceG1Affines = Vec<Self::HostG1Affine>;
    type DeviceG2Affines = Vec<Self::HostG2Affine>;

    type MSMStream = CudaStream;
    type G1MSMConfig = MSMConfig<'a>;
    type G2MSMConfig = MSMConfig<'a>;

    // UTILITIES
    fn create_stream() -> Self::MSMStream {
        CudaStream::create().unwrap()
    }

    fn create_g1_msm_config(stream: &'a Self::MSMStream) -> Self::G1MSMConfig {
        let mut config = get_default_msm_config::<CurveCfg>();
        config.ctx.stream = stream;
        config.is_async = true;
        config
    }

    fn create_g2_msm_config(stream: &'a Self::MSMStream) -> Self::G2MSMConfig {
        let mut config = get_default_msm_config::<G2CurveCfg>();
        config.ctx.stream = stream;
        config.is_async = true;
        config
    }

    fn host_scalar_is_zero(scalar: &Self::HostScalarField) -> bool {
        scalar.eq(&Self::HostScalarField::zero())
    }

    fn host_g1_is_zero(base: &Self::HostG1Affine) -> bool {
        base.x.eq(&BaseField::zero()) && base.y.eq(&BaseField::zero())
    }

    fn host_g2_is_zero(base: &Self::HostG2Affine) -> bool {
        base.x.eq(&G2BaseField::zero()) && base.y.eq(&G2BaseField::zero())
    }

    fn host_scalar_random<R: Rng>(rng: &mut R) -> Self::HostScalarField {
        Self::HostScalarField::from_ark(<Bn254 as Pairing>::ScalarField::rand(rng))
    }

    fn host_scalar_zero() -> Self::HostScalarField {
        Self::HostScalarField::zero()
    }

    fn host_g1_zero() -> Self::HostG1Affine {
        Self::HostG1Affine::zero()
    }

    // ARK to HOST
    fn ark_to_host_scalar(scalar: <Bn254 as Pairing>::ScalarField) -> Self::HostScalarField {
        Self::HostScalarField::from_ark(scalar)
    }

    fn ark_to_host_g1_affine(base: <Bn254 as Pairing>::G1Affine) -> Self::HostG1Affine {
        Self::HostG1Affine::from_ark(base)
    }

    fn ark_to_host_g2_affine(base: <Bn254 as Pairing>::G2Affine) -> Self::HostG2Affine {
        Self::HostG2Affine::from_ark(base)
    }

    fn ark_to_host_scalars(
        scalars: Vec<<Bn254 as Pairing>::ScalarField>,
    ) -> Vec<Self::HostScalarField> {
        scalars
            .into_iter()
            .map(|x| Self::HostScalarField::from_ark(x))
            .collect()
    }

    fn ark_to_host_g1_affines(bases: Vec<<Bn254 as Pairing>::G1Affine>) -> Vec<Self::HostG1Affine> {
        bases
            .into_iter()
            .map(|x| Self::HostG1Affine::from_ark(x))
            .collect()
    }

    fn ark_to_host_g2_affines(bases: Vec<<Bn254 as Pairing>::G2Affine>) -> Vec<Self::HostG2Affine> {
        bases
            .into_iter()
            .map(|x| Self::HostG2Affine::from_ark(x))
            .collect()
    }

    // HOST to ARK
    fn host_to_ark_scalar(scalar: Self::HostScalarField) -> <Bn254 as Pairing>::ScalarField {
        scalar.to_ark()
    }

    fn host_to_ark_g1_affine(base: Self::HostG1Affine) -> <Bn254 as Pairing>::G1Affine {
        base.to_ark()
    }

    fn host_to_ark_g2_affine(base: Self::HostG2Affine) -> <Bn254 as Pairing>::G2Affine {
        base.to_ark()
    }

    // HOST to DEVICE
    fn host_to_device_g1_affines(bases: &[Self::HostG1Affine]) -> Self::DeviceG1Affines {
        bases.to_vec()
    }

    fn host_to_device_g2_affines(bases: &[Self::HostG2Affine]) -> Self::DeviceG2Affines {
        bases.to_vec()
    }

    fn msm_g1(
        config: &Self::G1MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG1Affines,
    ) -> <Bn254 as Pairing>::G1 {
        assert!(scalars.len() == bases.len());

        let mut filtered_scalars = vec![];
        let mut filtered_bases = vec![];
        let _: () = scalars
            .into_iter()
            .zip(bases)
            .filter(|(s, _)| !Self::host_scalar_is_zero(s))
            .map(|(s, b)| {
                filtered_scalars.push(s);
                filtered_bases.push(b.clone());
            })
            .collect();

        let mut device_scalars = HostOrDeviceSlice::cuda_malloc(filtered_scalars.len()).unwrap();
        device_scalars.copy_from_host(&filtered_scalars).unwrap();

        let mut device_bases = HostOrDeviceSlice::cuda_malloc(filtered_bases.len()).unwrap();
        device_bases.copy_from_host(&filtered_bases).unwrap();

        let mut device_result = HostOrDeviceSlice::cuda_malloc(1).unwrap();
        msm(&device_scalars, &device_bases, &config, &mut device_result).unwrap();

        let mut result = vec![icicle_bn254::curve::G1Projective::zero()];
        device_result.copy_to_host(&mut result).unwrap();

        config.ctx.stream.synchronize().unwrap();
        result[0].to_ark()
    }

    fn msm_g2(
        config: &Self::G2MSMConfig,
        scalars: Vec<Self::HostScalarField>,
        bases: &Self::DeviceG2Affines,
    ) -> <Bn254 as Pairing>::G2 {
        assert!(scalars.len() == bases.len());

        let mut filtered_scalars = vec![];
        let mut filtered_bases = vec![];
        let _: () = scalars
            .into_iter()
            .zip(bases)
            .filter(|(s, _)| !Self::host_scalar_is_zero(s))
            .map(|(s, b)| {
                filtered_scalars.push(s);
                filtered_bases.push(b.clone());
            })
            .collect();

        let mut device_scalars = HostOrDeviceSlice::cuda_malloc(filtered_scalars.len()).unwrap();
        device_scalars.copy_from_host(&filtered_scalars).unwrap();

        let mut device_bases = HostOrDeviceSlice::cuda_malloc(filtered_bases.len()).unwrap();
        device_bases.copy_from_host(&filtered_bases).unwrap();

        let mut device_result = HostOrDeviceSlice::cuda_malloc(1).unwrap();
        msm(&device_scalars, &device_bases, &config, &mut device_result).unwrap();

        let mut result = vec![icicle_bn254::curve::G2Projective::zero()];
        device_result.copy_to_host(&mut result).unwrap();

        config.ctx.stream.synchronize().unwrap();
        result[0].to_ark()
    }
}
