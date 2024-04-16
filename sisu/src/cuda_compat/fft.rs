use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use ark_ff::{FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::cfg_iter;
use icicle_bn254::curve::ScalarField as IcicleFrBN254;
use icicle_core::fft::{fft_evaluate, fft_evaluate_multi, fft_interpolate, fft_interpolate_multi};
use icicle_cuda_runtime::bindings::{cudaError, cudaMemcpy, cudaMemcpyKind, cudaMemset};
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostOrDeviceSlice2D};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use sisulib::domain::find_root_of_unity;
use sisulib::field::FrBN254;

use crate::cuda_compat::slice::CudaSlice;
use crate::icicle_converter::IcicleConvertibleField;

pub trait A<K> {
    fn x<F: Field>(x: F) -> F;
}

pub trait FftEngine<F: IcicleConvertibleField>: Clone + Sync + Send {
    type Configs: Sync + Send;

    fn configs(n: usize) -> Self::Configs;

    fn evaluate(configs: &Self::Configs, coeffs: &mut CudaSlice<F>) -> CudaSlice<F>;
    fn interpolate(configs: &Self::Configs, evaluations: &mut CudaSlice<F>) -> CudaSlice<F>;

    fn evaluate_multi(config: &Self::Configs, coeffs: &mut [CudaSlice<F>]) -> Vec<CudaSlice<F>> {
        let mut result = vec![];

        for c in coeffs.iter_mut() {
            result.push(Self::evaluate(config, c));
        }

        result
    }

    fn interpolate_multi(
        config: &Self::Configs,
        evaluations: &mut [CudaSlice<F>],
    ) -> Vec<CudaSlice<F>> {
        let mut result = vec![];

        for eval in evaluations.iter_mut() {
            result.push(Self::interpolate(config, eval));
        }

        result
    }
}

pub struct FFTEnginePool<F: IcicleConvertibleField, Fft: FftEngine<F>> {
    engines: Arc<RwLock<HashMap<usize, Fft::Configs>>>,
    __phantom: PhantomData<F>,
}

impl<F: IcicleConvertibleField, Fft: FftEngine<F>> Clone for FFTEnginePool<F, Fft> {
    fn clone(&self) -> Self {
        Self {
            engines: self.engines.clone(),
            __phantom: PhantomData,
        }
    }
}

impl<F: IcicleConvertibleField, Fft: FftEngine<F>> FFTEnginePool<F, Fft> {
    pub fn new() -> Self {
        Self {
            engines: Arc::new(RwLock::new(HashMap::default())),
            __phantom: PhantomData,
        }
    }

    fn engine_reader(&self, size: usize) -> RwLockReadGuard<HashMap<usize, Fft::Configs>> {
        let engine = self.engines.read().unwrap();
        if !engine.contains_key(&size) {
            drop(engine);

            let mut engine = self.engines.write().unwrap();
            if !engine.contains_key(&size) {
                (*engine).insert(size, Fft::configs(size));
            }
        }

        self.engines.read().unwrap()
    }

    pub fn evaluate(&self, output_size: usize, coeffs: &mut CudaSlice<F>) -> CudaSlice<F> {
        let engine = self.engine_reader(output_size);
        Fft::evaluate(&engine[&output_size], coeffs)
    }

    pub fn interpolate(&self, evaluations: &mut CudaSlice<F>) -> CudaSlice<F> {
        let engine = self.engine_reader(evaluations.len());
        Fft::interpolate(&engine[&evaluations.len()], evaluations)
    }

    pub fn evaluate_multi(
        &self,
        output_size: usize,
        coeffs: &mut [CudaSlice<F>],
    ) -> Vec<CudaSlice<F>> {
        let engine = self.engine_reader(output_size);
        Fft::evaluate_multi(&engine[&output_size], coeffs)
    }

    pub fn interpolate_multi(&self, evaluations: &mut [CudaSlice<F>]) -> Vec<CudaSlice<F>> {
        for eval in evaluations.iter() {
            assert!(eval.len() == evaluations[0].len());
        }

        let engine = self.engine_reader(evaluations[0].len());
        Fft::interpolate_multi(&engine[&evaluations[0].len()], evaluations)
    }
}

#[derive(Clone)]
pub struct ArkFftEngine<F: FftField> {
    __phantom: PhantomData<F>,
}

impl<F: FftField + IcicleConvertibleField> FftEngine<F> for ArkFftEngine<F> {
    type Configs = GeneralEvaluationDomain<F>;

    fn configs(n: usize) -> Self::Configs {
        GeneralEvaluationDomain::new(n).unwrap()
    }

    fn evaluate(configs: &Self::Configs, coeffs: &mut CudaSlice<F>) -> CudaSlice<F> {
        CudaSlice::on_host(configs.fft(coeffs.as_ref_host()))
    }

    fn interpolate(configs: &Self::Configs, evaluations: &mut CudaSlice<F>) -> CudaSlice<F> {
        CudaSlice::on_host(configs.ifft(evaluations.as_ref_host()))
    }
}

#[derive(Clone)]
pub struct IcicleFftConfigs<F: IcicleConvertibleField> {
    size: usize,
    ws: Arc<HostOrDeviceSlice<F::IcicleField>>,
    ws_inv: Arc<HostOrDeviceSlice<F::IcicleField>>,
}

unsafe impl<F: IcicleConvertibleField> Sync for IcicleFftConfigs<F> {}
unsafe impl<F: IcicleConvertibleField> Send for IcicleFftConfigs<F> {}

impl IcicleFftConfigs<FrBN254> {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());

        let ws = precompute_w(n, false);
        let ws_inv = precompute_w(n, true);

        let mut device_ws = HostOrDeviceSlice::cuda_malloc(ws.len()).unwrap();
        device_ws.copy_from_host(&ws).unwrap();

        let mut device_ws_inv = HostOrDeviceSlice::cuda_malloc(ws_inv.len()).unwrap();
        device_ws_inv.copy_from_host(&ws_inv).unwrap();

        Self {
            size: n,
            ws: Arc::new(device_ws),
            ws_inv: Arc::new(device_ws_inv),
        }
    }
}

#[derive(Clone)]
pub struct IcicleFftEngine<F: IcicleConvertibleField> {
    __phantom: PhantomData<F>,
}

impl IcicleFftEngine<FrBN254> {
    fn padding_coeffs(
        configs: &<Self as FftEngine<FrBN254>>::Configs,
        coeffs: &HostOrDeviceSlice<IcicleFrBN254>,
    ) -> HostOrDeviceSlice<IcicleFrBN254> {
        let mut inout_slice =
            HostOrDeviceSlice::<IcicleFrBN254>::cuda_malloc(configs.size).unwrap();

        let coeffs_size = size_of::<IcicleFrBN254>() * coeffs.len();
        let padding_size = size_of::<IcicleFrBN254>() * (configs.size - coeffs.len());
        unsafe {
            let err = cudaMemcpy(
                inout_slice.as_mut_ptr() as *mut c_void,
                coeffs.as_ptr() as *const c_void,
                coeffs_size,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            );
            assert_eq!(
                err,
                cudaError::cudaSuccess,
                "failed to copy coeffs to device {:?}",
                err
            );

            if padding_size > 0 {
                let err = cudaMemset(
                    inout_slice.as_mut_ptr().wrapping_add(coeffs.len()) as *mut c_void,
                    0,
                    padding_size,
                );
                assert_eq!(
                    err,
                    cudaError::cudaSuccess,
                    "failed to copy coeffs to device {:?}",
                    err
                );
            }
        };

        inout_slice
    }
}

impl FftEngine<FrBN254> for IcicleFftEngine<FrBN254> {
    type Configs = IcicleFftConfigs<FrBN254>;

    fn configs(n: usize) -> Self::Configs {
        Self::Configs::new(n)
    }

    fn evaluate(configs: &Self::Configs, coeffs: &mut CudaSlice<FrBN254>) -> CudaSlice<FrBN254> {
        assert!(coeffs.len() <= configs.size);

        let mut inout_slice = Self::padding_coeffs(configs, coeffs.as_ref_device());

        fft_evaluate(&mut inout_slice, &configs.ws, configs.size as u32).unwrap();

        CudaSlice::on_device(inout_slice)
    }

    fn interpolate(
        configs: &Self::Configs,
        evaluations: &mut CudaSlice<FrBN254>,
    ) -> CudaSlice<FrBN254> {
        assert!(evaluations.len() == configs.size);

        let mut inout_slice = evaluations.clone().as_device();

        fft_interpolate(&mut inout_slice, &configs.ws_inv, configs.size as u32).unwrap();

        CudaSlice::on_device(inout_slice)
    }

    fn evaluate_multi(
        config: &Self::Configs,
        coeffs: &mut [CudaSlice<FrBN254>],
    ) -> Vec<CudaSlice<FrBN254>> {
        let mut coeffs_2d = vec![];
        for coeff in coeffs.iter_mut() {
            let inout_slice = Self::padding_coeffs(config, coeff.as_ref_device());
            coeffs_2d.push(inout_slice);
        }

        let mut coeffs_2d = HostOrDeviceSlice2D::ref_from(coeffs_2d).unwrap();

        fft_evaluate_multi(&mut coeffs_2d, &config.ws, config.size as u32).unwrap();

        let mut result = vec![];
        for coeff in coeffs_2d.into_iter() {
            result.push(CudaSlice::on_device(coeff))
        }

        result
    }

    fn interpolate_multi(
        config: &Self::Configs,
        evaluations: &mut [CudaSlice<FrBN254>],
    ) -> Vec<CudaSlice<FrBN254>> {
        let mut evaluations_2d = vec![];
        for eval in evaluations.iter() {
            evaluations_2d.push(eval.clone().as_device());
        }

        let mut evaluations_2d = HostOrDeviceSlice2D::ref_from(evaluations_2d).unwrap();

        fft_interpolate_multi(&mut evaluations_2d, &config.ws_inv, config.size as u32).unwrap();

        let mut result = vec![];
        for eval in evaluations_2d.into_iter() {
            result.push(CudaSlice::on_device(eval))
        }

        result
    }
}

pub fn precompute_w(
    n: usize,
    invert: bool,
) -> Vec<<FrBN254 as IcicleConvertibleField>::IcicleField> {
    let mut root = find_root_of_unity::<FrBN254>(n);
    if invert {
        root = root.inverse().unwrap();
    }

    let mut ws = Vec::with_capacity(n);

    let mut len = 2;
    while len <= n {
        let mut wlen = root.clone();
        let mut i = len;
        while i < n {
            wlen = wlen * wlen;
            i <<= 1;
        }

        let mut w = FrBN254::ONE;
        for _ in 0..len >> 1 {
            ws.push(w);
            w = w * wlen;
        }

        len <<= 1;
    }

    cfg_iter!(ws)
        .map(|x| <FrBN254 as IcicleConvertibleField>::IcicleField::from(x.into_bigint().0))
        .collect()
}

#[cfg(test)]
mod tests {
    use sisulib::field::FrBN254;
    use std::time::Instant;
    use std::{sync::Arc, thread};

    use crate::cuda_compat::slice::CudaSlice;

    use super::{ArkFftEngine, FftEngine, IcicleFftEngine};

    #[test]
    fn test_fft() {
        let n = 2usize.pow(21);
        let icicle_fft_configs = IcicleFftEngine::<FrBN254>::configs(n);
        let ark_fft_configs = Arc::new(ArkFftEngine::<FrBN254>::configs(n));

        let mut coeffs = vec![];
        for i in 0..n {
            coeffs.push(FrBN254::from((i + 1) as u64));
        }

        let coeffs = CudaSlice::on_host(coeffs);

        // let root_domain = RootDomain::<FrBN254>::new(n);
        // let domain = Domain::from(&root_domain);
        // let mut expected_evaluations = vec![];
        // for i in 0..domain.len() {
        //     let x = domain[i];
        //     let mut tmp = FrBN254::ONE;
        //     let mut s = FrBN254::ZERO;
        //     for j in 0..n {
        //         s += tmp * coeffs[j];
        //         tmp *= x;
        //     }
        //     expected_evaluations.push(s);
        // }

        thread::scope(|s| {
            for _ in 0..8 {
                let ark_fft_configs = ark_fft_configs.clone();
                let icicle_fft_configs = icicle_fft_configs.clone();
                let mut coeffs = coeffs.clone();

                s.spawn(move || {
                    let now = Instant::now();
                    let mut ark_evaluations = ArkFftEngine::evaluate(&ark_fft_configs, &mut coeffs);
                    println!("ARK evaluations: {:?}", now.elapsed());

                    let now = Instant::now();
                    let mut icicle_evaluations =
                        IcicleFftEngine::evaluate(&icicle_fft_configs, &mut coeffs);
                    println!("ICICLE evaluations: {:?}", now.elapsed());

                    assert_eq!(
                        ark_evaluations.as_ref_host(),
                        icicle_evaluations.as_ref_host()
                    );

                    let now = Instant::now();
                    let mut ark_coeffs =
                        ArkFftEngine::interpolate(&ark_fft_configs, &mut ark_evaluations);
                    println!("ARK interpolate: {:?}", now.elapsed());

                    let now = Instant::now();
                    let mut icicle_coeffs =
                        IcicleFftEngine::interpolate(&icicle_fft_configs, &mut icicle_evaluations);
                    println!("ICICLE interpolate: {:?}", now.elapsed());

                    assert_eq!(ark_coeffs.as_ref_host(), icicle_coeffs.as_ref_host());
                    assert_eq!(ark_coeffs.as_ref_host(), coeffs.as_ref_host());
                });
            }
        });
    }
}
