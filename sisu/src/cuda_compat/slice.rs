use std::{
    ffi::c_void,
    mem::size_of,
    ops::{Range, RangeFrom, RangeTo},
};

use ark_std::{cfg_into_iter, cfg_iter};
use icicle_core::traits::{FieldImpl, IcicleResultWrap};
use icicle_cuda_runtime::{
    bindings::{cudaMemcpy, cudaMemcpyKind},
    memory::HostOrDeviceSlice,
};

use crate::icicle_converter::IcicleConvertibleField;

#[derive(Clone)]
pub enum ConvertDirection {
    ToHost,
    ToDevice,
}

pub struct HostVec<T> {
    ptr: *mut T,
    size: usize,
    is_ref: bool,
}

impl<T> HostVec<T> {
    pub fn new(mut v: Vec<T>) -> Self {
        v.shrink_to_fit();

        let slice = Self {
            ptr: v.as_mut_ptr(),
            size: v.len(),
            is_ref: false,
        };

        std::mem::forget(v);

        slice
    }

    pub fn ref_from(other: &Self) -> Self {
        Self {
            ptr: other.ptr,
            size: other.size,
            is_ref: true,
        }
    }

    pub fn to_vec(mut self) -> Vec<T> {
        assert!(!self.is_ref, "reference is not allowed to convert to vec");
        self.is_ref = true;

        unsafe { Vec::from_raw_parts(self.ptr, self.size, self.size) }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

unsafe impl<T> Send for HostVec<T> {}
unsafe impl<T> Sync for HostVec<T> {}

impl<T: Clone> Clone for HostVec<T> {
    fn clone(&self) -> Self {
        Self::new(self.as_slice().to_vec())
    }
}

impl<T> Drop for HostVec<T> {
    fn drop(&mut self) {
        if !self.is_ref {
            // Convert this variable to vector, so then Rust will drop it
            // automatically.
            unsafe {
                Vec::from_raw_parts(self.ptr, self.size, self.size);
            }
        }
    }
}

#[derive(Clone)]
pub struct CudaSlice<F: IcicleConvertibleField> {
    is_all_one: bool,
    last_convert: ConvertDirection,
    host: Option<HostVec<F>>,
    device: Option<HostOrDeviceSlice<F::IcicleField>>,
}

impl<F: IcicleConvertibleField> Default for CudaSlice<F> {
    fn default() -> Self {
        Self {
            is_all_one: false,
            last_convert: ConvertDirection::ToHost,
            host: None,
            device: None,
        }
    }
}

impl<F: IcicleConvertibleField> CudaSlice<F> {
    pub fn new(h: Vec<F>, d: HostOrDeviceSlice<F::IcicleField>) -> Self {
        Self {
            last_convert: ConvertDirection::ToDevice,
            is_all_one: false,
            host: Some(HostVec::new(h)),
            device: Some(d),
        }
    }

    pub fn on_host(h: Vec<F>) -> Self {
        Self {
            last_convert: ConvertDirection::ToHost,
            is_all_one: false,
            host: Some(HostVec::new(h)),
            device: None,
        }
    }

    pub fn on_device(d: HostOrDeviceSlice<F::IcicleField>) -> Self {
        Self {
            last_convert: ConvertDirection::ToDevice,
            is_all_one: false,
            host: None,
            device: Some(d),
        }
    }

    pub fn ref_from(other: &Self) -> Self {
        let h = match &other.host {
            None => None,
            Some(h) => Some(HostVec::ref_from(h)),
        };

        let d = match &other.device {
            None => None,
            Some(d) => Some(HostOrDeviceSlice::device_ref(d)),
        };

        Self {
            is_all_one: other.is_all_one,
            last_convert: other.last_convert.clone(),
            host: h,
            device: d,
        }
    }

    pub fn zeros_on_host(size: usize) -> Self {
        Self {
            last_convert: ConvertDirection::ToHost,
            is_all_one: false,
            host: Some(HostVec::new(vec![F::ZERO; size])),
            device: None,
        }
    }

    pub fn zeros_on_device(size: usize) -> Self {
        Self {
            last_convert: ConvertDirection::ToDevice,
            is_all_one: false,
            host: None,
            device: Some(HostOrDeviceSlice::zeros_on_device(size).unwrap()),
        }
    }

    pub fn drop_host(&mut self) {
        self.host = None;
    }

    pub fn drop_device(&mut self) {
        self.device = None;
    }

    pub fn ones_on_host(size: usize) -> Self {
        Self {
            last_convert: ConvertDirection::ToHost,
            is_all_one: true,
            host: Some(HostVec::new(vec![F::ONE; size])),
            device: None,
        }
    }

    pub fn ones_on_device(size: usize) -> Self {
        let ones = vec![F::IcicleField::one(); size];
        let device = HostOrDeviceSlice::on_device(&ones).unwrap();

        Self {
            last_convert: ConvertDirection::ToDevice,
            is_all_one: true,
            host: None,
            device: Some(device),
        }
    }

    pub fn cuda_malloc(count: usize) -> Self {
        Self {
            last_convert: ConvertDirection::ToDevice,
            is_all_one: false,
            host: None,
            device: Some(HostOrDeviceSlice::cuda_malloc(count).unwrap()),
        }
    }

    pub fn is_one(&self) -> bool {
        self.is_all_one
    }

    pub fn is_host(&self) -> bool {
        self.host.is_some()
    }

    pub fn is_device(&self) -> bool {
        self.device.is_some()
    }

    pub fn len(&self) -> usize {
        match &self.host {
            Some(host) => host.len(),
            None => match &self.device {
                Some(device) => device.len(),
                None => panic!("invalid cuda slice"),
            },
        }
    }

    pub fn as_host(self) -> Vec<F> {
        match self.host {
            None => match self.device {
                None => panic!("invalid cuda slice"),
                Some(device) => {
                    let mut icicle_output = vec![F::IcicleField::zero(); device.len()];
                    device.copy_to_host(&mut icicle_output).unwrap();

                    cfg_into_iter!(icicle_output)
                        .map(|x| F::from_icicle(&x))
                        .collect()
                }
            },
            Some(host) => host.to_vec(),
        }
    }

    pub fn as_device(self) -> HostOrDeviceSlice<F::IcicleField> {
        match self.device {
            None => match self.host {
                None => panic!("invalid cuda slice"),
                Some(host) => {
                    let icicle_input: Vec<_> =
                        cfg_iter!(host.as_slice()).map(|x| x.to_icicle()).collect();
                    HostOrDeviceSlice::on_device(&icicle_input).unwrap()
                }
            },
            Some(device) => device,
        }
    }

    pub fn must_as_ref_host(&self) -> &[F] {
        self.host.as_ref().unwrap().as_slice()
    }

    pub fn must_as_ref_device(&self) -> &HostOrDeviceSlice<F::IcicleField> {
        self.device.as_ref().unwrap()
    }

    pub fn as_ref_host(&mut self) -> &[F] {
        self.last_convert = ConvertDirection::ToHost;

        if let None = &self.host {
            match &self.device {
                None => panic!("invalid cuda slice"),
                Some(device) => {
                    let mut icicle_output = vec![F::IcicleField::zero(); device.len()];
                    device.copy_to_host(&mut icicle_output).unwrap();

                    let host: Vec<_> = cfg_into_iter!(icicle_output)
                        .map(|x| F::from_icicle(&x))
                        .collect();

                    self.host = Some(HostVec::new(host));
                }
            }
        }

        self.must_as_ref_host()
    }

    pub fn as_ref_device(&mut self) -> &HostOrDeviceSlice<F::IcicleField> {
        self.last_convert = ConvertDirection::ToDevice;

        if let None = &self.device {
            match &self.host {
                None => panic!("invalid cuda slice"),
                Some(host) => {
                    let icicle_input: Vec<_> =
                        cfg_iter!(host.as_slice()).map(|x| x.to_icicle()).collect();
                    let device = HostOrDeviceSlice::on_device(&icicle_input).unwrap();

                    self.device = Some(device);
                }
            }
        }

        self.must_as_ref_device()
    }

    pub fn as_mut_host(&mut self) -> &mut [F] {
        self.last_convert = ConvertDirection::ToHost;

        self.as_ref_host();

        self.is_all_one = false;
        self.device = None;
        self.host.as_mut().unwrap().as_mut_slice()
    }

    pub fn as_mut_device(&mut self) -> &mut HostOrDeviceSlice<F::IcicleField> {
        self.as_ref_device();

        self.is_all_one = false;
        self.host = None;
        self.device.as_mut().unwrap()
    }

    pub fn at(&self, index: usize) -> F {
        match &self.host {
            Some(host) => return host.as_slice()[index].clone(),
            None => (),
        }

        match &self.device {
            Some(device) => {
                let mut icicle_output = vec![F::IcicleField::zero(); 1];
                device
                    .copy_to_host_partially(&mut icicle_output, index)
                    .unwrap();

                return F::from_icicle(&icicle_output[0]);
            }
            None => (),
        }

        panic!("invalid cuda slice");
    }

    pub fn at_range(&self, range: Range<usize>) -> CudaSlice<F> {
        if let ConvertDirection::ToHost = self.last_convert {
            self.at_range_host(range)
        } else {
            self.at_range_device(range)
        }
    }

    pub fn at_range_from(&self, from: RangeFrom<usize>) -> CudaSlice<F> {
        self.at_range(from.start..self.len())
    }

    pub fn at_range_to(&self, to: RangeTo<usize>) -> CudaSlice<F> {
        self.at_range(0..to.end)
    }

    pub fn at_range_from_host(&self, from: RangeFrom<usize>) -> CudaSlice<F> {
        self.at_range_host(from.start..self.len())
    }

    pub fn at_range_to_host(&self, to: RangeTo<usize>) -> CudaSlice<F> {
        self.at_range_host(0..to.end)
    }

    fn at_range_host(&self, range: Range<usize>) -> CudaSlice<F> {
        match &self.host {
            Some(host) => return CudaSlice::on_host(host.as_slice()[range].to_vec()),
            None => panic!("invalid memory host"),
        }
    }

    fn at_range_device(&self, range: Range<usize>) -> CudaSlice<F> {
        match &self.device {
            Some(device) => {
                let mut range_device =
                    HostOrDeviceSlice::cuda_malloc(range.end - range.start).unwrap();

                let size = range_device.len() * size_of::<F::IcicleField>();
                unsafe {
                    cudaMemcpy(
                        range_device.as_mut_ptr() as *mut c_void,
                        device.as_ptr().wrapping_add(range.start) as *const c_void,
                        size,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    )
                    .wrap()
                    .unwrap();
                }

                return CudaSlice::on_device(range_device);
            }
            None => panic!("invalid memory device"),
        }
    }
}
