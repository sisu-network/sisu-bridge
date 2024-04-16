use ark_ff::Field;
use icicle_cuda_runtime::memory::{
    HostOrDeviceSlice, HostOrDeviceSlice2D, HostOrDeviceSliceWrapper, ToCuda,
};

use sisulib::{
    circuit::{
        general_circuit::{circuit::GeneralCircuit, gate::LayerIndex, layer::LayerExtensions},
        CircuitParams,
    },
    common::{round_to_pow_of_two, split_number},
    mle::sparse::SisuSparseMultilinearExtension,
};

use crate::icicle_converter::IcicleConvertibleField;

use super::slice::CudaSlice;

trait OptionAsPtr {
    fn as_ptr(&self) -> *const Self;
}

pub struct CudaSparseMultilinearExtension<F: IcicleConvertibleField> {
    pub size: u32,
    pub z_num_vars: u32,
    pub x_num_vars: u32,
    pub y_num_vars: u32,

    pub point_z: HostOrDeviceSlice<u32>,
    pub point_x: HostOrDeviceSlice<u32>,
    pub point_y: HostOrDeviceSlice<u32>,
    pub evaluations: HostOrDeviceSlice<F::IcicleField>,

    pub z_indices_start: HostOrDeviceSlice<u32>,
    pub z_indices: HostOrDeviceSlice<u32>,

    pub x_indices_start: HostOrDeviceSlice<u32>,
    pub x_indices: HostOrDeviceSlice<u32>,

    pub y_indices_start: HostOrDeviceSlice<u32>,
    pub y_indices: HostOrDeviceSlice<u32>,
}

impl<'a, F: IcicleConvertibleField> CudaSparseMultilinearExtension<F> {
    pub fn from(
        mle: &SisuSparseMultilinearExtension<F>,
        circuit_params: &CircuitParams<F>,
        z_num_vars: usize,
        x_num_vars: usize,
        y_num_vars: usize,
    ) -> Self {
        assert!(z_num_vars + x_num_vars + y_num_vars == mle.num_vars());

        let mut point_z = vec![];
        let mut point_x = vec![];
        let mut point_y = vec![];
        let mut evaluations = vec![];

        let mut z_indices = vec![vec![]; 2usize.pow(z_num_vars as u32)];
        let mut x_indices = vec![vec![]; 2usize.pow(x_num_vars as u32)];
        let mut y_indices = vec![vec![]; 2usize.pow(y_num_vars as u32)];

        let mut z_indices_size = vec![0; 2usize.pow(z_num_vars as u32)];
        let mut x_indices_size = vec![0; 2usize.pow(x_num_vars as u32)];
        let mut y_indices_size = vec![0; 2usize.pow(y_num_vars as u32)];

        for (point, evaluation) in mle.evaluations.compile(circuit_params) {
            let (zx, y) = split_number(&point, y_num_vars);
            let (z, x) = split_number(&zx, x_num_vars);

            z_indices[z].push(point_z.len() as u32);
            x_indices[x].push(point_x.len() as u32);
            y_indices[y].push(point_y.len() as u32);

            z_indices_size[z] += 1;
            x_indices_size[x] += 1;
            y_indices_size[y] += 1;

            point_z.push(z as u32);
            point_x.push(x as u32);
            point_y.push(y as u32);
            evaluations.push(evaluation.to_icicle());
        }

        let device_point_z = HostOrDeviceSlice::on_device(&point_z).unwrap();
        let device_point_x = HostOrDeviceSlice::on_device(&point_x).unwrap();
        let device_point_y = HostOrDeviceSlice::on_device(&point_y).unwrap();
        let device_evaluations = HostOrDeviceSlice::on_device(&evaluations).unwrap();

        let flatten_z_indices = Self::flatten_indices(z_indices);
        let flatten_x_indices = Self::flatten_indices(x_indices);
        let flatten_y_indices = Self::flatten_indices(y_indices);

        let device_z_indices = HostOrDeviceSlice::on_device(&flatten_z_indices).unwrap();
        let device_x_indices = HostOrDeviceSlice::on_device(&flatten_x_indices).unwrap();
        let device_y_indices = HostOrDeviceSlice::on_device(&flatten_y_indices).unwrap();

        let z_indices_start = Self::size_to_start(z_indices_size);
        let x_indices_start = Self::size_to_start(x_indices_size);
        let y_indices_start = Self::size_to_start(y_indices_size);
        let device_z_indices_start = HostOrDeviceSlice::on_device(&z_indices_start).unwrap();
        let device_x_indices_start = HostOrDeviceSlice::on_device(&x_indices_start).unwrap();
        let device_y_indices_start = HostOrDeviceSlice::on_device(&y_indices_start).unwrap();

        Self {
            size: mle.num_evaluations() as u32,
            z_num_vars: z_num_vars as u32,
            x_num_vars: x_num_vars as u32,
            y_num_vars: y_num_vars as u32,
            point_z: device_point_z,
            point_x: device_point_x,
            point_y: device_point_y,
            evaluations: device_evaluations,
            z_indices_start: device_z_indices_start,
            z_indices: device_z_indices,
            x_indices_start: device_x_indices_start,
            x_indices: device_x_indices,
            y_indices_start: device_y_indices_start,
            y_indices: device_y_indices,
        }
    }

    fn size_to_start(size_arr: Vec<u32>) -> Vec<u32> {
        let mut result = vec![0];

        for i in 0..size_arr.len() {
            result.push(result[i] + size_arr[i]);
        }

        result
    }

    fn flatten_indices(indices_arr: Vec<Vec<u32>>) -> Vec<u32> {
        let mut result = vec![];

        for sub_arr in indices_arr {
            result.extend(sub_arr);
        }

        result
    }
}

impl<F: IcicleConvertibleField> ToCuda for CudaSparseMultilinearExtension<F> {
    type CudaRepr = icicle_core::sisu::SparseMultilinearExtension<F::IcicleField>;

    fn to_cuda(&self) -> icicle_core::sisu::SparseMultilinearExtension<F::IcicleField> {
        icicle_core::sisu::SparseMultilinearExtension {
            size: self.size,
            z_num_vars: self.z_num_vars,
            x_num_vars: self.x_num_vars,
            y_num_vars: self.y_num_vars,
            point_z: self.point_z.as_ptr(),
            point_x: self.point_x.as_ptr(),
            point_y: self.point_y.as_ptr(),
            evaluations: self.evaluations.as_ptr(),
            z_indices_start: self.z_indices_start.as_ptr(),
            z_indices: self.z_indices.as_ptr(),
            x_indices_start: self.x_indices_start.as_ptr(),
            x_indices: self.x_indices.as_ptr(),
            y_indices_start: self.y_indices_start.as_ptr(),
            y_indices: self.y_indices.as_ptr(),
        }
    }
}

pub struct CudaReverseSparseMultilinearExtension {
    pub size: u32,
    pub subset_num_vars: u32,
    pub real_num_vars: u32,

    pub point_subset: HostOrDeviceSlice<u32>,
    pub point_real: HostOrDeviceSlice<u32>,

    pub subset_position: HostOrDeviceSlice<u32>,
    pub real_position: HostOrDeviceSlice<u32>,
}

impl CudaReverseSparseMultilinearExtension {
    pub fn from<F: Field>(
        mle: &SisuSparseMultilinearExtension<F>,
        circuit_params: &CircuitParams<F>,
        z_num_vars: usize,
        x_num_vars: usize,
    ) -> Self {
        assert!(mle.num_vars() == 0 || z_num_vars + x_num_vars == mle.num_vars());
        assert!(mle.num_evaluations() <= 2usize.pow(z_num_vars as u32));

        let mut point_z = vec![];
        let mut point_x = vec![];

        let mut z_index = vec![u32::MAX; 2usize.pow(z_num_vars as u32)];
        let mut x_index = vec![u32::MAX; 2usize.pow(x_num_vars as u32)];

        for (point, _) in mle.evaluations.compile(circuit_params) {
            let (z, x) = split_number(&point, x_num_vars);

            // This index must not be occupied.
            assert!(z_index[z] == u32::MAX);
            assert!(x_index[x] == u32::MAX);

            z_index[z] = point_z.len() as u32;
            x_index[x] = point_x.len() as u32;

            point_z.push(z as u32);
            point_x.push(x as u32);
        }

        let device_point_z = HostOrDeviceSlice::on_device(&point_z).unwrap();
        let device_point_x = HostOrDeviceSlice::on_device(&point_x).unwrap();

        let device_z_indices = HostOrDeviceSlice::on_device(&z_index).unwrap();
        let device_x_indices = HostOrDeviceSlice::on_device(&x_index).unwrap();

        Self {
            size: mle.num_evaluations() as u32,
            subset_num_vars: z_num_vars as u32,
            real_num_vars: x_num_vars as u32,
            point_subset: device_point_z,
            point_real: device_point_x,
            subset_position: device_z_indices,
            real_position: device_x_indices,
        }
    }
}

impl ToCuda for CudaReverseSparseMultilinearExtension {
    type CudaRepr = icicle_core::sisu::ReverseSparseMultilinearExtension;

    fn to_cuda(&self) -> icicle_core::sisu::ReverseSparseMultilinearExtension {
        icicle_core::sisu::ReverseSparseMultilinearExtension {
            size: self.size,
            subset_num_vars: self.subset_num_vars,
            real_num_vars: self.real_num_vars,
            point_subset: self.point_subset.as_ptr(),
            point_real: self.point_real.as_ptr(),
            subset_position: self.subset_position.as_ptr(),
            real_position: self.real_position.as_ptr(),
        }
    }
}

pub struct CudaCircuitLayer<F: IcicleConvertibleField> {
    pub layer_index: u8,
    pub num_layers: u8,
    pub size: u32,

    pub constant_ext: HostOrDeviceSliceWrapper<CudaSparseMultilinearExtension<F>>,
    pub mul_ext: HostOrDeviceSliceWrapper<CudaSparseMultilinearExtension<F>>,
    pub forward_x_ext: HostOrDeviceSliceWrapper<CudaSparseMultilinearExtension<F>>,
    pub forward_y_ext: HostOrDeviceSliceWrapper<CudaSparseMultilinearExtension<F>>,
}

impl<F: IcicleConvertibleField> CudaCircuitLayer<F> {
    pub fn from(
        circuit: &GeneralCircuit<F>,
        circuit_params: &CircuitParams<F>,
        layer_index: usize,
        num_layers: usize,
    ) -> Self {
        let z_num_vars = circuit.num_vars_at(layer_index);
        let x_num_vars = circuit.num_vars_at(layer_index + 1);

        Self {
            layer_index: layer_index as u8,
            num_layers: num_layers as u8,
            size: round_to_pow_of_two(circuit.len_at(layer_index)) as u32,
            constant_ext: Self::handle_single_ext(
                &circuit.layer(layer_index).constant_ext,
                circuit_params,
                z_num_vars,
                x_num_vars,
            ),
            mul_ext: Self::handle_single_ext(
                &circuit.layer(layer_index).mul_ext,
                circuit_params,
                z_num_vars,
                x_num_vars,
            ),
            forward_x_ext: Self::handle_single_ext(
                &circuit.layer(layer_index).forward_x_ext,
                circuit_params,
                z_num_vars,
                x_num_vars,
            ),
            forward_y_ext: Self::handle_single_ext(
                &circuit.layer(layer_index).forward_y_ext,
                circuit_params,
                z_num_vars,
                x_num_vars,
            ),
        }
    }

    fn handle_single_ext(
        layer_exts: &LayerExtensions<F>,
        circuit_params: &CircuitParams<F>,
        z_num_vars: usize,
        x_num_vars: usize,
    ) -> HostOrDeviceSliceWrapper<CudaSparseMultilinearExtension<F>> {
        let mut result = vec![];

        for (i, ext) in layer_exts.as_slice().iter().enumerate() {
            let y_num_vars = layer_exts.subset_num_vars_at(&LayerIndex::Relative(i + 1));

            result.push(CudaSparseMultilinearExtension::from(
                ext,
                circuit_params,
                z_num_vars,
                x_num_vars,
                y_num_vars,
            ))
        }

        HostOrDeviceSliceWrapper::new(result).unwrap()
    }

    pub fn to_cuda(&self) -> icicle_core::sisu::Layer<F::IcicleField> {
        icicle_core::sisu::Layer {
            layer_index: self.layer_index,
            num_layers: self.num_layers,
            size: self.size,
            constant_ext: self.constant_ext.as_ptr(),
            mul_ext: self.mul_ext.as_ptr(),
            forward_x_ext: self.forward_x_ext.as_ptr(),
            forward_y_ext: self.forward_y_ext.as_ptr(),
        }
    }
}

pub struct CudaCircuit<F: IcicleConvertibleField> {
    pub num_layers: u8,

    pub layers: Vec<CudaCircuitLayer<F>>,
    pub cuda_layers: Vec<icicle_core::sisu::Layer<F::IcicleField>>,

    pub on_host_subset_num_vars: Vec<Vec<u32>>,
    pub on_host_subset_num_vars_ptr: Vec<*const u32>,
    pub reverse_exts:
        HostOrDeviceSliceWrapper<HostOrDeviceSliceWrapper<CudaReverseSparseMultilinearExtension>>,

    pub num_inputs: usize,
}

unsafe impl<F: IcicleConvertibleField> Sync for CudaCircuit<F> {}
unsafe impl<F: IcicleConvertibleField> Send for CudaCircuit<F> {}

impl<F: IcicleConvertibleField> CudaCircuit<F> {
    pub fn from(circuit: &GeneralCircuit<F>, circuit_params: &CircuitParams<F>) -> Self {
        let mut layers = vec![];

        for layer_index in 0..circuit.len() {
            layers.push(CudaCircuitLayer::from(
                circuit,
                circuit_params,
                layer_index,
                circuit.len(),
            ));
        }

        let mut cuda_layers = vec![];
        for i in 0..layers.len() {
            cuda_layers.push(layers[i].to_cuda());
        }

        let mut reverse_ext = vec![];
        let mut subset_num_vars = vec![];
        for target_layer_index in 0..circuit.len() + 1 {
            let (ext, num_vars) =
                Self::handle_reverse_ext(circuit, circuit_params, target_layer_index);
            reverse_ext.push(ext);
            subset_num_vars.push(num_vars);
        }

        let mut subset_num_vars_ptr = vec![];
        for i in 0..subset_num_vars.len() {
            subset_num_vars_ptr.push(subset_num_vars[i].as_ptr());
        }

        Self {
            num_layers: circuit.len() as u8,
            layers,
            cuda_layers,
            on_host_subset_num_vars: subset_num_vars,
            on_host_subset_num_vars_ptr: subset_num_vars_ptr,
            reverse_exts: HostOrDeviceSliceWrapper::new(reverse_ext).unwrap(),
            num_inputs: circuit.number_inputs,
        }
    }

    fn handle_reverse_ext(
        circuit: &GeneralCircuit<F>,
        circuit_params: &CircuitParams<F>,
        target_layer_index: usize,
    ) -> (
        HostOrDeviceSliceWrapper<CudaReverseSparseMultilinearExtension>,
        Vec<u32>,
    ) {
        let num_vars = circuit.num_vars_at(target_layer_index);

        let reverse_exts = if target_layer_index == circuit.len() {
            circuit.input_subset_reverse_ext.as_slice()
        } else {
            circuit
                .layer(target_layer_index)
                .subset_reverse_ext
                .as_slice()
        };

        let mut device_reverse_ext = vec![];
        let mut subset_num_vars_vec = vec![];

        for ext in reverse_exts {
            let subset_num_vars = if ext.num_vars() == 0 {
                0
            } else {
                ext.num_vars() - num_vars
            };
            device_reverse_ext.push(CudaReverseSparseMultilinearExtension::from(
                ext,
                circuit_params,
                subset_num_vars,
                num_vars,
            ));
            subset_num_vars_vec.push(subset_num_vars as u32);
        }

        (
            HostOrDeviceSliceWrapper::new(device_reverse_ext).unwrap(),
            subset_num_vars_vec,
        )
    }

    pub fn to_cuda(&self) -> icicle_core::sisu::Circuit<F::IcicleField> {
        icicle_core::sisu::Circuit {
            num_layers: self.num_layers,
            layers: self.cuda_layers.as_ptr(),
            on_host_subset_num_vars: self.on_host_subset_num_vars_ptr.as_ptr(),
            reverse_exts: self.reverse_exts.as_ptr(),
        }
    }
}

pub fn new_empty_cuda_circuit_evaluations<F: IcicleConvertibleField>(
    circuit: &CudaCircuit<F>,
    input: &mut [CudaSlice<F>],
) -> HostOrDeviceSlice2D<F::IcicleField> {
    let mut result = vec![];
    for i in 0..circuit.num_layers + 1 {
        let layer_size = if i == circuit.num_layers {
            circuit.num_inputs
        } else {
            circuit.layers[i as usize].size as usize
        };

        result.push(round_to_pow_of_two(layer_size) * input.len());
    }

    let mut device = HostOrDeviceSlice2D::zeros(result).unwrap();

    for (subcircuit_index, sub_input) in input.into_iter().enumerate() {
        assert!(sub_input.len() == circuit.num_inputs);

        device
            .at_mut(circuit.num_layers as usize)
            .device_copy_partially(
                sub_input.as_ref_device(),
                0,
                subcircuit_index * round_to_pow_of_two(circuit.num_inputs),
            )
            .unwrap();
    }

    device
}

pub fn new_empty_cuda_circuit_subset_evaluations<F: IcicleConvertibleField>(
    circuit: &CudaCircuit<F>,
    num_subcircuits: usize,
    layer_index: usize,
) -> HostOrDeviceSlice2D<F::IcicleField> {
    let mut result = vec![];
    for i in layer_index + 1..circuit.num_layers as usize + 1 {
        let layer_size = 1 << circuit.on_host_subset_num_vars[i][layer_index];

        result.push(num_subcircuits * layer_size);
    }

    HostOrDeviceSlice2D::zeros(result).unwrap()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, thread};

    use ark_std::cfg_iter;
    use icicle_bn254::curve::ScalarField as IcicleFrBN254;
    use icicle_core::{
        sisu::{circuit_evaluate, circuit_subset_evaluations},
        traits::FieldImpl,
    };
    use sisulib::{
        circuit::{general_circuit::examples::example_general_circuit, CircuitParams},
        common::round_to_pow_of_two,
        field::FrBN254,
    };

    use crate::{
        cuda_compat::{circuit::new_empty_cuda_circuit_subset_evaluations, slice::CudaSlice},
        icicle_converter::IcicleConvertibleField,
    };

    use super::{new_empty_cuda_circuit_evaluations, CudaCircuit};

    #[test]
    fn test_cuda_circuit() {
        thread::scope(|s| {
            for _ in 0..16 {
                s.spawn(|| {
                    let n_subcircuits = 16;

                    let mut circuit = example_general_circuit::<FrBN254>();
                    circuit.finalize(HashMap::default());

                    let mut witness = vec![];
                    for i in 0..n_subcircuits {
                        let mut sub_witness = vec![];
                        for j in 0..circuit.input_size() {
                            sub_witness.push(FrBN254::from((i * j + 3) as u64));
                        }

                        witness.push(CudaSlice::on_host(sub_witness));
                    }

                    let mut expected_evaluations = vec![];
                    for sub_witness in witness.iter_mut() {
                        let sub_evaluations =
                            circuit.evaluate(&CircuitParams::default(), sub_witness.as_ref_host());
                        expected_evaluations.push(sub_evaluations);
                    }

                    let gpu_circuit = CudaCircuit::from(&circuit, &CircuitParams::default());
                    let mut circuit_evaluations =
                        new_empty_cuda_circuit_evaluations(&gpu_circuit, &mut witness);

                    let cuda_circuit = gpu_circuit.to_cuda();
                    circuit_evaluate(&cuda_circuit, n_subcircuits, &mut circuit_evaluations)
                        .unwrap();

                    for subcircuit_index in 0..n_subcircuits {
                        let mut icicle_output =
                            vec![IcicleFrBN254::zero(); round_to_pow_of_two(circuit.len_at(0))];
                        circuit_evaluations[0]
                            .copy_to_host_partially(
                                &mut icicle_output,
                                subcircuit_index * round_to_pow_of_two(circuit.len_at(0)),
                            )
                            .unwrap();

                        let actual_output: Vec<_> = cfg_iter!(icicle_output)
                            .map(|x| FrBN254::from_icicle(x))
                            .collect();

                        assert_eq!(
                            actual_output,
                            expected_evaluations[subcircuit_index].at_layer(0, true),
                            "subcircuit {}",
                            subcircuit_index
                        );
                    }

                    let mut subset_evaluations =
                        new_empty_cuda_circuit_subset_evaluations(&gpu_circuit, n_subcircuits, 0);
                    circuit_subset_evaluations(
                        &cuda_circuit,
                        n_subcircuits,
                        0,
                        &circuit_evaluations,
                        &mut subset_evaluations,
                    )
                    .unwrap();

                    let mut wrapped_subset_evaluations = vec![];
                    for layer_eval in subset_evaluations.into_iter() {
                        wrapped_subset_evaluations
                            .push(CudaSlice::<FrBN254>::on_device(layer_eval));
                    }

                    for subcircuit_index in 0..n_subcircuits {
                        let expected_subset_evaluations =
                            expected_evaluations[subcircuit_index].w_subset_evaluations(0);

                        let mut actual_subset_evaluations = vec![];

                        for (layer_index, cuda_layer_eval) in
                            wrapped_subset_evaluations.iter().enumerate()
                        {
                            let layer_size = expected_subset_evaluations[layer_index].len();
                            let host_layer_eval = cuda_layer_eval.at_range(
                                subcircuit_index * layer_size..(subcircuit_index + 1) * layer_size,
                            );
                            actual_subset_evaluations.push(host_layer_eval.as_host());
                        }

                        assert_eq!(
                            expected_subset_evaluations, actual_subset_evaluations,
                            "subcircuit {}",
                            subcircuit_index
                        );
                    }
                });
            }
        });
    }
}
