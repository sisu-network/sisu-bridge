use ark_std::cfg_into_iter;
use icicle_core::sisu::{circuit_evaluate, circuit_subset_evaluations};

use sisulib::{
    circuit::{general_circuit::circuit::GeneralCircuit, CircuitParams},
    field::FrBN254,
};

use crate::{
    cuda_compat::{
        circuit::{
            new_empty_cuda_circuit_evaluations, new_empty_cuda_circuit_subset_evaluations,
            CudaCircuit,
        },
        slice::CudaSlice,
    },
    icicle_converter::IcicleConvertibleField,
};

pub struct CudaCircuitEvaluations<F: IcicleConvertibleField> {
    evaluations: Vec<CudaSlice<F>>,             //  layer_index
    subset_evaluations: Vec<Vec<CudaSlice<F>>>, //  source_layer_index - target_layer_index
}

impl<F: IcicleConvertibleField> CudaCircuitEvaluations<F> {
    pub fn from_host(circuit: &GeneralCircuit<F>, witness: &mut [CudaSlice<F>]) -> Self {
        let mut all_evaluations = vec![];
        let mut all_subset_evaluations = vec![vec![]; circuit.len()];

        for (subcircuit_index, sub_witness) in witness.iter_mut().enumerate() {
            let sub_evaluations =
                circuit.evaluate(&CircuitParams::default(), sub_witness.as_ref_host());
            for layer_index in 0..sub_evaluations.len() {
                if subcircuit_index == 0 {
                    all_evaluations.push(sub_evaluations.w_evaluations(layer_index).to_vec());
                } else {
                    all_evaluations[layer_index]
                        .extend_from_slice(sub_evaluations.w_evaluations(layer_index));
                }
            }

            for source_layer_index in 0..circuit.len() {
                for (target_layer_index, layer_eval) in sub_evaluations
                    .w_subset_evaluations(source_layer_index)
                    .into_iter()
                    .enumerate()
                {
                    if subcircuit_index == 0 {
                        all_subset_evaluations[source_layer_index].push(layer_eval.to_vec());
                    } else {
                        all_subset_evaluations[source_layer_index][target_layer_index]
                            .extend_from_slice(layer_eval);
                    }
                }
            }
        }

        let wrapped_evaluations = cfg_into_iter!(all_evaluations)
            .map(|x| CudaSlice::on_host(x))
            .collect();

        let wrapped_subset_evaluations = cfg_into_iter!(all_subset_evaluations)
            .map(|subset_eval| {
                cfg_into_iter!(subset_eval)
                    .map(|x| CudaSlice::on_host(x))
                    .collect()
            })
            .collect();

        Self {
            evaluations: wrapped_evaluations,
            subset_evaluations: wrapped_subset_evaluations,
        }
    }

    pub fn w_evaluation(&self, layer_index: usize) -> &CudaSlice<F> {
        &self.evaluations[layer_index]
    }

    pub fn w_subset_evaluations(&self, source_layer_index: usize) -> &[CudaSlice<F>] {
        &self.subset_evaluations[source_layer_index]
    }

    pub fn detach(self) -> (Vec<CudaSlice<F>>, Vec<Vec<CudaSlice<F>>>) {
        (self.evaluations, self.subset_evaluations)
    }
}

impl CudaCircuitEvaluations<FrBN254> {
    pub fn from_device(circuit: &CudaCircuit<FrBN254>, witness: &mut [CudaSlice<FrBN254>]) -> Self {
        let mut evaluations = new_empty_cuda_circuit_evaluations(&circuit, witness);

        circuit_evaluate(&circuit.to_cuda(), witness.len(), &mut evaluations).unwrap();

        let mut wrapped_subset_evaluations = vec![vec![]; circuit.num_layers as usize];
        for source_layer_index in 0..circuit.num_layers as usize {
            let mut subset_evaluations = new_empty_cuda_circuit_subset_evaluations(
                &circuit,
                witness.len(),
                source_layer_index,
            );

            circuit_subset_evaluations(
                &circuit.to_cuda(),
                witness.len(),
                source_layer_index as u8,
                &evaluations,
                &mut subset_evaluations,
            )
            .unwrap();

            for layer_eval in subset_evaluations.into_iter() {
                wrapped_subset_evaluations[source_layer_index]
                    .push(CudaSlice::on_device(layer_eval));
            }
        }

        let mut wrapped_evaluations = vec![];
        for layer_eval in evaluations.into_iter() {
            wrapped_evaluations.push(CudaSlice::on_device(layer_eval));
        }

        Self {
            evaluations: wrapped_evaluations,
            subset_evaluations: wrapped_subset_evaluations,
        }
    }
}
