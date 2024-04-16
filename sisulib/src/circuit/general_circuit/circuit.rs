use std::collections::HashMap;

use ark_ff::Field;

use crate::circuit::{CircuitGateType, CircuitParams, GateF};
use crate::codegen::generator::FuncGenerator;
use crate::common::{ilog2_ceil, padding_pow_of_two_size};

use super::gate::{GeneralCircuitGate, LayerIndex};
use super::layer::{GeneralCircuitLayer, SubsetReverseExtensions};

pub struct GeneralCircuitEvaluations<F: Field> {
    evaluations: Vec<Vec<F>>,
    layer_size: Vec<usize>,

    // source_layer_index - target_layer_index - evaluations
    subset_evaluations: Vec<Vec<Vec<F>>>,
}

impl<F: Field> GeneralCircuitEvaluations<F> {
    pub fn from(
        circuit: &GeneralCircuit<F>,
        circuit_params: &CircuitParams<F>,
        witness: &[F],
    ) -> Self {
        let mut circuit_evaluations = Self {
            evaluations: vec![vec![]; circuit.len()],
            layer_size: vec![0; circuit.len()],
            subset_evaluations: vec![vec![]; circuit.len()],
        };

        let mut input_layer = witness.to_vec();
        padding_pow_of_two_size(&mut input_layer);

        circuit_evaluations.evaluations.push(input_layer);
        circuit_evaluations.layer_size.push(witness.len());

        for i in 0..circuit.len() {
            circuit_evaluations.subset_evaluations[i] = vec![vec![]; circuit.len() - i];
        }

        // Init subset evaluations.
        for target_layer_index in 0..circuit.len() + 1 {
            for source_layer_index in 0..target_layer_index {
                let relative_target_layer_index =
                    LayerIndex::Relative(target_layer_index - source_layer_index);

                let subset_num_vars = circuit
                    .layer(source_layer_index)
                    .constant_ext
                    .subset_num_vars_at(&relative_target_layer_index);

                circuit_evaluations.subset_evaluations[source_layer_index]
                    [relative_target_layer_index.value_as_vector_index()] =
                    vec![F::ZERO; 2usize.pow(subset_num_vars as u32)];
            }
        }

        for (layer_index, layer) in circuit.layers.iter().enumerate().rev() {
            assert!(
                layer.is_finalized,
                "The layer must be finalized before evaluating"
            );

            circuit_evaluations.evaluate_layer(layer_index, layer, &circuit_params);
        }

        circuit_evaluations
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn at_layer(&self, layer_index: usize, padding: bool) -> &[F] {
        if padding {
            &self.evaluations[layer_index]
        } else {
            &self.evaluations[layer_index][..self.layer_size[layer_index]]
        }
    }

    fn evaluate_layer(
        &mut self,
        current_layer_index: usize,
        layer: &GeneralCircuitLayer<F>,
        params: &CircuitParams<F>,
    ) {
        for gate in layer.gates.iter() {
            let output = self.evaluate_gate(current_layer_index, gate, params);
            self.evaluations[current_layer_index].push(output);
            self.set_subset_evaluation(current_layer_index, gate);
        }

        padding_pow_of_two_size(&mut self.evaluations[current_layer_index]);
        self.layer_size[current_layer_index] = layer.len();
    }

    fn set_subset_evaluation(&mut self, current_layer_index: usize, gate: &GeneralCircuitGate<F>) {
        match &gate.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    self.set_subset_evaluation(current_layer_index, subgate);
                }
            }
            _ => {
                let right_value = self.evaluations
                    [gate.right.0.absolute_value(current_layer_index)][gate.right.1.value()];

                self.subset_evaluations[current_layer_index]
                    [gate.right.0.value_as_vector_index()][gate.right_subset_index] = right_value;
            }
        }
    }

    fn evaluate_gate(
        &self,
        current_layer_index: usize,
        gate: &GeneralCircuitGate<F>,
        params: &CircuitParams<F>,
    ) -> F {
        let in1 = self.evaluations[current_layer_index + 1][gate.left.value()];
        let in2 = self.evaluations[gate.right.0.absolute_value(current_layer_index)]
            [gate.right.1.value()];

        let output = match &gate.gate_type {
            CircuitGateType::Constant(constant) => constant.to_value(params),
            CircuitGateType::Add => in1 + in2,
            CircuitGateType::Sub => in1 - in2,
            CircuitGateType::CAdd(c) => c.to_value(params) + in1,
            CircuitGateType::CSub(c) => c.to_value(params) - in1,
            CircuitGateType::Mul(constant) => in1 * in2 * constant.to_value(params),
            CircuitGateType::Xor(constant) => {
                (in1 + in2 - F::from(2u64) * in1 * in2) * constant.to_value(params)
            }
            CircuitGateType::NAAB => (F::ONE - in1) * in2,
            CircuitGateType::Fourier(domain) => in1 + in2 * domain.to_value(params),
            CircuitGateType::ForwardX(constant) => in1 * constant.to_value(params),
            CircuitGateType::ForwardY(constant) => in2 * constant.to_value(params),
            CircuitGateType::Accumulation(accumulated_gates) => {
                let mut acc = F::ZERO;
                for g in accumulated_gates {
                    let value = self.evaluate_gate(current_layer_index, g, params);
                    acc += value;
                }

                acc
            }
            CircuitGateType::Zero => F::ZERO,
        };

        output
    }

    pub fn w_evaluations(&self, layer_index: usize) -> &[F] {
        self.at_layer(layer_index, true)
    }

    pub fn w_subset_evaluations(&self, source_layer_index: usize) -> &[Vec<F>] {
        &self.subset_evaluations[source_layer_index]
    }
}

#[derive(Clone)]
pub struct GeneralCircuit<F: Field> {
    pub layers: Vec<GeneralCircuitLayer<F>>, // The first layer is output, the last layer is input.
    pub number_inputs: usize,
    pub input_subset_reverse_ext: SubsetReverseExtensions<F>,
}

impl<F: Field> GeneralCircuit<F> {
    pub const fn default() -> Self {
        Self::new(0)
    }

    pub const fn new(number_inputs: usize) -> Self {
        Self {
            layers: vec![],
            number_inputs,
            input_subset_reverse_ext: SubsetReverseExtensions::default(),
        }
    }

    pub fn from_layers(layers: Vec<GeneralCircuitLayer<F>>, number_inputs: usize) -> Self {
        Self {
            layers,
            number_inputs,
            input_subset_reverse_ext: SubsetReverseExtensions::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn len_at(&self, layer_index: usize) -> usize {
        if layer_index == self.layers.len() {
            return self.number_inputs;
        }
        self.layers[layer_index].len()
    }

    pub fn input_size(&self) -> usize {
        self.number_inputs
    }

    pub fn num_vars_at(&self, layer_index: usize) -> usize {
        ilog2_ceil(self.len_at(layer_index))
    }

    pub fn layer(&self, layer_index: usize) -> &GeneralCircuitLayer<F> {
        &self.layers[layer_index]
    }

    /// Replace a current layer to a new one.
    pub fn replace(&mut self, layer_index: usize, layer: GeneralCircuitLayer<F>) {
        self.layers[layer_index] = layer.clone();
    }

    /// Push a new layer into circuit as output layer.
    pub fn push_layer(&mut self, layer: GeneralCircuitLayer<F>) {
        // Create a dummy layer to replace.
        self.layers.insert(0, GeneralCircuitLayer::default());
        self.replace(0, layer);
    }

    /// Append the current circuit by another circuit.
    pub fn push_circuit(&mut self, other: Self) {
        assert!(
            self.len() == 0 || self.len_at(0) >= other.number_inputs,
            "output of previous circuit must large than input of the current circuit ({} < {})",
            self.len_at(0),
            other.number_inputs,
        );

        if self.len() == 0 {
            self.number_inputs += other.number_inputs;
        }

        let new_len = self.len() + other.len();
        let other_len = other.len();
        for (i, mut layer) in other.layers.into_iter().enumerate().rev() {
            for gate in layer.gates.iter_mut() {
                gate.finalize_left(0);
                gate.finalize_right_layer(new_len, i);

                // When we push a circuit onto another one, the size of layers
                // doesn't change, so we don't need save them before push layer.
                //
                // However, the layer index is changed, we need adjust it before
                // extract the layer size.
                let right_target_layer_index = gate.right.0.absolute_value(i) - i - 1;

                // If the right_target_layer_index is one of new added layers,
                // the layer size will be zero, otherwise, we get the info from
                // the current circuit.
                let right_layer_size = if right_target_layer_index < other_len - i {
                    0
                } else {
                    self.len_at(right_target_layer_index)
                };

                gate.finalize_right_gate(right_layer_size);
            }

            self.push_layer(layer);
        }
    }

    /// Extend a circuit by another circuit.
    pub fn extend_by_circuit(
        &mut self,
        mut extended_layer_index: usize,
        other: Self,
        extend_input: bool,
    ) {
        assert!(
            extended_layer_index == other.len(),
            "Only support extending a circuit with length of {} at layer {}",
            extended_layer_index,
            extended_layer_index
        );

        if extended_layer_index == self.len() {
            if extend_input {
                self.number_inputs += other.number_inputs;
            } else {
                assert!(
                    self.number_inputs == other.number_inputs,
                    "Two circuits has different inputs"
                )
            }

            extended_layer_index -= 1;
        }

        let total_layers = self.len();

        // We need to determine the layer size here because when we adjust the
        // gate, the self object is inaccessible for get layer size.
        //
        // Moreover, when we extend a circuit, the layer size is changed, so we
        // also need to save the old layer size.
        let mut old_layer_size = vec![];
        for layer_index in 0..self.len() + 1 {
            old_layer_size.push(self.len_at(layer_index));
        }

        for (i, mut extended_layer) in other.layers.into_iter().enumerate() {
            let layer_index = extended_layer_index - i;
            for gate in extended_layer.gates.iter_mut().rev() {
                gate.finalize_left(old_layer_size[layer_index + 1]);
                gate.finalize_right_layer(total_layers, layer_index);

                let right_target_layer_index = gate.right.0.absolute_value(layer_index);
                gate.finalize_right_gate(old_layer_size[right_target_layer_index]);
            }

            let mut new_layer = GeneralCircuitLayer::default();
            new_layer.add_gates(self.layers[layer_index].gates.clone());
            new_layer.add_gates(extended_layer.gates);

            self.replace(layer_index, new_layer);
        }
    }

    pub fn make_subcircuit(&mut self) {
        let total_layers = self.len();
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            for gate in layer.gates.iter_mut() {
                gate.finalize_right_layer_in_subcircuit(total_layers, layer_index);
            }
        }
    }

    /// Finalize a layer to build mle for that layer.
    pub fn finalize_layer(
        &mut self,
        layer_index: usize,
        preset_subset_info: HashMap<LayerIndex, HashMap<usize, usize>>,
    ) {
        let total_layers = self.len();

        let left_input_num_vars = self.num_vars_at(layer_index + 1);
        self.layers[layer_index].finalize(
            total_layers,
            layer_index,
            left_input_num_vars,
            preset_subset_info,
        );
    }

    /// Finalize a circuit to build mle for all layers.
    pub fn finalize(
        &mut self,
        preset_subset_info: HashMap<usize, HashMap<LayerIndex, HashMap<usize, usize>>>,
    ) {
        self.input_subset_reverse_ext = SubsetReverseExtensions::new(self.len());

        for source_layer_index in (0..self.len()).rev() {
            self.finalize_layer(
                source_layer_index,
                preset_subset_info
                    .get(&source_layer_index)
                    .unwrap_or(&HashMap::new())
                    .clone(),
            );

            // Build the subset reverse extension.
            for gate_index in 0..self.layers[source_layer_index].len() {
                let gate = &self.layers[source_layer_index].gates[gate_index].clone();
                gate.check_input_indices(source_layer_index, self);

                self.build_subset_reverse_extension(source_layer_index, gate);
            }
        }
    }

    pub fn evaluate(
        &self,
        circuit_params: &CircuitParams<F>,
        witness: &[F],
    ) -> GeneralCircuitEvaluations<F> {
        assert_eq!(witness.len(), self.number_inputs);
        GeneralCircuitEvaluations::from(self, circuit_params, witness)
    }

    fn build_subset_reverse_extension(
        &mut self,
        source_layer_index: usize,
        gate: &GeneralCircuitGate<F>,
    ) {
        match &gate.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    self.build_subset_reverse_extension(source_layer_index, subgate);
                }
            }
            _ => {
                let target_layer_index = gate.right.0.absolute_value(source_layer_index);

                let target_gate_index = gate.right.1.value();
                let target_subset_gate_index = gate.right_subset_index;
                let target_subset_num_vars = self.layers[source_layer_index]
                    .constant_ext
                    .subset_num_vars_at(&gate.right.0);
                let num_vars = self.num_vars_at(target_layer_index);

                let subset_reverse_ext = if target_layer_index == self.len() {
                    &mut self.input_subset_reverse_ext
                } else {
                    &mut self.layers[target_layer_index].subset_reverse_ext
                };

                subset_reverse_ext.add_evaluation(
                    source_layer_index,
                    target_subset_num_vars,
                    num_vars,
                    (target_subset_gate_index, target_gate_index, GateF::ONE),
                )
            }
        }
    }

    pub fn gen_code(&self, gkr_index: usize) -> Vec<FuncGenerator<F>> {
        let mut funcs = vec![];

        let mut num_outputs_func =
            FuncGenerator::new("get_general_gkr__num_outputs", vec!["gkr_index"]);
        num_outputs_func.add_number(vec![gkr_index], self.len_at(0));
        funcs.push(num_outputs_func);

        for layer_index in 0..self.layers.len() {
            let f = self.layers[layer_index].gen_code(gkr_index, layer_index);
            funcs.extend(f);

            let mut num_vars_func = FuncGenerator::new(
                "get_general_gkr__single_subcircuit_num_vars",
                vec!["gkr_index", "src_layer_index"],
            );
            num_vars_func.add_number(vec![gkr_index, layer_index], self.num_vars_at(layer_index));
            funcs.push(num_vars_func);

            for i in 0..self.len() - layer_index {
                let subset_num_vars = self
                    .layer(layer_index)
                    .constant_ext
                    .subset_num_vars_at(&LayerIndex::Relative(i + 1));

                let mut y_num_vars_func = FuncGenerator::new(
                    "get_general_gkr__single_subcircuit_subset_num_vars",
                    vec!["gkr_index", "src_layer_index", "dst_layer_index"],
                );
                y_num_vars_func.add_number(
                    vec![gkr_index, layer_index, layer_index + i + 1],
                    subset_num_vars,
                );
                funcs.push(y_num_vars_func);
            }
        }

        let mut last_num_vars_func = FuncGenerator::new(
            "get_general_gkr__single_subcircuit_num_vars",
            vec!["gkr_index", "src_layer_index"],
        );
        last_num_vars_func.add_number(vec![gkr_index, self.len()], self.num_vars_at(self.len()));
        funcs.push(last_num_vars_func);

        let f = self
            .input_subset_reverse_ext
            .gen_code(gkr_index, self.len());
        funcs.extend(f);

        funcs
    }
}

#[cfg(test)]
pub mod tests {
    use std::collections::HashMap;

    use crate::{
        circuit::{general_circuit::examples::example_general_circuit, CircuitParams},
        field::Fp389,
    };

    use super::GeneralCircuit;

    #[test]
    fn test_circuit_evaluation() {
        let mut circuit = example_general_circuit();
        circuit.finalize(HashMap::default());

        let evaluation = circuit.evaluate(
            &CircuitParams::default(),
            &[Fp389::from(1), Fp389::from(2), Fp389::from(3)],
        );

        // check values of output (layer 0)
        assert_eq!(
            evaluation.at_layer(0, false),
            vec![
                Fp389::from(8),
                Fp389::from(36),
                Fp389::from(24),
                Fp389::from(102)
            ]
        );
    }

    #[test]
    fn test_circuit_append() {
        let mut subcircuit = example_general_circuit();
        subcircuit.make_subcircuit();

        let mut circuit = GeneralCircuit::default();
        circuit.push_circuit(subcircuit.clone());
        circuit.push_circuit(subcircuit);
        circuit.finalize(HashMap::default());

        let evaluation = circuit.evaluate(
            &CircuitParams::default(),
            &[Fp389::from(1), Fp389::from(2), Fp389::from(3)],
        );

        // check values of output (layer 0)
        assert_eq!(
            evaluation.at_layer(0, false),
            vec![
                Fp389::from(269),
                Fp389::from(348),
                Fp389::from(133),
                Fp389::from(312)
            ]
        );
    }
}
