use std::collections::HashMap;

use ark_ff::Field;
use sisulib::circuit::{
    general_circuit::{
        circuit::GeneralCircuit,
        gate::{GeneralCircuitGate, LayerIndex},
        layer::GeneralCircuitLayer,
    },
    GateIndex,
};

use crate::layer::Layer;
use crate::layer_input::LayerInput;

// This struct represents the first layer relative to the input layer, NOT the actual layer 1
// in the circuit. For example, if the input is at layer 4, this struct represents layer 3 in the
// circuit design.
pub struct Layer1Builder<'a, F: Field> {
    input_layer: &'a LayerInput<F>,
    pub indexes: HashMap<String, Vec<Vec<GateIndex>>>,
}

impl<'a, F: Field> Layer1Builder<'a, F> {
    pub fn new(input_layer: &'a LayerInput<F>) -> Self {
        return Self {
            input_layer,
            indexes: HashMap::<String, Vec<Vec<GateIndex>>>::default(),
        };
    }

    fn check_zero_one(&mut self, all_gates: &mut Vec<GeneralCircuitGate<F>>) {
        let input_indexes = &self.input_layer.indexes;
        let input_tags = Layer::<F>::input_tags();
        let mut all_indexes = vec![];
        for label in input_tags {
            let gate_indexes = input_indexes.get(&label).unwrap();
            let mut prev_indexes: Vec<GateIndex> = vec![];
            for gates in gate_indexes {
                prev_indexes.extend(gates);
            }

            let new_gates = Layer::new_xor_gate_array(
                &label,
                &prev_indexes,
                &prev_indexes,
                LayerIndex::Relative(1),
            );
            let forward_indexes = Layer::general_circuit_gate_to_index(&new_gates, all_gates.len());
            all_gates.extend(new_gates);
            all_indexes.extend(forward_indexes);
        }

        self.indexes
            .insert("zero_one_check".to_string(), vec![all_indexes]);
    }

    pub fn build(&mut self, circuit: &mut GeneralCircuit<F>) {
        let input_indexes = &self.input_layer.indexes;
        let mut all_gates: Vec<GeneralCircuitGate<F>> = vec![];
        let w_i_bits = input_indexes.get(&format!("w_i_bits")).unwrap();

        // forward zero-one-check.
        self.check_zero_one(&mut all_gates);

        // small_sigma0_step1
        Layer::add_new_gates(
            "small_sigma0_step1",
            &mut all_gates,
            &mut self.indexes,
            |label: &str, i: usize| {
                if i <= 15 {
                    return vec![];
                }

                let wi_15 = &w_i_bits[i - 15][..32].to_vec();
                let w_i15_rrt7 = Layer::<F>::right_rotate(&wi_15, 7);
                let w_i15_rrt18 = Layer::<F>::right_rotate(&wi_15, 18);
                Layer::<F>::new_xor_gate_array(
                    &format!("{label}[{i}]"),
                    &w_i15_rrt7,
                    &w_i15_rrt18,
                    LayerIndex::Relative(1),
                )
            },
        );

        // small_sigma1_step1
        Layer::add_new_gates(
            "small_sigma1_step1",
            &mut all_gates,
            &mut self.indexes,
            |label: &str, i: usize| {
                if i <= 15 {
                    return vec![];
                }

                let wi_2 = &w_i_bits[i - 2][..32].to_vec();
                let w_i2_rrt17 = Layer::<F>::right_rotate(&wi_2, 17);
                let w_i2_rrt19 = Layer::<F>::right_rotate(&wi_2, 19);
                Layer::<F>::new_xor_gate_array(
                    &format!("{label}[{i}]"),
                    &w_i2_rrt17,
                    &w_i2_rrt19,
                    LayerIndex::Relative(1),
                )
            },
        );

        // big_sigma0_step1
        Layer::with_shuffled_abcdefgh(
            "big_sigma0_step1",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, _: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let rrt2 = Layer::<F>::right_rotate(&abcdefgh_bits[0], 2);
                let rrt13 = Layer::<F>::right_rotate(&abcdefgh_bits[0], 13);
                Layer::<F>::new_xor_gate_array(label, &rrt2, &rrt13, LayerIndex::Relative(1))
            },
        );

        // big_sigma1_step1
        Layer::with_shuffled_abcdefgh(
            "big_sigma1_step1",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, _: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let rrt6 = Layer::<F>::right_rotate(&abcdefgh_bits[4], 6);
                let rrt11 = Layer::<F>::right_rotate(&abcdefgh_bits[4], 11);
                Layer::<F>::new_xor_gate_array(label, &rrt6, &rrt11, LayerIndex::Relative(1))
            },
        );

        // major_a_and_b_step1
        Layer::with_shuffled_abcdefgh(
            "major_a_and_b_step1",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, _: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let a_gates = &abcdefgh_bits[0];
                let b_gates = &abcdefgh_bits[1];
                Layer::<F>::new_mul_gate_array(label, &a_gates, &b_gates, LayerIndex::Relative(1))
            },
        );

        // major_a_and_c_step1
        Layer::with_shuffled_abcdefgh(
            "major_a_and_c_step1",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, _: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let a_gates = &abcdefgh_bits[0];
                let c_gates = &abcdefgh_bits[2];
                Layer::<F>::new_mul_gate_array(label, &a_gates, &c_gates, LayerIndex::Relative(1))
            },
        );

        // major_b_and_c_step1
        Layer::with_shuffled_abcdefgh(
            "major_b_and_c_step1",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, _: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let b_gates = &abcdefgh_bits[1];
                let c_gates = &abcdefgh_bits[2];
                Layer::<F>::new_mul_gate_array(label, &b_gates, &c_gates, LayerIndex::Relative(1))
            },
        );

        // ch_e_and_f
        Layer::with_shuffled_abcdefgh(
            "ch_e_and_f",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, _: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let e_gates = &abcdefgh_bits[4];
                let f_gates = &abcdefgh_bits[5];
                Layer::<F>::new_mul_gate_array(label, &e_gates, &f_gates, LayerIndex::Relative(1))
            },
        );

        // ch_not_e_and_g
        Layer::with_shuffled_abcdefgh(
            "ch_not_e_and_g",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, _: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let e_gates = &abcdefgh_bits[4];
                let g_gates = &abcdefgh_bits[6];
                Layer::<F>::new_naab_gate_array(label, &e_gates, &g_gates, LayerIndex::Relative(1))
            },
        );

        circuit.push_layer(GeneralCircuitLayer::new(all_gates));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use sisulib::{circuit::CircuitParams, field::FrBN254};

    use crate::{
        layer1::Layer1Builder,
        test_util::tests::{
            assign_input_values, check_output_by_labels, get_circuit_and_input_layer,
        },
    };

    #[test]
    fn build_layer() {
        type F = FrBN254;
        let circuit_result = get_circuit_and_input_layer::<FrBN254>();
        let mut circuit = circuit_result.0;
        let input_layer = circuit_result.1;

        // Layer 1
        let mut layer1_builder = Layer1Builder::<F>::new(&input_layer);
        layer1_builder.build(&mut circuit);

        circuit.finalize(HashMap::default());

        ////////////////////////////////////

        let a = "Hello".as_bytes();
        let input_values = assign_input_values(a, &input_layer);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false);

        let check_labels1 = vec![
            "big_sigma0_step1",
            "big_sigma1_step1",
            "small_sigma0_step1",
            "small_sigma1_step1",
            "major_a_and_b_step1",
            "major_a_and_c_step1",
            "major_b_and_c_step1",
            "ch_e_and_f",
            "ch_not_e_and_g",
        ];
        check_output_by_labels("fixtures/group1.txt", check_labels1, &circuit, output_evals);
    }
}
