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

use crate::layer_input::LayerInput;
use crate::{layer::Layer, layer1::Layer1Builder};

// This struct represents the first layer relative to the input layer, NOT the actual layer 1
// in the circuit. For example, if the input is at layer 4, this struct represents layer 3 in the
// circuit design.
pub struct Layer2Builder<'a, F: Field> {
    input_layer: &'a LayerInput<F>,
    layer1: &'a Layer1Builder<'a, F>,
    pub indexes: HashMap<String, Vec<Vec<GateIndex>>>,
}

impl<'a, F: Field> Layer2Builder<'a, F> {
    pub fn new(input_layer: &'a LayerInput<F>, layer1: &'a Layer1Builder<'a, F>) -> Self {
        return Self {
            input_layer,
            layer1,
            indexes: HashMap::<String, Vec<Vec<GateIndex>>>::default(),
        };
    }

    pub fn build(&mut self, circuit: &mut GeneralCircuit<F>) {
        let input_indexes = &self.input_layer.indexes;
        let layer1_indexes = &self.layer1.indexes;
        let mut all_gates: Vec<GeneralCircuitGate<F>> = vec![];
        let w_i_bits = input_indexes.get(&format!("w_i_bits")).unwrap();
        let input0 = input_indexes.get(&format!("input0")).unwrap()[0][0];

        // small_sigma0_final
        let small_sigma0_step1s = layer1_indexes.get("small_sigma0_step1").unwrap();
        Layer::add_new_gates(
            "small_sigma0_final",
            &mut all_gates,
            &mut self.indexes,
            |label: &str, i: usize| {
                if i <= 15 {
                    return vec![];
                }

                let small_sigma0_step1 = &small_sigma0_step1s[i - 16];
                let wi15 = w_i_bits[i - 15][..32].to_vec();
                let w15_shf3 = Layer::<F>::new_right_shift(&wi15, input0, 3);

                Layer::<F>::new_xor_gate_array(
                    &format!("{label}[{i}]"),
                    &small_sigma0_step1,
                    &w15_shf3,
                    LayerIndex::Relative(2),
                )
            },
        );

        // small_sigma1_final
        let small_sigma1_step1s = layer1_indexes.get("small_sigma1_step1").unwrap();
        Layer::add_new_gates(
            "small_sigma1_final",
            &mut all_gates,
            &mut self.indexes,
            |label: &str, i: usize| {
                if i <= 15 {
                    return vec![];
                }

                let small_sigma1_step1 = &small_sigma1_step1s[i - 16];
                let wi2 = w_i_bits[i - 2][..32].to_vec();
                let w2_shf10 = Layer::<F>::new_right_shift(&wi2, input0, 10);

                Layer::<F>::new_xor_gate_array(
                    &format!("{label}[{i}]"),
                    &small_sigma1_step1,
                    &w2_shf10,
                    LayerIndex::Relative(2),
                )
            },
        );

        // big_sigma0_final
        let big_sigma0_step1s = layer1_indexes.get("big_sigma0_step1").unwrap();
        Layer::with_shuffled_abcdefgh(
            "big_sigma0_final",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, i: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let rrt22 = Layer::<F>::right_rotate(&abcdefgh_bits[0], 22);
                Layer::<F>::new_xor_gate_array(
                    label,
                    &big_sigma0_step1s[i],
                    &rrt22,
                    LayerIndex::Relative(2),
                )
            },
        );

        // big_sigma1_final
        let big_sigma1_step1s = layer1_indexes.get("big_sigma1_step1").unwrap();
        Layer::with_shuffled_abcdefgh(
            "big_sigma1_final",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, i: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let rrt25 = Layer::<F>::right_rotate(&abcdefgh_bits[4], 25);
                Layer::<F>::new_xor_gate_array(
                    label,
                    &big_sigma1_step1s[i],
                    &rrt25,
                    LayerIndex::Relative(2),
                )
            },
        );

        // major_final
        // b_c[k] <== b[k]*c[k];
        // out[k] <== a[k] * (b[k]+c[k]-2*b_c[k]) + b_c[k];
        let ab = layer1_indexes.get("major_a_and_b_step1").unwrap();
        let ac = layer1_indexes.get("major_a_and_c_step1").unwrap();
        let bc = layer1_indexes.get("major_b_and_c_step1").unwrap();
        Layer::with_shuffled_abcdefgh(
            "major_final",
            &mut all_gates,
            &input_indexes,
            &mut self.indexes,
            |label: &str, i: usize, abcdefgh_bits: &Vec<Vec<GateIndex>>| {
                let ab_forward = Layer::<F>::new_forward_gate_array("ab", &ab[i], 1);
                let ac_forward = Layer::<F>::new_forward_gate_array("ac", &ac[i], 1);
                let bc_forward = Layer::<F>::new_forward_gate_array("bc", &bc[i], 1);

                let abc_mul = Layer::<F>::new_mul_gate_array_with_constant(
                    "abc_mul",
                    &bc[i],
                    &abcdefgh_bits[0],
                    2,
                    F::ZERO - F::from(2u128),
                );

                let nested = vec![ab_forward, ac_forward, bc_forward, abc_mul];
                Layer::new_accumulates(label, &nested)
            },
        );

        // ch_final
        let ch_e_and_fs = layer1_indexes.get("ch_e_and_f").unwrap();
        let ch_not_e_and_gs = layer1_indexes.get("ch_not_e_and_g").unwrap();
        Layer::add_new_gates(
            "ch_final",
            &mut all_gates,
            &mut self.indexes,
            |label: &str, i: usize| {
                let ch_e_and_f = &ch_e_and_fs[i];
                let ch_not_e_and_g = &ch_not_e_and_gs[i];

                Layer::<F>::new_xor_gate_array(
                    &format!("{label}[{i}]"),
                    &ch_e_and_f,
                    &ch_not_e_and_g,
                    LayerIndex::Relative(1),
                )
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
        layer2::Layer2Builder,
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

        // Layer 2
        let mut layer2_builder = Layer2Builder::<F>::new(&input_layer, &layer1_builder);
        layer2_builder.build(&mut circuit);

        circuit.finalize(HashMap::default());

        ////////////////////////////////////

        let a = "Hello".as_bytes();
        let input_values = assign_input_values(a, &input_layer);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false);

        let check_labels2 = vec![
            "small_sigma0_final",
            "small_sigma1_final",
            "big_sigma0_final",
            "big_sigma1_final",
            "ch_final",
        ];
        check_output_by_labels("fixtures/group2.txt", check_labels2, &circuit, output_evals);

        let check_labels3 = vec!["major_final"];
        check_output_by_labels("fixtures/group3.txt", check_labels3, &circuit, output_evals);
    }
}
