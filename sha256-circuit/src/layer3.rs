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

use crate::{layer::Layer, layer1::Layer1Builder};
use crate::{layer2::Layer2Builder, layer_input::LayerInput};

// This struct represents the first layer relative to the input layer, NOT the actual layer 1
// in the circuit. For example, if the input is at layer 4, this struct represents layer 3 in the
// circuit design.
pub struct Layer3Builder<'a, F: Field> {
    input_layer: &'a LayerInput<F>,
    layer1: &'a Layer1Builder<'a, F>,
    layer2: &'a Layer2Builder<'a, F>,
    pub indexes: HashMap<String, Vec<Vec<GateIndex>>>,
}

impl<'a, F: Field> Layer3Builder<'a, F> {
    pub fn new(
        input_layer: &'a LayerInput<F>,
        layer1: &'a Layer1Builder<'a, F>,
        layer2: &'a Layer2Builder<'a, F>,
    ) -> Self {
        return Self {
            input_layer,
            layer1,
            layer2,
            indexes: HashMap::<String, Vec<Vec<GateIndex>>>::default(),
        };
    }

    fn forward_zero_one(&self, all_gates: &mut Vec<GeneralCircuitGate<F>>) {
        let layer1_indexes = &self.layer1.indexes;
        let gates = &layer1_indexes.get("zero_one_check").unwrap()[0];
        let new_gates = Layer::<F>::new_forward_gate_array("zero_one_check", gates, 2);
        all_gates.extend(new_gates);
    }

    /// Forwards w[0-16], hin, houts word values that will be used to compare with corresponding
    /// values in the next subcircuit. These are the only values that are different from 0 in output
    /// layer.
    fn forward_w_hin_hout_gates(
        &mut self,
        w_i_bits32: &Vec<Vec<GateIndex>>,
        all_gates: &mut Vec<GeneralCircuitGate<F>>,
    ) {
        let input_indexes = &self.input_layer.indexes;
        let coeff128 = Layer::<F>::constant_coeff_128();

        // forward w_i.
        for i in 0..4 {
            let mut indexes: Vec<GateIndex> = vec![];
            let start = i * 4;
            indexes.extend(&w_i_bits32[start]);
            indexes.extend(&w_i_bits32[start + 1]);
            indexes.extend(&w_i_bits32[start + 2]);
            indexes.extend(&w_i_bits32[start + 3]);
            assert_eq!(128, indexes.len());
            let gates = Layer::new_forward_gate_array_with_coeffs("", &indexes, 3, &coeff128);
            let acc_gate = Layer::new_accumulate("forward_w_i", gates);
            all_gates.push(acc_gate);
        }

        // forward h_in.
        let h_i_bits = input_indexes.get(&format!("h_i_bits")).unwrap();
        for i in 0..2 {
            let mut indexes: Vec<GateIndex> = vec![];
            let start = i * 4;
            indexes.extend(&h_i_bits[start]);
            indexes.extend(&h_i_bits[start + 1]);
            indexes.extend(&h_i_bits[start + 2]);
            indexes.extend(&h_i_bits[start + 3]);
            assert_eq!(128, indexes.len());
            let gates = Layer::new_forward_gate_array_with_coeffs("", &indexes, 3, &coeff128);
            let acc_gate = Layer::new_accumulate("forward_hin", gates);
            all_gates.push(acc_gate);
        }

        // forward hout
        let hout_i_bits = input_indexes.get(&format!("hout_i_bits")).unwrap();
        for i in 0..2 {
            let mut indexes: Vec<GateIndex> = vec![];
            let start = i * 4;
            indexes.extend(&hout_i_bits[start][..32]);
            indexes.extend(&hout_i_bits[start + 1][..32]);
            indexes.extend(&hout_i_bits[start + 2][..32]);
            indexes.extend(&hout_i_bits[start + 3][..32]);
            assert_eq!(128, indexes.len());
            let gates = Layer::new_forward_gate_array_with_coeffs("", &indexes, 3, &coeff128);
            let acc_gate = Layer::new_accumulate("forward_hout", gates);
            all_gates.push(acc_gate);
        }
    }

    pub fn build(&mut self, circuit: &mut GeneralCircuit<F>) {
        let input_indexes = &self.input_layer.indexes;
        let layer2_indexes = &self.layer2.indexes;
        let mut all_gates: Vec<GeneralCircuitGate<F>> = vec![];

        let w_i_bits = input_indexes.get(&format!("w_i_bits")).unwrap();
        let w_i_bits32: Vec<Vec<GateIndex>> = w_i_bits.iter().map(|x| x[..32].to_vec()).collect();

        self.forward_w_hin_hout_gates(&w_i_bits32, &mut all_gates);

        self.forward_zero_one(&mut all_gates);

        let coeff_32 = Layer::<F>::constant_coeff_32();
        let negative_coeff_33 = Layer::<F>::constant_negative_coeff_33();
        let negative_coeff_34 = Layer::<F>::constant_negative_coeff_34();
        let negative_coeff_35 = Layer::<F>::constant_negative_coeff_35();
        let constants_k = Layer::<F>::constants_k();

        // w[i] := w[i-16] + s0 + w[i-7] + s1
        let small_sigma0_final = layer2_indexes.get(&format!("small_sigma0_final")).unwrap();
        let small_sigma1_final = layer2_indexes.get(&format!("small_sigma1_final")).unwrap();
        for i in 16..64 {
            // w_i16
            let w_i16_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("w_i16").as_str(),
                &w_i_bits32[i - 16],
                3,
                &coeff_32,
            );

            // w_i7
            let w_i7_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("w_i7").as_str(),
                &w_i_bits32[i - 7],
                3,
                &coeff_32,
            );

            // small_sigma0_final
            let small_sigma0_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("small_sigma0_final").as_str(),
                &small_sigma0_final[i - 16],
                1,
                &coeff_32,
            );

            // small_sigma1_final
            let small_sigma1_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("small_sigma1_final").as_str(),
                &small_sigma1_final[i - 16],
                1,
                &coeff_32,
            );

            let w_i_bits_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("w32").as_str(),
                &w_i_bits[i],
                3,
                &negative_coeff_34,
            );

            let mut nested = Vec::from(w_i16_gates);
            nested.extend(w_i7_gates);
            nested.extend(small_sigma0_final_gates);
            nested.extend(small_sigma1_final_gates);
            nested.extend(w_i_bits_gates);

            let acc_gates = Layer::new_accumulate("w_final_check", nested);
            all_gates.push(acc_gates);
        }

        let a_i_bits = input_indexes.get("a_i_bits").unwrap().clone();
        let e_i_bits = input_indexes.get("e_i_bits").unwrap().clone();
        let hs = Layer::<F>::get_abcdefgh_indexes(input_indexes, 7);
        let ds = Layer::<F>::get_abcdefgh_indexes(input_indexes, 3);

        // (h + big_sigma1 + ch + k + w) + (big_sigma0 + maj) - a = 0
        let big_sigma0_final = layer2_indexes.get(&format!("big_sigma0_final")).unwrap();
        let big_sigma1_final = layer2_indexes.get(&format!("big_sigma1_final")).unwrap();
        let ch_final = layer2_indexes.get(&format!("ch_final")).unwrap();
        let major_final = layer2_indexes.get(&format!("major_final")).unwrap();

        for i in 0..64 {
            // ch_final
            let ch_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("ch_final").as_str(),
                &ch_final[i],
                1,
                &coeff_32,
            );

            // big_sigma0_final
            let big_sigma0_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("big_sigma0_final").as_str(),
                &big_sigma0_final[i],
                1,
                &coeff_32,
            );

            // big_sigma1_final
            let big_sigma1_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("big_sigma1_final").as_str(),
                &big_sigma1_final[i],
                1,
                &coeff_32,
            );

            // major_final
            let major_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("major_final").as_str(),
                &major_final[i],
                1,
                &coeff_32,
            );

            ////////////

            // h gates
            let h_gates = Layer::new_forward_gate_array_with_coeffs("h", &hs[i], 3, &coeff_32);

            // w gate
            let w_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("w_i_bits").as_str(),
                &w_i_bits32[i],
                3,
                &coeff_32,
            );

            // constant k
            let k_gate = Layer::<F>::new_constant_gate_from_layer("k", constants_k[i], 3, 0);

            // a gate
            let a_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("a").as_str(),
                &a_i_bits[i],
                3,
                &negative_coeff_35,
            );

            let mut nested = vec![];
            nested.extend(big_sigma0_final_gates);
            nested.extend(big_sigma1_final_gates);
            nested.extend(ch_final_gates);
            nested.extend(major_final_gates);
            nested.push(k_gate);
            nested.extend(h_gates);
            nested.extend(w_gates);
            nested.extend(a_gates);

            let acc_gates = Layer::new_accumulate("(h+big_sigma1+ch+k+w+big_sigma0+maj)-a", nested);
            all_gates.push(acc_gates);
        }

        // big_sigma1 + ch + d + h + k[i] + w[i] - e[i] = 0
        for i in 0..64 {
            // ch_final
            let ch_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("ch_final").as_str(),
                &ch_final[i],
                1,
                &coeff_32,
            );

            // big_sigma1_final
            let big_sigma1_final_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("big_sigma1_final").as_str(),
                &big_sigma1_final[i],
                1,
                &coeff_32,
            );

            // k
            let k_gate = Layer::<F>::new_constant_gate_from_layer("k", constants_k[i], 3, 0);

            // d
            let d_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("d").as_str(),
                &ds[i],
                3,
                &coeff_32,
            );

            // h
            let h_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("h").as_str(),
                &hs[i],
                3,
                &coeff_32,
            );

            // w gate
            let w_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("w").as_str(),
                &w_i_bits32[i],
                3,
                &coeff_32,
            );

            // e gate
            let e_gates = Layer::<F>::new_forward_gate_array_with_coeffs(
                format!("e").as_str(),
                &e_i_bits[i],
                3,
                &negative_coeff_35,
            );

            let mut nested = vec![];
            nested.extend(big_sigma1_final_gates);
            nested.extend(ch_final_gates);
            nested.push(k_gate);
            nested.extend(d_gates);
            nested.extend(h_gates);
            nested.extend(w_gates);
            nested.extend(e_gates);

            let acc_gates = Layer::new_accumulate("(d+h+S1+ch+k[i]+w[i])-e", nested);
            all_gates.push(acc_gates);
        }

        // hout[i] = h_original[i] + abcdefgh[i]
        let mut abcdefgh_bits = input_indexes.get(&format!("h_i_bits")).unwrap().clone();
        let hout_i_bits = input_indexes.get(&format!("hout_i_bits")).unwrap().clone();
        let h_original = input_indexes.get(&format!("h_i_bits")).unwrap().clone();
        for i in 0..64 {
            Layer::<F>::update_abcdefgh(i, &mut abcdefgh_bits, &e_i_bits, &a_i_bits);
        }
        for i in 0..8 {
            let hout_i_bits32_gates = Layer::new_forward_gate_array_with_coeffs(
                "houts",
                &hout_i_bits[i],
                3,
                &negative_coeff_33,
            );
            let h_original_gates = Layer::new_forward_gate_array_with_coeffs(
                "h_original",
                &h_original[i],
                3,
                &coeff_32,
            );
            let abcdefgh_gates = Layer::new_forward_gate_array_with_coeffs(
                "abcdefgh",
                &abcdefgh_bits[i],
                3,
                &coeff_32,
            );

            let mut nested = vec![];
            nested.extend(hout_i_bits32_gates);
            nested.extend(h_original_gates);
            nested.extend(abcdefgh_gates);

            let acc_gates = Layer::new_accumulate("houts-(h_original+final_abcdefgh)", nested);
            all_gates.push(acc_gates);
        }

        // input0
        let input0 = input_indexes.get("input0").unwrap()[0][0];
        all_gates.push(Layer::new_forward_gate(
            "input0",
            F::ONE,
            (LayerIndex::Relative(3), input0),
        ));

        println!("Layer3: all_gates len = {}", all_gates.len());

        circuit.push_layer(GeneralCircuitLayer::new(all_gates));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ark_ff::Field;
    use sisulib::{circuit::CircuitParams, field::FrBN254};

    use crate::{
        layer1::Layer1Builder,
        layer2::Layer2Builder,
        layer3::Layer3Builder,
        test_util::tests::{assign_input_values, get_circuit_and_input_layer},
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

        // Layer 3
        let mut layer3_builder =
            Layer3Builder::<F>::new(&input_layer, &layer1_builder, &layer2_builder);
        layer3_builder.build(&mut circuit);

        circuit.finalize(HashMap::default());

        ////////////////////////////////////
        let a = "Hello".as_bytes();
        let input_values = assign_input_values(a, &input_layer);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false);

        // The first 8 output gates are w, hi & hout used to prove subcircuits connection.
        for i in 8..output_evals.len() {
            assert_eq!(F::ZERO, output_evals[i]);
        }
    }
}
