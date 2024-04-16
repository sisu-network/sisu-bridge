use circuitlib::{
    constant::{WORD_COUNT, WORD_SIZE},
    layer::Layer,
};
use std::marker::PhantomData;

use ark_ff::Field;
use sisulib::circuit::{general_circuit::gate::GeneralCircuitGate, GateIndex};

use crate::{
    circuit_layer::CircuitLayer, circuit_util::CircuitUtil, constant::NUM_LEN, op_add::OpAdd,
    op_add_input_num::OpAddInputNum, op_assemble_word::OpAssembleWord,
    op_assemble_word2::OpAssembleWord2, op_eq::OpEq, op_lt::OpLt, op_mul::OpMul,
};

pub struct CircuitBuilder<F: Field> {
    __f: PhantomData<F>,
    layers: Vec<CircuitLayer<F>>,
}

impl<F: Field> CircuitBuilder<F> {
    pub fn new() -> Self {
        Self {
            __f: PhantomData::default(),
            layers: vec![],
        }
    }

    fn core_input(&mut self) {
        let layer0 = self.layers.get_mut(0).unwrap();
        let core_labels = ["p", "q", "r"];
        for core_label in core_labels {
            let label = &format!("{core_label}_x");
            OpAddInputNum::build(label, layer0, NUM_LEN, true);
        }

        // y1 and y2 for each core labels.
        for core_label in core_labels {
            let labels = [
                format!("{core_label}_y1"),
                format!("{core_label}_y2"),
                format!("{core_label}_y"),
            ];
            for label in labels {
                OpAddInputNum::build(&label, layer0, NUM_LEN, false);
            }
        }

        let labels = &CircuitUtil::num384_labels();
        for label in labels {
            OpAddInputNum::build(&label, layer0, NUM_LEN, false);
        }

        // fp
        OpAddInputNum::build("fp", layer0, NUM_LEN, false);

        // input0.
        layer0.insert_indexes("input0", vec![GateIndex::Absolute(layer0.index)]);

        // input1.
        layer0.insert_indexes("input1", vec![GateIndex::Absolute(layer0.index)]);
    }

    fn verify_y_selection(&mut self, inout_label: &str) {
        let flag_indexes = &self.layers[0]
            .get_indexes(&format!("{inout_label}_x_flags"))
            .clone();
        let y_label = &format!("{inout_label}_y");
        let word_label = &CircuitUtil::word_label(y_label);
        OpAssembleWord2::build(
            &mut self.layers,
            &format!("{inout_label}_y1"),
            &format!("{inout_label}_y2"),
            y_label,
            flag_indexes[2],
        );

        OpAssembleWord::build(&mut self.layers, y_label, 1);

        let mut acc_gates = vec![];
        for i in 0..WORD_COUNT {
            let layer1_index = self.layers[1].get_indexes(word_label)[i];
            let layer2_index = self.layers[2].get_indexes(y_label)[i];
            let gate1 = Layer::new_forward_gate("", F::ONE, layer1_index, self.layers.len() - 2);
            let gate2 =
                Layer::new_forward_gate("", F::ZERO - F::ONE, layer2_index, self.layers.len() - 3);
            let acc_gate =
                Layer::new_accumulate(&format!("{inout_label}_y_check"), vec![gate1, gate2]);
            acc_gates.push(acc_gate);
        }

        let len = self.layers.len();
        self.layers[len - 1].add_gates(&format!("{inout_label}_y_check"), acc_gates);
    }

    // Forward p, q, r words.
    fn forward_pqr_words(&mut self) {
        let inout_labels = vec!["p", "q", "r"];
        let layer_len = self.layers.len();
        let coeffs96 = Layer::<F>::coeffs96();

        let mut acc_gates = vec![];
        for inout_label in inout_labels {
            let flag_indexes = self.layers[0].get_indexes(&format!("{inout_label}_x_flags"));
            let x_value = self.layers[0].get_indexes(&format!("{inout_label}_x"));
            let mut indexes: Vec<GateIndex> = vec![];
            indexes.extend(flag_indexes);
            indexes.extend(x_value);

            for i in 0..4 {
                let start = i * WORD_SIZE;
                let sub_indexes = indexes[start..start + WORD_SIZE].to_vec();
                let gates = Layer::new_forward_gate_array_with_coeffs(
                    "",
                    &sub_indexes,
                    layer_len - 1,
                    &coeffs96,
                );
                let acc_gate = Layer::new_accumulate(&format!("forward_{inout_label}[{i}]"), gates);
                acc_gates.push(acc_gate);
            }
        }
        assert_eq!(12, acc_gates.len());

        self.layers[layer_len - 1].add_gates("forward_words", acc_gates);
    }

    fn zero_one_check(&mut self) {
        let indexes = self.layers[0]
            .zero_one_indexes()
            .iter()
            .map(|x| GateIndex::Absolute(*x))
            .collect::<Vec<GateIndex>>();

        let zero_one_label = "zero_one_check";
        let new_gates = Layer::new_xor_gate_array(zero_one_label, &indexes, &indexes, 1);
        self.layers[1].add_gates(zero_one_label, new_gates);

        // Forward  all these values to the output layer.
        let indexes = self.layers[1].get_indexes(zero_one_label);
        let layer_len = self.layers.len();
        let new_gates = Layer::new_forward_gate_array(zero_one_label, indexes, layer_len - 2);
        self.layers[layer_len - 1].add_gates(zero_one_label, new_gates);
    }

    fn check_constant_zero_one(&mut self) {
        let layer_len = self.layers.len();

        let index = self.layers[0].get_indexes("input0")[0];
        let gate = Layer::new_forward_gate("input0", F::ONE, index, layer_len - 1);
        self.layers[layer_len - 1].add_gates("input0", vec![gate]);

        let index = self.layers[0].get_indexes("input1")[0];
        let one_gate = Layer::new_forward_gate("input1", F::ONE, index, layer_len - 1);
        let minus1_gate = Layer::new_constant_gate("minus1", F::ZERO - F::ONE);
        let acc_gate = Layer::new_accumulate("input1", vec![one_gate, minus1_gate]);
        self.layers[layer_len - 1].add_gates("input1", vec![acc_gate]);
    }

    fn check_fp(&mut self) {
        // 8047903782086192180586325942
        // 20826981314825584179608359615
        // 31935979117156477062286671870
        // 54880396502181392957329877675
        OpAssembleWord::build(&mut self.layers, "fp", 1);
        let fp_indexes = self.layers[1].get_indexes("fp_word");
        assert_eq!(WORD_COUNT, fp_indexes.len());
        let constants = vec![
            F::ZERO - F::from(8047903782086192180586325942u128),
            F::ZERO - F::from(20826981314825584179608359615u128),
            F::ZERO - F::from(31935979117156477062286671870u128),
            F::ZERO - F::from(54880396502181392957329877675u128),
        ];
        let mut acc_gates: Vec<GeneralCircuitGate<F>> = vec![];
        let layer_len = self.layers.len();
        for i in 0..4 {
            let forward_gate =
                Layer::new_forward_gate("fp_word", F::ONE, fp_indexes[i], layer_len - 2);
            let const_gate = Layer::new_constant_gate("fp_word", constants[i]);

            let nested = vec![forward_gate, const_gate];
            let acc_gate = Layer::new_accumulate(format!("fp_word_{i}").as_str(), nested);
            acc_gates.push(acc_gate);
        }
        self.layers[layer_len - 1].add_gates("fp_check", acc_gates);
    }

    pub fn build(&mut self) {
        // This circuit has n layers (including input)
        for i in 0..5 {
            self.layers.push(CircuitLayer::new(i == 0));
        }

        self.core_input();

        // Prove that r_x < fp.
        OpLt::build(&mut self.layers, "r_x", "fp");
        // Prove p_y1 < p_y2
        OpLt::build(&mut self.layers, "p_y1", "p_y2");
        // Prove q_y1 < q_y2
        OpLt::build(&mut self.layers, "q_y1", "q_y2");
        // Prove r_y1 < r_y2
        OpLt::build(&mut self.layers, "r_y1", "r_y2");

        // Prove that p_x != q_x
        OpEq::build(&mut self.layers, "p_x", "q_x");

        // Prove that p_y = if bigger_flag { p_y2 } else { p_y1 };
        self.verify_y_selection("p");
        self.verify_y_selection("q");
        self.verify_y_selection("r");

        // Verify multiplication.
        OpMul::build(&mut self.layers, "div_y_x_2", "div_y_x", "div_y_x", 2);
        OpMul::build(&mut self.layers, "mul_y_x_pr", "div_y_x", "diff_x_pr", 2);
        OpMul::build(&mut self.layers, "diff_y_qp", "div_y_x", "diff_x_qp", 2);

        // Verify add.
        OpAdd::build(&mut self.layers, "q_x", "p_x", "diff_x_qp", 2);
        OpAdd::build(&mut self.layers, "p_x", "r_x", "diff_x_pr", 2);
        OpAdd::build(&mut self.layers, "mul_y_x_pr", "p_y", "r_y", 2);
        OpAdd::build(&mut self.layers, "q_y", "p_y", "diff_y_qp", 2);
        OpAdd::build(&mut self.layers, "div_y_x_2_minus_p_x", "r_x", "q_x", 2);
        OpAdd::build(
            &mut self.layers,
            "div_y_x_2",
            "p_x",
            "div_y_x_2_minus_p_x",
            2,
        );

        self.check_constant_zero_one();

        // This should be the last operation to build when all the input values are added.
        self.zero_one_check();

        self.check_fp();

        println!("Input size = {}", self.layers[0].len());
    }
}

#[cfg(test)]
mod tests {
    use sisulib::{circuit::CircuitParams, field::FpSisu};

    use crate::test_util::tests::{assign_input_values, circuit_from_layers};

    use super::CircuitBuilder;
    use ark_ff::Field;

    #[test]
    fn build() {
        type F = FpSisu;
        let mut circuit_builder = CircuitBuilder::<F>::new();
        circuit_builder.build();

        let circuit = circuit_from_layers(&circuit_builder.layers);

        ////////
        let input_values = assign_input_values(&circuit_builder.layers[0]).0;
        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false).to_vec();

        println!("output_evals len = {}", output_evals.len());

        for i in 12..output_evals.len() {
            if output_evals[i] != F::ZERO {
                let len = circuit_builder.layers.len();
                let layer = &circuit_builder.layers[len - 1];
                println!(
                    "Test failed at gate label = {}, i = {}",
                    (layer.all_gates[i]).clone().label(),
                    i
                )
            }
            assert_eq!(F::ZERO, output_evals[i]);
        }
    }
}
