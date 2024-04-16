use std::marker::PhantomData;

use crate::op_assemble_word::OpAssembleWord;
use crate::{circuit_layer::CircuitLayer, circuit_util::CircuitUtil};
use ark_ff::Field;
use circuitlib::constant::WORD_COUNT;
use circuitlib::layer::Layer;
use sisulib::circuit::GateIndex;

pub struct OpAssembleWord2<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> OpAssembleWord2<F> {
    fn construct(
        layers: &mut Vec<CircuitLayer<F>>,
        a_label: &str,
        b_label: &str,
        word_label: &str,
        mt_flag_index: GateIndex,
    ) {
        OpAssembleWord::build(layers, a_label, 1);
        OpAssembleWord::build(layers, b_label, 1);

        let a_word_label = CircuitUtil::word_label(a_label);
        let b_word_label = CircuitUtil::word_label(b_label);
        let a_words = layers[1].get_indexes(&a_word_label);
        let b_words = layers[1].get_indexes(&b_word_label);

        let mut acc_gates = vec![];
        for i in 0..WORD_COUNT {
            let b_mul = Layer::new_mul_gate("", b_words[i], mt_flag_index, 2);
            let a_forward = Layer::new_forward_gate("", F::ONE, a_words[i], 1);
            let a_mul = Layer::new_mul_gate_with_constant(
                "",
                a_words[i],
                mt_flag_index,
                2,
                F::ZERO - F::ONE,
            );

            let acc_gate =
                Layer::new_accumulate(&format!("{word_label}[{i}]"), vec![a_mul, a_forward, b_mul]);
            acc_gates.push(acc_gate);
        }

        layers[2].add_gates(word_label, acc_gates);
    }

    /// If flag = 1 then out = a else out = b.
    pub fn build(
        layers: &mut Vec<CircuitLayer<F>>,
        a_label: &str,
        b_label: &str,
        word_label: &str,
        mt_flag_index: GateIndex,
    ) {
        // construct words in layer1.
        match layers[1].indexes.get(word_label) {
            Some(_) => {
                // We already had this word in the indexes. No need to reconstruct the word.
            }
            None => {
                Self::construct(layers, a_label, b_label, word_label, mt_flag_index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use circuitlib::constant::WORD_COUNT;
    use sisulib::circuit::general_circuit::circuit::GeneralCircuit;
    use sisulib::circuit::general_circuit::layer::GeneralCircuitLayer;
    use sisulib::circuit::{CircuitParams, GateIndex};
    use sisulib::field::FpSisu;

    use crate::circuit_layer::CircuitLayer;
    use crate::constant::NUM_LEN;
    use crate::op_assemble_word2::OpAssembleWord2;
    use ark_ff::{Field, PrimeField};

    fn run_test<F: PrimeField>(flag: F, x: F) -> Vec<F> {
        // generate random 381 bytes.
        let num1 = vec![F::ZERO; NUM_LEN];
        let mut num2 = vec![F::ZERO; NUM_LEN];
        num2[NUM_LEN - 1] = F::from(x);

        let mut layers: Vec<CircuitLayer<F>> = vec![];
        layers.push(CircuitLayer::new(true));
        layers.push(CircuitLayer::new(false));
        layers.push(CircuitLayer::new(false));

        layers[0].add_input_zero_one("a", NUM_LEN);
        layers[0].add_input_zero_one("b", NUM_LEN);
        layers[0].add_input_zero_one("flag", 1);
        layers[0].add_input_zero_one("input0", 1);

        OpAssembleWord2::build(
            &mut layers,
            "a",
            "b",
            "a_or_b",
            GateIndex::Absolute(NUM_LEN * 2),
        );

        let mut circuit = GeneralCircuit::<F>::new(NUM_LEN * 2 + 2);
        for i in 1..layers.len() {
            circuit.push_layer(GeneralCircuitLayer::new(layers[i].all_gates.clone()));
        }
        circuit.finalize(HashMap::default());

        ///////
        let mut input_values = Vec::from(num1);
        input_values.extend(num2);
        input_values.push(flag);
        input_values.push(F::ZERO);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false).to_vec();

        output_evals
    }
    #[test]
    fn build() {
        type F = FpSisu;
        let x = F::from(3u128);
        let output_evals = run_test::<F>(F::ONE, x);
        assert_eq!(WORD_COUNT, output_evals.len());
        assert_eq!(x, output_evals[3]);

        let output_evals = run_test::<F>(F::ZERO, x);
        assert_eq!(WORD_COUNT, output_evals.len());
        assert_eq!(F::ZERO, output_evals[3]);
    }
}
