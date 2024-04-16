use std::marker::PhantomData;

use ark_ff::Field;

use crate::op_diff::OpDiff;
use crate::{circuit_layer::CircuitLayer, circuit_util::CircuitUtil};
use circuitlib::constant::WORD_COUNT;
use circuitlib::layer::Layer;

pub struct OpEq<F: Field> {
    __f: PhantomData<F>,
}

/// This operation proves that a != b. We represent a & b as lists of words and show that at least
/// one a[i] - b[i] is not 0.
/// The algorithm to check if a number in is equal to 0 as follow:
///   inv <-- in!=0 ? 1/in : 0;
///   out <== -in*inv +1;
///   in*out === 0;
impl<F: Field> OpEq<F> {
    pub fn build(layers: &mut Vec<CircuitLayer<F>>, a_label: &str, b_label: &str) {
        // construct words in layer1.
        OpDiff::build(layers, a_label, b_label);

        let eq_label = CircuitUtil::eq_label(a_label, b_label);

        // Add the inverse to input.
        layers[0].add_input(&format!("{eq_label}_inv"), WORD_COUNT, false);

        let diff_label = CircuitUtil::diff_label(a_label, b_label);
        let diff_indexes = layers[1].get_indexes(&diff_label);
        let inv_indexes = layers[0].get_indexes(&format!("{eq_label}_inv"));
        let input1 = layers[0].get_indexes("input1")[0];

        let mut output_gates = vec![];
        for i in 0..WORD_COUNT {
            // output = -diff*inv + 1;
            let gate = Layer::new_mul_gate_with_constant(
                "",
                diff_indexes[i],
                inv_indexes[i],
                2,
                F::ZERO - F::ONE,
            );
            let one = Layer::new_forward_gate("", F::ONE, input1, 2);
            let acc_gate = Layer::new_accumulate(&format!("{eq_label}_out[i]"), vec![gate, one]);
            output_gates.push(acc_gate);
        }
        layers[2].add_gates(&format!("{eq_label}_out"), output_gates);

        // check that diff * out === 0 for all words
        let mut mul_gates = vec![];
        let diff_indexes = layers[1].get_indexes(&diff_label);
        let out_indexes = layers[2].get_indexes(&format!("{eq_label}_out"));
        for i in 0..WORD_COUNT {
            // Mul the out and the diff from input layer. Make sure result of mul is 0.
            let gate = Layer::new_mul_gate("", out_indexes[i], diff_indexes[i], 2);
            mul_gates.push(gate);
        }
        layers[3].add_gates(&format!("{eq_label}_mul"), mul_gates);

        // Forward the mul result to the last layer.
        let mul_indexes = layers[3].get_indexes(&format!("{eq_label}_mul"));
        let len = layers.len();
        let gates = Layer::<F>::new_forward_gate_array(
            &format!("{eq_label}_mul"),
            mul_indexes,
            len - 1 - 3,
        );
        layers[len - 1].add_gates(&format!("{eq_label}_mul"), gates);

        // Each out[i]'s value is either 0 or 1. We need to multiply all of them to make sure that
        // they are all 1.
        assert_eq!(4, WORD_COUNT);
        let out_indexes = layers[2].get_indexes(&format!("{eq_label}_out"));
        let out01_gate = Layer::new_mul_gate(
            &format!("{eq_label}_out01"),
            out_indexes[0],
            out_indexes[1],
            1,
        );
        let out23_gate = Layer::new_mul_gate(
            &format!("{eq_label}_out23"),
            out_indexes[2],
            out_indexes[3],
            1,
        );
        layers[3].add_gates(
            &format!("{eq_label}_out_01_23"),
            vec![out01_gate, out23_gate],
        );

        let out_01_23 = layers[3].get_indexes(&format!("{eq_label}_out_01_23"));
        let out01_index = out_01_23[0];
        let out23_index = out_01_23[1];
        let out_gate = Layer::new_mul_gate(
            &format!("{eq_label}_out_final"),
            out01_index,
            out23_index,
            1,
        );
        layers[4].add_gates(&format!("{eq_label}_out_final"), vec![out_gate]);
    }
}

#[cfg(test)]
mod tests {
    use circuitlib::layer::Layer;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use sisulib::circuit::general_circuit::circuit::GeneralCircuit;
    use sisulib::circuit::general_circuit::layer::GeneralCircuitLayer;
    use sisulib::circuit::{CircuitParams, GateIndex};
    use sisulib::field::FpSisu;
    use std::collections::HashMap;

    use crate::circuit_layer::CircuitLayer;
    use crate::circuit_util::CircuitUtil;
    use crate::constant::NUM_LEN;
    use crate::conversion::u8_to_words;
    use crate::op_eq::OpEq;
    use crate::test_util::tests::add_2_nums;
    use ark_ff::Field;
    use circuitlib::constant::WORD_COUNT;

    fn run_test<F: Field>(num1: &Vec<u8>, num2: &Vec<u8>) -> (Vec<F>, Vec<CircuitLayer<F>>) {
        let num_field1 = num1.iter().map(|x| F::from(*x as u128)).collect::<Vec<F>>();
        let num_field2 = num2.iter().map(|x| F::from(*x as u128)).collect::<Vec<F>>();
        let a_label = "a";
        let b_label = "b";

        let mut layers: Vec<CircuitLayer<F>> = vec![];
        add_2_nums(&mut layers, a_label, b_label);
        // layer 2, 3 & 4
        layers.push(CircuitLayer::new(false));
        layers.push(CircuitLayer::new(false));
        layers.push(CircuitLayer::new(false));

        // constant 1
        layers[0].add_input_zero_one("input1", 1);

        OpEq::build(&mut layers, a_label, b_label);
        assert_eq!(WORD_COUNT, layers[1].len());

        let eq_label = CircuitUtil::eq_label(a_label, b_label);
        // This is a dummy gate to work around a bug where there is one single connection from layer i -> j
        let dummy_gate = Layer::new_mul_gate(
            &format!("{eq_label}_dummy"),
            GateIndex::Absolute(0),
            GateIndex::Absolute(0),
            1,
        );
        layers[4].add_gates(&format!("{eq_label}_dummy"), vec![dummy_gate]);

        // Build circuit.
        let mut circuit = GeneralCircuit::<F>::new(2 * NUM_LEN + 2 + WORD_COUNT);
        for i in 1..layers.len() {
            circuit.push_layer(GeneralCircuitLayer::new(layers[i].all_gates.clone()));
        }
        circuit.finalize(HashMap::default());

        ///////
        let a_words_field = u8_to_words::<F>(num1);
        let b_words_field = u8_to_words::<F>(num2);

        let mut input_values = vec![];
        input_values.extend(num_field1);
        input_values.extend(num_field2);
        input_values.push(F::ZERO);
        input_values.push(F::ONE);
        // inverse
        for i in 0..WORD_COUNT {
            let diff = a_words_field[i] - b_words_field[i];
            if diff == F::ZERO {
                input_values.push(F::ZERO);
            } else {
                input_values.push(F::ONE / diff);
            }
        }

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false).to_vec();

        (output_evals, layers)
    }

    #[test]
    fn eq() {
        type F = FpSisu;
        // generate random 381 bytes.
        let mut rng = StdRng::seed_from_u64(42);
        let mut num = vec![];
        for _ in 0..NUM_LEN {
            let x = rng.gen_range(0..2) as u8;
            num.push(x);
        }

        let output_evals = run_test::<F>(&num, &num).0;
        for i in 0..4 {
            assert_eq!(F::ZERO, output_evals[i]);
        }
        assert_eq!(F::ONE, output_evals[4]);
    }

    #[test]
    fn not_eq() {
        type F = FpSisu;
        // generate random 381 bytes.
        let mut rng = StdRng::seed_from_u64(42);
        let mut num1 = vec![];
        for _ in 0..NUM_LEN {
            let x = rng.gen_range(0..2) as u8;
            num1.push(x);
        }

        let mut num2 = num1.clone();
        num2[NUM_LEN - 1] = 1 - num2[NUM_LEN - 1];

        let result = run_test::<F>(&num1, &num2);
        let output_evals = result.0;
        for i in 0..output_evals.len() {
            assert_eq!(F::ZERO, output_evals[i]);
        }
    }
}
