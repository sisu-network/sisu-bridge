use std::marker::PhantomData;

use ark_ff::Field;

use crate::{circuit_layer::CircuitLayer, circuit_util::CircuitUtil, constant::NUM_LEN};
use circuitlib::constant::WORD_COUNT;
use circuitlib::layer::Layer;

pub struct OpDiff<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> OpDiff<F> {
    fn construct(layers: &mut Vec<CircuitLayer<F>>, a_label: &str, b_label: &str) {
        let diff_label = CircuitUtil::diff_label(a_label, b_label);
        let a_indexes = layers[0].indexes.get(a_label).unwrap();
        let b_indexes = layers[0].indexes.get(b_label).unwrap();
        let input0 = layers[0].indexes.get("input0").unwrap()[0];
        assert_eq!(NUM_LEN, a_indexes.len());
        assert_eq!(NUM_LEN, b_indexes.len());

        let mut a_indexes_ext = vec![input0, input0, input0];
        a_indexes_ext.extend(a_indexes);
        let mut b_indexes_ext = vec![input0, input0, input0];
        b_indexes_ext.extend(b_indexes);

        let coeffs96 = Layer::<F>::coeffs96();
        let neg_coeffs96 = Layer::<F>::neg_coeffs96();

        let mut acc_gates = vec![];
        for i in 0..WORD_COUNT {
            let start = i * 96;
            let a_gates = Layer::new_forward_gate_array_with_coeffs(
                "",
                &a_indexes_ext[start..start + 96].to_vec(),
                1,
                &coeffs96,
            );
            let b_gates = Layer::new_forward_gate_array_with_coeffs(
                "",
                &b_indexes_ext[start..start + 96].to_vec(),
                1,
                &neg_coeffs96,
            );

            let mut gates = Vec::from(a_gates);
            gates.extend(b_gates);

            let acc_gate = Layer::new_accumulate(&format!("{diff_label}[{i}]"), gates);
            acc_gates.push(acc_gate);
        }

        layers[1].add_gates(&diff_label, acc_gates);
    }

    pub fn build(layers: &mut Vec<CircuitLayer<F>>, a_label: &str, b_label: &str) {
        let diff_label = CircuitUtil::diff_label(a_label, b_label);
        match layers[1].indexes.get(&diff_label) {
            Some(_) => {
                // We already had this word in the indexes. No need to reconstruct the word.
            }
            None => {
                Self::construct(layers, a_label, &b_label);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use sisulib::circuit::general_circuit::circuit::GeneralCircuit;
    use sisulib::circuit::general_circuit::layer::GeneralCircuitLayer;
    use sisulib::circuit::CircuitParams;
    use sisulib::field::FpSisu;
    use std::collections::HashMap;

    use crate::circuit_layer::CircuitLayer;
    use crate::constant::NUM_LEN;
    use crate::op_diff::OpDiff;
    use crate::test_util::tests::add_2_nums;
    use ark_ff::Field;
    use circuitlib::constant::WORD_COUNT;

    fn run_test<F: Field>(num_field1: &Vec<F>, num_field2: &Vec<F>) -> Vec<F> {
        let mut layers: Vec<CircuitLayer<F>> = vec![];
        add_2_nums(&mut layers, "a", "b");

        OpDiff::build(&mut layers, "a", "b");
        assert_eq!(WORD_COUNT, layers[1].len());

        let mut circuit = GeneralCircuit::<F>::new(2 * NUM_LEN + 1);
        circuit.push_layer(GeneralCircuitLayer::new(layers[1].all_gates.clone()));
        circuit.finalize(HashMap::default());

        ///////

        let mut input_values = vec![];
        input_values.extend(num_field1);
        input_values.extend(num_field2);
        input_values.push(F::ZERO);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false).to_vec();
        assert_eq!(WORD_COUNT, output_evals.len());

        output_evals
    }

    #[test]
    fn eq() {
        type F = FpSisu;
        // generate random 381 bytes.
        let mut rng = StdRng::seed_from_u64(42);
        let mut num_field = vec![];
        for _ in 0..NUM_LEN {
            let x = rng.gen_range(0..2) as u8;
            num_field.push(F::from(x));
        }

        let output_evals = run_test(&num_field, &num_field);
        for i in (0..4).rev() {
            assert_eq!(F::ZERO, output_evals[i]);
        }

        let mut num_field2 = num_field.clone();
        num_field2[NUM_LEN - 1] = F::ONE - num_field2[NUM_LEN - 1];
        let output_evals = run_test(&num_field, &num_field2);
        for i in (0..3).rev() {
            assert_eq!(F::ZERO, output_evals[i]);
        }
        assert_eq!(F::ZERO - F::ONE, output_evals[3]);
    }
}
