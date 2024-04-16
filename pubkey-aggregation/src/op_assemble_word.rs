use std::marker::PhantomData;

use crate::{circuit_layer::CircuitLayer, circuit_util::CircuitUtil, constant::NUM_LEN};
use ark_ff::Field;
use circuitlib::constant::WORD_COUNT;
use circuitlib::layer::Layer;

pub struct OpAssembleWord<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> OpAssembleWord<F> {
    fn construct(
        layers: &mut Vec<CircuitLayer<F>>,
        label: &str,
        word_label: &str,
        out_layer: usize,
    ) {
        let a_indexes = layers[0].indexes.get(label).unwrap();
        let input0 = layers[0].indexes.get("input0").unwrap()[0];
        assert_eq!(NUM_LEN, a_indexes.len());

        let mut indexes = vec![input0, input0, input0];
        indexes.extend(a_indexes);

        let coeffs96 = Layer::<F>::coeffs96();
        let mut acc_gates = vec![];
        for i in 0..WORD_COUNT {
            let start = i * 96;
            let gates = Layer::new_forward_gate_array_with_coeffs(
                "",
                &indexes[start..start + 96].to_vec(),
                out_layer,
                &coeffs96,
            );
            let acc_gate = Layer::new_accumulate(&format!("{word_label}[{i}]"), gates);
            acc_gates.push(acc_gate);
        }

        layers[out_layer].add_gates(word_label, acc_gates);
    }

    pub fn build(layers: &mut Vec<CircuitLayer<F>>, label: &str, out_layer: usize) {
        // construct words in layer1.
        let word_label = CircuitUtil::word_label(label);
        match layers[out_layer].indexes.get(&word_label) {
            Some(_) => {
                // We already had this word in the indexes. No need to reconstruct the word.
            }
            None => {
                Self::construct(layers, label, &word_label, out_layer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::str::FromStr;

    use num_bigint::BigUint;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use sisulib::circuit::general_circuit::circuit::GeneralCircuit;
    use sisulib::circuit::general_circuit::layer::GeneralCircuitLayer;
    use sisulib::circuit::CircuitParams;
    use sisulib::field::FpSisu;

    use crate::circuit_layer::CircuitLayer;
    use crate::constant::NUM_LEN;
    use crate::op_assemble_word::OpAssembleWord;
    use ark_ff::Field;
    use circuitlib::constant::WORD_COUNT;

    #[test]
    fn build() {
        type F = FpSisu;
        // generate random 381 bytes.
        let mut rng = StdRng::seed_from_u64(42);
        let mut num = vec![];
        let mut num_field = vec![];
        for _ in 0..NUM_LEN {
            let x = rng.gen_range(0..2) as u8;
            num.push(x);
            num_field.push(F::from(x));
        }

        let mut layers: Vec<CircuitLayer<F>> = vec![];
        layers.push(CircuitLayer::new(true));
        layers.push(CircuitLayer::new(false));

        layers[0].add_input_zero_one("p", NUM_LEN);
        layers[0].add_input_zero_one("input0", 1);

        OpAssembleWord::build(&mut layers, "p", 1);
        assert_eq!(WORD_COUNT, layers[1].len());

        let mut circuit = GeneralCircuit::<F>::new(NUM_LEN + 1);
        circuit.push_layer(GeneralCircuitLayer::new(layers[1].all_gates.clone()));
        circuit.finalize(HashMap::default());

        ///////

        let mut input_values = Vec::from(num_field);
        input_values.push(F::ZERO);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false).to_vec();
        assert_eq!(WORD_COUNT, output_evals.len());

        // let big_num = BigUint::from_bytes_be(num.as_slice());
        let mut big_num = BigUint::from_radix_be(num.as_slice(), 2).unwrap();
        let p96 = BigUint::from_str("79228162514264337593543950336").unwrap();

        for i in (0..4).rev() {
            let rem = &big_num % &p96;
            assert_eq!(rem.to_string(), output_evals[i].to_string());
            big_num = big_num / &p96;
        }
    }
}
