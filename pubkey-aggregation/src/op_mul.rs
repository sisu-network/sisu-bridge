use std::marker::PhantomData;

use ark_ff::Field;
use sisulib::circuit::general_circuit::gate::GeneralCircuitGate;

use crate::constant::NUM_LEN;
use crate::op_add_input_num::OpAddInputNum;
use crate::op_assemble_word::OpAssembleWord;
use crate::{circuit_layer::CircuitLayer, circuit_util::CircuitUtil};
use circuitlib::constant::{WORD_COUNT, WORD_SIZE};
use circuitlib::layer::Layer;

pub struct OpMul<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> OpMul<F> {
    fn assemble_carries(
        layers: &mut Vec<CircuitLayer<F>>,
        carry_label: &str,
        word_layer: usize,
        coeffs96: &Vec<F>,
    ) {
        let carry_gates = layers[0].get_indexes(carry_label);
        let carry_extra_gates = layers[0].get_indexes(&format!("{carry_label}_extra"));
        let mul_extra_bit_coeffs: Vec<F> = Layer::mul_extra_bit_coeffs();

        let mut acc_gates: Vec<GeneralCircuitGate<F>> = Vec::with_capacity(2 * WORD_COUNT - 1);
        for i in 0..2 * WORD_COUNT {
            let start = i * WORD_SIZE;

            let forward_gates = Layer::new_forward_gate_array_with_coeffs(
                carry_label,
                &carry_gates[start..start + WORD_SIZE].to_vec(),
                word_layer,
                &coeffs96,
            );

            // Add extra bit
            let extra_bit_gates = Layer::new_forward_gate_array_with_coeffs(
                carry_label,
                &carry_extra_gates[i * 2..i * 2 + 2].to_vec(),
                word_layer,
                &mul_extra_bit_coeffs,
            );

            let mut nested = Vec::with_capacity(WORD_SIZE + 2);
            nested.extend(forward_gates);
            nested.extend(extra_bit_gates);
            assert_eq!(WORD_SIZE + 2, nested.len());

            let acc_gate = Layer::new_accumulate(&format!("{carry_label}_word"), nested);
            acc_gates.push(acc_gate);
        }

        layers[word_layer].add_gates(&format!("{carry_label}_word"), acc_gates);
    }

    fn verify_mul(
        layers: &mut Vec<CircuitLayer<F>>,
        a_label: &str,
        b_label: &str,
        mul_label: &str,
        verify_layer: usize,
    ) {
        let pre_layer = verify_layer - 1;

        let word_indexes_a = layers[pre_layer].get_indexes(&format!("{a_label}_word"));
        let word_indexes_b = layers[pre_layer].get_indexes(&format!("{b_label}_word"));
        let word_indexes_c = layers[pre_layer].get_indexes(&format!("{mul_label}_word"));
        let word_indexes_d = layers[pre_layer].get_indexes(&format!("{mul_label}_mul_d_word"));

        let carry_left_indexes =
            layers[verify_layer - 1].get_indexes(&format!("{mul_label}_mul_carry_left_word"));
        let carry_right_indexes =
            layers[verify_layer - 1].get_indexes(&format!("{mul_label}_mul_carry_right_word"));

        let negative_p_381_coeffs: Vec<F> = Layer::constant_negative_p_381_coeffs();
        let minus_one: F = F::ZERO - F::ONE;
        let modulo: F = CircuitUtil::modulo();
        let minus_modulo: F = F::ZERO - modulo;
        let mut acc_gates: Vec<GeneralCircuitGate<F>> = Vec::with_capacity(7);

        for k in 0..2 * WORD_COUNT - 1 {
            // sum(ai * bj) + carry_left[k - 1] - carry_left[k] * modulo = sum(di * pj) + c[k] + carry_right[k - 1] - carry_right[k] * modulo
            // This equivalent to
            // sum(ai * bj) + carry_left[k - 1] - carry_left[k] * modulo - (sum(di * pj) + c[k] + carry_right[k - 1] - carry_right[k] * modulo) = 0
            // sum(ai * bj) - sum(di * pj) - c[k] + carry_left[k - 1] - carry_right[k - 1] - carry_left[k] * modulo + carry_right[k] * modulo = 0
            let mut gates: Vec<GeneralCircuitGate<F>> = vec![];
            for i in 0..k + 1 {
                let j = k - i;
                if i > WORD_COUNT - 1 || j > WORD_COUNT - 1 {
                    continue;
                }
                let index1 = WORD_COUNT - 1 - i;
                let index2 = WORD_COUNT - 1 - j;

                // ai * bj
                let gate: GeneralCircuitGate<F> = Layer::new_mul_gate(
                    format!("{mul_label}_a_{i}_b_{j}").as_str(),
                    word_indexes_a[index1],
                    word_indexes_b[index2],
                    1,
                );
                gates.push(gate);

                // - di * pj
                let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                    format!("{mul_label}_d_{i}_p_{j}").as_str(),
                    negative_p_381_coeffs[index1],
                    word_indexes_d[index2],
                    1,
                );
                gates.push(gate);
            }

            // - c
            if k < WORD_COUNT {
                let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                    format!("{mul_label}_minus_c_{k}").as_str(),
                    minus_one,
                    word_indexes_c[WORD_COUNT - 1 - k],
                    1,
                );
                gates.push(gate);
            }

            // + carry_left[previous] - carry_right[previous]
            if k > 0 {
                // + carry_left[k - 1]
                let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                    format!("{mul_label}_carry_left_{k}_minus_1").as_str(),
                    F::ONE,
                    carry_left_indexes[WORD_COUNT * 2 - k],
                    1,
                );
                gates.push(gate);

                // - carry_right[k - 1]
                let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                    format!("{mul_label}_carry_right_{k}_minus_1").as_str(),
                    minus_one,
                    carry_right_indexes[WORD_COUNT * 2 - k],
                    1,
                );
                gates.push(gate);
            }

            // - carry_left * modulo
            let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                format!("{mul_label}_carry_left_{k}_mul_modulo").as_str(),
                minus_modulo,
                carry_left_indexes[WORD_COUNT * 2 - 1 - k],
                1,
            );
            gates.push(gate);

            // + carry_right * modulo
            let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                format!("{mul_label}_carry_right_{k}_mul_modulo").as_str(),
                modulo,
                carry_right_indexes[WORD_COUNT * 2 - 1 - k],
                1,
            );
            gates.push(gate);

            let acc_gate =
                Layer::new_accumulate(format!("{mul_label}_accumulate_{k}").as_str(), gates);

            acc_gates.push(acc_gate);
        }
        acc_gates.reverse();

        let check_label = format!("{mul_label}_check");
        layers[verify_layer].add_gates(&check_label, acc_gates);

        let layer_len = layers.len();
        if verify_layer != layer_len - 1 {
            // forward this result to the final layer.
            let indexes = layers[verify_layer].get_indexes(&check_label);
            let gates = Layer::new_forward_gate_array(
                &format!("{mul_label}_check"),
                indexes,
                layer_len - 1 - verify_layer,
            );
            layers[layer_len - 1].add_gates(&check_label, gates);
        }
    }

    fn mul_carry_extra_bit_range_check(layers: &mut Vec<CircuitLayer<F>>, mul_label: &str) {
        let len = WORD_COUNT * 2;
        let gates = layers[0].get_indexes(mul_label);
        let mut forward_gates: Vec<GeneralCircuitGate<F>> = vec![];
        let mut mul_gates: Vec<GeneralCircuitGate<F>> = vec![];
        let layer_len = layers.len();

        for i in 1..len {
            let check_label = &format!("{mul_label}_{i}");
            let gate0 = gates[i * 2];
            let gate1 = gates[i * 2 + 1];
            if i == 1 || i == len - 1 {
                // Both extra bits should be 0.
                let new_gates =
                    Layer::new_forward_gate_array(mul_label, &vec![gate0, gate1], layer_len - 1);
                forward_gates.extend(new_gates);
            }
            if i == 2 || i == len - 2 {
                // The 2 bits should be 00 or 01. We only need to forward the first bit as the second
                // bit is included in the zero-one check.
                let new_gates =
                    Layer::new_forward_gate_array(mul_label, &vec![gate0], layer_len - 1);
                forward_gates.extend(new_gates);
            }
            if i == 3 || i == len - 3 {
                // The 2 bits could 00, 01 or 10. This is equivalent to bit1 * bit2 == 0.
                let mul_gate = Layer::new_mul_gate(check_label, gate0, gate1, 1);
                mul_gates.push(mul_gate);
            }
        }
        assert_eq!(6, forward_gates.len());

        layers[layer_len - 1].add_gates(&format!("{mul_label}"), forward_gates);

        // Do the mul in layer 1 and forward the result to layer output.
        layers[1].add_gates(&format!("{mul_label}"), mul_gates);
        let indexes = layers[1].get_indexes(&format!("{mul_label}"));
        let gates = Layer::new_forward_gate_array(&format!("{mul_label}"), indexes, layer_len - 2);

        layers[layer_len - 1].add_gates(&format!("{mul_label}_2"), gates);
    }

    fn check_mul_carry_left_right(
        layers: &mut Vec<CircuitLayer<F>>,
        mul_label: &str,
        word_layer: usize,
    ) {
        let left_label = &format!("{mul_label}_mul_carry_left_word");
        let right_label = &format!("{mul_label}_mul_carry_right_word");
        let layer_len = layers.len();

        let left_gates = layers[word_layer].get_indexes(left_label);
        let right_gates = layers[word_layer].get_indexes(right_label);
        assert_eq!(2 * WORD_COUNT, left_gates.len());
        assert_eq!(2 * WORD_COUNT, right_gates.len());
        let mut check_gates = vec![];

        let forward_left = Layer::new_forward_gate(
            left_label,
            F::ONE,
            left_gates[1],
            layer_len - 1 - word_layer,
        );
        let forward_right = Layer::new_forward_gate(
            right_label,
            F::ZERO - F::ONE,
            right_gates[1],
            layer_len - 1 - word_layer,
        );

        let nested = vec![forward_left, forward_right];
        let acc_gate = Layer::new_accumulate(&mul_label, nested);
        check_gates.push(acc_gate);

        layers[word_layer + 1].add_gates(
            &format!("{mul_label}_check_mul_carry_left_right"),
            check_gates,
        )
    }

    /// Adds necessary gates to various layers to prove that a * b = c (mod p).
    ///
    /// # Arguments
    ///
    /// * `word_layer` - The layer where words will be assembled.
    pub fn build(
        layers: &mut Vec<CircuitLayer<F>>,
        mul_label: &str,
        a_label: &str,
        b_label: &str,
        word_layer: usize,
    ) {
        OpAddInputNum::build(a_label, &mut layers[0], NUM_LEN, false);
        OpAddInputNum::build(b_label, &mut layers[0], NUM_LEN, false);
        OpAddInputNum::build(mul_label, &mut layers[0], NUM_LEN, false);
        OpAddInputNum::build(
            &format!("{mul_label}_mul_d"),
            &mut layers[0],
            NUM_LEN,
            false,
        );
        OpAddInputNum::build(
            &format!("{mul_label}_mul_carry_left"),
            &mut layers[0],
            WORD_SIZE * 8,
            false,
        );
        OpAddInputNum::build(
            &format!("{mul_label}_mul_carry_right"),
            &mut layers[0],
            WORD_SIZE * 8,
            false,
        );

        // extra
        OpAddInputNum::build(
            &format!("{mul_label}_mul_carry_left_extra"),
            &mut layers[0],
            2 * 8,
            false,
        );
        OpAddInputNum::build(
            &format!("{mul_label}_mul_carry_right_extra"),
            &mut layers[0],
            2 * 8,
            false,
        );

        OpAssembleWord::build(layers, a_label, word_layer);
        OpAssembleWord::build(layers, b_label, word_layer);
        OpAssembleWord::build(layers, mul_label, word_layer);
        // assemble d
        OpAssembleWord::build(layers, &format!("{mul_label}_mul_d"), word_layer);

        // carries
        let coeffs96 = Layer::<F>::coeffs96();
        Self::assemble_carries(
            layers,
            &format!("{mul_label}_mul_carry_left"),
            word_layer,
            &coeffs96,
        );
        Self::assemble_carries(
            layers,
            &format!("{mul_label}_mul_carry_right"),
            word_layer,
            &coeffs96,
        );

        // verify
        Self::verify_mul(layers, a_label, b_label, mul_label, word_layer + 1);

        // verify carries are within their bounds.
        Self::mul_carry_extra_bit_range_check(layers, &format!("{mul_label}_mul_carry_left_extra"));
        Self::mul_carry_extra_bit_range_check(
            layers,
            &format!("{mul_label}_mul_carry_right_extra"),
        );

        // Check that mul carry_left[1] = carry_right[1].
        Self::check_mul_carry_left_right(layers, mul_label, word_layer);
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use ark_ff::PrimeField;
    use sisulib::circuit::CircuitParams;
    use sisulib::field::FpSisu;

    use crate::test_util::tests::assign_input_with_labels;
    use crate::{
        circuit_layer::CircuitLayer, constant::tests::default_input,
        input_calculator::assign_input, test_util::tests::circuit_from_layers,
    };

    use super::OpMul;

    fn build_layers<F: PrimeField>(
        layers: &mut Vec<CircuitLayer<F>>,
        mul_label: &str,
        a_label: &str,
        b_label: &str,
    ) {
        layers[0].add_input_zero_one("input0", 1);
        layers[0].add_input_zero_one("input1", 1);

        OpMul::build(layers, mul_label, a_label, b_label, 1);
    }

    #[test]
    fn build() {
        type F = FpSisu;
        let mut layers: Vec<CircuitLayer<F>> = vec![];
        layers.push(CircuitLayer::new(true));
        layers.push(CircuitLayer::new(false));
        layers.push(CircuitLayer::new(false));

        // mul_y_x_pr = div_y_x * diff_x_pr
        build_layers::<F>(&mut layers, "mul_y_x_pr", "div_y_x", "diff_x_pr");
        let circuit = circuit_from_layers(&layers);

        let inputs = default_input();
        let assignments = assign_input::<F>(&inputs[0], &inputs[1], &inputs[2]);

        let mut input_values: Vec<F> = vec![F::ZERO; layers[0].len()];
        let indexes = &layers[0].indexes;
        let labels = vec![
            "div_y_x",
            "diff_x_pr",
            "mul_y_x_pr",
            "mul_y_x_pr_mul_d",
            "mul_y_x_pr_mul_carry_left",
            "mul_y_x_pr_mul_carry_right",
            "mul_y_x_pr_mul_carry_left_extra",
            "mul_y_x_pr_mul_carry_right_extra",
            "input0",
            "input1",
        ];
        assign_input_with_labels(&assignments, &labels, indexes, &mut input_values);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false).to_vec();

        for i in 0..output_evals.len() {
            assert_eq!(F::ZERO, output_evals[i])
        }
    }
}
