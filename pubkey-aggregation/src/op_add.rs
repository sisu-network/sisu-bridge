use std::marker::PhantomData;

use ark_ff::Field;
use sisulib::circuit::general_circuit::gate::GeneralCircuitGate;

use crate::constant::NUM_LEN;
use crate::op_add_input_num::OpAddInputNum;
use crate::op_assemble_word::OpAssembleWord;
use crate::{circuit_layer::CircuitLayer, circuit_util::CircuitUtil};
use circuitlib::constant::WORD_COUNT;
use circuitlib::layer::Layer;

pub struct OpAdd<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> OpAdd<F> {
    fn verify_add(
        layers: &mut Vec<CircuitLayer<F>>,
        add_label: &str,
        a_label: &str,
        b_label: &str,
        verify_layer: usize,
    ) {
        let pre_layer = verify_layer - 1;
        let word_indexes_a = layers[pre_layer].get_indexes(&format!("{a_label}_word"));
        let word_indexes_b = layers[pre_layer].get_indexes(&format!("{b_label}_word"));
        let word_indexes_c = layers[pre_layer].get_indexes(&format!("{add_label}_word"));
        let d_index = layers[0].get_indexes(&format!("{add_label}_add_d"))[0];

        let carry_left_indexes = layers[0].get_indexes(&format!("{add_label}_add_carry_left"));
        let carry_right_indexes = layers[0].get_indexes(&format!("{add_label}_add_carry_right"));

        let negative_p_381_coeffs: Vec<F> = Layer::constant_negative_p_381_coeffs();
        let minus_one: F = F::ZERO - F::ONE;
        let modulo: F = CircuitUtil::modulo();
        let minus_modulo: F = F::ZERO - modulo;
        let mut acc_gates: Vec<GeneralCircuitGate<F>> = Vec::with_capacity(WORD_COUNT);

        for k in 0..WORD_COUNT {
            let mut gates: Vec<GeneralCircuitGate<F>> = vec![];
            // a[k] + b[k]
            let gate: GeneralCircuitGate<F> = Layer::new_add_gate(
                add_label,
                word_indexes_a[WORD_COUNT - 1 - k],
                word_indexes_b[WORD_COUNT - 1 - k],
                1,
            );
            gates.push(gate);

            // - d * p381[k]
            let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                add_label,
                negative_p_381_coeffs[WORD_COUNT - 1 - k],
                d_index,
                verify_layer,
            );
            gates.push(gate);

            // - c
            let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                format!("{add_label}_minus_c_{k}").as_str(),
                minus_one,
                word_indexes_c[WORD_COUNT - 1 - k],
                1,
            );
            gates.push(gate);

            // + carry_left[k - 1] - carry_right[k - 1]
            if k > 0 {
                let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                    &add_label,
                    F::ONE,
                    carry_left_indexes[WORD_COUNT - k],
                    verify_layer,
                );
                gates.push(gate);

                let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                    &add_label,
                    minus_one,
                    carry_right_indexes[WORD_COUNT - k],
                    verify_layer,
                );
                gates.push(gate);
            }

            // - carry_left * modulo
            let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                format!("{add_label}_carry_left_{k}_add_modulo").as_str(),
                minus_modulo,
                carry_left_indexes[WORD_COUNT - 1 - k],
                verify_layer,
            );
            gates.push(gate);

            // + carry_right * modulo
            let gate: GeneralCircuitGate<F> = Layer::new_forward_gate(
                format!("{add_label}_carry_right_{k}_add_modulo").as_str(),
                modulo,
                carry_right_indexes[WORD_COUNT - 1 - k],
                verify_layer,
            );
            gates.push(gate);

            let acc_gate =
                Layer::new_accumulate(format!("{add_label}_accumulate_{k}").as_str(), gates);

            acc_gates.push(acc_gate);
        }

        let check_label = format!("{add_label}_check");
        layers[verify_layer].add_gates(&check_label, acc_gates);

        let layer_len = layers.len();
        if verify_layer != layer_len - 1 {
            // forward this result to the final layer.
            let indexes = layers[verify_layer].get_indexes(&check_label);
            let gates = Layer::new_forward_gate_array(
                &format!("{add_label}_check"),
                indexes,
                layer_len - 1 - verify_layer,
            );
            layers[layer_len - 1].add_gates(&check_label, gates);
        }
    }

    pub fn build(
        layers: &mut Vec<CircuitLayer<F>>,
        add_label: &str,
        a_label: &str,
        b_label: &str,
        word_layer: usize,
    ) {
        OpAddInputNum::build(a_label, &mut layers[0], NUM_LEN, false);
        OpAddInputNum::build(b_label, &mut layers[0], NUM_LEN, false);
        OpAddInputNum::build(add_label, &mut layers[0], NUM_LEN, false);
        OpAddInputNum::build(&format!("{add_label}_add_d"), &mut layers[0], 1, false);
        OpAddInputNum::build(
            &format!("{add_label}_add_carry_left"),
            &mut layers[0],
            WORD_COUNT,
            false,
        );
        OpAddInputNum::build(
            &format!("{add_label}_add_carry_right"),
            &mut layers[0],
            WORD_COUNT,
            false,
        );

        OpAssembleWord::build(layers, a_label, word_layer);
        OpAssembleWord::build(layers, b_label, word_layer);
        OpAssembleWord::build(layers, add_label, word_layer);

        Self::verify_add(layers, add_label, a_label, b_label, word_layer + 1);
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use ark_ff::PrimeField;
    use circuitlib::constant::WORD_COUNT;
    use sisulib::circuit::CircuitParams;
    use sisulib::field::FpSisu;

    use crate::test_util::tests::assign_input_with_labels;
    use crate::{
        circuit_layer::CircuitLayer, constant::tests::default_input,
        input_calculator::assign_input, test_util::tests::circuit_from_layers,
    };

    use super::OpAdd;

    fn build_layers<F: PrimeField>(
        layers: &mut Vec<CircuitLayer<F>>,
        add_label: &str,
        a_label: &str,
        b_label: &str,
    ) {
        layers[0].add_input_zero_one("input0", 1);
        layers[0].add_input_zero_one("input1", 1);

        OpAdd::build(layers, add_label, a_label, b_label, 1);
    }

    #[test]
    fn build() {
        type F = FpSisu;
        let mut layers: Vec<CircuitLayer<F>> = vec![];
        layers.push(CircuitLayer::new(true));
        layers.push(CircuitLayer::new(false));
        layers.push(CircuitLayer::new(false));

        let add_label = "mul_y_x_pr";
        let a_label = "p_y";
        let b_label = "r_y";

        build_layers::<F>(&mut layers, add_label, a_label, b_label);
        let circuit = circuit_from_layers(&layers);

        let inputs = default_input();
        let assignments = assign_input::<F>(&inputs[0], &inputs[1], &inputs[2]);
        let mut input_values: Vec<F> = vec![F::ZERO; layers[0].len()];
        let indexes = &layers[0].indexes;

        let d_label = &format!("{add_label}_add_d");
        let left_carry_label = &format!("{add_label}_add_carry_left");
        let right_carry_label = &format!("{add_label}_add_carry_right");
        let labels = vec![
            add_label,
            a_label,
            b_label,
            d_label,
            left_carry_label,
            right_carry_label,
            "input0",
            "input1",
        ];
        assign_input_with_labels(&assignments, &labels, indexes, &mut input_values);

        let evaluations = circuit.evaluate(&CircuitParams::default(), input_values.as_slice());
        let output_evals = evaluations.at_layer(0, false).to_vec();
        assert_eq!(WORD_COUNT, output_evals.len());

        let layer1_evals = evaluations.at_layer(1, false).to_vec();
        let indexes = layers[1].get_indexes("p_y_word");
        for i in 0..4 {
            println!(
                "word {} = {}",
                i,
                layer1_evals[indexes[i].value()].to_string()
            );
        }

        for i in 0..output_evals.len() {
            assert_eq!(F::ZERO, output_evals[i])
        }
    }
}
