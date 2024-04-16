use std::marker::PhantomData;

use ark_ff::Field;
use sisulib::circuit::GateIndex;

use crate::{
    circuit_layer::CircuitLayer, circuit_util::CircuitUtil, constant::NUM_LEN,
    op_add_input_num::OpAddInputNum,
};
use circuitlib::layer::Layer;

pub struct OpLt<F: Field> {
    __f: PhantomData<F>,
}

/// This operation adds necessary gates to a circuit to prove that one number is smaller than the other.
///   // ors[i] holds (lt[0] || (eq[0] && lt[1]) .. || (eq[0] && .. && lt[i]))
///   // ands[i] holds (eq[0] && .. && lt[i])
///   // eq_ands[i] holds (eq[0] && .. && eq[i])
///   for (var i = 1; i < k - 1; i++) {
///     ands[i] = AND();
///     eq_ands[i] = AND();
///     ors[i] = OR();
///
///     if (i == 1) {
///        ands[i].a <== eq[i - 1].out;
///        ands[i].b <== lt[i].out;
///        eq_ands[i].a <== eq[i - 1].out;
///        eq_ands[i].b <== eq[i].out;
///        ors[i].a <== lt[i - 1].out;
///        ors[i].b <== ands[i].out;
///     } else {
///        ands[i].a <== eq_ands[i - 1].out;
///        ands[i].b <== lt[i].out;
///        eq_ands[i].a <== eq_ands[i - 1].out;
///        eq_ands[i].b <== eq[i].out;
///        ors[i].a <== ors[i - 1].out;
///        ors[i].b <== ands[i].out;
///     }
/// }
/// out <== ors[k - 1].out;
impl<F: Field> OpLt<F> {
    fn add_lt_gate(layers: &mut Vec<CircuitLayer<F>>, a_label: &str, b_label: &str) {
        let cmp_label = CircuitUtil::cmp_lt_label(a_label, b_label);

        // construct lt and eq array in layer1.
        let a_indexes = layers[0].indexes.get(a_label).unwrap();
        let b_indexes = layers[0].indexes.get(b_label).unwrap();

        // lt[i] = (F::ONE - a[i]) * b[i] = b[i] - a[i] * b[i].
        let forward_b = Layer::new_forward_gate_array(b_label, b_indexes, 1);
        let mul = Layer::new_mul_gate_array_with_constant(
            "",
            &a_indexes,
            &b_indexes,
            1,
            F::ZERO - F::ONE,
        );
        let nested = vec![forward_b, mul];
        let acc_gates = Layer::new_accumulates(a_label, &nested);
        layers[1].add_gates(&format!("{cmp_label}_lt"), acc_gates);
    }

    fn add_eq_gate(layers: &mut Vec<CircuitLayer<F>>, a_label: &str, b_label: &str) {
        let cmp_label = CircuitUtil::cmp_lt_label(a_label, b_label);
        let input_indexes = &layers[0].indexes;
        // eq[i] =  a[i] * b[i] + (F::ONE - a[i]) * (F::ONE - b[i]) = 2 * a[i] * b[i] - a[i] - b[i] + F::ONE.
        let minus_one = F::ZERO - F::ONE;

        let a_indexes = input_indexes.get(a_label).unwrap();
        let b_indexes = input_indexes.get(b_label).unwrap();
        let n = a_indexes.len();
        // 2 * a[i] * b[i]
        let mul_ab = Layer::new_mul_gate_array_with_constant(
            "mul_ab",
            &a_indexes,
            &b_indexes,
            1,
            F::from(2u128),
        );
        // -a[i]
        let forward_a =
            Layer::new_forward_gate_array_with_coeffs(a_label, a_indexes, 1, &vec![minus_one; n]);
        // -b[i]
        let forward_b =
            Layer::new_forward_gate_array_with_coeffs(b_label, b_indexes, 1, &vec![minus_one; n]);
        // + F::ONE
        let one_gates = Layer::new_constant_gate_array("one", &vec![F::ONE; n]);

        let nested = vec![forward_a, forward_b, mul_ab, one_gates];
        let acc_gates = Layer::new_accumulates("eq", &nested);

        layers[1].add_gates(&format!("{cmp_label}_eq"), acc_gates);
    }

    /// This function calculated ands, eq_ands, ors array.
    fn check_eq_ands_ors(layers: &mut Vec<CircuitLayer<F>>, cmp_label: &str) {
        // check result in layer 2.
        // 1. ands:
        // if i == 1: ands[i] = eq[i - 1] * lt[i]
        // if i > 1: ands[i] = eq_ands[i - 1] * lt[i]
        let input0 = layers[0].indexes.get(&format!("input0")).unwrap()[0];
        let eq0 = layers[0].indexes.get(&format!("{cmp_label}_eq0")).unwrap()[0];
        let lt = layers[1].indexes.get(&format!("{cmp_label}_lt")).unwrap();
        let eq_ands = layers[0]
            .indexes
            .get(&format!("{cmp_label}_eq_ands"))
            .unwrap();
        let mut gate_indexes: Vec<GateIndex> = vec![];

        // i = 0.
        gate_indexes.push(input0);

        // i = 1. Forward the eq[0] gate from layer 0.
        gate_indexes.push(eq0);

        // gates
        for i in 2..NUM_LEN {
            gate_indexes.push(eq_ands[i - 1]);
        }

        let gates = Layer::new_mul_gate_array("", &lt, &gate_indexes, 2);
        layers[2].add_gates(&format!("{cmp_label}_ands"), gates);

        // 2. eq_ands
        let eq = layers[1].indexes.get(&format!("{cmp_label}_eq")).unwrap();
        let gates = Layer::new_mul_gate_array("", &eq, &gate_indexes, 2);
        layers[2].add_gates(&format!("{cmp_label}_eq_ands"), gates);

        // 3. ors
        let lt0 = layers[0].indexes.get(&format!("{cmp_label}_lt0")).unwrap()[0];
        let ors = layers[0].indexes.get(&format!("{cmp_label}_ors")).unwrap();
        let ands = layers[0].indexes.get(&format!("{cmp_label}_ands")).unwrap();
        let mut gate_indexes: Vec<GateIndex> = vec![];
        // i = 0.
        gate_indexes.push(input0);
        // i = 1. Forward the lt[0] gate from layer 0.
        gate_indexes.push(lt0);
        for i in 2..NUM_LEN {
            gate_indexes.push(ors[i - 1]);
        }

        // or(a, b) = a + b - a * b.
        // forward ands to layer 1
        let forward_ands = Layer::new_forward_gate_array("", &ands, 1);
        layers[1].add_gates(&format!("{cmp_label}_forward_ands"), forward_ands);

        let ands = layers[1]
            .indexes
            .get(&format!("{cmp_label}_forward_ands"))
            .unwrap();
        let forward_ands = Layer::new_forward_gate_array("", &ands, 1);
        let forward_ors = Layer::new_forward_gate_array("", &gate_indexes, 2);
        let mul_gates =
            Layer::new_mul_gate_array_with_constant("", &ands, &gate_indexes, 2, F::ZERO - F::ONE);

        let nested = vec![forward_ands, forward_ors, mul_gates];
        let acc_gates = Layer::new_accumulates("", &nested);

        layers[2].add_gates(&format!("{cmp_label}_ors"), acc_gates);
    }

    /// This method checks that the calculated values of ands, eq_ands, ors in layer 2 equal the
    /// values in the input layer.
    fn forward_final_checks(layers: &mut Vec<CircuitLayer<F>>, cmp_label: &str) {
        let op_labels = ["ands", "eq_ands", "ors"];
        let layer_count = layers.len();
        for op_label in op_labels {
            let label = &format!("{cmp_label}_{op_label}");
            let from_layer2 = layers[2].get_indexes(&label);
            let from_layer0 = layers[0].get_indexes(&label);
            assert_eq!(from_layer0.len(), from_layer2.len());

            let gates2 = Layer::new_forward_gate_array(&label, from_layer2, layer_count - 1 - 2);
            let gates0 = Layer::new_forward_gate_array_with_coeffs(
                &label,
                from_layer0,
                layer_count - 1,
                &vec![F::ZERO - F::ONE; from_layer0.len()],
            );

            let nested = vec![gates2, gates0];
            let acc_gates = Layer::new_accumulates(&format!("{label}"), &nested);

            layers[layer_count - 1].add_gates(label, acc_gates);
        }

        // Check that the lt0 and eq0 in the input layer equals the computed values in layer 1.
        for op_label in ["lt", "eq"] {
            let label = format!("{cmp_label}_{op_label}");
            let layer0_gate = layers[0].indexes.get(&format!("{label}0")).unwrap()[0];
            let layer1_gate = layers[1].indexes.get(&label).unwrap()[0];

            let gate0 = Layer::new_forward_gate("", F::ONE, layer0_gate, layer_count - 1);
            let gate1 = Layer::new_forward_gate("", F::ZERO - F::ONE, layer1_gate, layer_count - 2);
            let acc_gate = Layer::new_accumulate(&label, vec![gate0, gate1]);
            layers[layer_count - 1].add_gates(&label, vec![acc_gate]);
        }
    }

    pub fn build(layers: &mut Vec<CircuitLayer<F>>, a_label: &str, b_label: &str) {
        let cmp_label = CircuitUtil::cmp_lt_label(a_label, b_label);

        OpAddInputNum::build(a_label, &mut layers[0], NUM_LEN, false);
        OpAddInputNum::build(b_label, &mut layers[0], NUM_LEN, false);

        // precomputed ands, eq_ands, ors in layer 0
        let op_labels: Vec<String> =
            vec!["ands".to_string(), "eq_ands".to_string(), "ors".to_string()];
        for op_label in &op_labels {
            let label = format!("{cmp_label}_{op_label}");
            OpAddInputNum::build(&label, &mut layers[0], NUM_LEN, false);
        }

        // construct lt and eq array in layer 1.
        Self::add_lt_gate(layers, a_label, b_label);
        Self::add_eq_gate(layers, a_label, b_label);

        // add eq0 and lt0 in layer input.
        layers[0].add_input_zero_one(&format!("{cmp_label}_lt0"), 1);
        layers[0].add_input_zero_one(&format!("{cmp_label}_eq0"), 1);

        Self::check_eq_ands_ors(layers, &cmp_label);

        // Final check
        Self::forward_final_checks(layers, &cmp_label);
    }
}
