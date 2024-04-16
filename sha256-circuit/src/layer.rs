use std::collections::HashMap;
use std::marker::PhantomData;

use ark_ff::Field;
use sisulib::circuit::general_circuit::gate::{GeneralCircuitGate, LayerIndex};
use sisulib::circuit::{CircuitGateType, GateF, GateIndex};

use crate::constants::{
    COEFFS_128, COEFF_32, K_VALUES, NEGATIVE_COEFF_33, NEGATIVE_COEFF_34, NEGATIVE_COEFF_35,
};

pub struct Layer<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> Layer<F> {
    pub fn input_tags() -> Vec<String> {
        vec![
            "a_i_bits".to_string(),
            "e_i_bits".to_string(),
            "w_i_bits".to_string(),
            "h_i_bits".to_string(),
            "hout_i_bits".to_string(),
            "input0".to_string(),
        ]
    }

    pub fn constant_coeff_32() -> Vec<F> {
        COEFF_32.iter().map(|x| F::from(*x as u128)).collect()
    }

    pub fn constant_coeff_128() -> Vec<F> {
        let res: Vec<F> = COEFFS_128.iter().map(|x| F::from(*x)).collect();

        return res;
    }

    pub fn constant_negative_coeff_33() -> Vec<F> {
        let res: Vec<F> = NEGATIVE_COEFF_33
            .iter()
            .map(|x| F::ZERO - F::from((-(*x)) as u128))
            .collect();

        return res;
    }

    pub fn constant_negative_coeff_34() -> Vec<F> {
        let res: Vec<F> = NEGATIVE_COEFF_34
            .iter()
            .map(|x| F::ZERO - F::from((-(*x)) as u128))
            .collect();

        return res;
    }

    pub fn constant_negative_coeff_35() -> Vec<F> {
        let res: Vec<F> = NEGATIVE_COEFF_35
            .iter()
            .map(|x| F::ZERO - F::from((-(*x)) as u128))
            .collect();

        return res;
    }

    pub fn constants_k() -> Vec<F> {
        let mut ret: Vec<F> = vec![];
        for i in 0..K_VALUES.len() {
            ret.push(F::from(K_VALUES[i]))
        }
        ret
    }

    ///////////////////////////////////////////////////////
    // New gate functions
    ///////////////////////////////////////////////////////
    pub fn new_xor_gate_with_constant(
        label: &str,
        left: GateIndex,
        right: (LayerIndex, GateIndex),
        constant: F,
    ) -> GeneralCircuitGate<F> {
        return GeneralCircuitGate::new(
            label,
            CircuitGateType::Xor(GateF::C(constant)),
            left,
            right,
        );
    }

    pub fn new_mul_gate(
        label: &str,
        left: GateIndex,
        right: (LayerIndex, GateIndex),
    ) -> GeneralCircuitGate<F> {
        return Self::new_mul_gate_with_constant(label, left, right, F::ONE);
    }

    pub fn new_mul_gate_with_constant(
        label: &str,
        left: GateIndex,
        right: (LayerIndex, GateIndex),
        constant: F,
    ) -> GeneralCircuitGate<F> {
        return GeneralCircuitGate::new(
            label,
            CircuitGateType::Mul(GateF::C(constant)),
            left,
            right,
        );
    }

    pub fn new_forward_gate(
        label: &str,
        constant: F,
        right: (LayerIndex, GateIndex),
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(
            format!("{label}").as_str(),
            CircuitGateType::ForwardY(GateF::C(constant)),
            right.1,
            right,
        )
    }

    pub fn new_accumulate(
        label: &str,
        nested: Vec<GeneralCircuitGate<F>>,
    ) -> GeneralCircuitGate<F> {
        return GeneralCircuitGate::new_accumulate(label, nested);
    }

    pub fn new_accumulates(
        label: &str,
        nested: &Vec<Vec<GeneralCircuitGate<F>>>,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut ret = Vec::with_capacity(nested[0].len());
        for k in 0..nested[0].len() {
            let mut tmp: Vec<GeneralCircuitGate<F>> = Vec::with_capacity(nested.len());
            for i in 0..nested.len() {
                tmp.push(nested[i][k].clone());
            }

            ret.push(Self::new_accumulate(&format!("{label}"), tmp));
        }

        return ret;
    }

    pub fn new_xor_gate_array(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        in2_layer: LayerIndex,
    ) -> Vec<GeneralCircuitGate<F>> {
        Self::new_xor_gate_array_with_coeffs(
            label,
            input1,
            input2,
            in2_layer,
            &vec![F::ONE; input1.len()],
        )
    }

    pub fn new_xor_gate_array_with_coeffs(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        in2_layer: LayerIndex,
        coeffs: &Vec<F>,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut res: Vec<GeneralCircuitGate<F>> = vec![];

        for i in 0..input1.len() {
            let x = input1[i].clone();
            let y = input2[i].clone();
            let gate = Layer::new_xor_gate_with_constant(
                format!("{label}[{i}]").as_str(),
                x,
                (in2_layer.clone(), y),
                coeffs[i],
            );
            res.push(gate);
        }

        res
    }

    pub fn new_mul_gate_array(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        in2_layer: LayerIndex,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut res: Vec<GeneralCircuitGate<F>> = vec![];

        for i in 0..input1.len() {
            let x = input1[i].clone();
            let y = input2[i].clone();
            let gate =
                Layer::new_mul_gate(format!("{label}[{i}]").as_str(), x, (in2_layer.clone(), y));
            res.push(gate);
        }

        return res;
    }

    pub fn new_mul_gate_array_with_constant(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        in2_layer: usize,
        constant: F,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut res: Vec<GeneralCircuitGate<F>> = vec![];

        for i in 0..input1.len() {
            let x = input1[i].clone();
            let y = input2[i].clone();
            let gate = Layer::new_mul_gate_with_constant(
                format!("{label}[{i}]").as_str(),
                x,
                (LayerIndex::Relative(in2_layer), y),
                constant,
            );
            res.push(gate);
        }

        return res;
    }

    pub fn new_forward_gate_array(
        label: &str,
        gates: &Vec<GateIndex>,
        layer_relative_index: usize,
    ) -> Vec<GeneralCircuitGate<F>> {
        Self::new_forward_gate_array_with_coeffs(
            label,
            gates,
            layer_relative_index,
            &vec![F::ONE; gates.len()],
        )
    }

    pub fn new_forward_gate_array_with_coeffs(
        label: &str,
        gates: &Vec<GateIndex>,
        layer_relative_index: usize,
        coeffs: &Vec<F>,
    ) -> Vec<GeneralCircuitGate<F>> {
        return gates
            .iter()
            .enumerate()
            .map(|(i, x)| {
                return Self::new_forward_gate(
                    format!("{label}[{i}]").as_str(),
                    coeffs[i],
                    (LayerIndex::Relative(layer_relative_index), x.clone()),
                );
            })
            .collect();
    }

    pub fn new_constant_gate_from_layer(
        label: &str,
        constant: F,
        relative_layer: usize,
        gate_index: usize,
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(
            format!("{label}").as_str(),
            CircuitGateType::Constant(GateF::C(constant)),
            GateIndex::Absolute(0),
            (
                LayerIndex::Relative(relative_layer),
                GateIndex::Absolute(gate_index),
            ),
        )
    }

    pub fn right_rotate(x: &Vec<GateIndex>, len: usize) -> Vec<GateIndex> {
        let mut res: Vec<GateIndex> = vec![];
        res.extend_from_slice(&x[x.len() - len..]);
        res.extend_from_slice(&x[..x.len() - len]);
        return res;
    }

    pub fn new_naab_gate(
        label: &str,
        left: GateIndex,
        right: (LayerIndex, GateIndex),
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(label, CircuitGateType::NAAB, left, right)
    }

    pub fn new_naab_gate_array(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        in2_layer: LayerIndex,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut res: Vec<GeneralCircuitGate<F>> = vec![];

        for i in 0..input1.len() {
            let x = input1[i].clone();
            let y = input2[i].clone();
            let gate =
                Self::new_naab_gate(format!("{label}[{i}]").as_str(), x, (in2_layer.clone(), y));
            res.push(gate);
        }

        res
    }

    pub fn new_right_shift(
        gates: &Vec<GateIndex>,
        input0: GateIndex,
        shift: usize,
    ) -> Vec<GateIndex> {
        let mut res: Vec<GateIndex> = vec![];
        for _ in 0..shift {
            res.push(input0);
        }
        for i in 0..gates.len() - shift {
            res.push(gates[i]);
        }
        res
    }

    ///////////////////////////////////////////////////////
    // Other Functions
    ///////////////////////////////////////////////////////
    pub fn update_abcdefgh(
        i: usize,
        abcdefgh_bits: &mut Vec<Vec<GateIndex>>,
        e_i_bits: &Vec<Vec<GateIndex>>,
        a_i_bits: &Vec<Vec<GateIndex>>,
    ) {
        for j in (1..8).rev() {
            abcdefgh_bits[j] = abcdefgh_bits[j - 1].clone();
        }

        // e_bits = e_i_bits[i][..32]
        abcdefgh_bits[4] = e_i_bits[i][..32].to_vec();
        // a_bits = a_i_bits[i][..32]
        abcdefgh_bits[0] = a_i_bits[i][..32].to_vec();
    }

    pub fn get_abcdefgh_indexes(
        prev_indexes: &HashMap<String, Vec<Vec<GateIndex>>>,
        letter_index: usize,
    ) -> Vec<Vec<GateIndex>> {
        let a_i_bits = prev_indexes.get(&format!("a_i_bits")).unwrap();
        let e_i_bits = prev_indexes.get(&format!("e_i_bits")).unwrap();
        let mut abcdefgh_bits = prev_indexes.get(&format!("h_i_bits")).unwrap().clone();

        let mut temp_values: Vec<Vec<GateIndex>> = Vec::with_capacity(64);
        for i in 0..64 {
            let gates = abcdefgh_bits[letter_index].clone();
            temp_values.push(gates);
            Self::update_abcdefgh(i, &mut abcdefgh_bits, &e_i_bits, &a_i_bits);
        }

        temp_values
    }

    pub fn general_circuit_gate_to_index(
        gates: &Vec<GeneralCircuitGate<F>>,
        start: usize,
    ) -> Vec<GateIndex> {
        return gates
            .iter()
            .enumerate()
            .map(|(index, _)| GateIndex::Absolute(start + index))
            .collect();
    }

    pub fn add_new_gates<CallbackFn>(
        label: &str,
        all_gates: &mut Vec<GeneralCircuitGate<F>>,
        indexes: &mut HashMap<String, Vec<Vec<GateIndex>>>,
        mut callback: CallbackFn,
    ) where
        CallbackFn: FnMut(&str, usize) -> Vec<GeneralCircuitGate<F>>,
    {
        let mut temp_values: Vec<Vec<GateIndex>> = Vec::with_capacity(64);
        for i in 0..64 {
            let gates = callback(label, i);
            if gates.len() == 0 {
                continue;
            }
            temp_values.push(Self::general_circuit_gate_to_index(&gates, all_gates.len()));
            all_gates.extend(gates);
        }
        indexes.insert(label.to_string(), temp_values.clone());
    }

    // Loops through 64 rounds and in each round, adds a list of new gates associated with that
    // rounnd. This list of gates is returned by the callback function.
    // The callback function receives a list of all values of a, b, c, d, e, f, g, h associated with
    // the current round.
    pub fn with_shuffled_abcdefgh<CallbackFn>(
        label: &str,
        all_gates: &mut Vec<GeneralCircuitGate<F>>,
        prev_indexes: &HashMap<String, Vec<Vec<GateIndex>>>,
        indexes: &mut HashMap<String, Vec<Vec<GateIndex>>>,
        mut callback: CallbackFn,
    ) -> Vec<Vec<GateIndex>>
    where
        CallbackFn: FnMut(&str, usize, &Vec<Vec<GateIndex>>) -> Vec<GeneralCircuitGate<F>>,
    {
        let a_i_bits = prev_indexes.get(&format!("a_i_bits")).unwrap();
        let e_i_bits = prev_indexes.get(&format!("e_i_bits")).unwrap();
        let mut abcdefgh_bits = prev_indexes.get(&format!("h_i_bits")).unwrap().clone();

        let mut temp_values: Vec<Vec<GateIndex>> = Vec::with_capacity(64);
        for i in 0..64 {
            let gates = callback(label, i, &abcdefgh_bits);
            temp_values.push(Self::general_circuit_gate_to_index(&gates, all_gates.len()));
            all_gates.extend(gates);
            Self::update_abcdefgh(i, &mut abcdefgh_bits, &e_i_bits, &a_i_bits);
        }
        indexes.insert(label.to_string(), temp_values.clone());

        temp_values
    }
}
