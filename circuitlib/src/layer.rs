use std::collections::HashMap;
use std::marker::PhantomData;

use ark_ff::Field;
use sisulib::circuit::general_circuit::gate::{GeneralCircuitGate, LayerIndex};
use sisulib::circuit::{CircuitGateType, GateF, GateIndex};

use crate::constant::{COEFFS_128, COEFFS_96, MUL_EXTRA_BIT_COEFFS, P_381_COEFFS};

pub struct Layer<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> Layer<F> {
    pub fn mul_extra_bit_coeffs() -> Vec<F> {
        MUL_EXTRA_BIT_COEFFS.iter().map(|x| F::from(*x)).collect()
    }

    pub fn coeffs96() -> Vec<F> {
        let mut ret = Vec::with_capacity(COEFFS_96.len());
        for x in COEFFS_96 {
            ret.push(F::from(x));
        }

        ret
    }

    pub fn neg_coeffs96() -> Vec<F> {
        let mut ret = Vec::with_capacity(COEFFS_96.len());
        for x in COEFFS_96 {
            ret.push(F::ZERO - F::from(x));
        }

        ret
    }

    pub fn coeffs128() -> Vec<F> {
        let mut ret = Vec::with_capacity(COEFFS_128.len());
        for x in COEFFS_128 {
            ret.push(F::from(x));
        }

        ret
    }

    pub fn constant_negative_p_381_coeffs() -> Vec<F> {
        return P_381_COEFFS
            .iter()
            .map(|x| F::ZERO - F::from(*x as u128))
            .collect();
    }

    pub fn add_gates(
        label: &str,
        gates: Vec<GeneralCircuitGate<F>>,
        all_gates: &mut Vec<GeneralCircuitGate<F>>,
        indexes: &mut HashMap<String, Vec<GateIndex>>,
    ) {
        let mut temp_values: Vec<GateIndex> = vec![];
        temp_values.extend(Layer::general_circuit_gate_to_index(
            &gates,
            all_gates.len(),
        ));
        indexes.insert(format!("{label}"), temp_values);
        all_gates.extend(gates);
    }

    pub fn general_circuit_gate_to_index(
        gates: &Vec<GeneralCircuitGate<F>>,
        start: usize,
    ) -> Vec<GateIndex> {
        gates
            .iter()
            .enumerate()
            .map(|(index, _)| GateIndex::Absolute(start + index))
            .collect()
    }

    pub fn new_add_gate(
        label: &str,
        left: GateIndex,
        right: GateIndex,
        relative_layer: usize,
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(
            label,
            CircuitGateType::Add,
            left,
            (LayerIndex::Relative(relative_layer), right),
        )
    }

    pub fn new_xor_gate(
        label: &str,
        left: GateIndex,
        right: (LayerIndex, GateIndex),
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(label, CircuitGateType::Xor(GateF::C(F::ONE)), left, right)
    }

    pub fn new_xor_gate_array(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        relative_layer: usize,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut res: Vec<GeneralCircuitGate<F>> = vec![];

        for i in 0..input1.len() {
            let x = input1[i].clone();
            let y = input2[i].clone();
            let gate = Self::new_xor_gate(
                format!("{label}[{i}]").as_str(),
                x,
                (LayerIndex::Relative(relative_layer), y),
            );
            res.push(gate);
        }

        res
    }

    pub fn new_forward_gate(
        label: &str,
        constant: F,
        right: GateIndex,
        relative_layer: usize,
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(
            format!("{label}").as_str(),
            CircuitGateType::ForwardY(GateF::C(constant)),
            GateIndex::Dummy,
            (LayerIndex::Relative(relative_layer), right),
        )
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
        gates
            .iter()
            .enumerate()
            .map(|(i, x)| {
                Self::new_forward_gate(
                    format!("{label}[{i}]").as_str(),
                    coeffs[i],
                    x.clone(),
                    layer_relative_index,
                )
            })
            .collect()
    }

    pub fn new_mul_gate(
        label: &str,
        left: GateIndex,
        right: GateIndex,
        relative_layer: usize,
    ) -> GeneralCircuitGate<F> {
        Self::new_mul_gate_with_constant(label, left, right, relative_layer, F::ONE)
    }

    pub fn new_mul_gate_with_constant(
        label: &str,
        left: GateIndex,
        right: GateIndex,
        relative_layer: usize,
        constant: F,
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(
            label,
            CircuitGateType::Mul(GateF::C(constant)),
            left,
            (LayerIndex::Relative(relative_layer), right),
        )
    }

    pub fn new_mul_gate_array(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        in2_layer: usize,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut res: Vec<GeneralCircuitGate<F>> = vec![];

        for i in 0..input1.len() {
            let x = input1[i].clone();
            let y = input2[i].clone();
            let gate = Self::new_mul_gate(format!("{label}[{i}]").as_str(), x, y, in2_layer);
            res.push(gate);
        }

        res
    }

    pub fn new_mul_gate_array_with_constant(
        label: &str,
        input1: &Vec<GateIndex>,
        input2: &Vec<GateIndex>,
        in2_relative_layer: usize,
        constant: F,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut res: Vec<GeneralCircuitGate<F>> = vec![];

        for i in 0..input1.len() {
            let x = input1[i].clone();
            let y = input2[i].clone();
            let gate = Self::new_mul_gate_with_constant(
                format!("{label}[{i}]").as_str(),
                x,
                y,
                in2_relative_layer,
                constant,
            );
            res.push(gate);
        }

        res
    }

    pub fn new_accumulate(
        label: &str,
        nested: Vec<GeneralCircuitGate<F>>,
    ) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new_accumulate(label, nested)
    }

    pub fn new_accumulates(
        label: &str,
        nested: &Vec<Vec<GeneralCircuitGate<F>>>,
    ) -> Vec<GeneralCircuitGate<F>> {
        let mut ret = Vec::with_capacity(nested[0].len());
        for i in 0..nested.len() {
            assert_eq!(nested[0].len(), nested[i].len());
        }
        for k in 0..nested[0].len() {
            let mut tmp: Vec<GeneralCircuitGate<F>> = Vec::with_capacity(nested.len());
            for i in 0..nested.len() {
                tmp.push(nested[i][k].clone());
            }

            ret.push(Self::new_accumulate(&format!("{label}"), tmp));
        }

        ret
    }

    pub fn new_constant_gate(label: &str, c: F) -> GeneralCircuitGate<F> {
        GeneralCircuitGate::new(
            format!("{label}").as_str(),
            CircuitGateType::Constant(GateF::C(c)),
            GateIndex::Dummy,
            (LayerIndex::Relative(1), GateIndex::Absolute(0)),
        )
    }

    pub fn new_constant_gate_array(label: &str, constants: &Vec<F>) -> Vec<GeneralCircuitGate<F>> {
        let mut ret: Vec<GeneralCircuitGate<F>> = vec![];
        for i in 0..constants.len() {
            let gate = Self::new_constant_gate(label, constants[i]);
            ret.push(gate);
        }

        ret
    }
}
