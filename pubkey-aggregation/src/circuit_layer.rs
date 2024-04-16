use std::collections::HashMap;

use ark_ff::Field;
use circuitlib::layer::Layer;
use sisulib::circuit::{general_circuit::gate::GeneralCircuitGate, GateIndex};

pub struct CircuitLayer<F: Field> {
    pub is_input: bool,
    pub all_gates: Vec<GeneralCircuitGate<F>>,
    pub indexes: HashMap<String, Vec<GateIndex>>,
    pub index: usize,
    zero_one_indexes: Vec<usize>,
}

impl<F: Field> CircuitLayer<F> {
    pub fn new(is_input: bool) -> Self {
        Self {
            is_input,
            all_gates: vec![],
            indexes: HashMap::default(),
            index: 0,
            zero_one_indexes: vec![],
        }
    }

    pub fn add_gates(&mut self, label: &str, gates: Vec<GeneralCircuitGate<F>>) {
        let mut temp_values: Vec<GateIndex> = vec![];
        temp_values.extend(Layer::general_circuit_gate_to_index(
            &gates,
            self.all_gates.len(),
        ));
        self.insert_indexes(label, temp_values);
        self.all_gates.extend(gates);
    }

    pub fn insert_indexes(&mut self, label: &str, indexes: Vec<GateIndex>) {
        self.index += indexes.len();
        self.indexes.insert(label.to_string(), indexes);
    }

    pub fn add_input(&mut self, label: &str, len: usize, is_zero_one: bool) {
        let old_index = self.index;
        let mut indexes: Vec<GateIndex> = Vec::with_capacity(len);
        for i in 0..len {
            indexes.push(GateIndex::Absolute(self.index + i));
        }
        self.insert_indexes(label, indexes);

        if is_zero_one && self.is_input {
            for i in old_index..self.index {
                self.zero_one_indexes.push(i);
            }
        }
    }

    pub fn zero_one_indexes(&self) -> Vec<usize> {
        self.zero_one_indexes.clone()
    }

    pub fn add_input_zero_one(&mut self, label: &str, len: usize) {
        self.add_input(label, len, true);
    }

    pub fn get_indexes(&self, label: &str) -> &Vec<GateIndex> {
        self.indexes.get(label).unwrap()
    }

    pub fn len(&self) -> usize {
        if self.is_input {
            return self.index;
        }

        self.all_gates.len()
    }
}
