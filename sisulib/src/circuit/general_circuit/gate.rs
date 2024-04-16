use ark_ff::Field;

use crate::circuit::{CircuitGateType, GateIndex};

use super::circuit::GeneralCircuit;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum LayerIndex {
    #[default]
    Invalid, // Do not use this type.

    /// The index since the current layer. Not accept zero.
    Relative(usize),

    /// The index since input layer of global circuit. The real index is
    /// D - idx, where D is the number layers of global circuit.
    GlobalInput(usize),

    /// The index since input layer of sub-circuit. The real index is D - idx,
    /// where D is the number of layers of subcircuit.
    LocalInput(usize),

    /// The index since output layer of global circuit. Not accept zero. The
    /// real index is idx.
    GlobalOutput(usize),

    /// The index since output layer of sub-circuit. Not accept zero. The real
    /// index is idx.
    LocalOutput(usize),
}

impl LayerIndex {
    pub fn relative_value(&self) -> usize {
        match self {
            Self::Relative(idx) => *idx,
            _ => panic!("Cannot get value of a absolute index or input index"),
        }
    }

    /// Because the relative index doesn't accept zero value, the first layer
    /// after the current layer is indexed as ONE. So if we want to use this
    /// value as vector index, we must minus it by ONE.
    pub fn value_as_vector_index(&self) -> usize {
        self.relative_value() - 1
    }

    pub fn absolute_value(&self, current_layer_index: usize) -> usize {
        current_layer_index + self.relative_value()
    }

    pub fn finalize_subcircuit(
        &mut self,
        total_subcircuit_layers: usize,
        current_layer_index: usize,
    ) {
        match self {
            Self::LocalInput(idx) => {
                *self = Self::Relative(total_subcircuit_layers - current_layer_index - *idx);
            }
            Self::LocalOutput(idx) => {
                *self = Self::Relative(*idx - current_layer_index);
            }
            _ => {}
        }
    }

    pub fn finalize(&mut self, total_circuit_layers: usize, current_layer_index: usize) {
        match self {
            Self::GlobalInput(idx) => {
                *self = Self::Relative(total_circuit_layers - current_layer_index - *idx);
            }
            Self::GlobalOutput(idx) => {
                *self = Self::Relative(*idx - current_layer_index);
            }
            Self::Relative(_) => {}
            _ => panic!("Please finalize the subcircuit before append or extend"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GeneralCircuitGate<F: Field> {
    _label: String,
    pub gate_type: CircuitGateType<F, Self>,
    pub left: GateIndex,
    pub right: (LayerIndex, GateIndex),
    pub right_subset_index: usize,
}

impl<F: Field> GeneralCircuitGate<F> {
    pub fn new(
        label: &str,
        gate_type: CircuitGateType<F, Self>,
        left: GateIndex,
        right: (LayerIndex, GateIndex),
    ) -> Self {
        Self {
            _label: String::from(label),
            gate_type,
            left,
            right,
            right_subset_index: 0,
        }
    }

    pub fn new_dummy(label: &str) -> Self {
        Self::new(
            label,
            CircuitGateType::Zero,
            GateIndex::Dummy,
            (LayerIndex::Relative(1), GateIndex::Dummy),
        )
    }

    pub fn new_accumulate(label: &str, nested: Vec<GeneralCircuitGate<F>>) -> Self {
        Self::new(
            label,
            CircuitGateType::Accumulation(nested),
            GateIndex::Dummy,
            (LayerIndex::Relative(1), GateIndex::Dummy),
        )
    }

    pub fn label(self) -> String {
        return self._label;
    }

    pub fn finalize_left(&mut self, padding_offset: usize) {
        self.left.finalize(padding_offset);
    }

    pub fn finalize_right_layer_in_subcircuit(
        &mut self,
        total_layers: usize,
        current_layer_index: usize,
    ) {
        self.right
            .0
            .finalize_subcircuit(total_layers, current_layer_index);

        match &mut self.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    subgate.finalize_right_layer_in_subcircuit(total_layers, current_layer_index);
                }
            }
            _ => {}
        }
    }

    pub fn finalize_right_layer(&mut self, total_layers: usize, current_layer_index: usize) {
        self.right.0.finalize(total_layers, current_layer_index);

        match &mut self.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    subgate.finalize_right_layer(total_layers, current_layer_index);
                }
            }
            _ => {}
        }
    }

    pub fn finalize_right_gate(&mut self, padding_gate: usize) {
        self.right.1.finalize(padding_gate);

        match &mut self.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    subgate.finalize_right_gate(padding_gate);
                }
            }
            _ => {}
        }
    }

    pub fn check_input_indices(&self, current_layer_index: usize, circuit: &GeneralCircuit<F>) {
        match &self.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    subgate.check_input_indices(current_layer_index, circuit)
                }
            }
            _ => {
                let left_layer_size = circuit.len_at(current_layer_index + 1);
                if self.left.value() >= left_layer_size {
                    panic!("the left input of gate {} is use a outbound index: the previous layer size is {}, but the index is {}",
                     self._label, left_layer_size, self.left.value());
                }

                let right_target_layer_index = self.right.0.absolute_value(current_layer_index);
                let right_layer_size = circuit.len_at(right_target_layer_index);
                if self.right.1.value() >= right_layer_size {
                    panic!("the right input of gate {} is use a outbound index: the target layer is {} with size of {}, but the index is {}",
                     self._label, right_target_layer_index, right_layer_size, self.right.1.value());
                }
            }
        }
    }
}
