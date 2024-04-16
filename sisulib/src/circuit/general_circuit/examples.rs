use ark_ff::Field;

use crate::circuit::{CircuitGateType, GateF, GateIndex};

use super::{
    circuit::GeneralCircuit,
    gate::{GeneralCircuitGate, LayerIndex},
    layer::GeneralCircuitLayer,
};

pub fn example_general_circuit<F: Field>() -> GeneralCircuit<F> {
    GeneralCircuit::from_layers(
        vec![
            GeneralCircuitLayer::new(vec![
                GeneralCircuitGate::new(
                    // Output layer - layer 0
                    "0-0",
                    CircuitGateType::Add,
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(0)),
                ),
                GeneralCircuitGate::new(
                    // Output layer - layer 0
                    "0-1",
                    CircuitGateType::Add,
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(1)),
                ),
                GeneralCircuitGate::new(
                    // Output layer - layer 0
                    "0-2",
                    CircuitGateType::Mul(GateF::ONE),
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(0)),
                ),
                GeneralCircuitGate::new(
                    // Output layer - layer 0
                    "0-3",
                    CircuitGateType::Mul(GateF::ONE),
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(1)),
                ),
                GeneralCircuitGate::new(
                    "0-6",
                    CircuitGateType::Accumulation(vec![
                        GeneralCircuitGate::new(
                            "0-0",
                            CircuitGateType::Mul(GateF::ONE),
                            GateIndex::Absolute(0),
                            (LayerIndex::Relative(2), GateIndex::Absolute(0)),
                        ),
                        GeneralCircuitGate::new(
                            "0-1",
                            CircuitGateType::Mul(GateF::from(2u64)),
                            GateIndex::Absolute(1),
                            (LayerIndex::Relative(2), GateIndex::Absolute(1)),
                        ),
                        GeneralCircuitGate::new(
                            "0-2",
                            CircuitGateType::Mul(GateF::ONE),
                            GateIndex::Absolute(1),
                            (LayerIndex::Relative(2), GateIndex::Absolute(2)),
                        ),
                        GeneralCircuitGate::new(
                            "0-3",
                            CircuitGateType::Add,
                            GateIndex::Absolute(1),
                            (LayerIndex::Relative(2), GateIndex::Absolute(0)),
                        ),
                        GeneralCircuitGate::new(
                            "0-4",
                            CircuitGateType::Add,
                            GateIndex::Absolute(1),
                            (LayerIndex::Relative(2), GateIndex::Absolute(1)),
                        ),
                        GeneralCircuitGate::new(
                            "0-5",
                            CircuitGateType::Add,
                            GateIndex::Absolute(1),
                            (LayerIndex::Relative(2), GateIndex::Absolute(2)),
                        ),
                    ]),
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(0)),
                ),
            ]),
            GeneralCircuitLayer::new(vec![
                // layer 1
                GeneralCircuitGate::new(
                    "1-0",
                    CircuitGateType::Mul(GateF::ONE),
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(1)),
                ),
                GeneralCircuitGate::new(
                    "1-1",
                    CircuitGateType::Mul(GateF::ONE),
                    GateIndex::Absolute(1),
                    (LayerIndex::Relative(1), GateIndex::Absolute(2)),
                ),
                GeneralCircuitGate::new(
                    "1-2",
                    CircuitGateType::Mul(GateF::ONE),
                    GateIndex::Absolute(2),
                    (LayerIndex::Relative(1), GateIndex::Absolute(2)),
                ),
            ]),
            GeneralCircuitLayer::new(vec![
                GeneralCircuitGate::new(
                    "2-0",
                    CircuitGateType::Mul(GateF::ONE),
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(0)),
                ),
                GeneralCircuitGate::new(
                    "2-1",
                    CircuitGateType::Add,
                    GateIndex::Absolute(1),
                    (LayerIndex::Relative(1), GateIndex::Absolute(1)),
                ),
                GeneralCircuitGate::new(
                    "2-2",
                    CircuitGateType::Mul(GateF::ONE),
                    GateIndex::Absolute(0),
                    (LayerIndex::Relative(1), GateIndex::Absolute(2)),
                ),
                GeneralCircuitGate::new(
                    "2-3",
                    CircuitGateType::Add,
                    GateIndex::Absolute(1),
                    (LayerIndex::Relative(1), GateIndex::Absolute(2)),
                ),
            ]),
        ],
        3,
    )
}
