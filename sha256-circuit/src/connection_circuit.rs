use std::collections::HashMap;

use ark_ff::Field;
use sisulib::{
    circuit::{
        general_circuit::{
            circuit::GeneralCircuit,
            gate::{GeneralCircuitGate, LayerIndex},
            layer::GeneralCircuitLayer,
        },
        CircuitGateType, GateF, GateIndex,
    },
    common::dec2bin,
};

use crate::constants::INIT_H_VALUES;

pub fn create_merkle_tree_connection_circuit<F: Field>(path_size: usize) -> GeneralCircuit<F> {
    // For each SHA256 circuit, we only get these values as input of this
    // circuit:
    // W0 (256) + W1 (256) + H_IN(256) + H_OUT(256)
    let single_input_size = 1024;

    // For each element in path, we need two SHA256 circuits to hash 512bit
    // input.
    let n_sha256_circuits = path_size * 2;

    // Calculate the full circuit size.
    let input_size = n_sha256_circuits * single_input_size;

    let mut circuit = GeneralCircuit::new(input_size);

    // COMPRESSION LAYER
    //
    // Convert every 256-gate group into two numbers, each number is 128 bits.
    //
    // The reason why we compress a group to two numbers is that we are using
    // FrBN256, that represents for a number with maximum of 253 bits.
    //
    // After this layer, each single circuit has 8 gates:
    // 2 W0
    // 2 W1
    // 2 H_IN
    // 2 H_OUT
    let mut compression_layer = GeneralCircuitLayer::default();
    let n_groups = input_size / 256;
    for group_index in 0..n_groups {
        for number_index in 0..2 {
            let mut accumulated_gates = vec![];
            let mut pow = F::ONE;

            for gate_relative_index in 0..128 {
                let gate_index = group_index * 256 + number_index * 128 + gate_relative_index;

                accumulated_gates.push(GeneralCircuitGate::new(
                    &format!("{}", gate_index),
                    CircuitGateType::ForwardX(GateF::C(pow)),
                    GateIndex::Absolute(gate_index),
                    (LayerIndex::Relative(1), GateIndex::Absolute(gate_index)),
                ));

                pow = pow + pow;
            }

            compression_layer.add_gate(GeneralCircuitGate::new_accumulate(
                &format!("compression-{}-{}", group_index, number_index),
                accumulated_gates,
            ));
        }
    }
    circuit.push_layer(compression_layer);

    // CONSTANT
    let compression_layer_size_of_each_sha256 = 8;
    let compression_layer_w0_start_index = 0;
    let compression_layer_w1_start_index = 2;
    let compression_layer_h_in_start_index = 4;
    let compression_layer_h_out_start_index = 6;

    // COMPARISON LAYERS
    //
    // For each element in merkle path, we have two SHA256 circuit, call them as
    // SHA1 and SHA2. We must check the following conditions:
    // 1. SHA1.H_IN  === INIT_H
    // 2. SHA2.H_IN  === SHA1.H_OUT
    // 3. SHA2.W0    === PADDING
    // 4. SHA2.H_OUT === NEXT_SHA1.W0 or NEXT_SHA1.W1
    //
    // Gate order in COMPRESSION LAYER:
    // 1. SHA1.H_IN  - INIT_H.
    // 2. SHA2.H_IN  - SHA1.H_OUT
    // 3. SHA2.W0    - PADDING
    // 4. SHA2.H_OUT - NEXT_SHA1.W0
    // 5. SHA2.H_OUT - NEXT_SHA1.W1

    let mut comparison_layer = GeneralCircuitLayer::default();

    // COMPARISON LAYER: SHA1.H_IN === INIT_H.
    let mut init_h_bits: Vec<F> = vec![];
    for i in 0..8 {
        init_h_bits.extend(dec2bin::<_, F>(INIT_H_VALUES[i] as u64, 32));
    }

    // At this time, init_h_bits has the size of 32*8 = 256.
    // We need compress this value to two values as compression layer.
    let mut init_h_128bit_numbers = vec![];
    for number_index in 0..2 {
        let mut number = F::ZERO;
        let mut pow = F::ONE;
        for relative_idx in 0..128 {
            let idx = number_index * 128 + relative_idx;
            number += init_h_bits[idx] * pow;
            pow = pow + pow;
        }

        init_h_128bit_numbers.push(number);
    }

    for path_index in 0..path_size {
        for number_index in 0..2 {
            let gate_index = path_index * compression_layer_size_of_each_sha256 * 2
                + compression_layer_h_in_start_index
                + number_index;

            comparison_layer.add_gate(GeneralCircuitGate::new(
                &format!("comparison-SHA1[{}]-H_IN[{}]", path_index, number_index),
                CircuitGateType::CSub(GateF::C(init_h_128bit_numbers[number_index])),
                GateIndex::Absolute(gate_index),
                (LayerIndex::Relative(1), GateIndex::Absolute(gate_index)),
            ));
        }
    }

    // COMPARISON LAYER: SHA2.H_IN === SHA1.H_OUT.
    for path_index in 0..path_size {
        for number_index in 0..2 {
            let sha1_h_out_index = path_index * compression_layer_size_of_each_sha256 * 2
                + compression_layer_h_out_start_index
                + number_index;

            let sha2_h_in_index = path_index * compression_layer_size_of_each_sha256 * 2
                + compression_layer_size_of_each_sha256
                + compression_layer_h_in_start_index
                + number_index;

            comparison_layer.add_gate(GeneralCircuitGate::new(
                &format!("comparison-SHA2[{}]-H_IN[{}]", path_index, number_index),
                CircuitGateType::Sub,
                GateIndex::Absolute(sha2_h_in_index),
                (
                    LayerIndex::Relative(1),
                    GateIndex::Absolute(sha1_h_out_index),
                ),
            ));
        }
    }

    // COMPARISON LAYER: SHA2.W0 === PADDING.
    let mut padding = vec![];
    padding.push(0x80);
    while (padding.len() * 8 + 64) % 512 != 0 {
        padding.push(0x00);
    }
    padding.extend(512usize.to_be_bytes());
    assert!(padding.len() * 8 % 512 == 0);

    // At this time, padding has the size of 512/8 = 64.
    let mut padding_bits: Vec<F> = vec![];
    for i in 0..64 {
        padding_bits.extend(dec2bin::<_, F>(padding[i] as u64, 8));
    }

    // We need compress this value to two values as compression layer.
    let mut padding_128bit_numbers = vec![];
    for number_index in 0..2 {
        let mut number = F::ZERO;
        let mut pow = F::ONE;
        for relative_idx in 0..128 {
            let idx = number_index * 128 + relative_idx;
            number += padding_bits[idx] * pow;
            pow = pow + pow;
        }

        padding_128bit_numbers.push(number);
    }

    for path_index in 0..path_size {
        for number_index in 0..2 {
            let gate_index = path_index * compression_layer_size_of_each_sha256 * 2
                + compression_layer_size_of_each_sha256
                + compression_layer_w0_start_index
                + number_index;

            comparison_layer.add_gate(GeneralCircuitGate::new(
                &format!("comparison-SHA2[{}]-W0[{}]", path_index, number_index),
                CircuitGateType::CSub(GateF::C(padding_128bit_numbers[number_index])),
                GateIndex::Absolute(gate_index),
                (LayerIndex::Relative(1), GateIndex::Absolute(gate_index)),
            ));
        }
    }

    // COMPARISON LAYER:
    // + SHA2.H_OUT === NEXT_SHA1.W0.
    // + SHA2.H_OUT === NEXT_SHA1.W1.
    for path_index in 0..path_size - 1 {
        for number_index in 0..2 {
            let sha2_h_out_index = path_index * compression_layer_size_of_each_sha256 * 2
                + compression_layer_size_of_each_sha256
                + compression_layer_h_out_start_index
                + number_index;

            let next_sha1_w0_index = (path_index + 1) * compression_layer_size_of_each_sha256 * 2
                + compression_layer_w0_start_index
                + number_index;

            comparison_layer.add_gate(GeneralCircuitGate::new(
                &format!("comparison-SHA2[{}]-H_OUT[{}]-W0", path_index, number_index),
                CircuitGateType::Sub,
                GateIndex::Absolute(sha2_h_out_index),
                (
                    LayerIndex::Relative(1),
                    GateIndex::Absolute(next_sha1_w0_index),
                ),
            ));
        }

        for number_index in 0..2 {
            let sha2_h_out_index = path_index * compression_layer_size_of_each_sha256 * 2
                + compression_layer_size_of_each_sha256
                + compression_layer_h_out_start_index
                + number_index;

            let next_sha1_w1_index = (path_index + 1) * compression_layer_size_of_each_sha256 * 2
                + compression_layer_w1_start_index
                + number_index;

            comparison_layer.add_gate(GeneralCircuitGate::new(
                &format!("comparison-SHA2[{}]-H_OUT[{}]-W1", path_index, number_index),
                CircuitGateType::Sub,
                GateIndex::Absolute(sha2_h_out_index),
                (
                    LayerIndex::Relative(1),
                    GateIndex::Absolute(next_sha1_w1_index),
                ),
            ));
        }
    }

    circuit.push_layer(comparison_layer);
    circuit.finalize(HashMap::default());

    circuit
}
