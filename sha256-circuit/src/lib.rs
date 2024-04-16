use std::collections::HashMap;

use ark_ff::Field;
use sisulib::circuit::general_circuit::circuit::GeneralCircuit;

mod layer;
mod layer1;
mod layer2;
mod layer3;
mod test_util;

pub mod connection_circuit;
pub mod constants;
pub mod layer_input;
pub mod sha_calculation;

pub fn build_circuit<F: Field>() -> GeneralCircuit<F> {
    let input_len = 7177;
    let mut circuit = GeneralCircuit::<F>::new(input_len);

    let input_layer = layer_input::LayerInput::<F>::new();
    assert_eq!(input_len, input_layer.input_len());

    // Layer 1
    let mut layer1_builder = layer1::Layer1Builder::<F>::new(&input_layer);
    layer1_builder.build(&mut circuit);

    // Layer 2
    let mut layer2_builder = layer2::Layer2Builder::<F>::new(&input_layer, &layer1_builder);
    layer2_builder.build(&mut circuit);

    // Layer 3
    let mut layer3_builder =
        layer3::Layer3Builder::<F>::new(&input_layer, &layer1_builder, &layer2_builder);
    layer3_builder.build(&mut circuit);

    circuit.finalize(HashMap::default());

    circuit
}
