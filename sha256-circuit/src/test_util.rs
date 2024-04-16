use std::collections::HashMap;

use sisulib::circuit::GateIndex;

pub fn assign_values<AssignFn>(
    indexes_map: &HashMap<String, Vec<Vec<GateIndex>>>,
    values_map: &HashMap<String, Vec<Vec<i64>>>,
    mut assign_fn: AssignFn,
) where
    AssignFn: FnMut(GateIndex, i64),
{
    let vars = vec![
        "a_i_bits",
        "e_i_bits",
        "w_i_bits",
        "h_i_bits",
        "hout_i_bits",
    ];

    for x in vars {
        let values = values_map.get(x).unwrap();
        let indexes = indexes_map.get(x).unwrap();
        assert_eq!(values.len(), indexes.len());

        for i in 0..values.len() {
            for j in 0..values[i].len() {
                assign_fn(indexes[i][j].clone(), values[i][j]);
            }
        }
    }
}

#[cfg(test)]
pub mod tests {
    use serde::{Deserialize, Serialize};
    use std::{collections::HashMap, fs};

    use ark_ff::Field;
    use sisulib::circuit::{general_circuit::circuit::GeneralCircuit, GateIndex};

    use crate::{constants::INIT_H_VALUES, layer_input::LayerInput, sha_calculation::Sha256Cal};

    use super::assign_values;

    #[derive(Debug, Deserialize, Serialize)]
    pub struct OutputResult {
        pub values: HashMap<String, Vec<i64>>,
    }

    pub fn get_circuit_and_input_layer<'a, F: Field>() -> (GeneralCircuit<F>, LayerInput<F>) {
        let input_len = 7177;
        let circuit = GeneralCircuit::<F>::new(input_len);

        // layer input
        let input_layer = LayerInput::<F>::new();
        assert_eq!(input_len, input_layer.input_len());

        (circuit, input_layer)
    }

    pub fn assign_input_values<F: Field>(arr: &[u8], input_layer: &LayerInput<F>) -> Vec<F> {
        let m = Sha256Cal::preprocess_message(arr);
        let result = Sha256Cal::calculate(&INIT_H_VALUES, &m);
        let mut input_values: Vec<F> = vec![F::ZERO; input_layer.input_len()];

        // Assign values from input layer and the SHA256 calculation results.
        let assign_fn = |index: GateIndex, value: i64| {
            input_values[index.value()] = F::from(value as u128);
        };
        assign_values(&input_layer.indexes, &result.1, assign_fn);

        input_values
    }

    pub fn check_output_by_labels<F: Field>(
        expected_file: &str,
        labels: Vec<&str>,
        circuit: &GeneralCircuit<F>,
        output_evals: &[F],
    ) {
        // Verify other outputs
        // Read expected outputs.
        println!("expected_file = {}", expected_file);
        let data = fs::read_to_string(expected_file).expect("Unable to read file");
        let expected_output: OutputResult = serde_json::from_str(data.as_str()).unwrap();

        for label in labels {
            // Find the actual outputs
            let mut actual_values: Vec<F> = vec![];
            for (i, gate) in circuit.layers[0].gates.iter().enumerate() {
                if gate.clone().label().contains(label) {
                    actual_values.push(output_evals[i]);
                }
            }

            let expected_values = expected_output.values.get(label).unwrap();
            let diff = actual_values.len() - expected_values.len();
            for i in 0..expected_values.len() {
                // let equal = convert_fn(expected_values[i]) == actual_values[i + diff];
                let equal = F::from(expected_values[i] as u128) == actual_values[i + diff];
                if !equal {
                    println!(
                        "i = {}, expected = {:?}, actual = {:?}",
                        i, expected_values[i], actual_values[i]
                    );
                }
                assert!(equal);
            }
        }
    }
}
