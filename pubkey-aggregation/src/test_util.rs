#[cfg(test)]
pub mod tests {
    use sisulib::circuit::general_circuit::circuit::GeneralCircuit;
    use sisulib::circuit::general_circuit::layer::GeneralCircuitLayer;
    use std::fmt::Debug;
    use std::{collections::HashMap, str::FromStr};

    use crate::constant::tests::default_input;
    use crate::constant::NUM_LEN;
    use crate::input_calculator::assign_input;
    use ark_ff::{Field, PrimeField};
    use sisulib::circuit::GateIndex;

    use crate::circuit_layer::CircuitLayer;

    pub fn circuit_from_layers<F: Field>(layers: &Vec<CircuitLayer<F>>) -> GeneralCircuit<F> {
        let mut circuit = GeneralCircuit::<F>::new(layers[0].len());
        for i in 1..layers.len() {
            circuit.push_layer(GeneralCircuitLayer::new(layers[i].all_gates.clone()));
        }

        circuit.finalize(HashMap::default());

        circuit
    }

    fn assign<F: PrimeField>(
        label: &str,
        input_indexes: &HashMap<String, Vec<GateIndex>>,
        input_values: &mut Vec<F>,
        values: &Vec<u8>,
    ) {
        let indexes = input_indexes.get(label).unwrap();
        for i in 0..indexes.len() {
            input_values[indexes[i].value()] = F::from(values[i]);
        }
    }

    /// This function assign default test input values for an input layer.
    pub fn assign_input_values<F: PrimeField>(
        input_layer: &CircuitLayer<F>,
    ) -> (Vec<F>, HashMap<String, Vec<F>>)
    where
        <F as FromStr>::Err: Debug,
    {
        let input_indexes = &input_layer.indexes;
        let mut input_values: Vec<F> = vec![F::ZERO; input_layer.len()];

        let inputs = default_input();
        let assignments = assign_input::<F>(&inputs[0], &inputs[1], &inputs[2]);

        for (label, values) in &assignments {
            let indexes = input_indexes.get(label).unwrap();
            for i in 0..indexes.len() {
                input_values[indexes[i].value()] = values[i];
            }
        }

        (input_values, assignments)
    }

    pub fn assign_input_with_labels<F: Field>(
        assignments: &HashMap<String, Vec<F>>,
        labels: &Vec<&str>,
        input_indexes: &HashMap<String, Vec<GateIndex>>,
        input_values: &mut Vec<F>,
    ) {
        for label in labels {
            let indexes = input_indexes.get(*label).unwrap();
            let values = assignments.get(*label).unwrap();
            for i in 0..indexes.len() {
                input_values[indexes[i].value()] = values[i];
            }
        }
    }

    pub fn add_2_nums<F: Field>(layers: &mut Vec<CircuitLayer<F>>, a_label: &str, b_label: &str) {
        layers.push(CircuitLayer::new(true));
        layers.push(CircuitLayer::new(false));

        layers[0].add_input_zero_one(a_label, NUM_LEN);
        layers[0].add_input_zero_one(b_label, NUM_LEN);
        layers[0].add_input_zero_one("input0", 1);
    }
}
