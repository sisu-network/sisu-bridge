use ark_ff::Field;
use ark_std::rand::Rng;

use crate::circuit::{CircuitGateType, CircuitLayer, GateF};
use crate::codegen::generator::{CustomMLEGenerator, FuncGenerator};
use crate::common::{ilog2_ceil, padding_pow_of_two_size};
use crate::domain::Domain;

#[derive(Default)]
pub struct CircuitEvaluations<F> {
    evaluations: Vec<Vec<F>>,
    size: Vec<usize>,
}

impl<F: Copy> CircuitEvaluations<F> {
    pub fn new(size: Vec<usize>, evaluations: Vec<Vec<F>>) -> Self {
        Self { size, evaluations }
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn at_layer(&self, layer_index: usize, padding: bool) -> &[F] {
        if padding {
            &self.evaluations[layer_index]
        } else {
            &self.evaluations[layer_index][..self.size[layer_index]]
        }
    }
}

#[derive(Clone, Debug)]
pub struct CircuitParams<'a, F: Field> {
    pub one: F,
    pub f: Vec<F>,
    pub d: Vec<usize>,
    pub domain: Option<Domain<'a, F>>,
}

impl<'a, F: Field> CircuitParams<'a, F> {
    pub const fn default() -> Self {
        Self {
            one: F::ONE,
            f: vec![],
            d: vec![],
            domain: None,
        }
    }

    pub fn with_domain(domain: Domain<'a, F>) -> Self {
        Self {
            one: F::ONE,
            f: vec![],
            d: vec![],
            domain: Some(domain),
        }
    }

    pub fn get_domain(&self) -> &Domain<'a, F> {
        self.domain.as_ref().unwrap()
    }
}

#[derive(Clone)]
pub struct Circuit<F: Field> {
    pub layers: Vec<CircuitLayer<F>>, // The first layer is input, the last layer is output.
    pub number_inputs: usize,
}

impl<F: Field> Circuit<F> {
    pub fn default() -> Self {
        Self::new(0)
    }

    pub fn new(number_inputs: usize) -> Self {
        Self::new_with_layer(vec![], number_inputs)
    }

    pub fn new_with_layer(layers: Vec<CircuitLayer<F>>, number_inputs: usize) -> Self {
        let mut circuit = Self {
            layers: vec![],
            number_inputs,
        };

        for layer in layers.into_iter().rev() {
            circuit.push_layer(layer);
        }
        circuit
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn len_at(&self, layer_index: usize) -> usize {
        if layer_index == self.layers.len() {
            return self.number_inputs;
        }

        let layer_index = self.layers.len() - layer_index - 1;
        self.layers[layer_index].len()
    }

    pub fn input_size(&self) -> usize {
        self.number_inputs
    }

    pub fn num_vars_at(&self, layer_index: usize) -> usize {
        ilog2_ceil(self.len_at(layer_index))
    }

    pub fn layer(&self, layer_index: usize) -> &CircuitLayer<F> {
        let layer_index = self.layers.len() - layer_index - 1;
        &self.layers[layer_index]
    }

    /// Finalize a layer to build mle for that layer.
    pub fn finalize_layer(&mut self, layer_index: usize) {
        let input_num_vars = self.num_vars_at(layer_index + 1); // log2(next layer size)
        let n = self.layers.len() - layer_index - 1;
        self.layers[n].finalize(input_num_vars);
    }

    /// Finalize a circuit to build mle for all layers.
    pub fn finalize(&mut self) {
        for i in 0..self.len() {
            self.finalize_layer(i);
        }
    }

    /// Replace a current layer to a new one.
    pub fn replace(&mut self, layer_index: usize, layer: CircuitLayer<F>) {
        let n = self.layers.len() - layer_index - 1;
        self.layers[n] = layer.clone();
    }

    /// Push a new layer into circuit as output layer.
    pub fn push_layer(&mut self, layer: CircuitLayer<F>) {
        // Create a dummy layer to replace.
        self.layers.push(CircuitLayer::default());
        self.replace(0, layer);
    }

    /// Append the current circuit by another circuit.
    pub fn push_circuit(&mut self, other: Self) {
        assert!(
            self.len() == 0 || self.len_at(0) >= other.number_inputs,
            "output of previous circuit must large than input of the current circuit ({} < {})",
            self.len_at(0),
            other.number_inputs,
        );

        if self.len() == 0 {
            self.number_inputs += other.number_inputs;
        }

        for mut layer in other.layers {
            for gate in &mut layer.gates {
                gate.finalize(0);
            }

            self.push_layer(layer);
        }
    }

    /// Extend a layer by another layer.
    pub fn extend_at_layer(&mut self, layer_index: usize, layer: CircuitLayer<F>) {
        let prev_layer_size = self.len_at(layer_index + 1);

        let mut new_layer = self.layer(layer_index).clone();
        new_layer.is_finalized = false;

        for (gate_index, gate) in layer.gates.into_iter().enumerate() {
            assert!(
                gate.check_input_indices(prev_layer_size),
                "Gate {} of the layer {} is using a unbound input index",
                gate_index,
                layer_index,
            );

            new_layer.add_gate(gate);
        }

        self.replace(layer_index, new_layer);
    }

    /// Extend a circuit by another circuit.
    pub fn extend_by_circuit(
        &mut self,
        mut extended_layer_index: usize,
        other: Self,
        extend_input: bool,
    ) {
        assert!(
            extended_layer_index == other.len(),
            "Only support extending a circuit with length of {} at layer {}",
            extended_layer_index,
            extended_layer_index
        );

        if extended_layer_index == self.len() {
            if extend_input {
                self.number_inputs += other.number_inputs;
            } else {
                assert!(
                    self.number_inputs == other.number_inputs,
                    "Two circuits has different inputs"
                )
            }

            extended_layer_index -= 1;
        }

        let mut prev_layer_size = self.len_at(extended_layer_index + 1);
        for (i, mut extended_layer) in other.layers.into_iter().enumerate() {
            for gate in &mut extended_layer.gates {
                gate.finalize(prev_layer_size);
            }

            let layer_index = extended_layer_index - i;
            prev_layer_size = self.len_at(layer_index);
            self.extend_at_layer(layer_index, extended_layer);
        }
    }

    pub fn evaluate(
        &self,
        circuit_params: &CircuitParams<F>,
        witness: &[F],
    ) -> CircuitEvaluations<F> {
        assert_eq!(witness.len(), self.number_inputs);

        let mut layer_size = vec![self.number_inputs];

        let mut input_layer = witness.to_vec();
        padding_pow_of_two_size(&mut input_layer);

        let mut layer_evaluations = vec![input_layer];

        for layer in self.layers.iter() {
            assert!(
                layer.is_finalized,
                "The layer must be finalized before evaluate"
            );

            let current_input = layer_evaluations.last().unwrap();

            let mut output = vec![F::ZERO; layer.len()];
            for (gate_index, gate) in layer.gates.iter().enumerate() {
                let in1 = current_input[gate.input_indices[0].value()];
                let in2 = current_input[gate.input_indices[1].value()];
                output[gate_index] = match &gate.gate_type {
                    // CircuitGateType::Constant(constant) => {
                    //      constant.to_value(&self.params)
                    // }
                    CircuitGateType::Add => in1 + in2,
                    // CircuitGateType::Sub => in1 - in2,
                    CircuitGateType::Mul(constant) => in1 * in2 * constant.to_value(circuit_params),
                    // CircuitGateType::Xor(constant) => output.push(
                    //     (in1 + in2 - F::from(2u64) * in1 * in2) * constant.to_value(circuit_params),
                    // ),
                    // CircuitGateType::NAAB => (F::ONE - in1) * in2,
                    CircuitGateType::Fourier(domain) => in1 + in2 * domain.to_value(circuit_params),
                    CircuitGateType::ForwardX(constant) => {
                        if constant == &GateF::ONE {
                            in1
                        } else {
                            in1 * constant.to_value(circuit_params)
                        }
                    }
                    CircuitGateType::ForwardY(constant) => {
                        if constant == &GateF::ONE {
                            in2
                        } else {
                            in2 * constant.to_value(circuit_params)
                        }
                    }
                    CircuitGateType::Accumulation(accumulated_gates) => {
                        let mut acc = F::ZERO;
                        for acc_gate in accumulated_gates {
                            acc += match &acc_gate.gate_type {
                                CircuitGateType::Add => {
                                    current_input[acc_gate.input_indices[0].value()]
                                        + current_input[acc_gate.input_indices[1].value()]
                                }
                                CircuitGateType::Mul(constant) => {
                                    constant.to_value(circuit_params)
                                        * current_input[acc_gate.input_indices[0].value()]
                                        * current_input[acc_gate.input_indices[1].value()]
                                }
                                CircuitGateType::ForwardX(constant) => {
                                    current_input[acc_gate.input_indices[0].value()]
                                        * constant.to_value(circuit_params)
                                }
                                CircuitGateType::ForwardY(constant) => {
                                    current_input[acc_gate.input_indices[1].value()]
                                        * constant.to_value(circuit_params)
                                }
                                _ => {
                                    panic!(
                                        "Currently not support accumulate for gate {:?}",
                                        acc_gate.gate_type
                                    )
                                }
                            }
                        }
                        acc
                    }
                    CircuitGateType::Zero => F::ZERO,
                    _ => panic!("not support gate {:?}", gate.gate_type),
                }
            }

            padding_pow_of_two_size(&mut output);
            layer_size.push(layer.len());
            layer_evaluations.push(output);
        }

        layer_evaluations.reverse();
        layer_size.reverse();

        CircuitEvaluations {
            size: layer_size,
            evaluations: layer_evaluations,
        }
    }

    pub fn gen_code(
        &self,
        gkr_index: usize,
        circuit_params: &CircuitParams<F>,
    ) -> (Vec<FuncGenerator<F>>, Vec<CustomMLEGenerator>) {
        let mut funcs = vec![];
        let mut mle = vec![];

        for i in 0..self.len() {
            let (f, t) = self.layer(i).gen_code(circuit_params, gkr_index, i);
            funcs.extend(f);
            mle.extend(t);
        }

        (funcs, mle)
    }

    pub fn test_mle<R: Rng>(&self, rng: &mut R, circuit_params: &CircuitParams<F>) -> bool {
        for i in 0..self.len() {
            println!("[Layer {}]", i);
            println!("========================================");
            if !self.layer(i).test_mle(rng, circuit_params) {
                return false;
            }
            println!("");
        }

        true
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{
        circuit::{
            Circuit, CircuitGate, CircuitGateType, CircuitLayer, CircuitParams, GateF, GateIndex,
        },
        field::Fp389,
    };
    use ark_ff::Field;

    pub fn example_circuit<F: Field>() -> Circuit<F> {
        //        +      x
        //     x     x      +   dummy
        //  +     +     x      +
        //  01    11    02     13
        //
        //  1     2     3      4
        Circuit::new_with_layer(
            vec![
                CircuitLayer::new(vec![
                    CircuitGate::new(
                        "0-0",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(0), GateIndex::Absolute(1)],
                    ),
                    CircuitGate::new(
                        "0-1",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(1), GateIndex::Absolute(2)],
                    ),
                ]),
                CircuitLayer::new(vec![
                    CircuitGate::new(
                        "1-0",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(0), GateIndex::Absolute(1)],
                    ),
                    CircuitGate::new(
                        "1-1",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(1), GateIndex::Absolute(2)],
                    ),
                    CircuitGate::new(
                        "1-2",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(2), GateIndex::Absolute(3)],
                    ),
                    // This is a dummy gate so as to the layer size will be a power of 2
                    CircuitGate::new(
                        "1-3",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(0), GateIndex::Absolute(0)],
                    ),
                ]),
                CircuitLayer::new(vec![
                    CircuitGate::new(
                        "2-0",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(0), GateIndex::Absolute(1)],
                    ),
                    CircuitGate::new(
                        "2-1",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(1), GateIndex::Absolute(1)],
                    ),
                    CircuitGate::new(
                        "2-2",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(0), GateIndex::Absolute(2)],
                    ),
                    CircuitGate::new(
                        "2-3",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(1), GateIndex::Absolute(3)],
                    ),
                ]),
            ],
            4,
        )
    }

    #[test]
    fn circuit_evaluation() {
        let mut circuit = example_circuit();
        circuit.finalize();

        let evaluation = circuit.evaluate(
            &CircuitParams::default(),
            &[
                Fp389::from(1),
                Fp389::from(2),
                Fp389::from(3),
                Fp389::from(4),
            ],
        );

        // check values of output (layer 0)
        assert_eq!(
            evaluation.at_layer(0, false),
            vec![Fp389::from(24), Fp389::from(108)]
        );
    }
}
