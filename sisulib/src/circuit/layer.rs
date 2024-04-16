use crate::{
    circuit::CircuitGate,
    codegen::generator::{CustomMLEGenerator, FuncGenerator},
    common::{combine_integer, ilog2_ceil},
    mle::sparse::SisuSparseMultilinearExtension,
};
use ark_ff::Field;
use ark_std::rand::Rng;

use super::{CircuitGateType, CircuitParams, GateF};

#[derive(Clone)]
pub struct CircuitLayer<F: Field> {
    pub gates: Vec<CircuitGate<F>>,
    pub is_finalized: bool,

    pub constant_ext: SisuSparseMultilinearExtension<F>,
    pub mul_ext: SisuSparseMultilinearExtension<F>,
    pub forward_x_ext: SisuSparseMultilinearExtension<F>,
    pub forward_y_ext: SisuSparseMultilinearExtension<F>,
}

impl<F: Field> CircuitLayer<F> {
    pub fn default() -> Self {
        Self::new(vec![])
    }

    pub fn new(gates: Vec<CircuitGate<F>>) -> Self {
        Self {
            gates,
            is_finalized: false,
            constant_ext: SisuSparseMultilinearExtension::default(),
            mul_ext: SisuSparseMultilinearExtension::default(),
            forward_x_ext: SisuSparseMultilinearExtension::default(),
            forward_y_ext: SisuSparseMultilinearExtension::default(),
        }
    }

    pub fn finalize(&mut self, input_num_vars: usize) {
        if self.is_finalized {
            return;
        }

        let num_vars = self.num_vars() as usize + 2 * input_num_vars;
        self.constant_ext.set_num_vars(num_vars);
        self.mul_ext.set_num_vars(num_vars);
        self.forward_x_ext.set_num_vars(num_vars);
        self.forward_y_ext.set_num_vars(num_vars);

        let gates = self.gates.clone();
        for (i, gate) in gates.iter().enumerate() {
            self.handle_single_gate_mle(input_num_vars, i, &gate);
        }

        self.is_finalized = true;
    }

    /// The length of the layer.
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    pub fn add_gate(&mut self, gate: CircuitGate<F>) {
        assert!(
            !self.is_finalized,
            "The layer cannot add gate after finalized"
        );

        self.gates.push(gate);
    }

    pub fn add_gates(&mut self, gates: Vec<CircuitGate<F>>) {
        for gate in gates {
            self.gates.push(gate);
        }
    }

    fn num_vars(&self) -> usize {
        ilog2_ceil(self.gates.len())
    }

    fn handle_single_gate_mle(
        &mut self,
        input_num_vars: usize,
        out_index: usize,
        gate: &CircuitGate<F>,
    ) {
        let output_num_vars = self.num_vars();
        let point = combine_integer(vec![
            (out_index, output_num_vars),
            (gate.input_indices[0].value(), input_num_vars),
            (gate.input_indices[1].value(), input_num_vars),
        ]);

        match &gate.gate_type {
            CircuitGateType::Constant(c) => self.constant_ext.evaluations.push((point, c.clone())),
            CircuitGateType::Add => {
                let point_x = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                ]);
                self.forward_x_ext.evaluations.push((point_x, GateF::ONE));

                let point_y = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                ]);
                self.forward_x_ext.evaluations.push((point_y, GateF::ONE));
            }
            CircuitGateType::Sub => {
                let point_x = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                ]);
                self.forward_x_ext.evaluations.push((point_x, GateF::ONE));

                let point_y = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                ]);
                self.forward_x_ext
                    .evaluations
                    .push((point_y, GateF::C(F::ZERO - F::ONE)));
            }
            CircuitGateType::CAdd(c) => {
                self.constant_ext.evaluations.push((point, c.clone()));
                self.forward_x_ext.evaluations.push((point, GateF::ONE));
            }
            CircuitGateType::CSub(c) => {
                self.constant_ext.evaluations.push((point, c.clone()));
                self.forward_x_ext
                    .evaluations
                    .push((point, GateF::C(F::ZERO - F::ONE)));
            }
            CircuitGateType::Mul(c) => self.mul_ext.evaluations.push((point, c.clone())),
            CircuitGateType::ForwardX(c) => self.forward_x_ext.evaluations.push((point, c.clone())),
            CircuitGateType::ForwardY(c) => self.forward_y_ext.evaluations.push((point, c.clone())),
            CircuitGateType::Fourier(domain) => {
                // GATE FOURIER: forward(g, x, y) * V(x) + domain * forward(g, x, y) * V(y).

                let point_x = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                ]);
                self.forward_x_ext.evaluations.push((point_x, GateF::ONE));

                let point_y = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                ]);
                self.forward_x_ext
                    .evaluations
                    .push((point_y, domain.clone()));
            }
            CircuitGateType::Xor(c) => {
                // GATE XOR: xor * (1 - ((1-in1) * (1-in2) + in1*in2))
                // = xor * (1 - (1 - in1 - in2 + in1 * in2 + in1 * in2))
                // = forward_x * in1 + forward_y * in2 - 2*in1*in2

                // forward_x * in1 + forward_y * in2
                let point_x = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                    (gate.input_indices[0].value(), input_num_vars),
                ]);
                self.forward_x_ext.evaluations.push((point_x, c.clone()));

                let point_y = combine_integer(vec![
                    (out_index, output_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                    (gate.input_indices[1].value(), input_num_vars),
                ]);
                self.forward_x_ext.evaluations.push((point_y, c.clone()));

                // -2 * mul * in1 * in2
                self.mul_ext
                    .evaluations
                    .push((point, c.clone() * GateF::C(F::ZERO - F::from(2u64))));
            }
            CircuitGateType::NAAB => {
                // GATE XOR: naab * (1-in1) * in2
                // = naab * (in2 - in1*in2)

                // forward * in2
                self.forward_y_ext.evaluations.push((point, GateF::ONE));

                // - mul * in1 * in2
                self.mul_ext
                    .evaluations
                    .push((point, GateF::C(F::ZERO - F::ONE)));
            }
            CircuitGateType::Accumulation(accumulated_gates) => {
                for gate in accumulated_gates {
                    self.handle_single_gate_mle(input_num_vars, out_index, gate)
                }
            }
            CircuitGateType::Zero => {}
        }
    }

    pub fn gen_code(
        &self,
        params: &CircuitParams<F>,
        gkr_index: usize,
        layer_index: usize,
    ) -> (Vec<FuncGenerator<F>>, Vec<CustomMLEGenerator>) {
        let mut funcs = vec![];
        let mut mle = vec![];

        let (f, t) = self
            .constant_ext
            .gen_code_for_gkr(params, gkr_index, layer_index, 0);
        funcs.extend(f);
        mle.extend(t);

        let (f, t) = self
            .mul_ext
            .gen_code_for_gkr(params, gkr_index, layer_index, 1);
        funcs.extend(f);
        mle.extend(t);

        let (f, t) = self
            .forward_x_ext
            .gen_code_for_gkr(params, gkr_index, layer_index, 2);
        funcs.extend(f);
        mle.extend(t);

        let (f, t) = self
            .forward_y_ext
            .gen_code_for_gkr(params, gkr_index, layer_index, 3);
        funcs.extend(f);
        mle.extend(t);

        (funcs, mle)
    }

    pub fn test_mle<R: Rng>(&self, rng: &mut R, circuit_params: &CircuitParams<F>) -> bool {
        let mut final_result = true;

        if self.constant_ext.num_evaluations() > 0 {
            println!("--------------------- Constant Extension ----------------------");

            let result = self
                .constant_ext
                .test_over_boolean_hypercube(circuit_params);
            println!("BooleanHypercube: {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }

            let result = self.constant_ext.test_random(rng, circuit_params);
            println!("Random point:     {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }
        }

        if self.mul_ext.num_evaluations() > 0 {
            println!("--------------------- Mul Extension ----------------------");

            let result = self.mul_ext.test_over_boolean_hypercube(circuit_params);
            println!("BooleanHypercube: {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }

            let result = self.mul_ext.test_random(rng, circuit_params);
            println!("Random point:     {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }
        }

        if self.forward_x_ext.num_evaluations() > 0 {
            println!("--------------------- ForwardX Extension ----------------------");

            let result = self
                .forward_x_ext
                .test_over_boolean_hypercube(circuit_params);
            println!("BooleanHypercube: {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }

            let result = self.forward_x_ext.test_random(rng, circuit_params);
            println!("Random point:     {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }
        }

        if self.forward_y_ext.num_evaluations() > 0 {
            println!("--------------------- ForwardY Extension ----------------------");

            let result = self
                .forward_y_ext
                .test_over_boolean_hypercube(circuit_params);
            println!("BooleanHypercube: {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }

            let result = self.forward_y_ext.test_random(rng, circuit_params);
            println!("Random point:     {:?}", result);

            if let Some(r) = result {
                final_result = final_result && r;
            }
        }

        final_result
    }
}
