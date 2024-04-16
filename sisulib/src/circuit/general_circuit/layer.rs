use std::{collections::HashMap, ops::Index};

use ark_ff::Field;

use crate::{
    circuit::{
        general_circuit::gate::GeneralCircuitGate, CircuitGateType, CircuitParams, GateF, GateIndex,
    },
    codegen::generator::FuncGenerator,
    common::{combine_integer, ilog2_ceil, split_number},
    mle::{custom::CustomMultilinearExtensionHandler, sparse::SisuSparseMultilinearExtension},
};

use super::{circuit::GeneralCircuit, gate::LayerIndex};

#[derive(Clone)]
pub struct LayerExtensions<F: Field> {
    extensions: Vec<SisuSparseMultilinearExtension<F>>,
    subset_num_vars: Vec<usize>,
}

pub fn get_extension<'a, F: Field>(
    circuit: &'a GeneralCircuit<F>,
    source_layer_index: usize,
    target_layer_index: usize,
) -> HashMap<&'a str, Vec<(usize, usize, usize, F)>> {
    let layer = circuit.layer(source_layer_index);
    let in2_num_vars = layer.mul_ext.subset_num_vars_at(&LayerIndex::Relative(
        target_layer_index - source_layer_index,
    ));
    let in1_num_vars = circuit.num_vars_at(source_layer_index + 1);
    println!(
        "in1_num_vars = {}, in2_num_vars = {}",
        in1_num_vars, in2_num_vars
    );
    let mut ret: HashMap<&str, Vec<(usize, usize, usize, F)>> = HashMap::default();

    let exts = HashMap::from([
        ("constant_ext", &layer.constant_ext),
        ("mul_ext", &layer.mul_ext),
        ("forward_x_ext", &layer.forward_x_ext),
        ("forward_y_ext", &layer.forward_y_ext),
    ]);

    for (ext_name, ext) in exts.iter() {
        let mut v: Vec<(usize, usize, usize, F)> = vec![];
        for (point, evaluation) in ext[target_layer_index - source_layer_index - 1]
            .evaluations
            .compile(&CircuitParams::default())
        {
            // point = out | in1 | in2
            let (out_in1, in2) = split_number(&point, in2_num_vars);
            let (out, in1) = split_number(&out_in1, in1_num_vars);
            // println!("cout = {}, {}-{}-{} == {:?}", count, out, in1, in2, evaluation);
            v.push((out, in1, in2, evaluation));
        }
        ret.insert(ext_name, v.clone());
    }

    return ret;
}

impl<F: Field> LayerExtensions<F> {
    pub fn default() -> Self {
        Self {
            extensions: vec![],
            subset_num_vars: vec![],
        }
    }

    pub fn new(current_layer_index: usize, total_layers: usize) -> Self {
        Self {
            extensions: vec![
                SisuSparseMultilinearExtension::default();
                total_layers - current_layer_index
            ],
            subset_num_vars: vec![0; total_layers - current_layer_index],
        }
    }

    pub fn custom<C: CustomMultilinearExtensionHandler<F>>(
        &mut self,
        target_layer_index: &LayerIndex,
        custom_mle: C,
    ) {
        self.extensions[target_layer_index.value_as_vector_index()].custom(custom_mle);
    }

    pub fn setup_mle(
        &mut self,
        target_layer_index: &LayerIndex,
        out_num_vars: usize,
        left_num_vars: usize,
        right_num_vars: usize,
    ) {
        self.extensions[target_layer_index.value_as_vector_index()]
            .set_num_vars(out_num_vars + left_num_vars + right_num_vars);
        self.subset_num_vars[target_layer_index.value_as_vector_index()] = right_num_vars;
    }

    pub fn add_evaluation(
        &mut self,
        target_layer_index: &LayerIndex,
        evaluation: (usize, GateF<F>),
    ) {
        self.extensions[target_layer_index.value_as_vector_index()]
            .evaluations
            .push(evaluation);
    }

    pub fn subset_num_vars_at(&self, target_layer_index: &LayerIndex) -> usize {
        self.subset_num_vars[target_layer_index.value_as_vector_index()].clone()
    }

    pub fn as_slice(&self) -> &[SisuSparseMultilinearExtension<F>] {
        &self.extensions
    }

    pub fn is_non_zero(&self) -> bool {
        for i in 0..self.extensions.len() {
            if self.extensions[i].evaluations.len() > 0 {
                return true;
            }
        }

        false
    }

    pub fn gen_code(
        &self,
        gkr_index: usize,
        layer_index: usize,
        ext_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        // Currently, we have not supported mle of single evaluation. The reason
        // is caused by a bug of circom https://github.com/iden3/circom/issues/245.
        //
        // It will be wrong if there is a one-element array in return statement.
        assert!(self.is_valid(), "layer {}", layer_index);

        let mut funcs = vec![];

        for i in 0..self.extensions.len() {
            let f = self.extensions[i].gen_code_for_general_gkr(
                &CircuitParams::default(),
                gkr_index,
                layer_index,
                layer_index + i + 1,
                ext_index,
            );

            funcs.extend(f);
        }

        funcs
    }

    fn is_valid(&self) -> bool {
        for i in 0..self.extensions.len() {
            if self.extensions[i].num_evaluations() == 1 {
                return false;
            }
        }

        return true;
    }
}

impl<'a, F: Field> Index<usize> for LayerExtensions<F> {
    type Output = SisuSparseMultilinearExtension<F>;

    fn index(&self, index: usize) -> &Self::Output {
        self.extensions.get(index).unwrap()
    }
}

#[derive(Clone)]
pub struct SubsetReverseExtensions<F: Field> {
    extensions: Vec<SisuSparseMultilinearExtension<F>>,
}

impl<F: Field> SubsetReverseExtensions<F> {
    pub const fn default() -> Self {
        Self { extensions: vec![] }
    }

    pub fn new(current_layer_index: usize) -> Self {
        Self {
            extensions: vec![SisuSparseMultilinearExtension::default(); current_layer_index],
        }
    }

    pub fn custom<C: CustomMultilinearExtensionHandler<F>>(
        &mut self,
        source_layer_index: usize,
        custom_mle: C,
    ) {
        self.extensions[source_layer_index].custom(custom_mle);
    }

    pub fn add_evaluation(
        &mut self,
        source_layer_index: usize,
        subset_num_vars: usize,
        num_vars: usize,
        evaluation: (usize, usize, GateF<F>),
    ) {
        self.extensions[source_layer_index].set_num_vars(subset_num_vars + num_vars);

        let combined_point = combine_integer(vec![
            (evaluation.0, subset_num_vars),
            (evaluation.1, num_vars),
        ]);

        for (p, v) in self.extensions[source_layer_index].evaluations.raw() {
            if p == &combined_point {
                assert!(v == &evaluation.2);
                return;
            }
        }

        self.extensions[source_layer_index]
            .evaluations
            .push((combined_point, evaluation.2));
    }

    pub fn as_slice(&self) -> &[SisuSparseMultilinearExtension<F>] {
        &self.extensions
    }

    pub fn gen_code(&self, gkr_index: usize, layer_index: usize) -> Vec<FuncGenerator<F>> {
        // Currently, we have not supported mle of single evaluation. The reason
        // is caused by a bug of circom https://github.com/iden3/circom/issues/245.
        //
        // It will be wrong if there is a one-element array in return statement.
        assert!(self.is_valid(), "layer {}", layer_index);

        let mut func = vec![];

        for src_layer_index in 0..self.extensions.len() {
            let f = self.extensions[src_layer_index].gen_code_for_general_gkr_reverse_subset(
                &CircuitParams::default(),
                gkr_index,
                src_layer_index,
                layer_index,
            );

            func.extend(f);
        }

        func
    }

    fn is_valid(&self) -> bool {
        for i in 0..self.extensions.len() {
            if self.extensions[i].num_evaluations() == 1 {
                return false;
            }
        }

        return true;
    }
}

#[derive(Clone)]
pub struct GeneralCircuitLayer<F: Field> {
    pub gates: Vec<GeneralCircuitGate<F>>,
    pub is_finalized: bool,

    pub constant_ext: LayerExtensions<F>,
    pub mul_ext: LayerExtensions<F>,
    pub forward_x_ext: LayerExtensions<F>,
    pub forward_y_ext: LayerExtensions<F>,

    pub subset_reverse_ext: SubsetReverseExtensions<F>,
    pub subset_info: SubsetInfo,
}

impl<F: Field> GeneralCircuitLayer<F> {
    pub fn default() -> Self {
        Self::new(vec![])
    }

    pub fn new(gates: Vec<GeneralCircuitGate<F>>) -> Self {
        Self {
            gates,
            is_finalized: false,
            constant_ext: LayerExtensions::default(),
            mul_ext: LayerExtensions::default(),
            forward_x_ext: LayerExtensions::default(),
            forward_y_ext: LayerExtensions::default(),
            subset_reverse_ext: SubsetReverseExtensions::default(),
            subset_info: SubsetInfo::default(),
        }
    }

    pub fn finalize(
        &mut self,
        total_layers: usize,
        layer_index: usize,
        left_input_num_vars: usize,
        preset_subset_mapping: HashMap<LayerIndex, HashMap<usize, usize>>,
    ) {
        assert!(!self.is_finalized, "Do not finalize twice");

        self.constant_ext = LayerExtensions::new(layer_index, total_layers);
        self.mul_ext = LayerExtensions::new(layer_index, total_layers);
        self.forward_x_ext = LayerExtensions::new(layer_index, total_layers);
        self.forward_y_ext = LayerExtensions::new(layer_index, total_layers);
        self.subset_reverse_ext = SubsetReverseExtensions::new(layer_index);

        self.subset_info = SubsetInfo::new(layer_index, total_layers);
        for (preset_layer, preset_mapping) in preset_subset_mapping.into_iter() {
            self.subset_info.preset_subset(preset_layer, preset_mapping);
        }

        for gate in self.gates.iter_mut() {
            self.subset_info.add_and_adjust_gate(gate);
        }

        let out_num_vars = self.num_vars();
        for i in 0..total_layers - layer_index {
            let target_layer_index = LayerIndex::Relative(i + 1);
            let right_input_num_vars = self.subset_info.num_vars_at(&target_layer_index);
            self.setup_mle_num_vars(
                &target_layer_index,
                out_num_vars,
                left_input_num_vars,
                right_input_num_vars,
            );
        }

        for (out_index, gate) in self.gates.clone().iter().enumerate() {
            self.handle_single_gate_mle(layer_index, left_input_num_vars, out_index, gate);
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

    pub fn add_gate(&mut self, gate: GeneralCircuitGate<F>) -> usize {
        assert!(
            !self.is_finalized,
            "The layer cannot add gate after finalized"
        );

        self.gates.push(gate);
        self.gates.len() - 1
    }

    pub fn add_gates(&mut self, gates: Vec<GeneralCircuitGate<F>>) {
        for gate in gates {
            self.gates.push(gate);
        }
    }

    fn setup_mle_num_vars(
        &mut self,
        target_layer_index: &LayerIndex,
        out_num_vars: usize,
        left_input_num_vars: usize,
        right_input_num_vars: usize,
    ) {
        self.constant_ext.setup_mle(
            &target_layer_index,
            out_num_vars,
            left_input_num_vars,
            right_input_num_vars,
        );
        self.mul_ext.setup_mle(
            &target_layer_index,
            out_num_vars,
            left_input_num_vars,
            right_input_num_vars,
        );
        self.forward_x_ext.setup_mle(
            &target_layer_index,
            out_num_vars,
            left_input_num_vars,
            right_input_num_vars,
        );
        self.forward_y_ext.setup_mle(
            &target_layer_index,
            out_num_vars,
            left_input_num_vars,
            right_input_num_vars,
        );
    }

    fn num_vars(&self) -> usize {
        ilog2_ceil(self.gates.len())
    }

    fn handle_single_gate_mle(
        &mut self,
        current_layer_index: usize,
        left_input_num_vars: usize,
        out_index: usize,
        gate: &GeneralCircuitGate<F>,
    ) {
        let output_num_vars = self.num_vars();
        let target_layer_index = &gate.right.0;
        let right_input_num_vars = self.subset_info.num_vars_at(&gate.right.0);

        let point = combine_integer(vec![
            (out_index, output_num_vars),
            (gate.left.value(), left_input_num_vars),
            (gate.right_subset_index, right_input_num_vars),
        ]);

        match &gate.gate_type {
            CircuitGateType::Constant(c) => {
                self.constant_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));
            }
            CircuitGateType::Add => {
                self.forward_x_ext
                    .add_evaluation(target_layer_index, (point, GateF::ONE));
                self.forward_y_ext
                    .add_evaluation(target_layer_index, (point, GateF::ONE));
            }
            CircuitGateType::Sub => {
                self.forward_x_ext
                    .add_evaluation(target_layer_index, (point, GateF::ONE));
                self.forward_y_ext
                    .add_evaluation(target_layer_index, (point, GateF::C(F::ZERO - F::ONE)));
            }
            CircuitGateType::CAdd(c) => {
                self.constant_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));
                self.forward_x_ext
                    .add_evaluation(target_layer_index, (point, GateF::ONE));
            }
            CircuitGateType::CSub(c) => {
                self.constant_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));
                self.forward_x_ext
                    .add_evaluation(target_layer_index, (point, GateF::C(F::ZERO - F::ONE)));
            }
            CircuitGateType::Mul(c) => {
                self.mul_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));
            }
            CircuitGateType::ForwardX(c) => {
                self.forward_x_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));
            }
            CircuitGateType::ForwardY(c) => {
                self.forward_y_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));
            }
            CircuitGateType::Xor(c) => {
                // GATE XOR: xor * (1 - ((1-in1) * (1-in2) + in1*in2))
                // = xor * (1 - (1 - in1 - in2 + in1 * in2 + in1 * in2))
                // = add*(in1+in2) - 2*in1*in2

                // add * (in1 + in2)
                self.forward_x_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));
                self.forward_y_ext
                    .add_evaluation(target_layer_index, (point, c.clone()));

                // -2 * mul * in1 * in2
                self.mul_ext.add_evaluation(
                    target_layer_index,
                    (point, c.clone() * GateF::C(F::ZERO - F::from(2u64))),
                );
            }
            CircuitGateType::NAAB => {
                // GATE XOR: naab * (1-in1) * in2
                // = naab * (in2 - in1*in2)

                // forward * in2
                self.forward_y_ext
                    .add_evaluation(target_layer_index, (point, GateF::ONE));

                // - mul * in1 * in2
                self.mul_ext
                    .add_evaluation(target_layer_index, (point, GateF::C(F::ZERO - F::ONE)));
            }
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    self.handle_single_gate_mle(
                        current_layer_index,
                        left_input_num_vars,
                        out_index,
                        subgate,
                    );
                }
            }
            CircuitGateType::Zero => {}
            CircuitGateType::Fourier(_) => panic!("not support fourier gate in general circuit"),
        }
    }

    pub fn gen_code(&self, gkr_index: usize, layer_index: usize) -> Vec<FuncGenerator<F>> {
        let mut funcs = vec![];

        let f = self.subset_reverse_ext.gen_code(gkr_index, layer_index);
        funcs.extend(f);

        let f = self.constant_ext.gen_code(gkr_index, layer_index, 0);
        funcs.extend(f);

        let f = self.mul_ext.gen_code(gkr_index, layer_index, 1);
        funcs.extend(f);

        let f = self.forward_x_ext.gen_code(gkr_index, layer_index, 2);
        funcs.extend(f);

        let f = self.forward_y_ext.gen_code(gkr_index, layer_index, 3);
        funcs.extend(f);

        funcs
    }
}

#[derive(Clone, Debug, Default)]
pub struct SubsetInfo {
    subset_size: Vec<usize>,
    subset_index_mapping: Vec<HashMap<usize, usize>>,
}

impl SubsetInfo {
    fn new(current_layer_index: usize, total_layers: usize) -> Self {
        Self {
            subset_size: vec![0; total_layers - current_layer_index],
            subset_index_mapping: vec![HashMap::new(); total_layers - current_layer_index],
        }
    }

    fn preset_subset(&mut self, layer_index: LayerIndex, map: HashMap<usize, usize>) {
        self.subset_index_mapping[layer_index.value_as_vector_index()] = map;
    }

    fn add_and_adjust_gate<F: Field>(&mut self, gate: &mut GeneralCircuitGate<F>) {
        match &mut gate.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    self.add_and_adjust_gate(subgate);
                }
            }
            _ => {
                let target_layer_index = gate.right.0.value_as_vector_index();
                let target_gate_index = gate.right.1.value();

                let target_layer_size = self.subset_size.get_mut(target_layer_index).unwrap();
                let index_mapping = self
                    .subset_index_mapping
                    .get_mut(target_layer_index)
                    .unwrap();

                // Set the suitable subset index.
                gate.right_subset_index = if index_mapping.contains_key(&target_gate_index) {
                    *index_mapping.get(&target_gate_index).unwrap()
                } else {
                    let subset_index = target_layer_size.clone();
                    *target_layer_size += 1;
                    index_mapping.insert(target_gate_index, subset_index);

                    subset_index
                };
            }
        }
    }

    fn num_vars_at(&self, layer_index: &LayerIndex) -> usize {
        ilog2_ceil(
            self.subset_size
                .get(layer_index.value_as_vector_index())
                .unwrap()
                .clone(),
        )
    }
}
