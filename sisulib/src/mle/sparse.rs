use std::collections::HashMap;

use ark_ff::Field;
use ark_std::rand::Rng;

use crate::{
    circuit::{CircuitParams, GateEvaluations},
    codegen::generator::{CustomMLEGenerator, FuncGenerator},
    common::{combine_point, dec2bin},
};

use super::custom::{CustomMultilinearExtension, CustomMultilinearExtensionHandler};

/// utility: precompute f(x) = eq(g,x)
fn precompute_eq<F: Field>(g: &[F]) -> Vec<F> {
    let dim = g.len();

    let mut dp = vec![F::zero(); 1 << dim];
    dp[0] = F::one() - g[0];
    dp[1] = g[0];
    for i in 1..dim {
        for b in 0..(1 << i) {
            let prev = dp[b];
            dp[b + (1 << i)] = prev * g[i];
            dp[b] = prev - dp[b + (1 << i)];
        }
    }

    dp
}

#[derive(Clone)]
pub struct SisuSparseMultilinearExtension<F: Field> {
    pub evaluations: GateEvaluations<F>,
    num_vars: usize,
    custom: Option<CustomMultilinearExtension<F>>,
}

impl<F: Field> SisuSparseMultilinearExtension<F> {
    pub fn default() -> Self {
        Self {
            evaluations: GateEvaluations::default(),
            custom: None,
            num_vars: 0,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_evaluations(&self) -> usize {
        self.evaluations.len()
    }

    pub fn set_num_vars(&mut self, num_vars: usize) {
        self.num_vars = num_vars;
    }

    pub fn custom<C: CustomMultilinearExtensionHandler<F>>(&mut self, custom_mle: C) {
        self.custom = Some(CustomMultilinearExtension::new(custom_mle));
    }

    pub fn from_evaluations(num_vars: usize, evaluations: GateEvaluations<F>) -> Self {
        Self {
            evaluations,
            num_vars,
            custom: None,
        }
    }

    pub fn evaluate(&self, point: Vec<&[F]>, params: &CircuitParams<F>) -> F {
        if self.evaluations.len() == 0 {
            return F::ZERO;
        }

        if let Some(custom_ext) = &self.custom {
            custom_ext.evaluate(point, params)
        } else {
            self.raw_evaluate(point, params)
        }
    }

    // This is old evaluation function without the part that calculates values used for circom.
    fn raw_evaluate(&self, point: Vec<&[F]>, params: &CircuitParams<F>) -> F {
        if self.evaluations.len() == 0 {
            return F::ZERO;
        }

        let mut combined_point = combine_point(point);
        combined_point.reverse();

        let dim = combined_point.len();
        assert!(dim == self.num_vars, "invalid partial point dimension");

        let window = ark_std::log2(self.evaluations.len()) as usize;
        let mut points = combined_point.as_slice();
        let mut last = HashMap::from_iter(self.evaluations.compile(params));

        // batch evaluation
        while !points.is_empty() {
            let focus_length = if window > 0 && points.len() > window {
                window
            } else {
                points.len()
            };

            let focus = &points[..focus_length];
            points = &points[focus_length..];
            let pre = precompute_eq(focus);
            let dim = focus.len();
            let mut result = HashMap::new();

            for src_entry in last.iter() {
                let old_idx = *src_entry.0;
                let gz = pre[old_idx & ((1 << dim) - 1)];
                let new_idx = old_idx >> dim;
                let dst_entry = result.entry(new_idx).or_insert(F::ZERO);
                *dst_entry += gz * src_entry.1;
            }

            last = result;
        }

        assert_eq!(last.len(), 1);
        last[&0]
    }

    pub fn evaluations(&self) -> &GateEvaluations<F> {
        &self.evaluations
    }

    pub fn gen_code_for_gkr(
        &self,
        params: &CircuitParams<F>,
        gkr_index: usize,
        layer_index: usize,
        ext_index: usize,
    ) -> (Vec<FuncGenerator<F>>, Vec<CustomMLEGenerator>) {
        let mut funcs = vec![];
        let mut mle = vec![];
        let mut point_size_func = FuncGenerator::new(
            "get_custom_mle__point_size",
            vec!["gkr_index", "layer_index", "ext_index"],
        );

        if self.custom.is_none() {
            point_size_func.add_number(vec![gkr_index, layer_index, ext_index], 0);
            funcs.extend(self.gen_constants_for_gkr(params, gkr_index, layer_index, ext_index));
        } else {
            point_size_func.add_number(vec![gkr_index, layer_index, ext_index], self.num_vars);

            let custom_mle = self.custom.as_ref().unwrap();
            let mut mle_template = CustomMLEGenerator::new();
            mle_template.add_mle(
                (gkr_index, layer_index, ext_index),
                &custom_mle.circom_template_name,
            );
            mle.push(mle_template);
        }
        funcs.push(point_size_func);

        (funcs, mle)
    }

    pub fn gen_code_for_general_gkr(
        &self,
        params: &CircuitParams<F>,
        gkr_index: usize,
        src_layer_index: usize,
        dst_layer_index: usize,
        ext_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        self.gen_constants_for_general_gkr(
            params,
            gkr_index,
            src_layer_index,
            dst_layer_index,
            ext_index,
        )
    }

    pub fn gen_code_for_general_gkr_reverse_subset(
        &self,
        params: &CircuitParams<F>,
        gkr_index: usize,
        src_layer_index: usize,
        dst_layer_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        let funcs = self.gen_constants_for_general_gkr_reverse_subset(
            params,
            gkr_index,
            src_layer_index,
            dst_layer_index,
        );

        funcs
    }

    fn gen_constants_for_gkr(
        &self,
        params: &CircuitParams<F>,
        gkr_index: usize,
        layer_index: usize,
        ext_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        self.gen_constants(
            params,
            "gkr",
            vec!["gkr_index", "layer_index", "ext_index"],
            vec![gkr_index, layer_index, ext_index],
        )
    }

    fn gen_constants_for_general_gkr(
        &self,
        params: &CircuitParams<F>,
        gkr_index: usize,
        src_layer_index: usize,
        dst_layer_index: usize,
        ext_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        self.gen_constants(
            params,
            "general_gkr",
            vec![
                "gkr_index",
                "src_layer_index",
                "dst_layer_index",
                "ext_index",
            ],
            vec![gkr_index, src_layer_index, dst_layer_index, ext_index],
        )
    }

    fn gen_constants_for_general_gkr_reverse_subset(
        &self,
        params: &CircuitParams<F>,
        gkr_index: usize,
        src_layer_index: usize,
        dst_layer_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        self.gen_constants(
            params,
            "general_gkr__reverse_subset",
            vec!["gkr_index", "src_layer_index", "dst_layer_index"],
            vec![gkr_index, src_layer_index, dst_layer_index],
        )
    }

    fn gen_constants(
        &self,
        circuit_params: &CircuitParams<F>,
        tag: &str,
        param_names: Vec<&str>,
        param_values: Vec<usize>,
    ) -> Vec<FuncGenerator<F>> {
        let window = ark_std::log2(self.evaluations.len()) as usize;
        let mut last: Vec<usize> = self
            .evaluations
            .compile(circuit_params)
            .into_iter()
            .map(|x| x.0)
            .collect();

        let mut prev_output_indexes = Vec::with_capacity(last.len());
        let mut last_positions = Vec::with_capacity(last.len());
        let mut flattened_lens = Vec::with_capacity(last.len());
        let mut old_idxes = Vec::with_capacity(last.len());

        let mut point_len = self.num_vars;
        // batch evaluation
        if self.evaluations.len() > 0 {
            while point_len > 0 {
                let focus_length = if window > 0 && point_len > window {
                    window
                } else {
                    point_len
                };

                point_len -= focus_length;
                let dim = focus_length;
                let mut new_last: Vec<usize> = vec![];
                // Position of the FIRST appearance of an index in the last evaluation array.
                let mut first_position_map: HashMap<usize, usize> = HashMap::new();
                // Position of the LAST appearance of an index in the last evaluation array. This array
                // is used to create values used for circom.
                let mut last_position_map: HashMap<usize, usize> = HashMap::new();

                for i in 0..last.len() {
                    let old_idx = last[i];
                    let new_idx = old_idx >> dim;

                    old_idxes.push(old_idx);

                    if first_position_map.contains_key(&new_idx) {
                        let index = first_position_map.get(&new_idx).unwrap();
                        assert_eq!(new_last[*index], new_idx);

                        let last_pos = *last_position_map.get(&new_idx).unwrap();
                        last_positions.push(last_pos);
                    } else {
                        last_positions.push(usize::MAX);

                        // This is new index value. Insert into the new_last array.
                        new_last.push(new_idx);
                        first_position_map.insert(new_idx, new_last.len() - 1);
                    }

                    last_position_map.insert(new_idx, last_positions.len() - 1);
                }

                flattened_lens.push(last.len());
                for i in 0..new_last.len() {
                    let idx = new_last[i];
                    prev_output_indexes.push(*last_position_map.get(&idx).unwrap());
                }

                last = new_last;
            }
        }

        let mut funcs = vec![];

        let mut evaluations_size_func = FuncGenerator::new(
            &format!("get_{}__ext__evaluations_size", tag),
            param_names.to_vec(),
        );
        evaluations_size_func.add_number(param_values.to_vec(), self.evaluations.len());
        funcs.push(evaluations_size_func);

        let mut evaluations_func = FuncGenerator::new(
            &format!("get_{}__ext__evaluations", tag),
            param_names.to_vec(),
        );
        evaluations_func.add_field_array(
            param_values.to_vec(),
            self.evaluations
                .compile(circuit_params)
                .into_iter()
                .map(|(_, f)| f)
                .collect(),
        );
        funcs.push(evaluations_func);

        let mut prev_output_indexes_len_func = FuncGenerator::new(
            &format!("get_{}__ext__prev_output_indexes_size", tag),
            param_names.to_vec(),
        );
        prev_output_indexes_len_func.add_number(param_values.to_vec(), prev_output_indexes.len());
        funcs.push(prev_output_indexes_len_func);

        let mut prev_output_indexes_func = FuncGenerator::new(
            &format!("get_{}__ext__prev_output_indexes", tag),
            param_names.to_vec(),
        );
        prev_output_indexes_func.add_number_array(param_values.to_vec(), prev_output_indexes);
        funcs.push(prev_output_indexes_func);

        let mut last_positions_size_func = FuncGenerator::new(
            &format!("get_{}__ext__last_positions_size", tag),
            param_names.to_vec(),
        );
        last_positions_size_func.add_number(param_values.to_vec(), last_positions.len());
        funcs.push(last_positions_size_func);

        let mut last_positions_func = FuncGenerator::new(
            &format!("get_{}__ext__last_positions", tag),
            param_names.to_vec(),
        );
        last_positions_func.add_number_array(param_values.to_vec(), last_positions);
        funcs.push(last_positions_func);

        let mut flattern_size_size_func = FuncGenerator::new(
            &format!("get_{}__ext__flattened_size_size", tag),
            param_names.to_vec(),
        );
        flattern_size_size_func.add_number(param_values.to_vec(), flattened_lens.len());
        funcs.push(flattern_size_size_func);

        let mut flattern_size_func = FuncGenerator::new(
            &format!("get_{}__ext__flattened_size", tag),
            param_names.to_vec(),
        );
        flattern_size_func.add_number_array(param_values.to_vec(), flattened_lens);
        funcs.push(flattern_size_func);

        let mut old_indexes_size_func = FuncGenerator::new(
            &format!("get_{}__ext__old_indexes_size", tag),
            param_names.to_vec(),
        );
        old_indexes_size_func.add_number(param_values.to_vec(), old_idxes.len());
        funcs.push(old_indexes_size_func);

        let mut old_indexes_func = FuncGenerator::new(
            &format!("get_{}__ext__old_indexes", tag),
            param_names.to_vec(),
        );
        old_indexes_func.add_number_array(param_values.to_vec(), old_idxes);
        funcs.push(old_indexes_func);

        funcs
    }

    pub fn test_over_boolean_hypercube(&self, circuit_params: &CircuitParams<F>) -> Option<bool> {
        if self.custom.is_none() {
            return None;
        }

        let custom_ext = self.custom.as_ref().unwrap();

        let out_num_vars = custom_ext.out_num_vars;
        let in1_num_vars = custom_ext.in1_num_vars;
        let in2_num_vars = custom_ext.in2_num_vars;

        let mut result = true;
        for z in 0..2usize.pow(out_num_vars as u32) {
            for x in 0..2usize.pow(in1_num_vars as u32) {
                for y in 0..2usize.pow(in2_num_vars as u32) {
                    let zp = dec2bin(z as u64, out_num_vars);
                    let xp = dec2bin(x as u64, in1_num_vars);
                    let yp = dec2bin(y as u64, in2_num_vars);

                    let custom_mle = custom_ext.evaluate(vec![&zp, &xp, &yp], circuit_params);
                    let sparse_mle = self.raw_evaluate(vec![&zp, &xp, &yp], circuit_params);
                    if sparse_mle != custom_mle {
                        println!(
                            "Z={z} X={x} Y={y} ACTUAL={:?}  EXPECTED={:?}",
                            custom_mle, sparse_mle
                        );

                        result = false;
                    }
                }
            }
        }

        Some(result)
    }

    pub fn test_random<R: Rng>(
        &self,
        rng: &mut R,
        circuit_params: &CircuitParams<F>,
    ) -> Option<bool> {
        if self.custom.is_none() {
            return None;
        }

        let custom_ext = self.custom.as_ref().unwrap();

        let out_num_vars = custom_ext.out_num_vars;
        let in1_num_vars = custom_ext.in1_num_vars;
        let in2_num_vars = custom_ext.in2_num_vars;

        let mut z = vec![];
        for _ in 0..out_num_vars {
            z.push(F::rand(rng));
        }

        let mut x = vec![];
        for _ in 0..in1_num_vars {
            x.push(F::rand(rng));
        }

        let mut y = vec![];
        for _ in 0..in2_num_vars {
            y.push(F::rand(rng));
        }

        let mut result = true;
        let custom_mle = custom_ext.evaluate(vec![&z, &x, &y], circuit_params);
        let sparse_mle = self.raw_evaluate(vec![&z, &x, &y], circuit_params);
        if sparse_mle != custom_mle {
            println!(
                "Z={:?} X={:?} Y={:?} ACTUAL={:?}  EXPECTED={:?}",
                z, x, y, custom_mle, sparse_mle
            );

            result = false;
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {

    use crate::{circuit::GateF, codegen::generator::FileGenerator, field::FpSisu};

    use super::*;

    #[test]
    fn test_gen_mle_circom() {
        let num_vars = 20;
        let n = 2usize.pow(12);
        let mut evaluations = GateEvaluations::<FpSisu>::default();
        for i in 0..n {
            evaluations.push((i, GateF::ONE));
        }

        let mut input = vec![];
        for i in 0..num_vars {
            input.push(FpSisu::from(i as u64));
        }

        let sparse_mle = SisuSparseMultilinearExtension::from_evaluations(num_vars, evaluations);
        let (f, _) = sparse_mle.gen_code_for_gkr(&CircuitParams::default(), 0, 0, 0);

        let mut file_gen =
            FileGenerator::<FpSisu>::new("../bls-circom/circuit/sisu/configs.gen.circom");
        file_gen.extend_funcs(f);
        file_gen.create();
    }
}
