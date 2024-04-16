use std::collections::HashMap;
use std::marker::PhantomData;

use crate::layer::Layer;
use ark_ff::Field;
use sisulib::circuit::GateIndex;

use crate::constants::INIT_H_VALUES;
use crate::sha_calculation::Sha256Cal;
use crate::test_util::assign_values;

pub struct MessageInfo<F: Field> {
    pub input: Vec<F>,
    pub output: Vec<u8>,
}

pub struct LayerInput<F: Field> {
    index: usize,
    pub indexes: HashMap<String, Vec<Vec<GateIndex>>>,

    __phantom: PhantomData<F>,
}

impl<F: Field> LayerInput<F> {
    pub fn new() -> Self {
        let mut input_label = HashMap::<usize, String>::default();
        let mut indexes = HashMap::<String, Vec<Vec<GateIndex>>>::default();

        let mut index = 0usize;
        // w input
        let mut w_i_bits_indexes: Vec<Vec<GateIndex>> = Vec::with_capacity(64);
        for i in 0..16 {
            input_label.insert(index, format!("w_i_bits[{:?}]", i));
            let mut tmp: Vec<GateIndex> = Vec::with_capacity(32);
            for j in 0..32 {
                tmp.push(GateIndex::Absolute(index + j));
            }
            w_i_bits_indexes.push(tmp);
            index += 32;
        }

        // h0 -> h7
        let mut h_i_bits_indexes: Vec<Vec<GateIndex>> = Vec::with_capacity(8);
        for i in 0..8 {
            input_label.insert(index, format!("h_i_bits[{:?}]", i));
            let mut tmp: Vec<GateIndex> = Vec::with_capacity(32);
            for j in 0..32 {
                tmp.push(GateIndex::Absolute(index + j));
            }
            h_i_bits_indexes.push(tmp);
            index += 32;
        }

        // hout0 -> hout7
        let mut hout_i_bits_indexes: Vec<Vec<GateIndex>> = Vec::with_capacity(8);
        for i in 0..8 {
            input_label.insert(index, format!("hout_i_bits[{:?}]", i));
            let mut tmp: Vec<GateIndex> = Vec::with_capacity(33);
            for j in 0..32 {
                tmp.push(GateIndex::Absolute(index + j));
            }
            hout_i_bits_indexes.push(tmp);
            index += 32;
        }

        assert_eq!(1024, index);

        // w extension
        for i in 16..64 {
            input_label.insert(index, format!("w_i_bits[{:?}]", i));
            let mut tmp: Vec<GateIndex> = Vec::with_capacity(34);
            for j in 0..32 {
                tmp.push(GateIndex::Absolute(index + j));
            }
            w_i_bits_indexes.push(tmp);
            index += 32;
        }

        // a_i_bits
        let mut a_i_bits_indexes: Vec<Vec<GateIndex>> = Vec::with_capacity(64);
        for i in 0..64 {
            input_label.insert(index, format!("a_i_bits[{:?}]", i));
            let mut tmp: Vec<GateIndex> = Vec::with_capacity(35);
            for j in 0..35 {
                tmp.push(GateIndex::Absolute(index + j));
            }
            a_i_bits_indexes.push(tmp);
            index += 35;
        }

        // e_i_bits
        let mut e_i_bits_indexes: Vec<Vec<GateIndex>> = Vec::with_capacity(64);
        for i in 0..64 {
            input_label.insert(index, format!("e_i_bits[{:?}]", i));
            let mut tmp: Vec<GateIndex> = Vec::with_capacity(35);
            for j in 0..35 {
                tmp.push(GateIndex::Absolute(index + j));
            }
            e_i_bits_indexes.push(tmp);
            index += 35;
        }

        // 2 bits for w_i_bits each
        for i in 0..64 {
            input_label.insert(index, format!("w_i_bits_extra[{:?}]", i));
            for j in 0..2 {
                w_i_bits_indexes[i].push(GateIndex::Absolute(index + j));
            }
            index += 2;
        }

        // 1 bits for hout_i_bits each
        for i in 0..8 {
            input_label.insert(index, format!("hout_i_bits[{:?}]", i));
            hout_i_bits_indexes[i].push(GateIndex::Absolute(index));
            index += 1;
        }

        // input0
        let input0s = vec![vec![GateIndex::Absolute(index)]];
        index += 1;

        // Update all the indexes.
        indexes.insert("a_i_bits".to_string(), a_i_bits_indexes);
        indexes.insert("e_i_bits".to_string(), e_i_bits_indexes);
        indexes.insert("w_i_bits".to_string(), w_i_bits_indexes);
        indexes.insert("h_i_bits".to_string(), h_i_bits_indexes);
        indexes.insert("hout_i_bits".to_string(), hout_i_bits_indexes);
        indexes.insert("input0".to_string(), input0s);
        assert_eq!(Layer::<F>::input_tags().len(), indexes.len());

        Self {
            __phantom: Default::default(),
            index,
            indexes,
        }
    }

    pub fn input_len(&self) -> usize {
        self.index
    }

    pub fn build_message_info(&self, msg: &[u8]) -> Vec<MessageInfo<F>> {
        let mut result = vec![];
        let preprocessed_message = Sha256Cal::preprocess_message(msg);

        // Evaluation
        // 1a. Run SHA calculation.
        let mut init_h = INIT_H_VALUES.to_vec();
        for i in 0..preprocessed_message.len() / 64 {
            let (output, input_map) =
                Sha256Cal::calculate(&init_h, &preprocessed_message[i * 64..(i + 1) * 64]);
            let mut input: Vec<F> = vec![F::ZERO; self.index];

            // Assign values from input layer and the SHA256 calculation results.
            let assign_fn = |index: GateIndex, value: i64| {
                input[index.value()] = F::from(value as u128);
            };
            assign_values(&self.indexes, &input_map, assign_fn);

            let mut next_init_h = vec![];
            for i in 0..8 {
                let mut hi = 0;
                for j in 0..4 {
                    hi = (hi << 8) | output[i * 4 + j] as u32;
                }
                next_init_h.push(hi);
            }

            init_h = next_init_h;
            result.push(MessageInfo { input, output });
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use sisulib::{circuit::GateIndex, field::Fp389};

    use super::LayerInput;

    #[test]
    fn new() {
        let layer_input = LayerInput::<Fp389>::new();

        fn check(x: &Vec<Vec<GateIndex>>, dim1: usize, dim2: usize) {
            assert_eq!(dim1, x.len());
            for i in 0..dim1 {
                assert_eq!(dim2, x[i].len());
            }
        }

        check(layer_input.indexes.get("a_i_bits").unwrap(), 64, 35);
        check(layer_input.indexes.get("e_i_bits").unwrap(), 64, 35);
        check(layer_input.indexes.get("w_i_bits").unwrap(), 64, 34);
        check(layer_input.indexes.get("h_i_bits").unwrap(), 8, 32);
        check(layer_input.indexes.get("hout_i_bits").unwrap(), 8, 33);
    }
}
