use std::marker::PhantomData;

use ark_ff::Field;

use crate::circuit_layer::CircuitLayer;

pub struct OpAddInputNum<F: Field> {
    __f: PhantomData<F>,
}

impl<F: Field> OpAddInputNum<F> {
    pub fn build(label: &str, layer: &mut CircuitLayer<F>, size: usize, has_compress_flag: bool) {
        if layer.indexes.contains_key(label) {
            // This label has been added to the input.
            return;
        }

        if has_compress_flag {
            // The first 3 bits are used for compression info.
            layer.add_input_zero_one(&format!("{label}_flags"), 3);
        }

        // The input
        layer.add_input_zero_one(label, size);

        // Padding
        if has_compress_flag {
            layer.add_input_zero_one("dummy", 128);
        }
    }
}
