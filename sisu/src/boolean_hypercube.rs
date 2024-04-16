use std::marker::PhantomData;

use ark_ff::Field;
use bitvec::slice::BitSlice;

pub struct BooleanHypercube<F: Field> {
    num_vars: usize,
    current: usize,
    __phantom: PhantomData<F>,
}

impl<F: Field> BooleanHypercube<F> {
    pub fn new(n: usize) -> Self {
        Self {
            num_vars: n,
            current: 0,
            __phantom: PhantomData,
        }
    }
}

impl<F: Field> Iterator for BooleanHypercube<F> {
    type Item = Vec<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_vars == 0 || self.current == 2usize.pow(self.num_vars as u32) {
            return None;
        } else {
            let le_bytes = self.current.to_le_bytes();
            let s: &BitSlice<u8> = BitSlice::<u8>::from_slice(&le_bytes);
            self.current += 1;

            Some(
                s.iter()
                    .take(self.num_vars as usize)
                    .map(|v| match *v {
                        false => F::zero(),
                        true => F::one(),
                    })
                    .rev()
                    .collect(),
            )
        }
    }
}
