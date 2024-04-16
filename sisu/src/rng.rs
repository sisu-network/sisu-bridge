use ark_ff::Field;
use ark_std::rand::Rng;

pub trait RngF<F> {
    fn draw(&mut self) -> F;
}

impl<F: Field, T: Rng> RngF<F> for T {
    fn draw(&mut self) -> F {
        F::rand(self)
    }
}
