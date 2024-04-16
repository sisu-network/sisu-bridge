use std::rc::Rc;

use ark_ff::Field;

use crate::circuit::CircuitParams;

pub trait CustomMultilinearExtensionHandler<F: Field> {
    fn template(&self) -> String;
    fn handler(&self) -> Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F>;
    fn out_num_vars(&self) -> usize;
    fn in1_num_vars(&self) -> usize;
    fn in2_num_vars(&self) -> usize;
}

#[derive(Clone)]
pub struct CustomMultilinearExtension<F: Field> {
    handler: Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F>,
    pub circom_template_name: String,
    pub out_num_vars: usize,
    pub in1_num_vars: usize,
    pub in2_num_vars: usize,
}

unsafe impl<F: Field> Sync for CustomMultilinearExtension<F> {}
unsafe impl<F: Field> Send for CustomMultilinearExtension<F> {}

impl<F: Field> CustomMultilinearExtension<F> {
    pub fn new<C: CustomMultilinearExtensionHandler<F>>(mle: C) -> Self {
        Self {
            handler: mle.handler(),
            circom_template_name: mle.template(),
            out_num_vars: mle.out_num_vars(),
            in1_num_vars: mle.in1_num_vars(),
            in2_num_vars: mle.in2_num_vars(),
        }
    }

    pub fn evaluate(&self, points: Vec<&[F]>, circuit_params: &CircuitParams<F>) -> F {
        (self.handler)(points, circuit_params)
    }
}
