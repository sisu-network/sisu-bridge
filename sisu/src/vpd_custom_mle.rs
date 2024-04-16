use std::marker::PhantomData;
use std::rc::Rc;

use ark_ff::Field;
use sisulib::circuit::CircuitParams;
use sisulib::common::dec2bin;
use sisulib::common::ilog2_ceil;
use sisulib::mle::custom::CustomMultilinearExtensionHandler;

pub struct FFTComputeTMulExtension<F: Field> {
    stage: usize,
    l: usize,
    __phantom: PhantomData<F>,
}

impl<F: Field> FFTComputeTMulExtension<F> {
    pub fn new(stage: usize, l: usize) -> Self {
        Self {
            stage,
            l,
            __phantom: PhantomData,
        }
    }
}

impl<F: Field> CustomMultilinearExtensionHandler<F> for FFTComputeTMulExtension<F> {
    fn template(&self) -> String {
        format!("FFT_ComputeT_Mul({}, {})", self.stage, self.l)
    }

    fn out_num_vars(&self) -> usize {
        let scalars_num_vars = ilog2_ceil(self.l) + 1;
        let out_t_num_vars = self.stage + 1;

        let mut out_num_vars = scalars_num_vars + 1;
        if out_t_num_vars > scalars_num_vars {
            out_num_vars = out_t_num_vars + 1;
        }

        out_num_vars
    }

    fn in1_num_vars(&self) -> usize {
        let scalars_num_vars = ilog2_ceil(self.l) + 1;
        let in_t_num_vars = if self.stage == 0 { 1 } else { self.stage };

        let mut in_num_vars = scalars_num_vars + 1;
        if in_t_num_vars > scalars_num_vars {
            in_num_vars = in_t_num_vars + 1;
        }

        in_num_vars
    }

    fn in2_num_vars(&self) -> usize {
        let scalars_num_vars = ilog2_ceil(self.l) + 1;
        let in_t_num_vars = if self.stage == 0 { 1 } else { self.stage };

        let mut in_num_vars = scalars_num_vars + 1;
        if in_t_num_vars > scalars_num_vars {
            in_num_vars = in_t_num_vars + 1;
        }

        in_num_vars
    }

    fn handler(&self) -> std::rc::Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F> {
        let stage = self.stage.clone();
        let l = self.l.clone();
        let out_num_vars = self.out_num_vars();
        let in_num_vars = self.in1_num_vars();

        Rc::new(move |points, _| {
            assert!(points.len() == 3);
            let out = points[0];
            let in1 = points[1];
            let in2 = points[2];

            let out_t_num_vars = stage + 1;
            let in_t_num_vars = if stage == 0 { 1 } else { stage };

            assert_eq!(out.len(), out_num_vars);
            assert_eq!(in1.len(), in_num_vars);
            assert_eq!(in2.len(), in_num_vars);

            let mut result = F::ONE;

            // The size of scalar group is 2 * l and hence requires 1 + ceiling(log(l)) bits. These
            // bits in IN2 must equal to the binary representation of stage.
            let l_binary_len = ilog2_ceil(l);
            let scalar_len = l_binary_len + 1;

            // First out bit must be 0.
            // First bit of in1 must be 0.
            // First bit if on2 must be 1.
            result *= (F::ONE - out[0]) * (F::ONE - in1[0]) * in2[0];

            // The binary representation of OUT's T group has 2 parts:
            // 1a) k right most bits must match with the k right most bits of IN2.
            // 1b)  The  bits starting from bit (k + 1) from the right must match with bits in IN1

            // 1a) k right most bits of OUT's T must match with k right most bits of IN2.
            result *= out[out_num_vars - 1] * in2[in_num_vars - 1]
                + (F::ONE - out[out_num_vars - 1]) * (F::ONE - in2[in_num_vars - 1]);

            // 1b)
            if stage == 0 {
                result *= F::ONE - in1[in_num_vars - 1];
            } else {
                for i in 0..in_t_num_vars {
                    let in_index = in_num_vars - 1 - i;
                    let out_index = out_num_vars - 1 - i - 1;

                    // if from_binary(out) == 2 && from_binary(in1) == 0 && from_binary(in2) == 24 {}

                    result *= out[out_index] * in1[in_index]
                        + (F::ONE - out[out_index]) * (F::ONE - in1[in_index]);
                }
            }

            // in2 has k + ceiling(log(l - 1)) bits. We wire first k bits of in2 in 1a). Now we need to
            // wire ceiling(log(l - 1)) bits.
            let stage_binary = dec2bin(stage as u64, l_binary_len);
            for i in 0..l_binary_len {
                let in2_value = in2[in_num_vars - 1 - l_binary_len + i];
                let stage_value: F = stage_binary[i];

                result *= in2_value * stage_value + (F::ONE - in2_value) * (F::ONE - stage_value);
            }

            // Add the padding for IN1's T group or scalar group.
            if in_t_num_vars < scalar_len {
                // pad T group with dummy 0 bits in IN1
                for i in 0..scalar_len - in_t_num_vars {
                    result *= F::ONE - in1[in_num_vars - 1 - in_t_num_vars - i];
                }
            } else {
                // pad scalar group with dummy 0 bits for IN2
                for i in 0..in_t_num_vars - scalar_len {
                    result *= F::ONE - in2[in_num_vars - 1 - scalar_len - i];
                }
            }

            // Add padding for OUT's T group if needed.
            if out_t_num_vars < scalar_len {
                for i in 0..scalar_len - out_t_num_vars {
                    result *= F::ONE - out[out_num_vars - 1 - out_t_num_vars - i];
                }
            }

            result
        })
    }
}

pub struct FFTComputeTForwardYExtension<F: Field> {
    stage: usize,
    l: usize,
    __phantom: PhantomData<F>,
}

impl<F: Field> FFTComputeTForwardYExtension<F> {
    pub fn new(stage: usize, l: usize) -> Self {
        Self {
            stage,
            l,
            __phantom: PhantomData,
        }
    }
}

impl<F: Field> CustomMultilinearExtensionHandler<F> for FFTComputeTForwardYExtension<F> {
    fn template(&self) -> String {
        format!("FFT_ComputeT_ForwardY({}, {})", self.stage, self.l)
    }

    fn out_num_vars(&self) -> usize {
        let scalars_num_vars = ilog2_ceil(self.l) + 1;
        let out_t_num_vars = self.stage + 1;

        let mut out_num_vars = scalars_num_vars + 1;
        if out_t_num_vars > scalars_num_vars {
            out_num_vars = out_t_num_vars + 1;
        }

        out_num_vars
    }

    fn in1_num_vars(&self) -> usize {
        let scalars_num_vars = ilog2_ceil(self.l) + 1;
        let in_t_num_vars = if self.stage == 0 { 1 } else { self.stage };

        let mut in_num_vars = scalars_num_vars + 1;
        if in_t_num_vars > scalars_num_vars {
            in_num_vars = in_t_num_vars + 1;
        }

        in_num_vars
    }

    fn in2_num_vars(&self) -> usize {
        let scalars_num_vars = ilog2_ceil(self.l) + 1;
        let in_t_num_vars = if self.stage == 0 { 1 } else { self.stage };

        let mut in_num_vars = scalars_num_vars + 1;
        if in_t_num_vars > scalars_num_vars {
            in_num_vars = in_t_num_vars + 1;
        }

        in_num_vars
    }

    fn handler(&self) -> Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F> {
        let l = self.l.clone();
        let out_num_vars = self.out_num_vars();
        let in_num_vars = self.in1_num_vars();

        Rc::new(move |points: Vec<&[F]>, _| {
            assert!(points.len() == 3);
            let out = points[0];
            let in1 = points[1];
            let in2 = points[2];

            assert_eq!(out.len(), out_num_vars);
            assert_eq!(in1.len(), in_num_vars);
            assert_eq!(in2.len(), in_num_vars);

            let mut result = F::ONE;

            // First bit of out, in1 and in2 must be 1.
            result *= out[0] * in1[0] * in2[0];

            // The size of scalar group is (d + 1) * l and hence requires k + ceiling(log(l)) bits. These
            // bits in IN2 must equal to the binary representation of stage.
            let l_binary = dec2bin::<_, F>(l as u64 - 1, 0);
            let l_binary_len = l_binary.len() as usize;
            let scalar_len = l_binary_len + 1;

            // The last scalar_len bits of OUT and IN1 must be the same.
            for i in 0..scalar_len {
                let out_index = (out_num_vars - 1 - i) as usize;
                let in_index = (in_num_vars - 1 - i) as usize;

                result *= out[out_index] * in1[in_index] * in2[in_index]
                    + (F::ONE - out[out_index])
                        * (F::ONE - in1[in_index])
                        * (F::ONE - in2[in_index]);
            }

            // Add dummy bits for out.
            if out_num_vars > scalar_len + 1 {
                for i in 0..out_num_vars - scalar_len - 1 {
                    result *= F::ONE - out[i + 1];
                }
            }

            // Add dummy bits for in1 & in2.
            if in_num_vars > scalar_len + 1 {
                for i in 0..in_num_vars - scalar_len - 1 {
                    result *= (F::ONE - in1[i + 1]) * (F::ONE - in2[i + 1]);
                }
            }

            result
        })
    }
}

pub struct FFTDivideInverseForwardYExtension<F: Field> {
    factor: F,
    num_vars: usize,
}

impl<F: Field> FFTDivideInverseForwardYExtension<F> {
    pub fn new(factor: F, num_vars: usize) -> Self {
        Self { factor, num_vars }
    }
}

impl<F: Field> CustomMultilinearExtensionHandler<F> for FFTDivideInverseForwardYExtension<F> {
    fn template(&self) -> String {
        format!(
            "FFT_DivideInverse_ForwardY({}, {})",
            self.factor, self.num_vars
        )
    }

    fn out_num_vars(&self) -> usize {
        self.num_vars
    }

    fn in1_num_vars(&self) -> usize {
        self.num_vars + 1
    }

    fn in2_num_vars(&self) -> usize {
        self.num_vars + 1
    }

    fn handler(&self) -> Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F> {
        let factor = self.factor.clone();
        let num_vars = self.num_vars.clone();

        Rc::new(move |points, _| {
            assert!(points.len() == 3);
            let out = points[0];
            let in1 = points[1];
            let in2 = points[2];

            assert!(out.len() == num_vars);
            assert!(out.len() + 1 == in1.len());
            assert!(out.len() + 1 == in2.len());

            let mut result = F::ONE;

            result *= F::ONE - in1[0];
            result *= F::ONE - in2[0];

            for i in 0..out.len() {
                result *= out[i] * in1[i + 1] * in2[i + 1]
                    + (F::ONE - out[i]) * (F::ONE - in1[i + 1]) * (F::ONE - in2[i + 1]);
            }

            result * factor
        })
    }
}
