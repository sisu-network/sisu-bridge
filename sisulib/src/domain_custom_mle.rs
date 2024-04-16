use std::{marker::PhantomData, rc::Rc};

use ark_ff::Field;

use crate::{
    circuit::CircuitParams,
    common::{dec2bin, ilog2_ceil},
    domain::Domain,
    mle::custom::CustomMultilinearExtensionHandler,
};

pub struct FFTShuffleForwardYExtension<F: Field> {
    num_vars: usize,
    __phantom: PhantomData<F>,
}

impl<F: Field> FFTShuffleForwardYExtension<F> {
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            __phantom: PhantomData,
        }
    }
}

impl<F: Field> CustomMultilinearExtensionHandler<F> for FFTShuffleForwardYExtension<F> {
    fn template(&self) -> String {
        format!("FFT_Shuffle_ForwardY({})", self.num_vars)
    }

    fn out_num_vars(&self) -> usize {
        self.num_vars
    }

    fn in1_num_vars(&self) -> usize {
        self.num_vars
    }

    fn in2_num_vars(&self) -> usize {
        self.num_vars
    }

    fn handler(&self) -> Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F> {
        let num_vars = self.num_vars.clone();

        Rc::new(move |points, _| {
            assert!(points.len() == 3);
            let out = points[0];
            let in1 = points[1];
            let in2 = points[2];

            assert!(out.len() == num_vars);
            assert!(out.len() == in1.len());
            assert!(out.len() == in2.len());

            let mut result = F::ONE;

            for i in 0..out.len() {
                let out_index = i as usize;
                let in_index = (out.len() - 1 - i) as usize;

                result *= out[out_index] * in1[in_index] * in2[in_index]
                    + (F::ONE - out[out_index])
                        * (F::ONE - in1[in_index])
                        * (F::ONE - in2[in_index]);
            }

            result
        })
    }
}

fn get_bases<F: Field>(base: F, log_n: usize) -> Vec<F> {
    let mut bases = vec![F::ZERO; log_n];
    bases[0] = base;
    for i in 1..log_n {
        bases[i] = bases[i - 1] * bases[i - 1];
    }

    bases
}

pub struct FFTInterpolateForwardXExtension<F: Field> {
    generator: F,
    stage: usize,
    domain_size: usize,
}

impl<F: Field> FFTInterpolateForwardXExtension<F> {
    pub fn new(domain: &Domain<F>, stage: usize) -> Self {
        let n: usize = domain.len();
        let log_n = n.ilog2() as usize;
        let step_domain = 2usize.pow((log_n - stage - 1) as u32);

        Self {
            generator: domain[domain.len() - step_domain],
            domain_size: domain.len(),
            stage,
        }
    }
}

impl<F: Field> CustomMultilinearExtensionHandler<F> for FFTInterpolateForwardXExtension<F> {
    fn template(&self) -> String {
        format!(
            "FFT_Interpolate_ForwardX({}, {}, {})",
            self.generator.to_string(),
            self.stage,
            self.domain_size
        )
    }

    fn out_num_vars(&self) -> usize {
        self.domain_size.ilog2() as usize
    }

    fn in1_num_vars(&self) -> usize {
        self.domain_size.ilog2() as usize
    }

    fn in2_num_vars(&self) -> usize {
        self.domain_size.ilog2() as usize
    }

    fn handler(&self) -> Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F> {
        let generator = self.generator.clone();
        let stage = self.stage.clone();
        let domain_size = self.domain_size.clone();

        Rc::new(move |points, _| {
            assert!(points.len() == 3);
            let out = points[0];
            let in1 = points[1];
            let in2 = points[2];

            let num_vars = domain_size.ilog2() as usize;
            assert_eq!(out.len(), num_vars);
            assert_eq!(in1.len(), num_vars);
            assert_eq!(in2.len(), num_vars);

            let log_n = in1.len();

            // Calculate bases;
            let bases = get_bases(generator, log_n);

            // case1: the bit at position stage is 0 for in1, in2.
            let mut case1 = F::ONE;
            for i in 0..log_n {
                let index = log_n - 1 - i;

                if i == stage {
                    case1 *= (F::ONE - in1[index]) * (F::ONE - in2[index]);
                } else {
                    case1 *= out[index] * in1[index] * in2[index]
                        + (F::ONE - out[index]) * (F::ONE - in1[index]) * (F::ONE - in2[index]);
                }
            }

            // case2: the bit at position stage is 1 for in1, in2.
            let mut case2 = F::ONE;
            for i in 0..log_n {
                let index = log_n - 1 - i;

                if i == stage {
                    case2 *=
                        in1[index] * in2[index] * (out[index] * bases[i] + (F::ONE - out[index]));
                } else {
                    case2 *= out[index] * in1[index] * in2[index] * bases[i]
                        + (F::ONE - out[index]) * (F::ONE - in1[index]) * (F::ONE - in2[index]);
                }
            }

            // println!("stage = {}, case1 + case2 = {}", stage, case1 + case2);

            case1 + case2
        })
    }
}

pub struct FFTEvaluateForwardXExtension<F: Field> {
    coefficients_size: usize,
    evaluations_size: usize,
    __phantom: PhantomData<F>,
}

impl<F: Field> FFTEvaluateForwardXExtension<F> {
    pub fn new(coefficients_size: usize, evaluations_size: usize) -> Self {
        Self {
            coefficients_size,
            evaluations_size,
            __phantom: PhantomData,
        }
    }
}

impl<F: Field> CustomMultilinearExtensionHandler<F> for FFTEvaluateForwardXExtension<F> {
    fn template(&self) -> String {
        // This mle is a special one because it need a list of input (vector
        // inputs) as signal input in circom template.
        //
        // But we still need to treat is as a custom mle, otherwise, the code
        // generator will treat it as sparse mle and generate a huge function in
        // circom.
        format!("FFT_Evaluate_ForwardX_Dummy()")
    }

    fn out_num_vars(&self) -> usize {
        ilog2_ceil(self.evaluations_size * 2)
    }

    fn in1_num_vars(&self) -> usize {
        self.coefficients_size.ilog2() as usize
    }

    fn in2_num_vars(&self) -> usize {
        self.coefficients_size.ilog2() as usize
    }

    fn handler(&self) -> Rc<dyn Fn(Vec<&[F]>, &CircuitParams<F>) -> F> {
        let coeffs_size = self.coefficients_size.clone();

        Rc::new(move |points, circuit_params| {
            let mut inputs = vec![];
            for idx in circuit_params.d.iter() {
                inputs.push(circuit_params.get_domain()[*idx]);
            }

            assert!(points.len() == 3);
            let out = points[0];
            let in1 = points[1];
            let in2 = points[2];

            let out_num_vars = ilog2_ceil(inputs.len() * 2);
            let in_num_vars = coeffs_size.ilog2() as usize;
            assert_eq!(out.len(), out_num_vars);
            assert_eq!(in1.len(), in_num_vars);
            assert_eq!(in2.len(), in_num_vars);

            let log_n = in1.len();
            let mut bases = vec![];

            for i in 0..inputs.len() {
                bases.push(get_bases(inputs[i], log_n));
                bases.push(get_bases(F::ZERO - inputs[i], log_n));
            }

            // let mut cases = vec![];
            let mut ret = F::ZERO;
            for k in 0..bases.len() {
                let mut case = F::ONE;
                for i in 0..log_n {
                    let index = log_n - 1 - i;
                    case *= bases[k][i] * in1[index] * in2[index]
                        + (F::ONE - in1[index]) * (F::ONE - in2[index]);
                }

                // Convert k to binary.
                let k_binary: Vec<usize> = dec2bin(k as u32, out.len());
                let mut tmp = F::ONE;
                for j in 0..k_binary.len() {
                    if k_binary[j] == 0 {
                        tmp *= F::ONE - out[j];
                    } else {
                        tmp *= out[j];
                    }
                }

                ret += case * tmp;
            }

            ret
        })
    }
}
