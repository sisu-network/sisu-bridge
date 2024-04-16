use std::ops::{Mul, Sub};

use ark_ff::Field;

use super::CircuitParams;

#[derive(Clone, PartialEq, Eq, Debug, Hash, Default)]
pub enum GateF<F: Field> {
    #[default]
    ONE,
    C(F),

    /// Domain[params.d[k]+i].pow(j).
    DomainPd(bool, usize, usize, usize),
}

impl<F: Field> GateF<F> {
    pub fn from(x: u64) -> Self {
        Self::C(F::from(x))
    }

    pub fn to_value<'a>(&'a self, params: &'a CircuitParams<'a, F>) -> F {
        match &self {
            GateF::ONE => F::ONE,
            GateF::C(x) => x.clone(),
            GateF::DomainPd(positive, k, idx, pow) => {
                let d = params.d[*k];
                if *positive {
                    params.get_domain()[(d + idx) * pow]
                } else {
                    if pow & 1 == 0 {
                        params.get_domain()[(d + idx) * pow]
                    } else {
                        // Fast approach
                        let tmp = params.get_domain()[(d + idx) * (pow - 1)];
                        tmp * (F::ZERO - params.get_domain()[d + idx])

                        // Slow approach 1
                        // params.get_domain()[(params.d2 + idx) * pow]

                        // Slow approach 2
                        // params.get_domain()[(params.d1 + idx + domain.len()/2) * pow]
                    }
                }
            }
        }
    }
}

impl<F: Field> Mul for GateF<F> {
    type Output = GateF<F>;
    fn mul(self, rhs: Self) -> Self::Output {
        match self {
            GateF::ONE => rhs,
            GateF::C(x) => match rhs {
                GateF::ONE => self,
                GateF::C(y) => GateF::C(x * y),
                _ => panic!("not support"),
            },
            GateF::DomainPd(_, _, _, _) => match rhs {
                GateF::ONE => self,
                _ => panic!("not support"),
            },
        }
    }
}

impl<F: Field> Sub for GateF<F> {
    type Output = GateF<F>;
    fn sub(self, rhs: Self) -> Self::Output {
        match self {
            GateF::ONE => match rhs {
                GateF::ONE => GateF::C(F::ZERO),
                GateF::C(y) => GateF::C(F::ONE - y),
                _ => panic!("not support"),
            },
            GateF::C(x) => match rhs {
                GateF::ONE => GateF::C(x - F::ONE),
                GateF::C(y) => GateF::C(x - y),
                _ => panic!("not support"),
            },
            GateF::DomainPd(_, _, _, _) => panic!("not support"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash, Default)]
pub enum CircuitGateType<F: Field, Nested> {
    #[default]
    Zero,
    Constant(GateF<F>),
    Sub,
    Add,
    CAdd(GateF<F>),
    CSub(GateF<F>),
    Mul(GateF<F>),
    Xor(GateF<F>),
    NAAB,
    Fourier(GateF<F>),
    ForwardX(GateF<F>),
    ForwardY(GateF<F>),
    Accumulation(Vec<Nested>),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Copy)]
pub enum GateIndex {
    #[default]
    Dummy, // Only for Zero, Accumulate
    Absolute(usize),
    Relative(usize),
}

impl GateIndex {
    pub fn value(&self) -> usize {
        match self {
            Self::Absolute(idx) => *idx,
            Self::Dummy => 0,
            _ => panic!("Cannot get value of a relative index or dummy index"),
        }
    }

    pub fn finalize(&mut self, padding_offset: usize) {
        match self {
            Self::Relative(offset) => *self = Self::Absolute(padding_offset + *offset),
            _ => {}
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CircuitGate<F: Field> {
    _label: String,
    pub gate_type: CircuitGateType<F, Self>,
    pub input_indices: [GateIndex; 2],
}

impl<F: Field> CircuitGate<F> {
    pub fn new(
        label: &str,
        gate_type: CircuitGateType<F, Self>,
        input_indices: [GateIndex; 2],
    ) -> Self {
        Self {
            _label: String::from(label),
            gate_type,
            input_indices,
        }
    }

    pub fn new_dummy(label: &str) -> Self {
        Self::new(
            label,
            CircuitGateType::Zero,
            [GateIndex::Dummy, GateIndex::Dummy],
        )
    }

    pub fn finalize(&mut self, current_layer_size: usize) {
        match &mut self.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    subgate.finalize(current_layer_size);
                }
            }

            _ => {
                self.input_indices[0].finalize(current_layer_size);
                self.input_indices[1].finalize(current_layer_size);
            }
        }
    }

    pub fn check_input_indices(&self, current_layer_size: usize) -> bool {
        match &self.gate_type {
            CircuitGateType::Accumulation(subgates) => {
                for subgate in subgates {
                    if !subgate.check_input_indices(current_layer_size) {
                        return false;
                    }
                }

                true
            }

            _ => {
                self.input_indices[0].value() < current_layer_size
                    && self.input_indices[1].value() < current_layer_size
            }
        }
    }
}
