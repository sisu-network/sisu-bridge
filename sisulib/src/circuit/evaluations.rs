use ark_ff::Field;
use ark_std::iterable::Iterable;

use super::{CircuitParams, GateF};

/// If there are more than MAX_ONES_RATIO% number ONE in GateEvaluations, it is
/// treated as a major-ones-evaluations.
const MAJOR_ONES_RATIO: usize = 50;

#[derive(Debug)]
pub struct GateEvaluations<F: Field> {
    pub evaluations: Vec<(usize, GateF<F>)>,
    num_ones: usize,
}

impl<F: Field> Clone for GateEvaluations<F> {
    fn clone(&self) -> Self {
        Self {
            evaluations: self.evaluations.clone(),
            num_ones: self.num_ones,
        }
    }
}

impl<F: Field> GateEvaluations<F> {
    pub fn new(evaluations: Vec<(usize, GateF<F>)>) -> Self {
        let mut num_ones = 0;
        for i in 0..evaluations.len() {
            if evaluations[i].1 == GateF::ONE {
                num_ones += 1;
            }
        }

        Self {
            evaluations,
            num_ones,
        }
    }

    pub fn default() -> Self {
        Self {
            evaluations: vec![],
            num_ones: 0,
        }
    }

    pub fn push(&mut self, value: (usize, GateF<F>)) {
        if value.1 == GateF::ONE {
            self.num_ones += 1;
        }

        self.evaluations.push(value);
    }

    pub fn is_major_ones(&self) -> bool {
        self.len() * MAJOR_ONES_RATIO / 100 <= self.num_ones
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn raw(&self) -> &[(usize, GateF<F>)] {
        &self.evaluations
    }

    pub fn compile<'a>(&'a self, params: &'a CircuitParams<F>) -> GateEvaluationsIterator<'a, F> {
        GateEvaluationsIterator {
            inner: &self,
            params,
            cur: 0,
        }
    }
}

pub struct GateEvaluationsIterator<'a, F: Field> {
    inner: &'a GateEvaluations<F>,
    params: &'a CircuitParams<'a, F>,
    cur: usize,
}

impl<'a, F: Field> Iterable for GateEvaluationsIterator<'a, F> {
    type Item = (usize, F);
    type Iter = Self;

    fn iter(&self) -> Self::Iter {
        Self {
            inner: self.inner,
            params: self.params,
            cur: 0,
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, F: Field> Iterator for GateEvaluationsIterator<'a, F> {
    type Item = (usize, F);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.inner.len() {
            return None;
        }

        let (i, v) = &self.inner.evaluations[self.cur];
        self.cur += 1;

        Some((i.clone(), v.to_value(self.params)))
    }
}
