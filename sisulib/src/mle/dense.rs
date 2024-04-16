use std::ops::{Add, MulAssign, Sub};

use ark_ff::Field;

use crate::common::{combine_point, ilog2_ceil};

#[derive(Clone, Debug)]
pub struct SisuDenseMultilinearExtension<'a, F: Field> {
    evaluations: &'a [F],
    num_vars: usize,
}

impl<'a, F: Field> SisuDenseMultilinearExtension<'a, F> {
    pub fn from_slice(evaluations: &'a [F]) -> Self {
        assert!(evaluations.len() == 0 || evaluations.len().is_power_of_two());

        Self {
            num_vars: ilog2_ceil(evaluations.len()),
            evaluations,
        }
    }

    pub fn new_dummy(num_vars: usize) -> Self {
        Self {
            num_vars,
            evaluations: &[],
        }
    }

    pub fn evaluate(&self, point: Vec<&[F]>) -> F {
        if self.evaluations.len() == 0 {
            return F::ZERO;
        }

        if self.evaluations.len() == 1 {
            return self.evaluations[0];
        }

        let mut point = combine_point(point);
        point.reverse();

        assert!(
            self.evaluations.len().is_power_of_two()
                && self.evaluations.len().ilog2() as usize == self.num_vars
        );
        assert_eq!(point.len(), self.num_vars, "invalid size of partial point");
        let mut poly = self.evaluations.to_vec();
        let nv = self.num_vars;
        let dim = point.len();
        // evaluate single variable of partial point from left to right
        for i in 1..dim + 1 {
            let r = point[i - 1];
            for b in 0..(1 << (nv - i)) {
                let left = poly[b << 1];
                let right = poly[(b << 1) + 1];
                poly[b] = left + r * (right - left);
            }
        }

        poly[0]
    }

    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

#[derive(Clone, Debug)]
pub struct SisuDenseMajorZeroMultilinearExtension<F: Field> {
    groups: Vec<Vec<F>>,
    num_vars: usize,
}

impl<F: Field> SisuDenseMajorZeroMultilinearExtension<F> {
    pub fn from_groups(groups: Vec<Vec<F>>, super_group_size: usize) -> Self {
        assert!(groups.len().is_power_of_two());
        assert!(super_group_size.is_power_of_two());
        assert!(groups[0].len() <= super_group_size);

        for i in 0..groups.len() {
            assert!(groups[i].len() == 0 || groups[i].len().is_power_of_two());
            assert_eq!(groups[i].len(), groups[0].len());
        }

        let num_vars = (super_group_size * groups.len()).ilog2() as usize;

        Self { num_vars, groups }
    }

    pub fn evaluate(&self, point: Vec<&[F]>) -> F {
        let point = combine_point(point);

        // points = [connection_vars zero_vars group_vars]
        let group_num_vars = self.groups[0].len().ilog2() as usize;
        let connection_num_vars = self.groups.len().ilog2() as usize;
        let zero_num_vars = self.num_vars - connection_num_vars - group_num_vars;

        let mut group_mle = vec![];
        for group in self.groups.iter() {
            let dense_mle = SisuDenseMultilinearExtension::from_slice(group);
            group_mle.push(dense_mle.evaluate(vec![&point[connection_num_vars + zero_num_vars..]]));
        }

        let mut beta_zero = F::ONE;
        for i in 0..zero_num_vars {
            beta_zero *= F::ONE - point[connection_num_vars + i];
        }

        for i in 0..group_mle.len() {
            group_mle[i] *= beta_zero;
        }

        let dense_mle = SisuDenseMultilinearExtension::from_slice(&group_mle);
        dense_mle.evaluate(vec![&point[..connection_num_vars]])
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

/// Identity MLE returns the mle of function f(x, y), where f(x, y) == 1 iff x=y.
pub fn identity_mle<F>(rx: &[F], ry: &[F]) -> F
where
    F: Add<Output = F> + Sub<Output = F> + MulAssign + From<u8> + Copy,
{
    identity_mle_vec(vec![rx, ry])
}

/// Identity MLE returns the mle of function f(r...), where f(r...) == 1 iff r[0]==r[1]==..==r[n].
pub fn identity_mle_vec<F>(rs: Vec<&[F]>) -> F
where
    F: Add<Output = F> + Sub<Output = F> + MulAssign + From<u8> + Copy,
{
    for r in rs.iter() {
        assert!(r.len() == rs[0].len());
    }

    let mut result = F::from(1);
    for var_index in 0..rs[0].len() {
        let mut product = F::from(1);
        let mut one_minus_product = F::from(1);
        for r_index in 0..rs.len() {
            product *= rs[r_index][var_index];
            one_minus_product *= F::from(1) - rs[r_index][var_index];
        }

        result *= product + one_minus_product;
    }

    result
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::field::FpSisu;

    use super::*;

    #[test]
    fn test_identity_mle() {
        assert_eq!(identity_mle(&[1, 0, 0, 1], &[1, 0, 0, 1]), 1);
        assert_ne!(identity_mle(&[1, 0, 1, 1], &[1, 0, 0, 1]), 1);
    }

    #[test]
    fn test_dense_major_zero_mle() {
        let num_replicas = 32usize;
        let replica_size = 4096usize;
        let super_replica_size = 4096usize;

        let num_evaluations = super_replica_size * num_replicas;
        let mut evaluations = vec![FpSisu::ZERO; num_evaluations];
        let mut groups = vec![vec![FpSisu::ZERO; replica_size]; num_replicas];
        for replica_index in 0..num_replicas {
            for element_index in 0..replica_size {
                let value = FpSisu::from(((1 + replica_index) * (1 + element_index)) as u64);
                groups[replica_index][element_index] = value;
                evaluations[replica_index * super_replica_size + element_index] = value;
            }
        }

        let mut points = vec![];
        for i in 0..num_evaluations.ilog2() as usize {
            points.push(FpSisu::from((i + 1) as u64));
        }

        let now = Instant::now();
        let dense_major_zero_mle =
            SisuDenseMajorZeroMultilinearExtension::from_groups(groups, super_replica_size);
        let dense_major_zero_value = dense_major_zero_mle.evaluate(vec![&points]);
        println!("MAJOR: {:?}", now.elapsed());

        let now = Instant::now();
        let dense_mle = SisuDenseMultilinearExtension::from_slice(&evaluations);
        let dense_value = dense_mle.evaluate(vec![&points]);
        println!("NORMAL: {:?}", now.elapsed());

        assert_eq!(dense_major_zero_value, dense_value);
    }
}
