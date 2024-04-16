use std::{
    cmp::max,
    iter::Sum,
    marker::PhantomData,
    ops::{Add, AddAssign},
};

use icicle_bn254::curve::ScalarField as IcicleFrBN254;
use icicle_core::sisu::{
    bk_produce_case_1, bk_produce_case_2, bk_reduce, bk_sum_all_case_1, bk_sum_all_case_2,
    mul_by_scalar, SumcheckConfig,
};
use icicle_core::traits::FieldImpl;

use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_chunks;

use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use sisulib::{common::ilog2_ceil, field::FrBN254, mle::dense::SisuDenseMultilinearExtension};

use crate::{
    boolean_hypercube::BooleanHypercube, cuda_compat::slice::CudaSlice,
    icicle_converter::IcicleConvertibleField,
};

#[derive(Clone)]
pub struct FunctionEvaluations<F: Field> {
    pub evaluations: Vec<F>,
    is_const_one: bool,
}

impl<F: Field> FunctionEvaluations<F> {
    pub fn new(evaluations: Vec<F>) -> Self {
        Self {
            evaluations,
            is_const_one: false,
        }
    }

    pub fn constant(constant: F, len: usize) -> Self {
        Self {
            evaluations: vec![constant; len],
            is_const_one: constant == F::ONE,
        }
    }

    pub fn default() -> Self {
        Self::constant(F::ONE, 0)
    }

    pub fn is_constant_one(&self) -> bool {
        self.is_const_one
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn get(&self, index: usize) -> &F {
        &self.evaluations[index]
    }

    pub fn set(&mut self, evaluations: Vec<F>) {
        self.evaluations = evaluations;
    }

    pub fn mul(&mut self, c: F) {
        if c == F::ONE {
            return;
        }

        for i in 0..self.evaluations.len() {
            self.evaluations[i] *= c;
        }

        self.is_const_one = false;
    }

    pub fn mle_evaluate(&self, point: Vec<&[F]>) -> F {
        SisuDenseMultilinearExtension::from_slice(&self.evaluations).evaluate(point)
    }

    pub fn as_slice(&self) -> &[F] {
        &self.evaluations
    }
}

pub trait RootProductBookeepingTable<F: IcicleConvertibleField>:
    VariantProductBookeepingTable<F>
{
    fn new(constant: F, evaluations_1: CudaSlice<F>, evaluations_2: CudaSlice<F>) -> Self;

    fn new_dummy() -> Self;
}

pub trait VariantProductBookeepingTable<F: Field> {
    fn num_vars(&self) -> usize;
    fn sum_all(&self) -> F;
    fn produce(&self) -> QuadraticPolynomial<F>;
    fn reduce(&mut self, r: F);
    fn get_last_evaluations(&self) -> (Vec<F>, Vec<F>);
}

/// Generate bookeeping table of product constant*f1*f2.
#[derive(Clone)]
pub struct CPUProductBookeepingTable<F: IcicleConvertibleField> {
    num_vars: usize,
    current_round: usize,
    bookeeping_table_1: FunctionEvaluations<F>,
    bookeeping_table_2: FunctionEvaluations<F>,
}

impl<F: IcicleConvertibleField> CPUProductBookeepingTable<F> {
    fn bk1_is_not_constant_one(&self) -> bool {
        !self.bookeeping_table_1.is_constant_one()
    }

    fn bk2_is_not_constant_one(&self) -> bool {
        !self.bookeeping_table_2.is_constant_one()
    }

    fn both_bk_are_not_constant_one(&self) -> bool {
        self.bk1_is_not_constant_one() && self.bk2_is_not_constant_one()
    }
}

impl<F: IcicleConvertibleField> RootProductBookeepingTable<F> for CPUProductBookeepingTable<F> {
    fn new(constant: F, evaluations_1: CudaSlice<F>, evaluations_2: CudaSlice<F>) -> Self {
        assert!(evaluations_1.len() > 0 || evaluations_1.len() == evaluations_2.len());
        assert!(evaluations_1.len().is_power_of_two());

        let mut evaluations_1 = FunctionEvaluations::new(evaluations_1.as_host());
        let evaluations_2 = FunctionEvaluations::new(evaluations_2.as_host());

        evaluations_1.mul(constant);

        let num_vars = if evaluations_1.len() > 0 {
            evaluations_1.len().ilog2() as usize
        } else {
            evaluations_2.len().ilog2() as usize
        };

        Self {
            num_vars,
            current_round: 1,
            bookeeping_table_1: evaluations_1,
            bookeeping_table_2: evaluations_2,
        }
    }

    fn new_dummy() -> Self {
        Self {
            num_vars: 0,
            current_round: 1,
            bookeeping_table_1: FunctionEvaluations::default(),
            bookeeping_table_2: FunctionEvaluations::default(),
        }
    }
}

impl<F: IcicleConvertibleField> VariantProductBookeepingTable<F> for CPUProductBookeepingTable<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn sum_all(&self) -> F {
        let mut sum = F::ZERO;

        if self.both_bk_are_not_constant_one() {
            for i in 0..self.bookeeping_table_1.len() {
                sum += *self.bookeeping_table_1.get(i) * self.bookeeping_table_2.get(i);
            }
        } else if self.bk1_is_not_constant_one() {
            for i in 0..self.bookeeping_table_1.len() {
                sum += self.bookeeping_table_1.get(i);
            }
        } else if self.bk2_is_not_constant_one() {
            for i in 0..self.bookeeping_table_2.len() {
                sum += self.bookeeping_table_2.get(i);
            }
        }

        sum
    }

    fn produce(&self) -> QuadraticPolynomial<F> {
        let mut result = (F::ZERO, F::ZERO, F::ZERO);
        let n = 2usize.pow((self.num_vars - self.current_round + 1) as u32);

        if self.both_bk_are_not_constant_one() {
            for i0 in (0..n).step_by(2) {
                let i1 = i0 + 1;

                let a10 = self.bookeeping_table_1.get(i0);
                let a20 = self.bookeeping_table_2.get(i0);
                let a11 = self.bookeeping_table_1.get(i1);
                let a21 = self.bookeeping_table_2.get(i1);

                result.0 += *a10 * a20;
                result.1 += *a11 * a21;
                result.2 += (*a11 + a11 - a10) * (*a21 + a21 - a20);
            }
        } else if self.bk1_is_not_constant_one() {
            for i0 in (0..n).step_by(2) {
                let a10 = self.bookeeping_table_1.get(i0);
                let a11 = self.bookeeping_table_1.get(i0 + 1);

                result.0 += a10;
                result.1 += a11;
                result.2 += *a11 + a11 - a10;
            }
        } else if self.bk2_is_not_constant_one() {
            for i0 in (0..n).step_by(2) {
                let a20 = self.bookeeping_table_2.get(i0);
                let a21 = self.bookeeping_table_2.get(i0 + 1);

                result.0 += a20;
                result.1 += a21;
                result.2 += *a21 + a21 - a20;
            }
        }

        QuadraticPolynomial::from_evaluations(result.0, result.1, result.2)
    }

    fn reduce(&mut self, r: F) {
        if self.bk1_is_not_constant_one() {
            self.bookeeping_table_1.set(
                cfg_chunks!(self.bookeeping_table_1.as_slice(), 2)
                    .map(|a1| a1[0] + r * (a1[1] - a1[0]))
                    .collect(),
            );
        }

        if self.bk2_is_not_constant_one() {
            self.bookeeping_table_2.set(
                cfg_chunks!(self.bookeeping_table_2.as_slice(), 2)
                    .map(|a2| a2[0] + r * (a2[1] - a2[0]))
                    .collect(),
            );
        }

        self.current_round += 1;
    }

    fn get_last_evaluations(&self) -> (Vec<F>, Vec<F>) {
        assert!(self.current_round == self.num_vars + 1);

        (
            vec![self.bookeeping_table_1.get(0).clone()],
            vec![self.bookeeping_table_2.get(0).clone()],
        )
    }
}

pub struct GPUProductBookeepingTable<F: IcicleConvertibleField> {
    num_vars: usize,
    current_round: usize,
    bk1_is_constant_one: bool,
    bk2_is_constant_one: bool,
    bookeeping_table_1: HostOrDeviceSlice<F::IcicleField>,
    bookeeping_table_2: HostOrDeviceSlice<F::IcicleField>,
}

impl<F: IcicleConvertibleField> GPUProductBookeepingTable<F> {
    fn bk1_is_not_constant_one(&self) -> bool {
        !self.bk1_is_constant_one
    }

    fn bk2_is_not_constant_one(&self) -> bool {
        !self.bk2_is_constant_one
    }

    fn both_bk_are_not_constant_one(&self) -> bool {
        self.bk1_is_not_constant_one() && self.bk2_is_not_constant_one()
    }
}

impl<'a> RootProductBookeepingTable<FrBN254> for GPUProductBookeepingTable<FrBN254> {
    fn new(
        constant: FrBN254,
        mut evaluations_1: CudaSlice<FrBN254>,
        evaluations_2: CudaSlice<FrBN254>,
    ) -> Self {
        assert!(evaluations_1.len() > 0 || evaluations_1.len() == evaluations_2.len());
        assert!(evaluations_1.len().is_power_of_two());

        mul_by_scalar(evaluations_1.as_mut_device(), constant.to_icicle()).unwrap();

        let num_vars = if evaluations_1.len() > 0 {
            evaluations_1.len().ilog2() as usize
        } else {
            evaluations_2.len().ilog2() as usize
        };

        let bk1_is_constant_one = evaluations_1.is_one();
        let bk2_is_constant_one = evaluations_2.is_one();

        Self {
            num_vars,
            current_round: 1,
            bk1_is_constant_one,
            bk2_is_constant_one,
            bookeeping_table_1: evaluations_1.as_device(),
            bookeeping_table_2: evaluations_2.as_device(),
        }
    }

    fn new_dummy() -> Self {
        Self {
            num_vars: 0,
            current_round: 1,
            bk1_is_constant_one: true,
            bk2_is_constant_one: true,
            bookeeping_table_1: HostOrDeviceSlice::Host(vec![]),
            bookeeping_table_2: HostOrDeviceSlice::Host(vec![]),
        }
    }
}

impl VariantProductBookeepingTable<FrBN254> for GPUProductBookeepingTable<FrBN254> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn sum_all(&self) -> FrBN254 {
        if self.num_vars == 0 {
            return FrBN254::ZERO;
        }

        let n = 2usize.pow((self.num_vars - self.current_round + 1) as u32);
        let config = SumcheckConfig::default();
        let mut result = HostOrDeviceSlice::cuda_malloc(1).unwrap();

        if self.both_bk_are_not_constant_one() {
            bk_sum_all_case_1(
                &config,
                &self.bookeeping_table_1,
                &self.bookeeping_table_2,
                &mut result,
                n as u32,
            )
            .unwrap();
        } else if self.bk1_is_not_constant_one() {
            bk_sum_all_case_2(&config, &self.bookeeping_table_1, &mut result, n as u32).unwrap();
        } else if self.bk2_is_not_constant_one() {
            bk_sum_all_case_2(&config, &self.bookeeping_table_2, &mut result, n as u32).unwrap();
        }

        let mut host_result = vec![IcicleFrBN254::zero()];
        result.copy_to_host(&mut host_result).unwrap();

        FrBN254::from_icicle(&host_result[0])
    }

    fn produce(&self) -> QuadraticPolynomial<FrBN254> {
        let n = 2usize.pow((self.num_vars - self.current_round + 1) as u32);
        let config = SumcheckConfig::default();
        let mut result = HostOrDeviceSlice::cuda_malloc(3).unwrap();

        if self.both_bk_are_not_constant_one() {
            bk_produce_case_1(
                &config,
                &self.bookeeping_table_1,
                &self.bookeeping_table_2,
                &mut result,
                n as u32,
            )
            .unwrap();
        } else if self.bk1_is_not_constant_one() {
            bk_produce_case_2(&config, &self.bookeeping_table_1, &mut result, n as u32).unwrap();
        } else if self.bk2_is_not_constant_one() {
            bk_produce_case_2(&config, &self.bookeeping_table_2, &mut result, n as u32).unwrap();
        }

        let mut host_result = vec![IcicleFrBN254::zero(); 3];
        result.copy_to_host(&mut host_result).unwrap();

        QuadraticPolynomial::from_evaluations(
            FrBN254::from_icicle(&host_result[0]),
            FrBN254::from_icicle(&host_result[1]),
            FrBN254::from_icicle(&host_result[2]),
        )
    }

    fn reduce(&mut self, r: FrBN254) {
        let n = 2usize.pow((self.num_vars - self.current_round + 1) as u32);
        let r = IcicleFrBN254::from(r.into_bigint().0);
        let config = SumcheckConfig::default();

        if self.bk1_is_not_constant_one() {
            bk_reduce(&config, &mut self.bookeeping_table_1, n as u32, r).unwrap();
        }

        if self.bk2_is_not_constant_one() {
            let config = SumcheckConfig::default();
            bk_reduce(&config, &mut self.bookeeping_table_2, n as u32, r).unwrap();
        }

        self.current_round += 1;
    }

    fn get_last_evaluations(&self) -> (Vec<FrBN254>, Vec<FrBN254>) {
        assert!(self.current_round == self.num_vars + 1);

        let last_1 = if !self.bk1_is_constant_one {
            let mut result = vec![IcicleFrBN254::one()];
            self.bookeeping_table_1
                .copy_to_host_partially(&mut result, 0)
                .unwrap();

            FrBN254::from_icicle(&result[0])
        } else {
            FrBN254::ONE
        };

        let last_2 = if !self.bk2_is_constant_one {
            let mut result = vec![IcicleFrBN254::one()];
            self.bookeeping_table_2
                .copy_to_host_partially(&mut result, 0)
                .unwrap();

            FrBN254::from_icicle(&result[0])
        } else {
            FrBN254::ONE
        };

        (vec![last_1], vec![last_2])
    }
}

pub struct SumOfProductBookeepingTable<F: IcicleConvertibleField, BK: RootProductBookeepingTable<F>>
{
    bookeeping_tables: Vec<BK>,
    __phantom: PhantomData<F>,
}

impl<F: IcicleConvertibleField, BK: RootProductBookeepingTable<F>>
    SumOfProductBookeepingTable<F, BK>
{
    pub fn new(
        constant: F,
        evaluations_f: Vec<CudaSlice<F>>,
        evaluations_g: Vec<CudaSlice<F>>,
    ) -> Self {
        let expected_evaluation_size = evaluations_f[0].len();
        let mut bookeeping_tables = vec![];
        for (eval_f, eval_g) in evaluations_f.into_iter().zip(evaluations_g) {
            assert_eq!(expected_evaluation_size, eval_f.len());

            let bk_table = BK::new(constant, eval_f, eval_g);

            bookeeping_tables.push(bk_table);
        }

        Self {
            bookeeping_tables,
            __phantom: PhantomData,
        }
    }
}

impl<F: IcicleConvertibleField, BK: RootProductBookeepingTable<F>> VariantProductBookeepingTable<F>
    for SumOfProductBookeepingTable<F, BK>
{
    fn num_vars(&self) -> usize {
        self.bookeeping_tables[0].num_vars()
    }

    fn sum_all(&self) -> F {
        let mut sum_all = F::ZERO;
        for i in 0..self.bookeeping_tables.len() {
            sum_all += self.bookeeping_tables[i].sum_all();
        }
        sum_all
    }

    fn produce(&self) -> QuadraticPolynomial<F> {
        let mut final_polynomials = QuadraticPolynomial::default();

        for i in 0..self.bookeeping_tables.len() {
            final_polynomials += self.bookeeping_tables[i].produce();
        }

        final_polynomials
    }

    fn reduce(&mut self, r: F) {
        for i in 0..self.bookeeping_tables.len() {
            self.bookeeping_tables[i].reduce(r);
        }
    }

    fn get_last_evaluations(&self) -> (Vec<F>, Vec<F>) {
        let mut last_f = vec![];
        let mut last_g = vec![];
        for i in 0..self.bookeeping_tables.len() {
            let last_evaluation = self.bookeeping_tables[i].get_last_evaluations();
            last_f.push(last_evaluation.0[0]);
            last_g.push(last_evaluation.1[0]);
        }

        (last_f, last_g)
    }
}

pub struct ProductBookeepingTablePlus<F: IcicleConvertibleField, BK: RootProductBookeepingTable<F>>
{
    sum: Vec<F>,
    factor: Vec<F>,
    max_num_vars: usize,
    current_bk_index: usize,
    fixed_num_vars: usize,
    bookeeping_tables: Vec<BK>,
    reindex_map: Vec<usize>, // new index --> origin index
}

impl<F: IcicleConvertibleField, BK: RootProductBookeepingTable<F>>
    ProductBookeepingTablePlus<F, BK>
{
    pub fn new(
        constant: F,
        evaluation_f: Vec<CudaSlice<F>>,
        evaluation_g: Vec<CudaSlice<F>>,
    ) -> Self {
        let mut max_num_evaluations = 0;
        let mut bookeeping_tables = vec![];
        for (ef, eg) in evaluation_f.into_iter().zip(evaluation_g) {
            if ef.len() == 0 {
                continue;
            }

            max_num_evaluations = max(max_num_evaluations, ef.len());
            bookeeping_tables.push(BK::new(constant, ef, eg));
        }

        // Add one dummy bookeeping table with num_vars=0 for more convenient in
        // implement the algorithm.
        bookeeping_tables.push(BK::new_dummy());

        // Initialize origin index map.
        let mut reindex_map = vec![];
        for i in 0..bookeeping_tables.len() {
            reindex_map.push(i);
        }

        // Sort bookeeping table based on the num vars of evaluations. The first
        // bookeeping table is the one with largest num vars.
        let bk_len = bookeeping_tables.len();
        for i in 0..bk_len - 1 {
            for j in i + 1..bk_len {
                if bookeeping_tables[i].num_vars() < bookeeping_tables[j].num_vars() {
                    bookeeping_tables.swap(i, j);
                    reindex_map.swap(i, j);
                }
            }
        }

        let mut sum = vec![];
        let mut factor = vec![];
        for bookeeping_table in &bookeeping_tables {
            let s = bookeeping_table.sum_all();
            sum.push(s);
            factor.push(F::ONE);
        }

        let mut prover = Self {
            sum,
            factor,
            reindex_map,
            bookeeping_tables,
            current_bk_index: 0,
            fixed_num_vars: 0,
            max_num_vars: ilog2_ceil(max_num_evaluations),
        };
        prover.ignore_same_bookeeping_table();

        prover
    }

    pub fn num_rounds(&self) -> usize {
        self.max_num_vars
    }

    fn to_next_bk_index(&mut self) {
        let next_num_vars = self.bookeeping_tables[self.current_bk_index + 1].num_vars();
        if self.max_num_vars - self.fixed_num_vars > next_num_vars {
            // If we haven't fixed all random points of current bookeeping
            // table, we will not go to the next step.
            return;
        }

        self.current_bk_index += 1;
        self.ignore_same_bookeeping_table();
    }

    fn ignore_same_bookeeping_table(&mut self) {
        loop {
            let current_num_vars = self.bookeeping_tables[self.current_bk_index].num_vars();
            let next_num_vars = self.bookeeping_tables[self.current_bk_index + 1].num_vars();

            if current_num_vars > next_num_vars {
                break;
            }
            self.current_bk_index += 1;
        }
    }
}

impl<F: IcicleConvertibleField, BK: RootProductBookeepingTable<F>> VariantProductBookeepingTable<F>
    for ProductBookeepingTablePlus<F, BK>
{
    fn num_vars(&self) -> usize {
        self.max_num_vars
    }

    fn sum_all(&self) -> F {
        assert_eq!(self.fixed_num_vars, 0);
        self.sum.iter().sum()
    }

    fn produce(&self) -> QuadraticPolynomial<F> {
        let mut quadratic_poly = QuadraticPolynomial::default();

        // PRODUCE STEP
        for k in 0..self.current_bk_index + 1 {
            let mut etmp = self.bookeeping_tables[k].produce();
            etmp.mul(&self.factor[k]);
            quadratic_poly += etmp;
        }

        for k in self.current_bk_index + 1..self.bookeeping_tables.len() {
            let tmp = self.sum[k] * self.factor[k];
            let tmp_poly = QuadraticPolynomial::from_evaluations(F::ZERO, tmp, F::from(2u64) * tmp);
            quadratic_poly += tmp_poly;
        }

        quadratic_poly
    }

    fn reduce(&mut self, r: F) {
        // REDUCE STEP
        for k in 0..self.current_bk_index + 1 {
            self.bookeeping_tables[k].reduce(r);
        }

        for k in self.current_bk_index + 1..self.bookeeping_tables.len() {
            self.factor[k] *= r;
        }

        self.fixed_num_vars += 1;
        if self.fixed_num_vars < self.max_num_vars {
            self.to_next_bk_index();
        }
    }

    fn get_last_evaluations(&self) -> (Vec<F>, Vec<F>) {
        assert_eq!(self.fixed_num_vars, self.max_num_vars);

        let mut last_evaluations_f = vec![F::ZERO; self.bookeeping_tables.len() - 1];
        let mut last_evaluations_g = vec![F::ZERO; self.bookeeping_tables.len() - 1];

        for k in 0..self.bookeeping_tables.len() - 1 {
            let last_evaluation = self.bookeeping_tables[k].get_last_evaluations();
            last_evaluations_f[self.reindex_map[k]] = last_evaluation.0[0] * self.factor[k];
            last_evaluations_g[self.reindex_map[k]] = last_evaluation.1[0];
        }

        (last_evaluations_f, last_evaluations_g)
    }
}

#[derive(Clone)]
pub struct SisuSparseMultilinearPolynomial<F: Field> {
    pub num_vars: usize,
    terms: Vec<(F, Vec<usize>)>,
}

impl<F: Field> SisuSparseMultilinearPolynomial<F> {
    pub fn new(num_vars: usize, terms: Vec<(F, Vec<usize>)>) -> Self {
        Self { num_vars, terms }
    }

    pub fn evaluate(&self, point: &[F]) -> F {
        assert!(point.len() == self.num_vars, "invalid point length");

        let mut result = F::ZERO;
        for (coeff, term) in &self.terms {
            let mut tmp = *coeff;
            for t in term {
                if tmp == F::ZERO {
                    break;
                }
                tmp *= point[*t];
            }
            result += tmp;
        }

        result
    }

    pub fn evaluations_on_boolean_hypercube(&self) -> Vec<F> {
        let mut evaluations = vec![];
        for b in BooleanHypercube::new(self.num_vars) {
            evaluations.push(self.evaluate(&b));
        }

        evaluations
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Copy, Clone, Debug, Default)]
pub struct QuadraticPolynomial<F: Field>(pub F, pub F, pub F);

impl<F: Field> QuadraticPolynomial<F> {
    pub fn zero() -> Self {
        Self::new(F::ZERO, F::ZERO, F::ZERO)
    }

    pub fn new(a: F, b: F, c: F) -> Self {
        Self(a, b, c)
    }

    pub fn from_evaluations(f0: F, f1: F, f2: F) -> Self {
        // let c = f0;
        // let a = (f2 + f0) / (F::ONE + F::ONE) - f1;
        // let b = f1 - f0 - a;

        let a = f0;
        let c = (f2 + f0) / (F::ONE + F::ONE) - f1;
        let b = f1 - f0 - c;

        Self::new(a, b, c)
    }

    pub fn mul(&mut self, c: &F) {
        self.0 *= c;
        self.1 *= c;
        self.2 *= c;
    }

    pub fn evaluate(&self, point: F) -> F {
        self.0 + self.1 * point + self.2 * point * point
    }

    pub fn to_vec(&self) -> Vec<F> {
        vec![self.0, self.1, self.2]
    }
}

impl<F: Field> AddAssign for QuadraticPolynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl<F: Field> Add for QuadraticPolynomial<F> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
        self
    }
}

impl<F: Field> Sum for QuadraticPolynomial<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |prev, cur| prev + cur)
    }
}

#[cfg(test)]
mod tests {
    use sisulib::field::FpFRI;

    use crate::hash::SisuMimc;

    use super::QuadraticPolynomial;

    #[test]
    fn test_poly_hash() {
        let poly = QuadraticPolynomial::new(
            FpFRI::from(34u128),
            FpFRI::from(2113u128),
            FpFRI::from(192u128),
        );

        let mimc = SisuMimc::<FpFRI>::default();
        assert_eq!(
            FpFRI::from(11073192571180991937u128),
            mimc.hash_array(&poly.to_vec())
        );
    }
}
