use ark_ff::Field;
use ark_std::cfg_into_iter;
use sisulib::codegen::generator::FuncGenerator;
use sisulib::common::Error;
use std::time::Instant;
use std::{marker::PhantomData, ops::AddAssign};

use crate::cuda_compat::slice::CudaSlice;
use crate::icicle_converter::IcicleConvertibleField;
use crate::{
    channel::{MasterNode, NoChannel, SisuReceiver, SisuSender, WorkerNode},
    fiat_shamir::{DummyFiatShamirEngine, FiatShamirEngine, Transcript, TranscriptIter},
    polynomial::{
        CPUProductBookeepingTable, ProductBookeepingTablePlus, QuadraticPolynomial,
        RootProductBookeepingTable, SumOfProductBookeepingTable, VariantProductBookeepingTable,
    },
};

pub struct MultiProductSumcheckWorkerProver<
    'a,
    F: Field,
    B: VariantProductBookeepingTable<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    worker: Option<&'a WorkerNode<S, R>>,
    is_new_protocol: bool,
    num_rounds: usize,
    bookeeping_tables: Vec<Option<B>>,
    __phantom: PhantomData<F>,
}

impl<'a, F: Field, B: VariantProductBookeepingTable<F>, S: SisuSender, R: SisuReceiver>
    MultiProductSumcheckWorkerProver<'a, F, B, S, R>
{
    pub fn default() -> Self {
        Self {
            worker: None,
            is_new_protocol: true,
            bookeeping_tables: vec![],
            num_rounds: 0,
            __phantom: PhantomData,
        }
    }

    pub fn new(worker: &'a WorkerNode<S, R>) -> Self {
        Self {
            worker: Some(worker),
            is_new_protocol: true,
            bookeeping_tables: vec![],
            num_rounds: 0,
            __phantom: PhantomData,
        }
    }

    pub fn not_new_protocol(&mut self) {
        self.is_new_protocol = false;
    }

    /// Check if this is a worker prover.
    pub fn is_worker(&self) -> bool {
        self.worker.is_some()
    }

    /// Check if this is a standalone prover.
    pub fn is_standalone(&self) -> bool {
        self.worker.is_none()
    }

    pub fn len(&self) -> usize {
        self.bookeeping_tables.len()
    }

    pub fn add(&mut self, bookeeping_table: Option<B>) {
        if bookeeping_table.is_some() {
            if self.num_rounds > 0 {
                assert_eq!(
                    bookeeping_table.as_ref().unwrap().num_vars(),
                    self.num_rounds,
                );
            } else {
                self.num_rounds = bookeeping_table.as_ref().unwrap().num_vars();
            }
        }

        self.bookeeping_tables.push(bookeeping_table);
    }

    pub fn get_bookeeping_tables(&self) -> &[Option<B>] {
        &self.bookeeping_tables
    }

    pub fn worker_unchecked(&self) -> &'a WorkerNode<S, R> {
        self.worker.as_ref().unwrap()
    }

    pub fn send_sum_all(&self, sum_all: &[F]) {
        if self.worker.is_some() {
            self.worker_unchecked()
                .send_to_master_and_done(&sum_all)
                .unwrap();
        }
    }

    pub fn send_poly_and_recv_final_poly(
        &self,
        polynomials: Vec<QuadraticPolynomial<F>>,
    ) -> QuadraticPolynomial<F> {
        if self.worker.is_some() {
            self.worker_unchecked()
                .send_to_master(&polynomials)
                .unwrap();
            self.worker_unchecked().recv_from_master().unwrap()
        } else {
            polynomials.into_iter().sum()
        }
    }

    pub fn send_last_evaluations(&self, last_1: &[Vec<F>], last_2: &[Vec<F>]) {
        if self.worker.is_some() {
            self.worker_unchecked()
                .send_to_master_and_done(&last_1)
                .unwrap();
            self.worker_unchecked()
                .send_to_master_and_done(&last_2)
                .unwrap();
        }
    }

    /// Run the sumcheck, return the random points and transcript .
    pub fn run<FS: FiatShamirEngine<F>>(
        &mut self,
        fiat_shamir_engine: &mut FS,
    ) -> (Vec<F>, Vec<Vec<F>>, Vec<Vec<F>>, Transcript) {
        if self.is_new_protocol {
            fiat_shamir_engine.begin_protocol();
        }

        let mut transcript = Transcript::default();

        let mut sum_all = vec![];
        for bk in self.bookeeping_tables.iter() {
            if bk.is_some() {
                let s = bk.as_ref().unwrap().sum_all();
                transcript.serialize_and_push(&s);
                sum_all.push(s);
            }
        }
        self.send_sum_all(&sum_all);

        let mut random_points = vec![];
        for _ in 0..self.num_rounds {
            let mut polynomials = vec![];
            for bk in self.bookeeping_tables.iter() {
                if bk.is_some() {
                    let quadratic_poly = bk.as_ref().unwrap().produce();
                    transcript.serialize_and_push(&quadratic_poly);
                    polynomials.push(quadratic_poly);
                }
            }

            let final_poly = self.send_poly_and_recv_final_poly(polynomials);

            let r = fiat_shamir_engine.reduce_and_hash_to_field(&final_poly.to_vec());

            for bk in self.bookeeping_tables.iter_mut() {
                if bk.is_some() {
                    bk.as_mut().unwrap().reduce(r);
                }
            }

            random_points.push(r);
        }

        let mut last_evaluations_1 = vec![];
        let mut last_evaluations_2 = vec![];
        for i in 0..self.bookeeping_tables.len() {
            if self.bookeeping_tables[i].is_some() {
                let last_evaluations = self.bookeeping_tables[i]
                    .as_ref()
                    .unwrap()
                    .get_last_evaluations();
                last_evaluations_1.push(last_evaluations.0);
                last_evaluations_2.push(last_evaluations.1);
            }
        }

        self.send_last_evaluations(&last_evaluations_1, &last_evaluations_2);

        random_points.reverse();

        if self.worker.is_some() {
            let mut master_random_points = self
                .worker_unchecked()
                .recv_from_master::<Vec<F>>()
                .unwrap();
            master_random_points.extend(random_points);
            random_points = master_random_points;

            last_evaluations_1 = self.worker_unchecked().recv_from_master().unwrap();
            last_evaluations_2 = self.worker_unchecked().recv_from_master().unwrap();
        }

        let mut idx = 0;
        let mut final_last_evaluations_1 = vec![];
        let mut final_last_evaluations_2 = vec![];
        for i in 0..self.bookeeping_tables.len() {
            if self.bookeeping_tables[i].is_some() {
                final_last_evaluations_1.push(last_evaluations_1[idx].clone());
                final_last_evaluations_2.push(last_evaluations_2[idx].clone());
                idx += 1;
            } else {
                final_last_evaluations_1.push(vec![]);
                final_last_evaluations_2.push(vec![]);
            }
        }

        (
            random_points,
            final_last_evaluations_1,
            final_last_evaluations_2,
            transcript,
        )
    }
}

pub struct MultiProductSumcheckMasterProver<
    'a,
    F: IcicleConvertibleField,
    BK: RootProductBookeepingTable<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    master: &'a MasterNode<S, R>,
    num_workers: usize,
    num_worker_rounds: usize,

    /// Common coefficients for all sumchecks. It will be multiplied by last_evaluations_1 before
    /// run the master sumcheck.
    common_coeffs: Vec<F>,

    /// Coefficicents for each sumchecks. It will be multiplied by last_evaluations_2 before
    /// run the master sumcheck.
    coeffs: Vec<Vec<F>>,

    /// Store status of each bookeeping table, it will be used when return the last evaluations.
    /// Which bookeeping tables are none will generate an empty last evaluations.
    bookeeping_table_statuses: Vec<Option<()>>,

    __phantom: PhantomData<BK>,
}

impl<
        'a,
        F: IcicleConvertibleField,
        BK: RootProductBookeepingTable<F>,
        S: SisuSender,
        R: SisuReceiver,
    > MultiProductSumcheckMasterProver<'a, F, BK, S, R>
{
    pub fn new(
        master: &'a MasterNode<S, R>,
        num_workers: usize,
        num_worker_rounds: usize,
        common_coeffs: Vec<F>,
    ) -> Self {
        assert!(common_coeffs.len() == num_workers || common_coeffs.len() == 0);

        Self {
            master,
            num_workers,
            num_worker_rounds,
            common_coeffs,
            coeffs: vec![],
            bookeeping_table_statuses: vec![],
            __phantom: PhantomData,
        }
    }

    pub fn add(&mut self, coeff: Option<Vec<F>>) {
        if coeff.is_some() {
            self.bookeeping_table_statuses.push(Some(()));

            let coeff_len = coeff.as_ref().unwrap().len();
            assert!(coeff_len == self.num_workers || coeff_len == 0);
            self.coeffs.push(coeff.unwrap());
        } else {
            self.bookeeping_table_statuses.push(None);
        }
    }

    fn get_common_coeffs(&self, worker_index: usize) -> F {
        self.common_coeffs
            .get(worker_index)
            .unwrap_or(&F::ONE)
            .clone()
    }

    fn get_coeffs(&self, worker_index: usize, sumcheck_index: usize) -> F {
        self.coeffs[sumcheck_index]
            .get(worker_index)
            .unwrap_or(&F::ONE)
            .clone()
    }

    fn synthetize_polynomials(
        &self,
        worker_polynomials: &mut Vec<Vec<QuadraticPolynomial<F>>>,
    ) -> Vec<QuadraticPolynomial<F>> {
        let num_sumchecks = self.coeffs.len();

        let mut final_polynomials = vec![QuadraticPolynomial::default(); num_sumchecks];
        for worker_index in 0..self.num_workers {
            assert!(worker_polynomials[worker_index].len() == num_sumchecks);
            for sumcheck_index in 0..num_sumchecks {
                worker_polynomials[worker_index][sumcheck_index]
                    .mul(&self.get_common_coeffs(worker_index));

                worker_polynomials[worker_index][sumcheck_index]
                    .mul(&self.get_coeffs(worker_index, sumcheck_index));

                final_polynomials[sumcheck_index] +=
                    worker_polynomials[worker_index][sumcheck_index];
            }
        }

        final_polynomials
    }

    /// Run the sumcheck, return the random points and transcript .
    pub fn run<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
    ) -> (CudaSlice<F>, Vec<Vec<F>>, Vec<Vec<F>>, Transcript) {
        fiat_shamir_engine.begin_protocol();

        let num_sumchecks = self.coeffs.len();

        let mut transcript = Transcript::default();

        let worker_sum_all = self.master.recv_from_workers_and_done::<Vec<F>>().unwrap();

        let mut master_sum_all = vec![F::ZERO; num_sumchecks];
        for worker_index in 0..self.num_workers {
            assert_eq!(worker_sum_all[worker_index].len(), num_sumchecks);
            for sumcheck_index in 0..num_sumchecks {
                master_sum_all[sumcheck_index] += self.get_common_coeffs(worker_index)
                    * self.get_coeffs(worker_index, sumcheck_index)
                    * worker_sum_all[worker_index][sumcheck_index];
            }
        }

        for s in master_sum_all.iter() {
            transcript.serialize_and_push(s);
        }

        let mut random_points = vec![];
        let mut prev_g_r = master_sum_all.clone();
        let mut final_last_g_r = vec![vec![F::ZERO; self.num_workers]; num_sumchecks];
        for round_index in 0..self.num_worker_rounds {
            let mut worker_polynomials = self
                .master
                .recv_from_workers::<Vec<QuadraticPolynomial<F>>>()
                .unwrap();

            let final_polynomials = self.synthetize_polynomials(&mut worker_polynomials);

            let sum_final_polynomials: QuadraticPolynomial<F> =
                final_polynomials.clone().into_iter().sum();
            self.master.send_to_workers(&sum_final_polynomials).unwrap();

            for sumcheck_index in 0..num_sumchecks {
                assert_eq!(
                    prev_g_r[sumcheck_index],
                    final_polynomials[sumcheck_index].evaluate(F::ZERO)
                        + final_polynomials[sumcheck_index].evaluate(F::ONE),
                    "failed to compare final poly with prev g_r at round {}-{}",
                    round_index,
                    sumcheck_index
                );
            }
            for p in final_polynomials.iter() {
                transcript.serialize_and_push(p);
            }

            let r = fiat_shamir_engine.reduce_and_hash_to_field(&sum_final_polynomials.to_vec());
            for sumcheck_index in 0..num_sumchecks {
                prev_g_r[sumcheck_index] = final_polynomials[sumcheck_index].evaluate(r);
            }

            random_points.push(r);

            if round_index == self.num_worker_rounds - 1 {
                for sumcheck_index in 0..num_sumchecks {
                    for worker_index in 0..self.num_workers {
                        final_last_g_r[sumcheck_index][worker_index] =
                            worker_polynomials[worker_index][sumcheck_index].evaluate(r);
                    }
                }
            }
        }

        // num_workers - num_real_sumchecks - num_functions
        let last_evaluations_1 = self
            .master
            .recv_from_workers_and_done::<Vec<Vec<F>>>()
            .unwrap();
        let last_evaluations_2 = self
            .master
            .recv_from_workers_and_done::<Vec<Vec<F>>>()
            .unwrap();

        let num_functions = last_evaluations_1[0][0].len();

        // num_sumchecks - num_functions - num_workers
        let mut reordered_last_evaluations_1 = vec![vec![vec![]; num_functions]; num_sumchecks];
        let mut reordered_last_evaluations_2 = vec![vec![vec![]; num_functions]; num_sumchecks];

        for sumcheck_index in 0..num_sumchecks {
            for function_index in 0..num_functions {
                for worker_index in 0..self.num_workers {
                    reordered_last_evaluations_1[sumcheck_index][function_index].push(
                        last_evaluations_1[worker_index][sumcheck_index][function_index]
                            * self.get_common_coeffs(worker_index),
                    );

                    reordered_last_evaluations_2[sumcheck_index][function_index].push(
                        last_evaluations_2[worker_index][sumcheck_index][function_index]
                            * self.get_coeffs(worker_index, sumcheck_index),
                    );
                }
            }
        }

        // Sum all functions in every sumcheck.
        for sumcheck_index in 0..num_sumchecks {
            for worker_index in 0..self.num_workers {
                let mut final_value = F::ZERO;
                for function_index in 0..num_functions {
                    final_value += reordered_last_evaluations_1[sumcheck_index][function_index]
                        [worker_index]
                        * reordered_last_evaluations_2[sumcheck_index][function_index]
                            [worker_index];
                }

                assert_eq!(final_last_g_r[sumcheck_index][worker_index], final_value);
            }
        }

        random_points.reverse();

        let mut master_prover =
            MultiProductSumcheckWorkerProver::<_, _, NoChannel, NoChannel>::default();
        master_prover.not_new_protocol();

        for (evaluations_f, evaluations_g) in reordered_last_evaluations_1
            .into_iter()
            .zip(reordered_last_evaluations_2.into_iter())
        {
            master_prover.add(Some(SumOfProductBookeepingTable::<F, BK>::new(
                F::ONE,
                cfg_into_iter!(evaluations_f)
                    .map(|x| CudaSlice::on_host(x))
                    .collect(),
                cfg_into_iter!(evaluations_g)
                    .map(|x| CudaSlice::on_host(x))
                    .collect(),
            )));
        }

        // Separate this fiat_shamir because after this time, only master node
        // run the fiat-shamir and worker nodes don't know about it.
        let mut fiat_shamir_clone: FS = fiat_shamir_engine.freeze();
        let (mut master_random_points, last_1, last_2, master_transcript) =
            master_prover.run(&mut fiat_shamir_clone);

        let mut master_transcript_iter: TranscriptIter = master_transcript.into_iter();

        // We will not include this information of worker sumchecks in the synthetized sumcheck.
        // Only ONE sum_all is included at the beginning of the transcript.
        for _ in 0..num_sumchecks {
            let _sum_all = master_transcript_iter.pop_and_deserialize::<F>();
        }

        for _ in 0..master_transcript_iter.remaining_len() {
            transcript.push(master_transcript_iter.pop().to_vec());
        }

        self.master.send_to_workers(&master_random_points).unwrap();
        self.master.send_to_workers(&last_1).unwrap();
        self.master.send_to_workers(&last_2).unwrap();

        // Adjust last_evaluations.
        let mut idx = 0;
        let mut final_last_evaluations_1 = vec![];
        let mut final_last_evaluations_2 = vec![];
        for i in 0..self.bookeeping_table_statuses.len() {
            if self.bookeeping_table_statuses[i].is_some() {
                final_last_evaluations_1.push(last_1[idx].clone());
                final_last_evaluations_2.push(last_2[idx].clone());
                idx += 1;
            } else {
                final_last_evaluations_1.push(vec![]);
                final_last_evaluations_2.push(vec![]);
            }
        }

        master_random_points.extend(random_points);

        (
            CudaSlice::on_host(master_random_points),
            final_last_evaluations_1,
            final_last_evaluations_2,
            transcript,
        )
    }
}

pub struct MultiSumcheckVerifier<F: Field> {
    num_sumchecks: usize,
    num_master_rounds: usize,
    num_worker_rounds: usize,
    __phantom: PhantomData<F>,
}

impl<F: Field> MultiSumcheckVerifier<F> {
    pub fn new(num_sumchecks: usize, num_master_rounds: usize, num_worker_rounds: usize) -> Self {
        Self {
            num_sumchecks,
            num_master_rounds,
            num_worker_rounds,
            __phantom: PhantomData,
        }
    }

    // Run the sumcheck, return the random points, last values (depend on phase) and transcript .
    pub fn verify<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        mut transcript: TranscriptIter,
    ) -> Result<(Vec<F>, F, F), Error> {
        fiat_shamir_engine.begin_protocol();

        let mut sum_all = F::ZERO;
        let mut prev_poly_at_r = vec![];
        for _ in 0..self.num_sumchecks {
            let s = transcript.pop_and_deserialize();
            sum_all += s;
            prev_poly_at_r.push(s);
        }

        let mut random_points = vec![];
        for round_index in 0..self.num_worker_rounds {
            let mut final_poly = QuadraticPolynomial::default();
            let mut all_polynomials = vec![];
            for i in 0..self.num_sumchecks {
                let poly = transcript.pop_and_deserialize::<QuadraticPolynomial<F>>();

                if poly.evaluate(F::ZERO) + poly.evaluate(F::ONE) != prev_poly_at_r[i] {
                    return Err(Error::Sumcheck(format!(
                        "sum of two evaluations at round {} doesn't equal to the pre-computed sum",
                        round_index
                    )));
                }

                final_poly += poly;
                all_polynomials.push(poly);
            }

            let r = fiat_shamir_engine.reduce_and_hash_to_field(&final_poly.to_vec());

            for i in 0..self.num_sumchecks {
                prev_poly_at_r[i] = all_polynomials[i].evaluate(r);
            }

            random_points.push(r);
        }

        // Separate this fiat_shamir because after this time, only master node
        // run the fiat-shamir and worker nodes don't know about it.
        let mut fiat_shamir_engine = fiat_shamir_engine.freeze();
        for round_index in 0..self.num_master_rounds {
            let mut final_poly = QuadraticPolynomial::default();
            let mut all_polynomials = vec![];
            for i in 0..self.num_sumchecks {
                let poly = transcript.pop_and_deserialize::<QuadraticPolynomial<F>>();

                if poly.evaluate(F::ZERO) + poly.evaluate(F::ONE) != prev_poly_at_r[i] {
                    return Err(Error::Sumcheck(format!(
                        "sum of two evaluations at round {} doesn't equal to the pre-computed sum",
                        round_index
                    )));
                }

                final_poly += poly;
                all_polynomials.push(poly);
            }

            let r = fiat_shamir_engine.reduce_and_hash_to_field(&final_poly.to_vec());

            for i in 0..self.num_sumchecks {
                prev_poly_at_r[i] = all_polynomials[i].evaluate(r);
            }

            random_points.push(r);
        }

        let mut final_value = F::ZERO;
        for i in 0..self.num_sumchecks {
            final_value += prev_poly_at_r[i];
        }

        random_points.reverse();

        Ok((random_points, final_value, sum_all))
    }

    pub fn extract_transcript(&self, mut transcript: TranscriptIter) -> MultiSumcheckTranscript<F> {
        let mut sum_all = vec![];
        for _ in 0..self.num_sumchecks {
            sum_all.push(transcript.pop_and_deserialize());
        }

        let mut polynomials = vec![vec![]; self.num_sumchecks];
        for _ in 0..self.num_worker_rounds + self.num_master_rounds {
            for i in 0..self.num_sumchecks {
                let poly = transcript.pop_and_deserialize::<QuadraticPolynomial<F>>();
                polynomials[i].push(poly);
            }
        }

        MultiSumcheckTranscript {
            sum_all,
            polynomials,
        }
    }

    pub fn configs(&self) -> MultiSumcheckConfigs {
        MultiSumcheckConfigs {
            num_worker_rounds: self.num_worker_rounds,
            num_master_rounds: self.num_master_rounds,
            num_sumchecks: self.num_sumchecks,
        }
    }
}

#[derive(Default)]
pub struct MultiSumcheckTranscript<F: Field> {
    sum_all: Vec<F>,
    polynomials: Vec<Vec<QuadraticPolynomial<F>>>,
}

impl<F: Field> MultiSumcheckTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];

        result.extend_from_slice(&self.sum_all);
        for j in 0..self.polynomials[0].len() {
            for i in 0..self.polynomials.len() {
                result.extend_from_slice(&self.polynomials[i][j].to_vec());
            }
        }

        result
    }
}

#[derive(Default)]
pub struct MultiSumcheckConfigs {
    num_worker_rounds: usize,
    num_master_rounds: usize,
    num_sumchecks: usize,
}

impl MultiSumcheckConfigs {
    pub fn gen_code<F: Field>(&self, sumcheck_index: usize) -> Vec<FuncGenerator<F>> {
        let mut result = vec![];

        let mut n_worker_rounds_func =
            FuncGenerator::new("get_sumcheck__n_worker_rounds", vec!["sumcheck_index"]);
        n_worker_rounds_func.add_number(vec![sumcheck_index], self.num_worker_rounds);
        result.push(n_worker_rounds_func);

        let mut n_master_rounds_func =
            FuncGenerator::new("get_sumcheck__n_master_rounds", vec!["sumcheck_index"]);
        n_master_rounds_func.add_number(vec![sumcheck_index], self.num_master_rounds);
        result.push(n_master_rounds_func);

        let mut n_sumchecks_func =
            FuncGenerator::new("get_sumcheck__n_sumchecks", vec!["sumcheck_index"]);
        n_sumchecks_func.add_number(vec![sumcheck_index], self.num_sumchecks);
        result.push(n_sumchecks_func);

        result
    }
}

/// This function runs sumcheck of c * sum(fi(x) * gi(x))
/// (constant, evaluations_f and evaluations_g corresponding). Return all random
/// numbers r, f(r), g(r), and transcripts.
pub fn generate_sumcheck_product_transcript<F: IcicleConvertibleField, FS: FiatShamirEngine<F>>(
    fiat_shamir_engine: &mut FS,
    constant: F,
    evaluations_f: Vec<CudaSlice<F>>,
    evaluations_g: Vec<CudaSlice<F>>,
    trusted_r: &[F],
    v: bool,
) -> (Vec<F>, (Vec<F>, Vec<F>), Transcript) {
    assert_ne!(evaluations_f.len(), 0);
    assert_eq!(evaluations_f.len(), evaluations_g.len());

    let num_vars = evaluations_f[0].len().ilog2() as usize;

    if trusted_r.len() == 0 && num_vars > 0 {
        fiat_shamir_engine.begin_protocol();
    }

    let expected_evaluation_size = evaluations_f[0].len();
    assert!(
        trusted_r.len() == 0 || trusted_r.len() == num_vars,
        "Provide an invalid trusted random points (expected 0 or {}, but got {})",
        num_vars,
        trusted_r.len()
    );

    // println!(
    //     "num_vars = {}, trusted_r.len() = {}",
    //     num_vars,
    //     trusted_r.len()
    // );

    let mut sum_all = F::ZERO;
    let mut bookeeping_tables = vec![];
    let global_now = Instant::now();
    let now = Instant::now();
    for (eval_f, eval_g) in evaluations_f.into_iter().zip(evaluations_g) {
        assert_eq!(expected_evaluation_size, eval_f.len());

        let bk_table = CPUProductBookeepingTable::new(constant, eval_f, eval_g);
        sum_all += bk_table.sum_all();
        bookeeping_tables.push(bk_table);
    }
    if v {
        println!("==== compute sum: {:?}", now.elapsed());
    }

    let mut transcript = Transcript::new();
    transcript.serialize_and_push(&sum_all);

    let mut random_points = vec![];
    for round in 0..num_vars {
        let now = Instant::now();
        // Evaluate a[t] = sum(fi[r, t, b] * gi[r, t, b]) for b over boolean
        // hypercube, t = [0, 1, 2].
        let mut quadratic_poly = QuadraticPolynomial::default();
        for i in 0..bookeeping_tables.len() {
            quadratic_poly += bookeeping_tables[i].produce();
        }

        if v {
            println!("==== round {} produce value: {:?}", round, now.elapsed());
        }

        let now = Instant::now();

        // Generate random r.
        let r = if trusted_r.len() == 0 {
            fiat_shamir_engine.reduce_and_hash_to_field(&quadratic_poly.to_vec())
        } else {
            trusted_r[trusted_r.len() - round - 1]
        };

        for i in 0..bookeeping_tables.len() {
            bookeeping_tables[i].reduce(r.clone());
        }

        // Send quadratic_poly to verifer.
        // transcript.serialize_and_push(&quadratic_poly);
        transcript.serialize_and_push(&quadratic_poly);

        if v {
            println!("==== round {} fix variable: {:?}", round, now.elapsed());
        }
        random_points.push(r);
    }

    let mut last_f = vec![];
    let mut last_g = vec![];
    for i in 0..bookeeping_tables.len() {
        let last_evaluation = bookeeping_tables[i].get_last_evaluations();
        last_f.push(last_evaluation.0[0]);
        last_g.push(last_evaluation.1[0]);
    }

    random_points.reverse();
    if v {
        println!("==== GLOBAL finish: {:?}", global_now.elapsed());
    }
    (random_points, (last_f, last_g), transcript)
}

pub fn verify_product_sumcheck_transcript<F: Field, FS: FiatShamirEngine<F>>(
    fiat_shamir_engine: &mut FS,
    mut transcript: TranscriptIter,
    trusted_r: &[F],
) -> Result<(Vec<F>, F, F), Error> {
    assert!(
        trusted_r.len() == 0 || trusted_r.len() == transcript.len() - 1,
        "Provide an invalid trusted random points"
    );
    if trusted_r.len() == 0 {
        fiat_shamir_engine.begin_protocol();
    }

    let sum_all = transcript.pop_and_deserialize::<F>();

    let mut prev_g_r = sum_all.clone();
    let mut random_points = vec![];
    for round in 0..transcript.remaining_len() {
        let quadratic_poly = transcript.pop_and_deserialize::<QuadraticPolynomial<F>>();
        let f0 = quadratic_poly.evaluate(F::ZERO);
        let f1 = quadratic_poly.evaluate(F::ONE);

        if f0 + f1 != prev_g_r {
            return Err(Error::Sumcheck(format!(
                "sum of two evaluations at round {round} doesn't equal to the pre-computed sum"
            )));
        }

        let r = if trusted_r.len() == 0 {
            fiat_shamir_engine.reduce_and_hash_to_field(&quadratic_poly.to_vec())
        } else {
            trusted_r[trusted_r.len() - round - 1]
        };

        random_points.push(r);
        prev_g_r = quadratic_poly.evaluate(r);
    }

    random_points.reverse();
    Ok((random_points, prev_g_r, sum_all))
}

/// Given f1(yk1), g1(yk1), ..., fm(ykm), gm(ykm). This function runs sumcheck
/// product of:
/// constant * (F1(y)*G1(y) + F2(y)*G2(y) + ... + Fm(y)*Gm(y)).
/// Where:
/// Fi(y)*Gi(y) = yki+1*...*yl*fi(y1..yki)*gi(y1..yki).
pub fn generate_sumcheck_product_plus_transcript<
    F: IcicleConvertibleField,
    FS: FiatShamirEngine<F>,
    BK: RootProductBookeepingTable<F>,
>(
    fiat_shamir_engine: &mut FS,
    constant: F,
    evaluation_f: Vec<CudaSlice<F>>,
    evaluation_g: Vec<CudaSlice<F>>,
    trusted_r: &[F],
) -> (Vec<F>, (Vec<F>, Vec<F>), Transcript) {
    let mut bookeeping_table =
        ProductBookeepingTablePlus::<F, BK>::new(constant, evaluation_f, evaluation_g);

    let mut transcript = Transcript::default();
    transcript.serialize_and_push(&bookeeping_table.sum_all());

    let mut random_points = vec![];
    let num_rounds = bookeeping_table.num_rounds();
    for round in 0..num_rounds {
        let poly = bookeeping_table.produce();
        transcript.serialize_and_push(&poly);

        let r = if trusted_r.len() == 0 {
            fiat_shamir_engine.reduce_and_hash_to_field(&poly.to_vec())
        } else {
            trusted_r[num_rounds - round - 1]
        };

        bookeeping_table.reduce(r);
        random_points.push(r);
    }

    random_points.reverse();

    let (last_evaluations_f, last_evaluations_g) = bookeeping_table.get_last_evaluations();

    (
        random_points,
        (last_evaluations_f, last_evaluations_g),
        transcript,
    )
}

pub fn generate_synthetic_product_sumcheck_transcript<F: IcicleConvertibleField>(
    raw_transcripts: Vec<Transcript>,
    coeffs_f: &[F],                  // coefficients for f functions in each replica.
    coeffs_g: &[F],                  // coefficients for g functions in each replica.
    last_evaluations_f: Vec<Vec<F>>, // f11 - f1N, ..., ft1 - ftN (where t is num replicas, N is num functions).
    last_evaluations_g: Vec<Vec<F>>, // g11 - g1N, ..., gt1 - gtN (where t is num replicas, N is num functions).
    trusted_r: &[F],
) -> ((Vec<F>, Vec<F>), Transcript) {
    assert!(raw_transcripts.len().is_power_of_two());
    assert!(raw_transcripts.len() == last_evaluations_f.len());

    let extra_num_vars = raw_transcripts.len().ilog2() as usize;

    // This sumcheck algorithm need to be provided trusted random points.
    assert!(
        trusted_r.len() == extra_num_vars + raw_transcripts[0].len() - 1,
        "Provide invalid trusted random points"
    );
    assert!(
        coeffs_f.len() == 0 || coeffs_f.len() == raw_transcripts.len(),
        "Provide invalid coefficients"
    );
    assert!(
        coeffs_g.len() == 0 || coeffs_g.len() == raw_transcripts.len(),
        "Provide invalid coefficients"
    );

    let mut transcripts = Transcript::into_vec_iter(&raw_transcripts);

    let mut sum_all = F::ZERO;
    let sub_sum = TranscriptIter::pop_and_deserialize_vec::<F>(&mut transcripts);
    for i in 0..sub_sum.len() {
        sum_all +=
            *coeffs_f.get(i).unwrap_or(&F::ONE) * coeffs_g.get(i).unwrap_or(&F::ONE) * sub_sum[i];
    }

    // Run synthetic sumcheck for all sub-transcripts.
    let mut synthetic_transcript = Transcript::default();
    synthetic_transcript.serialize_and_push(&sum_all);

    let num_rounds = transcripts[0].remaining_len();
    let mut prev_g_r = sum_all;
    for round in 0..num_rounds {
        let mut synthetic_quadratic_poly = QuadraticPolynomial::zero();

        for (replica_index, transcript) in transcripts.iter_mut().enumerate() {
            let mut quadratic_poly = transcript.pop_and_deserialize::<QuadraticPolynomial<F>>();
            quadratic_poly.mul(coeffs_f.get(replica_index).unwrap_or(&F::ONE));
            quadratic_poly.mul(coeffs_g.get(replica_index).unwrap_or(&F::ONE));
            synthetic_quadratic_poly.add_assign(quadratic_poly);

            // If this is the last round.
            if round == num_rounds - 1 {
                let transcript_value =
                    quadratic_poly.evaluate(trusted_r[trusted_r.len() - round - 1]);

                let mut calculated_value = F::ZERO;
                for func_index in 0..last_evaluations_f[replica_index].len() {
                    calculated_value += *coeffs_f.get(replica_index).unwrap_or(&F::ONE)
                        * coeffs_g.get(replica_index).unwrap_or(&F::ONE)
                        * last_evaluations_f[replica_index][func_index]
                        * last_evaluations_g[replica_index][func_index];
                }

                assert_eq!(transcript_value, calculated_value);
            }
        }

        assert_eq!(
            synthetic_quadratic_poly.evaluate(F::ZERO) + synthetic_quadratic_poly.evaluate(F::ONE),
            prev_g_r,
            "invalid synthetic poly at round {}",
            round,
        );

        prev_g_r = synthetic_quadratic_poly.evaluate(trusted_r[trusted_r.len() - round - 1]);
        // synthetic_transcript.serialize_and_push(&synthetic_quadratic_poly);
        synthetic_transcript.serialize_and_push(&synthetic_quadratic_poly);
    }

    let mut f_evaluations = vec![];
    let mut g_evaluations = vec![];
    for func_index in 0..last_evaluations_f[0].len() {
        let mut f_eval = vec![];
        let mut g_eval = vec![];
        for replica_index in 0..transcripts.len() {
            f_eval.push(
                *coeffs_f.get(replica_index).unwrap_or(&F::ONE)
                    * last_evaluations_f[replica_index][func_index],
            );
            g_eval.push(
                *coeffs_g.get(replica_index).unwrap_or(&F::ONE)
                    * last_evaluations_g[replica_index][func_index],
            );
        }

        f_evaluations.push(CudaSlice::on_host(f_eval));
        g_evaluations.push(CudaSlice::on_host(g_eval));
    }

    // Run sumcheck for H' = sum(f_i(r)).
    // Because we provided trusted_r, so no need a fiat_shamir_engine here
    let (_, (last_f1, last_f2), extra_transcript) = generate_sumcheck_product_transcript(
        &mut DummyFiatShamirEngine::default(),
        F::ONE,
        f_evaluations,
        g_evaluations,
        &trusted_r[..extra_num_vars],
        false,
    );

    let mut extra_transcript_iter = extra_transcript.into_iter();
    extra_transcript_iter.pop(); // remove sum_all data.
    for _ in 0..extra_transcript_iter.remaining_len() {
        synthetic_transcript.push(extra_transcript_iter.pop().to_vec());
    }

    ((last_f1, last_f2), synthetic_transcript)
}

#[derive(Default)]
pub struct ProductSumcheckRandomPoints<F: Field> {
    pub r: Vec<F>,
}

#[derive(Default)]
pub struct ProductSumcheckConfigs {
    pub n_rounds: usize,
}

impl ProductSumcheckConfigs {
    pub fn gen_code<F: Field>(
        &self,
        gkr_index: usize,
        layer_index: usize,
        ext_index: usize,
        phase: usize,
    ) -> Vec<FuncGenerator<F>> {
        let mut functions = vec![];

        let mut n_rounds_func = FuncGenerator::new(
            "get_product_sumcheck__n_rounds",
            vec!["gkr_index", "layer_index", "ext_index", "phase"],
        );
        n_rounds_func.add_number(
            vec![gkr_index, layer_index, ext_index, phase],
            self.n_rounds,
        );
        functions.push(n_rounds_func);

        functions
    }
}

#[derive(Default)]
pub struct ProductSumcheckTranscript<F: Field> {
    sum_all: F,
    polynomials: Vec<QuadraticPolynomial<F>>,
}

impl<F: Field> ProductSumcheckTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        if self.polynomials.len() == 0 {
            return vec![];
        }

        let mut result = vec![self.sum_all];

        for i in 0..self.polynomials.len() {
            result.extend(self.polynomials[i].to_vec());
        }

        result
    }
}

pub fn extract_product_sumcheck_transcript<F: Field>(
    mut transcript: TranscriptIter,
) -> ProductSumcheckTranscript<F> {
    let mut circom_transcript = ProductSumcheckTranscript::default();

    let sum_all = transcript.pop_and_deserialize::<F>();
    circom_transcript.sum_all = sum_all;

    for _ in 0..transcript.remaining_len() {
        let poly = transcript.pop_and_deserialize::<QuadraticPolynomial<F>>();
        circom_transcript.polynomials.push(poly);
    }

    circom_transcript
}

pub fn calc_product_sumcheck_random_points<F: Field, FS: FiatShamirEngine<F>>(
    fiat_shamir_engine: &mut FS,
    mut transcript: TranscriptIter,
) -> ProductSumcheckRandomPoints<F> {
    let mut random_points = ProductSumcheckRandomPoints::default();

    let mut fiat_shamir_data = transcript.pop_and_deserialize::<F>();

    for _ in 0..transcript.remaining_len() {
        random_points
            .r
            .push(fiat_shamir_engine.hash_to_field(&fiat_shamir_data));
        fiat_shamir_data = fiat_shamir_engine.reduce_g(
            &transcript
                .pop_and_deserialize::<QuadraticPolynomial<F>>()
                .to_vec(),
        );
    }

    random_points
}

#[cfg(test)]
mod tests {
    use crate::{
        channel::NoChannel,
        cuda_compat::slice::CudaSlice,
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine},
        polynomial::{
            CPUProductBookeepingTable, RootProductBookeepingTable, SisuSparseMultilinearPolynomial,
        },
        sumcheck::{MultiProductSumcheckWorkerProver, MultiSumcheckVerifier},
    };

    use ark_ff::Field;
    use ark_ff::PrimeField;
    use rand::{self, Rng};

    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use sisulib::{
        codegen::generator::FileGenerator,
        common::{convert_field_to_string, convert_vec_field_to_string},
        field::FpSisu,
        mle::dense::SisuDenseMultilinearExtension,
    };

    fn big_random_poly<F: Field>(
        rng: &mut impl Rng,
        num_vars: usize,
        num_terms: usize,
    ) -> SisuSparseMultilinearPolynomial<F> {
        if num_vars == 0 {
            return SisuSparseMultilinearPolynomial::new(0, vec![]);
        }

        let mut terms = vec![];
        for _ in 0..num_terms {
            let coeff = rng.gen_range(0..100);
            let mut v = vec![];
            for j in 0..num_vars {
                let power = rng.gen_range(0..=2);
                if power >= 1 {
                    v.push(j);
                }
            }
            terms.push((F::from(coeff as u32), v));
        }

        SisuSparseMultilinearPolynomial::new(num_vars, terms)
    }

    fn big_poly<F: Field>(num_vars: usize, num_terms: usize) -> SisuSparseMultilinearPolynomial<F> {
        if num_vars == 0 {
            return SisuSparseMultilinearPolynomial::new(0, vec![]);
        }

        let mut terms = vec![];
        for i in 0..num_terms {
            let coeff = i + 1;
            let mut v = vec![];
            for j in 0..num_vars {
                let power = (i + j) % 2;
                if power >= 1 {
                    v.push(j);
                }
            }
            terms.push((F::from(coeff as u32), v));
        }

        SisuSparseMultilinearPolynomial::new(num_vars, terms)
    }

    #[test]
    fn test_sumcheck() {
        let num_vars = 15;
        let num_terms = 100;

        let mut rng = StdRng::seed_from_u64(42);

        let polynomial1 = big_random_poly(&mut rng, num_vars, num_terms);
        let polynomial2 = big_random_poly(&mut rng, num_vars, num_terms);
        let evaluation1 = polynomial1.evaluations_on_boolean_hypercube();
        let evaluation2 = polynomial2.evaluations_on_boolean_hypercube();

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::from(3u64));

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::from(3u64));

        let (_, _, transcript) = crate::sumcheck::generate_sumcheck_product_transcript(
            &mut prover_fiat_shamir,
            FpSisu::ONE,
            vec![CudaSlice::on_host(evaluation1.clone())],
            vec![CudaSlice::on_host(evaluation2.clone())],
            &[],
            false,
        );

        match crate::sumcheck::verify_product_sumcheck_transcript(
            &mut verifier_fiat_shamir,
            transcript.into_iter(),
            &[],
        ) {
            Ok((r, last_evaluation, _)) => {
                let x = polynomial1.evaluate(&r);
                let y = polynomial2.evaluate(&r);

                let ext1 = SisuDenseMultilinearExtension::from_slice(&evaluation1);
                let ext2 = SisuDenseMultilinearExtension::from_slice(&evaluation2);

                assert_eq!(x * y, last_evaluation);
                assert_eq!(
                    ext1.evaluate(vec![&r]) * ext2.evaluate(vec![&r]),
                    last_evaluation
                );
            }
            Err(e) => panic!("{e}"),
        };
    }

    #[test]
    fn test_fixed_sumcheck() {
        let num_vars = 15;
        let num_terms = 100;

        let polynomial1 = big_poly(num_vars, num_terms);
        let polynomial2 = big_poly(num_vars, num_terms);
        let evaluation1 = polynomial1.evaluations_on_boolean_hypercube();
        let evaluation2 = polynomial2.evaluations_on_boolean_hypercube();

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::from(3));

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::from(3));

        let mut sumcheck_prover =
            MultiProductSumcheckWorkerProver::<_, _, NoChannel, NoChannel>::default();
        sumcheck_prover.add(Some(CPUProductBookeepingTable::new(
            FpSisu::ONE,
            CudaSlice::on_host(evaluation1.clone()),
            CudaSlice::on_host(evaluation2.clone()),
        )));
        let (_, _, _, transcript) = sumcheck_prover.run(&mut prover_fiat_shamir);

        let sumcheck_verifier = MultiSumcheckVerifier::new(1, 0, num_vars);
        match sumcheck_verifier.verify(&mut verifier_fiat_shamir, transcript.into_iter()) {
            Ok((r, final_value, sum_all)) => {
                println!("RANDOM_POINTS: {:?}", convert_vec_field_to_string(&r));
                println!("LAST_G_R: {:?}", convert_field_to_string(&final_value));
                println!("SUM_ALL: {:?}", convert_field_to_string(&sum_all));

                let x = polynomial1.evaluate(&r);
                let y = polynomial2.evaluate(&r);

                let ext1 = SisuDenseMultilinearExtension::from_slice(&evaluation1);
                let ext2 = SisuDenseMultilinearExtension::from_slice(&evaluation2);

                assert_eq!(x * y, final_value);
                assert_eq!(
                    ext1.evaluate(vec![&r]) * ext2.evaluate(vec![&r]),
                    final_value
                );

                let mut random_points = vec![];
                for i in (0..r.len()).rev() {
                    random_points.push(r[i].into_bigint().0[0].to_string());
                }
            }
            Err(e) => panic!("{e}"),
        };

        let configs = sumcheck_verifier.configs();
        let mut file_gen =
            FileGenerator::<FpSisu>::new("../bls-circom/circuit/sisu/configs.gen.circom");
        file_gen.extend_funcs(configs.gen_code(0));
        file_gen.create();

        let sumcheck_transcript = sumcheck_verifier.extract_transcript(transcript.into_iter());
        println!(
            "{:?}",
            convert_vec_field_to_string(&sumcheck_transcript.to_vec())
        );
    }

    #[test]
    fn test_synthetic_sumcheck() {
        let num_vars = 2;
        let num_terms = 3;
        let num_replicas = 8;

        let mut rng = StdRng::seed_from_u64(42);

        let mut polynomials1 = vec![];
        let mut polynomials2 = vec![];
        for _ in 0..num_replicas {
            polynomials1.push(big_random_poly(&mut rng, num_vars, num_terms));
            polynomials2.push(big_random_poly(&mut rng, num_vars, num_terms));
        }

        let mut evaluations1 = vec![];
        let mut evaluations2 = vec![];
        for i in 0..num_replicas {
            evaluations1.push(polynomials1[i].evaluations_on_boolean_hypercube());
            evaluations2.push(polynomials2[i].evaluations_on_boolean_hypercube());
        }

        let mut sub_transcripts = vec![];
        let mut last_evaluations_f1 = vec![];
        let mut last_evaluations_f2 = vec![];

        let mut fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        let trusted_r = fiat_shamir.hash_to_fields(
            &FpSisu::ZERO,
            (evaluations1.len().ilog2() + evaluations1[0].len().ilog2()) as usize,
        );
        for i in 0..evaluations1.len() {
            let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
            let (_, (last_f1, last_f2), transcript) =
                crate::sumcheck::generate_sumcheck_product_transcript(
                    &mut prover_fiat_shamir,
                    FpSisu::ONE,
                    vec![CudaSlice::on_host(evaluations1[i].clone())],
                    vec![CudaSlice::on_host(evaluations2[i].clone())],
                    &trusted_r[evaluations1.len().ilog2() as usize..],
                    true,
                );

            sub_transcripts.push(transcript);
            last_evaluations_f1.push(last_f1);
            last_evaluations_f2.push(last_f2);
        }

        let (_, transcript) = crate::sumcheck::generate_synthetic_product_sumcheck_transcript(
            sub_transcripts,
            &[],
            &[],
            last_evaluations_f1,
            last_evaluations_f2,
            &trusted_r,
        );

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        match crate::sumcheck::verify_product_sumcheck_transcript(
            &mut verifier_fiat_shamir,
            transcript.into_iter(),
            &trusted_r,
        ) {
            Ok((r, last_point, sum)) => {
                let mut expected_sum = FpSisu::from(0);
                for i in 0..evaluations1.len() {
                    for j in 0..evaluations1[0].len() {
                        expected_sum += evaluations1[i][j] * evaluations2[i][j];
                    }
                }

                assert_eq!(expected_sum, sum);

                let mut synthetic_e1 = vec![];
                let mut synthetic_e2 = vec![];
                for i in 0..evaluations1.len() {
                    synthetic_e1.extend_from_slice(&evaluations1[i]);
                    synthetic_e2.extend_from_slice(&evaluations2[i]);
                }

                let ext1 = SisuDenseMultilinearExtension::from_slice(&synthetic_e1);

                let ext2 = SisuDenseMultilinearExtension::from_slice(&synthetic_e2);

                assert_eq!(
                    ext1.evaluate(vec![&r]) * ext2.evaluate(vec![&r]),
                    last_point
                );
            }
            Err(e) => panic!("{e}"),
        };
    }

    #[test]
    fn test_sumcheck_with_provided_random() {
        let mut rng = StdRng::seed_from_u64(42);

        let polynomial = big_random_poly(&mut rng, 15, 100);
        let evaluation = polynomial.evaluations_on_boolean_hypercube();

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);

        let (_, _, transcript) = crate::sumcheck::generate_sumcheck_product_transcript(
            &mut prover_fiat_shamir,
            FpSisu::ONE,
            vec![CudaSlice::on_host(evaluation.clone())],
            vec![CudaSlice::on_host(evaluation)],
            &[],
            true,
        );

        match crate::sumcheck::verify_product_sumcheck_transcript(
            &mut verifier_fiat_shamir,
            transcript.into_iter(),
            &[],
        ) {
            Ok((r, sum, _)) => {
                let x = polynomial.evaluate(&r);
                assert_eq!(x * x, sum);
            }
            Err(e) => panic!("{e}"),
        };
    }

    #[test]
    fn test_sumcheck_plus() {
        let total_num_vars = 3;
        let num_polynomials = 3;

        let mut rng = StdRng::seed_from_u64(42);
        let mut poly_f = vec![];
        let mut poly_g = vec![];
        for i in 0..num_polynomials {
            let num_vars = if i == 0 {
                total_num_vars
            } else {
                rand::thread_rng().gen_range(0..=total_num_vars)
            };

            poly_f.push(big_random_poly(&mut rng, num_vars, 2));
            poly_g.push(big_random_poly(&mut rng, num_vars, 2));
        }

        let mut eval_f = vec![];
        let mut eval_g = vec![];
        let mut func_eval_f = vec![];
        let mut func_eval_g = vec![];
        for i in 0..num_polynomials {
            eval_f.push(poly_f[i].evaluations_on_boolean_hypercube());
            eval_g.push(poly_g[i].evaluations_on_boolean_hypercube());

            func_eval_f.push(CudaSlice::on_host(eval_f[eval_f.len() - 1].clone()));
            func_eval_g.push(CudaSlice::on_host(eval_g[eval_g.len() - 1].clone()));
        }

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);

        let (_, (last_f, last_g), transcript) =
            crate::sumcheck::generate_sumcheck_product_plus_transcript::<
                _,
                _,
                CPUProductBookeepingTable<_>,
            >(
                &mut prover_fiat_shamir,
                FpSisu::ONE,
                func_eval_f,
                func_eval_g,
                &[],
            );
        match crate::sumcheck::verify_product_sumcheck_transcript(
            &mut verifier_fiat_shamir,
            transcript.into_iter(),
            &[],
        ) {
            Ok((r, value_at_r, sum_on_boolean_hypercube)) => {
                let mut expected_value_at_r = FpSisu::from(0);
                for i in 0..poly_f.len() {
                    let x = poly_f[i].evaluate(&r[..poly_f[i].num_vars])
                        * poly_g[i].evaluate(&r[..poly_g[i].num_vars]);

                    let mut y = FpSisu::from(1);
                    for j in poly_f[i].num_vars..total_num_vars {
                        y *= r[j];
                    }

                    expected_value_at_r += x * y;
                }

                let mut expected_last_value = FpSisu::from(0);
                for i in 0..last_f.len() {
                    expected_last_value += last_f[i] * last_g[i];
                }

                let mut expected_sum = FpSisu::from(0);
                for i in 0..eval_f.len() {
                    for j in 0..eval_f[i].len() {
                        expected_sum += eval_f[i][j] * eval_g[i][j];
                    }
                }

                assert_eq!(expected_last_value, value_at_r);
                assert_eq!(expected_value_at_r, value_at_r);
                assert_eq!(expected_sum, sum_on_boolean_hypercube);
            }
            Err(e) => panic!("{e}"),
        };
    }

    #[test]
    fn test_synthetic_sumcheck_plus() {
        let total_num_vars = 6;
        let num_polynomials_per_replica = 3;
        let num_replicas = 8;
        let mut rng = StdRng::seed_from_u64(42);

        let mut poly_f = vec![vec![]; num_replicas];
        let mut poly_g = vec![vec![]; num_replicas];

        let mut eval_f = vec![vec![]; num_replicas];
        let mut eval_g = vec![vec![]; num_replicas];
        let mut func_eval_f = vec![vec![]; num_replicas];
        let mut func_eval_g = vec![vec![]; num_replicas];

        let mut poly_num_vars = vec![total_num_vars];
        for _ in 1..num_polynomials_per_replica {
            poly_num_vars.push(rand::thread_rng().gen_range(0..=total_num_vars));
        }

        let mut coeffs = vec![];
        for _ in 0..num_replicas {
            coeffs.push(FpSisu::from(rand::thread_rng().gen_range(0..100) as u64));
        }

        for replica_index in 0..num_replicas {
            for poly_index in 0..num_polynomials_per_replica {
                poly_f[replica_index].push(big_random_poly(&mut rng, poly_num_vars[poly_index], 2));
                poly_g[replica_index].push(big_random_poly(&mut rng, poly_num_vars[poly_index], 2));

                eval_f[replica_index]
                    .push(poly_f[replica_index][poly_index].evaluations_on_boolean_hypercube());
                eval_g[replica_index]
                    .push(poly_g[replica_index][poly_index].evaluations_on_boolean_hypercube());

                func_eval_f[replica_index].push(CudaSlice::on_host(
                    eval_f[replica_index][poly_index].clone(),
                ));
                func_eval_g[replica_index].push(CudaSlice::on_host(
                    eval_g[replica_index][poly_index].clone(),
                ));
            }
        }

        let mut fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        let trusted_r = fiat_shamir.hash_to_fields(
            &FpSisu::ZERO,
            total_num_vars + num_replicas.ilog2() as usize,
        );

        let mut sub_transcripts = vec![];
        let mut last_evaluations_f = vec![];
        let mut last_evaluations_g = vec![];

        for replica_index in 0..num_replicas {
            let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
            let (_, (last_f, last_g), transcript) =
                crate::sumcheck::generate_sumcheck_product_plus_transcript::<
                    _,
                    _,
                    CPUProductBookeepingTable<_>,
                >(
                    &mut prover_fiat_shamir,
                    FpSisu::ONE,
                    func_eval_f[replica_index].clone(),
                    func_eval_g[replica_index].clone(),
                    &trusted_r[num_replicas.ilog2() as usize..],
                );

            sub_transcripts.push(transcript);
            last_evaluations_f.push(last_f);
            last_evaluations_g.push(last_g);
        }

        let (_, transcript) = crate::sumcheck::generate_synthetic_product_sumcheck_transcript(
            sub_transcripts,
            &coeffs,
            &[],
            last_evaluations_f,
            last_evaluations_g,
            &trusted_r,
        );

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_sha256();
        match crate::sumcheck::verify_product_sumcheck_transcript(
            &mut verifier_fiat_shamir,
            transcript.into_iter(),
            &trusted_r,
        ) {
            Ok((r, value_at_r, sum_on_boolean_hypercube)) => {
                let mut expected_value_at_r = FpSisu::from(0);
                let extra_num_vars = num_replicas.ilog2() as usize;

                let mut synthetic_eval_f = vec![];
                let mut synthetic_eval_g = vec![];
                for func_index in 0..eval_f[0].len() {
                    let mut eval_f_i = vec![];
                    let mut eval_g_i = vec![];
                    for replica_index in 0..num_replicas {
                        for evaluation_index in 0..eval_f[replica_index][func_index].len() {
                            eval_f_i.push(
                                coeffs[replica_index]
                                    * eval_f[replica_index][func_index][evaluation_index],
                            );
                            eval_g_i.push(eval_g[replica_index][func_index][evaluation_index]);
                        }
                    }

                    synthetic_eval_f.push(eval_f_i);
                    synthetic_eval_g.push(eval_g_i);
                }

                let mut ext_f = vec![];
                let mut ext_g = vec![];
                for func_index in 0..eval_f[0].len() {
                    ext_f.push(SisuDenseMultilinearExtension::from_slice(
                        &synthetic_eval_f[func_index],
                    ));

                    ext_g.push(SisuDenseMultilinearExtension::from_slice(
                        &synthetic_eval_g[func_index],
                    ));
                }

                for func_index in 0..poly_f[0].len() {
                    let num_vars = poly_f[0][func_index].num_vars + extra_num_vars;
                    let x = ext_f[func_index].evaluate(vec![&r[..num_vars]])
                        * ext_g[func_index].evaluate(vec![&r[..num_vars]]);

                    let mut y = FpSisu::from(1);
                    for j in num_vars..total_num_vars + extra_num_vars {
                        y *= r[j];
                    }

                    expected_value_at_r += x * y;
                }

                let mut expected_sum = FpSisu::from(0);
                for replica_index in 0..num_replicas {
                    for func_index in 0..eval_f[replica_index].len() {
                        for evaluation_index in 0..eval_f[replica_index][func_index].len() {
                            expected_sum += coeffs[replica_index]
                                * eval_f[replica_index][func_index][evaluation_index]
                                * eval_g[replica_index][func_index][evaluation_index];
                        }
                    }
                }

                assert_eq!(expected_sum, sum_on_boolean_hypercube);
                assert_eq!(expected_value_at_r, value_at_r);
            }
            Err(e) => panic!("{e}"),
        };
    }
}
