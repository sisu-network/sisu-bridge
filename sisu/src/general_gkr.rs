use std::{cmp::max, marker::PhantomData, time::Instant};

use ark_ff::Field;
use ark_serialize::CanonicalSerialize;
use sisulib::{
    circuit::{
        general_circuit::{circuit::GeneralCircuit, gate::LayerIndex},
        CircuitParams,
    },
    codegen::generator::FuncGenerator,
    common::{padding_pow_of_two_size, round_to_pow_of_two, Error},
    mle::dense::{
        identity_mle, identity_mle_vec, SisuDenseMajorZeroMultilinearExtension,
        SisuDenseMultilinearExtension,
    },
};

use crate::{
    channel::{SisuReceiver, SisuSender, WorkerNode},
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    general_gkr_sumcheck::{
        GeneralGKRSumcheckNextRoundWorkerProver, GeneralGKRSumcheckPhaseOneWorkerProver,
        GeneralGKRSumcheckPhaseTwoWorkerProver,
    },
    icicle_converter::IcicleConvertibleField,
    sisu_engine::{GateExtensionType, SisuEngine},
    sumcheck::{MultiSumcheckConfigs, MultiSumcheckTranscript, MultiSumcheckVerifier},
};

#[derive(Default)]
struct GeneralProverGKRParams<F: IcicleConvertibleField> {
    random_points: Vec<Vec<CudaSlice<F>>>,
}

impl<F: IcicleConvertibleField> GeneralProverGKRParams<F> {
    pub fn new(total_layers: usize) -> Self {
        let mut params = Self {
            random_points: vec![vec![]; total_layers + 1],
        };

        for i in 1..params.random_points.len() {
            params.random_points[i] = vec![CudaSlice::on_host(vec![]); i + 1];
        }

        params
    }

    pub fn add_random_point(
        &mut self,
        source_layer_index: usize,
        target_layer_index: usize,
        random_point: CudaSlice<F>,
    ) {
        self.random_points[target_layer_index][source_layer_index] = random_point;
    }

    pub fn add_last_random_point(&mut self, target_layer_index: usize, random_point: CudaSlice<F>) {
        self.random_points[target_layer_index][target_layer_index] = random_point;
    }

    pub fn get_random_points_of(&self, target_layer_index: usize) -> &[CudaSlice<F>] {
        &self.random_points[target_layer_index]
    }
}

#[derive(Default)]
struct GeneralVerifierGKRParams<F: Field> {
    random_points: Vec<Vec<Vec<F>>>,
    claims: Vec<Vec<F>>,
}

impl<F: Field> GeneralVerifierGKRParams<F> {
    pub fn new(total_layers: usize) -> Self {
        let mut params = Self {
            random_points: vec![vec![]; total_layers + 1],
            claims: vec![vec![]; total_layers + 1],
        };

        for i in 1..params.claims.len() {
            params.random_points[i] = vec![vec![]; i + 1];
            params.claims[i] = vec![F::ZERO; i + 1];
        }

        params
    }

    pub fn add_claim(
        &mut self,
        source_layer_index: usize,
        target_layer_index: usize,
        r: Vec<F>,
        claim: F,
    ) {
        self.random_points[target_layer_index][source_layer_index] = r;
        self.claims[target_layer_index][source_layer_index] = claim;
    }

    pub fn add_last_claim(&mut self, target_layer_index: usize, r: Vec<F>, claim: F) {
        self.random_points[target_layer_index][target_layer_index] = r;
        self.claims[target_layer_index][target_layer_index] = claim;
    }

    pub fn get_claims_of(&self, target_layer_index: usize) -> &[F] {
        &self.claims[target_layer_index]
    }

    pub fn get_random_points_of(&self, target_layer_index: usize) -> &[Vec<F>] {
        &self.random_points[target_layer_index]
    }
}

pub struct GeneralGKRProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    engine: &'a Engine,
    circuit: &'a GeneralCircuit<F>,
    num_replicas: usize,

    num_workers: usize,
    worker_index: usize,

    worker: Option<&'a WorkerNode<S, R>>,

    __phantom: PhantomData<Engine>,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    GeneralGKRProver<'a, F, Engine, S, R>
{
    pub fn new(engine: &'a Engine, circuit: &'a GeneralCircuit<F>, num_replicas: usize) -> Self {
        assert!(num_replicas.is_power_of_two());

        Self {
            engine,
            circuit,
            num_replicas,
            num_workers: 1,
            worker_index: 0,
            worker: None,
            __phantom: PhantomData,
        }
    }

    pub fn is_worker(&self) -> bool {
        self.worker.is_some()
    }

    pub fn total_num_vars(&self, layer_index: usize) -> usize {
        self.total_extra_num_vars() + self.circuit_num_vars(layer_index)
    }

    pub fn total_extra_num_vars(&self) -> usize {
        self.worker_extra_num_vars() + self.replica_extra_num_vars()
    }

    pub fn worker_extra_num_vars(&self) -> usize {
        self.num_workers.ilog2() as usize
    }

    pub fn replica_extra_num_vars(&self) -> usize {
        self.num_replicas.ilog2() as usize
    }

    pub fn circuit_num_vars(&self, layer_index: usize) -> usize {
        self.circuit.num_vars_at(layer_index)
    }

    pub fn to_worker(
        &mut self,
        worker_node: &'a WorkerNode<S, R>,
        num_workers: usize,
        worker_index: usize,
    ) {
        assert!(num_workers.is_power_of_two());
        assert!(worker_index < num_workers);

        self.num_workers = num_workers;
        self.worker_index = worker_index;

        self.worker = Some(worker_node);
    }

    pub fn get_worker(&self) -> &WorkerNode<S, R> {
        self.worker.unwrap()
    }

    fn send_to_master_and_done_if_any<T: CanonicalSerialize>(&self, obj: &T) {
        if self.worker.is_some() {
            self.worker.unwrap().send_to_master_and_done(obj).unwrap();
        }
    }

    /// Run a sumcheck on V(g) = f(g, x, y).
    fn run_sumcheck_phase_1<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        wx_next_evaluations: CudaSlice<F>,
        wy_next_evaluations: Vec<CudaSlice<F>>,
        bookeeping_identity_g: CudaSlice<F>,
        verbosity: bool,
    ) -> (CudaSlice<F>, Vec<F>, Option<F>, Transcript) {
        // V(g) =
        //   constant
        // + add * (V(x) + V(y))
        // + mul * V(x) * V(y)
        // + forward * V(y).
        let now = Instant::now();
        let mut phase_1_sumcheck_prover = GeneralGKRSumcheckPhaseOneWorkerProver::new(
            self.engine,
            self.circuit,
            round,
            self.worker,
            bookeeping_identity_g,
            self.num_replicas,
            wx_next_evaluations,
            wy_next_evaluations,
        );
        if verbosity {
            println!("[General GKR]: Setup phase 1: {:?}", now.elapsed(),);
        }

        let now = Instant::now();
        // GATE CONSTANT: constant(g, x, y) * constant_1(x) * constant1(y)
        phase_1_sumcheck_prover.add(GateExtensionType::Constant, false);

        // GATE MUL: mul(g, x, y) * V(x) * V(y).
        phase_1_sumcheck_prover.add(GateExtensionType::Mul, true);

        // GATE FORWARD X: forward(g, x, y) * V(x) * V_constant_1(y).
        phase_1_sumcheck_prover.add(GateExtensionType::ForwardX, true);

        // GATE FORWARD Y: forward(g, x, y) * V_constant_1(x) * V(y).
        phase_1_sumcheck_prover.add(GateExtensionType::ForwardY, false);
        if verbosity {
            println!("[General GKR]: Prepare phase 1: {:?}", now.elapsed());
        }

        let now = Instant::now();
        let (random_u, t_at_u, wx, transcript) = phase_1_sumcheck_prover.run(fiat_shamir_engine);
        if verbosity {
            println!("[General GKR]: Run phase 1: {:?}", now.elapsed());
        }

        (CudaSlice::on_host(random_u), t_at_u, wx, transcript)
    }

    /// Run a sumcheck on V(g) = f(g, x, y).
    fn run_sumcheck_phase_2<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        wy_next_evaluations: Vec<CudaSlice<F>>,
        bookeeping_identity_g: CudaSlice<F>,
        random_u: &mut CudaSlice<F>,
        t_at_u: Vec<F>,
        verbosity: bool,
    ) -> (CudaSlice<F>, Option<Vec<F>>, Transcript) {
        // V(g) =
        //   constant
        // + add * (V(x) + V(y))
        // + mul * V(x) * V(y)
        // + forward * V(y).

        let now = Instant::now();
        let mut phase_2_sumcheck_prover = GeneralGKRSumcheckPhaseTwoWorkerProver::new(
            self.engine,
            self.worker,
            self.circuit,
            round,
            bookeeping_identity_g,
            random_u,
            t_at_u,
            self.num_replicas,
            wy_next_evaluations,
        );
        if verbosity {
            println!("[General GKR]: Setup phase 2: {:?}", now.elapsed());
        }

        let now = Instant::now();
        // GATE CONSTANT: constant(g, x, y) * constant_1(x) * constant1(y)
        phase_2_sumcheck_prover.add(GateExtensionType::Constant, false);

        // GATE MUL: mul(g, x, y) * V(x) * V(y).
        phase_2_sumcheck_prover.add(GateExtensionType::Mul, true);

        // GATE FORWARD X: forward(g, x, y) * V(x) * V_constant_1(y).
        phase_2_sumcheck_prover.add(GateExtensionType::ForwardX, false);

        // GATE FORWARD Y: forward(g, x, y) * V_constant_1(x) * V(y).
        phase_2_sumcheck_prover.add(GateExtensionType::ForwardY, true);
        if verbosity {
            println!("[General GKR]: Prepare phase 2: {:?}", now.elapsed());
        }

        let now = Instant::now();
        let (random_v, _, wy, transcript) = phase_2_sumcheck_prover.run(fiat_shamir_engine);
        if verbosity {
            println!("[General GKR]: Run phase 2: {:?}", now.elapsed());
        }

        (CudaSlice::on_host(random_v), wy, transcript)
    }

    /// Run a sumcheck on V(g) = f(g, x, y).
    fn run_sumcheck<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        wx_next_evaluations: CudaSlice<F>,
        wy_next_evaluations: Vec<CudaSlice<F>>,
        bookeeping_identity_g: CudaSlice<F>,
        w_g: F,
        verbosity: bool,
    ) -> (
        CudaSlice<F>,
        CudaSlice<F>,
        Option<F>,
        Option<Vec<F>>,
        Transcript,
    ) {
        fiat_shamir_engine.set_seed(w_g);
        let (random_u, t_at_u, w_u, phase_1_transcript) = self.run_sumcheck_phase_1(
            fiat_shamir_engine,
            round,
            wx_next_evaluations,
            wy_next_evaluations.clone(),
            bookeeping_identity_g.clone(),
            verbosity,
        );

        fiat_shamir_engine.set_seed(w_g);
        let (random_v, w_v, phase_2_transcript) = self.run_sumcheck_phase_2(
            fiat_shamir_engine,
            round,
            wy_next_evaluations,
            bookeeping_identity_g,
            &mut random_u.at_range_from(self.worker_extra_num_vars()..),
            t_at_u,
            verbosity,
        );

        let mut transcript = Transcript::default();

        transcript.serialize_and_push(&phase_1_transcript);
        transcript.serialize_and_push(&phase_2_transcript);

        (random_u, random_v, w_u, w_v, transcript)
    }

    fn generate_round<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        alpha: &[F],
        gkr_params: &mut GeneralProverGKRParams<F>,
        mut wx_next_evaluations: CudaSlice<F>,
        mut wy_next_evaluations: Vec<CudaSlice<F>>,
        random_g: &CudaSlice<F>,
        w_g: F,
    ) -> (CudaSlice<F>, F, Transcript) {
        let v_round = 0;

        let now = Instant::now();

        if round == v_round {
            println!("[General GKR]: Prepare wx, wy: {:?}", now.elapsed());
        }

        let now = Instant::now();
        let bookeeping_identity_g = Engine::precompute_bookeeping(
            F::ONE,
            &mut random_g.at_range_from(self.worker_extra_num_vars()..),
        );
        if round == v_round {
            println!(
                "[General GKR]: precompute bookeeping g: {:?}",
                now.elapsed()
            );
        }

        let now = Instant::now();
        let (random_u, random_v, w_u, w_v, round_transcript) = self.run_sumcheck(
            fiat_shamir_engine,
            round,
            wx_next_evaluations.clone(),
            wy_next_evaluations.clone(),
            bookeeping_identity_g,
            w_g,
            round == v_round,
        );
        if round == v_round {
            println!("[General GKR]: Total run sumcheck: {:?}", now.elapsed());
        }

        let now = Instant::now();
        let w_u = match w_u {
            Some(value) => value,
            None => {
                let wx_mle =
                    SisuDenseMultilinearExtension::from_slice(wx_next_evaluations.as_ref_host());
                wx_mle.evaluate(vec![&random_u
                    .at_range_from(self.worker_extra_num_vars()..)
                    .as_ref_host()])
            }
        };

        let w_v = match w_v {
            Some(values) => values,
            None => {
                // panic!("invalid w_v");
                let mut w_v = vec![];
                for i in 0..wy_next_evaluations.len() {
                    let y_num_vars = self
                        .circuit
                        .layer(round)
                        .constant_ext
                        .subset_num_vars_at(&LayerIndex::Relative(i + 1));

                    let mut random_v_i = random_v.at_range(
                        self.worker_extra_num_vars()..self.total_extra_num_vars() + y_num_vars,
                    );
                    let wy_mle = SisuDenseMultilinearExtension::from_slice(
                        &wy_next_evaluations[i].as_ref_host(),
                    );
                    w_v.push(wy_mle.evaluate(vec![random_v_i.as_ref_host()]));
                }

                w_v
            }
        };
        self.send_to_master_and_done_if_any(&w_v);
        self.send_to_master_and_done_if_any(&w_u);

        for i in 0..self.circuit.layer(round).constant_ext.as_slice().len() {
            let y_num_vars = self
                .circuit
                .layer(round)
                .constant_ext
                .subset_num_vars_at(&LayerIndex::Relative(i + 1));

            let full_random_v_i = random_v.at_range_to(..self.total_extra_num_vars() + y_num_vars);
            gkr_params.add_random_point(round, round + i + 1, full_random_v_i);
        }
        gkr_params.add_last_random_point(round + 1, random_u);
        if round == v_round {
            println!("[General GKR]: get all claims: {:?}", now.elapsed());
        }
        let now = Instant::now();

        let bookeeping_qx = self.engine.initialize_combining_point(
            self.circuit,
            self.worker_index,
            self.num_replicas,
            self.num_workers,
            round + 1,
            gkr_params.get_random_points_of(round + 1),
            &alpha,
        );
        if round == v_round {
            println!(
                "[General GKR]: initialize combining point: {:?}",
                now.elapsed()
            );
        }
        let now = Instant::now();

        let mut combining_sumcheck = GeneralGKRSumcheckNextRoundWorkerProver::<
            _,
            Engine::RootProductBookeepingTable,
            _,
            _,
        >::new(self.worker);

        fiat_shamir_engine.set_seed(w_g);
        let (random_next_g, w_at_next_g, combining_transcript) =
            combining_sumcheck.run(fiat_shamir_engine, bookeeping_qx, wx_next_evaluations);

        if round == v_round {
            println!(
                "[General GKR]: sumcheck combining point: {:?}",
                now.elapsed()
            );
        }
        let now = Instant::now();

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&round_transcript);
        transcript.serialize_and_push(&w_u);
        transcript.serialize_and_push(&w_v);
        transcript.serialize_and_push(&combining_transcript);

        if round == v_round {
            println!("[General GKR]: finalize: {:?}", now.elapsed());
        }

        (CudaSlice::on_host(random_next_g), w_at_next_g, transcript)
    }

    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        witness: &mut [CudaSlice<F>],
        num_non_zero_outputs: Option<usize>,
    ) -> (CudaSlice<F>, F, Transcript) {
        fiat_shamir_engine.begin_protocol();

        let num_non_zero_outputs =
            num_non_zero_outputs.unwrap_or(round_to_pow_of_two(self.circuit.len_at(0)));

        let now = Instant::now();

        let circuit_evaluations = self.engine.evaluate_circuit(&self.circuit, witness);

        let mut all_non_zero_outputs = vec![];
        for subcircuit_index in 0..witness.len() {
            let start = subcircuit_index * round_to_pow_of_two(self.circuit.len_at(0));
            all_non_zero_outputs.push(
                circuit_evaluations
                    .w_evaluation(0)
                    .at_range(start..start + num_non_zero_outputs)
                    .as_host(),
            );
        }

        let (mut evaluations, subset_evaluations) = circuit_evaluations.detach();

        // Remove output from evaluations.
        let mut new_evaluations = vec![];
        for (i, eval) in evaluations.into_iter().enumerate() {
            if i > 0 {
                new_evaluations.push(eval);
            }
        }
        evaluations = new_evaluations;

        println!("[General GKR]: run circuit: {:?}", now.elapsed());

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&all_non_zero_outputs);

        // Compute W_output(g).
        let mut random_g =
            CudaSlice::on_host(fiat_shamir_engine.hash_to_fields(&F::ZERO, self.total_num_vars(0)));

        let output_ext = SisuDenseMajorZeroMultilinearExtension::from_groups(
            all_non_zero_outputs.clone(),
            round_to_pow_of_two(self.circuit.len_at(0)),
        );
        let worker_w_g = output_ext.evaluate(vec![random_g
            .at_range_from(self.worker_extra_num_vars()..)
            .as_ref_host()]);

        let mut w_g = if self.is_worker() {
            self.get_worker().send_to_master(&worker_w_g).unwrap();
            self.get_worker().recv_from_master().unwrap()
        } else {
            worker_w_g
        };

        if self.is_worker() {
            self.get_worker()
                .send_to_master_and_done(&all_non_zero_outputs)
                .unwrap();
        }

        let mut round_transcript: Transcript;
        let mut gkr_params = GeneralProverGKRParams::new(self.circuit.len());
        for (round, (eval, subset_eval)) in evaluations
            .into_iter()
            .zip(subset_evaluations.into_iter())
            .enumerate()
        {
            let now = Instant::now();
            let alpha = fiat_shamir_engine.hash_to_fields(&w_g, round + 2);
            (random_g, w_g, round_transcript) = self.generate_round(
                fiat_shamir_engine,
                round,
                &alpha,
                &mut gkr_params,
                eval,
                subset_eval,
                &random_g,
                w_g,
            );
            transcript.serialize_and_push(&round_transcript);
            println!(
                "[General GKR]: round {} ({}): {:?}",
                round,
                self.circuit.len_at(round),
                now.elapsed()
            );
        }

        // Return the last random g and the transcript
        (random_g, w_g, transcript)
    }
}

pub struct GeneralGKRVerifier<'a, F: Field> {
    circuit: &'a GeneralCircuit<F>,
    master_extra_num_vars: usize,
    worker_extra_num_vars: usize,
}

impl<'a, F: Field> GeneralGKRVerifier<'a, F> {
    pub fn new(circuit: &'a GeneralCircuit<F>) -> Self {
        Self {
            circuit,
            master_extra_num_vars: 0,
            worker_extra_num_vars: 0,
        }
    }

    pub fn replicate(&mut self, num_replicas_per_worker: usize, num_workers: usize) {
        assert!(num_replicas_per_worker >= 1 && num_replicas_per_worker.is_power_of_two());
        self.worker_extra_num_vars = num_replicas_per_worker.ilog2() as usize;

        assert!(num_workers >= 1 && num_workers.is_power_of_two());
        self.master_extra_num_vars = num_workers.ilog2() as usize;
    }

    pub fn total_extra_num_vars(&self) -> usize {
        self.worker_extra_num_vars + self.master_extra_num_vars
    }

    fn verify_sumcheck<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        mut transcript: TranscriptIter,
        round: usize,
        w_g: F,
    ) -> Result<(Vec<F>, Vec<F>, F, F), Error> {
        let mut num_sumchecks = 0;

        if self.circuit.layer(round).constant_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).mul_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).forward_x_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).forward_y_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        let x_num_vars = self.circuit.num_vars_at(round + 1);
        let mut y_num_vars = 0;
        for target_layer_index in 1..=self.circuit.len() - round {
            let target_layer_index = LayerIndex::Relative(target_layer_index);

            y_num_vars = max(
                y_num_vars,
                self.circuit
                    .layer(round)
                    .mul_ext
                    .subset_num_vars_at(&target_layer_index),
            );
        }

        let sumcheck_phase_1_transcript = transcript.pop_and_deserialize::<Transcript>();
        let sumcheck_phase_1_verifier = MultiSumcheckVerifier::new(
            num_sumchecks,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + x_num_vars,
        );

        fiat_shamir_engine.set_seed(w_g);
        let (random_u, _, sum_over_boolean_hypercube) = sumcheck_phase_1_verifier
            .verify(fiat_shamir_engine, sumcheck_phase_1_transcript.into_iter())?;

        let sumcheck_phase_2_transcript = transcript.pop_and_deserialize::<Transcript>();
        let sumcheck_phase_2_verifier = MultiSumcheckVerifier::new(
            num_sumchecks,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + y_num_vars,
        );

        fiat_shamir_engine.set_seed(w_g);
        let (random_v, final_value, _) = sumcheck_phase_2_verifier
            .verify(fiat_shamir_engine, sumcheck_phase_2_transcript.into_iter())?;

        Ok((random_u, random_v, final_value, sum_over_boolean_hypercube))
    }

    fn verify_round<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        mut transcript: TranscriptIter,
        round: usize,
        random_g: &[F],
        w_g: F,
        alpha: &[F],
        gkr_params: &mut GeneralVerifierGKRParams<F>,
    ) -> Result<(Vec<F>, F), Error> {
        let sumcheck_transcript = transcript.pop_and_deserialize::<Transcript>();
        let w_u = transcript.pop_and_deserialize::<F>();
        let w_v = transcript.pop_and_deserialize::<Vec<F>>();
        let combining_transcript = transcript.pop_and_deserialize::<Transcript>();

        let (random_u, random_v, final_value, sum_all) = self.verify_sumcheck(
            fiat_shamir_engine,
            sumcheck_transcript.into_iter(),
            round,
            w_g,
        )?;

        if w_g != sum_all {
            return Err(Error::GRK(format!(
                    "previous w_g is different from the sum over boolean hypercube of round {} sumcheck protocol",
                    round,
                )));
        }

        let constant_ext = self.circuit.layer(round).constant_ext.as_slice();
        let mul_ext = self.circuit.layer(round).mul_ext.as_slice();
        let forward_x_ext = self.circuit.layer(round).forward_x_ext.as_slice();
        let forward_y_ext = self.circuit.layer(round).forward_y_ext.as_slice();

        let mut oracle_access = F::ZERO;
        let effective_random_g = &random_g[self.total_extra_num_vars()..];
        let effective_random_u = &random_u[self.total_extra_num_vars()..];

        let extra_g = &random_g[..self.total_extra_num_vars()];
        let extra_u = &random_u[..self.total_extra_num_vars()];
        for i in 0..constant_ext.len() {
            let y_num_vars = self
                .circuit
                .layer(round)
                .constant_ext
                .subset_num_vars_at(&LayerIndex::Relative(i + 1));

            let random_v_i = &random_v[..self.total_extra_num_vars() + y_num_vars];
            let effecitve_random_v_i = &random_v_i[self.total_extra_num_vars()..];

            let effective_points =
                vec![effective_random_g, effective_random_u, effecitve_random_v_i];

            let constant =
                constant_ext[i].evaluate(effective_points.clone(), &CircuitParams::default());
            let mul = mul_ext[i].evaluate(effective_points.clone(), &CircuitParams::default());
            let forward_x =
                forward_x_ext[i].evaluate(effective_points.clone(), &CircuitParams::default());
            let forward_y = forward_y_ext[i].evaluate(effective_points, &CircuitParams::default());

            // compute yk * ... * yN.
            let mut remaining_v_product = F::ONE;
            for j in self.total_extra_num_vars() + y_num_vars..random_v.len() {
                remaining_v_product *= random_v[j];
            }

            // This mle returns ONE if extra_g == extra_u == extra_v on boolean
            // hypercube.
            let extra_v = &random_v[..self.total_extra_num_vars()];
            let extra_mle_value = identity_mle_vec(vec![extra_g, extra_u, extra_v]);

            oracle_access += extra_mle_value
                * remaining_v_product
                * (constant + mul * w_u * w_v[i] + forward_x * w_u + forward_y * w_v[i]);

            gkr_params.add_claim(round, round + i + 1, random_v_i.to_vec(), w_v[i]);
        }
        gkr_params.add_last_claim(round + 1, random_u, w_u);

        if oracle_access != final_value {
            return Err(Error::GRK(format!(
                "oracle sum is different from final value at the end of round {} sumcheck protocol",
                round
            )));
        }

        let next_sumcheck_verifier = MultiSumcheckVerifier::new(
            1,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + self.circuit.num_vars_at(round + 1),
        );
        // Reduce all claims to one point of next round.
        fiat_shamir_engine.set_seed(w_g);
        let (random_next_g, final_value_of_reduce_sumcheck, sum_of_reduce_sumcheck) =
            next_sumcheck_verifier.verify(fiat_shamir_engine, combining_transcript.into_iter())?;

        let all_claims = gkr_params.get_claims_of(round + 1);
        assert!(all_claims.len() == alpha.len());

        let mut expected_sum_of_reduce_sumcheck = F::ZERO;
        for (i, claim) in all_claims.iter().enumerate() {
            expected_sum_of_reduce_sumcheck += alpha[i] * claim;
        }

        if sum_of_reduce_sumcheck != expected_sum_of_reduce_sumcheck {
            return Err(Error::GRK(format!(
                "sum of reduce is different from the expected result at round {}",
                round,
            )));
        }

        let subset_reverse_ext = if round + 1 == self.circuit.len() {
            self.circuit.input_subset_reverse_ext.as_slice()
        } else {
            self.circuit.layer(round + 1).subset_reverse_ext.as_slice()
        };

        let above_random_points = gkr_params.get_random_points_of(round + 1);

        // alpha[i+1] * Ci+1(r, x),  where Ci+1(r, x) === extra_mle.
        let mut g_r = alpha[subset_reverse_ext.len()]
            * identity_mle(&above_random_points.last().unwrap(), &random_next_g);

        // sum(alpha[i] * Ci(r, x)), where Ci(r, x) === extra_mle * subset_mle.
        let extra_next_g = &random_next_g[..self.total_extra_num_vars()];
        let effective_next_g = &random_next_g[self.total_extra_num_vars()..];
        for i in 0..subset_reverse_ext.len() {
            let extra_random_point = &above_random_points[i][..self.total_extra_num_vars()];
            let effective_random_point = &above_random_points[i][self.total_extra_num_vars()..];

            let extra_mle = identity_mle(extra_random_point, extra_next_g);

            g_r += alpha[i]
                * extra_mle
                * subset_reverse_ext[i].evaluate(
                    vec![&effective_random_point, &effective_next_g],
                    &CircuitParams::default(),
                );
        }

        let next_w_g = final_value_of_reduce_sumcheck / g_r;
        Ok((random_next_g, next_w_g))
    }

    pub fn verify_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        input: &[F],
        mut transcript: TranscriptIter,
    ) -> Result<(Vec<F>, F, Vec<Vec<F>>), Error> {
        fiat_shamir_engine.begin_protocol();

        let mut random_g = fiat_shamir_engine.hash_to_fields(
            &F::ZERO,
            self.total_extra_num_vars() + self.circuit.num_vars_at(0),
        );

        let now = Instant::now();

        let recv_output = transcript.pop_and_deserialize::<Vec<Vec<F>>>();
        println!("VERIFY GKR 1: {:?}", now.elapsed());

        let output = recv_output.clone();

        let output_ext = SisuDenseMajorZeroMultilinearExtension::from_groups(
            recv_output,
            round_to_pow_of_two(self.circuit.len_at(0)),
        );
        let mut w_g = output_ext.evaluate(vec![&random_g]);

        let mut gkr_params = GeneralVerifierGKRParams::new(self.circuit.len());

        for round in 0..transcript.remaining_len() {
            let now = Instant::now();
            let alpha = fiat_shamir_engine.hash_to_fields(&w_g, round + 2);
            let round_transcript = transcript.pop_and_deserialize::<Transcript>();

            // All last_w_u and last_w_v are ignored, except the last one.
            (random_g, w_g) = self.verify_round(
                fiat_shamir_engine,
                round_transcript.into_iter(),
                round,
                &random_g,
                w_g,
                &alpha,
                &mut gkr_params,
            )?;

            println!("VERIFY GKR 3-{}: {:?}", round, now.elapsed());
        }

        let total_replicas = 2usize.pow(self.total_extra_num_vars() as u32);
        if input.len() == round_to_pow_of_two(self.circuit.input_size()) * total_replicas {
            let mut padding_input = input.to_vec();
            padding_pow_of_two_size(&mut padding_input);

            let w_d = SisuDenseMultilinearExtension::from_slice(&padding_input);
            println!("VERIFY GKR 4: {:?}", now.elapsed());

            let w_d_g = w_d.evaluate(vec![&random_g]);
            if w_d_g != w_g {
                return Err(Error::GRK(format!(
                    "w_g {:?} != expected w_g {:?}",
                    w_d_g, w_g,
                )));
            }
            println!("VERIFY GKR 5: {:?}", now.elapsed());
        }

        Ok((random_g, w_g, output))
    }
}

impl<'a, F: Field> GeneralGKRVerifier<'a, F> {
    fn extract_sumcheck_transcript(
        &self,
        mut transcript: TranscriptIter,
        round: usize,
    ) -> GeneralGKRSumcheckTranscript<F> {
        let mut num_sumchecks = 0;

        if self.circuit.layer(round).constant_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).mul_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).forward_x_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).forward_y_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        let x_num_vars = self.circuit.num_vars_at(round + 1);
        let mut y_num_vars = 0;
        for target_layer_index in 1..=self.circuit.len() - round {
            let target_layer_index = LayerIndex::Relative(target_layer_index);

            y_num_vars = max(
                y_num_vars,
                self.circuit
                    .layer(round)
                    .mul_ext
                    .subset_num_vars_at(&target_layer_index),
            );
        }

        let sumcheck_phase_1_transcript = transcript.pop_and_deserialize::<Transcript>();
        let sumcheck_phase_1_verifier = MultiSumcheckVerifier::new(
            num_sumchecks,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + x_num_vars,
        );

        let sumcheck_phase_2_transcript = transcript.pop_and_deserialize::<Transcript>();
        let sumcheck_phase_2_verifier = MultiSumcheckVerifier::new(
            num_sumchecks,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + y_num_vars,
        );

        GeneralGKRSumcheckTranscript {
            phase_1: sumcheck_phase_1_verifier
                .extract_transcript(sumcheck_phase_1_transcript.into_iter()),
            phase_2: sumcheck_phase_2_verifier
                .extract_transcript(sumcheck_phase_2_transcript.into_iter()),
        }
    }

    fn extract_round_transcript(
        &self,
        mut transcript: TranscriptIter,
        round: usize,
    ) -> GeneralGKRRoundTranscript<F> {
        let sumcheck_transcript = transcript.pop_and_deserialize::<Transcript>();
        let w_u = transcript.pop_and_deserialize::<F>();
        let w_v = transcript.pop_and_deserialize::<Vec<F>>();
        let combining_transcript = transcript.pop_and_deserialize::<Transcript>();

        let sumcheck_transcript =
            self.extract_sumcheck_transcript(sumcheck_transcript.into_iter(), round);

        let combining_sumcheck_verifier = MultiSumcheckVerifier::new(
            1,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + self.circuit.num_vars_at(round + 1),
        );

        GeneralGKRRoundTranscript {
            sumcheck: sumcheck_transcript,
            w_u,
            w_v,
            combining_sumcheck: combining_sumcheck_verifier
                .extract_transcript(combining_transcript.into_iter()),
        }
    }

    pub fn extract_transcript(&self, mut transcript: TranscriptIter) -> GeneralGKRTranscript<F> {
        let mut gkr_transcript = GeneralGKRTranscript::default();

        let recv_output = transcript.pop_and_deserialize::<Vec<Vec<F>>>();
        gkr_transcript.output = recv_output;

        for round in 0..transcript.remaining_len() {
            let round_transcript = transcript.pop_and_deserialize::<Transcript>();
            // All last_w_u and last_w_v are ignored, except the last one.
            let gkr_round_transcript =
                self.extract_round_transcript(round_transcript.into_iter(), round);

            gkr_transcript.round_transcripts.push(gkr_round_transcript);
        }

        gkr_transcript
    }
}

impl<'a, F: Field> GeneralGKRVerifier<'a, F> {
    fn sumcheck_configs(&self, round: usize) -> GeneralGKRSumcheckConfigs {
        let mut num_sumchecks = 0;

        if self.circuit.layer(round).constant_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).mul_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).forward_x_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        if self.circuit.layer(round).forward_y_ext.is_non_zero() {
            num_sumchecks += 1;
        }

        let x_num_vars = self.circuit.num_vars_at(round + 1);
        let mut y_num_vars = 0;
        for target_layer_index in 1..=self.circuit.len() - round {
            let target_layer_index = LayerIndex::Relative(target_layer_index);

            y_num_vars = max(
                y_num_vars,
                self.circuit
                    .layer(round)
                    .mul_ext
                    .subset_num_vars_at(&target_layer_index),
            );
        }

        let sumcheck_phase_1_verifier = MultiSumcheckVerifier::<F>::new(
            num_sumchecks,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + x_num_vars,
        );

        let sumcheck_phase_2_verifier = MultiSumcheckVerifier::<F>::new(
            num_sumchecks,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + y_num_vars,
        );

        GeneralGKRSumcheckConfigs {
            phase_1: sumcheck_phase_1_verifier.configs(),
            phase_2: sumcheck_phase_2_verifier.configs(),
        }
    }

    fn round_configs(&self, round: usize) -> GeneralGKRRoundConfigs {
        let sumcheck_configs = self.sumcheck_configs(round);

        let combining_sumcheck_verifier = MultiSumcheckVerifier::<F>::new(
            1,
            self.master_extra_num_vars,
            self.worker_extra_num_vars + self.circuit.num_vars_at(round + 1),
        );

        GeneralGKRRoundConfigs {
            phase_sumcheck: sumcheck_configs,
            combining_sumcheck: combining_sumcheck_verifier.configs(),
        }
    }

    pub fn configs(&self, num_non_zero_outputs: Option<usize>) -> GeneralGKRConfigs<'a, F> {
        let mut gkr_configs = GeneralGKRConfigs::new(&self.circuit);
        gkr_configs.num_non_zero_outputs =
            num_non_zero_outputs.unwrap_or(round_to_pow_of_two(self.circuit.len_at(0)));
        gkr_configs.num_replicas = 2usize.pow(self.total_extra_num_vars() as u32);

        for round in 0..self.circuit.len() {
            // All last_w_u and last_w_v are ignored, except the last one.
            let gkr_round_configs = self.round_configs(round);
            gkr_configs.round_configs.push(gkr_round_configs);
        }

        gkr_configs
    }
}

pub struct GeneralGKRSumcheckConfigs {
    phase_1: MultiSumcheckConfigs,
    phase_2: MultiSumcheckConfigs,
}

impl GeneralGKRSumcheckConfigs {
    pub fn gen_code<F: Field>(
        &self,
        gkr_index: usize,
        round_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        let mut funcs = vec![];

        let f = self
            .phase_1
            .gen_code(gkr_index * 1000 + round_index * 10 + 1);
        funcs.extend(f);

        let f = self
            .phase_2
            .gen_code(gkr_index * 1000 + round_index * 10 + 2);
        funcs.extend(f);

        funcs
    }
}

#[derive(Default)]
pub struct GeneralGKRSumcheckTranscript<F: Field> {
    phase_1: MultiSumcheckTranscript<F>,
    phase_2: MultiSumcheckTranscript<F>,
}

impl<F: Field> GeneralGKRSumcheckTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];
        result.extend(self.phase_1.to_vec());
        result.extend(self.phase_2.to_vec());

        result
    }
}

pub struct GeneralGKRRoundConfigs {
    phase_sumcheck: GeneralGKRSumcheckConfigs,
    combining_sumcheck: MultiSumcheckConfigs,
}

impl GeneralGKRRoundConfigs {
    pub fn gen_code<F: Field>(
        &self,
        gkr_index: usize,
        round_index: usize,
    ) -> Vec<FuncGenerator<F>> {
        let mut funcs = vec![];

        let f = self.phase_sumcheck.gen_code(gkr_index, round_index);
        funcs.extend(f);

        let f = self
            .combining_sumcheck
            .gen_code(gkr_index * 1000 + round_index * 10 + 3);
        funcs.extend(f);

        funcs
    }
}

#[derive(Default)]
pub struct GeneralGKRRoundTranscript<F: Field> {
    sumcheck: GeneralGKRSumcheckTranscript<F>,
    w_u: F,
    w_v: Vec<F>,
    combining_sumcheck: MultiSumcheckTranscript<F>,
}

impl<F: Field> GeneralGKRRoundTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];
        result.extend(self.sumcheck.to_vec());
        result.push(self.w_u);
        result.extend(self.w_v.clone());
        result.extend(self.combining_sumcheck.to_vec());

        result
    }
}

pub struct GeneralGKRConfigs<'a, F: Field> {
    circuit: &'a GeneralCircuit<F>,
    num_replicas: usize,
    num_non_zero_outputs: usize,
    round_configs: Vec<GeneralGKRRoundConfigs>,
}

impl<'a, F: Field> GeneralGKRConfigs<'a, F> {
    pub fn new(circuit: &'a GeneralCircuit<F>) -> Self {
        Self {
            num_replicas: 0,
            circuit,
            num_non_zero_outputs: 0,
            round_configs: vec![],
        }
    }

    pub fn gen_code(&self, gkr_index: usize) -> Vec<FuncGenerator<F>> {
        let mut funcs = vec![];

        let mut num_non_zero_outputs_func =
            FuncGenerator::new("get_general_gkr__num_non_zero_outputs", vec!["gkr_index"]);
        num_non_zero_outputs_func.add_number(vec![gkr_index], self.num_non_zero_outputs);
        funcs.push(num_non_zero_outputs_func);

        let mut num_rounds_func =
            FuncGenerator::new("get_general_gkr__num_rounds", vec!["gkr_index"]);
        num_rounds_func.add_number(vec![gkr_index], self.round_configs.len());
        funcs.push(num_rounds_func);

        let mut num_outputs_func =
            FuncGenerator::new("get_general_gkr__num_outputs", vec!["gkr_index"]);
        num_outputs_func.add_number(vec![gkr_index], round_to_pow_of_two(self.circuit.len_at(0)));
        funcs.push(num_outputs_func);

        let mut num_replicas_func =
            FuncGenerator::new("get_general_gkr__num_replicas", vec!["gkr_index"]);
        num_replicas_func.add_number(vec![gkr_index], self.num_replicas);
        funcs.push(num_replicas_func);

        for i in 0..self.round_configs.len() {
            let f = self.round_configs[i].gen_code(gkr_index, i);
            funcs.extend(f);
        }

        let f = self.circuit.gen_code(gkr_index);
        funcs.extend(f);

        funcs
    }
}

#[derive(Default)]
pub struct GeneralGKRTranscript<F: Field> {
    output: Vec<Vec<F>>,
    round_transcripts: Vec<GeneralGKRRoundTranscript<F>>,
}

impl<F: Field> GeneralGKRTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];
        for o in self.output.iter() {
            result.extend(o);
        }

        for round in self.round_transcripts.iter() {
            result.extend(round.to_vec());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use sisulib::{
        circuit::general_circuit::examples::example_general_circuit,
        codegen::generator::FileGenerator, common::convert_vec_field_to_string, field::FpSisu,
    };

    use crate::{
        channel::NoChannel,
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine},
        sisu_engine::GPUFrBN254SisuEngine,
    };

    use super::*;

    #[test]
    fn test_gkr() {
        let num_replicas = 4;

        let mut circuit = example_general_circuit();
        circuit.finalize(HashMap::default());

        let mut witness = vec![];
        let mut expected_output = vec![];
        for i in 0..num_replicas {
            let mut tmp = vec![];
            for j in 0..circuit.input_size() {
                tmp.push(FpSisu::from(((i + 1) * (j + 1)) as u64));
            }
            let evaluations = circuit.evaluate(&CircuitParams::default(), &tmp);
            let output = evaluations.at_layer(0, true).to_vec();

            witness.push(CudaSlice::on_host(tmp));
            expected_output.push(output);
        }

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::from(3));

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::from(3));

        let engine = GPUFrBN254SisuEngine::new();
        let prover =
            GeneralGKRProver::<_, _, NoChannel, NoChannel>::new(&engine, &circuit, num_replicas);
        let (_, _, transcript) =
            prover.generate_transcript(&mut prover_fiat_shamir, &mut witness, None);

        let mut verifier = GeneralGKRVerifier::new(&circuit);
        verifier.replicate(num_replicas, 1);

        let result =
            verifier.verify_transcript(&mut verifier_fiat_shamir, &[], transcript.into_iter());
        match result {
            Ok((_, _, output)) => {
                assert_eq!(output, expected_output, "wrong output");
            }
            Err(e) => panic!("{e}"),
        }

        let gkr_transcript = verifier.extract_transcript(transcript.into_iter());
        println!(
            "Transcript: {:?} {}",
            convert_vec_field_to_string(&gkr_transcript.to_vec()),
            gkr_transcript.to_vec().len(),
        );

        let gkr_configs = verifier.configs(None);
        let mut file_gen =
            FileGenerator::<FpSisu>::new("../bls-circom/circuit/sisu/configs.gen.circom");

        let f = gkr_configs.gen_code(0);
        file_gen.extend_funcs(f);
        file_gen.create();
    }
}
