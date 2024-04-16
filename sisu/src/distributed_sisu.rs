use std::{
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use ark_ff::Field;
use sisulib::{
    circuit::{general_circuit::circuit::GeneralCircuit, CircuitParams},
    codegen::generator::{CustomMLEGenerator, FuncGenerator},
    common::{dec2bin_limit, padding_pow_of_two_size, Error},
    domain::{Domain, RootDomain},
    mle::dense::SisuDenseMultilinearExtension,
};

use crate::{
    channel::{self, MasterNode, SisuReceiver, SisuSender, WorkerNode},
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine, Transcript, TranscriptIter},
    general_gkr::{GeneralGKRConfigs, GeneralGKRProver, GeneralGKRVerifier},
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    mempool::{MasterSharedMemPool, SharedMemPool, WorkerSharedMemPool},
    multi_vpd::{
        MultiVPDCentralCommitment, MultiVPDConfigs, MultiVPDMasterProver, MultiVPDVerifier,
        MultiVPDWorkerCommitment, MultiVPDWorkerProver,
    },
    sisu_engine::SisuEngine,
    sisu_merkle_tree::SisuMerkleTreeEngine,
    synthetic_gkr::SyntheticGeneralGKRProver,
    vpd::generate_ldt_root_domain,
};

pub struct DistributedSisuWorkerCommitment<
    F: IcicleConvertibleField,
    MTEngine: SisuMerkleTreeEngine<F>,
> {
    vpd_commitment: MultiVPDWorkerCommitment<F, MTEngine>,
    circuit_inputs: Vec<CudaSlice<F>>,
}

pub struct DistributedSisuCentralCommitment<
    F: IcicleConvertibleField,
    MTEngine: SisuMerkleTreeEngine<F>,
> {
    vpd_commitment: MultiVPDCentralCommitment<F, MTEngine>,
}

impl<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>>
    DistributedSisuCentralCommitment<F, MTEngine>
{
    pub fn to_transcript(&self) -> Transcript {
        self.vpd_commitment.to_transcript()
    }
}

pub struct SisuExtraVPDParams<F: Field> {
    pub input_position: usize,
    pub input_size: usize,
    pub replica_distance: usize,

    input: Option<Vec<Vec<F>>>,
    expected_value: Option<F>,
}

impl<F: Field> SisuExtraVPDParams<F> {
    pub fn new(input_position: usize, replica_distance: usize) -> Self {
        assert!(replica_distance.is_power_of_two());

        Self {
            input_position,
            replica_distance,
            input_size: 0,
            input: None,
            expected_value: None,
        }
    }

    pub fn with_input(mut self, input: &[Vec<F>]) -> Self {
        assert!(input[0].len().is_power_of_two());
        self.input_size = input[0].len();
        self.input = Some(input.to_vec());

        self
    }

    pub fn with_expected(mut self, input_size: usize, expected_value: F) -> Self {
        assert!(input_size.is_power_of_two());
        self.input_size = input_size;
        self.expected_value = Some(expected_value);
        self
    }

    pub fn get_expected_value(&self, r: &[F]) -> Option<F> {
        match &self.expected_value {
            Some(v) => Some(v.clone()),
            None => match &self.input {
                Some(input) => {
                    let mut single_input_mle_values = vec![];
                    let combining_size = r.len() - self.input_size.ilog2() as usize;

                    for replica_index in 0..input.len() {
                        if replica_index % self.replica_distance == 0 {
                            let sub_mle =
                                SisuDenseMultilinearExtension::from_slice(&input[replica_index]);
                            single_input_mle_values
                                .push(sub_mle.evaluate(vec![&r[combining_size..]]));
                        }
                    }

                    let expected_input_mle =
                        SisuDenseMultilinearExtension::from_slice(&single_input_mle_values);
                    Some(expected_input_mle.evaluate(vec![
                        &r[..combining_size - self.replica_distance.ilog2() as usize],
                    ]))
                }
                None => None,
            },
        }
    }
}

pub struct SisuWorker<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    engine: &'a Engine,
    circuit: &'a GeneralCircuit<F>,
    worker: &'a WorkerNode<S, R>,

    vpd_prover: MultiVPDWorkerProver<'a, F, Engine, S, R>,
    total_replicas: usize,
    num_workers: usize,
    worker_index: usize,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    SisuWorker<'a, F, Engine, S, R>
{
    pub fn new(
        circuit: &'a GeneralCircuit<F>,
        engine: &'a Engine,
        mempool: WorkerSharedMemPool<'a, F, S, R>,
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_repetitions: usize,
        worker_node: &'a WorkerNode<S, R>,
        total_replicas: usize,
        num_workers: usize,
        worker_index: usize,
    ) -> Self {
        assert!(total_replicas.is_power_of_two());
        assert!(num_workers.is_power_of_two());
        assert!(num_workers <= total_replicas);

        Self {
            engine,
            circuit,
            worker: worker_node,
            vpd_prover: MultiVPDWorkerProver::new(
                worker_node,
                engine,
                mempool,
                ldt_domain,
                ldt_rate,
                num_repetitions,
                num_workers,
                worker_index,
            ),
            total_replicas,
            num_workers,
            worker_index,
        }
    }

    pub fn worker_commit(
        &self,
        circuit_input: Vec<Vec<F>>,
    ) -> DistributedSisuWorkerCommitment<F, Engine::MerkleTree> {
        assert!(circuit_input.len() == self.total_replicas / self.num_workers);
        for i in 0..circuit_input.len() {
            assert!(self.circuit.input_size() == circuit_input[i].len());
        }

        let mut flat_circuit_input = vec![];
        let mut cuda_circuit_input = vec![];
        for mut input in circuit_input {
            cuda_circuit_input.push(CudaSlice::on_host(input.clone()));

            padding_pow_of_two_size(&mut input);
            flat_circuit_input.extend(input);
        }

        DistributedSisuWorkerCommitment {
            vpd_commitment: self
                .vpd_prover
                .worker_commit(CudaSlice::on_host(flat_circuit_input)),
            circuit_inputs: cuda_circuit_input,
        }
    }

    fn contribute_gkr<FS: FiatShamirEngine<F>>(
        &'a self,
        fiat_shamir_engine: &mut FS,
        circuit_inputs: &mut [CudaSlice<F>],
        num_non_zero_outputs: Option<usize>,
    ) -> CudaSlice<F> {
        let num_replicas_per_worker = self.total_replicas / self.num_workers;
        assert!(circuit_inputs.len() == num_replicas_per_worker);

        let mut worker_gkr_prover = GeneralGKRProver::<_, Engine, _, _>::new(
            self.engine,
            &self.circuit,
            num_replicas_per_worker,
        );
        worker_gkr_prover.to_worker(self.worker, self.num_workers, self.worker_index);
        let (random_g, _w_g, _transcript) = worker_gkr_prover.generate_transcript(
            fiat_shamir_engine,
            circuit_inputs,
            num_non_zero_outputs,
        );

        random_g
    }

    pub fn contribute_transcript<FS: FiatShamirEngine<F>>(
        &'a self,
        fiat_shamir_engine: &mut FS,
        mut worker_commitment: DistributedSisuWorkerCommitment<F, Engine::MerkleTree>,
        extra_vpd_params: &[SisuExtraVPDParams<F>],
        num_non_zero_outputs: Option<usize>,
    ) {
        fiat_shamir_engine.begin_protocol();

        let now = Instant::now();

        fiat_shamir_engine.inherit_seed();
        let mut g = self.contribute_gkr(
            fiat_shamir_engine,
            &mut worker_commitment.circuit_inputs,
            num_non_zero_outputs,
        );
        println!(
            "[Distributed Sisu Worker] DONE CONTRIBUTE GKR: {:?}",
            now.elapsed()
        );

        // Step 1: Contribute V(g) VPD transcript to master.
        let now = Instant::now();
        fiat_shamir_engine.reduce_and_set_seed(g.as_ref_host());
        self.vpd_prover.contribute_transcript(
            fiat_shamir_engine,
            &mut worker_commitment.vpd_commitment,
            &g,
        );
        println!(
            "[Distributed Sisu Worker] DONE CONTRIBUTE VPD: {:?}",
            now.elapsed()
        );

        // Step 2: Run extra VPD.
        for (i, extra_vpd_param) in extra_vpd_params.into_iter().enumerate() {
            let now = Instant::now();
            let v_extra_random_points = CudaSlice::on_host(generate_v_input_random_points(
                fiat_shamir_engine,
                extra_vpd_param,
                self.total_replicas,
                &self.circuit,
            ));

            fiat_shamir_engine.reduce_and_set_seed(g.as_ref_host());
            self.vpd_prover.contribute_transcript(
                fiat_shamir_engine,
                &mut worker_commitment.vpd_commitment,
                &v_extra_random_points,
            );
            println!(
                "[Distributed Sisu Worker] DONE CONTRIBUTE EXTRA VPD [{}]: {:?}",
                i,
                now.elapsed()
            );
        }
    }
}

pub struct SisuMaster<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    total_replicas: usize,
    num_workers: usize,
    circuit: &'a GeneralCircuit<F>,
    master: &'a MasterNode<S, R>,
    vpd_prover: MultiVPDMasterProver<'a, F, Engine, S, R>,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    SisuMaster<'a, F, Engine, S, R>
{
    pub fn new(
        circuit: &'a GeneralCircuit<F>,
        master_node: &'a MasterNode<S, R>,
        engine: &'a Engine,
        mempool: MasterSharedMemPool<'a, S, R>,
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_repetitions: usize,
        num_workers: usize,
        total_replicas: usize,
    ) -> Self {
        assert!(num_workers.is_power_of_two());

        Self {
            total_replicas,
            num_workers,
            circuit,
            master: master_node,
            vpd_prover: MultiVPDMasterProver::new(
                master_node,
                engine,
                mempool,
                ldt_domain,
                ldt_rate,
                num_repetitions,
                num_workers,
            ),
        }
    }

    pub fn central_commit(&self) -> DistributedSisuCentralCommitment<F, Engine::MerkleTree> {
        DistributedSisuCentralCommitment {
            vpd_commitment: self.vpd_prover.central_commit(),
        }
    }

    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &'a self,
        fiat_shamir_engine: &mut FS,
        central_commitment: &DistributedSisuCentralCommitment<F, Engine::MerkleTree>,
        extra_vpd_params: &[SisuExtraVPDParams<F>],
    ) -> Transcript {
        fiat_shamir_engine.begin_protocol();

        let mut transcript = Transcript::default();

        let now = Instant::now();
        let num_replicas_per_worker = self.total_replicas / self.num_workers;
        let mut gkr_prover = SyntheticGeneralGKRProver::<_, Engine, _, _>::new(
            &self.master,
            &self.circuit,
            self.num_workers,
            num_replicas_per_worker,
        );

        fiat_shamir_engine.inherit_seed();
        let (mut g, gkr_transcript) = gkr_prover.generate_transcript(fiat_shamir_engine);
        transcript.serialize_and_push(&gkr_transcript);
        println!(
            "[Distributed Sisu Master] DONE SYNTHETIZE GKR: {:?}",
            now.elapsed()
        );

        // Step 1: Run V(g) VPD.
        let now = Instant::now();
        fiat_shamir_engine.reduce_and_set_seed(g.as_ref_host());
        let vg_vpd_transcript = self.vpd_prover.generate_transcript(
            fiat_shamir_engine,
            &central_commitment.vpd_commitment,
            g.as_ref_host(),
        );
        transcript.serialize_and_push(&vg_vpd_transcript);
        println!(
            "[Distributed Sisu Master] DONE SYNTHETIZE VPD: {:?}",
            now.elapsed()
        );

        // Step 2: Run extra VPD.
        for (i, extra_vpd_param) in extra_vpd_params.iter().enumerate() {
            let now = Instant::now();
            let v_extra_random_points = generate_v_input_random_points(
                fiat_shamir_engine,
                extra_vpd_param,
                self.total_replicas,
                &self.circuit,
            );

            fiat_shamir_engine.reduce_and_set_seed(g.as_ref_host());
            let v_extra_vpd_transcript = self.vpd_prover.generate_transcript(
                fiat_shamir_engine,
                &central_commitment.vpd_commitment,
                &v_extra_random_points,
            );
            transcript.serialize_and_push(&v_extra_vpd_transcript);
            transcript.serialize_and_push(&vg_vpd_transcript);
            println!(
                "[Distributed Sisu Master] DONE SYNTHETIZE EXTRA VPD [{}]: {:?}",
                i,
                now.elapsed()
            );
        }

        transcript
    }
}

pub struct DistributedSisuVerifier<'a, F: IcicleConvertibleField, H: SisuHasher<F>> {
    circuit: &'a GeneralCircuit<F>,
    num_replicas_per_worker: usize,
    num_workers: usize,

    vpd_verifier: MultiVPDVerifier<'a, F, H>,
}

impl<'a, F: IcicleConvertibleField, H: SisuHasher<F>> DistributedSisuVerifier<'a, F, H> {
    pub fn new(
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_repetitions: usize,
        circuit: &'a GeneralCircuit<F>,
        num_replicas_per_worker: usize,
        num_workers: usize,
    ) -> Self {
        assert!(num_replicas_per_worker.is_power_of_two());
        assert!(num_workers.is_power_of_two());

        Self {
            circuit,
            num_workers,
            num_replicas_per_worker,
            vpd_verifier: MultiVPDVerifier::new(ldt_domain, ldt_rate, num_repetitions, num_workers),
        }
    }

    pub fn verify_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        commitment_transcript: TranscriptIter,
        mut transcript: TranscriptIter,
        extra_vpd_params: &[SisuExtraVPDParams<F>],
    ) -> Result<(Vec<Vec<F>>, Vec<F>), Error> {
        fiat_shamir_engine.begin_protocol();

        // Step 1: Run GKR.
        let gkr_transcript = transcript.pop_and_deserialize::<Transcript>();

        let mut gkr_verifier = GeneralGKRVerifier::new(&self.circuit);
        gkr_verifier.replicate(self.num_replicas_per_worker, self.num_workers);

        fiat_shamir_engine.inherit_seed();
        let (last_g, last_vg, output) = gkr_verifier.verify_transcript(
            fiat_shamir_engine,
            &[], // We will check the input later, so do not pass input here.
            gkr_transcript.into_iter(),
        )?;

        // Step 2: Run V(g) VPD.
        fiat_shamir_engine.reduce_and_set_seed(&last_g);
        let vg_vpd_transcript = transcript.pop_and_deserialize::<Transcript>();
        let open_vg = self.vpd_verifier.verify_transcript(
            fiat_shamir_engine,
            commitment_transcript.clone(),
            vg_vpd_transcript.into_iter(),
            &last_g,
        )?;

        if open_vg != last_vg {
            return Err(Error::Sisu(String::from(
                "Vd at g of vpd commitment is diffrerent from expected result",
            )));
        }

        // Step 3: Run extra VPD.
        let mut all_open_v_extra = vec![];
        for extra_vpd_param in extra_vpd_params {
            let v_extra_random_points = generate_v_input_random_points(
                fiat_shamir_engine,
                extra_vpd_param,
                self.num_replicas_per_worker * self.num_workers,
                &self.circuit,
            );

            let v_extra_vpd_transcript = transcript.pop_and_deserialize::<Transcript>();
            fiat_shamir_engine.reduce_and_set_seed(&last_g);
            let open_v_extra = self.vpd_verifier.verify_transcript(
                fiat_shamir_engine,
                commitment_transcript.clone(),
                v_extra_vpd_transcript.into_iter(),
                &v_extra_random_points,
            )?;

            let expected_extra_mle_value =
                extra_vpd_param.get_expected_value(&v_extra_random_points);

            if let Some(value) = expected_extra_mle_value {
                if open_v_extra != value {
                    return Err(Error::Sisu(String::from(
                        "Vd of check input of vpd commitment is diffrerent from expected result",
                    )));
                }
            }

            all_open_v_extra.push(open_v_extra);
        }

        Ok((output, all_open_v_extra))
    }

    // pub fn configs(
    //     &self,
    //     circuit_params: CircuitParams<'a, F>,
    //     num_non_zero_outputs: Option<usize>,
    // ) -> DistributedSisuConfigs<F> {
    //     let mut gkr_verifier = GeneralGKRVerifier::new(&self.circuit, circuit_params);
    //     gkr_verifier.replicate(self.num_replicas_per_worker, self.num_workers);

    //     DistributedSisuConfigs {
    //         gkr_configs: gkr_verifier.configs(num_non_zero_outputs),
    //         vpd_configs: self.vpd_verifier.configs(fiat_shamir_engine, transcript),
    //     }
    // }
}

pub struct DistributedSisuConfigs<'a, F: Field> {
    gkr_configs: GeneralGKRConfigs<'a, F>,
    vpd_configs: MultiVPDConfigs<'a, F>,
}

impl<'a, F: Field> DistributedSisuConfigs<'a, F> {
    pub fn gen_code(&self, sisu_index: usize) -> (Vec<FuncGenerator<F>>, Vec<CustomMLEGenerator>) {
        let mut funcs = vec![];

        let f = self.gkr_configs.gen_code(sisu_index);
        funcs.extend(f);

        let (f, mle) = self.vpd_configs.gen_code(sisu_index);
        funcs.extend(f);

        (funcs, mle)
    }
}

pub fn generate_distributed_root_domain<F: Field>(
    circuit: &GeneralCircuit<F>,
    ldt_rate: usize,
    num_replicas_per_worker: usize,
) -> RootDomain<F> {
    assert!(num_replicas_per_worker.is_power_of_two());
    let input_num_vars =
        circuit.num_vars_at(circuit.len()) + num_replicas_per_worker.ilog2() as usize;
    generate_ldt_root_domain(input_num_vars, ldt_rate)
}

pub fn generate_v_input_random_points<F: Field, FS: FiatShamirEngine<F>>(
    fiat_shamir_engine: &mut FS,
    param: &SisuExtraVPDParams<F>,
    total_replicas: usize,
    circuit: &GeneralCircuit<F>,
) -> Vec<F> {
    assert!(total_replicas.is_power_of_two());

    // Input and witness example:
    // [w00 i0 w01 w02; w10 i1 w11 w12; ...; wN0 iN wN1 wN2]
    //
    // Let:
    // - n: single public input num vars.
    // - d: replica distance num vars (for example, we only use replicas 0, 2, 4, 6 instead of all replicas).
    // - N: single circuit num vars (both input + witness).
    // - K: total num vars (after all, the random point has K-size).
    //
    // random_points = [(K-N-d) random + d zeros + (N-n-d) input index points + d zeros + n random], where
    // - the first (K-N-d) random points represent for combining all replicas.
    // - the next d zeros represent for replica distance.
    // - the next (N-n) points represent for indexing input groups in
    //   input_witness.
    // - the last n random points represents for every single replica.
    let single_input_num_vars = param.input_size.ilog2() as usize;
    let distance_num_vars = param.replica_distance.ilog2() as usize;
    let circuit_num_vars = circuit.num_vars_at(circuit.len());
    let total_num_vars = total_replicas.ilog2() as usize + circuit_num_vars;

    let combination_random_points = fiat_shamir_engine.hash_to_fields(
        &F::ZERO,
        total_num_vars - circuit_num_vars - distance_num_vars,
    );
    let replica_index_random_points = vec![F::ZERO; distance_num_vars];
    let input_index_random_points: Vec<F> = dec2bin_limit(
        param.input_position as u64,
        circuit_num_vars - single_input_num_vars,
    )
    .into_iter()
    .rev()
    .collect();
    let group_random_points = fiat_shamir_engine.hash_to_fields(&F::ZERO, single_input_num_vars);

    let mut random_points = vec![];
    random_points.extend(combination_random_points);
    random_points.extend(replica_index_random_points);
    random_points.extend(input_index_random_points);
    random_points.extend(group_random_points);

    random_points
}

pub trait DefaultSisuRunner<F: IcicleConvertibleField>: Sync {
    fn domain(&self) -> Domain<F>;
    fn ldt_rate(&self) -> usize;
    fn num_repetitions(&self) -> usize;
    fn circuit(&self) -> &GeneralCircuit<F>;
    fn num_workers(&self) -> usize;
    fn num_replicas_per_worker(&self) -> usize;
    fn num_non_zero_outputs(&self) -> Option<usize>;

    fn run_sisu<Engine: SisuEngine<F>, FSH: SisuHasher<F>>(
        &self,
        witness: Vec<Vec<F>>,
        engine: Engine,
    ) {
        #[cfg(feature = "parallel")]
        println!("[Sisu] In parallel mode");

        let num_workers = self.num_workers();
        let num_replicas_per_worker = self.num_replicas_per_worker();
        let total_replicas = num_workers * num_replicas_per_worker;
        assert!(witness.len() == total_replicas);

        let root_mempool = SharedMemPool::new(num_workers);

        let now = Instant::now();
        let mut expected_output = vec![];
        for i in 0..total_replicas {
            let evaluations = self
                .circuit()
                .evaluate(&CircuitParams::default(), &witness[i]);
            expected_output.push(evaluations.at_layer(0, true).to_vec());
        }
        println!("=====================PREPARE OUTPUT: {:?}", now.elapsed());

        // Prepare long-live variables.
        let now = Instant::now();
        let mut worker_witness = vec![];
        for worker_index in 0..num_workers {
            let mut single_worker_witness = vec![];
            for i in 0..num_replicas_per_worker {
                let replica_index = worker_index * num_replicas_per_worker + i;

                let mut tmp = witness[replica_index].clone();
                padding_pow_of_two_size(&mut tmp);
                single_worker_witness.extend(tmp);
            }
            worker_witness.push(single_worker_witness);
        }
        println!(
            "=====================PREPARE LONG LIVE VAR: {:?}",
            now.elapsed()
        );

        // TODO: VPD for check public input.
        let extra_vpd_params = vec![];
        // if input.len() > 0 {
        //     extra_vpd_params.push(SisuExtraVPDParams::new(0, 1).with_input(&input));
        // }
        let extra_vpd_params = Arc::new(extra_vpd_params);
        let thread_witness = Arc::new(witness);

        // Step 0: Setup node.
        let now = Instant::now();
        let (master_sender, master_receiver) = channel::default();
        let mut master_node = MasterNode::from_channel(master_receiver);

        let mut worker_nodes = vec![];
        for i in 0..num_workers {
            let (worker_sender, worker_receiver) = channel::default();
            let worker_node = WorkerNode::from_channel(i, worker_receiver, &master_sender);
            worker_nodes.push(worker_node);
            master_node.add_worker(&worker_sender);
        }

        println!("=====================PREPARE NODE: {:?}", now.elapsed());

        // Step 1: SetupWorker generates communication data.
        let central_commitment_transcript = Arc::new(Mutex::new(Transcript::default()));
        let master_transcript = Arc::new(Mutex::new(Transcript::default()));

        let now = Instant::now();
        let master_engine = engine.clone();
        let master = SisuMaster::<_, Engine, _, _>::new(
            self.circuit(),
            &master_node,
            &master_engine,
            MasterSharedMemPool::new(&master_node),
            self.domain(),
            self.ldt_rate(),
            self.num_repetitions(),
            self.num_workers(),
            total_replicas,
        );
        println!("=====================GEN MASTER: {:?}", now.elapsed());

        let now = Instant::now();
        thread::scope(|scope| {
            for (worker_index, worker_node) in worker_nodes.into_iter().enumerate() {
                let witness = thread_witness.clone();
                let extra_vpd_params = extra_vpd_params.clone();
                let root_mempool = root_mempool.clone();
                let engine = engine.clone();

                scope.spawn(move || {
                    let worker_mempool =
                        WorkerSharedMemPool::clone_from(worker_index, &worker_node, root_mempool);

                    let worker_prover = SisuWorker::<_, Engine, _, _>::new(
                        self.circuit(),
                        &engine,
                        worker_mempool,
                        self.domain(),
                        self.ldt_rate(),
                        self.num_repetitions(),
                        &worker_node,
                        total_replicas,
                        self.num_workers(),
                        worker_index,
                    );

                    let mut worker_witness = vec![];
                    for i in 0..num_replicas_per_worker {
                        let replica_index = worker_index * num_replicas_per_worker + i;
                        worker_witness.push(witness[replica_index].clone());
                    }

                    let now = Instant::now();
                    let worker_commitment = worker_prover.worker_commit(worker_witness);
                    println!("[Sisu RUNNER] WORKER COMMIT: {:?}", now.elapsed());

                    // Step 3: All workers contributes GKR information to the
                    // master.
                    let now = Instant::now();
                    let mut fiat_shamir_engine = DefaultFiatShamirEngine::<_, FSH>::default();
                    fiat_shamir_engine.set_seed(F::ZERO);

                    worker_prover.contribute_transcript(
                        &mut fiat_shamir_engine,
                        worker_commitment,
                        &extra_vpd_params,
                        self.num_non_zero_outputs(),
                    );
                    println!(
                        "[Sisu RUNNER] WORKER CONTRIBUTE TRANSCRIPT: {:?}",
                        now.elapsed()
                    );
                });
            }

            // Step 5: Master synthetizes the GKR information of workers and
            // generate the GKR transcript.
            let now = Instant::now();
            let central_commitment = master.central_commit();
            println!("[Sisu RUNNER] MASTER CENTRAL COMMIT: {:?}", now.elapsed());

            // Step 8: Master synthetizes and generates VPD transcript.
            let mut fiat_shamir_engine = DefaultFiatShamirEngine::<_, FSH>::default();
            fiat_shamir_engine.set_seed(F::ZERO);
            let now = Instant::now();
            let transcript = master.generate_transcript(
                &mut fiat_shamir_engine,
                &central_commitment,
                &extra_vpd_params,
            );
            println!(
                "[Sisu RUNNER] MASTER SYNTHETIZE VPD TRANSCRIPT: {:?}",
                now.elapsed()
            );

            let mut central_commitment_transcript = central_commitment_transcript.lock().unwrap();
            *central_commitment_transcript = central_commitment.to_transcript();

            let mut master_transcript = master_transcript.lock().unwrap();
            *master_transcript = transcript;
        });
        println!("[Sisu RUNNER] DONE SISU PROVER: {:?}", now.elapsed());

        let central_commitment_transcript = Arc::try_unwrap(central_commitment_transcript)
            .unwrap()
            .into_inner()
            .unwrap();

        let master_transcript = Arc::try_unwrap(master_transcript)
            .unwrap()
            .into_inner()
            .unwrap();

        // Step 9: Verifier verifies transcript from MasterProver.

        let now = Instant::now();
        let sisu_verifier = DistributedSisuVerifier::<_, FSH>::new(
            self.domain(),
            self.ldt_rate(),
            self.num_repetitions(),
            self.circuit(),
            self.num_replicas_per_worker(),
            self.num_workers(),
        );

        let mut fiat_shamir_engine = DefaultFiatShamirEngine::<_, FSH>::default();
        fiat_shamir_engine.set_seed(F::ZERO);
        let result = sisu_verifier.verify_transcript(
            &mut fiat_shamir_engine,
            central_commitment_transcript.into_iter(),
            master_transcript.into_iter(),
            &extra_vpd_params,
        );
        println!("[Sisu RUNNER] VERIFIER DONE: {:?}", now.elapsed());

        let (output, _) = result.unwrap();

        match self.num_non_zero_outputs() {
            None => assert_eq!(output, expected_output),
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use sisulib::{circuit::general_circuit::examples::example_general_circuit, field::FpSisu};

    use crate::{hash::SisuMimc, sisu_engine::GPUFrBN254SisuEngine};

    use super::*;

    struct SisuTest<F: Field> {
        circuit: GeneralCircuit<F>,
        domain: RootDomain<F>,
        ldt_rate: usize,
        num_repetitions: usize,
        num_workers: usize,
        num_replicas_per_worker: usize,
    }

    impl<F: Field> SisuTest<F> {
        pub fn new(
            num_replicas_per_worker: usize,
            ldt_rate: usize,
            num_repetitions: usize,
            num_workers: usize,
        ) -> Self {
            let mut circuit = example_general_circuit();
            circuit.finalize(HashMap::default());

            let root_domain =
                generate_distributed_root_domain(&circuit, num_replicas_per_worker, ldt_rate);
            Self {
                circuit,
                domain: root_domain,
                ldt_rate,
                num_repetitions,
                num_replicas_per_worker,
                num_workers,
            }
        }
    }

    impl<F: IcicleConvertibleField> DefaultSisuRunner<F> for SisuTest<F> {
        fn circuit(&self) -> &GeneralCircuit<F> {
            &self.circuit
        }

        fn ldt_rate(&self) -> usize {
            self.ldt_rate
        }

        fn num_repetitions(&self) -> usize {
            self.num_repetitions
        }

        fn domain(&self) -> Domain<F> {
            Domain::from(&self.domain)
        }

        fn num_workers(&self) -> usize {
            self.num_workers
        }

        fn num_replicas_per_worker(&self) -> usize {
            self.num_replicas_per_worker
        }

        fn num_non_zero_outputs(&self) -> Option<usize> {
            None
        }
    }

    #[test]
    fn test_sisu() {
        let sisu_test_runner = SisuTest::new(2usize.pow(10), 8, 1, 2);

        let single_witness_size: usize = sisu_test_runner.circuit().input_size();

        let total_replicas =
            sisu_test_runner.num_replicas_per_worker() * sisu_test_runner.num_workers();

        // Prepare WITNESS
        let mut witness = vec![];
        for i in 0..total_replicas {
            let mut single_witness = vec![];
            for j in 0..single_witness_size {
                single_witness.push(FpSisu::from(((i + 1) * (j + 1)) as u64))
            }

            witness.push(single_witness);
        }

        // let engine = CPUSisuEngine::<_, SisuMimc<_>>::new();
        let engine = GPUFrBN254SisuEngine::new();

        sisu_test_runner.run_sisu::<_, SisuMimc<FpSisu>>(witness, engine);
    }
}
