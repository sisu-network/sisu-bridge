use std::time::Instant;

use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sisulib::{
    circuit::{Circuit, CircuitParams},
    codegen::generator::{CustomMLEGenerator, FuncGenerator},
    common::{dec2bin, ilog2_ceil, padding_pow_of_two_size, Error},
    domain::Domain,
    mle::dense::{identity_mle, SisuDenseMultilinearExtension},
};

use crate::{
    channel::{MasterNode, SisuReceiver, SisuSender, WorkerNode},
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    gkr::{GKRConfigs, GKRProver, GKRTranscript, GKRVerifier},
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    mempool::{MasterSharedMemPool, WorkerSharedMemPool},
    multi_fri::{
        FRICentralCommitment, MultiFRICommitmentTranscript, MultiFRIConfigs, MultiFRIMasterProver,
        MultiFRITranscript, MultiFRIVerifier, MultiFRIWorkerProver,
    },
    sisu_engine::SisuEngine,
    sisu_merkle_tree::{DummyMerkleTreeEngine, MultiMerkleTreeQuery, SisuMerkleTreeEngine, Tree},
    vpd::{
        compute_l_mul_q, compute_p, compute_rational_constraint, divide_by_vanishing_poly,
        generate_gkr_circuit,
    },
};

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MultiLDTFirstQuery<F: Field> {
    positive_query: MultiMerkleTreeQuery<F>,
    negative_query: MultiMerkleTreeQuery<F>,
}

impl<F: Field> MultiLDTFirstQuery<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];

        result.extend(self.positive_query.to_vec());
        result.extend(self.negative_query.to_vec());

        result
    }

    pub fn verify<H: SisuHasher<F>>(
        &self,
        root: &F,
        index: usize,
        op_index: usize,
    ) -> Result<(), Error> {
        self.positive_query.verify::<H>(root, index)?;
        self.negative_query.verify::<H>(root, op_index)?;

        Ok(())
    }
}

pub struct MultiVPDWorkerCommitment<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>> {
    pub evaluations: CudaSlice<F>,
    pub l2_evaluations: CudaSlice<F>,

    l_evaluations: CudaSlice<F>,
    l_worker_merkle_tree: MTEngine::Tree,
}

impl<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>>
    MultiVPDWorkerCommitment<F, MTEngine>
{
    pub fn get_polynomial<'a>(&'a mut self) -> SisuDenseMultilinearExtension<'a, F> {
        SisuDenseMultilinearExtension::from_slice(self.evaluations.as_ref_host())
    }
}

#[derive(Clone)]
pub struct MultiVPDCentralCommitment<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>> {
    l_central_merkle_tree: MTEngine::Tree,
}

impl<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>>
    MultiVPDCentralCommitment<F, MTEngine>
{
    pub fn to_transcript(&self) -> Transcript {
        Transcript::from_vec(vec![&self.l_central_merkle_tree.root()])
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> F {
        transcript.pop_and_deserialize()
    }
}

pub struct MultiVPDWorkerProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    worker: &'a WorkerNode<S, R>,
    num_repetitions: usize,
    num_workers: usize,
    sumcheck_domain: Domain<'a, F>,
    ldt_prover: MultiFRIWorkerProver<'a, F, Engine, S, R>,
    engine: &'a Engine,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    MultiVPDWorkerProver<'a, F, Engine, S, R>
{
    pub fn new(
        worker_node: &'a WorkerNode<S, R>,
        engine: &'a Engine,
        mempool: WorkerSharedMemPool<'a, F, S, R>,
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_repetitions: usize,
        num_workers: usize,
        worker_index: usize,
    ) -> Self {
        assert!(ldt_rate.is_power_of_two());
        assert!(num_workers.is_power_of_two());

        let mut sumcheck_domain = ldt_domain.clone();

        let mut ratio = ldt_rate;
        while ratio > 1 {
            ratio = ratio / 2;
            sumcheck_domain = sumcheck_domain.square();
        }

        Self {
            worker: worker_node,
            num_workers,
            sumcheck_domain,
            num_repetitions,
            engine,
            ldt_prover: MultiFRIWorkerProver::new(
                worker_node,
                engine,
                mempool,
                ldt_domain,
                ldt_rate,
                num_workers,
                worker_index,
            ),
        }
    }

    pub fn worker_commit(
        &self,
        mut evaluations: CudaSlice<F>,
    ) -> MultiVPDWorkerCommitment<F, Engine::MerkleTree> {
        println!(
            "[VPD Worker Commit]: size = 2^{:?}",
            evaluations.len().ilog2()
        );

        let now = Instant::now();
        let mut l_coeffs = self.engine.fft_engine_pool().interpolate(&mut evaluations);
        println!("[VPD Worker Commit]: interpolate L {:?}", now.elapsed());

        // Double the size of sumcheck domain.
        let now = Instant::now();
        let l2_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate(self.sumcheck_domain.len() * 2, &mut l_coeffs);
        println!(
            "[VPD Worker Commit]: evaluate L over 2|H| {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let mut l_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate(self.ldt_prover.ldt_domain.len(), &mut l_coeffs);
        println!(
            "[VPD Worker Commit]: evaluate L over LDT domain {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let l_worker_merkle_tree = self
            .ldt_prover
            .multi_merkle_tree_worker
            .commit(&mut l_evaluations);
        println!("[VPD Worker Commit]: commit L {:?}", now.elapsed());

        MultiVPDWorkerCommitment {
            evaluations,
            l2_evaluations,
            l_evaluations,
            l_worker_merkle_tree,
        }
    }

    pub fn contribute_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        commitment: &mut MultiVPDWorkerCommitment<F, Engine::MerkleTree>,
        input: &CudaSlice<F>,
    ) {
        fiat_shamir_engine.begin_protocol();
        let polynomial = commitment.get_polynomial();
        assert_eq!(
            input.len(),
            ilog2_ceil(self.num_workers) + polynomial.num_vars()
        );

        // Keep only the effective input at this worker.
        let now = Instant::now();
        let mut input = input.at_range_from(self.num_workers.ilog2() as usize..);
        let output = polynomial.evaluate(vec![input.as_ref_host()]);
        self.worker.send_to_master_and_done(&output).unwrap();
        println!(
            "[VPD Worker Transcript]: Generate output and send to master {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let q_polynomial = interpolate_q_polynomial_by_fft_engine(self.engine, &mut input);
        println!(
            "[VPD Worker Transcript]: Interpolate Q poly {:?}",
            now.elapsed()
        );

        // Let f(x) = l(x) * q(x).
        let now = Instant::now();
        let mut f_polynomial = compute_l_mul_q(
            self.engine.fft_engine_pool(),
            &mut commitment.l2_evaluations,
            q_polynomial,
        );
        println!("[VPD Worker Transcript]: Compute F=L*Q {:?}", now.elapsed());

        // The remainer g_polynomial is only used for asserting its degree.
        let now = Instant::now();
        let mut h_polynomial =
            divide_by_vanishing_poly(&mut f_polynomial, self.sumcheck_domain.len());
        println!("[VPD Worker Transcript]: Compute H=F/Z {:?}", now.elapsed());

        let now = Instant::now();
        // Compute the rational constraint polynomial p(x).
        let mut p_polynomial = compute_rational_constraint(
            self.sumcheck_domain.len(),
            &mut f_polynomial,
            &mut h_polynomial,
            output,
        );
        println!(
            "[VPD Worker Transcript]: Compute rational constraint P {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let p_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate(self.ldt_prover.ldt_domain.len(), &mut p_polynomial);
        println!("[VPD Worker Transcript]: Evaluate P {:?}", now.elapsed());

        let now = Instant::now();
        fiat_shamir_engine.inherit_seed();
        let (p_ldt_worker_commitment, _) = self.ldt_prover.worker_commit(
            fiat_shamir_engine,
            self.sumcheck_domain.len(),
            p_evaluations,
            &[],
            true,
        );
        println!(
            "[VPD Worker Transcript]: LDT worker commit {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let mut h_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate(self.ldt_prover.ldt_domain.len(), &mut h_polynomial);
        println!(
            "[VPD Worker Transcript]: Worker evaluate H {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let h_merkle_tree = self
            .ldt_prover
            .multi_merkle_tree_worker
            .commit(&mut h_evaluations);
        println!(
            "[VPD Worker Transcript]: Worker commit H evaluations {:?}",
            now.elapsed()
        );

        for _ in 0..self.num_repetitions {
            // TODO: we must sumary all final constants of all workers to generate
            // slice index instead using the zero value.
            let (index, _) = generate_index(
                fiat_shamir_engine,
                &self.ldt_prover.ldt_domain,
                p_ldt_worker_commitment
                    .central_roots
                    .last()
                    .unwrap()
                    .clone(),
                F::ZERO,
            );

            let now = Instant::now();
            self.contribute_worker_first_query_proof(
                &commitment.l_evaluations,
                &commitment.l_worker_merkle_tree,
                index,
            );
            println!(
                "[VPD Worker Transcript]: Generate indexed L evaluations and send to master {:?}",
                now.elapsed()
            );

            let now = Instant::now();
            self.contribute_worker_first_query_proof(&h_evaluations, &h_merkle_tree, index);
            println!(
                "[VPD Worker Transcript]: Generate indexed H evaluations and send to master {:?}",
                now.elapsed()
            );

            let now = Instant::now();
            self.ldt_prover
                .contribute_transcript(&p_ldt_worker_commitment, index);
            println!(
                "[VPD Worker Transcript]: Contribute LDT transcript {:?}",
                now.elapsed()
            );
        }

        // We do not run GKR protocol for Q_CIRCUIT here because it is identical
        // for every worker.
        // Remember that the Q_CIRCUIT is identical and the input for every
        // worker is also the same.
        // So we only need to run GKR for Q_CIRCUIT at the master worker in one
        // time.
    }

    pub fn contribute_worker_first_query_proof(
        &self,
        evaluations: &CudaSlice<F>,
        worker_merkle_tree: &<Engine::MerkleTree as SisuMerkleTreeEngine<F>>::Tree,
        index: usize,
    ) {
        let op_index = self.ldt_prover.ldt_domain.get_opposite_index_of(index);

        let (positive_evaluations, positve_proof) = self
            .ldt_prover
            .multi_merkle_tree_worker
            .generate_worker_proof(evaluations, worker_merkle_tree, index);

        let (negative_evaluations, negative_proof) = self
            .ldt_prover
            .multi_merkle_tree_worker
            .generate_worker_proof(evaluations, worker_merkle_tree, op_index);

        self.worker
            .send_to_master_and_done(&positive_evaluations)
            .unwrap();
        self.worker.send_to_master_and_done(&positve_proof).unwrap();
        self.worker
            .send_to_master_and_done(&negative_evaluations)
            .unwrap();
        self.worker
            .send_to_master_and_done(&negative_proof)
            .unwrap();
    }
}

pub struct MultiVPDMasterProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    pub q_circuit: Circuit<F>,
    pub sumcheck_domain: Domain<'a, F>,

    engine: &'a Engine,
    master: &'a MasterNode<S, R>,
    num_repetitions: usize,
    ldt_prover: MultiFRIMasterProver<'a, F, Engine, S, R>,
    num_workers: usize,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    MultiVPDMasterProver<'a, F, Engine, S, R>
{
    pub fn new(
        master_node: &'a MasterNode<S, R>,
        engine: &'a Engine,
        mempool: MasterSharedMemPool<'a, S, R>,
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_repetitions: usize,
        num_workers: usize,
    ) -> Self {
        assert!(num_workers.is_power_of_two());
        let mut sumcheck_domain = ldt_domain.clone();

        let mut ratio = ldt_rate;
        while ratio > 1 {
            ratio = ratio / 2;
            sumcheck_domain = sumcheck_domain.square();
        }

        let num_vars = sumcheck_domain.len().ilog2() as usize;
        let q_circuit = generate_gkr_circuit(&sumcheck_domain, 1, num_vars, num_repetitions);

        Self {
            master: master_node,
            engine,
            q_circuit,
            sumcheck_domain,
            num_workers,
            num_repetitions,
            ldt_prover: MultiFRIMasterProver::new(
                master_node,
                engine,
                mempool,
                ldt_domain,
                ldt_rate,
                num_workers,
            ),
        }
    }

    pub fn central_commit(&self) -> MultiVPDCentralCommitment<F, Engine::MerkleTree> {
        MultiVPDCentralCommitment {
            l_central_merkle_tree: self.ldt_prover.multi_merkle_tree_master.commit(),
        }
    }

    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        central_commitment: &MultiVPDCentralCommitment<F, Engine::MerkleTree>,
        input: &[F],
    ) -> Transcript {
        fiat_shamir_engine.begin_protocol();

        let mut transcript = Transcript::default();

        let now = Instant::now();
        let outputs = self.master.recv_from_workers_and_done::<F>().unwrap();
        transcript.serialize_and_push(&outputs);
        println!(
            "[VPD Master Transcript]: receives output {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        fiat_shamir_engine.inherit_seed();
        let p_ldt_commitment =
            self.ldt_prover
                .central_commit(fiat_shamir_engine, self.sumcheck_domain.len(), true);
        transcript.serialize_and_push(&p_ldt_commitment.to_transcript());
        println!(
            "[VPD Master Transcript]: LDT central commit {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let h_central_merkle_tree = self.ldt_prover.multi_merkle_tree_master.commit();
        println!(
            "[VPD Master Transcript]: commit h evaluations {:?}",
            now.elapsed()
        );

        // L CENTRAL MERKLE TREE ROOT IS SENT IN COMMITMENT TRANSCRIPT;
        transcript.serialize_and_push(&h_central_merkle_tree.root());

        let mut params_d = vec![];
        let mut ldt_transcripts = vec![];
        let mut first_l_ldt_queries = vec![];
        let mut first_h_ldt_queries = vec![];
        for _ in 0..self.num_repetitions {
            let (index, _) = generate_index(
                fiat_shamir_engine,
                &self.ldt_prover.ldt_domain,
                p_ldt_commitment.merkle_trees.last().unwrap().root().clone(),
                F::ZERO,
            );

            let now = Instant::now();
            let first_l_ldt_query =
                self.generate_ldt_first_query(index, &central_commitment.l_central_merkle_tree);
            println!(
                "[VPD Master Transcript]: generate l first query proof {:?}",
                now.elapsed()
            );

            let now = Instant::now();
            let first_h_ldt_query = self.generate_ldt_first_query(index, &h_central_merkle_tree);
            println!(
                "[VPD Master Transcript]: generate h first query proof {:?}",
                now.elapsed()
            );

            first_l_ldt_queries.push(first_l_ldt_query);
            first_h_ldt_queries.push(first_h_ldt_query);

            let now = Instant::now();
            let ldt_transcript = self
                .ldt_prover
                .generate_transcript(index, &p_ldt_commitment);

            // We delay pushing ldt_transcript into the final transcript, we
            // will put this after FFT GKR transcript.
            // The reason is that the verifier need to verify FFT GKR before he
            // verify LDT.
            ldt_transcripts.push(ldt_transcript);
            println!(
                "[VPD Master Transcript]: generate LDT transcript {:?}",
                now.elapsed()
            );

            params_d.push(index);
        }

        // At this time, the worker prover completed their protocol, so the
        // fiat-shamir should be ended here. But we have an incompleted FFT GKR,
        // we should clone another fiat-shamir engine here to not affect to
        // next protocol.
        let mut fiat_shamir_engine: FS = fiat_shamir_engine.freeze();

        // Generate proof to k points in ldt domain of q(x).
        //
        // Note that we only run GKR for Q_CIRCUIT once here because it is
        // identical for every worker and no need to run at each worker
        // separately.
        let now = Instant::now();
        let mut circuit_params = CircuitParams::with_domain(self.ldt_prover.ldt_domain.clone());
        circuit_params.d = params_d;
        let q_gkr_prover =
            GKRProver::<_, Engine>::new(self.engine, &self.q_circuit, circuit_params);
        let q_gkr_input = generate_gkr_input(&input[self.num_workers.ilog2() as usize..]);

        // Seed for fiat-shamir of FFT GKR.
        fiat_shamir_engine.set_seed(p_ldt_commitment.merkle_trees.last().unwrap().root());

        let (_, _, _, _, q_gkr_transcript) =
            q_gkr_prover.generate_transcript(&mut fiat_shamir_engine, &q_gkr_input);
        transcript.serialize_and_push(&q_gkr_transcript);
        println!(
            "[VPD Master Transcript]: Generate FFT GKR transcript {:?}",
            now.elapsed()
        );

        for repetition_index in 0..self.num_repetitions {
            transcript.serialize_and_push(&first_l_ldt_queries[repetition_index]);
            transcript.serialize_and_push(&first_h_ldt_queries[repetition_index]);
            transcript.serialize_and_push(&ldt_transcripts[repetition_index]);
        }

        transcript
    }

    pub fn generate_ldt_first_query(
        &self,
        index: usize,
        central_merkle_tree: &<Engine::MerkleTree as SisuMerkleTreeEngine<F>>::Tree,
    ) -> MultiLDTFirstQuery<F> {
        let op_index = self.ldt_prover.ldt_domain.get_opposite_index_of(index);

        let positive_evaluations_hash_path = self
            .ldt_prover
            .multi_merkle_tree_master
            .generate_master_proof(index, self.ldt_prover.ldt_domain.len(), central_merkle_tree);

        let negative_evaluations_hash_path = self
            .ldt_prover
            .multi_merkle_tree_master
            .generate_master_proof(
                op_index,
                self.ldt_prover.ldt_domain.len(),
                central_merkle_tree,
            );

        let positive_worker_evaluations = self.master.recv_from_workers_and_done().unwrap();
        let positive_worker_proofs = self.master.recv_from_workers_and_done().unwrap();
        let negative_worker_evaluations = self.master.recv_from_workers_and_done().unwrap();
        let negative_worker_proofs = self.master.recv_from_workers_and_done().unwrap();

        let positive_query = self.ldt_prover.multi_merkle_tree_master.combine_proofs(
            positive_worker_evaluations,
            positive_worker_proofs,
            positive_evaluations_hash_path,
            index,
            self.ldt_prover.ldt_domain.len(),
        );

        let negative_query = self.ldt_prover.multi_merkle_tree_master.combine_proofs(
            negative_worker_evaluations,
            negative_worker_proofs,
            negative_evaluations_hash_path,
            op_index,
            self.ldt_prover.ldt_domain.len(),
        );

        MultiLDTFirstQuery {
            positive_query,
            negative_query,
        }
    }
}

pub struct MultiVPDVerifier<'a, F: IcicleConvertibleField, H: SisuHasher<F>> {
    q_circuit: Circuit<F>,

    num_workers: usize,
    sumcheck_domain: Domain<'a, F>,
    ldt_verifier: MultiFRIVerifier<'a, F, H>,
    num_repetitions: usize,
}

impl<'a, F: IcicleConvertibleField, H: SisuHasher<F>> MultiVPDVerifier<'a, F, H> {
    pub fn new(
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_repetitions: usize,
        num_workers: usize,
    ) -> Self {
        assert!(ldt_rate.is_power_of_two());
        assert!(num_workers.is_power_of_two());

        let mut sumcheck_domain = ldt_domain.clone();

        let mut ratio = ldt_rate;
        while ratio > 1 {
            ratio = ratio / 2;
            sumcheck_domain = sumcheck_domain.square();
        }

        let num_vars = sumcheck_domain.len().ilog2() as usize;
        let q_circuit = generate_gkr_circuit(&sumcheck_domain, 1, num_vars, num_repetitions);

        let ldt_verifier = MultiFRIVerifier::new(ldt_domain, ldt_rate);

        Self {
            num_workers,
            sumcheck_domain,
            q_circuit,
            num_repetitions,
            ldt_verifier,
        }
    }

    pub fn verify_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        commitment_transcript: TranscriptIter,
        mut transcript: TranscriptIter,
        input: &[F],
    ) -> Result<F, Error> {
        fiat_shamir_engine.begin_protocol();

        assert!(
            input.len() == (self.num_workers.ilog2() + self.sumcheck_domain.len().ilog2()) as usize
        );

        let l_central_root = MultiVPDCentralCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
            commitment_transcript,
        );

        let now = Instant::now();
        let outputs = transcript.pop_and_deserialize::<Vec<F>>();

        let p_ldt_commitment = transcript.pop_and_deserialize::<Transcript>();
        let (p_ldt_commitment_roots, _) =
            FRICentralCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
                p_ldt_commitment.into_iter(),
            );

        let h_central_root = transcript.pop_and_deserialize();

        fiat_shamir_engine.inherit_seed();
        let ldt_random_points = self.ldt_verifier.recover_random_points(
            fiat_shamir_engine,
            true,
            p_ldt_commitment.into_iter(),
        );

        let mut params_d = vec![];
        let mut indexes = vec![];
        let mut op_indexes = vec![];

        for _ in 0..self.num_repetitions {
            let (index, op_index) = generate_index(
                fiat_shamir_engine,
                &self.ldt_verifier.ldt_domain,
                p_ldt_commitment_roots.last().unwrap().clone(),
                F::ZERO,
            );

            params_d.push(index);
            indexes.push(index);
            op_indexes.push(op_index);
        }

        // Run FFT GKR.
        let mut circuit_params = CircuitParams::with_domain(self.ldt_verifier.ldt_domain.clone());
        circuit_params.d = params_d;
        let q_gkr_transcript = transcript.pop_and_deserialize::<Transcript>();
        let q_gkr_input = generate_gkr_input(&input[self.num_workers.ilog2() as usize..]);
        let q_gkr_verifier = GKRVerifier::new(&self.q_circuit, circuit_params);

        // Seed for fiat-shamir of FFT GKR.
        let mut fiat_shamir_engine: FS = fiat_shamir_engine.freeze();
        fiat_shamir_engine.set_seed(p_ldt_commitment_roots.last().unwrap().clone());

        let (_, _, _, _, q_oracle_access) = q_gkr_verifier.verify_transcript(
            &mut fiat_shamir_engine,
            &q_gkr_input,
            q_gkr_transcript.into_iter(),
        )?;
        println!("VERIFY 5: {:?}", now.elapsed());

        for repetition_index in 0..self.num_repetitions {
            let l_ldt_first_query = transcript.pop_and_deserialize::<MultiLDTFirstQuery<F>>();
            l_ldt_first_query.verify::<H>(
                &l_central_root,
                indexes[repetition_index],
                op_indexes[repetition_index],
            )?;

            let h_ldt_first_query = transcript.pop_and_deserialize::<MultiLDTFirstQuery<F>>();
            h_ldt_first_query.verify::<H>(
                &h_central_root,
                indexes[repetition_index],
                op_indexes[repetition_index],
            )?;

            // We pop the p_ldt_query_transcript here and use it later at the end of
            // protocol (after FFT GKR).
            let p_ldt_query = transcript.pop_and_deserialize::<Transcript>();

            let mut first_queries = (vec![], vec![]);

            for worker_index in 0..self.num_workers {
                let p_first_positive_evaluation = compute_p(
                    &self.sumcheck_domain,
                    &self.ldt_verifier.ldt_domain,
                    indexes[repetition_index],
                    &q_oracle_access[repetition_index * 2],
                    &l_ldt_first_query.positive_query.evaluations[worker_index],
                    &h_ldt_first_query.positive_query.evaluations[worker_index],
                    &outputs[worker_index],
                );
                let p_first_negative_evaluation = compute_p(
                    &self.sumcheck_domain,
                    &self.ldt_verifier.ldt_domain,
                    op_indexes[repetition_index],
                    &q_oracle_access[repetition_index * 2 + 1],
                    &l_ldt_first_query.negative_query.evaluations[worker_index],
                    &h_ldt_first_query.negative_query.evaluations[worker_index],
                    &outputs[worker_index],
                );
                first_queries.0.push(p_first_positive_evaluation);
                first_queries.1.push(p_first_negative_evaluation);
            }

            self.ldt_verifier.verify(
                self.sumcheck_domain.len(),
                indexes[repetition_index],
                p_ldt_commitment.into_iter(),
                p_ldt_query.into_iter(),
                first_queries,
                &ldt_random_points,
            )?;
        }
        println!("VERIFY 4: {:?}", now.elapsed());

        let mut output = F::ZERO;
        let worker_num_vars = self.num_workers.ilog2() as usize;
        let worker_input = &input[..worker_num_vars];
        for worker_index in 0..outputs.len() {
            let worker_index_binary = dec2bin(worker_index as u64, worker_num_vars);
            let beta = identity_mle(worker_input, &worker_index_binary);
            output += beta * outputs[worker_index];
        }

        Ok(output)
    }

    // TODO: Until Custom MLE for first round of FFT GKR is completed, we must
    // include transcript to
    pub fn configs<FS: FiatShamirEngine<F>>(
        &'a self,
        fiat_shamir_engine: &mut FS,
        mut transcript: TranscriptIter,
    ) -> MultiVPDConfigs<'a, F> {
        fiat_shamir_engine.begin_protocol();

        let _outputs = transcript.pop_and_deserialize::<Vec<F>>();
        let p_ldt_commitment = transcript.pop_and_deserialize::<Transcript>();
        let (p_ldt_commitment_roots, _) =
            FRICentralCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
                p_ldt_commitment.into_iter(),
            );

        let _h_central_root = transcript.pop_and_deserialize::<F>();

        fiat_shamir_engine.inherit_seed();
        let _ldt_random_points = self.ldt_verifier.recover_random_points(
            fiat_shamir_engine,
            true,
            p_ldt_commitment.into_iter(),
        );

        let mut params_d = vec![];
        for _ in 0..self.num_repetitions {
            let (index, _) = generate_index(
                fiat_shamir_engine,
                &self.ldt_verifier.ldt_domain,
                p_ldt_commitment_roots.last().unwrap().clone(),
                F::ZERO,
            );

            params_d.push(index);
        }

        let mut circuit_params = CircuitParams::with_domain(self.ldt_verifier.ldt_domain.clone());
        circuit_params.d = params_d;
        let q_gkr_verifier = GKRVerifier::new(&self.q_circuit, circuit_params);

        MultiVPDConfigs {
            num_workers: self.num_workers,
            num_repetitions: self.num_repetitions,
            sumcheck_domain: self.sumcheck_domain.clone(),
            ldt_domain: self.ldt_verifier.ldt_domain.clone(),
            fri_configs: self
                .ldt_verifier
                .configs(self.sumcheck_domain.len(), self.num_workers),
            q_gkr_configs: q_gkr_verifier.configs(),
        }
    }

    pub fn extract_commitment(
        &self,
        commitment_transcript: TranscriptIter,
    ) -> MultiVPDCommitmentTranscript<F> {
        let mut vpd_commitment_transcript = MultiVPDCommitmentTranscript::default();

        vpd_commitment_transcript.l_root =
            MultiVPDCentralCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
                commitment_transcript,
            );

        vpd_commitment_transcript
    }

    pub fn extract_transcript(&self, mut transcript: TranscriptIter) -> MultiVPDTranscript<F> {
        let mut vpd_transcript = MultiVPDTranscript::default();

        vpd_transcript.outputs = transcript.pop_and_deserialize::<Vec<F>>();

        let p_ldt_commitment = transcript.pop_and_deserialize::<Transcript>();
        vpd_transcript.p_fri_commitment_transcript = self
            .ldt_verifier
            .extract_commitment(p_ldt_commitment.into_iter());

        vpd_transcript.h_root = transcript.pop_and_deserialize();

        // Run FFT GKR.
        let q_gkr_verifier = GKRVerifier::new(&self.q_circuit, CircuitParams::default());
        let q_gkr_transcript = transcript.pop_and_deserialize::<Transcript>();
        vpd_transcript.q_gkr_transcript =
            q_gkr_verifier.extract_transcript(q_gkr_transcript.into_iter());

        for _ in 0..self.num_repetitions {
            vpd_transcript
                .l_first_queries
                .push(transcript.pop_and_deserialize::<MultiLDTFirstQuery<F>>());

            vpd_transcript
                .h_first_queries
                .push(transcript.pop_and_deserialize::<MultiLDTFirstQuery<F>>());

            // We pop the p_ldt_query_transcript here and use it later at the end of
            // protocol (after FFT GKR).
            let p_ldt_query = transcript.pop_and_deserialize::<Transcript>();
            vpd_transcript.p_fri_query_transcript.push(
                self.ldt_verifier
                    .extract_transcript(p_ldt_query.into_iter()),
            );
        }

        vpd_transcript
    }
}

fn generate_gkr_input<F: Field>(input: &[F]) -> Vec<F> {
    let mut precomputed_input_pow = vec![];
    for t in input {
        precomputed_input_pow.extend_from_slice(&[F::ONE - t, *t]);
    }

    padding_pow_of_two_size(&mut precomputed_input_pow);

    let mut circuit_input = vec![];
    circuit_input.extend(vec![F::ONE; precomputed_input_pow.len()]);
    circuit_input.extend(precomputed_input_pow);

    circuit_input
}

pub fn generate_index<F: Field, FS: FiatShamirEngine<F>>(
    fiat_shamir_engine: &mut FS,
    ldt_domain: &Domain<F>,
    last_p_ldt_root: F,
    final_constant: F,
) -> (usize, usize) {
    let g = fiat_shamir_engine.reduce_g(&[last_p_ldt_root, final_constant]);
    let index = fiat_shamir_engine.hash_to_u64(&g, ldt_domain.len() as u64) as usize;

    let op_index = ldt_domain.get_opposite_index_of(index);

    (index, op_index)
}

pub fn interpolate_q_polynomial_by_fft_engine<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
>(
    engine: &Engine,
    input: &mut CudaSlice<F>,
) -> CudaSlice<F> {
    let input = input.as_ref_host();

    let mut q_evaluations = vec![F::ONE];
    for t in input {
        let mut new_evaluations = vec![];
        for v in q_evaluations {
            let tmp = v * t;
            new_evaluations.push(v - tmp);
            new_evaluations.push(tmp);
        }
        q_evaluations = new_evaluations;
    }

    let q_coeffs = engine
        .fft_engine_pool()
        .interpolate(&mut CudaSlice::on_host(q_evaluations));
    q_coeffs
}

pub struct MultiVPDConfigs<'a, F: Field> {
    num_workers: usize,
    num_repetitions: usize,
    sumcheck_domain: Domain<'a, F>,
    ldt_domain: Domain<'a, F>,
    fri_configs: MultiFRIConfigs<'a, F>,
    q_gkr_configs: GKRConfigs<'a, F>,
}

impl<'a, F: Field> MultiVPDConfigs<'a, F> {
    pub fn gen_code(&self, vpd_index: usize) -> (Vec<FuncGenerator<F>>, Vec<CustomMLEGenerator>) {
        let mut funcs = vec![];

        let mut num_workers_func = FuncGenerator::new("get_vpd__num_workers", vec!["vpd_index"]);
        num_workers_func.add_number(vec![vpd_index], self.num_workers);
        funcs.push(num_workers_func);

        let mut num_repetitions_func =
            FuncGenerator::new("get_vpd__num_repetitions", vec!["vpd_index"]);
        num_repetitions_func.add_number(vec![vpd_index], self.num_repetitions);
        funcs.push(num_repetitions_func);

        let mut first_query_path_size_func =
            FuncGenerator::new("get_vpd__first_query_path_size", vec!["vpd_index"]);
        first_query_path_size_func
            .add_number(vec![vpd_index], self.ldt_domain.len().ilog2() as usize);
        funcs.push(first_query_path_size_func);

        let mut single_input_size_func =
            FuncGenerator::new("get_vpd__single_input_size", vec!["vpd_index"]);
        single_input_size_func
            .add_number(vec![vpd_index], self.sumcheck_domain.len().ilog2() as usize);
        funcs.push(single_input_size_func);

        let mut ldt_domain_size_func =
            FuncGenerator::new("get_vpd__ldt_domain_size", vec!["vpd_index"]);
        ldt_domain_size_func.add_number(vec![vpd_index], self.ldt_domain.len());
        funcs.push(ldt_domain_size_func);

        let mut sumcheck_domain_size_func =
            FuncGenerator::new("get_vpd__sumcheck_domain_size", vec!["vpd_index"]);
        sumcheck_domain_size_func.add_number(vec![vpd_index], self.sumcheck_domain.len());
        funcs.push(sumcheck_domain_size_func);

        let mut sumcheck_domain_size_inv_func =
            FuncGenerator::new("get_vpd__sumcheck_domain_size_inv", vec!["vpd_index"]);
        sumcheck_domain_size_inv_func.add_field(
            vec![vpd_index],
            F::from(self.sumcheck_domain.len() as u64)
                .inverse()
                .unwrap(),
        );
        funcs.push(sumcheck_domain_size_inv_func);

        let precomputed_generators = self.ldt_domain.precomputed_generators();
        let mut precomputed_domain_generator_size_func = FuncGenerator::new(
            "get_vpd__precomputed_ldt_domain_generators_size",
            vec!["vpd_index"],
        );
        precomputed_domain_generator_size_func
            .add_number(vec![vpd_index], precomputed_generators.len());
        funcs.push(precomputed_domain_generator_size_func);

        let mut precomputed_domain_generator_func = FuncGenerator::new(
            "get_vpd__precomputed_ldt_domain_generators",
            vec!["vpd_index"],
        );
        precomputed_domain_generator_func.add_field_array(vec![vpd_index], precomputed_generators);
        funcs.push(precomputed_domain_generator_func);

        let f = self.fri_configs.gen_code(vpd_index);
        funcs.extend(f);

        // All GKR INDEX of VPD has the form of 1000 + vpd_index. It helps us
        // differentiate FFT GKR and GENERAL GKR.
        let (f, mle) = self.q_gkr_configs.gen_code(1000 + vpd_index);
        funcs.extend(f);

        (funcs, mle)
    }
}

#[derive(Default)]
pub struct MultiVPDCommitmentTranscript<F: Field> {
    l_root: F,
}

impl<F: Field> MultiVPDCommitmentTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        vec![self.l_root]
    }
}

#[derive(Default)]
pub struct MultiVPDTranscript<F: Field> {
    outputs: Vec<F>,
    p_fri_commitment_transcript: MultiFRICommitmentTranscript<F>,
    h_root: F,
    q_gkr_transcript: GKRTranscript<F>,
    l_first_queries: Vec<MultiLDTFirstQuery<F>>,
    h_first_queries: Vec<MultiLDTFirstQuery<F>>,
    p_fri_query_transcript: Vec<MultiFRITranscript<F>>,
}

impl<F: Field> MultiVPDTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];
        result.extend_from_slice(&self.outputs);
        result.extend(self.p_fri_commitment_transcript.to_vec());
        result.push(self.h_root);
        result.extend(self.q_gkr_transcript.to_vec());

        for i in 0..self.l_first_queries.len() {
            result.extend(self.l_first_queries[i].to_vec());
            result.extend(self.h_first_queries[i].to_vec());
            result.extend(self.p_fri_query_transcript[i].to_vec());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        channel::{self, MasterNode, WorkerNode},
        cuda_compat::slice::CudaSlice,
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine, Transcript},
        hash::SisuMimc,
        mempool::{MasterSharedMemPool, SharedMemPool, WorkerSharedMemPool},
        multi_vpd::{MultiVPDMasterProver, MultiVPDVerifier, MultiVPDWorkerProver},
        sisu_engine::CPUSisuEngine,
        vpd::generate_ldt_root_domain,
    };

    use ark_std::test_rng;
    use sisulib::{
        circuit::CircuitParams,
        codegen::generator::FileGenerator,
        common::{convert_field_to_string, convert_vec_field_to_string},
        domain::Domain,
        field::{FpSisu, FrBN254},
        mle::dense::SisuDenseMultilinearExtension,
    };
    use std::{
        sync::{Arc, Mutex},
        thread,
        time::Instant,
    };

    #[test]
    fn test_vpd_mle() {
        type Engine = CPUSisuEngine<FrBN254, SisuMimc<FrBN254>>;

        let ldt_rate = 2;
        let num_vars = 4;
        let worker_num_vars = 1;
        let num_repetitions = 2;
        let num_workers = 2usize.pow(worker_num_vars as u32);

        let engine = Engine::new();
        let root_mempool = SharedMemPool::new(num_workers);
        let root_domain = generate_ldt_root_domain(num_vars, ldt_rate);
        let ldt_domain = Domain::from(&root_domain);

        let mut random_evaluations = vec![];
        for i in 0..2usize.pow((num_vars + worker_num_vars) as u32) {
            random_evaluations.push(FpSisu::from((i + 1) as u64));
        }

        let mle = SisuDenseMultilinearExtension::from_slice(&random_evaluations);
        let mut input = vec![];
        for i in 0..worker_num_vars + num_vars {
            input.push(FpSisu::from((i + 1) as u64));
        }
        let verifier_input = input.clone();

        // Step 0: Setup node.
        let (master_sender, master_receiver) = channel::default();
        let mut master_node = MasterNode::from_channel(master_receiver);

        let mut worker_nodes = vec![];
        for i in 0..num_workers {
            let (worker_sender, worker_receiver) = channel::default();
            let worker_node = WorkerNode::from_channel(i, worker_receiver, &master_sender);
            worker_nodes.push(worker_node);
            master_node.add_worker(&worker_sender);
        }

        let now = Instant::now();

        // Step 1: Worker commits.
        let commitment_transcript = Arc::new(Mutex::new(Transcript::default()));
        let query_transcript = Arc::new(Mutex::new(Transcript::default()));
        thread::scope(|scope| {
            for (worker_index, worker_node) in worker_nodes.into_iter().enumerate() {
                let evaluations_size = 2usize.pow(num_vars as u32);
                let worker_evaluations = CudaSlice::on_host(
                    random_evaluations
                        [worker_index * evaluations_size..(worker_index + 1) * evaluations_size]
                        .to_vec(),
                );
                let input = input.clone();
                let engine = engine.clone();
                let root_mempool = root_mempool.clone();

                scope.spawn(move || {
                    let worker_mempool =
                        WorkerSharedMemPool::clone_from(worker_index, &worker_node, root_mempool);

                    let setup_worker = MultiVPDWorkerProver::<_, Engine, _, _>::new(
                        &worker_node,
                        &engine,
                        worker_mempool,
                        ldt_domain.clone(),
                        ldt_rate,
                        num_repetitions,
                        num_workers,
                        worker_index,
                    );
                    let mut commitment = setup_worker.worker_commit(worker_evaluations);
                    let mut prover_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
                    prover_fiat_shamir_engine.set_seed(FpSisu::from(3));

                    setup_worker.contribute_transcript(
                        &mut prover_fiat_shamir_engine,
                        &mut commitment,
                        &CudaSlice::on_host(input),
                    );
                });
            }

            let commitment_transcript = commitment_transcript.clone();
            let query_transcript = query_transcript.clone();
            scope.spawn(move || {
                let master_mempool = MasterSharedMemPool::new(&master_node);
                let master = MultiVPDMasterProver::<_, Engine, _, _>::new(
                    &master_node,
                    &engine,
                    master_mempool,
                    ldt_domain.clone(),
                    ldt_rate,
                    num_repetitions,
                    num_workers,
                );

                // Step 3a: Master central commits.
                let central_commitment = master.central_commit();

                // Step 3b: The master generates setup central commitments
                // (corresponding to step 2).
                let mut prover_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
                prover_fiat_shamir_engine.set_seed(FpSisu::from(3));

                let transcript = master.generate_transcript(
                    &mut prover_fiat_shamir_engine,
                    &central_commitment,
                    &input,
                );

                let mut commitment_transcript = commitment_transcript.lock().unwrap();
                *commitment_transcript = central_commitment.to_transcript();

                let mut query_transcript = query_transcript.lock().unwrap();
                *query_transcript = transcript;
            });
        });

        println!("GENERATE TRANSCRIPT {:?}", now.elapsed());

        // Step 3: Verifier verifies transcript from MasterProver.
        let mut verifier_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir_engine.set_seed(FpSisu::from(3));

        let vpd_verifier = MultiVPDVerifier::<_, SisuMimc<_>>::new(
            ldt_domain.clone(),
            ldt_rate,
            num_repetitions,
            num_workers,
        );

        let commitment_transcript = Arc::try_unwrap(commitment_transcript)
            .unwrap()
            .into_inner()
            .unwrap();
        let query_transcript = Arc::try_unwrap(query_transcript)
            .unwrap()
            .into_inner()
            .unwrap();

        let output = vpd_verifier.verify_transcript(
            &mut verifier_fiat_shamir_engine,
            commitment_transcript.into_iter(),
            query_transcript.into_iter(),
            &verifier_input,
        );
        println!("VERIFY TRANSCRIPT {:?}", now.elapsed());

        let output = output.unwrap();
        assert_eq!(output, mle.evaluate(vec![&verifier_input]));

        let vpd_commitment_transcript =
            vpd_verifier.extract_commitment(commitment_transcript.into_iter());
        let vpd_transcript = vpd_verifier.extract_transcript(query_transcript.into_iter());

        println!(
            "COMMITMENT TRANSCRIPT: {:?}",
            convert_vec_field_to_string(&vpd_commitment_transcript.to_vec())
        );
        println!(
            "TRANSCRIPT: {:?}",
            convert_vec_field_to_string(&vpd_transcript.to_vec())
        );
        println!("INPUT: {:?}", convert_vec_field_to_string(&verifier_input));
        println!("OUTPUT: {:?}", convert_field_to_string(&output));

        let mut config_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
        config_fiat_shamir_engine.set_seed(FpSisu::from(3));
        let configs =
            vpd_verifier.configs(&mut config_fiat_shamir_engine, query_transcript.into_iter());

        let (funcs, mle) = configs.gen_code(0);
        let mut config_file_gen =
            FileGenerator::new("../bls-circom/circuit/sisu/configs.gen.circom");
        config_file_gen.extend_funcs(funcs);
        config_file_gen.create();

        let mut mle_file_gen =
            FileGenerator::<FpSisu>::new("../bls-circom/circuit/sisu/gkr/mle.gen.circom");
        mle_file_gen.include("../configs.gen.circom");
        mle_file_gen.include("./custom_mle.circom");
        mle_file_gen.init_mle();
        mle_file_gen.extend_mle(mle);
        mle_file_gen.create();
    }

    #[test]
    fn test_fft_circuit_custom_mle() {
        let ldt_rate = 2;
        let num_vars = 4;
        let num_repetitions = 2;

        let root_domain = generate_ldt_root_domain::<FpSisu>(num_vars, ldt_rate);
        let ldt_domain = Domain::from(&root_domain);

        let vpd_verifier =
            MultiVPDVerifier::<_, SisuMimc<_>>::new(ldt_domain, ldt_rate, num_repetitions, 1);

        let mut rng = test_rng();
        let mut circuit_params =
            CircuitParams::with_domain(vpd_verifier.ldt_verifier.ldt_domain.clone());
        for i in 0..num_repetitions {
            circuit_params.d.push(i + 3);
        }
        assert!(vpd_verifier.q_circuit.test_mle(&mut rng, &circuit_params));
    }
}
