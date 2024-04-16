/// Let we have N different functions, called f1, f2, ..., fN.
/// We need to run FRI protocol on these N functions. Naively, we only need to
/// run FRI protocol for each function separately, but this approach must
/// generate O(N) merkle paths.
///
/// Assumming that when we run FRI protocol on each function, the random point
/// z_index is identical for every function. We can leverage this assumption to
/// reduce the total merkle paths to O(1).
///
/// In FRI protocol commitment phase, we need to generate merkle tree for all
/// layer evaluations. Instead of committing evaluations of each function
/// separately, we will combine evaluations at the same index into each other,
/// then committing these combinations.
///
/// For example, at layer 0, we has four evaluations corresponding to four
/// different functions:
/// - ev1 = [a0, a1, a2, a3]
/// - ev2 = [b0, b1, b2, b3]
/// - ev3 = [c0, c1, c2, c3]
/// - ev4 = [d0, d1, d2, d3]
///
/// Instead of MT.Commit(ev1), MT.Commit(ev2), MT.Commit(ev3), MT.Commit(ev4).
/// Let:
/// - e_index_0 = [a0, b0, c0, d0]
/// - e_index_1 = [a1, b1, c1, d1]
/// - e_index_2 = [a2, b2, c2, d2]
/// - e_index_3 = [a3, b3, c3, d3]
/// Then run r_i = MT.Commit(e_index_i) where i = 0,1,2,3.
/// And finally, run r = MT.Commit([r0, r1, r2, r3]).
///
/// Now, we has only one merkle root for 4 functions at layer 0.
///
/// At query phase, as we assume that all z_index to query is identical, let it
/// is 2.
/// Instead of MT.Open(ev1, a2), ..., MT.Open(ev4, d2). We will send only one
/// merkle path pi=MT.Open(r, r2) and four evaluation [a2, b2, c2, d2] to the
/// verifier.
///
/// The verifier will check if MT.Verify(r2, pi) === r and
///                            MT.Commit([a2, b2, c2, d2]) === r2.
///
/// If the check passes, Verifier ensures that four evaluation a2, b2, c2, d2 is
/// exact with only one merkle path.
use std::marker::PhantomData;

use ark_ff::Field;
use ark_std::cfg_iter;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sisulib::{
    codegen::generator::FuncGenerator,
    common::{round_to_pow_of_two, Error},
    domain::Domain,
};

use crate::{
    channel::{MasterNode, SisuReceiver, SisuSender, WorkerNode},
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    fri::fold_v1,
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    mempool::{MasterSharedMemPool, WorkerSharedMemPool},
    sisu_engine::SisuEngine,
    sisu_merkle_tree::{
        DummyMerkleTreeEngine, MultiMerkleTreeMaster, MultiMerkleTreeQuery, MultiMerkleTreeWorker,
        SisuMerkleTreeEngine, Tree,
    },
};

pub struct FRIWorkerCommitment<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>> {
    pub central_roots: Vec<F>,
    pub final_constant: F,
    is_ignore_first_evaluations: bool,
    layer_evaluations: Vec<CudaSlice<F>>, // layer - evaluations
    layer_merkle_trees: Vec<MTEngine::Tree>,
}

pub struct FRICentralCommitment<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>> {
    pub merkle_trees: Vec<MTEngine::Tree>, // layer - tree
    pub final_constants: Vec<F>,
    is_ignore_first_evaluations: bool,
}

impl<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>>
    FRICentralCommitment<F, MTEngine>
{
    pub fn to_transcript(&self) -> Transcript {
        let roots: Vec<F> = self.merkle_trees.iter().map(|tree| tree.root()).collect();

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&roots);
        transcript.serialize_and_push(&self.final_constants);

        transcript
    }

    pub fn to_roots(&self) -> Vec<F> {
        self.merkle_trees.iter().map(|tree| tree.root()).collect()
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> (Vec<F>, Vec<F>) {
        let roots = transcript.pop_and_deserialize::<Vec<F>>();
        let final_constants = transcript.pop_and_deserialize::<Vec<F>>();
        (roots, final_constants)
    }
}

pub struct MultiFRIWorkerProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    pub multi_merkle_tree_worker: MultiMerkleTreeWorker<'a, F, Engine::MerkleTree, S, R>,
    pub ldt_domain: Domain<'a, F>,
    pub ldt_rate: usize,

    worker: &'a WorkerNode<S, R>,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    MultiFRIWorkerProver<'a, F, Engine, S, R>
{
    pub fn new(
        worker_node: &'a WorkerNode<S, R>,
        engine: &'a Engine,
        mempool: WorkerSharedMemPool<'a, F, S, R>,
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_workers: usize,
        worker_index: usize,
    ) -> Self {
        assert!(ldt_domain.len().is_power_of_two(),);
        assert!(num_workers.is_power_of_two());
        assert!(worker_index < num_workers);

        Self {
            worker: worker_node,
            ldt_domain,
            ldt_rate,
            multi_merkle_tree_worker: MultiMerkleTreeWorker::new(
                worker_node,
                engine.merkle_tree_engine(),
                mempool,
                num_workers,
                worker_index,
            ),
        }
    }

    /// Generate all evaluations at all layers in FRI commitment phase. However,
    /// instead of committing these evaluations, it only returns them for
    /// communicating with other workers later.
    /// At this phase, this worker must determine which index belongs to which
    /// worker.
    /// Specifically:
    /// - evaluation_index % total_worker == worker_index.
    pub fn worker_commit<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        degree_bound: usize,
        evaluations: CudaSlice<F>,
        trusted_r: &[F],
        is_ignore_first_evaluations: bool,
    ) -> (FRIWorkerCommitment<F, Engine::MerkleTree>, Vec<F>) {
        assert!(
            degree_bound.is_power_of_two() && degree_bound * self.ldt_rate <= self.ldt_domain.len()
        );
        fiat_shamir_engine.begin_protocol();

        let total_layers = degree_bound.ilog2() as usize;

        let mut domain = self.ldt_domain.clone();
        let mut evaluations = evaluations;

        let inverse_two = F::from(2u64).inverse().unwrap();
        let mut random_points = vec![];
        let mut central_roots = vec![];
        let mut layer_evaluations = vec![];
        let mut layer_merkle_trees = vec![];
        for layer_index in 0..total_layers {
            let mut fiat_shamir_data = F::ZERO;
            if layer_index > 0 || !is_ignore_first_evaluations {
                let (central_root, worker_merkle_tree) = self
                    .multi_merkle_tree_worker
                    .commit_and_wait_root(&mut evaluations);

                fiat_shamir_data = central_root;

                central_roots.push(central_root);
                layer_merkle_trees.push(worker_merkle_tree);
            }

            let random_point = if trusted_r.len() > 0 {
                trusted_r[layer_index]
            } else {
                fiat_shamir_engine.hash_to_field(&fiat_shamir_data)
            };

            let half_domain_size = domain.len() / 2;

            let evaluations_ref = evaluations.as_ref_host();

            let new_evaluations: Vec<_> = cfg_iter!(evaluations_ref[..half_domain_size])
                .zip(cfg_iter!(evaluations_ref[half_domain_size..]))
                .enumerate()
                .map(|(i, (positive_evaluation, negative_evaluation))| {
                    fold_v1(
                        &domain,
                        random_point,
                        &inverse_two,
                        i,
                        positive_evaluation,
                        negative_evaluation,
                    )
                })
                .collect();

            if layer_index > 0 || !is_ignore_first_evaluations {
                layer_evaluations.push(evaluations);
            }

            domain = domain.square();
            evaluations = CudaSlice::on_host(new_evaluations);
            random_points.push(random_point);
        }

        // At the end of commit phase, the evaluation has only one constant value.
        let last_evaluations = evaluations.as_ref_host();
        for e in last_evaluations.iter() {
            assert_eq!(
                e, &last_evaluations[0],
                "The last evaluations must be constant"
            );
        }

        // Send the final constant to master to complete the commitment.
        self.worker
            .send_to_master_and_done(&last_evaluations[0])
            .unwrap();

        (
            FRIWorkerCommitment {
                is_ignore_first_evaluations,
                layer_evaluations,
                layer_merkle_trees,
                central_roots,
                final_constant: last_evaluations[0],
            },
            random_points, // All random points.
        )
    }

    pub fn contribute_transcript(
        &self,
        worker_commitment: &FRIWorkerCommitment<F, Engine::MerkleTree>,
        mut index: usize,
    ) {
        let mut num_of_layers = worker_commitment.layer_evaluations.len();
        if worker_commitment.is_ignore_first_evaluations {
            num_of_layers += 1;
        }

        let mut domain = self.ldt_domain.clone();
        for layer_index in 0..num_of_layers {
            if layer_index > 0 || !worker_commitment.is_ignore_first_evaluations {
                let mut commitment_index = layer_index;
                if worker_commitment.is_ignore_first_evaluations {
                    commitment_index = layer_index - 1;
                }

                let (z_evaluations, z_proofs) =
                    self.multi_merkle_tree_worker.generate_worker_proof(
                        &worker_commitment.layer_evaluations[commitment_index],
                        &worker_commitment.layer_merkle_trees[commitment_index],
                        index,
                    );
                self.worker.send_to_master_and_done(&z_evaluations).unwrap();
                self.worker.send_to_master_and_done(&z_proofs).unwrap();

                let op_index = domain.get_opposite_index_of(index);
                let (op_z_evaluations, op_z_proofs) =
                    self.multi_merkle_tree_worker.generate_worker_proof(
                        &worker_commitment.layer_evaluations[commitment_index],
                        &worker_commitment.layer_merkle_trees[commitment_index],
                        op_index,
                    );
                self.worker
                    .send_to_master_and_done(&op_z_evaluations)
                    .unwrap();
                self.worker.send_to_master_and_done(&op_z_proofs).unwrap();
            }

            index = domain.get_square_index_of(index) / 2;
            domain = domain.square();
        }
    }
}

pub struct MultiFRIMasterProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    pub multi_merkle_tree_master: MultiMerkleTreeMaster<'a, F, Engine::MerkleTree, S, R>,
    pub ldt_domain: Domain<'a, F>,
    pub ldt_rate: usize,

    master: &'a MasterNode<S, R>,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    MultiFRIMasterProver<'a, F, Engine, S, R>
{
    pub fn new(
        node: &'a MasterNode<S, R>,
        engine: &'a Engine,
        mempool: MasterSharedMemPool<'a, S, R>,
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_workers: usize,
    ) -> Self {
        assert!(ldt_domain.len().is_power_of_two(),);

        Self {
            master: node,
            multi_merkle_tree_master: MultiMerkleTreeMaster::new(
                node,
                engine.merkle_tree_engine(),
                mempool,
                num_workers,
            ),
            ldt_domain,
            ldt_rate,
        }
    }

    pub fn central_commit<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        degree_bound: usize,
        is_ignore_first_evaluations: bool,
    ) -> FRICentralCommitment<F, Engine::MerkleTree> {
        assert!(degree_bound.is_power_of_two() && degree_bound <= self.ldt_domain.len());
        fiat_shamir_engine.begin_protocol();

        let total_layers = degree_bound.ilog2() as usize;
        let mut merkle_trees = vec![];
        for layer_index in 0..total_layers {
            let mut fiat_shamir_data = F::ZERO;
            if layer_index > 0 || !is_ignore_first_evaluations {
                let tree = self.multi_merkle_tree_master.commit_and_send_root();

                fiat_shamir_data = tree.root();
                merkle_trees.push(tree);
            }

            // Run fiat-shamir to synchronize with worker.
            fiat_shamir_engine.hash_to_field(&fiat_shamir_data);
        }

        let final_constants = self.master.recv_from_workers_and_done().unwrap();

        FRICentralCommitment {
            merkle_trees,
            final_constants,
            is_ignore_first_evaluations,
        }
    }

    /// This function returns the query transcript.
    pub fn generate_transcript(
        &self,
        mut index: usize,
        central_commitment: &FRICentralCommitment<F, Engine::MerkleTree>,
    ) -> Transcript {
        let mut transcript = Transcript::default();

        let mut number_of_layers = central_commitment.merkle_trees.len();
        if central_commitment.is_ignore_first_evaluations {
            number_of_layers += 1;
        }

        let mut domain = self.ldt_domain.clone();

        for layer_index in 0..number_of_layers {
            if layer_index > 0 || !central_commitment.is_ignore_first_evaluations {
                let mut commitment_index = layer_index;
                if central_commitment.is_ignore_first_evaluations {
                    commitment_index = layer_index - 1;
                }

                let z_query = self.construct_merkle_tree_query(
                    index,
                    domain.len(),
                    &central_commitment.merkle_trees[commitment_index],
                );

                let op_index = domain.get_opposite_index_of(index);
                let op_z_query = self.construct_merkle_tree_query(
                    op_index,
                    domain.len(),
                    &central_commitment.merkle_trees[commitment_index],
                );

                transcript.serialize_and_push(&(z_query, op_z_query));
            }

            index = domain.get_square_index_of(index) / 2;
            domain = domain.square();
        }

        transcript
    }

    pub fn construct_merkle_tree_query(
        &self,
        index: usize,
        evaluations_size: usize,
        master_merkle_tree: &<Engine::MerkleTree as SisuMerkleTreeEngine<F>>::Tree,
    ) -> MultiMerkleTreeQuery<F> {
        let master_path = self.multi_merkle_tree_master.generate_master_proof(
            index,
            evaluations_size,
            master_merkle_tree,
        );

        let all_worker_evaluations = self.master.recv_from_workers_and_done().unwrap();
        let all_worker_proofs = self.master.recv_from_workers_and_done().unwrap();
        let query = self.multi_merkle_tree_master.combine_proofs(
            all_worker_evaluations,
            all_worker_proofs,
            master_path,
            index,
            evaluations_size,
        );

        query
    }
}

pub struct MultiFRIVerifier<'a, F: Field, H: SisuHasher<F>> {
    pub ldt_domain: Domain<'a, F>,
    pub ldt_rate: usize,
    __phantom: PhantomData<H>,
}

impl<'a, F: IcicleConvertibleField, H: SisuHasher<F>> MultiFRIVerifier<'a, F, H> {
    pub fn new(ldt_domain: Domain<'a, F>, ldt_rate: usize) -> Self {
        Self {
            ldt_domain,
            ldt_rate,
            __phantom: PhantomData,
        }
    }

    pub fn recover_random_points<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        is_ignore_first_evaluations: bool,
        commitment_transcript: TranscriptIter,
    ) -> Vec<F> {
        fiat_shamir_engine.begin_protocol();

        let (roots, _) = FRICentralCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
            commitment_transcript,
        );

        let mut r = vec![];
        if is_ignore_first_evaluations {
            r.push(fiat_shamir_engine.hash_to_field(&F::ZERO));
        }

        for root in roots.iter() {
            r.push(fiat_shamir_engine.hash_to_field(&root));
        }

        r
    }

    pub fn verify(
        &self,
        degree_bound: usize,
        mut index: usize,
        commitment_transcript: TranscriptIter,
        mut query_transcript: TranscriptIter,
        first_queries: (Vec<F>, Vec<F>),
        random_points: &[F],
    ) -> Result<(), Error> {
        let (roots, final_constants) =
            FRICentralCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
                commitment_transcript.clone(),
            );

        let is_ignore_first_evaluations = first_queries.0.len() > 0;
        let num_workers = final_constants.len();
        let mut number_of_layers = query_transcript.len();
        if is_ignore_first_evaluations {
            number_of_layers += 1;
        }

        if is_ignore_first_evaluations {
            if first_queries.0.len() != num_workers || first_queries.1.len() != num_workers {
                return Err(Error::FRI(format!(
                    "invalid first_queries size {} {}",
                    first_queries.0.len(),
                    first_queries.1.len(),
                )));
            }
        }

        if degree_bound != 2usize.pow(number_of_layers as u32) {
            return Err(Error::FRI(format!(
                "invalid degree bound ({} != {})",
                degree_bound,
                2usize.pow(number_of_layers as u32)
            )));
        }

        let inverse_two = F::from(2u64).inverse().unwrap();
        let mut sum = vec![None; num_workers];
        let mut domain = self.ldt_domain.clone();
        for layer_index in 0..number_of_layers {
            let op_index = domain.get_opposite_index_of(index);

            let (z_evaluations, op_z_evaluations) = if layer_index > 0
                || !is_ignore_first_evaluations
            {
                let mut commitment_index = layer_index;
                if is_ignore_first_evaluations {
                    commitment_index = layer_index - 1;
                }

                let (z_query, op_z_query) = query_transcript
                    .pop_and_deserialize::<(MultiMerkleTreeQuery<F>, MultiMerkleTreeQuery<F>)>();

                let root = &roots[commitment_index];
                z_query.verify::<H>(root, index)?;
                op_z_query.verify::<H>(root, op_index)?;

                (z_query.evaluations, op_z_query.evaluations)
            } else {
                first_queries.clone()
            };

            for worker_index in 0..z_evaluations.len() {
                if let Some(s) = sum[worker_index] {
                    if s != z_evaluations[worker_index] {
                        return Err(Error::FRI(format!(
                            "The result of layer {} at worker {} is invalid",
                            layer_index, worker_index
                        )));
                    }
                }
            }

            for worker_index in 0..z_evaluations.len() {
                sum[worker_index] = Some(fold_v1(
                    &domain,
                    random_points[layer_index].clone(),
                    &inverse_two,
                    index,
                    &z_evaluations[worker_index],
                    &op_z_evaluations[worker_index],
                ));
            }

            index = domain.get_square_index_of(index) / 2;
            domain = domain.square();
        }

        // Check the final sum with the constant in commitment.
        for worker_index in 0..num_workers {
            if sum[worker_index].unwrap() != final_constants[worker_index] {
                return Err(Error::FRI(format!(
                    "The final sum is different from final constant at worker {}",
                    worker_index,
                )));
            }
        }

        Ok(())
    }

    pub fn extract_commitment(
        &self,
        commitment_transcript: TranscriptIter,
    ) -> MultiFRICommitmentTranscript<F> {
        let mut fri_commitment_transcript = MultiFRICommitmentTranscript::default();

        let (roots, final_constants) =
            FRICentralCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
                commitment_transcript.clone(),
            );

        fri_commitment_transcript.layer_roots = roots;
        fri_commitment_transcript.worker_final_constants = final_constants;

        fri_commitment_transcript
    }

    pub fn extract_transcript(
        &self,
        mut query_transcript: TranscriptIter,
    ) -> MultiFRITranscript<F> {
        let mut fri_transcript = MultiFRITranscript::default();

        for _ in 0..query_transcript.len() {
            let (z_query, op_z_query) = query_transcript
                .pop_and_deserialize::<(MultiMerkleTreeQuery<F>, MultiMerkleTreeQuery<F>)>();
            fri_transcript.layer_z_queries.push(z_query);
            fri_transcript.layer_op_z_queries.push(op_z_query);
        }

        fri_transcript
    }

    pub fn configs(&self, degree_bound: usize, num_workers: usize) -> MultiFRIConfigs<'a, F> {
        let mut configs = MultiFRIConfigs::default();
        configs.n_layers = round_to_pow_of_two(degree_bound).ilog2() as usize;
        configs.n_workers = num_workers;
        let mut domain = self.ldt_domain.clone();
        for _ in 0..configs.n_layers {
            configs.layer_domains.push(domain.clone());
            domain = domain.square();
        }

        configs
    }
}

#[derive(Default)]
pub struct MultiFRIConfigs<'a, F: Field> {
    n_layers: usize,
    n_workers: usize,
    layer_domains: Vec<Domain<'a, F>>,
}

impl<'a, F: Field> MultiFRIConfigs<'a, F> {
    pub fn gen_code(&self, vpd_index: usize) -> Vec<FuncGenerator<F>> {
        let mut result = vec![];

        for layer_index in 0..self.layer_domains.len() {
            let mut domain_size_func =
                FuncGenerator::new("get_fri__domain_size", vec!["vpd_index", "layer_index"]);
            domain_size_func.add_number(
                vec![vpd_index, layer_index],
                self.layer_domains[layer_index].len(),
            );
            result.push(domain_size_func);

            let precomputed_generators = self.layer_domains[layer_index].precomputed_generators();
            let mut precomputed_domain_generator_size_func = FuncGenerator::new(
                "get_fri__precomputed_domain_generators_size",
                vec!["vpd_index", "layer_index"],
            );
            precomputed_domain_generator_size_func
                .add_number(vec![vpd_index, layer_index], precomputed_generators.len());
            result.push(precomputed_domain_generator_size_func);

            let mut precomputed_domain_generator_func = FuncGenerator::new(
                "get_fri__precomputed_domain_generators",
                vec!["vpd_index", "layer_index"],
            );
            precomputed_domain_generator_func
                .add_field_array(vec![vpd_index, layer_index], precomputed_generators);
            result.push(precomputed_domain_generator_func);
        }

        for layer_index in 1..self.n_layers {
            let mut query_size_func =
                FuncGenerator::new("get_fri__query_size", vec!["vpd_index", "layer_index"]);
            query_size_func.add_number(
                vec![vpd_index, layer_index],
                self.n_workers // evaluations size
                + 1 // evaluation hash
                + self.layer_domains[layer_index].len().ilog2() as usize, // path size
            );
            result.push(query_size_func);
        }

        let mut n_layers_func = FuncGenerator::new("get_fri__n_layers", vec!["vpd_index"]);
        n_layers_func.add_number(vec![vpd_index], self.n_layers);
        result.push(n_layers_func);

        let mut n_workers_func = FuncGenerator::new("get_fri__n_workers", vec!["vpd_index"]);
        n_workers_func.add_number(vec![vpd_index], self.n_workers);
        result.push(n_workers_func);

        result
    }
}

#[derive(Default)]
pub struct MultiFRITranscript<F: Field> {
    layer_z_queries: Vec<MultiMerkleTreeQuery<F>>,
    layer_op_z_queries: Vec<MultiMerkleTreeQuery<F>>,
}

impl<F: Field> MultiFRITranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];

        for (z_queries, op_z_queries) in self
            .layer_z_queries
            .iter()
            .zip(self.layer_op_z_queries.iter())
        {
            result.extend(z_queries.to_vec());
            result.extend(op_z_queries.to_vec());
        }

        result
    }
}

#[derive(Default)]
pub struct MultiFRICommitmentTranscript<F: Field> {
    layer_roots: Vec<F>,
    worker_final_constants: Vec<F>,
}

impl<F: Field> MultiFRICommitmentTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];

        result.extend_from_slice(&self.layer_roots);
        result.extend_from_slice(&self.worker_final_constants);

        result
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        thread,
    };

    use crate::{
        channel::{self, MasterNode, WorkerNode},
        cuda_compat::slice::CudaSlice,
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine, Transcript},
        hash::SisuMimc,
        mempool::{MasterSharedMemPool, SharedMemPool, WorkerSharedMemPool},
        multi_fri::{MultiFRIMasterProver, MultiFRIVerifier},
        sisu_engine::CPUSisuEngine,
    };
    use sisulib::{
        codegen::generator::FileGenerator,
        common::{convert_field_to_string, convert_vec_field_to_string},
        domain::{Domain, RootDomain},
        field::{FpSisu, FrBN254},
    };

    use super::MultiFRIWorkerProver;

    #[test]
    fn test_fri() {
        type Engine = CPUSisuEngine<FrBN254, SisuMimc<FrBN254>>;

        let num_workers = 4;
        let domain_size = 2usize.pow(4);
        let ldt_rate = 4;
        let polynomial_degree = domain_size / ldt_rate;
        let index = 1;

        let engine = Engine::new();
        let root_mempool = SharedMemPool::new(num_workers);
        let root_domain = RootDomain::new(domain_size);
        let fri_domain = Domain::from(&root_domain);

        let mut evaluations = vec![];
        for i in 0..num_workers {
            let mut p = vec![];
            for j in 0..polynomial_degree {
                p.push(FpSisu::from(((i + 1) * (j + 1)) as u64));
            }
            evaluations.push(fri_domain.evaluate(&p));
        }

        let (master_sender, master_receiver) = channel::default();
        let mut master_node = MasterNode::from_channel(master_receiver);

        let mut worker_nodes = vec![];
        for i in 0..num_workers {
            let (worker_sender, worker_receiver) = channel::default();
            let worker_node = WorkerNode::from_channel(i, worker_receiver, &master_sender);
            worker_nodes.push(worker_node);
            master_node.add_worker(&worker_sender);
        }

        let commitment_transcript = Arc::new(Mutex::new(Transcript::default()));
        let query_transcript = Arc::new(Mutex::new(Transcript::default()));
        let scope_root_domain = root_domain.clone();
        thread::scope(|scope| {
            for (i, worker_node) in worker_nodes.into_iter().enumerate() {
                let worker_evaluations = CudaSlice::on_host(evaluations[i].clone());
                let scope_root_domain = scope_root_domain.clone();
                let root_mempool = root_mempool.clone();
                let engine = engine.clone();

                scope.spawn(move || {
                    let worker_mempool =
                        WorkerSharedMemPool::clone_from(i, &worker_node, root_mempool);

                    let prover = MultiFRIWorkerProver::<_, Engine, _, _>::new(
                        &worker_node,
                        &engine,
                        worker_mempool,
                        Domain::from(&scope_root_domain),
                        ldt_rate,
                        num_workers,
                        i,
                    );

                    let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
                    prover_fiat_shamir.set_seed(FpSisu::from(3));

                    let (commitment, _) = prover.worker_commit(
                        &mut prover_fiat_shamir,
                        polynomial_degree,
                        worker_evaluations,
                        &[],
                        true,
                    );

                    prover.contribute_transcript(&commitment, index);
                });
            }

            let commitment_transcript = commitment_transcript.clone();
            let query_transcript = query_transcript.clone();
            scope.spawn(move || {
                let master_mempool = MasterSharedMemPool::new(&master_node);
                let master_prover = MultiFRIMasterProver::<_, Engine, _, _>::new(
                    &master_node,
                    &engine,
                    master_mempool,
                    Domain::from(&scope_root_domain),
                    ldt_rate,
                    num_workers,
                );

                let mut master_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
                master_fiat_shamir.set_seed(FpSisu::from(3));

                let central_commitent =
                    master_prover.central_commit(&mut master_fiat_shamir, polynomial_degree, true);
                let transcript = master_prover.generate_transcript(index, &central_commitent);

                let mut query_transcript = query_transcript.lock().unwrap();
                (*query_transcript) = transcript;
                let mut commitment_transcript = commitment_transcript.lock().unwrap();
                (*commitment_transcript) = central_commitent.to_transcript();
            });
        });

        let query_transcript = Arc::try_unwrap(query_transcript)
            .unwrap()
            .into_inner()
            .unwrap();
        let commitment_transcript = Arc::try_unwrap(commitment_transcript)
            .unwrap()
            .into_inner()
            .unwrap();

        let verifier =
            MultiFRIVerifier::<FpSisu, SisuMimc<FpSisu>>::new(Domain::from(&root_domain), ldt_rate);
        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::from(3));

        let ldt_random_points = verifier.recover_random_points(
            &mut verifier_fiat_shamir,
            true,
            commitment_transcript.into_iter(),
        );

        let op_index = fri_domain.get_opposite_index_of(index);
        let mut first_queries = (vec![], vec![]);
        for worker_index in 0..num_workers {
            let first_positive_queries = evaluations[worker_index][index];
            let first_negative_queries = evaluations[worker_index][op_index];

            first_queries.0.push(first_positive_queries);
            first_queries.1.push(first_negative_queries);
        }

        let result = verifier.verify(
            polynomial_degree, // degree bound
            index,             // z index
            commitment_transcript.into_iter(),
            query_transcript.into_iter(),
            first_queries.clone(),
            &ldt_random_points,
        );

        let fri_commitment_transcript =
            verifier.extract_commitment(commitment_transcript.into_iter());

        let fri_query_transcript = verifier.extract_transcript(query_transcript.into_iter());

        match result {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        }

        let multi_fri_configs = verifier.configs(polynomial_degree, num_workers);
        let mut file_gen = FileGenerator::new("../bls-circom/circuit/sisu/configs.gen.circom");
        file_gen.extend_funcs(multi_fri_configs.gen_code(0));
        file_gen.create();

        println!(
            "fri_commitment_transcript: {:?}",
            convert_vec_field_to_string(&fri_commitment_transcript.to_vec())
        );

        println!(
            "fri_query_transcript: {:?} ({})",
            convert_vec_field_to_string(&fri_query_transcript.to_vec()),
            &fri_query_transcript.to_vec().len(),
        );

        println!(
            "first_z_evaluations: {:?}",
            convert_vec_field_to_string(&first_queries.0)
        );

        println!(
            "first_op_z_evaluations: {:?}",
            convert_vec_field_to_string(&first_queries.1)
        );

        println!(
            "random_points: {:?}",
            convert_vec_field_to_string(&ldt_random_points)
        );

        println!("index: {:?}", index);

        println!(
            "DOMANI[4]: {:?}",
            convert_field_to_string(&multi_fri_configs.layer_domains[0][4])
        );
    }

    #[test]
    fn test_gen_big_fpsisu() {
        let num_workers = 128;
        let domain_size = 2usize.pow(23);
        let ldt_rate = 32;
        let polynomial_degree = domain_size / ldt_rate;

        let root_domain = RootDomain::new(domain_size);

        let verifier =
            MultiFRIVerifier::<FpSisu, SisuMimc<FpSisu>>::new(Domain::from(&root_domain), ldt_rate);

        let multi_fri_configs = verifier.configs(polynomial_degree, num_workers);
        let mut file_gen = FileGenerator::new("../bls-circom/circuit/sisu/fri/configs.gen.circom");
        file_gen.extend_funcs(multi_fri_configs.gen_code(0));
        file_gen.create();

        println!("{:?}", multi_fri_configs.layer_domains[0][123]);
    }
}
