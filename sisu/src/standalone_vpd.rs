use std::time::Instant;

use crate::{
    cuda_compat::slice::CudaSlice,
    multi_vpd::generate_index,
    sisu_engine::SisuEngine,
    sisu_merkle_tree::{DummyMerkleTreeEngine, ExchangeTree, MerkleProof, SisuMerkleTreeEngine},
    standalone_fri::{StandaloneFRICommitment, StandaloneFRIProver, StandaloneFRIVerifier},
    vpd::{compute_p, generate_gkr_circuit, generate_gkr_input},
};
use ark_ff::Field;

use sisulib::{
    circuit::{Circuit, CircuitParams},
    common::{dec2bin, Error},
    domain::Domain,
    mle::dense::identity_mle,
};

use crate::{
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    gkr::{GKRProver, GKRVerifier},
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
};

pub struct StandaloneLDTFirstQuery<F: Field> {
    positive_evaluations: Vec<F>,
    positive_proof: MerkleProof<F>,
    negative_evaluations: Vec<F>,
    negative_proof: MerkleProof<F>,
}

impl<F: IcicleConvertibleField> StandaloneLDTFirstQuery<F> {
    pub fn from_commitment<'a, Engine: SisuEngine<F>>(
        ldt_prover: &StandaloneFRIProver<'a, F, Engine>,
        commitment: &ExchangeTree<F, Engine::MerkleTree>,
        index: usize,
    ) -> Self {
        let op_slice_index = ldt_prover.ldt_domain.get_opposite_index_of(index);

        let (positive_evaluations, positive_path) = commitment.prove(index);
        let (negative_evaluations, negative_path) = commitment.prove(op_slice_index);

        Self {
            positive_evaluations,
            positive_proof: positive_path,
            negative_evaluations,
            negative_proof: negative_path,
        }
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> Self {
        let positive_evaluations = transcript.pop_and_deserialize::<Vec<F>>();
        let positive_proof = transcript.pop_and_deserialize::<MerkleProof<F>>();
        let negative_evaluations = transcript.pop_and_deserialize::<Vec<F>>();
        let negative_proof = transcript.pop_and_deserialize::<MerkleProof<F>>();

        Self {
            positive_evaluations,
            positive_proof,
            negative_evaluations,
            negative_proof,
        }
    }

    pub fn to_transcript(&self) -> Transcript {
        let mut transcript = Transcript::default();

        transcript.serialize_and_push(&self.positive_evaluations);
        transcript.serialize_and_push(&self.positive_proof);
        transcript.serialize_and_push(&self.negative_evaluations);
        transcript.serialize_and_push(&self.negative_proof);

        transcript
    }

    pub fn verify<H: SisuHasher<F>>(&self, poly_name: &str, root: &F) -> Result<(), Error> {
        let positive_ok = ExchangeTree::<F, DummyMerkleTreeEngine>::verify::<H>(
            &self.positive_evaluations,
            &self.positive_proof,
            root,
        );

        let negative_ok = ExchangeTree::<F, DummyMerkleTreeEngine>::verify::<H>(
            &self.negative_evaluations,
            &self.negative_proof,
            root,
        );

        if !positive_ok || !negative_ok {
            return Err(Error::VPD(format!(
                "invalid proof of {}: positive ({}) negative ({})",
                poly_name, positive_ok, negative_ok
            )));
        }

        Ok(())
    }
}

pub struct StandaloneVPDCommitment<F: IcicleConvertibleField, MT: SisuMerkleTreeEngine<F>> {
    origin_evaluations: Vec<CudaSlice<F>>,

    l2_evaluations: Vec<CudaSlice<F>>,
    l_commitment: ExchangeTree<F, MT>,
}

impl<F: IcicleConvertibleField, MT: SisuMerkleTreeEngine<F>> StandaloneVPDCommitment<F, MT> {
    pub fn to_transcript(&self) -> Transcript {
        Transcript::from_vec(vec![&self.l_commitment.root()])
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> F {
        transcript.pop_and_deserialize()
    }
}

pub struct StandaloneVPDProver<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> {
    pub q_circuit: Circuit<F>,

    pub sumcheck_domain: Domain<'a, F>,

    engine: &'a Engine,
    ldt_prover: StandaloneFRIProver<'a, F, Engine>,
    num_repetitions: usize,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> StandaloneVPDProver<'a, F, Engine> {
    pub fn new(
        engine: &'a Engine,
        ldt_domain: Domain<'a, F>,
        ldt_rate: usize,
        num_repetitions: usize,
    ) -> Self {
        assert!(ldt_rate.is_power_of_two());

        let mut sumcheck_domain = ldt_domain.clone();

        let mut ratio = ldt_rate;
        while ratio > 1 {
            ratio = ratio / 2;
            sumcheck_domain = sumcheck_domain.square();
        }

        let num_vars = sumcheck_domain.len().ilog2() as usize;
        let q_circuit = generate_gkr_circuit(&sumcheck_domain, 1, num_vars, num_repetitions);

        Self {
            sumcheck_domain,
            ldt_prover: StandaloneFRIProver::new(engine, ldt_rate, ldt_domain),
            q_circuit,
            engine,
            num_repetitions,
        }
    }

    pub fn commit(
        &self,
        mut evaluations: Vec<CudaSlice<F>>,
    ) -> StandaloneVPDCommitment<F, Engine::MerkleTree> {
        for i in 0..evaluations.len() {
            assert!(evaluations[i].len() == self.sumcheck_domain.len());
        }

        let now = Instant::now();
        let mut l_coeffs = self
            .engine
            .fft_engine_pool()
            .interpolate_multi(&mut evaluations);
        println!("[VPD Commit]: Interpolate L {:?}", now.elapsed());
        let now = Instant::now();

        // Double the size of sumcheck domain.
        let l2_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate_multi(self.sumcheck_domain.len() * 2, &mut l_coeffs);
        println!("[VPD Commit]: Evaluate L2 {:?}", now.elapsed());

        let now = Instant::now();
        let mut l_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate_multi(self.ldt_prover.ldt_domain.len(), &mut l_coeffs);

        let l_commitment = self
            .engine
            .merkle_tree_engine()
            .exchange_and_create(&mut l_evaluations, evaluations.len());
        println!("[VPD Commit]: Evalaute L and commit {:?}", now.elapsed());

        StandaloneVPDCommitment {
            origin_evaluations: evaluations,
            l2_evaluations,
            l_commitment,
        }
    }

    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        commitment: &mut StandaloneVPDCommitment<F, Engine::MerkleTree>,
        input: &[F],
    ) -> Transcript {
        fiat_shamir_engine.begin_protocol();
        let mut transcript = Transcript::default();

        let input = input[commitment.origin_evaluations.len().ilog2() as usize..].to_vec();

        let now = Instant::now();
        let output = Engine::dense_mle_multi(commitment.origin_evaluations.clone(), &input);
        transcript.serialize_and_push(&output);
        println!("[VPD Transcript]: Compute output: {:?}", now.elapsed());

        // Interpolate q from input.
        let now = Instant::now();
        let q_polynomial = self.engine.interpolate_q(&input);
        println!("[VPD Transcript]: Compute Q: {:?}", now.elapsed());

        let now = Instant::now();
        // Let f(x) = l(x) * q(x).
        let mut f_polynomials = self
            .engine
            .compute_l_mul_q_multi(commitment.l2_evaluations.clone(), q_polynomial);
        println!("[VPD Transcript]: Compute F = L*Q: {:?}", now.elapsed());

        // The vanishing polynomial over sumcheck domain is Z = X^N - 1.
        let now = Instant::now();
        let mut h_polynomials =
            Engine::divide_by_vanishing_poly_multi(&mut f_polynomials, self.sumcheck_domain.len());
        println!("[VPD Transcript]: Compute H=F/Z: {:?}", now.elapsed());

        let mut h_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate_multi(self.ldt_prover.ldt_domain.len(), &mut h_polynomials);

        let now = Instant::now();
        let mut p_polynomials = Engine::compute_rational_constraint_multi(
            self.sumcheck_domain.len(),
            f_polynomials,
            h_polynomials,
            output,
        );
        println!(
            "[VPD Transcript]: Compute rational constraint P: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let p_evaluations = self
            .engine
            .fft_engine_pool()
            .evaluate_multi(self.ldt_prover.ldt_domain.len(), &mut p_polynomials);

        fiat_shamir_engine.inherit_seed();
        let (p_ldt_commitment, _) = self.ldt_prover.commit(
            fiat_shamir_engine,
            self.sumcheck_domain.len(),
            p_evaluations,
            true,
        );
        transcript.serialize_and_push(&p_ldt_commitment.to_transcript());
        println!("[VPD Transcript]: LDT commit: {:?}", now.elapsed());

        let h_commitment = self
            .engine
            .merkle_tree_engine()
            .exchange_and_create(&mut h_evaluations, commitment.origin_evaluations.len());

        transcript.serialize_and_push(&h_commitment.root());

        let mut d_params = vec![];
        let mut l_query_transcripts = vec![];
        let mut h_query_transcripts = vec![];
        let mut p_ldt_transcripts = vec![];
        for _ in 0..self.num_repetitions {
            let (index, _) = generate_index(
                fiat_shamir_engine,
                &self.ldt_prover.ldt_domain,
                p_ldt_commitment.commitments.last().unwrap().root(),
                F::ZERO,
            );
            d_params.push(index);

            let now = Instant::now();
            let l_ldt_first_query = StandaloneLDTFirstQuery::from_commitment(
                &self.ldt_prover,
                &commitment.l_commitment,
                index,
            );
            l_query_transcripts.push(l_ldt_first_query.to_transcript());
            println!(
                "[VPD Transcript]: Commit first L query: {:?}",
                now.elapsed()
            );

            let now = Instant::now();
            let h_ldt_first_query =
                StandaloneLDTFirstQuery::from_commitment(&self.ldt_prover, &h_commitment, index);
            h_query_transcripts.push(h_ldt_first_query.to_transcript());
            println!(
                "[VPD Transcript]: Commit first H query: {:?}",
                now.elapsed()
            );

            let now = Instant::now();
            // Generate proof for low degree test on the rational constraint p.
            let p_ldt_transcript = self
                .ldt_prover
                .generate_transcript(index, &p_ldt_commitment);
            p_ldt_transcripts.push(p_ldt_transcript);
            println!(
                "[VPD Transcript]: Generate LDT transcript: {:?}",
                now.elapsed()
            );
        }

        let now = Instant::now();
        // Seed for fiat-shamir of FFT GKR.
        fiat_shamir_engine.set_seed(p_ldt_commitment.commitments.last().unwrap().root());

        let mut circuit_params = CircuitParams::with_domain(self.ldt_prover.ldt_domain.clone());
        circuit_params.d = d_params;

        let q_gkr_prover = GKRProver::new(self.engine, &self.q_circuit, circuit_params);
        let q_gkr_input = generate_gkr_input(&input);

        let (_, _, _, _, q_gkr_transcript) =
            q_gkr_prover.generate_transcript(fiat_shamir_engine, &q_gkr_input);
        transcript.serialize_and_push(&q_gkr_transcript);
        println!(
            "[VPD Transcript]: Generate FFT GKR transcript: {:?}",
            now.elapsed()
        );

        for i in 0..self.num_repetitions {
            transcript.serialize_and_push(&l_query_transcripts[i]);
            transcript.serialize_and_push(&h_query_transcripts[i]);
            transcript.serialize_and_push(&p_ldt_transcripts[i]);
        }

        transcript
    }
}

pub struct StandaloneVPDVerifier<'a, F: IcicleConvertibleField> {
    q_circuit: Circuit<F>,

    sumcheck_domain: Domain<'a, F>,
    ldt_verifier: StandaloneFRIVerifier<'a, F>,
    num_repetitions: usize,
}

impl<'a, F: IcicleConvertibleField> StandaloneVPDVerifier<'a, F> {
    pub fn new(ldt_domain: Domain<'a, F>, ldt_rate: usize, num_repetitions: usize) -> Self {
        assert!(ldt_rate.is_power_of_two());

        let mut sumcheck_domain = ldt_domain.clone();

        let mut ratio = ldt_rate;
        while ratio > 1 {
            ratio = ratio / 2;
            sumcheck_domain = sumcheck_domain.square();
        }

        let num_vars = sumcheck_domain.len().ilog2() as usize;
        let q_circuit = generate_gkr_circuit(&sumcheck_domain, 1, num_vars, num_repetitions);

        let ldt_verifier = StandaloneFRIVerifier::new(ldt_rate, ldt_domain);

        Self {
            sumcheck_domain,
            q_circuit,
            num_repetitions,
            ldt_verifier,
        }
    }

    pub fn verify_transcript<H: SisuHasher<F>, FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        commitment_transcript: TranscriptIter,
        mut transcript: TranscriptIter,
        input: &[F],
    ) -> Result<F, Error> {
        fiat_shamir_engine.begin_protocol();

        let now = Instant::now();
        let outputs = transcript.pop_and_deserialize::<Vec<F>>();

        let p_ldt_commitment = transcript.pop_and_deserialize::<Transcript>();
        let (p_ldt_commitment_roots, p_ldt_final_constants) =
            StandaloneFRICommitment::<F, DummyMerkleTreeEngine>::from_transcript(
                p_ldt_commitment.into_iter(),
            );

        let num_workers = p_ldt_final_constants.len();
        assert!(input.len() == (num_workers.ilog2() + self.sumcheck_domain.len().ilog2()) as usize);

        let l_central_root = StandaloneVPDCommitment::<F, DummyMerkleTreeEngine>::from_transcript(
            commitment_transcript,
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
        let q_gkr_input = generate_gkr_input(&input[num_workers.ilog2() as usize..]);
        let q_gkr_verifier = GKRVerifier::new(&self.q_circuit, circuit_params);

        // Seed for fiat-shamir of FFT GKR.
        fiat_shamir_engine.set_seed(p_ldt_commitment_roots.last().unwrap().clone());

        let (_, _, _, _, q_oracle_access) = q_gkr_verifier.verify_transcript(
            fiat_shamir_engine,
            &q_gkr_input,
            q_gkr_transcript.into_iter(),
        )?;
        println!("VERIFY 5: {:?}", now.elapsed());

        for repetition_index in 0..self.num_repetitions {
            let l_ldt_first_query = transcript.pop_and_deserialize::<Transcript>();
            let l_ldt_first_query =
                StandaloneLDTFirstQuery::from_transcript(l_ldt_first_query.into_iter());
            l_ldt_first_query.verify::<H>("l_poly", &l_central_root)?;

            let h_ldt_first_query = transcript.pop_and_deserialize::<Transcript>();
            let h_ldt_first_query =
                StandaloneLDTFirstQuery::from_transcript(h_ldt_first_query.into_iter());
            h_ldt_first_query.verify::<H>("h_poly", &h_central_root)?;

            // We pop the p_ldt_query_transcript here and use it later at the end of
            // protocol (after FFT GKR).
            let p_ldt_query = transcript.pop_and_deserialize::<Transcript>();

            let mut first_queries = (vec![], vec![]);

            for worker_index in 0..num_workers {
                let p_first_positive_evaluation = compute_p(
                    &self.sumcheck_domain,
                    &self.ldt_verifier.ldt_domain,
                    indexes[repetition_index],
                    &q_oracle_access[repetition_index * 2],
                    &l_ldt_first_query.positive_evaluations[worker_index],
                    &h_ldt_first_query.positive_evaluations[worker_index],
                    &outputs[worker_index],
                );
                let p_first_negative_evaluation = compute_p(
                    &self.sumcheck_domain,
                    &self.ldt_verifier.ldt_domain,
                    op_indexes[repetition_index],
                    &q_oracle_access[repetition_index * 2 + 1],
                    &l_ldt_first_query.negative_evaluations[worker_index],
                    &h_ldt_first_query.negative_evaluations[worker_index],
                    &outputs[worker_index],
                );
                first_queries.0.push(p_first_positive_evaluation);
                first_queries.1.push(p_first_negative_evaluation);
            }

            self.ldt_verifier.verify::<H>(
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
        let worker_num_vars = num_workers.ilog2() as usize;
        let worker_input = &input[..worker_num_vars];
        for worker_index in 0..outputs.len() {
            let worker_index_binary = dec2bin(worker_index as u64, worker_num_vars);
            let beta = identity_mle(worker_input, &worker_index_binary);
            output += beta * outputs[worker_index];
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cuda_compat::slice::CudaSlice,
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine},
        hash::SisuMimc,
        sisu_engine::GPUFrBN254SisuEngine,
        standalone_vpd::{StandaloneVPDProver, StandaloneVPDVerifier},
        vpd::generate_ldt_root_domain,
    };
    use ark_ff::Field;
    use sisulib::{domain::Domain, field::FpSisu, mle::dense::SisuDenseMultilinearExtension};
    use std::time::Instant;

    #[test]
    fn test_vpd_mle() {
        let num_workers = 4;
        let ldt_rate = 2;
        let num_vars = 4;
        let num_repetitions = 1;

        let root_domain = generate_ldt_root_domain(num_vars, ldt_rate);
        let ldt_domain = Domain::from(&root_domain);

        let mut worker_evaluations = vec![];
        let mut all_evaluations = vec![];
        for i in 0..num_workers {
            let mut tmp = vec![];
            for j in 0..2u64.pow(num_vars as u32) {
                tmp.push(FpSisu::from((i + 1) * (j + 1)))
            }

            all_evaluations.extend_from_slice(&tmp);
            worker_evaluations.push(CudaSlice::on_host(tmp));
        }

        let mut input = vec![];
        for i in 0..num_vars + num_workers.ilog2() as usize {
            input.push(FpSisu::from(i as u32));
        }

        // let engine = CPUSisuEngine::<_, SisuMimc<_>>::new();
        let engine = GPUFrBN254SisuEngine::new();

        let prover = StandaloneVPDProver::new(&engine, ldt_domain, ldt_rate, num_repetitions);
        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);

        let mut commitment = prover.commit(worker_evaluations);

        let now = Instant::now();
        let transcript =
            prover.generate_transcript(&mut prover_fiat_shamir, &mut commitment, &input);
        println!("GENERATE TRANSCRIPT {:?}", now.elapsed());

        let now = Instant::now();
        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);
        let verifier = StandaloneVPDVerifier::new(ldt_domain, ldt_rate, num_repetitions);
        let output = verifier.verify_transcript::<SisuMimc<_>, _>(
            &mut verifier_fiat_shamir,
            commitment.to_transcript().into_iter(),
            transcript.into_iter(),
            &input,
        );
        println!("VERIFY TRANSCRIPT {:?}", now.elapsed());

        let mle = SisuDenseMultilinearExtension::from_slice(&all_evaluations);
        assert_eq!(output.unwrap(), mle.evaluate(vec![&input]));
    }
}
