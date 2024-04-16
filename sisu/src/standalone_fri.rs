use ark_ff::Field;
use sisulib::{common::Error, domain::Domain};

use crate::{
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    fri::fold_v1,
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    sisu_engine::SisuEngine,
    sisu_merkle_tree::{DummyMerkleTreeEngine, ExchangeTree, MerkleProof, SisuMerkleTreeEngine},
};

pub struct StandaloneFRICommitment<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>> {
    pub commitments: Vec<ExchangeTree<F, MTEngine>>,
    pub final_constants: Vec<F>,
    is_ignore_first_evaluations: bool,
}

impl<F: IcicleConvertibleField, MTEngine: SisuMerkleTreeEngine<F>>
    StandaloneFRICommitment<F, MTEngine>
{
    pub fn to_transcript(&self) -> Transcript {
        let commitments: Vec<F> = self.commitments.iter().map(|c| c.root()).collect();

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&commitments);
        transcript.serialize_and_push(&self.final_constants);

        transcript
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> (Vec<F>, Vec<F>) {
        let commitments = transcript.pop_and_deserialize::<Vec<F>>();
        let final_constants = transcript.pop_and_deserialize::<Vec<F>>();
        (commitments, final_constants)
    }
}

pub struct StandaloneFRIProver<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> {
    pub ldt_rate: usize,
    pub ldt_domain: Domain<'a, F>,

    engine: &'a Engine,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> StandaloneFRIProver<'a, F, Engine> {
    pub fn new(engine: &'a Engine, ldt_rate: usize, ldt_domain: Domain<'a, F>) -> Self {
        assert!(
            ldt_domain.len().is_power_of_two(),
            "degree pow must be a power of two",
        );

        Self {
            ldt_rate,
            ldt_domain,
            engine,
        }
    }

    /// This function commits a polynomial for low degree test. It returns
    /// polynomial commitments of all layers and the final constant when the
    /// last polynomial becomes constant polynomial.
    ///
    /// is_ignore_first_evaluations: If it could validate the first evaluations
    /// by another way, enable this parameter.
    pub fn commit<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        degree_bound: usize,
        mut evaluations: Vec<CudaSlice<F>>,
        is_ignore_first_evaluations: bool,
    ) -> (StandaloneFRICommitment<F, Engine::MerkleTree>, Vec<F>) {
        assert!(degree_bound * self.ldt_rate <= self.ldt_domain.len());
        assert!(degree_bound.is_power_of_two());

        fiat_shamir_engine.begin_protocol();

        let slice_size = evaluations.len();
        let mut domain = self.ldt_domain.clone();
        let mut layer_commitments = vec![];
        let mut random_points = vec![];

        for round in 0..degree_bound.ilog2() as usize {
            let mut fiat_shamir_data = F::ZERO;
            if round > 0 || !is_ignore_first_evaluations {
                // We only commits the first polynomial if this isn't a
                // rational constraint polynomial.
                let commitment = self
                    .engine
                    .merkle_tree_engine()
                    .exchange_and_create(&mut evaluations, slice_size);

                layer_commitments.push(commitment);
                fiat_shamir_data = layer_commitments.last().unwrap().root();
            }

            let random_point = fiat_shamir_engine.hash_to_field(&fiat_shamir_data);

            evaluations = self
                .engine
                .fold_multi(&domain, random_point, &mut evaluations);

            random_points.push(random_point);
            domain = domain.square();
        }

        // At the end of commit phase, the polynomial must be a constant.
        let mut final_constants = vec![];
        for i in 0..evaluations.len() {
            let last_evaluations = evaluations[i].as_ref_host();

            for e in last_evaluations {
                assert_eq!(
                    e, &last_evaluations[0],
                    "The last evaluations must be constant"
                );
            }

            final_constants.push(last_evaluations[0]);
        }

        (
            StandaloneFRICommitment {
                is_ignore_first_evaluations,
                commitments: layer_commitments,
                final_constants, // The last constant.
            },
            random_points,
        )
    }

    /// This function returns the query transcript.
    pub fn generate_transcript(
        &self,
        slice_index: usize,
        commitment: &StandaloneFRICommitment<F, Engine::MerkleTree>,
    ) -> Transcript {
        let mut z_index = slice_index; // rename
        let mut transcript = Transcript::default();

        let mut number_of_layers = commitment.commitments.len();
        if commitment.is_ignore_first_evaluations {
            number_of_layers += 1;
        }

        let mut domain = self.ldt_domain.clone();

        for round in 0..number_of_layers {
            let op_z_index = domain.get_opposite_index_of(z_index);

            if round > 0 || !commitment.is_ignore_first_evaluations {
                let mut c_index = round;
                if commitment.is_ignore_first_evaluations {
                    c_index = round - 1;
                }

                let positive_proof = commitment.commitments[c_index].prove(z_index);
                let negative_proof = commitment.commitments[c_index].prove(op_z_index);

                transcript.serialize_and_push(&positive_proof);
                transcript.serialize_and_push(&negative_proof);
            }

            z_index = domain.get_square_index_of(z_index) / 2;
            domain = domain.square();
        }

        transcript
    }
}

pub struct StandaloneFRIVerifier<'a, F: Field> {
    pub ldt_rate: usize,
    pub ldt_domain: Domain<'a, F>,
}

impl<'a, F: IcicleConvertibleField> StandaloneFRIVerifier<'a, F> {
    pub fn new(ldt_rate: usize, ldt_domain: Domain<'a, F>) -> Self {
        assert!(
            ldt_domain.len().is_power_of_two(),
            "degree pow must be a power of two",
        );

        Self {
            ldt_rate,
            ldt_domain,
        }
    }

    pub fn recover_random_points<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        is_ignore_first_evaluations: bool,
        commitment_transcript: TranscriptIter,
    ) -> Vec<F> {
        fiat_shamir_engine.begin_protocol();

        let (commitments, _) = StandaloneFRICommitment::<F, DummyMerkleTreeEngine>::from_transcript(
            commitment_transcript,
        );

        let mut r = vec![];
        if is_ignore_first_evaluations {
            r.push(fiat_shamir_engine.hash_to_field(&F::ZERO));
        }

        for commitment in commitments.iter() {
            r.push(fiat_shamir_engine.hash_to_field(commitment));
        }

        r
    }

    pub fn verify<H: SisuHasher<F>>(
        &self,
        degree_bound: usize,
        index: usize,
        commitment_transcript: TranscriptIter,
        mut query_transcript: TranscriptIter,
        first_queries: (Vec<F>, Vec<F>),
        random_points: &[F],
    ) -> Result<(), Error> {
        let (commitments, final_constants) =
            StandaloneFRICommitment::<F, DummyMerkleTreeEngine>::from_transcript(
                commitment_transcript.clone(),
            );

        assert!(first_queries.0.len() == 0 || first_queries.0.len() == final_constants.len());
        assert!(first_queries.0.len() == first_queries.1.len());
        let is_ignore_first_evaluations = first_queries.0.len() > 0;

        let mut number_of_layers = query_transcript.len() / 2;
        if is_ignore_first_evaluations {
            number_of_layers += 1;
        }

        if degree_bound != 2usize.pow(number_of_layers as u32) {
            return Err(Error::FRI(format!(
                "invalid degree bound ({} != 2^{})",
                degree_bound, number_of_layers
            )));
        }

        let inverse_two = F::from(2u64).inverse().unwrap();
        let mut z_index = index; // rename
        let mut sum = vec![None; final_constants.len()];
        let mut domain = self.ldt_domain.clone();
        for layer_index in 0..number_of_layers {
            let (positive_evaluations, negative_evaluations) =
                if layer_index == 0 && is_ignore_first_evaluations {
                    first_queries.clone()
                } else {
                    let mut c_index = layer_index;
                    if is_ignore_first_evaluations {
                        c_index = layer_index - 1;
                    }

                    let (positive_evaluations, positive_proof) =
                        query_transcript.pop_and_deserialize::<(Vec<F>, MerkleProof<F>)>();

                    let (negative_evaluations, negative_proof) =
                        query_transcript.pop_and_deserialize::<(Vec<F>, MerkleProof<F>)>();

                    if !ExchangeTree::<_, DummyMerkleTreeEngine>::verify::<H>(
                        &positive_evaluations,
                        &positive_proof,
                        &commitments[c_index],
                    ) {
                        return Err(Error::FRI(format!(
                            "invalid positive proof at layer {}",
                            layer_index
                        )));
                    }

                    if !ExchangeTree::<_, DummyMerkleTreeEngine>::verify::<H>(
                        &negative_evaluations,
                        &negative_proof,
                        &commitments[c_index],
                    ) {
                        return Err(Error::FRI(format!(
                            "invalid negative proof at layer {}",
                            layer_index
                        )));
                    }

                    (positive_evaluations, negative_evaluations)
                };

            for idx in 0..final_constants.len() {
                if let Some(sum) = sum[idx] {
                    if sum != positive_evaluations[idx] {
                        return Err(Error::FRI(format!(
                            "The result of layer {} is invalid",
                            layer_index
                        )));
                    }
                }

                sum[idx] = Some(fold_v1(
                    &domain,
                    random_points[layer_index],
                    &inverse_two,
                    z_index,
                    &positive_evaluations[idx],
                    &negative_evaluations[idx],
                ));
            }

            z_index = domain.get_square_index_of(z_index) / 2;
            domain = domain.square();
        }

        // Check the final sum with the constant in commitment.
        for i in 0..sum.len() {
            match sum[i] {
                Some(sum) => {
                    if sum != final_constants[i] {
                        return Err(Error::FRI(String::from(
                            "The final sum is different from final constant",
                        )));
                    }
                }
                None => panic!("invalid sum"),
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use sisulib::{
        common::round_to_pow_of_two,
        domain::{Domain, RootDomain},
        field::FpSisu,
    };

    use crate::{
        cuda_compat::slice::CudaSlice,
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine},
        hash::SisuMimc,
        sisu_engine::GPUFrBN254SisuEngine,
    };

    use super::{StandaloneFRIProver, StandaloneFRIVerifier};

    #[test]
    fn test_fri() {
        let n_workers = 4usize;
        let degree = 6;
        let index = 1;

        let mut poly_coeffs = vec![];
        for i in 0..n_workers {
            let mut poly = vec![];
            for j in 0..degree {
                poly.push(FpSisu::from(((i + 1) * (j + 1)) as u32));
            }

            poly_coeffs.push(poly);
        }

        let ldt_rate = 2;
        let root_domain = RootDomain::new(round_to_pow_of_two(degree) * ldt_rate);
        let fri_domain = Domain::from(&root_domain);

        let mut evaluations = vec![];
        for i in 0..poly_coeffs.len() {
            evaluations.push(CudaSlice::on_host(fri_domain.evaluate(&poly_coeffs[i])));
        }

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);

        //  let engine = CPUSisuEngine::<_, SisuMimc<FpSisu>>::new();
        let engine = GPUFrBN254SisuEngine::new();

        let prover = StandaloneFRIProver::new(&engine, ldt_rate, fri_domain);
        let (commitment, _) = prover.commit(
            &mut prover_fiat_shamir,
            round_to_pow_of_two(degree),
            evaluations.clone(),
            true,
        );
        let query_transcript = prover.generate_transcript(index, &commitment);

        let op_index = fri_domain.get_opposite_index_of(index);
        let mut first_positive_queries = vec![];
        let mut first_negative_queries = vec![];
        for i in 0..n_workers {
            first_positive_queries.push(evaluations[i].at(index));
            first_negative_queries.push(evaluations[i].at(op_index));
        }

        let verifier = StandaloneFRIVerifier::new(ldt_rate, fri_domain);
        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);

        let ldt_random_points = verifier.recover_random_points(
            &mut verifier_fiat_shamir,
            true,
            commitment.to_transcript().into_iter(),
        );

        let result = verifier.verify::<SisuMimc<FpSisu>>(
            round_to_pow_of_two(degree),
            index,
            commitment.to_transcript().into_iter(),
            query_transcript.into_iter(),
            (first_positive_queries, first_negative_queries),
            &ldt_random_points,
        );

        match result {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        }
    }
}
