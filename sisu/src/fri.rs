use std::marker::PhantomData;

use ark_ff::Field;
use sisulib::{common::Error, domain::Domain};

use crate::{
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    hash::SisuHasher,
    sisu_merkle_tree::SisuMerkleTree,
};

pub struct FRICommitment<F: Field, H: SisuHasher<F>> {
    pub commitments: Vec<SisuMerkleTree<F, H>>,
    pub final_constant: F,
    is_ignore_first_evaluations: bool,
}

impl<F: Field, H: SisuHasher<F>> FRICommitment<F, H> {
    pub fn to_transcript(&self) -> Transcript {
        let commitments: Vec<F> = self.commitments.iter().map(|c| c.root()).collect();

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&commitments);
        transcript.serialize_and_push(&self.final_constant);

        transcript
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> (Vec<F>, F) {
        let commitments = transcript.pop_and_deserialize::<Vec<F>>();
        let final_constant = transcript.pop_and_deserialize::<F>();
        (commitments, final_constant)
    }
}

pub struct FRIProver<'a, F: Field, H: SisuHasher<F>> {
    pub ldt_rate: usize,
    pub ldt_domain: Domain<'a, F>,
    __phantom: PhantomData<H>,
}

impl<'a, F: Field, H: SisuHasher<F>> FRIProver<'a, F, H> {
    pub fn setup(ldt_rate: usize, ldt_domain: Domain<'a, F>) -> Self {
        assert!(
            ldt_domain.len().is_power_of_two(),
            "degree pow must be a power of two",
        );

        Self {
            ldt_rate,
            ldt_domain,
            __phantom: PhantomData,
        }
    }

    pub fn get_verifier(&self) -> FRIVerifier<'a, F, H> {
        FRIVerifier {
            ldt_rate: self.ldt_rate,
            ldt_domain: self.ldt_domain.clone(),
            __phantom: PhantomData,
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
        mut evaluations: Vec<F>,
        trusted_r: &[F],
        is_ignore_first_evaluations: bool,
    ) -> (FRICommitment<F, H>, Vec<F>) {
        assert!(degree_bound * self.ldt_rate <= self.ldt_domain.len());
        assert!(degree_bound.is_power_of_two());

        fiat_shamir_engine.begin_protocol();

        let inverse_two = F::from(2u64).inverse().unwrap();
        let mut domain = self.ldt_domain.clone();
        let mut layer_commitments = vec![];
        let mut random_points = vec![];

        for round in 0..degree_bound.ilog2() as usize {
            let mut fiat_shamir_data = F::ZERO;
            if round > 0 || !is_ignore_first_evaluations {
                // We only commits the first polynomial if this isn't a
                // rational constraint polynomial.
                layer_commitments.push(SisuMerkleTree::<F, H>::from_vec(
                    evaluations.clone(),
                    self.ldt_rate,
                ));
                fiat_shamir_data = layer_commitments.last().unwrap().root()
            }

            let random_point = if trusted_r.len() == 0 {
                fiat_shamir_engine.hash_to_field(&fiat_shamir_data)
            } else {
                trusted_r[round as usize]
            };

            let half_domain_size = domain.len() / 2;
            let mut new_evaluations = vec![F::ZERO; half_domain_size];
            for i in 0..half_domain_size {
                new_evaluations[i] = fold_v1(
                    &domain,
                    random_point,
                    &inverse_two,
                    i,
                    &evaluations[i],
                    &evaluations[i + half_domain_size],
                );
            }

            random_points.push(random_point);
            domain = domain.square();
            evaluations = new_evaluations;
        }

        // At the end of commit phase, the polynomial must be a constant.
        for e in evaluations.iter() {
            assert_eq!(e, &evaluations[0], "The last evaluations must be constant");
        }

        (
            FRICommitment {
                is_ignore_first_evaluations,
                commitments: layer_commitments, // Commitments of evaluations.
                final_constant: evaluations[0].clone(), // The last constant.
            },
            random_points,
        )
    }

    /// This function returns the query transcript.
    pub fn generate_transcript(
        &self,
        slice_index: usize,
        commitment: &FRICommitment<F, H>,
    ) -> Transcript {
        let mut z_slice_index = slice_index; // rename
        let mut transcript = Transcript::default();

        let mut slice_domain = generate_slice_domain(&self.ldt_domain, self.ldt_rate);
        let mut number_of_layers = commitment.commitments.len();
        if commitment.is_ignore_first_evaluations {
            number_of_layers += 1;
        }

        for round in 0..number_of_layers {
            let op_z_slice_index = slice_domain.get_opposite_index_of(z_slice_index);

            if round > 0 || !commitment.is_ignore_first_evaluations {
                let mut c_index = round;
                if commitment.is_ignore_first_evaluations {
                    c_index = round - 1;
                }

                let positive_proof = commitment.commitments[c_index].path_of(z_slice_index);
                let negative_proof = commitment.commitments[c_index].path_of(op_z_slice_index);

                transcript.serialize_and_push(&(positive_proof, negative_proof));
            }

            z_slice_index = slice_domain.get_square_index_of(z_slice_index) / 2;
            slice_domain = slice_domain.square();
        }

        transcript
    }
}

pub struct FRIVerifier<'a, F: Field, H: SisuHasher<F>> {
    pub ldt_rate: usize,
    pub ldt_domain: Domain<'a, F>,
    __phantom: PhantomData<H>,
}

impl<'a, F: Field, H: SisuHasher<F>> FRIVerifier<'a, F, H> {
    pub fn recover_random_points<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        is_ignore_first_evaluations: bool,
        commitment_transcript: TranscriptIter,
    ) -> Vec<F> {
        fiat_shamir_engine.begin_protocol();

        let (commitments, _) = FRICommitment::<F, H>::from_transcript(commitment_transcript);

        let mut r = vec![];
        if is_ignore_first_evaluations {
            r.push(fiat_shamir_engine.hash_to_field(&F::ZERO));
        }

        for commitment in commitments.iter() {
            r.push(fiat_shamir_engine.hash_to_field(commitment));
        }

        r
    }

    pub fn verify(
        &self,
        degree_bound: usize,
        slice_index: usize,
        commitment_transcript: TranscriptIter,
        mut query_transcript: TranscriptIter,
        first_queries: (Vec<F>, Vec<F>),
        random_points: &[F],
    ) -> Result<(), Error> {
        assert!(first_queries.0.len() == 0 || first_queries.0.len() == self.ldt_rate);
        assert!(first_queries.0.len() == first_queries.1.len());
        let is_ignore_first_evaluations = first_queries.0.len() > 0;

        let mut number_of_layers = query_transcript.len();
        if is_ignore_first_evaluations {
            number_of_layers += 1;
        }

        if degree_bound != 2usize.pow(number_of_layers as u32) {
            return Err(Error::FRI(format!(
                "invalid degree bound ({} != 2^{})",
                degree_bound, number_of_layers
            )));
        }

        let (commitments, final_constant) =
            FRICommitment::<F, H>::from_transcript(commitment_transcript.clone());

        let inverse_two = F::from(2u64).inverse().unwrap();
        let mut z_slice_index = slice_index; // rename
        let mut sum = vec![None; self.ldt_rate];
        let mut domain = self.ldt_domain.clone();
        let mut slice_domain = generate_slice_domain(&self.ldt_domain, self.ldt_rate);
        for layer_index in 0..number_of_layers {
            let op_z_slice_index = slice_domain.get_opposite_index_of(z_slice_index);

            let (positive_evaluations, negative_evaluations) = if layer_index == 0
                && is_ignore_first_evaluations
            {
                first_queries.clone()
            } else {
                let mut c_index = layer_index;
                if is_ignore_first_evaluations {
                    c_index = layer_index - 1;
                }

                let (
                    (positive_evaluations, positive_proof),
                    (negative_evaluations, negative_proof),
                ) = query_transcript.pop_and_deserialize::<((Vec<F>, Vec<F>), (Vec<F>, Vec<F>))>();

                let positive_ok = SisuMerkleTree::<F, H>::verify_path(
                    &commitments[c_index],
                    z_slice_index,
                    &positive_evaluations,
                    &positive_proof,
                );

                let negative_ok = SisuMerkleTree::<F, H>::verify_path(
                    &commitments[c_index],
                    op_z_slice_index,
                    &negative_evaluations,
                    &negative_proof,
                );

                if !positive_ok || !negative_ok {
                    return Err(Error::FRI(format!(
                        "invalid proof at layer {}: positive ({}) negative({})",
                        layer_index, positive_ok, negative_ok
                    )));
                }

                (positive_evaluations, negative_evaluations)
            };

            let first_z_index = z_slice_index * self.ldt_rate;
            for idx in 0..self.ldt_rate {
                if let Some(sum) = sum[idx] {
                    if sum != positive_evaluations[idx] {
                        return Err(Error::FRI(format!(
                            "The result of layer {} is invalid",
                            layer_index
                        )));
                    }
                }

                let next_sum = fold_v1(
                    &domain,
                    random_points[layer_index],
                    &inverse_two,
                    first_z_index + idx,
                    &positive_evaluations[idx],
                    &negative_evaluations[idx],
                );

                sum[idx] = Some(next_sum);
            }

            z_slice_index = slice_domain.get_square_index_of(z_slice_index) / 2;
            slice_domain = slice_domain.square();
            domain = domain.square();
        }

        // Check the final sum with the constant in commitment.
        for i in 0..sum.len() {
            match sum[i] {
                Some(sum) => {
                    if sum != final_constant {
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

pub fn generate_slice_domain<'a, F: Field>(
    ldt_domain: &Domain<'a, F>,
    ldt_rate: usize,
) -> Domain<'a, F> {
    let mut result = ldt_domain.clone();
    for _ in 0..ldt_rate.ilog2() {
        result = result.square();
    }

    result
}

// A higher performance version of verifier_fold.
pub fn fold_v2<F: Field>(
    inverse_two: &F, // for performance purpose
    r_inverse_p: &F, // for performance purpose
    positive_evaluation: &F,
    negative_evaluation: &F,
) -> F {
    (*r_inverse_p * (*positive_evaluation - negative_evaluation)
        + positive_evaluation
        + negative_evaluation)
        * inverse_two
}

pub fn fold_v1<F: Field>(
    domain: &Domain<F>,
    r: F,
    inverse_two: &F, // for performance purpose
    point_index: usize,
    positive_evaluation: &F,
    negative_evaluation: &F,
) -> F {
    let inverse_p = domain[domain.get_inverse_index_of(point_index)];

    // (r * (*positive_evaluation - negative_evaluation)
    //     + domain[point_index] * (*positive_evaluation + negative_evaluation))
    //     * (inverse_p * inverse_two)

    *inverse_two
        * (r * inverse_p * (*positive_evaluation - negative_evaluation)
            + positive_evaluation
            + negative_evaluation)
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use sha2::Sha256;
    use sisulib::{
        domain::{Domain, RootDomain},
        field::FpSisu,
    };

    use crate::{
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine},
        fri::FRIProver,
    };

    use super::generate_slice_domain;

    #[test]
    fn test_fri() {
        let polynomial_coeffs = vec![
            FpSisu::from(1),
            FpSisu::from(2),
            FpSisu::from(3),
            FpSisu::from(4),
            FpSisu::from(5),
            FpSisu::from(6),
        ];

        let ldt_rate = 2;
        let root_domain = RootDomain::new(16);
        let fri_domain = Domain::from(&root_domain);

        let evaluations = fri_domain.evaluate(&polynomial_coeffs);

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);

        let slice_index = 1;

        let prover = FRIProver::<_, Sha256>::setup(ldt_rate, fri_domain);
        let (commitment, _) = prover.commit(&mut prover_fiat_shamir, 8, evaluations, &[], false);
        let query_transcript = prover.generate_transcript(slice_index, &commitment);

        let verifier = prover.get_verifier();
        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);

        let ldt_random_points = verifier.recover_random_points(
            &mut verifier_fiat_shamir,
            false,
            commitment.to_transcript().into_iter(),
        );

        let result = verifier.verify(
            8,
            slice_index,
            commitment.to_transcript().into_iter(),
            query_transcript.into_iter(),
            (vec![], vec![]),
            &ldt_random_points,
        );

        match result {
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn test_fri_ignore_first_evaluations() {
        let polynomial_coeffs = vec![
            FpSisu::from(1),
            FpSisu::from(2),
            FpSisu::from(3),
            FpSisu::from(4),
            FpSisu::from(5),
            FpSisu::from(6),
        ];

        let ldt_rate = 2;
        let root_domain = RootDomain::new(16);
        let ldt_domain = Domain::from(&root_domain);

        let slice_domain = generate_slice_domain(&ldt_domain, ldt_rate);
        let slice_index = 1;
        let op_slice_index = slice_domain.get_opposite_index_of(slice_index);

        let evaluations = ldt_domain.evaluate(&polynomial_coeffs);
        let first_positive_queries =
            evaluations[slice_index * ldt_rate..(slice_index + 1) * ldt_rate].to_vec();
        let first_negative_queries =
            evaluations[op_slice_index * ldt_rate..(op_slice_index + 1) * ldt_rate].to_vec();

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);

        let prover = FRIProver::<_, Sha256>::setup(ldt_rate, ldt_domain);
        let (commitment, _) =
            prover.commit(&mut prover_fiat_shamir, 8, evaluations.clone(), &[], true);
        let query_transcript = prover.generate_transcript(slice_index, &commitment);

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);

        let verifier = prover.get_verifier();
        let ldt_random_points = verifier.recover_random_points(
            &mut verifier_fiat_shamir,
            true,
            commitment.to_transcript().into_iter(),
        );

        let result = verifier.verify(
            8,
            slice_index,
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
