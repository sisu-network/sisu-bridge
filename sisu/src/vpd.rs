use std::{
    cmp::max,
    ops::{Mul, Sub},
    time::Instant,
};

use crate::{
    cuda_compat::slice::CudaSlice,
    hash::DummyHash,
    sisu_engine::{CPUSisuEngine, SisuEngine},
};
use ark_ff::{FftField, Field};
use ark_poly::{univariate, DenseUVPolynomial, Polynomial};
use sisulib::{
    circuit::{
        Circuit, CircuitGate, CircuitGateType, CircuitLayer, CircuitParams, GateF, GateIndex,
    },
    common::{divide_dense_by_sparse, padding_pow_of_two_size, round_to_pow_of_two, Error},
    domain::{Domain, RootDomain},
    mle::dense::SisuDenseMultilinearExtension,
};

use crate::{
    cuda_compat::fft::{ArkFftEngine, FFTEnginePool, FftEngine},
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    fri::{generate_slice_domain, FRICommitment, FRIProver, FRIVerifier},
    gkr::{GKRProver, GKRVerifier},
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    sisu_merkle_tree::SisuMerkleTree,
    vpd_custom_mle::{
        FFTComputeTForwardYExtension, FFTComputeTMulExtension, FFTDivideInverseForwardYExtension,
    },
};

pub struct LDTFirstQuery<F: Field> {
    root: F,
    positive_evaluations: Vec<F>,
    positive_proof: Vec<F>,
    negative_evaluations: Vec<F>,
    negative_proof: Vec<F>,
}

impl<F: Field + FftField> LDTFirstQuery<F> {
    pub fn from_poly<'a, H: SisuHasher<F>>(
        ldt_prover: &FRIProver<'a, F, H>,
        poly_coeffs: &[F],
        slice_index: usize,
    ) -> Self {
        let evaluations = ldt_prover.ldt_domain.evaluate(&poly_coeffs);
        let commitment = SisuMerkleTree::<F, H>::from_vec(evaluations, ldt_prover.ldt_rate);
        Self::from_commitment(ldt_prover, &commitment, slice_index)
    }
}

impl<F: Field> LDTFirstQuery<F> {
    pub fn from_commitment<'a, H: SisuHasher<F>>(
        ldt_prover: &FRIProver<'a, F, H>,
        commitment: &SisuMerkleTree<F, H>,
        slice_index: usize,
    ) -> Self {
        let slice_domain = generate_slice_domain(&ldt_prover.ldt_domain, ldt_prover.ldt_rate);
        let op_slice_index = slice_domain.get_opposite_index_of(slice_index);

        let (positive_evaluations, positive_path) = commitment.path_of(slice_index);
        let (negative_evaluations, negative_path) = commitment.path_of(op_slice_index);

        Self {
            root: commitment.root(),
            positive_evaluations,
            positive_proof: positive_path,
            negative_evaluations,
            negative_proof: negative_path,
        }
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> Self {
        let commitment = transcript.pop_and_deserialize::<F>();
        let (positive_evaluations, positive_path) =
            transcript.pop_and_deserialize::<(Vec<F>, Vec<F>)>();
        let (negative_evaluations, negative_path) =
            transcript.pop_and_deserialize::<(Vec<F>, Vec<F>)>();

        Self {
            root: commitment,
            positive_evaluations,
            positive_proof: positive_path,
            negative_evaluations,
            negative_proof: negative_path,
        }
    }

    pub fn to_transcript(&self) -> Transcript {
        let mut transcript = Transcript::default();

        transcript.serialize_and_push(&self.root);
        transcript.serialize_and_push(&(
            self.positive_evaluations.as_slice(),
            self.positive_proof.as_slice(),
        ));
        transcript.serialize_and_push(&(
            self.negative_evaluations.as_slice(),
            self.negative_proof.as_slice(),
        ));

        transcript
    }

    pub fn verify<H: SisuHasher<F>>(
        &self,
        poly_name: &str,
        slice_index: usize,
        op_slice_index: usize,
    ) -> Result<(), Error> {
        let positive_ok = SisuMerkleTree::<F, H>::verify_path(
            &self.root,
            slice_index,
            &self.positive_evaluations,
            &self.positive_proof,
        );

        let negative_ok = SisuMerkleTree::<F, H>::verify_path(
            &self.root,
            op_slice_index,
            &self.negative_evaluations,
            &self.negative_proof,
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

pub struct VPDCommitment<'a, F: IcicleConvertibleField, H: SisuHasher<F>> {
    origin_mle: SisuDenseMultilinearExtension<'a, F>,

    l2_evaluations: CudaSlice<F>,
    l_commitment: SisuMerkleTree<F, H>,
}

impl<'a, F: IcicleConvertibleField, H: SisuHasher<F>> VPDCommitment<'a, F, H> {
    pub fn to_transcript(&self) -> Transcript {
        Transcript::from_vec(vec![&self.l_commitment.root()])
    }

    pub fn from_transcript(mut transcript: TranscriptIter) -> F {
        transcript.pop_and_deserialize()
    }
}

pub struct VPDProver<'a, F: Field, H: SisuHasher<F>> {
    pub q_circuit: Circuit<F>,

    pub sumcheck_domain: Domain<'a, F>,
    ldt_prover: FRIProver<'a, F, H>,
}

impl<'a, F: IcicleConvertibleField + FftField, H: SisuHasher<F>> VPDProver<'a, F, H> {
    pub fn setup(ldt_domain: Domain<'a, F>, ldt_rate: usize) -> Self {
        assert!(ldt_rate.is_power_of_two());

        let mut sumcheck_domain = ldt_domain.clone();

        let mut ratio = ldt_rate;
        while ratio > 1 {
            ratio = ratio / 2;
            sumcheck_domain = sumcheck_domain.square();
        }

        let num_vars = sumcheck_domain.len().ilog2() as usize;
        let q_circuit = generate_gkr_circuit(&sumcheck_domain, ldt_rate, num_vars, 1);

        Self {
            sumcheck_domain,
            ldt_prover: FRIProver::setup(ldt_rate, ldt_domain),
            q_circuit,
        }
    }

    pub fn get_verifier(&self) -> VPDVerifier<'a, F, H> {
        VPDVerifier {
            q_circuit: self.q_circuit.clone(),
            sumcheck_domain: self.sumcheck_domain.clone(),
            ldt_verifier: self.ldt_prover.get_verifier(),
        }
    }

    pub fn commit<'b>(&self, evaluations: &'b [F]) -> VPDCommitment<'b, F, H> {
        assert!(evaluations.len() == self.sumcheck_domain.len());
        let now = Instant::now();
        let l_coeffs = self.sumcheck_domain.interpolate(evaluations.to_vec());
        println!("[VPD Commit]: Interpolate L {:?}", now.elapsed());
        let now = Instant::now();

        // Double the size of sumcheck domain.
        let fft_l2_domain = self.sumcheck_domain.sqrt();
        let l2_evaluations = fft_l2_domain.evaluate(&l_coeffs);
        println!("[VPD Commit]: Evaluate L2 {:?}", now.elapsed());

        let now = Instant::now();
        let l_evaluations = self.ldt_prover.ldt_domain.evaluate(&l_coeffs);
        let l_commitment = SisuMerkleTree::from_vec(l_evaluations, self.ldt_prover.ldt_rate);
        println!("[VPD Commit]: Evalaute L and commit {:?}", now.elapsed());

        VPDCommitment {
            origin_mle: SisuDenseMultilinearExtension::from_slice(evaluations),
            l2_evaluations: CudaSlice::on_host(l2_evaluations),
            l_commitment,
        }
    }

    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        commitment: &mut VPDCommitment<F, H>,
        input: &[F],
    ) -> Transcript {
        fiat_shamir_engine.begin_protocol();
        let mut transcript = Transcript::default();

        let now = Instant::now();
        let output = commitment.origin_mle.evaluate(vec![input]);
        transcript.serialize_and_push(&output);
        println!("[VPD Transcript]: Compute output: {:?}", now.elapsed());

        // Interpolate q from input.
        let now = Instant::now();
        let q_polynomial = interpolate_q_polynomial(&self.sumcheck_domain, input);
        println!("[VPD Transcript]: Compute Q: {:?}", now.elapsed());

        let now = Instant::now();
        // Let f(x) = l(x) * q(x).
        let mut f_polynomial = compute_l_mul_q(
            &FFTEnginePool::<F, ArkFftEngine<F>>::new(),
            &mut commitment.l2_evaluations,
            CudaSlice::on_host(q_polynomial),
        );
        println!("[VPD Transcript]: Compute F = L*Q: {:?}", now.elapsed());

        // The vanishing polynomial over sumcheck domain is Z = X^N - 1.
        let now = Instant::now();
        let mut h_polynomial =
            divide_by_vanishing_poly(&mut f_polynomial, self.sumcheck_domain.len());
        println!("[VPD Transcript]: Compute H=F/Z: {:?}", now.elapsed());

        let now = Instant::now();
        let mut p_polynomial = compute_rational_constraint(
            self.sumcheck_domain.len(),
            &mut f_polynomial,
            &mut h_polynomial,
            output,
        );
        println!(
            "[VPD Transcript]: Compute rational constraint P: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let p_evaluations = self
            .ldt_prover
            .ldt_domain
            .evaluate(p_polynomial.as_ref_host());
        fiat_shamir_engine.inherit_seed();
        let (p_ldt_commitment, _) = self.ldt_prover.commit(
            fiat_shamir_engine,
            self.sumcheck_domain.len(),
            p_evaluations,
            &[],
            true,
        );
        transcript.serialize_and_push(&p_ldt_commitment.to_transcript());
        println!("[VPD Transcript]: LDT commit: {:?}", now.elapsed());

        let (slice_index, _) = generate_slice_index(
            fiat_shamir_engine,
            &self.ldt_prover.ldt_domain,
            self.ldt_prover.ldt_rate,
            &p_ldt_commitment.commitments.last().unwrap().root(),
            &p_ldt_commitment.final_constant,
        );

        let now = Instant::now();
        let l_ldt_first_query =
            LDTFirstQuery::from_commitment(&self.ldt_prover, &commitment.l_commitment, slice_index);
        transcript.serialize_and_push(&l_ldt_first_query.to_transcript());
        println!(
            "[VPD Transcript]: Commit first L query: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let h_ldt_first_query =
            LDTFirstQuery::from_poly(&self.ldt_prover, h_polynomial.as_ref_host(), slice_index);
        transcript.serialize_and_push(&h_ldt_first_query.to_transcript());
        println!(
            "[VPD Transcript]: Commit first H query: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        // Seed for fiat-shamir of FFT GKR.
        fiat_shamir_engine.reduce_and_set_seed(&[
            p_ldt_commitment.commitments.last().unwrap().root(),
            p_ldt_commitment.final_constant,
        ]);

        let mut circuit_params = CircuitParams::with_domain(self.ldt_prover.ldt_domain.clone());
        circuit_params.d = vec![slice_index * self.ldt_prover.ldt_rate];

        let engine = CPUSisuEngine::<_, H>::new();
        let q_gkr_prover = GKRProver::new(&engine, &self.q_circuit, circuit_params);
        let q_gkr_input = generate_gkr_input(input);

        let (_, _, _, _, q_gkr_transcript) =
            q_gkr_prover.generate_transcript(fiat_shamir_engine, &q_gkr_input);
        transcript.serialize_and_push(&q_gkr_transcript);
        println!(
            "[VPD Transcript]: Generate FFT GKR transcript: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        // Generate proof for low degree test on the rational constraint p.
        let p_ldt_transcript = self
            .ldt_prover
            .generate_transcript(slice_index, &p_ldt_commitment);
        transcript.serialize_and_push(&p_ldt_transcript);
        println!(
            "[VPD Transcript]: Generate LDT transcript: {:?}",
            now.elapsed()
        );

        transcript
    }
}

pub struct VPDVerifier<'a, F: Field, H: SisuHasher<F>> {
    q_circuit: Circuit<F>,

    sumcheck_domain: Domain<'a, F>,
    ldt_verifier: FRIVerifier<'a, F, H>,
}

impl<'a, F: IcicleConvertibleField, H: SisuHasher<F>> VPDVerifier<'a, F, H> {
    pub fn verify_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        commitment_transcript: TranscriptIter,
        mut transcript: TranscriptIter,
        input: &[F],
    ) -> Result<F, Error> {
        fiat_shamir_engine.begin_protocol();

        let l_root = VPDCommitment::<F, H>::from_transcript(commitment_transcript);

        let now = Instant::now();
        let output = transcript.pop_and_deserialize::<F>();

        let p_ldt_commitment = transcript.pop_and_deserialize::<Transcript>();
        let (p_ldt_commitment_roots, final_constant) =
            FRICommitment::<F, H>::from_transcript(p_ldt_commitment.into_iter());

        fiat_shamir_engine.inherit_seed();
        let ldt_random_points = self.ldt_verifier.recover_random_points(
            fiat_shamir_engine,
            true,
            p_ldt_commitment.into_iter(),
        );

        let (slice_index, op_slice_index) = generate_slice_index(
            fiat_shamir_engine,
            &self.ldt_verifier.ldt_domain,
            self.ldt_verifier.ldt_rate,
            p_ldt_commitment_roots.last().unwrap(),
            &final_constant,
        );

        let l_ldt_first_query_transcript = transcript.pop_and_deserialize::<Transcript>();
        let l_ldt_first_query =
            LDTFirstQuery::<F>::from_transcript(l_ldt_first_query_transcript.into_iter());
        if l_ldt_first_query.root != l_root {
            return Err(Error::VPD(String::from("invalid commitment")));
        }

        l_ldt_first_query.verify::<H>("l_poly", slice_index, op_slice_index)?;

        let h_ldt_first_query_transcript = transcript.pop_and_deserialize::<Transcript>();
        let h_ldt_first_query =
            LDTFirstQuery::<F>::from_transcript(h_ldt_first_query_transcript.into_iter());
        h_ldt_first_query.verify::<H>("h_poly", slice_index, op_slice_index)?;

        // Seed for fiat-shamir of FFT GKR.
        fiat_shamir_engine.reduce_and_set_seed(&[
            p_ldt_commitment_roots.last().unwrap().clone(),
            final_constant,
        ]);

        let mut circuit_params = CircuitParams::with_domain(self.ldt_verifier.ldt_domain.clone());
        circuit_params.d = vec![slice_index * self.ldt_verifier.ldt_rate];
        let q_gkr_transcript = transcript.pop_and_deserialize::<Transcript>();
        let q_gkr_input = generate_gkr_input(input);
        let q_gkr_verifier = GKRVerifier::new(&self.q_circuit, circuit_params);

        let (_, _, _, _, q_oracle_access) = q_gkr_verifier.verify_transcript(
            fiat_shamir_engine,
            &q_gkr_input,
            q_gkr_transcript.into_iter(),
        )?;
        println!("VERIFY 5: {:?}", now.elapsed());

        let p_first_positive_evaluations = compute_p_evaluations(
            &self.sumcheck_domain,
            &self.ldt_verifier.ldt_domain,
            self.ldt_verifier.ldt_rate,
            slice_index,
            &q_oracle_access[..self.ldt_verifier.ldt_rate],
            &l_ldt_first_query.positive_evaluations,
            &h_ldt_first_query.positive_evaluations,
            &output,
        );
        let p_first_negative_evaluations = compute_p_evaluations(
            &self.sumcheck_domain,
            &self.ldt_verifier.ldt_domain,
            self.ldt_verifier.ldt_rate,
            op_slice_index,
            &q_oracle_access[self.ldt_verifier.ldt_rate..],
            &l_ldt_first_query.negative_evaluations,
            &h_ldt_first_query.negative_evaluations,
            &output,
        );

        let p_ldt_query = transcript.pop_and_deserialize::<Transcript>();

        self.ldt_verifier.verify(
            self.sumcheck_domain.len(),
            slice_index,
            p_ldt_commitment.into_iter(),
            p_ldt_query.into_iter(),
            (p_first_positive_evaluations, p_first_negative_evaluations),
            &ldt_random_points,
        )?;
        println!("VERIFY 4: {:?}", now.elapsed());

        Ok(output)
    }
}

pub fn generate_ldt_root_domain<F: Field>(num_vars: usize, ldt_rate: usize) -> RootDomain<F> {
    assert!(ldt_rate.is_power_of_two());

    let n = ldt_rate * 2usize.pow(num_vars as u32);
    RootDomain::new(n)
}

fn build_evaluation_in_circuit<'a, F: Field>(
    q_circuit: &mut Circuit<F>,
    sumcheck_domain: &Domain<'a, F>,
    slice_size: usize,
    num_repetitions: usize,
) {
    let layer = sumcheck_domain.build_short_evaluation_layer(slice_size, num_repetitions);

    q_circuit.replace(0, layer);
}

pub fn generate_gkr_circuit<'a, F: Field>(
    sumcheck_domain: &Domain<'a, F>,
    slice_size: usize,
    num_vars: usize,
    num_repetitions: usize,
) -> Circuit<F> {
    let mut circuit = generate_precompute_circuit(num_vars);

    // Upon this line, the circuit will output evaluations of all monomials
    // of polynomial.
    // Notation:
    //   - n: size of sumcheck domain.
    //   - N: size of ldt domain.
    // [evaluations(n)]

    // Build C = evaluate(interpolate(evaluations, sumcheck_domain), ldt_domain)
    let ifft_circuit = sumcheck_domain.build_interpolation_circuit(&sumcheck_domain); // no padding
    circuit.push_circuit(ifft_circuit);

    // Put a dummy output layer to the circuit, we will replace this output
    // when the protocol determine indices of k oracle access points.

    let now = Instant::now();
    build_evaluation_in_circuit(&mut circuit, &sumcheck_domain, slice_size, num_repetitions);
    println!("[VPD] Build FFT Circuit: {:?}", now.elapsed());
    circuit.finalize();

    circuit
}

fn generate_precompute_circuit<'a, F: Field>(num_vars: usize) -> Circuit<F> {
    let mut circuit = generate_monomial_evaluations::<F>(num_vars);
    let swap_circuit = generate_divide_evaluations::<F>(num_vars);

    circuit.push_circuit(swap_circuit);
    circuit
}

fn generate_monomial_evaluations<'a, F: Field>(num_vars: usize) -> Circuit<F> {
    // Build monomial evaluations. Number of layers = num_vars.
    // Circuit input: [1..., input....]                 size = 2*2*n
    // Circuit output: [(d+1)^n evaluations, input...]  size = 2^n + 2*n
    let input_size = 2 * num_vars;
    let input_max_size = round_to_pow_of_two(input_size);

    let mut circuit = Circuit::new(2 * input_max_size);

    for var_index in 0..num_vars {
        let mut layer = CircuitLayer::default();
        layer
            .mul_ext
            .custom(FFTComputeTMulExtension::new(var_index, num_vars));
        layer
            .forward_y_ext
            .custom(FFTComputeTForwardYExtension::new(var_index, num_vars));

        // Multiply previous evaluations with ti^0, ti^1, ..., ti^d.
        let prev_evaluations_size = 2usize.pow(var_index as u32);
        let prev_evaluations_max_size = max(prev_evaluations_size, input_max_size);

        // Compute evaluation T1, T2, ..., Tn from input t.
        for prev_evaluation_index in 0..prev_evaluations_size {
            for d in 0..2 {
                layer.add_gate(CircuitGate::new(
                    &format!("evaluation {}*{}", prev_evaluation_index, d),
                    CircuitGateType::Mul(GateF::ONE),
                    [
                        GateIndex::Absolute(prev_evaluation_index),
                        GateIndex::Absolute(prev_evaluations_max_size + var_index * 2 + d),
                    ],
                ));
            }
        }

        // Padding dummy gates to evaluation group such that it has the same
        // size with the input group.
        let current_evaluations_size = 2usize.pow(var_index as u32 + 1);
        for i in current_evaluations_size..input_max_size {
            layer.add_gate(CircuitGate::new_dummy(&format!("dummy {}", i)));
        }

        // Map the input from previous layer -> current layer.
        for input_index in 0..input_max_size {
            layer.add_gate(CircuitGate::new(
                &format!("input {}", input_index),
                CircuitGateType::ForwardY(GateF::ONE),
                [
                    GateIndex::Absolute(prev_evaluations_max_size + input_index),
                    GateIndex::Absolute(prev_evaluations_max_size + input_index),
                ],
            ));
        }

        circuit.push_layer(layer);
    }

    circuit
}

fn generate_divide_evaluations<'a, F: Field>(num_vars: usize) -> Circuit<F> {
    // Swap monomial evaluations. Number of layers = 1.
    // Circuit input:  [2^n evaluations, input]
    // Circuit output: [2^n evaluations * inverse_N]

    let evaluations_size = 2usize.pow(num_vars as u32);
    let input_size = round_to_pow_of_two(2usize * num_vars);
    let inverse_n = F::from(evaluations_size as u64).inverse().unwrap();

    let mut circuit = Circuit::new(evaluations_size + input_size);
    let mut layer = CircuitLayer::default();
    layer
        .forward_y_ext
        .custom(FFTDivideInverseForwardYExtension::new(inverse_n, num_vars));

    for index in 0..evaluations_size {
        layer.add_gate(CircuitGate::new(
            &format!("evaluations {}", index),
            CircuitGateType::ForwardY(GateF::C(inverse_n)),
            [GateIndex::Absolute(index), GateIndex::Absolute(index)],
        ));
    }

    circuit.push_layer(layer);
    circuit
}

pub fn interpolate_q_polynomial<F: IcicleConvertibleField + FftField>(
    sumcheck_domain: &Domain<F>,
    input: &[F],
) -> Vec<F> {
    let q_evaluations = CPUSisuEngine::<_, DummyHash>::precompute_bookeeping(
        F::ONE,
        &mut CudaSlice::on_host(input.to_vec()),
    );

    sumcheck_domain.interpolate(q_evaluations.as_host())
}

pub fn generate_gkr_input<F: Field>(input: &[F]) -> Vec<F> {
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

pub fn compute_rational_constraint<F: IcicleConvertibleField>(
    sumcheck_domain_size: usize,
    f_polynomial: &mut CudaSlice<F>,
    h_polynomial: &mut CudaSlice<F>,
    output: F,
) -> CudaSlice<F> {
    // ============= COMPUTE THE RATIONAL CONSTRAINT p(x).
    let f_polynomial =
        univariate::DensePolynomial::from_coefficients_slice(f_polynomial.as_ref_host());
    let h_polynomial =
        univariate::DensePolynomial::from_coefficients_slice(h_polynomial.as_ref_host());

    let domain_size = F::from(sumcheck_domain_size as u64);

    // f1 = |H| * f(x)
    let f1 = f_polynomial.mul(domain_size);

    // f2 = |H| * h(x) * Z(x) = |H| * h(x) * (x^n-1)
    //    = low_coeffs(-|H| * h(x))
    //    + high_coeffs(|H| * h(x)).

    let domain_size_mul_h = h_polynomial.mul(domain_size);
    let mut f2 = domain_size_mul_h.mul(F::ZERO - F::ONE);

    // Padding f2 with zero coeffs.
    f2.coeffs
        .extend(vec![F::ZERO; sumcheck_domain_size - f2.coeffs.len()]);
    // Then, append with h(x), we will have f2.
    f2.coeffs.extend(&domain_size_mul_h.coeffs);

    // f3 = f1 - f2 - output
    let mut f3 = f1.sub(&f2);
    if f3.coeffs.len() == 0 {
        f3.coeffs.push(F::ZERO);
    }

    f3.coeffs[0] -= output;

    // p(x)*x = f3(x)/|H|
    let mut p_mul_x_polynomial = f3.mul(domain_size.inverse().unwrap());
    if p_mul_x_polynomial.coeffs.len() == 0 {
        p_mul_x_polynomial.coeffs.push(F::ZERO);
    }

    assert!(
        p_mul_x_polynomial.coeffs[0] == F::ZERO,
        "The constant coefficent of rational constraint p(x) must be 0"
    );

    CudaSlice::on_host(p_mul_x_polynomial[1..].to_vec())
}

pub fn compute_l_mul_q<'a, F: IcicleConvertibleField, Fft: FftEngine<F>>(
    fft_engine_pool: &FFTEnginePool<F, Fft>,
    l2_evaluations: &mut CudaSlice<F>,
    mut q_polynomial: CudaSlice<F>,
) -> CudaSlice<F> {
    let q2_evaluations = fft_engine_pool.evaluate(q_polynomial.len() * 2, &mut q_polynomial);
    let mut q2_evaluations = q2_evaluations.as_host();
    let l2_evaluations = l2_evaluations.as_ref_host();

    assert_eq!(l2_evaluations.len(), q2_evaluations.len());
    for i in 0..q2_evaluations.len() {
        q2_evaluations[i] *= l2_evaluations[i];
    }

    fft_engine_pool.interpolate(&mut CudaSlice::on_host(q2_evaluations))
}

pub fn divide_by_vanishing_poly<F: IcicleConvertibleField>(
    f_polynomial: &mut CudaSlice<F>,
    vanishing_degree: usize,
) -> CudaSlice<F> {
    let z_polynomial = univariate::SparsePolynomial::from_coefficients_slice(&[
        (vanishing_degree, F::ONE), //  1 * X^N
        (0, F::ZERO - F::ONE),      // -1 * X^0
    ]);

    // The remainer g_polynomial is only used for asserting its degree.
    let f_polynomial =
        univariate::DensePolynomial::from_coefficients_slice(f_polynomial.as_ref_host());
    let (mut h_polynomial, g_polynomial) = divide_dense_by_sparse(&f_polynomial, &z_polynomial);

    assert!(g_polynomial.degree() < vanishing_degree);
    assert!(h_polynomial.degree() < vanishing_degree - 1);

    h_polynomial
        .coeffs
        .extend(vec![F::ZERO; vanishing_degree - h_polynomial.coeffs.len()]);

    CudaSlice::on_host(h_polynomial.coeffs)
}

pub fn compute_p<F: Field>(
    sumcheck_domain: &Domain<F>,
    ldt_domain: &Domain<F>,
    index: usize,
    q: &F,
    l: &F,
    h: &F,
    output: &F,
) -> F {
    let domain_size = F::from(sumcheck_domain.len() as u64);
    let domain_size_inv = domain_size.inverse().unwrap();
    let z_polynomial = univariate::SparsePolynomial::from_coefficients_slice(&[
        (sumcheck_domain.len(), F::ONE), //  1 * X^N
        (0, F::ZERO - F::ONE),           // -1 * X^0
    ]);

    let x = ldt_domain[index];
    let x_inv = ldt_domain[ldt_domain.get_inverse_index_of(index)];
    let z = z_polynomial.evaluate(&x);

    (domain_size * l * q - output - domain_size * z * h) * (domain_size_inv * x_inv)
}

pub fn compute_p_evaluations<F: Field>(
    sumcheck_domain: &Domain<F>,
    ldt_domain: &Domain<F>,
    ldt_rate: usize,
    slice_index: usize,
    q: &[F],
    l: &[F],
    h: &[F],
    output: &F,
) -> Vec<F> {
    let mut p_evaluations = vec![];
    let domain_size = F::from(sumcheck_domain.len() as u64);
    let domain_size_inv = domain_size.inverse().unwrap();
    let z_polynomial = univariate::SparsePolynomial::from_coefficients_slice(&[
        (sumcheck_domain.len(), F::ONE), //  1 * X^N
        (0, F::ZERO - F::ONE),           // -1 * X^0
    ]);
    for i in 0..ldt_rate {
        let point_index = slice_index * ldt_rate + i;
        let q = q[i];
        let x = ldt_domain[point_index];
        let x_inv = ldt_domain[ldt_domain.get_inverse_index_of(point_index)];
        let l = &l[i];
        let h = &h[i];
        let z = z_polynomial.evaluate(&x);

        p_evaluations
            .push((domain_size * l * q - output - domain_size * z * h) * (domain_size_inv * x_inv));
    }

    p_evaluations
}

pub fn generate_slice_index<F: Field, FS: FiatShamirEngine<F>>(
    fiat_shamir_engine: &mut FS,
    ldt_domain: &Domain<F>,
    ldt_rate: usize,
    last_p_ldt_root: &F,
    final_constant: &F,
) -> (usize, usize) {
    let slice_index = fiat_shamir_engine.reduce_and_hash_to_u64(
        &[last_p_ldt_root.clone(), final_constant.clone()],
        (ldt_domain.len() / ldt_rate) as u64,
    ) as usize;

    let slice_domain = generate_slice_domain(ldt_domain, ldt_rate);
    let op_slice_index = slice_domain.get_opposite_index_of(slice_index);

    (slice_index, op_slice_index)
}

#[cfg(test)]
mod tests {
    use crate::{
        fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine},
        vpd::{generate_ldt_root_domain, generate_precompute_circuit, VPDProver},
    };
    use ark_ff::Field;
    use sha2::Sha256;
    use sisulib::{
        circuit::CircuitParams,
        domain::Domain,
        field::{Fp97, FpSisu},
        mle::dense::SisuDenseMultilinearExtension,
    };
    use std::time::Instant;

    #[test]
    fn test_vpd_mle() {
        let ldt_rate = 2;
        let num_vars = 4;

        let root_domain = generate_ldt_root_domain(num_vars, ldt_rate);
        let ldt_domain = Domain::from(&root_domain);

        let mut random_evaluations = vec![];
        for i in 0..2u64.pow(num_vars as u32) {
            random_evaluations.push(FpSisu::from(i))
        }

        let mle = SisuDenseMultilinearExtension::from_slice(&random_evaluations);
        let input = vec![FpSisu::from(1); num_vars];

        let prover = VPDProver::<_, Sha256>::setup(ldt_domain, ldt_rate);
        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);

        println!("Layer size = {}", prover.q_circuit.layers.len());

        let mut commitment = prover.commit(mle.evaluations());

        let now = Instant::now();
        let transcript =
            prover.generate_transcript(&mut prover_fiat_shamir, &mut commitment, &input);
        println!("GENERATE TRANSCRIPT {:?}", now.elapsed());

        let now = Instant::now();
        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);
        let verifier = prover.get_verifier();
        let output = verifier.verify_transcript(
            &mut verifier_fiat_shamir,
            commitment.to_transcript().into_iter(),
            transcript.into_iter(),
            &input,
        );
        println!("VERIFY TRANSCRIPT {:?}", now.elapsed());

        assert_eq!(output.unwrap(), mle.evaluate(vec![&input]));
    }

    #[test]
    fn test_combine_circuit() {
        let circuit = generate_precompute_circuit(3);

        let witness = [
            Fp97::from(1),
            Fp97::from(1),
            Fp97::from(2),
            Fp97::from(1),
            Fp97::from(3),
            Fp97::from(1),
            Fp97::from(4),
            Fp97::from(85), // inverse of 8
            Fp97::from(1),
            Fp97::from(8), // generator
        ];

        let evaluations = circuit.evaluate(&CircuitParams::default(), &witness);

        assert_eq!(
            evaluations.at_layer(0, false),
            vec![
                Fp97::from(85),
                Fp97::from(49),
                Fp97::from(61),
                Fp97::from(50),
                Fp97::from(73),
                Fp97::from(1),
                Fp97::from(25),
                Fp97::from(3),
                Fp97::from(1),
                Fp97::from(8),
                Fp97::from(64),
                Fp97::from(27),
                Fp97::from(22),
                Fp97::from(79),
                Fp97::from(50),
                Fp97::from(12),
                Fp97::from(96),
                Fp97::from(89),
                Fp97::from(33),
                Fp97::from(70),
                Fp97::from(75),
                Fp97::from(18),
                Fp97::from(47),
                Fp97::from(85),
            ]
        )
    }
}
