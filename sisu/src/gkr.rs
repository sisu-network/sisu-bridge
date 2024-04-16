use std::time::Instant;

use ark_ff::Field;
use sisulib::{
    circuit::{Circuit, CircuitEvaluations, CircuitParams},
    codegen::generator::{CustomMLEGenerator, FuncGenerator},
    common::{padding_pow_of_two_size, round_to_pow_of_two, Error},
    mle::dense::SisuDenseMultilinearExtension,
};

use crate::{
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    gkr_sumcheck::{precompute_bookeeping_with_linear_combination, GKRSumcheckProver},
    icicle_converter::IcicleConvertibleField,
    sisu_engine::SisuEngine,
    sumcheck::{
        MultiSumcheckConfigs, MultiSumcheckTranscript, MultiSumcheckVerifier,
        ProductSumcheckTranscript,
    },
};

pub struct GKRProver<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> {
    engine: &'a Engine,
    circuit: &'a Circuit<F>,
    circuit_params: CircuitParams<'a, F>,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> GKRProver<'a, F, Engine> {
    pub fn new(
        engine: &'a Engine,
        circuit: &'a Circuit<F>,
        circuit_params: CircuitParams<'a, F>,
    ) -> Self {
        Self {
            engine,
            circuit,
            circuit_params,
        }
    }

    /// Run a sumcheck phase 1 sumcheck on V(g, x, y) where y run on boolean hypercube.
    fn sumcheck_phase_1<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        sumcheck_prover: &mut GKRSumcheckProver<F, Engine>,
        round: usize,
    ) -> (CudaSlice<F>, Vec<F>, Option<F>, Transcript) {
        sumcheck_prover.reset();

        // GATE CONSTANT: one(g, x, y) * constant_1(x) * constant_1(y)
        let constant_evaluations = &self.circuit.layer(round).constant_ext.evaluations;
        sumcheck_prover.add(constant_evaluations, false);

        // GATE MUL: mul(g, x, y) * V(x) * V(y).
        let mul_evaluations = &self.circuit.layer(round).mul_ext.evaluations;
        sumcheck_prover.add(mul_evaluations, true);

        // GATE FORWARD X: forward(g, x, y) * V(x) * V_constant_1(y).
        let forward_x_evaluations = &self.circuit.layer(round).forward_x_ext.evaluations;
        sumcheck_prover.add(forward_x_evaluations, true);

        // GATE FORWARD Y: forward(g, x, y) * V_constant_1(x) * V(y).
        let forward_y_evaluations = &self.circuit.layer(round).forward_y_ext.evaluations;
        sumcheck_prover.add(forward_y_evaluations, false);

        let (random_u, f2_at_u, wu, transcript) = sumcheck_prover.run(fiat_shamir_engine);

        (CudaSlice::on_host(random_u), f2_at_u, wu, transcript)
    }

    /// Run a sumcheck phase 2 sumcheck on V(g, u, y).
    fn sumcheck_phase_2<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        sumcheck_prover: &mut GKRSumcheckProver<F, Engine>,
        round: usize,
    ) -> (CudaSlice<F>, Option<F>, Transcript) {
        // GATE CONSTANT: one(g, x, y) * constant_1(x) * constant_1(y)
        let constant_evaluations = &self.circuit.layer(round).constant_ext.evaluations;
        sumcheck_prover.add(constant_evaluations, false);

        // GATE MUL: mul(g, x, y) * V(x) * V(y).
        let mul_evaluations = &self.circuit.layer(round).mul_ext.evaluations;
        sumcheck_prover.add(mul_evaluations, true);

        // GATE FORWARD X: forward(g, x, y) * V(x) * V_constant_1(y).
        let forward_x_evaluations = &self.circuit.layer(round).forward_x_ext.evaluations;
        sumcheck_prover.add(forward_x_evaluations, false);

        // GATE FORWARD Y: forward(g, x, y) * V_constant_1(x) * V(y).
        let forward_y_evaluations = &self.circuit.layer(round).forward_y_ext.evaluations;
        sumcheck_prover.add(forward_y_evaluations, true);

        let (random_v, _, wv, transcript) = sumcheck_prover.run(fiat_shamir_engine);

        (CudaSlice::on_host(random_v), wv, transcript)
    }

    /// Run a sumcheck on V(g) = f(g, x, y).
    fn run_sumcheck<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        w_next_evaluations: CudaSlice<F>,
        bookeeping_identity_g: CudaSlice<F>,
        fiat_shamir_seed: F,
    ) -> (CudaSlice<F>, CudaSlice<F>, Option<F>, Option<F>, Transcript) {
        let mut sumcheck_prover = GKRSumcheckProver::new(
            self.engine,
            &self.circuit_params,
            bookeeping_identity_g,
            w_next_evaluations,
        );

        fiat_shamir_engine.set_seed(fiat_shamir_seed);
        sumcheck_prover.init_phase_1();
        let (mut random_u, f2_at_u, wu, transcript_phase_1) =
            self.sumcheck_phase_1(fiat_shamir_engine, &mut sumcheck_prover, round);

        fiat_shamir_engine.set_seed(fiat_shamir_seed);
        sumcheck_prover.init_phase_2(&mut random_u, f2_at_u);
        let (random_v, wv, transcript_phase_2) =
            self.sumcheck_phase_2(fiat_shamir_engine, &mut sumcheck_prover, round);

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&transcript_phase_1);
        transcript.serialize_and_push(&transcript_phase_2);

        (random_u, random_v, wu, wv, transcript)
    }

    fn generate_first_round<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        circuit_evaluations: &CircuitEvaluations<F>,
        g: &mut CudaSlice<F>,
        w_g: F,
    ) -> (CudaSlice<F>, CudaSlice<F>, F, F, Transcript) {
        let mut w_next_evaluations =
            CudaSlice::on_host(circuit_evaluations.at_layer(1, true).to_vec());
        let bookeeping_identity_g = Engine::precompute_bookeeping(F::ONE, g);

        let (mut random_u, mut random_v, w_u, w_v, round_transcript) = self.run_sumcheck(
            fiat_shamir_engine,
            0,
            w_next_evaluations.clone(),
            bookeeping_identity_g,
            w_g,
        );

        let w_next_ext =
            SisuDenseMultilinearExtension::from_slice(w_next_evaluations.as_ref_host());
        let w_u = w_u.unwrap_or_else(|| w_next_ext.evaluate(vec![random_u.as_ref_host()]));
        let w_v = w_v.unwrap_or_else(|| w_next_ext.evaluate(vec![random_v.as_ref_host()]));

        let mut transcript = Transcript::new();
        transcript.serialize_and_push(&round_transcript);
        transcript.serialize_and_push(&w_u);
        transcript.serialize_and_push(&w_v);

        (random_u, random_v, w_u, w_v, transcript)
    }

    /// Run sumcheck on
    /// alpha * V_i(u) + beta * V_i(v)
    /// =  alpha * (add(u, x, y) * (V(x) + V(y)) + mul(u, x, y) * V(x) * V(y))
    ///  + beta  * (add(v, x, y) * (V(x) + V(y)) + mul(v, x, y) * V(x) * V(y))
    fn generate_round<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        alpha: F,
        beta: F,
        prev_u: &mut CudaSlice<F>,
        prev_v: &mut CudaSlice<F>,
        prev_w_u: F,
        prev_w_v: F,
        evaluations: &CircuitEvaluations<F>,
    ) -> (CudaSlice<F>, CudaSlice<F>, F, F, Transcript) {
        assert!(round > 0, "This function only works for non-zero round");
        let mut w_next_evaluations =
            CudaSlice::on_host(evaluations.at_layer(round + 1, true).to_vec());
        let bookeeping_identity_g =
            precompute_bookeeping_with_linear_combination::<_, Engine>(alpha, prev_u, beta, prev_v);

        let fiat_shamir_seed = alpha * prev_w_u + beta * prev_w_v;
        let (mut random_u, mut random_v, w_u, w_v, round_transcript) = self.run_sumcheck(
            fiat_shamir_engine,
            round,
            w_next_evaluations.clone(),
            bookeeping_identity_g,
            fiat_shamir_seed,
        );

        let w_next_ext =
            SisuDenseMultilinearExtension::from_slice(w_next_evaluations.as_ref_host());
        let w_u = w_u.unwrap_or_else(|| w_next_ext.evaluate(vec![random_u.as_ref_host()]));
        let w_v = w_v.unwrap_or_else(|| w_next_ext.evaluate(vec![random_v.as_ref_host()]));

        let mut transcript = Transcript::new();
        transcript.serialize_and_push(&round_transcript);
        transcript.serialize_and_push(&w_u);
        transcript.serialize_and_push(&w_v);

        (random_u, random_v, w_u, w_v, transcript)
    }

    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        witness: &[F],
    ) -> (CudaSlice<F>, CudaSlice<F>, F, F, Transcript) {
        let now = Instant::now();
        fiat_shamir_engine.begin_protocol();

        let circuit_evaluations = self.circuit.evaluate(&self.circuit_params, witness);

        let output = circuit_evaluations.at_layer(0, false);
        println!("[GKR]: run circuit: {:?}", now.elapsed());
        let now = Instant::now();

        let mut transcript = Transcript::new();
        transcript.serialize_and_push(&output.to_vec());

        let mut g = CudaSlice::on_host(
            fiat_shamir_engine
                .reduce_and_hash_to_fields(&output, self.circuit.num_vars_at(0) as usize),
        );
        let output_ext = SisuDenseMultilinearExtension::from_slice(&output);
        let w_g = output_ext.evaluate(vec![g.as_ref_host()]);

        println!("[GKR]: run output extension: {:?}", now.elapsed());
        let now = Instant::now();

        let (mut prev_u, mut prev_v, mut prev_w_u, mut prev_w_v, mut round_transcript) =
            self.generate_first_round(fiat_shamir_engine, &circuit_evaluations, &mut g, w_g);
        // transcript.serialize_and_push(&round_transcript);
        transcript.serialize_and_push(&round_transcript);
        println!("[GKR]: first round: {:?}", now.elapsed());

        let mut alpha: F;
        let mut beta: F;
        for i in 1..=self.circuit.len() - 1 {
            let now = Instant::now();
            let data = prev_w_u + prev_w_v;
            alpha = fiat_shamir_engine.hash_to_field(&data);
            beta = fiat_shamir_engine.hash_to_field(&data);
            (prev_u, prev_v, prev_w_u, prev_w_v, round_transcript) = self.generate_round(
                fiat_shamir_engine,
                i,
                alpha,
                beta,
                &mut prev_u,
                &mut prev_v,
                prev_w_u,
                prev_w_v,
                &circuit_evaluations,
            );
            // transcript.serialize_and_push(&round_transcript);
            transcript.serialize_and_push(&round_transcript);
            println!(
                "[GKR]: round {} ({}): {:?}",
                i,
                self.circuit.len_at(i),
                now.elapsed()
            );
        }

        (prev_u, prev_v, prev_w_u, prev_w_v, transcript)
    }
}

pub struct GKRVerifier<'a, F: Field> {
    circuit: &'a Circuit<F>,
    circuit_params: CircuitParams<'a, F>,
}

impl<'a, F: Field> GKRVerifier<'a, F> {
    pub fn new(circuit: &'a Circuit<F>, circuit_params: CircuitParams<'a, F>) -> Self {
        Self {
            circuit,
            circuit_params,
        }
    }

    fn verify_sumcheck<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        mut transcript: TranscriptIter,
        fiat_shamir_seed: F,
    ) -> Result<(Vec<F>, Vec<F>, F, F), Error> {
        let mut num_sumchecks = 0;

        // GATE CONSTANT: constant(g, x, y)
        let constant_evaluations = &self.circuit.layer(round).constant_ext.evaluations;
        num_sumchecks += if constant_evaluations.len() > 0 { 1 } else { 0 };

        // GATE MUL: mul(g, x, y) * V(x) * V(y)
        let mul_evaluations = &self.circuit.layer(round).mul_ext.evaluations;
        num_sumchecks += if mul_evaluations.len() > 0 { 1 } else { 0 };

        // GATE FORWARD X: forward(g, x, y) * V(x)
        let forw_x_evaluations = &self.circuit.layer(round).forward_x_ext.evaluations;
        num_sumchecks += if forw_x_evaluations.len() > 0 { 1 } else { 0 };

        // GATE FORWARD Y: forward(g, x, y) * V(y)
        let forw_y_evaluations = &self.circuit.layer(round).forward_y_ext.evaluations;
        num_sumchecks += if forw_y_evaluations.len() > 0 { 1 } else { 0 };

        let sumcheck_verifier =
            MultiSumcheckVerifier::new(num_sumchecks, 0, self.circuit.num_vars_at(round + 1));

        // PHASE 1
        fiat_shamir_engine.set_seed(fiat_shamir_seed);
        let sumcheck_phase_1_transcript = transcript.pop_and_deserialize::<Transcript>();
        let (random_u, _, sum_over_boolean_hypercube) = sumcheck_verifier
            .verify(fiat_shamir_engine, sumcheck_phase_1_transcript.into_iter())?;

        // PHASE 2
        fiat_shamir_engine.set_seed(fiat_shamir_seed);
        let sumcheck_phase_2_transcript = transcript.pop_and_deserialize::<Transcript>();
        let (random_v, final_value, _) = sumcheck_verifier
            .verify(fiat_shamir_engine, sumcheck_phase_2_transcript.into_iter())?;

        Ok((random_u, random_v, final_value, sum_over_boolean_hypercube))
    }

    fn verify_first_round<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        output: &[F],
        mut transcript: TranscriptIter,
        g: &[F],
    ) -> Result<(Vec<F>, Vec<F>, F, F), Error> {
        let now = Instant::now();
        let sumcheck_transcript = transcript.pop_and_deserialize::<Transcript>();

        let output_ext = SisuDenseMultilinearExtension::from_slice(&output);
        let w_g = output_ext.evaluate(vec![g]);

        let (random_u, random_v, final_value, sum_all) =
            self.verify_sumcheck(fiat_shamir_engine, 0, sumcheck_transcript.into_iter(), w_g)?;

        if w_g != sum_all {
            return Err(Error::GRK(format!(
                    "the evaluation of output extension is different from the sum over boolean hypercube of first sumcheck protocol"
                )));
        }
        println!("VERIFY_FIRST_ROUND 1: {:?}", now.elapsed());

        let constant_ext = &self.circuit.layer(0).constant_ext;
        let mul_ext = &self.circuit.layer(0).mul_ext;
        let forward_x_ext = &self.circuit.layer(0).forward_x_ext;
        let forward_y_ext = &self.circuit.layer(0).forward_y_ext;

        println!("VERIFY_FIRST_ROUND 2: {:?}", now.elapsed());
        let constant = constant_ext.evaluate(vec![g, &random_u, &random_v], &self.circuit_params);
        let mul = mul_ext.evaluate(vec![g, &random_u, &random_v], &self.circuit_params);
        let forward_x = forward_x_ext.evaluate(vec![g, &random_u, &random_v], &self.circuit_params);
        let forward_y = forward_y_ext.evaluate(vec![g, &random_u, &random_v], &self.circuit_params);

        let w_u = transcript.pop_and_deserialize::<F>();
        let w_v = transcript.pop_and_deserialize::<F>();
        println!("VERIFY_FIRST_ROUND 3: {:?}", now.elapsed());

        let oracle_access = constant + mul * w_u * w_v + forward_x * w_u + forward_y * w_v;
        if oracle_access != final_value {
            return Err(Error::GRK(format!(
                "oracle sum is different from the value at point (u, v) at the end of first sumcheck protocol"
            )));
        }

        Ok((random_u, random_v, w_u, w_v))
    }

    fn verify_round<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        mut transcript: TranscriptIter,
        prev_w_u: &F,
        prev_w_v: &F,
        prev_u: &[F],
        prev_v: &[F],
        alpha: F,
        beta: F,
    ) -> Result<(Vec<F>, Vec<F>, F, F), Error> {
        let now = Instant::now();
        let sumcheck_transcript = transcript.pop_and_deserialize::<Transcript>();

        let fiat_shamir_seed = alpha * prev_w_u + beta * prev_w_v;
        let (random_u, random_v, final_value, sum_all) = self.verify_sumcheck(
            fiat_shamir_engine,
            round,
            sumcheck_transcript.into_iter(),
            fiat_shamir_seed,
        )?;

        if alpha * prev_w_u + beta * prev_w_v != sum_all {
            return Err(Error::GRK(format!(
                "the sum of round {round} is different from the sum over boolean hypercube of sumcheck protocol"
            )));
        }

        let constant_ext = &self.circuit.layer(round).constant_ext;
        let mul_ext = &self.circuit.layer(round).mul_ext;
        let forward_x_ext = &self.circuit.layer(round).forward_x_ext;
        let forward_y_ext = &self.circuit.layer(round).forward_y_ext;

        let constant_alpha =
            constant_ext.evaluate(vec![prev_u, &random_u, &random_v], &self.circuit_params);
        let mul_alpha = mul_ext.evaluate(vec![prev_u, &random_u, &random_v], &self.circuit_params);
        let forward_x_alpha =
            forward_x_ext.evaluate(vec![prev_u, &random_u, &random_v], &self.circuit_params);
        let forward_y_alpha =
            forward_y_ext.evaluate(vec![prev_u, &random_u, &random_v], &self.circuit_params);

        println!("VERIFY ROUND 4a: {:?}", now.elapsed());
        let constant_beta =
            constant_ext.evaluate(vec![prev_v, &random_u, &random_v], &self.circuit_params);
        let mul_beta = mul_ext.evaluate(vec![prev_v, &random_u, &random_v], &self.circuit_params);
        let forward_x_beta =
            forward_x_ext.evaluate(vec![prev_v, &random_u, &random_v], &self.circuit_params);
        let forward_y_beta =
            forward_y_ext.evaluate(vec![prev_v, &random_u, &random_v], &self.circuit_params);

        let w_u = transcript.pop_and_deserialize::<F>();
        let w_v = transcript.pop_and_deserialize::<F>();
        println!("VERIFY ROUND 4b: {:?}", now.elapsed());
        let oracle_sum = F::ZERO // The zero value is here for more readable only.
            + alpha
                * (constant_alpha
                    + mul_alpha * w_u * w_v
                    + forward_x_alpha * w_u
                    + forward_y_alpha * w_v)
            + beta
                * (constant_beta
                    + mul_beta * w_u * w_v
                    + forward_x_beta * w_u
                    + forward_y_beta * w_v);

        if oracle_sum != final_value {
            return Err(Error::GRK(format!(
                "oracle sum is different with the sum at (u, v) at the end of round {round} sumcheck protocol"
            )));
        }

        Ok((random_u, random_v, w_u, w_v))
    }

    pub fn verify_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        input: &[F],
        mut transcript: TranscriptIter,
    ) -> Result<(Vec<F>, Vec<F>, F, F, Vec<F>), Error> {
        fiat_shamir_engine.begin_protocol();

        let now = Instant::now();
        let output = transcript.pop_and_deserialize::<Vec<F>>();
        let mut padding_output = output.clone();
        padding_pow_of_two_size(&mut padding_output);
        println!("VERIFY GKR 1: {:?}", now.elapsed());

        let g = fiat_shamir_engine
            .reduce_and_hash_to_fields(&output, self.circuit.num_vars_at(0) as usize);

        let first_round_transcript = transcript.pop_and_deserialize::<Transcript>();
        let (mut u, mut v, mut last_w_u, mut last_w_v) = self.verify_first_round(
            fiat_shamir_engine,
            &padding_output,
            first_round_transcript.into_iter(),
            &g,
        )?;
        println!("VERIFY GKR 2: {:?}", now.elapsed());

        let mut alpha: F;
        let mut beta: F;
        for round in 0..transcript.remaining_len() {
            let now = Instant::now();
            let data = last_w_u + last_w_v;
            alpha = fiat_shamir_engine.hash_to_field(&data);
            beta = fiat_shamir_engine.hash_to_field(&data);

            let round_transcript = transcript.pop_and_deserialize::<Transcript>();

            // All last_w_u and last_w_v are ignored, except the last one.
            (u, v, last_w_u, last_w_v) = self.verify_round(
                fiat_shamir_engine,
                round + 1,
                round_transcript.into_iter(),
                &last_w_u,
                &last_w_v,
                &u,
                &v,
                alpha,
                beta,
            )?;

            println!("VERIFY GKR 3-{}: {:?}", round, now.elapsed());
        }

        if input.len() == self.circuit.input_size() {
            let mut padding_input = input.to_vec();
            padding_pow_of_two_size(&mut padding_input);

            let v_d = SisuDenseMultilinearExtension::from_slice(&padding_input);
            println!("VERIFY GKR 4: {:?}", now.elapsed());

            let v_d_u = v_d.evaluate(vec![&u]);
            if v_d_u != last_w_u {
                return Err(Error::GRK(format!(
                    "w_u {:?} != sumcheck w_u {:?}",
                    v_d_u, last_w_u,
                )));
            }
            println!("VERIFY GKR 5: {:?}", now.elapsed());

            let v_d_v = v_d.evaluate(vec![&v]);
            if v_d_v != last_w_v {
                return Err(Error::GRK(format!(
                    "w_v {:?} != sumcheck w_v {:?}",
                    v_d_v, last_w_v,
                )));
            }

            println!("VERIFY GKR 6: {:?}", now.elapsed());
        }

        Ok((u, v, last_w_u, last_w_v, output))
    }

    fn extract_round_transcript(
        &self,
        mut transcript: TranscriptIter,
        round: usize,
    ) -> GKRRoundTranscript<F> {
        let sumcheck_transcript = transcript.pop_and_deserialize::<Transcript>();
        let mut sumcheck_transcript_iter = sumcheck_transcript.into_iter();

        let mut num_sumchecks = 0;

        // GATE CONSTANT: constant(g, x, y)
        let constant_evaluations = &self.circuit.layer(round).constant_ext.evaluations;
        num_sumchecks += if constant_evaluations.len() > 0 { 1 } else { 0 };

        // GATE MUL: mul(g, x, y) * V(x) * V(y)
        let mul_evaluations = &self.circuit.layer(round).mul_ext.evaluations;
        num_sumchecks += if mul_evaluations.len() > 0 { 1 } else { 0 };

        // GATE FORWARD X: forward(g, x, y) * V(x)
        let forw_x_evaluations = &self.circuit.layer(round).forward_x_ext.evaluations;
        num_sumchecks += if forw_x_evaluations.len() > 0 { 1 } else { 0 };

        // GATE FORWARD Y: forward(g, x, y) * V(y)
        let forw_y_evaluations = &self.circuit.layer(round).forward_y_ext.evaluations;
        num_sumchecks += if forw_y_evaluations.len() > 0 { 1 } else { 0 };

        let sumcheck_verifier =
            MultiSumcheckVerifier::new(num_sumchecks, 0, self.circuit.num_vars_at(round + 1));

        let phase_1_transcript = sumcheck_transcript_iter.pop_and_deserialize::<Transcript>();
        let phase_1 = sumcheck_verifier.extract_transcript(phase_1_transcript.into_iter());

        let phase_2_transcript = sumcheck_transcript_iter.pop_and_deserialize::<Transcript>();
        let phase_2 = sumcheck_verifier.extract_transcript(phase_2_transcript.into_iter());

        let w_u = transcript.pop_and_deserialize::<F>();
        let w_v = transcript.pop_and_deserialize::<F>();

        GKRRoundTranscript {
            phase_1,
            phase_2,
            w_u,
            w_v,
        }
    }

    pub fn extract_transcript(&self, mut transcript: TranscriptIter) -> GKRTranscript<F> {
        let mut gkr_transcript = GKRTranscript::default();

        let output = transcript.pop_and_deserialize::<Vec<F>>();
        let mut padding_output = output.clone();
        padding_pow_of_two_size(&mut padding_output);
        gkr_transcript.output = padding_output;

        for round in 0..transcript.remaining_len() {
            let round_transcript = transcript.pop_and_deserialize::<Transcript>();

            // All last_w_u and last_w_v are ignored, except the last one.
            let gkr_round_transcript =
                self.extract_round_transcript(round_transcript.into_iter(), round);
            gkr_transcript.round_transcripts.push(gkr_round_transcript);
        }

        gkr_transcript
    }

    pub fn configs(&self) -> GKRConfigs<'a, F> {
        let mut gkr_configs = GKRConfigs::new(&self.circuit);
        gkr_configs.circuit_params = self.circuit_params.clone();
        gkr_configs.output_size = round_to_pow_of_two(self.circuit.len_at(0));
        gkr_configs.input_size = round_to_pow_of_two(self.circuit.input_size());

        for round in 0..self.circuit.len() {
            let mut num_sumchecks = 0;

            // GATE CONSTANT: constant(g, x, y)
            let constant_evaluations = &self.circuit.layer(round).constant_ext.evaluations;
            num_sumchecks += if constant_evaluations.len() > 0 { 1 } else { 0 };

            // GATE MUL: mul(g, x, y) * V(x) * V(y)
            let mul_evaluations = &self.circuit.layer(round).mul_ext.evaluations;
            num_sumchecks += if mul_evaluations.len() > 0 { 1 } else { 0 };

            // GATE FORWARD X: forward(g, x, y) * V(x)
            let forw_x_evaluations = &self.circuit.layer(round).forward_x_ext.evaluations;
            num_sumchecks += if forw_x_evaluations.len() > 0 { 1 } else { 0 };

            // GATE FORWARD Y: forward(g, x, y) * V(y)
            let forw_y_evaluations = &self.circuit.layer(round).forward_y_ext.evaluations;
            num_sumchecks += if forw_y_evaluations.len() > 0 { 1 } else { 0 };

            let sumcheck_verifier = MultiSumcheckVerifier::<F>::new(
                num_sumchecks,
                0,
                self.circuit.num_vars_at(round + 1),
            );

            gkr_configs
                .round_sumcheck_configs
                .push(sumcheck_verifier.configs());
        }

        gkr_configs
    }
}

#[derive(Default)]
pub struct GKRLinearProductSumcheckTranscript<F: Field> {
    phase_1_transcript: ProductSumcheckTranscript<F>,
    phase_2_transcript: ProductSumcheckTranscript<F>,
}

impl<F: Field> GKRLinearProductSumcheckTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];

        result.extend(self.phase_1_transcript.to_vec());
        result.extend(self.phase_2_transcript.to_vec());

        result
    }
}

#[derive(Default)]
pub struct GKRRoundTranscript<F: Field> {
    phase_1: MultiSumcheckTranscript<F>,
    phase_2: MultiSumcheckTranscript<F>,
    w_u: F,
    w_v: F,
}

impl<F: Field> GKRRoundTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];

        result.extend(self.phase_1.to_vec());
        result.extend(self.phase_2.to_vec());

        result.push(self.w_u);
        result.push(self.w_v);

        result
    }
}

pub struct GKRConfigs<'a, F: Field> {
    circuit: &'a Circuit<F>,
    circuit_params: CircuitParams<'a, F>,
    output_size: usize,
    input_size: usize,
    round_sumcheck_configs: Vec<MultiSumcheckConfigs>,
}

impl<'a, F: Field> GKRConfigs<'a, F> {
    pub fn new(circuit: &'a Circuit<F>) -> Self {
        Self {
            circuit,
            circuit_params: CircuitParams::default(),
            output_size: 0,
            input_size: 0,
            round_sumcheck_configs: vec![],
        }
    }

    pub fn gen_code(&self, gkr_index: usize) -> (Vec<FuncGenerator<F>>, Vec<CustomMLEGenerator>) {
        let mut funcs = vec![];
        let mut mle = vec![];

        // SUMCHECK_ID OF GKR:
        // gkr_index * 100 + layer_index.
        for layer_index in 0..self.round_sumcheck_configs.len() {
            funcs.extend(
                self.round_sumcheck_configs[layer_index].gen_code(gkr_index * 1000 + layer_index),
            );
        }

        let mut n_rounds_func = FuncGenerator::new("get_gkr__n_rounds", vec!["gkr_index"]);
        n_rounds_func.add_number(vec![gkr_index], self.round_sumcheck_configs.len());
        funcs.push(n_rounds_func);

        let mut output_size_func = FuncGenerator::new("get_gkr__output_size", vec!["gkr_index"]);
        output_size_func.add_number(vec![gkr_index], self.output_size);
        funcs.push(output_size_func);

        let mut input_size_func = FuncGenerator::new("get_gkr__input_size", vec!["gkr_index"]);
        input_size_func.add_number(vec![gkr_index], self.input_size);
        funcs.push(input_size_func);

        let (f, t) = self.circuit.gen_code(gkr_index, &self.circuit_params);
        funcs.extend(f);
        mle.extend(t);

        (funcs, mle)
    }
}

#[derive(Default)]
pub struct GKRTranscript<F: Field> {
    output: Vec<F>,
    round_transcripts: Vec<GKRRoundTranscript<F>>,
}

impl<F: Field> GKRTranscript<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];

        result.extend_from_slice(&self.output);
        for i in 0..self.round_transcripts.len() {
            result.extend(self.round_transcripts[i].to_vec());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::fiat_shamir::DefaultFiatShamirEngine;
    use crate::hash::DummyHash;
    use crate::sisu_engine::CPUSisuEngine;

    use ark_ff::Field;
    use sisulib::circuit::{
        CircuitGate, CircuitGateType, CircuitLayer, CircuitParams, GateEvaluations, GateF,
        GateIndex,
    };
    use sisulib::codegen::generator::FileGenerator;
    use sisulib::common::{
        convert_field_to_string, convert_vec_field_to_string, dec2bin, dec2bin_limit, split_number,
    };
    use sisulib::field::{FpFRI, FpSisu};
    use sisulib::mle::sparse::SisuSparseMultilinearExtension;

    use super::*;
    use rand::SeedableRng;
    use rand::{rngs::StdRng, Rng};

    #[test]
    fn test_dense_extension() {
        let r_index = 5;
        let num_vars = 9;
        let sub_num_vars = 4;
        let mut rng = StdRng::seed_from_u64(60);

        let mut evaluations = vec![];
        for _ in 0..2usize.pow(num_vars as u32) {
            evaluations.push(FpFRI::from(rng.gen_range(0..10000)));
        }

        let mut sub_random_point = vec![];
        for _ in 0..sub_num_vars {
            sub_random_point.push(FpFRI::from(rng.gen_range(0..10000)));
        }

        let sub_evaluations = &evaluations
            [2usize.pow(sub_num_vars) * r_index..2usize.pow(sub_num_vars) * (r_index + 1)];

        let sub_ext = SisuDenseMultilinearExtension::from_slice(sub_evaluations);

        let r_index_var = dec2bin(r_index as u64, num_vars - sub_num_vars as usize);

        let mut random_point = r_index_var.clone();
        random_point.extend_from_slice(&sub_random_point);

        let sub_mle_value = sub_ext.evaluate(vec![&sub_random_point]);

        let ext = SisuDenseMultilinearExtension::from_slice(&evaluations);
        let mle_value = ext.evaluate(vec![&random_point]);

        assert_eq!(mle_value, sub_mle_value);
    }

    #[test]
    fn test_dense_extension_combine() {
        let num_vars = 5usize;
        let sub_num_vars = 2usize;

        let mut evaluations = vec![];
        for _ in 0..2usize.pow(num_vars as u32) {
            evaluations.push(FpSisu::from(rand::thread_rng().gen_range(0..100)));
        }

        let mut random_point = vec![];
        for _ in 0..num_vars {
            random_point.push(FpSisu::from(rand::thread_rng().gen_range(0..100)));
        }

        let mut sub_evaluations = vec![];
        for i in 0..2usize.pow(num_vars as u32 - sub_num_vars as u32) {
            sub_evaluations.push(
                &evaluations[2usize.pow(sub_num_vars as u32) * i
                    ..2usize.pow(sub_num_vars as u32) * (i + 1)],
            );
        }

        let mut sub_ext = vec![];
        for i in 0..2usize.pow(num_vars as u32 - sub_num_vars as u32) {
            sub_ext.push(SisuDenseMultilinearExtension::from_slice(
                sub_evaluations[i],
            ));
        }

        let mut combine_evaluations = vec![];
        for i in 0..2usize.pow(num_vars as u32 - sub_num_vars as u32) {
            let sub_mle_value = sub_ext[i].evaluate(vec![&random_point[num_vars - sub_num_vars..]]);
            combine_evaluations.push(sub_mle_value);
        }

        let combine_ext = SisuDenseMultilinearExtension::from_slice(&combine_evaluations);
        let combine_value = combine_ext.evaluate(vec![&random_point[..num_vars - sub_num_vars]]);

        let ext = SisuDenseMultilinearExtension::from_slice(&evaluations);
        let mle_value = ext.evaluate(vec![&random_point]);

        assert_eq!(mle_value, combine_value);
    }

    #[test]
    fn test_dense_extension_manual() {
        let evaluations = vec![
            FpSisu::from(3), // 0
            FpSisu::from(2),
            FpSisu::from(6),
            FpSisu::from(7),
            FpSisu::from(4), // 1
            FpSisu::from(1),
            FpSisu::from(5),
            FpSisu::from(8),
            FpSisu::from(9), // 2
            FpSisu::from(3),
            FpSisu::from(2),
            FpSisu::from(4),
            FpSisu::from(3), // 3
            FpSisu::from(1),
            FpSisu::from(8),
            FpSisu::from(5),
            FpSisu::from(3), // 4
            FpSisu::from(2),
            FpSisu::from(6),
            FpSisu::from(7),
            FpSisu::from(4), // 5
            FpSisu::from(1),
            FpSisu::from(5),
            FpSisu::from(8),
            FpSisu::from(9), // 6
            FpSisu::from(3),
            FpSisu::from(2),
            FpSisu::from(4),
            FpSisu::from(3), // 7
            FpSisu::from(1),
            FpSisu::from(8),
            FpSisu::from(5),
        ];

        let r_index_1 = 0;
        let r_index_2 = 2;
        let r_index_3 = 4;
        let r_index_4 = 6;

        let sub_evaluations_0 = &evaluations[4 * r_index_1..4 * (r_index_1 + 1)];
        let sub_evaluations_2 = &evaluations[4 * r_index_2..4 * (r_index_2 + 1)];
        let sub_evaluations_4 = &evaluations[4 * r_index_3..4 * (r_index_3 + 1)];
        let sub_evaluations_6 = &evaluations[4 * r_index_4..4 * (r_index_4 + 1)];

        let ext = SisuDenseMultilinearExtension::from_slice(&evaluations);
        let sub_ext_0 = SisuDenseMultilinearExtension::from_slice(sub_evaluations_0);
        let sub_ext_2 = SisuDenseMultilinearExtension::from_slice(sub_evaluations_2);
        let sub_ext_4 = SisuDenseMultilinearExtension::from_slice(sub_evaluations_4);
        let sub_ext_6 = SisuDenseMultilinearExtension::from_slice(sub_evaluations_6);

        let sub_var = vec![FpSisu::from(31), FpSisu::from(32)];

        // 0 2 4 6 (r_index)
        //
        // 0 0 0 0 (same bit) - represent for the group.
        // 0 1 0 1 (diff bit at 0-2, 4-6)
        // 0 0 1 1 (diff bit at 02 - 46)
        let mle_value = ext.evaluate(vec![&[
            FpSisu::from(7),  // a random point because of this is a different bit of 02 - 46.
            FpSisu::from(3),  // a random point because of this is a different bit of 0-2, 4-6.
            FpSisu::from(0),  // point representing for the group of 4 indexes (the same bit).
            FpSisu::from(31), // common point for sub_mle and mle.
            FpSisu::from(32), // common point for sub_mle and mle.
        ]]); // r_C_sig = [r_index as binary, 31, 32]
        let sub_mle_value_0 = sub_ext_0.evaluate(vec![&sub_var]); // r_C_membership [31, 32]
        let sub_mle_value_2 = sub_ext_2.evaluate(vec![&sub_var]); // r_C_membership [31, 32]
        let sub_mle_value_4 = sub_ext_4.evaluate(vec![&sub_var]); // r_C_membership [31, 32]
        let sub_mle_value_6 = sub_ext_6.evaluate(vec![&sub_var]); // r_C_membership [31, 32]

        // We need to evaluate for two pair (02 and 46) at the point third
        // random points.
        let pair02 = sub_mle_value_0 * (FpSisu::from(1) - FpSisu::from(3))
            + sub_mle_value_2 * FpSisu::from(3);

        let pair46 = sub_mle_value_4 * (FpSisu::from(1) - FpSisu::from(3))
            + sub_mle_value_6 * FpSisu::from(3);

        assert!(
            mle_value == pair02 * (FpSisu::from(1) - FpSisu::from(7)) + pair46 * FpSisu::from(7)
        );
    }

    #[test]
    fn test_dense_extension_general() {
        println!(); // a new line for more readable.
        let evaluations = vec![
            FpSisu::from(3), // 0   0
            FpSisu::from(2),
            FpSisu::from(6),
            FpSisu::from(7),
            FpSisu::from(4), // 1
            FpSisu::from(1),
            FpSisu::from(5),
            FpSisu::from(8),
            FpSisu::from(9), // 2   1
            FpSisu::from(3),
            FpSisu::from(2),
            FpSisu::from(4),
            FpSisu::from(3), // 3
            FpSisu::from(1),
            FpSisu::from(8),
            FpSisu::from(5),
            FpSisu::from(3), // 4   2
            FpSisu::from(2),
            FpSisu::from(6),
            FpSisu::from(7),
            FpSisu::from(4), // 5
            FpSisu::from(1),
            FpSisu::from(5),
            FpSisu::from(8),
            FpSisu::from(9), // 6   3
            FpSisu::from(3),
            FpSisu::from(2),
            FpSisu::from(4),
            FpSisu::from(3), // 7
            FpSisu::from(1),
            FpSisu::from(8),
            FpSisu::from(5),
        ];

        let single_input_size = 2;
        let input_position = 3;
        let replica_distance = 2usize;
        let total_replicas = 4;

        // --->
        let replica_size = evaluations.len() / total_replicas;
        let replica_num_vars = replica_size.ilog2() as usize;

        let ext = SisuDenseMultilinearExtension::from_slice(&evaluations);
        let mut sub_ext = vec![];
        for r_index in 0..total_replicas {
            if r_index % replica_distance == 0 {
                let start_index = r_index * replica_size + input_position * single_input_size;
                let end_index = start_index + single_input_size;
                let sub_evaluations = &evaluations[start_index..end_index];
                sub_ext.push(SisuDenseMultilinearExtension::from_slice(sub_evaluations));
            }
        }

        // Input and witness example:
        // [w00 i0 w01 w02; w10 i1 w11 w12; ...; wN0 iN wN1 wN2]
        //
        // Let:
        // - n: single public input num vars.
        // - d: replica distance num vars (for example, we only use replicas 0, 2, 4, 6 instead of all replicas).
        // - N: single circuit num vars (both input + witness).
        // - K: total num vars (after all, the random point has K-size).
        //
        // random_points = [(K-N-d) random + (N-n-d) input index points + d zeros + n random], where
        // - the first (K-N) random points represent for combining all replicas.
        // - the next (N-n-d) points represent for indexing input groups in
        //   input_witness.
        // - the last n random points represents for every single replica.
        let single_input_num_vars = single_input_size.ilog2() as usize;
        let distance_num_vars = replica_distance.ilog2() as usize;
        let total_num_vars = total_replicas.ilog2() as usize + replica_num_vars;

        let mut fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
        let combination_random_points = fiat_shamir_engine.hash_to_fields(
            &FpSisu::ZERO,
            total_num_vars - replica_num_vars - distance_num_vars,
        );
        let input_index_random_points: Vec<FpSisu> = dec2bin_limit(
            input_position as u64,
            replica_num_vars - single_input_num_vars,
        )
        .into_iter()
        .rev()
        .collect();
        let replica_index_random_points = vec![FpSisu::from(0); distance_num_vars];
        let group_random_points =
            fiat_shamir_engine.hash_to_fields(&FpSisu::ZERO, single_input_num_vars);

        let mut random_points = vec![];
        random_points.extend(combination_random_points);
        random_points.extend(replica_index_random_points);
        random_points.extend(input_index_random_points);
        random_points.extend(group_random_points);

        println!("total_num_vars: {}", total_num_vars);
        println!("single_input_num_vars: {}", single_input_num_vars);
        println!("distance_num_vars: {}", distance_num_vars);
        println!("random_points: {:?}", random_points);

        // 0 2 4 6 (r_index)
        //
        // 0 0 0 0 (same bit) - represent for the group.
        // 0 1 0 1 (diff bit at 0-2, 4-6)
        // 0 0 1 1 (diff bit at 02 - 46)
        let mle_value = ext.evaluate(vec![&random_points]); // r_C_sig = [r_index as binary, 31, 32]
        let mut sub_mle_values = vec![];
        for i in 0..sub_ext.len() {
            sub_mle_values.push(sub_ext[i].evaluate(vec![
                &random_points[total_num_vars - single_input_num_vars..],
            ]));
        }

        // We need to evaluate for two pair (02 and 46) at the point third
        // random points.
        let all_sub_ext = SisuDenseMultilinearExtension::from_slice(&sub_mle_values);
        let all_sub_mle_value = all_sub_ext.evaluate(vec![
            &random_points[..total_num_vars - replica_num_vars - distance_num_vars],
        ]);

        assert_eq!(mle_value, all_sub_mle_value);
    }

    #[test]
    fn test_sparse_extension() {
        let sub_ext = SisuSparseMultilinearExtension::<FpSisu>::from_evaluations(
            4,
            GateEvaluations::new(vec![
                (0, GateF::from(1)),  // 00 0 0
                (3, GateF::from(2)),  // 00 1 1
                (7, GateF::from(3)),  // 01 1 1
                (13, GateF::from(4)), // 11 0 1
            ]),
        );

        let composed_ext = SisuSparseMultilinearExtension::<FpSisu>::from_evaluations(
            10,
            GateEvaluations::new(vec![
                // subcircuit 1            0100 010 010
                (0, GateF::from(1)),   //  0000 000 000
                (9, GateF::from(2)),   //  0000 001 001
                (73, GateF::from(3)),  //  0001 001 001
                (193, GateF::from(4)), //  0011 000 001
                // subcircuit 2
                (274, GateF::from(1)), //  0100 010 010
                (283, GateF::from(2)), //  0100 011 011
                (347, GateF::from(3)), //  0101 011 011
                (467, GateF::from(4)), //  0111 010 011
                // subcircuit 3
                (548, GateF::from(1)), //  1000 100 100
                (557, GateF::from(2)), //  1000 101 101
                (621, GateF::from(3)), //  1001 101 101
                (741, GateF::from(4)), //  1011 100 101
                // subcircuit 4
                (822, GateF::from(1)),  // 1100 110 110
                (831, GateF::from(2)),  // 1100 111 111
                (895, GateF::from(3)),  // 1101 111 111
                (1015, GateF::from(4)), // 1111 110 111
            ]),
        );

        let out = vec![
            FpSisu::from(1),
            FpSisu::from(2),
            FpSisu::from(3),
            FpSisu::from(4),
        ];
        let in1 = vec![FpSisu::from(5), FpSisu::from(6), FpSisu::from(7)];
        let in2 = vec![FpSisu::from(8), FpSisu::from(9), FpSisu::from(10)];

        let sub_value: FpSisu = sub_ext.evaluate(
            vec![&out[2..], &in1[2..], &in2[2..]],
            &CircuitParams::default(),
        );
        let expected_value =
            composed_ext.evaluate(vec![&out, &in1, &in2], &CircuitParams::default());

        let computed_value = sub_value
            * (out[0] * in1[0] * in2[0]
                + (FpSisu::from(1) - out[0])
                    * (FpSisu::from(1) - in1[0])
                    * (FpSisu::from(1) - in2[0]))
            * (out[1] * in1[1] * in2[1]
                + (FpSisu::from(1) - out[1])
                    * (FpSisu::from(1) - in1[1])
                    * (FpSisu::from(1) - in2[1]));

        assert_eq!(expected_value, computed_value);
    }

    fn circuit_from_book<F: Field>() -> Circuit<F> {
        let mut circuit = Circuit::new_with_layer(
            vec![
                CircuitLayer::new(vec![
                    CircuitGate::new(
                        "0-0",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(1), GateIndex::Absolute(0)],
                    ),
                    CircuitGate::new(
                        "0-1",
                        CircuitGateType::Add,
                        [GateIndex::Absolute(0), GateIndex::Absolute(1)],
                    ),
                    CircuitGate::new(
                        "0-2",
                        CircuitGateType::Fourier(GateF::ONE),
                        [GateIndex::Absolute(1), GateIndex::Absolute(0)],
                    ),
                    CircuitGate::new(
                        "0-3",
                        CircuitGateType::Fourier(GateF::from(32)),
                        [GateIndex::Absolute(0), GateIndex::Absolute(1)],
                    ),
                ]),
                CircuitLayer::new(vec![
                    CircuitGate::new(
                        "1-0",
                        CircuitGateType::Fourier(GateF::from(24)),
                        [GateIndex::Absolute(0), GateIndex::Absolute(1)],
                    ),
                    CircuitGate::new(
                        "1-1",
                        CircuitGateType::Fourier(GateF::from(12)),
                        [GateIndex::Absolute(2), GateIndex::Absolute(3)],
                    ),
                ]),
                CircuitLayer::new(vec![
                    CircuitGate::new(
                        "2-0",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(0), GateIndex::Absolute(0)],
                    ),
                    CircuitGate::new(
                        "2-1",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(1), GateIndex::Absolute(1)],
                    ),
                    CircuitGate::new(
                        "2-2",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(1), GateIndex::Absolute(2)],
                    ),
                    CircuitGate::new(
                        "2-3",
                        CircuitGateType::Mul(GateF::ONE),
                        [GateIndex::Absolute(3), GateIndex::Absolute(3)],
                    ),
                ]),
            ],
            5,
        );

        circuit.finalize();
        circuit
    }

    #[test]
    fn test_precompute_identity() {
        let mut result = CPUSisuEngine::<_, DummyHash>::precompute_bookeeping(
            FpSisu::from(1),
            &mut CudaSlice::on_host(vec![FpSisu::from(7), FpSisu::from(3)]),
        );
        assert_eq!(
            result.as_ref_host(),
            &[
                FpSisu::from(12),
                FpSisu::from(371),
                FpSisu::from(375),
                FpSisu::from(21),
            ]
        )
    }

    #[test]
    fn test_split_number() {
        let z = 13; //1101.

        let (x, y) = split_number(&z, 2);
        assert_eq!(x, 3);
        assert_eq!(y, 1);
    }

    // #[test]
    // fn test_initialize_phase_1() {
    //     let g = &[FpSisu::from(7), FpSisu::from(3)];

    //     let f1 = &[(1usize, FpSisu::ONE), (27, FpSisu::ONE), (41, FpSisu::ONE)];

    //     let f3 = &[
    //         FpSisu::from(3),
    //         FpSisu::from(2),
    //         FpSisu::from(6),
    //         FpSisu::from(7),
    //     ];

    //     let result = initialize_phase_1(6, &f1, &f3, &precompute_bookeeping(g));
    //     assert_eq!(
    //         result,
    //         &[
    //             FpSisu::from(24),
    //             FpSisu::from(0),
    //             FpSisu::from(235),
    //             FpSisu::from(0),
    //         ]
    //     )
    // }

    #[test]
    fn test_gkr() {
        let circuit = circuit_from_book();
        let witness = [
            FpSisu::from(3),
            FpSisu::from(2),
            FpSisu::from(3),
            FpSisu::from(1),
            FpSisu::from(2),
        ];

        let evaluations = circuit.evaluate(&CircuitParams::default(), &witness);
        let expected_output = evaluations.at_layer(0, false);

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::from(3));

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::from(3));

        let engine = CPUSisuEngine::<_, DummyHash>::new();
        let prover = GKRProver::new(&engine, &circuit, CircuitParams::default());
        let (_, _, _, _, transcript) =
            prover.generate_transcript(&mut prover_fiat_shamir, &witness);

        let verifier = GKRVerifier::new(&circuit, CircuitParams::default());
        let result =
            verifier.verify_transcript(&mut verifier_fiat_shamir, &[], transcript.into_iter());
        match result {
            Ok((_, _, _, _, output)) => {
                assert_eq!(output, expected_output, "wrong output");
            }
            Err(e) => panic!("{e}"),
        }

        let gkr_transcript = verifier.extract_transcript(transcript.into_iter());
        println!(
            "Transcript: {:?}",
            convert_vec_field_to_string(&gkr_transcript.to_vec())
        );
        println!("Circuit Input: {:?}", convert_vec_field_to_string(&witness));

        let mut calc_r_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        calc_r_fiat_shamir.set_seed(FpSisu::from(3));

        println!(
            "prev_w_u: {:?}",
            convert_field_to_string(&gkr_transcript.round_transcripts[1].w_u)
        );
        println!(
            "prev_w_v: {:?}",
            convert_field_to_string(&gkr_transcript.round_transcripts[1].w_v)
        );

        let gkr_configs = verifier.configs();
        let mut file_gen =
            FileGenerator::<FpSisu>::new("../bls-circom/circuit/sisu_frbn254/configs.gen.circom");

        let (f, _) = gkr_configs.gen_code(0);
        file_gen.extend_funcs(f);

        let dummy_vpd_func = FuncGenerator::new("get_vpd__num_repetitions", vec!["vpd_index"]);
        file_gen.add_func(dummy_vpd_func);

        file_gen.create();
    }
}
