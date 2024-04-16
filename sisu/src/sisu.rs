use ark_ff::{FftField, Field};
use sisulib::{
    circuit::{general_circuit::circuit::GeneralCircuit, Circuit, CircuitParams},
    common::{padding_pow_of_two_size, round_to_pow_of_two, Error},
    domain::{Domain, RootDomain},
    mle::dense::SisuDenseMultilinearExtension,
};

use crate::{
    channel::NoChannel,
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript, TranscriptIter},
    general_gkr::{GeneralGKRProver, GeneralGKRVerifier},
    gkr::{GKRProver, GKRVerifier},
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    sisu_engine::CPUSisuEngine,
    vpd::{generate_ldt_root_domain, VPDProver, VPDVerifier},
};

#[derive(Clone)]
pub enum GKRCircuit<F: Field> {
    Regular(Circuit<F>),
    General(GeneralCircuit<F>),
}

impl<F: Field> GKRCircuit<F> {
    pub fn len(&self) -> usize {
        match self {
            Self::Regular(c) => c.len(),
            Self::General(c) => c.len(),
        }
    }

    pub fn len_at(&self, layer_index: usize) -> usize {
        match self {
            Self::Regular(c) => c.len_at(layer_index),
            Self::General(c) => c.len_at(layer_index),
        }
    }

    pub fn input_size(&self) -> usize {
        match self {
            Self::Regular(c) => c.input_size(),
            Self::General(c) => c.input_size(),
        }
    }
}

pub struct SisuProver<'a, F: Field, H: SisuHasher<F>> {
    vpd_prover: VPDProver<'a, F, H>,
    circuit: GKRCircuit<F>,
}

impl<'a, F: IcicleConvertibleField + FftField, H: SisuHasher<F>> SisuProver<'a, F, H> {
    pub fn setup(domain: &'a RootDomain<F>, ldt_rate: usize, circuit: GKRCircuit<F>) -> Self {
        Self {
            circuit,
            vpd_prover: VPDProver::setup(Domain::from(domain), ldt_rate),
        }
    }

    pub fn get_verifier(&self) -> SisuVerifier<F, H> {
        SisuVerifier {
            vpd_verifier: self.vpd_prover.get_verifier(),
            circuit: self.circuit.clone(),
        }
    }

    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        circuit_params: CircuitParams<F>,
        witness: &[F],
    ) -> Transcript {
        assert!(
            witness.len() == self.circuit.input_size(),
            "Invalid size of input and witness ({} != {})",
            witness.len(),
            self.circuit.input_size()
        );

        fiat_shamir_engine.begin_protocol();

        let mut transcript = Transcript::default();

        let mut vd = witness.to_vec();
        padding_pow_of_two_size(&mut vd);

        let mut vpd_commitment = self.vpd_prover.commit(&vd);
        transcript.serialize_and_push(&vpd_commitment.to_transcript());

        let lc_w = match &self.circuit {
            GKRCircuit::Regular(circuit) => {
                let engine = CPUSisuEngine::<_, H>::new();
                let gkr_prover = GKRProver::new(&engine, circuit, circuit_params);
                let (mut u, mut v, w_u, w_v, gkr_transcript) =
                    gkr_prover.generate_transcript(fiat_shamir_engine, witness);
                transcript.serialize_and_push(&gkr_transcript);

                fiat_shamir_engine.set_seed(w_u + w_v);
                let open_w_u_transcript = self.vpd_prover.generate_transcript(
                    fiat_shamir_engine,
                    &mut vpd_commitment,
                    u.as_ref_host(),
                );

                fiat_shamir_engine.set_seed(w_u + w_v);
                let open_w_v_transcript = self.vpd_prover.generate_transcript(
                    fiat_shamir_engine,
                    &mut vpd_commitment,
                    v.as_ref_host(),
                );

                transcript.serialize_and_push(&open_w_u_transcript);
                transcript.serialize_and_push(&open_w_v_transcript);

                w_u + w_v
            }
            GKRCircuit::General(circuit) => {
                let engine = CPUSisuEngine::<_, H>::new();
                let gkr_prover =
                    GeneralGKRProver::<_, _, NoChannel, NoChannel>::new(&engine, circuit, 1);
                let (mut g, w_g, gkr_transcript) = gkr_prover.generate_transcript(
                    fiat_shamir_engine,
                    &mut [CudaSlice::on_host(witness.to_vec())],
                    None,
                );
                transcript.serialize_and_push(&gkr_transcript);

                fiat_shamir_engine.set_seed(w_g.clone());
                let open_w_g_transcript = self.vpd_prover.generate_transcript(
                    fiat_shamir_engine,
                    &mut vpd_commitment,
                    g.as_ref_host(),
                );

                transcript.serialize_and_push(&open_w_g_transcript);
                w_g
            }
        };

        // For Vd (a MLE of input and witness), the Verifier doesn't know the
        // witness, but he knows the public input. So he needs to check if the
        // Vd is organized from input or somethingelse.
        //
        // Call input as x, witness as w.
        // Both Prover and Verifier will generate a random point with size of x,
        // called rx.
        // Then padding rx with |w| zeros, the new point is called rxw. Now rxw
        // has the length of |x| + |w|. Prover opens Vd at rxw and send that to
        // Verifier.
        //
        // Verifier constructs a MLE from the public input only, called Vx.
        // Verifier calculates Vx(rx), then compare its result to Vd(rxw)
        // received from Prover. If the check is not passed, the protocol fails.
        if witness.len() > 0 {
            let input_num_vars = round_to_pow_of_two(witness.len()).ilog2() as usize;
            let all_input_num_vars =
                round_to_pow_of_two(witness.len() + witness.len()).ilog2() as usize;
            let mut rxw = fiat_shamir_engine.hash_to_fields(&lc_w, input_num_vars);
            rxw.extend(vec![F::ZERO; all_input_num_vars - input_num_vars]);

            fiat_shamir_engine.set_seed(lc_w);
            let open_rx_transcript =
                self.vpd_prover
                    .generate_transcript(fiat_shamir_engine, &mut vpd_commitment, &rxw);
            transcript.serialize_and_push(&open_rx_transcript);
        }

        transcript
    }
}

pub struct SisuVerifier<'a, F: IcicleConvertibleField, H: SisuHasher<F>> {
    vpd_verifier: VPDVerifier<'a, F, H>,
    circuit: GKRCircuit<F>,
}

impl<'a, F: IcicleConvertibleField, H: SisuHasher<F>> SisuVerifier<'a, F, H> {
    pub fn verify_transcript<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        mut transcript: TranscriptIter,
        circuit_params: CircuitParams<'a, F>,
        input: &[F],
    ) -> Result<Vec<F>, Error> {
        fiat_shamir_engine.begin_protocol();

        let vpd_commitment_transcript = transcript.pop_and_deserialize::<Transcript>();

        let gkr_transcript = transcript.pop_and_deserialize::<Transcript>();

        let gkr_output: Vec<F>;
        let lc_w =
            match &self.circuit {
                GKRCircuit::Regular(circuit) => {
                    let gkr_verifier = GKRVerifier::new(circuit, circuit_params);
                    let (last_u, last_v, last_w_u, last_w_v, output) = gkr_verifier
                        .verify_transcript(fiat_shamir_engine, &[], gkr_transcript.into_iter())?;

                    gkr_output = output;
                    let open_w_u_transcript = transcript.pop_and_deserialize::<Transcript>();
                    let open_w_v_transcript = transcript.pop_and_deserialize::<Transcript>();

                    fiat_shamir_engine.set_seed(last_w_u + last_w_v);
                    let open_w_u = self.vpd_verifier.verify_transcript(
                        fiat_shamir_engine,
                        vpd_commitment_transcript.into_iter(),
                        open_w_u_transcript.into_iter(),
                        &last_u,
                    )?;

                    fiat_shamir_engine.set_seed(last_w_u + last_w_v);
                    let open_w_v = self.vpd_verifier.verify_transcript(
                        fiat_shamir_engine,
                        vpd_commitment_transcript.into_iter(),
                        open_w_v_transcript.into_iter(),
                        &last_v,
                    )?;

                    if open_w_u != last_w_u {
                        return Err(Error::Sisu(String::from(
                            "Vd at u of vpd commitment is diffrerent from GKR result",
                        )));
                    }

                    if open_w_v != last_w_v {
                        return Err(Error::Sisu(String::from(
                            "Vd at v of vpd commitment is diffrerent from GKR result",
                        )));
                    }

                    last_w_u + last_w_v
                }
                GKRCircuit::General(circuit) => {
                    let gkr_verifier = GeneralGKRVerifier::new(circuit);
                    let (last_g, last_w_g, output) = gkr_verifier.verify_transcript(
                        fiat_shamir_engine,
                        &[],
                        gkr_transcript.into_iter(),
                    )?;

                    gkr_output = output.into_iter().flatten().collect();
                    let open_w_g_transcript = transcript.pop_and_deserialize::<Transcript>();

                    fiat_shamir_engine.set_seed(last_w_g);
                    let open_w_u = self.vpd_verifier.verify_transcript(
                        fiat_shamir_engine,
                        vpd_commitment_transcript.into_iter(),
                        open_w_g_transcript.into_iter(),
                        &last_g,
                    )?;

                    if open_w_u != last_w_g {
                        return Err(Error::Sisu(String::from(
                            "Vd at u of vpd commitment is diffrerent from GKR result",
                        )));
                    }

                    last_w_g
                }
            };

        if input.len() > 0 {
            let all_input_num_vars =
                round_to_pow_of_two(self.circuit.input_size()).ilog2() as usize;
            let input_num_vars = round_to_pow_of_two(input.len()).ilog2() as usize;
            let rx = fiat_shamir_engine.hash_to_fields(&lc_w, input_num_vars);
            let mut rxw = rx.clone();
            rxw.extend(vec![F::ZERO; all_input_num_vars - input_num_vars]);

            let open_vrxw_transcript = transcript.pop_and_deserialize::<Transcript>();

            fiat_shamir_engine.set_seed(lc_w);
            let open_wrxw = self.vpd_verifier.verify_transcript(
                fiat_shamir_engine,
                vpd_commitment_transcript.into_iter(),
                open_vrxw_transcript.into_iter(),
                &rxw,
            )?;

            let mut padding_input = input.to_vec();
            padding_pow_of_two_size(&mut padding_input);
            let wx_mle = SisuDenseMultilinearExtension::from_slice(&padding_input);
            let wrx = wx_mle.evaluate(vec![&rx]);

            if open_wrxw != wrx {
                return Err(Error::Sisu(String::from(
                    "The input of prover is different from input of verifier",
                )));
            }
        }

        Ok(gkr_output)
    }
}

pub fn generate_root_domain<F: Field>(circuit: &GKRCircuit<F>, ldt_rate: usize) -> RootDomain<F> {
    let input_size = round_to_pow_of_two(circuit.input_size());
    let input_num_vars = input_size.ilog2() as usize;
    generate_ldt_root_domain(input_num_vars, ldt_rate)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::fiat_shamir::{DefaultFiatShamirEngine, FiatShamirEngine};

    use super::*;
    use sha2::Sha256;
    use sisulib::{circuit::general_circuit::examples::example_general_circuit, field::FpSisu};

    #[test]
    fn test_sisu() {
        let ldt_rate = 8;

        let mut circuit = example_general_circuit();
        circuit.finalize(HashMap::default());

        let circuit = GKRCircuit::General(circuit);

        let witness = [FpSisu::from(1), FpSisu::from(2), FpSisu::from(3)];
        let expected_output = [
            FpSisu::from(8),
            FpSisu::from(36),
            FpSisu::from(24),
            FpSisu::from(102),
        ];

        let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir.set_seed(FpSisu::ZERO);
        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::ZERO);

        let root_domain = generate_root_domain(&circuit, ldt_rate);
        let sisu_prover = SisuProver::<_, Sha256>::setup(&root_domain, ldt_rate, circuit);

        let transcript = sisu_prover.generate_transcript(
            &mut prover_fiat_shamir,
            CircuitParams::default(),
            &witness,
        );

        let sisu_verifier = sisu_prover.get_verifier();
        let result = sisu_verifier.verify_transcript(
            &mut verifier_fiat_shamir,
            transcript.into_iter(),
            CircuitParams::default(),
            &witness,
        );
        match result {
            Ok(output) => {
                assert_eq!(output, expected_output, "wrong output");
            }
            Err(e) => panic!("{e}"),
        }
    }
}
