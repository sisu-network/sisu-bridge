use std::{cmp::max, marker::PhantomData, time::Instant};

use sisulib::{
    circuit::general_circuit::{circuit::GeneralCircuit, gate::LayerIndex},
    mle::dense::SisuDenseMultilinearExtension,
};

use crate::{
    channel::{MasterNode, SisuReceiver, SisuSender},
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript},
    icicle_converter::IcicleConvertibleField,
    sisu_engine::SisuEngine,
    sumcheck::MultiProductSumcheckMasterProver,
};

pub struct SyntheticGeneralGKRProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    pub circuit: &'a GeneralCircuit<F>,
    num_workers: usize,
    num_replicas_per_worker: usize,
    master: &'a MasterNode<S, R>,
    __phantom: PhantomData<Engine>,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    SyntheticGeneralGKRProver<'a, F, Engine, S, R>
{
    pub fn new(
        master: &'a MasterNode<S, R>,
        circuit: &'a GeneralCircuit<F>,
        num_workers: usize,
        num_replicas_per_worker: usize,
    ) -> Self {
        assert!(num_workers.is_power_of_two());
        assert!(num_replicas_per_worker.is_power_of_two());

        Self {
            master,
            circuit,
            num_workers,
            num_replicas_per_worker,
            __phantom: PhantomData,
        }
    }

    pub fn worker_extra_num_vars(&self) -> usize {
        self.num_workers.ilog2() as usize
    }

    pub fn total_extra_num_vars(&self) -> usize {
        self.worker_extra_num_vars() + self.num_replicas_per_worker.ilog2() as usize
    }

    pub fn total_num_vars(&self, layer_index: usize) -> usize {
        self.total_extra_num_vars() + self.circuit.num_vars_at(layer_index)
    }

    /// Run a sumcheck on V(g) = f(g, x, y).
    fn run_sumcheck<FS: FiatShamirEngine<F>>(
        &self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        extra_random_g: &mut CudaSlice<F>,
        w_g: F,
    ) -> (CudaSlice<F>, CudaSlice<F>, Transcript) {
        // V(g) =
        //   constant
        // + add * (V(x) + V(y))
        // + mul * V(x) * V(y)
        // + forward * V(y).

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

        let bookeeping_extra_g = Engine::precompute_bookeeping(F::ONE, extra_random_g);
        let mut phase_1_sumcheck_prover =
            MultiProductSumcheckMasterProver::<_, Engine::RootProductBookeepingTable, _, _>::new(
                &self.master,
                self.num_workers,
                self.num_replicas_per_worker.ilog2() as usize + x_num_vars,
                bookeeping_extra_g.clone().as_host(),
            );

        // GATE CONSTANT: constant(g, x, y) * constant_1(x) * constant1(y)
        if self.circuit.layer(round).constant_ext.is_non_zero() {
            phase_1_sumcheck_prover.add(Some(vec![]));
        } else {
            phase_1_sumcheck_prover.add(None)
        }

        // GATE MUL: mul(g, x, y) * V(x) * V(y).
        if self.circuit.layer(round).mul_ext.is_non_zero() {
            phase_1_sumcheck_prover.add(Some(vec![]));
        } else {
            phase_1_sumcheck_prover.add(None)
        }

        // GATE FORWARD X: forward(g, x, y) * V(x) * V_constant_1(y).
        if self.circuit.layer(round).forward_x_ext.is_non_zero() {
            phase_1_sumcheck_prover.add(Some(vec![]));
        } else {
            phase_1_sumcheck_prover.add(None)
        }

        // GATE FORWARD Y: forward(g, x, y) * V_constant_1(x) * V(y).
        if self.circuit.layer(round).forward_y_ext.is_non_zero() {
            phase_1_sumcheck_prover.add(Some(vec![]));
        } else {
            phase_1_sumcheck_prover.add(None)
        }

        fiat_shamir_engine.set_seed(w_g);
        let (random_u, _h_at_u, t_at_u, phase_1_transcript) =
            phase_1_sumcheck_prover.run(fiat_shamir_engine);

        let bookeeping_extra_u = Engine::precompute_bookeeping(
            F::ONE,
            &mut random_u.at_range_to(..extra_random_g.len()),
        );
        let bookeeping_extra_g_u = bookeeping_extra_g
            .as_host()
            .into_iter()
            .zip(bookeeping_extra_u.as_host().into_iter())
            .map(|(g, u)| g * u)
            .collect();

        let mut phase_2_sumcheck_prover =
            MultiProductSumcheckMasterProver::<_, Engine::RootProductBookeepingTable, _, _>::new(
                &self.master,
                self.num_workers,
                self.num_replicas_per_worker.ilog2() as usize + y_num_vars,
                bookeeping_extra_g_u,
            );
        for value in t_at_u {
            if value.len() == 0 {
                phase_2_sumcheck_prover.add(None);
            } else {
                assert_eq!(value.len(), 1);
                phase_2_sumcheck_prover.add(Some(vec![value[0]; self.num_workers]))
            }
        }

        fiat_shamir_engine.set_seed(w_g);
        let (random_v, _f_at_v, _s_at_v, phase_2_transcript) =
            phase_2_sumcheck_prover.run(fiat_shamir_engine);

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&phase_1_transcript);
        transcript.serialize_and_push(&phase_2_transcript);

        (random_u, random_v, transcript)
    }

    fn generate_round<FS: FiatShamirEngine<F>>(
        &mut self,
        fiat_shamir_engine: &mut FS,
        round: usize,
        random_g: &CudaSlice<F>,
        w_g: F,
    ) -> (CudaSlice<F>, F, Transcript) {
        let (random_u, random_v, synthetic_round_transcript) = self.run_sumcheck(
            fiat_shamir_engine,
            round,
            &mut random_g.at_range_to(..self.worker_extra_num_vars()),
            w_g,
        );

        let sub_w_v = self.master.recv_from_workers_and_done::<Vec<F>>().unwrap();
        let mut synthetic_w_v = vec![];

        for layer_index in 0..sub_w_v[0].len() {
            let mut sub_w_v_i = vec![];
            for replica_index in 0..sub_w_v.len() {
                sub_w_v_i.push(sub_w_v[replica_index][layer_index]);
            }

            let sub_w_v_i_ext = SisuDenseMultilinearExtension::from_slice(&sub_w_v_i);
            let synthetic_w_v_i = sub_w_v_i_ext.evaluate(vec![random_v
                .at_range_to(..self.worker_extra_num_vars())
                .as_ref_host()]);
            synthetic_w_v.push(synthetic_w_v_i);
        }

        let sub_w_u = self.master.recv_from_workers_and_done().unwrap();
        let sub_w_u_ext = SisuDenseMultilinearExtension::from_slice(&sub_w_u);
        let synthetic_w_u = sub_w_u_ext.evaluate(vec![random_u
            .at_range_to(..self.worker_extra_num_vars())
            .as_ref_host()]);

        let mut next_round_sumcheck_prover =
            MultiProductSumcheckMasterProver::<_, Engine::RootProductBookeepingTable, _, _>::new(
                &self.master,
                self.num_workers,
                self.num_replicas_per_worker.ilog2() as usize + self.circuit.num_vars_at(round + 1),
                vec![],
            );
        // because we has only ONE sumcheck in this multi-sumcheck prover,
        // so we only need to pass a vector with len of 1 to this function.
        //
        // vec![vec[]; num_sumchecks] === vec![vec![]; 1] == vec![vec![]]
        next_round_sumcheck_prover.add(Some(vec![]));

        fiat_shamir_engine.set_seed(w_g);
        let (
            random_next_g,
            synthetic_w_at_next_g,
            _synthetic_q_at_next_g,
            synthetic_combining_transcript,
        ) = next_round_sumcheck_prover.run(fiat_shamir_engine);

        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&synthetic_round_transcript);
        transcript.serialize_and_push(&synthetic_w_u);
        transcript.serialize_and_push(&synthetic_w_v);
        transcript.serialize_and_push(&synthetic_combining_transcript);

        (random_next_g, synthetic_w_at_next_g[0][0], transcript)
    }

    /// Note that fiat shamir's input must be set before calling this method.
    pub fn generate_transcript<FS: FiatShamirEngine<F>>(
        &mut self,
        fiat_shamir_engine: &mut FS,
    ) -> (CudaSlice<F>, Transcript) {
        fiat_shamir_engine.begin_protocol();
        let mut synthetic_transcript = Transcript::default();

        // Compute W_output(random_g).
        let mut random_g =
            CudaSlice::on_host(fiat_shamir_engine.hash_to_fields(&F::ZERO, self.total_num_vars(0)));

        let now = Instant::now();

        // Calculate worker w_g
        let worker_w_g = self.master.recv_from_workers::<F>().unwrap();
        let synthetic_w_g_ext = SisuDenseMultilinearExtension::from_slice(&worker_w_g);
        let mut w_g = synthetic_w_g_ext.evaluate(vec![random_g
            .at_range_to(..self.worker_extra_num_vars())
            .as_ref_host()]);

        self.master.send_to_workers(&w_g).unwrap();

        println!("[General GKR Master] Run output ext: {:?}", now.elapsed());

        // Synthetize output
        let now = Instant::now();
        let mut output = vec![];
        let sub_outputs = self
            .master
            .recv_from_workers_and_done::<Vec<Vec<F>>>()
            .unwrap();
        println!("[General GKR Master] wait output: {:?}", now.elapsed());

        let now = Instant::now();
        for o in sub_outputs {
            output.extend(o);
        }

        synthetic_transcript.serialize_and_push(&output);
        println!("[General GKR Master] serialize output: {:?}", now.elapsed());

        let mut round_transcript: Transcript;
        for round in 0..self.circuit.len() {
            let now = Instant::now();
            // Ignore random alpha because we don't need this point for
            // synthetizing transcripts, but we must run this method to ensure
            // that ALL random points generated later is correct.
            let _alpha = fiat_shamir_engine.hash_to_fields(&w_g, round + 2);
            (random_g, w_g, round_transcript) =
                self.generate_round(fiat_shamir_engine, round, &random_g, w_g);
            synthetic_transcript.serialize_and_push(&round_transcript);
            println!("[General GKR Master] round {}: {:?}", round, now.elapsed());
        }

        (random_g, synthetic_transcript)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
        thread,
    };

    use sisulib::{
        circuit::{general_circuit::examples::example_general_circuit, CircuitParams},
        codegen::generator::FileGenerator,
        common::{convert_vec_field_to_string, padding_pow_of_two_size},
        field::FpSisu,
    };

    use crate::{
        channel::{self, WorkerNode},
        fiat_shamir::DefaultFiatShamirEngine,
        general_gkr::{GeneralGKRProver, GeneralGKRVerifier},
        hash::DummyHash,
        sisu_engine::CPUSisuEngine,
    };

    use super::*;

    #[test]
    fn test_synthetic_gkr() {
        let num_replicas_per_worker = 2usize.pow(1);
        let num_workers = 2usize;
        let w_output = None; //Some(Fp389::from(375));

        let mut circuit = example_general_circuit();
        circuit.finalize(HashMap::default());

        let mut all_witness = vec![];
        let mut synthetic_witness = vec![];
        let mut synthetic_output = vec![];
        for worker_index in 0..num_workers {
            let mut worker_witness = vec![];
            for replica_index in 0..num_replicas_per_worker {
                let mut replica_witness = vec![];
                for j in 0..circuit.input_size() {
                    replica_witness.push(FpSisu::from(
                        ((worker_index + 1) * (replica_index + 1) * (j + 1)) as u64,
                    ));
                }

                let evaluations = circuit.evaluate(&CircuitParams::default(), &replica_witness);
                let output = evaluations.at_layer(0, true).to_vec();
                synthetic_output.push(output);

                worker_witness.push(CudaSlice::on_host(replica_witness.clone()));

                padding_pow_of_two_size(&mut replica_witness);
                synthetic_witness.extend_from_slice(&replica_witness);
            }

            all_witness.push(worker_witness);
        }

        let (master_sender, master_receiver) = channel::default();
        let mut master_node = MasterNode::from_channel(master_receiver);

        let mut worker_nodes = vec![];
        for i in 0..num_workers {
            let (worker_sender, worker_receiver) = channel::default();
            let worker_node = WorkerNode::from_channel(i, worker_receiver, &master_sender);

            master_node.add_worker(&worker_sender);
            worker_nodes.push(worker_node);
        }

        let transcript = Arc::new(Mutex::new(Transcript::default()));

        thread::scope(|scope| {
            let master_circuit = circuit.clone();

            let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
            prover_fiat_shamir.set_seed(FpSisu::from(3));
            let transcript = transcript.clone();
            scope.spawn(move || {
                let mut master_prover =
                    SyntheticGeneralGKRProver::<_, CPUSisuEngine<_, DummyHash>, _, _>::new(
                        &master_node,
                        &master_circuit,
                        num_workers,
                        num_replicas_per_worker,
                    );

                let mut transcript = transcript.lock().unwrap();
                (_, *transcript) = master_prover.generate_transcript(&mut prover_fiat_shamir);
            });

            for (worker_index, worker_node) in worker_nodes.into_iter().enumerate() {
                let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
                prover_fiat_shamir.set_seed(FpSisu::from(3));

                let mut worker_witness = all_witness[worker_index].to_vec();

                let worker_circuit = circuit.clone();
                scope.spawn(move || {
                    let engine = CPUSisuEngine::<_, DummyHash>::new();

                    let mut worker_provers =
                        GeneralGKRProver::new(&engine, &worker_circuit, num_replicas_per_worker);
                    worker_provers.to_worker(&worker_node, num_workers, worker_index);
                    worker_provers.generate_transcript(
                        &mut prover_fiat_shamir,
                        &mut worker_witness,
                        w_output.clone(),
                    );
                });
            }
        });

        let transcript = Arc::try_unwrap(transcript).unwrap().into_inner().unwrap();

        let mut verifier_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir.set_seed(FpSisu::from(3));
        let mut verifier = GeneralGKRVerifier::new(&circuit);
        verifier.replicate(num_replicas_per_worker, num_workers);
        let result = verifier.verify_transcript(
            &mut verifier_fiat_shamir,
            &synthetic_witness,
            transcript.into_iter(),
        );
        match result {
            Ok((_, _, output)) => match w_output {
                None => assert_eq!(output, synthetic_output, "wrong output"),
                _ => (),
            },
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
