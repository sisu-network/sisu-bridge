use std::{
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use ark_ec::pairing::Pairing;
use ark_groth16::{Groth16, ProvingKey};
use ark_snark::SNARK;
use ark_std::rand::SeedableRng;
use circom_compat::{CircomBuilder, CircomCircuit, CircomConfig};
use rand_chacha::ChaCha8Rng;

use crate::{
    builder::{gen_wtns, CircomCustomBuilder},
    distributed::{DistributedGroth16, WorkerDeviceKeys},
    msm::MSMEngine,
};

pub trait Groth16TestWrapper<E: Pairing> {
    fn gen_key(&self) -> ProvingKey<E>;
    fn gen_wtns(&self) -> CircomCircuit<E>;

    fn run(circuit: CircomCircuit<E>, pk: &ProvingKey<E>) {
        println!();

        let now = Instant::now();
        print!("generating public inputs...");
        let inputs = circuit.get_public_inputs().unwrap();
        println!("done {:?}", now.elapsed());

        let now = Instant::now();
        print!("generating proof...");
        let proof = Groth16::<E>::create_proof_with_reduction_no_zk(circuit, &pk).unwrap();
        println!("done {:?} {:?}", now.elapsed(), proof);

        let now = Instant::now();
        print!("verifying...");
        let pvk = Groth16::<E>::process_vk(&pk.vk).unwrap();
        let verified = Groth16::<E>::verify_with_processed_vk(&pvk, &inputs, &proof).unwrap();
        println!("done {:?}", now.elapsed());

        assert!(verified);
    }

    fn run_distributed<'a, M: MSMEngine<'a, E> + 'a>(
        circuit: CircomCircuit<E>,
        pk: &ProvingKey<E>,
        num_workers: usize,
        stream: &'a M::MSMStream,
    ) {
        println!();

        let mut groth16_distributed = DistributedGroth16::<'a, E, M>::new(num_workers);

        let now = Instant::now();
        print!("distribute key...");
        let (master_pk, worker_host_pks) = groth16_distributed.setup(&pk);
        let mut worker_device_pks = vec![];
        for i in 0..worker_host_pks.len() {
            worker_device_pks.push(WorkerDeviceKeys::from_host_keys(&worker_host_pks[i]));
        }
        println!("done {:?}", now.elapsed());

        let now = Instant::now();
        print!("generating public inputs...");
        let inputs = circuit.get_public_inputs().unwrap();
        println!("done {:?}", now.elapsed());

        let now = Instant::now();
        print!("generate assignments...");
        let (r, s, worker_assignments) = groth16_distributed.generate_assignments_non_zk(circuit);
        println!("done {:?}", now.elapsed());

        let now = Instant::now();
        print!("generating worker proofs...");
        let worker_proofs = Arc::new(Mutex::new(vec![]));
        thread::scope(|scope| {
            for (key, assignment) in worker_device_pks.iter().zip(worker_assignments.into_iter()) {
                let worker_proofs = worker_proofs.clone();
                let g1_msm_config = M::create_g1_msm_config(stream);
                let g2_msm_config = M::create_g2_msm_config(stream);

                scope.spawn(move || {
                    let worker_proof = DistributedGroth16::<E, M>::worker_prove(
                        key,
                        assignment,
                        &g1_msm_config,
                        &g2_msm_config,
                    );

                    let mut worker_proofs = worker_proofs.lock().unwrap();
                    (*worker_proofs).push(worker_proof);
                });
            }
        });
        println!("done {:?}", now.elapsed());

        let worker_proofs = Arc::try_unwrap(worker_proofs)
            .unwrap()
            .into_inner()
            .unwrap();

        let now = Instant::now();
        print!("generating master proofs...");
        let proof = DistributedGroth16::<E, M>::master_prove(&master_pk, r, s, worker_proofs);
        println!("done {:?} {:?}", now.elapsed(), proof);

        let now = Instant::now();
        print!("verifying...");
        let pvk = Groth16::<E>::process_vk(&pk.vk).unwrap();
        let verified = Groth16::<E>::verify_with_processed_vk(&pvk, &inputs, &proof).unwrap();
        println!("done {:?}", now.elapsed());

        assert!(verified);
    }
}

pub struct GenericGroth16Wrapper<E: Pairing> {
    pub use_wtns_c_plus_plus: bool,
    path: String,
    cfg: CircomConfig<E>,
}

impl<E: Pairing> GenericGroth16Wrapper<E> {
    pub fn new(path: &str) -> Self {
        let wasm_path = path.to_owned() + ".wasm";
        let r1cs_path = path.to_owned() + ".r1cs";

        println!("wasm file: {}", wasm_path);
        println!("r1cs file: {}", r1cs_path);

        let now = Instant::now();
        print!("reading wasm and r1cs...");
        let mut cfg = CircomConfig::<E>::new(&wasm_path, &r1cs_path).unwrap();
        cfg.sanity_check = true;
        println!("done {:?}", now.elapsed());

        Self {
            path: path.to_owned(),
            use_wtns_c_plus_plus: false,
            cfg,
        }
    }
}

impl<E: Pairing> Groth16TestWrapper<E> for GenericGroth16Wrapper<E> {
    fn gen_key(&self) -> ProvingKey<E> {
        // let zkey_path = self.path.to_owned() + ".zkey";
        // println!("zkey file: {}", zkey_path);

        let now = Instant::now();
        print!("generate zkey...");
        let builder = CircomBuilder::new(self.cfg.clone());
        let circom = builder.setup();
        // let mut rng = thread_rng();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let pk = Groth16::<E>::generate_random_parameters_with_reduction(circom, &mut rng).unwrap();
        println!("done {:?}", now.elapsed());

        pk
    }

    fn gen_wtns(&self) -> CircomCircuit<E> {
        let input_path = self.path.to_owned() + ".json";
        println!("input file: {}", input_path);

        let mut builder = CircomBuilder::new(self.cfg.clone());

        let now = Instant::now();
        let circuit = if self.use_wtns_c_plus_plus {
            let wtns_path = gen_wtns(&self.path);

            print!("gen + parse wtns...");
            builder.build_with_wtns(&wtns_path).unwrap()
        } else {
            print!("gen wtns by arkworks...");
            builder.push_inputs_from_json_file(&input_path).unwrap();
            builder.build().unwrap()
        };
        println!("done {:?}", now.elapsed());

        circuit
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Bn254;
    use circom_compat::{CircomBuilder, CircomConfig};
    use icicle_cuda_runtime::stream::CudaStream;

    use crate::{
        builder::{parse_wtns, CircomCustomBuilder},
        msm::{ArkMSMEngine, FilterZeroIcicleMSMEngine, IcicleMSMEngine},
    };

    use super::{GenericGroth16Wrapper, Groth16TestWrapper};

    #[test]
    fn test_groth16_bigint_add() {
        let mut groth16_engine = GenericGroth16Wrapper::<Bn254>::new("../files/bigint/add");
        groth16_engine.use_wtns_c_plus_plus = true;

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();

        GenericGroth16Wrapper::run(circuit, &pk);
    }

    #[test]
    fn test_groth16_bigint_add_distributed() {
        let mut groth16_engine = GenericGroth16Wrapper::<Bn254>::new("../files/bigint/add");
        groth16_engine.use_wtns_c_plus_plus = true;

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();

        println!("=========== Ark MSM ==============");
        GenericGroth16Wrapper::run_distributed::<ArkMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            12,
            &true,
        );

        let stream = CudaStream::create().unwrap();

        println!("=========== Filter Zero Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<FilterZeroIcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            12,
            &stream,
        );

        println!("=========== Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<IcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            12,
            &stream,
        );
    }

    #[test]
    fn test_groth16_mycircuit() {
        let groth16_engine = GenericGroth16Wrapper::<Bn254>::new("../files/mycircuit/mycircuit");

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();
        GenericGroth16Wrapper::run(circuit, &pk);
    }

    #[test]
    fn test_groth16_hash_to_field() {
        let mut groth16_engine =
            GenericGroth16Wrapper::<Bn254>::new("../files/bls_signature/test_hash_to_field");
        groth16_engine.use_wtns_c_plus_plus = false;

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();
        GenericGroth16Wrapper::run(circuit, &pk);
    }

    #[test]
    fn test_groth16_hash_to_field_distributed() {
        let mut groth16_engine =
            GenericGroth16Wrapper::<Bn254>::new("../files/bls_signature/test_hash_to_field");
        groth16_engine.use_wtns_c_plus_plus = true;

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();

        println!("=========== Ark MSM ==============");
        GenericGroth16Wrapper::run_distributed::<ArkMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            1,
            &true,
        );

        let stream = CudaStream::create().unwrap();

        println!("=========== Filter Zero Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<FilterZeroIcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            1,
            &stream,
        );

        println!("=========== Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<IcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            1,
            &stream,
        );
    }

    #[test]
    fn test_groth16_fp_multiply() {
        let groth16_engine = GenericGroth16Wrapper::<Bn254>::new("../files/fp/fp_multiply");

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();
        GenericGroth16Wrapper::run(circuit, &pk);
    }

    #[test]
    fn test_groth16_fp_multiply_distributed() {
        let groth16_engine = GenericGroth16Wrapper::<Bn254>::new("../files/fp/fp_multiply");

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();

        println!("=========== Ark MSM ==============");
        GenericGroth16Wrapper::run_distributed::<ArkMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &true,
        );

        let stream = CudaStream::create().unwrap();

        println!("=========== Filter Zero Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<FilterZeroIcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &stream,
        );

        println!("=========== Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<IcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &stream,
        );
    }

    #[test]
    fn test_groth16_fp2_multiply() {
        let groth16_engine = GenericGroth16Wrapper::<Bn254>::new("../files/fp/fp2_multiply");

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();
        GenericGroth16Wrapper::run(circuit, &pk);
    }

    #[test]
    fn test_groth16_fp2_multiply_distributed() {
        let groth16_engine = GenericGroth16Wrapper::<Bn254>::new("../files/fp/fp2_multiply");

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();

        println!("=========== Ark MSM ==============");
        GenericGroth16Wrapper::run_distributed::<ArkMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &true,
        );

        let stream = CudaStream::create().unwrap();

        println!("=========== Filter Zero Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<FilterZeroIcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &stream,
        );

        println!("=========== Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<IcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &stream,
        );
    }

    #[test]
    fn test_parse_wtns() {
        let path = "../files/bls_signature/test_hash_to_field";

        let wasm_path = path.to_owned() + ".wasm";
        let r1cs_path = path.to_owned() + ".r1cs";
        let input_path = path.to_owned() + ".json";
        let wtns_path = path.to_owned() + ".wtns";

        println!("wasm file: {}", wasm_path);
        println!("r1cs file: {}", r1cs_path);
        println!("input file: {}", input_path);

        print!("reading wasm and r1cs...");
        let mut cfg = CircomConfig::<Bn254>::new(&wasm_path, &r1cs_path).unwrap();
        cfg.sanity_check = true;

        let mut builder = CircomBuilder::new(cfg);
        builder.push_inputs_from_json_file(&input_path).unwrap();

        let circuit = builder.build().unwrap();

        let x = parse_wtns::<Bn254>(&wtns_path);
        assert!(x == circuit.witness.unwrap());
    }

    #[test]
    fn test_groth16_bls_signature() {
        let mut groth16_engine =
            GenericGroth16Wrapper::<Bn254>::new("../files/bls_signature/bls_signature");
        groth16_engine.use_wtns_c_plus_plus = true;

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();
        GenericGroth16Wrapper::run(circuit, &pk);
    }

    #[test]
    fn test_groth16_bls_signature_distributed() {
        let mut groth16_engine =
            GenericGroth16Wrapper::<Bn254>::new("../files/bls_signature/bls_signature");
        groth16_engine.use_wtns_c_plus_plus = true;

        let circuit = groth16_engine.gen_wtns();
        let pk = groth16_engine.gen_key();

        println!("=========== Ark MSM ==============");
        GenericGroth16Wrapper::run_distributed::<ArkMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &true,
        );

        let stream = CudaStream::create().unwrap();

        println!("=========== Filter Zero Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<FilterZeroIcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &stream,
        );

        println!("=========== Icicle MSM ==============");
        GenericGroth16Wrapper::run_distributed::<IcicleMSMEngine<Bn254>>(
            circuit.clone(),
            &pk,
            4,
            &stream,
        );
    }
}
