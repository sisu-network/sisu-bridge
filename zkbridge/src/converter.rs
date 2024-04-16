use std::fs;

use ark_serialize::Read;

const VALIDATOR_MERKLE_PROOF_SIZE: usize = 552;

#[derive(Default, Debug)]
pub struct MerkleProof {
    pub index: u64,
    pub leaf: [u8; 32],
    pub path: Vec<Vec<u8>>,
}

impl MerkleProof {
    pub fn deserialize(&mut self, buf: &[u8]) {
        assert_eq!((buf.len() - 40) % 32, 0);

        let mut index_buf = [0; 8];
        index_buf.copy_from_slice(&buf[0..8]);

        // The index in buf is a generalized index, so we need to convert to
        // the normal index.
        self.index = u64::from_be_bytes(index_buf) - 2u64.pow(16);
        self.leaf.copy_from_slice(&buf[8..40]);

        self.path = vec![];
        let mut i = 40;
        while i < buf.len() {
            self.path.push(buf[i..i + 32].to_vec());

            i += 32;
        }
    }

    pub fn path_size(&self) -> usize {
        self.path.len()
    }
}

pub fn read_validator_merkle_path(path: &str) -> Vec<MerkleProof> {
    let mut f = fs::File::open(path).unwrap();

    let mut buf = [0; VALIDATOR_MERKLE_PROOF_SIZE];
    let mut result = vec![];

    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }

        if n != buf.len() {
            panic!("invalid size {}", n)
        }

        let mut path = MerkleProof::default();
        path.deserialize(&buf);

        assert_eq!(path.path_size(), 16);

        result.push(path);
    }

    result
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use sisu::cuda_compat::slice::CudaSlice;
    use sisu::hash::DummyHash;
    use sisu::sisu_engine::CPUSisuEngine;
    use sisulib::codegen::generator::FileGenerator;
    use sisulib::common::convert_vec_field_to_string;
    use sisulib::field::FpSisu;

    use super::read_validator_merkle_path;
    use crate::merkle_path::MerklePathCircuitRunner;
    use sisu::fiat_shamir::FiatShamirEngine;
    use sisu::{
        channel::NoChannel,
        distributed_sisu::DefaultSisuRunner,
        fiat_shamir::DefaultFiatShamirEngine,
        general_gkr::{GeneralGKRProver, GeneralGKRVerifier},
    };

    #[test]
    fn test_read_validator_merkle_proof() {
        let proofs = read_validator_merkle_path("fixtures/validator_merrkle_path.bin");

        let now = Instant::now();
        println!(
            "=====================Create Merkle Tree: {:?}",
            now.elapsed()
        );

        let ldt_rate = 8;
        let num_repetitions = 2;

        let now = Instant::now();
        let merkle_path_sisu = MerklePathCircuitRunner::<FpSisu>::new(
            ldt_rate,
            proofs.len(),
            proofs[0].path_size(),
            num_repetitions,
        );
        println!(
            "=====================Create Merkle Path Circuit: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let mut all_witness = vec![];
        for i in 0..proofs.len() {
            let (witness, _) = merkle_path_sisu.generate_witness(
                proofs[i].index,
                &proofs[i].leaf,
                &proofs[i].path,
            );

            for w in witness {
                all_witness.push(CudaSlice::on_host(w))
            }
        }
        println!("=====================GENERATE WITNESS: {:?}", now.elapsed());

        let engine = CPUSisuEngine::<_, DummyHash>::new();
        let gkr_prover = GeneralGKRProver::<_, _, NoChannel, NoChannel>::new(
            &engine,
            merkle_path_sisu.circuit(),
            merkle_path_sisu.num_replicas_per_worker(),
        );

        let mut prover_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
        prover_fiat_shamir_engine.set_seed(FpSisu::from(3));
        let (_, _, transcript) = gkr_prover.generate_transcript(
            &mut prover_fiat_shamir_engine,
            &mut all_witness[..merkle_path_sisu.num_replicas_per_worker()],
            merkle_path_sisu.num_non_zero_outputs(),
        );

        let mut verifier = GeneralGKRVerifier::new(merkle_path_sisu.circuit());
        verifier.replicate(merkle_path_sisu.num_replicas_per_worker(), 1);

        let mut verifier_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
        verifier_fiat_shamir_engine.set_seed(FpSisu::from(3));
        let (_, _, output) = verifier
            .verify_transcript(
                &mut verifier_fiat_shamir_engine,
                &[],
                transcript.into_iter(),
            )
            .unwrap();

        for i in 0..output.len() {
            println!(
                "OUTPUT[{}]: {:?}",
                i,
                convert_vec_field_to_string(&output[i])
            );
        }

        let gkr_transcript = verifier.extract_transcript(transcript.into_iter());
        println!(
            "Transcript: {:?} {}",
            convert_vec_field_to_string(&gkr_transcript.to_vec()),
            gkr_transcript.to_vec().len(),
        );

        let gkr_configs = verifier.configs(merkle_path_sisu.num_non_zero_outputs());
        let mut file_gen =
            FileGenerator::<FpSisu>::new("../bls-circom/circuit/sisu/configs.gen.circom");

        let f = gkr_configs.gen_code(0);
        file_gen.extend_funcs(f);
        file_gen.create();
    }
}
