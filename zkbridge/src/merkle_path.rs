use ark_ff::{Field, PrimeField};
use sha256_circuit::{build_circuit, layer_input::LayerInput};
use sisu::{
    distributed_sisu::{generate_distributed_root_domain, DefaultSisuRunner},
    general_gkr::GeneralGKRVerifier,
    icicle_converter::IcicleConvertibleField,
};
use sisulib::{
    circuit::general_circuit::circuit::GeneralCircuit,
    codegen::generator::FuncGenerator,
    domain::{Domain, RootDomain},
};

// A info object containing merkle path circuit. It will generate sisu objects
// such that each machine will handle exactly ONE path.
pub struct MerklePathCircuitRunner<F: Field> {
    circuit: GeneralCircuit<F>,
    input_generator: LayerInput<F>,
    domain: RootDomain<F>,
    ldt_rate: usize,
    num_repetitions: usize,
    num_paths: usize,
    path_size: usize,
}

impl<F: IcicleConvertibleField + PrimeField> MerklePathCircuitRunner<F> {
    pub fn new(
        ldt_rate: usize,
        num_paths: usize,
        path_size: usize,
        num_repetitions: usize,
    ) -> Self {
        let circuit = build_circuit();
        let domain = generate_distributed_root_domain(&circuit, ldt_rate, path_size * 2);
        Self {
            circuit,
            ldt_rate,
            num_repetitions,
            input_generator: LayerInput::new(),
            domain,
            num_paths,
            path_size,
        }
    }

    pub fn generate_witness(
        &self,
        mut index: u64,
        hash: &[u8],
        path: &[Vec<u8>],
    ) -> (Vec<Vec<F>>, Vec<u8>) {
        assert_eq!(path.len(), self.path_size);

        let mut witness = vec![];

        let mut output = hash.to_vec();
        for i in 0..path.len() {
            let mut msg = vec![];
            if index % 2 == 0 {
                msg.extend(output);
                msg.extend(&path[i]);
            } else {
                msg.extend(&path[i]);
                msg.extend(output);
            }

            let info = self.input_generator.build_message_info(&msg);
            assert!(info.len() == 2);

            witness.push(info[0].input.clone());
            witness.push(info[1].input.clone());

            output = info[1].output.clone();
            index = index / 2;
        }

        (witness, output)
    }

    pub fn gen_code(&self, merkle_path_index: usize) -> Vec<FuncGenerator<F>> {
        let mut verifier = GeneralGKRVerifier::new(self.circuit());
        verifier.replicate(self.num_replicas_per_worker(), 1);

        let gkr_configs = verifier.configs(self.num_non_zero_outputs());

        let mut funcs = vec![];

        // circuit_index of merkle path is the merkle_path_index
        let f = gkr_configs.gen_code(merkle_path_index);
        funcs.extend(f);

        funcs
    }
}

impl<F: IcicleConvertibleField + PrimeField> DefaultSisuRunner<F> for MerklePathCircuitRunner<F> {
    fn domain(&self) -> Domain<F> {
        Domain::from(&self.domain)
    }

    fn ldt_rate(&self) -> usize {
        self.ldt_rate
    }

    fn num_repetitions(&self) -> usize {
        self.num_repetitions
    }

    fn circuit(&self) -> &GeneralCircuit<F> {
        &self.circuit
    }

    fn num_workers(&self) -> usize {
        self.num_paths
    }

    fn num_replicas_per_worker(&self) -> usize {
        self.path_size * 2
    }

    fn num_non_zero_outputs(&self) -> Option<usize> {
        Some(8) // W0(2) + W1(2) + H_IN(2) + H_OUT(2)
    }
}

#[cfg(test)]
mod tests {
    use crate::merkle_tree::MerkleTree;
    use ark_ff::Field;
    use sha2::digest::DynDigest;
    use sha2::Sha256;
    use sha256_circuit::constants::INIT_H_VALUES;
    use sha256_circuit::sha_calculation::Sha256Cal;
    use sisu::distributed_sisu::DefaultSisuRunner;
    use sisu::hash::SisuMimc;
    use sisu::sisu_engine::GPUFrBN254SisuEngine;
    use sisulib::common::convert_vec_field_to_string;
    use sisulib::common::dec2bin;
    use sisulib::common::serialize;
    use sisulib::field::FpSisu;
    use std::time::Instant;

    use crate::merkle_path::MerklePathCircuitRunner;

    #[test]
    fn test_merkletree_sisu() {
        println!();
        let values = vec![
            FpSisu::from(0),
            FpSisu::from(1),
            FpSisu::from(2),
            FpSisu::from(3),
            // FpSisu::from(4),
            // FpSisu::from(5),
            // FpSisu::from(6),
            // FpSisu::from(7),
            // FpSisu::from(8),
            // FpSisu::from(9),
            // FpSisu::from(10),
            // FpSisu::from(11),
            // FpSisu::from(12),
            // FpSisu::from(13),
            // FpSisu::from(14),
            // FpSisu::from(15),
        ];

        let num_paths = 2;
        let path_size = values.len().ilog2() as usize;

        let now = Instant::now();
        let merkle_tree: MerkleTree<FpSisu> = MerkleTree::from_vec(values);
        println!(
            "=====================Create Merkle Tree: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let merkle_path_sisu = MerklePathCircuitRunner::<FpSisu>::new(8, num_paths, path_size, 2);
        println!(
            "=====================Create Merkle Path Circuit: {:?}",
            now.elapsed()
        );

        let now = Instant::now();
        let expected_root = merkle_tree.root();
        let mut all_witness = vec![];
        for i in 0..num_paths {
            let (v, path) = merkle_tree.path_of(i);
            let mut hasher = Sha256::default();
            hasher.update(&serialize(&v));
            let hv = hasher.finalize_reset();

            let (witness, root) = merkle_path_sisu.generate_witness(i as u64, &hv, &path);

            assert_eq!(expected_root, root);
            all_witness.extend(witness);
        }
        println!("=====================GENERATE WITNESS: {:?}", now.elapsed());

        // let engine = CPUSisuEngine::new();

        let engine = GPUFrBN254SisuEngine::new();
        let now = Instant::now();
        merkle_path_sisu.run_sisu::<_, SisuMimc<FpSisu>>(all_witness, engine);
        println!("=====================SISU: {:?}", now.elapsed());
    }

    #[test]
    fn test_gen_init_h() {
        let mut h_bits = vec![];
        for h in INIT_H_VALUES {
            let b = dec2bin::<_, FpSisu>(h as u64, 32);
            h_bits.extend(b);
        }

        assert!(h_bits.len() == 256);

        let mut init_h_numbers = vec![];
        for i in 0..2 {
            let mut value = FpSisu::from(0);
            let mut pow = FpSisu::from(2).pow([127u64]);
            for j in 0..128 {
                value += h_bits[i * 128 + j] * pow;
                pow /= FpSisu::from(2);
            }

            init_h_numbers.push(value);
        }

        println!(
            "H_VALUES: {:?}",
            convert_vec_field_to_string(&init_h_numbers)
        );
    }

    #[test]
    fn test_gen_padding_w() {
        let original_msg = vec![0; 64]; // size in bytes
        let m = Sha256Cal::preprocess_message(&original_msg);

        let mut m_bits = vec![];
        for i in 0..m.len() {
            m_bits.extend(dec2bin::<_, FpSisu>(m[i], 8));
        }

        let m_bits = &m_bits[512..];

        assert!(m_bits.len() % 128 == 0);
        let n_numbers = m_bits.len() / 128;

        let mut padding_w_numbers = vec![];
        for i in 0..n_numbers {
            let mut value = FpSisu::from(0);
            let mut pow = FpSisu::from(2).pow([127u64]);
            for j in 0..128 {
                value += FpSisu::from(m_bits[i * 128 + j]) * pow;
                pow /= FpSisu::from(2);
            }

            padding_w_numbers.push(value);
        }

        println!(
            "PADDING_W_VALUES: {:?}",
            convert_vec_field_to_string(&padding_w_numbers)
        );
    }
}
