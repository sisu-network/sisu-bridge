use ark_bls12_381::Bls12_381;
use ark_bn254::Bn254;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::{pairing::Pairing, Group};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{field_hashers::DefaultFieldHasher, Field, Zero};
use ark_ff::{BigInt, BigInteger, PrimeField, UniformRand};
use ark_poly::{
    multivariate::{self, Term},
    univariate::DensePolynomial,
    DenseUVPolynomial, Polynomial,
};
use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain};
use ark_poly_commit::kzg10::{Powers, UniversalParams, VerifierKey, KZG10};
use ark_poly_commit::marlin_pc::MarlinKZG10;
use ark_poly_commit::{Error, PolynomialCommitment};
use ark_std::{cfg_iter, test_rng};
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use rand::Rng;
use sha2::{digest::DynDigest, Sha256};
use sisu::channel::NoChannel;
use sisu::cuda_compat::fft::{ArkFftEngine, FFTEnginePool, IcicleFftEngine};
use sisu::cuda_compat::slice::{CudaSlice, HostVec};
use sisu::distributed_sisu::DefaultSisuRunner;
use sisu::fiat_shamir::FiatShamirEngine;
use sisu::general_gkr::{GeneralGKRProver, GeneralGKRVerifier};
use sisu::hash::{DummyHash, Hasher, SisuMimc};
use sisu::polynomial::{CPUProductBookeepingTable, GPUProductBookeepingTable};
use sisu::sisu_engine::{CPUSisuEngine, GPUFrBN254SisuEngine};
use sisu::sisu_merkle_tree::{CPUMerkleTreeEngine, GPUMerkleTreeEngine};
use sisu::standalone_fri::{StandaloneFRIProver, StandaloneFRIVerifier};
use sisu::standalone_vpd::{StandaloneVPDProver, StandaloneVPDVerifier};
use sisulib::circuit::{CircuitParams, GateF};
use sisulib::common::{
    convert_field_to_string, le2be, round_to_pow_of_two, serialize, split_number,
};
use sisulib::domain::{Domain, RootDomain};
use sisulib::field::{Fp11111119, FpFRI, FpSisu, FrBN254};
use sisulib::mle::dense::SisuDenseMultilinearExtension;
use std::ops::{Div, Mul};
use std::sync::{Arc, Mutex};
use std::{
    thread,
    time::{Duration, Instant},
};
use zkbridge::converter::read_validator_merkle_path;
use zkbridge::merkle_path::MerklePathCircuitRunner;
use zkbridge::merkle_tree::MerkleTree;
// use sisu::distributed_sisu::DefaultSisuRunner;
use sisu::vpd::VPDProver;
use sisu::{
    fiat_shamir::DefaultFiatShamirEngine, sisu_merkle_tree::SisuMerkleTree,
    vpd::generate_ldt_root_domain,
};

//use zkbridge::merkle_path::MerklePathCircuitRunner;

fn test_le2be_performance() {
    let n = 2000000;

    let mut evaluations = vec![];
    for _ in 0..n {
        evaluations.push(rand::thread_rng().gen_range(0..10000000) as usize);
    }

    let now = Instant::now();
    let mut xxx = vec![FpFRI::from(1); 3];
    let mut yyy = (FpFRI::from(0), FpFRI::from(0));
    //for _ in 0..100 {
    let mut x = evaluations[0];
    for i in 0..n - 10 {
        le2be(evaluations[i], 20);
    }
    //}

    println!("ELAPSED: {:?}", now.elapsed());
}

fn test_calculation_performance() {
    let n = 1048576 / 2;

    let mut evaluations = vec![];
    for _ in 0..n {
        evaluations.push(FpFRI::from(rand::thread_rng().gen_range(0..10000000)));
    }
    let two = FpFRI::ONE + FpFRI::ONE;

    let now = Instant::now();
    let mut yyy = (FpFRI::ZERO, FpFRI::ZERO, FpFRI::ZERO);
    //for _ in 0..100 {
    for i in 0..n - 10 {
        for i in [i, i + 1] {
            let a10 = evaluations[i];
            let a20 = evaluations[i + 1];
            let a11 = evaluations[(i + n / 2) % n];
            let a21 = evaluations[(i + n / 2 + 1) % n];

            yyy.0 += a10 * a20;
            yyy.1 += a11 * a21;
            yyy.2 += (a11 * two - a10) * (a21 * two - a20);
        }
    }

    println!("ELAPSED: {:?} {:?}", yyy, now.elapsed());
}

fn test_sha_performance() {
    type F = FrBN254;

    let slice_size = 2usize.pow(5);
    let real_n = 2usize.pow(18 + 5);
    let n = real_n / slice_size;

    let now = Instant::now();
    let mut hasher = Sha256::default();
    let mut c = 0;
    let t = F::from(123456);
    let data = serialize(&t);

    println!("DATA SIZE: {}", data.len());

    // SHA256 BUNDLE
    for _ in 0..n {
        for _ in 0..slice_size {
            hasher.update(&data);
        }
        let h = hasher.finalize_reset().to_vec();
        c += h.len();
    }
    println!(
        "{} n {} SHA256 BUNDLE ELAPSED:  {:?}",
        c,
        real_n,
        now.elapsed()
    );

    // SHA256
    for _ in 0..real_n {
        hasher.update(&data);
        let h = hasher.finalize_reset().to_vec();
        c += h.len();
    }
    println!("{} n {} SHA256 ELAPSED:  {:?}", c, real_n, now.elapsed());

    // MIMC BUNDLE
    let now = Instant::now();
    let mut x = F::ZERO;
    for _ in 0..n {
        let mut tmp = vec![];
        for _ in 0..slice_size {
            tmp.push(t);
        }
        x += batch_sisu_mimc(&tmp);
    }
    println!(
        "{:?} n {} MIMC BUNDLE ELAPSED:  {:?}",
        x,
        real_n,
        now.elapsed()
    );

    // MIMC
    let now = Instant::now();
    let mut x = F::ZERO;
    for _ in 0..real_n {
        x += sisu_mimc(t);
    }
    println!("{:?} n {} MIMC ELAPSED:  {:?}", x, real_n, now.elapsed());

    // ORIGIN MIMC
    let now = Instant::now();
    let mut x = F::ZERO;
    for _ in 0..real_n {
        x += sisu_mimc(t);
    }
    println!("{:?} n {} MIMC ELAPSED:  {:?}", x, real_n, now.elapsed());
}

fn test_serialize_performance() {
    let n = 1000000;

    let now = Instant::now();
    let mut hasher = Sha256::default();
    for _ in 0..n {
        let mut t = FpFRI::from(0);
        let x = serialize(&t);
    }

    println!("ELAPSED:  {:?}", now.elapsed());
}

/// Copied from ark-poly-commit-test crate
fn trim_kzg_ark<E: Pairing>(
    pp: &UniversalParams<E>,
    mut supported_degree: usize,
) -> (Powers<E>, VerifierKey<E>) {
    if supported_degree == 1 {
        supported_degree += 1;
    }
    let powers_of_g = pp.powers_of_g[..=supported_degree].to_vec();
    let powers_of_gamma_g = (0..=supported_degree)
        .map(|i| pp.powers_of_gamma_g[&i])
        .collect();

    let powers = Powers {
        powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
        powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
    };
    let vk = VerifierKey {
        g: pp.powers_of_g[0],
        gamma_g: pp.powers_of_gamma_g[&0],
        h: pp.h,
        beta_h: pp.beta_h,
        prepared_h: pp.prepared_h.clone(),
        prepared_beta_h: pp.prepared_beta_h.clone(),
    };
    (powers, vk)
}

fn test_kzg_ark() {
    let degree = 2usize.pow(18);
    type Curve = Bn254;
    type ScalarField = <Curve as Pairing>::ScalarField;
    type Polynomial = DensePolynomial<ScalarField>;

    let rng = &mut test_rng();
    let now = Instant::now();
    let pp = KZG10::<Curve, Polynomial>::setup(degree, false, rng).unwrap();
    println!("SETUP {:?}", now.elapsed());
    let (ck, vk) = trim_kzg_ark(&pp, degree);
    let p = Polynomial::rand(degree, rng);
    let hiding_bound = Some(1);
    let now = Instant::now();
    let (comm, rand) =
        KZG10::<Curve, Polynomial>::commit(&ck, &p, hiding_bound, Some(rng)).unwrap();
    println!("COMMIT {:?}", now.elapsed());
    // let point = ScalarField::rand(rng);
    // let value = p.evaluate(&point);
    // let proof = KZG10::open(&ck, &p, point, &rand).unwrap();
    // assert!(
    //     KZG10::check(&vk, &comm, point, value, &proof).unwrap(),
    //     "proof was incorrect for max_degree = {}, polynomial_degree = {}, hiding_bound = {:?}",
    //     degree,
    //     p.degree(),
    //     hiding_bound,
    // );
}

fn test_vpd() {
    let ldt_rate = 32;
    let num_vars = 18;

    let root_domain = generate_ldt_root_domain(num_vars, ldt_rate);
    let ldt_domain = Domain::from(&root_domain);

    let mut random_evaluations = vec![];
    for _ in 0..2usize.pow(num_vars as u32) {
        random_evaluations.push(FpSisu::from(rand::thread_rng().gen_range(0..100) as u64))
    }

    let mle = SisuDenseMultilinearExtension::from_slice(&random_evaluations);
    let mut input = vec![];
    for i in 0..num_vars {
        input.push(FpSisu::from(i as u64));
    }

    let mut prover = VPDProver::<FpSisu, Sha256>::setup(ldt_domain, ldt_rate);
    let mut prover_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
    prover_fiat_shamir_engine.set_seed(FpSisu::ZERO);

    println!("BEGIN");
    let now = Instant::now();
    let mut commitment = prover.commit(mle.evaluations());
    println!("COMMIT {:?}", now.elapsed());

    let transcript =
        prover.generate_transcript(&mut prover_fiat_shamir_engine, &mut commitment, &input);
    println!("GENERATE TRANSCRIPT {:?}", now.elapsed());

    let now = Instant::now();
    let mut verifier_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
    verifier_fiat_shamir_engine.set_seed(FpSisu::ZERO);

    let verifier = prover.get_verifier();
    let output = verifier.verify_transcript(
        &mut verifier_fiat_shamir_engine,
        commitment.to_transcript().into_iter(),
        transcript.into_iter(),
        &input,
    );
    println!("VERIFY TRANSCRIPT {:?}", now.elapsed());

    assert_eq!(output.unwrap(), mle.evaluate(vec![&input]));
}

fn test_merkle_tree_sisu(ldt_rate: usize, num_repetitions: usize) {
    println!();

    let num_paths = 2;
    let path_size = 16usize;
    let mut values = vec![];
    for i in 0..2usize.pow(path_size as u32) {
        values.push(FpSisu::from(i as u64));
    }

    let now = Instant::now();
    let merkle_tree: MerkleTree<FpSisu> = MerkleTree::from_vec(values);
    println!(
        "=====================Create Merkle Tree: {:?}",
        now.elapsed()
    );

    let now = Instant::now();
    let merkle_path_sisu =
        MerklePathCircuitRunner::<FpSisu>::new(ldt_rate, num_paths, path_size, num_repetitions);
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

    // let mt_engine = CPUMerkleTreeEngine::<_, SisuMimc<FpSisu>>::new();
    // let engine = CPUSisuEngine::<FpSisu, SisuMimc<FpSisu>>::new();
    let engine = GPUFrBN254SisuEngine::new();

    let now = Instant::now();
    merkle_path_sisu.run_sisu::<_, SisuMimc<FpSisu>>(all_witness, engine);
    println!("=====================SISU: {:?}", now.elapsed());
}

fn test_merkle_tree_gkr() {
    println!();

    let num_paths = 8;
    let path_size = 16usize;
    let mut values = vec![];
    for i in 0..2usize.pow(path_size as u32) {
        values.push(FpSisu::from(i as u64));
    }

    let now = Instant::now();
    let merkle_tree: MerkleTree<FpSisu> = MerkleTree::from_vec(values);
    println!(
        "=====================Create Merkle Tree: {:?}",
        now.elapsed()
    );

    let now = Instant::now();
    let merkle_path_sisu = MerklePathCircuitRunner::<FpSisu>::new(1, num_paths, path_size, 1);
    println!(
        "=====================Create Merkle Path Circuit: {:?}",
        now.elapsed()
    );

    let circuit = merkle_path_sisu.circuit();
    for i in 0..circuit.len() {
        let mut num_gates = vec![0; circuit.len_at(i + 1)];
        for gate in circuit.layer(i).gates.iter() {
            num_gates[gate.left.value()] = 1;
        }

        let used_gates: usize = num_gates.iter().sum();
        println!(
            "USED GATES: {}   TOTAL GATES: {}",
            used_gates,
            circuit.len_at(i)
        );
    }

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

        for w in witness {
            all_witness.push(CudaSlice::on_host(w));
        }
    }
    println!("=====================GENERATE WITNESS: {:?}", now.elapsed());

    // let engine = CPUSisuEngine::<FpSisu, SisuMimc<FpSisu>>::new();
    let engine = GPUFrBN254SisuEngine::new();
    let now = Instant::now();
    let gkr_prover = GeneralGKRProver::<_, _, NoChannel, NoChannel>::new(
        &engine,
        merkle_path_sisu.circuit(),
        merkle_path_sisu.num_replicas_per_worker() * merkle_path_sisu.num_workers(),
    );

    let mut prover_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
    prover_fiat_shamir_engine.set_seed(FpSisu::from(3));
    let (_, _, transcript) = gkr_prover.generate_transcript(
        &mut prover_fiat_shamir_engine,
        &mut all_witness,
        merkle_path_sisu.num_non_zero_outputs(),
    );
    println!(
        "=====================DONE FIRST PROVER: {:?}",
        now.elapsed()
    );

    let now = Instant::now();
    let mut prover_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
    prover_fiat_shamir_engine.set_seed(FpSisu::from(3));
    let (_, _, transcript) = gkr_prover.generate_transcript(
        &mut prover_fiat_shamir_engine,
        &mut all_witness,
        merkle_path_sisu.num_non_zero_outputs(),
    );
    println!(
        "=====================DONE FINAL PROVER: {:?}",
        now.elapsed()
    );

    let mut verifier = GeneralGKRVerifier::new(merkle_path_sisu.circuit());
    verifier.replicate(
        merkle_path_sisu.num_replicas_per_worker() * merkle_path_sisu.num_workers(),
        1,
    );

    let mut verifier_fiat_shamir_engine = DefaultFiatShamirEngine::default_fpsisu();
    verifier_fiat_shamir_engine.set_seed(FpSisu::from(3));
    verifier
        .verify_transcript(
            &mut verifier_fiat_shamir_engine,
            &[],
            transcript.into_iter(),
        )
        .unwrap();
}

fn test_fri() {
    let ldt_rate = 8;
    let n_workers = 8usize;
    let degree = 2usize.pow(18);
    let index = 1;

    let mut poly_coeffs = vec![];
    for i in 0..n_workers {
        let mut poly = vec![];
        for j in 0..degree {
            poly.push(FpSisu::from(((i + 1) * (j + 1)) as u32));
        }

        poly_coeffs.push(poly);
    }

    let root_domain = RootDomain::new(round_to_pow_of_two(degree) * ldt_rate);
    let fri_domain = Domain::from(&root_domain);

    let mut evaluations = vec![];
    for i in 0..poly_coeffs.len() {
        let mut slice = CudaSlice::on_host(fri_domain.evaluate(&poly_coeffs[i]));
        slice.as_ref_device();
        evaluations.push(slice);
    }

    let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
    prover_fiat_shamir.set_seed(FpSisu::ZERO);

    //  let engine = CPUSisuEngine::<_, SisuMimc<FpSisu>>::new();
    let engine = GPUFrBN254SisuEngine::new();

    let now = Instant::now();
    let prover = StandaloneFRIProver::new(&engine, ldt_rate, fri_domain);
    let (commitment, _) = prover.commit(
        &mut prover_fiat_shamir,
        round_to_pow_of_two(degree),
        evaluations.clone(),
        true,
    );
    println!("FIRST COMMIT: {:?}", now.elapsed());

    let now = Instant::now();
    let query_transcript = prover.generate_transcript(index, &commitment);
    println!("FIRST PROVE: {:?}", now.elapsed());

    let now = Instant::now();
    let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
    prover_fiat_shamir.set_seed(FpSisu::ZERO);

    let (commitment, _) = prover.commit(
        &mut prover_fiat_shamir,
        round_to_pow_of_two(degree),
        evaluations.clone(),
        true,
    );
    println!("COMMIT: {:?}", now.elapsed());

    let now = Instant::now();
    let query_transcript = prover.generate_transcript(index, &commitment);
    println!("PROVE: {:?}", now.elapsed());

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

fn test_vpd_mle() {
    let num_workers = 8;
    let ldt_rate = 8;
    let num_vars = 18;
    let num_repetitions = 8;

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
        let mut x = CudaSlice::on_host(tmp);
        x.as_ref_device();
        worker_evaluations.push(x);
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

    let now = Instant::now();
    let mut commitment = prover.commit(worker_evaluations.clone());
    println!("COMMITMENT {:?}", now.elapsed());

    let now = Instant::now();
    let transcript = prover.generate_transcript(&mut prover_fiat_shamir, &mut commitment, &input);
    println!("GENERATE TRANSCRIPT {:?}", now.elapsed());

    let mut prover_fiat_shamir = DefaultFiatShamirEngine::default_fpsisu();
    prover_fiat_shamir.set_seed(FpSisu::ZERO);

    let now = Instant::now();
    let mut commitment = prover.commit(worker_evaluations);
    println!("FINAL COMMITMENT {:?}", now.elapsed());

    let now = Instant::now();
    let transcript = prover.generate_transcript(&mut prover_fiat_shamir, &mut commitment, &input);
    println!("FINAL GENERATE TRANSCRIPT {:?}", now.elapsed());

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

fn test_read_validator_merkle_proof() {
    let proofs = read_validator_merkle_path("./zkbridge/fixtures/validator_merrkle_path.bin");

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
        let (witness, _) =
            merkle_path_sisu.generate_witness(proofs[i].index, &proofs[i].leaf, &proofs[i].path);

        for w in witness {
            all_witness.push(CudaSlice::on_host(w));
        }
    }
    println!("=====================GENERATE WITNESS: {:?}", now.elapsed());

    let engine = GPUFrBN254SisuEngine::new();
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
    verifier
        .verify_transcript(
            &mut verifier_fiat_shamir_engine,
            &[],
            transcript.into_iter(),
        )
        .unwrap();
}

fn num_bit_of(base: usize) -> usize {
    let mut base_num_bit = 0;
    loop {
        base_num_bit += 1;
        if base >> (base_num_bit) == 0 {
            break;
        }
    }
    base_num_bit
}

fn test_mul_solution() {
    let num_bit = 10;
    let base = 911;

    let b = FpFRI::from(base as u64);
    println!(
        "DIFF BIT: {}-{}={}",
        num_bit,
        num_bit_of(base),
        num_bit - num_bit_of(base)
    );
    thread::sleep(Duration::from_secs(3));

    for i in 0..2usize.pow(2 * num_bit as u32) {
        let z = FpFRI::from(i as u64);
        let mut count = 0;
        for j in 0..2usize.pow(num_bit as u32) {
            let y = FpFRI::from(j as u64);
            let x = (z - y) / b;

            let t = format!("{:?}", x);
            let start = t.find('[').unwrap();
            let end = t.find(']').unwrap();

            let t = t[start + 1..end].to_string().parse::<usize>().unwrap();
            if t < 2usize.pow(num_bit as u32 + 2) {
                // println!("{:?} = {:?} * 97 + {:?}", z, x, y);
                count += 1;
            }
        }

        println!("{:?}: {}", z, count);
    }
}

fn test_split_number() {
    let size = 18;
    let a = 1234566776;
    let mut s = 0;

    let now = Instant::now();

    for _ in 0..2usize.pow(size as u32) {
        let (x, y) = split_number(&a, size);
        let (z, t) = split_number(&x, size);
        s += y + z + t;
    }
    println!("{} {:?}", s, now.elapsed());
}

fn test_fft() {
    type F = FrBN254;

    let poly_size = 2usize.pow(18);
    let eval_size = 2usize.pow(23);

    let mut poly_coeffs = vec![];
    for i in 0..poly_size {
        poly_coeffs.push(F::from(i as u64));
    }

    let root_domain = RootDomain::<F>::new(eval_size);
    let xdomain = Domain::from(&root_domain);

    let now = Instant::now();
    let evaluations = xdomain.evaluate(&poly_coeffs);

    println!(
        "my evaluation_size={:?}  time={:?}",
        evaluations.len(),
        now.elapsed()
    );

    let domain = GeneralEvaluationDomain::<F>::new(eval_size).unwrap();
    let now = Instant::now();
    let evaluations = domain.fft(&poly_coeffs);

    println!(
        "ark evaluation_size={:?}  time={:?}",
        evaluations.len(),
        now.elapsed()
    );
}

fn test_domain() {
    type F = FpFRI;

    let eval_size = 2usize.pow(23);

    let root_domain = RootDomain::<F>::new(eval_size);
    let xdomain = Some(Domain::from(&root_domain));
    let evaluations = vec![GateF::<F>::DomainPd(false, 0, 1, 100); 16000000];

    let mut params = CircuitParams::default();
    params.domain = xdomain;

    let now = Instant::now();
    let mut x = 0;
    let mut t = F::ONE;
    for (i, k) in evaluations.iter().enumerate() {
        let f = xdomain.unwrap()[xdomain.unwrap().get_opposite_index_of(i)];
        let y = k.to_value(&params);
        t += y;
        t += f;
    }

    println!(
        "ark evaluation_size={:?} {:?}  time={:?}",
        x,
        t,
        now.elapsed()
    );
}

fn batch_sisu_mimc<F: Field>(xs: &[F]) -> F {
    let mut result = F::ZERO;
    for x in xs {
        result += sisu_mimc(result + x.clone());
    }

    result
}

fn batch_sisu_mimc_2<F: Field>(xs: &[F]) -> F {
    let mut result = F::from(13u64);
    for (i, x) in xs.into_iter().enumerate() {
        result = (result + x.pow(&[5u64]) + F::from(i as u64)).pow(&[5u64]);
    }

    result
}

fn sisu_mimc<F: Field>(x: F) -> F {
    x.pow(&[3u64])
}

fn mimc<F: Field>(x: F) -> F {
    let mut r = F::ZERO;
    for _ in 0..91 {
        r = (r + x).pow(&[3u64])
    }
    r
}

fn mimc_origin<F: Field>(mut xl: F, mut xr: F, constants: &[F]) -> F {
    assert_eq!(constants.len(), 322);

    for i in 0..322 {
        let mut tmp1 = xl;
        tmp1.add_assign(&constants[i]);
        let mut tmp2 = tmp1;
        tmp2.square_in_place();
        tmp2.mul_assign(&tmp1);
        tmp2.add_assign(&xr);
        xr = xl;
        xl = tmp2;
    }

    xl
}

fn test_collision_mimc() {
    type F = Fp11111119;
    let p = 11111119;

    for i in 0..p {
        let x = F::from(i);
        for j in 0..p {
            if i == j {
                continue;
            }

            let y = F::from(j);
            // if batch_sisu_mimc(&[x]) == batch_sisu_mimc(&[y]) {
            //     println!("{:?} {:?}", x, y);
            // }
            // if sisu_mimc(x) == sisu_mimc(y) {
            //     println!("{:?} {:?}", x, y);
            // }
            // if mimc(x) == mimc(y) {
            //     println!("{:?} {:?}", x, y);
            // }
            if x.pow(&[3u64]) == y.pow(&[3u64]) {
                println!("{:?} {:?}", x, y);
            }
        }
        println!("{}", i);
    }
}

fn test_batch_mimc() {
    type F = Fp11111119;
    let p = 11111119;

    let mut c = vec![];
    for i in 0..322 {
        c.push(F::from(i));
    }

    for i in 0..p {
        let x = F::from(i);
        for j in 0..p {
            let y = F::from(j);

            for ki in 0..p {
                let kx = F::from(ki);

                for kj in 0..p {
                    let ky = F::from(kj);

                    if i == ki && j == kj {
                        continue;
                    }

                    // if mimc_origin(x, y, &c) == mimc_origin(kx, ky, &c) {
                    //     println!("{:?} {:?} == {:?} {:?}", x, y, kx, ky);
                    // }

                    if batch_sisu_mimc_2(&[x, y]) == batch_sisu_mimc_2(&[kx, ky]) {
                        println!("{:?} {:?} == {:?} {:?}", x, y, kx, ky);
                    }
                }
            }
            println!("{} {}", i, j);
        }
    }
}

fn test_msm_performance<E: Pairing>(n: usize) {
    let super_secret = E::ScalarField::from(1234567u64);
    let mut scalar_secrets = vec![E::ScalarField::ONE];
    for _ in 1..n {
        scalar_secrets.push(*scalar_secrets.last().unwrap() * super_secret);
    }

    let scalar_size = E::ScalarField::MODULUS_BIT_SIZE as usize;
    let window_g1 = FixedBase::get_mul_window_size(n);

    let g1_generator = E::G1::generator();
    let secret_g1_table = FixedBase::get_window_table(scalar_size, window_g1, g1_generator);

    let secrets_pow_g1 = FixedBase::msm(scalar_size, window_g1, &secret_g1_table, &scalar_secrets);

    let now = Instant::now();
    let secrets_pow_g1_affine = E::G1::normalize_batch(&secrets_pow_g1);
    let x = <E::G1 as VariableBaseMSM>::msm(&secrets_pow_g1_affine, &scalar_secrets)
        .unwrap()
        .into_affine();
    println!("--msm mul {:?}", now.elapsed());

    let num_thread = 2;
    let size_each_thread = scalar_secrets.len() / num_thread;
    let total_sum = Arc::new(Mutex::new(E::G1::zero()));

    let now = Instant::now();
    thread::scope(|scope| {
        for i in 0..num_thread {
            let worker_scalar =
                scalar_secrets[i * size_each_thread..(i + 1) * size_each_thread].to_vec();
            let worker_pow =
                secrets_pow_g1_affine[i * size_each_thread..(i + 1) * size_each_thread].to_vec();
            let total_sum = total_sum.clone();
            scope.spawn(move || {
                let now = Instant::now();
                let s = <E::G1 as VariableBaseMSM>::msm(&worker_pow, &worker_scalar)
                    .unwrap()
                    .into_affine();
                println!("--msm mul thread {} {:?}", i, now.elapsed());

                let mut total_sum = total_sum.lock().unwrap();
                *total_sum += s;
            });
        }
    });
    println!("--msm thread {:?} {:?} {:?}", now.elapsed(), x, total_sum);
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn test_parallel() {
    let n = 2usize.pow(26);
    let mut v = vec![];

    for i in 0..n {
        v.push(FpFRI::from(i as u64));
    }

    let s = &mut FpFRI::zero();

    let now = Instant::now();
    let _: Vec<_> = v.iter().map(|x| *s += x).collect();
    println!("NO PARALLEL: {:?} {:?}", now.elapsed(), s);
}

fn test_thread_write_same_slice() {
    let n_workers = 16;
    let size_per_workers = 32;

    let mut common_vec = vec![];
    for i in 0..n_workers {
        common_vec.push(Mutex::new(vec![0; size_per_workers]));
    }

    let common_vec = Arc::new(common_vec);

    thread::scope(|s| {
        for i in 0..n_workers {
            let common_vec = common_vec.clone();
            s.spawn(move || {
                for j in 0..size_per_workers {
                    let mut v = common_vec[i].lock().unwrap();
                    (*v)[j] = i * size_per_workers + j;
                }
            });
        }
    });

    println!("{:?}", common_vec);
}

fn gen_ah(x: FrBN254) -> FrBN254 {
    let mut r = FrBN254::zero();

    for i in 0..2 {
        r = (r + x + FrBN254::from(u32::MAX)).pow(&[3]);
    }

    r
}

fn field_to_bit<F: PrimeField>(x: F) -> Vec<bool> {
    x.into_bigint().to_bits_le()
}

fn what_interval<F: PrimeField>(x: F, interval: BigUint) -> usize {
    let y: BigUint = x.into_bigint().into();

    let k = y.div(interval);

    let t: u64 = k.to_u64().unwrap();
    t as usize
}

fn test_bigint() {
    let dist_size = 1000;
    let n = 1000000;

    let mut distribute = vec![0; dist_size];
    let modulus: BigUint = FrBN254::MODULUS.into();
    let interval = modulus.div(BigUint::from(dist_size));
    let mut bit_change = vec![0; 254];
    let mut rng = test_rng();

    for i in 0..n {
        let z = rng.gen_range(0..u32::MAX);

        let x = FrBN254::from(z);
        let r0 = gen_ah(x);
        let r1 = gen_ah(x + FrBN254::ONE);

        let b0 = field_to_bit(r0);
        let b1 = field_to_bit(r1);

        for j in 0..254 {
            if b0[j] != b1[j] {
                bit_change[j] += 1;
            }
        }

        distribute[what_interval(r0, interval.clone())] += 1;
    }

    for r in bit_change {
        println!("AVG RATIO: {}", r as f64 / n as f64);
    }

    println!("{:?}", distribute);
}

fn main() {
    #[cfg(feature = "parallel")]
    println!("In parallel mode");

    //   test_bigint();
    // test_batch_mimc();
    // test_collision_mimc();
    // test_fft();
    // test_parallel();
    // test_kzg_ark();
    // test_sha_performance();
    // test_merkle_tree_sisu(8, 1);
    // test_merkle_tree_gkr();
    //test_fri();
    test_vpd_mle();

    // test_read_validator_merkle_proof();
    // test_mimc_new();
    // test_thread_write_same_slice();

    // test_domain();
    // test_vpd();
    // test_split_number();
    // println!("{x:?}");
    // test_mul_solution();
    // test_msm_performance::<Bn254>(2usize.pow(18));
}
