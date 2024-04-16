use std::{marker::PhantomData, time::Instant};

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, Group};
use ark_ff::PrimeField;
use ark_groth16::{
    r1cs_to_qap::{LibsnarkReduction, R1CSToQAP},
    Proof, ProvingKey,
};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem, OptimizationGoal};
use ark_std::{
    end_timer,
    ops::{AddAssign, Mul},
    rand::Rng,
    start_timer, Zero,
};

use crate::msm::MSMEngine;

const G1_PERFORMANCE_RATE: f64 = 1.0;
const G2_PERFORMANCE_RATE: f64 = 2.0;

pub enum DistributedKeyType {
    A,
    B1,
    B2,
    H,
    L,
}

#[derive(Clone, Debug)]
pub enum DistributedKeySetup {
    A(usize, usize, Vec<usize>),
    B1(usize, usize, Vec<usize>),
    B2(usize, usize, Vec<usize>),
    H(usize, usize, Vec<usize>),
    L(usize, usize, Vec<usize>),
}

#[derive(Clone)]
pub enum DistributedHostKey<'a, E: Pairing, M: MSMEngine<'a, E>> {
    /// The elements `a_i * G` in `E::G1`.
    A(Vec<M::HostG1Affine>),
    /// The elements `b_i * G` in `E::G1`.
    B1(Vec<M::HostG1Affine>),
    /// The elements `b_i * H` in `E::G2`.
    B2(Vec<M::HostG2Affine>),
    /// The elements `h_i * G` in `E::G1`.
    H(Vec<M::HostG1Affine>),
    /// The elements `l_i * G` in `E::G1`.
    L(Vec<M::HostG1Affine>),
}

#[derive(Clone)]
pub enum DistributedDeviceKey<'a, E: Pairing, M: MSMEngine<'a, E>> {
    /// The elements `a_i * G` in `E::G1`.
    A(M::DeviceG1Affines),
    /// The elements `b_i * G` in `E::G1`.
    B1(M::DeviceG1Affines),
    /// The elements `b_i * H` in `E::G2`.
    B2(M::DeviceG2Affines),
    /// The elements `h_i * G` in `E::G1`.
    H(M::DeviceG1Affines),
    /// The elements `l_i * G` in `E::G1`.
    L(M::DeviceG1Affines),
}

impl<'a, E: Pairing, M: MSMEngine<'a, E>> DistributedDeviceKey<'a, E, M> {
    pub fn from_host_key(hk: &DistributedHostKey<'a, E, M>) -> Self {
        match hk {
            DistributedHostKey::A(x) => DistributedDeviceKey::A(M::host_to_device_g1_affines(x)),
            DistributedHostKey::B1(x) => DistributedDeviceKey::B1(M::host_to_device_g1_affines(x)),
            DistributedHostKey::B2(x) => DistributedDeviceKey::B2(M::host_to_device_g2_affines(x)),
            DistributedHostKey::H(x) => DistributedDeviceKey::H(M::host_to_device_g1_affines(x)),
            DistributedHostKey::L(x) => DistributedDeviceKey::L(M::host_to_device_g1_affines(x)),
        }
    }
}

#[derive(Debug)]
pub enum DistributedProof<E: Pairing> {
    A(E::G1),
    B1(E::G1),
    B2(E::G2),
    H(E::G1),
    L(E::G1),
}

pub struct WorkerHostKeys<'a, E: Pairing, M: MSMEngine<'a, E>> {
    keys: Vec<DistributedHostKey<'a, E, M>>,
}

impl<'a, E: Pairing, M: MSMEngine<'a, E>> WorkerHostKeys<'a, E, M> {
    pub fn default() -> Self {
        Self { keys: vec![] }
    }

    pub fn push(&mut self, key: DistributedHostKey<'a, E, M>) {
        self.keys.push(key);
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn get(&self, index: usize) -> &DistributedHostKey<'a, E, M> {
        &self.keys[index]
    }
}

pub struct WorkerDeviceKeys<'a, E: Pairing, M: MSMEngine<'a, E>> {
    keys: Vec<DistributedDeviceKey<'a, E, M>>,
}

impl<'a, E: Pairing, M: MSMEngine<'a, E>> WorkerDeviceKeys<'a, E, M> {
    pub fn from_host_keys(hk: &WorkerHostKeys<'a, E, M>) -> Self {
        let mut device_keys = Self { keys: vec![] };
        for key in hk.keys.iter() {
            device_keys
                .keys
                .push(DistributedDeviceKey::from_host_key(key))
        }

        device_keys
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn get(&self, index: usize) -> &DistributedDeviceKey<'a, E, M> {
        &self.keys[index]
    }
}

#[derive(Clone)]
pub struct WorkerAssignments<'a, E: Pairing, M: MSMEngine<'a, E>> {
    assignments: Vec<Vec<M::HostScalarField>>,
}

impl<'a, E: Pairing, M: MSMEngine<'a, E>> WorkerAssignments<'a, E, M> {
    pub fn default() -> Self {
        Self {
            assignments: vec![],
        }
    }

    pub fn push(&mut self, assignment: Vec<M::HostScalarField>) {
        self.assignments.push(assignment);
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }
}

#[derive(Debug)]
pub struct WorkerProofs<E: Pairing> {
    proofs: Vec<DistributedProof<E>>,
}

impl<E: Pairing> WorkerProofs<E> {
    pub fn default() -> Self {
        Self { proofs: vec![] }
    }

    pub fn push(&mut self, proof: DistributedProof<E>) {
        self.proofs.push(proof);
    }
}

pub struct MasterKey<E: Pairing> {
    // First elements in A, B1, B2 query is always multiply with ONE, so put
    // them in the master.
    pub first_a_query: E::G1Affine,
    pub first_b1_query: E::G1Affine,
    pub first_b2_query: E::G2Affine,

    /// The `alpha * G`, where `G` is the generator of `E::G1`.
    pub alpha_g1: E::G1Affine,
    /// The `alpha * H`, where `H` is the generator of `E::G2`.
    pub beta_g2: E::G2Affine,
    /// The `gamma * H`, where `H` is the generator of `E::G2`.
    pub gamma_g2: E::G2Affine,
    /// The `delta * H`, where `H` is the generator of `E::G2`.
    pub delta_g2: E::G2Affine,
    /// The element `beta * G` in `E::G1`.
    pub beta_g1: E::G1Affine,
    /// The element `delta * G` in `E::G1`.
    pub delta_g1: E::G1Affine,
}

impl<E: Pairing> MasterKey<E> {
    pub fn from(pk: &ProvingKey<E>) -> Self {
        Self {
            first_a_query: pk.a_query[0].clone(),
            first_b1_query: pk.b_g1_query[0].clone(),
            first_b2_query: pk.b_g2_query[0].clone(),
            alpha_g1: pk.vk.alpha_g1.clone(),
            beta_g2: pk.vk.beta_g2.clone(),
            gamma_g2: pk.vk.gamma_g2.clone(),
            delta_g2: pk.vk.delta_g2.clone(),
            beta_g1: pk.beta_g1.clone(),
            delta_g1: pk.delta_g1.clone(),
        }
    }
}

#[derive(Debug)]
pub struct DistributedGroth16<
    'a,
    E: Pairing,
    M: MSMEngine<'a, E>,
    QAP: R1CSToQAP = LibsnarkReduction,
> {
    key_settings: Vec<Vec<DistributedKeySetup>>, // mapping worker_index to its keys.
    __phantom: &'a PhantomData<(E, QAP, M)>,
}

impl<'a, E: Pairing, M: MSMEngine<'a, E>, QAP: R1CSToQAP> DistributedGroth16<'a, E, M, QAP> {
    pub fn new(num_workers: usize) -> Self {
        Self {
            key_settings: vec![vec![]; num_workers],
            __phantom: &PhantomData,
        }
    }

    #[inline]
    pub fn num_workers(&self) -> usize {
        self.key_settings.len()
    }

    pub fn setup(&mut self, pk: &ProvingKey<E>) -> (MasterKey<E>, Vec<WorkerHostKeys<'a, E, M>>) {
        let (a_num_non_zeros, a_is_zero_map) = num_non_zeros(&pk.a_query[1..]);
        let (b1_num_non_zeros, b1_is_zero_map) = num_non_zeros(&pk.b_g1_query[1..]);
        let (b2_num_non_zeros, b2_is_zero_map) = num_non_zeros(&pk.b_g2_query[1..]);
        let (h_num_non_zeros, h_is_zero_map) = num_non_zeros(&pk.h_query);
        let (l_num_non_zeros, l_is_zero_map) = num_non_zeros(&pk.l_query);

        let a_query_points = G1_PERFORMANCE_RATE * a_num_non_zeros as f64;
        let b1_query_points = G1_PERFORMANCE_RATE * b1_num_non_zeros as f64;
        let b2_query_points = G2_PERFORMANCE_RATE * b2_num_non_zeros as f64;
        let h_query_points = G1_PERFORMANCE_RATE * h_num_non_zeros as f64;
        let l_query_points = G1_PERFORMANCE_RATE * l_num_non_zeros as f64;

        let mut total_points = 0f64;
        total_points += a_query_points;
        total_points += b1_query_points;
        total_points += b2_query_points;
        total_points += h_query_points;
        total_points += l_query_points;

        let a_query_num_workers = cal_num_workers(a_query_points, total_points, self.num_workers());
        let b1_query_num_workers =
            cal_num_workers(b1_query_points, total_points, self.num_workers());
        let b2_query_num_workers =
            cal_num_workers(b2_query_points, total_points, self.num_workers());
        let h_query_num_workers = cal_num_workers(h_query_points, total_points, self.num_workers());
        let l_query_num_workers = cal_num_workers(l_query_points, total_points, self.num_workers());

        let mut current_worker_index = 0;
        let mut worker_keys = vec![];
        for _ in 0..self.num_workers() {
            worker_keys.push(WorkerHostKeys::default());
        }

        self.add_setup_g1(
            &M::ark_to_host_g1_affines(pk.a_query[1..].to_vec()),
            a_num_non_zeros,
            &a_is_zero_map,
            a_query_num_workers,
            DistributedKeyType::A,
            &mut current_worker_index,
            &mut worker_keys,
        );
        self.add_setup_g1(
            &M::ark_to_host_g1_affines(pk.b_g1_query[1..].to_vec()),
            b1_num_non_zeros,
            &b1_is_zero_map,
            b1_query_num_workers,
            DistributedKeyType::B1,
            &mut current_worker_index,
            &mut worker_keys,
        );
        self.add_setup_g2(
            &M::ark_to_host_g2_affines(pk.b_g2_query[1..].to_vec()),
            &b2_is_zero_map,
            b2_num_non_zeros,
            b2_query_num_workers,
            &mut current_worker_index,
            &mut worker_keys,
        );
        self.add_setup_g1(
            &M::ark_to_host_g1_affines(pk.h_query.to_vec()),
            h_num_non_zeros,
            &h_is_zero_map,
            h_query_num_workers,
            DistributedKeyType::H,
            &mut current_worker_index,
            &mut worker_keys,
        );
        self.add_setup_g1(
            &M::ark_to_host_g1_affines(pk.l_query.to_vec()),
            l_num_non_zeros,
            &l_is_zero_map,
            l_query_num_workers,
            DistributedKeyType::L,
            &mut current_worker_index,
            &mut worker_keys,
        );

        (MasterKey::from(pk), worker_keys)
    }

    pub fn generate_assignments_zk<R: Rng, C: ConstraintSynthesizer<E::ScalarField>>(
        &self,
        rng: &mut R,
        circuit: C,
    ) -> (
        M::HostScalarField,
        M::HostScalarField,
        Vec<WorkerAssignments<'a, E, M>>,
    ) {
        let r = M::host_scalar_random(rng);
        let s = M::host_scalar_random(rng);
        self.generate_assignments(circuit, r, s)
    }

    pub fn generate_assignments_non_zk<C: ConstraintSynthesizer<E::ScalarField>>(
        &self,
        circuit: C,
    ) -> (
        M::HostScalarField,
        M::HostScalarField,
        Vec<WorkerAssignments<'a, E, M>>,
    ) {
        self.generate_assignments(circuit, M::host_scalar_zero(), M::host_scalar_zero())
    }

    fn generate_assignments<C: ConstraintSynthesizer<E::ScalarField>>(
        &self,
        circuit: C,
        r: M::HostScalarField,
        s: M::HostScalarField,
    ) -> (
        M::HostScalarField,
        M::HostScalarField,
        Vec<WorkerAssignments<'a, E, M>>,
    ) {
        let prover_time = start_timer!(|| "Groth16::Prover");
        let cs = ConstraintSystem::new_ref();

        // Set the optimization goal
        cs.set_optimization_goal(OptimizationGoal::Constraints);

        // Synthesize the circuit.
        circuit.generate_constraints(cs.clone()).unwrap();
        debug_assert!(cs.is_satisfied().unwrap());

        let lc_time = start_timer!(|| "Inlining LCs");
        cs.finalize();
        end_timer!(lc_time);

        let witness_map_time = Instant::now();
        let h_assignments =
            QAP::witness_map::<E::ScalarField, GeneralEvaluationDomain<E::ScalarField>>(cs.clone())
                .unwrap();
        let h_assignments = M::ark_to_host_scalars(h_assignments);
        println!("R1CS to QAP witness map: {:?}", witness_map_time.elapsed());

        let prover = cs.borrow().unwrap();

        let now = Instant::now();
        let input_assignments = M::ark_to_host_scalars(prover.instance_assignment[1..].to_vec());
        let witness_assignments = M::ark_to_host_scalars(prover.witness_assignment.clone());
        let total_assignments = [&input_assignments[..], &witness_assignments[..]].concat();
        println!("Convert ARK to HOST: {:?}", now.elapsed());

        let now = Instant::now();
        let mut assignments = vec![WorkerAssignments::default(); self.num_workers()];
        for worker_index in 0..self.num_workers() {
            for setting in &self.key_settings[worker_index] {
                let (full_assignments, mut start, end, zero_indexes) = match &setting {
                    DistributedKeySetup::A(start, end, zero_indexes) => {
                        (&total_assignments, *start, *end, zero_indexes)
                    }
                    DistributedKeySetup::B1(start, end, zero_indexes) => {
                        if M::host_scalar_is_zero(&r) {
                            (&total_assignments, 0usize, 0usize, zero_indexes)
                        } else {
                            (&total_assignments, *start, *end, zero_indexes)
                        }
                    }
                    DistributedKeySetup::B2(start, end, zero_indexes) => {
                        (&total_assignments, *start, *end, zero_indexes)
                    }
                    DistributedKeySetup::H(start, end, zero_indexes) => {
                        (&h_assignments, *start, *end, zero_indexes)
                    }
                    DistributedKeySetup::L(start, end, zero_indexes) => {
                        (&witness_assignments, *start, *end, zero_indexes)
                    }
                };

                let mut values = vec![];
                if start != 0 || end != 0 {
                    let mut indexes = zero_indexes.clone();
                    indexes.push(end);

                    for i in 0..indexes.len() {
                        let end = indexes[i];
                        values.extend_from_slice(&full_assignments[start..end]);

                        start = end + 1;
                    }
                }

                // let mut values = vec![];
                // let mut next_check_zero_index = 0;
                // for i in start..end {
                //     if next_check_zero_index < zero_indexes.len()
                //         && zero_indexes[next_check_zero_index] == i
                //     {
                //         next_check_zero_index += 1; // ignore if the value at this index in zero.
                //     } else {
                //         values.push(full_assignments[i].clone());
                //     }
                // }

                // if values_x.len() != values.len() {
                //     println!("{}", values_x.len());
                // }

                assignments[worker_index].push(values);
            }
        }
        println!("Distribute Assignments: {:?}", now.elapsed());

        end_timer!(prover_time);
        (r, s, assignments)
    }

    pub fn worker_prove(
        pk: &WorkerDeviceKeys<'a, E, M>,
        assignments: WorkerAssignments<'a, E, M>,
        g1_msm_config: &M::G1MSMConfig,
        g2_msm_config: &M::G2MSMConfig,
    ) -> WorkerProofs<E> {
        assert!(pk.len() == assignments.len());

        let mut proofs = WorkerProofs::default();
        for (i, assignment) in assignments.assignments.into_iter().enumerate() {
            let proof = match pk.get(i) {
                DistributedDeviceKey::A(a_query) => {
                    DistributedProof::A(M::msm_g1(g1_msm_config, assignment, a_query))
                }
                DistributedDeviceKey::B1(b1_query) => {
                    if assignment.len() == 0 {
                        DistributedProof::B1(E::G1::zero())
                    } else {
                        DistributedProof::B1(M::msm_g1(g1_msm_config, assignment, b1_query))
                    }
                }
                DistributedDeviceKey::B2(b2_query) => {
                    DistributedProof::B2(M::msm_g2(g2_msm_config, assignment, b2_query))
                }
                DistributedDeviceKey::H(h_query) => {
                    DistributedProof::H(M::msm_g1(g1_msm_config, assignment, h_query))
                }
                DistributedDeviceKey::L(l_query) => {
                    DistributedProof::L(M::msm_g1(g1_msm_config, assignment, l_query))
                }
            };

            proofs.push(proof)
        }

        proofs
    }

    pub fn master_prove(
        pk: &MasterKey<E>,
        r: M::HostScalarField,
        s: M::HostScalarField,
        worker_proofs: Vec<WorkerProofs<E>>,
    ) -> Proof<E> {
        let mut assignments_mul_a = pk.first_a_query.into_group();
        let mut assignments_mul_b1 = pk.first_b1_query.into_group();
        let mut assignments_mul_b2 = pk.first_b2_query.into_group();

        let mut h = E::G1::zero();
        let mut witness_mul_l = E::G1::zero();

        for proofs in worker_proofs {
            for proof in proofs.proofs {
                match proof {
                    DistributedProof::A(x) => assignments_mul_a += x,
                    DistributedProof::B1(x) => assignments_mul_b1 += x,
                    DistributedProof::B2(x) => assignments_mul_b2 += x,
                    DistributedProof::H(x) => h += x,
                    DistributedProof::L(x) => witness_mul_l += x,
                }
            }
        }

        let r = M::host_to_ark_scalar(r);
        let s = M::host_to_ark_scalar(s);

        // Compute A = alpha_1 + assignments*A + r*delta_1.
        let mut a = pk.delta_g1.mul(r);
        a.add_assign(assignments_mul_a);
        a.add_assign(pk.alpha_g1);

        // Compute B = beta_2 + w*B2 + s*delta_2.
        let mut b = pk.delta_g2.mul(s);
        b.add_assign(assignments_mul_b2);
        b.add_assign(pk.beta_g2);

        // Compute C = witness*L + H + s*assignments*A + r*assignments*B1 + r*s*delta_1.
        let mut c = pk
            .delta_g1
            .into_group()
            .mul_bigint(&r.into_bigint())
            .mul_bigint(&s.into_bigint());
        c.add_assign(witness_mul_l);
        c.add_assign(h);
        c.add_assign(assignments_mul_a.mul_bigint(&s.into_bigint()));
        c.add_assign(assignments_mul_b1.mul_bigint(&r.into_bigint()));

        Proof {
            a: a.into_affine(),
            b: b.into_affine(),
            c: c.into_affine(),
        }
    }

    fn add_setup_g1(
        &mut self,
        val: &[M::HostG1Affine],
        num_non_zeros: usize,
        is_zero_map: &[bool],
        arrange_num_workers: usize,
        keytype: DistributedKeyType,
        current_worker_index: &mut usize,
        keys: &mut Vec<WorkerHostKeys<'a, E, M>>,
    ) {
        let filtered_keys = split_keys(val, is_zero_map, num_non_zeros, arrange_num_workers);
        assert!(filtered_keys.len() == arrange_num_workers);

        for filtered_key in filtered_keys {
            let (start, end, zero_indexes, val) = filtered_key;
            match keytype {
                DistributedKeyType::A => {
                    keys[*current_worker_index].push(DistributedHostKey::A(val));
                    self.key_settings[*current_worker_index].push(DistributedKeySetup::A(
                        start,
                        end,
                        zero_indexes,
                    ));
                }
                DistributedKeyType::B1 => {
                    keys[*current_worker_index].push(DistributedHostKey::B1(val));
                    self.key_settings[*current_worker_index].push(DistributedKeySetup::B1(
                        start,
                        end,
                        zero_indexes,
                    ));
                }
                DistributedKeyType::H => {
                    keys[*current_worker_index].push(DistributedHostKey::H(val));
                    self.key_settings[*current_worker_index].push(DistributedKeySetup::H(
                        start,
                        end,
                        zero_indexes,
                    ));
                }
                DistributedKeyType::L => {
                    keys[*current_worker_index].push(DistributedHostKey::L(val));
                    self.key_settings[*current_worker_index].push(DistributedKeySetup::L(
                        start,
                        end,
                        zero_indexes,
                    ));
                }
                DistributedKeyType::B2 => {
                    panic!("must use add_setup_g2 for b2");
                }
            }
            *current_worker_index = (*current_worker_index + 1) % self.num_workers();
        }
    }

    /// This function is always for b_query in G2.
    fn add_setup_g2(
        &mut self,
        val: &[M::HostG2Affine],
        is_zero_map: &[bool],
        num_non_zeros: usize,
        arrange_num_workers: usize,
        current_worker_index: &mut usize,
        keys: &mut Vec<WorkerHostKeys<'a, E, M>>,
    ) {
        let filtered_keys = split_keys(val, is_zero_map, num_non_zeros, arrange_num_workers);
        assert!(filtered_keys.len() == arrange_num_workers);

        for filtered_key in filtered_keys {
            let (start, end, zero_indexes, val) = filtered_key;

            keys[*current_worker_index].push(DistributedHostKey::B2(val));
            self.key_settings[*current_worker_index].push(DistributedKeySetup::B2(
                start,
                end,
                zero_indexes,
            ));

            *current_worker_index = (*current_worker_index + 1) % self.num_workers();
        }
    }
}

fn num_non_zeros<T: AffineRepr>(values: &[T]) -> (usize, Vec<bool>) {
    let mut non_zero_len = 0;
    let mut is_zero_map = vec![false; values.len()];
    for (i, v) in values.into_iter().enumerate() {
        if v.is_zero() {
            is_zero_map[i] = true;
        } else {
            non_zero_len += 1;
        }
    }
    (non_zero_len, is_zero_map)
}

fn split_keys<T: Clone>(
    points: &[T],
    is_zero_map: &[bool],
    num_non_zeros: usize,
    num_workers: usize,
) -> Vec<(usize, usize, Vec<usize>, Vec<T>)> {
    let worker_size = num_non_zeros / num_workers;
    let mut result = vec![];

    let mut start = 0;
    for worker_index in 0..num_workers {
        let mut end = start;
        let mut worker_points = vec![];
        let mut worker_zero_indexes = vec![];
        while worker_points.len() < worker_size
            || (worker_index == num_workers - 1 && end < points.len())
        {
            if is_zero_map[end] {
                worker_zero_indexes.push(end);
            } else {
                worker_points.push(points[end].clone());
            }

            end += 1;
        }

        result.push((start, end, worker_zero_indexes, worker_points));
        start = end;
    }

    result
}

fn cal_num_workers(points: f64, total_points: f64, num_workers: usize) -> usize {
    let ratio = points / total_points;
    let mut val = (ratio * num_workers as f64).round() as usize;
    if val == 0 {
        val = 1;
    }
    val
}
