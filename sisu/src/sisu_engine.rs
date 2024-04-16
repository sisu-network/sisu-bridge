use std::{
    collections::HashMap,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use ark_ff::{FftField, Field};
use ark_std::{cfg_iter, cfg_iter_mut};
use icicle_core::sisu::{
    dense_mle_multi, fold_multi, initialize_combining_point, mul_arr_multi, mul_by_scalar_multi,
    sub_arr_multi,
};
use icicle_cuda_runtime::memory::{
    HostOrDeviceSlice, HostOrDeviceSlice2D, HostOrDeviceSliceWrapper,
};

use sisulib::{
    circuit::{
        general_circuit::{circuit::GeneralCircuit, layer::GeneralCircuitLayer},
        CircuitParams,
    },
    common::{dec2bin, ilog2_ceil, split_number},
    domain::{Domain, RootDomain},
    field::FrBN254,
    mle::{
        dense::{identity_mle, SisuDenseMultilinearExtension},
        sparse::SisuSparseMultilinearExtension,
    },
};

use crate::{
    circuit_evaluation::CudaCircuitEvaluations,
    cuda_compat::{
        circuit::{CudaCircuit, CudaCircuitLayer, CudaSparseMultilinearExtension},
        fft::{ArkFftEngine, FFTEnginePool, FftEngine, IcicleFftEngine},
        slice::CudaSlice,
    },
    fri::fold_v1,
    general_gkr_sumcheck::replica_point,
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    polynomial::{
        CPUProductBookeepingTable, GPUProductBookeepingTable, RootProductBookeepingTable,
    },
    sisu_merkle_tree::{CPUMerkleTreeEngine, GPUMerkleTreeEngine, SisuMerkleTreeEngine},
    vpd::{compute_rational_constraint, divide_by_vanishing_poly},
};

pub enum GateExtensionType {
    Constant,
    Mul,
    ForwardX,
    ForwardY,
}

impl GateExtensionType {
    pub fn to_extensions<'a, F: Field>(
        &self,
        layer: &'a GeneralCircuitLayer<F>,
    ) -> &'a [SisuSparseMultilinearExtension<F>] {
        match &self {
            Self::Constant => layer.constant_ext.as_slice(),
            Self::Mul => layer.mul_ext.as_slice(),
            Self::ForwardX => layer.forward_x_ext.as_slice(),
            Self::ForwardY => layer.forward_y_ext.as_slice(),
        }
    }

    pub fn to_cuda_extensions<'a>(
        &self,
        layer: &'a CudaCircuitLayer<FrBN254>,
    ) -> &'a HostOrDeviceSliceWrapper<CudaSparseMultilinearExtension<FrBN254>> {
        match &self {
            Self::Constant => &layer.constant_ext,
            Self::Mul => &layer.mul_ext,
            Self::ForwardX => &layer.forward_x_ext,
            Self::ForwardY => &layer.forward_y_ext,
        }
    }
}

pub struct OnesSlicePool<F: IcicleConvertibleField> {
    pool: Arc<RwLock<HashMap<usize, CudaSlice<F>>>>,
}

impl<F: IcicleConvertibleField> Clone for OnesSlicePool<F> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<F: IcicleConvertibleField> OnesSlicePool<F> {
    pub fn new() -> Self {
        Self {
            pool: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    pub fn get(&self, size: usize) -> CudaSlice<F> {
        {
            let read_pool = self.pool.read().unwrap();
            if !read_pool.contains_key(&size) {
                drop(read_pool);

                let host_slice = vec![F::ONE; size];

                let icicle_slice = vec![F::ONE.to_icicle(); size];
                let device_slice = if let Ok(device) = HostOrDeviceSlice::on_device(&icicle_slice) {
                    device
                } else {
                    HostOrDeviceSlice::Host(vec![])
                };

                let mut write_pool = self.pool.write().unwrap();
                (*write_pool).insert(size, CudaSlice::new(host_slice, device_slice));
            }
        }

        let read_pool = self.pool.read().unwrap();

        CudaSlice::ref_from(read_pool.get(&size).unwrap())
    }
}

#[derive(Clone)]
pub struct DomainPool<F: IcicleConvertibleField> {
    pool: Arc<RwLock<HashMap<usize, HostOrDeviceSlice<F::IcicleField>>>>,
}

impl<F: IcicleConvertibleField> DomainPool<F> {
    pub fn new() -> Self {
        Self {
            pool: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    pub fn get(&self, size: usize) -> HostOrDeviceSlice<F::IcicleField> {
        {
            let read_pool = self.pool.read().unwrap();
            if !read_pool.contains_key(&size) {
                drop(read_pool);

                let domain = RootDomain::<F>::new(size);
                let icicle_domain: Vec<_> = cfg_iter!(domain).map(|x| x.to_icicle()).collect();
                let device_slice = HostOrDeviceSlice::on_device(&icicle_domain).unwrap();

                let mut write_pool = self.pool.write().unwrap();
                (*write_pool).insert(size, device_slice);
            }
        }

        let read_pool = self.pool.read().unwrap();

        HostOrDeviceSlice::device_ref(read_pool.get(&size).unwrap())
    }
}

pub struct GeneralCircuitHashMap<T> {
    map: Arc<RwLock<HashMap<u64, T>>>,
}

impl<T> Clone for GeneralCircuitHashMap<T> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<T> GeneralCircuitHashMap<T> {
    pub fn new() -> Self {
        Self {
            map: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    pub fn read_or_insert<'a, F: Field, InsertFunc: Fn() -> T>(
        &'a self,
        circuit: &GeneralCircuit<F>,
        insert_func: InsertFunc,
    ) -> (RwLockReadGuard<HashMap<u64, T>>, u64) {
        let circuit_ptr = circuit as *const GeneralCircuit<F>;
        let circuit_id = circuit_ptr as u64;

        let read_map = self.map.read().unwrap();
        if !read_map.contains_key(&circuit_id) {
            drop(read_map);

            let t = insert_func();

            let mut write_map = self.map.write().unwrap();
            (*write_map).insert(circuit_id, t);
        } else {
            drop(read_map);
        }

        (self.map.read().unwrap(), circuit_id)
    }
}

pub trait SisuEngine<F: IcicleConvertibleField>: Clone + Sync + Send {
    type Fft: FftEngine<F>;
    type MerkleTree: SisuMerkleTreeEngine<F>;
    type RootProductBookeepingTable: RootProductBookeepingTable<F>;

    fn fft_engine_pool(&self) -> &FFTEnginePool<F, Self::Fft>;
    fn merkle_tree_engine(&self) -> &Self::MerkleTree;
    fn cuda_slice_ones(&self, size: usize) -> CudaSlice<F>;

    fn evaluate_circuit(
        &self,
        circuit: &GeneralCircuit<F>,
        witness: &mut [CudaSlice<F>],
    ) -> CudaCircuitEvaluations<F>;

    fn precompute_bookeeping(init: F, g: &mut CudaSlice<F>) -> CudaSlice<F>;

    fn initialize_phase_1_plus(
        &self,
        circuit: &GeneralCircuit<F>,
        num_replicas: usize,
        layer_index: usize,
        gate_type: GateExtensionType,
        s_evaluations: &mut [CudaSlice<F>],
        bookeeping_g: &mut CudaSlice<F>,
    ) -> CudaSlice<F>;

    fn initialize_phase_2_plus(
        &self,
        circuit: &GeneralCircuit<F>,
        num_replicas: usize,
        layer_index: usize,
        gate_type: GateExtensionType,
        bookeeping_g: &mut CudaSlice<F>,
        bookeeping_u: &mut CudaSlice<F>,
    ) -> Vec<CudaSlice<F>>;

    fn initialize_combining_point(
        &self,
        circuit: &GeneralCircuit<F>,
        worker_index: usize,
        num_replicas: usize,
        num_workers: usize,
        layer_index: usize,
        rs: &[CudaSlice<F>],
        alpha: &[F],
    ) -> CudaSlice<F>;

    fn fold_multi(
        &self,
        domain: &Domain<F>,
        random_point: F,
        evaluations: &mut [CudaSlice<F>],
    ) -> Vec<CudaSlice<F>>;

    fn dense_mle_multi(evaluations: Vec<CudaSlice<F>>, input: &[F]) -> Vec<F>;

    fn interpolate_q(&self, input: &[F]) -> CudaSlice<F> {
        let mut q_evaluations =
            Self::precompute_bookeeping(F::ONE, &mut CudaSlice::on_host(input.to_vec()));

        self.fft_engine_pool().interpolate(&mut q_evaluations)
    }

    fn divide_by_vanishing_poly_multi(
        f_polynomials: &mut [CudaSlice<F>],
        vanishing_degree: usize,
    ) -> Vec<CudaSlice<F>> {
        let mut result = vec![];

        for i in 0..f_polynomials.len() {
            result.push(divide_by_vanishing_poly(
                &mut f_polynomials[i],
                vanishing_degree,
            ));
        }

        result
    }

    fn compute_l_mul_q_multi(
        &self,
        l2_evaluations: Vec<CudaSlice<F>>,
        q_polynomial: CudaSlice<F>,
    ) -> Vec<CudaSlice<F>>;

    fn compute_rational_constraint_multi(
        domain_size: usize,
        f_polynomials: Vec<CudaSlice<F>>,
        h_polynomials: Vec<CudaSlice<F>>,
        output: Vec<F>,
    ) -> Vec<CudaSlice<F>>;
}

#[derive(Clone)]
pub struct CPUSisuEngine<F: IcicleConvertibleField + FftField, H: SisuHasher<F>> {
    fft_pool: FFTEnginePool<F, ArkFftEngine<F>>,
    merkle_tree: CPUMerkleTreeEngine<F, H>,
    ones_slice_pool: OnesSlicePool<F>,
}

impl<F: IcicleConvertibleField + FftField, H: SisuHasher<F>> CPUSisuEngine<F, H> {
    pub fn new() -> Self {
        Self {
            fft_pool: FFTEnginePool::new(),
            merkle_tree: CPUMerkleTreeEngine::new(),
            ones_slice_pool: OnesSlicePool::new(),
        }
    }
}

impl<F: IcicleConvertibleField + FftField, H: SisuHasher<F>> SisuEngine<F> for CPUSisuEngine<F, H> {
    type Fft = ArkFftEngine<F>;
    type MerkleTree = CPUMerkleTreeEngine<F, H>;
    type RootProductBookeepingTable = CPUProductBookeepingTable<F>;

    fn fft_engine_pool(&self) -> &FFTEnginePool<F, Self::Fft> {
        &self.fft_pool
    }

    fn merkle_tree_engine(&self) -> &Self::MerkleTree {
        &self.merkle_tree
    }

    fn cuda_slice_ones(&self, size: usize) -> CudaSlice<F> {
        self.ones_slice_pool.get(size)
    }

    fn evaluate_circuit(
        &self,
        circuit: &GeneralCircuit<F>,
        witness: &mut [CudaSlice<F>],
    ) -> CudaCircuitEvaluations<F> {
        CudaCircuitEvaluations::from_host(circuit, witness)
    }

    fn precompute_bookeeping(init: F, g: &mut CudaSlice<F>) -> CudaSlice<F> {
        let mut bookeeping_g = vec![init];
        for gi in g.as_ref_host() {
            let mut new_bookeeping_g = vec![F::ZERO; 2 * bookeeping_g.len()];
            for (idx, prev_value) in bookeeping_g.iter().enumerate() {
                let b0 = (idx << 1) + 0;
                let b1 = (idx << 1) + 1;
                let tmp = *prev_value * gi;

                new_bookeeping_g[b0] = *prev_value - tmp;
                new_bookeeping_g[b1] = tmp;
            }

            bookeeping_g = new_bookeeping_g;
        }

        CudaSlice::on_host(bookeeping_g)
    }

    fn initialize_phase_1_plus(
        &self,
        circuit: &GeneralCircuit<F>,
        num_replicas: usize,
        layer_index: usize,
        gate_type: GateExtensionType,
        s_evaluations: &mut [CudaSlice<F>],
        bookeeping_g: &mut CudaSlice<F>,
    ) -> CudaSlice<F> {
        let f_extensions = gate_type.to_extensions(circuit.layer(layer_index));

        let bookeeping_g = bookeeping_g.as_ref_host();
        let s_evaluations: Vec<_> = cfg_iter_mut!(s_evaluations)
            .map(|x| x.as_ref_host())
            .collect();

        assert!(f_extensions.len() == s_evaluations.len());

        let replica_num_vars = num_replicas.ilog2() as usize;
        let replica_z_num_vars = bookeeping_g.len().ilog2() as usize;
        let z_num_vars = replica_z_num_vars - replica_num_vars;

        let replica_x_num_vars = 3 * replica_num_vars + f_extensions[0].num_vars()
            - replica_z_num_vars
            - ilog2_ceil(s_evaluations[0].len());
        let x_num_vars = replica_x_num_vars - replica_num_vars;

        for i in 0..s_evaluations.len() {
            assert_eq!(
                replica_x_num_vars,
                3 * replica_num_vars + f_extensions[i].num_vars()
                    - replica_z_num_vars
                    - ilog2_ceil(s_evaluations[i].len())
            );
        }

        let mut bookeeping_h = vec![F::ZERO; 2usize.pow(replica_x_num_vars as u32)];
        for (layer_index, f_ext) in f_extensions.iter().enumerate() {
            let replica_y_num_vars = ilog2_ceil(s_evaluations[layer_index].len());

            let y_num_vars = replica_y_num_vars - replica_num_vars;

            for (point, evaluation_fi_at_this_point) in
                f_ext.evaluations.compile(&CircuitParams::default())
            {
                // The point is in form (z, x, y).
                let (zx, y) = split_number(&point, y_num_vars);
                let (z, x) = split_number(&zx, x_num_vars);

                for replica_index in 0..num_replicas {
                    bookeeping_h[replica_point(x, x_num_vars, replica_index)] += bookeeping_g
                        [replica_point(z, z_num_vars, replica_index)]
                        * s_evaluations[layer_index][replica_point(y, y_num_vars, replica_index)]
                        * evaluation_fi_at_this_point;
                }
            }
        }

        CudaSlice::on_host(bookeeping_h)
    }

    fn initialize_phase_2_plus(
        &self,
        circuit: &GeneralCircuit<F>,
        num_replicas: usize,
        layer_index: usize,
        gate_type: GateExtensionType,
        bookeeping_g: &mut CudaSlice<F>,
        bookeeping_u: &mut CudaSlice<F>,
    ) -> Vec<CudaSlice<F>> {
        let f_extensions = gate_type.to_extensions(circuit.layer(layer_index));

        let bookeeping_g = bookeeping_g.as_ref_host();
        let bookeeping_u = bookeeping_u.as_ref_host();

        let replica_num_vars = num_replicas.ilog2() as usize;
        let replica_z_num_vars = bookeeping_g.len().ilog2() as usize;
        let replica_x_num_vars = bookeeping_u.len().ilog2() as usize;
        let z_num_vars = replica_z_num_vars - replica_num_vars;
        let x_num_vars = replica_x_num_vars - replica_num_vars;

        let mut bookeeping_all_f = vec![];

        for f_ext in f_extensions {
            let replica_y_num_vars =
                3 * replica_num_vars + f_ext.num_vars() - replica_z_num_vars - replica_x_num_vars;

            let y_num_vars = replica_y_num_vars - replica_num_vars;

            let mut bookeeping_fi = vec![F::ZERO; 2usize.pow(replica_y_num_vars as u32)];

            for (point, evaluation_fi_at_this_point) in
                f_ext.evaluations.compile(&CircuitParams::default())
            {
                // The point is in form (z, x, y).
                let (zx, y) = split_number(&point, y_num_vars);
                let (z, x) = split_number(&zx, x_num_vars);

                for replica_index in 0..num_replicas {
                    bookeeping_fi[replica_point(y, y_num_vars, replica_index)] += bookeeping_g
                        [replica_point(z, z_num_vars, replica_index)]
                        * bookeeping_u[replica_point(x, x_num_vars, replica_index)]
                        * evaluation_fi_at_this_point;
                }
            }

            bookeeping_all_f.push(CudaSlice::on_host(bookeeping_fi));
        }

        bookeeping_all_f
    }

    fn initialize_combining_point(
        &self,
        circuit: &GeneralCircuit<F>,
        worker_index: usize,
        num_replicas: usize,
        num_workers: usize,
        layer_index: usize,
        rs: &[CudaSlice<F>],
        alpha: &[F],
    ) -> CudaSlice<F> {
        let subset_reverse_ext = if layer_index == circuit.len() {
            &circuit.layer(layer_index).subset_reverse_ext
        } else {
            &circuit.input_subset_reverse_ext
        };

        let worker_extra_num_vars = num_workers.ilog2() as usize;
        let replica_extra_num_vars = num_replicas.ilog2() as usize;

        assert!(rs.len() == subset_reverse_ext.as_slice().len() + 1);
        assert!(worker_index < num_workers);

        let mut bookeeping_table_rs = vec![];
        let identity_point = dec2bin(worker_index as u64, worker_extra_num_vars);
        for (i, r) in rs.iter().enumerate() {
            let identity_value = identity_mle(
                &identity_point,
                r.at_range_to(..worker_extra_num_vars).as_ref_host(),
            );

            bookeeping_table_rs.push(
                Self::precompute_bookeeping(
                    alpha[i] * identity_value,
                    &mut r.at_range_from(worker_extra_num_vars..),
                )
                .as_host(),
            );
        }

        // At this time, the last random points in `rs` is the random point u in
        // the evaluation V(u).
        let replica_x_num_vars = ilog2_ceil(num_replicas * circuit.len_at(layer_index));
        let x_num_vars = replica_x_num_vars - replica_extra_num_vars;
        assert!(rs[rs.len() - 1].len() == replica_x_num_vars + worker_extra_num_vars);

        // Because V(u) is evaluated on all gates of this layer (rather than only
        // a subset), such that:
        // + C(z, x) == 1 iff z==x and z < layer size.
        // + C(z, x) == 0 iff z != x or z >= layer size.
        //
        // So the bookeeping table g can be quickly calculated by copying the
        // bookeeping table of last random point, except for points which are larger
        // than layer size (set them by zero instead).
        //
        // let mut bookeeping_table_g = bookeeping_table_rs[rs.len() - 1][..layer_size].to_vec();
        // bookeeping_table_g.extend(vec![F::ZERO; 2usize.pow(x_num_vars as u32) - layer_size]);
        let mut bookeeping_table_g = bookeeping_table_rs[rs.len() - 1].to_vec();

        let num_replicas = 2usize.pow(replica_extra_num_vars as u32);
        for (k, subset_ext) in subset_reverse_ext.as_slice().iter().enumerate() {
            let subset_ext_num_vars = subset_ext.num_vars();
            if subset_ext_num_vars == 0 {
                continue;
            }

            let t_num_vars = subset_ext_num_vars - x_num_vars;

            for (point, _) in subset_ext.evaluations.compile(&CircuitParams::default()) {
                // point = (t, x)
                // k = source_layer_index.
                // t = target_subset_gate_index.
                // x = target_gate_index.
                //
                // C(t, x) == 1 iff t (subset index) and x (real index) indicates
                // the same gate in this layer.
                let (t, x) = split_number(&point, x_num_vars);

                for replica_index in 0..num_replicas {
                    bookeeping_table_g[replica_point(x, x_num_vars, replica_index)] +=
                        bookeeping_table_rs[k][replica_point(t, t_num_vars, replica_index)];
                }
            }
        }

        CudaSlice::on_host(bookeeping_table_g)
    }

    fn fold_multi(
        &self,
        domain: &Domain<F>,
        random_point: F,
        evaluations: &mut [CudaSlice<F>],
    ) -> Vec<CudaSlice<F>> {
        let mut result = vec![];

        let inverse_two = F::from(2u8).inverse().unwrap();

        for k in 0..evaluations.len() {
            let mut new_evaluations = vec![];

            let evaluations = evaluations[k].as_ref_host();
            for i in 0..evaluations.len() / 2 {
                new_evaluations.push(fold_v1(
                    domain,
                    random_point,
                    &inverse_two,
                    i,
                    &evaluations[i],
                    &evaluations[i + evaluations.len() / 2],
                ))
            }

            result.push(CudaSlice::on_host(new_evaluations));
        }

        result
    }

    fn dense_mle_multi(mut evaluations: Vec<CudaSlice<F>>, input: &[F]) -> Vec<F> {
        let mut result = vec![];
        for i in 0..evaluations.len() {
            let mle = SisuDenseMultilinearExtension::from_slice(evaluations[i].as_ref_host());
            result.push(mle.evaluate(vec![input]));
        }

        result
    }

    fn compute_l_mul_q_multi(
        &self,
        l2_evaluations: Vec<CudaSlice<F>>,
        mut q_polynomial: CudaSlice<F>,
    ) -> Vec<CudaSlice<F>> {
        let mut result = vec![];
        let mut q2_evaluations = self
            .fft_engine_pool()
            .evaluate(q_polynomial.len() * 2, &mut q_polynomial);

        for l2 in l2_evaluations {
            let q2_evaluations = q2_evaluations.as_ref_host();
            let mut l2_evaluations = l2.as_host();

            assert_eq!(l2_evaluations.len(), q2_evaluations.len());
            for i in 0..q2_evaluations.len() {
                // f = q*l
                l2_evaluations[i] *= q2_evaluations[i];
            }

            result.push(
                self.fft_engine_pool()
                    .interpolate(&mut CudaSlice::on_host(l2_evaluations)),
            );
        }

        result
    }

    fn compute_rational_constraint_multi(
        domain_size: usize,
        mut f_polynomials: Vec<CudaSlice<F>>,
        mut h_polynomials: Vec<CudaSlice<F>>,
        output: Vec<F>,
    ) -> Vec<CudaSlice<F>> {
        let mut result = vec![];
        for i in 0..f_polynomials.len() {
            result.push(compute_rational_constraint(
                domain_size,
                &mut f_polynomials[i],
                &mut h_polynomials[i],
                output[i],
            ));
        }

        result
    }
}

#[derive(Clone)]
pub struct GPUFrBN254SisuEngine {
    domain_pool: DomainPool<FrBN254>,
    cuda_circuit_map: GeneralCircuitHashMap<CudaCircuit<FrBN254>>,
    fft_pool: FFTEnginePool<FrBN254, IcicleFftEngine<FrBN254>>,
    merkle_tree: GPUMerkleTreeEngine<FrBN254>,
    ones_slice_pool: OnesSlicePool<FrBN254>,
}

impl GPUFrBN254SisuEngine {
    pub fn new() -> Self {
        Self {
            domain_pool: DomainPool::new(),
            cuda_circuit_map: GeneralCircuitHashMap::new(),
            fft_pool: FFTEnginePool::new(),
            merkle_tree: GPUMerkleTreeEngine::new(),
            ones_slice_pool: OnesSlicePool::new(),
        }
    }
}

impl SisuEngine<FrBN254> for GPUFrBN254SisuEngine {
    type Fft = IcicleFftEngine<FrBN254>;
    type MerkleTree = GPUMerkleTreeEngine<FrBN254>;
    type RootProductBookeepingTable = GPUProductBookeepingTable<FrBN254>;

    fn fft_engine_pool(&self) -> &FFTEnginePool<FrBN254, Self::Fft> {
        &self.fft_pool
    }

    fn merkle_tree_engine(&self) -> &Self::MerkleTree {
        &self.merkle_tree
    }

    fn evaluate_circuit(
        &self,
        circuit: &GeneralCircuit<FrBN254>,
        witness: &mut [CudaSlice<FrBN254>],
    ) -> CudaCircuitEvaluations<FrBN254> {
        let (cuda_circuit_reader, circuit_id) =
            self.cuda_circuit_map.read_or_insert(circuit, || {
                CudaCircuit::from(circuit, &CircuitParams::default())
            });

        let cuda_circuit = cuda_circuit_reader.get(&circuit_id).unwrap();

        CudaCircuitEvaluations::from_device(&cuda_circuit, witness)
    }

    fn cuda_slice_ones(&self, size: usize) -> CudaSlice<FrBN254> {
        self.ones_slice_pool.get(size)
    }

    fn precompute_bookeeping(init: FrBN254, g: &mut CudaSlice<FrBN254>) -> CudaSlice<FrBN254> {
        let mut output = HostOrDeviceSlice::cuda_malloc(1 << g.len()).unwrap();
        icicle_core::sisu::precompute_bookeeping(init.to_icicle(), g.as_ref_device(), &mut output)
            .unwrap();
        CudaSlice::on_device(output)
    }

    fn initialize_phase_1_plus(
        &self,
        circuit: &GeneralCircuit<FrBN254>,
        num_replicas: usize,
        layer_index: usize,
        gate_type: GateExtensionType,
        s_evaluations: &mut [CudaSlice<FrBN254>],
        bookeeping_g: &mut CudaSlice<FrBN254>,
    ) -> CudaSlice<FrBN254> {
        let (cuda_circuit_reader, circuit_id) =
            self.cuda_circuit_map.read_or_insert(circuit, || {
                CudaCircuit::from(circuit, &CircuitParams::default())
            });

        let cuda_circuit = cuda_circuit_reader.get(&circuit_id).unwrap();

        let output_size =
            num_replicas * (1 << cuda_circuit.layers[layer_index].constant_ext[0].x_num_vars);
        let mut output = HostOrDeviceSlice::zeros_on_device(output_size).unwrap();

        let f_extensions = gate_type
            .to_cuda_extensions(&cuda_circuit.layers[layer_index])
            .as_device();

        let s_ptr: Vec<_> = cfg_iter_mut!(s_evaluations)
            .map(|x| HostOrDeviceSlice::device_ref(x.as_ref_device()))
            .collect();
        let device_s_ptr = HostOrDeviceSlice2D::ref_from(s_ptr).unwrap();

        icicle_core::sisu::initialize_phase_1_plus(
            s_evaluations.len() as u32,
            output_size as u32,
            f_extensions,
            &device_s_ptr,
            bookeeping_g.as_ref_device(),
            &mut output,
        )
        .unwrap();

        CudaSlice::on_device(output)
    }

    fn initialize_phase_2_plus(
        &self,
        circuit: &GeneralCircuit<FrBN254>,
        num_replicas: usize,
        layer_index: usize,
        gate_type: GateExtensionType,
        bookeeping_g: &mut CudaSlice<FrBN254>,
        bookeeping_u: &mut CudaSlice<FrBN254>,
    ) -> Vec<CudaSlice<FrBN254>> {
        let (cuda_circuit_reader, circuit_id) =
            self.cuda_circuit_map.read_or_insert(circuit, || {
                CudaCircuit::from(circuit, &CircuitParams::default())
            });

        let cuda_circuit = cuda_circuit_reader.get(&circuit_id).unwrap();

        let f_extensions = gate_type
            .to_cuda_extensions(&cuda_circuit.layers[layer_index])
            .as_device();

        let mut on_host_output_size = vec![];
        for i in 0..f_extensions.len() {
            on_host_output_size.push(
                num_replicas * (1 << cuda_circuit.layers[layer_index].constant_ext[i].y_num_vars),
            );
        }

        let mut output = HostOrDeviceSlice2D::zeros(on_host_output_size.clone()).unwrap();

        icicle_core::sisu::initialize_phase_2_plus(
            f_extensions.len() as u32,
            on_host_output_size.into_iter().map(|x| x as u32).collect(),
            f_extensions,
            bookeeping_g.as_ref_device(),
            bookeeping_u.as_ref_device(),
            &mut output,
        )
        .unwrap();

        output
            .into_iter()
            .map(|x| CudaSlice::on_device(x))
            .collect()
    }

    fn initialize_combining_point(
        &self,
        circuit: &GeneralCircuit<FrBN254>,
        worker_index: usize,
        _: usize,
        num_workers: usize,
        layer_index: usize,
        rs: &[CudaSlice<FrBN254>],
        alpha: &[FrBN254],
    ) -> CudaSlice<FrBN254> {
        assert!(worker_index < num_workers);

        let (cuda_circuit_reader, circuit_id) =
            self.cuda_circuit_map.read_or_insert(circuit, || {
                CudaCircuit::from(circuit, &CircuitParams::default())
            });
        let cuda_circuit = cuda_circuit_reader.get(&circuit_id).unwrap();

        let worker_extra_num_vars = num_workers.ilog2() as usize;

        let mut bookeeping_table_rs = vec![];
        let identity_point = dec2bin(worker_index as u64, worker_extra_num_vars);
        for (i, r) in rs.iter().enumerate() {
            let identity_value = identity_mle(
                &identity_point,
                r.at_range_to(..worker_extra_num_vars).as_ref_host(),
            );

            bookeeping_table_rs.push(Self::precompute_bookeeping(
                alpha[i] * identity_value,
                &mut r.at_range_from(worker_extra_num_vars..),
            ));
        }

        // Because V(u) is evaluated on all gates of this layer (rather than only
        // a subset), such that:
        // + C(z, x) == 1 iff z==x and z < layer size.
        // + C(z, x) == 0 if z != x or z >= layer size.
        //
        // So the bookeeping table g can be quickly calculated by copying the
        // bookeeping table of last random point, except for points which are larger
        // than layer size (set them by zero instead).
        //
        // let mut bookeeping_table_g = bookeeping_table_rs[rs.len() - 1][..layer_size].to_vec();
        // bookeeping_table_g.extend(vec![F::ZERO; 2usize.pow(x_num_vars as u32) - layer_size]);
        let mut bookeeping_table_g = HostOrDeviceSlice::on_host(vec![]);
        let mut bookeeping_rs = vec![];
        let mut on_host_bookeeping_rs_size = vec![];
        for (i, bk) in bookeeping_table_rs.into_iter().enumerate() {
            if i == layer_index {
                bookeeping_table_g = bk.as_device();
            } else {
                on_host_bookeeping_rs_size.push(bk.len() as u32);
                bookeeping_rs.push(bk.as_device());
            }
        }

        initialize_combining_point(
            cuda_circuit.reverse_exts[layer_index].len() as u32,
            on_host_bookeeping_rs_size,
            &HostOrDeviceSlice2D::ref_from(bookeeping_rs).unwrap(),
            &cuda_circuit.reverse_exts[layer_index].as_device(),
            &mut bookeeping_table_g,
        )
        .unwrap();

        CudaSlice::on_device(bookeeping_table_g)
    }

    fn fold_multi(
        &self,
        domain: &Domain<FrBN254>,
        random_point: FrBN254,
        evaluations: &mut [CudaSlice<FrBN254>],
    ) -> Vec<CudaSlice<FrBN254>> {
        let domain = self.domain_pool.get(domain.root_size());

        let mut evaluations_2d = vec![];
        for i in 0..evaluations.len() {
            evaluations_2d.push(HostOrDeviceSlice::device_ref(
                evaluations[i].as_ref_device(),
            ));
        }

        let evaluations_2d = HostOrDeviceSlice2D::ref_from(evaluations_2d).unwrap();

        let mut size = vec![];
        for i in 0..evaluations.len() {
            size.push(evaluations[i].len() / 2);
        }
        let mut output = HostOrDeviceSlice2D::malloc(size).unwrap();

        fold_multi(
            &domain,
            random_point.to_icicle(),
            &evaluations_2d,
            &mut output,
        )
        .unwrap();

        let mut result = vec![];
        for o in output.into_iter() {
            result.push(CudaSlice::on_device(o));
        }

        result
    }

    fn dense_mle_multi(evaluations: Vec<CudaSlice<FrBN254>>, input: &[FrBN254]) -> Vec<FrBN254> {
        let num_mle = evaluations.len();
        let mut evaluations_2d = vec![];
        for eval in evaluations {
            evaluations_2d.push(eval.as_device());
        }

        let evaluations_2d = HostOrDeviceSlice2D::ref_from(evaluations_2d).unwrap();

        let mut output = HostOrDeviceSlice::cuda_malloc(num_mle).unwrap();
        let icicle_input = cfg_iter!(input).map(|x| x.to_icicle()).collect();

        dense_mle_multi(&mut output, evaluations_2d, icicle_input).unwrap();

        CudaSlice::on_device(output).as_host()
    }

    fn compute_l_mul_q_multi(
        &self,
        l2_evaluations: Vec<CudaSlice<FrBN254>>,
        mut q_polynomial: CudaSlice<FrBN254>,
    ) -> Vec<CudaSlice<FrBN254>> {
        let mut q2 = self
            .fft_engine_pool()
            .evaluate(q_polynomial.len() * 2, &mut q_polynomial);

        let mut l2_2d = cuda_slices_to_device_2d(l2_evaluations);
        let mut q2_2d = vec![];
        for _ in 0..l2_2d.len() {
            q2_2d.push(HostOrDeviceSlice::device_ref(q2.as_ref_device()));
        }
        let q2_2d = HostOrDeviceSlice2D::ref_from(q2_2d).unwrap();

        mul_arr_multi(&mut l2_2d, &q2_2d).unwrap();

        self.fft_engine_pool()
            .interpolate_multi(&mut device_2d_to_vec_slices(l2_2d))
    }

    fn compute_rational_constraint_multi(
        domain_size: usize,
        f_polynomials: Vec<CudaSlice<FrBN254>>,
        h_polynomials: Vec<CudaSlice<FrBN254>>,
        output: Vec<FrBN254>,
    ) -> Vec<CudaSlice<FrBN254>> {
        let mut f_poly = cuda_slices_to_device_2d(f_polynomials);
        let mut low_h_poly = cuda_slices_to_device_2d(h_polynomials);
        let mut high_h_poly = low_h_poly.clone();

        let domain_size = FrBN254::from(domain_size as u64);

        // f1 = |H| * f(x)
        mul_by_scalar_multi(&mut f_poly, domain_size.to_icicle()).unwrap();
        let mut f1_poly = f_poly; // rename

        mul_by_scalar_multi(&mut low_h_poly, (FrBN254::ZERO - domain_size).to_icicle()).unwrap();
        mul_by_scalar_multi(&mut high_h_poly, domain_size.to_icicle()).unwrap();

        // f2 = |H| * h(x) * Z(x) = |H| * h(x) * (x^n-1)
        //    = low_coeffs(-|H| * h(x))
        //    + high_coeffs(|H| * h(x)).
        let mut f2_poly = vec![];
        for (low_h, high_h) in low_h_poly.into_iter().zip(high_h_poly.into_iter()) {
            let mut f2 = HostOrDeviceSlice::cuda_malloc(low_h.len() * 2).unwrap();
            f2.device_copy_partially(&low_h, 0, 0).unwrap();
            f2.device_copy_partially(&high_h, 0, low_h.len()).unwrap();

            f2_poly.push(f2);
        }

        // f3 = f1 - f2 - output.
        sub_arr_multi(
            &mut f1_poly,
            &HostOrDeviceSlice2D::ref_from(f2_poly).unwrap(),
        )
        .unwrap();

        let mut f3 = device_2d_to_vec_slices::<FrBN254>(f1_poly); // rename
        for (i, sub_f3) in f3.iter_mut().enumerate() {
            assert_eq!(sub_f3.at(0), output[i]);

            // f3 = f3_tmp - output;
            // set sub_f3[0] = 0;
            sub_f3
                .as_mut_device()
                .copy_from_host_partially(&[FrBN254::ZERO.to_icicle()], 0)
                .unwrap();
        }

        // p(x)*x = f3(x)/|H|
        let inverse_domain_size = domain_size.inverse().unwrap();
        let mut f3 = cuda_slices_to_device_2d(f3);
        mul_by_scalar_multi(&mut f3, inverse_domain_size.to_icicle()).unwrap();

        let mut result = vec![];
        for subf3 in f3.into_iter() {
            let mut px = HostOrDeviceSlice::cuda_malloc(subf3.len() - 1).unwrap();
            px.device_copy_partially(&subf3, 1, 0).unwrap();
            result.push(CudaSlice::on_device(px));
        }

        result
    }
}

fn cuda_slices_to_device_2d<F: IcicleConvertibleField>(
    slices: Vec<CudaSlice<F>>,
) -> HostOrDeviceSlice2D<F::IcicleField> {
    let slices_2d = slices.into_iter().map(|x| x.as_device()).collect();

    HostOrDeviceSlice2D::ref_from(slices_2d).unwrap()
}

fn device_2d_to_vec_slices<F: IcicleConvertibleField>(
    d: HostOrDeviceSlice2D<F::IcicleField>,
) -> Vec<CudaSlice<F>> {
    d.into_iter().map(|x| CudaSlice::on_device(x)).collect()
}
