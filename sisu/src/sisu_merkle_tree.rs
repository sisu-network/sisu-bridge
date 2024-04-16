use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_chunks, cfg_iter};
use icicle_bn254::curve::ScalarField as IcicleFrBN254;
use icicle_core::{
    sisu::{build_merkle_tree, exchange_evaluations, hash_merkle_tree_slice, MerkleTreeConfig},
    traits::FieldImpl,
};

use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostOrDeviceSlice2D};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sisulib::{
    common::{div_round, Error},
    field::FrBN254,
};

use crate::{
    channel::{MasterNode, SisuReceiver, SisuSender, WorkerNode},
    cuda_compat::slice::CudaSlice,
    hash::SisuHasher,
    icicle_converter::IcicleConvertibleField,
    mempool::{MasterSharedMemPool, WorkerSharedMemPool},
    mimc_k,
};

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct MerkleProof<F: Field> {
    pub index: usize,
    pub leaf: F,
    pub path: Vec<F>,
}

impl<F: Field> MerkleProof<F> {
    pub fn verify<H: SisuHasher<F>>(&self, root: &F) -> bool {
        let mut hasher = H::default();

        let mut index = self.index;
        let mut hv = self.leaf;
        for neighbor in self.path.iter() {
            hv = if index % 2 == 0 {
                hasher.hash_two(&hv, &neighbor)
            } else {
                hasher.hash_two(&neighbor, &hv)
            };

            index = index / 2;
        }

        &hv == root
    }
}

pub trait Tree<F: Field> {
    fn root(&self) -> F;
    fn prove(&self, index: usize) -> MerkleProof<F>;
    fn num_leaves(&self) -> usize;
}

pub struct ExchangeTree<F: IcicleConvertibleField, MT: SisuMerkleTreeEngine<F>> {
    exchange_leaves: CudaSlice<F>,
    tree: MT::Tree,
}

impl<F: IcicleConvertibleField, MT: SisuMerkleTreeEngine<F>> ExchangeTree<F, MT> {
    pub fn root(&self) -> F {
        self.tree.root()
    }

    pub fn prove(&self, index: usize) -> (Vec<F>, MerkleProof<F>) {
        let slice_size = self.exchange_leaves.len() / self.tree.num_leaves();
        let values = self
            .exchange_leaves
            .at_range(index * slice_size..(index + 1) * slice_size)
            .as_host();

        (values, self.tree.prove(index))
    }

    pub fn verify<H: SisuHasher<F>>(values: &[F], proof: &MerkleProof<F>, root: &F) -> bool {
        let mut hasher = H::default();
        let values_hash = hasher.hash_slice(values);

        if values_hash != proof.leaf {
            return false;
        }

        proof.verify::<H>(root)
    }
}

pub trait SisuMerkleTreeEngine<F: IcicleConvertibleField>: Clone {
    type Tree: Tree<F>;

    fn dummy(&self) -> Self::Tree;
    fn reduce(&self, leaves: &mut CudaSlice<F>, slice_size: usize) -> CudaSlice<F>;
    fn create(&self, leaves: CudaSlice<F>) -> Self::Tree;

    /// Return the evaluations after exchange and the tree.
    fn exchange_and_create(
        &self,
        leaves: &mut [CudaSlice<F>],
        slice_size: usize,
    ) -> ExchangeTree<F, Self>;
}

pub struct SisuMerkleNode<F: Field, H: SisuHasher<F>> {
    value: F,
    left: Option<SisuMerkelNodeRef<F, H>>,
    right: Option<SisuMerkelNodeRef<F, H>>,
}

pub type SisuMerkelNodeRef<F, H> = Rc<RefCell<SisuMerkleNode<F, H>>>;

pub struct SisuMerkleTree<F: Field, H: SisuHasher<F>> {
    leaves: Vec<F>,
    slice_size: usize,
    root: SisuMerkelNodeRef<F, H>,
}

impl<F: Field, H: SisuHasher<F>> Clone for SisuMerkleTree<F, H> {
    fn clone(&self) -> Self {
        Self {
            leaves: self.leaves.clone(),
            slice_size: self.slice_size,
            root: self.root.clone(),
        }
    }
}

impl<F: Field, H: SisuHasher<F>> SisuMerkleTree<F, H> {
    pub fn reduce(leaves: &[F], slice_size: usize) -> Vec<F> {
        assert!(leaves.len() % slice_size == 0);

        let mut hasher = H::default();
        let mut result = vec![];
        for i in (0..leaves.len()).step_by(slice_size) {
            result.push(hasher.hash_slice(&leaves[i..i + slice_size]));
        }

        result
    }

    pub fn from_vec(leaves: Vec<F>, slice_size: usize) -> Self {
        assert!(
            leaves.len() == 0 || (leaves.len() / slice_size).is_power_of_two(),
            "leaves_size={}  slice_size={}",
            leaves.len(),
            slice_size
        );

        let reduced_leaves = Self::reduce(&leaves, slice_size);
        let tree = Self::from_vec_no_hash_leaves(reduced_leaves);

        Self {
            leaves,
            slice_size,
            root: tree.root,
        }
    }

    pub fn from_vec_no_hash_leaves(leaves: Vec<F>) -> Self {
        assert!(leaves.len().is_power_of_two() || leaves.len() == 0);

        if leaves.len() == 0 {
            return Self {
                root: Rc::new(RefCell::new(SisuMerkleNode {
                    value: F::ZERO,
                    left: None,
                    right: None,
                })),
                leaves,
                slice_size: 1,
            };
        }

        if leaves.len() == 1 {
            return Self {
                root: Rc::new(RefCell::new(SisuMerkleNode {
                    value: leaves[0].clone(),
                    left: None,
                    right: None,
                })),
                leaves,
                slice_size: 1,
            };
        }

        let mut queue: Vec<SisuMerkelNodeRef<F, H>> = vec![];

        let mut hasher = H::default();

        for i in 0..leaves.len() {
            queue.push(Rc::new(RefCell::new(SisuMerkleNode {
                value: leaves[i].clone(),
                left: None,
                right: None,
            })));
        }

        while queue.len() > 1 {
            let mut i = 0;
            let mut new_queue = vec![];
            while i < queue.len() {
                let l = &queue[i];
                let mut r = &queue[i];
                if i + 1 < queue.len() {
                    r = &queue[i + 1];
                }

                let hv = hasher.hash_two(&l.borrow().value, &r.borrow().value);

                new_queue.push(Rc::new(RefCell::new(SisuMerkleNode {
                    value: hv,
                    left: Some(Rc::clone(&l)),
                    right: Some(Rc::clone(&r)),
                })));

                i = i + 2;
            }
            queue = new_queue;
        }

        assert!(
            queue.len() == 1,
            "Cannot calculate the exact root of merkle tree"
        );

        Self {
            leaves,
            slice_size: 1,
            root: Rc::clone(&queue[0]),
        }
    }

    pub fn from_vec_no_slice(leaves: Vec<F>) -> Self {
        Self::from_vec(leaves, 1)
    }

    pub fn root(&self) -> F {
        self.root.borrow().value.clone()
    }

    pub fn path_of(&self, slice_index: usize) -> (Vec<F>, Vec<F>) {
        let mut path = vec![];

        let mut current_layer = (self.leaves.len() / self.slice_size).ilog2() as usize;
        let mut current_node = Rc::clone(&self.root);
        while current_layer > 0 {
            let current_bit = (slice_index >> (current_layer - 1)) & 1;

            // Got the leaf node if left (or/and right) child is None.
            if let None = current_node.borrow().left {
                break;
            }

            let l: SisuMerkelNodeRef<F, H>;
            if let Some(node) = &current_node.borrow().left {
                l = Rc::clone(node);
            } else {
                panic!("Not found left node");
            }

            let r: SisuMerkelNodeRef<F, H>;
            if let Some(node) = &current_node.borrow().right {
                r = Rc::clone(node);
            } else {
                panic!("Not found right node")
            }

            let l_borrow = l.borrow();
            let r_borrow = r.borrow();

            // current_bit = 0 -> left.
            // current_bit = 1 -> right.
            if current_bit == 0 {
                current_node = Rc::clone(&l);
                path.push(r_borrow.value.clone());
            } else {
                current_node = Rc::clone(&r);
                path.push(l_borrow.value.clone());
            }

            current_layer -= 1;
        }

        (
            self.leaves[slice_index * self.slice_size..(slice_index + 1) * self.slice_size]
                .to_vec(),
            path.into_iter().rev().collect(),
        )
    }

    pub fn path_of_no_slice(&self, index: usize) -> (F, Vec<F>) {
        assert!(self.slice_size == 1);
        let mut path = vec![];

        let mut current_layer = (self.leaves.len() / self.slice_size).ilog2() as usize;
        let mut current_node = Rc::clone(&self.root);
        while current_layer > 0 {
            let current_bit = (index >> (current_layer - 1)) & 1;

            // Got the leaf node if left (or/and right) child is None.
            if let None = current_node.borrow().left {
                break;
            }

            let l: SisuMerkelNodeRef<F, H>;
            if let Some(node) = &current_node.borrow().left {
                l = Rc::clone(node);
            } else {
                panic!("Not found left node");
            }

            let r: SisuMerkelNodeRef<F, H>;
            if let Some(node) = &current_node.borrow().right {
                r = Rc::clone(node);
            } else {
                panic!("Not found right node")
            }

            let l_borrow = l.borrow();
            let r_borrow = r.borrow();

            // current_bit = 0 -> left.
            // current_bit = 1 -> right.
            if current_bit == 0 {
                current_node = Rc::clone(&l);
                path.push(r_borrow.value.clone());
            } else {
                current_node = Rc::clone(&r);
                path.push(l_borrow.value.clone());
            }

            current_layer -= 1;
        }

        (self.leaves[index], path.into_iter().rev().collect())
    }

    pub fn verify_path(root: &F, mut slice_index: usize, v: &[F], path: &[F]) -> bool {
        let mut hasher = H::default();
        let mut hv = hasher.hash_slice(v);

        for neighbor in path {
            hv = if slice_index % 2 == 0 {
                hasher.hash_two(&hv, &neighbor)
            } else {
                hasher.hash_two(&neighbor, &hv)
            };

            slice_index = slice_index / 2;
        }

        &hv == root
    }

    pub fn verify_path_no_slice(root: &F, mut index: usize, v: &F, path: &[F]) -> bool {
        let mut hasher = H::default();
        let mut hv = v.clone();

        for neighbor in path {
            hv = if index % 2 == 0 {
                hasher.hash_two(&hv, &neighbor)
            } else {
                hasher.hash_two(&neighbor, &hv)
            };

            index = index / 2;
        }

        &hv == root
    }

    pub fn get_leaves(&self) -> &[F] {
        &self.leaves
    }
}

impl<F: Field, H: SisuHasher<F>> Tree<F> for SisuMerkleTree<F, H> {
    fn prove(&self, index: usize) -> MerkleProof<F> {
        let (leaf, path) = self.path_of_no_slice(index);
        MerkleProof { index, leaf, path }
    }

    fn root(&self) -> F {
        self.root()
    }

    fn num_leaves(&self) -> usize {
        self.leaves.len()
    }
}

#[derive(Clone)]
pub struct DummyMerkleTreeEngine();

impl<F: Field> Tree<F> for DummyMerkleTreeEngine {
    fn prove(&self, _: usize) -> MerkleProof<F> {
        panic!("not implemented");
    }

    fn root(&self) -> F {
        panic!("not implemented");
    }

    fn num_leaves(&self) -> usize {
        panic!("not implemented");
    }
}

impl<F: IcicleConvertibleField> SisuMerkleTreeEngine<F> for DummyMerkleTreeEngine {
    type Tree = Self;

    fn dummy(&self) -> Self::Tree {
        panic!("not implemented");
    }

    fn create(&self, _: CudaSlice<F>) -> Self::Tree {
        panic!("not implemented");
    }

    fn reduce(&self, _: &mut CudaSlice<F>, _: usize) -> CudaSlice<F> {
        panic!("not implemented");
    }

    fn exchange_and_create(&self, _: &mut [CudaSlice<F>], _: usize) -> ExchangeTree<F, Self> {
        panic!("not implemented")
    }
}

#[derive(Clone)]
pub struct CPUMerkleTreeEngine<F: Field, H: SisuHasher<F>> {
    __phantom: PhantomData<(F, H)>,
}

impl<F: Field, H: SisuHasher<F>> CPUMerkleTreeEngine<F, H> {
    pub fn new() -> Self {
        Self {
            __phantom: PhantomData,
        }
    }
}

impl<F: IcicleConvertibleField, H: SisuHasher<F>> SisuMerkleTreeEngine<F>
    for CPUMerkleTreeEngine<F, H>
{
    type Tree = SisuMerkleTree<F, H>;

    fn dummy(&self) -> Self::Tree {
        Self::Tree {
            leaves: vec![],
            slice_size: 0,
            root: Rc::new(RefCell::new(SisuMerkleNode {
                value: F::ZERO,
                left: None,
                right: None,
            })),
        }
    }

    fn reduce(&self, leaves: &mut CudaSlice<F>, slice_size: usize) -> CudaSlice<F> {
        if slice_size == 0 {
            leaves.clone()
        } else {
            CudaSlice::on_host(SisuMerkleTree::<_, H>::reduce(
                leaves.as_ref_host(),
                slice_size,
            ))
        }
    }

    fn create(&self, leaves: CudaSlice<F>) -> Self::Tree {
        SisuMerkleTree::from_vec_no_hash_leaves(leaves.as_host())
    }

    fn exchange_and_create(
        &self,
        leaves: &mut [CudaSlice<F>],
        slice_size: usize,
    ) -> ExchangeTree<F, Self> {
        let mut exchanged_leaves = vec![F::ZERO; leaves.len() * leaves[0].len()];

        for i in 0..exchanged_leaves.len() {
            let machine_index = i % leaves.len();
            let element_index = i / leaves.len();

            exchanged_leaves[i] = leaves[machine_index].as_ref_host()[element_index];
        }

        let mut exchange_leaves = CudaSlice::on_host(exchanged_leaves);

        let reduced_output =
            <Self as SisuMerkleTreeEngine<F>>::reduce(&self, &mut exchange_leaves, slice_size);

        ExchangeTree {
            exchange_leaves,
            tree: self.create(reduced_output),
        }
    }
}

#[derive(Clone)]
pub struct GPUMerkleTreeEngine<F: IcicleConvertibleField> {
    parameter_k: Arc<HostOrDeviceSlice<F::IcicleField>>,
    parameter_d: Arc<HostOrDeviceSlice<u32>>,
}

impl GPUMerkleTreeEngine<FrBN254> {
    pub fn new() -> Self {
        let icicle_param_k: Vec<_> = cfg_iter!(mimc_k::K_BN254)
            .map(|x| IcicleFrBN254::from(x.0 .0))
            .collect();

        let u32_param_d: Vec<_> = cfg_iter!(mimc_k::D).map(|x| *x as u32).collect();

        let mut device_param_k = HostOrDeviceSlice::cuda_malloc(icicle_param_k.len()).unwrap();
        device_param_k.copy_from_host(&icicle_param_k).unwrap();

        let mut device_param_d = HostOrDeviceSlice::cuda_malloc(u32_param_d.len()).unwrap();
        device_param_d.copy_from_host(&u32_param_d).unwrap();

        Self {
            parameter_k: Arc::new(device_param_k),
            parameter_d: Arc::new(device_param_d),
        }
    }
}

type GPUTree<T> = HostOrDeviceSlice<T>;

impl<F: IcicleConvertibleField> Tree<F> for GPUTree<F::IcicleField> {
    fn prove(&self, index: usize) -> MerkleProof<F> {
        let leaf = extract_single_field_from_device_slice(self, index);

        // Get the path
        let mut layer_size = (self.len() + 1) / 2;
        let mut extract_index = index;
        let mut offset = 0;
        let mut path = vec![];
        let num_layers = layer_size.ilog2() as usize;
        for _ in 0..num_layers {
            if extract_index % 2 == 0 {
                path.push(extract_single_field_from_device_slice(
                    self,
                    offset + extract_index + 1,
                ));
            } else {
                path.push(extract_single_field_from_device_slice(
                    self,
                    offset + extract_index - 1,
                ));
            }

            offset += layer_size;
            extract_index = extract_index / 2;
            layer_size /= 2;
        }

        MerkleProof { index, leaf, path }
    }

    fn root(&self) -> F {
        if self.len() == 0 {
            F::ZERO
        } else {
            extract_single_field_from_device_slice(self, self.len() - 1)
        }
    }

    fn num_leaves(&self) -> usize {
        (self.len() + 1) / 2
    }
}

impl SisuMerkleTreeEngine<FrBN254> for GPUMerkleTreeEngine<FrBN254> {
    type Tree = GPUTree<IcicleFrBN254>;

    fn dummy(&self) -> Self::Tree {
        HostOrDeviceSlice::Host(vec![])
    }

    fn reduce(&self, leaves: &mut CudaSlice<FrBN254>, slice_size: usize) -> CudaSlice<FrBN254> {
        assert!(slice_size == 0 || leaves.len() % slice_size == 0);

        let mut device_output: HostOrDeviceSlice<IcicleFrBN254>;
        if slice_size == 0 {
            let merkle_tree_size = leaves.len() * 2 - 1;

            // We allocate a memory enough to create a merkle tree so that we don't
            // need to allocate memory many times.
            device_output = HostOrDeviceSlice::cuda_malloc(merkle_tree_size).unwrap();
            device_output
                .device_copy_partially(leaves.as_ref_device(), 0, 0)
                .unwrap();
        } else {
            let config = MerkleTreeConfig::default_for_device(&self.parameter_k, &self.parameter_d);

            let reduced_leaves_size = leaves.len() / slice_size;
            let reduced_merkle_tree_size = reduced_leaves_size * 2 - 1;

            // We allocate a memory enough to create a merkle tree so that we don't
            // need to allocate memory many times.
            device_output = HostOrDeviceSlice::cuda_malloc(reduced_merkle_tree_size).unwrap();

            let leaves_size = leaves.len();
            hash_merkle_tree_slice(
                &config,
                leaves.as_ref_device(),
                &mut device_output,
                leaves_size as u32,
                slice_size as u32,
            )
            .unwrap();
        }

        CudaSlice::on_device(device_output)
    }

    fn create(&self, mut leaves: CudaSlice<FrBN254>) -> Self::Tree {
        let config = MerkleTreeConfig::default_for_device(&self.parameter_k, &self.parameter_d);

        // The memory size of parameter leaves is enough to build a merkle tree.
        // We need to calculated the actual leaves size to pass into this
        // function.
        let actual_leaves_size = (leaves.len() + 1) / 2;
        build_merkle_tree(&config, leaves.as_mut_device(), actual_leaves_size as u32).unwrap();

        leaves.as_device()
    }

    fn exchange_and_create(
        &self,
        leaves: &mut [CudaSlice<FrBN254>],
        slice_size: usize,
    ) -> ExchangeTree<FrBN254, Self> {
        let mut leaves_2d = vec![];
        let first_leaves_size = leaves[0].len();
        for l in leaves.iter_mut() {
            assert!(l.len() == first_leaves_size);
            leaves_2d.push(HostOrDeviceSlice::device_ref(l.as_ref_device()));
        }

        let leaves_2d = HostOrDeviceSlice2D::ref_from(leaves_2d).unwrap();

        let mut exchanged_leaves =
            HostOrDeviceSlice::cuda_malloc(leaves.len() * leaves[0].len()).unwrap();

        exchange_evaluations(&leaves_2d, &mut exchanged_leaves).unwrap();

        let mut exchange_leaves = CudaSlice::on_device(exchanged_leaves);

        let reduced_leaves = self.reduce(&mut exchange_leaves, slice_size);

        ExchangeTree {
            exchange_leaves,
            tree: self.create(reduced_leaves),
        }
    }
}

fn extract_single_field_from_device_slice<F: IcicleConvertibleField>(
    device: &HostOrDeviceSlice<F::IcicleField>,
    index: usize,
) -> F {
    let mut f = vec![F::IcicleField::zero()];
    device.copy_to_host_partially(&mut f, index).unwrap();
    F::from_icicle(&f[0])
}

/// Returns a vector of form <to_worker_index - relative_slice_index - root>
pub fn split_elements_for_multi_merkle_tree<F: IcicleConvertibleField>(
    evaluations: &mut CudaSlice<F>,
    num_workers: usize,
) -> Vec<Box<Vec<F>>> {
    let num_elements_per_worker = div_round(evaluations.len(), num_workers);

    let mut split_elements: Vec<_> =
        cfg_chunks!(evaluations.as_ref_host(), num_elements_per_worker)
            .map(|x| Box::new(x.to_vec()))
            .collect();
    split_elements.extend(vec![Box::new(vec![]); num_workers - split_elements.len()]);

    split_elements
}

/// Receives a vector of form <from_worker_index - evaluations>.
/// Returns a vector of form <relative_slice_index - evaluations>
pub fn synthetized_elements_for_central_merkle_tree<F: Field>(
    recv_elements: Vec<Vec<F>>,
) -> Vec<Vec<F>> {
    let num_elements_of_this_worker = recv_elements[0].len();

    let mut elements = vec![vec![]; num_elements_of_this_worker];
    for single_recv_elements in recv_elements {
        assert!(single_recv_elements.len() == num_elements_of_this_worker);

        for (element_index, single_element) in single_recv_elements.into_iter().enumerate() {
            elements[element_index].push(single_element);
        }
    }

    elements
}

pub struct MultiMerkleTreeWorker<
    'a,
    F: IcicleConvertibleField,
    MTEngine: SisuMerkleTreeEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    worker: &'a WorkerNode<S, R>,
    mempool: WorkerSharedMemPool<'a, F, S, R>,

    mt_engine: &'a MTEngine,

    num_workers: usize,
    worker_index: usize,

    __phantom: PhantomData<F>,
}

impl<
        'a,
        F: IcicleConvertibleField,
        MTEngine: SisuMerkleTreeEngine<F>,
        S: SisuSender,
        R: SisuReceiver,
    > MultiMerkleTreeWorker<'a, F, MTEngine, S, R>
{
    pub fn new(
        worker_node: &'a WorkerNode<S, R>,
        mt_engine: &'a MTEngine,
        mempool: WorkerSharedMemPool<'a, F, S, R>,
        num_workers: usize,
        worker_index: usize,
    ) -> Self {
        assert!(num_workers.is_power_of_two());

        Self {
            worker: worker_node,
            mempool,
            mt_engine,
            num_workers,
            worker_index,
            __phantom: PhantomData,
        }
    }

    pub fn commit<'b>(&self, evaluations: &'b mut CudaSlice<F>) -> MTEngine::Tree {
        let worker_merkle_tree = self.commit_uncheck(evaluations);
        self.worker.recv_from_master::<usize>().unwrap(); // dummy recv

        worker_merkle_tree
    }

    pub fn commit_and_wait_root<'b>(
        &self,
        evaluations: &'b mut CudaSlice<F>,
    ) -> (F, MTEngine::Tree) {
        let worker_merkle_tree = self.commit_uncheck(evaluations);

        let second_root = self.worker.recv_from_master::<F>().unwrap();

        (second_root, worker_merkle_tree)
    }

    fn commit_uncheck<'b>(&self, evaluations: &'b mut CudaSlice<F>) -> MTEngine::Tree {
        let splitted_elements = split_elements_for_multi_merkle_tree(evaluations, self.num_workers);

        self.mempool.share(splitted_elements);

        let recv_elements = self.mempool.synthetize();

        // At this phase, each worker has been received some
        // evaluations of the same slice index of other workers, the
        // evaluation-index depends on the current worker index.
        // This worker must commit these evaluations, then sends all of
        // them to the master worker for the central commitment.
        let synthetized_elements = synthetized_elements_for_central_merkle_tree(recv_elements);

        let worker_merkle_tree = if synthetized_elements.len() > 0 {
            let slice_size = synthetized_elements[0].len();
            let synthetized_elements: Vec<_> = synthetized_elements.into_iter().flatten().collect();

            let reduced_synthetized_elements = self
                .mt_engine
                .reduce(&mut CudaSlice::on_host(synthetized_elements), slice_size);

            let worker_merkle_tree = self.mt_engine.create(reduced_synthetized_elements);

            worker_merkle_tree
        } else {
            self.mt_engine.dummy()
        };

        self.worker
            .send_to_master(&worker_merkle_tree.root())
            .unwrap();

        worker_merkle_tree
    }

    /// Returns evaluations + worker_proof_if_any (evaluations_hash + worker_path)
    pub fn generate_worker_proof(
        &self,
        evaluations: &CudaSlice<F>,
        worker_merkle_tree: &MTEngine::Tree,
        index: usize,
    ) -> (F, Option<(F, Vec<F>)>) {
        let mut worker_proof = None;

        let num_groups_per_worker = div_round(evaluations.len(), self.num_workers);
        if index / num_groups_per_worker == self.worker_index {
            let true_index = index % num_groups_per_worker;

            let proof = worker_merkle_tree.prove(true_index);

            worker_proof = Some((proof.leaf, proof.path));
        }

        (evaluations.at(index), worker_proof)
    }
}

pub struct MultiMerkleTreeMaster<
    'a,
    F: IcicleConvertibleField,
    MTEngine: SisuMerkleTreeEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    master_node: &'a MasterNode<S, R>,
    mt_engine: &'a MTEngine,
    mempool: MasterSharedMemPool<'a, S, R>,
    num_workers: usize,
    __phantom: PhantomData<F>,
}

impl<
        'a,
        F: IcicleConvertibleField,
        MTEngine: SisuMerkleTreeEngine<F>,
        S: SisuSender,
        R: SisuReceiver,
    > MultiMerkleTreeMaster<'a, F, MTEngine, S, R>
{
    pub fn new(
        master: &'a MasterNode<S, R>,
        mt_engine: &'a MTEngine,
        mempool: MasterSharedMemPool<'a, S, R>,
        num_workers: usize,
    ) -> Self {
        Self {
            master_node: master,
            mt_engine,
            mempool,
            num_workers,
            __phantom: PhantomData,
        }
    }

    pub fn commit_and_send_root(&self) -> MTEngine::Tree {
        self.mempool.block_all_workers_until_done();

        let worker_roots = self.master_node.recv_from_workers::<F>().unwrap();

        let prepared_worker_roots = self
            .mt_engine
            .reduce(&mut CudaSlice::on_host(worker_roots), 0);
        let central_merkle_tree = self.mt_engine.create(prepared_worker_roots);

        self.master_node
            .send_to_workers(&central_merkle_tree.root())
            .unwrap();

        central_merkle_tree
    }

    pub fn commit(&self) -> MTEngine::Tree {
        self.mempool.block_all_workers_until_done();

        let worker_roots = self.master_node.recv_from_workers::<F>().unwrap();
        self.master_node.send_to_workers(&1usize).unwrap(); // dummy send

        let prepared_worker_roots = self
            .mt_engine
            .reduce(&mut CudaSlice::on_host(worker_roots), 0);
        self.mt_engine.create(prepared_worker_roots)
    }

    /// Return the merkle path corresponding to worker index.
    pub fn generate_master_proof(
        &self,
        index: usize,
        evaluations_size: usize,
        central_merkle_tree: &MTEngine::Tree,
    ) -> Vec<F> {
        let num_elements_per_worker = div_round(evaluations_size, self.num_workers);
        let worker_index = index / num_elements_per_worker;

        let proof = central_merkle_tree.prove(worker_index);

        proof.path
    }

    /// Returns evaluations + paths.
    pub fn combine_proofs(
        &self,
        all_worker_elements: Vec<F>,
        worker_proofs: Vec<Option<(F, Vec<F>)>>,
        master_path: Vec<F>,
        index: usize,
        evaluations_size: usize,
    ) -> MultiMerkleTreeQuery<F> {
        assert_eq!(all_worker_elements.len(), self.num_workers);

        let num_elements_per_worker = div_round(evaluations_size, worker_proofs.len());
        let mut evaluations_hash = F::ZERO;
        let mut path = vec![];
        let mut has_one_worker_provided_proof = false;
        for (worker_index, worker_proof) in worker_proofs.into_iter().enumerate() {
            if index / num_elements_per_worker == worker_index {
                assert!(
                    !has_one_worker_provided_proof && worker_proof.is_some(),
                    "expect that worker {} must send the path",
                    worker_index
                );

                let (worker_evaluation_hash, worker_path) = worker_proof.unwrap();
                evaluations_hash = worker_evaluation_hash;
                path = worker_path;
                has_one_worker_provided_proof = true;
            } else {
                assert!(
                    worker_proof.is_none(),
                    "expect that worker {} must not send the path",
                    worker_index
                );
            }
        }

        assert!(has_one_worker_provided_proof);
        path = [path, master_path].concat();

        MultiMerkleTreeQuery {
            evaluations: all_worker_elements,
            evaluations_hash,
            evaluations_hash_path: path,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct MultiMerkleTreeQuery<F: Field> {
    pub evaluations: Vec<F>,           // true leaves
    pub evaluations_hash: F,           // intermidate root
    pub evaluations_hash_path: Vec<F>, // path of intermediate root
}

impl<F: Field> MultiMerkleTreeQuery<F> {
    pub fn to_vec(&self) -> Vec<F> {
        let mut result = vec![];
        result.extend_from_slice(&self.evaluations);
        result.push(self.evaluations_hash);
        result.extend_from_slice(&self.evaluations_hash_path);

        result
    }

    pub fn verify<H: SisuHasher<F>>(&self, root: &F, index: usize) -> Result<(), Error> {
        let mut hasher = H::default();

        let evaluations_root = hasher.hash_slice(&self.evaluations);
        if evaluations_root != self.evaluations_hash {
            return Err(Error::FRI(format!("The evaluations root at is invalid")));
        }

        if !SisuMerkleTree::<F, H>::verify_path_no_slice(
            root,
            index,
            &self.evaluations_hash,
            &self.evaluations_hash_path,
        ) {
            return Err(Error::FRI(format!("The query point is invalid",)));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use sha2::Sha256;
    use sisulib::{
        common::{convert_field_to_string, convert_vec_field_to_string},
        field::{Fp41, FpSisu},
    };

    use crate::{hash::SisuMimc, sisu_merkle_tree::SisuMerkleTree};

    #[test]
    fn test_merkletree() {
        let values = vec![
            FpSisu::from(1),
            FpSisu::from(2),
            FpSisu::from(3),
            FpSisu::from(4),
            FpSisu::from(5),
            FpSisu::from(6),
            FpSisu::from(7),
            FpSisu::from(8),
        ];

        let merkle_tree = SisuMerkleTree::<_, SisuMimc<FpSisu>>::from_vec_no_hash_leaves(values);
        let (v, path) = merkle_tree.path_of_no_slice(2);
        assert!(
            SisuMerkleTree::<FpSisu, SisuMimc<FpSisu>>::verify_path_no_slice(
                &merkle_tree.root(),
                2,
                &v,
                &path
            )
        );
        println!("ROOT: {:?}", convert_field_to_string(&merkle_tree.root()));
        println!("VALUE: {:?}", convert_field_to_string(&v));
        println!("PATH: {:?}", convert_vec_field_to_string(&path));
    }

    #[test]
    fn test_multi_merkletree() {
        let values = vec![
            FpSisu::from(1),
            FpSisu::from(2),
            FpSisu::from(3),
            FpSisu::from(4),
            FpSisu::from(5),
            FpSisu::from(6),
            FpSisu::from(7),
            FpSisu::from(8),
        ];

        let merkle_tree = SisuMerkleTree::<_, SisuMimc<FpSisu>>::from_vec_no_hash_leaves(values);
        let (v, path) = merkle_tree.path_of_no_slice(2);
        assert!(
            SisuMerkleTree::<FpSisu, SisuMimc<FpSisu>>::verify_path_no_slice(
                &merkle_tree.root(),
                2,
                &v,
                &path
            )
        );
        println!("ROOT: {:?}", convert_field_to_string(&merkle_tree.root()));
        println!("VALUE: {:?}", convert_field_to_string(&v));
        println!("PATH: {:?}", convert_vec_field_to_string(&path));
    }

    #[test]
    fn test_merkletree_vec_full_slice() {
        let values = vec![
            Fp41::from(1),
            Fp41::from(2),
            Fp41::from(3),
            Fp41::from(4),
            Fp41::from(5),
            Fp41::from(6),
            Fp41::from(7),
            Fp41::from(8),
        ];

        let merkle_tree: SisuMerkleTree<Fp41, Sha256> =
            SisuMerkleTree::<_, Sha256>::from_vec(values, 8);
        let (v, path) = merkle_tree.path_of(0);
        assert!(SisuMerkleTree::<Fp41, Sha256>::verify_path(
            &merkle_tree.root(),
            0,
            &v,
            &path
        ));
    }

    #[test]
    fn test_cfg_macros() {
        #[cfg(feature = "parallel")]
        println!("In parallel mode");
    }
}
