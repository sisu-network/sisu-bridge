use ark_serialize::CanonicalSerialize;
use sha2::digest::DynDigest;
use sha2::Sha256;
use sisulib::common::serialize;
use std::{cell::RefCell, rc::Rc};

pub struct MerkleNode<Leaf> {
    value: Vec<u8>,
    left: Option<MerkelNodeRef<Leaf>>,
    right: Option<MerkelNodeRef<Leaf>>,
}

pub type MerkelNodeRef<Leaf> = Rc<RefCell<MerkleNode<Leaf>>>;

pub struct MerkleTree<Leaf: Clone> {
    leaves: Vec<Leaf>,
    root: MerkelNodeRef<Leaf>,
}

impl<Leaf: Clone> Clone for MerkleTree<Leaf> {
    fn clone(&self) -> Self {
        Self {
            leaves: self.leaves.clone(),
            root: self.root.clone(),
        }
    }
}

impl<Leaf: CanonicalSerialize + Clone> MerkleTree<Leaf> {
    pub fn from_vec(leaves: Vec<Leaf>) -> Self {
        assert!(leaves.len().is_power_of_two());

        let mut queue: Vec<MerkelNodeRef<Leaf>> = vec![];

        let mut hasher = Sha256::default();

        for i in 0..leaves.len() {
            hasher.update(&serialize(&leaves[i]));

            queue.push(Rc::new(RefCell::new(MerkleNode {
                value: hasher.finalize_reset().to_vec(),
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

                hasher.update(&l.borrow().value);
                hasher.update(&r.borrow().value);

                new_queue.push(Rc::new(RefCell::new(MerkleNode {
                    value: hasher.finalize_reset().to_vec(),
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
            root: Rc::clone(&queue[0]),
        }
    }

    pub fn root(&self) -> Vec<u8> {
        self.root.borrow().value.clone()
    }

    pub fn path_of(&self, slice_index: usize) -> (Leaf, Vec<Vec<u8>>) {
        let mut path = vec![];

        let mut current_layer = self.leaves.len().ilog2() as usize;
        let mut current_node = Rc::clone(&self.root);
        while current_layer > 0 {
            let current_bit = (slice_index >> (current_layer - 1)) & 1;

            // Got the leaf node if left (or/and right) child is None.
            if let None = current_node.borrow().left {
                break;
            }

            let l: MerkelNodeRef<Leaf>;
            if let Some(node) = &current_node.borrow().left {
                l = Rc::clone(node);
            } else {
                panic!("Not found left node");
            }

            let r: MerkelNodeRef<Leaf>;
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
            self.leaves[slice_index].clone(),
            path.into_iter().rev().collect(),
        )
    }

    pub fn verify_path(root: &[u8], mut slice_index: usize, v: &Leaf, path: &[Vec<u8>]) -> bool {
        let mut hasher = Sha256::default();
        hasher.update(&serialize(v));
        let mut hv = hasher.finalize_reset().to_vec();

        for neighbor in path {
            if slice_index % 2 == 0 {
                hasher.update(&hv);
                hasher.update(&neighbor);
            } else {
                hasher.update(&neighbor);
                hasher.update(&hv);
            };

            hv = hasher.finalize_reset().to_vec();

            slice_index = slice_index / 2;
        }

        &hv == root
    }
}
