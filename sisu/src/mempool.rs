use std::sync::{Arc, RwLock};

use crate::channel::{MasterNode, SisuReceiver, SisuSender, WorkerNode};

#[derive(Clone)]
pub struct SharedMemPool<T: Clone> {
    storage: Arc<Vec<Vec<RwLock<Box<Vec<T>>>>>>, // storage[receiver_idx][sender_idx] = vec[].
}

impl<T: Clone> SharedMemPool<T> {
    pub fn new(num_workers: usize) -> Self {
        let mut storage = vec![];
        for _receiver_id in 0..num_workers {
            let mut worker_storage = vec![];
            for _sender_id in 0..num_workers {
                worker_storage.push(RwLock::new(Box::new(vec![])));
            }

            storage.push(worker_storage);
        }

        Self {
            storage: Arc::new(storage),
        }
    }
}

pub struct MasterSharedMemPool<'a, S: SisuSender, R: SisuReceiver> {
    master: &'a MasterNode<S, R>,
}

impl<'a, S: SisuSender, R: SisuReceiver> MasterSharedMemPool<'a, S, R> {
    pub fn new(master: &'a MasterNode<S, R>) -> Self {
        Self { master }
    }

    pub fn block_all_workers_until_done(&self) {
        self.master.recv_from_workers_and_done::<usize>().unwrap(); // Share
        self.master.recv_from_workers_and_done::<usize>().unwrap(); // Synthetize
    }
}

pub struct WorkerSharedMemPool<'a, T: Clone, S: SisuSender, R: SisuReceiver> {
    worker: &'a WorkerNode<S, R>,
    worker_id: usize,
    storage: Arc<Vec<Vec<RwLock<Box<Vec<T>>>>>>, // storage[receiver_idx][sender_idx] = vec[].
}

impl<'a, T: Clone, S: SisuSender, R: SisuReceiver> WorkerSharedMemPool<'a, T, S, R> {
    pub fn clone_from(
        worker_id: usize,
        worker_node: &'a WorkerNode<S, R>,
        root_mempool: SharedMemPool<T>,
    ) -> Self {
        Self {
            worker: worker_node,
            worker_id,
            storage: root_mempool.storage,
        }
    }

    /// Share to each worker a vector of values.
    /// Input: values[to_id] = vec[];
    pub fn share(&self, values: Vec<Box<Vec<T>>>) {
        assert_eq!(values.len(), self.storage.len());

        let sender_id = self.worker_id;
        for (receiver_id, recv_value) in values.into_iter().enumerate() {
            let mut single_storage = self.storage[receiver_id][sender_id].write().unwrap();
            *single_storage = recv_value;
        }

        self.worker.send_to_master_and_done(&1usize).unwrap(); // Blocking until every workers shared.
    }

    /// Synthetize all received values.
    pub fn synthetize(&self) -> Vec<Vec<T>> {
        let receiver_id = self.worker_id;
        let mut result = vec![];

        for recv_storage in self.storage[receiver_id].iter() {
            let recv_value = recv_storage.read().unwrap();
            result.push(recv_value.to_vec());
        }

        self.worker.send_to_master_and_done(&1usize).unwrap(); // Blocking until every workers synthetized.

        result
    }
}
