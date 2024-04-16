use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sisulib::common::{deserialize, serialize, Error};

use crate::fiat_shamir::Transcript;

pub trait SisuSender: Send + Sync + Clone {
    fn send<T: CanonicalSerialize>(&self, obj: &T) -> Result<(), Error>;
}

pub trait SisuReceiver {
    fn recv<T: CanonicalDeserialize>(&self) -> Result<T, Error>;
}

#[derive(Clone)]
pub struct NoChannel {}

impl SisuSender for NoChannel {
    fn send<T: CanonicalSerialize>(&self, _: &T) -> Result<(), Error> {
        Err(Error::ChannelIO(String::from("not implemented")))
    }
}

impl SisuReceiver for NoChannel {
    fn recv<T: CanonicalDeserialize>(&self) -> Result<T, Error> {
        Err(Error::ChannelIO(String::from("not implemented")))
    }
}

pub type DefaultSender = std::sync::mpsc::Sender<Vec<u8>>;
pub type DefaultReceiver = std::sync::mpsc::Receiver<Vec<u8>>;

pub fn default() -> (DefaultSender, DefaultReceiver) {
    std::sync::mpsc::channel()
}

impl SisuSender for DefaultSender {
    fn send<T: CanonicalSerialize>(&self, obj: &T) -> Result<(), Error> {
        match self.send(serialize(obj)) {
            Ok(()) => Ok(()),
            Err(err) => Err(Error::SendChannelIO(err)),
        }
    }
}

impl SisuReceiver for DefaultReceiver {
    fn recv<T: CanonicalDeserialize>(&self) -> Result<T, Error> {
        match self.recv() {
            Ok(data) => Ok(deserialize(&data)),
            Err(err) => Err(Error::RecvChannelIO(err)),
        }
    }
}

pub struct MasterNode<S: SisuSender, R: SisuReceiver> {
    receiver: R,
    workers: Vec<S>,
}

impl Default for MasterNode<NoChannel, NoChannel> {
    fn default() -> Self {
        Self {
            receiver: NoChannel {},
            workers: vec![],
        }
    }
}

impl MasterNode<DefaultSender, DefaultReceiver> {
    pub fn from_channel(receiver: DefaultReceiver) -> Self {
        Self {
            receiver,
            workers: vec![],
        }
    }
}

impl<S: SisuSender, R: SisuReceiver> MasterNode<S, R> {
    pub fn add_worker(&mut self, worker: &S) {
        self.workers.push(worker.clone());
    }

    /// This node is a master and receives values from all workers.
    pub fn recv_from_workers<T: CanonicalDeserialize + Default>(&self) -> Result<Vec<T>, Error> {
        let mut results = vec![];
        for _ in 0..self.workers.len() {
            results.push(T::default());
        }

        let mut received = 0;
        while received < self.workers.len() {
            let transcript = self.receiver.recv::<Transcript>()?;

            let mut transcript = transcript.into_iter();
            let worker_id = transcript.pop_and_deserialize::<usize>();
            results[worker_id] = transcript.pop_and_deserialize();
            received += 1;
        }

        Ok(results)
    }

    fn dummy_send_to_workers(&self) -> Result<(), Error> {
        self.send_to_workers(&1usize)
    }

    pub fn recv_from_workers_and_done<T: CanonicalDeserialize + Default>(
        &self,
    ) -> Result<Vec<T>, Error> {
        let result = self.recv_from_workers();
        self.dummy_send_to_workers()?;
        result
    }

    pub fn send_to_workers<T: CanonicalSerialize>(&self, obj: &T) -> Result<(), Error> {
        for i in 0..self.workers.len() {
            if let Err(err) = self.workers[i].send(obj) {
                return Err(Error::ChannelIO(format!(
                    "An error occurs when send to worker {}: {}",
                    i, err
                )));
            }
        }

        Ok(())
    }
}

pub struct WorkerNode<S: SisuSender, R: SisuReceiver> {
    id: usize,
    receiver: R,
    master: S,
}

impl WorkerNode<DefaultSender, DefaultReceiver> {
    pub fn from_channel(id: usize, receiver: DefaultReceiver, master: &DefaultSender) -> Self {
        Self {
            id,
            receiver,
            master: master.clone(),
        }
    }
}

impl<S: SisuSender, R: SisuReceiver> WorkerNode<S, R> {
    /// This node is a worker and receives a value from master.
    pub fn recv_from_master<T: CanonicalDeserialize>(&self) -> Result<T, Error> {
        self.receiver.recv()
    }

    fn dummy_recv_from_master(&self) -> Result<(), Error> {
        self.recv_from_master::<usize>()?;
        Ok(())
    }

    pub fn send_to_master<T: CanonicalSerialize>(&self, obj: &T) -> Result<(), Error> {
        let mut transcript = Transcript::default();
        transcript.serialize_and_push(&self.id);
        transcript.serialize_and_push(obj);

        self.master.send(&transcript)
    }

    pub fn send_to_master_and_done<T: CanonicalSerialize>(&self, obj: &T) -> Result<(), Error> {
        self.send_to_master(obj)?;
        self.dummy_recv_from_master()
    }
}

pub struct PeerNode<S: SisuSender, R: SisuReceiver> {
    id: usize,
    receiver: R,
    peers: Vec<Option<S>>,
}

impl PeerNode<DefaultSender, DefaultReceiver> {
    pub fn from_channel(id: usize, receiver: DefaultReceiver, num_peers: usize) -> Self {
        Self {
            id,
            receiver,
            peers: vec![None; num_peers],
        }
    }
}

impl<S: SisuSender, R: SisuReceiver> PeerNode<S, R> {
    pub fn add_peer(&mut self, peer_info: (usize, &S)) {
        assert_ne!(self.id, peer_info.0, "do not add myself to peer list");
        if let Some(_) = self.peers[peer_info.0] {
            panic!("add a duplicated peer")
        }

        self.peers[peer_info.0] = Some(peer_info.1.clone());
    }

    /// This node is a peer and receives values from all other peers.
    pub fn recv_from_peers<T: CanonicalDeserialize + Default>(
        &self,
        my_value: T,
    ) -> Result<Vec<T>, Error> {
        let mut results = vec![];
        for _ in 0..self.peers.len() {
            results.push(T::default());
        }

        results[self.id] = my_value;

        let mut received = 1;
        while received < self.peers.len() {
            let transcript = self.receiver.recv::<Transcript>()?;

            let mut transcript = transcript.into_iter();
            let peer_id = transcript.pop_and_deserialize::<usize>();
            assert_ne!(peer_id, self.id);

            results[peer_id] = transcript.pop_and_deserialize();
            received += 1;
        }

        Ok(results)
    }

    pub fn send_to_peers<T: CanonicalSerialize>(&self, objects: &[T]) -> Result<(), Error> {
        if objects.len() != self.peers.len() {
            return Err(Error::ChannelIO(format!(
                "Expected the list of objects has the same length with peers"
            )));
        }

        for (i, obj) in objects.iter().enumerate() {
            let mut transcript = Transcript::default();
            transcript.serialize_and_push(&self.id);
            transcript.serialize_and_push(obj);

            if let Some(peer) = &self.peers[i] {
                if let Err(err) = peer.send(&transcript) {
                    return Err(Error::ChannelIO(format!(
                        "An error occurs when send to peer {}: {}",
                        i, err
                    )));
                }
            } else {
                assert!(i == self.id);
            }
        }

        Ok(())
    }
}
