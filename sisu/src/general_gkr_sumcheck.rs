use sisulib::circuit::general_circuit::circuit::GeneralCircuit;

use crate::{
    channel::{SisuReceiver, SisuSender, WorkerNode},
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript},
    icicle_converter::IcicleConvertibleField,
    polynomial::{ProductBookeepingTablePlus, RootProductBookeepingTable},
    sisu_engine::{GateExtensionType, SisuEngine},
    sumcheck::MultiProductSumcheckWorkerProver,
};

pub struct GeneralGKRSumcheckPhaseOneWorkerProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    engine: &'a Engine,

    circuit: &'a GeneralCircuit<F>,
    layer_index: usize,

    num_replicas: usize,

    wx_evaluations: CudaSlice<F>,
    wx_constant_one: CudaSlice<F>,
    wy_evaluations: Vec<CudaSlice<F>>,
    wy_constant_ones: Vec<CudaSlice<F>>,
    bookeeping_g: CudaSlice<F>,

    // all last evaluations at bookeeping table at the true index must be the same value.
    assert_w: Vec<bool>,
    sumcheck_prover:
        MultiProductSumcheckWorkerProver<'a, F, Engine::RootProductBookeepingTable, S, R>,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    GeneralGKRSumcheckPhaseOneWorkerProver<'a, F, Engine, S, R>
{
    pub fn new(
        engine: &'a Engine,
        circuit: &'a GeneralCircuit<F>,
        layer_index: usize,
        worker: Option<&'a WorkerNode<S, R>>,
        bookeeping_g: CudaSlice<F>,
        num_replicas: usize,
        wx_evaluations: CudaSlice<F>,
        wy_evaluations: Vec<CudaSlice<F>>,
    ) -> Self {
        let sumcheck_prover = if worker.is_none() {
            MultiProductSumcheckWorkerProver::default()
        } else {
            MultiProductSumcheckWorkerProver::new(worker.unwrap())
        };

        Self {
            engine,
            circuit,
            layer_index,
            num_replicas,
            bookeeping_g,
            sumcheck_prover,
            assert_w: vec![],
            wx_constant_one: engine.cuda_slice_ones(wx_evaluations.len()),
            wx_evaluations,
            wy_constant_ones: wy_evaluations
                .iter()
                .map(|wy| engine.cuda_slice_ones(wy.len()))
                .collect(),
            wy_evaluations,
        }
    }

    pub fn add(&mut self, gate_extensions: GateExtensionType, assert_w: bool) {
        let mut is_zero = true;

        for ext in gate_extensions.to_extensions(self.circuit.layer(self.layer_index)) {
            if ext.evaluations.len() > 0 {
                is_zero = false;
            }
        }
        if is_zero {
            self.sumcheck_prover.add(None);
            self.assert_w.push(false);
            return;
        }

        let gate_type = self.sumcheck_prover.len();

        let (f2, f3) = if gate_type == 0 {
            (&mut self.wx_constant_one, &mut self.wy_constant_ones)
        } else if gate_type == 1 {
            (&mut self.wx_evaluations, &mut self.wy_evaluations)
        } else if gate_type == 2 {
            (&mut self.wx_evaluations, &mut self.wy_constant_ones)
        } else if gate_type == 3 {
            (&mut self.wx_constant_one, &mut self.wy_evaluations)
        } else {
            panic!("got an invalid gate type {}", gate_type);
        };

        self.sumcheck_prover.add(Some(
            <Engine::RootProductBookeepingTable as RootProductBookeepingTable<F>>::new(
                F::ONE,
                self.engine.initialize_phase_1_plus(
                    self.circuit,
                    self.num_replicas,
                    self.layer_index,
                    gate_extensions,
                    f3,
                    &mut self.bookeeping_g,
                ),
                f2.clone(),
            ),
        ));
        self.assert_w.push(assert_w);
    }

    // Run the sumcheck, return the random points, last values (depend on phase) and transcript .
    pub fn run<FS: FiatShamirEngine<F>>(
        &mut self,
        fiat_shamir_engine: &mut FS,
    ) -> (Vec<F>, Vec<F>, Option<F>, Transcript) {
        let (random_points, last_f, last_g, transcript) =
            self.sumcheck_prover.run(fiat_shamir_engine);

        let mut all_w = vec![];
        let mut final_w = None;

        if self.sumcheck_prover.is_standalone() {
            for (sumcheck_index, (f, g)) in last_f.into_iter().zip(last_g.into_iter()).enumerate() {
                assert_eq!(f.len(), g.len());
                if g.len() == 0 {
                    all_w.push(F::ZERO);
                    continue;
                }

                assert_eq!(g.len(), 1);

                // Currently, g[0] == t_at_u.

                if self.assert_w[sumcheck_index] {
                    if final_w.is_none() {
                        final_w = Some(g[0]);
                    } else {
                        assert!(final_w.unwrap() == g[0]);
                    }
                }

                all_w.push(g[0]);
            }
        }

        (random_points, all_w, final_w, transcript)
    }
}

pub struct GeneralGKRSumcheckPhaseTwoWorkerProver<
    'a,
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    engine: &'a Engine,

    num_replicas: usize,
    circuit: &'a GeneralCircuit<F>,
    layer_index: usize,

    wy_evaluations: Vec<CudaSlice<F>>,
    wy_constant_ones: Vec<CudaSlice<F>>,
    bookeeping_g: CudaSlice<F>,
    bookeeping_u: CudaSlice<F>,
    t_at_u: Vec<F>,

    // all last evaluations at bookeeping table at the true index must be the same value.
    assert_w: Vec<bool>,
    sumcheck_prover: MultiProductSumcheckWorkerProver<
        'a,
        F,
        ProductBookeepingTablePlus<F, Engine::RootProductBookeepingTable>,
        S,
        R,
    >,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>, S: SisuSender, R: SisuReceiver>
    GeneralGKRSumcheckPhaseTwoWorkerProver<'a, F, Engine, S, R>
{
    pub fn new(
        engine: &'a Engine,
        worker: Option<&'a WorkerNode<S, R>>,
        circuit: &'a GeneralCircuit<F>,
        layer_index: usize,
        bookeeping_g: CudaSlice<F>,
        random_u: &mut CudaSlice<F>,
        t_at_u: Vec<F>,
        num_replicas: usize,
        wy_evaluations: Vec<CudaSlice<F>>,
    ) -> Self {
        let sumcheck_prover = if worker.is_none() {
            MultiProductSumcheckWorkerProver::default()
        } else {
            MultiProductSumcheckWorkerProver::new(worker.unwrap())
        };

        Self {
            engine,
            num_replicas,
            circuit,
            layer_index,
            bookeeping_g,
            bookeeping_u: Engine::precompute_bookeeping(F::ONE, random_u),
            t_at_u,
            assert_w: vec![],
            sumcheck_prover,
            wy_constant_ones: wy_evaluations
                .iter()
                .map(|wy| engine.cuda_slice_ones(wy.len()))
                .collect(),
            wy_evaluations,
        }
    }

    pub fn add(&mut self, gate_extension: GateExtensionType, assert_w: bool) {
        let mut is_zero = true;
        for ext in gate_extension.to_extensions(self.circuit.layer(self.layer_index)) {
            if ext.evaluations.len() > 0 {
                is_zero = false;
                break;
            }
        }
        if is_zero {
            self.sumcheck_prover.add(None);
            self.assert_w.push(false);
            return;
        }

        let gate_type = self.sumcheck_prover.len();

        let f3 = if gate_type == 0 {
            &self.wy_constant_ones
        } else if gate_type == 1 {
            &self.wy_evaluations
        } else if gate_type == 2 {
            &self.wy_constant_ones
        } else if gate_type == 3 {
            &self.wy_evaluations
        } else {
            panic!("got an invalid gate type {}", gate_type);
        };

        let t_at_u = if self.sumcheck_prover.is_worker() {
            F::ONE
        } else {
            self.t_at_u[gate_type]
        };

        self.sumcheck_prover
            .add(Some(ProductBookeepingTablePlus::new(
                t_at_u,
                self.engine.initialize_phase_2_plus(
                    &self.circuit,
                    self.num_replicas,
                    self.layer_index,
                    gate_extension,
                    &mut self.bookeeping_g,
                    &mut self.bookeeping_u,
                ),
                f3.clone(),
            )));
        self.assert_w.push(assert_w);
    }

    // Run the sumcheck, return the random points, last values (depend on phase) and transcript .
    pub fn run<FS: FiatShamirEngine<F>>(
        &mut self,
        fiat_shamir_engine: &mut FS,
    ) -> (Vec<F>, Vec<Vec<F>>, Option<Vec<F>>, Transcript) {
        let (random_points, last_f, last_g, transcript) =
            self.sumcheck_prover.run(fiat_shamir_engine);

        let mut all_w = vec![];
        let mut final_w = None;
        if self.sumcheck_prover.is_standalone() {
            for (sumcheck_index, (f, g)) in last_f.into_iter().zip(last_g.into_iter()).enumerate() {
                assert_eq!(f.len(), g.len());
                if f.len() == 0 {
                    all_w.push(vec![]);
                    continue;
                }

                // Currently, g == s_at_v.
                if self.assert_w[sumcheck_index] {
                    if final_w.is_none() {
                        final_w = Some(g.clone());
                    } else {
                        assert!(final_w.as_ref().unwrap() == &g);
                    }
                }

                all_w.push(g);
            }
        }

        (random_points, all_w, final_w, transcript)
    }
}

pub struct GeneralGKRSumcheckNextRoundWorkerProver<
    'a,
    F: IcicleConvertibleField,
    BK: RootProductBookeepingTable<F>,
    S: SisuSender,
    R: SisuReceiver,
> {
    sumcheck_prover: MultiProductSumcheckWorkerProver<'a, F, BK, S, R>,
}

impl<
        'a,
        F: IcicleConvertibleField,
        BK: RootProductBookeepingTable<F>,
        S: SisuSender,
        R: SisuReceiver,
    > GeneralGKRSumcheckNextRoundWorkerProver<'a, F, BK, S, R>
{
    pub fn new(worker: Option<&'a WorkerNode<S, R>>) -> Self {
        let sumcheck_prover = if worker.is_none() {
            MultiProductSumcheckWorkerProver::default()
        } else {
            MultiProductSumcheckWorkerProver::new(worker.unwrap())
        };

        Self { sumcheck_prover }
    }

    // Run the sumcheck, return the random points, last values (depend on phase) and transcript .
    pub fn run<FS: FiatShamirEngine<F>>(
        &mut self,
        fiat_shamir_engine: &mut FS,
        bookeeping_q: CudaSlice<F>,
        wx_evaluations: CudaSlice<F>,
    ) -> (Vec<F>, F, Transcript) {
        self.sumcheck_prover
            .add(Some(BK::new(F::ONE, wx_evaluations, bookeeping_q)));

        let (random_points, last_f, _last_g, transcript) =
            self.sumcheck_prover.run(fiat_shamir_engine);

        // At this time, we has only ONE sumcheck in above multi-sumcheck, and
        // this sumcheck has only ONE function too.
        // last_f === last_w_at_g.
        assert_eq!(last_f.len(), 1);
        assert_eq!(last_f[0].len(), 1);

        (random_points, last_f[0][0], transcript)
    }
}

#[inline]
pub fn replica_point(x: usize, num_vars: usize, replica_index: usize) -> usize {
    (replica_index << num_vars) | x
}
