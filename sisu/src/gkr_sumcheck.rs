use sisulib::{
    circuit::{CircuitParams, GateEvaluations},
    common::split_number,
};

use crate::{
    channel::NoChannel,
    cuda_compat::slice::CudaSlice,
    fiat_shamir::{FiatShamirEngine, Transcript},
    icicle_converter::IcicleConvertibleField,
    polynomial::RootProductBookeepingTable,
    sisu_engine::SisuEngine,
    sumcheck::MultiProductSumcheckWorkerProver,
};

pub struct GKRSumcheckProver<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> {
    phase: usize,
    circuit_params: &'a CircuitParams<'a, F>,

    f2_at_u: Vec<F>,
    wx_evaluations: CudaSlice<F>,
    wy_evaluations: CudaSlice<F>,
    wx_constant_one: CudaSlice<F>,
    wy_constant_one: CudaSlice<F>,
    bookeeping_g: CudaSlice<F>,
    bookeeping_u: CudaSlice<F>,

    // all last evaluations at bookeeping table at the true index must be the same value.
    assert_w: Vec<bool>,
    sumcheck_prover: MultiProductSumcheckWorkerProver<
        'a,
        F,
        Engine::RootProductBookeepingTable,
        NoChannel,
        NoChannel,
    >,
}

impl<'a, F: IcicleConvertibleField, Engine: SisuEngine<F>> GKRSumcheckProver<'a, F, Engine> {
    pub fn new(
        engine: &Engine,
        circuit_params: &'a CircuitParams<F>,
        bookeeping_g: CudaSlice<F>,
        w_evaluations: CudaSlice<F>,
    ) -> Self {
        Self {
            phase: 0,
            wx_constant_one: engine.cuda_slice_ones(w_evaluations.len()),
            wy_constant_one: engine.cuda_slice_ones(w_evaluations.len()),
            wx_evaluations: w_evaluations.clone(),
            wy_evaluations: w_evaluations,
            circuit_params,
            bookeeping_g,
            bookeeping_u: CudaSlice::on_host(vec![]),
            f2_at_u: vec![],
            sumcheck_prover: MultiProductSumcheckWorkerProver::default(),
            assert_w: vec![],
        }
    }

    pub fn reset(&mut self) {
        self.sumcheck_prover = MultiProductSumcheckWorkerProver::default();
        self.assert_w = vec![];
    }

    pub fn init_phase_1(&mut self) {
        self.phase = 1;
    }

    pub fn init_phase_2(&mut self, random_u: &mut CudaSlice<F>, f2_at_u: Vec<F>) {
        self.reset();
        assert!(f2_at_u.len() == 4); // const - mul - forward_x - forward_y.

        self.phase = 2;
        self.bookeeping_u = Engine::precompute_bookeeping(F::ONE, random_u);
        self.f2_at_u = f2_at_u;
    }

    fn add_phase_1(&mut self, gate_evaluations: &GateEvaluations<F>, assert_last_value: bool) {
        assert!(self.phase == 1);

        if gate_evaluations.len() == 0 {
            self.sumcheck_prover.add(None);
            self.assert_w.push(false);
            return;
        }

        let gate_type = self.sumcheck_prover.len();

        let (f2, f3) = if gate_type == 0 {
            (&mut self.wx_constant_one, &mut self.wy_constant_one)
        } else if gate_type == 1 {
            (&mut self.wx_evaluations, &mut self.wy_evaluations)
        } else if gate_type == 2 {
            (&mut self.wx_evaluations, &mut self.wy_constant_one)
        } else if gate_type == 3 {
            (&mut self.wx_constant_one, &mut self.wx_evaluations)
        } else {
            panic!("got an invalid gate type {}", gate_type);
        };

        self.sumcheck_prover
            .add(Some(Engine::RootProductBookeepingTable::new(
                F::ONE,
                f2.clone(),
                initialize_phase_1(
                    &self.circuit_params,
                    gate_evaluations,
                    f3,
                    &mut self.bookeeping_g,
                ),
            )));
        self.assert_w.push(assert_last_value);
    }

    fn add_phase_2(&mut self, gate_evaluations: &GateEvaluations<F>, assert_w: bool) {
        assert!(self.phase == 2);

        if gate_evaluations.len() == 0 {
            self.sumcheck_prover.add(None);
            self.assert_w.push(false);
            return;
        }

        let gate_type = self.sumcheck_prover.len();

        let f3 = if gate_type == 0 {
            &self.wx_constant_one
        } else if gate_type == 1 {
            &self.wx_evaluations
        } else if gate_type == 2 {
            &self.wx_constant_one
        } else if gate_type == 3 {
            &self.wx_evaluations
        } else {
            panic!("got an invalid gate type {}", gate_type);
        };

        self.sumcheck_prover
            .add(Some(Engine::RootProductBookeepingTable::new(
                self.f2_at_u[gate_type],
                initialize_phase_2(
                    &self.circuit_params,
                    gate_evaluations,
                    &mut self.bookeeping_g,
                    &mut self.bookeeping_u,
                ),
                f3.clone(),
            )));
        self.assert_w.push(assert_w);
    }

    pub fn add(&mut self, gate_evaluations: &GateEvaluations<F>, assert_w: bool) {
        match self.phase {
            0 => panic!("not yet init phase"),
            1 => self.add_phase_1(gate_evaluations, assert_w),
            2 => self.add_phase_2(gate_evaluations, assert_w),
            _ => panic!("invalid phase"),
        }
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
        for (sumcheck_index, (f, g)) in last_f.into_iter().zip(last_g.into_iter()).enumerate() {
            assert_eq!(f.len(), g.len());
            if f.len() == 0 {
                all_w.push(F::ZERO);
                continue;
            }

            assert_eq!(f.len(), 1);
            let f = f[0];
            let g = g[0];

            let w = if self.phase == 1 {
                f // f2_at_u
            } else {
                g // f3_at_v
            };

            if self.assert_w[sumcheck_index] {
                if final_w.is_none() {
                    final_w = Some(w);
                } else {
                    assert!(final_w.unwrap() == w);
                }
            }

            all_w.push(w);
        }

        (random_points, all_w, final_w, transcript)
    }
}

pub fn precompute_bookeeping_with_linear_combination<
    F: IcicleConvertibleField,
    Engine: SisuEngine<F>,
>(
    alpha: F,
    g1: &mut CudaSlice<F>,
    beta: F,
    g2: &mut CudaSlice<F>,
) -> CudaSlice<F> {
    let mut bookeeping_g1 = Engine::precompute_bookeeping(alpha, g1).as_host();
    let bookeeping_g2 = Engine::precompute_bookeeping(beta, g2).as_host();

    for i in 0..bookeeping_g1.len() {
        bookeeping_g1[i] = bookeeping_g1[i] + bookeeping_g2[i];
    }

    CudaSlice::on_host(bookeeping_g1)
}

// Evaluate all values of h(x) = f1(g, x, y) * f3(y) on boolean hypercube.
// Where f1 is a sparse multilinear extension. f3 is multilinear extension.
fn initialize_phase_1<F: IcicleConvertibleField>(
    circuit_params: &CircuitParams<F>,
    f1: &GateEvaluations<F>,
    f3: &mut CudaSlice<F>,
    bookeeping_g: &mut CudaSlice<F>,
) -> CudaSlice<F> {
    let bookeeping_g = bookeeping_g.as_ref_host();
    let f3_is_one = f3.is_one();
    let f3 = f3.as_ref_host();

    let mut bookeeping_h = vec![F::ZERO; f3.len()];
    let f3_num_vars = f3.len().ilog2() as usize;

    if f3_is_one {
        if f1.is_major_ones() {
            for (point, evaluation_f1_at_this_point) in f1.compile(circuit_params) {
                // The point is in form (z, x, y).
                let (zx, _) = split_number(&point, f3_num_vars);
                let (z, x) = split_number(&zx, f3_num_vars);

                if evaluation_f1_at_this_point == F::ONE {
                    bookeeping_h[x] += bookeeping_g[z];
                } else {
                    bookeeping_h[x] += bookeeping_g[z] * evaluation_f1_at_this_point;
                };
            }
        } else {
            for (point, evaluation_f1_at_this_point) in f1.compile(circuit_params) {
                // The point is in form (z, x, y).
                let (zx, _) = split_number(&point, f3_num_vars);
                let (z, x) = split_number(&zx, f3_num_vars);

                bookeeping_h[x] += bookeeping_g[z] * evaluation_f1_at_this_point;
            }
        }
    } else {
        if f1.is_major_ones() {
            for (point, evaluation_f1_at_this_point) in f1.compile(circuit_params) {
                // The point is in form (z, x, y).
                let (zx, y) = split_number(&point, f3_num_vars);
                let (z, x) = split_number(&zx, f3_num_vars);

                if evaluation_f1_at_this_point == F::ONE {
                    bookeeping_h[x] += bookeeping_g[z] * f3[y];
                } else {
                    bookeeping_h[x] += bookeeping_g[z] * evaluation_f1_at_this_point * f3[y];
                };
            }
        } else {
            for (point, evaluation_f1_at_this_point) in f1.compile(circuit_params) {
                // The point is in form (z, x, y).
                let (zx, y) = split_number(&point, f3_num_vars);
                let (z, x) = split_number(&zx, f3_num_vars);

                bookeeping_h[x] += bookeeping_g[z] * evaluation_f1_at_this_point * f3[y];
            }
        }
    }

    CudaSlice::on_host(bookeeping_h)
}

// Evaluate all values of f1(g, u, y) on boolean hypercube. Where f1 is a sparse
// multilinear extension.
fn initialize_phase_2<F: IcicleConvertibleField>(
    circuit_params: &CircuitParams<F>,
    f1: &GateEvaluations<F>,
    bookeeping_g: &mut CudaSlice<F>,
    bookeeping_u: &mut CudaSlice<F>,
) -> CudaSlice<F> {
    let bookeeping_g = bookeeping_g.as_ref_host();
    let bookeeping_u = bookeeping_u.as_ref_host();

    let f2_num_vars = bookeeping_u.len().ilog2() as usize;
    let mut bookeeping_f1 = vec![F::ZERO; bookeeping_u.len()];

    if f1.is_major_ones() {
        for (point, evaluation_f1_at_this_point) in f1.compile(circuit_params) {
            // The point is in form (z, x, y).
            let (zx, y) = split_number(&point, f2_num_vars);
            let (z, x) = split_number(&zx, f2_num_vars);

            if evaluation_f1_at_this_point == F::ONE {
                bookeeping_f1[y] += bookeeping_g[z] * bookeeping_u[x];
            } else {
                bookeeping_f1[y] += bookeeping_g[z] * bookeeping_u[x] * evaluation_f1_at_this_point;
            }
        }
    } else {
        for (point, evaluation_f1_at_this_point) in f1.compile(circuit_params) {
            // The point is in form (z, x, y).
            let (zx, y) = split_number(&point, f2_num_vars);
            let (z, x) = split_number(&zx, f2_num_vars);

            bookeeping_f1[y] += bookeeping_g[z] * bookeeping_u[x] * evaluation_f1_at_this_point;
        }
    }

    CudaSlice::on_host(bookeeping_f1)
}
