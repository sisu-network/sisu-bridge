use std::{collections::HashMap, ops::Index, slice::Iter};

use ark_ff::{BigInteger, FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain};

use crate::{
    circuit::{Circuit, CircuitGate, CircuitGateType, CircuitLayer, GateF, GateIndex},
    common::round_to_pow_of_two,
    domain_custom_mle::{
        FFTEvaluateForwardXExtension, FFTInterpolateForwardXExtension, FFTShuffleForwardYExtension,
    },
};

pub fn find_root_of_unity<F: Field>(n: usize) -> F {
    assert!(n.is_power_of_two(), "Require n is a power of 2");

    // Calculate p = (MODULUS - 1)/n;
    let mut p = F::BasePrimeField::MODULUS;
    p.sub_with_borrow(&<F::BasePrimeField as PrimeField>::BigInt::from(1u8));
    p.divn(n.ilog2() as u32);

    // Calculate n = n/2;
    let half_n = n as u64 / 2;

    // Find primitive number g.
    let mut g: F;
    let mut x = F::ONE + F::ONE;
    loop {
        g = x.pow(p);
        if g.pow(&[half_n]) != F::ONE {
            break;
        }
        x = x + F::ONE;
    }

    g
}

#[derive(PartialEq, Debug, Clone, Default)]
pub struct RootDomain<F: Field> {
    elements: Box<Vec<F>>,
}

/// Call n is size of domain, the domain has the following properties:
/// - a[0] = 1
/// - a[i] + a[i + n/2] = 0
/// - a[i] * a[n-i] = 1 (any i != 0 and i != n/2).
impl<F: Field> RootDomain<F> {
    // Generate a new multiplicative group size n, n must be a power of two.
    // This will generate all n-th roots of unity in the field.
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "Require n is a power of 2");
        let g = find_root_of_unity::<F>(n);

        // Generate all roots of unity.
        let elements = Box::new((0..n).map(|i| g.pow(&[i as u64])).collect::<Vec<F>>());

        Self { elements }
    }

    pub fn iter(&self) -> Iter<F> {
        self.elements.iter()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Domain<'a, F: Field> {
    root_domain: &'a RootDomain<F>,
    offset: usize,
    step: usize,
    len: usize,
}

impl<'a, F: Field> Domain<'a, F> {
    pub fn precomputed_generators(&self) -> Vec<F> {
        let mut precomputed_domain_generators = vec![];
        for i in 0..self.len().ilog2() {
            precomputed_domain_generators.push(self[2usize.pow(i)]);
        }
        precomputed_domain_generators
    }
}

impl<'a, F: Field> Domain<'a, F> {
    pub fn from(root_domain: &'a RootDomain<F>) -> Self {
        let len = root_domain.elements.len();
        Self {
            root_domain,
            offset: 0,
            step: 1,
            len,
        }
    }

    /// Check if this domain is the subdomain of another.
    pub fn belongs_to(&self, other: &Self) -> bool {
        if self.root_domain == other.root_domain {
            if self.offset == other.offset && self.step >= other.step {
                return true;
            }
        }

        false
    }

    pub fn root_size(&self) -> usize {
        self.root_domain.len()
    }

    pub fn generator(&self) -> F {
        self[1]
    }

    fn iter(&self) -> DomainIter<'a, F> {
        DomainIter::from_ref(&self)
    }
}

impl<F: Field> Index<usize> for Domain<'_, F> {
    type Output = F;
    fn index(&self, index: usize) -> &Self::Output {
        &self.root_domain.elements[self.offset + (index % self.len) * self.step]
    }
}

impl<'a, F: Field> IntoIterator for Domain<'a, F> {
    type Item = &'a F;
    type IntoIter = DomainIter<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        DomainIter::from(self)
    }
}

impl<F: Field> Domain<'_, F> {
    /// Returns the size of domain.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns all elements in this domain.
    pub fn elements(&self) -> Vec<F> {
        self.into_iter().map(|v| v.clone()).collect()
    }

    /// Note that this takes O(N) time to find.
    pub fn find_elements_index(&self, elements: &[F]) -> Vec<usize> {
        let mut index_map = HashMap::new();

        for (i, element) in self.iter().enumerate() {
            for j in 0..elements.len() {
                if element == &elements[j] {
                    index_map.insert(j, i);
                }
            }
        }

        let mut indexes = vec![];
        for j in 0..elements.len() {
            indexes.push(index_map.get(&j).unwrap().clone())
        }

        indexes
    }

    /// Return a new domain which includes square of all elements in the current
    /// domain. The new domain has the half size of current domain.
    pub fn square(&self) -> Self {
        Self {
            root_domain: self.root_domain,
            offset: self.offset,
            step: self.step * 2,
            len: self.len / 2,
        }
    }

    /// Return a new domain which includes sqrt of all elements in the current
    /// domain. The new domain has the double size of current domain.
    pub fn sqrt(&self) -> Self {
        assert!(self.step % 2 == 0);
        Self {
            root_domain: self.root_domain,
            offset: self.offset,
            step: self.step / 2,
            len: self.len * 2,
        }
    }

    /// Given index of number a, return index of number b such that a+b=0.
    pub fn get_opposite_index_of(&self, index: usize) -> usize {
        (index + self.len / 2) % self.len()
    }

    /// Given index of number a, return index of number b such that a^2=b.
    pub fn get_square_index_of(&self, index: usize) -> usize {
        (index * 2) % self.len()
    }

    /// Given index of number a, return index of number b such that a^-1=b.
    pub fn get_inverse_index_of(&self, index: usize) -> usize {
        if index == 0 || index == self.len / 2 {
            index
        } else {
            self.len() - index
        }
    }
}

impl<F: Field + FftField> Domain<'_, F> {
    // Evaluate a polynomial over the current domain by FFT.
    pub fn evaluate(&self, polynomial_coeffs: &[F]) -> Vec<F> {
        let domain = GeneralEvaluationDomain::<F>::new(self.len()).unwrap();
        domain.fft(polynomial_coeffs)
    }

    // Interpolate evaluations over the current domain by IFFT.
    pub fn interpolate(&self, evaluations: Vec<F>) -> Vec<F> {
        let domain = GeneralEvaluationDomain::new(self.len()).unwrap();

        let evaluations = Evaluations::from_vec_and_domain(evaluations, domain);
        evaluations.interpolate().coeffs
    }
}

impl<'a, F: Field> Domain<'a, F> {
    pub fn generate_input_to_circuit(&self, evaluations: &[F]) -> Vec<F> {
        let mut input = vec![];
        input.extend_from_slice(evaluations);
        input.extend(vec![F::ZERO; self.len() - evaluations.len()]);

        input
    }

    pub fn build_short_evaluation_layer(
        &self,
        slice_size: usize,
        num_repititions: usize,
    ) -> CircuitLayer<F> {
        let mut layer = CircuitLayer::new(vec![]);
        // layer
        //     .forward_x_ext
        //     .custom(FFTEvaluateForwardXExtension::new(
        //         self.len(),
        //         num_repititions,
        //     ));

        for k in 0..num_repititions {
            for sign in [true, false] {
                for i in 0..round_to_pow_of_two(slice_size) {
                    let mut accumulate_gates = vec![];
                    for j in 0..self.len() {
                        accumulate_gates.push(CircuitGate::new(
                            "",
                            // +/- (params.d[k] + i)^j
                            CircuitGateType::ForwardX(GateF::DomainPd(sign, k, i, j)),
                            [
                                GateIndex::Absolute(j as usize),
                                GateIndex::Absolute(j as usize),
                            ],
                        ));
                    }

                    layer.add_gate(CircuitGate::new(
                        &format!("output {}", i),
                        CircuitGateType::Accumulation(accumulate_gates),
                        [GateIndex::Dummy, GateIndex::Dummy],
                    ));
                }
            }
        }

        layer
    }

    pub fn build_evaluation_circuit(&self) -> Circuit<F> {
        self.build_fft_circuit(false)
    }

    pub fn build_interpolation_circuit(&'a self, padding_to_domain: &'a Self) -> Circuit<F> {
        let mut circuit = self.build_fft_circuit(true);

        let mut layer = CircuitLayer::default();
        for i in 0..self.len() {
            layer.add_gate(CircuitGate::new(
                &format!("ifft coeffs {}", i),
                CircuitGateType::ForwardY(GateF::ONE),
                [GateIndex::Absolute(i), GateIndex::Absolute(i)],
            ));
        }

        for i in self.len()..padding_to_domain.len() {
            layer.add_gate(CircuitGate::new_dummy(&format!("dummy {}", i)))
        }

        circuit.push_layer(layer);
        circuit
    }

    fn build_fft_circuit(&'a self, inverse: bool) -> Circuit<F> {
        //   INPUT:
        //   [coeffs]
        //
        //   OUTPUT:
        //   [evaluations]

        // We must shuffle the input before run fft.
        let mut circuit = self.build_shuffle_circuit();

        let total_layers = self.len().ilog2();
        for layer_index in 0..total_layers {
            let step_evaluation = 2usize.pow(layer_index);
            let step_domain = 2usize.pow(total_layers - layer_index - 1);
            let mut even = vec![];
            let mut odd = vec![];

            let mut i = 0;
            while i < self.len() {
                for j in 0..step_evaluation {
                    even.push(i + j);
                }
                i += step_evaluation;

                for j in 0..step_evaluation {
                    odd.push(i + j);
                }
                i += step_evaluation;
            }

            let mut layer = CircuitLayer::default();
            // layer
            //     .forward_x_ext
            //     .custom(FFTInterpolateForwardXExtension::new(
            //         &self,
            //         layer_index as usize,
            //     ));

            let mut idx = 0;
            while idx < even.len() {
                // EVEN + ODD * DOMAIN.
                for j in 0..step_evaluation {
                    let mut positive_domain_idx = ((idx + j) * step_domain) % (self.len() / 2);
                    if inverse {
                        positive_domain_idx = self.get_inverse_index_of(positive_domain_idx);
                    }

                    layer.add_gate(CircuitGate::new(
                        "",
                        // &format!(
                        // "{} + {}*{:?}",
                        // even[idx + j],
                        // odd[idx + j],
                        // self.index(positive_domain_idx)
                        // ),
                        CircuitGateType::Fourier(GateF::C(self.index(positive_domain_idx).clone())),
                        [
                            GateIndex::Absolute(even[idx + j]),
                            GateIndex::Absolute(odd[idx + j]),
                        ],
                    ));
                }

                // EVEN - ODD * DOMAIN.
                for j in 0..step_evaluation {
                    let mut positive_domain_idx = ((idx + j) * step_domain) % (self.len() / 2);
                    if inverse {
                        positive_domain_idx = self.get_inverse_index_of(positive_domain_idx);
                    }
                    let negative_domain_idx = self.get_opposite_index_of(positive_domain_idx);

                    layer.add_gate(CircuitGate::new(
                        "", // &format!(
                        // "{} + {}*{:?}",
                        // even[idx + j],
                        // odd[idx + j],
                        //self.index(negative_domain_idx)
                        //),
                        CircuitGateType::Fourier(GateF::C(self.index(negative_domain_idx).clone())),
                        [
                            GateIndex::Absolute(even[idx + j]),
                            GateIndex::Absolute(odd[idx + j]),
                        ],
                    ));
                }

                idx += step_evaluation;
            }

            circuit.push_layer(layer);
        }

        println!("========================================================");

        circuit
    }

    fn build_shuffle_circuit(&'a self) -> Circuit<F> {
        // Swap monomial evaluations. Number of layers = 1.
        // Circuit input : [evaluations (n)]
        // Circuit output: [evaluations after shuffle (n)]
        let evaluations_size = self.len();

        let mut circuit = Circuit::new(evaluations_size);

        let mut new_index = vec![0; evaluations_size];
        for i in 0..evaluations_size {
            new_index[i] = i;
        }

        // We need the new index after swapping.
        let mut j = 0;
        for i in 1..evaluations_size {
            let mut bit = evaluations_size >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }

            j ^= bit;

            if i < j {
                (new_index[i], new_index[j]) = (new_index[j], new_index[i]);
            }
        }

        // Add gates after calculating new index.
        let mut layer = CircuitLayer::default();
        // layer
        //     .forward_y_ext
        //     .custom(FFTShuffleForwardYExtension::new(self.len().ilog2() as usize));

        for index in new_index {
            layer.add_gate(CircuitGate::new(
                "", //  &format!("evaluations {}", index),
                CircuitGateType::ForwardY(GateF::ONE),
                [GateIndex::Absolute(index), GateIndex::Absolute(index)],
            ));
        }

        circuit.push_layer(layer);

        circuit
    }
}

pub struct DomainIter<'a, F: Field> {
    domain: Domain<'a, F>,
    current: usize,
}

impl<'a, F: Field> DomainIter<'a, F> {
    pub fn from(domain: Domain<'a, F>) -> Self {
        Self {
            domain,
            current: domain.offset,
        }
    }

    pub fn from_ref(domain: &Domain<'a, F>) -> Self {
        Self {
            domain: domain.clone(),
            current: domain.offset,
        }
    }
}

impl<'a, F: Field> Iterator for DomainIter<'a, F> {
    type Item = &'a F;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.domain.root_domain.elements.len() {
            None
        } else {
            let index = self.current;
            self.current += self.domain.step;
            Some(&self.domain.root_domain.elements[index])
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circuit::CircuitParams,
        domain::{Domain, RootDomain},
        field::{Fp337, Fp97},
    };

    use ark_poly::{univariate, DenseUVPolynomial, Polynomial};

    #[test]
    fn test_fft_with_subdomain() {
        let expected_polynomial = vec![
            Fp337::from(3),
            Fp337::from(1),
            Fp337::from(4),
            Fp337::from(1),
            Fp337::from(5),
            Fp337::from(9),
            Fp337::from(2),
            Fp337::from(6),
        ];

        let root_domain = RootDomain::new(expected_polynomial.len());
        let domain = Domain::from(&root_domain);

        let poly = univariate::DensePolynomial::from_coefficients_slice(&expected_polynomial);
        let expected_evaluations: Vec<Fp337> = domain
            .elements()
            .iter()
            .map(|p| poly.evaluate(&p))
            .collect();

        let evaluations = domain.evaluate(&expected_polynomial);

        assert_eq!(evaluations, expected_evaluations);
    }

    #[test]
    fn test_fft_97() {
        let expected_polynomial = vec![
            Fp97::from(3),
            Fp97::from(1),
            Fp97::from(4),
            Fp97::from(1),
            Fp97::from(5),
            Fp97::from(9),
            Fp97::from(2),
            Fp97::from(6),
        ];

        let root_domain = RootDomain::new(expected_polynomial.len());
        let domain = Domain::from(&root_domain);
        println!("DOMAIN: {:?}", domain.elements());

        let poly = univariate::DensePolynomial::from_coefficients_slice(&expected_polynomial);
        let expected_evaluations: Vec<Fp97> = domain
            .elements()
            .iter()
            .map(|p| poly.evaluate(&p))
            .collect();

        let evaluations = domain.evaluate(&expected_polynomial);
        let interpolated_polynomials = domain.interpolate(evaluations.clone());

        assert_eq!(evaluations, expected_evaluations);
        println!("EVALUATIONS: {:?}", expected_evaluations);
        assert_eq!(interpolated_polynomials, expected_polynomial);
    }

    #[test]
    fn test_fft_97_with_subdomain() {
        let expected_polynomial = vec![
            Fp97::from(3),
            Fp97::from(1),
            Fp97::from(4),
            Fp97::from(1),
            Fp97::from(5),
            Fp97::from(9),
            Fp97::from(2),
            Fp97::from(6),
        ];

        let root_domain = RootDomain::new(2 * expected_polynomial.len());
        let domain = Domain::from(&root_domain);

        println!("DOMAIN: {:?}", domain.elements());

        let poly = univariate::DensePolynomial::from_coefficients_slice(&expected_polynomial);
        let expected_evaluations: Vec<Fp97> = domain
            .elements()
            .iter()
            .map(|p| poly.evaluate(&p))
            .collect();

        let evaluations = domain.evaluate(&expected_polynomial);

        assert_eq!(evaluations, expected_evaluations);
    }

    #[test]
    fn test_fft_circuit() {
        let expected_polynomial = vec![
            Fp337::from(3),
            Fp337::from(1),
            Fp337::from(4),
            Fp337::from(1),
            Fp337::from(5),
            Fp337::from(9),
            Fp337::from(2),
            Fp337::from(6),
        ];

        let mut expected_evaluations = vec![
            Fp337::from(31),
            Fp337::from(70),
            Fp337::from(109),
            Fp337::from(74),
            Fp337::from(334),
            Fp337::from(181),
            Fp337::from(232),
            Fp337::from(4),
        ];

        let root_domain = RootDomain::new(8);
        let main_domain = Domain::from(&root_domain);

        let fft_circuit = main_domain.build_evaluation_circuit();
        let ifft_circuit = main_domain.build_interpolation_circuit(&main_domain);

        let evaluations = fft_circuit.evaluate(
            &CircuitParams::default(),
            &main_domain.generate_input_to_circuit(&expected_polynomial),
        );
        assert_eq!(evaluations.at_layer(0, false)[..8], expected_evaluations);

        for i in 0..expected_evaluations.len() {
            expected_evaluations[i] *= Fp337::from(295); // divide evaluations by 8.
        }

        let evaluations = ifft_circuit.evaluate(
            &CircuitParams::default(),
            &main_domain.generate_input_to_circuit(&expected_evaluations),
        );
        assert_eq!(evaluations.at_layer(0, false)[..8], expected_polynomial);
    }

    #[test]
    fn test_fft_circuit_with_subdomain() {
        let expected_polynomial = vec![
            Fp337::from(3),
            Fp337::from(1),
            Fp337::from(4),
            Fp337::from(1),
            Fp337::from(5),
            Fp337::from(9),
            Fp337::from(2),
            Fp337::from(6),
        ];

        let expected_evaluations = vec![
            Fp337::from(31),
            Fp337::from(70),
            Fp337::from(109),
            Fp337::from(74),
            Fp337::from(334),
            Fp337::from(181),
            Fp337::from(232),
            Fp337::from(4),
        ];

        let root_domain = RootDomain::new(16);
        let main_domain = Domain::from(&root_domain);
        let fft_domain = main_domain.square();

        let ifft_circuit = fft_domain.build_interpolation_circuit(&main_domain);
        let fft_circuit = main_domain.build_evaluation_circuit();

        let mut ifft_input = expected_evaluations.clone();
        for i in 0..ifft_input.len() {
            ifft_input[i] *= Fp337::from(295); // divide evaluations by 8.
        }

        let evaluations = ifft_circuit.evaluate(
            &CircuitParams::default(),
            &fft_domain.generate_input_to_circuit(&ifft_input),
        );
        assert_eq!(evaluations.at_layer(0, false)[..8], expected_polynomial);

        let evaluations = fft_circuit.evaluate(
            &CircuitParams::default(),
            &main_domain.generate_input_to_circuit(&expected_polynomial),
        );
        assert_eq!(
            evaluations
                .at_layer(0, false)
                .iter()
                .step_by(2)
                .take(8)
                .map(|v| *v)
                .collect::<Vec<Fp337>>(),
            expected_evaluations
        );
    }

    #[test]
    fn test_fft_circuit_97() {
        let expected_polynomial = vec![
            Fp97::from(3),
            Fp97::from(1),
            Fp97::from(4),
            Fp97::from(1),
            Fp97::from(5),
            Fp97::from(9),
            Fp97::from(2),
            Fp97::from(6),
        ];

        let mut expected_evaluations = vec![
            Fp97::from(31),
            Fp97::from(56),
            Fp97::from(68),
            Fp97::from(10),
            Fp97::from(94),
            Fp97::from(28),
            Fp97::from(33),
            Fp97::from(92),
        ];

        let root_domain = RootDomain::new(8);
        let main_domain = Domain::from(&root_domain);

        let fft_circuit = main_domain.build_evaluation_circuit();
        let ifft_circuit = main_domain.build_interpolation_circuit(&main_domain);

        let evaluations = fft_circuit.evaluate(
            &CircuitParams::default(),
            &main_domain.generate_input_to_circuit(&expected_polynomial),
        );
        assert_eq!(evaluations.at_layer(0, false)[..8], expected_evaluations);

        for i in 0..expected_evaluations.len() {
            expected_evaluations[i] *= Fp97::from(85); // divide evaluations by 8.
        }

        let evaluations = ifft_circuit.evaluate(
            &CircuitParams::default(),
            &main_domain.generate_input_to_circuit(&expected_evaluations),
        );
        assert_eq!(evaluations.at_layer(0, false)[..8], expected_polynomial);
    }
}
