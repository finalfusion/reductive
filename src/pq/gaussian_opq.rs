use std::iter::Sum;

use lax::Lapack;
use ndarray::{ArrayBase, Data, Ix2, NdFloat};
use ndarray_linalg::types::Scalar;
use num_traits::AsPrimitive;
use rand::{CryptoRng, RngCore, SeedableRng};

use super::{Pq, TrainPq, OPQ};

/// Optimized product quantizer for Gaussian variables (Ge et al., 2013).
///
/// A product quantizer is a vector quantizer that slices a vector and
/// assigns to the *i*-th slice the index of the nearest centroid of the
/// *i*-th subquantizer. Vector reconstruction consists of concatenating
/// the centroids that represent the slices.
///
/// This quantizer learns a orthonormal matrix that rotates the input
/// space in order to balance variances over subquantizers. The
/// optimization procedure assumes that the variables have a Gaussian
/// distribution. The `OPQ` quantizer provides a non-parametric,
/// albeit slower to train implementation of optimized product
/// quantization.
pub struct GaussianOPQ;

impl<A> TrainPq<A> for GaussianOPQ
where
    A: Lapack + NdFloat + Scalar + Sum,
    A::Real: NdFloat,
    usize: AsPrimitive<A>,
{
    fn train_pq_using<S, R>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
        rng: &mut R,
    ) -> Result<Pq<A>, rand::Error>
    where
        S: Sync + Data<Elem = A>,
        R: CryptoRng + RngCore + SeedableRng + Send,
    {
        Pq::check_quantizer_invariants(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            instances.view(),
        );

        let projection = OPQ::create_projection_matrix(instances.view(), n_subquantizers);
        let rx = instances.dot(&projection);
        let pq = Pq::train_pq_using(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            rx,
            rng,
        )?;

        Ok(Pq {
            projection: Some(projection),
            quantizers: pq.quantizers,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ArrayView2};
    use rand::distributions::Uniform;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::GaussianOPQ;
    use crate::linalg::EuclideanDistance;
    use crate::ndarray_rand::RandomExt;
    use crate::pq::{Pq, QuantizeVector, ReconstructVector, TrainPq};

    /// Calculate the average euclidean distances between the the given
    /// instances and the instances returned by quantizing and then
    /// reconstructing the instances.
    fn avg_euclidean_loss(instances: ArrayView2<f32>, quantizer: &Pq<f32>) -> f32 {
        let mut euclidean_loss = 0f32;

        let quantized: Array2<u8> = quantizer.quantize_batch(instances);
        let reconstructions = quantizer.reconstruct_batch(quantized);

        for (instance, reconstruction) in instances.outer_iter().zip(reconstructions.outer_iter()) {
            euclidean_loss += instance.euclidean_distance(reconstruction);
        }

        euclidean_loss / instances.nrows() as f32
    }

    #[test]
    fn quantize_with_gaussian_opq() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let uniform = Uniform::new(0f32, 1f32);
        let instances = Array2::random_using((256, 20), uniform, &mut rng);
        let pq = GaussianOPQ::train_pq_using(10, 7, 10, 1, instances.view(), &mut rng).unwrap();
        let loss = avg_euclidean_loss(instances.view(), &pq);
        // Loss is around 0.1.
        assert!(loss < 0.12);
    }
}
