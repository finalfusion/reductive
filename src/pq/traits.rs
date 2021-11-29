use std::iter::Sum;

use ndarray::{Array1, Array2, ArrayBase, ArrayViewMut1, ArrayViewMut2, Data, Ix1, Ix2, NdFloat};
use num_traits::{AsPrimitive, Bounded, Zero};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::pq::Pq;

/// Training triat for product quantizers.
///
/// This traits specifies the training functions for product
/// quantizers.
pub trait TrainPq<A> {
    /// Train a product quantizer with the xorshift PRNG.
    ///
    /// Train a product quantizer with `n_subquantizers` subquantizers
    /// on `instances`. Each subquantizer has 2^`quantizer_bits`
    /// centroids.  The subquantizers are trained with `n_iterations`
    /// k-means iterations. Each subquantizer is trained `n_attempts`
    /// times, where the best clustering is used.
    fn train_pq<S>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
    ) -> Result<Pq<A>, rand::Error>
    where
        S: Sync + Data<Elem = A>,
    {
        Self::train_pq_using(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            instances,
            &mut ChaCha8Rng::from_entropy(),
        )
    }

    /// Train a product quantizer.
    ///
    /// Train a product quantizer with `n_subquantizers` subquantizers
    /// on `instances`. Each subquantizer has 2^`quantizer_bits`
    /// centroids.  The subquantizers are trained with `n_iterations`
    /// k-means iterations. Each subquantizer is trained `n_attempts`
    /// times, where the best clustering is used.
    ///
    /// `rng` is used for picking the initial cluster centroids of
    /// each subquantizer. Multiple RNGs are seeded from this RNG,
    /// so we require a cryptographic RNG to avoid correlations between
    /// the seeded RNGs.
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
        R: CryptoRng + RngCore + SeedableRng + Send;
}

/// Vector quantization.
pub trait QuantizeVector<A> {
    /// Quantize a batch of vectors.
    fn quantize_batch<I, S>(&self, x: ArrayBase<S, Ix2>) -> Array2<I>
    where
        I: AsPrimitive<usize> + Bounded + Zero,
        S: Data<Elem = A>,
        usize: AsPrimitive<I>;

    /// Quantize a batch of vectors into an existing matrix.
    fn quantize_batch_into<I, S>(&self, x: ArrayBase<S, Ix2>, quantized: ArrayViewMut2<I>)
    where
        I: AsPrimitive<usize> + Bounded + Zero,
        S: Data<Elem = A>,
        usize: AsPrimitive<I>;

    /// Quantize a vector.
    fn quantize_vector<I, S>(&self, x: ArrayBase<S, Ix1>) -> Array1<I>
    where
        I: AsPrimitive<usize> + Bounded + Zero,
        S: Data<Elem = A>,
        usize: AsPrimitive<I>;

    /// Get the length of a vector after quantization.
    fn quantized_len(&self) -> usize;
}

/// Vector reconstruction.
pub trait ReconstructVector<A>
where
    A: NdFloat + Sum,
{
    /// Reconstruct a batch of vectors.
    ///
    /// The vectors are reconstructed from the quantization indices.
    fn reconstruct_batch<I, S>(&self, quantized: ArrayBase<S, Ix2>) -> Array2<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        let mut reconstructions = Array2::zeros((quantized.nrows(), self.reconstructed_len()));
        self.reconstruct_batch_into(quantized, reconstructions.view_mut());
        reconstructions
    }

    /// Reconstruct a batch of vectors into an existing matrix.
    ///
    /// The vectors are reconstructed from the quantization indices.
    fn reconstruct_batch_into<I, S>(
        &self,
        quantized: ArrayBase<S, Ix2>,
        reconstructions: ArrayViewMut2<A>,
    ) where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>;

    /// Reconstruct a vectors.
    ///
    /// The vector is reconstructed from the quantization indices.
    fn reconstruct_vector<I, S>(&self, quantized: ArrayBase<S, Ix1>) -> Array1<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        let mut reconstruction = Array1::zeros((self.reconstructed_len(),));
        self.reconstruct_vector_into(quantized, reconstruction.view_mut());
        reconstruction
    }

    /// Reconstruct a vector into an existing vector.
    ///
    /// The vector is reconstructed from the quantization indices.
    fn reconstruct_vector_into<I, S>(
        &self,
        quantized: ArrayBase<S, Ix1>,
        reconstruction: ArrayViewMut1<A>,
    ) where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>;

    /// Get the length of a vector after reconstruction.
    fn reconstructed_len(&self) -> usize;
}
