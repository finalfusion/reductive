use std::iter;
use std::iter::Sum;

use log::info;
use ndarray::{
    concatenate, s, Array1, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, ArrayViewMut1,
    ArrayViewMut2, Axis, Data, Ix1, Ix2, NdFloat,
};
use num_traits::{AsPrimitive, Bounded, Zero};
use ordered_float::OrderedFloat;
use rand::{Rng, RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

use super::primitives;
use super::{QuantizeVector, Reconstruct, TrainPq};
use crate::error::ReductiveError;
use crate::kmeans::{
    InitialCentroids, KMeansWithCentroids, NIterationsCondition, RandomInstanceCentroids,
};

/// Product quantizer (Jégou et al., 2011).
///
/// A product quantizer is a vector quantizer that slices a vector and
/// assigns to the *i*-th slice the index of the nearest centroid of the
/// *i*-th subquantizer. Vector reconstruction consists of concatenating
/// the centroids that represent the slices.
#[derive(Clone, Debug, PartialEq)]
pub struct Pq<A> {
    pub(crate) projection: Option<Array2<A>>,
    pub(crate) quantizers: Array3<A>,
}

impl<A> Pq<A>
where
    A: NdFloat,
{
    pub fn new(projection: Option<Array2<A>>, quantizers: Array3<A>) -> Self {
        assert!(
            !quantizers.is_empty(),
            "Attempted to construct a product quantizer without quantizers."
        );

        let reconstructed_len = primitives::reconstructed_len(quantizers.view());

        if let Some(ref projection) = projection {
            assert_eq!(
                projection.shape(),
                [reconstructed_len; 2],
                "Incorrect projection matrix shape, was: {:?}, should be [{}, {}]",
                projection.shape(),
                reconstructed_len,
                reconstructed_len
            );
        }

        Pq {
            projection,
            quantizers,
        }
    }

    pub(crate) fn check_quantizer_invariants(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayView2<A>,
    ) -> Result<(), ReductiveError> {
        if n_subquantizers == 0 || n_subquantizers > instances.ncols() {
            return Err(ReductiveError::NSubquantizersOutsideRange {
                n_subquantizers,
                max_subquantizers: instances.ncols(),
            });
        }

        let max_subquantizer_bits = (instances.nrows() as f64).log2().trunc() as u32;
        if n_subquantizer_bits == 0 || n_subquantizer_bits > max_subquantizer_bits {
            return Err(ReductiveError::IncorrectNSubquantizerBits {
                max_subquantizer_bits,
            });
        }

        if instances.ncols() % n_subquantizers != 0 {
            return Err(ReductiveError::IncorrectNumberSubquantizers {
                n_subquantizers,
                n_columns: instances.ncols(),
            });
        }

        if n_iterations == 0 {
            return Err(ReductiveError::IncorrectNIterations);
        }

        if n_attempts == 0 {
            return Err(ReductiveError::IncorrectNAttempts);
        }

        Ok(())
    }

    /// Get the number of centroids per quantizer.
    pub fn n_quantizer_centroids(&self) -> usize {
        self.quantizers.len_of(Axis(1))
    }

    /// Get the projection matrix (if used).
    pub fn projection(&self) -> Option<ArrayView2<A>> {
        self.projection.as_ref().map(Array2::view)
    }

    /// Create initial centroids for a single quantizer.
    ///
    /// `subquantizer_idx` is the subquantizer index for which the initial
    /// centroids should be picked. `subquantizer_idx < n_subquantizers`,
    /// the total number of subquantizers.
    pub(crate) fn subquantizer_initial_centroids<S>(
        subquantizer_idx: usize,
        n_subquantizers: usize,
        codebook_len: usize,
        instances: ArrayBase<S, Ix2>,
        rng: &mut impl Rng,
    ) -> Array2<A>
    where
        S: Data<Elem = A>,
    {
        let sq_dims = instances.ncols() / n_subquantizers;

        let mut random_centroids = RandomInstanceCentroids::new(rng);

        let offset = subquantizer_idx * sq_dims;
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sq_instances = instances.slice(s![.., offset..offset + sq_dims]);
        random_centroids.initial_centroids(sq_instances, Axis(0), codebook_len)
    }

    /// Train a subquantizer.
    ///
    /// `subquantizer_idx` is the index of the subquantizer, where
    /// `subquantizer_idx < n_subquantizers`, the overall number of
    /// subquantizers. `codebook_len` is the code book size of the
    /// quantizer.
    fn train_subquantizer(
        subquantizer_idx: usize,
        n_subquantizers: usize,
        codebook_len: usize,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayView2<A>,
        mut rng: impl Rng,
    ) -> Array2<A>
    where
        A: Sum,
        usize: AsPrimitive<A>,
    {
        assert!(n_attempts > 0, "Cannot train a subquantizer in 0 attempts.");

        info!("Training PQ subquantizer {}", subquantizer_idx);

        let sq_dims = instances.ncols() / n_subquantizers;

        let offset = subquantizer_idx * sq_dims;
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sq_instances = instances.slice(s![.., offset..offset + sq_dims]);

        iter::repeat_with(|| {
            let mut quantizer = Pq::subquantizer_initial_centroids(
                subquantizer_idx,
                n_subquantizers,
                codebook_len,
                instances,
                &mut rng,
            );
            let loss = sq_instances.kmeans_with_centroids(
                Axis(0),
                quantizer.view_mut(),
                NIterationsCondition(n_iterations),
            );
            (loss, quantizer)
        })
        .take(n_attempts)
        .map(|(loss, quantizer)| (OrderedFloat(loss), quantizer))
        .min_by_key(|attempt| attempt.0)
        .unwrap()
        .1
    }

    /// Get the subquantizer centroids.
    pub fn subquantizers(&self) -> ArrayView3<A> {
        self.quantizers.view()
    }
}

impl<A> TrainPq<A> for Pq<A>
where
    A: NdFloat + Sum,
    usize: AsPrimitive<A>,
{
    fn train_pq_using<S, R>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
        mut rng: &mut R,
    ) -> Result<Pq<A>, ReductiveError>
    where
        S: Sync + Data<Elem = A>,
        R: RngCore + SeedableRng + Send,
    {
        Self::check_quantizer_invariants(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            instances.view(),
        )?;

        let rngs = iter::repeat_with(|| XorShiftRng::from_rng(&mut rng))
            .take(n_subquantizers)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ReductiveError::ConstructRng)?;

        let quantizers = rngs
            .into_par_iter()
            .enumerate()
            .map(|(idx, rng)| {
                Self::train_subquantizer(
                    idx,
                    n_subquantizers,
                    2usize.pow(n_subquantizer_bits),
                    n_iterations,
                    n_attempts,
                    instances.view(),
                    rng,
                )
                .insert_axis(Axis(0))
            })
            .collect::<Vec<_>>();

        let views = quantizers.iter().map(|a| a.view()).collect::<Vec<_>>();

        Ok(Pq {
            projection: None,
            quantizers: concatenate(Axis(0), &views).expect("Cannot concatenate subquantizers"),
        })
    }
}

impl<A> QuantizeVector<A> for Pq<A>
where
    A: NdFloat + Sum,
{
    fn quantize_batch<I, S>(&self, x: ArrayBase<S, Ix2>) -> Array2<I>
    where
        I: AsPrimitive<usize> + Bounded + Zero,
        S: Data<Elem = A>,
        usize: AsPrimitive<I>,
    {
        let mut quantized = Array2::zeros((x.nrows(), self.quantized_len()));
        self.quantize_batch_into(x, quantized.view_mut());
        quantized
    }

    /// Quantize a batch of vectors into an existing matrix.
    fn quantize_batch_into<I, S>(&self, x: ArrayBase<S, Ix2>, mut quantized: ArrayViewMut2<I>)
    where
        I: AsPrimitive<usize> + Bounded + Zero,
        S: Data<Elem = A>,
        usize: AsPrimitive<I>,
    {
        match self.projection {
            Some(ref projection) => {
                let rx = x.dot(projection);
                primitives::quantize_batch_into(self.quantizers.view(), rx, quantized.view_mut());
            }
            None => {
                primitives::quantize_batch_into(self.quantizers.view(), x, quantized.view_mut());
            }
        }
    }

    fn quantize_vector<I, S>(&self, x: ArrayBase<S, Ix1>) -> Array1<I>
    where
        I: AsPrimitive<usize> + Bounded + Zero,
        S: Data<Elem = A>,
        usize: AsPrimitive<I>,
    {
        match self.projection {
            Some(ref projection) => {
                let rx = x.dot(projection);
                primitives::quantize(self.quantizers.view(), self.reconstructed_len(), rx)
            }
            None => primitives::quantize(self.quantizers.view(), self.reconstructed_len(), x),
        }
    }

    fn quantized_len(&self) -> usize {
        self.quantizers.len_of(Axis(0))
    }
}

impl<A> Reconstruct<A> for Pq<A>
where
    A: NdFloat + Sum,
{
    fn reconstruct_batch_into<I, S>(
        &self,
        quantized: ArrayBase<S, Ix2>,
        mut reconstructions: ArrayViewMut2<A>,
    ) where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        primitives::reconstruct_batch_into(
            self.quantizers.view(),
            quantized,
            reconstructions.view_mut(),
        );

        if let Some(ref projection) = self.projection {
            let projected_reconstruction = reconstructions.dot(&projection.t());
            reconstructions.assign(&projected_reconstruction);
        }
    }

    fn reconstruct_into<I, S>(
        &self,
        quantized: ArrayBase<S, Ix1>,
        mut reconstruction: ArrayViewMut1<A>,
    ) where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        primitives::reconstruct_into(self.quantizers.view(), quantized, reconstruction.view_mut());

        if let Some(ref projection) = self.projection {
            let projected_reconstruction = reconstruction.dot(&projection.t());
            reconstruction.assign(&projected_reconstruction);
        }
    }

    fn reconstructed_len(&self) -> usize {
        primitives::reconstructed_len(self.quantizers.view())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array1, Array2, Array3, ArrayView2};
    use rand::distributions::Uniform;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::Pq;
    use crate::linalg::EuclideanDistance;
    use crate::ndarray_rand::RandomExt;
    use crate::pq::{QuantizeVector, Reconstruct, TrainPq};

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

    fn test_vectors() -> Array2<f32> {
        array![
            [0., 2., 0., -0.5, 0., 0.],
            [1., -0.2, 0., 0.5, 0.5, 0.],
            [-0.2, 0.2, 0., 0., -2., 0.],
            [1., 0.2, 0., 0., -2., 0.],
        ]
    }

    fn test_quantizations() -> Array2<usize> {
        array![[1, 1], [0, 1], [1, 0], [0, 0]]
    }

    fn test_reconstructions() -> Array2<f32> {
        array![
            [0., 1., 0., 0., 1., 0.],
            [1., 0., 0., 0., 1., 0.],
            [0., 1., 0., 1., -1., 0.],
            [1., 0., 0., 1., -1., 0.]
        ]
    }

    fn test_pq() -> Pq<f32> {
        let quantizers = array![[[1., 0., 0.], [0., 1., 0.]], [[1., -1., 0.], [0., 1., 0.]],];

        Pq {
            projection: None,
            quantizers,
        }
    }

    #[test]
    fn quantize_batch_with_predefined_codebook() {
        let pq = test_pq();

        assert_eq!(
            pq.quantize_batch::<usize, _>(test_vectors()),
            test_quantizations()
        );
    }

    #[test]
    fn quantize_with_predefined_codebook() {
        let pq = test_pq();

        for (vector, quantization) in test_vectors()
            .outer_iter()
            .zip(test_quantizations().outer_iter())
        {
            assert_eq!(pq.quantize_vector::<usize, _>(vector), quantization);
        }
    }

    #[test]
    fn quantize_with_pq() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let uniform = Uniform::new(0f32, 1f32);
        let instances = Array2::random_using((256, 20), uniform, &mut rng);
        let pq = Pq::train_pq_using(10, 7, 10, 1, instances.view(), &mut rng).unwrap();
        let loss = avg_euclidean_loss(instances.view(), &pq);
        // Loss is around 0.077.
        assert!(loss < 0.08);
    }

    #[test]
    fn quantize_with_type() {
        let uniform = Uniform::new(0f32, 1f32);
        let pq = Pq {
            projection: None,
            quantizers: Array3::random((1, 256, 10), uniform),
        };
        pq.quantize_vector::<u8, _>(Array1::random((10,), uniform));
    }

    #[test]
    #[should_panic]
    fn quantize_with_too_narrow_type() {
        let uniform = Uniform::new(0f32, 1f32);
        let pq = Pq {
            projection: None,
            quantizers: Array3::random((1, 257, 10), uniform),
        };
        pq.quantize_vector::<u8, _>(Array1::random((10,), uniform));
    }

    #[test]
    fn quantizer_lens() {
        let quantizer = test_pq();

        assert_eq!(quantizer.quantized_len(), 2);
        assert_eq!(quantizer.reconstructed_len(), 6);
    }

    #[test]
    fn reconstruct_batch_with_predefined_codebook() {
        let pq = test_pq();
        assert_eq!(
            pq.reconstruct_batch(test_quantizations()),
            test_reconstructions()
        );
    }

    #[test]
    fn reconstruct_with_predefined_codebook() {
        let pq = test_pq();

        for (quantization, reconstruction) in test_quantizations()
            .outer_iter()
            .zip(test_reconstructions().outer_iter())
        {
            assert_eq!(pq.reconstruct(quantization), reconstruction);
        }
    }
}
