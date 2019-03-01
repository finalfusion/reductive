//! Product quantization.

use std::iter;
use std::iter::Sum;

use log::info;
use ndarray::{
    azip, s, Array1, Array2, ArrayBase, ArrayView2, ArrayViewMut2, Axis, Data, Ix1, Ix2, NdFloat,
};
#[cfg(feature = "opq-train")]
use ndarray_linalg::{eigh::Eigh, lapack_traits::UPLO, svd::SVD, types::Scalar};
use ndarray_parallel::prelude::*;
use num_traits::{AsPrimitive, Bounded, Zero};
use ordered_float::OrderedFloat;
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;

#[cfg(feature = "opq-train")]
use crate::kmeans::KMeansIteration;
use crate::kmeans::{
    cluster_assignment, cluster_assignments, InitialCentroids, KMeansWithCentroids,
    NIterationsCondition, RandomInstanceCentroids,
};
#[cfg(feature = "opq-train")]
use crate::linalg::Covariance;

/// Training triat for product quantizers.
///
/// This traits specifies the training functions for product
/// quantizers.
pub trait TrainPQ<A> {
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
    ) -> PQ<A>
    where
        S: Sync + Data<Elem = A>,
    {
        let mut rng = XorShiftRng::from_entropy();
        Self::train_pq_using(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            instances,
            &mut rng,
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
    /// each subquantizer.
    fn train_pq_using<S>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
        rng: &mut impl Rng,
    ) -> PQ<A>
    where
        S: Sync + Data<Elem = A>;
}

/// Vector quantization.
pub trait QuantizeVector<A> {
    /// Quantize a batch of vectors.
    fn quantize_batch<I, S>(&self, x: ArrayBase<S, Ix2>) -> Array2<I>
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
pub trait ReconstructVector<A> {
    /// Reconstruct a vector.
    ///
    /// The vectors are reconstructed from the quantization indices.
    fn reconstruct_batch<I, S>(&self, quantized: ArrayBase<S, Ix2>) -> Array2<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>;

    /// Reconstruct a batch of vectors.
    ///
    /// The vector is reconstructed from the quantization indices.
    fn reconstruct_vector<I, S>(&self, quantized: ArrayBase<S, Ix1>) -> Array1<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>;

    /// Get the length of a vector after reconstruction.
    fn reconstructed_len(&self) -> usize;
}

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
#[cfg(feature = "opq-train")]
pub struct GaussianOPQ;

#[cfg(feature = "opq-train")]
impl<A> TrainPQ<A> for GaussianOPQ
where
    A: NdFloat + Scalar + Sum,
    A::Real: NdFloat,
    usize: AsPrimitive<A>,
{
    fn train_pq_using<S>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
        rng: &mut impl Rng,
    ) -> PQ<A>
    where
        S: Sync + Data<Elem = A>,
    {
        PQ::check_quantizer_invariants(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            instances.view(),
        );

        let projection = create_projection_matrix(instances.view(), n_subquantizers);
        let rx = instances.dot(&projection);
        let pq = PQ::train_pq_using(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            rx,
            rng,
        );

        PQ {
            projection: Some(projection),
            quantizer_len: pq.quantizer_len,
            quantizers: pq.quantizers,
        }
    }
}

/// Optimized product quantizer (Ge et al., 2013).
///
/// A product quantizer is a vector quantizer that slices a vector and
/// assigns to the *i*-th slice the index of the nearest centroid of the
/// *i*-th subquantizer. Vector reconstruction consists of concatenating
/// the centroids that represent the slices.
///
/// This quantizer learns a orthonormal matrix that rotates the input
/// space in order to balance variances over subquantizers. If the
/// variables have a Gaussian distribution, `GaussianOPQ` is faster to
/// train than this quantizer.
///
/// This quantizer always trains the quantizer in one attempt, so the
/// `n_attempts` argument of the `TrainPQ` constructors currently has
/// no effect.
#[cfg(feature = "opq-train")]
pub struct OPQ;

#[cfg(feature = "opq-train")]
impl<A> TrainPQ<A> for OPQ
where
    A: NdFloat + Scalar + Sum,
    A::Real: NdFloat,
    usize: AsPrimitive<A>,
{
    fn train_pq_using<S>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        _n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
        rng: &mut impl Rng,
    ) -> PQ<A>
    where
        S: Sync + Data<Elem = A>,
    {
        PQ::check_quantizer_invariants(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            1,
            instances.view(),
        );

        // Find initial projection matrix, which will be refined iteratively.
        let mut projection = create_projection_matrix(instances.view(), n_subquantizers);
        let rx = instances.dot(&projection);

        // Pick centroids.
        let mut centroids = initial_centroids(
            n_subquantizers,
            2usize.pow(n_subquantizer_bits),
            rx.view(),
            rng,
        );

        // Iteratively refine the clusters and the projection matrix.
        for i in 0..n_iterations {
            info!("Train iteration {}", i);
            Self::train_iteration(projection.view_mut(), &mut centroids, instances.view());
        }

        PQ {
            projection: Some(projection),
            quantizers: centroids,
            quantizer_len: instances.cols(),
        }
    }
}

#[cfg(feature = "opq-train")]
impl OPQ {
    fn train_iteration<A>(
        mut projection: ArrayViewMut2<A>,
        centroids: &mut [Array2<A>],
        instances: ArrayView2<A>,
    ) where
        A: NdFloat + Scalar + Sum,
        A::Real: NdFloat,
        usize: AsPrimitive<A>,
    {
        info!("Updating subquantizers");

        // Perform one iteration of cluster updates, using regular k-means.
        let rx = instances.dot(&projection);
        Self::update_subquantizers(centroids, rx.view());

        info!("Updating projection matrix");

        // Do a quantization -> reconstruction roundtrip. We recycle the
        // projection matrix to avoid (re)allocations.
        let quantized = quantize_batch::<_, usize, _>(centroids, instances.cols(), rx.view());
        let mut reconstructed = rx;
        reconstruct_batch_into(centroids, quantized, reconstructed.view_mut());

        // Find the new projection matrix using the instances and their
        // (projected) reconstructions. See (the text below) Eq 7 in
        // Ge et al., 2013.
        let (u, _, vt) = instances.t().dot(&reconstructed).svd(true, true).unwrap();
        projection.assign(&u.unwrap().dot(&vt.unwrap()));
    }

    fn update_subquantizers<A, S>(centroids: &mut [Array2<A>], instances: ArrayBase<S, Ix2>)
    where
        A: NdFloat + Scalar + Sum,
        A::Real: NdFloat,
        usize: AsPrimitive<A>,
        S: Sync + Data<Elem = A>,
    {
        centroids
            .into_par_iter()
            .enumerate()
            .for_each(|(sq, sq_centroids)| {
                let offset = sq * sq_centroids.cols();
                // ndarray#474
                #[allow(clippy::deref_addrof)]
                let sq_instances = instances.slice(s![.., offset..offset + sq_centroids.cols()]);
                sq_instances.kmeans_iteration(Axis(0), sq_centroids.view_mut());
            });
    }
}

/// Product quantizer (Jégou et al., 2011).
///
/// A product quantizer is a vector quantizer that slices a vector and
/// assigns to the *i*-th slice the index of the nearest centroid of the
/// *i*-th subquantizer. Vector reconstruction consists of concatenating
/// the centroids that represent the slices.
#[derive(Clone, Debug, PartialEq)]
pub struct PQ<A> {
    projection: Option<Array2<A>>,
    quantizer_len: usize,
    quantizers: Vec<Array2<A>>,
}

impl<A> PQ<A>
where
    A: NdFloat + Sum,
    usize: AsPrimitive<A>,
{
    pub fn new(projection: Option<Array2<A>>, quantizers: Vec<Array2<A>>) -> Self {
        assert!(
            !quantizers.is_empty(),
            "Attempted to construct a product quantizer without quantizers."
        );

        // Check that subquantizers have the same shapes.
        let mut shape_iter = quantizers.iter().map(|q| q.shape());
        let first_shape = shape_iter.next().unwrap();
        assert!(
            shape_iter.all(|shape| shape == first_shape),
            "Diverging quantizer shapes."
        );

        let quantizer_len = quantizers.len() * quantizers[0].cols();

        if let Some(ref projection) = projection {
            assert_eq!(
                projection.shape(),
                [quantizer_len; 2],
                "Incorrect projection matrix shape, was: {:?}, should be [{}, {}]",
                projection.shape(),
                quantizer_len,
                quantizer_len
            );
        }

        PQ {
            projection,
            quantizer_len,
            quantizers,
        }
    }

    fn check_quantizer_invariants(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayView2<A>,
    ) {
        assert!(
            n_subquantizers > 0 && n_subquantizers <= instances.cols(),
            "The number of subquantizers should at least be 1 and at most be {}.",
            instances.cols()
        );
        assert!(
            n_subquantizer_bits > 0,
            "Number of quantizer bits should at least be one."
        );
        assert!(
            instances.cols() % n_subquantizers == 0,
            "The number of subquantizers should evenly divide each instance."
        );
        assert!(
            n_iterations > 0,
            "The subquantizers should be optimized for at least one iteration."
        );
        assert!(
            n_attempts > 0,
            "The subquantizers should be optimized for at least one attempt."
        );
    }

    /// Get the number of centroids per quantizer.
    pub fn n_quantizer_centroids(&self) -> usize {
        self.quantizers[0].rows()
    }

    /// Get the projection matrix (if used).
    pub fn projection(&self) -> Option<ArrayView2<A>> {
        self.projection.as_ref().map(Array2::view)
    }

    /// Train a subquantizer.
    ///
    /// `sq` is the index of the subquantizer, `sq_dims` the number of
    /// dimensions that are quantized, and `codebook_len` the code book
    /// size of the quantizer.
    fn train_subquantizer(
        idx: usize,
        quantizer: Array2<A>,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayView2<A>,
    ) -> Array2<A> {
        assert!(n_attempts > 0, "Cannot train a subquantizer in 0 attempts.");

        info!("Training PQ subquantizer {}", idx);

        let sq_dims = quantizer.cols();

        let offset = idx * sq_dims;
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sq_instances = instances.slice(s![.., offset..offset + sq_dims]);

        iter::repeat_with(|| {
            let mut quantizer = quantizer.to_owned();
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
    pub fn subquantizers(&self) -> &[Array2<A>] {
        &self.quantizers
    }
}

impl<A> TrainPQ<A> for PQ<A>
where
    A: NdFloat + Sum,
    usize: AsPrimitive<A>,
{
    fn train_pq_using<S>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
        rng: &mut impl Rng,
    ) -> PQ<A>
    where
        S: Sync + Data<Elem = A>,
    {
        Self::check_quantizer_invariants(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            n_attempts,
            instances.view(),
        );

        let quantizers = initial_centroids(
            n_subquantizers,
            2usize.pow(n_subquantizer_bits),
            instances.view(),
            rng,
        );

        let quantizers = quantizers
            .into_par_iter()
            .enumerate()
            .map(|(idx, quantizer)| {
                Self::train_subquantizer(idx, quantizer, n_iterations, n_attempts, instances.view())
            })
            .collect();

        PQ {
            projection: None,
            quantizer_len: instances.cols(),
            quantizers,
        }
    }
}

impl<A> QuantizeVector<A> for PQ<A>
where
    A: NdFloat + Sum,
{
    fn quantize_batch<I, S>(&self, x: ArrayBase<S, Ix2>) -> Array2<I>
    where
        I: AsPrimitive<usize> + Bounded + Zero,
        S: Data<Elem = A>,
        usize: AsPrimitive<I>,
    {
        match self.projection {
            Some(ref projection) => {
                let rx = x.dot(projection);
                quantize_batch(&self.quantizers, self.quantizer_len, rx)
            }
            None => quantize_batch(&self.quantizers, self.quantizer_len, x),
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
                quantize(&self.quantizers, self.quantizer_len, rx)
            }
            None => quantize(&self.quantizers, self.quantizer_len, x),
        }
    }

    fn quantized_len(&self) -> usize {
        self.quantizers.len()
    }
}

impl<A> ReconstructVector<A> for PQ<A>
where
    A: NdFloat + Sum,
{
    fn reconstruct_batch<I, S>(&self, quantized: ArrayBase<S, Ix2>) -> Array2<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        let reconstruction = reconstruct_batch(&self.quantizers, quantized);
        match self.projection {
            Some(ref projection) => reconstruction.dot(&projection.t()),
            None => reconstruction,
        }
    }

    fn reconstruct_vector<I, S>(&self, quantized: ArrayBase<S, Ix1>) -> Array1<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        let reconstruction = reconstruct(&self.quantizers, quantized);
        match self.projection {
            Some(ref projection) => reconstruction.dot(&projection.t()),
            None => reconstruction,
        }
    }

    fn reconstructed_len(&self) -> usize {
        self.quantizer_len
    }
}

#[cfg(feature = "opq-train")]
fn bucket_eigenvalues<S, A>(eigenvalues: ArrayBase<S, Ix1>, n_buckets: usize) -> Vec<Vec<usize>>
where
    S: Data<Elem = A>,
    A: NdFloat,
{
    assert!(
        n_buckets > 0,
        "Cannot distribute eigenvalues over zero buckets."
    );
    assert!(
        eigenvalues.len() >= n_buckets,
        "At least one eigenvalue is required per bucket"
    );
    assert_eq!(
        eigenvalues.len() % n_buckets,
        0,
        "The number of eigenvalues should be a multiple of the number of buckets."
    );

    let mut eigenvalue_indices: Vec<usize> = (0..eigenvalues.len()).collect();
    eigenvalue_indices
        .sort_unstable_by(|l, r| OrderedFloat(eigenvalues[*l]).cmp(&OrderedFloat(eigenvalues[*r])));

    // Only handle positive values, to switch to log-space. This is
    // ok for our purposes, since we only eigendecompose covariance
    // matrices.
    assert!(
        eigenvalues[eigenvalue_indices[0]] >= A::zero(),
        "Bucketing is only supported for positive eigenvalues."
    );

    // Do eigenvalue multiplication in log-space to avoid over/underflow.
    let mut eigenvalues = eigenvalues.map(|&v| (v + A::epsilon()).ln());

    // Make values positive, this is so that we can treat eigenvalues
    // (0,1] and [1,] in the same manner.
    let smallest = eigenvalues
        .iter()
        .cloned()
        .min_by_key(|&v| OrderedFloat(v))
        .unwrap();
    eigenvalues.map_mut(|v| *v -= smallest);

    let mut assignments = vec![vec![]; n_buckets];
    let mut products = vec![A::zero(); n_buckets];
    let max_assignments = eigenvalues.len_of(Axis(0)) / n_buckets;

    while let Some(eigenvalue_idx) = eigenvalue_indices.pop() {
        // Find non-full bucket with the smallest product.
        let (idx, _) = assignments
            .iter()
            .enumerate()
            .filter(|(_, a)| a.len() < max_assignments)
            .min_by_key(|(idx, _)| OrderedFloat(products[*idx]))
            .unwrap();

        assignments[idx].push(eigenvalue_idx);
        products[idx] += eigenvalues[eigenvalue_idx];
    }

    assignments
}

#[cfg(feature = "opq-train")]
fn create_projection_matrix<A>(instances: ArrayView2<A>, n_subquantizers: usize) -> Array2<A>
where
    A: NdFloat + Scalar,
    A::Real: NdFloat,
    usize: AsPrimitive<A>,
{
    info!(
        "Creating projection matrix ({} instances, {} dimensions, {} subquantizers)",
        instances.rows(),
        instances.cols(),
        n_subquantizers
    );

    // Compute the covariance matrix.
    let cov = instances.covariance(Axis(0));

    // Find eigenvalues/vectors.
    let (eigen_values, eigen_vectors) = cov.eigh(UPLO::Upper).unwrap();

    // Order principal components by their eigenvalues
    let buckets = bucket_eigenvalues(eigen_values.view(), n_subquantizers);

    let mut transformations = Array2::zeros((eigen_values.len(), eigen_values.len()));
    for (idx, direction_idx) in buckets.into_iter().flatten().enumerate() {
        transformations
            .index_axis_mut(Axis(1), idx)
            .assign(&eigen_vectors.index_axis(Axis(1), direction_idx));
    }

    transformations
}

fn initial_centroids<S, A>(
    n_subquantizers: usize,
    codebook_len: usize,
    instances: ArrayBase<S, Ix2>,
    rng: &mut impl Rng,
) -> Vec<Array2<A>>
where
    S: Data<Elem = A>,
    A: NdFloat,
{
    let sq_dims = instances.cols() / n_subquantizers;

    let mut random_centroids = RandomInstanceCentroids::new(rng);

    (0..n_subquantizers)
        .map(|sq| {
            let offset = sq * sq_dims;
            // ndarray#474
            #[allow(clippy::deref_addrof)]
            let sq_instances = instances.slice(s![.., offset..offset + sq_dims]);
            random_centroids.initial_centroids(sq_instances, Axis(0), codebook_len)
        })
        .collect()
}

fn quantize<A, I, S>(
    quantizers: &[Array2<A>],
    quantizer_len: usize,
    x: ArrayBase<S, Ix1>,
) -> Array1<I>
where
    A: NdFloat + Sum,
    I: 'static + AsPrimitive<usize> + Bounded + Copy + Zero,
    S: Data<Elem = A>,
    usize: AsPrimitive<I>,
{
    assert_eq!(
        quantizer_len,
        x.len(),
        "Quantizer and vector length mismatch"
    );

    assert!(
        quantizers[0].rows() - 1 <= I::max_value().as_(),
        "Cannot store centroids in quantizer index type"
    );

    let mut indices = Array1::zeros(quantizers.len());

    let mut offset = 0;
    for (quantizer, index) in quantizers.iter().zip(indices.iter_mut()) {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sub_vec = x.slice(s![offset..offset + quantizer.cols()]);
        *index = cluster_assignment(quantizer.view(), sub_vec).as_();

        offset += quantizer.cols();
    }

    indices
}

fn quantize_batch<A, I, S>(
    quantizers: &[Array2<A>],
    quantizer_len: usize,
    x: ArrayBase<S, Ix2>,
) -> Array2<I>
where
    A: NdFloat + Sum,
    I: 'static + AsPrimitive<usize> + Bounded + Copy + Zero,
    S: Data<Elem = A>,
    usize: AsPrimitive<I>,
{
    assert_eq!(
        quantizer_len,
        x.cols(),
        "Quantizer and vector length mismatch"
    );

    let mut quantized = Array2::zeros((x.rows(), quantizers.len()));

    let mut offset = 0;
    for (quantizer, mut quantized) in quantizers.iter().zip(quantized.axis_iter_mut(Axis(1))) {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sub_matrix = x.slice(s![.., offset..offset + quantizer.cols()]);
        let assignments = cluster_assignments(quantizer.view(), sub_matrix, Axis(0));
        azip!(mut quantized, assignments in { *quantized = assignments.as_() });

        offset += quantizer.cols();
    }

    quantized
}

fn reconstruct<A, I, S>(quantizers: &[Array2<A>], quantized: ArrayBase<S, Ix1>) -> Array1<A>
where
    A: NdFloat,
    I: AsPrimitive<usize>,
    S: Data<Elem = I>,
{
    assert_eq!(
        quantizers.len(),
        quantized.len(),
        "Quantization length does not match number of subquantizers"
    );

    let mut reconstruction = Array1::zeros((quantizers.len() * quantizers[0].cols(),));

    let mut offset = 0;
    for (&centroid, quantizer) in quantized.into_iter().zip(quantizers.iter()) {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let mut sub_vec = reconstruction.slice_mut(s![offset..offset + quantizer.cols()]);
        sub_vec.assign(&quantizer.index_axis(Axis(0), centroid.as_()));
        offset += quantizer.cols();
    }

    reconstruction
}

fn reconstruct_batch<A, I, S>(quantizers: &[Array2<A>], quantized: ArrayBase<S, Ix2>) -> Array2<A>
where
    A: NdFloat,
    I: AsPrimitive<usize>,
    S: Data<Elem = I>,
{
    let mut reconstructions =
        Array2::zeros((quantized.rows(), quantizers.len() * quantizers[0].cols()));

    reconstruct_batch_into(quantizers, quantized, reconstructions.view_mut());

    reconstructions
}

fn reconstruct_batch_into<A, I, S>(
    quantizers: &[Array2<A>],
    quantized: ArrayBase<S, Ix2>,
    mut reconstructions: ArrayViewMut2<A>,
) where
    A: NdFloat,
    I: AsPrimitive<usize>,
    S: Data<Elem = I>,
{
    assert_eq!(
        quantizers.len(),
        quantized.cols(),
        "Quantization length does not match number of subquantizers"
    );

    for (quantized, mut reconstruction) in
        quantized.outer_iter().zip(reconstructions.outer_iter_mut())
    {
        reconstruction.assign(&reconstruct(quantizers, quantized));
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array1, Array2};
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    use super::{QuantizeVector, ReconstructVector, PQ};

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

    fn test_pq() -> PQ<f32> {
        let quantizers = vec![
            array![[1., 0., 0.], [0., 1., 0.]],
            array![[1., -1., 0.], [0., 1., 0.]],
        ];

        PQ {
            projection: None,
            quantizer_len: 6,
            quantizers,
        }
    }

    #[test]
    #[cfg(feature = "opq-train")]
    fn bucket_eigenvalues() {
        // Some fake eigenvalues.
        let eigenvalues = array![0.2, 0.6, 0.4, 0.1, 0.3, 0.5];
        assert_eq!(
            super::bucket_eigenvalues(eigenvalues.view(), 3),
            vec![vec![1, 3], vec![5, 0], vec![2, 4]]
        );
    }

    #[test]
    #[cfg(feature = "opq-train")]
    fn bucket_large_eigenvalues() {
        let eigenvalues = array![11174., 23450., 30835., 1557., 32425., 5154.];
        assert_eq!(
            super::bucket_eigenvalues(eigenvalues.view(), 3),
            vec![vec![4, 3], vec![2, 5], vec![1, 0]]
        );
    }

    #[test]
    #[should_panic]
    #[cfg(feature = "opq-train")]
    fn bucket_eigenvalues_uneven() {
        // Some fake eigenvalues.
        let eigenvalues = array![0.2, 0.6, 0.4, 0.1, 0.3, 0.5];
        super::bucket_eigenvalues(eigenvalues.view(), 4);
    }

    #[test]
    fn quantize_batch_with_predefined_codebook() {
        let pq = test_pq();

        assert_eq!(pq.quantize_batch(test_vectors()), test_quantizations());
    }

    #[test]
    fn quantize_with_predefined_codebook() {
        let pq = test_pq();

        for (vector, quantization) in test_vectors()
            .outer_iter()
            .zip(test_quantizations().outer_iter())
        {
            assert_eq!(pq.quantize_vector(vector), quantization);
        }
    }

    #[test]
    fn quantize_with_type() {
        let uniform = Uniform::new(0f32, 1f32);
        let pq = PQ {
            projection: None,
            quantizer_len: 10,
            quantizers: vec![Array2::random((256, 10), uniform)],
        };
        pq.quantize_vector::<u8, _>(Array1::random((10,), uniform));
    }

    #[test]
    #[should_panic]
    fn quantize_with_too_narrow_type() {
        let uniform = Uniform::new(0f32, 1f32);
        let pq = PQ {
            projection: None,
            quantizer_len: 10,
            quantizers: vec![Array2::random((257, 10), uniform)],
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
            assert_eq!(pq.reconstruct_vector(quantization), reconstruction);
        }
    }
}
