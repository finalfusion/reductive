//! Product quantization.

use std::iter::Sum;

use log::info;
use ndarray::{
    s, stack, Array2, ArrayBase, ArrayView2, ArrayViewMut2, ArrayViewMut3, Axis, Data, Ix1, Ix2,
    NdFloat,
};
use ndarray_linalg::{
    eigh::Eigh,
    lapack::{Lapack, UPLO},
    svd::SVD,
    types::Scalar,
};
use num_traits::AsPrimitive;
use ordered_float::OrderedFloat;
use rand::{Rng, RngCore};
use rayon::prelude::*;

use crate::kmeans::KMeansIteration;
use crate::linalg::Covariance;

use super::primitives;
use super::{TrainPQ, PQ};

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
pub struct OPQ;

impl<A> TrainPQ<A> for OPQ
where
    A: Lapack + NdFloat + Scalar + Sum,
    A::Real: NdFloat,
    usize: AsPrimitive<A>,
{
    fn train_pq_using<S, R>(
        n_subquantizers: usize,
        n_subquantizer_bits: u32,
        n_iterations: usize,
        _n_attempts: usize,
        instances: ArrayBase<S, Ix2>,
        mut rng: R,
    ) -> PQ<A>
    where
        S: Sync + Data<Elem = A>,
        R: RngCore,
    {
        PQ::check_quantizer_invariants(
            n_subquantizers,
            n_subquantizer_bits,
            n_iterations,
            1,
            instances.view(),
        );

        // Find initial projection matrix, which will be refined iteratively.
        let mut projection = Self::create_projection_matrix(instances.view(), n_subquantizers);
        let rx = instances.dot(&projection);

        // Pick centroids.
        let centroids = Self::initial_centroids(
            n_subquantizers,
            2usize.pow(n_subquantizer_bits),
            rx.view(),
            &mut rng,
        );

        let views = centroids
            .iter()
            .map(|c| c.view().insert_axis(Axis(0)))
            .collect::<Vec<_>>();
        let mut quantizers = stack(Axis(0), &views).expect("Cannot stack subquantizers");

        // Iteratively refine the clusters and the projection matrix.
        for i in 0..n_iterations {
            info!("Train iteration {}", i);
            Self::train_iteration(
                projection.view_mut(),
                quantizers.view_mut(),
                instances.view(),
            );
        }

        PQ {
            projection: Some(projection),
            quantizers,
        }
    }
}

impl OPQ {
    pub(crate) fn create_projection_matrix<A>(
        instances: ArrayView2<A>,
        n_subquantizers: usize,
    ) -> Array2<A>
    where
        A: Lapack + NdFloat + Scalar,
        A::Real: NdFloat,
        usize: AsPrimitive<A>,
    {
        info!(
            "Creating projection matrix ({} instances, {} dimensions, {} subquantizers)",
            instances.nrows(),
            instances.ncols(),
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
        (0..n_subquantizers)
            .map(|sq| {
                PQ::subquantizer_initial_centroids(
                    sq,
                    n_subquantizers,
                    codebook_len,
                    instances.view(),
                    rng,
                )
            })
            .collect()
    }

    fn train_iteration<A>(
        mut projection: ArrayViewMut2<A>,
        mut centroids: ArrayViewMut3<A>,
        instances: ArrayView2<A>,
    ) where
        A: Lapack + NdFloat + Scalar + Sum,
        A::Real: NdFloat,
        usize: AsPrimitive<A>,
    {
        info!("Updating subquantizers");

        // Perform one iteration of cluster updates, using regular k-means.
        let rx = instances.dot(&projection);
        Self::update_subquantizers(centroids.view_mut(), rx.view());

        info!("Updating projection matrix");

        // Do a quantization -> reconstruction roundtrip. We recycle the
        // projection matrix to avoid (re)allocations.
        let quantized = primitives::quantize_batch::<_, usize, _>(centroids.view(), rx.view());
        let mut reconstructed = rx;
        primitives::reconstruct_batch_into(centroids.view(), quantized, reconstructed.view_mut());

        // Find the new projection matrix using the instances and their
        // (projected) reconstructions. See (the text below) Eq 7 in
        // Ge et al., 2013.
        let (u, _, vt) = instances.t().dot(&reconstructed).svd(true, true).unwrap();
        projection.assign(&u.unwrap().dot(&vt.unwrap()));
    }

    fn update_subquantizers<A, S>(mut centroids: ArrayViewMut3<A>, instances: ArrayBase<S, Ix2>)
    where
        A: NdFloat + Scalar + Sum,
        A::Real: NdFloat,
        usize: AsPrimitive<A>,
        S: Sync + Data<Elem = A>,
    {
        centroids
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(sq, mut sq_centroids)| {
                let offset = sq * sq_centroids.ncols();
                // ndarray#474
                #[allow(clippy::deref_addrof)]
                let sq_instances = instances.slice(s![.., offset..offset + sq_centroids.ncols()]);
                sq_instances.kmeans_iteration(Axis(0), sq_centroids.view_mut());
            });
    }
}

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

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2, ArrayView2};
    use rand::distributions::Uniform;

    use super::OPQ;
    use crate::linalg::EuclideanDistance;
    use crate::ndarray_rand::RandomExt;
    use crate::pq::{QuantizeVector, ReconstructVector, TrainPQ, PQ};

    /// Calculate the average euclidean distances between the the given
    /// instances and the instances returned by quantizing and then
    /// reconstructing the instances.
    fn avg_euclidean_loss(instances: ArrayView2<f32>, quantizer: &PQ<f32>) -> f32 {
        let mut euclidean_loss = 0f32;

        let quantized: Array2<u8> = quantizer.quantize_batch(instances);
        let reconstructions = quantizer.reconstruct_batch(quantized);

        for (instance, reconstruction) in instances.outer_iter().zip(reconstructions.outer_iter()) {
            euclidean_loss += instance.euclidean_distance(reconstruction);
        }

        euclidean_loss / instances.nrows() as f32
    }

    #[test]
    fn bucket_eigenvalues() {
        // Some fake eigenvalues.
        let eigenvalues = array![0.2, 0.6, 0.4, 0.1, 0.3, 0.5];
        assert_eq!(
            super::bucket_eigenvalues(eigenvalues.view(), 3),
            vec![vec![1, 3], vec![5, 0], vec![2, 4]]
        );
    }

    #[test]
    fn bucket_large_eigenvalues() {
        let eigenvalues = array![11174., 23450., 30835., 1557., 32425., 5154.];
        assert_eq!(
            super::bucket_eigenvalues(eigenvalues.view(), 3),
            vec![vec![4, 3], vec![2, 5], vec![1, 0]]
        );
    }

    #[test]
    #[should_panic]
    fn bucket_eigenvalues_uneven() {
        // Some fake eigenvalues.
        let eigenvalues = array![0.2, 0.6, 0.4, 0.1, 0.3, 0.5];
        super::bucket_eigenvalues(eigenvalues.view(), 4);
    }

    #[test]
    fn quantize_with_opq() {
        let uniform = Uniform::new(0f32, 1f32);
        let instances = Array2::random((256, 20), uniform);
        let pq = OPQ::train_pq(10, 7, 10, 1, instances.view());
        let loss = avg_euclidean_loss(instances.view(), &pq);
        // Loss is around 0.09.
        assert!(loss < 0.1);
    }
}
