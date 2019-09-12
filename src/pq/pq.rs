use std::iter;
use std::iter::Sum;

use log::info;
use ndarray::{
    azip, s, stack, Array1, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, ArrayViewMut2, Axis,
    Data, Ix1, Ix2, NdFloat,
};
use num_traits::{AsPrimitive, Bounded, Zero};
use ordered_float::OrderedFloat;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;

use super::{QuantizeVector, ReconstructVector, TrainPQ};
use crate::kmeans::{
    cluster_assignment, cluster_assignments, InitialCentroids, KMeansWithCentroids,
    NIterationsCondition, RandomInstanceCentroids,
};
use crate::rng::ReseedOnCloneRng;

/// Product quantizer (Jégou et al., 2011).
///
/// A product quantizer is a vector quantizer that slices a vector and
/// assigns to the *i*-th slice the index of the nearest centroid of the
/// *i*-th subquantizer. Vector reconstruction consists of concatenating
/// the centroids that represent the slices.
#[derive(Clone, Debug, PartialEq)]
pub struct PQ<A> {
    pub(crate) projection: Option<Array2<A>>,
    pub(crate) quantizers: Array3<A>,
}

impl<A> PQ<A>
where
    A: NdFloat,
{
    pub fn new(projection: Option<Array2<A>>, quantizers: Array3<A>) -> Self {
        assert!(
            !quantizers.is_empty(),
            "Attempted to construct a product quantizer without quantizers."
        );

        let quantizer_len = quantizers.len_of(Axis(0)) * quantizers.len_of(Axis(2));

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
            quantizers,
        }
    }

    pub(crate) fn check_quantizer_invariants(
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
        let sq_dims = instances.cols() / n_subquantizers;

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

        let sq_dims = instances.cols() / n_subquantizers;

        let offset = subquantizer_idx * sq_dims;
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sq_instances = instances.slice(s![.., offset..offset + sq_dims]);

        iter::repeat_with(|| {
            let mut quantizer = PQ::subquantizer_initial_centroids(
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

impl<A> TrainPQ<A> for PQ<A>
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
        rng: R,
    ) -> PQ<A>
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
        );

        let rng = ReseedOnCloneRng(rng);

        let rngs = iter::repeat_with(|| rng.clone())
            .take(n_subquantizers)
            .collect::<Vec<_>>();

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

        PQ {
            projection: None,
            quantizers: stack(Axis(0), &views).expect("Cannot stack subquantizers"),
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
                quantize_batch(self.quantizers.view(), self.reconstructed_len(), rx)
            }
            None => quantize_batch(self.quantizers.view(), self.reconstructed_len(), x),
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
                quantize(self.quantizers.view(), self.reconstructed_len(), rx)
            }
            None => quantize(self.quantizers.view(), self.reconstructed_len(), x),
        }
    }

    fn quantized_len(&self) -> usize {
        self.quantizers.len_of(Axis(0))
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
        let reconstruction = reconstruct_batch(self.quantizers.view(), quantized);
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
        let reconstruction = reconstruct(self.quantizers.view(), quantized);
        match self.projection {
            Some(ref projection) => reconstruction.dot(&projection.t()),
            None => reconstruction,
        }
    }

    fn reconstructed_len(&self) -> usize {
        self.quantizers.len_of(Axis(0)) * self.quantizers.len_of(Axis(2))
    }
}

fn quantize<A, I, S>(
    quantizers: ArrayView3<A>,
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
        quantizers.len_of(Axis(1)) - 1 <= I::max_value().as_(),
        "Cannot store centroids in quantizer index type"
    );

    let mut indices = Array1::zeros(quantizers.len_of(Axis(0)));

    let mut offset = 0;
    for (quantizer, index) in quantizers.outer_iter().zip(indices.iter_mut()) {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sub_vec = x.slice(s![offset..offset + quantizer.cols()]);
        *index = cluster_assignment(quantizer.view(), sub_vec).as_();

        offset += quantizer.cols();
    }

    indices
}

pub(crate) fn quantize_batch<A, I, S>(
    quantizers: ArrayView3<A>,
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

    let mut quantized = Array2::zeros((x.rows(), quantizers.len_of(Axis(0))));

    let mut offset = 0;
    for (quantizer, mut quantized) in quantizers
        .outer_iter()
        .zip(quantized.axis_iter_mut(Axis(1)))
    {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sub_matrix = x.slice(s![.., offset..offset + quantizer.cols()]);
        let assignments = cluster_assignments(quantizer.view(), sub_matrix, Axis(0));
        azip!(mut quantized, assignments in { *quantized = assignments.as_() });

        offset += quantizer.cols();
    }

    quantized
}

fn reconstruct<A, I, S>(quantizers: ArrayView3<A>, quantized: ArrayBase<S, Ix1>) -> Array1<A>
where
    A: NdFloat,
    I: AsPrimitive<usize>,
    S: Data<Elem = I>,
{
    assert_eq!(
        quantizers.len_of(Axis(0)),
        quantized.len(),
        "Quantization length does not match number of subquantizers"
    );

    let mut reconstruction =
        Vec::with_capacity(quantizers.len_of(Axis(0)) * quantizers.len_of(Axis(2)));
    for (&centroid, quantizer) in quantized.into_iter().zip(quantizers.outer_iter()) {
        reconstruction.extend(quantizer.index_axis(Axis(0), centroid.as_()));
    }

    Array1::from_vec(reconstruction)
}

fn reconstruct_batch<A, I, S>(quantizers: ArrayView3<A>, quantized: ArrayBase<S, Ix2>) -> Array2<A>
where
    A: NdFloat,
    I: AsPrimitive<usize>,
    S: Data<Elem = I>,
{
    let mut reconstructions = Array2::zeros((
        quantized.rows(),
        quantizers.len_of(Axis(0)) * quantizers.len_of(Axis(2)),
    ));

    reconstruct_batch_into(quantizers, quantized, reconstructions.view_mut());

    reconstructions
}

pub(crate) fn reconstruct_batch_into<A, I, S>(
    quantizers: ArrayView3<A>,
    quantized: ArrayBase<S, Ix2>,
    mut reconstructions: ArrayViewMut2<A>,
) where
    A: NdFloat,
    I: AsPrimitive<usize>,
    S: Data<Elem = I>,
{
    assert_eq!(
        quantizers.len_of(Axis(0)),
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
    use ndarray::{array, Array1, Array2, Array3, ArrayView2};
    use rand::distributions::Uniform;

    use super::PQ;
    use crate::linalg::EuclideanDistance;
    use crate::ndarray_rand::RandomExt;
    use crate::pq::{QuantizeVector, ReconstructVector, TrainPQ};

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

        euclidean_loss / instances.rows() as f32
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

    fn test_pq() -> PQ<f32> {
        let quantizers = array![[[1., 0., 0.], [0., 1., 0.]], [[1., -1., 0.], [0., 1., 0.]],];

        PQ {
            projection: None,
            quantizers,
        }
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
    fn quantize_with_pq() {
        let uniform = Uniform::new(0f32, 1f32);
        let instances = Array2::random((256, 20), uniform);
        let pq = PQ::train_pq(10, 7, 10, 1, instances.view());
        let loss = avg_euclidean_loss(instances.view(), &pq);
        // Loss is around 0.077.
        assert!(loss < 0.08);
    }

    #[test]
    fn quantize_with_type() {
        let uniform = Uniform::new(0f32, 1f32);
        let pq = PQ {
            projection: None,
            quantizers: Array3::random((1, 256, 10), uniform),
        };
        pq.quantize_vector::<u8, _>(Array1::random((10,), uniform));
    }

    #[test]
    #[should_panic]
    fn quantize_with_too_narrow_type() {
        let uniform = Uniform::new(0f32, 1f32);
        let pq = PQ {
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
            assert_eq!(pq.reconstruct_vector(quantization), reconstruction);
        }
    }
}
