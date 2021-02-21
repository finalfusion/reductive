//! K-means clustering.

use std::collections::HashSet;
use std::iter::Sum;

use ndarray::{
    Array1, Array2, ArrayBase, ArrayView2, ArrayViewMut2, Axis, Data, Ix1, Ix2, NdFloat,
};
use num_traits::AsPrimitive;
use ordered_float::OrderedFloat;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use crate::linalg::SquaredEuclideanDistance;

/// Initial centroid selection.
pub trait InitialCentroids<A> {
    /// Pick *k* initial centroids for k-mean clustering.
    ///
    /// The initial centroid selection can use the provided `data`, where
    /// instances are along `instance_axis`.
    ///
    /// Implementations return a matrix with centroids as rows.
    fn initial_centroids<S>(
        &mut self,
        data: ArrayBase<S, Ix2>,
        instance_axis: Axis,
        k: usize,
    ) -> Array2<A>
    where
        S: Data<Elem = A>;
}

/// Pick random data set instances as centroids.
pub struct RandomInstanceCentroids<R>(R);

impl<R> RandomInstanceCentroids<R>
where
    R: Rng,
{
    /// Construct `RandomInstanceCentroids` from a random number generator.
    pub fn new(rng: R) -> Self {
        RandomInstanceCentroids(rng)
    }
}

impl<A, R> InitialCentroids<A> for RandomInstanceCentroids<R>
where
    A: NdFloat,
    R: Rng,
{
    fn initial_centroids<S>(
        &mut self,
        data: ArrayBase<S, Ix2>,
        instance_axis: Axis,
        k: usize,
    ) -> Array2<A>
    where
        S: Data<Elem = A>,
    {
        assert!(k > 0, "Cannot pick 0 random centroids");
        assert!(
            k < data.len_of(instance_axis),
            "Cannot pick more centroids than instances: {} instances, {} centroids",
            data.len_of(instance_axis),
            k
        );
        assert!(
            data.len() / data.len_of(instance_axis) > 0,
            "Cannot pick centroids from zero-length instances"
        );

        // Use random instances as centroids.
        let uniform = Uniform::new(0, data.len_of(instance_axis));
        let mut initial_indices = HashSet::new();
        while initial_indices.len() != k {
            initial_indices.insert(uniform.sample(&mut self.0));
        }

        // Assign instances.
        let mut centroids = Array2::zeros((k, data.len() / data.len_of(instance_axis)));
        for (idx, mut centroid) in initial_indices.iter().zip(centroids.outer_iter_mut()) {
            centroid.assign(&data.index_axis(instance_axis, *idx));
        }

        centroids
    }
}

/// k-means stopping conditions.
pub trait StopCondition<A> {
    /// Returns `true` when k-means clustering should stop.
    fn should_stop(&mut self, iteration: usize, loss: A) -> bool;
}

/// Condition that stops clustering after N iterations.
#[derive(Copy, Clone, Debug)]
pub struct NIterationsCondition(pub usize);

impl<A> StopCondition<A> for NIterationsCondition {
    fn should_stop(&mut self, iteration: usize, _loss: A) -> bool {
        iteration >= self.0
    }
}

/// Find nearest cluster centroid for an instance.
///
/// Find nearest centroid for each instance along `instance_axis` of
/// `instances`. Returns for each instance the index of the nearest
/// cluster centroid.
pub(crate) fn cluster_assignment<A, S>(
    centroids: ArrayView2<A>,
    instance: ArrayBase<S, Ix1>,
) -> usize
where
    A: NdFloat + Sum,
    S: Data<Elem = A>,
{
    instance
        .squared_euclidean_distance(centroids)
        .iter()
        .enumerate()
        .min_by_key(|v| OrderedFloat(*v.1))
        .unwrap()
        .0
}

/// Find nearest cluster centroid for each instance.
///
/// Find nearest centroid for each instance along `instance_axis` of
/// `instances`. Returns for each instance the index of the nearest
/// cluster centroid.
pub(crate) fn cluster_assignments<A>(
    centroids: ArrayView2<A>,
    instances: ArrayView2<A>,
    instance_axis: Axis,
) -> Array1<usize>
where
    A: NdFloat + Sum,
{
    let mut assignments = Array1::zeros(instances.len_of(instance_axis));

    let dists = if instance_axis == Axis(0) {
        instances.squared_euclidean_distance(centroids)
    } else {
        instances.t().squared_euclidean_distance(centroids)
    };

    for (assignment, inst_dists) in assignments.iter_mut().zip(dists.outer_iter()) {
        *assignment = inst_dists
            .iter()
            .enumerate()
            .min_by_key(|v| OrderedFloat(*v.1))
            .unwrap()
            .0;
    }

    assignments
}

/// Update centroids to the mean of the assigned data points.
///
/// `instance_axis` is the instance axis of `data`. The centroids
/// are row-based. `assignments` contains an assignment for each
/// data point.
fn update_centroids<A, S>(
    mut centroids: ArrayViewMut2<A>,
    data: ArrayView2<A>,
    instance_axis: Axis,
    assignments: ArrayBase<S, Ix1>,
) where
    A: NdFloat,
    S: Data<Elem = usize>,
{
    assert_eq!(
        assignments.len(),
        data.len_of(instance_axis),
        "The number of assignments should be equal to the number of instances."
    );

    centroids.fill(A::zero());

    let mut centroid_counts = Array1::zeros(centroids.nrows());

    for (instance, assignment) in data.axis_iter(instance_axis).zip(assignments.iter()) {
        let mut centroid = centroids.index_axis_mut(Axis(0), *assignment);
        centroid += &instance;
        centroid_counts[*assignment] += A::one();
    }

    for (mut centroid, centroid_count) in
        centroids.outer_iter_mut().zip(centroid_counts.outer_iter())
    {
        if centroid_count[()] > A::zero() {
            centroid /= &centroid_count;
        }
    }
}

/// Trait for types that implement k-means clustering.
pub trait KMeans<A> {
    /// Perform k-means clustering.
    ///
    /// Performs k-means clustering on the matrix of instances along the
    /// given `instance_axis`.
    ///
    /// Returns the *k x d* matrix of cluster centroids and the mean
    /// mean-squared error.
    fn k_means(
        &self,
        instance_axis: Axis,
        k: usize,
        initial_centroids: impl InitialCentroids<A>,
        stop_condition: impl StopCondition<A>,
    ) -> (Array2<A>, A);
}

impl<'a, S, A> KMeans<A> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: NdFloat + Sum,
    usize: AsPrimitive<A>,
{
    fn k_means(
        &self,
        instance_axis: Axis,
        k: usize,
        mut initial_centroids: impl InitialCentroids<A>,
        stop_condition: impl StopCondition<A>,
    ) -> (Array2<A>, A) {
        assert!(
            k <= self.len_of(instance_axis) && k != 0,
            "k cannot be larger than the number of data points or zero"
        );

        let mut centroids = initial_centroids.initial_centroids(self.view(), instance_axis, k);
        let loss = self.kmeans_with_centroids(instance_axis, centroids.view_mut(), stop_condition);
        (centroids, loss)
    }
}

/// Trait for k-means clustering with an initial set of centroids.
///
/// Performs k-means clustering on the matrix of instances along
/// `instance_axis` using the given `centroids`.
///
/// Returns the mean squared error.
pub trait KMeansWithCentroids<A> {
    fn kmeans_with_centroids(
        &self,
        instance_axis: Axis,
        centroids: ArrayViewMut2<A>,
        stop_condition: impl StopCondition<A>,
    ) -> A;
}

impl<S, A> KMeansWithCentroids<A> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: NdFloat + Sum,
    usize: AsPrimitive<A>,
{
    fn kmeans_with_centroids(
        &self,
        instance_axis: Axis,
        mut centroids: ArrayViewMut2<A>,
        mut stop_condition: impl StopCondition<A>,
    ) -> A {
        assert!(
            centroids.nrows() > 0,
            "Cannot cluster instances with zero centroids."
        );
        assert_eq!(
            centroids.ncols(),
            self.len_of(Axis(instance_axis.index() ^ 1)),
            "Centroid and instance lengths differ."
        );

        for iter in 0.. {
            let loss = self.kmeans_iteration(instance_axis, centroids.view_mut());
            if stop_condition.should_stop(iter + 1, loss) {
                return loss;
            }
        }

        unreachable!()
    }
}

/// Trait for types that implement a single k-means step.
pub trait KMeansIteration<A> {
    /// Perform a single iteration of k-means clustering.
    ///
    /// Performs a single iteration of k-means clustering on the
    /// matrix of instances along`instance_axis` using the given
    /// `centroids`.
    ///
    /// Returns the mean squared error.
    fn kmeans_iteration(&self, instance_axis: Axis, centroids: ArrayViewMut2<A>) -> A;
}

impl<S, A> KMeansIteration<A> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: NdFloat + Sum,
    usize: AsPrimitive<A>,
{
    fn kmeans_iteration(&self, instance_axis: Axis, mut centroids: ArrayViewMut2<A>) -> A {
        assert!(
            centroids.nrows() > 0,
            "Cannot cluster instances with zero centroids."
        );
        assert_eq!(
            centroids.ncols(),
            self.len_of(Axis(instance_axis.index() ^ 1)),
            "Centroid and instance lengths differ."
        );

        let assignments = cluster_assignments(centroids.view(), self.view(), instance_axis);
        update_centroids(
            centroids.view_mut(),
            self.view(),
            instance_axis,
            assignments.view(),
        );
        mean_squared_error(centroids.view(), self.view(), instance_axis, assignments)
    }
}

fn mean_squared_error<A, S>(
    centroids: ArrayView2<A>,
    instances: ArrayView2<A>,
    instance_axis: Axis,
    assignments: ArrayBase<S, Ix1>,
) -> A
where
    A: NdFloat + Sum,
    usize: AsPrimitive<A>,
    S: Data<Elem = usize>,
{
    // Get the centroids representing the instances. Future: do not
    // construct an explicit matrix. Though I guess that it is
    // potentially optimized away due to the fold below.
    let mut errors = centroids.select(
        Axis(0),
        assignments.as_slice().expect("Non-contiguous vector"),
    );

    // Absolute errors.
    match instance_axis {
        Axis(0) => errors -= &instances,
        Axis(1) => errors -= &instances.t(),
        _ => unreachable!(),
    }

    // Summed squared error
    let sse = errors.into_iter().map(|&v| v * v).sum::<A>();

    sse / instances.len().as_()
}

#[cfg(test)]
mod tests {
    use ndarray::{array, concatenate, Array2, ArrayBase, Axis, Data, Ix2};
    use rand::{Rng, SeedableRng};
    use rand_distr::Normal;
    use rand_xorshift::XorShiftRng;

    use super::{
        cluster_assignments, mean_squared_error, update_centroids, KMeans, NIterationsCondition,
        RandomInstanceCentroids,
    };
    use crate::ndarray_rand::RandomExt;

    const SEED: [u8; 16] = [
        0xd3, 0x68, 0x34, 0x05, 0xf2, 0x6e, 0xa4, 0x45, 0x2b, 0x2b, 0xea, 0x1f, 0x08, 0xce, 0x88,
        0xf6,
    ];

    #[test]
    fn correct_cluster_assignments() {
        let centroids = array![[0.5, 0., 0.], [0., -1., 0.], [0., 0., 1.], [0., 1., 1.]];
        let instances = array![
            [0., 0.5, 0.],
            [0., 0., 2.],
            [1., 0., 0.],
            [0., 0., 1.],
            [0., -2., 0.],
            [0., 0.7, 0.7],
            [0., 0., 0.]
        ];

        // Test instances along axis 0.
        let assignments = cluster_assignments(centroids.view(), instances.view(), Axis(0));
        assert_eq!(assignments, array![0, 2, 0, 2, 1, 3, 0]);

        // Test instances along axis 1.
        let assignments = cluster_assignments(centroids.view(), instances.t(), Axis(1));
        assert_eq!(assignments, array![0, 2, 0, 2, 1, 3, 0]);
    }

    #[test]
    fn correct_update_centroids() {
        let mut centroids = array![[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let instances = array![
            [-1., -1., 0.],
            [1., 1., 0.],
            [-2., -1., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 2.],
        ];
        let assignments = array![1, 0, 1, 0, 2, 2];

        // Test instances along axis 0.
        update_centroids(
            centroids.view_mut(),
            instances.view(),
            Axis(0),
            assignments.view(),
        );

        assert_eq!(
            centroids,
            array![[0.5, 0.5, 0.], [-1.5, -1., 0.], [0., 0., 1.5]]
        );

        // Test instances along axis 1.
        update_centroids(centroids.view_mut(), instances.t(), Axis(1), assignments);

        assert_eq!(
            centroids,
            array![[0.5, 0.5, 0.], [-1.5, -1., 0.], [0., 0., 1.5]]
        );
    }

    fn gaussian_spheres<S>(centers: ArrayBase<S, Ix2>, mut rng: &mut impl Rng) -> Array2<f64>
    where
        S: Data<Elem = f64>,
    {
        let n_samples = 11;

        let mut spheres = Vec::new();
        for center in centers.outer_iter() {
            let mut sphere = Array2::random_using(
                (n_samples, center.len()),
                Normal::new(0., 0.01).unwrap(),
                &mut rng,
            );
            sphere += &center;
            spheres.push(sphere);
        }

        let sphere_views: Vec<_> = spheres.iter().map(|s| s.view()).collect();

        concatenate(Axis(0), &sphere_views).expect("Shapes of gaussian spheres do not match")
    }

    #[test]
    fn k_means_3() {
        let mut rng = XorShiftRng::from_seed(SEED);

        let gaussians = gaussian_spheres(array![[0., 0.], [1., 0.], [1., 1.]], &mut rng);

        let random_centroids = RandomInstanceCentroids::new(rng);
        let mut centroids: Vec<_> = gaussians
            .k_means(Axis(0), 3, random_centroids, NIterationsCondition(10))
            .0
            // Round centroids to nearest integer.
            .map(|v| v.round() as isize)
            // Convert rows to Vec.
            .outer_iter()
            .map(|r| r.to_vec())
            .collect();
        centroids.sort();

        // k-means can find a worse local minimum, but we are using a fixed seed.
        assert_eq!(centroids, [[0, 0], [1, 0], [1, 1]]);
    }

    #[test]
    fn k_means_3_axis1() {
        let mut rng = XorShiftRng::from_seed(SEED);

        let gaussians = gaussian_spheres(array![[0., 0.], [1., 0.], [1., 1.]], &mut rng);

        let random_centroids = RandomInstanceCentroids::new(rng);
        let mut centroids: Vec<_> = gaussians
            .t()
            .k_means(Axis(1), 3, random_centroids, NIterationsCondition(10))
            .0
            // Round centroids to nearest integer.
            .map(|v| v.round() as isize)
            // Convert rows to Vec.
            .outer_iter()
            .map(|r| r.to_vec())
            .collect();
        centroids.sort();

        // k-means can find a worse local minimum, but we are using a fixed seed.
        assert_eq!(centroids, [[0, 0], [1, 0], [1, 1]]);
    }

    #[test]
    fn correct_mean_squared_error() {
        let centroids = array![[-1., 2., 0.], [0., -1., 1.]];
        let instances = array![[-1., 1., 1.], [0., 1., 0.]];

        let mse = mean_squared_error(centroids.view(), instances.view(), Axis(0), array![1, 0]);
        assert_eq!(mse, 7. / 6.);

        let mse = mean_squared_error(
            centroids.view(),
            instances.view().t(),
            Axis(1),
            array![1, 0],
        );
        assert_eq!(mse, 7. / 6.);
    }
}
