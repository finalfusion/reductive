//! K-means clustering.

use std::collections::HashSet;
use std::iter::Sum;

use ndarray::{Array1, Array2, ArrayBase, ArrayView2, ArrayViewMut2, Axis, Data, Ix2, NdFloat};
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

/// Find nearest cluster centroid for each instance.
///
/// Find nearest centroid for each instance along `instance_axis` of
/// `instances`. Returns for each instance the index of the nearest
/// cluster centroid.
fn cluster_assignments<A>(
    centroids: ArrayView2<A>,
    instances: ArrayView2<A>,
    instance_axis: Axis,
) -> Vec<usize>
where
    A: NdFloat + Sum,
{
    let mut assignments = Vec::with_capacity(instances.len_of(instance_axis));

    let dists = if instance_axis == Axis(0) {
        instances.squared_euclidean_distance(centroids)
    } else {
        instances.t().squared_euclidean_distance(centroids)
    };

    for inst_dists in dists.outer_iter() {
        assignments.push(
            inst_dists
                .iter()
                .enumerate()
                .min_by_key(|v| OrderedFloat(*v.1))
                .unwrap()
                .0,
        );
    }

    assignments
}

/// Update centroids to the mean of the assigned data points.
///
/// `instance_axis` is the instance axis of `data`. The centroids
/// are row-based. `assignments` contains an assignment for each
/// data point.
fn update_centroids<A>(
    mut centroids: ArrayViewMut2<A>,
    data: ArrayView2<A>,
    instance_axis: Axis,
    assignments: &[usize],
) where
    A: NdFloat,
{
    assert_eq!(
        assignments.len(),
        data.len_of(instance_axis),
        "The number of assignments should be equal to the number of instances."
    );

    centroids.fill(A::zero());

    let mut centroid_counts = Array1::zeros(centroids.rows());

    for (instance, assignment) in data.axis_iter(instance_axis).zip(assignments) {
        let mut centroid = centroids.index_axis_mut(Axis(0), *assignment);
        centroid += &instance;
        centroid_counts[*assignment] += A::one();
    }

    for (mut centroid, centroid_count) in centroids
        .axis_iter_mut(instance_axis)
        .zip(centroid_counts.outer_iter())
    {
        if centroid_count[()] > A::zero() {
            centroid /= &centroid_count;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis};

    use super::{cluster_assignments, update_centroids};

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
        assert_eq!(assignments, &[0, 2, 0, 2, 1, 3, 0]);

        // Test instances along axis 1.
        let assignments = cluster_assignments(centroids.view(), instances.t(), Axis(1));
        assert_eq!(assignments, &[0, 2, 0, 2, 1, 3, 0]);
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
        let assignments = vec![1, 0, 1, 0, 2, 2];

        // Test instances along axis 0.
        update_centroids(
            centroids.view_mut(),
            instances.view(),
            Axis(0),
            &assignments,
        );

        assert_eq!(
            centroids,
            array![[0.5, 0.5, 0.], [-1.5, -1., 0.], [0., 0., 1.5]]
        );

        // Test instances along axis 1.
        update_centroids(centroids.view_mut(), instances.t(), Axis(1), &assignments);

        assert_eq!(
            centroids,
            array![[0.5, 0.5, 0.], [-1.5, -1., 0.], [0., 0., 1.5]]
        );
    }
}
