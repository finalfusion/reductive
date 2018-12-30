//! K-means clustering.

use std::collections::HashSet;

use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, NdFloat};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

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
