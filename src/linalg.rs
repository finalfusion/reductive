//! Various linear algebra utility traits.

use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, NdFloat};
use num_traits::AsPrimitive;

/// Trait for computing covariance matrices.
pub trait Covariance<A> {
    /// Compute the covariance matrix the matrix.
    ///
    /// Consider an *n × m* matrix A. If *n* is the observation axis and
    /// *m* the variable axis, then this method returns an *m × m* covariance
    /// matrix *C*. *C_ij* is the the covariance between variables *i* and *j*
    /// and *C_ii* the variance of variable *i*.
    fn covariance(self, observation_axis: Axis) -> Array2<A>;
}

impl<S, A> Covariance<A> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: NdFloat,
    usize: AsPrimitive<A>,
{
    fn covariance(self, observation_axis: Axis) -> Array2<A> {
        // Center the data
        let means = self.mean_axis(observation_axis);
        let mut centered = self.to_owned();
        centered
            .axis_iter_mut(observation_axis)
            .for_each(|mut o| o -= &means);

        let normalization = self.len_of(observation_axis).as_() - A::one();

        // Compute the covariance matrix.
        if observation_axis == Axis(0) {
            centered.t().dot(&centered.map(|v| *v / normalization))
        } else {
            centered.dot(&centered.t().map(|v| *v / normalization))
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis};

    use super::Covariance;

    #[test]
    fn covariance() {
        let x = array![[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]];
        let cov = x.view().covariance(Axis(0));
        assert_eq!(cov, array![[1., -1.], [-1., 1.]]);

        let cov = x.t().covariance(Axis(1));
        assert_eq!(cov, array![[1., -1.], [-1., 1.]]);
    }
}
