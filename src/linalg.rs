//! Various linear algebra utility traits.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2, NdFloat};
use num_traits::{AsPrimitive, FromPrimitive};

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
    A: FromPrimitive + NdFloat,
    usize: AsPrimitive<A>,
{
    fn covariance(self, observation_axis: Axis) -> Array2<A> {
        assert!(
            self.len_of(observation_axis) != 0,
            "Cannot compute a covariance from zero observations"
        );

        // Center the data
        let means = self.mean_axis(observation_axis).unwrap();
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

/// Squared euclidean distance *|u-v|^2*.
///
/// Computes the squared euclidean distances between two arrays.
///
/// * If `self` and `other` are vectors, a scalar is returned.
/// * If `self` is a vector and `other` a matrix, a vector of distances
///  between `self` and the rows of `other` is returned.
/// * If `self` and `other` are both matrices, a matrix of distances
///   is returned were *(i, j)* is the distance between row *i* of
///   `self` and row *j* of `other`.
pub trait SquaredEuclideanDistance<A, D> {
    type Output;

    /// Compute the squared Euclidean distance(s).
    fn squared_euclidean_distance<S>(&self, other: ArrayBase<S, D>) -> Self::Output
    where
        S: Data<Elem = A>;
}

// Note on the implementations of SquaredEuclideanDistance below.
//
// Rather than calculating the squared Euclidean distance in the
// obvious manner (|u-v|^2), we rewrite the distance in terms of the
// dot product u·v:
//
//   |u-v|^2
// = (u-v)·(u-v)
// = ∑[(u_i-v_i)(u_i-v_i)]
// = ∑[u_i^2 + v_i^2 - 2 u_i v_i]
// = u·u + v·v - 2u·v
// = |u|^2 + |v|^2 - 2u·v
//
// This is just the law of cosines, since
// u·v = |u| |v| cos ∠(u,v)
//
// When computing the distances between every row in the self matrix
// and every row in the other matrix, we can rely on this formulation
// to do the heavy lifting in matrix multiplication, which uses
// highly-optimized kernels in BLAS implementations such as Intel MKL
// and OpenBLAS, but also in vanilla ndarray.

impl<A, S1> SquaredEuclideanDistance<A, Ix1> for ArrayBase<S1, Ix1>
where
    A: NdFloat,
    S1: Data<Elem = A>,
{
    type Output = A;

    #[allow(clippy::suspicious_operation_groupings)]
    fn squared_euclidean_distance<S2>(&self, other: ArrayBase<S2, Ix1>) -> A
    where
        S2: Data<Elem = A>,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "Cannot compute (squared) euclidean distance of vectors with different lengths."
        );

        // Also compute the euclidean distance using the law of cosines for
        // vectors. It is slightly faster than subtracting the vectors and
        // computing the norm.

        let self_sqn = self.dot(self);
        let other_sqn = other.dot(&other);
        let dp = self.dot(&other);

        self_sqn + other_sqn - (dp + dp)
    }
}

impl<A, S1> SquaredEuclideanDistance<A, Ix2> for ArrayBase<S1, Ix1>
where
    A: NdFloat,
    S1: Data<Elem = A>,
{
    type Output = Array1<A>;

    fn squared_euclidean_distance<S2>(&self, other: ArrayBase<S2, Ix2>) -> Self::Output
    where
        S2: Data<Elem = A>,
    {
        assert_eq!(
            self.len(),
            other.ncols(),
            "Cannot compute (squared) euclidean distance when the number of vector components and matrix columns differ."
        );

        // Compute the squared norms of self and the rows of the other matrix.
        let self_sqn = self.dot(self);
        let other_sqn: Array1<_> = other.outer_iter().map(|r| r.dot(&r)).collect();

        // Se comments for matrix-matrix squared euclidean distance:
        // |u - v|^2 = |u|^2 + |v|^2 - 2u·v
        let mut distances = other.dot(self);
        for i in 0..distances.len() {
            distances[i] = self_sqn + other_sqn[i] - (distances[i] + distances[i]);
        }

        distances
    }
}

impl<A, S1> SquaredEuclideanDistance<A, Ix2> for ArrayBase<S1, Ix2>
where
    A: NdFloat,
    S1: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn squared_euclidean_distance<S2>(&self, other: ArrayBase<S2, Ix2>) -> Self::Output
    where
        S2: Data<Elem = A>,
    {
        assert_eq!(
            self.ncols(),
            other.ncols(),
            "Cannot compute (squared) euclidean distance of matrices with different numbers of columns."
        );

        let self_sqn: Array1<_> = self.outer_iter().map(|r| r.dot(&r)).collect();
        let other_sqn: Array1<_> = other.outer_iter().map(|r| r.dot(&r)).collect();

        let mut distances = self.dot(&other.t());
        for i in 0..distances.nrows() {
            for j in 0..distances.ncols() {
                distances[(i, j)] =
                    self_sqn[i] + other_sqn[j] - (distances[(i, j)] + distances[(i, j)]);
            }
        }

        distances
    }
}

/// Trait for computing the euclidean distance *|u-v|*.
///
/// Computes the euclidean distances between two arrays.
///
/// * If `self` and `other` are vectors, a scalar is returned.
/// * If `self` is a vector and `other` a matrix, a vector of distances
///  between `self` and the rows of `other` is returned.
/// * If `self` and `other` are both matrices, a matrix of distances
///   is returned were *(i, j)* is the distance between row *i* of
///   `self` and row *j* of `other`.
pub trait EuclideanDistance<A, D> {
    type Output;

    /// Compute the Euclidean distance(s).
    fn euclidean_distance<S>(&self, other: ArrayBase<S, D>) -> Self::Output
    where
        S: Data<Elem = A>;
}

impl<A, S1> EuclideanDistance<A, Ix1> for ArrayBase<S1, Ix1>
where
    A: NdFloat,
    S1: Data<Elem = A>,
{
    type Output = A;

    fn euclidean_distance<S2>(&self, other: ArrayBase<S2, Ix1>) -> A
    where
        S2: Data<Elem = A>,
    {
        self.squared_euclidean_distance(other).sqrt()
    }
}

impl<A, S1> EuclideanDistance<A, Ix2> for ArrayBase<S1, Ix1>
where
    A: NdFloat,
    S1: Data<Elem = A>,
{
    type Output = Array1<A>;

    fn euclidean_distance<S>(&self, other: ArrayBase<S, Ix2>) -> Self::Output
    where
        S: Data<Elem = A>,
    {
        self.squared_euclidean_distance(other).mapv_into(A::sqrt)
    }
}

impl<A, S1> EuclideanDistance<A, Ix2> for ArrayBase<S1, Ix2>
where
    A: NdFloat,
    S1: Data<Elem = A>,
{
    type Output = Array2<A>;

    fn euclidean_distance<S>(&self, other: ArrayBase<S, Ix2>) -> Self::Output
    where
        S: Data<Elem = A>,
    {
        self.squared_euclidean_distance(other).mapv_into(A::sqrt)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Axis};

    use super::{Covariance, EuclideanDistance, SquaredEuclideanDistance};

    #[test]
    fn covariance() {
        let x = array![[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]];
        let cov = x.view().covariance(Axis(0));
        assert_eq!(cov, array![[1., -1.], [-1., 1.]]);

        let cov = x.t().covariance(Axis(1));
        assert_eq!(cov, array![[1., -1.], [-1., 1.]]);
    }

    #[test]
    fn euclidean_distance_ix1_ix1() {
        let a = array![1., 2., 3.];
        let b = array![0., 2., 0.];
        assert_eq!(a.euclidean_distance(b), 10f32.sqrt());
    }

    #[test]
    fn euclidean_distance_ix1_ix2() {
        let a = array![1., 2., 3.];
        let b = array![[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]];
        assert!(a
            .euclidean_distance(b)
            .abs_diff_eq(&array![14f32.sqrt(), 10f32.sqrt(), 6f32.sqrt()], 1e-6));
    }

    #[test]
    fn euclidean_distance_ix2_ix2() {
        let a = array![[1., 2., 3.], [3., 2., 1.]];
        let b = array![[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]];
        assert!(a.euclidean_distance(b).abs_diff_eq(
            &array![
                [14f32.sqrt(), 10f32.sqrt(), 6f32.sqrt()],
                [6f32.sqrt(), 10f32.sqrt(), 14f32.sqrt()]
            ],
            1e-6
        ));
    }

    #[test]
    fn squared_euclidean_distance_ix1_ix1() {
        let a = array![1., 2., 3.];
        let b = array![0., 2., 0.];
        assert_eq!(a.squared_euclidean_distance(b), 10f32);
    }

    #[test]
    fn squared_euclidean_distances_ix1_ix2() {
        let a = array![1., 2., 3.];
        let b = array![[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]];
        assert_eq!(a.squared_euclidean_distance(b), array![14., 10., 6.]);
    }

    #[test]
    fn squared_euclidean_distances_ix2_ix2() {
        let a = array![[1., 2., 3.], [3., 2., 1.]];
        let b = array![[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]];
        assert_eq!(
            a.squared_euclidean_distance(b),
            array![[14., 10., 6.], [6.0, 10.0, 14.0]]
        );
    }
}
