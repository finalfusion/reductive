use std::convert::TryFrom;

use ndarray::{s, Array1, Array2, Array3, ArrayBase, Data, Ix1, Ix2, LinalgScalar, NdFloat};
use num_traits::AsPrimitive;

use super::{ReconstructVector, PQ};

#[derive(Clone, Debug, PartialEq)]
pub enum PrecomputeError {
    NoProjection,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrecomputedPQ<A> {
    quantizer_len: usize,

    // In a non-precomputed quantizer, each subquantizer has a shape
    // of:
    //
    // n_centroids x dimensionality. In a precomputed
    //
    // subquantizer, we store for each centroid the dot product of
    // that centroid and and each (sub)-row of the projection
    // matrix. Thus leading to a subquantizer of:
    //
    // n_centroids x n_projection_rows x 1
    //
    // So even though the dimensionality of the subquantizers appear
    // the same, the meaning of the inner dimension is different.
    quantizers: Array3<A>,
}

impl<A> ReconstructVector<A> for PrecomputedPQ<A>
where
    A: NdFloat,
{
    fn reconstruct_batch<I, S>(&self, quantized: ArrayBase<S, Ix2>) -> Array2<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        assert_eq!(
            self.quantizers.dim().0,
            quantized.cols(),
            "Quantization length does not match number of subquantizers"
        );
        let mut reconstructions = Array2::zeros((quantized.rows(), self.quantizer_len));

        for (quantized, mut reconstruction) in
            quantized.outer_iter().zip(reconstructions.outer_iter_mut())
        {
            reconstruction.assign(&self.reconstruct_vector(quantized));
        }

        reconstructions
    }

    fn reconstruct_vector<I, S>(&self, quantized: ArrayBase<S, Ix1>) -> Array1<A>
    where
        I: AsPrimitive<usize>,
        S: Data<Elem = I>,
    {
        let mut reconstructed: Array1<A> = Array1::zeros((self.quantizer_len,));

        for (&centroid, quantizer) in quantized.into_iter().zip(self.quantizers.outer_iter()) {
            reconstructed += &quantizer.row(centroid.as_())
        }

        reconstructed
    }

    fn reconstructed_len(&self) -> usize {
        self.quantizer_len
    }
}

impl<A> TryFrom<PQ<A>> for PrecomputedPQ<A>
where
    A: LinalgScalar,
{
    type Error = PrecomputeError;

    fn try_from(pq: PQ<A>) -> Result<Self, Self::Error> {
        let projection = match pq.projection {
            Some(projection) => projection,
            None => return Err(PrecomputeError::NoProjection),
        };

        let n_subquantizers = pq.quantizers.len();
        // Get the reconstruction length and number of centroids of the subquantizer.
        let (n_centroids, sq_len) = pq.quantizers[0].dim();

        let mut quantizers = Vec::with_capacity(n_subquantizers * n_centroids * projection.rows());

        for (sq_idx, sq) in pq.quantizers.into_iter().enumerate() {
            // Get the relevant slice of the projection matrix.
            let offset = sq_idx * sq_len;
            let projection_slice = projection.slice(s![.., offset..offset + sq_len]);

            for centroid in sq.outer_iter() {
                for projection_row in projection_slice.outer_iter() {
                    quantizers.push(projection_row.dot(&centroid));
                }
            }
        }

        Ok(PrecomputedPQ {
            quantizer_len: pq.quantizer_len,
            quantizers: Array3::from_shape_vec(
                (n_subquantizers, n_centroids, projection.rows()),
                quantizers,
            )
            .expect("Incorrect subquantizer shape"),
        })
    }
}

#[cfg(test)]
#[cfg(feature = "opq-train")]
mod tests {
    use std::convert::TryInto;

    use ndarray::{Array2, ArrayView2};
    use rand::distributions::Uniform;

    use super::PrecomputedPQ;
    use crate::linalg::EuclideanDistance;
    use crate::ndarray_rand::RandomExt;
    use crate::pq::{QuantizeVector, ReconstructVector, TrainPQ, OPQ};

    /// calculate the average euclidean distances between the the given
    /// instances and the instances returned by quantizing and then
    /// reconstructing the instances.
    fn avg_euclidean_loss(instances: ArrayView2<f32>, reconstructions: ArrayView2<f32>) -> f32 {
        let mut euclidean_loss = 0f32;

        for (instance, reconstruction) in instances.outer_iter().zip(reconstructions.outer_iter()) {
            euclidean_loss += instance.euclidean_distance(reconstruction);
        }

        euclidean_loss / instances.rows() as f32
    }

    #[test]
    fn precompute_from_opq() {
        let uniform = Uniform::new(0f32, 1f32);
        let instances = Array2::random((256, 20), uniform);
        let pq = OPQ::train_pq(10, 7, 10, 1, instances.view());
        let quantized: Array2<u8> = pq.quantize_batch(instances.view());

        let precomputed_pq: PrecomputedPQ<f32> = pq.try_into().unwrap();
        let reconstructed = precomputed_pq.reconstruct_batch(quantized);

        let loss = avg_euclidean_loss(instances.view(), reconstructed.view());
        // loss is around 0.09.
        assert!(loss < 0.1);
    }
}
