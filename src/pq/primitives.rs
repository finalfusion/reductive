use std::iter::Sum;

#[cfg(feature = "opq-train")]
use ndarray::Array2;
use ndarray::{
    azip, s, Array1, ArrayBase, ArrayView3, ArrayViewMut2, Axis, Data, Ix1, Ix2, NdFloat,
};

use num_traits::{AsPrimitive, Bounded, Zero};

use crate::kmeans::{cluster_assignment, cluster_assignments};

pub fn quantize<A, I, S>(
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

#[cfg(feature = "opq-train")]
pub fn quantize_batch<A, I, S>(quantizers: ArrayView3<A>, x: ArrayBase<S, Ix2>) -> Array2<I>
where
    A: NdFloat + Sum,
    I: 'static + AsPrimitive<usize> + Bounded + Copy + Zero,
    S: Data<Elem = A>,
    usize: AsPrimitive<I>,
{
    let mut quantized = Array2::zeros((x.rows(), quantizers.len_of(Axis(0))));
    quantize_batch_into(quantizers, x, quantized.view_mut());
    quantized
}

pub fn quantize_batch_into<A, I, S>(
    quantizers: ArrayView3<A>,
    x: ArrayBase<S, Ix2>,
    mut quantized: ArrayViewMut2<I>,
) where
    A: NdFloat + Sum,
    I: 'static + AsPrimitive<usize> + Bounded + Copy + Zero,
    S: Data<Elem = A>,
    usize: AsPrimitive<I>,
{
    assert_eq!(
        reconstructed_len(quantizers.view()),
        x.cols(),
        "Quantizer and vector length mismatch"
    );

    assert!(
        quantized.rows() == x.rows() && quantized.cols() == quantizers.len_of(Axis(0)),
        "Quantized matrix has incorrect shape, expected: ({}, {}), got: ({}, {})",
        x.rows(),
        quantizers.len_of(Axis(0)),
        quantized.rows(),
        quantized.cols()
    );

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
}

pub fn reconstructed_len<A>(quantizers: ArrayView3<A>) -> usize {
    quantizers.len_of(Axis(0)) * quantizers.len_of(Axis(2))
}

pub fn reconstruct<A, I, S>(quantizers: ArrayView3<A>, quantized: ArrayBase<S, Ix1>) -> Array1<A>
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

    let mut reconstruct = Vec::with_capacity(reconstructed_len(quantizers.view()));
    for (&centroid, quantizer) in quantized.into_iter().zip(quantizers.outer_iter()) {
        reconstruct.extend(quantizer.index_axis(Axis(0), centroid.as_()));
    }

    Array1::from_vec(reconstruct)
}

pub fn reconstruct_batch_into<A, I, S>(
    quantizers: ArrayView3<A>,
    quantized: ArrayBase<S, Ix2>,
    mut reconstructions: ArrayViewMut2<A>,
) where
    A: NdFloat,
    I: AsPrimitive<usize>,
    S: Data<Elem = I>,
{
    assert!(
        reconstructions.rows() == quantized.rows()
            && reconstructions.cols() == reconstructed_len(quantizers.view()),
        "Reconstructions matrix has incorrect shape, expected: ({}, {}), got: ({}, {})",
        quantized.rows(),
        reconstructed_len(quantizers.view()),
        reconstructions.rows(),
        reconstructions.cols()
    );

    for (quantized, mut reconstruction) in
        quantized.outer_iter().zip(reconstructions.outer_iter_mut())
    {
        reconstruction.assign(&reconstruct(quantizers, quantized));
    }
}
