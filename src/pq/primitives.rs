use std::iter::Sum;

#[cfg(feature = "opq-train")]
use ndarray::Array2;
use ndarray::{
    s, Array1, ArrayBase, ArrayView3, ArrayViewMut2, Axis, Data, Ix1, Ix2, NdFloat, Zip,
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
        let sub_vec = x.slice(s![offset..offset + quantizer.ncols()]);
        *index = cluster_assignment(quantizer.view(), sub_vec).as_();

        offset += quantizer.ncols();
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
    let mut quantized = Array2::zeros((x.nrows(), quantizers.len_of(Axis(0))));
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
        x.ncols(),
        "Quantizer and vector length mismatch"
    );

    assert!(
        quantized.nrows() == x.nrows() && quantized.ncols() == quantizers.len_of(Axis(0)),
        "Quantized matrix has incorrect shape, expected: ({}, {}), got: ({}, {})",
        x.nrows(),
        quantizers.len_of(Axis(0)),
        quantized.nrows(),
        quantized.ncols()
    );

    let mut offset = 0;
    for (quantizer, mut quantized) in quantizers
        .outer_iter()
        .zip(quantized.axis_iter_mut(Axis(1)))
    {
        // ndarray#474
        #[allow(clippy::deref_addrof)]
        let sub_matrix = x.slice(s![.., offset..offset + quantizer.ncols()]);
        let assignments = cluster_assignments(quantizer.view(), sub_matrix, Axis(0));
        Zip::from(&mut quantized)
            .and(&assignments)
            .for_each(|quantized, assignment| *quantized = assignment.as_());

        offset += quantizer.ncols();
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

    Array1::from(reconstruct)
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
        reconstructions.nrows() == quantized.nrows()
            && reconstructions.ncols() == reconstructed_len(quantizers.view()),
        "Reconstructions matrix has incorrect shape, expected: ({}, {}), got: ({}, {})",
        quantized.nrows(),
        reconstructed_len(quantizers.view()),
        reconstructions.nrows(),
        reconstructions.ncols()
    );

    for (quantized, mut reconstruction) in
        quantized.outer_iter().zip(reconstructions.outer_iter_mut())
    {
        reconstruction.assign(&reconstruct(quantizers, quantized));
    }
}
