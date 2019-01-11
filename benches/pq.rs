#![feature(test)]

extern crate test;

use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use test::Bencher;

use reductive::pq::{QuantizeVector, TrainPQ, PQ};

#[bench]
fn pq_quantize(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((100, 128), Normal::new(0., 1.));
    let pq = PQ::train_pq(16, 4, 10, 1, data.view());

    bencher.iter(|| {
        for v in data.outer_iter() {
            pq.quantize_vector(v);
        }
    })
}

#[bench]
fn pq_quantize_batch(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((100, 128), Normal::new(0., 1.));
    let pq = PQ::train_pq(16, 4, 10, 1, data.view());

    bencher.iter(|| {
        pq.quantize_batch(data.view());
    })
}
