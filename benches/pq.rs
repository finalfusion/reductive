#![feature(test)]

extern crate test;

use ndarray::{Array1, Array2};
use rand_distr::Normal;
use test::Bencher;

use reductive::ndarray_rand::RandomExt;
use reductive::pq::{QuantizeVector, TrainPq, Pq};

#[bench]
fn pq_quantize(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((100, 128), Normal::new(0., 1.).unwrap());
    let pq = Pq::train_pq(16, 4, 10, 1, data.view()).unwrap();

    bencher.iter(|| {
        for v in data.outer_iter() {
            let _: Array1<u8> = pq.quantize_vector(v);
        }
    })
}

#[bench]
fn pq_quantize_batch(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((100, 128), Normal::new(0., 1.).unwrap());
    let pq = Pq::train_pq(16, 4, 10, 1, data.view()).unwrap();

    bencher.iter(|| {
        let _: Array2<u8> = pq.quantize_batch(data.view());
    })
}
