#![feature(test)]

extern crate test;

use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use test::Bencher;

use reductive::linalg::Covariance;

#[bench]
fn covariance_axis0(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((50, 100), Normal::new(1., 0.2));

    bencher.iter(|| {
        data.view().covariance(Axis(0));
    })
}

#[bench]
fn covariance_axis1(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((100, 50), Normal::new(1., 0.2));

    bencher.iter(|| {
        data.view().covariance(Axis(1));
    })
}
