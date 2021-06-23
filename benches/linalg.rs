#![feature(test)]

extern crate test;

use ndarray::{Array2, Axis};
use rand_distr::Normal;
use test::Bencher;

use reductive::linalg::{Covariance, SquaredEuclideanDistance};
use reductive::ndarray_rand::RandomExt;

#[bench]
fn covariance_axis0(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((50, 100), Normal::new(1., 0.2).unwrap());

    bencher.iter(|| {
        data.view().covariance(Axis(0));
    })
}

#[bench]
fn covariance_axis1(bencher: &mut Bencher) {
    let data: Array2<f64> = Array2::random((100, 50), Normal::new(1., 0.2).unwrap());

    bencher.iter(|| {
        data.view().covariance(Axis(1));
    })
}

#[bench]
fn squared_euclidean_distance_ix1_ix1(bencher: &mut Bencher) {
    let data1: Array2<f64> = Array2::random((200, 50), Normal::new(1., 0.2).unwrap());
    let data2: Array2<f64> = Array2::random((50, 50), Normal::new(1., 0.2).unwrap());

    bencher.iter(|| {
        for row1 in data1.outer_iter() {
            for row2 in data2.outer_iter() {
                row1.squared_euclidean_distance(row2);
            }
        }
    })
}

#[bench]
fn squared_euclidean_distance_ix1_ix2(bencher: &mut Bencher) {
    let data1: Array2<f64> = Array2::random((200, 50), Normal::new(1., 0.2).unwrap());
    let data2: Array2<f64> = Array2::random((50, 50), Normal::new(1., 0.2).unwrap());

    bencher.iter(|| {
        for row in data1.outer_iter() {
            row.squared_euclidean_distance(data2.view());
        }
    })
}

#[bench]
fn squared_euclidean_distance_ix2_ix2(bencher: &mut Bencher) {
    let data1: Array2<f64> = Array2::random((200, 50), Normal::new(1., 0.2).unwrap());
    let data2: Array2<f64> = Array2::random((50, 50), Normal::new(1., 0.2).unwrap());

    bencher.iter(|| {
        data1.view().squared_euclidean_distance(data2.view());
    })
}
