use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Axis};
use rand_distr::Normal;

use reductive::linalg::{Covariance, SquaredEuclideanDistance};
use reductive::ndarray_rand::RandomExt;

fn covariance_axis0(c: &mut Criterion) {
    let data: Array2<f64> = Array2::random((50, 100), Normal::new(1., 0.2).unwrap());

    c.bench_function("covariance_axis0", |b| {
        b.iter(|| data.view().covariance(Axis(0)))
    });
}

fn covariance_axis1(c: &mut Criterion) {
    let data: Array2<f64> = Array2::random((100, 50), Normal::new(1., 0.2).unwrap());

    c.bench_function("covariance_axis1", |b| {
        b.iter(|| data.view().covariance(Axis(1)))
    });
}

fn squared_euclidean_distance_ix1_ix1(c: &mut Criterion) {
    let data1: Array2<f64> = Array2::random((200, 50), Normal::new(1., 0.2).unwrap());
    let data2: Array2<f64> = Array2::random((50, 50), Normal::new(1., 0.2).unwrap());

    c.bench_function("squared_euclidean_distance_ix1_ix1", |b| {
        b.iter(|| {
            for row1 in data1.outer_iter() {
                for row2 in data2.outer_iter() {
                    row1.squared_euclidean_distance(row2);
                }
            }
        })
    });
}

fn squared_euclidean_distance_ix1_ix2(c: &mut Criterion) {
    let data1: Array2<f64> = Array2::random((200, 50), Normal::new(1., 0.2).unwrap());
    let data2: Array2<f64> = Array2::random((50, 50), Normal::new(1., 0.2).unwrap());

    c.bench_function("squared_euclidean_distance_ix1_ix2", |b| {
        b.iter(|| {
            for row in data1.outer_iter() {
                row.squared_euclidean_distance(data2.view());
            }
        })
    });
}

fn squared_euclidean_distance_ix2_ix2(c: &mut Criterion) {
    let data1: Array2<f64> = Array2::random((200, 50), Normal::new(1., 0.2).unwrap());
    let data2: Array2<f64> = Array2::random((50, 50), Normal::new(1., 0.2).unwrap());

    c.bench_function("squared_euclidean_distance_ix2_ix2", |b| {
        b.iter(|| data1.view().squared_euclidean_distance(data2.view()))
    });
}

criterion_group!(
    benches,
    covariance_axis0,
    covariance_axis1,
    squared_euclidean_distance_ix1_ix2,
    squared_euclidean_distance_ix2_ix2,
    squared_euclidean_distance_ix1_ix1
);
criterion_main!(benches);
