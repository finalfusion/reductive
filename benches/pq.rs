use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use rand_distr::Normal;

use reductive::ndarray_rand::RandomExt;
use reductive::pq::{Pq, QuantizeVector, TrainPq};

fn pq_quantize(c: &mut Criterion) {
    let data: Array2<f64> = Array2::random((100, 128), Normal::new(0., 1.).unwrap());
    let pq = Pq::train_pq(16, 4, 10, 1, data.view()).unwrap();

    c.bench_function("pq_quantize", |b| {
        b.iter(|| {
            for v in data.outer_iter() {
                let _: Array1<u8> = pq.quantize_vector(v);
            }
        })
    });
}

fn pq_quantize_batch(c: &mut Criterion) {
    let data: Array2<f64> = Array2::random((100, 128), Normal::new(0., 1.).unwrap());
    let pq = Pq::train_pq(16, 4, 10, 1, data.view()).unwrap();

    c.bench_function("pq_quantize_batch", |b| {
        b.iter(|| pq.quantize_batch::<u8, _>(data.view()))
    });
}

criterion_group!(benches, pq_quantize, pq_quantize_batch);
criterion_main!(benches);
