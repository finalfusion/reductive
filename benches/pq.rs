use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use rand_distr::Normal;

use reductive::ndarray_rand::RandomExt;
use reductive::pq::{Pq, QuantizeVector, Reconstruct, TrainPq};

pub fn create_quantizer() -> (Pq<f32>, Array2<f32>) {
    let data: Array2<f32> = Array2::random((100, 128), Normal::new(0., 1.).unwrap());
    let pq = Pq::train_pq(16, 4, 10, 1, data.view()).unwrap();

    (pq, data)
}

fn pq_quantize(c: &mut Criterion) {
    let (pq, data) = create_quantizer();

    c.bench_function("pq_quantize", |b| {
        b.iter(|| {
            for v in data.outer_iter() {
                let _: Array1<u8> = pq.quantize_vector(v);
            }
        })
    });
}

fn pq_quantize_batch(c: &mut Criterion) {
    let (pq, data) = create_quantizer();

    c.bench_function("pq_quantize_batch", |b| {
        b.iter(|| pq.quantize_batch::<u8, _>(data.view()))
    });
}

fn pq_reconstruct(c: &mut Criterion) {
    let (pq, data) = create_quantizer();
    let quantized = pq.quantize_batch::<u8, _>(data);

    c.bench_function("pq_reconstruct", |b| {
        b.iter(|| {
            for q in quantized.outer_iter() {
                pq.reconstruct(q);
            }
        })
    });
}

fn pq_reconstruct_batch(c: &mut Criterion) {
    let (pq, data) = create_quantizer();
    let quantized = pq.quantize_batch::<u8, _>(data);

    c.bench_function("pq_reconstruct_batch", |b| {
        b.iter(|| pq.reconstruct_batch(quantized.view()))
    });
}

criterion_group!(
    benches,
    pq_quantize,
    pq_quantize_batch,
    pq_reconstruct,
    pq_reconstruct_batch
);
criterion_main!(benches);
