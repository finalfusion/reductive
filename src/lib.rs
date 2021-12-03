#[cfg(feature = "accelerate-test")]
extern crate accelerate_src;

pub mod error;

pub mod kmeans;

pub mod linalg;

#[doc(hidden)]
pub mod ndarray_rand;

pub mod pq;
