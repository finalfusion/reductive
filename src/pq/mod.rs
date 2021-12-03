//! Product quantization.

#[cfg(feature = "opq-train")]
mod gaussian_opq;
#[cfg(feature = "opq-train")]
pub use gaussian_opq::GaussianOpq;

#[cfg(feature = "opq-train")]
mod opq;
#[cfg(feature = "opq-train")]
pub use self::opq::Opq;

pub(crate) mod primitives;

#[allow(clippy::module_inception)]
mod pq;
pub use self::pq::Pq;

mod traits;
pub use self::traits::{QuantizeVector, Reconstruct, TrainPq};
