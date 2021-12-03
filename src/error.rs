use thiserror::Error;

/// Reductive error type.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ReductiveError {
    #[error("The number of quantization attempts per iteration must be >= 1")]
    IncorrectNAttempts,

    #[error("The number of quantization iterations must be >= 1")]
    IncorrectNIterations,

    #[error(
        "The number of subquantizers bits must be between 1 and {}",
        max_subquantizer_bits
    )]
    IncorrectNSubquantizerBits { max_subquantizer_bits: u32 },

    #[error(
        "The number of columns ({}) is not exactly dividable by the number of subquantizers ({})",
        n_subquantizers,
        n_columns
    )]
    IncorrectNumberSubquantizers {
        n_subquantizers: usize,
        n_columns: usize,
    },

    #[error(
        "The number of subquantizers must be between 1 and {}, was {}",
        max_subquantizers,
        n_subquantizers
    )]
    NSubquantizersOutsideRange {
        n_subquantizers: usize,
        max_subquantizers: usize,
    },

    #[error("Cannot initialize random number generator for quantization")]
    ConstructRng(#[source] rand::Error),
}
