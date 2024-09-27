use ndarray_linalg::error::LinalgError;
use thiserror_no_std::Error;

/// Error manager.
#[derive(Error, Debug)]
pub enum NjangError {
    /// Linear algebra error
    #[error(transparent)]
    Linalg(#[from] LinalgError),

    /// Not supported item.
    #[error("{item} is not supported")]
    NotSupported { item: String },

    /// Shape
    #[error("The number of samples {xlen} and {ylen} should match.")]
    NotMatchedLength { xlen: usize, ylen: usize },
}
