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
    NotSupported { item: &'static str },
}
