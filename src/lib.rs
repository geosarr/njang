#![cfg_attr(not(feature = "std"), no_std)]

pub mod linear_model;
mod solver;
mod traits;
mod utils;

#[allow(unused)]
pub(crate) use utils::{l2_diff, l2_diff2, l2_norm1, l2_norm2};

pub use linear_model::{Regression, RegressionSettings, RegressionSolver};
pub use traits::RegressionModel;
