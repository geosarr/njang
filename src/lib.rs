#![cfg_attr(not(feature = "std"), no_std)]

mod linear_model;
mod traits;
mod utils;

#[allow(unused)]
pub(crate) use utils::{l2_diff, l2_diff2, l2_norm1, l2_norm2};

pub use linear_model::{
    LinearRegression, LinearRegressionHyperParameter, LinearRegressionSolver, RidgeRegression,
    RidgeRegressionHyperParameter, RidgeRegressionSolver,
};
pub use traits::RegressionModel;
