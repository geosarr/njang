#![no_std]

mod linear_model;
mod traits;
mod utils;

pub(crate) use utils::{l2_diff, l2_diff2};

pub use linear_model::{
    LinearRegression, LinearRegressionHyperParameter, LinearRegressionSolver, RidgeRegression,
    RidgeRegressionHyperParameter, RidgeRegressionSolver,
};
pub use traits::RegressionModel;
