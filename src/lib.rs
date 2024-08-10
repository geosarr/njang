// #![no_std]

mod linear_model;
mod traits;

pub use linear_model::{
    LinearRegression, LinearRegressionHyperParameter, LinearRegressionSolver, RidgeRegression,
    RidgeRegressionHyperParameter, RidgeRegressionSolver,
};
pub use traits::RegressionModel;
