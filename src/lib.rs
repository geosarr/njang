// #![no_std]

mod linear_model;
mod traits;

pub use linear_model::{
    LinearRegression, LinearRegressionSolver, RidgeRegression, RidgeRegressionSolver,
};
pub use traits::RegressionModel;
