#![cfg_attr(not(feature = "std"), no_std)]

mod error;
pub mod linear_model;
mod neighbors;
mod solver;
mod traits;
mod utils;

#[allow(unused)]
pub(crate) use utils::{l2_diff, l2_diff2, l2_norm1, l2_norm2};

pub use linear_model::{
    LinearModelSolver, LinearRegression, LinearRegressionSettings, LogisticRegression,
    LogisticRegressionSettings, RidgeClassification, RidgeClassificationSettings,
};
pub use neighbors::KdTree;
pub use traits::{ClassificationModel, RegressionModel};
