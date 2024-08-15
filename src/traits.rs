use core::ops::Mul;
use core::ops::Not;
use ndarray::{linalg::Dot, Array1, Array2, ScalarOperand};
use ndarray_linalg::{Cholesky, Inverse, Lapack, QR};
use num_traits::{Float, FromPrimitive};
/// Implements classic steps of a regression model.
pub trait RegressionModel {
    type X;
    type Y;
    type FitResult;
    type PredictResult;
    /// Trains the model, with possibly warm start.
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult;
    /// Predicts instances if possible.
    fn predict(&self, x: &Self::X) -> Self::PredictResult;
}

// pub trait ClassificationModel {
//     type X;
//     type Y;
//     type FitResult;
//     type PredictResult;
//     type PredictProbaResult;
//     fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult;
//     fn predict(&self, x: &Self::X) -> Self::PredictResult;
//     fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult;
// }

/// Implements some methods on vectors and matrices.
pub trait Info {
    type MeanOutput;
    type RowOutput;
    type ColOutput;
    type ShapeOutput;
    type ColMut;
    type NcolsOutput;
    type NrowsOutput;
    /// Mean of each column for 2d containers and mean of all elements for 1d containers.
    fn mean(&self) -> Self::MeanOutput;
    /// Like copy, view of a "row".
    fn get_row(&self, i: usize) -> Self::RowOutput;
    /// Like copy, view of a "column".
    fn get_col(&self, i: usize) -> Self::ColOutput;
    /// Like a pair (number of rows, number of columns) for 2d containers and (n_elements) for 1d containers.
    fn shape(&self) -> Self::ShapeOutput;
    /// Mutate column number idx of a 2d container with elem.
    fn col_mut(&mut self, idx: usize, elem: Self::ColMut);
    /// Number of columns for 2d containers.
    fn get_ncols(&self) -> Self::NcolsOutput;
    /// Number of rows for 2d containers.
    fn get_nrows(&self) -> Self::NrowsOutput;
}

pub(crate) trait Scalar<X>
where
    for<'a> Self: Lapack
        + Float
        + ScalarOperand
        + Mul<X, Output = X>
        + Mul<&'a X, Output = X>
        + core::fmt::Debug
        + FromPrimitive,
{
}
macro_rules! impl_scalar {
    ($t:ty) => {
        impl Scalar<Array1<$t>> for $t {}
        impl Scalar<Array2<$t>> for $t {}
    };
}
impl_scalar!(f32);
impl_scalar!(f64);
