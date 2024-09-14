use core::ops::Mul;
use ndarray::*;
use ndarray_linalg::Lapack;
use num_traits::{Float, FromPrimitive};
/// Implements classic steps of a regression model.
pub trait RegressionModel {
    type X;
    type Y;
    type FitResult;
    type PredictResult;
    /// Trains the model.
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
    type RowMut;
    type NcolsOutput;
    type NrowsOutput;
    type SliceRowOutput;
    /// Mean of each column for 2d containers and mean of all elements for 1d
    /// containers.
    fn mean(&self) -> Self::MeanOutput;
    /// Like copy, view of a "row".
    fn get_row(&self, i: usize) -> Self::RowOutput;
    /// Like copy, view of a "column".
    fn get_col(&self, i: usize) -> Self::ColOutput;
    /// Like a pair (number of rows, number of columns) for 2d containers and
    /// (n_elements) for 1d containers.
    fn shape(&self) -> Self::ShapeOutput;
    /// Mutate column number idx of a 2d container with elem.
    fn col_mut(&mut self, idx: usize, elem: Self::ColMut);
    /// Mutate row number idx of a 2d container with elem.
    fn row_mut(&mut self, idx: usize, elem: Self::RowMut);
    /// Slices rows of a matrix, taking all columns
    fn slice_row(&self, start: usize, end: usize) -> Self::SliceRowOutput;
    /// Number of columns for 2d containers.
    fn get_ncols(&self) -> Self::NcolsOutput;
    /// Number of rows for 2d containers.
    fn get_nrows(&self) -> Self::NrowsOutput;
}

pub trait Container {
    type Elem;
    type SelectionOutput;
    fn dimension(&self) -> &[usize];
    fn selection(&self, axis: usize, indices: &[usize]) -> Self::SelectionOutput;
}

impl<S, D> Container for ArrayBase<S, D>
where
    S: Data,
    S::Elem: Clone,
    D: Dimension + RemoveAxis,
{
    type Elem = S::Elem;
    type SelectionOutput = Array<S::Elem, D>;
    fn dimension(&self) -> &[usize] {
        Self::shape(&self)
    }
    fn selection(&self, axis: usize, indices: &[usize]) -> Self::SelectionOutput {
        Self::select(self, Axis(axis), indices)
    }
}

pub trait Algebra: Container {
    type MeanAxisOutput;
    fn powi(&self, n: i32) -> Self;
    fn mean_axis(&self, axis: usize) -> Self::MeanAxisOutput;
    fn mean(&self) -> Self::Elem;
    fn l2_norm(&self) -> Self::Elem;
    fn sign(&self) -> Self;
}

impl<T: Float + FromPrimitive> Algebra for Array1<T> {
    type MeanAxisOutput = Array0<T>;
    fn powi(&self, n: i32) -> Self {
        self.map(|v| v.powi(n))
    }
    fn mean(&self) -> Self::Elem {
        Self::mean(self).unwrap()
    }
    fn mean_axis(&self, axis: usize) -> Self::MeanAxisOutput {
        Self::mean_axis(self, Axis(axis)).unwrap()
    }
    fn l2_norm(&self) -> Self::Elem {
        self.powi(2).sum().sqrt()
    }
    fn sign(&self) -> Self {
        self.map(|v| {
            if v.abs() > T::epsilon() {
                T::zero()
            } else {
                v.signum()
            }
        })
    }
}

impl<T: Float + FromPrimitive> Algebra for Array2<T> {
    type MeanAxisOutput = Array1<T>;
    fn powi(&self, n: i32) -> Self {
        self.map(|v| v.powi(n))
    }
    fn mean(&self) -> Self::Elem {
        Self::mean(self).unwrap()
    }
    fn mean_axis(&self, axis: usize) -> Self::MeanAxisOutput {
        Self::mean_axis(self, Axis(axis)).unwrap()
    }
    fn l2_norm(&self) -> Self::Elem {
        self.powi(2).sum().sqrt()
    }
    fn sign(&self) -> Self {
        self.map(|v| {
            if v.abs() > T::epsilon() {
                T::zero()
            } else {
                v.signum()
            }
        })
    }
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
