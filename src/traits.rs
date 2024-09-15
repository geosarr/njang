use ndarray::*;
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
    fn l2_norm(&self) -> Self::Elem;
    fn sign(&self) -> Self;
}

impl<T: Float + FromPrimitive> Algebra for Array1<T> {
    type MeanAxisOutput = Array0<T>;
    fn powi(&self, n: i32) -> Self {
        self.map(|v| v.powi(n))
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
