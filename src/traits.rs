use ndarray::*;
use num_traits::{Float, FromPrimitive, Zero};

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

/// Implements classic steps of a classification model.
pub trait ClassificationModel {
    type X;
    type Y;
    type FitResult;
    type PredictResult;
    type PredictProbaResult;
    /// Trains the model.
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult;
    /// Predicts instances if possible.
    fn predict(&self, x: &Self::X) -> Self::PredictResult;
    /// Estimates the probability(ies) of instances if possible.
    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult;
}

/// Base trait handling the modelling data structures.
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

/// Trait implementing operations on modelling data structures.
pub trait Algebra: Container {
    type MeanAxisOutput;
    type PowiOutput;
    type SignOutput;
    fn powi(&self, n: i32) -> Self::PowiOutput;
    fn mean_axis(&self, axis: usize) -> Self::MeanAxisOutput;
    fn l2_norm(&self) -> Self::Elem;
    fn sign(&self) -> Self::SignOutput;
}

impl<S, D> Algebra for ArrayBase<S, D>
where
    S: Data,
    S::Elem: Float + Zero + FromPrimitive,
    D: Dimension + RemoveAxis,
{
    type MeanAxisOutput = Array<S::Elem, D::Smaller>;
    type PowiOutput = Array<S::Elem, D>;
    type SignOutput = Array<S::Elem, D>;
    fn powi(&self, n: i32) -> Self::PowiOutput {
        self.map(|v| v.powi(n))
    }
    fn mean_axis(&self, axis: usize) -> Self::MeanAxisOutput {
        Self::mean_axis(self, Axis(axis)).unwrap()
    }
    fn l2_norm(&self) -> S::Elem {
        self.powi(2).sum().sqrt()
    }
    fn sign(&self) -> Self::SignOutput {
        self.map(|v| {
            if v.abs() > S::Elem::epsilon() {
                S::Elem::zero()
            } else {
                v.signum()
            }
        })
    }
}
