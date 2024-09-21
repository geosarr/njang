use ndarray::*;
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive, Zero};

pub trait Model<'a> {
    type Data;
    type FitResult;
    /// Trains the model.
    fn fit(&mut self, data: &'a Self::Data) -> Self::FitResult;
}

/// Implements classic steps of a regression model.
pub trait RegressionModel: for<'a> Model<'a> {
    type X;
    type Y;
    type PredictResult;
    /// Trains the model.
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> <Self as Model<'_>>::FitResult;
    /// Predicts instances if possible.
    fn predict(&self, x: &Self::X) -> Self::PredictResult;
}

/// Implements classic steps of a classification model.
pub trait ClassificationModel: for<'a> Model<'a> {
    type X;
    type Y;
    type PredictResult;
    type PredictProbaResult;
    /// Trains the model.
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> <Self as Model<'_>>::FitResult;
    /// Predicts instances if possible.
    fn predict(&self, x: &Self::X) -> Self::PredictResult;
    /// Estimates the probability(ies) of instances if possible.
    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult;
}

/// Trait to handle float-pointing numbers.
pub trait Scalar:
    Lapack + PartialOrd + Float + ScalarOperand + SampleUniform + core::fmt::Debug
{
}
impl Scalar for f32 {}
impl Scalar for f64 {}

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
    fn linf_norm(&self) -> Self::Elem;
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
    fn linf_norm(&self) -> S::Elem {
        let mut norm = S::Elem::zero();
        self.map(|x| {
            let xabs = x.abs();
            if xabs > norm {
                norm = xabs
            }
            xabs
        });
        norm
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
