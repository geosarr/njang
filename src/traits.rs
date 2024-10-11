use ndarray::*;
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive, One, Zero};

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

/// Handles float-pointing numbers.
pub trait Scalar:
    Lapack + PartialOrd + Float + ScalarOperand + SampleUniform + core::fmt::Debug
{
}
impl Scalar for f32 {}
impl Scalar for f64 {}

/// Handles label types for classification tasks.
pub trait Label: Eq + Ord + core::hash::Hash + Copy + core::fmt::Debug {}
macro_rules! impl_label(
    ( $( $t:ty ),* )=> {
        $(
            impl Label for $t {}
        )*
    }
);
impl_label!(
    usize,
    u8,
    u16,
    u32,
    u64,
    isize,
    i8,
    i16,
    i32,
    i64,
    &'static str
);

/// Base trait handling the modeling data structures.
pub trait Container {
    type Elem;
    type SelectionOutput;
    type LenghtOutput;
    fn dimension(&self) -> &[usize];
    fn selection(&self, axis: usize, indices: &[usize]) -> Self::SelectionOutput;
    fn length(&self) -> Self::LenghtOutput;
}

impl<S, D> Container for ArrayBase<S, D>
where
    S: Data,
    S::Elem: Clone,
    D: Dimension + RemoveAxis,
{
    type Elem = S::Elem;
    type SelectionOutput = Array<S::Elem, D>;
    type LenghtOutput = usize;
    fn dimension(&self) -> &[usize] {
        Self::shape(self)
    }
    fn selection(&self, axis: usize, indices: &[usize]) -> Self::SelectionOutput {
        Self::select(self, Axis(axis), indices)
    }
    fn length(&self) -> Self::LenghtOutput {
        self.len()
    }
}

/// Trait implementing operations on modeling data structures.
pub trait Algebra: Container {
    type MeanAxisOutput;
    type PowiOutput;
    type SignOutput;
    type SoftmaxOutput;
    fn powi(&self, n: i32) -> Self::PowiOutput;
    fn mean_axis(&self, axis: usize) -> Self::MeanAxisOutput;
    fn l2_norm(&self) -> Self::Elem;
    fn minkowsky(&self, p: Self::Elem) -> Self::Elem;
    fn linf_norm(&self) -> Self::Elem;
    fn sign(&self) -> Self::SignOutput;
    fn softmax(&self, max: Option<Self::Elem>, axis: usize) -> Self::SoftmaxOutput;
}

impl<S, D> Algebra for ArrayBase<S, D>
where
    S: Data,
    S::Elem: Float + Zero + One + FromPrimitive + ScalarOperand,
    D: Dimension + RemoveAxis,
{
    type MeanAxisOutput = Array<S::Elem, D::Smaller>;
    type PowiOutput = Array<S::Elem, D>;
    type SignOutput = Array<S::Elem, D>;
    type SoftmaxOutput = Array<S::Elem, D>;
    fn powi(&self, n: i32) -> Self::PowiOutput {
        self.map(|v| v.powi(n))
    }
    fn mean_axis(&self, axis: usize) -> Self::MeanAxisOutput {
        Self::mean_axis(self, Axis(axis)).unwrap()
    }
    fn l2_norm(&self) -> S::Elem {
        self.powi(2).sum().sqrt()
    }
    fn minkowsky(&self, p: S::Elem) -> Self::Elem {
        assert!(
            p >= S::Elem::one(),
            "p should be >= 1 for Minkowsky distance."
        );
        self.map(|v| v.abs().powf(p)).sum().powf(S::Elem::one() / p)
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
            if v.abs() <= S::Elem::epsilon() {
                S::Elem::zero()
            } else {
                v.signum()
            }
        })
    }
    fn softmax(&self, max: Option<Self::Elem>, axis: usize) -> Self::SoftmaxOutput {
        let exponentials = if let Some(m) = max {
            self.map(|x| (*x - m).exp())
        } else {
            self.map(|x| (*x).exp())
        };
        let denom = exponentials.sum_axis(Axis(axis));
        exponentials / denom
    }
}
