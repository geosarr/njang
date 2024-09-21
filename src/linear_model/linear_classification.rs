use core::hash::Hash;
mod unit_test;

use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use rand_chacha::ChaCha20Rng;

use crate::{
    error::NjangError,
    traits::{ClassificationModel, Container, Model, RegressionModel, Scalar},
};

use super::{LinearRegression, LinearRegressionSettings, LinearRegressionSolver};
use std::collections::HashSet;

#[derive(Default, Debug, Clone, Copy)]
pub enum LinearClassificationSolver {
    /// Uses Stochastic Gradient Descent
    ///
    /// The user should standardize the input predictors, otherwise the
    /// algorithm may not converge.
    ///
    /// **This solver supports all models.**
    #[default]
    Sgd,
    /// Uses Batch Gradient Descent
    ///
    /// The user should standardize the input predictors, otherwise the
    /// algorithm may not converge.
    ///
    /// **This solver supports all models.**
    Bgd,
    /// Uses Stochastic Average Gradient
    ///
    /// The user should standardize the input predictors, otherwise the
    /// algorithm may not converge.
    Sag,
    ///
    Exact,
    ///
    Svd,
    ///
    Qr,
    ///
    Cholesky,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LinearClassificationParameter<C, I> {
    /// Non-intercept weight(s).
    pub coef: Option<C>,
    /// Intercept weight(s) of the model.
    pub intercept: Option<I>,
}

#[derive(Debug, Clone, Copy)]
pub enum LinearClassificationLoss {
    /// Ridge Classification
    Square,
    /// Logistic/SoftMax Classification
    LogLoss,
}

#[derive(Debug, Clone)]
pub(crate) struct LinearClassificationInternal<T> {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_targets: usize,
    pub l1_penalty: Option<T>,
    pub l2_penalty: Option<T>,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub rng: Option<ChaCha20Rng>,
    pub max_iter: Option<usize>,
}

/// Hyperparameters used in a linear classification model.
#[derive(Debug, Clone, Copy)]
pub struct LinearClassificationSettings<T> {
    /// If it is `true` then the model fits with an intercept, `false` without  
    /// /// an intercept.
    pub fit_intercept: bool,
    /// Optimization method, see [`LinearClassificationSolver`].
    pub solver: LinearClassificationSolver,
    /// Loss minimized by the `solver`, see [`LinearClassificationLoss`].
    pub loss: LinearClassificationLoss,
    /// If it is `None`, then no L1-penalty is added to the loss objective
    /// function. Otherwise, if it is equal to `Some(value)`, then `value *
    /// ||coef||`<sub>1</sub> is added to the loss objective function. Instead
    /// of setting `l1_penalty = Some(0.)`, it may be preferable to set
    /// `l1_penalty = None` to avoid useless computations and numerical issues.
    pub l1_penalty: Option<T>,
    /// If it is `None`, then no L2-penalty is added to the loss objective
    /// function. Otherwise, if it is equal to `Some(value)`, then `0.5 *
    /// value * ||coef||`<sub>2</sub><sup>2</sup> is added to the loss objective
    /// function. Instead of setting `l2_penalty = Some(0.)`, it may be
    /// preferable to set `l2_penalty = None` to avoid useless computations
    /// and numerical issues.
    pub l2_penalty: Option<T>,
    /// Tolerance parameter.
    /// - Gradient descent solvers (like
    ///   [Sgd][`LinearClassificationSolver::Sgd`],
    ///   [Bgd][`LinearClassificationSolver::Bgd`], etc) stop when the relative
    ///   variation of consecutive iterates is lower than **tol**, that is:
    ///     - `||coef_next - coef_current|| <= tol *||coef_current||`
    /// - No impact on the other algorithms:
    ///     - [Exact][`LinearClassificationSolver::Exact`]
    ///     - [Svd][`LinearClassificationSolver::Svd`]
    ///     - [Qr][`LinearClassificationSolver::Qr`]
    ///     - [Cholesky][`LinearClassificationSolver::Cholesky`]
    pub tol: Option<T>,
    /// Step size used in gradient descent algorithms.
    pub step_size: Option<T>,
    /// Seed of random generators used in gradient descent algorithms.
    pub random_state: Option<u32>,
    /// Maximum number of iterations used in gradient descent algorithms.
    pub max_iter: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LinearClassification<C, I>
where
    C: Container,
{
    pub parameter: LinearClassificationParameter<C, I>,
    pub settings: LinearClassificationSettings<C::Elem>,
    internal: LinearClassificationInternal<C::Elem>,
}

pub type RidgeClassificationSolver = LinearRegressionSolver;
pub struct RidgeClassificationSettings<T> {
    pub fit_intercept: bool,
    pub solver: RidgeClassificationSolver,
    pub l2_penalty: Option<T>,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

pub struct RidgeClassification<C, I, L = i32>
where
    C: Container,
{
    model: LinearRegression<C, I>,
    labels: Vec<L>,
}

impl<C: Container, I, L> RidgeClassification<C, I, L> {
    pub fn new(settings: RidgeClassificationSettings<C::Elem>) -> Self
    where
        C::Elem: Float,
    {
        let lin_settings = LinearRegressionSettings {
            fit_intercept: settings.fit_intercept,
            solver: settings.solver,
            l1_penalty: None,
            l2_penalty: settings.l2_penalty,
            tol: settings.tol,
            step_size: settings.step_size,
            random_state: settings.random_state,
            max_iter: settings.max_iter,
        };
        Self {
            model: LinearRegression::new(lin_settings),
            labels: Vec::new(),
        }
    }
    /// Coefficients of the model
    pub fn coef(&self) -> Option<&C> {
        self.model.parameter.coef.as_ref()
    }
    /// Intercept of the model
    pub fn intercept(&self) -> Option<&I> {
        self.model.parameter.intercept.as_ref()
    }
}

fn unique_labels<L>(labels: L) -> Vec<L::Item>
where
    L: IntoIterator,
    L::Item: Eq + Hash + Ord,
{
    let unique_labels = labels.into_iter().collect::<HashSet<_>>();
    let mut unique_labels = unique_labels.into_iter().collect::<Vec<_>>();
    unique_labels.sort();
    unique_labels
}

fn argmax<I, T>(iterable: I) -> Result<usize, isize>
where
    I: IntoIterator<Item = (usize, T)>,
    T: Float,
{
    let mut m = -1;
    let mut max = T::neg_infinity();
    iterable
        .into_iter()
        .map(|(ix, val)| {
            if val > max {
                m = ix as isize;
                max = val;
            }
        })
        .for_each(drop);
    if m >= 0 {
        Ok(m as usize)
    } else {
        Err(m) // Empty iterator
    }
}

fn dummies<T: Scalar, L: Ord>(y: &Array1<L>, labels: &[L]) -> Array2<T> {
    let mut y_reg = Array2::<T>::zeros((y.len(), labels.len()));
    y_reg
        .axis_iter_mut(Axis(0))
        .enumerate()
        .map(|(sample, mut row)| {
            let pos = labels.binary_search(&y[sample]).unwrap(); // Safe to .unwrap() since self.labels is built from y.
            row[pos] = T::one();
        })
        .for_each(drop);
    y_reg
}
impl<'a, T: Scalar, L: Eq + Hash + Ord + Copy + 'a> Model<'a>
    for RidgeClassification<Array2<T>, Array1<T>, L>
{
    type FitResult = Result<(), NjangError>;
    type Data = (&'a Array2<T>, &'a Array1<L>);
    fn fit(&mut self, data: &Self::Data) -> Self::FitResult {
        let (x, y) = data;
        self.labels = unique_labels((*y).iter()).into_iter().copied().collect();
        let y_reg = dummies(*y, &self.labels);
        RegressionModel::fit(&mut self.model, *x, &y_reg)
    }
}
impl<T: Scalar, L: Eq + Ord + Hash + Copy + 'static> ClassificationModel
    for RidgeClassification<Array2<T>, Array1<T>, L>
{
    type X = Array2<T>;
    type Y = Array1<L>;
    type PredictResult = Result<Array1<L>, ()>;
    type PredictProbaResult = Result<Array2<T>, ()>;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> <Self as Model<'_>>::FitResult {
        let data = (x, y);
        <Self as Model>::fit(self, &data)
    }
    fn predict(&self, x: &Self::X) -> Self::PredictResult {
        let raw_prediction = self.model.predict(x)?;
        Ok(raw_prediction
            .axis_iter(Axis(0))
            .map(|pred| self.labels[argmax(pred.iter().copied().enumerate()).unwrap()])
            .collect())
    }
    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult {
        let mut raw_prediction = self.model.predict(x)?.map(|x| Float::exp(*x));
        raw_prediction
            .axis_iter_mut(Axis(0))
            .map(|mut row| {
                let norm = row.sum();
                row.iter_mut().for_each(|p| *p = *p / norm);
            })
            .for_each(drop);
        Ok(raw_prediction)
    }
}

#[test]
fn code() {
    use ndarray::array;

    let x = array![[0., 0., 1.], [1., 0., 0.], [1., 0., 1.]];
    let y = ndarray::Array1::from_shape_fn(x.nrows(), core::convert::identity);
    println!("y:\n{:?}", y);

    let settings = RidgeClassificationSettings {
        fit_intercept: false,
        solver: RidgeClassificationSolver::Sag,
        l2_penalty: None,
        tol: Some(1e-6),
        step_size: Some(1e-3),
        random_state: Some(0),
        max_iter: Some(100000),
    };
    let mut model = RidgeClassification::<Array2<_>, _, _>::new(settings);
    match Model::fit(&mut model, &(&x, &y)) {
        Ok(_) => {
            println!("{:?}", model.coef());
        }
        Err(error) => {
            println!("{:?}", error);
        }
    };

    assert_eq!(model.predict(&x).unwrap(), y);
}
