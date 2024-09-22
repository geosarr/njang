use core::hash::Hash;
mod ridge_classification;
mod unit_test;
use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use rand_chacha::ChaCha20Rng;

use super::{LinearRegression, LinearRegressionSettings, LinearRegressionSolver};
use crate::{
    error::NjangError,
    traits::{ClassificationModel, Container, Model, RegressionModel, Scalar},
};
pub use ridge_classification::{
    RidgeClassification, RidgeClassificationSettings, RidgeClassificationSolver,
};
use std::collections::HashSet;

// #[derive(Default, Debug, Clone, Copy)]
// pub enum LinearClassificationSolver {
//     /// Uses Stochastic Gradient Descent
//     ///
//     /// The user should standardize the input predictors, otherwise the
//     /// algorithm may not converge.
//     ///
//     /// **This solver supports all models.**
//     #[default]
//     Sgd,
//     /// Uses Batch Gradient Descent
//     ///
//     /// The user should standardize the input predictors, otherwise the
//     /// algorithm may not converge.
//     ///
//     /// **This solver supports all models.**
//     Bgd,
//     /// Uses Stochastic Average Gradient
//     ///
//     /// The user should standardize the input predictors, otherwise the
//     /// algorithm may not converge.
//     Sag,
//     ///
//     Exact,
//     ///
//     Svd,
//     ///
//     Qr,
//     ///
//     Cholesky,
// }

// #[derive(Debug, Default, Clone, Copy)]
// pub struct LinearClassificationParameter<C, I> {
//     /// Non-intercept weight(s).
//     pub coef: Option<C>,
//     /// Intercept weight(s) of the model.
//     pub intercept: Option<I>,
// }

// #[derive(Debug, Clone, Copy)]
// pub enum LinearClassificationLoss {
//     /// Ridge Classification
//     Square,
//     // /// Logistic/SoftMax Classification
//     // LogLoss,
// }

// #[derive(Debug, Clone)]
// pub(crate) struct LinearClassificationInternal<T> {
//     pub n_samples: usize,
//     pub n_features: usize,
//     pub n_targets: usize,
//     pub l1_penalty: Option<T>,
//     pub l2_penalty: Option<T>,
//     pub tol: Option<T>,
//     pub step_size: Option<T>,
//     pub rng: Option<ChaCha20Rng>,
//     pub max_iter: Option<usize>,
// }

// /// Hyperparameters used in a linear classification model.
// #[derive(Debug, Clone, Copy)]
// pub struct LinearClassificationSettings<T> {
//     /// If it is `true` then the model fits with an intercept, `false`
// without     /// /// an intercept.
//     pub fit_intercept: bool,
//     /// Optimization method, see [`LinearClassificationSolver`].
//     pub solver: LinearClassificationSolver,
//     /// Loss minimized by the `solver`, see [`LinearClassificationLoss`].
//     pub loss: LinearClassificationLoss,
//     /// If it is `None`, then no L1-penalty is added to the loss objective
//     /// function. Otherwise, if it is equal to `Some(value)`, then `value *
//     /// ||coef||`<sub>1</sub> is added to the loss objective function.
// Instead     /// of setting `l1_penalty = Some(0.)`, it may be preferable to
// set     /// `l1_penalty = None` to avoid useless computations and numerical
// issues.     pub l1_penalty: Option<T>,
//     /// If it is `None`, then no L2-penalty is added to the loss objective
//     /// function. Otherwise, if it is equal to `Some(value)`, then `0.5 *
//     /// value * ||coef||`<sub>2</sub><sup>2</sup> is added to the loss
// objective     /// function. Instead of setting `l2_penalty = Some(0.)`, it
// may be     /// preferable to set `l2_penalty = None` to avoid useless
// computations     /// and numerical issues.
//     pub l2_penalty: Option<T>,
//     /// Tolerance parameter.
//     /// - Gradient descent solvers (like
//     ///   [Sgd][`LinearClassificationSolver::Sgd`],
//     ///   [Bgd][`LinearClassificationSolver::Bgd`], etc) stop when the
// relative     ///   variation of consecutive iterates is lower than **tol**,
// that is:     ///     - `||coef_next - coef_current|| <= tol
// *||coef_current||`     /// - No impact on the other algorithms:
//     ///     - [Exact][`LinearClassificationSolver::Exact`]
//     ///     - [Svd][`LinearClassificationSolver::Svd`]
//     ///     - [Qr][`LinearClassificationSolver::Qr`]
//     ///     - [Cholesky][`LinearClassificationSolver::Cholesky`]
//     pub tol: Option<T>,
//     /// Step size used in gradient descent algorithms.
//     pub step_size: Option<T>,
//     /// Seed of random generators used in gradient descent algorithms.
//     pub random_state: Option<u32>,
//     /// Maximum number of iterations used in gradient descent algorithms.
//     pub max_iter: Option<usize>,
// }

// #[derive(Debug, Clone)]
// pub struct LinearClassification<C, I, M>
// where
//     C: Container,
//     M: Model,
// {
//     pub parameter: LinearClassificationParameter<C, I>,
//     pub settings: LinearClassificationSettings<C::Elem>,
//     internal: LinearClassificationInternal<C::Elem>,
//     model: M,
// }

// impl<C: Container, I, M: Model> LinearClassification<C, I, M> {
//     pub fn new(settings: LinearClassificationSettings<C::Elem>) -> Self {
//         let loss = settings.loss;
//         match loss {
//             LinearClassificationLoss::Square => Self {
//                 parameter: LinearClassificationParameter::new(),
//                 settings: settings,
//                 internal: LinearClassificationInternal::new(),
//                 model: RidgeClassification::new(settings),
//             },
//         }
//     }
// }

pub(crate) fn unique_labels<L>(labels: L) -> Vec<L::Item>
where
    L: IntoIterator,
    L::Item: Eq + Hash + Ord,
{
    let unique_labels = labels.into_iter().collect::<HashSet<_>>();
    let mut unique_labels = unique_labels.into_iter().collect::<Vec<_>>();
    unique_labels.sort();
    unique_labels
}

pub(crate) fn argmax<I, T>(iterable: I) -> Result<usize, isize>
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

pub(crate) fn dummies<T: Scalar, L: Ord>(y: &Array1<L>, labels: &[L]) -> Array2<T> {
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
