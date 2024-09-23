use rand_chacha::ChaCha20Rng;

use crate::{linear_model::LinearModelParameter, traits::Container};
use core::ops::Sub;

#[derive(Default, Debug, Clone, Copy)]
pub enum LogisticRegressionSolver {
    /// Uses Stochastic Gradient Descent
    ///
    /// The user should standardize the input predictors, otherwise the
    /// algorithm may not converge.
    #[default]
    Sgd,
    // /// Uses Batch Gradient Descent
    // ///
    // /// The user should standardize the input predictors, otherwise the
    // /// algorithm may not converge.
    // Bgd,
    // /// Uses Stochastic Average Gradient
    // ///
    // /// The user should standardize the input predictors, otherwise the
    // /// algorithm may not converge.
    // Sag,
}

use crate::linear_model::ModelInternal;

#[derive(Debug, Default, Clone, Copy)]
pub struct LogisticRegressionParameter<C, I> {
    /// Non-intercept weight(s).
    pub coef: Option<C>,
    /// Intercept weight(s) of the model.
    pub intercept: Option<I>,
}

#[derive(Debug, Clone)]
pub(crate) struct LogisticRegressionInternal<T> {
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

impl<T> LogisticRegressionInternal<T> {
    pub fn new() -> Self {
        Self {
            n_samples: 0,
            n_features: 0,
            n_targets: 0,
            l1_penalty: None,
            l2_penalty: None,
            tol: None,
            step_size: None,
            rng: None,
            max_iter: None,
        }
    }
}

/// Hyperparameters used in a linear classification model.
#[derive(Debug, Clone, Copy)]
pub struct LogisticRegressionSettings<T> {
    /// If it is `true` then the model fits with an intercept, `false`
    /// without an intercept.
    pub fit_intercept: bool,
    /// Optimization method, see [`LogisticRegressionSolver`].
    pub solver: LogisticRegressionSolver,
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
    /// - Gradient descent solvers (like [Sgd][`LogisticRegressionSolver::Sgd`],
    ///   [Bgd][`LogisticRegressionSolver::Bgd`], etc) stop when the relative
    ///   variation of consecutive iterates is lower than **tol**, that is:
    ///     - `||coef_next - coef_current|| <= tol *||coef_current||`
    pub tol: Option<T>,
    /// Step size used in gradient descent algorithms.
    pub step_size: Option<T>,
    /// Seed of random generators used in gradient descent algorithms.
    pub random_state: Option<u32>,
    /// Maximum number of iterations used in gradient descent algorithms.
    pub max_iter: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LogisticRegression<C, I>
where
    C: Container,
{
    pub parameter: LinearModelParameter<C, I>,
    pub settings: LogisticRegressionSettings<C::Elem>,
    pub(crate) internal: ModelInternal<C::Elem>,
}
