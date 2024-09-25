use rand_chacha::ChaCha20Rng;

use core::ops::Sub;
use num_traits::Float;

use crate::{
    error::NjangError,
    linear_model::{
        classification::{argmax, dummies, unique_labels},
        preprocess, randu_2d, LinearModelParameter,
    },
    solver::stochastic_gradient_descent,
    traits::{Container, Model, Scalar},
};

use ndarray::{Array1, Array2, ScalarOperand};

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

impl<C: Container, I> LogisticRegression<C, I> {
    pub fn new(settings: LogisticRegressionSettings<C::Elem>) -> Self
    where
        C::Elem: Float,
    {
        Self {
            parameter: LinearModelParameter {
                coef: None,
                intercept: None,
            },
            settings,
            internal: ModelInternal::new(),
        }
    }
}

use num_traits::{FromPrimitive, Zero};

impl<'a, T: Scalar + Float + Zero + FromPrimitive + ScalarOperand> Model<'a>
    for LogisticRegression<Array2<T>, Array1<T>>
{
    type FitResult = Result<(), NjangError>;
    type Data = (&'a Array2<T>, &'a Array1<i32>);
    fn fit(&mut self, data: &Self::Data) -> Self::FitResult {
        let (x, y) = data;
        let labels = unique_labels((*y).iter())
            .into_iter()
            .copied()
            .collect::<Vec<_>>();
        let y_reg = dummies::<T, i32>(*y, &labels);
        // let (x_centered, x_mean, y_centered, y_mean) = preprocess(*x, *y);
        self.parameter.coef = match self.settings.solver {
            LogisticRegressionSolver::Sgd => {
                self.set_internal(*x, &y_reg);
                // Rescale step_size and penalty(ies) to scale (sub)gradients correctly
                self.scale_step_size();
                self.scale_penalty();
                let (n_labels, n_features, rng) = (
                    self.internal.n_targets,
                    self.internal.n_features,
                    self.internal.rng.as_mut().unwrap(),
                );
                let coef = randu_2d::<T, _>(&[n_features, n_labels], rng);
                Some(stochastic_gradient_descent(
                    *x,
                    &y_reg,
                    coef,
                    self.gradient_function(),
                    &self.internal,
                ))
            }
        };
        Ok(())
    }
}

#[test]
fn log() {
    use ndarray::array;

    let x = array![[0., 0., 1.], [1., 0., 0.], [1., 0., 1.]];
    let y = array![0, 1, 2];
    println!("y:\n{:?}", y);

    let settings = LogisticRegressionSettings {
        fit_intercept: false,
        solver: LogisticRegressionSolver::Sgd,
        l1_penalty: None,
        l2_penalty: None,
        tol: Some(1e-6),
        step_size: Some(1e-3),
        random_state: Some(0),
        max_iter: Some(100000),
    };
    let mut model = LogisticRegression::<Array2<_>, _>::new(settings);
    match Model::fit(&mut model, &(&x, &y)) {
        Ok(_) => {
            println!("{:?}", model.coef());
        }
        Err(error) => {
            println!("{:?}", error);
        }
    };
    // assert_eq!(model.predict(&x).unwrap(), y);
}
