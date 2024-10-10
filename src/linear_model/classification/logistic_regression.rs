extern crate alloc;

use alloc::fmt::format;
use num_traits::Float;

use crate::linear_model::LinearModelSolver;
use crate::solver::batch_gradient_descent;
use crate::traits::Algebra;
use crate::{
    error::NjangError,
    linear_model::{
        classification::{argmax, dummies, unique_labels},
        randu_2d, LinearModelParameter,
    },
    solver::stochastic_gradient_descent,
    traits::{ClassificationModel, Container, Label, Scalar},
};

use ndarray::{Array1, Array2, Axis};

use crate::linear_model::LinearModelInternal;

/// Hyperparameters used in a logistic regression model.
#[derive(Debug, Clone, Copy)]
pub struct LogisticRegressionSettings<T> {
    /// If it is `true` then the model fits with an intercept, `false`
    /// without an intercept.
    pub fit_intercept: bool,
    /// Optimization method, see [`LinearModelSolver`].
    pub solver: LinearModelSolver,
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
    /// - Gradient descent solvers (like [Sgd][`LinearModelSolver::Sgd`],
    ///   [Bgd][`LinearModelSolver::Bgd`], etc) stop when the relative variation
    ///   of consecutive iterates is lower than **tol**, that is:
    ///     - `||coef_next - coef_current|| <= tol *||coef_current||`
    pub tol: Option<T>,
    /// Step size used in gradient descent algorithms.
    pub step_size: Option<T>,
    /// Seed of random generators used in gradient descent algorithms.
    pub random_state: Option<u32>,
    /// Maximum number of iterations used in gradient descent algorithms.
    pub max_iter: Option<usize>,
}

/// Logistic regression eventually penalized (with L1-penalty, or with
/// L2-penalty or with both L1 and L2 penalties).
#[derive(Debug, Clone)]
pub struct LogisticRegression<C, I, L>
where
    C: Container,
{
    pub parameter: LinearModelParameter<C, I>,
    pub settings: LogisticRegressionSettings<C::Elem>,
    pub(crate) internal: LinearModelInternal<C::Elem>,
    pub labels: Vec<L>,
}

impl<C: Container, I, L> LogisticRegression<C, I, L> {
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
            internal: LinearModelInternal::new(),
            labels: Vec::new(),
        }
    }
}

impl<T: Scalar, L: Label> ClassificationModel for LogisticRegression<Array2<T>, Array1<T>, L> {
    type X = Array2<T>;
    type Y = Array1<L>;
    type FitResult = Result<(), NjangError>;
    type PredictResult = Result<Array1<L>, ()>;
    type PredictProbaResult = Result<Array2<T>, ()>;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
        let (xlen, ylen) = (x.nrows(), y.len());
        if xlen != ylen {
            return Err(NjangError::NotMatchedLength { xlen, ylen });
        }
        self.labels = unique_labels(y.iter()).into_iter().copied().collect();
        let y_reg = dummies(y, &self.labels);
        // let (x_centered, x_mean, y_centered, y_mean) = preprocess(*x, *y);
        self.parameter.coef = match self.settings.solver {
            LinearModelSolver::Sgd => {
                self.set_internal(x, &y_reg);
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
                    x,
                    &y_reg,
                    coef,
                    self.gradient_function(),
                    &self.internal,
                ))
            }
            LinearModelSolver::Bgd => {
                self.set_internal(x, &y_reg);
                let (n_targets, n_features, rng) = (
                    self.internal.n_targets,
                    self.internal.n_features,
                    self.internal.rng.as_mut().unwrap(),
                );
                let coef = randu_2d(&[n_features, n_targets], rng);
                Some(batch_gradient_descent(
                    x,
                    &y_reg,
                    coef,
                    self.gradient_function(),
                    &self.internal,
                ))
            }
            _ => {
                return Err(NjangError::NotSupported {
                    item: format!("Solver {:?}", self.settings.solver),
                })
            }
        };
        Ok(())
    }
    fn predict(&self, x: &Self::X) -> Self::PredictResult {
        if let Some(coef) = self.coef() {
            let raw_prediction = x.dot(coef).softmax(None, 0);
            Ok(raw_prediction
                .axis_iter(Axis(0))
                .map(|pred| self.labels[argmax(pred.iter().copied().enumerate()).unwrap()])
                .collect())
        } else {
            Err(())
        }
    }
    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult {
        if let Some(coef) = self.coef() {
            Ok(x.dot(coef).softmax(None, 0))
        } else {
            Err(())
        }
    }
}

#[test]
fn log() {
    use ndarray::array;

    let x = array![[0., 0., 1.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
    let y = array!["C", "A", "B", "Z"];
    println!("y:\n{:?}", y);

    let settings = LogisticRegressionSettings {
        fit_intercept: false,
        solver: LinearModelSolver::Svd,
        l1_penalty: None,
        l2_penalty: Some(0.),
        tol: Some(1e-6),
        step_size: Some(1e-3),
        random_state: Some(0),
        max_iter: Some(100000),
    };
    let mut model = LogisticRegression::<Array2<_>, _, _>::new(settings);
    match model.fit(&x, &y) {
        Ok(_) => {
            println!("{:?}", model.predict_proba(&x));
            let y_pred = model.predict(&x);
            println!("\n{:?}", y_pred);
            assert_eq!(y_pred.unwrap(), y);
        }
        Err(error) => {
            println!("{:?}", error);
        }
    };
}
