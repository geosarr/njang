mod unit_test;
use crate::error::NjangError;
use crate::{
    linear_model::{cholesky, exact, preprocess, qr, randu_1d, randu_2d},
    solver::{batch_gradient_descent, stochastic_average_gradient, stochastic_gradient_descent},
    traits::{Algebra, Container, Model, RegressionModel, Scalar},
};
use core::convert::identity;
use core::ops::Add;
use ndarray::{s, Array, Array1, Array2, ArrayView2, Axis, Ix0, Ix1, Ix2, ScalarOperand};
use ndarray_linalg::{error::LinalgError, Lapack, LeastSquaresSvd};
use num_traits::Float;

use super::{LinearModelInternal, LinearModelParameter, ModelInternal};

#[derive(Debug, Default, Clone, Copy)]
pub enum LinearRegressionSolver {
    /// Uses Singular Value Decomposition
    ///
    /// **This solver is available only for Linear regression and Ridge
    /// regression**.
    Svd,
    /// Computes the exact solution
    ///
    /// **This solver is available only for Linear regression and Ridge
    /// regression**.
    Exact,
    /// Uses QR decomposition to solve the problem.
    ///
    /// **This solver is available only for Linear regression and Ridge
    /// regression**.
    Qr,
    /// Uses Cholesky decomposition
    ///
    /// **This solver is available only for Linear regression and Ridge
    /// regression**.
    Cholesky,
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
}

/// Hyperparameters used in a linear regression model.
///
/// **It is important to note** that the fitted model depends on how the user
/// sets the fields `*_penalty`:
/// - if `l1_penalty = None` and `l2_penalty = None`, then the fitted model is
///   **Linear regression** (without penalty).
/// - if `l1_penalty = None` and `l2_penalty = Some(value)`, then the fitted
///   model is **Ridge regression**.
/// - if `l1_penalty = Some(value)` and `l2_penalty = None`, then the fitted
///   model is **Lasso regression**.
/// - otherwise (i.e `l1_penalty = Some(value)` and `l2_penalty = Some(value)`),
///   then the fitted model is **Elastic Net regression**.
#[derive(Debug, Clone, Copy)]
pub struct LinearRegressionSettings<T> {
    /// If it is `true` then the model fits with an intercept, `false` without
    /// an intercept. The matrix [coef][`LinearRegression`] of non-intercept
    /// weights satisfies:
    /// - if `fit_intercept = false`, then the prediction of a sample `x` is
    ///   given by `x.dot(coef)`
    /// - if `fit_intercept = true`, then the prediction of a sample `x` is
    ///   given by `x.dot(coef) + intercept`
    /// where `intercept = Y_mean - X_mean.dot(coef)`, with `X_mean`
    /// designating the average of the training predictors `X` (i.e features
    /// mean) and `Y_mean` designating the average(s) of target(s) `Y`, `dot`
    /// is the matrix multiplication.
    pub fit_intercept: bool,
    /// Optimization method, see [`LinearRegressionSolver`].
    pub solver: LinearRegressionSolver,
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
    /// - Gradient descent solvers (like [Sgd][`LinearRegressionSolver::Sgd`],
    ///   [Bgd][`LinearRegressionSolver::Bgd`], etc) stop when the relative
    ///   variation of consecutive iterates is lower than **tol**, that is:
    ///     - `||coef_next - coef_current|| <= tol * ||coef_current||`
    /// - No impact on the other algorithms:
    ///     - [Exact][`LinearRegressionSolver::Exact`]
    ///     - [Svd][`LinearRegressionSolver::Svd`]
    ///     - [Qr][`LinearRegressionSolver::Qr`]
    ///     - [Cholesky][`LinearRegressionSolver::Cholesky`]
    pub tol: Option<T>,
    /// Step size used in gradient descent algorithms.
    pub step_size: Option<T>,
    /// Seed of random generators used in gradient descent algorithms.
    pub random_state: Option<u32>,
    /// Maximum number of iterations used in gradient descent algorithms.
    pub max_iter: Option<usize>,
}

impl<T> Default for LinearRegressionSettings<T> {
    /// Defaults to linear regression without penalty
    /// ```
    /// use ndarray::{Array0, Array1, Array2};
    /// use njang::{LinearRegression, LinearRegressionSettings};
    ///
    /// let settings = LinearRegressionSettings::default();
    /// let model = LinearRegression::<Array1<f32>, Array0<f32>>::new(settings);
    /// assert!(model.is_linear());
    /// ```
    fn default() -> Self {
        Self {
            fit_intercept: true,
            solver: LinearRegressionSolver::default(),
            l1_penalty: None,
            l2_penalty: None,
            tol: None,
            step_size: None,
            random_state: None,
            max_iter: None,
        }
    }
}

/// Ordinary Least Squares (OLS) eventually penalized (Lasso with L1-penalty,
/// Ridge with L2-penalty and Elastic Net with L1 and L2 penalties).
///
/// Minimization of the objective function `loss`:
/// - `loss(coef) = 0.5 * ||X.dot(coef) - Y||`<sup>2</sup><sub>2</sub> `+
///   penalty(coef)`
///
/// with respect to the matrix `coef` of non-intercept weights, with
/// regressors/predictors `X` and targets `Y`, dot designates the matrix
/// multiplication.
///
/// **It is important to note** that the fitted model depends on how the user
/// sets the fields `*_penalty` in [LinearRegressionSettings]. See
/// [LinearRegressionSettings] for more details.
///
/// It is able to fit at once many linear regressions with the same input
/// predictors `X`, when `X` and `Y` are of type `Array2<T>` from ndarray crate.
/// **In this case, the same settings apply to all regressions involved**.
/// ```
/// use ndarray::{array, Array1, Array2};
/// use njang::{
///     LinearRegression, LinearRegressionSettings, LinearRegressionSolver, RegressionModel,
/// };
/// let x = array![[0., 1.], [1., -1.], [-2., 3.]];
/// let coef = array![[10., 30.], [20., 40.]];
/// let intercept = 1.;
/// let y = x.dot(&coef) + intercept;
/// // multiple linear regression models with intercept.
/// let mut model = LinearRegression::<Array2<f32>, Array1<f32>>::new(LinearRegressionSettings {
///     fit_intercept: true,
///     solver: LinearRegressionSolver::Exact,
///     ..Default::default()
/// });
/// model.fit(&x, &y);
/// assert!(
///     (model.coef().unwrap() - &coef)
///         .map(|error: &f32| error.powi(2))
///         .sum()
///         .sqrt()
///         < 1e-4
/// );
/// assert!(
///     (model.intercept().unwrap() - intercept)
///         .map(|error| error.abs())
///         .sum()
///         < 1e-4
/// );
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegression<C, I>
where
    C: Container,
{
    pub parameter: LinearModelParameter<C, I>,
    pub settings: LinearRegressionSettings<C::Elem>,
    pub(crate) internal: ModelInternal<C::Elem>,
}

macro_rules! impl_regression {
    ($ix:ty, $ix_smaller:ty, $randu:ident, $reshape_to_normal:ident, $reshape_to_2d:ident) => {
        impl<T: Scalar> LinearRegression<Array<T, $ix>, Array<T, $ix_smaller>> {
            fn linalg_solve<S>(
                &mut self,
                x: &Array2<T>,
                y: &Array<T, $ix>,
                solver: S,
            ) -> Result<Array<T, $ix>, NjangError>
            where
                S: Fn(
                    Array2<T>,
                    ArrayView2<T>,
                    &Array<T, $ix>,
                ) -> Result<Array<T, $ix>, LinalgError>,
            {
                if self.is_linear() | self.is_ridge() {
                    self.set_sample_to_internal(x, y);
                    let (n_features, n_samples) =
                        (self.internal.n_features, self.internal.n_samples);
                    let xt = x.t();
                    if n_samples >= n_features {
                        if self.is_linear() {
                            Ok(match solver(xt.dot(x), xt, y) {
                                Err(error) => return Err(NjangError::Linalg(error)),
                                Ok(value) => value,
                            })
                        } else {
                            self.set_l2_penalty_to_internal();
                            let penalty = self.internal.l2_penalty.unwrap();
                            Ok(
                                match solver((xt.dot(x) + Array2::eye(n_features) * penalty), xt, y)
                                {
                                    Err(error) => return Err(NjangError::Linalg(error)),
                                    Ok(value) => value,
                                },
                            )
                        }
                    } else {
                        if self.is_linear() {
                            let dual = match solver(x.dot(&xt), Array2::eye(n_samples).view(), y) {
                                Err(error) => return Err(NjangError::Linalg(error)),
                                Ok(value) => value,
                            };
                            Ok(xt.dot(&dual))
                        } else {
                            self.set_l2_penalty_to_internal();
                            let penalty = self.internal.l2_penalty.unwrap();
                            let eye = Array2::eye(n_samples);
                            let dual = match solver(x.dot(&xt) + &eye * penalty, eye.view(), y) {
                                Err(error) => return Err(NjangError::Linalg(error)),
                                Ok(value) => value,
                            };
                            Ok(xt.dot(&dual))
                        }
                    }
                } else {
                    return Err(NjangError::NotSupported { item: "Solver" });
                }
            }

            fn solve(
                &mut self,
                x: &Array2<T>,
                y: &Array<T, $ix>,
                fit_intercept: bool,
            ) -> Result<Array<T, $ix>, NjangError> {
                let solver = self.settings.solver;
                match solver {
                    LinearRegressionSolver::Svd => {
                        if self.is_linear() {
                            Ok(x.least_squares(y)?.solution)
                        } else if self.is_ridge() {
                            self.set_l2_penalty_to_internal();
                            let (xt, alpha) = (x.t(), self.internal.l2_penalty.unwrap());
                            let xtx_pen = xt.dot(x) + Array2::eye(x.ncols()) * alpha;
                            let xty = xt.dot(y);
                            Ok(match xtx_pen.least_squares(&xty) {
                                Ok(value) => value.solution,
                                Err(error) => return Err(NjangError::Linalg(error)),
                            })
                        } else {
                            return Err(NjangError::NotSupported { item: "Solver" });
                        }
                    }
                    LinearRegressionSolver::Exact => self.linalg_solve(x, y, exact),
                    LinearRegressionSolver::Qr => self.linalg_solve(x, y, qr),
                    LinearRegressionSolver::Cholesky => self.linalg_solve(x, y, cholesky),
                    LinearRegressionSolver::Sgd => {
                        self.set_internal(x, y);
                        // Rescale step_size and penalty(ies) to scale (sub)gradients correctly
                        self.scale_step_size();
                        self.scale_penalty();
                        let (n_targets, n_features, rng) = (
                            self.internal.n_targets,
                            self.internal.n_features,
                            self.internal.rng.as_mut().unwrap(),
                        );
                        let coef = $randu(&[n_features, n_targets], rng);
                        Ok(stochastic_gradient_descent(
                            x,
                            y,
                            coef,
                            self.gradient_function(),
                            &self.internal,
                        ))
                    }
                    LinearRegressionSolver::Bgd => {
                        self.set_internal(x, y);
                        let (n_targets, n_features, rng) = (
                            self.internal.n_targets,
                            self.internal.n_features,
                            self.internal.rng.as_mut().unwrap(),
                        );
                        let coef = $randu(&[n_features, n_targets], rng);
                        Ok(batch_gradient_descent(
                            x,
                            y,
                            coef,
                            self.gradient_function(),
                            &self.internal,
                        ))
                    }
                    LinearRegressionSolver::Sag => {
                        self.set_internal(x, y);
                        // Rescale step_size and penalty(ies) to scale gradients correctly
                        self.scale_step_size();
                        self.scale_penalty();
                        let n_targets = self.internal.n_targets;
                        if self.is_ridge() | self.is_elastic_net() {
                            // This scope resets the step size following the paper Schmidt, M.,
                            // Roux, N. L., & Bach, F. (2013)
                            let mut max_squared_sum = T::zero();
                            x.axis_iter(Axis(0))
                                .map(|r| {
                                    let norm = r.l2_norm();
                                    if max_squared_sum < norm {
                                        max_squared_sum = norm;
                                    }
                                })
                                .for_each(drop);
                            max_squared_sum = Float::powi(max_squared_sum, 2);
                            let fi = T::from(usize::from(fit_intercept)).unwrap();
                            let n_targets_in_t = T::from(n_targets).unwrap();
                            // At this stage the Ridge penalty is l2_penalty * n_targets / n_samples
                            // due to the call of self.scale_penalty(). So to get the normalized
                            // penalty l2_penalty / n_samples divide by n_targets.
                            let alpha_norm = self.internal.l2_penalty.unwrap() / n_targets_in_t;
                            // The step size should be scaled by 1 / n_targets due to scale
                            // correctly the (sub)gradient.
                            self.internal.step_size.as_mut().map(|p| {
                                *p = T::one()
                                    / ((max_squared_sum + fi + alpha_norm) * n_targets_in_t)
                            });
                        }
                        // TODO: Improvement needed in reshape_to_2d to avoid copy or broadcasting
                        // Do we really need to reshape to a 2d container ?
                        let y_2d = $reshape_to_2d(y);
                        let (n_features, rng) = (
                            self.internal.n_features,
                            self.internal.rng.as_mut().unwrap(),
                        );
                        let coef = randu_2d(&[n_features, n_targets], rng);
                        let gradient_function = self.gradient_function();
                        let (gradients, sum_gradients) =
                            init_grad(x, &y_2d, &coef, &gradient_function, &self.internal);
                        let coef = stochastic_average_gradient(
                            x,
                            &y_2d,
                            coef,
                            gradient_function,
                            &self.internal,
                            gradients,
                            sum_gradients,
                        );
                        Ok($reshape_to_normal(coef))
                    }
                }
            }
        }

        impl<'a, T: Scalar> Model<'a> for LinearRegression<Array<T, $ix>, Array<T, $ix_smaller>> {
            type FitResult = Result<(), NjangError>;
            type Data = (&'a Array2<T>, &'a Array<T, $ix>);
            fn fit(&mut self, data: &Self::Data) -> Self::FitResult {
                let (x, y) = data;
                let fit_intercept = self.settings.fit_intercept;
                if fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(*x, *y);
                    let coef = match self.solve(&x_centered, &y_centered, fit_intercept) {
                        Err(error) => return Err(error),
                        Ok(value) => value,
                    };
                    self.parameter.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.parameter.coef = Some(coef);
                } else {
                    self.parameter.coef = Some(match self.solve(x, y, fit_intercept) {
                        Err(error) => return Err(error),
                        Ok(value) => value,
                    });
                }
                Ok(())
            }
        }
        impl<'a, T: Scalar> RegressionModel
            for LinearRegression<Array<T, $ix>, Array<T, $ix_smaller>>
        {
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Result<Array<T, $ix>, ()>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> <Self as Model<'_>>::FitResult {
                let data = (x, y);
                <Self as Model>::fit(self, &data)
            }
            fn predict(&self, x: &Self::X) -> Self::PredictResult {
                if self.settings.fit_intercept {
                    if let Some(ref coef) = &self.parameter.coef {
                        if let Some(ref intercept) = &self.parameter.intercept {
                            return Ok(intercept + x.dot(coef));
                        }
                    }
                } else {
                    if let Some(ref coef) = &self.parameter.coef {
                        return Ok(x.dot(coef));
                    }
                }
                Err(())
            }
        }
    };
}

pub fn reshape_to_2d<T, Y>(y: &Y) -> Array2<T>
where
    T: Float,
    Y: Container<Elem = T>,
    for<'a> &'a Y: Add<Array2<T>, Output = Array2<T>>,
{
    let shape = y.dimension().to_vec();
    if shape.len() == 1 {
        (y + Array2::zeros((1, shape[0]))).t().to_owned()
    } else {
        y + Array2::zeros((shape[0], shape[1]))
    }
}

pub fn reshape_to_1d<T: Clone>(y: Array2<T>) -> Array1<T> {
    y.column(0).to_owned()
}
pub(crate) fn init_grad<T, G, S>(
    x: &Array2<T>,
    y: &Array2<T>,
    coef: &Array2<T>,
    scaled_grad: G,
    settings: &S,
) -> (Array2<T>, Array2<T>)
where
    for<'a> T: Lapack + ScalarOperand,
    G: Fn(&Array2<T>, &Array2<T>, &Array2<T>, &S) -> Array2<T>,
    S: LinearModelInternal<Scalar = T>,
{
    let (n_samples, n_features, n_targets) = (x.nrows(), x.ncols(), y.ncols());
    let mut grad = Array2::<_>::zeros((n_features, n_samples * n_targets));
    let mut sum_grad = Array2::<T>::zeros((n_features, n_targets));
    for k in 0..n_samples {
        let xi = x.selection(0, &[k]).to_owned();
        let yi = y.selection(0, &[k]).to_owned();
        let gradient = scaled_grad(&xi, &yi, &coef, settings);
        for t in 0..n_targets {
            let start = t * n_samples;
            (gradient.column(t).to_owned()).assign_to(grad.slice_mut(s!(.., start + k)));
        }
    }
    for t in 0..n_targets {
        grad.slice(s!(.., t * n_samples..(t + 1) * n_samples))
            .sum_axis(Axis(1))
            .assign_to(sum_grad.slice_mut(s!(.., t)));
    }
    return (grad, sum_grad);
}
impl_regression!(Ix1, Ix0, randu_1d, reshape_to_1d, reshape_to_2d);
impl_regression!(Ix2, Ix1, randu_2d, identity, reshape_to_2d);

// #[test]
// fn code() {
//     use ndarray::array;

//     let x = array![[0., 0., 1.], [1., 0., 0.]];
//     println!("x:\n{:?}\n", x);
//     println!("xxt:\n{:?}\n", x.dot(&x.t()));
//     let coef = array![[1., 1.], [2., 2.], [3., 3.]];
//     println!("coef:\n{:?}\n", coef);
//     let y = x.dot(&coef);
//     println!("y:\n{:?}\n", y);
//     println!("xty:\n{:?}\n", x.t().dot(&y));

//     let settings = LinearRegressionSettings {
//         fit_intercept: false,
//         solver: LinearRegressionSolver::Svd,
//         l1_penalty: None,
//         l2_penalty: None, //Some(1e-6),
//         tol: Some(1e-6),
//         step_size: Some(1e-3),
//         random_state: Some(0),
//         max_iter: Some(100000),
//     };
//     let mut model = LinearRegression::<Array2<_>, _>::new(settings);
//     match model.fit(&x, &y) {
//         Ok(_) => {
//             println!("{:?}", model.coef());
//             println!("{:?}", model.intercept());
//         }
//         Err(error) => {
//             println!("{:?}", error);
//         }
//     };
// }
