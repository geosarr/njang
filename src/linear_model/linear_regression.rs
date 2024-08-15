use ndarray::{linalg::Dot, Array, Array2, Ix0, Ix1, Ix2, ScalarOperand};

use crate::{linear_model::preprocess, traits::Info, RegressionModel};
use ndarray_linalg::{error::LinalgError, Inverse, Lapack, LeastSquaresSvd, QR};

/// Solver to use when fitting a linear regression model (Ordinary Least Squares, OLS).
#[derive(Debug, Default)]
pub enum LinearRegressionSolver {
    /// Solves the problem using Singular Value Decomposition
    #[default]
    SVD,
    /// Computes the exact solution:
    /// - `x.t().dot(x).inverse().dot(x.t()).dot(y)`
    EXACT,
    /// Uses QR decomposition to solve the problem.
    QR,
}

/// Hyperparameters used in a linear regression model
///
/// - **fit_intercept**: `true` means fit with an intercept, `false` without an intercept.
/// - **solver**: optimization method see [`LinearRegressionSolver`].
#[derive(Debug, Default)]
pub struct LinearRegressionHyperParameter {
    pub fit_intercept: bool,
    pub solver: LinearRegressionSolver,
}

/// Ordinary Least Squares (OLS).
///
/// Minimization of the L2-norm `||Xb - Y||`<sup/>2</sup> with respect to `b`, for regressors/predictors `X` and targets `Y`.
///
/// The vector of coefficients satisfies:
/// - if `self.fit_intercept = false`, then `Xb = X*self.coef`
/// - if `self.fit_intercept = true`, then `Xb = X*self.coef + self.intercept`.
///
/// It is able to fit at once many regressions with the same input regressors `X`, when `X` and `Y` are of type `Array2<T>` from ndarray crate.
/// The same hyperparameter `fit_intercept` applies for all regressions involved.
/// ```
/// use ndarray::{array, Array1, Array2};
/// use njang::{LinearRegression, LinearRegressionHyperParameter, LinearRegressionSolver, RegressionModel};
/// let x = array![[0., 1.], [1., -1.], [-2., 3.]];
/// let coef = array![[10., 30.], [20., 40.]];
/// let y = x.dot(&coef) + 1.;
/// // multiple linear regression models with intercept.
/// let mut model = LinearRegression::<Array2<f32>, Array1<f32>>::new(LinearRegressionHyperParameter {
///     fit_intercept: true,
///     solver: LinearRegressionSolver::EXACT,
/// });
/// model.fit(&x, &y);
/// assert!((model.coef().unwrap() - &coef).map(|error: &f32| error.powi(2)).sum().sqrt() < 1e-6);
/// ```
#[derive(Debug)]
pub struct LinearRegression<C, I> {
    coef: Option<C>,
    intercept: Option<I>,
    settings: LinearRegressionHyperParameter,
}

impl<C, I> LinearRegression<C, I> {
    pub fn new(settings: LinearRegressionHyperParameter) -> Self {
        Self {
            settings,
            coef: None,
            intercept: None,
        }
    }
    /// Coefficients of the model
    pub fn coef(&self) -> Option<&C> {
        self.coef.as_ref()
    }
    /// Intercept of the model
    pub fn intercept(&self) -> Option<&I> {
        self.intercept.as_ref()
    }
}
macro_rules! impl_lin_reg {
    ($ix:ty, $ix_smaller:ty) => {
        impl<T> RegressionModel for LinearRegression<Array<T, $ix>, Array<T, $ix_smaller>>
        where
            T: Lapack + ScalarOperand,
            Array2<T>: Dot<Array2<T>, Output = Array2<T>>
                + Dot<Array<T, Ix1>, Output = Array<T, Ix1>>
                + Info<MeanOutput = Array<T, Ix1>>
                + QR<Q = Array2<T>, R = Array2<T>>,
            Array<T, Ix1>: Info<MeanOutput = Array<T, Ix0>>,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Option<Array<T, $ix>>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                if self.settings.fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(x, y);
                    let coef = match self.settings.solver {
                        LinearRegressionSolver::SVD => {
                            x_centered.least_squares(&y_centered)?.solution
                        }
                        LinearRegressionSolver::EXACT => {
                            let xct = x_centered.t();
                            match xct.dot(&x_centered).inv() {
                                Ok(mat) => mat.dot(&xct).dot(&y_centered),
                                Err(error) => return Err(error),
                            }
                        }
                        LinearRegressionSolver::QR => {
                            let (q, r) = match x_centered.qr() {
                                Ok((q, r)) => (q, r),
                                Err(error) => return Err(error),
                            };
                            match r.inv() {
                                Ok(inv_r) => inv_r.dot(&q.t().dot(&y_centered)),
                                Err(error) => return Err(error),
                            }
                        }
                    };
                    self.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.coef = Some(coef);
                } else {
                    match self.settings.solver {
                        LinearRegressionSolver::SVD => {
                            let res = x.least_squares(&y)?;
                            self.coef = Some(res.solution);
                        }
                        LinearRegressionSolver::EXACT => {
                            let xt = x.t();
                            self.coef = Some(match xt.dot(x).inv() {
                                Ok(mat) => mat.dot(&xt).dot(y),
                                Err(error) => return Err(error),
                            });
                        }
                        LinearRegressionSolver::QR => {
                            let (q, r) = match x.qr() {
                                Ok((q, r)) => (q, r),
                                Err(error) => return Err(error),
                            };
                            self.coef = Some(match r.inv() {
                                Ok(inv_r) => inv_r.dot(&q.t().dot(y)),
                                Err(error) => return Err(error),
                            });
                        }
                    }
                }
                Ok(())
            }
            fn predict(&self, x: &Self::X) -> Self::PredictResult {
                if self.settings.fit_intercept {
                    if let Some(ref coef) = &self.coef {
                        if let Some(ref intercept) = &self.intercept {
                            return Some(intercept + x.dot(coef));
                        }
                    }
                } else {
                    if let Some(ref coef) = &self.coef {
                        return Some(x.dot(coef));
                    }
                }
                None
            }
        }
    };
}
impl_lin_reg!(Ix1, Ix0);
impl_lin_reg!(Ix2, Ix1);
