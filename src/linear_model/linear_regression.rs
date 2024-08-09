use ndarray::{linalg::Dot, Array, Array2, Ix0, Ix1, Ix2, ScalarOperand};

use crate::{linear_model::preprocess, traits::Info, RegressionModel};
use ndarray_linalg::{error::LinalgError, Inverse, Lapack, LeastSquaresSvd};

/// Solver to use when fitting a linear regression model (Ordinary Least Squares, OLS).
#[derive(Debug, Default)]
pub enum LinearRegressionSolver {
    /// Solves the problem using Singular Value Decomposition
    #[default]
    Svd,
    /// Exact solution of the OLS problem x.t().dot(x).inverse().dot(x.t()).dot(y)
    Exact,
}

/// Hyperparameters used in a linear regression model
#[derive(Debug)]
pub struct LinearRegressionHyperParameter {
    pub fit_intercept: bool,
    pub solver: LinearRegressionSolver,
}

/// Ordinary Least Squares (OLS)
///
/// Minimization of the L2-norm `||xb - y||`<sup/>2</sup> with respect to `b`, for regressors/predictors `x` and targets `y`.
///
/// The vector of coefficients `b = self.coef` if `self.fit_intercept = false` else  `b = (self.intercept, self.coef)'`.
///
/// It is able to fit at once many regressions with the same input regressors `x`, when `x` and `y` are of type `Array2<T>` from ndarray crate.
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
                + Info<MeanOutput = Array<T, Ix1>>,
            Array<T, Ix1>: Info<MeanOutput = Array<T, Ix0>>,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Option<Array<T, $ix>>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                if self.settings.fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(x, y);
                    match self.settings.solver {
                        LinearRegressionSolver::Svd => {
                            let res = x_centered.least_squares(&y_centered)?;
                            self.intercept = Some(y_mean - x_mean.dot(&res.solution));
                            self.coef = Some(res.solution);
                        }
                        LinearRegressionSolver::Exact => {
                            let xct = x_centered.t();
                            let coef = match xct.dot(&x_centered).inv() {
                                Ok(mat) => mat.dot(&xct).dot(&y_centered),
                                Err(error) => return Err(error),
                            };
                            self.intercept = Some(y_mean - x_mean.dot(&coef));
                            self.coef = Some(coef);
                        }
                    }
                } else {
                    match self.settings.solver {
                        LinearRegressionSolver::Svd => {
                            let res = x.least_squares(&y)?;
                            self.coef = Some(res.solution);
                        }
                        LinearRegressionSolver::Exact => {
                            let xt = x.t();
                            self.coef = Some(match xt.dot(x).inv() {
                                Ok(mat) => mat.dot(&xt).dot(y),
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
