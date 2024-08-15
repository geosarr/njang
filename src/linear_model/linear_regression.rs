use ndarray::{linalg::Dot, Array, Array2, Ix0, Ix1, Ix2, ScalarOperand};

use crate::{
    linear_model::{
        preprocess, solve_chol1, solve_chol2, solve_exact1, solve_exact2, solve_qr1, solve_qr2,
    },
    traits::Info,
    RegressionModel,
};
use ndarray_linalg::{error::LinalgError, Lapack, LeastSquaresSvd, QR};

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
    /// Uses Cholesky decomposition
    CHOLESKY,
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
/// assert!((model.coef().unwrap() - &coef).map(|error: &f32| error.powi(2)).sum().sqrt() < 1e-4);
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
    ($ix:ty, $ix_smaller:ty, $exact_name:ident, $qr_name:ident, $chol_name:ident) => {
        impl<T> RegressionModel for LinearRegression<Array<T, $ix>, Array<T, $ix_smaller>>
        where
            T: Lapack + ScalarOperand,
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
                            $exact_name(xct.dot(&x_centered), xct, &y_centered)?
                        }
                        LinearRegressionSolver::QR => {
                            let xct = x_centered.t();
                            $qr_name(xct.dot(&x_centered), xct, &y_centered)?
                        }
                        LinearRegressionSolver::CHOLESKY => {
                            let xct = x_centered.t();
                            $chol_name(xct.dot(&x_centered), xct, &y_centered)?
                        }
                    };
                    self.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.coef = Some(coef);
                } else {
                    let coef = match self.settings.solver {
                        LinearRegressionSolver::SVD => x.least_squares(&y)?.solution,
                        LinearRegressionSolver::EXACT => {
                            let xt = x.t();
                            $exact_name(xt.dot(x), xt, y)?
                        }
                        LinearRegressionSolver::QR => {
                            let xt = x.t();
                            $qr_name(xt.dot(x), xt, y)?
                        }
                        LinearRegressionSolver::CHOLESKY => {
                            let xt = x.t();
                            $chol_name(xt.dot(x), xt, y)?
                        }
                    };
                    self.coef = Some(coef);
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
impl_lin_reg!(Ix1, Ix0, solve_exact1, solve_qr1, solve_chol1);
impl_lin_reg!(Ix2, Ix1, solve_exact2, solve_qr2, solve_chol2);
