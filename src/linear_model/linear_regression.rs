use ndarray::{linalg::Dot, Array, Array2, Axis, Ix0, Ix1, Ix2, ScalarOperand};

use crate::RegressionModel;
use ndarray_linalg::{error::LinalgError, Inverse, Lapack, LeastSquaresSvd};

/// Solver to use when fitting a linear regression model
pub enum LinearRegressionSolver {
    Svd,
    Exact,
}
impl Default for LinearRegressionSolver {
    fn default() -> Self {
        LinearRegressionSolver::Svd
    }
}

/// Ordinary-Least-Squares: minimization of L2-norm ||xb - y|| with respect to b.
///
/// The vector of coefficients b = self.coef if `self.fit_intercept = false` else (self.intercept, self.coef)'.
///
/// It is able to fit at once many regressions with the same input regressors `x`.
pub struct LinearRegression<C, I> {
    solver: LinearRegressionSolver,
    coef: Option<C>,
    intercept: Option<I>,
    pub(crate) fit_intercept: bool,
}

impl<C, I> LinearRegression<C, I> {
    pub fn new(fit_intercept: bool, solver: LinearRegressionSolver) -> Self {
        Self {
            solver,
            coef: None,
            intercept: None,
            fit_intercept,
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
            Array2<T>: Dot<Array2<T>, Output = Array2<T>>,
            Array2<T>: Dot<Array<T, Ix1>, Output = Array<T, Ix1>>,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Result<Array<T, $ix>, ()>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                match self.solver {
                    LinearRegressionSolver::Svd => {
                        if self.fit_intercept {
                            let x_mean = x.mean_axis(Axis(0)).unwrap();
                            let x_centered = x - &x_mean;
                            let y_mean = y.mean_axis(Axis(0)).unwrap();
                            let y_centered = y - &y_mean;
                            let res = x_centered.least_squares(&y_centered)?;
                            self.intercept = Some(y_mean - x_mean.dot(&res.solution));
                            self.coef = Some(res.solution);
                        } else {
                            let res = x.least_squares(&y)?;
                            self.coef = Some(res.solution);
                        }
                    }
                    LinearRegressionSolver::Exact => {
                        if self.fit_intercept {
                            let x_mean = x.mean_axis(Axis(0)).unwrap();
                            let x_centered = x - &x_mean;
                            let y_mean = y.mean_axis(Axis(0)).unwrap();
                            let y_centered = y - &y_mean;
                            let xct = x_centered.t();
                            let coef = match x_centered.dot(&xct).inv() {
                                Ok(mat) => mat.dot(&xct).dot(&y_centered),
                                Err(error) => return Err(error),
                            };
                            self.intercept = Some(y_mean - x_mean.dot(&coef));
                            self.coef = Some(coef);
                        } else {
                            let xt = x.t();
                            self.coef = Some(match x.dot(&xt).inv() {
                                Ok(mat) => mat.dot(&xt).dot(y),
                                Err(error) => return Err(error),
                            });
                        }
                    }
                }
                Ok(())
            }
            fn predict(&self, x: &Self::X) -> Self::PredictResult {
                if self.fit_intercept {
                    if let Some(ref coef) = &self.coef {
                        if let Some(ref intercept) = &self.intercept {
                            Ok(intercept + x.dot(coef))
                        } else {
                            panic!("No intercept")
                        }
                    } else {
                        panic!("No coef")
                    }
                } else {
                    if let Some(ref coef) = &self.coef {
                        Ok(x.dot(coef))
                    } else {
                        panic!("No coef")
                    }
                }
            }
        }
    };
}
impl_lin_reg!(Ix1, Ix0);
impl_lin_reg!(Ix2, Ix1);
