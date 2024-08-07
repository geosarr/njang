use ndarray::{linalg::Dot, Array, Array2, Axis, Ix0, Ix1, Ix2, ScalarOperand};

mod unit_test;
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
    fit_intercept: bool,
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
            type PredictResult = Result<(), ()>;
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
            fn predict(&self, _x: &Self::X) -> Self::PredictResult {
                Ok(())
            }
        }
    };
}
impl_lin_reg!(Ix1, Ix0);
impl_lin_reg!(Ix2, Ix1);

// fn gradient_descent<F, T>(f: F, x: &Array2<T>, y: &Array1<T>)
// where
//     F: Fn(&Array1<T>) -> T,
// {
// }
// impl LinearRegression<Array1<f32>, f32>
// // where
// //     T: Number,
// {
//     pub fn fit_mse(&mut self, x: &Array2<f32>, y: &Array1<f32>)
//     // where
//     // for<'a> &'a Array1<T>:
//     //     Sub<&'a Array1<T>, Output = Array1<T>> + Sub<Array1<T>, Output = Array1<T>>,
//     // Array2<T>: ndarray::linalg::Dot<Array1<T>, Output = Array1<T>>,
//     // T: Float,
//     // T: Scalar<Array2<T>> + Scalar<Array1<T>> + Default,
//     {
//         use tuutal::steepest_descent;
//         let (n_features, n_samples) = (x.ncols(), x.nrows());
//         // let xt = T::cast_from_f32(2. / (n_samples as f32)) * x.t().to_owned();
//         // let eta = T::cast_from_f32(0.001);
//         let xt = (2. / (n_samples as f32)) * x.t().to_owned();
//         let f = |coef: &Array1<f32>| (x.dot(coef) - y).map(|delta| delta.powi(2)).sum();
//         let gradf = |coef: &Array1<f32>| xt.dot(&(x.dot(coef) - y));
//         let f_gradf = |coef: &Array1<f32>| {
//             let error = x.dot(coef) - y;
//             (
//                 error.map(|delta| delta.powi(2)).sum(),
//                 xt.dot(&error),
//                 error,
//             )
//         };
//         // self.coef = Array1::default(n_features);
//         let x0 = Array1::default(n_features);
//         let x0 = steepest_descent(f, gradf, &x0, &Default::default(), 1e-4, 10000);
//         println!("{:?}", x0);
//         // for k in 0..10000 {
//         //     self.coef = eta * xt.dot(&(y - &x.dot(&self.coef))) + &self.coef;
//         // }
//     }
// }

// // macro_rules! impl_scalar(
// //     ( $( $t:ident ),* )=> {
// //         $(
// //           impl Scalar<Array1<$t>> for $t {}
// //           impl Scalar<Array2<$t>> for $t {}
// //         )*
// //     }
// //   );
// // impl_scalar!(f32, f64);
