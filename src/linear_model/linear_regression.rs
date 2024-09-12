use core::ops::{Mul, Sub};

use crate::{
    batch_gradient_descent,
    linear_model::{
        preprocess, solve_chol1, solve_chol2, solve_exact1, solve_exact2, solve_qr1, solve_qr2,
    },
    stochastic_gradient_descent,
    traits::Container,
    RegressionModel,
};
use ndarray::{linalg::Dot, Array, Array2, ArrayView2, Ix0, Ix1, Ix2, ScalarOperand};
use ndarray_linalg::{error::LinalgError, Lapack, LeastSquaresSvd};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive};

use super::{randn_1d, randn_2d, square_loss_gradient};
use rand_chacha::ChaCha20Rng;

use ndarray_rand::rand::SeedableRng;
// use ndarray_rand::RandomExt;
/// Solver to use when fitting a linear regression model (Ordinary Least
/// Squares, OLS).
#[derive(Debug, Default, Clone, Copy)]
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
    /// Stochastic Gradient Descent
    SGD,
    /// Batch Gradient Descent
    BGD,
}

/// Hyperparameters used in a linear regression model
///
/// - **fit_intercept**: `true` means fit with an intercept, `false` without an
///   intercept.
/// - **solver**: optimization method see [`LinearRegressionSolver`].
/// - etc
/// ```
/// panic!("Add doc")
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct LinearRegressionSettings<T> {
    pub fit_intercept: bool,
    pub solver: LinearRegressionSolver,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LinearRegressionParameter<C, I> {
    pub coef: Option<C>,
    pub intercept: Option<I>,
}

/// Ordinary Least Squares (OLS).
///
/// Minimization of the L2-norm `||Xb - Y||`<sup/>2</sup> with respect to `b`,
/// for regressors/predictors `X` and targets `Y`.
///
/// The vector of coefficients satisfies:
/// - if `self.fit_intercept = false`, then `Xb = X*self.coef`
/// - if `self.fit_intercept = true`, then `Xb = X*self.coef + self.intercept`.
///
/// It is able to fit at once many regressions with the same input regressors
/// `X`, when `X` and `Y` are of type `Array2<T>` from ndarray crate.
/// The same hyperparameter `fit_intercept` applies for all regressions
/// involved.
/// ```
/// use ndarray::{array, Array1, Array2};
/// use njang::{
///     LinearRegression, LinearRegressionSettings, LinearRegressionSolver, RegressionModel,
/// };
/// let x = array![[0., 1.], [1., -1.], [-2., 3.]];
/// let coef = array![[10., 30.], [20., 40.]];
/// let y = x.dot(&coef) + 1.;
/// // multiple linear regression models with intercept.
/// let mut model = LinearRegression::<Array2<f32>, Array1<f32>>::new(LinearRegressionSettings {
///     fit_intercept: true,
///     solver: LinearRegressionSolver::EXACT,
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
/// ```
#[derive(Debug, Clone)]
pub struct LinearRegression<C, I>
where
    C: Container,
{
    pub parameter: LinearRegressionParameter<C, I>,
    pub settings: LinearRegressionSettings<C::Elem>,
    internal: LinearRegressionInternal<C::Elem>,
}

/// This is responsible for processing settings, setting default values
#[derive(Debug, Clone)]
pub(crate) struct LinearRegressionInternal<T> {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_targets: usize,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub rng: Option<ChaCha20Rng>,
    pub max_iter: Option<usize>,
}

impl<T> LinearRegressionInternal<T> {
    pub fn new() -> Self {
        Self {
            n_samples: 0,
            n_features: 0,
            n_targets: 0,
            tol: None,
            step_size: None,
            rng: None,
            max_iter: None,
        }
    }
}

impl<C: Container, I> LinearRegression<C, I> {
    pub fn new(settings: LinearRegressionSettings<C::Elem>) -> Self
    where
        C::Elem: Float,
    {
        Self {
            parameter: LinearRegressionParameter {
                coef: None,
                intercept: None,
            },
            settings,
            internal: LinearRegressionInternal::new(),
        }
    }
    /// Coefficients of the model
    pub fn coef(&self) -> Option<&C> {
        self.parameter.coef.as_ref()
    }
    /// Intercept of the model
    pub fn intercept(&self) -> Option<&I> {
        self.parameter.intercept.as_ref()
    }
    fn set_internal<X: Container>(&mut self, x: &X, y: &C)
    where
        C::Elem: Copy + FromPrimitive,
    {
        self.set_sample_to_internal(x, y);
        self.set_settings_to_internal();
    }
    fn set_sample_to_internal<X: Container>(&mut self, x: &X, y: &C) {
        // x should be a 2 dimensional container, and <X as Container>::dimension should
        // return something like [nrows, ncols] of X.
        let dim = x.dimension();
        let (n_samples, n_features) = (dim[0], dim[1]);
        self.internal.n_features = n_features;
        self.internal.n_samples = n_samples;
        let shape = y.dimension();
        let n_targets = if shape.len() == 2 { shape[1] } else { 1 };
        self.internal.n_targets = n_targets;
    }
    fn set_settings_to_internal(&mut self)
    where
        C::Elem: Copy + FromPrimitive,
    {
        self.internal.tol = Some(
            self.settings
                .tol
                .unwrap_or(C::Elem::from_f32(1e-4).unwrap()),
        );
        self.internal.step_size = Some(
            self.settings
                .step_size
                .unwrap_or(C::Elem::from_f32(1e-3).unwrap()),
        );
        let random_state = self.settings.random_state.unwrap_or(0);
        self.internal.rng = Some(ChaCha20Rng::seed_from_u64(random_state as u64));
        self.internal.max_iter = Some(self.settings.max_iter.unwrap_or(1000));
        // sel
    }
}

macro_rules! impl_lin_reg {
    ($ix:ty, $ix_smaller:ty, $exact_name:ident, $qr_name:ident, $chol_name:ident, $randn:ident) => {
        impl<T> RegressionModel for LinearRegression<Array<T, $ix>, Array<T, $ix_smaller>>
        where
            T: Lapack + ScalarOperand + PartialOrd + Float + SampleUniform,
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
                        LinearRegressionSolver::SGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            // Rescale step_size to scale gradients correctly
                            // [Specific to this algorithm]
                            self.internal
                                .step_size
                                .as_mut()
                                .map(|s| *s = *s / T::from(n_targets).unwrap());
                            let coef = $randn(n_features, y.dimension(), rng);
                            stochastic_gradient_descent(
                                &x_centered,
                                &y_centered,
                                coef,
                                linear_regression_gradient,
                                &self.internal,
                            )
                        }
                        LinearRegressionSolver::BGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            // Rescale step_size to scale gradients correctly
                            // [Specific to this algorithm]
                            self.internal
                                .step_size
                                .as_mut()
                                .map(|s| *s = *s / T::from(n_targets).unwrap());
                            let coef = $randn(n_features, y.dimension(), rng);
                            batch_gradient_descent(
                                &x_centered,
                                &y_centered,
                                coef,
                                linear_regression_gradient,
                                &self.internal,
                            )
                        }
                    };
                    self.parameter.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.parameter.coef = Some(coef);
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
                        LinearRegressionSolver::SGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            // Rescale step_size to scale gradients correctly
                            // [Specific to this algorithm]
                            self.internal
                                .step_size
                                .as_mut()
                                .map(|s| *s = *s / T::from(n_targets).unwrap());
                            let coef = $randn(n_features, y.dimension(), rng);
                            stochastic_gradient_descent(
                                x,
                                y,
                                coef,
                                linear_regression_gradient,
                                &self.internal,
                            )
                        }
                        LinearRegressionSolver::BGD => {
                            self.set_internal(x, y);
                            let (n_features, rng) = (
                                self.internal.n_features,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            let coef = $randn(n_features, y.dimension(), rng);
                            batch_gradient_descent(
                                x,
                                y,
                                coef,
                                linear_regression_gradient,
                                &self.internal,
                            )
                        }
                    };
                    self.parameter.coef = Some(coef);
                }
                Ok(())
            }
            fn predict(&self, x: &Self::X) -> Self::PredictResult {
                if self.settings.fit_intercept {
                    if let Some(ref coef) = &self.parameter.coef {
                        if let Some(ref intercept) = &self.parameter.intercept {
                            return Some(intercept + x.dot(coef));
                        }
                    }
                } else {
                    if let Some(ref coef) = &self.parameter.coef {
                        return Some(x.dot(coef));
                    }
                }
                None
            }
        }
    };
}
impl_lin_reg!(Ix1, Ix0, solve_exact1, solve_qr1, solve_chol1, randn_1d);
impl_lin_reg!(Ix2, Ix1, solve_exact2, solve_qr2, solve_chol2, randn_2d);

fn linear_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &LinearRegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y> + Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    return square_loss_gradient(x, y, coef) * (-step_size);
}

#[test]
fn code() {
    use ndarray::*;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use rand_chacha::ChaCha20Rng;
    let settings = LinearRegressionSettings {
        fit_intercept: true,
        max_iter: Some(10000),
        solver: LinearRegressionSolver::SGD,
        tol: Some(1e-20),
        random_state: Some(0),
        step_size: Some(1e-3),
    };
    let mut model: LinearRegression<Array1<f32>, Array0<f32>> = LinearRegression {
        parameter: LinearRegressionParameter {
            coef: None,
            intercept: None,
        },
        settings: settings,
        internal: LinearRegressionInternal::new(),
    };
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let p = 10;
    let x = Array::<f32, Ix2>::random_using((100000, p), StandardNormal, &mut rng);
    let coef = Array1::from((1..p + 1).map(|val| val as f32).collect::<Vec<_>>());
    let y = x.dot(&coef);
    model.fit(&x, &y);

    let mut model: LinearRegression<Array2<f32>, Array1<f32>> = LinearRegression {
        parameter: LinearRegressionParameter {
            coef: None,
            intercept: None,
        },
        internal: LinearRegressionInternal::new(),
        settings: settings,
    };

    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let p = 10;
    let x = Array::<f32, Ix2>::random_using((100000, p), StandardNormal, &mut rng);
    let r = 10;
    // let x = (&x - x.mean_axis(Axis(0)).unwrap()) / x.std_axis(Axis(0), 0.);
    let coef = Array2::from_shape_vec(
        (p, r),
        (1..p * r + 1).map(|val| val as f32).collect::<Vec<_>>(),
    )
    .unwrap();
    let intercept = Array1::from_iter((1..r + 1).map(|val| val as f32));
    let y = x.dot(&coef) + intercept;
    model.fit(&x, &y);
    println!("{:?}", model.coef());
    println!("{:?}", model.intercept());
}
