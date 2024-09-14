use crate::{
    linear_model::{cholesky, exact, preprocess, qr, randn_1d, randn_2d, square_loss_gradient},
    solver::{batch_gradient_descent, stochastic_gradient_descent},
    traits::{Algebra, Container, RegressionModel},
};
use core::ops::{Add, Mul, Sub};
use ndarray::{
    linalg::Dot, Array, Array2, ArrayView2, Dimension, Ix0, Ix1, Ix2, OwnedRepr, ScalarOperand,
};
use ndarray_linalg::{error::LinalgError, Lapack, LeastSquaresSvd};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive};
use rand_chacha::ChaCha20Rng;

#[derive(Debug, Default, Clone, Copy)]
pub enum RegressionSolver {
    /// Uses Singular Value Decomposition
    Svd,
    // /// Computes the exact solution
    // Exact,
    // /// Uses QR decomposition to solve the problem.
    // Qr,
    // /// Uses Cholesky decomposition
    // Cholesky,
    /// Uses Stochastic Gradient Descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm
    /// may not converge.
    #[default]
    Sgd,
    /// Uses Batch Gradient Descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm
    /// may not converge.
    Bgd,
    // /// Solves the problem Stochastic Average Gradient
    // ///
    // /// Make sure to standardize the input predictors, otherwise the algorithm
    // /// may not converge.
    // SAG,
}

/// This is responsible for processing settings, setting default values
#[derive(Debug, Clone)]
pub(crate) struct RegressionInternal<T> {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_targets: usize,
    pub l1_penalty: Option<T>, // for Lasso Regression
    pub l2_penalty: Option<T>, // for Ridge Regression
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub rng: Option<ChaCha20Rng>,
    pub max_iter: Option<usize>,
}

impl<T> RegressionInternal<T> {
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

/// Hyperparameters used in a regression model
///
/// - **fit_intercept**: `true` means fit with an intercept, `false` without an
///   intercept.
/// - **solver**: optimization method see [`RegressionSolver`].
/// - etc
/// ```
/// panic!("Add doc")
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct RegressionSettings<T> {
    pub fit_intercept: bool,
    pub solver: RegressionSolver,
    pub l1_penalty: Option<T>, // for Lasso Regression
    pub l2_penalty: Option<T>, // for Ridge Regression
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RegressionParameter<C, I> {
    pub coef: Option<C>,
    pub intercept: Option<I>,
}

/// Ordinary Least Squares (OLS) eventually penalized (Lasso with L1-penalty,
/// Ridge with L2-penalty and Elastic Net with L1 and L2 penalties).
///
/// Minimization of the L2-norm `||Xb - Y||`<sup/>2</sup> with respect to `b`,
/// for regressors/predictors `X` and targets `Y`.
///
/// The vector of coefficients satisfies:
/// - if `self.fit_intercept = false`, then `Xb = X*self.coef`
/// - if `self.fit_intercept = true`, then `Xb = X*self.coef + self.intercept`.
///
/// It is able to fit at once many regressions with the same input regressors
/// `X`, when `X` and `Y` are of type `Array2<T>` from ndarray crate. The same
/// hyperparameters apply to all regressions involved.
/// ```
/// use ndarray::{array, Array1, Array2};
/// use njang::{Regression, RegressionModel, RegressionSettings, RegressionSolver};
/// let x = array![[0., 1.], [1., -1.], [-2., 3.]];
/// let coef = array![[10., 30.], [20., 40.]];
/// let y = x.dot(&coef) + 1.;
/// // multiple regression models with intercept.
/// let mut model = Regression::<Array2<f32>, Array1<f32>>::new(RegressionSettings {
///     fit_intercept: true,
///     solver: RegressionSolver::Exact,
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
pub struct Regression<C, I>
where
    C: Container,
{
    pub parameter: RegressionParameter<C, I>,
    pub settings: RegressionSettings<C::Elem>,
    internal: RegressionInternal<C::Elem>,
}

impl<C: Container, I> Regression<C, I> {
    pub fn new(settings: RegressionSettings<C::Elem>) -> Self
    where
        C::Elem: Float,
    {
        Self {
            parameter: RegressionParameter {
                coef: None,
                intercept: None,
            },
            settings,
            internal: RegressionInternal::new(),
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
    fn set_rng_to_internal(&mut self) {
        let random_state = self.settings.random_state.unwrap_or(0);
        self.internal.rng = Some(ChaCha20Rng::seed_from_u64(random_state as u64));
    }
    fn set_max_iter_to_internal(&mut self) {
        self.internal.max_iter = Some(self.settings.max_iter.unwrap_or(1000));
    }
    fn scale_step_size(&mut self)
    where
        C::Elem: Float + FromPrimitive,
    {
        let n_targets = C::Elem::from_usize(self.internal.n_targets).unwrap();
        self.internal
            .step_size
            .as_mut()
            .map(|s| *s = *s / n_targets);
    }
    fn scale_l1_penalty(&mut self)
    where
        C::Elem: Float + FromPrimitive,
    {
        let n_targets = C::Elem::from_usize(self.internal.n_targets).unwrap();
        let n_samples = C::Elem::from_usize(self.internal.n_samples).unwrap();
        self.internal
            .l1_penalty
            .as_mut()
            .map(|p| *p = *p * n_targets / n_samples);
    }
    fn scale_l2_penalty(&mut self)
    where
        C::Elem: Float + FromPrimitive,
    {
        let n_targets = C::Elem::from_usize(self.internal.n_targets).unwrap();
        let n_samples = C::Elem::from_usize(self.internal.n_samples).unwrap();
        self.internal
            .l2_penalty
            .as_mut()
            .map(|p| *p = *p * n_targets / n_samples);
    }
}

macro_rules! impl_settings_to_internal {
    ($setter_name:ident, $field_name:ident, $default:ident) => {
        impl<C: Container, I> Regression<C, I>
        where
            C::Elem: Copy + FromPrimitive,
        {
            fn $setter_name(&mut self) {
                self.internal.$field_name = Some(
                    self.settings
                        .$field_name
                        .unwrap_or(C::Elem::from_f32($default).unwrap()),
                );
            }
        }
    };
}
const DEFAULT_L1: f32 = 1.;
const DEFAULT_L2: f32 = 1.;
const DEFAULT_TOL: f32 = 1e-3;
const DEFAULT_STEP_SIZE: f32 = 1e-3;
impl_settings_to_internal!(set_l1_penalty_to_internal, l1_penalty, DEFAULT_L1);
impl_settings_to_internal!(set_l2_penalty_to_internal, l2_penalty, DEFAULT_L2);
impl_settings_to_internal!(set_tol_to_internal, tol, DEFAULT_TOL);
impl_settings_to_internal!(set_step_size_to_internal, tol, DEFAULT_STEP_SIZE);

macro_rules! impl_regression {
    ($ix:ty, $ix_smaller:ty, $randn:ident) => {
        impl<T: Clone> Regression<Array<T, $ix>, Array<T, $ix_smaller>> {
            fn solve(
                &mut self,
                x: &Array2<T>,
                y: &Array<T, $ix>,
            ) -> Result<Array<T, $ix>, LinalgError>
            where
                T: Lapack + PartialOrd + Float + ScalarOperand + SampleUniform,
            {
                match self.settings.solver {
                    RegressionSolver::Svd => Ok(x.least_squares(y)?.solution),
                    // RegressionSolver::Exact => {
                    //     let xct = x.t();
                    //     exact(xct.dot(&x), xct, &y)?
                    // }
                    // RegressionSolver::Qr => {
                    //     let xct = x.t();
                    //     qr(xct.dot(&x), xct, &y)?
                    // }
                    // RegressionSolver::Cholesky => {
                    //     let xct = x.t();
                    //     cholesky(xct.dot(&x), xct, &y)?
                    // }
                    RegressionSolver::Sgd => {
                        // self.set_internal(x, y);
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
                        let coef = $randn(&[n_features, n_targets], rng);
                        Ok(stochastic_gradient_descent(
                            x,
                            y,
                            coef,
                            linear_regression_gradient, // TODO change
                            &self.internal,
                        ))
                    }
                    RegressionSolver::Bgd => {
                        // self.set_internal(x, y);
                        let (n_targets, n_features, rng) = (
                            self.internal.n_targets,
                            self.internal.n_features,
                            self.internal.rng.as_mut().unwrap(),
                        );
                        let coef = $randn(&[n_features, n_targets], rng);
                        Ok(batch_gradient_descent(
                            x,
                            y,
                            coef,
                            linear_regression_gradient, // TODO change
                            &self.internal,
                        ))
                    }
                }
            }
        }
        impl<T: Clone> RegressionModel for Regression<Array<T, $ix>, Array<T, $ix_smaller>>
        where
            T: Lapack + ScalarOperand + PartialOrd + Float + SampleUniform,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Result<Array<T, $ix>, ()>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                if self.settings.fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(x, y);
                    let coef = self.solve(&x_centered, &y_centered)?;
                    self.parameter.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.parameter.coef = Some(coef);
                } else {
                    self.parameter.coef = Some(self.solve(x, y)?);
                }
                Ok(())
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
impl_regression!(Ix1, Ix0, randn_1d);
impl_regression!(Ix2, Ix1, randn_2d);
// fn solve<T, R>(
//     x: &Array2<T>,
//     y: &Array<T, D>,
//     settings: RegressionSettings<T>,
//     internal: &mut RegressionInternal<T>,
//     rand_init: R,
// ) -> Result<Array<T, D>, LinalgError>
// where
//     T: Lapack + PartialOrd + Float + ScalarOperand + SampleUniform,
//     R: Fn(&[usize], &mut ChaCha20Rng) -> Array<T, D>,
//     D: Dimension,
//     Array2<T>: LeastSquaresSvd<OwnedRepr<T>, T, D> + Dot<Array<T, D>, Output
// = Array<T, D>>,     for<'a> ArrayView2<'a, T>: Dot<Array<T, D>, Output =
// Array<T, D>>,     Array<T, D>: Algebra<Elem = T, SelectionOutput = Array<T,
// D>>, {
//     match settings.solver {
//         RegressionSolver::Svd => Ok(x.least_squares(y)?.solution),
//         // RegressionSolver::Exact => {
//         //     let xct = x.t();
//         //     exact(xct.dot(&x), xct, &y)?
//         // }
//         // RegressionSolver::Qr => {
//         //     let xct = x.t();
//         //     qr(xct.dot(&x), xct, &y)?
//         // }
//         // RegressionSolver::Cholesky => {
//         //     let xct = x.t();
//         //     cholesky(xct.dot(&x), xct, &y)?
//         // }
//         RegressionSolver::Sgd => {
//             // self.set_internal(x, y);
//             let (n_targets, n_features, rng) = (
//                 internal.n_targets,
//                 internal.n_features,
//                 internal.rng.as_mut().unwrap(),
//             );
//             // Rescale step_size to scale gradients correctly
//             // [Specific to this algorithm]
//             internal
//                 .step_size
//                 .as_mut()
//                 .map(|s| *s = *s / T::from(n_targets).unwrap());
//             let coef = rand_init(&[n_features, n_targets], rng);
//             Ok(stochastic_gradient_descent(
//                 x,
//                 y,
//                 coef,
//                 linear_regression_gradient,
//                 &*internal,
//             ))
//         }
//         RegressionSolver::Bgd => {
//             // self.set_internal(x, y);
//             let (n_targets, n_features, rng) = (
//                 internal.n_targets,
//                 internal.n_features,
//                 internal.rng.as_mut().unwrap(),
//             );
//             let coef = rand_init(&[n_features, n_targets], rng);
//             Ok(batch_gradient_descent(
//                 x,
//                 y,
//                 coef,
//                 linear_regression_gradient,
//                 &*internal,
//             ))
//         }
//     }
// }

fn linear_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &RegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y> + Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    return square_loss_gradient(x, y, coef) * (-step_size);
}

fn ridge_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &RegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y> + Add<Y, Output = Y> + Mul<T, Output = Y>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    let l2_penalty = settings.l2_penalty.unwrap();
    return (square_loss_gradient(x, y, coef) + coef * l2_penalty) * (-step_size);
}

fn lasso_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &RegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y> + Add<Y, Output = Y> + Mul<T, Output = Y> + Algebra<Elem = T>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    let l1_penalty = settings.l1_penalty.unwrap();
    return (square_loss_gradient(x, y, coef) + coef.sign() * l1_penalty) * (-step_size);
}

fn elastic_net_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &RegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y> + Add<Y, Output = Y> + Mul<T, Output = Y> + Algebra<Elem = T>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    let (l1_penalty, l2_penalty) = (settings.l1_penalty.unwrap(), settings.l2_penalty.unwrap());
    return (square_loss_gradient(x, y, coef) + coef.sign() * l1_penalty + coef * l2_penalty)
        * (-step_size);
}
