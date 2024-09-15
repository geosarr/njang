use crate::{
    linear_model::{
        cholesky, exact, init_grad_2d, preprocess, qr, randn_1d, randn_2d, square_loss_gradient,
    },
    solver::{batch_gradient_descent, stochastic_average_gradient, stochastic_gradient_descent},
    traits::{Algebra, Container, RegressionModel},
};
use core::convert::identity;
use core::ops::{Add, Mul, Sub};
use ndarray::{
    linalg::Dot, s, Array, Array1, Array2, ArrayView2, Axis, Ix0, Ix1, Ix2, OwnedRepr,
    ScalarOperand,
};
use ndarray_linalg::{error::LinalgError, Lapack, LeastSquaresSvd};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive};
use rand_chacha::ChaCha20Rng;

const DEFAULT_L1: f32 = 1.;
const DEFAULT_L2: f32 = 1.;
const DEFAULT_TOL: f32 = 1e-3;
const DEFAULT_STEP_SIZE: f32 = 1e-3;
const DEFAULT_STATE: u32 = 0;
const DEFAULT_MAX_ITER: usize = 1000;

#[derive(Debug, Default, Clone, Copy)]
pub enum RegressionSolver {
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

/// This is responsible for processing settings, setting default values
#[derive(Debug, Clone)]
pub(crate) struct RegressionInternal<T> {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_targets: usize,
    pub l1_penalty: Option<T>,
    pub l2_penalty: Option<T>,
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

/// Hyperparameters used in a regression model.
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
pub struct RegressionSettings<T> {
    /// If it is `true` then the model fits with an intercept, `false` without
    /// an intercept. The matrix [coef][`Regression`] of non-intercept weights
    /// satisfies:
    /// - if `fit_intercept = false`, then the prediction of a sample `x` is
    ///   given by `x.dot(coef)`
    /// - if `fit_intercept = true`, then the prediction of a sample `x` is
    ///   given by `x.dot(coef) + intercept`
    /// where `intercept = Y_mean - X_mean.dot(coef)`, with `X_mean`
    /// designating the average of the training predictors `X` (i.e features
    /// mean) and `Y_mean` designating the average(s) of target(s) `Y`, `dot`
    /// is the matrix multiplication.
    pub fit_intercept: bool,
    /// Optimization method, see [`RegressionSolver`].
    pub solver: RegressionSolver,
    /// If it is `None`, then no L1-penalty is added to the loss objective
    /// function. Otherwise, if it is equal to `Some(value)`, then `value *
    /// ||coef||`<sub>1</sub> is added to the loss objective function. Instead
    /// of setting `l1_penalty = Some(0.)`, it may be preferable to set
    /// `l1_penalty = None` to avoid useless computations and numerical issues.
    pub l1_penalty: Option<T>, // for Lasso Regression
    /// If it is `None`, then no L2-penalty is added to the loss objective
    /// function. Otherwise, if it is equal to `Some(value)`, then `0.5 *
    /// value * ||coef||`<sub>2</sub><sup>2</sup> is added to the loss objective
    /// function. Instead of setting `l2_penalty = Some(0.)`, it may be
    /// preferable to set `l2_penalty = None` to avoid useless computations
    /// and numerical issues.
    pub l2_penalty: Option<T>, // for Ridge Regression
    /// Tolerance parameter.
    /// - Gradient descent solvers (like [Sgd][`RegressionSolver::Sgd`],
    ///   [Bgd][`RegressionSolver::Bgd`], etc) stop when the relative variation
    ///   of consecutive iterates is lower than **tol**, that is:
    ///     - `||coef_next - coef_curr||`<sub>2</sub> `<= tol *
    ///       ||coef_curr||`<sub>2</sub>
    /// - No impact on the other algorithms:
    ///     - [Exact][`RegressionSolver::Exact`]
    ///     - [Svd][`RegressionSolver::Svd`]
    ///     - [Qr][`RegressionSolver::Qr`]
    ///     - [Cholesky][`RegressionSolver::Cholesky`]
    pub tol: Option<T>,
    /// Step size used in gradient descent algorithms.
    pub step_size: Option<T>,
    /// Seed of random generators used in gradient descent algorithms.
    pub random_state: Option<u32>,
    /// Maximum number of iterations used in gradient descent algorithms.
    pub max_iter: Option<usize>,
}

impl<T> Default for RegressionSettings<T> {
    /// Defaults to linear regression without penalty
    /// ```
    /// use ndarray::{Array0, Array1, Array2};
    /// use njang::{Regression, RegressionSettings};
    ///
    /// let settings = RegressionSettings::default();
    /// let model = Regression::<Array1<f32>, Array0<f32>>::new(settings);
    /// assert!(model.is_linear());
    /// ```
    fn default() -> Self {
        Self {
            fit_intercept: true,
            solver: RegressionSolver::default(),
            l1_penalty: None,
            l2_penalty: None,
            tol: None,
            step_size: None,
            random_state: None,
            max_iter: None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RegressionParameter<C, I> {
    /// Non-intercept weight(s).
    pub coef: Option<C>,
    /// Intercept weight(s) of the model.
    pub intercept: Option<I>,
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
/// sets the fields `*_penalty` in [RegressionSettings]. See
/// [RegressionSettings] for more details.
///
/// It is able to fit at once many regressions with the same input predictors
/// `X`, when `X` and `Y` are of type `Array2<T>` from ndarray crate. **In this
/// case, the same settings apply to all regressions involved**.
/// ```
/// use ndarray::{array, Array1, Array2};
/// use njang::{Regression, RegressionModel, RegressionSettings, RegressionSolver};
/// let x = array![[0., 1.], [1., -1.], [-2., 3.]];
/// let coef = array![[10., 30.], [20., 40.]];
/// let intercept = 1.;
/// let y = x.dot(&coef) + intercept;
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
/// assert!(
///     (model.intercept().unwrap() - intercept)
///         .map(|error| error.abs())
///         .sum()
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
    /// Whether or not the model is an Elastic Net regression.
    pub fn is_elastic_net(&self) -> bool {
        self.settings.l1_penalty.is_some() && self.settings.l2_penalty.is_some()
    }
    /// Whether or not the model is a Ridge regression.
    pub fn is_ridge(&self) -> bool {
        self.settings.l1_penalty.is_none() && self.settings.l2_penalty.is_some()
    }
    /// Whether or not the model is a Lasso regression.
    pub fn is_lasso(&self) -> bool {
        self.settings.l1_penalty.is_some() && self.settings.l2_penalty.is_none()
    }
    /// Whether or not the model is a Linear regression (without any penalty).
    pub fn is_linear(&self) -> bool {
        self.settings.l1_penalty.is_none() && self.settings.l2_penalty.is_none()
    }
    /// Sets information on the input samples to the internal settings. Argument
    /// `x` should be a 2 dimensional container, and `<X as
    /// Container>::dimension(x)` should return something like [nrows, ncols] of
    /// x.
    fn set_sample_to_internal<X: Container>(&mut self, x: &X, y: &C) {
        let dim = x.dimension();
        let (n_samples, n_features) = (dim[0], dim[1]);
        self.internal.n_features = n_features;
        self.internal.n_samples = n_samples;
        let shape = y.dimension();
        let n_targets = if shape.len() == 2 { shape[1] } else { 1 };
        self.internal.n_targets = n_targets;
    }
    fn set_rng_to_internal(&mut self) {
        let random_state = self.settings.random_state.unwrap_or(DEFAULT_STATE);
        self.internal.rng = Some(ChaCha20Rng::seed_from_u64(random_state as u64));
    }
    fn set_max_iter_to_internal(&mut self) {
        self.internal.max_iter = Some(self.settings.max_iter.unwrap_or(DEFAULT_MAX_ITER));
    }
    fn set_penalty_to_internal(&mut self)
    where
        C::Elem: Copy + FromPrimitive + core::fmt::Debug,
    {
        // Use here a match pattern with enum instead of if else's ?
        if self.is_lasso() {
            self.set_l1_penalty_to_internal();
        } else if self.is_ridge() {
            self.set_l2_penalty_to_internal();
        } else if self.is_elastic_net() {
            self.set_l1_penalty_to_internal();
            self.set_l2_penalty_to_internal();
        }
    }
    fn set_internal<X: Container>(&mut self, x: &X, y: &C)
    where
        C::Elem: Float + FromPrimitive + core::fmt::Debug,
    {
        self.set_sample_to_internal(x, y);
        self.set_rng_to_internal();
        self.set_max_iter_to_internal();
        self.set_tol_to_internal();
        self.set_step_size_to_internal();
        self.set_penalty_to_internal();
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
    fn scale_penalty(&mut self)
    where
        C::Elem: Float + FromPrimitive,
    {
        if self.is_lasso() {
            // L2-penalty
            self.scale_l1_penalty();
        } else if self.is_ridge() {
            // L1-penalty
            self.scale_l2_penalty();
        } else if self.is_elastic_net() {
            // L1 and L2 penalties
            self.scale_l1_penalty();
            self.scale_l2_penalty();
        }
    }
    fn gradient_function<T, Y>(&self) -> impl Fn(&Array2<T>, &Y, &Y, &RegressionInternal<T>) -> Y
    where
        T: Lapack,
        for<'a> Y:
            Sub<&'a Y, Output = Y> + Add<Y, Output = Y> + Mul<T, Output = Y> + Algebra<Elem = T>,
        for<'a> &'a Y: Mul<T, Output = Y>,
        Array2<T>: Dot<Y, Output = Y>,
        for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
    {
        // Use here a match pattern with enum instead of if else's ?
        if self.is_linear() {
            linear_regression_gradient
        } else if self.is_ridge() {
            ridge_regression_gradient
        } else if self.is_lasso() {
            lasso_regression_gradient
        } else {
            elastic_net_regression_gradient
        }
    }
}

macro_rules! impl_settings_to_internal {
    ($setter_name:ident, $field_name:ident, $default:ident) => {
        impl<C: Container, I> Regression<C, I>
        where
            C::Elem: Copy + FromPrimitive + core::fmt::Debug,
        {
            fn $setter_name(&mut self) {
                self.internal.$field_name = Some(
                    self.settings
                        .$field_name
                        .unwrap_or(C::Elem::from_f32($default).unwrap()),
                );
                // println!("\nField: {:?}\n", self.internal.$field_name);
            }
        }
    };
}
impl_settings_to_internal!(set_l1_penalty_to_internal, l1_penalty, DEFAULT_L1);
impl_settings_to_internal!(set_l2_penalty_to_internal, l2_penalty, DEFAULT_L2);
impl_settings_to_internal!(set_tol_to_internal, tol, DEFAULT_TOL);
impl_settings_to_internal!(set_step_size_to_internal, step_size, DEFAULT_STEP_SIZE);

macro_rules! impl_scale_penalty {
    ($scaler_name:ident, $field:ident) => {
        impl<C: Container, I> Regression<C, I>
        where
            C::Elem: Float + FromPrimitive,
        {
            fn $scaler_name(&mut self)
            where
                C::Elem: Float + FromPrimitive,
            {
                let n_targets = C::Elem::from_usize(self.internal.n_targets).unwrap();
                let n_samples = C::Elem::from_usize(self.internal.n_samples).unwrap();
                self.internal
                    .$field
                    .as_mut()
                    .map(|p| *p = *p * n_targets / n_samples);
            }
        }
    };
}
impl_scale_penalty!(scale_l1_penalty, l1_penalty);
impl_scale_penalty!(scale_l2_penalty, l2_penalty);

macro_rules! impl_regression {
    ($ix:ty, $ix_smaller:ty, $randn:ident, $reshape_to_normal:ident, $reshape_to_2d:ident) => {
        impl<T: Clone> Regression<Array<T, $ix>, Array<T, $ix_smaller>> {
            fn solve(
                &mut self,
                x: &Array2<T>,
                y: &Array<T, $ix>,
            ) -> Result<Array<T, $ix>, LinalgError>
            where
                T: Lapack + PartialOrd + Float + ScalarOperand + SampleUniform + core::fmt::Debug,
            {
                match self.settings.solver {
                    RegressionSolver::Svd => {
                        if self.is_ridge() | self.is_linear() {
                            Ok(x.least_squares(y)?.solution)
                        } else {
                            // TODO: use an error wrapper instead.
                            panic!("Not supported.");
                        }
                    }
                    RegressionSolver::Exact => {
                        if self.is_ridge() | self.is_linear() {
                            let xct = x.t();
                            exact(xct.dot(x), xct, y)
                        } else {
                            // TODO: use an error wrapper instead.
                            panic!("Not supported.");
                        }
                    }
                    RegressionSolver::Qr => {
                        if self.is_ridge() | self.is_linear() {
                            let xct = x.t();
                            qr(xct.dot(x), xct, y)
                        } else {
                            // TODO: use an error wrapper instead.
                            panic!("Not supported.");
                        }
                    }
                    RegressionSolver::Cholesky => {
                        if self.is_ridge() | self.is_linear() {
                            let xct = x.t();
                            cholesky(xct.dot(x), xct, y)
                        } else {
                            // TODO: use an error wrapper instead.
                            panic!("Not supported.");
                        }
                    }
                    RegressionSolver::Sgd => {
                        self.set_internal(x, y);
                        // Rescale step_size and penalty(ies) to scale gradients correctly
                        self.scale_step_size();
                        self.scale_penalty();
                        // println!("Internal after setting:\n{:?}", self.internal.clone());
                        let (n_targets, n_features, rng) = (
                            self.internal.n_targets,
                            self.internal.n_features,
                            self.internal.rng.as_mut().unwrap(),
                        );
                        let coef = $randn(&[n_features, n_targets], rng);
                        Ok(stochastic_gradient_descent(
                            x,
                            y,
                            coef,
                            self.gradient_function(),
                            &self.internal,
                        ))
                    }
                    RegressionSolver::Bgd => {
                        self.set_internal(x, y);
                        // println!("Internal after setting:\n{:?}", self.internal.clone());
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
                            self.gradient_function(),
                            &self.internal,
                        ))
                    }
                    RegressionSolver::Sag => {
                        self.set_internal(x, y);
                        // Rescale step_size and penalty(ies) to scale gradients correctly
                        self.scale_step_size();
                        self.scale_penalty();
                        println!("Internal after setting:\n{:?}", self.internal.clone());
                        let (n_targets, n_features, n_samples, rng) = (
                            self.internal.n_targets,
                            self.internal.n_features,
                            self.internal.n_samples,
                            self.internal.rng.as_mut().unwrap(),
                        );
                        let _y = $reshape_to_2d(y); //
                        let coef = randn_2d(&[n_features, n_targets], rng);
                        let grad = Array2::<_>::zeros((n_samples * n_targets, n_features));
                        let (gradients, sum_gradients) = init_grad(x, &_y, grad, &coef, T::zero());
                        let coef = stochastic_average_gradient(
                            x,
                            &_y,
                            coef,
                            self.gradient_function(),
                            &self.internal,
                            gradients,
                            sum_gradients,
                        );
                        Ok($reshape_to_normal(coef))
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
pub(crate) fn init_grad<T>(
    x: &Array2<T>,
    y: &Array2<T>,
    mut grad: Array2<T>,
    coef: &Array2<T>,
    alpha: T,
) -> (Array2<T>, Array2<T>)
where
    for<'a> T: Lapack + ScalarOperand, /*
                                        * + Mul<Array1<T>, Output = Array1<T>>
                                        * + Mul<&'a Array2<T>, Output = Array2<T>>
                                        * + Mul<&'a Array1<T>, Output = Array1<T>>, */
{
    let (n_samples, n_regressions) = (x.nrows(), y.ncols());
    for k in 0..n_samples {
        let xi = x.row(k).to_owned();
        let yi = y.row(k);
        let error = xi.dot(coef) - yi;
        let grad_norm = coef * alpha;
        for r in 0..n_regressions {
            let start = r * n_samples;
            (grad_norm.column(r).to_owned() + &xi * error[r])
                .assign_to(grad.slice_mut(s!(start + k, ..)));
        }
    }
    let mut sum_grad = Array2::<T>::zeros((n_regressions, x.ncols()));
    for r in 0..n_regressions {
        grad.slice(s!(r * n_samples..(r + 1) * n_samples, ..))
            .sum_axis(Axis(0))
            .assign_to(sum_grad.slice_mut(s!(r, ..)));
    }
    return (grad, sum_grad);
}
impl_regression!(Ix1, Ix0, randn_1d, reshape_to_1d, reshape_to_2d);
impl_regression!(Ix2, Ix1, randn_2d, identity, reshape_to_2d);

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

#[test]
fn code() {
    use crate::linear_model::RegressionInternal;
    use ndarray::{array, Array1};

    let x = array![[0., 1.], [1., -1.], [-2., 3.]];
    println!("x:\n{:?}\n", x);
    let coef = array![[1.], [2.]];
    println!("coef:\n{:?}\n", coef);
    let y = x.dot(&coef);
    println!("y:\n{:?}\n", y);
    let n_targets = if y.shape().len() == 1 {
        1
    } else {
        y.shape()[1]
    };

    let grad = Array2::<_>::zeros((x.nrows() * n_targets, x.ncols()));
    let (gradients, sum_gradients) = init_grad_2d(&x, &y, grad, &coef, 0.);
    let result = stochastic_average_gradient(
        &x,
        &y,
        Array2::zeros((coef.nrows(), n_targets)),
        linear_regression_gradient,
        &RegressionInternal {
            n_samples: x.nrows(),
            n_features: x.ncols(),
            n_targets: n_targets,
            l1_penalty: None,
            l2_penalty: None,
            tol: Some(1e-6),
            step_size: Some(1e-3),
            rng: Some(ChaCha20Rng::seed_from_u64(0)),
            max_iter: Some(100000),
        },
        gradients,
        sum_gradients,
    );
    // println!("{:?}\n", core::convert::identity(x));
    println!("{:?}\n", result);
    // println!("{:?}", reshape_to_1d(result, 1, 0));
}
