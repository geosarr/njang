use crate::l2_norm1;
use crate::linear_model::{randn_1d, randn_2d, square_loss_gradient};
use crate::traits::{Container, Scalar};
use crate::RegressionModel;
use crate::{
    linear_model::{cholesky, exact, preprocess, qr},
    solver::{batch_gradient_descent, stochastic_gradient_descent},
    traits::Info,
};
use ndarray::ArrayView2;
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::uniform::SampleUniform;

#[allow(unused)]
use core::{
    marker::{Send, Sync},
    ops::{Add, Div, Mul, Sub},
};
use ndarray::{linalg::Dot, Array, Array1, Array2, Axis, Ix0, Ix1, Ix2};
use ndarray_linalg::{error::LinalgError, LeastSquaresSvd};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use num_traits::{Float, FromPrimitive, Zero};
use rand_chacha::ChaCha20Rng;

use super::{init_grad_1d, init_grad_2d};

/// Solver to use when fitting a ridge regression model (L2-penalty with
/// Ordinary Least Squares).
///
/// Here `alpha` is the magnitude of the penalty and `eye` is the identity
/// matrix.
#[derive(Debug, Default, Clone)]
pub enum RidgeRegressionSolver {
    /// Solves the problem using Stochastic Gradient Descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm
    /// may not converge.
    #[default]
    SGD,
    /// Solves the problem using Batch Gradient Descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm
    /// may not converge.
    BGD,
    /// Solves the problem using Singular Value Decomposition
    SVD,
    /// Computes the solution:
    /// - `(x.t().dot(x) + alpha * eye).inverse().dot(x.t().dot(y))`
    EXACT,
    /// Uses QR decomposition of the matrix `x.t().dot(x) + alpha * eye` to
    /// solve the problem:
    /// - `(x.t().dot(x) + alpha * eye) * coef = x.t().dot(y)` with respect to
    ///   `coef`
    QR,
    /// Solves the problem using Stochastic Average Gradient
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm
    /// may not converge.
    SAG,
    /// Uses Cholesky decomposition of the matrix `x.t().dot(x) + alpha * eye`
    /// to solve the problem:
    /// - `(x.t().dot(x) + alpha * eye) * coef = x.t().dot(y)` with respect to
    ///   `coef`
    CHOLESKY,
}

/// Hyperparameters used in a Ridge regression.
///
/// - **alpha**: L2-norm penalty magnitude.
/// - **fit_intercept**: `true` means fit with an intercept, `false` without an
///   intercept.
/// - **solver**: optimization method see [`RidgeRegressionSolver`].
/// - **tol**: tolerance parameter:
///     - stochastic optimization solvers (like SGD) stop when the relative
///       variation of consecutive iterates is lower than **tol**:
///         - `||coef_next - coef_curr|| <= tol * ||coef_curr||`
///     - No impact on the other algorithms.
/// - **random_state**: seed of random generators.
/// - **max_iter**: maximum number of iterations.
#[derive(Debug, Clone)]
pub struct RidgeRegressionSettings<T> {
    pub alpha: T,
    pub fit_intercept: bool,
    pub solver: RidgeRegressionSolver,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

impl<T> Default for RidgeRegressionSettings<T>
where
    T: Default + FromPrimitive,
{
    fn default() -> Self {
        Self {
            alpha: T::from_f32(0.).unwrap(),
            fit_intercept: true,
            solver: Default::default(),
            tol: Some(T::from_f32(0.0001).unwrap()),
            step_size: Some(T::from_f32(0.001).unwrap()),
            random_state: Some(0),
            max_iter: Some(1000),
        }
    }
}
impl<T> RidgeRegressionSettings<T> {
    /// Creates a new instance of EXACT solver.
    pub fn new_exact(alpha: T, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            solver: RidgeRegressionSolver::EXACT,
            tol: None,
            step_size: None,
            random_state: None,
            max_iter: None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RidgeRegressionParameter<C, I> {
    pub coef: Option<C>,
    pub intercept: Option<I>,
}

/// L2-norm penalized Ordinary Least Squares.
#[derive(Debug)]
pub struct RidgeRegression<C, I>
where
    C: Container,
{
    pub parameter: RidgeRegressionParameter<C, I>,
    pub settings: RidgeRegressionSettings<C::Elem>,
    internal: RidgeRegressionInternal<C::Elem>,
}

#[derive(Debug, Clone)]
pub(crate) struct RidgeRegressionInternal<T> {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_targets: usize,
    pub alpha: T,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub rng: Option<ChaCha20Rng>,
    pub max_iter: Option<usize>,
}

impl<T: Zero> RidgeRegressionInternal<T> {
    pub fn new() -> Self {
        Self {
            n_samples: 0,
            n_features: 0,
            n_targets: 0,
            alpha: T::zero(),
            tol: None,
            step_size: None,
            rng: None,
            max_iter: None,
        }
    }
}

impl<C: Container, I> RidgeRegression<C, I> {
    /// Creates a new instance of `Self`.
    ///
    /// See also: [RidgeRegressionSettings], [RidgeRegressionSolver],
    /// [RegressionModel].
    /// ```
    /// use ndarray::{array, Array0, Array1};
    /// use njang::{RegressionModel, RidgeRegression, RidgeRegressionSettings, RidgeRegressionSolver};
    /// // Initial model
    /// let mut model = RidgeRegression::<Array1<f32>, Array0<f32>>::new(RidgeRegressionSettings {
    ///     alpha: 0.01,
    ///     tol: Some(0.0001),
    ///     solver: RidgeRegressionSolver::SGD,
    ///     fit_intercept: true,
    ///     random_state: Some(123),
    ///     max_iter: Some(1),
    ///     step_size: None,
    /// });
    /// // Dataset
    /// let x0 = array![[1., 2.], [-3., -4.], [0., 7.], [-2., 5.]];
    /// let y0 = array![0.5, -1., 2., 3.5];
    /// model.fit(&x0, &y0);
    /// ```
    pub fn new(settings: RidgeRegressionSettings<C::Elem>) -> Self
    where
        C::Elem: Zero,
    {
        Self {
            parameter: RidgeRegressionParameter {
                coef: None,
                intercept: None,
            },
            settings,
            internal: RidgeRegressionInternal::new(),
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
    }
}

macro_rules! impl_ridge_reg {
    ($ix:ty, $ix_smaller:ty, $randn:ident, $norm:ident, $init_grad:ident, $grad:ident, $sag:ident) => {
        impl<T: Clone> RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>> {
            pub(crate) fn init_stochastic_algo(
                &self,
                x: &Array2<T>,
                y: &Array<T, $ix>,
            ) -> (
                T,
                usize,
                T,
                Array1<usize>,
                Array<T, $ix>,
                Option<Array2<T>>,
                Option<Array<T, $ix>>,
                Option<T>,
            )
            where
                T: Scalar<Array2<T>> + Scalar<Array1<T>> + SampleUniform,
            {
                let mut rng =
                    ChaCha20Rng::seed_from_u64(self.settings.random_state.unwrap_or(0).into());
                let (n_samples, n_features) = (x.nrows(), x.ncols());
                let n_targets = if y.shape().len() == 2 {
                    y.shape()[1]
                } else {
                    1
                };
                let coef = $randn(&[n_features, n_targets], &mut rng);
                let nf = T::from(n_samples).unwrap(); // critical when number of samples > int(f32::MAX) ?
                let alpha_norm = self.settings.alpha / nf;
                let (gradients, sum_gradients) = match self.settings.solver {
                    RidgeRegressionSolver::SAG => {
                        let shape = y.shape();
                        let nb_reg = if shape.len() == 1 { 1 } else { shape[1] };
                        let grad = Array2::<T>::zeros((n_samples * nb_reg, n_features));
                        let (grad, sum_grad) = $init_grad(x, y, grad, &coef, alpha_norm);
                        (Some(grad), Some(sum_grad))
                    }
                    _ => (None, None),
                };
                let lambda = match self.settings.solver {
                    RidgeRegressionSolver::SAG => {
                        let mut max_sum_squared = T::neg_infinity();
                        x.map(|xi| Float::powi(*xi, 2))
                            .axis_iter(Axis(0))
                            .map(|row| {
                                let s = row.sum();
                                if s > max_sum_squared {
                                    max_sum_squared = s
                                }
                            })
                            .for_each(drop);
                        let fit_intercept = T::from(usize::from(self.settings.fit_intercept));
                        Some(T::one() / (max_sum_squared + fit_intercept.unwrap() + alpha_norm))
                    }
                    _ => T::from(1e-3),
                };
                let max_iter = self.settings.max_iter.unwrap_or(1000);
                let samples = Array::<usize, _>::random_using(
                    max_iter,
                    Uniform::from(0..n_samples),
                    &mut rng,
                );
                let tol = self.settings.tol.unwrap_or(T::from(1e-4).unwrap());
                (
                    alpha_norm,
                    max_iter,
                    tol,
                    samples,
                    coef,
                    gradients,
                    sum_gradients,
                    lambda,
                )
            }
        }
        impl<T> RegressionModel for RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>>
        where
            T: Scalar<Array2<T>> + Scalar<Array1<T>> + SampleUniform,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Result<Array<T, $ix>, ()>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                if self.settings.fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(x, y);
                    let coef = match self.settings.solver {
                        RidgeRegressionSolver::SGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, n_samples, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.n_samples,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            let n_targets_in_t = T::from(n_targets).unwrap();
                            let n_samples_in_t = T::from(n_samples).unwrap();
                            // Rescale step_size and l2 penalty coefficient to scale gradients
                            // correctly [Specific to this algorithm]
                            self.internal
                                .step_size
                                .as_mut()
                                .map(|s| *s = *s / n_targets_in_t);
                            self.internal.alpha *= n_targets_in_t / n_samples_in_t;
                            let coef = $randn(&[n_features, n_targets], rng);
                            stochastic_gradient_descent(
                                &x_centered,
                                &y_centered,
                                coef,
                                ridge_regression_gradient,
                                &self.internal,
                            )
                        }
                        RidgeRegressionSolver::BGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, n_samples, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.n_samples,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            let coef = $randn(&[n_features, n_targets], rng);
                            batch_gradient_descent(
                                &x_centered,
                                &y_centered,
                                coef,
                                ridge_regression_gradient,
                                &self.internal,
                            )
                        }
                        RidgeRegressionSolver::SVD => {
                            let (xct, alpha) = (x_centered.t(), self.settings.alpha);
                            let xctyc = xct.dot(&y_centered);
                            (xct.dot(&x_centered) + alpha * Array2::eye(x.ncols()))
                                .least_squares(&xctyc)?
                                .solution
                        }
                        RidgeRegressionSolver::EXACT => {
                            let (xct, alpha) = (x_centered.t(), self.settings.alpha);
                            exact(
                                xct.dot(&x_centered) + alpha * Array2::eye(x.ncols()),
                                xct,
                                &y_centered,
                            )?
                        }
                        RidgeRegressionSolver::QR => {
                            let (xct, alpha) = (x_centered.t(), self.settings.alpha);
                            qr(
                                xct.dot(&x_centered) + alpha * Array2::eye(x.ncols()),
                                xct,
                                &y_centered,
                            )?
                        }
                        RidgeRegressionSolver::SAG => {
                            let (
                                alpha_norm,
                                max_iter,
                                tol,
                                samples,
                                coef,
                                mut gradients,
                                mut sum_grad,
                                lambda,
                            ) = self.init_stochastic_algo(x, y);
                            $grad(
                                &x_centered,
                                &y_centered,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                (
                                    $sag,
                                    $norm,
                                    tol,
                                    &mut gradients.as_mut(),
                                    &mut sum_grad.as_mut(),
                                    lambda,
                                ), // use a struct ?
                            )
                        }
                        RidgeRegressionSolver::CHOLESKY => {
                            let (xct, alpha) = (x_centered.t(), self.settings.alpha);
                            cholesky(
                                xct.dot(&x_centered) + alpha * Array2::eye(x.ncols()),
                                xct,
                                &y_centered,
                            )?
                        }
                    };
                    self.parameter.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.parameter.coef = Some(coef);
                } else {
                    let coef = match self.settings.solver {
                        RidgeRegressionSolver::SGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, n_samples, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.n_samples,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            let n_targets_in_t = T::from(n_targets).unwrap();
                            let n_samples_in_t = T::from(n_samples).unwrap();
                            // Rescale step_size and l2 penalty coefficient to scale gradients
                            // correctly [Specific to this algorithm]
                            self.internal
                                .step_size
                                .as_mut()
                                .map(|s| *s = *s / n_targets_in_t);
                            self.internal.alpha *= n_targets_in_t / n_samples_in_t;
                            let coef = $randn(&[n_features, n_targets], rng);
                            stochastic_gradient_descent(
                                x,
                                y,
                                coef,
                                ridge_regression_gradient,
                                &self.internal,
                            )
                        }
                        RidgeRegressionSolver::BGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, n_samples, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.n_samples,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            let coef = $randn(&[n_features, n_targets], rng);
                            batch_gradient_descent(
                                x,
                                y,
                                coef,
                                ridge_regression_gradient,
                                &self.internal,
                            )
                        }
                        RidgeRegressionSolver::SVD => {
                            let (xt, alpha) = (x.t(), self.settings.alpha);
                            let xty = xt.dot(y);
                            (xt.dot(x) + alpha * Array2::eye(x.ncols()))
                                .least_squares(&xty)?
                                .solution
                        }
                        RidgeRegressionSolver::EXACT => {
                            let (xt, alpha) = (x.t(), self.settings.alpha);
                            exact(xt.dot(x) + alpha * Array2::eye(x.ncols()), xt, y)?
                        }
                        RidgeRegressionSolver::QR => {
                            let (xt, alpha) = (x.t(), self.settings.alpha);
                            qr(xt.dot(x) + alpha * Array2::eye(x.ncols()), xt, y)?
                        }
                        RidgeRegressionSolver::SAG => {
                            let (
                                alpha_norm,
                                max_iter,
                                tol,
                                samples,
                                coef,
                                mut gradients,
                                mut sum_grad,
                                lambda,
                            ) = self.init_stochastic_algo(x, y);
                            $grad(
                                x,
                                y,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                (
                                    $sag,
                                    $norm,
                                    tol,
                                    &mut gradients.as_mut(),
                                    &mut sum_grad.as_mut(),
                                    lambda,
                                ), // use a struct ?
                            )
                        }
                        RidgeRegressionSolver::CHOLESKY => {
                            let (xt, alpha) = (x.t(), self.settings.alpha);
                            cholesky(xt.dot(x) + alpha * Array2::eye(x.ncols()), xt, y)?
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
impl_ridge_reg!(
    Ix1,
    Ix0,
    randn_1d,
    l2_norm1,
    init_grad_1d,
    grad_1d,
    sag_updator
);

impl_ridge_reg!(
    Ix2,
    Ix1,
    randn_2d,
    l2_norm1,
    init_grad_2d,
    grad_2d,
    sag_updator
);

// fn sgd_updator<T, X, Y>(
//     x: &X,
//     y: &Y,
//     coef: &Y,
//     i: usize,
//     alpha: T,
//     lambda: Option<T>,
//     _gradients: (&mut Option<&mut X>, &mut Option<&mut Y>),
// ) -> Y
// where
//     Y: Info<RowOutput = T> + Dot<Y, Output = T> + Add<Y, Output = Y>,
//     for<'a> &'a Y: Sub<Y, Output = Y>,
//     X: Info<RowOutput = Y>,
//     for<'a> T: Sub<T, Output = T> + Copy + Mul<Y, Output = Y> + Mul<&'a Y,
// Output = Y>, {
//     let xi = x.get_row(i);
//     let yi = y.get_row(i);
//     let pre_update = alpha * coef + (xi.dot(coef) - yi) * xi;
//     if let Some(lamb) = lambda {
//         lamb * pre_update
//     } else {
//         pre_update
//     }
// }

fn sag_updator<T, X, Y>(
    x: &X,
    y: &Y,
    coef: &Y,
    i: usize,
    alpha: T,
    lambda: Option<T>,
    gradients: (&mut Option<&mut X>, &mut Option<&mut Y>),
) -> Y
where
    Y: Info<RowOutput = T> + Dot<Y, Output = T> + Add<Y, Output = Y> + Sub<Y, Output = Y>,
    for<'a> &'a Y: Sub<Y, Output = Y> + Add<Y, Output = Y>,
    X: Info<RowOutput = Y, MeanOutput = Y, ColMut = Y, RowMut = Y, NrowsOutput = usize>,
    for<'a> T: Sub<T, Output = T>
        + Div<T, Output = T>
        + Copy
        + Mul<T, Output = T>
        + Mul<Y, Output = Y>
        + Mul<&'a Y, Output = Y>
        + FromPrimitive,
{
    let (gradients, sum_gradients) = gradients;
    let xi = x.get_row(i);
    let yi = y.get_row(i);
    let gradi = alpha * coef + (xi.dot(coef) - yi) * xi;
    let scale = T::from_f32(1.).unwrap() / T::from_usize(x.get_nrows()).unwrap();
    if let Some(grad) = gradients {
        if let Some(sgrad) = sum_gradients {
            let sum = &**sgrad + (&gradi - grad.get_row(i));
            let update = if let Some(lamb) = lambda {
                lamb * scale * &sum
            } else {
                scale * &sum // To improve by adaptive step_size ?
            };
            grad.row_mut(i, gradi);
            **sgrad = sum;
            update
        } else {
            panic!("")
        }
    } else {
        panic!("No gradients provided");
    }
}

fn grad_1d<T, X, Y, U, N>(
    x: &X,
    y: &Y,
    mut coef: Y,
    max_iter: usize,
    samples: &Array1<usize>,
    alpha_norm: T,
    updators: (U, N, T, &mut Option<&mut X>, &mut Option<&mut Y>, Option<T>),
) -> Y
where
    Y: Info<RowOutput = T> + Dot<Y, Output = T> + Add<Y, Output = Y>,
    for<'a> &'a Y: Sub<Y, Output = Y>,
    X: Info<RowOutput = Y>,
    for<'a> T: Sub<T, Output = T>
        + Copy
        + Mul<Y, Output = Y>
        + Mul<&'a Y, Output = Y>
        + FromPrimitive
        + Float
        + core::fmt::Debug,
    U: Fn(&X, &Y, &Y, usize, T, Option<T>, (&mut Option<&mut X>, &mut Option<&mut Y>)) -> Y,
    N: Fn(&Y) -> T,
{
    let (updator, norm_func, tol, gradients, sum_grad, lambda) = updators;
    for k in 0..max_iter {
        let i = samples[k];
        let update = updator(x, y, &coef, i, alpha_norm, lambda, (gradients, sum_grad));
        if norm_func(&update).abs() <= tol * norm_func(&coef) {
            break;
        }
        coef = &coef - update;
    }
    coef
}

fn grad_2d<T, X, Xs, U, N>(
    x: &X,
    y: &X,
    mut coef: X,
    max_iter: usize,
    samples: &Array1<usize>,
    alpha_norm: T,
    updators: (U, N, T, &mut Option<&mut X>, &mut Option<&mut X>, Option<T>),
) -> X
where
    Xs: Info<RowOutput = T> + Dot<Xs, Output = T> + Add<Xs, Output = Xs>,
    for<'a> &'a Xs: Sub<Xs, Output = Xs>,
    X: Info<
        RowOutput = Xs,
        ColOutput = Xs,
        NcolsOutput = usize,
        ColMut = Xs,
        NrowsOutput = usize,
        SliceRowOutput = X,
    >,
    for<'a> T: Sub<T, Output = T>
        + Copy
        + Mul<Xs, Output = Xs>
        + Mul<&'a Xs, Output = Xs>
        + FromPrimitive
        + Float
        + core::fmt::Debug,
    U: Fn(&X, &Xs, &Xs, usize, T, Option<T>, (&mut Option<&mut X>, &mut Option<&mut Xs>)) -> Xs,
    N: Fn(&Xs) -> T,
{
    let nb_reg = coef.get_ncols();
    let n_samples = x.get_nrows();
    let (updator, norm_func, tol, gradients, sum_grad, lambda) = updators;
    (0..nb_reg)
        .map(|r| {
            let coefr = coef.get_col(r);
            let mut gradr = if let Some(grad) = gradients {
                let start = r * n_samples;
                let end = start + n_samples;
                Some(grad.slice_row(start, end))
            } else {
                None
            };
            let yr = y.get_col(r);
            let mut sum_gradr = if let Some(sgrad) = sum_grad {
                Some(sgrad.get_row(r))
            } else {
                None
            };
            coef.col_mut(
                r,
                grad_1d(
                    x,
                    &yr,
                    coefr,
                    max_iter,
                    samples,
                    alpha_norm,
                    (
                        &updator,
                        &norm_func,
                        tol,
                        &mut gradr.as_mut(),
                        &mut sum_gradr.as_mut(),
                        lambda,
                    ),
                ),
            );
        })
        .for_each(drop);
    coef
}

#[cfg(feature = "std")]
#[cfg(feature = "rayon")]
fn par_grad_2d<T>(
    x: &Array2<T>,
    y: &Array2<T>,
    mut coef: Array2<T>,
    max_iter: usize,
    samples: &Array1<usize>,
    alpha_norm: T,
    lambda: T,
) -> Array2<T>
where
    Array1<T>:
        Info<RowOutput = T> + Dot<Array1<T>, Output = T> + Add<Array1<T>, Output = Array1<T>>,
    for<'a> &'a Array1<T>: Sub<Array1<T>, Output = Array1<T>>,
    Array2<T>: Info<RowOutput = Array1<T>>,
    T: Sub<T, Output = T> + Copy + Send + Sync,
    for<'a> T: Mul<Array1<T>, Output = Array1<T>> + Mul<&'a Array1<T>, Output = Array1<T>>,
{
    use rayon::prelude::*;
    use std::sync::Mutex;
    let nb_reg = coef.ncols();
    let coef = Mutex::new(coef);
    let y = Mutex::new(y);
    (0..nb_reg)
        .into_par_iter()
        .map(|r| {
            let coefr = coef.lock().unwrap().column(r).to_owned();
            let yr = y.lock().unwrap().column(r).to_owned();
            coef.lock().unwrap().column_mut(r).assign(&grad_1d(
                x, &yr, coefr, max_iter, &samples, alpha_norm, lambda,
            ));
        })
        .for_each(drop);
    let coef = coef.lock().unwrap();
    coef.clone()
}

fn ridge_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &RidgeRegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y> + Add<Y, Output = Y> + Mul<T, Output = Y>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    let l2_penalty = settings.alpha;
    return (square_loss_gradient(x, y, coef) + coef * l2_penalty) * (-step_size);
}

#[test]
fn code() {
    use ndarray::*;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use rand_chacha::ChaCha20Rng;
    let settings = RidgeRegressionSettings {
        alpha: 0.1,
        fit_intercept: true,
        max_iter: Some(10000),
        solver: RidgeRegressionSolver::SGD,
        tol: Some(1e-20),
        random_state: Some(0),
        step_size: Some(1e-3),
    };
    let mut model: RidgeRegression<Array1<f32>, Array0<f32>> = RidgeRegression {
        parameter: RidgeRegressionParameter {
            coef: None,
            intercept: None,
        },
        settings: settings.clone(),
        internal: RidgeRegressionInternal::new(),
    };
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let p = 100;
    let x = Array::<f32, Ix2>::random_using((10, p), StandardNormal, &mut rng);
    let coef = Array1::from((1..p + 1).map(|val| val as f32).collect::<Vec<_>>());
    let y = x.dot(&coef);
    let _ = model.fit(&x, &y);
    println!("coef:\n{:?}", model.coef());
    println!("intercept:\n{:?}\n\n", model.intercept());

    // let mut model: RidgeRegression<Array2<f32>, Array1<f32>> =
    // RidgeRegression {     parameter: RidgeRegressionParameter {
    //         coef: None,
    //         intercept: None,
    //     },
    //     internal: RidgeRegressionInternal::new(),
    //     settings: settings,
    // };

    // let mut rng = ChaCha20Rng::seed_from_u64(0);
    // let p = 10;
    // let x = Array::<f32, Ix2>::random_using((100000, p), StandardNormal, &mut
    // rng); let r = 10;
    // // let x = (&x - x.mean_axis(Axis(0)).unwrap()) / x.std_axis(Axis(0),
    // 0.); let coef = Array2::from_shape_vec(
    //     (p, r),
    //     (1..p * r + 1).map(|val| val as f32).collect::<Vec<_>>(),
    // )
    // .unwrap();
    // let intercept = Array1::from_iter((1..r + 1).map(|val| val as f32));
    // let y = x.dot(&coef) + intercept;
    // let _ = model.fit(&x, &y);
    // println!("coef:\n{:?}", model.coef());
    // println!("intercept:\n{:?}\n\n", model.intercept());
}
