use crate::l2_norm1;
use crate::linear_model::{randn_1d, randn_2d};
use crate::traits::Scalar;
use crate::RegressionModel;
use crate::{
    linear_model::{
        preprocess, solve_chol1, solve_chol2, solve_exact1, solve_exact2, solve_qr1, solve_qr2,
    },
    traits::Info,
};
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::uniform::SampleUniform;

#[allow(unused)]
use core::{
    marker::{Send, Sync},
    ops::{Add, Div, Mul, Sub},
};
use ndarray::{linalg::Dot, Array, Array1, Array2, Axis, Ix0, Ix1, Ix2};
use ndarray_linalg::error::LinalgError;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use num_traits::{Float, FromPrimitive};
use rand_chacha::ChaCha20Rng;

use super::{init_grad_1d, init_grad_2d};

/// Solver to use when fitting a ridge regression model (L2-penalty with
/// Ordinary Least Squares).
///
/// Here `alpha` is the magnitude of the penalty and `eye` is the identity
/// matrix.
#[derive(Debug, Default)]
pub enum RidgeRegressionSolver {
    /// Solves the problem using Stochastic Gradient Descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm
    /// may not converge.
    #[default]
    SGD,
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
#[derive(Debug)]
pub struct RidgeRegressionSettings<T> {
    pub alpha: T,
    pub fit_intercept: bool,
    pub solver: RidgeRegressionSolver,
    pub tol: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

impl<T> Default for RidgeRegressionSettings<T>
where
    T: Default + FromPrimitive,
{
    fn default() -> Self {
        Self {
            alpha: T::from_f32(1.).unwrap(),
            fit_intercept: true,
            solver: Default::default(),
            tol: Some(T::from_f32(0.0001).unwrap()),
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
            random_state: None,
            max_iter: None,
        }
    }
}

/// L2-norm penalized Ordinary Least Squares.
#[derive(Debug)]
pub struct RidgeRegression<C, I, T = f32> {
    coef: Option<C>,
    intercept: Option<I>,
    settings: RidgeRegressionSettings<T>,
}

impl<C, I, T> RidgeRegression<C, I, T> {
    /// Creates a new instance of `Self`.
    ///
    /// See also: [RidgeRegressionSettings], [RidgeRegressionSolver],
    /// [RegressionModel].
    /// ```
    /// use ndarray::{array, Array0, Array1};
    /// use njang::{RegressionModel, RidgeRegression, RidgeRegressionSettings, RidgeRegressionSolver};
    /// // Initial model
    /// let mut model =
    ///     RidgeRegression::<Array1<f32>, Array0<f32>, f32>::new(RidgeRegressionSettings {
    ///         alpha: 0.01,
    ///         tol: Some(0.0001),
    ///         solver: RidgeRegressionSolver::SGD,
    ///         fit_intercept: true,
    ///         random_state: Some(123),
    ///         max_iter: Some(1),
    ///     });
    /// // Dataset
    /// let x0 = array![[1., 2.], [-3., -4.], [0., 7.], [-2., 5.]];
    /// let y0 = array![0.5, -1., 2., 3.5];
    /// model.fit(&x0, &y0);
    /// // ... once model is fit, it can be trained again from where it stopped.
    /// let x1 = array![[0., 0.], [-1., -1.], [0.5, -5.], [-1., 3.]];
    /// let y1 = array![1.5, -1., 0., 1.];
    /// model.fit(&x1, &y1);
    /// ```
    pub fn new(settings: RidgeRegressionSettings<T>) -> Self {
        Self {
            coef: None,
            intercept: None,
            settings,
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

macro_rules! impl_ridge_reg {
    ($ix:ty, $ix_smaller:ty, $randn:ident, $norm:ident, $init_grad:ident, $grad:ident, $sgd:ident, $sag:ident, $exact_name:ident, $qr_name:ident, $chol_name:ident) => {
        impl<T> RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>, T> {
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
                let coef = $randn(n_features, y.shape(), &mut rng);
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
        impl<T> RegressionModel for RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>, T>
        where
            T: Scalar<Array2<T>> + Scalar<Array1<T>> + SampleUniform,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Option<Array<T, $ix>>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                if self.settings.fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(x, y);
                    let coef = match self.settings.solver {
                        RidgeRegressionSolver::SGD => {
                            let (alpha_norm, max_iter, tol, samples, coef, _, _, lambda) =
                                self.init_stochastic_algo(x, y);
                            $grad(
                                &x_centered,
                                &y_centered,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                ($sgd, $norm, tol, &mut None, &mut None, lambda), // use a struct ?
                            )
                        }
                        RidgeRegressionSolver::EXACT => {
                            let (xct, alpha) = (x_centered.t(), self.settings.alpha);
                            $exact_name(
                                xct.dot(&x_centered) + alpha * Array2::eye(x.ncols()),
                                xct,
                                &y_centered,
                            )?
                        }
                        RidgeRegressionSolver::QR => {
                            let (xct, alpha) = (x_centered.t(), self.settings.alpha);
                            $qr_name(
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
                            $chol_name(
                                xct.dot(&x_centered) + alpha * Array2::eye(x.ncols()),
                                xct,
                                &y_centered,
                            )?
                        }
                    };
                    self.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.coef = Some(coef);
                } else {
                    let coef = match self.settings.solver {
                        RidgeRegressionSolver::SGD => {
                            let (alpha_norm, max_iter, tol, samples, coef, _, _, lambda) =
                                self.init_stochastic_algo(x, y);
                            $grad(
                                x,
                                y,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                ($sgd, $norm, tol, &mut None, &mut None, lambda),
                            )
                        }
                        RidgeRegressionSolver::EXACT => {
                            let (xt, alpha) = (x.t(), self.settings.alpha);
                            $exact_name(xt.dot(x) + alpha * Array2::eye(x.ncols()), xt, y)?
                        }
                        RidgeRegressionSolver::QR => {
                            let (xt, alpha) = (x.t(), self.settings.alpha);
                            $qr_name(xt.dot(x) + alpha * Array2::eye(x.ncols()), xt, y)?
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
                            $chol_name(xt.dot(x) + alpha * Array2::eye(x.ncols()), xt, y)?
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
impl_ridge_reg!(
    Ix1,
    Ix0,
    randn_1d,
    l2_norm1,
    init_grad_1d,
    grad_1d,
    sgd_updator,
    sag_updator,
    solve_exact1,
    solve_qr1,
    solve_chol1
);

impl_ridge_reg!(
    Ix2,
    Ix1,
    randn_2d,
    l2_norm1,
    init_grad_2d,
    grad_2d,
    sgd_updator,
    sag_updator,
    solve_exact2,
    solve_qr2,
    solve_chol2
);

fn sgd_updator<T, X, Y>(
    x: &X,
    y: &Y,
    coef: &Y,
    i: usize,
    alpha: T,
    lambda: Option<T>,
    _gradients: (&mut Option<&mut X>, &mut Option<&mut Y>),
) -> Y
where
    Y: Info<RowOutput = T> + Dot<Y, Output = T> + Add<Y, Output = Y>,
    for<'a> &'a Y: Sub<Y, Output = Y>,
    X: Info<RowOutput = Y>,
    for<'a> T: Sub<T, Output = T> + Copy + Mul<Y, Output = Y> + Mul<&'a Y, Output = Y>,
{
    let xi = x.get_row(i);
    let yi = y.get_row(i);
    let pre_update = alpha * coef + (xi.dot(coef) - yi) * xi;
    if let Some(lamb) = lambda {
        lamb * pre_update
    } else {
        pre_update
    }
}

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
