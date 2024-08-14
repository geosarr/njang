use crate::linear_model::{randn_1d, randn_2d};
use crate::RegressionModel;
use crate::{l2_norm1, l2_norm2};
use crate::{linear_model::preprocess, traits::Info};
#[allow(unused)]
use core::{
    marker::{Send, Sync},
    ops::{Add, Mul, Sub},
};
use ndarray::{linalg::Dot, s, Array, Array1, Array2, Ix0, Ix1, Ix2, ScalarOperand};
use ndarray_linalg::{error::LinalgError, Inverse, Lapack, QR};
use ndarray_rand::{
    rand::{distributions::Distribution, SeedableRng},
    rand_distr::{StandardNormal, Uniform},
    RandomExt,
};
use num_traits::{Float, FromPrimitive};
use rand_chacha::ChaCha20Rng;

/// Solver to use when fitting a ridge regression model (L2-penalty with Ordinary Least Squares).
#[derive(Debug, Default)]
pub enum RidgeRegressionSolver {
    /// Solves the problem using Stochastic Gradient Descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm may not converge.
    #[default]
    SGD,
    /// Computes the solution: (x.t().dot(x) + alpha * eye).inverse().dot(x.t().dot(y))
    EXACT,
    /// Uses QR decomposition of the matrix x.t().dot(x) + alpha * eye to solve the problem
    /// (x.t().dot(x) + alpha * eye) * coef = x.t().dot(y) with respect to coef
    QR,
    /// Solves the problemen using Stochastic Average Gradient
    SAG,
}

/// Hyperparameters used in a Ridge regression.
///
/// - **alpha**: L2-norm penalty magnitude.
/// - **fit_intercept**: `true` means fit with an intercept, `false` without an intercept.
/// - **solver**: optimization method see [`RidgeRegressionSolver`].
/// - **tol**: tolerance parameter:
///     - stochastic algorithms (like SGD) stop when the relative variation of consecutive
///       iterates L2-norms is lower than **tol**.
///     - No impact on the other algorithms.
/// - **random_state**: seed of random generators.
/// - **max_iter**: maximum number of iterations.
/// - **warm_start**: whether or not warm starting is allowed.
#[derive(Debug)]
pub struct RidgeRegressionHyperParameter<T> {
    pub alpha: T,
    pub fit_intercept: bool,
    pub solver: RidgeRegressionSolver,
    pub tol: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
    pub warm_start: bool,
}

impl<T> Default for RidgeRegressionHyperParameter<T>
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
            warm_start: true,
        }
    }
}
impl<T> RidgeRegressionHyperParameter<T> {
    pub fn new_exact(alpha: T, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            solver: RidgeRegressionSolver::EXACT,
            tol: None,
            random_state: None,
            max_iter: None,
            warm_start: false,
        }
    }
}

#[derive(Debug)]
pub struct RidgeRegression<C, I, T = f32> {
    coef: Option<C>,
    intercept: Option<I>,
    settings: RidgeRegressionHyperParameter<T>,
}

impl<C, I, T> RidgeRegression<C, I, T> {
    /// Creates a new instance of `Self`.
    ///
    /// See also: [RidgeRegressionHyperParameter], [RidgeRegressionSolver], [RegressionModel].
    /// ```
    /// use njang::{RidgeRegression, RidgeRegressionHyperParameter, RidgeRegressionSolver, RegressionModel};
    /// use ndarray::{Array1, Array0, array};
    /// // Initial model
    /// let mut model = RidgeRegression::<Array1<f32>, Array0<f32>, f32>::new(
    ///     RidgeRegressionHyperParameter{
    ///         alpha: 0.01,
    ///         tol: Some(0.0001),
    ///         solver: RidgeRegressionSolver::SGD,
    ///         fit_intercept: true,
    ///         random_state: Some(123),
    ///         max_iter: Some(1),
    ///         warm_start: true,
    /// });
    /// // Dataset
    /// let x0 = array![[1., 2.], [-3., -4.], [0., 7.], [-2., 5.]];
    /// let y0 = array![0.5, -1., 2., 3.5];
    /// model.fit(&x0, &y0);
    /// // ... once model is fit, it can be trained again from where it stopped.
    /// let x1 = array![[0., 0.], [-1., -1.], [0.5, -5.], [-1., 3.]];
    /// let y1 = array![1.5, -1., 0., 1.];
    /// model.fit(&x1, &y1);
    /// ```
    pub fn new(settings: RidgeRegressionHyperParameter<T>) -> Self {
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
    ($ix:ty, $ix_smaller:ty,  $randn:ident, $norm:ident, $grad:ident, $sgd:ident, $sag:ident) => {
        impl<T> RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>, T> {
            fn init_stochastic_algo(
                &self,
                x: &Array2<T>,
                y: &Array<T, $ix>,
            ) -> (T, usize, T, Array1<usize>, Array<T, $ix>, Option<Array2<T>>)
            where
                for<'a> T: Lapack
                    + Clone
                    + Mul<&'a Array<T, Ix2>, Output = Array<T, Ix2>>
                    + Mul<&'a Array<T, Ix1>, Output = Array<T, Ix1>>,
                StandardNormal: Distribution<T>,
            {
                let mut rng =
                    ChaCha20Rng::seed_from_u64(self.settings.random_state.unwrap_or(0).into());
                let (n_samples, n_features) = (x.nrows(), x.ncols());
                let coef = if let Some(coef) = &self.coef {
                    // warm start is activated
                    if self.settings.warm_start {
                        coef.clone()
                    } else {
                        $randn(n_features, y.shape(), &mut rng)
                    }
                } else {
                    $randn(n_features, y.shape(), &mut rng)
                };
                let nf = T::from_f32(n_samples as f32).unwrap(); // critical when number of samples > int(f3::MAX) ?
                let alpha_norm = self.settings.alpha / nf;
                let gradients: Option<Array2<T>> = match self.settings.solver {
                    RidgeRegressionSolver::SAG => {
                        let shape = y.shape();
                        let nb_reg = if shape.len() == 1 { 1 } else { shape[1] };
                        let mut grad = Array2::<T>::zeros((n_samples * nb_reg, n_features));
                        // for r in 0..nb_reg {
                        //     let (start, end) = (r * n_samples, (r + 1) * n_samples);
                        //     // for k in 0..n_samples {
                        //     alpha_norm * &coef + (x.dot(&coef) - y) * x;
                        //     Array2::<T>::zeros((n_samples, n_features))
                        //         .assign_to(grad.slice_mut(s!(start..end, ..)));
                        //     // }
                        // }
                        // for r in 0..n_features{
                        //     grad.row_mut(r, );
                        // }
                        Some(grad)
                    }
                    _ => None,
                };
                let max_iter = self.settings.max_iter.unwrap_or(1000);
                let samples = Array::<usize, _>::random_using(
                    max_iter,
                    Uniform::from(0..n_samples),
                    &mut rng,
                );
                let tol = self.settings.tol.unwrap_or(T::from_f32(1e-4).unwrap());
                (alpha_norm, max_iter, tol, samples, coef, gradients)
            }
        }
        impl<T> RegressionModel for RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>, T>
        where
            for<'a> T: Lapack
                + Float
                + ScalarOperand
                + Mul<Array1<T>, Output = Array1<T>>
                + Mul<Array2<T>, Output = Array2<T>>
                + Mul<&'a Array<T, Ix2>, Output = Array<T, Ix2>>
                + Mul<&'a Array<T, Ix1>, Output = Array<T, Ix1>>
                + core::fmt::Debug,
            StandardNormal: Distribution<T>,
            Array2<T>: Dot<Array2<T>, Output = Array2<T>>
                + Inverse<Output = Array2<T>>
                + QR<Q = Array2<T>, R = Array2<T>>
                + Dot<Array1<T>, Output = Array1<T>>
                + Info<
                    MeanOutput = Array1<T>,
                    RowOutput = Array1<T>,
                    ColOutput = Array1<T>,
                    ColMut = Array1<T>,
                    NcolsOutput = usize,
                    NrowsOutput = usize,
                >,
            Array<T, Ix1>: Info<MeanOutput = Array<T, Ix0>, RowOutput = T>,
            for<'a> T: Mul<&'a Array1<T>, Output = Array1<T>>,
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
                            let (alpha_norm, max_iter, tol, samples, coef, _) =
                                self.init_stochastic_algo(x, y);
                            $grad(
                                &x_centered,
                                &y_centered,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                ($sgd, $norm, tol, &None), // use a struct ?
                            )
                        }
                        RidgeRegressionSolver::EXACT => {
                            let xct = x_centered.t();
                            match (xct.dot(&x_centered)
                                + self.settings.alpha * Array2::eye(x.ncols()))
                            .inv()
                            {
                                Ok(mat) => mat.dot(&xct).dot(&y_centered),
                                Err(error) => return Err(error),
                            }
                        }
                        RidgeRegressionSolver::QR => {
                            let xct = x_centered.t();
                            let (q, r) = match (xct.dot(&x_centered)
                                + self.settings.alpha * Array2::eye(x.ncols()))
                            .qr()
                            {
                                Ok((q, r)) => (q, r),
                                Err(error) => return Err(error),
                            };
                            match r.inv() {
                                Ok(inv_r) => inv_r.dot(&q.t().dot(&xct).dot(&y_centered)),
                                Err(error) => return Err(error),
                            }
                        }
                        RidgeRegressionSolver::SAG => {
                            let (alpha_norm, max_iter, tol, samples, coef, gradients) =
                                self.init_stochastic_algo(x, y);
                            $grad(
                                &x_centered,
                                &y_centered,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                ($sag, $norm, tol, &None), // use a struct ?
                            )
                        }
                    };
                    self.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.coef = Some(coef);
                } else {
                    match self.settings.solver {
                        RidgeRegressionSolver::SGD => {
                            let (alpha_norm, max_iter, tol, samples, coef, _) =
                                self.init_stochastic_algo(x, y);
                            self.coef = Some($grad(
                                x,
                                y,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                ($sgd, $norm, tol, &None),
                            ));
                        }
                        RidgeRegressionSolver::EXACT => {
                            let xt = x.t();
                            self.coef = Some(
                                match (xt.dot(x) + self.settings.alpha * Array2::eye(x.ncols()))
                                    .inv()
                                {
                                    Ok(mat) => mat.dot(&xt).dot(y),
                                    Err(error) => return Err(error),
                                },
                            );
                        }
                        RidgeRegressionSolver::QR => {
                            let (q, r) = match x.qr() {
                                Ok((q, r)) => (q, r),
                                Err(error) => return Err(error),
                            };
                            self.coef = Some(match r.inv() {
                                Ok(inv_r) => inv_r.dot(&q.t().dot(y)),
                                Err(error) => return Err(error),
                            });
                        }
                        RidgeRegressionSolver::SAG => {
                            let (alpha_norm, max_iter, tol, samples, coef, gradients) =
                                self.init_stochastic_algo(x, y);
                            self.coef = Some($grad(
                                x,
                                y,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                ($sag, $norm, tol, &None), // use a struct ?
                            ));
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
impl_ridge_reg!(
    Ix1,
    Ix0,
    randn_1d,
    l2_norm1,
    grad_1d,
    sgd_updator,
    sag_updator
);

impl_ridge_reg!(
    Ix2,
    Ix1,
    randn_2d,
    l2_norm1,
    grad_2d,
    sgd_updator,
    sag_updator
);

fn sgd_updator<T, X, Y>(
    x: &X,
    y: &Y,
    coef: &Y,
    i: usize,
    alpha: T,
    lambda: Option<T>,
    _gradients: &Option<&mut X>,
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
    gradients: &Option<&mut X>,
) -> Y
where
    Y: Info<RowOutput = T> + Dot<Y, Output = T> + Add<Y, Output = Y> + Sub<Y, Output = Y>,
    for<'a> &'a Y: Sub<Y, Output = Y> + Add<Y, Output = Y>,
    X: Info<RowOutput = Y, MeanOutput = Y, ColMut = Y, NrowsOutput = usize>,
    for<'a> T:
        Sub<T, Output = T> + Copy + Mul<Y, Output = Y> + Mul<&'a Y, Output = Y> + FromPrimitive,
{
    let xi = x.get_row(i);
    let yi = y.get_row(i);
    let gradi = alpha * coef + (xi.dot(coef) - yi) * xi;
    // Safe to .unwrap() ?
    let scale = T::from_f32(1. / (x.get_nrows() as f32)).unwrap();
    let update = if let Some(grad) = gradients {
        let pre_update = scale * (&gradi - grad.get_row(i)) + grad.mean();
        if let Some(lamb) = lambda {
            lamb * pre_update
        } else {
            pre_update
        }
    } else {
        panic!("No gradients provided");
    };
    // TODO update gradients
    // if let Some(grad) = gradients {
    //     (*grad).col_mut(i, gradi); // To change to row_mut;
    // }
    update
}

fn grad_1d<T, X, Y, U, N>(
    x: &X,
    y: &Y,
    mut coef: Y,
    max_iter: usize,
    samples: &Array1<usize>,
    alpha_norm: T,
    updators: (U, N, T, &Option<&mut X>),
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
    U: Fn(&X, &Y, &Y, usize, T, Option<T>, &Option<&mut X>) -> Y,
    N: Fn(&Y) -> T,
{
    let (updator, norm_func, tol, gradients) = updators;
    for k in 0..max_iter {
        let i = samples[k];
        let update = updator(x, y, &coef, i, alpha_norm, T::from_f32(0.001), gradients);
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
    updators: (U, N, T, &Option<&mut X>),
) -> X
where
    Xs: Info<RowOutput = T> + Dot<Xs, Output = T> + Add<Xs, Output = Xs>,
    for<'a> &'a Xs: Sub<Xs, Output = Xs>,
    X: Info<RowOutput = Xs, ColOutput = Xs, NcolsOutput = usize, ColMut = Xs>,
    for<'a> T: Sub<T, Output = T>
        + Copy
        + Mul<Xs, Output = Xs>
        + Mul<&'a Xs, Output = Xs>
        + FromPrimitive
        + Float
        + core::fmt::Debug,
    U: Fn(&X, &Xs, &Xs, usize, T, Option<T>, &Option<&mut X>) -> Xs,
    N: Fn(&Xs) -> T,
{
    let nb_reg = coef.get_ncols();
    let (updator, norm_func, tol, gradients) = updators;
    (0..nb_reg)
        .map(|r| {
            let coefr = coef.get_col(r);
            let yr = y.get_col(r);
            coef.col_mut(
                r,
                grad_1d(
                    x,
                    &yr,
                    coefr,
                    max_iter,
                    samples,
                    alpha_norm,
                    (&updator, &norm_func, tol, gradients),
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
