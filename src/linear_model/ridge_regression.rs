use ndarray::{linalg::Dot, Array, Array1, Array2, Ix0, Ix1, Ix2, ScalarOperand};
use ndarray_rand::rand::Rng;

use crate::RegressionModel;
use crate::{linear_model::preprocess, traits::Info};
extern crate alloc;
use alloc::boxed::Box;
use core::{
    marker::{Send, Sync},
    ops::{Add, Mul, Sub},
};
use ndarray_linalg::{error::LinalgError, Inverse, Lapack, QR};
use ndarray_rand::{
    rand::{distributions::Distribution, SeedableRng},
    rand_distr::{StandardNormal, Uniform},
    RandomExt,
};
use num_traits::FromPrimitive;
use rand_chacha::ChaCha20Rng;

/// Solver to use when fitting a ridge regression model (L2-penalty with Ordinary Least Squares).
#[derive(Debug, Default)]
pub enum RidgeRegressionSolver {
    /// Solves using stochastic gradient descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm may not converge.
    #[default]
    Sgd,
    /// Computing the exaction solution (x.t().dot(x) + alpha * eye).inverse().dot(x.t()).dot(y)
    Exact,
    /// Uses QR decomposition of the matrix x.t().dot(x) + alpha * eye to solve the problem
    Qr,
}

/// Hyperparameters used in a Ridge regression.
#[derive(Debug)]
pub struct RidgeRegressionHyperParameter<T> {
    pub alpha: T,
    pub fit_intercept: bool,
    pub solver: RidgeRegressionSolver,
    pub tol: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
    pub warm_start: Option<Box<Self>>,
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
            warm_start: None,
        }
    }
}
impl<T> RidgeRegressionHyperParameter<T> {
    pub fn new_exact(alpha: T, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            solver: RidgeRegressionSolver::Exact,
            tol: None,
            random_state: None,
            max_iter: None,
            warm_start: None,
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
    ($ix:ty, $ix_smaller:ty,  $randn:ident, $grad:ident) => {
        impl<T> RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>, T> {
            fn init_stochastic_algo(
                &self,
                x: &Array2<T>,
                y: &Array<T, $ix>,
            ) -> (T, usize, T, Array1<usize>, Array<T, $ix>)
            where
                T: Lapack,
                StandardNormal: Distribution<T>,
            {
                let mut rng =
                    ChaCha20Rng::seed_from_u64(self.settings.random_state.unwrap_or(0).into());
                let (n_samples, n_features) = (x.nrows(), x.ncols());
                let coef = $randn(n_features, y.shape(), &mut rng);
                let max_iter = self.settings.max_iter.unwrap_or(1000);
                let nf = T::from_f32(n_samples as f32).unwrap(); // critical when number of samples > int(f3::MAX) ?
                let lambda = T::from_f32(0.001).unwrap(); // to determine automatically.
                let samples = Array::<usize, _>::random_using(
                    max_iter,
                    Uniform::from(0..n_samples),
                    &mut rng,
                );
                let alpha_norm = self.settings.alpha / nf;
                (alpha_norm, max_iter, lambda, samples, coef)
            }
        }
        impl<T> RegressionModel for RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>, T>
        where
            T: Lapack
                + ScalarOperand
                + Mul<Array1<T>, Output = Array1<T>>
                + Mul<Array2<T>, Output = Array2<T>>,
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
                        RidgeRegressionSolver::Sgd => {
                            let (alpha_norm, max_iter, lambda, samples, coef) =
                                self.init_stochastic_algo(x, y);
                            $grad(
                                &x_centered,
                                &y_centered,
                                coef,
                                max_iter,
                                &samples,
                                alpha_norm,
                                lambda,
                            )
                        }
                        RidgeRegressionSolver::Exact => {
                            let xct = x_centered.t();
                            match (xct.dot(&x_centered)
                                + self.settings.alpha * Array2::eye(x.ncols()))
                            .inv()
                            {
                                Ok(mat) => mat.dot(&xct).dot(&y_centered),
                                Err(error) => return Err(error),
                            }
                        }
                        RidgeRegressionSolver::Qr => {
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
                    };
                    self.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.coef = Some(coef);
                } else {
                    match self.settings.solver {
                        RidgeRegressionSolver::Sgd => {
                            let (alpha_norm, max_iter, lambda, samples, coef) =
                                self.init_stochastic_algo(x, y);
                            self.coef =
                                Some($grad(x, y, coef, max_iter, &samples, alpha_norm, lambda));
                        }
                        RidgeRegressionSolver::Exact => {
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
                        RidgeRegressionSolver::Qr => {
                            let (q, r) = match x.qr() {
                                Ok((q, r)) => (q, r),
                                Err(error) => return Err(error),
                            };
                            self.coef = Some(match r.inv() {
                                Ok(inv_r) => inv_r.dot(&q.t().dot(y)),
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
impl_ridge_reg!(Ix1, Ix0, randn_1d, grad_1d);
impl_ridge_reg!(Ix2, Ix1, randn_2d, grad_2d);

fn grad_1d<T, X, Y>(
    x: &X,
    y: &Y,
    mut coef: Y,
    max_iter: usize,
    samples: &Array1<usize>,
    alpha_norm: T,
    lambda: T,
) -> Y
where
    Y: Info<RowOutput = T> + Dot<Y, Output = T> + Add<Y, Output = Y>,
    for<'a> &'a Y: Sub<Y, Output = Y>,
    X: Info<RowOutput = Y>,
    T: Sub<T, Output = T> + Copy,
    for<'a> T: Mul<Y, Output = Y> + Mul<&'a Y, Output = Y>,
{
    for k in 0..max_iter {
        let i = samples[k];
        let xi = x.get_row(i);
        let yi = y.get_row(i);
        let g_cost = alpha_norm * &coef + (xi.dot(&coef) - yi) * xi;
        coef = &coef - lambda * g_cost;
    }
    coef
}

fn grad_2d<T, X, Xsmaller>(
    x: &X,
    y: &X,
    mut coef: X,
    max_iter: usize,
    samples: &Array1<usize>,
    alpha_norm: T,
    lambda: T,
) -> X
where
    Xsmaller: Info<RowOutput = T> + Dot<Xsmaller, Output = T> + Add<Xsmaller, Output = Xsmaller>,
    for<'a> &'a Xsmaller: Sub<Xsmaller, Output = Xsmaller>,
    X: Info<RowOutput = Xsmaller, ColOutput = Xsmaller, NcolsOutput = usize, ColMut = Xsmaller>,
    T: Sub<T, Output = T> + Copy,
    for<'a> T: Mul<Xsmaller, Output = Xsmaller> + Mul<&'a Xsmaller, Output = Xsmaller>,
{
    let nb_reg = coef.get_ncols();
    (0..nb_reg)
        .into_iter()
        .map(|r| {
            let coefr = coef.get_col(r);
            let yr = y.get_col(r);
            coef.col_mut(
                r,
                grad_1d(x, &yr, coefr, max_iter, &samples, alpha_norm, lambda),
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

fn randn_1d<T, R: Rng>(n: usize, _m: &[usize], rng: &mut R) -> Array<T, Ix1>
where
    StandardNormal: Distribution<T>,
{
    Array::<T, Ix1>::random_using(n, StandardNormal, rng)
}

fn randn_2d<T, R: Rng>(n: usize, m: &[usize], rng: &mut R) -> Array<T, Ix2>
where
    StandardNormal: Distribution<T>,
{
    Array::<T, Ix2>::random_using((n, m[1]), StandardNormal, rng)
}
