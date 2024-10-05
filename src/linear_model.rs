//! This module implements some  classification and regression linear models.
mod classification;
mod gradient;
mod regression;
extern crate alloc;
use crate::traits::{Algebra, Container};
pub use classification::*;
use core::ops::{Add, Mul, Sub};
use gradient::*;
use ndarray::{linalg::Dot, Array, Array2, ArrayView2, Ix1, Ix2};
use ndarray_linalg::{error::LinalgError, Cholesky, Inverse, Lapack, QR, UPLO};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{
    rand::Rng,
    rand_distr::{uniform::SampleUniform, Uniform},
    RandomExt,
};
use num_traits::{Float, FromPrimitive};
use rand_chacha::ChaCha20Rng;
pub use regression::*;

const DEFAULT_L1: f32 = 1.;
const DEFAULT_L2: f32 = 1.;
const DEFAULT_TOL: f32 = 1e-3;
const DEFAULT_STEP_SIZE: f32 = 1e-3;
const DEFAULT_STATE: u32 = 0;
const DEFAULT_MAX_ITER: usize = 1000;

/// Used to preprocess data for linear models
pub(crate) fn preprocess<X, Y, MX, MY>(x: &X, y: &Y) -> (X, MX, Y, MY)
where
    X: Algebra<MeanAxisOutput = MX>,
    Y: Algebra<MeanAxisOutput = MY>,
    for<'a> &'a X: Sub<&'a MX, Output = X>,
    for<'a> &'a Y: Sub<&'a MY, Output = Y>,
{
    let x_mean = x.mean_axis(0);
    let y_mean = y.mean_axis(0);
    let x_centered = x - &x_mean;
    let y_centered = y - &y_mean;
    (x_centered, x_mean, y_centered, y_mean)
}

/// Parameters used in a linear model.
#[derive(Debug, Default, Clone, Copy)]
pub struct LinearModelParameter<C, I> {
    /// Non-intercept weight(s).
    pub coef: Option<C>,
    /// Intercept weight(s) of the model.
    pub intercept: Option<I>,
}

/// Solvers used in a linear model.
#[derive(Debug, Default, Clone, Copy)]
pub enum LinearModelSolver {
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
pub(crate) struct LinearModelInternal<T> {
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
impl<T> LinearModelInternal<T> {
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

pub(crate) fn exact<T, Y>(x: Array2<T>, z: ArrayView2<T>, y: &Y) -> Result<Y, LinalgError>
where
    T: Lapack,
    Array2<T>: Dot<Y, Output = Y> + for<'a> Dot<ArrayView2<'a, T>, Output = Array2<T>>,
{
    match x.inv() {
        Ok(mat) => Ok(mat.dot(&z).dot(y)),
        Err(error) => Err(error),
    }
}
pub(crate) fn qr<T, Y>(x: Array2<T>, z: ArrayView2<T>, y: &Y) -> Result<Y, LinalgError>
where
    T: Lapack,
    Array2<T>: Dot<Y, Output = Y>,
{
    match x.qr() {
        Ok((q, r)) => match r.inv() {
            Ok(inv_r) => Ok(inv_r.dot(&q.t().dot(&z).dot(y))),
            Err(error) => Err(error),
        },
        Err(error) => Err(error),
    }
}
pub(crate) fn cholesky<T, Y>(x: Array2<T>, z: ArrayView2<T>, y: &Y) -> Result<Y, LinalgError>
where
    T: Lapack,
    Array2<T>: Dot<Y, Output = Y> + for<'a> Dot<ArrayView2<'a, T>, Output = Array2<T>>,
{
    match x.cholesky(UPLO::Lower) {
        Ok(mat) => match mat.inv() {
            Ok(inv_m) => Ok(inv_m.t().dot(&inv_m).dot(&z).dot(y)),
            Err(error) => Err(error),
        },
        Err(error) => Err(error),
    }
}

pub(crate) fn randu_1d<T: Float + SampleUniform, R: Rng>(
    m: &[usize],
    rng: &mut R,
) -> Array<T, Ix1> {
    let sqrt_n = T::from(m[0]).unwrap().sqrt();
    let high = T::one() / sqrt_n;
    Array::<T, Ix1>::random_using(m[0], Uniform::new_inclusive(-high, high), rng)
}

pub(crate) fn randu_2d<T: Float + SampleUniform, R: Rng>(
    m: &[usize],
    rng: &mut R,
) -> Array<T, Ix2> {
    let sqrt = T::from(m[0]).unwrap().sqrt();
    let high = T::one() / sqrt;
    Array::<T, Ix2>::random_using((m[0], m[1]), Uniform::new_inclusive(-high, high), rng)
}

macro_rules! impl_partial_linear_model {
    ($model:ident, $settings:ident, $internal:ident, $linear_grad:ident, $lasso_grad:ident, $ridge_grad:ident, $elastic_grad:ident, $($container:ident),*) => {
        impl<$( $container ),*> $model<$( $container ),*>
        where C: Container
        {
            /// Coefficients of the model
            pub fn coef(&self) -> Option<&C> {
                self.parameter.coef.as_ref()
            }
            /// Intercept of the model
            pub fn intercept(&self) -> Option<&I> {
                self.parameter.intercept.as_ref()
            }
            /// Whether or not the model has an Elastic Net penalty.
            pub fn is_elastic_net(&self) -> bool {
                self.settings.l1_penalty.is_some() && self.settings.l2_penalty.is_some()
            }
            /// Whether or not the model has a Ridge penalty.
            pub fn is_ridge(&self) -> bool {
                self.settings.l1_penalty.is_none() && self.settings.l2_penalty.is_some()
            }
            /// Whether or not the model has a Lasso penalty.
            pub fn is_lasso(&self) -> bool {
                self.settings.l1_penalty.is_some() && self.settings.l2_penalty.is_none()
            }
            /// Whether or not the model has penalty.
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
                    self.scale_l1_penalty();
                } else if self.is_ridge() {
                    self.scale_l2_penalty();
                } else if self.is_elastic_net() {
                    self.scale_l1_penalty();
                    self.scale_l2_penalty();
                }
            }
            fn gradient_function<T, Y>(&self) -> impl Fn(&Array2<T>, &Y, &Y, &$internal<T>) -> Y
            where
                T: Lapack,
                for<'a> Y: Sub<&'a Y, Output = Y>
                    + Add<Y, Output = Y>
                    + Mul<T, Output = Y>
                    + Algebra<Elem = T, SignOutput = Y, SoftmaxOutput = Y>,
                for<'a> &'a Y: Mul<T, Output = Y>,
                Array2<T>: Dot<Y, Output = Y>,
                for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
            {
                // Use here a match pattern with enum instead of if else's ?
                if self.is_linear() {
                    $linear_grad
                } else if self.is_ridge() {
                    $ridge_grad
                } else if self.is_lasso() {
                    $lasso_grad
                } else {
                    $elastic_grad
                }
            }
        }
    };
}
macro_rules! impl_scale_penalty {
    ($model:ident, $scaler_name:ident, $field:ident, $($container:ident),*) => {
        impl<$($container),*> $model<$($container),*>
        where
            C: Container,
            C::Elem: Float + FromPrimitive,
        {
            fn $scaler_name(&mut self) {
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
macro_rules! impl_settings_to_internal {
    ($model:ident, $setter_name:ident, $field_name:ident, $default:ident, $($container:ident),*) => {
        impl<$($container),*> $model<$($container),*>
        where
            C: Container,
            C::Elem: Copy + FromPrimitive + core::fmt::Debug,
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

macro_rules! impl_all_linear_model {
    ($model:ident, $settings:ident, $internal:ident, $linear_grad:ident, $lasso_grad:ident, $ridge_grad:ident, $elastic_grad:ident, $($container:ident),*) => {
        impl_scale_penalty!($model, scale_l1_penalty, l1_penalty, $($container),*);
        impl_scale_penalty!($model, scale_l2_penalty, l2_penalty, $($container),*);
        impl_settings_to_internal!($model, set_l1_penalty_to_internal, l1_penalty, DEFAULT_L1, $($container),*);
        impl_settings_to_internal!($model, set_l2_penalty_to_internal, l2_penalty, DEFAULT_L2, $($container),*);
        impl_settings_to_internal!($model, set_tol_to_internal, tol, DEFAULT_TOL, $($container),*);
        impl_settings_to_internal!(
            $model,
            set_step_size_to_internal,
            step_size,
            DEFAULT_STEP_SIZE,
            $($container),*
        );
        impl_partial_linear_model!(
            $model,
            $settings,
            $internal,
            $linear_grad,
            $lasso_grad,
            $ridge_grad,
            $elastic_grad,
            $($container),*
        );
    };
}

impl_all_linear_model!(
    LinearRegression,
    LinearRegressionSettings,
    LinearModelInternal,
    linear_regression_gradient,
    lasso_regression_gradient,
    ridge_regression_gradient,
    elastic_net_regression_gradient,
    C,
    I
);
impl_all_linear_model!(
    LogisticRegression,
    LogisticRegressionSettings,
    LinearModelInternal,
    logistic_regression_gradient,
    logistic_lasso_regression_gradient,
    logistic_ridge_regression_gradient,
    logistic_elastic_net_regression_gradient,
    C,
    I,
    L
);
