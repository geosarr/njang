mod gradient;
mod linear_classification;
mod linear_regression;
mod unit_test;
extern crate alloc;
use crate::traits::Algebra;
use core::ops::Sub;
use gradient::*;
pub use linear_regression::*;
use ndarray::{linalg::Dot, Array, Array2, ArrayView2, Ix1, Ix2};
use ndarray_linalg::{error::LinalgError, Cholesky, Inverse, Lapack, QR, UPLO};
use ndarray_rand::{
    rand::Rng,
    rand_distr::{uniform::SampleUniform, Uniform},
    RandomExt,
};
use num_traits::Float;
use rand_chacha::ChaCha20Rng;

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

/// Used to compute gradient of the square loss function for linear models (like
/// linear regression, Ridge regression, etc.)
pub(crate) fn square_loss_gradient<T: Lapack, Y>(x: &Array2<T>, y: &Y, coef: &Y) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    return x.t().dot(&(x.dot(coef) - y));
}

pub(crate) trait LinearModelInternal {
    type Scalar;
    fn max_iter(&self) -> Option<usize> {
        None
    }
    fn tol(&self) -> Option<Self::Scalar> {
        None
    }
    fn rng(&self) -> Option<ChaCha20Rng> {
        None
    }
    fn n_targets(&self) -> Option<usize> {
        None
    }
    fn n_samples(&self) -> Option<usize> {
        None
    }
}

macro_rules! impl_settings {
    ($settings:ident) => {
        impl<T: Copy> LinearModelInternal for $settings<T> {
            type Scalar = T;
            fn max_iter(&self) -> Option<usize> {
                self.max_iter
            }
            fn tol(&self) -> Option<Self::Scalar> {
                self.tol
            }
            fn rng(&self) -> Option<ChaCha20Rng> {
                self.rng.clone()
            }
            fn n_targets(&self) -> Option<usize> {
                Some(self.n_targets)
            }
            fn n_samples(&self) -> Option<usize> {
                Some(self.n_samples)
            }
        }
    };
}
impl_settings!(LinearRegressionInternal);

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
