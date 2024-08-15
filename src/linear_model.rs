mod linear_regression;
mod ridge_regression;
mod unit_test;
use core::ops::{Add, Div, Not, Sub};
extern crate alloc;
use crate::traits::Info;
use alloc::vec::Vec;
pub use linear_regression::{
    LinearRegression, LinearRegressionHyperParameter, LinearRegressionSolver,
};
use ndarray::{linalg::Dot, Array, Array1, Array2, ArrayView2, Axis, Ix0, Ix1, Ix2, ScalarOperand};
use ndarray_linalg::{error::LinalgError, Cholesky, Inverse, Lapack, QR, UPLO};
use ndarray_rand::{
    rand::{distributions::Distribution, Rng},
    rand_distr::StandardNormal,
    RandomExt,
};
use num_traits::{FromPrimitive, Zero};
pub use ridge_regression::{RidgeRegression, RidgeRegressionHyperParameter, RidgeRegressionSolver};

/// Used to preprocess data for linear models
pub(crate) fn preprocess<X, Y, MX, MY>(x: &X, y: &Y) -> (X, MX, Y, MY)
where
    X: Info<MeanOutput = MX>,
    Y: Info<MeanOutput = MY>,
    for<'a> &'a X: Sub<&'a MX, Output = X>,
    for<'a> &'a Y: Sub<&'a MY, Output = Y>,
{
    let x_mean = x.mean();
    let y_mean = y.mean();
    let x_centered = x - &x_mean;
    let y_centered = y - &y_mean;
    (x_centered, x_mean, y_centered, y_mean)
}

macro_rules! impl_linalg {
    ($exact_name:ident, $qr_name:ident, $chol_name:ident, $ix:ty) => {
        pub(crate) fn $exact_name<T>(
            x: Array2<T>,
            z: ArrayView2<T>,
            y: &Array<T, $ix>,
        ) -> Result<Array<T, $ix>, LinalgError>
        where
            T: Lapack,
        {
            match x.inv() {
                Ok(mat) => Ok(mat.dot(&z).dot(y)),
                Err(error) => Err(error),
            }
        }
        pub(crate) fn $qr_name<T>(
            x: Array2<T>,
            z: ArrayView2<T>,
            y: &Array<T, $ix>,
        ) -> Result<Array<T, $ix>, LinalgError>
        where
            T: Lapack,
        {
            match x.qr() {
                Ok((q, r)) => match r.inv() {
                    Ok(inv_r) => Ok(inv_r.dot(&q.t().dot(&z).dot(y))),
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            }
        }
        pub(crate) fn $chol_name<T>(
            x: Array2<T>,
            z: ArrayView2<T>,
            y: &Array<T, $ix>,
        ) -> Result<Array<T, $ix>, LinalgError>
        where
            T: Lapack,
        {
            match x.cholesky(UPLO::Lower) {
                Ok(mat) => match mat.inv() {
                    Ok(inv_m) => Ok(inv_m.t().dot(&inv_m).dot(&z).dot(y)),
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            }
        }
    };
}
impl_linalg!(solve_exact1, solve_qr1, solve_chol1, Ix1);
impl_linalg!(solve_exact2, solve_qr2, solve_chol2, Ix2);
// pub(crate) fn cholesky<T>(x: &Array2<T>, y: &Y) {}

pub(crate) fn randn_1d<T, R: Rng>(n: usize, _m: &[usize], rng: &mut R) -> Array<T, Ix1>
where
    StandardNormal: Distribution<T>,
{
    Array::<T, Ix1>::random_using(n, StandardNormal, rng)
}

pub(crate) fn randn_2d<T, R: Rng>(n: usize, m: &[usize], rng: &mut R) -> Array<T, Ix2>
where
    StandardNormal: Distribution<T>,
{
    Array::<T, Ix2>::random_using((n, m[1]), StandardNormal, rng)
}

impl<T> Info for Array<T, Ix1>
where
    T: Copy + Zero + FromPrimitive + Add<Output = T> + Div<Output = T>,
{
    type MeanOutput = Array<T, Ix0>;
    type RowOutput = T;
    type ColOutput = T;
    type ShapeOutput = Vec<usize>;
    type ColMut = ();
    type NcolsOutput = ();
    type NrowsOutput = ();
    fn mean(&self) -> Self::MeanOutput {
        self.mean_axis(Axis(0)).unwrap()
    }
    fn get_row(&self, i: usize) -> Self::RowOutput {
        self[i]
    }
    fn get_col(&self, i: usize) -> Self::ColOutput {
        self[i]
    }
    fn shape(&self) -> Self::ShapeOutput {
        Array::<T, Ix1>::shape(self).into()
    }
    fn col_mut(&mut self, _idx: usize, _elem: ()) {}
    fn get_ncols(&self) {}
    fn get_nrows(&self) {}
}

impl<T> Info for Array<T, Ix2>
where
    T: Copy + Zero + FromPrimitive + Add<Output = T> + Div<Output = T>,
{
    type MeanOutput = Array<T, Ix1>;
    type RowOutput = Array<T, Ix1>;
    type ColOutput = Array<T, Ix1>;
    type ShapeOutput = Vec<usize>;
    type ColMut = Array1<T>;
    type NcolsOutput = usize;
    type NrowsOutput = usize;
    fn mean(&self) -> Self::MeanOutput {
        self.mean_axis(Axis(0)).unwrap()
    }
    fn get_row(&self, i: usize) -> Self::RowOutput {
        self.row(i).to_owned()
    }
    fn get_col(&self, i: usize) -> Self::ColOutput {
        self.column(i).to_owned()
    }
    fn shape(&self) -> Self::ShapeOutput {
        Array::<T, Ix2>::shape(self).into()
    }
    fn col_mut(&mut self, idx: usize, elem: Self::ColMut) {
        self.column_mut(idx).assign(&elem);
    }
    fn get_ncols(&self) -> Self::NcolsOutput {
        self.ncols()
    }
    fn get_nrows(&self) -> Self::NrowsOutput {
        self.nrows()
    }
}
