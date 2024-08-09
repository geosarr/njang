mod linear_regression;
mod ridge_regression;
mod unit_test;
use core::ops::{Add, Div, Sub};

use crate::traits::Info;
pub use linear_regression::{
    LinearRegression, LinearRegressionHyperParameter, LinearRegressionSolver,
};
use ndarray::{Array, Array2, Axis, Ix0, Ix1, Ix2};
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

impl<T> Info for Array<T, Ix1>
where
    T: Copy + Zero + FromPrimitive + Add<Output = T> + Div<Output = T>,
{
    type MeanOutput = Array<T, Ix0>;
    type RowOutput = T;
    fn mean(&self) -> Self::MeanOutput {
        self.mean_axis(Axis(0)).unwrap()
    }
    fn get_row(&self, i: usize) -> Self::RowOutput {
        self[i]
    }
}

impl<T> Info for Array<T, Ix2>
where
    T: Copy + Zero + FromPrimitive + Add<Output = T> + Div<Output = T>,
{
    type MeanOutput = Array<T, Ix1>;
    type RowOutput = Array<T, Ix1>;
    fn mean(&self) -> Self::MeanOutput {
        self.mean_axis(Axis(0)).unwrap()
    }
    fn get_row(&self, i: usize) -> Self::RowOutput {
        self.row(i).to_owned()
    }
}
// macro_rules! impl_stat {
//     ($ix:ty, $ix_smaller:ty) => {
//         impl<T> Info for Array<T, $ix> {
//             type MeanOutput = Array<$ft, $ix_smaller>;
//             // type RowOutput = Array<$ft, $ix_smaller>;
//             fn mean(&self) -> Self::MeanOutput {
//                 self.mean_axis(Axis(0)).unwrap()
//             }
//             // fn row(&self, i: usize) -> {

//             // }
//         }
//     };
// }
// impl_stat!(Ix1, Ix0);
// impl_stat!(Ix2, Ix1);
