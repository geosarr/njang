use crate::{linear_model::square_loss_gradient, traits::Algebra};
use core::ops::{Add, Mul, Sub};
use ndarray::{linalg::Dot, Array2, ArrayView2};
use ndarray_linalg::Lapack;

use super::RegressionInternal;

pub(crate) fn linear_regression_gradient<T: Lapack, Y>(
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

pub(crate) fn ridge_regression_gradient<T: Lapack, Y>(
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

pub(crate) fn lasso_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &RegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y>
        + Add<Y, Output = Y>
        + Mul<T, Output = Y>
        + Algebra<Elem = T, SignOutput = Y>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    let l1_penalty = settings.l1_penalty.unwrap();
    return (square_loss_gradient(x, y, coef) + coef.sign() * l1_penalty) * (-step_size);
}

pub(crate) fn elastic_net_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &RegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y>
        + Add<Y, Output = Y>
        + Mul<T, Output = Y>
        + Algebra<Elem = T, SignOutput = Y>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    let (l1_penalty, l2_penalty) = (settings.l1_penalty.unwrap(), settings.l2_penalty.unwrap());
    return (square_loss_gradient(x, y, coef) + coef.sign() * l1_penalty + coef * l2_penalty)
        * (-step_size);
}
