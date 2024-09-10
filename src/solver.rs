use core::ops::{Add, Mul, Sub};

use linalg::Dot;
use ndarray::*;
use ndarray_linalg::Lapack;
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Distribution, Uniform},
};
use rand_chacha::ChaCha20Rng;

use crate::{
    linear_model::LinearModelSettings,
    traits::{Algebra, Container},
    LinearRegressionSettings,
};

pub(crate) fn batch_gradient_descent<T: Lapack + PartialOrd, Y>(
    x: &Array2<T>,
    y: &Y,
    mut coef: Y,
    settings: &LinearRegressionSettings<T>,
) -> Y
where
    for<'a> Y: Algebra<Elem = T> + Sub<&'a Y, Output = Y> + Add<Y, Output = Y> + Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let (step_size, max_iter, tol) = (
        settings.step_size.unwrap(),
        settings.max_iter.unwrap(),
        settings.tol.unwrap(),
    );
    let xt = x.t();
    for _ in 0..max_iter {
        let update = xt.dot(&(x.dot(&coef) - y)) * (-step_size);
        if update.l2_norm() < tol * coef.l2_norm() {
            break;
        }
        coef = coef + update;
    }
    coef
}

pub(crate) fn stochastic_gradient_descent<
    T: Lapack + PartialOrd,
    Y,
    G,
    S: LinearModelSettings<Scalar = T>,
>(
    x: &Array2<T>,
    y: &Y,
    mut coef: Y,
    scaled_grad: G,
    settings: &S,
) -> Y
where
    for<'a> Y: Algebra<Elem = T, SelectionOutput = Y>
        + Sub<&'a Y, Output = Y>
        + Add<Y, Output = Y>
        + Mul<T, Output = Y>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Container<SelectionOutput = Array2<T>> + Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
    G: Fn(&Array2<T>, &Y, &Y, &S) -> Y,
{
    let (max_iter, tol, random_state) = (
        settings.max_iter().unwrap(),
        settings.tol().unwrap(),
        settings.random_state().unwrap() as u64,
    );
    let mut rng = ChaCha20Rng::seed_from_u64(random_state);
    let unif = Uniform::<usize>::new(0, x.nrows());
    let shape = y.dimension();
    let mut indices = if shape.len() == 2 {
        (0..shape[1]).collect::<Vec<usize>>()
    } else {
        vec![0]
    };
    for _ in 0..max_iter {
        let index = unif.sample(&mut rng);
        indices.fill(index);
        let xi = x.selection(0, &indices);
        let yi = y.selection(0, &indices);
        let update = scaled_grad(&xi, &yi, &coef, &settings); // (minus step size should be mutiplied to scale_grad function output).
        if update.l2_norm() < tol * coef.l2_norm() {
            break;
        }
        coef = coef + update;
    }
    coef
}
