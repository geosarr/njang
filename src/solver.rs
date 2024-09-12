use core::ops::{Add, Mul, Sub};

use linalg::Dot;
use ndarray::*;
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::{Distribution, Uniform};

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

pub(crate) fn stochastic_gradient_descent<T, Y, G, S>(
    x: &Array2<T>,
    y: &Y,
    mut coef: Y,
    scaled_grad: G,
    settings: &S,
) -> Y
where
    T: Lapack + PartialOrd,
    for<'a> Y: Algebra<Elem = T, SelectionOutput = Y>
        + Sub<&'a Y, Output = Y>
        + Add<Y, Output = Y>
        + Mul<T, Output = Y>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Container<SelectionOutput = Array2<T>> + Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
    G: Fn(&Array2<T>, &Y, &Y, &S) -> Y,
    S: LinearModelSettings<Scalar = T>,
{
    // Users have to make sure that these settings are provided in the object
    // `settings` before using this function
    let (max_iter, tol, mut rng) = (
        settings.max_iter().unwrap(),
        settings.tol().unwrap(),
        settings.rng().unwrap(),
    );
    let unif = Uniform::<usize>::new(0, x.nrows());
    let shape = y.dimension();
    // When sampling among matrix x rows, the same sample is duplicated as many
    // times as there are regression models to fit, i.e the number of columns in
    // matrix `y` (or `coef`) is equal to the number of row duplications.
    // The indices are stored in the object `indices`.
    let nb_reg = if shape.len() == 2 { shape[1] } else { 1 };
    let mut indices = (0..nb_reg).collect::<Vec<usize>>();
    for _ in 0..max_iter {
        let index = unif.sample(&mut rng);
        indices.fill(index);
        let xi = x.selection(0, &indices);
        let yi = y.selection(0, &indices);
        // minus step size should be multiplied to scaled_grad function output.
        let update = scaled_grad(&xi, &yi, &coef, settings);
        if update.l2_norm() < tol * coef.l2_norm() {
            break;
        }
        coef = coef + update;
    }
    coef
}
