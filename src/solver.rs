use core::ops::Add;

use ndarray::*;
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::{Distribution, Uniform};

use crate::{
    linear_model::LinearModelInternal,
    traits::{Algebra, Container},
};

/// Callers have to make sure that the settings needed are provided in the
/// object `settings` before using this function.
pub(crate) fn batch_gradient_descent<T, Y, G, S>(
    x: &Array2<T>,
    y: &Y,
    mut coef: Y,
    scaled_grad: G,
    settings: &S,
) -> Y
where
    T: Lapack + PartialOrd,
    Y: Algebra<Elem = T>,
    for<'a> &'a Y: Add<Y, Output = Y>,
    G: Fn(&Array2<T>, &Y, &Y, &S) -> Y,
    S: LinearModelInternal<Scalar = T>,
{
    // Callers have to make sure that these settings are provided in the object
    // `settings` before using this function
    let (max_iter, tol) = (settings.max_iter().unwrap(), settings.tol().unwrap());
    for _ in 0..max_iter {
        // minus step size should be multiplied to scaled_grad function output.
        let update = scaled_grad(x, y, &coef, settings);
        if update.l2_norm() < tol * coef.l2_norm() {
            break;
        }
        coef = &coef + update;
    }
    coef
}

/// Callers have to make sure that the settings needed are provided in the
/// object `settings` before using this function.
pub(crate) fn stochastic_gradient_descent<T, Y, G, S>(
    x: &Array2<T>,
    y: &Y,
    mut coef: Y,
    scaled_grad: G,
    settings: &S,
) -> Y
where
    T: Lapack + PartialOrd,
    Y: Algebra<Elem = T, SelectionOutput = Y>,
    for<'a> &'a Y: Add<Y, Output = Y>,
    Array2<T>: Container<SelectionOutput = Array2<T>>,
    G: Fn(&Array2<T>, &Y, &Y, &S) -> Y,
    S: LinearModelInternal<Scalar = T>,
{
    let (max_iter, tol, mut rng, n_targets) = (
        settings.max_iter().unwrap(),
        settings.tol().unwrap(),
        settings.rng().unwrap(),
        settings.n_targets().unwrap(),
    );
    let unif = Uniform::<usize>::new(0, x.nrows());
    // When sampling among matrix x rows, the same sample is duplicated as many
    // times as there are regression models to fit, i.e the number of columns in
    // matrix `y` (or `coef`) is equal to the number of row duplications. This is
    // done in order to facilitate gradients computations. The indices are stored in
    // the object `indices`.
    let mut indices = (0..n_targets).collect::<Vec<usize>>();
    for _ in 0..max_iter {
        indices.fill(unif.sample(&mut rng));
        let xi = x.selection(0, &indices);
        let yi = y.selection(0, &indices);
        // minus step size should be multiplied to `scaled_grad` function output.
        let update = scaled_grad(&xi, &yi, &coef, settings);
        if update.l2_norm() < tol * coef.l2_norm() {
            break;
        }
        coef = &coef + update;
    }
    coef
}

/// Callers have to make sure that the settings needed are provided in the
/// object `settings` before using this function.
///
/// `sum_gradients` and `gradients` arguments should be multiplied before the
/// first iteration by minus step size. The caller shoud use the same
/// `scaled_grad` function to sonstruct these arguments in order to take into
/// account the step size. The shape of `gradients` should be [n_features;
/// n_samples * n_targets] and the shape of `sum_gradients` should be
/// [n_features; n_targets] before the first iteration of this function.
pub(crate) fn stochastic_average_gradient<T, G, S>(
    x: &Array2<T>,
    y: &Array2<T>,
    mut coef: Array2<T>,
    scaled_grad: G,
    settings: &S,
    mut gradients: Array2<T>,
    mut sum_gradients: Array2<T>,
) -> Array2<T>
where
    T: Lapack + PartialOrd + ScalarOperand,
    Array2<T>: Algebra<Elem = T, SelectionOutput = Array2<T>>,
    G: Fn(&Array2<T>, &Array2<T>, &Array2<T>, &S) -> Array2<T>,
    S: LinearModelInternal<Scalar = T>,
{
    let (max_iter, tol, mut rng, n_targets, n_samples) = (
        settings.max_iter().unwrap(),
        settings.tol().unwrap(),
        settings.rng().unwrap(),
        settings.n_targets().unwrap(),
        settings.n_samples().unwrap(),
    );
    let unif = Uniform::<usize>::new(0, x.nrows());
    // When sampling among matrix x rows, the same sample is duplicated as many
    // times as there are regression models to fit, i.e the number of columns in
    // matrix `y` (or `coef`) is equal to the number of row duplications. This is
    // done in order to facilitate gradients computations. The indices are stored in
    // the object `indices`.
    let mut indices = (0..n_targets).collect::<Vec<usize>>();
    // The object `grad_indices` stores the indices of the selected samples
    // gradients.
    let mut grad_indices = (0..n_targets).collect::<Vec<usize>>();
    let scale = T::one() / T::from_usize(n_samples).unwrap();
    for _ in 0..max_iter {
        let index = unif.sample(&mut rng);
        indices.fill(index);
        let xi = x.selection(0, &indices);
        let yi = y.selection(0, &indices);
        // minus step size should be multiplied to `scaled_grad` function output.
        let gradi = scaled_grad(&xi, &yi, &coef, settings);
        // Put inside `grad_indices` the values: `index`; `n_samples + index`; ...;
        // `(n_targets-1)*n_samples + index`, in order to take the `index`-th gradient
        // for each target model.
        grad_indices
            .iter_mut()
            .enumerate()
            .map(|(pos, gi)| *gi = pos * n_samples + index)
            .for_each(drop);
        let previous_grad = gradients.selection(1, &grad_indices);
        let sum = &sum_gradients + (&gradi - previous_grad);
        let update = &sum * scale; // Average gradient
        if update.l2_norm() < tol * coef.l2_norm() {
            break;
        }
        coef = &coef + update;
        sum_gradients = sum;
        // gradi.assign_to(gradients.slice_mut(s!(.., &grad_indices)));
        grad_indices
            .iter()
            .enumerate()
            .map(|(pos, i)| gradients.column_mut(*i).assign(&gradi.column(pos)))
            .for_each(drop);
    }
    coef
}
