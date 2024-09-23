use core::hash::Hash;
mod logistic_regression;
mod ridge_classification;
mod unit_test;
pub use logistic_regression::*;
use ndarray::{Array1, Array2, Axis};
use num_traits::Float;

use super::{LinearRegressionSettings, LinearRegressionSolver};
use crate::traits::{ClassificationModel, Container, Model, RegressionModel, Scalar};
pub use ridge_classification::{
    RidgeClassification, RidgeClassificationSettings, RidgeClassificationSolver,
};
use std::collections::HashSet;

pub(crate) fn unique_labels<L>(labels: L) -> Vec<L::Item>
where
    L: IntoIterator,
    L::Item: Eq + Hash + Ord,
{
    let unique_labels = labels.into_iter().collect::<HashSet<_>>();
    let mut unique_labels = unique_labels.into_iter().collect::<Vec<_>>();
    unique_labels.sort();
    unique_labels
}

pub(crate) fn argmax<I, T>(iterable: I) -> Result<usize, isize>
where
    I: IntoIterator<Item = (usize, T)>,
    T: Float,
{
    let mut m = -1;
    let mut max = T::neg_infinity();
    iterable
        .into_iter()
        .map(|(ix, val)| {
            if val > max {
                m = ix as isize;
                max = val;
            }
        })
        .for_each(drop);
    if m >= 0 {
        Ok(m as usize)
    } else {
        Err(m) // Empty iterator
    }
}

pub(crate) fn dummies<T: Scalar, L: Ord>(y: &Array1<L>, labels: &[L]) -> Array2<T> {
    let mut y_reg = Array2::<T>::zeros((y.len(), labels.len()));
    y_reg
        .axis_iter_mut(Axis(0))
        .enumerate()
        .map(|(sample, mut row)| {
            let pos = labels.binary_search(&y[sample]).unwrap(); // Safe to .unwrap() since self.labels is built from y.
            row[pos] = T::one();
        })
        .for_each(drop);
    y_reg
}
