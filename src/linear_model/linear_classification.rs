use core::{hash::Hash, marker::PhantomData};

use ndarray::{Array, Array1, Array2, Axis, Ix0, Ix1, ScalarOperand};
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{zero, Float};

use crate::{
    error::NjangError,
    traits::{ClassificationModel, Container, RegressionModel},
};

use super::{LinearRegression, LinearRegressionSettings, LinearRegressionSolver};
use std::collections::HashSet;

#[derive(Default, Debug, Clone, Copy)]
pub enum LinearClassificationSolver {
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
    // ///
    // Exact
    // ///
    // Svd
    // ///
    // Qr
    // ///
    // Cholesky
}

pub type RidgeClassificationSolver = LinearRegressionSolver;
pub struct RidgeClassificationSettings<T> {
    pub fit_intercept: bool,
    pub solver: RidgeClassificationSolver,
    pub l2_penalty: Option<T>,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

pub struct RidgeClassification<C, I, L = i32>
where
    C: Container,
{
    model: LinearRegression<C, I>,
    labels: Vec<L>,
}

impl<C: Container, I, L> RidgeClassification<C, I, L> {
    pub fn new(settings: RidgeClassificationSettings<C::Elem>) -> Self
    where
        C::Elem: Float,
    {
        let lin_settings = LinearRegressionSettings {
            fit_intercept: settings.fit_intercept,
            solver: settings.solver,
            l1_penalty: None,
            l2_penalty: settings.l2_penalty,
            tol: settings.tol,
            step_size: settings.step_size,
            random_state: settings.random_state,
            max_iter: settings.max_iter,
        };
        Self {
            model: LinearRegression::new(lin_settings),
            labels: Vec::new(),
        }
    }
    /// Coefficients of the model
    pub fn coef(&self) -> Option<&C> {
        self.model.parameter.coef.as_ref()
    }
    /// Intercept of the model
    pub fn intercept(&self) -> Option<&I> {
        self.model.parameter.intercept.as_ref()
    }
}

fn unique_labels<L>(labels: L) -> Vec<L::Item>
where
    L: IntoIterator,
    L::Item: Eq + Hash + Ord + Copy,
{
    let unique_labels = labels.into_iter().collect::<HashSet<_>>();
    let mut unique_labels = unique_labels.into_iter().collect::<Vec<_>>();
    unique_labels.sort();
    unique_labels
}

fn argmax<I>(iterable: I) -> Vec<L::Item>
where
    I: IntoIterator,
    I::Item: Eq + Ord + Copy,
{
    let mut container = iterable.into_iter().collect::<Vec<_>>();
    container.arg
}

impl<T: Lapack + ScalarOperand + PartialOrd + Float + SampleUniform, L: Eq + Hash + Ord + Copy>
    ClassificationModel for RidgeClassification<Array2<T>, Array1<T>, L>
{
    type FitResult = Result<(), NjangError>;
    type X = Array2<T>;
    type Y = Array1<L>;
    type PredictResult = Result<Array2<T>, ()>;
    type PredictProbaResult = Result<Array2<T>, ()>;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
        self.labels = unique_labels(y.iter()).into_iter().copied().collect();
        let mut y_reg = Array2::<T>::zeros((y.len(), self.labels.len()));
        y_reg
            .axis_iter_mut(Axis(0))
            .enumerate()
            .map(|(sample, mut row)| {
                let pos = self.labels.binary_search(&y[sample]).unwrap(); // Safe to .unwrap() since self.labels is built from y.
                row[pos] = T::one();
            })
            .for_each(drop);
        println!("y_reg:\n{:?}", y_reg);
        self.model.fit(x, &y_reg)
        // Ok(())
    }
    fn predict(&self, x: &Self::X) -> Self::PredictResult {
        let prediction = self
            .model
            .predict(x)?
            .axis_iter(Axis(0))
            .map(|row| self.labels[argmax(row.iter().enumerate())]);

        // self.model.predict(x)
    }
    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult {
        Err(())
    }
}

#[test]
fn code() {
    use ndarray::array;
    use std::collections::HashSet;
    let kmers: Vec<u8> = vec![64, 64, 64, 65, 65, 65];
    let nodes = kmers.iter().copied().count();
    println!("{:?}", nodes);
    // println!("{:?}", y);
    // println!("{:?}", unique_labels(y.iter()));
    // println!("{:?}",);

    let x = array![[0., 0., 1.], [1., 0., 0.]];
    let y = array!["sale", "propre"];
    // println!("x:\n{:?}\n", x);
    // println!("xxt:\n{:?}\n", x.dot(&x.t()));
    // let coef = array![1., 2., 3.];
    // println!("coef:\n{:?}\n", coef);
    // let y = x.dot(&coef);
    // println!("y:\n{:?}\n", y);
    // println!("xty:\n{:?}\n", x.t().dot(&y));

    let settings = RidgeClassificationSettings {
        fit_intercept: false,
        solver: RidgeClassificationSolver::Svd,
        l2_penalty: None,
        tol: Some(1e-6),
        step_size: Some(1e-3),
        random_state: Some(0),
        max_iter: Some(100000),
    };
    let mut model = RidgeClassification::<Array2<_>, _, &str>::new(settings);
    match model.fit(&x, &y) {
        Ok(_) => {
            println!("{:?}", model.coef());
        }
        Err(error) => {
            println!("{:?}", error);
        }
    };

    match model.predict(&x) {
        Ok(value) => {
            println!("\ny_pred:\n{:?}", value);
        }
        Err(error) => {
            println!("{:?}", error);
        }
    };
}
