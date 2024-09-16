use ndarray::{Array, Array1, Array2, Ix0, Ix1, ScalarOperand};
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::Float;

use crate::{
    error::NjangError,
    traits::{ClassificationModel, Container, RegressionModel},
};

use super::{LinearRegression, LinearRegressionSettings, LinearRegressionSolver};

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

pub struct RidgeClassification<C, I>
where
    C: Container,
{
    model: LinearRegression<C, I>,
}

impl<C: Container, I> RidgeClassification<C, I> {
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

impl<T: Lapack + ScalarOperand + PartialOrd + Float + SampleUniform> ClassificationModel
    for RidgeClassification<Array<T, Ix1>, Array<T, Ix0>>
{
    type FitResult = Result<(), NjangError>;
    type X = Array2<T>;
    type Y = Array<T, Ix1>;
    type PredictResult = Result<Array<T, Ix1>, ()>;
    type PredictProbaResult = Result<Array<T, Ix1>, ()>;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
        self.model.fit(x, y)
    }
    fn predict(&self, x: &Self::X) -> Self::PredictResult {
        self.model.predict(x)
    }
    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult {
        Err(())
    }
}

#[test]
fn code() {
    use ndarray::array;

    let x = array![[0., 0., 1.], [1., 0., 0.]];
    println!("x:\n{:?}\n", x);
    println!("xxt:\n{:?}\n", x.dot(&x.t()));
    let coef = array![1., 2., 3.];
    println!("coef:\n{:?}\n", coef);
    let y = x.dot(&coef);
    println!("y:\n{:?}\n", y);
    println!("xty:\n{:?}\n", x.t().dot(&y));

    let settings = RidgeClassificationSettings {
        fit_intercept: false,
        solver: RidgeClassificationSolver::Svd,
        l2_penalty: None,
        tol: Some(1e-6),
        step_size: Some(1e-3),
        random_state: Some(0),
        max_iter: Some(100000),
    };
    let mut model = RidgeClassification::<Array1<_>, _>::new(settings);
    match model.fit(&x, &y) {
        Ok(_) => {
            println!("{:?}", model.coef());
        }
        Err(error) => {
            println!("{:?}", error);
        }
    };
}
