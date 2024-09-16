use num_traits::Float;

use crate::traits::Container;

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
