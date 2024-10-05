use core::hash::Hash;
use ndarray::{Array1, Array2, Axis};
use num_traits::Float;

use crate::{
    error::NjangError,
    linear_model::{LinearModelSolver, LinearRegression, LinearRegressionSettings},
    traits::{ClassificationModel, Container, Label, Model, RegressionModel, Scalar},
};

use super::{argmax, dummies, unique_labels};

/// Settings used in Ridge classification model.
pub struct RidgeClassificationSettings<T> {
    pub fit_intercept: bool,
    pub solver: LinearModelSolver,
    pub l2_penalty: Option<T>,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

/// Classification with a multi-output Ridge regression.
pub struct RidgeClassification<C, I, L = i32>
where
    C: Container,
{
    pub model: LinearRegression<C, I>,
    pub labels: Vec<L>,
}

impl<'a, T: Scalar, L: Label> Model<'a> for RidgeClassification<Array2<T>, Array1<T>, L> {
    type FitResult = Result<(), NjangError>;
    type Data = (&'a Array2<T>, &'a Array1<L>);
    fn fit(&mut self, data: &Self::Data) -> Self::FitResult {
        let (x, y) = data;
        self.labels = unique_labels((*y).iter()).into_iter().copied().collect();
        let y_reg = dummies(*y, &self.labels);
        RegressionModel::fit(&mut self.model, *x, &y_reg)
    }
}
impl<T: Scalar, L: Label> ClassificationModel for RidgeClassification<Array2<T>, Array1<T>, L> {
    type X = Array2<T>;
    type Y = Array1<L>;
    type PredictResult = Result<Array1<L>, ()>;
    type PredictProbaResult = Result<Array2<T>, ()>;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> <Self as Model<'_>>::FitResult {
        let data = (x, y);
        <Self as Model>::fit(self, &data)
    }
    fn predict(&self, x: &Self::X) -> Self::PredictResult {
        let raw_prediction = self.model.predict(x)?;
        Ok(raw_prediction
            .axis_iter(Axis(0))
            .map(|pred| self.labels[argmax(pred.iter().copied().enumerate()).unwrap()])
            .collect())
    }
    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult {
        let mut raw_prediction = self.model.predict(x)?.map(|x| Float::exp(*x));
        raw_prediction
            .axis_iter_mut(Axis(0))
            .map(|mut row| {
                let norm = row.sum();
                row.iter_mut().for_each(|p| *p = *p / norm);
            })
            .for_each(drop);
        Ok(raw_prediction)
    }
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

#[test]
fn code() {
    use ndarray::array;

    let x = array![[0., 0., 1.], [1., 0., 0.], [1., 0., 1.]];
    let y = ndarray::Array1::from_shape_fn(x.nrows(), core::convert::identity);
    println!("y:\n{:?}", y);

    let settings = RidgeClassificationSettings {
        fit_intercept: false,
        solver: LinearModelSolver::Sag,
        l2_penalty: None,
        tol: Some(1e-6),
        step_size: Some(1e-3),
        random_state: Some(0),
        max_iter: Some(100000),
    };
    let mut model = RidgeClassification::<Array2<_>, _, _>::new(settings);
    match ClassificationModel::fit(&mut model, &x, &y) {
        Ok(_) => {
            println!("{:?}", model.coef());
        }
        Err(error) => {
            println!("{:?}", error);
        }
    };

    assert_eq!(model.predict(&x).unwrap(), y);
}
