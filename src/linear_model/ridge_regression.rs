use ndarray::{linalg::Dot, Array, Array1, Array2, Ix0, Ix1, Ix2, ScalarOperand};
use ndarray_rand::rand::Rng;

use crate::RegressionModel;
use crate::{linear_model::preprocess, traits::Info};
use core::ops::Mul;
use ndarray_linalg::{error::LinalgError, Inverse, Lapack};
use ndarray_rand::{
    rand::{distributions::Distribution, SeedableRng},
    rand_distr::{StandardNormal, Uniform},
    RandomExt,
};
use num_traits::FromPrimitive;
use rand_chacha::ChaCha20Rng;

/// Solver to use when fitting a ridge regression model (L2-penalty with Ordinary Least Squares).
#[derive(Debug, Default)]
pub enum RidgeRegressionSolver {
    /// Solves using stochastic gradient descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm may not converge.
    #[default]
    Sgd,
    /// Computing the exaction solution (x.t().dot(x) + alpha * eye).inverse().dot(x.t()).dot(y)
    Exact,
}

/// Hyperparameters used in a Ridge regression.
#[derive(Debug)]
pub struct RidgeRegressionHyperParameter<T> {
    pub alpha: T,
    pub fit_intercept: bool,
    pub solver: RidgeRegressionSolver,
    pub tol: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

impl<T> Default for RidgeRegressionHyperParameter<T>
where
    T: Default + FromPrimitive,
{
    fn default() -> Self {
        Self {
            alpha: T::from_f32(1.).unwrap(),
            fit_intercept: true,
            solver: Default::default(),
            tol: Some(T::from_f32(0.0001).unwrap()),
            random_state: Some(0),
            max_iter: Some(1000),
        }
    }
}
impl<T> RidgeRegressionHyperParameter<T> {
    pub fn new_exact(alpha: T, fit_intercept: bool) -> Self {
        Self {
            alpha,
            fit_intercept,
            solver: RidgeRegressionSolver::Exact,
            tol: None,
            random_state: None,
            max_iter: None,
        }
    }
}

#[derive(Debug)]
pub struct RidgeRegression<C, I, T = f32> {
    coef: Option<C>,
    intercept: Option<I>,
    settings: RidgeRegressionHyperParameter<T>,
}

impl<C, I, T> RidgeRegression<C, I, T> {
    pub fn new(settings: RidgeRegressionHyperParameter<T>) -> Self {
        Self {
            coef: None,
            intercept: None,
            settings,
        }
    }
    /// Coefficients of the model
    pub fn coef(&self) -> Option<&C> {
        self.coef.as_ref()
    }
    /// Intercept of the model
    pub fn intercept(&self) -> Option<&I> {
        self.intercept.as_ref()
    }
}

macro_rules! impl_ridge_reg {
    ($ix:ty, $ix_smaller:ty,  $randn:ident) => {
        impl<T> RegressionModel for RidgeRegression<Array<T, $ix>, Array<T, $ix_smaller>, T>
        where
            for<'a> T: Lapack
                + ScalarOperand
                + FromPrimitive
                + Mul<&'a Array<T, $ix>, Output = Array<T, $ix>>
                + Mul<Array1<T>, Output = Array1<T>>
                + Mul<Array2<T>, Output = Array2<T>>,
            StandardNormal: Distribution<T>,
            Array2<T>: Dot<Array2<T>, Output = Array2<T>>
                + Inverse<Output = Array2<T>>
                + Dot<Array1<T>, Output = Array1<T>>
                + Info<MeanOutput = Array1<T>, RowOutput = Array1<T>>,
            Array<T, Ix1>: Info<MeanOutput = Array<T, Ix0>, RowOutput = T>,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Option<Array<T, $ix>>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                if self.settings.fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(x, y);
                    let coef = match self.settings.solver {
                        RidgeRegressionSolver::Sgd => {
                            let mut rng = ChaCha20Rng::seed_from_u64(
                                self.settings.random_state.unwrap_or(0).into(),
                            );
                            let mut coef = $randn(x.ncols(), y.shape(), &mut rng);
                            let max_iter = self.settings.max_iter.unwrap_or(1000);
                            let n = x.nrows();
                            let nf = T::from_f32(n as f32).unwrap();
                            let lambda = T::from_f32(0.001).unwrap(); // to determine automatically.
                            let samples = Array::<usize, _>::random_using(
                                max_iter,
                                Uniform::from(0..n),
                                &mut rng,
                            );
                            let alpha_norm = self.settings.alpha / nf;
                            for k in 0..max_iter {
                                let i = samples[k];
                                let xi = x_centered.get_row(i);
                                let yi = y_centered.get_row(i);
                                let g_cost = alpha_norm * &coef + (xi.dot(&coef) - yi) * xi;
                                coef = &coef - lambda * g_cost;
                            }
                            coef
                        }
                        RidgeRegressionSolver::Exact => {
                            let xct = x_centered.t();
                            match (xct.dot(&x_centered)
                                + self.settings.alpha * Array2::eye(x.ncols()))
                            .inv()
                            {
                                Ok(mat) => mat.dot(&xct).dot(&y_centered),
                                Err(error) => return Err(error),
                            }
                        }
                    };
                    self.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.coef = Some(coef);
                } else {
                    match self.settings.solver {
                        RidgeRegressionSolver::Sgd => {}
                        RidgeRegressionSolver::Exact => {
                            let xt = x.t();
                            self.coef = Some(
                                match (xt.dot(x) + self.settings.alpha * Array2::eye(x.ncols()))
                                    .inv()
                                {
                                    Ok(mat) => mat.dot(&xt).dot(y),
                                    Err(error) => return Err(error),
                                },
                            );
                        }
                    }
                }
                Ok(())
            }
            fn predict(&self, x: &Self::X) -> Self::PredictResult {
                if self.settings.fit_intercept {
                    if let Some(ref coef) = &self.coef {
                        if let Some(ref intercept) = &self.intercept {
                            return Some(intercept + x.dot(coef));
                        }
                    }
                } else {
                    if let Some(ref coef) = &self.coef {
                        return Some(x.dot(coef));
                    }
                }
                None
            }
        }
    };
}
impl_ridge_reg!(Ix1, Ix0, randn_1d);
impl_ridge_reg!(Ix2, Ix1, randn_2d);

// fn row_dup_1reg<T>(row: Array<T, Ix1>, d: usize) -> Array<T, Ix1> {
//     row
// }

// fn row_dup_dreg<T>(row: Array<T, Ix1>, d: usize) -> Array<T, Ix2> {
//     row
// }

fn randn_1d<T, R: Rng>(n: usize, _m: &[usize], rng: &mut R) -> Array<T, Ix1>
where
    StandardNormal: Distribution<T>,
{
    return Array::<T, Ix1>::random_using(n, StandardNormal, rng);
}

fn randn_2d<T, R: Rng>(n: usize, m: &[usize], rng: &mut R) -> Array<T, Ix2>
where
    StandardNormal: Distribution<T>,
{
    return Array::<T, Ix2>::random_using((n, m[1]), StandardNormal, rng);
}
