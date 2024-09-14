use crate::l2_norm1;
use crate::linear_model::{preprocess, randn_1d, randn_2d, square_loss_gradient};
use crate::traits::{Algebra, Container, Scalar};
use crate::RegressionModel;
use crate::{solver::stochastic_gradient_descent, traits::Info};
use ndarray::{ArrayView2, ScalarOperand};
use ndarray_linalg::Lapack;
use ndarray_rand::rand_distr::uniform::SampleUniform;

#[allow(unused)]
use core::{
    marker::{Send, Sync},
    ops::{Add, Div, Mul, Sub},
};
use ndarray::{linalg::Dot, Array, Array1, Array2, Axis, Ix0, Ix1, Ix2};
use ndarray_linalg::{error::LinalgError, LeastSquaresSvd};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use num_traits::{Float, FromPrimitive, Zero};
use rand_chacha::ChaCha20Rng;

/// Solver to use when fitting a lasso regression model (L2-penalty with
/// Ordinary Least Squares).
///
/// Here `alpha` is the magnitude of the penalty and `eye` is the identity
/// matrix.
#[derive(Debug, Default, Clone)]
pub enum LassoRegressionSolver {
    /// Solves the problem using Stochastic Gradient Descent
    ///
    /// Make sure to standardize the input predictors, otherwise the algorithm
    /// may not converge.
    #[default]
    SGD,
}

/// Hyperparameters used in a lasso regression.
///
/// - **alpha**: L2-norm penalty magnitude.
/// - **fit_intercept**: `true` means fit with an intercept, `false` without an
///   intercept.
/// - **solver**: optimization method see [`LassoRegressionSolver`].
/// - **tol**: tolerance parameter:
///     - stochastic optimization solvers (like SGD) stop when the relative
///       variation of consecutive iterates is lower than **tol**:
///         - `||coef_next - coef_curr|| <= tol * ||coef_curr||`
///     - No impact on the other algorithms.
/// - **random_state**: seed of random generators.
/// - **max_iter**: maximum number of iterations.
#[derive(Debug, Clone)]
pub struct LassoRegressionSettings<T> {
    pub alpha: T,
    pub fit_intercept: bool,
    pub solver: LassoRegressionSolver,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub random_state: Option<u32>,
    pub max_iter: Option<usize>,
}

impl<T> Default for LassoRegressionSettings<T>
where
    T: Default + FromPrimitive,
{
    fn default() -> Self {
        Self {
            alpha: T::from_f32(0.).unwrap(),
            fit_intercept: true,
            solver: Default::default(),
            tol: Some(T::from_f32(0.0001).unwrap()),
            step_size: Some(T::from_f32(0.001).unwrap()),
            random_state: Some(0),
            max_iter: Some(1000),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LassoRegressionParameter<C, I> {
    pub coef: Option<C>,
    pub intercept: Option<I>,
}

/// L2-norm penalized Ordinary Least Squares.
#[derive(Debug)]
pub struct LassoRegression<C, I>
where
    C: Container,
{
    pub parameter: LassoRegressionParameter<C, I>,
    pub settings: LassoRegressionSettings<C::Elem>,
    internal: LassoRegressionInternal<C::Elem>,
}

#[derive(Debug, Clone)]
pub(crate) struct LassoRegressionInternal<T> {
    pub n_samples: usize,
    pub n_features: usize,
    pub n_targets: usize,
    pub alpha: T,
    pub tol: Option<T>,
    pub step_size: Option<T>,
    pub rng: Option<ChaCha20Rng>,
    pub max_iter: Option<usize>,
}

impl<T: Zero> LassoRegressionInternal<T> {
    pub fn new() -> Self {
        Self {
            n_samples: 0,
            n_features: 0,
            n_targets: 0,
            alpha: T::zero(),
            tol: None,
            step_size: None,
            rng: None,
            max_iter: None,
        }
    }
}

impl<C: Container, I> LassoRegression<C, I> {
    /// Creates a new instance of `Self`.
    ///
    /// See also: [LassoRegressionSettings], [LassoRegressionSolver],
    /// [RegressionModel].
    /// ```
    /// use ndarray::{array, Array0, Array1};
    /// use njang::{LassoRegression, LassoRegressionSettings, LassoRegressionSolver, RegressionModel};
    /// // Initial model
    /// let mut model = LassoRegression::<Array1<f32>, Array0<f32>>::new(LassoRegressionSettings {
    ///     alpha: 0.01,
    ///     tol: Some(0.0001),
    ///     solver: LassoRegressionSolver::SGD,
    ///     fit_intercept: true,
    ///     random_state: Some(123),
    ///     max_iter: Some(1),
    ///     step_size: None,
    /// });
    /// // Dataset
    /// let x0 = array![[1., 2.], [-3., -4.], [0., 7.], [-2., 5.]];
    /// let y0 = array![0.5, -1., 2., 3.5];
    /// model.fit(&x0, &y0);
    /// ```
    pub fn new(settings: LassoRegressionSettings<C::Elem>) -> Self
    where
        C::Elem: Zero,
    {
        Self {
            parameter: LassoRegressionParameter {
                coef: None,
                intercept: None,
            },
            settings,
            internal: LassoRegressionInternal::new(),
        }
    }
    /// Coefficients of the model
    pub fn coef(&self) -> Option<&C> {
        self.parameter.coef.as_ref()
    }
    /// Intercept of the model
    pub fn intercept(&self) -> Option<&I> {
        self.parameter.intercept.as_ref()
    }
    fn set_internal<X: Container>(&mut self, x: &X, y: &C)
    where
        C::Elem: Copy + FromPrimitive,
    {
        self.set_sample_to_internal(x, y);
        self.set_settings_to_internal();
    }
    fn set_sample_to_internal<X: Container>(&mut self, x: &X, y: &C) {
        // x should be a 2 dimensional container, and <X as Container>::dimension should
        // return something like [nrows, ncols] of X.
        let dim = x.dimension();
        let (n_samples, n_features) = (dim[0], dim[1]);
        self.internal.n_features = n_features;
        self.internal.n_samples = n_samples;
        let shape = y.dimension();
        let n_targets = if shape.len() == 2 { shape[1] } else { 1 };
        self.internal.n_targets = n_targets;
    }
    fn set_settings_to_internal(&mut self)
    where
        C::Elem: Copy + FromPrimitive,
    {
        self.internal.tol = Some(
            self.settings
                .tol
                .unwrap_or(C::Elem::from_f32(1e-4).unwrap()),
        );
        self.internal.step_size = Some(
            self.settings
                .step_size
                .unwrap_or(C::Elem::from_f32(1e-3).unwrap()),
        );
        let random_state = self.settings.random_state.unwrap_or(0);
        self.internal.rng = Some(ChaCha20Rng::seed_from_u64(random_state as u64));
        self.internal.max_iter = Some(self.settings.max_iter.unwrap_or(1000));
    }
}

macro_rules! impl_lasso_reg {
    ($ix:ty, $ix_smaller:ty, $randn:ident) => {
        impl<T> RegressionModel for LassoRegression<Array<T, $ix>, Array<T, $ix_smaller>>
        where
            T: Lapack + ScalarOperand + PartialOrd + Float + SampleUniform,
        {
            type FitResult = Result<(), LinalgError>;
            type X = Array2<T>;
            type Y = Array<T, $ix>;
            type PredictResult = Result<Array<T, $ix>, ()>;
            fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
                if self.settings.fit_intercept {
                    let (x_centered, x_mean, y_centered, y_mean) = preprocess(x, y);
                    let coef = match self.settings.solver {
                        LassoRegressionSolver::SGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, n_samples, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.n_samples,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            let n_targets_in_t = T::from(n_targets).unwrap();
                            let n_samples_in_t = T::from(n_samples).unwrap();
                            // Rescale step_size and l2 penalty coefficient to scale gradients
                            // correctly [Specific to this algorithm]
                            self.internal
                                .step_size
                                .as_mut()
                                .map(|s| *s = *s / n_targets_in_t);
                            self.internal.alpha *= n_targets_in_t / n_samples_in_t;
                            let coef = $randn(&[n_features, n_targets], rng);
                            stochastic_gradient_descent(
                                &x_centered,
                                &y_centered,
                                coef,
                                lasso_regression_gradient,
                                &self.internal,
                            )
                        }
                    };
                    self.parameter.intercept = Some(y_mean - x_mean.dot(&coef));
                    self.parameter.coef = Some(coef);
                } else {
                    let coef = match self.settings.solver {
                        LassoRegressionSolver::SGD => {
                            self.set_internal(x, y);
                            let (n_targets, n_features, n_samples, rng) = (
                                self.internal.n_targets,
                                self.internal.n_features,
                                self.internal.n_samples,
                                self.internal.rng.as_mut().unwrap(),
                            );
                            let n_targets_in_t = T::from(n_targets).unwrap();
                            let n_samples_in_t = T::from(n_samples).unwrap();
                            // Rescale step_size and l2 penalty coefficient to scale gradients
                            // correctly [Specific to this algorithm]
                            self.internal
                                .step_size
                                .as_mut()
                                .map(|s| *s = *s / n_targets_in_t);
                            self.internal.alpha *= n_targets_in_t / n_samples_in_t;
                            let coef = $randn(&[n_features, n_targets], rng);
                            stochastic_gradient_descent(
                                x,
                                y,
                                coef,
                                lasso_regression_gradient,
                                &self.internal,
                            )
                        }
                    };
                    self.parameter.coef = Some(coef);
                }
                Ok(())
            }
            fn predict(&self, x: &Self::X) -> Self::PredictResult {
                if self.settings.fit_intercept {
                    if let Some(ref coef) = &self.parameter.coef {
                        if let Some(ref intercept) = &self.parameter.intercept {
                            return Ok(intercept + x.dot(coef));
                        }
                    }
                } else {
                    if let Some(ref coef) = &self.parameter.coef {
                        return Ok(x.dot(coef));
                    }
                }
                Err(())
            }
        }
    };
}

impl_lasso_reg!(Ix1, Ix0, randn_1d);
impl_lasso_reg!(Ix2, Ix1, randn_2d);

fn lasso_regression_gradient<T: Lapack, Y>(
    x: &Array2<T>,
    y: &Y,
    coef: &Y,
    settings: &LassoRegressionInternal<T>,
) -> Y
where
    for<'a> Y: Sub<&'a Y, Output = Y> + Add<Y, Output = Y> + Mul<T, Output = Y> + Algebra<Elem = T>,
    for<'a> &'a Y: Mul<T, Output = Y>,
    Array2<T>: Dot<Y, Output = Y>,
    for<'a> ArrayView2<'a, T>: Dot<Y, Output = Y>,
{
    let step_size = settings.step_size.unwrap();
    let l1_penalty = settings.alpha;
    return (square_loss_gradient(x, y, coef) + coef.sign() * l1_penalty) * (-step_size);
}

#[test]
fn code() {
    use ndarray::*;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use rand_chacha::ChaCha20Rng;
    let settings = LassoRegressionSettings {
        alpha: 0.1,
        fit_intercept: true,
        max_iter: Some(10000),
        solver: LassoRegressionSolver::SGD,
        tol: Some(1e-20),
        random_state: Some(0),
        step_size: Some(1e-3),
    };
    let mut model: LassoRegression<Array1<f32>, Array0<f32>> = LassoRegression {
        parameter: LassoRegressionParameter {
            coef: None,
            intercept: None,
        },
        settings: settings.clone(),
        internal: LassoRegressionInternal::new(),
    };
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let p = 100;
    let x = Array::<f32, Ix2>::random_using((10, p), StandardNormal, &mut rng);
    let coef = Array1::from((1..p + 1).map(|val| val as f32).collect::<Vec<_>>());
    let y = x.dot(&coef);
    let _ = model.fit(&x, &y);
    println!("coef:\n{:?}", model.coef());
    println!("intercept:\n{:?}\n\n", model.intercept());

    let mut model: LassoRegression<Array2<f32>, Array1<f32>> = LassoRegression {
        parameter: LassoRegressionParameter {
            coef: None,
            intercept: None,
        },
        internal: LassoRegressionInternal::new(),
        settings: settings,
    };

    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let p = 10;
    let x = Array::<f32, Ix2>::random_using((100000, p), StandardNormal, &mut rng);
    let r = 10;
    // let x = (&x - x.mean_axis(Axis(0)).unwrap()) / x.std_axis(Axis(0), 0.);
    let coef = Array2::from_shape_vec(
        (p, r),
        (1..p * r + 1).map(|val| val as f32).collect::<Vec<_>>(),
    )
    .unwrap();
    let intercept = Array1::from_iter((1..r + 1).map(|val| val as f32));
    let y = x.dot(&coef) + intercept;
    let _ = model.fit(&x, &y);
    println!("coef:\n{:?}", model.coef());
    println!("intercept:\n{:?}\n\n", model.intercept());
}
