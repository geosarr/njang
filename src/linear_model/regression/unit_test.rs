use super::super::*;
use crate::linear_model::LinearRegression;
use crate::traits::RegressionModel;
extern crate alloc;
use crate::traits::Algebra;
use alloc::vec::Vec;
use ndarray::{Array, Array0, Array1, Array2, Axis, Ix0, Ix1, Ix2};
use ndarray_linalg::Inverse;

const REGRESSION_SOLVERS: [LinearModelSolver; 7] = [
    LinearModelSolver::Sgd,
    LinearModelSolver::Bgd,
    LinearModelSolver::Sag,
    LinearModelSolver::Svd,
    LinearModelSolver::Qr,
    LinearModelSolver::Exact,
    LinearModelSolver::Cholesky,
];

// from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
#[allow(unused)]
fn predictor() -> Array2<f32> {
    let mut x = Vec::new();
    x.extend_from_slice(&[1f32, 1., 1., 2., 2., 2., 2., 3.]);
    Array2::from_shape_vec((4, 2), x).unwrap()
}
#[allow(unused)]
fn one_reg_dataset(intercept: f32) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
    let x = predictor();
    let coef = Array1::from_iter([1., 2.]);
    // y = 1. * x_0 + 2. * x_1 + intercept.
    let y = x.dot(&coef) + intercept;
    (x, y, coef)
}
#[allow(unused)]
fn multi_reg_dataset(intercept: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let x = predictor();
    let mut coef = Vec::new();
    coef.extend_from_slice(&[1., 2., 3., 4.]);
    let coef = Array2::from_shape_vec((2, 2), coef).unwrap();
    // println!("{:?}", coef);
    // y = 1. * x_0 + 3. * x_1 + intercept for regression 1
    // y = 2. * x_0 + 4. * x_1 + intercept for regression 2
    let y = x.dot(&coef) + intercept;
    (x, y, coef)
}
macro_rules! impl_assert_reg {
    ($assert_name:ident, $ix:ty, $ix_smaller:ty) => {
        fn $assert_name<M>(
            model: &M,
            x: &Array2<f32>,
            y: &Array<f32, $ix>,
            tol: f32,
            true_coef: &Array<f32, $ix>,
            fitted_coef: &Array<f32, $ix>,
            true_intercept: Option<Array<f32, $ix_smaller>>,
            fitted_intercept: Option<&Array<f32, $ix_smaller>>,
            with_prediction: bool,
        ) where
            M: RegressionModel<X = Array2<f32>, PredictResult = Result<Array<f32, $ix>, ()>>,
        {
            // println!("true coef:\n{:?}", true_coef);
            // println!("fitted coef:\n{:?}", fitted_coef);
            // println!("{:?}", (fitted_coef - true_coef).l2_norm());
            assert!((fitted_coef - true_coef).l2_norm() < tol);
            if let Some(true_inter) = true_intercept {
                if let Some(fitted_inter) = fitted_intercept {
                    let error = (fitted_inter - true_inter).map(|x| x.abs()).sum(); // println!("{:?}", error);
                    assert!(error < tol);
                }
            }
            // println!("\n");
            if with_prediction {
                let pred_error = (model.predict(&x).unwrap() - y).l2_norm();
                assert!(pred_error < tol);
            }
        }
    };
}
impl_assert_reg!(assert_one_reg, Ix1, Ix0);
impl_assert_reg!(assert_multi_reg, Ix2, Ix1);

macro_rules! impl_test {
    ($assert_name:ident, $test_name:ident, $model_name:ident,
$model_settings:ident, $dataset:ident, $solvers:ident, $ix:ty,
$ix_smaller:ty) => {
        fn $test_name(
            intercept: f32,
            tol: f32,
            l1: Option<f32>,
            l2: Option<f32>,
            true_coef: Option<Array<f32, $ix>>,
            true_intercept: Option<Array<f32, $ix_smaller>>,
            with_prediction: bool,
        ) {
            let (x, y, coef) = $dataset(intercept);
            let coef = if let Some(_coef) = true_coef {
                _coef
            } else {
                coef
            };
            for solver in $solvers {
                // println!("{:?}", solver);
                let mut model = $model_name::<_, _>::new($model_settings {
                    fit_intercept: intercept.abs() > 0.,
                    solver,
                    max_iter: Some(100000),
                    tol: Some(1e-10),
                    l1_penalty: l1,
                    l2_penalty: l2,
                    ..Default::default()
                });
                // println!("{:?}", model);
                let _ = model.fit(&x, &y);
                let (fitted_coef, fitted_intercept) = (model.coef().unwrap(), model.intercept());
                $assert_name(
                    &model,
                    &x,
                    &y,
                    tol,
                    &coef,
                    fitted_coef,
                    true_intercept.clone(),
                    fitted_intercept,
                    with_prediction,
                );
            }
        }
    };
}
impl_test!(
    assert_one_reg,
    one_reg,
    LinearRegression,
    LinearRegressionSettings,
    one_reg_dataset,
    REGRESSION_SOLVERS,
    Ix1,
    Ix0
);
impl_test!(
    assert_multi_reg,
    multi_reg,
    LinearRegression,
    LinearRegressionSettings,
    multi_reg_dataset,
    REGRESSION_SOLVERS,
    Ix2,
    Ix1
);
// Test Linear regression without penalties.
#[test]
fn test_lin_one_reg_with_intercept() {
    let intercept = Array0::from_elem((), 3.);
    one_reg(
        intercept.sum(),
        1e-3,
        None,
        None,
        None,
        Some(intercept),
        true,
    )
}
#[test]
fn test_lin_one_reg_without_intercept() {
    let intercept = Array0::from_elem((), 0.);
    one_reg(intercept.sum(), 1e-3, None, None, None, None, true)
}
#[test]
fn test_lin_multi_reg_with_intercept() {
    let intercept = Array1::from_elem(1, 3.);
    multi_reg(
        intercept.sum(),
        5e-3,
        None,
        None,
        None,
        Some(intercept),
        true,
    )
}
#[test]
fn test_lin_multi_reg_without_intercept() {
    multi_reg(0., 5e-3, None, None, None, None, true)
}

// Test Ridge regression.
#[test]
fn test_ridge_reg_with_intercept() {
    let intercept = 1.;
    let penalty = 1.;
    let (x, y, _) = one_reg_dataset(intercept);
    let xm = x.mean_axis(Axis(0)).unwrap();
    let ym = y.mean_axis(Axis(0)).unwrap();
    let (x, y) = (x - &xm, y - &ym);
    let xt = x.t();
    let right = xt.dot(&y);
    let expected_coef = (xt.dot(&x) + Array2::<f32>::eye(x.ncols()) * penalty)
        .inv()
        .unwrap()
        .dot(&right);
    let expected_intercept = ym - xm.dot(&expected_coef);
    one_reg(
        intercept,
        1e-2,
        None,
        Some(penalty),
        Some(expected_coef),
        Some(expected_intercept),
        false,
    );
    let (x, y, _) = multi_reg_dataset(intercept);
    let xm = x.mean_axis(Axis(0)).unwrap();
    let ym = y.mean_axis(Axis(0)).unwrap();
    let (x, y) = (x - &xm, y - &ym);
    let xt = x.t();
    let right = xt.dot(&y);
    let expected_coef = (xt.dot(&x) + Array2::<f32>::eye(x.ncols()) * penalty)
        .inv()
        .unwrap()
        .dot(&right);
    let expected_intercept = ym - xm.dot(&expected_coef);
    multi_reg(
        intercept,
        5e-2,
        None,
        Some(penalty),
        Some(expected_coef),
        Some(expected_intercept),
        false,
    );
}
#[test]
fn test_ridge_reg_without_intercept() {
    let intercept = 0.;
    let penalty = 1.;
    let (x, y, _) = one_reg_dataset(intercept);
    let xt = x.t();
    let right = xt.dot(&y);
    let expected_coef = (xt.dot(&x) + Array2::<f32>::eye(x.ncols()) * penalty)
        .inv()
        .unwrap()
        .dot(&right);
    one_reg(
        intercept,
        5e-3,
        None,
        Some(penalty),
        Some(expected_coef),
        None,
        false,
    );
    let (x, y, _) = multi_reg_dataset(intercept);
    let xt = x.t();
    let right = xt.dot(&y);
    let expected_coef = (xt.dot(&x) + Array2::<f32>::eye(x.ncols()) * penalty)
        .inv()
        .unwrap()
        .dot(&right);
    multi_reg(
        intercept,
        5e-2,
        None,
        Some(penalty),
        Some(expected_coef),
        None,
        false,
    );
}

#[test]
fn test_lin_reg_with_intercept_wide_dataset() {
    let x = Array2::from_shape_vec((2, 3), Vec::from_iter([0., 0., 1., 1., 0., 0.])).unwrap();
    let xxt = x.dot(&x.t());
    let coef = Array2::from_shape_vec((3, 2), Vec::from_iter([1., 1., 2., 2., 3., 3.])).unwrap();
    let y = x.dot(&coef);
    let expected_coef = x.t().dot(&xxt).dot(&y);
    for solver in [
        LinearModelSolver::Svd,
        LinearModelSolver::Qr,
        LinearModelSolver::Exact,
        LinearModelSolver::Cholesky,
    ] {
        println!("\n{:?}", solver);
        let settings = LinearRegressionSettings {
            fit_intercept: false,
            solver: solver,
            l1_penalty: None,
            l2_penalty: None,
            ..Default::default()
        };
        let mut model = LinearRegression::<Array2<_>, _>::new(settings);
        match model.fit(&x, &y) {
            Ok(_) => {
                assert!((expected_coef.clone() - model.coef().unwrap()).l2_norm() < 1e-5)
            }
            Err(_) => panic!("Should not be an error"),
        };
    }
}
