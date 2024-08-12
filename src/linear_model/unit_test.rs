mod tests {
    use super::super::*;
    use crate::RegressionModel;
    extern crate alloc;
    use crate::{l2_diff, l2_diff2};
    use alloc::vec::Vec;
    use ndarray::{Array, Array0, Array1, Array2, Ix0, Ix1, Ix2};

    // from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
    fn predictor() -> Array2<f32> {
        let mut x = Vec::new();
        x.extend_from_slice(&[1f32, 1., 1., 2., 2., 2., 2., 3.]);
        Array2::from_shape_vec((4, 2), x).unwrap()
    }
    fn one_reg_dataset(intercept: f32) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        let x = predictor();
        let coef = Array1::from_iter([1., 2.]);
        // y = 1. * x_0 + 2. * x_1 + intercept.
        let y = x.dot(&coef) + intercept;
        (x, y, coef)
    }
    fn multi_reg_dataset(intercept: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let x = predictor();
        let mut coef = Vec::new();
        coef.extend_from_slice(&[1., 2., 3., 4.]);
        let coef = Array2::from_shape_vec((2, 2), coef).unwrap();
        // y = 1. * x_0 + 3. * x_1 + intercept for regression 1
        // y = 2. * x_0 + 4. * x_1 + intercept for regression 2
        let y = x.dot(&coef) + intercept;
        (x, y, coef)
    }

    macro_rules! impl_assert_reg {
        ($name:ident, $l2_norm:ident, $ix:ty, $ix_smaller:ty) => {
            fn $name<M>(
                model: &M,
                x: &Array2<f32>,
                y: &Array<f32, $ix>,
                tol: f32,
                true_coef: &Array<f32, $ix>,
                fitted_coef: &Array<f32, $ix>,
                true_intercept: Option<f32>,
                fitted_intercept: Option<&Array<f32, $ix_smaller>>,
            ) where
                M: RegressionModel<X = Array2<f32>, PredictResult = Option<Array<f32, $ix>>>,
            {
                assert!($l2_norm(fitted_coef, true_coef) < tol);
                if let Some(true_inter) = true_intercept {
                    if let Some(fitted_inter) = fitted_intercept {
                        assert!((fitted_inter - true_inter).map(|x| x.abs()).sum() < tol);
                    }
                }
                let pred_error = $l2_norm(&model.predict(&x).unwrap(), &y);
                assert!(pred_error < tol);
            }
        };
    }
    impl_assert_reg!(assert_one_reg, l2_diff, Ix1, Ix0);
    impl_assert_reg!(assert_multi_reg, l2_diff2, Ix2, Ix1);

    fn lin_one_reg(intercept: f32, tol: f32) {
        let (x, y, coef) = one_reg_dataset(intercept);
        let solvers = [
            LinearRegressionSolver::Svd,
            LinearRegressionSolver::Qr,
            LinearRegressionSolver::Exact,
        ];
        for solver in solvers {
            let mut model =
                LinearRegression::<Array1<_>, Array0<_>>::new(LinearRegressionHyperParameter {
                    fit_intercept: intercept.abs() > 0.,
                    solver,
                });
            let _ = model.fit(&x, &y);
            let (fitted_coef, fitted_intercept) = (model.coef().unwrap(), model.intercept());
            assert_one_reg(
                &model,
                &x,
                &y,
                tol,
                &coef,
                fitted_coef,
                Some(intercept),
                fitted_intercept,
            );
        }
    }
    #[test]
    fn test_lin_one_reg_with_intercept() {
        lin_one_reg(3., 1e-6)
    }
    #[test]
    fn test_lin_one_reg_without_intercept() {
        lin_one_reg(0., 2e-5)
    }

    fn lin_multi_reg(intercept: f32, tol: f32) {
        let (x, y, coef) = multi_reg_dataset(intercept);
        let solvers = [
            LinearRegressionSolver::Svd,
            LinearRegressionSolver::Qr,
            LinearRegressionSolver::Exact,
        ];
        for solver in solvers {
            let mut model =
                LinearRegression::<Array2<_>, Array1<_>>::new(LinearRegressionHyperParameter {
                    fit_intercept: intercept.abs() > 0.,
                    solver,
                });
            let _ = model.fit(&x, &y);
            let (fitted_coef, fitted_intercept) = (model.coef().unwrap(), model.intercept());
            assert_multi_reg(
                &model,
                &x,
                &y,
                tol,
                &coef,
                fitted_coef,
                Some(intercept),
                fitted_intercept,
            );
        }
    }

    #[test]
    fn test_lin_multi_reg_with_intercept() {
        lin_multi_reg(3., 1e-5)
    }
    #[test]
    fn test_lin_multi_reg_without_intercept() {
        lin_multi_reg(0., 6e-5)
    }

    fn ridge_one_reg(intercept: f32, tol: f32) {
        let (x, y, coef) = one_reg_dataset(intercept);
        let solvers = [RidgeRegressionSolver::Exact, RidgeRegressionSolver::Sgd];
        for solver in solvers {
            let mut model =
                RidgeRegression::<Array1<_>, Array0<_>>::new(RidgeRegressionHyperParameter {
                    // Some attributes are not needed for Exact solver
                    alpha: 0.,
                    tol: Some(0.01),
                    solver: solver,
                    fit_intercept: intercept.abs() > 0.,
                    random_state: None,
                    max_iter: Some(100000),
                });
            let _ = model.fit(&x, &y);
            let (fitted_coef, fitted_intercept) = (model.coef().unwrap(), model.intercept());
            assert_one_reg(
                &model,
                &x,
                &y,
                tol,
                &coef,
                fitted_coef,
                Some(intercept),
                fitted_intercept,
            );
        }
    }
    #[test]
    fn test_ridge_one_reg_with_intercept() {
        ridge_one_reg(3., 5e-4)
    }
    #[test]
    fn test_ridge_one_reg_without_intercept() {
        ridge_one_reg(0., 2e-4)
    }

    #[test]
    fn test_ridge_multi_reg() {
        let intercept = 0.;
        let (x, y, coef) = multi_reg_dataset(intercept);
        let mut ridge = RidgeRegression::<Array2<_>, _>::new(
            RidgeRegressionHyperParameter::new_exact(0., intercept.abs() > 0.),
        );
        let _ = ridge.fit(&x, &y);
        let (fitted_coef, fitted_intercept) = (ridge.coef().unwrap(), ridge.intercept());
        assert_multi_reg(
            &ridge,
            &x,
            &y,
            1e-4,
            &coef,
            fitted_coef,
            Some(intercept),
            fitted_intercept,
        );
        let mut ridge = RidgeRegression::<Array2<_>, _>::new(RidgeRegressionHyperParameter {
            // Some attributes are not needed for Exact solver
            alpha: 0.,
            tol: Some(0.01),
            solver: RidgeRegressionSolver::Sgd,
            fit_intercept: intercept.abs() > 0.,
            random_state: None,
            max_iter: Some(100000),
        });
        let _ = ridge.fit(&x, &y);
        assert!(l2_diff2(&coef, &ridge.coef().unwrap()) < 5e-3);
    }
}
