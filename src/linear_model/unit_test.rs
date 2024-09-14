mod tests {
    use super::super::*;
    use crate::RegressionModel;
    extern crate alloc;
    use crate::traits::Algebra;
    use alloc::vec::Vec;
    use ndarray::{Array, Array1, Array2, Ix0, Ix1, Ix2};

    const LINEAR_REGRESSION_SOLVERS: [LinearRegressionSolver; 6] = [
        LinearRegressionSolver::SVD,
        LinearRegressionSolver::SGD,
        LinearRegressionSolver::BGD,
        LinearRegressionSolver::QR,
        LinearRegressionSolver::EXACT,
        LinearRegressionSolver::CHOLESKY,
    ];

    const RIDGE_REGRESSION_SOLVERS: [RidgeRegressionSolver; 7] = [
        RidgeRegressionSolver::SVD,
        RidgeRegressionSolver::SAG,
        RidgeRegressionSolver::SGD,
        RidgeRegressionSolver::BGD,
        RidgeRegressionSolver::QR,
        RidgeRegressionSolver::EXACT,
        RidgeRegressionSolver::CHOLESKY,
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
                true_intercept: Option<f32>,
                fitted_intercept: Option<&Array<f32, $ix_smaller>>,
            ) where
                M: RegressionModel<X = Array2<f32>, PredictResult = Result<Array<f32, $ix>, ()>>,
            {
                // println!("{:?}", true_coef);
                // println!("{:?}", fitted_coef);
                println!("{:?}", (fitted_coef - true_coef).l2_norm());
                assert!((fitted_coef - true_coef).l2_norm() < tol);
                if let Some(true_inter) = true_intercept {
                    if let Some(fitted_inter) = fitted_intercept {
                        let error = (fitted_inter - true_inter).map(|x| x.abs()).sum();
                        println!("{:?}", error);
                        assert!(error < tol);
                    }
                }
                println!("\n");
                let pred_error = (model.predict(&x).unwrap() - y).l2_norm();
                assert!(pred_error < tol);
            }
        };
    }
    impl_assert_reg!(assert_one_reg, Ix1, Ix0);
    impl_assert_reg!(assert_multi_reg, Ix2, Ix1);

    macro_rules! impl_test {
        ($assert_name:ident, $test_name:ident, $model_name:ident, $model_settings:ident, $dataset:ident, $solvers:ident, $ix:ty, $ix_smaller:ty) => {
            fn $test_name(intercept: f32, tol: f32) {
                let (x, y, coef) = $dataset(intercept);
                for solver in $solvers {
                    println!("{:?}", solver);
                    let mut model = $model_name::<_, _>::new($model_settings {
                        fit_intercept: intercept.abs() > 0.,
                        solver,
                        max_iter: Some(100000),
                        tol: Some(1e-10),
                        ..Default::default() // default value of penalties should be 0.
                    });
                    let _ = model.fit(&x, &y);
                    let (fitted_coef, fitted_intercept) =
                        (model.coef().unwrap(), model.intercept());
                    $assert_name(
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
        };
    }
    impl_test!(
        assert_one_reg,
        lin_one_reg,
        LinearRegression,
        LinearRegressionSettings,
        one_reg_dataset,
        LINEAR_REGRESSION_SOLVERS,
        Ix1,
        Ix0
    );
    impl_test!(
        assert_one_reg,
        ridge_one_reg,
        RidgeRegression,
        RidgeRegressionSettings,
        one_reg_dataset,
        RIDGE_REGRESSION_SOLVERS,
        Ix1,
        Ix0
    );
    impl_test!(
        assert_multi_reg,
        lin_multi_reg,
        LinearRegression,
        LinearRegressionSettings,
        multi_reg_dataset,
        LINEAR_REGRESSION_SOLVERS,
        Ix2,
        Ix1
    );
    impl_test!(
        assert_multi_reg,
        ridge_multi_reg,
        RidgeRegression,
        RidgeRegressionSettings,
        multi_reg_dataset,
        RIDGE_REGRESSION_SOLVERS,
        Ix2,
        Ix1
    );

    #[test]
    fn test_lin_one_reg_with_intercept() {
        lin_one_reg(3., 1e-3)
    }
    #[test]
    fn test_lin_one_reg_without_intercept() {
        lin_one_reg(0., 1e-3)
    }
    #[test]
    fn test_lin_multi_reg_with_intercept() {
        lin_multi_reg(3., 1e-3)
    }
    #[test]
    fn test_lin_multi_reg_without_intercept() {
        lin_multi_reg(0., 1e-3)
    }
    #[test]
    fn test_ridge_one_reg_with_intercept() {
        ridge_one_reg(3., 1e-3)
    }
    #[test]
    fn test_ridge_one_reg_without_intercept() {
        ridge_one_reg(0., 1e-3)
    }
    #[test]
    fn test_ridge_multi_reg_with_intercept() {
        ridge_multi_reg(3., 1e-3)
    }
    #[test]
    fn test_ridge_multi_reg_without_intercept() {
        ridge_multi_reg(0., 1e-3)
    }
}
