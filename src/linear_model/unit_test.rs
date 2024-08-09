mod tests {
    use ridge_regression::RidgeRegressionHyperParameter;

    use super::super::*;
    use crate::RegressionModel;
    extern crate alloc;
    use alloc::vec::Vec;

    #[test]
    fn test_lin_reg() {
        // from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
        use ndarray::{Array1, Array2};
        let mut x = Vec::new();
        x.extend_from_slice(&[1f32, 1., 1., 2., 2., 2., 2., 3.]);
        let x = Array2::from_shape_vec((4, 2), x).unwrap();
        let coef = Array1::from_iter([1., 2.]); // Array2::from_shape_vec((2, 2), vec![1., 2., 1., 2.]).unwrap(); //
        let intercept = 3.;
        let y = x.dot(&coef) + intercept; // y = 1. * x_0 + 2. * x_1 + 3.
        let mut lin_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionHyperParameter {
            fit_intercept: true,
            solver: LinearRegressionSolver::Svd,
        });
        let _ = lin_reg.fit(&x, &y);
        let coef_star = lin_reg.coef().unwrap();
        let intercept_star = lin_reg.intercept().unwrap();
        assert!((coef_star[0] - coef[0]).abs() < 1e-6);
        assert!((coef_star[1] - coef[1]).abs() < 1e-6);
        let pred_error = lin_reg
            .predict(&x)
            .unwrap()
            .into_iter()
            .zip(&y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(pred_error < 1e-6);
        // println!("{:?}", intercept_star - intercept);
    }

    #[test]
    fn test_ridge_reg() {
        use ndarray::{Array1, Array2, Axis};
        let mut x = Vec::new();
        x.extend_from_slice(&[1f32, 1., 1., 2., 2., 2., 2., 3.]);
        let x = Array2::from_shape_vec((4, 2), x).unwrap();
        let mut c = Vec::new();
        c.extend_from_slice(&[1., 2., 1., 2.]);
        let coef = Array1::from_iter([1., 2.]); // Array2::from_shape_vec((2, 2), c).unwrap(); //
        let intercept = 0.;
        let y = x.dot(&coef) + intercept; // y = 1. * x_0 + 2. * x_1 + 3.
        let mut ridge_sgd = RidgeRegression::<Array1<_>, _>::new(RidgeRegressionHyperParameter {
            alpha: 0.,
            tol: Some(0.01),
            solver: RidgeRegressionSolver::Sgd,
            fit_intercept: false,
            random_state: None,
            max_iter: Some(1000000),
        });
        let _ = ridge_sgd.fit(&x, &y);
        let mut ridge_exact = RidgeRegression::<Array1<_>, _>::new(
            RidgeRegressionHyperParameter::new_exact(0., false),
        );
        let _ = ridge_exact.fit(&x, &y);
        println!("{:?}", ridge_exact.coef());
        // println!("{:?}", ridge_exact.intercept());
        // println!("{:?}", ridge_sgd.coef());
        // println!("{:?}", ridge_sgd.intercept());
    }
    #[test]
    fn test_() {
        use ndarray::Array1;
        let mut c = Vec::new();
        c.extend_from_slice(&[1., 2., 1., 2.]);
        println!("{:?}", Array2::from_shape_vec((2, 2), c).unwrap().shape());
    }
}
