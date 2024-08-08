mod tests {
    use super::super::*;
    use crate::RegressionModel;
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
        let mut lin_reg = LinearRegression::<Array1<_>, _>::new(true, LinearRegressionSolver::Svd);
        let _ = lin_reg.fit(&x, &y);
        let coef_star = lin_reg.coef().unwrap();
        let intercept_star = lin_reg.intercept().unwrap();
        assert!((coef_star[0] - coef[0]).abs() < 1e-6);
        assert!((coef_star[1] - coef[1]).abs() < 1e-6);
        println!("{:?}", intercept_star - intercept);
    }
}
