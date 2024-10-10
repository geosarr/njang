#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use crate::linear_model::classification::{RidgeClassification, RidgeClassificationSettings};
    use crate::prelude::{ClassificationModel, LinearModelSolver};

    #[test]
    fn test_ridge_classifier() {
        let x = array![[0., 0., 1.], [1., 0., 0.], [1., 0., 1.]];
        let y = array!["A", "B", "C"];

        let settings = RidgeClassificationSettings {
            fit_intercept: false,
            solver: LinearModelSolver::Sag,
            l2_penalty: None,
            tol: Some(1e-6),
            step_size: Some(1e-3),
            random_state: Some(0),
            max_iter: Some(100000),
        };
        let mut model = RidgeClassification::<Array2<_>, _, &str>::new(settings);
        match model.fit(&x, &y) {
            Ok(_) => {
                println!("{:?}", model.coef());
            }
            Err(error) => {
                println!("{:?}", error);
            }
        };

        assert_eq!(model.predict(&x).unwrap(), y);
    }
}
