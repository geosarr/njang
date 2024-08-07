pub trait RegressionModel {
    type FitResult;
    type X;
    type Y;
    type PredictResult;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult;
    fn predict(&self, x: &Self::X) -> Self::PredictResult;
}

// pub trait ClassificationModel {
//     type FitResult;
//     type X;
//     type Y;
//     type PredictResult;
//     type PredictProbaResult;
//     fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult;
//     fn predict(&self, x: &Self::X) -> Self::PredictResult;
//     fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult;
// }
