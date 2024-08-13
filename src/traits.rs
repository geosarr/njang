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

pub trait Info {
    type MeanOutput;
    type RowOutput;
    type ColOutput;
    type ShapeOutput;
    type ColMut;
    type NcolsOutput;
    type NrowsOutput;
    fn mean(&self) -> Self::MeanOutput; // Mean of each column for 2d containers and mean of all elements for 1d containers.
    fn get_row(&self, i: usize) -> Self::RowOutput; // Like copy, view of a "row",
    fn get_col(&self, i: usize) -> Self::ColOutput; // Like copy, view of a "column"
    fn shape(&self) -> Self::ShapeOutput; // Like (nrows, ncols) for 2d containers and (n_elements) for 1d containers.
    fn col_mut(&mut self, idx: usize, elem: Self::ColMut); // Mutate column number idx of a 2d container with elem;
    fn get_ncols(&self) -> Self::NcolsOutput; // Number of columns for 2d containers
    fn get_nrows(&self) -> Self::NrowsOutput; // Number of rows for 2d containers
}
