#![feature(test)]
extern crate ndarray;
extern crate ndarray_rand;
extern crate njang;
extern crate rand_chacha;
extern crate test;

use ndarray::{Array1, Array2};
use ndarray_rand::{rand::SeedableRng, rand_distr::StandardNormal, RandomExt};
use njang::{LinearRegression, LinearRegressionSettings, LinearRegressionSolver, RegressionModel};
use rand_chacha::ChaCha20Rng;
use test::Bencher;

const N: usize = 10000;
const P: usize = 50;

fn dataset() -> (Array2<f32>, Array1<f32>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let x = Array2::<f32>::random_using((N, P), StandardNormal, &mut rng);
    let y = x.dot(&Array1::from(vec![1.; P]));
    (x, y)
}

// #[bench]
// fn fit_ridge_reg_exact_bench(bench: &mut Bencher) {
//     let (x, y) = dataset();
//     let mut ridge_reg = RidgeRegression::<Array1<_>,
// _>::new(RidgeLinearRegressionSettings {         fit_intercept: false,
//         solver: RidgeLinearRegressionSolver::EXACT,
//         ..Default::default()
//     });
//     bench.iter(|| {
//         let _solution = ridge_reg.fit(&x, &y);
//     });
// }

// #[bench]
// fn fit_ridge_reg_sgd_bench(bench: &mut Bencher) {
//     let (x, y) = dataset();
//     let mut ridge_reg = RidgeRegression::<Array1<_>,
// _>::new(RidgeLinearRegressionSettings {         fit_intercept: false,
//         solver: RidgeLinearRegressionSolver::SGD,
//         max_iter: Some(10000),
//         ..Default::default()
//     });
//     bench.iter(|| {
//         let _solution = ridge_reg.fit(&x, &y);
//     });
// }

// #[bench]
// fn fit_ridge_reg_qr_bench(bench: &mut Bencher) {
//     let (x, y) = dataset();
//     let mut ridge_reg = RidgeRegression::<Array1<_>,
// _>::new(RidgeLinearRegressionSettings {         fit_intercept: false,
//         solver: RidgeLinearRegressionSolver::QR,
//         ..Default::default()
//     });
//     bench.iter(|| {
//         let _solution = ridge_reg.fit(&x, &y);
//     });
// }
