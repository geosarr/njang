#![feature(test)]
extern crate ndarray;
extern crate ndarray_rand;
extern crate njang;
extern crate rand_chacha;
extern crate test;

use ndarray::{Array1, Array2};
use ndarray_rand::{rand::SeedableRng, rand_distr::StandardNormal, RandomExt};
use njang::{
    LinearRegression, LinearRegressionHyperParameter, LinearRegressionSolver, RegressionModel,
};
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

#[bench]
fn fit_lin_reg_exact_bench(bench: &mut Bencher) {
    let (x, y) = dataset();
    let mut lin_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionHyperParameter {
        fit_intercept: false,
        solver: LinearRegressionSolver::EXACT,
    });
    bench.iter(|| {
        let _solution = lin_reg.fit(&x, &y);
    });
}

#[bench]
fn fit_lin_reg_svd_bench(bench: &mut Bencher) {
    let (x, y) = dataset();
    let mut lin_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionHyperParameter {
        fit_intercept: false,
        solver: LinearRegressionSolver::SVD,
    });
    bench.iter(|| {
        let _solution = lin_reg.fit(&x, &y);
    });
}

#[bench]
fn fit_lin_reg_qr_bench(bench: &mut Bencher) {
    let (x, y) = dataset();
    let mut lin_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionHyperParameter {
        fit_intercept: false,
        solver: LinearRegressionSolver::QR,
    });
    bench.iter(|| {
        let _solution = lin_reg.fit(&x, &y);
    });
}

#[bench]
fn fit_lin_reg_chol_bench(bench: &mut Bencher) {
    let (x, y) = dataset();
    let mut lin_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionHyperParameter {
        fit_intercept: false,
        solver: LinearRegressionSolver::CHOLESKY,
    });
    bench.iter(|| {
        let _solution = lin_reg.fit(&x, &y);
    });
}
