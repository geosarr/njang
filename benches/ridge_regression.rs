#![feature(test)]
extern crate ndarray;
extern crate ndarray_rand;
extern crate njang;
extern crate rand_chacha;
extern crate test;

use ndarray::{Array1, Array2};
use ndarray_rand::{rand::SeedableRng, rand_distr::StandardNormal, RandomExt};
use njang::{LinearModelSolver, LinearRegression, LinearRegressionSettings, RegressionModel};
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
fn fit_ridge_reg_exact_bench(bench: &mut Bencher) {
    let (x, y) = dataset();
    let mut ridge_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionSettings {
        fit_intercept: false,
        solver: LinearModelSolver::Exact,
        l2_penalty: Some(1.),
        ..Default::default()
    });
    bench.iter(|| {
        let _solution = ridge_reg.fit(&x, &y);
    });
}

#[bench]
fn fit_ridge_reg_sgd_bench(bench: &mut Bencher) {
    let (x, y) = dataset();
    let mut ridge_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionSettings {
        fit_intercept: false,
        solver: LinearModelSolver::Sgd,
        l2_penalty: Some(1.),
        max_iter: Some(100000),
        tol: Some(1e-6),
        ..Default::default()
    });
    bench.iter(|| {
        let _solution = ridge_reg.fit(&x, &y);
    });
}

#[bench]
fn fit_ridge_reg_qr_bench(bench: &mut Bencher) {
    let (x, y) = dataset();
    let mut ridge_reg = LinearRegression::<Array1<_>, _>::new(LinearRegressionSettings {
        fit_intercept: false,
        solver: LinearModelSolver::Qr,
        l2_penalty: Some(1.),
        ..Default::default()
    });
    bench.iter(|| {
        let _solution = ridge_reg.fit(&x, &y);
    });
}
