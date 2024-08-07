#![feature(test)]
extern crate ndarray;
extern crate njang;
extern crate test;

use ndarray::{Array0, Array1, Array2};
use njang::{LinearRegression, RegressionModel};
use test::Bencher;

#[bench]
fn fit_lin_reg_bench(bench: &mut Bencher) {
    const N: usize = 100000;
    const P: usize = 10;
    let x = (0..N).map(|x| x as f32).collect();
    let x = Array2::from_shape_vec((N / P, P), x).unwrap();
    let y = x.dot(&Array1::from(vec![1.; P]));
    let mut lin_reg = LinearRegression::<Array1<_>, _>::new(false, Default::default());
    bench.iter(|| {
        let _solution = lin_reg.fit(&x, &y);
    });
}
