use pyo3::prelude::*;
mod linear_model;
mod neighbors;
use linear_model::{LinearRegression, LogisticRegression, RidgeRegression};
use neighbors::{BallTree, KDTree};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn njang(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<LinearRegression>().unwrap();
    m.add_class::<RidgeRegression>().unwrap();
    m.add_class::<LogisticRegression>().unwrap();
    m.add_class::<KDTree>().unwrap();
    m.add_class::<BallTree>().unwrap();
    Ok(())
}
