use ndarray::{Array0, Array1, Array2};
use njang::{
    LinearRegression as LinReg, LinearRegressionHyperParameter, LinearRegressionSolver,
    RegressionModel, RidgeRegression as RidgeReg, RidgeRegressionHyperParameter,
    RidgeRegressionSolver,
};
use numpy::{IntoPyArray, PyArray0, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct LinearRegression {
    inner: LinReg<Array1<f64>, Array0<f64>>,
}
#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new(fit_intercept: bool, solver: &str) -> PyResult<Self> {
        let solver = if solver == "svd" {
            LinearRegressionSolver::Svd
        } else if solver == "exact" {
            LinearRegressionSolver::Exact
        } else if solver == "qr" {
            LinearRegressionSolver::Qr
        } else {
            return Err(PyValueError::new_err("`{solver}` solver not supported"));
        };
        let settings = LinearRegressionHyperParameter {
            fit_intercept,
            solver: LinearRegressionSolver::Qr,
        };
        Ok(Self {
            inner: LinReg::new(settings),
        })
    }
    pub fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let _ = self
            .inner
            .fit(&x.as_array().to_owned(), &y.as_array().to_owned());
        Ok(())
    }
    #[getter]
    pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray0<f64>>> {
        match self.inner.intercept() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no intercept")),
        }
    }
    #[getter]
    pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.coef() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no coefficient")),
        }
    }
}

#[pyclass]
pub struct RidgeRegression {
    inner: RidgeReg<Array1<f64>, Array0<f64>, f64>,
}
#[pymethods]
impl RidgeRegression {
    #[new]
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        maxiter: Option<usize>,
        tol: Option<f64>,
        solver: Option<&str>,
        random_state: Option<u32>,
    ) -> PyResult<Self> {
        let solver = if let Some(solvr) = solver {
            if solvr == "sgd" {
                RidgeRegressionSolver::Sgd
            } else if solvr == "exact" {
                RidgeRegressionSolver::Exact
            } else {
                return Err(PyValueError::new_err("`{solvr}` solver not supported"));
            }
        } else {
            RidgeRegressionSolver::Sgd
        };
        let settings = RidgeRegressionHyperParameter {
            alpha,
            fit_intercept,
            solver,
            tol: Some(tol.unwrap_or(0.0001)),
            random_state: Some(random_state.unwrap_or(0)),
            max_iter: Some(maxiter.unwrap_or(1000)),
        };
        Ok(Self {
            inner: RidgeReg::new(settings),
        })
    }
    pub fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let _ = self
            .inner
            .fit(&x.as_array().to_owned(), &y.as_array().to_owned());
        Ok(())
    }
    #[getter]
    pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray0<f64>>> {
        match self.inner.intercept() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no intercept")),
        }
    }
    #[getter]
    pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.coef() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no coefficient")),
        }
    }
}
