use ndarray::{Array0, Array1, Array2};
use njang::{
    RegressionModel, RidgeRegression as RidgeReg, RidgeRegressionHyperParameter,
    RidgeRegressionSolver,
};
use numpy::{IntoPyArray, PyArray0, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct LinearRegression {
    reg: RidgeRegression,
}
#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new(
        fit_intercept: bool,
        maxiter: Option<usize>,
        tol: Option<f64>,
        solver: Option<&str>,
        random_state: Option<u32>,
    ) -> PyResult<Self> {
        // RidgeRegression seems to be faster than plain LinearRegression when penalty is set to 0., why ?
        Ok(Self {
            reg: RidgeRegression::new(0., fit_intercept, maxiter, tol, solver, random_state)?,
        })
    }
    pub fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.reg.fit(x, y)
    }
    #[getter]
    pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray0<f64>>> {
        self.reg.intercept(py)
    }
    #[getter]
    pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.reg.coef(py)
    }
}

#[pyclass]
pub struct RidgeRegression {
    reg: RidgeReg<Array1<f64>, Array0<f64>, f64>,
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
            } else if solvr == "qr" {
                RidgeRegressionSolver::Qr
            } else {
                return Err(PyValueError::new_err(format!(
                    "solver `{}` not supported",
                    solvr
                )));
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
            max_iter: Some(maxiter.unwrap_or(100000)),
        };
        Ok(Self {
            reg: RidgeReg::new(settings),
        })
    }
    pub fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let _ = self
            .reg
            .fit(&x.as_array().to_owned(), &y.as_array().to_owned());
        Ok(())
    }
    #[getter]
    pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray0<f64>>> {
        match self.reg.intercept() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no intercept")),
        }
    }
    #[getter]
    pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.reg.coef() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no coefficient")),
        }
    }
}
