use ndarray::{Array0, Array1, Array2};
use njang::{
    RegressionModel, RidgeRegression as RidgeReg, RidgeRegressionHyperParameter,
    RidgeRegressionSolver,
};
use numpy::{IntoPyArray, PyArray0, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct LinearRegression {
    reg: RidgeRegression1d,
    // reg_2d: RidgeRegression,
}

#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new(
        fit_intercept: bool,
        max_iter: Option<usize>,
        tol: Option<f64>,
        solver: Option<&str>,
        random_state: Option<u32>,
        warm_start: Option<bool>,
    ) -> PyResult<Self> {
        // RidgeRegression seems to be faster than plain LinearRegression when penalty is set to 0., why ?
        Ok(Self {
            reg: RidgeRegression1d::new(
                0.,
                fit_intercept,
                max_iter,
                tol,
                solver,
                random_state,
                warm_start,
            )?,
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
    reg_1d: RidgeRegression1d,
    reg_2d: RidgeRegression2d,
    fitted_1d: bool,
    fitted_2d: bool,
}

#[pymethods]
impl RidgeRegression {
    #[new]
    pub fn new(
        alpha: f64,
        fit_intercept: bool,
        max_iter: Option<usize>,
        tol: Option<f64>,
        solver: Option<&str>,
        random_state: Option<u32>,
        warm_start: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self {
            reg_1d: RidgeRegression1d::new(
                alpha,
                fit_intercept,
                max_iter,
                tol,
                solver,
                random_state,
                warm_start,
            )?,
            reg_2d: RidgeRegression2d::new(
                alpha,
                fit_intercept,
                max_iter,
                tol,
                solver,
                random_state,
                warm_start,
            )?,
            fitted_1d: false,
            fitted_2d: false,
        })
    }
    pub fn fit_1d(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let _ = self.reg_1d.fit(x, y);
        self.fitted_1d = true;
        self.fitted_2d = false;
        Ok(())
    }
    pub fn fit_2d(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> PyResult<()> {
        let _ = self.reg_2d.fit(x, y);
        self.fitted_2d = true;
        self.fitted_1d = false;
        Ok(())
    }
    #[getter]
    pub fn intercept_1d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray0<f64>>> {
        if self.fitted_1d {
            self.reg_1d.intercept(py)
        } else {
            Err(PyValueError::new_err("no intercept"))
        }
    }
    #[getter]
    pub fn intercept_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if self.fitted_2d {
            self.reg_2d.intercept(py)
        } else {
            Err(PyValueError::new_err("no intercept"))
        }
    }
    #[getter]
    pub fn coef_1d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if self.fitted_1d {
            self.reg_1d.coef(py)
        } else {
            Err(PyValueError::new_err("no coef"))
        }
    }
    #[getter]
    pub fn coef_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.fitted_2d {
            self.reg_2d.coef(py)
        } else {
            Err(PyValueError::new_err("no coef"))
        }
    }
}

macro_rules! impl_ridge_reg {
    ($name:ident, $rust_ix:ty, $rust_ix_smalr:ty, $py_ix:ty, $py_ix_smalr:ty, $pyr_ix:ty, $pyr_ix_smalr:ty) => {
        #[pyclass]
        pub struct $name {
            reg: RidgeReg<$rust_ix, $rust_ix_smalr, f64>,
        }
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(
                alpha: f64,
                fit_intercept: bool,
                max_iter: Option<usize>,
                tol: Option<f64>,
                solver: Option<&str>,
                random_state: Option<u32>,
                warm_start: Option<bool>,
            ) -> PyResult<Self> {
                let solver = if let Some(solvr) = solver {
                    if solvr == "sgd" {
                        RidgeRegressionSolver::SGD
                    } else if solvr == "ols" {
                        RidgeRegressionSolver::EXACT
                    } else if solvr == "qr" {
                        RidgeRegressionSolver::QR
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "solver `{}` not supported",
                            solvr
                        )));
                    }
                } else {
                    RidgeRegressionSolver::SGD
                };
                let settings = RidgeRegressionHyperParameter {
                    alpha,
                    fit_intercept,
                    solver,
                    tol: Some(tol.unwrap_or(0.0001)),
                    random_state: Some(random_state.unwrap_or(0)),
                    max_iter: Some(max_iter.unwrap_or(100000)),
                    warm_start: warm_start.unwrap_or(false),
                };
                Ok(Self {
                    reg: RidgeReg::new(settings),
                })
            }
            pub fn fit(&mut self, x: $pyr_ix, y: $pyr_ix_smalr) -> PyResult<()> {
                let _ = self
                    .reg
                    .fit(&x.as_array().to_owned(), &y.as_array().to_owned());
                Ok(())
            }
            #[getter]
            pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, $py_ix_smalr>> {
                match self.reg.intercept() {
                    Some(value) => Ok(value.clone().into_pyarray_bound(py)),
                    None => Err(PyValueError::new_err("no intercept")),
                }
            }
            #[getter]
            pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, $py_ix>> {
                match self.reg.coef() {
                    Some(value) => Ok(value.clone().into_pyarray_bound(py)),
                    None => Err(PyValueError::new_err("no coefficient")),
                }
            }
        }
    };
}
impl_ridge_reg!(
    RidgeRegression1d,
    Array1<f64>,
    Array0<f64>,
    PyArray1<f64>,
    PyArray0<f64>,
    PyReadonlyArray2<f64>,
    PyReadonlyArray1<f64>
);

impl_ridge_reg!(
    RidgeRegression2d,
    Array2<f64>,
    Array1<f64>,
    PyArray2<f64>,
    PyArray1<f64>,
    PyReadonlyArray2<f64>,
    PyReadonlyArray2<f64>
);
