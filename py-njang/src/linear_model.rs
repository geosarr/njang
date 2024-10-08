use ndarray::{Array0, Array1, Array2};
use njang::prelude::{
    ClassificationModel, LinearModelSolver, LinearRegression as LinReg, LinearRegressionSettings,
    LogisticRegression as LogReg, LogisticRegressionSettings, RegressionModel,
};
use numpy::{IntoPyArray, PyArray0, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct LinearRegression {
    reg: RidgeRegression,
}

#[pyclass]
pub struct LogisticRegression {
    reg: LogReg<Array2<f64>, Array1<f64>, isize>,
}

#[pymethods]
impl LogisticRegression {
    #[new]
    pub fn new(
        fit_intercept: bool,
        max_iter: Option<usize>,
        l1_penalty: Option<f64>,
        l2_penalty: Option<f64>,
        tol: Option<f64>,
        solver: Option<&str>,
        random_state: Option<u32>,
    ) -> PyResult<Self> {
        let solver = if let Some(solvr) = solver {
            if solvr == "sgd" {
                LinearModelSolver::Sgd
            } else if solvr == "bgd" {
                LinearModelSolver::Bgd
            } else {
                return Err(PyValueError::new_err(format!(
                    "solver `{}` not supported",
                    solvr
                )));
            }
        } else {
            LinearModelSolver::Sgd
        };
        let settings = LogisticRegressionSettings {
            l1_penalty,
            l2_penalty,
            step_size: Some(0.001),
            fit_intercept,
            solver,
            tol: Some(tol.unwrap_or(1e-4)),
            random_state,
            max_iter: Some(max_iter.unwrap_or(100000)),
        };
        Ok(Self {
            reg: LogReg::new(settings),
        })
    }
    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<isize>,
    ) -> PyResult<()> {
        let _ = self
            .reg
            .fit(&x.as_array().to_owned(), &y.as_array().to_owned());
        Ok(())
    }
    #[getter]
    pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.reg.intercept() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no intercept")),
        }
    }
    #[getter]
    pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        match self.reg.coef() {
            Some(value) => Ok(value.clone().into_pyarray_bound(py)),
            None => Err(PyValueError::new_err("no coefficient")),
        }
    }
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
    ) -> PyResult<Self> {
        // RidgeRegression seems to be faster than plain LinearRegression when penalty
        // is set to 0., why ?
        Ok(Self {
            reg: RidgeRegression::new(0., fit_intercept, max_iter, tol, solver, random_state)?,
        })
    }
    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyObject,
    ) -> PyResult<()> {
        self.reg.fit(py, x, y)
    }
    #[getter]
    pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.reg.intercept(py)
    }
    #[getter]
    pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
    ) -> PyResult<Self> {
        Ok(Self {
            reg_1d: RidgeRegression1d::new(
                alpha,
                fit_intercept,
                max_iter,
                tol,
                solver,
                random_state,
            )?,
            reg_2d: RidgeRegression2d::new(
                alpha,
                fit_intercept,
                max_iter,
                tol,
                solver,
                random_state,
            )?,
            fitted_1d: false,
            fitted_2d: false,
        })
    }
    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyObject,
    ) -> PyResult<()> {
        let dimension = y
            .getattr(py, "shape")?
            .call_method0(py, "__len__")?
            .extract::<usize>(py)?;
        if dimension == 1 {
            let _ = self.reg_1d.fit(x, y.extract::<PyReadonlyArray1<f64>>(py)?);
            self.fitted_1d = true;
            self.fitted_2d = false
        } else {
            let _ = self.reg_2d.fit(x, y.extract::<PyReadonlyArray2<f64>>(py)?);
            self.fitted_2d = true;
            self.fitted_1d = false
        }
        Ok(())
    }
    #[getter]
    pub fn intercept<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.fitted_1d {
            let val = self.reg_1d.intercept(py)?;
            Ok(val.as_any().clone())
        } else {
            let val = self.reg_2d.intercept(py)?;
            Ok(val.as_any().clone())
        }
    }
    #[getter]
    pub fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.fitted_1d {
            let val = self.reg_1d.coef(py)?;
            Ok(val.as_any().clone())
        } else {
            let val = self.reg_2d.coef(py)?;
            Ok(val.as_any().clone())
        }
    }
}

macro_rules! impl_ridge_reg {
    ($name:ident, $rust_ix:ty, $rust_ix_smalr:ty, $py_ix:ty, $py_ix_smalr:ty, $pyr_ix:ty, $pyr_ix_smalr:ty) => {
        #[pyclass]
        pub struct $name {
            reg: LinReg<$rust_ix, $rust_ix_smalr>,
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
            ) -> PyResult<Self> {
                let solver = if let Some(solvr) = solver {
                    if solvr == "sgd" {
                        LinearModelSolver::Sgd
                    } else if solvr == "exact" {
                        LinearModelSolver::Exact
                    } else if solvr == "svd" {
                        LinearModelSolver::Svd
                    } else if solvr == "qr" {
                        LinearModelSolver::Qr
                    } else if solvr == "cholesky" {
                        LinearModelSolver::Cholesky
                    } else if solvr == "sag" {
                        LinearModelSolver::Sag
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "solver `{}` not supported",
                            solvr
                        )));
                    }
                } else {
                    LinearModelSolver::Sgd
                };
                let settings = LinearRegressionSettings {
                    l1_penalty: None,
                    l2_penalty: Some(alpha),
                    step_size: Some(0.001),
                    fit_intercept,
                    solver,
                    tol,
                    random_state,
                    max_iter: Some(max_iter.unwrap_or(100000)),
                };
                Ok(Self {
                    reg: LinReg::new(settings),
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
