use ndarray::{Array0, Array1, Array2, Axis};
use njang::KdTree;

use numpy::{IntoPyArray, PyArray0, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
pub struct KDTree {
    tree: KdTree<Array1<f64>, f64>,
}

#[pymethods]
impl KDTree {
    #[new]
    pub fn new(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let mut tree = KdTree::new();
        let x = x.as_array();
        let y = y.as_array();
        x.axis_iter(Axis(0))
            .enumerate()
            .map(|(i, row)| tree.insert(row.to_owned(), y[i]))
            .for_each(drop);
        Ok(Self { tree })
    }
    pub fn nearest_neighbor<'py>(
        &self,
        key: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if let Some(key) = self.tree.nearest_neighbor(&key.as_array().to_owned()) {
            Ok(key.clone().into_pyarray_bound(py))
        } else {
            Err(PyValueError::new_err("no nearest neighbor"))
        }
    }
}
