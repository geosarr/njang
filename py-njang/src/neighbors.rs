use ndarray::{Array0, Array1, Array2, Axis};
use njang::{Algebra, KdTree};
use numpy::{IntoPyArray, PyArray0, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
#[pyclass]
pub struct KDTree {
    tree: KdTree<Array1<f64>>,
}

#[pymethods]
impl KDTree {
    #[new]
    pub fn new(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let mut tree = KdTree::new();
        let x = x.as_array();
        let y = y.as_array();
        x.axis_iter(Axis(0))
            .map(|row| tree.insert(row.to_owned()))
            .for_each(drop);
        Ok(Self { tree })
    }
    pub fn k_nearest_neighbors(
        &self,
        key: PyReadonlyArray1<f64>,
        k: isize,
    ) -> PyResult<Vec<(usize, f64)>> {
        if k >= 0 {
            if let Some(mut heap) =
                self.tree
                    .k_nearest_neighbors(&key.as_array().to_owned(), k as usize, |a, b| {
                        (a - b).l2_norm()
                    })
            {
                let mut res = heap
                    .to_vec()
                    .iter()
                    .map(|n| (n.number, n.dist))
                    .collect::<Vec<_>>();
                res.sort_by(|a, b| a.1.total_cmp(&b.1));
                Ok(res)
            } else {
                Err(PyValueError::new_err("no nearest neighbors"))
            }
        } else {
            Err(PyValueError::new_err("`k` should be >= 0"))
        }
    }
}
