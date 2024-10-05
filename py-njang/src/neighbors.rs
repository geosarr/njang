use ndarray::{Array1, Axis};
use njang::{Algebra, KdTree};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
#[pyclass]
pub struct KDTree {
    tree: KdTree<Array1<f64>>,
}

#[pymethods]
impl KDTree {
    #[new]
    pub fn new(x: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let mut tree = KdTree::new();
        let x = x.as_array();
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
            if let Some(heap) =
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
                Ok(vec![])
            }
        } else {
            Err(PyValueError::new_err("`k` should be >= 0"))
        }
    }
}
