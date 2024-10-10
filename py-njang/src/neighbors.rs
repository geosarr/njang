use ndarray::{Array1, Axis};
use njang::prelude::{Algebra, BallTree as RustBallTree, KdTree};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Class responsible for building a K-dimensional tree.
///
/// It builds a generalized binary search tree for which the dataset is split
/// according to the ordering of one dimension for each node. The dimension can
/// be chosen `naively` (cyclically starting for first dimension, then second
/// dimension, etc.). This method is fast to build and is useful when the entire
/// dataset is not known when building the tree. However the tree is not
/// guaranteed to be balanced, so the fast query is not guaranteed. The choice
/// of the dimension can be done in a more `robust` way by sorting recursively
/// the data points according to the dimension of the maximum spread then
/// splitting the current node at the median point. This method is slower to
/// build but guarantees a(n) (almost) balanced tree, so querying nearest
/// neighbors is faster.
#[pyclass]
pub struct KDTree {
    tree: KdTree<Array1<f64>>,
}

#[pyclass]
pub struct BallTree {
    tree: RustBallTree<Array1<f64>>,
}

#[pymethods]
impl KDTree {
    #[new]
    pub fn new(x: PyReadonlyArray2<f64>, method: &str) -> PyResult<Self> {
        let x = x.as_array();
        if method == "naive" {
            let mut tree = KdTree::new();
            x.axis_iter(Axis(0))
                .map(|row| tree.insert(row.to_owned()))
                .for_each(drop);
            Ok(Self { tree })
        } else if method == "robust" {
            let keys = x.axis_iter(Axis(0)).map(|row| row.to_owned());
            Ok(Self {
                tree: KdTree::from(keys).unwrap(),
            })
        } else {
            Err(PyValueError::new_err(
                "Only `naive`and `robust` tree building are supported",
            ))
        }
    }
    pub fn query(&self, key: PyReadonlyArray1<f64>, k: isize) -> PyResult<Vec<(usize, f64)>> {
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
                    .map(|n| (n.point, n.dist))
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

#[pymethods]
impl BallTree {
    #[new]
    pub fn new(x: PyReadonlyArray2<f64>, leaf_size: isize) -> PyResult<Self> {
        let x = x.as_array();
        let keys = x.axis_iter(Axis(0)).map(|row| row.to_owned());
        Ok(Self {
            tree: RustBallTree::from(keys, |a, b| (a - b).l2_norm(), leaf_size as usize).unwrap(),
        })
    }
    pub fn query(&self, key: PyReadonlyArray1<f64>, k: isize) -> PyResult<Vec<(usize, f64)>> {
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
                    .map(|n| (n.point.number, n.dist))
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
