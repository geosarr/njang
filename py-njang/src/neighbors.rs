use ndarray::{Array1, Array2, Axis};
use njang::prelude::{
    Algebra, BallTree as RustBallTree, KdTree, NearestNeighbors, NearestNeighborsSettings,
    NearestNeighborsSolver, RegressionModel, Scalar,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

type Distance<T> = fn(&Array1<T>, &Array1<T>) -> T;
type RustNearestNeighbors<T, Y> = NearestNeighbors<Distance<T>, Array1<T>, Y>;

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

#[pyclass]
pub struct KnnRegressor {
    model: RustNearestNeighbors<f64, Array2<f64>>,
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

pub fn l2_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).minkowsky(2.)
}
fn new_supervised_knn<D>(
    n_neighbors: isize,
    algorithm: &str,
    mut leaf_size: Option<isize>,
    distance: D,
) -> PyResult<NearestNeighbors<D, Array1<f64>, Array2<f64>>>
where
    D: Fn(&Array1<f64>, &Array1<f64>) -> f64,
{
    let solver = if algorithm == "kdtree" {
        NearestNeighborsSolver::KdTree
    } else if algorithm == "balltree" {
        let lsize = if let Some(lsize) = leaf_size {
            if lsize <= 0 {
                return Err(PyValueError::new_err("`leaf_size` should be >= 1"));
            }
        } else {
            leaf_size = Some(40);
        };
        NearestNeighborsSolver::BallTree
    } else {
        return Err(PyValueError::new_err(
            "The only supported algorithms are [`balltree`, `kdtree`]",
        ));
    };
    if n_neighbors <= 0 {
        return Err(PyValueError::new_err(
            "`n_neighbors` and `leaf_size` should be >= 1",
        ));
    }
    let settings = NearestNeighborsSettings {
        solver,
        distance,
        n_neighbors: n_neighbors as usize,
        leaf_size: leaf_size.map(|v| v as usize),
    };
    Ok(NearestNeighbors::new(settings))
}

#[pymethods]
impl KnnRegressor {
    #[new]
    pub fn new(n_neighbors: isize, algorithm: &str, leaf_size: Option<isize>) -> PyResult<Self> {
        // if p < 1. {
        //     return Err(PyValueError::new_err("`p` should be >= 1."));
        // }
        // let minkowsky_distance = |a: &Array1<f64>, b: &Array1<f64>| (a -
        // b).minkowsky(p);
        let distance: Distance<f64> = l2_distance;
        let model = match new_supervised_knn(n_neighbors, algorithm, leaf_size, distance) {
            Ok(model) => model,
            Err(error) => return Err(error),
        };
        Ok(Self { model })
    }

    pub fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) {
        self.model
            .fit(&x.as_array().to_owned(), &y.as_array().to_owned());
    }
}
