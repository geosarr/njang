//! Neighboors finding algorithms.

use core::{
    cmp::Ordering,
    mem::replace,
    ops::{Add, Mul, Sub},
};
use std::os::raw;
mod ball_tree;
pub use ball_tree::*;

mod kd_tree;
use hashbrown::HashMap;
pub use kd_tree::*;
use ndarray::{Array1, Array2, Axis};
use num_traits::{Float, FromPrimitive, Zero};

use crate::traits::{Algebra, ClassificationModel, Container, Label, RegressionModel, Scalar};

#[derive(Debug, Clone, Copy)]
pub enum NearestNeighborsSolver {
    KdTree,
    BallTree,
}
// #[derive(Debug, Clone)]
enum Tree<K: Container> {
    KdTree(KdTree<K>),
    BallTree(BallTree<K>),
}

// #[derive(Debug, Clone)]
pub struct NearestNeighborsSettings<D> {
    solver: NearestNeighborsSolver,
    distance: D,
    n_neighbors: usize,
    leaf_size: Option<usize>,
}
pub struct NearestNeighbors<D, K: Container, Y> {
    pub settings: NearestNeighborsSettings<D>,
    tree: Option<Tree<K>>,
    y: Option<Y>,
}

impl<D: Fn(&K, &K) -> K::Elem, K: Container, Y> NearestNeighbors<D, K, Y> {
    pub fn new(settings: NearestNeighborsSettings<D>) -> Self {
        Self {
            settings,
            tree: None,
            y: None,
        }
    }
}

fn fit<T, D, Y>(
    solver: NearestNeighborsSolver,
    x: &Array2<T>,
    y: &Y,
    distance: Option<&D>,
    leaf_size: Option<usize>,
) -> (Option<Tree<Array1<T>>>, Y)
where
    Y: Clone,
    T: Scalar + Mul<Array1<T>, Output = Array1<T>>,
    D: Fn(&Array1<T>, &Array1<T>) -> T,
{
    let keys = x.axis_iter(Axis(0)).map(|row| row.to_owned());
    let tree = match solver {
        NearestNeighborsSolver::BallTree => {
            let mut tree = BallTree::from(keys, distance.unwrap(), leaf_size.unwrap());
            if let Some(tree) = tree {
                Some(Tree::BallTree(tree))
            } else {
                None
            }
        }
        NearestNeighborsSolver::KdTree => {
            let mut tree = KdTree::from(keys);
            if let Some(tree) = tree {
                Some(Tree::KdTree(tree))
            } else {
                None
            }
        }
    };
    (tree, y.clone())
}

fn modal<L: Label>(labels: &Array1<L>) -> Option<L> {
    let mut counter = HashMap::<L, usize>::with_capacity(1 + labels.len());
    let (mut modal_label, mut max_count) = (None, 0);
    for label in labels {
        let count = if counter.contains_key(label) {
            counter[label] + 1
        } else {
            1
        };
        if count > max_count {
            modal_label = Some(*label);
            max_count = count;
        }
        counter.insert(*label, count);
    }
    modal_label
}

impl<D, T> RegressionModel for NearestNeighbors<D, Array1<T>, Array2<T>>
where
    T: Scalar + Mul<Array1<T>, Output = Array1<T>>,
    D: Fn(&Array1<T>, &Array1<T>) -> T,
{
    type FitResult = Result<(), ()>;
    type PredictResult = Result<Array2<T>, ()>;
    type X = Array2<T>;
    type Y = Array2<T>;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
        let (tree, y_copied) = fit(
            self.settings.solver,
            x,
            y,
            Some(&self.settings.distance),
            self.settings.leaf_size,
        );
        self.tree = tree;
        self.y = Some(y_copied);
        Ok(())
    }
    fn predict(&self, x: &Self::X) -> Self::PredictResult {
        let y = self.y.as_ref().unwrap();
        let mut predictions = Array2::zeros((x.nrows(), y.ncols()));
        for (k, point) in x.axis_iter(Axis(0)).enumerate() {
            if let Some(ref tree) = self.tree {
                let prediction = match tree {
                    Tree::KdTree(kd_tree) => {
                        let mut tree = kd_tree
                            .k_nearest_neighbors(
                                &point.to_owned(),
                                self.settings.n_neighbors,
                                &self.settings.distance,
                            )
                            .unwrap();
                        let mut indices = Vec::with_capacity(tree.len() + 1);
                        while !tree.is_empty() {
                            indices.push(tree.delete().unwrap().point);
                        }
                        y.select(Axis(0), &indices).mean_axis(Axis(0)).unwrap()
                    }
                    Tree::BallTree(ball_tree) => {
                        let mut tree = ball_tree
                            .k_nearest_neighbors(
                                &point.to_owned(),
                                self.settings.n_neighbors,
                                &self.settings.distance,
                            )
                            .unwrap();
                        let mut indices = Vec::with_capacity(tree.len() + 1);
                        while !tree.is_empty() {
                            indices.push(tree.delete().unwrap().point.number);
                        }
                        y.select(Axis(0), &indices).mean_axis(Axis(0)).unwrap()
                    }
                };
                predictions.row_mut(k).assign(&prediction);
            }
        }
        Ok(predictions)
    }
}

impl<D, T, L> ClassificationModel for NearestNeighbors<D, Array1<T>, Array1<L>>
where
    T: Scalar + Mul<Array1<T>, Output = Array1<T>>,
    L: Label + Zero,
    D: Fn(&Array1<T>, &Array1<T>) -> T,
{
    type FitResult = Result<(), ()>;
    type PredictResult = Result<Array1<L>, ()>;
    type PredictProbaResult = Result<(), ()>;
    type X = Array2<T>;
    type Y = Array1<L>;
    fn fit(&mut self, x: &Self::X, y: &Self::Y) -> Self::FitResult {
        let (tree, y_copied) = fit(
            self.settings.solver,
            x,
            y,
            Some(&self.settings.distance),
            self.settings.leaf_size,
        );
        self.tree = tree;
        self.y = Some(y_copied);
        Ok(())
    }
    fn predict(&self, x: &Self::X) -> Self::PredictResult {
        let y = self.y.as_ref().unwrap();
        let mut predictions = Array1::zeros(x.nrows());
        for (k, point) in x.axis_iter(Axis(0)).enumerate() {
            if let Some(ref tree) = self.tree {
                let prediction = match tree {
                    Tree::KdTree(kd_tree) => {
                        let indices = if let Some(heap) = kd_tree.k_nearest_neighbors(
                            &point.to_owned(),
                            self.settings.n_neighbors,
                            &self.settings.distance,
                        ) {
                            let mut res = heap.to_vec().into_iter().collect::<Vec<_>>();
                            res.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
                            res.into_iter().map(|n| n.point).collect()
                        } else {
                            Vec::new()
                        };
                        modal(&y.select(Axis(0), &indices))
                    }
                    Tree::BallTree(ball_tree) => {
                        let indices = if let Some(heap) = ball_tree.k_nearest_neighbors(
                            &point.to_owned(),
                            self.settings.n_neighbors,
                            &self.settings.distance,
                        ) {
                            let mut res = heap.to_vec().into_iter().collect::<Vec<_>>();
                            res.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
                            res.into_iter().map(|n| n.point.number).collect()
                        } else {
                            Vec::new()
                        };
                        modal(&y.select(Axis(0), &indices))
                    }
                };
                predictions[k] = prediction.unwrap();
            }
        }
        Ok(predictions)
    }

    fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult {
        Ok(())
    }
}

#[test]
fn neighbors_() {
    use ndarray::array;
    let x = array![[5., 4.], [2., 6.], [13., 3.], [3., 1.], [10., 2.], [8., 7.]];
    let y = array![0, 1, 2, 3, 4, 5];
    let settings = NearestNeighborsSettings {
        solver: NearestNeighborsSolver::BallTree,
        distance: |a: &Array1<f32>, b: &Array1<f32>| (a - b).l2_norm(),
        n_neighbors: 1,
        leaf_size: Some(1),
    };
    let mut neighbor = NearestNeighbors::new(settings);
    ClassificationModel::fit(&mut neighbor, &x, &y);
    print!("{:#?}", neighbor.predict(&x));

    assert_eq!(neighbor.predict(&x).unwrap(), y);
}

#[derive(Debug, Clone)]
pub struct Point<K> {
    pub number: usize,
    pub value: K,
}

impl<K: PartialEq> PartialEq for Point<K> {
    fn eq(&self, other: &Self) -> bool {
        self.number.eq(&other.number) && self.value.eq(&other.value)
    }
}
/// Represents a nearest neighbor point
#[derive(Debug, PartialEq, Clone)]
pub struct KthNearestNeighbor<P, D> {
    /// Id/value of this point.
    pub point: P,
    /// Distance from a point.
    pub dist: D,
}
impl<P: PartialEq, D: PartialOrd> PartialOrd for KthNearestNeighbor<P, D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

/// Implementation of priority queues using a `Vec` structure
/// # Examples
/// ```
/// use njang::prelude::*;
/// let mut bhqueue = BinaryHeap::with_capacity(3);
/// assert_eq!(bhqueue.len(), 0);
/// bhqueue.insert(0);
/// bhqueue.insert(1);
/// bhqueue.insert(2);
/// assert_eq!(bhqueue.len(), 3);
/// assert_eq!(bhqueue.delete(), Some(2));
/// assert_eq!(bhqueue.delete(), Some(1));
/// assert_eq!(bhqueue.len(), 1);
/// ```
#[derive(Debug, Default, Clone)]
pub struct BinaryHeap<T>
where
    T: PartialOrd,
{
    // vector of objects
    vec: Vec<Option<T>>,
    // position of the next object in the heap (or 1 + number of objects)
    n: usize,
    // Remarks:
    // - objects are nodes of the tree
    // - in the implementation objects are stored in self.vec from index = 1 to index = capacity so
    //   that index = 0 is always None object and:
    //     - each node k's parent is at position k/2
    //     - each node k's children are at positions 2k and 2k+1
    // - in the max oriented binary heap (with kind = HeapOrient::Max), parents are larger than
    //   their children (smaller for min oriented heap)
}

impl<T: PartialOrd> BinaryHeap<T> {
    /// Creates a new empty binary heap with an initial size.
    /// # Panics
    /// If `capacity = 0`, then it panics.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let bhqueue = BinaryHeap::<&str>::with_capacity(1);
    /// assert_eq!(bhqueue.len(), 0);
    /// ```
    /// # Time complexity
    /// This is expected to run in `O(capacity)`
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity > 0 {
            let mut vector = Vec::with_capacity(capacity + 1);
            for _ in 0..capacity + 1 {
                vector.push(None);
            }

            Self { vec: vector, n: 1 }
        } else {
            panic!("capacity shoul be > 0");
        }
    }

    /// Tests whether or not the binary heap is empty.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let mut bhqueue = BinaryHeap::<usize>::with_capacity(1);
    /// bhqueue.insert(1);
    /// assert!(!bhqueue.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.n == 1
    }

    /// Gives the number of objects in the binary heap.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3);
    /// bhqueue.insert(-1);
    /// bhqueue.insert(-2);
    /// bhqueue.insert(-4);
    /// assert_eq!(bhqueue.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.n - 1
    }

    /// Returns a reference of the maximum object in the binary heap, if any.
    /// Returns `None` otherwise.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3);
    /// bhqueue.insert(0);
    /// bhqueue.insert(1);
    /// assert_eq!(bhqueue.maximum(), Some(&1));
    /// ```
    /// # Time complexity
    /// This is expected to run in `O(1)`
    pub fn maximum(&self) -> Option<&T> {
        self.vec[1].as_ref()
    }

    /// Doubles the size of the binary heap. Run time complexity is expected
    /// to be `O(N)`
    fn double(&mut self) {
        let mut vector = Vec::with_capacity(self.vec.len());
        for _ in 0..self.vec.len() {
            vector.push(None);
        }
        self.vec.append(&mut vector);
    }

    /// Halves the size of the binary heap. Run time complexity is expected to
    /// be `O(N)`
    fn halve(&mut self) {
        self.vec.truncate(self.vec.len() / 2);
    }
}

impl<T: PartialOrd + Clone> BinaryHeap<T> {
    /// Moves data at position k up in the "tree" following the Peter principle:
    /// Nodes are promoted to their level of incompetence.
    ///
    /// Run time complexity is expected to be `O(log(N))`
    fn swim(&mut self, mut k: usize) {
        while k > 1 && self.vec[k] > self.vec[k / 2] {
            let val = self.vec[k].clone();
            self.vec[k] = replace(&mut self.vec[k / 2], val);
            k /= 2;
        }
    }

    /// Inserts an object into the binary heap.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3);
    /// bhqueue.insert(-1);
    /// bhqueue.insert(-2);
    /// assert_eq!(bhqueue.len(), 2);
    /// ```
    /// # Time complexity
    /// This is expected to run in `O(log(N))` on average (without doubling the
    /// heap). Doubling is `O(N)`. If the definitive size of heap is known in
    /// advance, it is better to use [.with_capacity][Self::with_capacity]
    /// method to build the heap.
    pub fn insert(&mut self, key: T) {
        if self.n < self.vec.len() {
            self.vec[self.n] = Some(key);
            self.swim(self.n);
            self.n += 1;
            if self.n == self.vec.len() {
                // resize the stack to allow more capacity
                self.double();
            }
        } else {
            panic!("cannot push, stack is full or has capacity 0");
        }
    }

    /// Moves data at position k down in the "tree" following the Power struggle
    /// principle: Better nodes are promoted Nodes beyond node n are
    /// untouched. Run time complexity is expected to be O(log(N)).
    fn sink(&mut self, mut k: usize, n: usize) {
        if !self.is_empty() {
            while 2 * k < n {
                let mut j = 2 * k;
                // find the largest child of node k
                if j < n - 1 && self.vec[j] < self.vec[j + 1] {
                    j += 1;
                }
                // compare it to node k
                if self.vec[k] >= self.vec[j] {
                    break;
                }
                // exchange them if it is larger than node k
                let val = self.vec[k].clone();
                self.vec[k] = replace(&mut self.vec[j], val);
                k = j;
            }
        }
    }

    /// Deletes and returns the maximum object in the binary heap, if any.
    /// Returns `None` otherwise.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3);
    /// bhqueue.insert(0);
    /// bhqueue.insert(1);
    /// assert_eq!(bhqueue.delete(), Some(1));
    /// ```
    /// # Time complexity
    /// This is expected to run in O(log(N)) on average
    pub fn delete(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let res = self.vec[1].clone();
            // Put the last object at the beginning of the root of the tree
            self.vec[1] = self.vec[self.n - 1].take();
            // sink the root object
            self.sink(1, self.n);
            self.n -= 1;
            if self.n <= self.vec.len() / 4 {
                self.halve();
            }
            res
        }
    }

    /// Converts the binary heap to `Vec`.
    pub fn to_vec(mut self) -> Vec<T> {
        let mut res = Vec::with_capacity(self.len());
        for _ in 0..self.len() {
            res.push(self.delete().expect("Failed to delete"));
        }
        res
    }
}
