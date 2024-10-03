use ndarray::*;
use rand_chacha::{ChaCha20Core, ChaCha20Rng};

use core::{
    cmp::Ordering,
    fmt::Debug,
    ops::{Index, Mul, Sub},
};
// use std::collections::BinaryHeap;

use crate::{
    error::NjangError,
    traits::{Algebra, Container, Label, Model, Scalar},
    ClassificationModel,
};

enum KnnSolver {
    Brute,
}

struct Knn<'a, T, L> {
    k: usize,
    data: Option<(&'a Array2<T>, &'a Array1<L>)>,
    solver: KnnSolver,
}

impl<'a, 'b: 'a, T, L: 'a + 'b> Model<'b> for Knn<'a, T, L>
where
    T: Scalar,
{
    type Data = (&'b Array2<T>, &'b Array1<L>);
    type FitResult = Result<(), NjangError>;
    fn fit(&mut self, data: &'b Self::Data) -> Self::FitResult {
        let (x, y) = data;
        match self.solver {
            KnnSolver::Brute => {
                self.data = Some((*x, *y));
            }
        };
        Ok(())
    }
}

// impl<'a, 'b, T: Scalar, L: Label> ClassificationModel for Knn<'a, T, L>
// where
//     'b: 'a,
// {
//     type PredictProbaResult = Result<(), ()>;
//     type PredictResult = Result<(), ()>;
//     type X = Array2<T>;
//     type Y = Array1<L>;
//     fn fit(&mut self, x: &Self::X, y: &Self::Y) -> <Self as
// Model<'b>>::FitResult {         let data = &(x, y);
//         <Self as Model>::fit(self, data)
//     }
//     fn predict(&self, x: &Self::X) -> Self::PredictResult {
//         Ok(())
//     }
//     fn predict_proba(&self, x: &Self::X) -> Self::PredictProbaResult {
//         Ok(())
//     }
// }

#[test]
fn knn() {
    let x = array![[1., 1.], [0., 1.], [-1., 0.]];
    let y = array![1, 1, 0];
    let mut model = Knn {
        k: 2,
        data: None,
        solver: KnnSolver::Brute,
    };
    let data = (&x, &y);
    Model::fit(&mut model, &data);
    println!("{:?}", model.data.unwrap().0);
}

#[derive(Clone, Debug, PartialEq)]
struct Node<K> {
    key: K,
    number: usize,
    left: Option<Box<Node<K>>>,
    right: Option<Box<Node<K>>>,
}

impl<K> Node<K> {
    pub fn new(key: K, number: usize) -> Self {
        Self {
            key,
            number,
            left: None,
            right: None,
        }
    }
}

/// Implementation of a Kd-tree.
///
/// The caller must make sure there are no duplicate keys inserted in the tree.
/// # Example
/// ```
/// use njang::KdTree;
/// let mut bt = KdTree::new();
/// bt.insert([0]);
/// bt.insert([1]);
/// bt.insert([2]);
/// assert_eq!(bt.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct KdTree<K>
where
    K: Container,
{
    pub root: Option<Box<Node<K>>>,
    pub len: usize,
}
impl<K: Container> Default for KdTree<K> {
    fn default() -> Self {
        Self::new()
    }
}
impl<K: Container> KdTree<K> {
    /// Creates an empty tree instance.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::<[usize; 1]>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }
    /// Creates a new tree with an initial (key, value) pair.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::init(["btree"]);
    /// assert_eq!(bt.len(), 1);
    /// ```
    pub fn init(key: K) -> Self {
        Self {
            root: Some(Box::new(Node::new(key, 0))),
            len: 1,
        }
    }
    /// Gives the number of (key, value) pairs in the tree.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::<[f32; 2]>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }
    /// Tests whether or not the tree is empty.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let mut bt = KdTree::new();
    /// bt.insert([1]);
    /// assert!(!bt.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K> KdTree<K>
where
    K: Index<usize> + Container<LenghtOutput = usize>,
    K::Output: PartialOrd + Copy,
{
    fn put<'a>(
        node: &mut Option<Box<Node<K>>>,
        key: K,
        number: usize,
        level: &'a mut usize,
    ) -> Option<&'a mut Box<Node<K>>> {
        match node {
            None => *node = Some(Box::new(Node::new(key, number))),
            Some(ref mut nod) => match key[*level].partial_cmp(&nod.key[*level]) {
                Some(Ordering::Less) => {
                    *level = (*level + 1) % key.length();
                    return Self::put(&mut nod.left, key, number, level);
                }
                Some(Ordering::Greater) => {
                    *level = (*level + 1) % key.length();
                    return Self::put(&mut nod.right, key, number, level);
                }
                Some(Ordering::Equal) => {
                    // Used to overwrite the current node's value, but doing so would change
                    // (possibly) the value of the current node's key, which changes the
                    // label/target of the predictor in the nearest neighbors algorithms.
                    // nod.value = value;
                    // return Some(nod);

                    // Possibility to put key and value in the left branch also.
                    *level = (*level + 1) % key.length();
                    return Self::put(&mut nod.right, key, number, level);
                }
                None => return None, //panic!("Unknown situation"),
            },
        }
        None
    }
    /// Inserts a `(key, value)` pair in the tree. The caller must make sure
    /// that the tree does not contain `key`.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let mut bt = KdTree::<[isize; 1]>::new();
    /// bt.insert([-1]);
    /// bt.insert([-2]);
    /// assert_eq!(bt.len(), 2);
    /// ```
    pub fn insert(&mut self, key: K) {
        let mut level = 0;
        Self::put(&mut self.root, key, self.len, &mut level);
        self.len += 1;
    }
    /// Searches the nearest neighbor of a `key` in the tree.
    ///
    /// Adapted from [this youtube channel][br].
    ///
    /// [br]: https://www.youtube.com/watch?v=Glp7THUpGow
    pub fn nearest_neighbor(&self, key: &K) -> Option<&K>
    where
        K: Algebra<Elem = K::Output, LenghtOutput = usize> + Debug,
        K::Output: Sub<Output = K::Output> + Mul<Output = K::Output> + Debug,
        for<'a> &'a K: Sub<&'a K, Output = K>,
    {
        let (best_squared_dist, best_key) = if let Some(ref root) = self.root {
            ((&root.key - key).squared_l2_norm(), &root.key)
        } else {
            return None;
        };
        let result = nearest_neighbor(&self.root, key, best_key, best_squared_dist, 0);
        Some(result.0)
    }
}

fn nearest_neighbor<'a, K>(
    node: &'a Option<Box<Node<K>>>,
    key: &K,
    mut best_key: &'a K,
    mut best_squared_dist: K::Elem,
    level: usize,
) -> (&'a K, K::Elem)
where
    K: Index<usize, Output = K::Elem> + Algebra<LenghtOutput = usize> + Debug,
    K::Elem: PartialOrd + Copy + Sub<Output = K::Elem> + Mul<Output = K::Elem> + Debug,
    for<'b> &'b K: Sub<&'b K, Output = K>,
{
    if let Some(nod) = node {
        let coordinate = level % key.length();
        let (next, other) = if key[coordinate] < nod.key[coordinate] {
            (&nod.left, &nod.right)
        } else {
            (&nod.right, &nod.left)
        };
        let (mut best, mut best_squared_dist) =
            nearest_neighbor(next, &key, best_key, best_squared_dist, level + 1);
        let dist = key[coordinate] - nod.key[coordinate];
        if dist * dist <= best_squared_dist {
            let radius_squared = (&nod.key - key).squared_l2_norm();
            if radius_squared < best_squared_dist {
                best_squared_dist = radius_squared;
                best = &nod.key;
            }
            let (temp, temp_smallest_dist) =
                nearest_neighbor(other, &key, best, best_squared_dist, level + 1);
            (best, best_squared_dist) = (temp, temp_smallest_dist);
        }
        return (best, best_squared_dist);
    } else {
        return (best_key, best_squared_dist);
    };
}

#[test]
fn partial() {
    let mut bt = KdTree::<_>::new();
    bt.insert(array![5., 4.]);
    bt.insert(array![2., 6.]);
    bt.insert(array![13., 3.]);
    bt.insert(array![3., 1.]);
    bt.insert(array![10., 2.]);
    bt.insert(array![8., 7.]);
    println!("{:?}", bt.len());
    println!("{:#?}\n", bt);
    println!("{:#?}", bt.nearest_neighbor(&array![9., 4.]).unwrap());
}
