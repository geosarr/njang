use ndarray::*;
use rand_chacha::{ChaCha20Core, ChaCha20Rng};

use core::{
    cmp::Ordering,
    fmt::Debug,
    ops::{Index, Mul, Sub},
};
use std::collections::HashSet;
// use std::collections::BinaryHeap;

use crate::{
    error::NjangError,
    traits::{Algebra, Container, Label, Model, Scalar},
    ClassificationModel,
};

use core::mem::replace;

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
    pub fn k_nearest_neighbors(&self, key: &K, k: usize) -> Option<Vec<(usize, K::Elem)>>
    where
        K: Index<usize, Output = K::Elem> + Algebra<LenghtOutput = usize> + Debug,
        K::Elem: PartialOrd + Copy + Sub<Output = K::Elem> + Mul<Output = K::Elem> + Debug,
        for<'b> &'b K: Sub<&'b K, Output = K>,
    {
        let best_squared_dist = if let Some(ref root) = self.root {
            (&root.key - key).squared_l2_norm()
        } else {
            return None;
        };
        let mut the_bests = HashSet::with_capacity(k + 1);
        let mut the_bests_with_dist = Vec::with_capacity(k + 1);
        for i in 0..k {
            the_bests_with_dist.push(k_nearest_neighbors(
                &self.root,
                key,
                0,
                best_squared_dist,
                0,
                &the_bests,
            ));
            the_bests.insert(the_bests_with_dist.last().unwrap().0);
        }
        Some(the_bests_with_dist)
    }
}

#[derive(Debug, PartialEq)]
pub struct KthNearestNeighbor<D> {
    number: usize,
    dist: D,
}
impl<D: PartialOrd> PartialOrd for KthNearestNeighbor<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

fn k_nearest_neighbors<'a, K>(
    node: &'a Option<Box<Node<K>>>,
    key: &K,
    mut best_key: usize,
    mut best_squared_dist: K::Elem,
    level: usize,
    the_bests: &HashSet<usize>,
) -> (usize, K::Elem)
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
        let (mut best, mut best_squared_dist) = k_nearest_neighbors(
            next,
            &key,
            best_key,
            best_squared_dist,
            level + 1,
            the_bests,
        );
        let dist = key[coordinate] - nod.key[coordinate];
        if (dist * dist <= best_squared_dist) & !the_bests.contains(&nod.number) {
            let radius_squared = (&nod.key - key).squared_l2_norm();
            if radius_squared < best_squared_dist {
                best_squared_dist = radius_squared;
                best = nod.number;
            }
            let (temp, temp_smallest_dist) =
                k_nearest_neighbors(other, &key, best, best_squared_dist, level + 1, the_bests);
            (best, best_squared_dist) = (temp, temp_smallest_dist);
        }
        return (best, best_squared_dist);
    } else {
        return (best_key, best_squared_dist);
    };
}

/// Defines the orientation of a binary heap (min oriented or max oriented)
#[derive(Debug, Clone, Default)]
pub enum HeapOrient {
    #[default]
    /// Max-oriented binary heap
    Max,
    /// Min-oriented binary heap
    Min,
}

/// Implementation of priority queues using a `Vec` structure
/// # Examples
/// ```
/// use njang::{BinaryHeap, HeapOrient};
/// let mut bhqueue = BinaryHeap::with_capacity(3, HeapOrient::Max);
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
    // type of binary heap
    kind: HeapOrient,
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
    /// use njang::{BinaryHeap, HeapOrient};
    /// let bhqueue = BinaryHeap::<&str>::with_capacity(1, HeapOrient::Min);
    /// assert_eq!(bhqueue.len(), 0);
    /// ```
    pub fn with_capacity(capacity: usize, k: HeapOrient) -> Self {
        // running time complexity: O(N)
        if capacity > 0 {
            let mut vector = Vec::with_capacity(capacity + 1);
            for _ in 0..capacity + 1 {
                vector.push(None);
            }

            Self {
                vec: vector,
                kind: k,
                n: 1,
            }
        } else {
            panic!("capacity shoul be > 0");
        }
    }

    /// Tests whether or not the binary heap is empty.
    /// # Example
    /// ```
    /// use njang::{BinaryHeap, HeapOrient};
    /// let mut bhqueue = BinaryHeap::<usize>::with_capacity(1, HeapOrient::Min);
    /// bhqueue.insert(1);
    /// assert!(!bhqueue.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.n == 1
    }

    /// Gives the number of objects in the binary heap.
    /// # Example
    /// ```
    /// use njang::{BinaryHeap, HeapOrient};
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3, HeapOrient::Min);
    /// bhqueue.insert(-1);
    /// bhqueue.insert(-2);
    /// bhqueue.insert(-4);
    /// assert_eq!(bhqueue.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        // number of objects in the heap
        // run time complexity O(1)
        self.n - 1
    }

    /// Returns the extremal (smallest in min oriented heap
    /// and largest in max oriented heap) object in the binary heap, if any.
    /// Returns `None` otherwise.
    /// # Example
    /// ```
    /// use njang::{BinaryHeap, HeapOrient};
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3, HeapOrient::Min);
    /// bhqueue.insert(0);
    /// bhqueue.insert(1);
    /// assert_eq!(bhqueue.extremum(), Some(&0));
    /// ```
    /// # Time complexity
    /// This is expected to run in O(1)
    pub fn extremum(&self) -> Option<&T> {
        // run time complexity O(1)
        self.vec[1].as_ref()
    }

    fn double(&mut self) {
        // run time complexity O(N)
        // doubling the size of the binary heap
        let mut vector = Vec::with_capacity(self.vec.len());
        for _ in 0..self.vec.len() {
            vector.push(None);
        }
        self.vec.append(&mut vector);
    }

    fn halve(&mut self) {
        // run time complexity O(N)
        // halving the size of the binary heap
        self.vec.truncate(self.vec.len() / 2);
    }
}

impl<T: PartialOrd + Clone> BinaryHeap<T> {
    fn swim(&mut self, mut k: usize) {
        // moves data at position k up in the "tree" following the
        // Peter principle: Nodes are promoted to their level of incompetence
        // run time complexity O(log(N))
        match self.kind {
            HeapOrient::Max => {
                while k > 1 && self.vec[k] > self.vec[k / 2] {
                    let val = self.vec[k].clone();
                    self.vec[k] = replace(&mut self.vec[k / 2], val);
                    k /= 2;
                }
            }
            HeapOrient::Min => {
                while k > 1 && self.vec[k] < self.vec[k / 2] {
                    let val = self.vec[k].clone();
                    self.vec[k] = replace(&mut self.vec[k / 2], val);
                    k /= 2;
                }
            }
        }
    }

    /// Inserts an object into the binary heap.
    /// # Example
    /// ```
    /// use njang::{BinaryHeap, HeapOrient};
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3, HeapOrient::Min);
    /// bhqueue.insert(-1);
    /// bhqueue.insert(-2);
    /// assert_eq!(bhqueue.len(), 2);
    /// ```
    /// # Time complexity
    /// This is expected to run in O(log(N)) on average
    pub fn insert(&mut self, key: T) {
        // run time complexity O(log(N)) (without resizing)
        // and O(N) with resizing
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

    fn sink(&mut self, mut k: usize, n: usize) {
        // moves data at position k down in the "tree" following the
        // Power struggle principle: Better nodes are promoted
        // Nodes beyond node n are untouched.
        // run time complexity O(log(N))
        if self.is_empty() {
            panic!("cannot sink data, heap is empty.")
        } else {
            match self.kind {
                HeapOrient::Max => {
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
                HeapOrient::Min => {
                    while 2 * k < n {
                        let mut j = 2 * k;
                        // find the smallest child of node k
                        if j < n - 1 && self.vec[j] > self.vec[j + 1] {
                            j += 1;
                        }
                        // compare it to node k
                        if self.vec[k] <= self.vec[j] {
                            break;
                        }
                        // exchange them if it is smaller than node k
                        let val = self.vec[k].clone();
                        self.vec[k] = replace(&mut self.vec[j], val);
                        k = j;
                    }
                }
            }
        }
    }

    /// Deletes and returns the extremal (smallest in min oriented heap
    /// and largest in max oriented heap) object in the binary heap, if any.
    /// Returns `None` otherwise.
    /// # Example
    /// ```
    /// use njang::{BinaryHeap, HeapOrient};
    /// let mut bhqueue = BinaryHeap::<isize>::with_capacity(3, HeapOrient::Min);
    /// bhqueue.insert(0);
    /// bhqueue.insert(1);
    /// assert_eq!(bhqueue.delete(), Some(0));
    /// ```
    /// # Time complexity
    /// This is expected to run in O(log(N)) on average
    pub fn delete(&mut self) -> Option<T> {
        // delete the extremal value and returns it
        // run time complexity O(log(N))
        if self.is_empty() {
            panic!("cannot delete, heap is empty");
        } else {
            let res = self.vec[1].clone();
            // Put the last object at the beginning of the root of the tree
            self.vec[1] = replace(&mut self.vec[self.n - 1], None);
            // sink the root object
            self.sink(1, self.n);
            self.n -= 1;
            if self.n <= self.vec.len() / 4 {
                self.halve();
            }
            res
        }
    }
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
    let knn = bt.k_nearest_neighbors(&array![6., 4.], 2).unwrap();
    println!("{:#?}", knn[0]);
    println!("{:#?}", knn[1]);
}
