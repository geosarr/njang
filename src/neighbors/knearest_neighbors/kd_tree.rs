use ndarray::*;
use num_traits::Zero;
use rand_chacha::{ChaCha20Core, ChaCha20Rng};

use core::{
    cmp::Ordering,
    fmt::Debug,
    ops::{Index, Mul, Sub},
};

use crate::{
    error::NjangError,
    traits::{Algebra, Container},
};

use core::mem::replace;

#[derive(Clone, Debug, PartialEq)]
struct Node<K> {
    key: K,
    number: usize,
    coordinate: Option<usize>,
    left: Option<Box<Node<K>>>,
    right: Option<Box<Node<K>>>,
}

impl<K> Node<K> {
    pub fn new(key: K, number: usize, coordinate: Option<usize>) -> Self {
        Self {
            key,
            number,
            coordinate,
            left: None,
            right: None,
        }
    }
}

/// Represents a nearest neighbor point
#[derive(Debug, PartialEq, Clone)]
pub struct KthNearestNeighbor<D> {
    /// Id of point.
    pub number: usize,
    /// Distance from query point.
    pub dist: D,
}
impl<D: PartialOrd> PartialOrd for KthNearestNeighbor<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

/// Implementation of a Kd-tree.
///
/// The caller must make sure there are no duplicate keys inserted in the tree.
/// # Example
/// ```
/// use njang::prelude::*;
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
    root: Option<Box<Node<K>>>,
    len: usize,
}
impl<K: Container> Default for KdTree<K> {
    /// Creates an empty tree.
    ///
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let bt = KdTree::<[usize; 1]>::default();
    /// assert_eq!(bt.len(), 0);
    /// ```
    fn default() -> Self {
        Self::new()
    }
}
impl<K: Container> KdTree<K> {
    /// Creates an empty tree instance.
    ///
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let bt = KdTree::<[usize; 1]>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// Gives the number of (key, value) pairs in the tree.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let bt = KdTree::<[f32; 2]>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether or not the tree is empty.
    /// # Example
    /// ```
    /// use njang::prelude::*;
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
    K: Index<usize, Output = K::Elem> + Container<LenghtOutput = usize>,
    K::Output: PartialOrd + Copy,
{
    fn put(
        node: &mut Option<Box<Node<K>>>,
        key: K,
        number: usize,
        coordinate: usize,
    ) -> Option<&mut Box<Node<K>>> {
        match node {
            None => *node = Some(Box::new(Node::new(key, number, Some(coordinate)))),
            Some(ref mut nod) => match key[coordinate].partial_cmp(&nod.key[coordinate]) {
                Some(Ordering::Less) => {
                    let coordinate = (nod.coordinate.unwrap() + 1) % key.length();
                    return Self::put(&mut nod.left, key, number, coordinate);
                }
                Some(Ordering::Greater) => {
                    let coordinate = (nod.coordinate.unwrap() + 1) % key.length();
                    return Self::put(&mut nod.right, key, number, coordinate);
                }
                Some(Ordering::Equal) => {
                    // Used to overwrite the current node's value, but doing so would change
                    // (possibly) the value of the current node's key, which changes the
                    // label/target of the predictor in the nearest neighbors algorithms.
                    // nod.value = value;
                    // return Some(nod);

                    // Possibility to put key and value in the left branch also.
                    let coordinate = (nod.coordinate.unwrap() + 1) % key.length();
                    return Self::put(&mut nod.right, key, number, coordinate);
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
    /// use njang::prelude::*;
    /// let mut bt = KdTree::<[isize; 1]>::new();
    /// bt.insert([-1]);
    /// bt.insert([-2]);
    /// assert_eq!(bt.len(), 2);
    /// ```
    pub fn insert(&mut self, key: K) {
        let mut level = 0;
        Self::put(&mut self.root, key, self.len, level);
        self.len += 1;
    }
    /// Searches the nearest neighbors of a `key` in the tree.
    ///
    /// Adapted from [Parallel k Nearest Neighbor Graph Construction Using
    /// Tree-Based Data Structures][paper]. The distance metric is provided by
    /// the caller.
    ///
    /// It returns a max oriented binary heap collecting the k nearest neighbors
    /// of `key` located in the tree. It means that the top element of the heap
    /// (accessible in O(1) running time) is the furthest from `key`.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use njang::prelude::*;
    /// let mut bt = KdTree::<_>::new();
    /// let a = array![5., 4.];
    /// let b = array![2., 6.];
    /// let c = array![13., 3.];
    /// let d = array![3., 1.];
    /// let e = array![10., 2.];
    /// let f = array![8., 7.];
    /// bt.insert(a.view());
    /// bt.insert(b.view());
    /// bt.insert(c.view());
    /// bt.insert(d.view());
    /// bt.insert(e.view());
    /// bt.insert(f.view());
    /// let mut knn = bt
    ///     .k_nearest_neighbors(&array![9., 4.].view(), 4, |a, b| (a - b).minkowsky(2.))
    ///     .unwrap();
    /// assert_eq!(2, knn.delete().unwrap().number);
    /// assert_eq!(0, knn.delete().unwrap().number);
    /// assert_eq!(5, knn.delete().unwrap().number);
    /// assert_eq!(4, knn.delete().unwrap().number);
    /// ```
    ///
    /// [paper]: http://dx.doi.org/10.5821/hpgm15.1
    pub fn k_nearest_neighbors<D>(
        &self,
        key: &K,
        k: usize,
        distance: D,
    ) -> Option<BinaryHeap<KthNearestNeighbor<K::Elem>>>
    where
        K: Algebra + Debug,
        K::Elem: Sub<Output = K::Elem> + Mul<Output = K::Elem> + Debug,
        D: Fn(&K, &K) -> K::Elem,
    {
        if self.root.is_none() | (k == 0) {
            return None;
        }
        Some(k_nearest_neighbors(
            &self.root,
            key,
            BinaryHeap::with_capacity(k + 1),
            k,
            &distance,
        ))
    }
    pub fn from_vec(keys: &mut Vec<(usize, K)>) -> Option<Self>
    where
        K: Clone + Index<usize> + Debug,
        K::Elem: Zero + Sub<K::Elem, Output = K::Elem> + Debug,
    {
        if keys.is_empty() {
            return None;
        }
        let dimension = keys[0].1.length();
        let n = keys.len();
        let dim_max_spread = find_dim_max_spread(keys, 0, n, dimension);
        let median = n / 2;
        keys[0..n].sort_by(|a, b| {
            a.1[dim_max_spread]
                .partial_cmp(&b.1[dim_max_spread])
                .unwrap()
        });
        let root_key = keys[median].1.clone();
        let mut root = Some(Box::new(Node::new(
            root_key,
            keys[median].0,
            Some(dim_max_spread),
        )));
        build_tree(&mut root, ChildType::Left, keys, 0, median, dimension);
        build_tree(&mut root, ChildType::Right, keys, median + 1, n, dimension);
        Some(Self { root: root, len: n })
    }
}

#[derive(Debug)]
enum ChildType {
    Left,
    Right,
}
fn build_tree<K>(
    node: &mut Option<Box<Node<K>>>,
    child_type: ChildType,
    keys: &mut Vec<(usize, K)>,
    start: usize,
    end: usize,
    dimension: usize,
) where
    K: Clone + Index<usize> + Debug,
    K::Output: PartialOrd + Zero + Copy + Sub<K::Output, Output = K::Output> + Debug,
{
    if start >= end {
        return;
    }
    if end == start + 1 {
        let (number, key) = (&keys[start].0, &keys[start].1);
        let new_node = Some(Box::new(Node::new(key.clone(), *number, None)));
        match child_type {
            ChildType::Left => {
                if let Some(ref mut nod) = node {
                    nod.left = new_node;
                }
            }
            ChildType::Right => {
                if let Some(ref mut nod) = node {
                    nod.right = new_node;
                }
            }
        };
        return;
    }
    let dim_max_spread = find_dim_max_spread(keys, start, end, dimension);
    keys[start..end].sort_by(|a, b| {
        a.1[dim_max_spread]
            .partial_cmp(&b.1[dim_max_spread])
            .unwrap()
    });
    let median = start + (end - start) / 2;
    match child_type {
        ChildType::Left => {
            if let Some(ref mut nod) = node {
                nod.left = Some(Box::new(Node::new(
                    keys[median].1.clone(),
                    keys[median].0,
                    Some(dim_max_spread),
                )));
                build_tree(
                    &mut nod.left,
                    ChildType::Left,
                    keys,
                    start,
                    median,
                    dimension,
                );
                build_tree(
                    &mut nod.left,
                    ChildType::Right,
                    keys,
                    median + 1,
                    end,
                    dimension,
                );
            }
        }
        ChildType::Right => {
            if let Some(ref mut nod) = node {
                nod.right = Some(Box::new(Node::new(
                    keys[median].1.clone(),
                    keys[median].0,
                    Some(dim_max_spread),
                )));
                build_tree(
                    &mut nod.right,
                    ChildType::Left,
                    keys,
                    start,
                    median,
                    dimension,
                );
                build_tree(
                    &mut nod.right,
                    ChildType::Right,
                    keys,
                    median + 1,
                    end,
                    dimension,
                );
            }
        }
    };
}

fn find_dim_max_spread<K>(
    keys: &mut Vec<(usize, K)>,
    start: usize,
    end: usize,
    dimension: usize,
) -> usize
where
    K: Clone + Index<usize> + Debug,
    K::Output: PartialOrd + Zero + Copy + Sub<K::Output, Output = K::Output> + Debug,
{
    let (mut dim_max_spread, mut max_spread) = (0, K::Output::zero());
    for dim in 0..dimension {
        let (mut min, mut max) = (keys[start].1[dim], keys[start].1[dim]);
        for index in start + 1..end {
            let val = keys[index].1[dim];
            if val > max {
                max = val;
            }
            if val < min {
                min = val;
            }
        }
        let spread = max - min;
        if spread > max_spread {
            dim_max_spread = dim;
            max_spread = spread;
        }
    }
    dim_max_spread
}
fn k_nearest_neighbors<K, D>(
    node: &Option<Box<Node<K>>>,
    key: &K,
    mut the_bests: BinaryHeap<KthNearestNeighbor<K::Elem>>,
    k: usize,
    distance: &D,
) -> BinaryHeap<KthNearestNeighbor<K::Elem>>
where
    K: Index<usize, Output = K::Elem> + Algebra<LenghtOutput = usize> + Debug,
    K::Elem: PartialOrd + Copy + Sub<Output = K::Elem> + Mul<Output = K::Elem> + Debug,
    D: Fn(&K, &K) -> K::Elem,
{
    if let Some(nod) = node {
        let dist = distance(&nod.key, key);
        if the_bests.len() < k {
            the_bests.insert(KthNearestNeighbor {
                number: nod.number,
                dist,
            });
        } else if dist < the_bests.maximum().unwrap().dist {
            // .unwrap() is safe here as long as k >= 1 because when k >= 1 the heap is not
            // empty, which guaranties the existence of a maximum.
            the_bests.delete();
            the_bests.insert(KthNearestNeighbor {
                number: nod.number,
                dist,
            });
        }
        if let Some(coordinate) = nod.coordinate {
            let (next, other, dist) = if key[coordinate] < nod.key[coordinate] {
                (&nod.left, &nod.right, nod.key[coordinate] - key[coordinate])
            } else {
                (&nod.right, &nod.left, key[coordinate] - nod.key[coordinate])
            };
            the_bests = k_nearest_neighbors(next, key, the_bests, k, distance);
            if (dist <= the_bests.maximum().unwrap().dist) | (the_bests.len() < k) {
                the_bests = k_nearest_neighbors(other, key, the_bests, k, distance);
            }
        }
    }
    the_bests
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

    /// Converts the binary heap to `Vec`.
    pub fn to_vec(mut self) -> Vec<T> {
        let mut res = Vec::with_capacity(self.len());
        for _ in 0..self.len() {
            res.push(self.delete().expect("Failed to delete"));
        }
        res
    }
}

#[test]
fn kd() {
    let mut v = [
        array![5., 4.],
        array![2., 6.],
        array![13., 3.],
        array![3., 1.],
        array![10., 2.],
        array![8., 7.],
    ]
    .into_iter()
    .enumerate()
    .collect::<Vec<_>>();
    let k = KdTree::from_vec(&mut v).unwrap();
    print!("{:#?}", k);
    let mut neighbors = k
        .k_nearest_neighbors(&array![9., 4.], 4, |a, b| (a - b).minkowsky(2.))
        .unwrap();
    print!("{:#?}", neighbors.delete());
    print!("{:#?}", neighbors.delete());
    print!("{:#?}", neighbors.delete());
    print!("{:#?}", neighbors.delete());
}
