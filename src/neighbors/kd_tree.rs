use num_traits::Zero;

use core::{
    cmp::Ordering,
    fmt::Debug,
    ops::{Index, Mul, Sub},
};

use crate::traits::{Algebra, Container};

use crate::neighbors::BinaryHeap;

use super::{KthNearestNeighbor, Point};

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

/// Implementation of a K-dimensional tree.
///
/// The caller must make sure there are no duplicate keys inserted in the tree.
/// # Example
/// ```
/// use ndarray::array;
/// use njang::prelude::*;
/// let mut bt = KdTree::new();
/// bt.insert(array![0f32]);
/// bt.insert(array![1.]);
/// bt.insert(array![2.]);
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
    /// use ndarray::Array1;
    /// use njang::prelude::*;
    /// let bt = KdTree::<Array1<f32>>::default();
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
    /// use ndarray::Array1;
    /// use njang::prelude::*;
    /// let bt = KdTree::<Array1<f32>>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// Gives the number of keys in the tree.
    /// # Example
    /// ```
    /// use ndarray::Array1;
    /// use njang::prelude::*;
    /// let bt = KdTree::<Array1<f32>>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether or not the tree is empty.
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use njang::prelude::*;
    /// let mut bt = KdTree::new();
    /// bt.insert(array![1f32]);
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

    /// Inserts a `key` in the tree.
    ///
    /// The caller must make sure that the tree does not already contain `key`,
    /// to avoid duplicates and error when retrieving nearest neighbors.
    ///
    /// This method is useful for online tree construction, when all the
    /// points to insert are not available at the same time. Building the
    /// tree with this mehod does not guarantee a balanced tree, so the
    /// caller may expect poor performance when retrieving the nearest
    /// neighbors, however inserting an element is fast on average.
    ///
    /// # Example
    /// ```
    /// use ndarray::{array, Array1};
    /// use njang::prelude::*;
    /// let mut bt = KdTree::<Array1<f32>>::new();
    /// bt.insert(array![-1.]);
    /// bt.insert(array![-2.]);
    /// assert_eq!(bt.len(), 2);
    /// ```
    pub fn insert(&mut self, key: K) {
        let level = 0;
        Self::put(&mut self.root, key, self.len, level);
        self.len += 1;
    }

    /// Inserts `keys` in the tree.
    ///
    /// The caller must make sure that the `keys` does not have duplicate
    /// elements to avoid errors when retrieving nearest neighbors.
    ///
    /// This method is useful for offline tree construction, when all the
    /// points to insert are available at the same time. Building the
    /// tree with this mehod guarantees an (almost) balanced tree, so the caller
    /// may expect faster nearest neighbors retrieval, at the cost of
    /// slower tree construction.
    ///
    /// Adapted from the paper: [Parallel k Nearest Neighbor Graph Construction
    /// Using Tree-Based Data Structures][paper].
    ///
    /// # Example
    /// ```
    /// use ndarray::{array, Array1};
    /// use njang::prelude::*;
    /// let mut bt = KdTree::<Array1<f32>>::new();
    /// bt.insert(array![-1.]);
    /// bt.insert(array![-2.]);
    /// assert_eq!(bt.len(), 2);
    /// ```
    ///
    /// [paper]: http://dx.doi.org/10.5821/hpgm15.1
    pub fn from<Keys>(keys: Keys) -> Option<Self>
    where
        Keys: IntoIterator<Item = K>,
        K: Clone + Index<usize> + Debug,
        K::Elem: Zero + Sub<K::Elem, Output = K::Elem> + Debug,
    {
        let mut keys = keys
            .into_iter()
            .enumerate()
            .map(|(number, value)| Point { number, value })
            .collect::<Vec<_>>();
        if keys.is_empty() {
            return None;
        }
        let dimension = keys[0].value.length();
        let len = keys.len();
        let dim_max_spread = find_dim_max_spread(&keys, 0, len, dimension);
        let median = len / 2;
        keys[0..len].sort_by(|a, b| {
            a.value[dim_max_spread]
                .partial_cmp(&b.value[dim_max_spread])
                .unwrap()
        });
        let root_key = keys[median].value.clone();
        let mut root = Some(Box::new(Node::new(
            root_key,
            keys[median].number,
            Some(dim_max_spread),
        )));
        build_tree(&mut root, ChildType::Left, &mut keys, 0, median, dimension);
        build_tree(
            &mut root,
            ChildType::Right,
            &mut keys,
            median + 1,
            len,
            dimension,
        );
        Some(Self { root, len })
    }
    /// Searches the nearest neighbors of a `key` in the tree.
    ///
    /// The distance metric is provided by the caller.
    ///
    /// It returns a max oriented binary heap collecting the k nearest neighbors
    /// of `key` located in the tree. It means that the top element of the heap
    /// (whose reference is accessible in O(1) running time) is the furthest
    /// from `key`.
    ///
    /// Adapted from the paper: [Parallel k Nearest Neighbor Graph Construction
    /// Using Tree-Based Data Structures][paper].
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
    /// assert_eq!(2, knn.delete().unwrap().point); // Third point inserted c, is the third closest to key.
    /// assert_eq!(0, knn.delete().unwrap().point); // First point inserted a, is the third closest to key.
    /// assert_eq!(5, knn.delete().unwrap().point); // Sixth point inserted f, is the second closest to key.
    /// assert_eq!(4, knn.delete().unwrap().point); // Fifth point inserted e, is closest to key
    /// ```
    ///
    /// [paper]: http://dx.doi.org/10.5821/hpgm15.1
    pub fn k_nearest_neighbors<D>(
        &self,
        key: &K,
        k: usize,
        distance: D,
    ) -> Option<BinaryHeap<KthNearestNeighbor<usize, K::Elem>>>
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
}

#[derive(Debug)]
enum ChildType {
    Left,
    Right,
}
fn build_tree<K>(
    node: &mut Option<Box<Node<K>>>,
    child_type: ChildType,
    keys: &mut Vec<Point<K>>,
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
        let (number, key) = (&keys[start].number, &keys[start].value);
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
    let dim_max_spread = find_dim_max_spread(&*keys, start, end, dimension);
    keys[start..end].sort_by(|a, b| {
        a.value[dim_max_spread]
            .partial_cmp(&b.value[dim_max_spread])
            .unwrap()
    });
    let median = start + (end - start) / 2;
    match child_type {
        ChildType::Left => {
            if let Some(ref mut nod) = node {
                nod.left = Some(Box::new(Node::new(
                    keys[median].value.clone(),
                    keys[median].number,
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
                    keys[median].value.clone(),
                    keys[median].number,
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

fn find_dim_max_spread<K>(keys: &[Point<K>], start: usize, end: usize, dimension: usize) -> usize
where
    K: Clone + Index<usize> + Debug,
    K::Output: PartialOrd + Zero + Copy + Sub<K::Output, Output = K::Output> + Debug,
{
    let (mut dim_max_spread, mut max_spread) = (0, K::Output::zero());
    for dim in 0..dimension {
        let (mut min, mut max) = (keys[start].value[dim], keys[start].value[dim]);
        for key in keys.iter().take(end).skip(start + 1) {
            let val = key.value[dim];
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
    mut the_bests: BinaryHeap<KthNearestNeighbor<usize, K::Elem>>,
    k: usize,
    distance: &D,
) -> BinaryHeap<KthNearestNeighbor<usize, K::Elem>>
where
    K: Index<usize, Output = K::Elem> + Algebra<LenghtOutput = usize> + Debug,
    K::Elem: PartialOrd + Copy + Sub<Output = K::Elem> + Mul<Output = K::Elem> + Debug,
    D: Fn(&K, &K) -> K::Elem,
{
    if let Some(nod) = node {
        let dist = distance(&nod.key, key);
        if the_bests.len() < k {
            the_bests.insert(KthNearestNeighbor {
                point: nod.number,
                dist,
            });
        } else if dist < the_bests.maximum().unwrap().dist {
            // .unwrap() is safe here as long as k >= 1 because when k >= 1 the heap is not
            // empty, which guaranties the existence of a maximum.
            the_bests.delete();
            the_bests.insert(KthNearestNeighbor {
                point: nod.number,
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

#[test]
fn kd() {
    use ndarray::array;
    let v = [
        array![5., 4.],
        array![2., 6.],
        array![13., 3.],
        array![3., 1.],
        array![10., 2.],
        array![8., 7.],
    ];
    let k = KdTree::from(v).unwrap();
    print!("{:#?}", k);
    let mut neighbors = k
        .k_nearest_neighbors(&array![9., 4.], 4, |a, b| (a - b).minkowsky(2.))
        .unwrap();
    print!("{:#?}", neighbors.delete());
    print!("{:#?}", neighbors.delete());
    print!("{:#?}", neighbors.delete());
    print!("{:#?}", neighbors.delete());
}
