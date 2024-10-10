use core::ops::{Add, Index, Mul, Sub};

use num_traits::{Float, FromPrimitive, One, Zero};

use crate::neighbors::KthNearestNeighbor;
use crate::traits::{Algebra, Container};

use super::BinaryHeap;
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

#[derive(Clone, Debug)]
struct Node<K, T> {
    points: Option<Vec<Point<K>>>,
    pivot: Option<K>,
    radius: Option<T>,
    child1: Option<Box<Node<K, T>>>,
    child2: Option<Box<Node<K, T>>>,
}

impl<K, T> Node<K, T> {
    pub fn new() -> Self {
        Self {
            points: None,
            pivot: None,
            radius: None,
            child1: None,
            child2: None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.points.is_some()
            & self.child1.is_none()
            & self.child2.is_none()
            & self.pivot.is_none()
            & self.radius.is_none()
    }
}

#[derive(Debug)]
pub struct BallTree<K: Container> {
    root: Option<Box<Node<K, K::Elem>>>,
    len: usize,
}
impl<K: Container> Default for BallTree<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Container> BallTree<K> {
    /// Creates an empty tree instance.
    ///
    /// # Example
    /// ```
    /// use ndarray::Array1;
    /// use njang::prelude::*;
    /// let bt = BallTree::<Array1<f32>>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// Gives the number of keys in the tree.
    ///
    /// # Example
    /// ```
    /// use ndarray::Array1;
    /// use njang::prelude::*;
    /// let bt = BallTree::<Array1<f32>>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether or not the tree is empty.
    /// # Example
    /// ```
    /// use ndarray::Array1;
    /// use njang::prelude::*;
    /// let mut bt = BallTree::<Array1<f32>>::new();
    /// assert!(bt.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Builds a tree from an iterator of points.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use njang::prelude::*;
    /// let points = [array![0f32, 1.], array![1., 1.]];
    /// let tree = BallTree::<_>::from(points, |a, b| (a - b).minkowsky(2.), 2);
    /// println!("{:#?}", tree);
    /// ```
    pub fn from<D, Keys>(keys: Keys, distance: D, leaf_size: usize) -> Option<Self>
    where
        Keys: IntoIterator<Item = K>,
        K: Clone + Index<usize> + core::fmt::Debug + Algebra,
        K::Elem: Float + FromPrimitive + Mul<K, Output = K> + Zero,
        for<'a> K: Add<&'a K, Output = K> + Clone,
        for<'a> &'a K: Sub<&'a K, Output = K>,
        D: Fn(&K, &K) -> K::Elem,
    {
        let points = keys
            .into_iter()
            .enumerate()
            .map(|(number, value)| Point { number, value })
            .collect::<Vec<_>>();
        if points.is_empty() {
            return None;
        }
        let len = points.len();
        let root = Some(Box::new(build_tree(
            Node::new(),
            points,
            &distance,
            leaf_size,
        )));
        Some(Self { root, len })
    }

    pub fn k_nearest_neighbors<D>(
        &self,
        key: &K,
        k: usize,
        distance: D,
    ) -> Option<BinaryHeap<KthNearestNeighbor<Point<K>, K::Elem>>>
    where
        K: Container + core::fmt::Debug + PartialEq + Clone,
        K::Elem: Float,
        D: Fn(&K, &K) -> K::Elem,
    {
        if self.root.is_none() | (k == 0) {
            return None;
        }
        Some(k_nearest_neighbors(
            self.root.as_ref().unwrap(),
            key,
            BinaryHeap::with_capacity(k + 1),
            k,
            &distance,
        ))
    }
}

fn centroid<K>(points: &[Point<K>]) -> Option<K>
where
    K::Elem: Float + FromPrimitive + Mul<K, Output = K>,
    for<'a> K: Container + Add<&'a K, Output = K> + Clone,
{
    if points.is_empty() {
        return None;
    }
    let inv_len = K::Elem::one() / K::Elem::from_usize(points.len()).unwrap();
    points
        .iter()
        .fold(None, |accumulator, x| {
            if let Some(acc) = accumulator {
                Some(acc + &x.value)
            } else {
                Some(x.value.clone())
            }
        })
        .map(|sum_points| inv_len * sum_points)
}

fn fursthest_from<'a, D, K: Container>(
    pivot: &K,
    points: &'a [Point<K>],
    distance: D,
) -> (K::Elem, Option<&'a Point<K>>)
where
    D: Fn(&K, &K) -> K::Elem,
    K::Elem: Float + Zero,
{
    let mut child = None;
    let mut radius = K::Elem::zero();
    for point in points {
        let dist = distance(&point.value, pivot);
        if dist > radius {
            radius = dist;
            child = Some(point);
        }
    }
    (radius, child)
}
fn build_tree<D, K>(
    mut node: Node<K, K::Elem>,
    points: Vec<Point<K>>,
    distance: &D,
    leaf_size: usize,
) -> Node<K, K::Elem>
where
    K: Clone + Index<usize> + core::fmt::Debug + Algebra,
    K::Elem: Float + FromPrimitive + Mul<K, Output = K> + Zero,
    for<'a> K: Add<&'a K, Output = K> + Clone,
    for<'a> &'a K: Sub<&'a K, Output = K>,
    D: Fn(&K, &K) -> K::Elem,
{
    let pivot = centroid(&points).unwrap();
    let (radius, child1) = fursthest_from(&pivot, &points, distance);
    node.pivot = Some(pivot);
    node.radius = Some(radius);
    // println!("{:#?}\n", root);
    let child2 = fursthest_from(&child1.unwrap().value, &points, distance).1;
    let (child1, child2) = (child1.unwrap().clone(), child2.unwrap().clone());
    let mut child1_points = Vec::with_capacity(1 + points.len() / 2);
    let mut child2_points = Vec::with_capacity(1 + points.len() / 2);
    for point in points {
        if distance(&child1.value, &point.value) <= distance(&child2.value, &point.value) {
            child1_points.push(point);
        } else {
            child2_points.push(point);
        }
    }
    if (child1_points.len() <= leaf_size) && !child1_points.is_empty() {
        let mut leaf = Node::new();
        leaf.points = Some(child1_points);
        node.child1 = Some(Box::new(leaf));
    } else if !child1_points.is_empty() {
        let child = build_tree(Node::new(), child1_points, distance, leaf_size);
        node.child1 = Some(Box::new(child));
    }
    if (child2_points.len() <= leaf_size) && !child2_points.is_empty() {
        let mut leaf = Node::new();
        leaf.points = Some(child2_points);
        node.child2 = Some(Box::new(leaf));
    } else if !child2_points.is_empty() {
        let child = build_tree(Node::new(), child2_points, distance, leaf_size);
        node.child2 = Some(Box::new(child));
    }
    node
}

fn k_nearest_neighbors<K, D>(
    node: &Node<K, K::Elem>,
    key: &K,
    mut the_bests: BinaryHeap<KthNearestNeighbor<Point<K>, K::Elem>>,
    k: usize,
    distance: &D,
) -> BinaryHeap<KthNearestNeighbor<Point<K>, K::Elem>>
where
    K: Container + core::fmt::Debug + PartialEq + Clone,
    K::Elem: Float,
    D: Fn(&K, &K) -> K::Elem,
{
    if !the_bests.is_empty() && !node.is_leaf() {
        let dist_key_from_node = distance(key, node.pivot.as_ref().unwrap()) - node.radius.unwrap();
        if dist_key_from_node >= distance(key, &the_bests.maximum().unwrap().point.value) {
            return the_bests;
        }
    }
    if node.is_leaf() {
        for point in node.points.as_ref().unwrap() {
            let dist = distance(key, &point.value);
            if !the_bests.is_empty() {
                if dist < distance(key, &the_bests.maximum().unwrap().point.value) {
                    the_bests.insert(KthNearestNeighbor {
                        point: point.clone(),
                        dist,
                    });
                }
            } else {
                the_bests.insert(KthNearestNeighbor {
                    point: point.clone(),
                    dist,
                });
            }
            if the_bests.len() > k {
                the_bests.delete();
            }
        }
        return the_bests;
    }
    // if let Some(ref pivot) = node.pivot{
    if let Some(ref child1) = node.child1 {
        the_bests = k_nearest_neighbors(child1, key, the_bests, k, distance);
    }
    if let Some(ref child2) = node.child2 {
        the_bests = k_nearest_neighbors(child2, key, the_bests, k, distance);
    }
    // };
    the_bests
}

#[test]
fn ball() {
    use ndarray::*;
    let points = [
        array![5., 4.],
        array![2., 6.],
        array![13., 3.],
        array![3., 1.],
        array![10., 2.],
        array![8., 7.],
    ];
    let tree = BallTree::<Array1<f32>>::from(points, |a, b| (a - b).minkowsky(2.), 2).unwrap();
    println!("{:#?}", tree);
    let knn = tree.k_nearest_neighbors(&array![9., 4.], 4, |a, b| (a - b).minkowsky(2.));
    println!("\n{:#?}", knn);
}
