use core::ops::{Add, Div, Index, Mul, Sub};
use std::process::Output;

use num_traits::{Float, FromPrimitive, One, Zero};

use crate::neighbors::KthNearestNeighbor;
use crate::traits::{Algebra, Container};
#[derive(Debug, Clone)]
struct Point<K> {
    number: usize,
    value: K,
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
    root: Option<Node<K, K::Elem>>,
    len: usize,
}

impl<K: Container> BallTree<K> {
    /// Creates an empty tree instance.
    ///
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let bt = BallTree::<[usize; 1]>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// Gives the number of keys in the tree.
    ///
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let bt = BallTree::<[f32; 2]>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether or not the tree is empty.
    /// # Example
    /// ```
    /// use njang::prelude::*;
    /// let mut bt = BallTree::<[f32; 3]>::new();
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
        let root = Some(build_tree(Node::new(), points, &distance, leaf_size));
        Some(Self { root, len })
    }
}

fn centroid<K: Container>(points: &[Point<K>]) -> Option<K>
where
    K::Elem: Float + FromPrimitive + Mul<K, Output = K>,
    for<'a> K: Add<&'a K, Output = K> + Clone,
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
        let dist = distance(&point.value, &pivot);
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
    let mut radius = K::Elem::zero();
    let (radius, child1) = fursthest_from(&pivot, &points, &distance);
    node.pivot = Some(pivot);
    node.radius = Some(radius);
    // println!("{:#?}\n", root);
    let child2 = fursthest_from(&child1.unwrap().value, &points, &distance).1;
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
        node.child2 = Some(Box::new(build_tree(
            Node::new(),
            child2_points,
            distance,
            leaf_size,
        )));
    }
    return node;
}

#[test]
fn ball() {
    use ndarray::*;
    let points = vec![array![0f32, 1.], array![1., 1.]];
    println!("{:?}", points);
    let tree = BallTree::<Array1<f32>>::from(points, |a, b| (a - b).minkowsky(2.), 2);
    println!("{:#?}", tree);
}
