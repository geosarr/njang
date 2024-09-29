use ndarray::*;

use core::{cmp::Ordering, ops::Index};

use crate::{
    error::NjangError,
    traits::{Container, Label, Model, Scalar},
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
struct Node<K, V> {
    key: K,
    value: V,
    left: Option<Box<Node<K, V>>>,
    right: Option<Box<Node<K, V>>>,
}

impl<K, V> Node<K, V> {
    pub fn init(key: K, value: V) -> Self {
        Self {
            key,
            value,
            left: None,
            right: None,
        }
    }
}

///// Implementation of a Kd-tree
///// # Example
///// ```
///// use njang::KdTree;
///// let mut bt = KdTree::new();
///// bt.insert([0], "1");
///// bt.insert([1], "2");
///// bt.insert([2], "3");
///// assert_eq!(bt.len(), 3);
///// assert!(bt.contains(&[0]));
///// assert_eq!(bt.get(&[2]), Some(&"3"));
///// ```
#[derive(Debug, Clone)]
pub struct KdTree<K, V>
where
    K: Container,
{
    root: Option<Box<Node<K, V>>>,
    len: usize,
}
impl<K: Container, V> Default for KdTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<K: Container, V> KdTree<K, V> {
    /// Creates an empty tree instance.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::<[usize; 1], isize>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }
    /// Creates a new tree with an initial (key, value) pair.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::init(["btree"], 0);
    /// assert_eq!(bt.len(), 1);
    /// ```
    pub fn init(key: K, value: V) -> Self {
        Self {
            root: Some(Box::new(Node::init(key, value))),
            len: 1,
        }
    }
    /// Gives the number of (key, value) pairs in the tree.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::<[f32; 2], f32>::new();
    /// assert_eq!(bt.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }
    // /// Tests whether or not the tree is empty.
    // /// # Example
    // /// ```
    // /// use njang::KdTree;
    // /// let mut bt = KdTree::new();
    // /// bt.insert(1, 1);
    // /// assert!(!bt.is_empty());
    // /// ```
    // pub fn is_empty(&self) -> bool {
    //     self.len() == 0
    // }
}
impl<K, V> KdTree<K, V>
where
    K: Index<usize> + Container<LenghtOutput = usize>,
    K::Output: PartialOrd + Copy,
{
    /// Tests whether or not the tree contains a given key.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::init(["btree"], "one");
    /// assert!(bt.contains(&["btree"]));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        if let Some(_) = self.get(key) {
            true
        } else {
            false
        }
    }
    /// Returns a reference of the value associated to a key if any exists in
    /// the tree. Returns `None` otherwise.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let bt = KdTree::init(["btree"], "one");
    /// assert_eq!(bt.get(&["no btree"]), None);
    /// assert_eq!(bt.get(&["btree"]), Some(&"one"));
    /// ```
    pub fn get(&self, key: &K) -> Option<&V> {
        // gets the value associated to key if key is in
        // the tree, otherwise returns None,
        // run time complexity on average O(log(N)), O(N) guaranteed (unbalanced tree)
        let mut node = &self.root;
        let mut level = 0;
        while node.is_some() {
            let temp_node = node.as_ref().unwrap();
            match key[level].partial_cmp(&temp_node.key[level]) {
                Some(Ordering::Less) => node = &temp_node.left,
                Some(Ordering::Greater) => node = &temp_node.right,
                Some(Ordering::Equal) => return Some(&temp_node.value),
                None => return None, //panic!("Unknown situation"),
            }
            level = (level + 1) % key.length();
        }
        None
    }
    fn put<'a>(
        node: &'a mut Option<Box<Node<K, V>>>,
        key: K,
        value: V,
        level: &mut usize,
    ) -> Option<&'a mut Box<Node<K, V>>> {
        match node {
            None => *node = Some(Box::new(Node::init(key, value))),
            Some(ref mut nod) => match key[*level].partial_cmp(&nod.key[*level]) {
                Some(Ordering::Less) => {
                    *level += 1;
                    return Self::put(&mut nod.left, key, value, level);
                }
                Some(Ordering::Greater) => {
                    *level += 1;
                    return Self::put(&mut nod.right, key, value, level);
                }
                Some(Ordering::Equal) => {
                    nod.value = value;
                    return Some(nod);
                }
                None => return None, //panic!("Unknown situation"),
            },
        }
        None
    }
    /// Inserts a (key, value) pair in the tree. When the input key is
    /// already on the map, then it replaces the old value with the new one
    /// specified.
    /// # Example
    /// ```
    /// use njang::KdTree;
    /// let mut bt = KdTree::<[isize; 1], usize>::new();
    /// bt.insert([-1], 2);
    /// bt.insert([-2], 3);
    /// bt.insert([-1], 4);
    /// assert_eq!(bt.len(), 2);
    /// //assert_eq!(bt.get(&[-2]), Some(&3));
    /// ```
    pub fn insert(&mut self, key: K, value: V) {
        let mut level = 0;
        if Self::put(&mut self.root, key, value, &mut level).is_none() {
            self.len += 1;
        }
    }
}

// #[derive(PartialEq)]
// struct Predictor<X> {
//     x: X,
//     level: usize,
// }

// impl<X> PartialOrd for Predictor<X>
// where
//     X: Index<usize> + PartialEq,
//     X::Output: PartialOrd,
// {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         self.x[other.level].partial_cmp(&other.x[other.level])
//     }
// }

// pub struct KdTree<X, Y> {
//     tree: KdTree<X, Y>,
// }

// impl KdTree<Predictor<Array1<f32>>, Array1<f32>> {
//     pub fn new() -> Self {
//         Self {
//             tree: KdTree::new(),
//         }
//     }
//     pub fn from_pair(x: Array1<f32>, y: Array1<f32>) -> Self {
//         let mut tree = KdTree::new();
//         tree.insert(Predictor { x, level: 0 }, y);
//         Self { tree }
//     }
// }
#[test]
fn partial() {
    let mut bt = KdTree::<Array1<isize>, usize>::new();
    bt.insert(array![-1], 2);
    bt.insert(array![-2], 3);
    bt.insert(array![-1], 4);
    println!("{:?}", bt.len());
    println!("{:?}", bt.get(&array![-1]));
    // let x = array![2., 3., -10.];
    // let p = Predictor { x: &x, level: 1 };

    // let q = Predictor {
    //     x: &array![1., -1., 0.],
    //     level: 2,
    // };
    // assert!(q < p);
}
