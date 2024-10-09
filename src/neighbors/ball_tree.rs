use core::hash::Hash;

use hashbrown::HashSet;

use crate::traits::Container;

#[derive(Debug, Clone, Eq)]
struct Point<K> {
    number: usize,
    value: K,
}
impl<K> Hash for Point<K> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.number.hash(state);
    }
}

impl<K> PartialEq for Point<K> {
    fn eq(&self, other: &Self) -> bool {
        self.number.eq(&other.number)
    }
}

#[derive(Clone, Debug)]
struct Node<K: Container> {
    data: HashSet<Point<K>>,
    pivot: K,
    radius: K::Elem,
    child1: HashSet<Point<K>>,
    child2: HashSet<Point<K>>,
}

// fn build_tree<K>(node: &mut Node<K>, points: &[Point<K>], leaf_size: usize) {
//     node.data =
// }

// impl<K> Node<K> {
//     pub fn new(key: K, number: usize, coordinate: Option<usize>) -> Self {
//         Self {
//             key,
//             number,
//             coordinate,
//             left: None,
//             right: None,
//         }
//     }
// }
