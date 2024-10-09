//! Neighboors finding algorithms.

use core::mem::replace;
mod ball_tree;
pub use ball_tree::*;

mod kd_tree;
pub use kd_tree::*;

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
