pub struct DecisionNode<T> {
    pub feature: usize,
    pub threshold: T,
    pub value: T,
    pub left: Box<DecisionNode<T>>,
    pub right: Box<DecisionNode<T>>,
}

pub struct DecisionTreeSettings<T> {}
