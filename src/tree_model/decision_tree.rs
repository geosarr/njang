pub struct DecisionNode<T> {
    pub feature: usize,
    pub threshold: T,
    pub value: T,
    pub left: Box<DecisionNode<T>>,
    pub right: Box<DecisionNode<T>>,
}

pub struct DecisionTreeSettings<T> {
    pub min_samples_split: T,
    pub min_impurity: T,
    pub max_depth: T,
    pub impurity_calculation: T,
    pub leaf_value_calculation: T,
    pub one_dim: T,
    // pub loss: T,
}
