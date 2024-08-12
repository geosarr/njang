use ndarray::{Array1, Array2};

macro_rules! diff {
    ($name:ident, $arr:ty) => {
        pub(crate) fn $name<T>(a: &$arr, b: &$arr) -> T
        where
            for<'a> &'a T: core::ops::Sub<Output = T>,
            T: num_traits::Float + core::ops::Mul<Output = T> + core::iter::Sum,
        {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<T>()
                .sqrt()
        }
    };
}
diff!(l2_diff, Array1<T>);
diff!(l2_diff2, Array2<T>);
