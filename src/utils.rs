use ndarray::{Array1, Array2};

macro_rules! diff {
    ($name_diff:ident, $name_norm:ident, $arr:ty) => {
        pub(crate) fn $name_diff<T>(a: &$arr, b: &$arr) -> T
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
        pub(crate) fn $name_norm<T>(a: &$arr) -> T
        where
            T: num_traits::Float + core::iter::Sum,
        {
            a.map(|x| x.powi(2)).sum().sqrt()
        }
    };
}
diff!(l2_diff, l2_norm1, Array1<T>);
diff!(l2_diff2, l2_norm2, Array2<T>);
