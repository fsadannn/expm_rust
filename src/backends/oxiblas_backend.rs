use crate::traits::{MathError, MatrixOps};
#[cfg(feature = "oxiblas-backend")]
use oxiblas::prelude::*;

#[cfg(feature = "oxiblas-backend")]
impl MatrixOps for Mat<f64> {
    type Scalar = f64;

    #[inline]
    fn copy_from(&mut self, x: &Self) {
        copy(x.raw_data(), self.raw_data_mut());
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    #[inline]
    fn identity(size: usize) -> Self {
        MatBuilder::identity(size)
    }

    #[inline]
    fn zeros(size: usize) -> Self {
        MatBuilder::zeros(size, size)
    }

    #[inline]
    fn diag(size: usize, val: Self::Scalar) -> Self {
        let mut m = MatBuilder::zeros(size, size);
        for i in 0..size {
            m[(i, i)] = val;
        }
        m
    }

    #[inline]
    fn scale_assign(&mut self, alpha: Self::Scalar) {
        scal(alpha, self.raw_data_mut());
    }

    #[inline]
    fn axpy(&mut self, alpha: Self::Scalar, x: &Self) {
        debug_assert_eq!(
            self.shape(),
            x.shape(),
            "{}",
            MathError::DimensionMismatch {
                expected: self.shape(),
                found: x.shape(),
            }
        );

        axpy(alpha, x.raw_data(), self.raw_data_mut());
    }

    #[inline]
    fn from_axpy(&mut self, alpha: Self::Scalar, x: &Self, y: &Self) {
        debug_assert_eq!(
            self.shape(),
            x.shape(),
            "{}",
            MathError::DimensionMismatch {
                expected: self.shape(),
                found: x.shape(),
            }
        );
        copy(y.raw_data(), self.raw_data_mut());
        axpy(alpha, x.raw_data(), self.raw_data_mut());
    }

    #[inline]
    fn gemm(&mut self, alpha: Self::Scalar, a: &Self, b: &Self, beta: Self::Scalar) {
        let (m, _) = a.shape();
        let (_, n) = b.shape();
        debug_assert_eq!(
            a.shape(),
            b.shape(),
            "{}",
            MathError::IncompatibleInnerDimensions {
                lhs: a.shape(),
                rhs: b.shape(),
            }
        );

        debug_assert_eq!(
            self.shape(),
            (m, n),
            "{}",
            MathError::DimensionMismatch {
                expected: (m, n),
                found: self.shape(),
            }
        );

        gemm(alpha, a.as_ref(), b.as_ref(), beta, self.as_mut());
    }

    #[inline]
    fn solve_in_place(&self, b: &mut Self) -> Result<(), MathError> {
        let (m, n) = self.shape();
        if m != n {
            return Err(MathError::DimensionMismatch {
                expected: (m, m),
                found: (m, n),
            });
        }

        let lu = Lu::compute(self.as_ref()).map_err(|_| MathError::SingularMatrix)?;

        let x_sol = lu
            .solve(b.as_ref())
            .map_err(|_| MathError::SingularMatrix)?;

        b.as_mut().copy_from(&x_sol.as_ref());
        Ok(())
    }
}
