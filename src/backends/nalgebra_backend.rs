//! [`MatrixOps`] backend implementation for [`nalgebra`].
//!
//! Enabled by the `nalgebra-backend` feature flag.

use crate::traits::{MathError, MatrixOps};
use nalgebra::{DefaultAllocator, Dim, DimMin, OMatrix, allocator::Allocator};

impl<D> MatrixOps for OMatrix<f64, D, D>
where
    D: Dim + DimMin<D, Output = D>,
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    type Scalar = f64;

    #[inline]
    fn copy_from(&mut self, x: &Self) {
        Self::copy_from(self, x);
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        Self::shape(&self)
    }

    #[inline]
    fn identity(size: usize) -> Self {
        Self::identity_generic(D::from_usize(size), D::from_usize(size))
    }

    #[inline]
    fn zeros(size: usize) -> Self {
        Self::zeros_generic(D::from_usize(size), D::from_usize(size))
    }

    #[inline]
    fn diag(size: usize, val: Self::Scalar) -> Self {
        Self::from_diagonal_element_generic(D::from_usize(size), D::from_usize(size), val)
    }

    #[inline]
    fn scale_assign(&mut self, alpha: Self::Scalar) {
        self.scale_mut(alpha);
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

        for (s_val, x_val) in self.iter_mut().zip(x.iter()) {
            *s_val += alpha * x_val;
        }
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

        OMatrix::gemm(self, alpha, a, b, beta);
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

        // `.lu()` consumes the matrix, so we `.clone()` our `&self` reference.
        // It computes the LU decomposition with partial pivoting.
        // `.solve_mut(b)` applies the result directly into `b` with zero extra allocations.
        if self.clone().lu().solve_mut(b) {
            return Ok(());
        }
        // Matrix is singular / not invertible
        Err(MathError::DimensionMismatch {
            expected: self.shape(),
            found: (0, 0), // Note: Consider adding a SingularMatrix variant to MathError
        })
    }
}
