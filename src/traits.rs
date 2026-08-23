use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MathError {
    DimensionMismatch {
        expected: (usize, usize),
        found: (usize, usize),
    },
    IncompatibleInnerDimensions {
        lhs: (usize, usize),
        rhs: (usize, usize),
    },
    NotSquare {
        rows: usize,
        cols: usize,
    },
    SingularMatrix,
    InvalidPolDeg(usize),
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathError::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}x{}, found {}x{}",
                    expected.0, expected.1, found.0, found.1
                )
            }
            MathError::IncompatibleInnerDimensions { lhs, rhs } => {
                write!(
                    f,
                    "Incompatible inner dimensions for gemm: {}x{} and {}x{}",
                    lhs.0, lhs.1, rhs.0, rhs.1
                )
            }
            MathError::NotSquare { rows, cols } => {
                write!(f, "Expected square matrix, but got {}x{}", rows, cols)
            }
            MathError::SingularMatrix => {
                write!(f, "Matrix is singular (non-invertible)")
            }
            MathError::InvalidPolDeg(deg) => {
                write!(f, "Invalid polynomial degree {}", deg)
            }
        }
    }
}

impl std::error::Error for MathError {}

pub trait MatrixOps: Sized + Clone {
    type Scalar: Copy;

    fn shape(&self) -> (usize, usize);

    /// Zeros Matrix (Square n x n)
    fn zeros(size: usize) -> Self;

    /// Identity Matrix (Square n x n)
    fn identity(size: usize) -> Self;

    /// Diagonal Matrix with `val` in the diagonal (Square n x n)
    fn diag(size: usize, val: Self::Scalar) -> Self;

    /// Scale matrix by a scalar factor: M = alpha * M
    fn scale_assign(&mut self, alpha: Self::Scalar);

    /// AXPY: Y = alpha * X + Y
    fn axpy(&mut self, alpha: Self::Scalar, x: &Self);

    /// AXPY: self = alpha * X + Y
    fn from_axpy(&mut self, alpha: Self::Scalar, x: &Self, y: &Self);

    /// General Matrix Multiplication: C = alpha * (A * B) + beta * C
    fn gemm(&mut self, alpha: Self::Scalar, a: &Self, b: &Self, beta: Self::Scalar);

    /// In-place Solves A * X = B, overwriting B with X
    fn solve_in_place(&self, b: &mut Self) -> Result<(), MathError>;

    /// copy the data from other matrix in this existing instance
    fn copy_from(&mut self, x: &Self);
}
