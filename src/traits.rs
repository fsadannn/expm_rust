use std::fmt;

/// Errors that can occur during matrix operations and Padé matrix exponential computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MathError {
    /// Raised when matrix dimensions do not match the expected dimensions.
    DimensionMismatch {
        /// Expected `(rows, cols)` shape.
        expected: (usize, usize),
        /// Found `(rows, cols)` shape.
        found: (usize, usize),
    },
    /// Raised during matrix multiplication (GEMM) when the column dimension of the left-hand matrix
    /// does not equal the row dimension of the right-hand matrix.
    IncompatibleInnerDimensions {
        /// Left-hand matrix `(rows, cols)` shape.
        lhs: (usize, usize),
        /// Right-hand matrix `(rows, cols)` shape.
        rhs: (usize, usize),
    },
    /// Raised when a matrix is required to be square ($N \times N$), but has non-equal dimensions.
    NotSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
    /// Raised when solving a linear system $A X = B$ and the matrix $A$ is singular or non-invertible.
    SingularMatrix,
    /// Raised when an unsupported Padé polynomial degree is requested.
    ///
    /// Currently supported degrees are $3, 4, 5, 6, 7, 9, 13$.
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

/// Core abstraction trait defining the fundamental linear algebra primitives required
/// for Padé matrix exponential evaluation.
///
/// Any matrix type implementing `MatrixOps` can be used directly with [`crate::pade_pp`]
/// and [`crate::pade_pp_f64`].
///
/// # Requirements
/// Implementors must provide:
/// - Basic constructors: zero matrix ([`zeros`](MatrixOps::zeros)), identity matrix ([`identity`](MatrixOps::identity)), and diagonal matrix ([`diag`](MatrixOps::diag)).
/// - In-place scalar scaling ([`scale_assign`](MatrixOps::scale_assign)).
/// - BLAS Level-1 AXPY operations ([`axpy`](MatrixOps::axpy) and [`from_axpy`](MatrixOps::from_axpy)).
/// - BLAS Level-3 General Matrix Multiplication ([`gemm`](MatrixOps::gemm)): $C \leftarrow \alpha (A B) + \beta C$.
/// - Linear system solver ([`solve_in_place`](MatrixOps::solve_in_place)): solves $A X = B$ overwriting $B$ with $X$.
/// - Deep copy buffer reuse ([`copy_from`](MatrixOps::copy_from)).
pub trait MatrixOps: Sized + Clone {
    /// The scalar element type of the matrix (e.g., `f64`, `f32`, `f128`, etc.).
    type Scalar: Copy;

    /// Returns the shape of the matrix as `(rows, cols)`.
    fn shape(&self) -> (usize, usize);

    /// Creates an $N \times N$ square zero matrix.
    ///
    /// # Arguments
    /// * `size` - Dimension $N$ of the square matrix.
    fn zeros(size: usize) -> Self;

    /// Creates an $N \times N$ square identity matrix $I$.
    ///
    /// # Arguments
    /// * `size` - Dimension $N$ of the square matrix.
    fn identity(size: usize) -> Self;

    /// Creates an $N \times N$ square diagonal matrix with scalar `val` along the main diagonal:
    ///
    /// $$M = \text{diag}(\text{val}, \dots, \text{val})$$
    ///
    /// # Arguments
    /// * `size` - Dimension $N$ of the square matrix.
    /// * `val` - Value to place on the diagonal entries.
    fn diag(size: usize, val: Self::Scalar) -> Self;

    /// In-place scalar multiplication: $M \leftarrow \alpha M$.
    ///
    /// # Arguments
    /// * `alpha` - Scalar factor to multiply every element of `self`.
    fn scale_assign(&mut self, alpha: Self::Scalar);

    /// In-place vector/matrix addition (AXPY):
    ///
    /// $$Y \leftarrow \alpha X + Y$$
    ///
    /// where $Y$ is `self`.
    ///
    /// # Arguments
    /// * `alpha` - Scalar multiplier.
    /// * `x` - Matrix $X$ to scale and add into `self`.
    fn axpy(&mut self, alpha: Self::Scalar, x: &Self);

    /// Assigns the result of AXPY into `self` without accumulating into existing values:
    ///
    /// $$\text{self} \leftarrow \alpha X + Y$$
    ///
    /// # Arguments
    /// * `alpha` - Scalar multiplier for $X$.
    /// * `x` - Matrix $X$.
    /// * `y` - Matrix $Y$.
    fn from_axpy(&mut self, alpha: Self::Scalar, x: &Self, y: &Self) {
        self.copy_from(y);
        self.axpy(alpha, x);
    }

    /// General Matrix Multiplication (GEMM):
    ///
    /// $$C \leftarrow \alpha (A \cdot B) + \beta C$$
    ///
    /// where $C$ is `self`.
    ///
    /// # Arguments
    /// * `alpha` - Scalar multiplier for matrix product $(A \cdot B)$.
    /// * `a` - Left-hand matrix $A$.
    /// * `b` - Right-hand matrix $B$.
    /// * `beta` - Scalar multiplier for matrix $C$ (`self`).
    fn gemm(&mut self, alpha: Self::Scalar, a: &Self, b: &Self, beta: Self::Scalar);

    /// Solves the linear system $A X = B$ in place, overwriting $B$ with the solution $X$.
    ///
    /// Here `self` is matrix $A$ and `b` is matrix $B$.
    ///
    /// # Arguments
    /// * `b` - Right-hand side matrix $B$, which is overwritten with the solution $X$.
    ///
    /// # Errors
    /// Returns [`MathError::SingularMatrix`] if the system cannot be inverted,
    /// or [`MathError::DimensionMismatch`] if dimensions are incompatible.
    fn solve_in_place(&self, b: &mut Self) -> Result<(), MathError>;

    /// Copies data from matrix `x` into `self` (overwriting existing content in `self`).
    ///
    /// # Arguments
    /// * `x` - Source matrix to copy from.
    fn copy_from(&mut self, x: &Self);
}
