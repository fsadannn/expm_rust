//! # expm_rust
//!
//! `expm_rust` is a high-performance, backend-agnostic Rust library for computing the **matrix exponential**
//! ($\exp(A) = e^A$) using the **scaling and squaring method** with **diagonal $[p/p]$ Padé rational approximants**.
//!
//! ## Key Features
//!
//! - **Generic Backend Architecture**: Designed around the [`MatrixOps`] trait. Bring your own matrix library
//!   (such as `nalgebra`, `faer`, `ndarray`, custom SIMD buffers, or fixed arrays) by simply implementing the trait.
//! - **Built-in `oxiblas` Backend**: Ready-to-use, high-performance BLAS/SIMD implementation for [`oxiblas::prelude::Mat<f64>`]
//!   under the `oxiblas-backend` feature flag (enabled by default).
//! - **Optimized Polynomial Evaluations**: Supports Padé degrees $p \in \{3, 4, 5, 6, 7, 9, 13\}$ using Paterson-Stockmeyer
//!   and Horner matrix evaluation schemes that minimize matrix-matrix multiplications.
//! - **Compile-Time Exact Rational Coefficients**: Padé coefficients are computed at compile time as exact fractions
//!   using 128-bit integer arithmetic ([`pade_coef::Rational`]), completely avoiding floating-point error in coefficient definitions.
//! - **Extensible to `f128`, `f32`, and Custom Precision**: Easily compute exact Padé coefficients for 128-bit floats
//!   or arbitrary precision scalars using [`pade_coef::compute_pade_array`].
//! - **Minimal Allocations**: Reuses matrix buffers in-place during scaling, polynomial accumulation, and squaring stages.
//!
//! ---
//!
//! ## Mathematical Method
//!
//! For a square matrix $A \in \mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$):
//!
//! 1. **Scaling**: Choose a non-negative integer $s$ such that $\|2^{-s} A\|$ is sufficiently small, and scale:
//!    $$A_s = 2^{-s} A$$
//!
//! 2. **Padé Approximation**: Approximate $e^{A_s} \approx R_{p,p}(A_s) = [V_p(A_s)]^{-1} U_p(A_s)$, where:
//!    $$U_p(X) = \sum_{k=0}^p c_k X^k, \quad V_p(X) = \sum_{k=0}^p c_k (-X)^k$$
//!    and coefficients satisfy $c_k = c_{k-1} \frac{p - k + 1}{k(2p - k + 1)}$.
//!
//!    *(Note: $c_0 = 1$ and $c_1 = 0.5$ are embedded directly into the base identity/diagonal matrices;
//!    the coefficient arrays passed to [`pade_pp`] omit these two and provide $[c_2, \dots, c_p]$).*
//!
//! 3. **Linear System Solve**: Overwrite $U_p(A_s)$ by solving:
//!    $$V_p(A_s) \cdot X = U_p(A_s)$$
//!
//! 4. **Repeated Squaring**: Recover $e^A$ through $s$ successive matrix squarings:
//!    $$e^A = \left(R_{p,p}(A_s)\right)^{2^s}$$
//!
//! ---
//!
//! ## Quick Start
//!
//! Add `expm_rust` to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! expm_rust = "0.1"
//! ```
//!
//! Compute the matrix exponential using the default `oxiblas` backend:
//!
//! ```rust
//! # #[cfg(feature = "oxiblas-backend")]
//! # {
//! use expm_rust::pade_pp_f64;
//! use oxiblas::prelude::*;
//!
//! // Create a 2x2 matrix:
//! // [ 0.0  1.0 ]
//! // [ 0.0  0.0 ]
//! let mut a = MatBuilder::<f64>::zeros(2, 2);
//! a[(0, 1)] = 1.0;
//!
//! // exp(A) for a nilpotent matrix [[0, 1], [0, 0]] is [[1, 1], [0, 1]]
//! let exp_a = pade_pp_f64(&mut a, 6, 0).expect("Exponential computation failed");
//!
//! assert!((exp_a[(0, 0)] - 1.0).abs() < 1e-14);
//! assert!((exp_a[(0, 1)] - 1.0).abs() < 1e-14);
//! assert!((exp_a[(1, 0)] - 0.0).abs() < 1e-14);
//! assert!((exp_a[(1, 1)] - 1.0).abs() < 1e-14);
//! # }
//! ```
//!
//! ---
//!
//! ## Implementing a Custom Backend
//!
//! To use your own matrix data structures, implement [`MatrixOps`] for your matrix type:
//!
//! ```rust
//! use expm_rust::{MatrixOps, MathError};
//!
//! #[derive(Clone)]
//! struct SimpleMat2x2 {
//!     data: [f64; 4], // row-major: [m00, m01, m10, m11]
//! }
//!
//! impl MatrixOps for SimpleMat2x2 {
//!     type Scalar = f64;
//!
//!     fn shape(&self) -> (usize, usize) { (2, 2) }
//!
//!     fn zeros(_size: usize) -> Self {
//!         Self { data: [0.0; 4] }
//!     }
//!
//!     fn identity(_size: usize) -> Self {
//!         Self { data: [1.0, 0.0, 0.0, 1.0] }
//!     }
//!
//!     fn diag(_size: usize, val: f64) -> Self {
//!         Self { data: [val, 0.0, 0.0, val] }
//!     }
//!
//!     fn scale_assign(&mut self, alpha: f64) {
//!         for x in &mut self.data { *x *= alpha; }
//!     }
//!
//!     fn axpy(&mut self, alpha: f64, x: &Self) {
//!         for (y, &xi) in self.data.iter_mut().zip(x.data.iter()) {
//!             *y += alpha * xi;
//!         }
//!     }
//!
//!     fn from_axpy(&mut self, alpha: f64, x: &Self, y: &Self) {
//!         for (res, (&xi, &yi)) in self.data.iter_mut().zip(x.data.iter().zip(y.data.iter())) {
//!             *res = alpha * xi + yi;
//!         }
//!     }
//!
//!     fn gemm(&mut self, alpha: f64, a: &Self, b: &Self, beta: f64) {
//!         let (a00, a01, a10, a11) = (a.data[0], a.data[1], a.data[2], a.data[3]);
//!         let (b00, b01, b10, b11) = (b.data[0], b.data[1], b.data[2], b.data[3]);
//!         let prod = [
//!             a00 * b00 + a01 * b10,
//!             a00 * b01 + a01 * b11,
//!             a10 * b00 + a11 * b10,
//!             a10 * b01 + a11 * b11,
//!         ];
//!         for (c, &p) in self.data.iter_mut().zip(prod.iter()) {
//!             *c = alpha * p + beta * *c;
//!         }
//!     }
//!
//!     fn solve_in_place(&self, b: &mut Self) -> Result<(), MathError> {
//!         // Direct 2x2 solve: X = A^(-1) * B
//!         let det = self.data[0] * self.data[3] - self.data[1] * self.data[2];
//!         if det.abs() < 1e-15 { return Err(MathError::SingularMatrix); }
//!         let inv = [
//!              self.data[3] / det, -self.data[1] / det,
//!             -self.data[2] / det,  self.data[0] / det,
//!         ];
//!         let (b00, b01, b10, b11) = (b.data[0], b.data[1], b.data[2], b.data[3]);
//!         b.data = [
//!             inv[0] * b00 + inv[1] * b10,
//!             inv[0] * b01 + inv[1] * b11,
//!             inv[2] * b00 + inv[3] * b10,
//!             inv[2] * b01 + inv[3] * b11,
//!         ];
//!         Ok(())
//!     }
//!
//!     fn copy_from(&mut self, x: &Self) {
//!         self.data = x.data;
//!     }
//! }
//! ```
//!
//! ---
//!
//! ## Feature Flags
//!
//! - `oxiblas-backend` (*default*): Enables [`oxiblas`] dependency and implements [`MatrixOps`] for `Mat<f64>`.
//!   Disable default features (`default-features = false`) if you only want the generic traits and algorithms.

pub mod backends;
pub mod pade_coef;
mod pade_expm;
mod traits;

pub use pade_expm::{pade_pp, pade_pp_f64};
pub use traits::{MathError, MatrixOps};

