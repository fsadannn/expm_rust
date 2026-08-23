# expm_rust

[![Crates.io](https://img.shields.io/crates/v/expm_rust.svg)](https://crates.io/crates/expm_rust)
[![Documentation](https://docs.rs/expm_rust/badge.svg)](https://docs.rs/expm_rust)
[![License](https://img.shields.io/badge/license-BSD--3--Clause%20OR%20Apache--2.0-blue.svg)](Cargo.toml)

High-performance, generic matrix exponential ($\exp(A) = e^A$) implementation in pure Rust using **diagonal $[p/p]$ Padé rational approximation** with the **scaling and squaring method**.

---

## Features

- **Generic Matrix Backend**: Decoupled from any single linear algebra framework via the [`MatrixOps`](https://docs.rs/expm_rust/latest/expm_rust/trait.MatrixOps.html) trait. Bring your own matrix type (`nalgebra`, `faer`, `ndarray`, custom SIMD arrays, or embedded fixed-size matrices).
- **Built-in `oxiblas` Support**: Includes a ready-to-use backend for [`oxiblas::prelude::Mat<f64>`](https://crates.io/crates/oxiblas) featuring fast BLAS/SIMD operations.
- **Optimized Polynomial Evaluation**: Implements Paterson-Stockmeyer and Horner evaluation schemes for degrees $p \in \{3, 4, 5, 6, 7, 9, 13\}$ to minimize matrix multiplications.
- **Compile-Time Exact Rational Coefficients**: Coefficients are generated at compile time as exact irreducible fractions using 128-bit unsigned integers (`u128`), avoiding floating-point rounding errors.
- **`f128`, `f32`, and Custom Precision Ready**: Exact fractions can be evaluated at compile time or runtime for custom scalar types (e.g. 128-bit quadruple precision floats or arbitrary-precision arithmetic).
- **In-Place Buffer Reuse**: Designed for minimal memory allocations during scaling, polynomial accumulation, and squaring stages.

---

## Mathematical Background

Given a square matrix $A \in \mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$), the matrix exponential is defined by the power series:

$$e^A = \sum_{k=0}^{\infty} \frac{1}{k!} A^k = I + A + \frac{1}{2} A^2 + \frac{1}{6} A^3 + \dots$$

### Scaling & Squaring Algorithm

Directly truncating the Taylor series for large $\|A\|$ is numerically unstable. Instead, `expm_rust` uses the scaling and squaring method with Padé approximants:

1. **Scaling**: Compute an integer scale parameter $s \ge 0$ such that the scaled matrix $A_s = 2^{-s} A$ has a sufficiently small matrix norm $\|A_s\|$.
2. **Diagonal $[p/p]$ Padé Approximation**:
   $$e^{A_s} \approx R_{p,p}(A_s) = [V_p(A_s)]^{-1} U_p(A_s)$$
   where:
   $$U_p(X) = \sum_{k=0}^p c_k X^k, \quad V_p(X) = \sum_{k=0}^p c_k (-X)^k$$
   and the recurrence for coefficients is:
   $$c_0 = 1, \quad c_k = c_{k-1} \frac{p - k + 1}{k(2p - k + 1)}$$
3. **Linear System Solve**: Rather than inverting $V_p(A_s)$, solve the linear matrix equation $V_p(A_s) X = U_p(A_s)$ in-place for $X$.
4. **Squaring**: Recover $e^A$ through $s$ repeated matrix squarings:
   $$e^A = (e^{A_s})^{2^s} = \underbrace{\left(\left(R_{p,p}(A_s)\right)^2\right)^2 \dots}_{s \text{ times}}$$

---

## Coefficient Conventions & Precision

### Coefficient Array Structure
In `expm_rust`, the first two Padé coefficients:
- $c_0 = 1.0$ (corresponds to identity matrix $I$)
- $c_1 = 0.5$ (corresponds to diagonal scaling $0.5 I$)

are structurally embedded in the base initialization of polynomial matrices $U$ and $V$. Therefore, the coefficient slices in [`PADE_COEFFS`](https://docs.rs/expm_rust/latest/expm_rust/pade_coef/static.PADE_COEFFS.html) omit $c_0$ and $c_1$, and contain $[c_2, c_3, \dots, c_p]$ (length $p - 1$).

### Higher-Precision (`f128` / Custom Float Types)
The [`pade_coef::Rational`](https://docs.rs/expm_rust/latest/expm_rust/pade_coef/struct.Rational.html) struct stores exact numerator and denominator as `u128` integers. You can compute exact Padé coefficients for custom scalar types using [`compute_pade_array`](https://docs.rs/expm_rust/latest/expm_rust/pade_coef/fn.compute_pade_array.html):

```rust
use expm_rust::pade_coef::{compute_pade_array, Rational};

// Exact rational coefficients for degree 7 (omits c0 and c1, length = 6)
const RATIONALS_7: [Rational; 6] = compute_pade_array::<7, 6>();

// If using f128 or custom precision:
// let c2 = RATIONALS_7[0].num as f128 / RATIONALS_7[0].den as f128;
```

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
expm_rust = "0.1"
```

To use only the generic traits without the default `oxiblas` backend:

```toml
[dependencies]
expm_rust = { version = "0.1", default-features = false }
```

---

## Quickstart

### Using the Built-In `oxiblas` Backend

```rust
use expm_rust::pade_pp_f64;
use oxiblas::prelude::*;

fn main() {
    // Define a 2x2 matrix
    let mut a = MatBuilder::<f64>::zeros(2, 2);
    a[(0, 0)] = 1.0;
    a[(0, 1)] = 2.0;
    a[(1, 0)] = 0.0;
    a[(1, 1)] = 1.0;

    // Degree 6, scaling power s = 1 (A is scaled by 2^-1 = 0.5)
    let exp_a = pade_pp_f64(&mut a, 6, 1).expect("Calculation failed");

    println!("exp(A) =");
    println!("[{:.4}, {:.4}]", exp_a[(0, 0)], exp_a[(0, 1)]);
    println!("[{:.4}, {:.4}]", exp_a[(1, 0)], exp_a[(1, 1)]);
}
```

---

## Implementing a Custom Backend

To integrate `expm_rust` with any third-party or custom matrix type, implement the [`MatrixOps`](https://docs.rs/expm_rust/latest/expm_rust/trait.MatrixOps.html) trait:

```rust
use expm_rust::{MatrixOps, MathError, pade_pp, pade_coef::PADE_COEFFS};

#[derive(Clone, Debug)]
struct Mat2x2 {
    data: [f64; 4], // [m00, m01, m10, m11]
}

impl MatrixOps for Mat2x2 {
    type Scalar = f64;

    fn shape(&self) -> (usize, usize) { (2, 2) }
    fn zeros(_size: usize) -> Self { Self { data: [0.0; 4] } }
    fn identity(_size: usize) -> Self { Self { data: [1.0, 0.0, 0.0, 1.0] } }
    fn diag(_size: usize, val: f64) -> Self { Self { data: [val, 0.0, 0.0, val] } }

    fn scale_assign(&mut self, alpha: f64) {
        for x in &mut self.data { *x *= alpha; }
    }

    fn axpy(&mut self, alpha: f64, x: &Self) {
        for (y, &xi) in self.data.iter_mut().zip(x.data.iter()) {
            *y += alpha * xi;
        }
    }

    fn from_axpy(&mut self, alpha: f64, x: &Self, y: &Self) {
        for (res, (&xi, &yi)) in self.data.iter_mut().zip(x.data.iter().zip(y.data.iter())) {
            *res = alpha * xi + yi;
        }
    }

    fn gemm(&mut self, alpha: f64, a: &Self, b: &Self, beta: f64) {
        let (a00, a01, a10, a11) = (a.data[0], a.data[1], a.data[2], a.data[3]);
        let (b00, b01, b10, b11) = (b.data[0], b.data[1], b.data[2], b.data[3]);
        let prod = [
            a00 * b00 + a01 * b10,
            a00 * b01 + a01 * b11,
            a10 * b00 + a11 * b10,
            a10 * b01 + a11 * b11,
        ];
        for (c, &p) in self.data.iter_mut().zip(prod.iter()) {
            *c = alpha * p + beta * *c;
        }
    }

    fn solve_in_place(&self, b: &mut Self) -> Result<(), MathError> {
        let det = self.data[0] * self.data[3] - self.data[1] * self.data[2];
        if det.abs() < 1e-15 { return Err(MathError::SingularMatrix); }
        let inv = [
             self.data[3] / det, -self.data[1] / det,
            -self.data[2] / det,  self.data[0] / det,
        ];
        let (b00, b01, b10, b11) = (b.data[0], b.data[1], b.data[2], b.data[3]);
        b.data = [
            inv[0] * b00 + inv[1] * b10,
            inv[0] * b01 + inv[1] * b11,
            inv[2] * b00 + inv[3] * b10,
            inv[2] * b01 + inv[3] * b11,
        ];
        Ok(())
    }

    fn copy_from(&mut self, x: &Self) {
        self.data = x.data;
    }
}

fn main() {
    let mut mat = Mat2x2 { data: [0.0, 1.0, 0.0, 0.0] };
    let exp_mat = pade_pp(&mut mat, 6, 0, &PADE_COEFFS).unwrap();
    println!("Result: {:?}", exp_mat);
}
```

---

## Examples

Run the provided examples:

```bash
# Run the basic exponential example
cargo run --example basic_expm

# Run the custom backend example
cargo run --example custom_backend
```

---

## Supported Padé Degrees

The following polynomial degrees are supported:
- **Degree 3**: 2 coefficients ($c_2, c_3$)
- **Degree 4**: 3 coefficients ($c_2, c_3, c_4$)
- **Degree 5**: 4 coefficients ($c_2, \dots, c_5$)
- **Degree 6**: 5 coefficients ($c_2, \dots, c_6$)
- **Degree 7**: 6 coefficients ($c_2, \dots, c_7$)
- **Degree 9**: 8 coefficients ($c_2, \dots, c_9$)
- **Degree 13**: 12 coefficients ($c_2, \dots, c_{13}$)

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- BSD 3-Clause License ([LICENSE-BSD-C-3](LICENSE-BSD-C-3) or <https://opensource.org/licenses/BSD-3-Clause>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
