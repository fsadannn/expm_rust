//! Custom backend implementation example for expm_rust.
//!
//! Demonstrates how to implement the `MatrixOps` trait for a custom matrix type
//! without any external linear algebra dependencies.
//!
//! Run with:
//! ```bash
//! cargo run --example custom_backend
//! ```

use expm_rust::{MathError, MatrixOps, pade_coef::PADE_COEFFS, pade_pp};

/// Minimal 2x2 row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomMat2x2 {
    pub data: [f64; 4], // [m00, m01, m10, m11]
}

impl CustomMat2x2 {
    pub fn new(m00: f64, m01: f64, m10: f64, m11: f64) -> Self {
        Self {
            data: [m00, m01, m10, m11],
        }
    }
}

impl MatrixOps for CustomMat2x2 {
    type Scalar = f64;

    fn shape(&self) -> (usize, usize) {
        (2, 2)
    }

    fn zeros(_size: usize) -> Self {
        Self { data: [0.0; 4] }
    }

    fn identity(_size: usize) -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 1.0],
        }
    }

    fn diag(_size: usize, val: f64) -> Self {
        Self {
            data: [val, 0.0, 0.0, val],
        }
    }

    fn scale_assign(&mut self, alpha: f64) {
        for x in &mut self.data {
            *x *= alpha;
        }
    }

    fn axpy(&mut self, alpha: f64, x: &Self) {
        for (y, &xi) in self.data.iter_mut().zip(x.data.iter()) {
            *y += alpha * xi;
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
        if det.abs() < 1e-15 {
            return Err(MathError::SingularMatrix);
        }
        let inv = [
            self.data[3] / det,
            -self.data[1] / det,
            -self.data[2] / det,
            self.data[0] / det,
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
    println!("=== expm_rust Custom Backend Example ===");

    // Matrix [[0, 1], [-1, 0]] which generates rotation matrix:
    // exp([[0, t], [-t, 0]]) = [[cos(t), sin(t)], [-sin(t), cos(t)]]
    let t = std::f64::consts::FRAC_PI_2; // pi / 2
    let mut rot_gen = CustomMat2x2::new(0.0, t, -t, 0.0);

    println!("\nGenerator Matrix A (for t = pi/2):");
    println!("  [{:10.6}, {:10.6}]", rot_gen.data[0], rot_gen.data[1]);
    println!("  [{:10.6}, {:10.6}]", rot_gen.data[2], rot_gen.data[3]);

    let exp_rot = pade_pp(&mut rot_gen, 9, 2, &PADE_COEFFS).expect("Failed to compute exponential");

    println!("\nResult exp(A) (Expected 90 deg rotation [[0, 1], [-1, 0]]):");
    println!("  [{:10.6}, {:10.6}]", exp_rot.data[0], exp_rot.data[1]);
    println!("  [{:10.6}, {:10.6}]", exp_rot.data[2], exp_rot.data[3]);

    // Check analytical values: cos(pi/2) = 0, sin(pi/2) = 1
    assert!((exp_rot.data[0] - 0.0).abs() < 1e-12);
    assert!((exp_rot.data[1] - 1.0).abs() < 1e-12);
    assert!((exp_rot.data[2] - (-1.0)).abs() < 1e-12);
    assert!((exp_rot.data[3] - 0.0).abs() < 1e-12);

    println!("\nVerification successful!");
}
