use crate::{MathError, MatrixOps, pade_coef::PADE_COEFFS};

/// Computes the matrix exponential $e^A$ using a $[p/p]$ Padé approximant and scaling and squaring.
///
/// This generic function allows supplying custom coefficient arrays (e.g. for `f128`, `f32`, or custom scalar types).
/// For standard `f64` computations with built-in coefficients, see [`pade_pp_f64`].
///
/// # Mathematical Method
///
/// The algorithm computes the matrix exponential in three main stages:
/// 1. **Scaling**: Scales $A \leftarrow 2^{-s} A$ so that the matrix norm is small enough for accurate Padé approximation.
/// 2. **Padé Approximation**: Evaluates matrix polynomials $U_p(A)$ and $V_p(A)$ using optimized
///    Paterson-Stockmeyer / Horner evaluation schemes, then solves the matrix system:
///    $$V_p(A) \cdot R_{p,p}(A) = U_p(A) \implies R_{p,p}(A) = [V_p(A)]^{-1} U_p(A)$$
/// 3. **Squaring**: Repeatedly squares the result $s$ times:
///    $$e^A = \left(R_{p,p}(2^{-s} A)\right)^{2^s}$$
///
/// # Coefficient Array Convention
///
/// The `coefficients` parameter expects an array of 13 slices indexed by `pol_deg - 1`.
/// Each slice contains $[c_2, c_3, \dots, c_p]$ (omitting the first two implicit coefficients $c_0 = 1$ and $c_1 = 0.5$,
/// which are structurally embedded in the base matrix initialization).
///
/// # Arguments
///
/// * `A` - Mutable reference to the square matrix $A$. Note that $A$ is scaled in-place and use as scratch memory during computation.
/// * `pol_deg` - Degree of the Padé approximant. Currently supported degrees are **3, 4, 5, 6, 7, 9, 13**.
/// * `s` - Non-negative integer scaling factor ($2^s$).
/// * `coefficients` - Array of slices containing Padé coefficients for each degree.
///
/// # Errors
///
/// Returns [`MathError::NotSquare`] if $A$ is not square.
/// Returns [`MathError::InvalidPolDeg`] if `pol_deg` is not one of $\{3, 4, 5, 6, 7, 9, 13\}$.
/// Returns [`MathError::SingularMatrix`] if matrix $V_p(A)$ is singular during the linear solve.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "oxiblas-backend")]
/// # {
/// use expm_rust::{pade_pp, pade_coef::PADE_COEFFS};
/// use oxiblas::prelude::*;
///
/// // Create a 2x2 zero matrix; exp(0) = I
/// let mut a = MatBuilder::<f64>::zeros(2, 2);
/// let exp_a = pade_pp(&mut a, 6, 0, &PADE_COEFFS).unwrap();
///
/// assert!((exp_a[(0, 0)] - 1.0).abs() < 1e-15);
/// assert!((exp_a[(1, 1)] - 1.0).abs() < 1e-15);
/// # }
/// ```
#[allow(non_snake_case)]
pub fn pade_pp<M, C>(
    A: &mut M,
    pol_deg: usize,
    s: u32,
    coefficients: &[&[C]; 13],
) -> Result<M, MathError>
where
    M: MatrixOps,
    C: Copy,
    M::Scalar: From<C> + From<f64>,
{
    let (row, col) = A.shape();
    if row != col {
        return Err(MathError::NotSquare {
            rows: row,
            cols: col,
        });
    }
    if pol_deg < 3 || pol_deg > 13 {
        return Err(MathError::InvalidPolDeg(pol_deg));
    }
    let coef = coefficients[pol_deg - 1];
    if coef.is_empty() {
        return Err(MathError::InvalidPolDeg(pol_deg));
    }

    /* ps = 2^s; */
    let ps: f64 = 2.0f64.powf(s as f64);
    let is: f64 = 1.0f64 / ps;

    A.scale_assign(is.into());
    let mut powA = M::zeros(row);
    // A^2
    powA.gemm(1.0f64.into(), A, A, 0.0f64.into());

    let mut u = M::identity(row);
    let mut v = M::diag(row, 0.5.into());

    match pol_deg {
        3 | 4 => {
            let mut cf = unsafe { *coef.get_unchecked(0) };
            u.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(1) };
            v.axpy(cf.into(), &powA);
            let mut temp = M::zeros(row);
            temp.gemm(1.0f64.into(), A, &v, 0.0f64.into());
            if pol_deg == 4 {
                A.gemm(1.0f64.into(), &powA, &powA, 0.0f64.into());
                cf = unsafe { *coef.get_unchecked(2) };
                u.axpy(cf.into(), &A);
            }
            v.from_axpy((-1.0f64).into(), &temp, &u);
            u.axpy(1.0f64.into(), &temp);
        }
        5 | 6 => {
            let mut cf = unsafe { *coef.get_unchecked(0) };
            u.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(1) };
            v.axpy(cf.into(), &powA);
            let mut copy_of_A = A.clone();
            A.gemm(1.0f64.into(), &powA, &powA, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(2) };
            u.axpy(cf.into(), &A);
            cf = unsafe { *coef.get_unchecked(3) };
            v.axpy(cf.into(), &A);
            let mut temp = M::zeros(row);
            temp.gemm(1.0f64.into(), &copy_of_A, &v, 0.0f64.into());
            if pol_deg == 6 {
                copy_of_A.gemm(1.0f64.into(), &A, &powA, 0.0f64.into());
                cf = unsafe { *coef.get_unchecked(4) };
                u.axpy(cf.into(), &copy_of_A);
            }
            v.from_axpy((-1.0f64).into(), &temp, &u);
            u.axpy(1.0f64.into(), &temp);
        }
        7 => {
            let mut cf = unsafe { *coef.get_unchecked(0) };
            u.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(1) };
            v.axpy(cf.into(), &powA);
            let copy_of_A = A.clone();
            // A^4
            A.gemm(1.0f64.into(), &powA, &powA, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(2) };
            u.axpy(cf.into(), &A);
            cf = unsafe { *coef.get_unchecked(3) };
            v.axpy(cf.into(), &A);
            let mut temp = M::zeros(row);
            // A^6
            temp.gemm(1.0f64.into(), &A, &powA, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(4) };
            u.axpy(cf.into(), &temp);
            cf = unsafe { *coef.get_unchecked(5) };
            v.axpy(cf.into(), &temp);

            temp.gemm(1.0f64.into(), &copy_of_A, &v, 0.0f64.into());
            v.from_axpy((-1.0f64).into(), &temp, &u);
            u.axpy(1.0f64.into(), &temp);
        }
        9 => {
            let mut cf = unsafe { *coef.get_unchecked(0) };
            u.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(1) };
            v.axpy(cf.into(), &powA);
            let copy_of_A = A.clone();
            // A^4
            A.gemm(1.0f64.into(), &powA, &powA, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(2) };
            u.axpy(cf.into(), &A);
            cf = unsafe { *coef.get_unchecked(3) };
            v.axpy(cf.into(), &A);
            let mut temp = M::zeros(row);
            // A^6
            temp.gemm(1.0f64.into(), &A, &powA, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(4) };
            u.axpy(cf.into(), &temp);
            cf = unsafe { *coef.get_unchecked(5) };
            v.axpy(cf.into(), &temp);

            // A^8
            powA.gemm(1.0f64.into(), &A, &A, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(6) };
            u.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(7) };
            v.axpy(cf.into(), &powA);

            temp.gemm(1.0f64.into(), &copy_of_A, &v, 0.0f64.into());
            v.from_axpy((-1.0f64).into(), &temp, &u);
            u.axpy(1.0f64.into(), &temp);
        }
        13 => {
            let mut u2 = powA.clone();
            let mut v2 = powA.clone();

            let mut cf = unsafe { *coef.get_unchecked(0) };
            u.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(1) };
            v.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(6) };
            u2.scale_assign(cf.into());
            cf = unsafe { *coef.get_unchecked(7) };
            v2.scale_assign(cf.into());

            let copy_of_A = A.clone();
            // A^4
            A.gemm(1.0f64.into(), &powA, &powA, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(2) };
            u.axpy(cf.into(), &A);
            cf = unsafe { *coef.get_unchecked(3) };
            v.axpy(cf.into(), &A);
            cf = unsafe { *coef.get_unchecked(8) };
            u2.axpy(cf.into(), &A);
            cf = unsafe { *coef.get_unchecked(9) };
            v2.axpy(cf.into(), &A);

            let mut temp = M::zeros(row);
            // A^6
            temp.gemm(1.0f64.into(), &A, &powA, 0.0f64.into());
            cf = unsafe { *coef.get_unchecked(4) };
            u.axpy(cf.into(), &temp);
            cf = unsafe { *coef.get_unchecked(5) };
            v.axpy(cf.into(), &temp);
            cf = unsafe { *coef.get_unchecked(10) };
            u2.axpy(cf.into(), &temp);
            cf = unsafe { *coef.get_unchecked(11) };
            v2.axpy(cf.into(), &temp);

            u.gemm(1.0f64.into(), &temp, &u2, 1.0f64.into());
            v.gemm(1.0f64.into(), &temp, &v2, 1.0f64.into());

            temp.gemm(1.0f64.into(), &copy_of_A, &v, 0.0f64.into());
            v.from_axpy((-1.0f64).into(), &temp, &u);
            u.axpy(1.0f64.into(), &temp);
        }
        _ => {
            todo!("not implemented yet")
        }
    }

    v.solve_in_place(&mut u)?;

    let scale_it = s / 2;
    for _k in 0..scale_it {
        v.gemm(1.0f64.into(), &u, &u, 0.0f64.into());
        u.gemm(1.0f64.into(), &v, &v, 0.0f64.into());
    }

    if s & 1 != 0 {
        v.gemm(1.0f64.into(), &u, &u, 0.0f64.into());
        u.copy_from(&v);
    }

    Ok(u)
}

/// Computes the matrix exponential $e^A$ for `f64` scalar matrix types using standard precomputed Padé coefficients.
///
/// This is a convenience wrapper around [`pade_pp`] using [`PADE_COEFFS`].
///
/// # Arguments
///
/// * `A` - Mutable reference to the square matrix $A$. Note that $A$ is scaled in-place.
/// * `pol_deg` - Degree of the Padé approximant. Must be one of $\{3, 4, 5, 6, 7, 9, 13\}$.
/// * `s` - Non-negative integer scaling factor ($2^s$).
///
/// # Errors
///
/// Returns [`MathError`] if dimensions are invalid or the matrix is singular.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "oxiblas-backend")]
/// # {
/// use expm_rust::pade_pp_f64;
/// use oxiblas::prelude::*;
///
/// // Create a 2x2 identity matrix; exp(I) = diag(e, e)
/// let mut a = MatBuilder::<f64>::identity(2);
/// let exp_a = pade_pp_f64(&mut a, 13, 1).unwrap();
///
/// let e = std::f64::consts::E;
/// assert!((exp_a[(0, 0)] - e).abs() < 1e-14);
/// assert!((exp_a[(1, 1)] - e).abs() < 1e-14);
/// # }
/// ```
#[inline(always)]
#[allow(non_snake_case)]
pub fn pade_pp_f64<M>(A: &mut M, pol_deg: usize, s: u32) -> Result<M, MathError>
where
    M: MatrixOps,
    M::Scalar: From<f64>,
{
    pade_pp(A, pol_deg, s, &PADE_COEFFS)
}

#[cfg(all(test, feature = "oxiblas-backend"))]
mod tests {
    use std::f64;

    use super::*;
    use oxiblas::prelude::*;

    /// utils functions for the test
    #[inline]
    pub fn frexp(x: f64) -> (f64, i32) {
        let mut y = x.to_bits();
        let ee = ((y >> 52) & 0x7ff) as i32;

        if ee == 0 {
            if x != 0.0 {
                let x1p64 = f64::from_bits(0x43f0000000000000);
                let (x, e) = frexp(x * x1p64);
                return (x, e - 64);
            }
            return (x, 0);
        } else if ee == 0x7ff {
            return (x, 0);
        }

        let e = ee - 0x3fe;
        y &= 0x800fffffffffffff;
        y |= 0x3fe0000000000000;
        (f64::from_bits(y), e)
    }

    fn get_scale(a: &Mat<f64>) -> u32 {
        let norm_a = norm_inf(a.as_ref());
        let (_, e) = frexp(norm_a);
        let s: u32 = std::cmp::max(0, e + 1).try_into().unwrap();
        s
    }

    fn err(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
        a.raw_data()
            .iter()
            .zip(b.raw_data().iter())
            .map(|(a, b)| (*a - *b).abs())
            .reduce(f64::max)
            .unwrap_or(0f64)
    }

    const POL_DEG: [usize; 7] = [3, 4, 5, 6, 7, 9, 13];
    /// test functions
    #[test]
    fn test_zero() {
        let a = MatBuilder::<f64>::zeros(2, 2);
        let sol = MatBuilder::<f64>::identity(2);

        for &deg in POL_DEG.iter() {
            let mut mat = a.clone();
            let _res = pade_pp_f64(&mut mat, deg, 0);
            assert!(_res.is_ok());
            let res = _res.unwrap();
            // println!("deg: {deg}");
            // println!("tol: {}", 1e-16);
            // println!("err: {}", err(&res, &sol));
            assert!(err(&res, &sol) < 1e-16f64);
        }
    }

    #[test]
    fn test_identity() {
        let a = MatBuilder::<f64>::identity(2);
        let s = get_scale(&a);
        let mut sol = MatBuilder::<f64>::zeros(2, 2);
        sol[(0, 0)] = f64::consts::E;
        sol[(1, 1)] = f64::consts::E;
        const TOLERANCE: [f64; 7] = [
            1e-8f64, 1e-11f64, 1e-15f64, 1e-15f64, 1e-15f64, 1e-15f64, 1e-15f64,
        ];

        for (&deg, &tol) in POL_DEG.iter().zip(TOLERANCE.iter()) {
            let mut mat = a.clone();
            let _res = pade_pp_f64(&mut mat, deg, s);
            assert!(_res.is_ok());
            let res = _res.unwrap();
            // println!("deg: {deg}");
            // println!("tol: {tol}");
            // println!("err: {}", err(&res, &sol));
            assert!(err(&res, &sol) < tol);
        }
    }

    #[test]
    fn test_upper_triangular() {
        let mut a = MatBuilder::<f64>::zeros(3, 3);
        a[(0, 0)] = 0.346358384327981;
        a[(0, 1)] = 0.388260875523650;
        a[(0, 2)] = 0.917847165891965;
        a[(1, 1)] = 0.031418391315215;
        a[(1, 2)] = 0.467599106573163;
        a[(2, 2)] = 0.078284436848147;
        let s = get_scale(&a);

        let mut sol = MatBuilder::<f64>::zeros(3, 3);
        sol[(0, 0)] = 1.413909247943398;
        sol[(0, 1)] = 0.470923306918821;
        sol[(0, 2)] = 1.244297682904218;
        sol[(1, 1)] = 1.031917158757147;
        sol[(1, 2)] = 0.494009253649905;
        sol[(2, 2)] = 1.081430213529469;

        const TOLERANCE: [f64; 7] = [
            1e-10f64, 1e-15f64, 1e-15f64, 1e-15f64, 1e-15f64, 1e-15f64, 1e-15f64,
        ];

        for (&deg, &tol) in POL_DEG.iter().zip(TOLERANCE.iter()) {
            let mut mat = a.clone();
            let _res = pade_pp_f64(&mut mat, deg, s);

            assert!(_res.is_ok());
            let res = _res.unwrap();
            // println!("deg: {deg}");
            // println!("tol: {tol}");
            // println!("err: {}", err(&res, &sol));
            assert!(err(&res, &sol) < tol);
        }
    }
}
