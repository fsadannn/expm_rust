use crate::{MathError, MatrixOps, pade_coef::PADE_COEFFS};

#[allow(non_snake_case)]
pub fn pade_pp<M>(A: &mut M, pol_deg: usize, s: u32) -> Result<M, MathError>
where
    M: MatrixOps + std::fmt::Debug,
    M::Scalar: From<f64>,
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
    let coef = PADE_COEFFS[pol_deg - 1];
    if coef.is_empty() {
        return Err(MathError::InvalidPolDeg(pol_deg));
    }

    /* ps = 2^s; */
    let ps: f64 = 2.0f64.powf(s as f64);
    let is: f64 = 1.0f64 / ps;

    A.scale_assign(is.into());
    let mut powA = M::zeros(row);
    powA.gemm(1.0f64.into(), A, A, 0.0f64.into());

    let mut u = M::identity(row);
    let mut v = M::diag(row, 0.5.into());

    match pol_deg {
        3 => {
            let mut cf = unsafe { *coef.get_unchecked(0) };
            u.axpy(cf.into(), &powA);
            cf = unsafe { *coef.get_unchecked(1) };
            v.axpy(cf.into(), &powA);
            let mut temp = M::zeros(row);
            temp.gemm(1.0f64.into(), A, &v, 0.0f64.into());
            v.from_axpy((-1.0f64).into(), &temp, &u);
            u.axpy(1.0f64.into(), &temp);
        }
        _ => {
            todo!("not implemented yet")
        }
    }
    // println!("u: {:?}", u);
    // println!("v: {:?}", v);
    v.solve_in_place(&mut u)?;
    // println!("u: {:?}", u);

    let scale_it = s / 2;
    for _k in 0..scale_it {
        v.gemm(1.0f64.into(), &u, &u, 0.0f64.into());
        u.gemm(1.0f64.into(), &v, &v, 0.0f64.into());
    }

    if s & 1 != 0 {
        v.gemm(1.0f64.into(), &u, &u, 0.0f64.into());
        u.copy_from(&v);
    }

    // if s > 0 {
    //     println!("u scaled: {:?}", u);
    // }

    Ok(u)
}

#[cfg(test)]
#[cfg(not(feature = "oxiblas-backend"))]
compile_error!("Tests must be run with `--features oxiblas-backend`.");

#[cfg(test)]
mod tests {
    use std::f64;

    use super::*;
    use oxiblas::prelude::*;

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

    #[test]
    fn test_zero() {
        let mut a = MatBuilder::<f64>::zeros(2, 2);
        let res = pade_pp::<Mat<f64>>(&mut a, 3, 0);
        assert!(res.is_ok());
        let sol = MatBuilder::<f64>::identity(2);

        assert!(err(&res.unwrap(), &sol) < 1e-16f64);
    }

    #[test]
    fn test_identity() {
        let mut a = MatBuilder::<f64>::identity(2);
        let s = get_scale(&a);
        let res = pade_pp::<Mat<f64>>(&mut a, 3, s);
        assert!(res.is_ok());
        let mut sol = MatBuilder::<f64>::zeros(2, 2);
        sol[(0, 0)] = f64::consts::E;
        sol[(1, 1)] = f64::consts::E;

        assert!(err(&res.unwrap(), &sol) < 1e-8f64);
    }

    #[test]
    fn test_general() {
        let mut a = MatBuilder::<f64>::zeros(3, 3);
        a[(0, 0)] = 0.346358384327981;
        a[(0, 1)] = 0.388260875523650;
        a[(0, 2)] = 0.917847165891965;
        a[(1, 1)] = 0.031418391315215;
        a[(1, 2)] = 0.467599106573163;
        a[(2, 2)] = 0.078284436848147;
        let s = get_scale(&a);

        let res = pade_pp::<Mat<f64>>(&mut a, 3, s);
        assert!(res.is_ok());

        let mut sol = MatBuilder::<f64>::zeros(3, 3);
        sol[(0, 0)] = 1.413909247945447;
        sol[(0, 1)] = 0.470923306921347;
        sol[(0, 2)] = 1.244297682915636;
        sol[(1, 1)] = 1.031917158757147;
        sol[(1, 2)] = 0.494009253649905;
        sol[(2, 2)] = 1.081430213529470;

        assert!(err(&res.unwrap(), &sol) < 1e-14f64);
    }
}
