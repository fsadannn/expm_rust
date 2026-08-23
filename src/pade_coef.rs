const fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[derive(Copy, Clone)]
pub struct Rational {
    pub num: u128,
    pub den: u128,
}

impl Rational {
    const DEFAULT: Self = Rational { num: 0, den: 1 };

    pub const fn new(num: u128, den: u128) -> Self {
        let g = gcd(num, den);
        Self {
            num: num / g,
            den: den / g,
        }
    }

    pub const fn mul_int(self, num_mul: u128, den_mul: u128) -> Self {
        let new_num = self.num * num_mul;
        let new_den = self.den * den_mul;
        Self::new(new_num, new_den)
    }
}

pub const fn compute_pade_array<const P: usize, const LEN: usize>() -> [Rational; LEN] {
    let mut arr = [Rational::DEFAULT; LEN];

    let mut current = Rational::new(1, 1);

    let mut k = 1;
    while k <= P {
        // a_k = a_{k-1} * (p - k + 1) / (k * (2p - k + 1))
        let num_mul = (P - k + 1) as u128;
        let den_mul = (k * (2 * P - k + 1)) as u128;

        current = current.mul_int(num_mul, den_mul);

        if k >= 2 {
            arr[k - 2] = current;
        }
        k += 1;
    }

    arr
}

const fn u128_div_to_f64(num: u128, den: u128) -> f64 {
    if den == 0 {
        return f64::INFINITY;
    }
    if num == 0 {
        return 0.0;
    }

    let num_lz = num.leading_zeros();
    let den_lz = den.leading_zeros();

    let mut n = num << num_lz;
    let d = den << den_lz;
    let mut q: u64 = 0;
    let exp_adj = if n >= d {
        n -= d;
        q = 1;
        0
    } else {
        1
    };

    let iters = if exp_adj == 0 { 53 } else { 54 };
    let mut i = 0;
    while i < iters {
        let msb_set = (n >> 127) != 0;
        n = (n & !(1 << 127)) << 1;

        if msb_set || n >= d {
            n = n.wrapping_sub(d);
            q = (q << 1) | 1;
        } else {
            q <<= 1;
        }
        i += 1;
    }

    let round_bit = (q & 1) != 0;
    let mut mantissa = q >> 1;
    let sticky_bit = n != 0;

    let mut exp = (den_lz as i32 - num_lz as i32) + 1023 - exp_adj;
    if round_bit && (sticky_bit || (mantissa & 1) != 0) {
        mantissa += 1;
        if mantissa >= (1 << 53) {
            mantissa >>= 1;
            exp += 1;
        }
    }

    if exp <= 0 {
        return 0.0;
    }
    if exp >= 2047 {
        return f64::INFINITY;
    }

    let bits = ((exp as u64) << 52) | (mantissa & 0x000F_FFFF_FFFF_FFFF);
    f64::from_bits(bits)
}

const fn rational_arr_to_f64<const LEN: usize>(r: [Rational; LEN]) -> [f64; LEN] {
    let mut arr = [0.0f64; LEN];
    let mut i: usize = 0;
    while i < LEN {
        arr[i] = u128_div_to_f64(r[i].num, r[i].den);
        i += 1;
    }

    arr
}

// Precompute the individual static coefficient arrays at compile time.
// We only compute active orders: 3..=6 and 7, 9, 13.
// We store P - 1 elements, skipping a_0 (1.0) and a_1 (0.5).
const PADE_3: [f64; 2] = rational_arr_to_f64::<2>(compute_pade_array::<3, 2>());
const PADE_4: [f64; 3] = rational_arr_to_f64::<3>(compute_pade_array::<4, 3>());
const PADE_5: [f64; 4] = rational_arr_to_f64::<4>(compute_pade_array::<5, 4>());
const PADE_6: [f64; 5] = rational_arr_to_f64::<5>(compute_pade_array::<6, 5>());
const PADE_7: [f64; 6] = rational_arr_to_f64::<6>(compute_pade_array::<7, 6>());
const PADE_9: [f64; 8] = rational_arr_to_f64::<8>(compute_pade_array::<9, 8>());
const PADE_13: [f64; 12] = rational_arr_to_f64::<12>(compute_pade_array::<13, 12>());

/// Array of 13 static slices mapping degree m (for m = 1..=13) to its Padé coefficients.
///
/// Index `m - 1` holds the slice `&[f64]` for degree `m`.
/// Orders 1, 2, 8, 10, and 12 are left as empty slices `&[]`.
pub static PADE_COEFFS: [&'static [f64]; 13] = [
    &[],      // m = 1 (unused)
    &[],      // m = 2 (unused)
    &PADE_3,  // m = 3
    &PADE_4,  // m = 4
    &PADE_5,  // m = 5
    &PADE_6,  // m = 6
    &PADE_7,  // m = 7
    &[],      // m = 8  (skipped)
    &PADE_9,  // m = 9
    &[],      // m = 10 (skipped)
    &[],      // m = 11 (skipped)
    &[],      // m = 12 (skipped)
    &PADE_13, // m = 13
];

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_approx_eq {
        ($actual:expr, $expected:expr) => {
            let actual = $actual;
            let expected = $expected;

            let margin = expected * f64::EPSILON * 2.0;
            let diff = (actual - expected).abs();

            assert!(
                diff <= margin,
                "Assertion failed: values are not approximately equal\n  actual: `{}`\n  expected: `{}`\n  diff: `{}`\n  allowed margin: `{}`",
                actual, expected, diff, margin
            );
        };
    }

    #[test]
    fn test_const_evaluation() {
        const CONST_RES: f64 = u128_div_to_f64(10, 2);
        assert_eq!(CONST_RES, 5.0);
    }

    #[test]
    fn test_edge_cases() {
        // Divide by zero
        assert_eq!(u128_div_to_f64(1, 0), f64::INFINITY);
        assert_eq!(u128_div_to_f64(u128::MAX, 0), f64::INFINITY);

        // Zero numerator
        assert_eq!(u128_div_to_f64(0, 1), 0.0);
        assert_eq!(u128_div_to_f64(0, u128::MAX), 0.0);
    }

    #[test]
    fn test_exact_integers() {
        assert_eq!(u128_div_to_f64(10, 2), 5.0);
        assert_eq!(u128_div_to_f64(100, 10), 10.0);
        assert_eq!(u128_div_to_f64(1, 1), 1.0);
    }

    #[test]
    fn test_exact_fractions() {
        // Fractions with clean power-of-two denominators yield exact floating point matches
        assert_eq!(u128_div_to_f64(1, 2), 0.5);
        assert_eq!(u128_div_to_f64(1, 4), 0.25);
        assert_eq!(u128_div_to_f64(3, 4), 0.75);
    }

    #[test]
    fn test_powers_of_two() {
        let num = 1u128 << 100;
        let den = 1u128 << 50;

        // 2^100 / 2^50 = 2^50
        assert_eq!(u128_div_to_f64(num, den), (1u64 << 50) as f64);

        // 2^50 / 2^100 = 2^-50
        let res = u128_div_to_f64(den, num);
        assert_eq!(res, 1.0 / ((1u64 << 50) as f64));
    }

    #[test]
    fn test_large_numbers() {
        let max = u128::MAX;
        let half = max / 2;

        // max / 1 -> ~3.402823669209385e38
        assert_approx_eq!(u128_div_to_f64(max, 1), max as f64);

        // max / max -> 1.0
        assert_approx_eq!(u128_div_to_f64(max, max), 1.0);

        // max / (max / 2) -> 2.0 (plus a microscopic fraction, handled by assert_approx_eq)
        assert_approx_eq!(u128_div_to_f64(max, half), 2.0);
    }

    #[test]
    fn test_complex_truncation_precision() {
        // A messy fraction to ensure our 53-bit mantissa alignment doesn't scramble bits
        let num: u128 = 123456789123456789;
        let den: u128 = 987654321987654321;

        let expected = (num as f64) / (den as f64);
        let actual = u128_div_to_f64(num, den);

        assert_approx_eq!(actual, expected);
    }
}
