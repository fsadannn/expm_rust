mod backends;
pub mod pade_coef;
mod pade_expm;
mod traits;
pub use pade_expm::{pade_pp, pade_pp_f64};
pub use traits::{MathError, MatrixOps};
