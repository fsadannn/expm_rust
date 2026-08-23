//! Basic matrix exponential example using the default oxiblas backend.
//!
//! Run with:
//! ```bash
//! cargo run --example basic_expm
//! ```

#[cfg(feature = "oxiblas-backend")]
use expm_rust::pade_pp_f64;
#[cfg(feature = "oxiblas-backend")]
use oxiblas::prelude::*;

fn main() {
    #[cfg(feature = "oxiblas-backend")]
    {
        println!("=== expm_rust Basic Matrix Exponential Example ===");

        // Create a 3x3 upper triangular test matrix
        let mut a = MatBuilder::<f64>::identity(3);

        println!("\nInput Matrix A:");
        for i in 0..3 {
            println!(
                "  [{:10.6}, {:10.6}, {:10.6}]",
                a[(i, 0)],
                a[(i, 1)],
                a[(i, 2)]
            );
        }

        // Estimate matrix norm & scaling parameter s
        let norm_inf_val = norm_inf(a.as_ref());
        println!("\nInfinity Norm ||A||_inf = {:.6}", norm_inf_val);

        // Compute exponential with Padé degree 13 and scaling s = 1
        let exp_a = pade_pp_f64(&mut a, 13, 1).expect("Failed to compute matrix exponential");

        println!("\nOutput Matrix exp(A):");
        for i in 0..3 {
            println!(
                "  [{:10.6}, {:10.6}, {:10.6}]",
                exp_a[(i, 0)],
                exp_a[(i, 1)],
                exp_a[(i, 2)]
            );
        }
    }

    #[cfg(not(feature = "oxiblas-backend"))]
    {
        println!("This example requires the 'oxiblas-backend' feature.");
    }
}
