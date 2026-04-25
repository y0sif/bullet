//! ReLU-KAN LUT (Lookup Table) sampling and export (arXiv 2406.02075).
//!
//! Mirrors [`crate::nn::kan_lut`] for B-spline KAN. The structural difference
//! is the basis function:
//!
//! ```text
//! R_i(x) = [ReLU(e_i - x) * ReLU(x - s_i)]^2 * 16 / (e_i - s_i)^4
//! ```
//!
//! Variant E uses pure-basis layers (no base path), so the LUT sample is just:
//! `lut[i][k][j] = sum_b sw[j, i*nb + b] * R_b(x_k)`.
//!
//! The normalizer `16 / (e_i - s_i)^4` is absorbed into the stored LUT values
//! (no runtime multiply needed at inference).

use acyclib::graph::save::SavedFormat;

/// Build the (s_grid, e_grid) pair for a ReLU-KAN layer.
///
/// Matches `relu_kan_layer()` in `relu_kan.rs`:
/// - `s_i = grid_min + (i - k) / G * span`
/// - `e_i = s_i + (k + 1) / G * span`
fn make_relu_kan_grid(
    grid_size: usize,
    support_width: usize,
    grid_range: (f32, f32),
) -> (Vec<f32>, Vec<f32>) {
    let num_basis = grid_size + support_width;
    let (grid_min, grid_max) = grid_range;
    let span = grid_max - grid_min;
    let g = grid_size as f32;
    let k = support_width as f32;

    let mut s_grid = Vec::with_capacity(num_basis);
    let mut e_grid = Vec::with_capacity(num_basis);
    for i in 0..num_basis {
        let s_i = grid_min + (i as f32 - k) / g * span;
        s_grid.push(s_i);
        e_grid.push(s_i + (k + 1.0) / g * span);
    }
    (s_grid, e_grid)
}

/// Evaluate ReLU-KAN basis functions at a single point.
///
/// Returns a vector of `num_basis = grid_size + support_width` values. Each value
/// already has the `16 / (e_i - s_i)^4` normalizer baked in, so the basis peaks
/// at exactly 1.0 at the midpoint of `(s_i, e_i)` and is zero outside.
fn eval_relu_kan_basis(x: f32, s_grid: &[f32], e_grid: &[f32]) -> Vec<f32> {
    debug_assert_eq!(s_grid.len(), e_grid.len());
    let num_basis = s_grid.len();
    let mut out = vec![0.0f32; num_basis];

    for i in 0..num_basis {
        let s = s_grid[i];
        let e = e_grid[i];
        let u = (e - x).max(0.0);
        let v = (x - s).max(0.0);
        if u == 0.0 || v == 0.0 {
            continue;
        }
        let prod = u * v;
        let span = e - s;
        let normalizer = 16.0 / (span * span * span * span);
        out[i] = prod * prod * normalizer;
    }
    out
}

/// Sample the full ReLU-KAN LUT from raw spline-weight data.
///
/// For each edge (i, j), evaluates the learned activation at `num_samples`
/// evenly-spaced points in `sample_range`:
///
/// ```text
/// lut[i][k][j] = sw[j, i*nb..(i+1)*nb] · R(x_k)
/// ```
///
/// Returns a flat `Vec<f32>` with layout `[in_features][num_samples][out_features]`,
/// matching [`crate::nn::kan_lut::sample_kan_lut`] for engine-side reuse.
pub fn sample_relu_kan_lut(
    sw_vals: &[f32],
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    support_width: usize,
    grid_range: (f32, f32),
    sample_range: (f32, f32),
    num_samples: usize,
) -> Vec<f32> {
    let num_basis = grid_size + support_width;
    let (s_grid, e_grid) = make_relu_kan_grid(grid_size, support_width, grid_range);

    assert_eq!(sw_vals.len(), out_features * in_features * num_basis);

    let (s_min, s_max) = sample_range;
    let sample_points: Vec<f32> = (0..num_samples)
        .map(|k| {
            if num_samples == 1 {
                s_min
            } else {
                s_min + (s_max - s_min) * k as f32 / (num_samples - 1) as f32
            }
        })
        .collect();

    let basis_at_samples: Vec<Vec<f32>> = sample_points
        .iter()
        .map(|&x| eval_relu_kan_basis(x, &s_grid, &e_grid))
        .collect();

    let total_size = in_features * num_samples * out_features;
    let mut lut = vec![0.0f32; total_size];

    for i in 0..in_features {
        for k in 0..num_samples {
            for j in 0..out_features {
                let mut val = 0.0f32;
                for b in 0..num_basis {
                    // Column-major: sw[j, i*nb + b] = sw_vals[(i*nb + b) * out_features + j]
                    let sw_idx = (i * num_basis + b) * out_features + j;
                    val += sw_vals[sw_idx] * basis_at_samples[k][b];
                }
                lut[i * num_samples * out_features + k * out_features + j] = val;
            }
        }
    }

    lut
}

/// Generate `SavedFormat` entry for a ReLU-KAN layer's LUT export (i8, cache-friendly layout).
///
/// Same i8 / `[in][sample][out]` layout as [`crate::nn::kan_lut::kan_lut_save_format`],
/// so the engine inference path is identical apart from the LUT contents.
pub fn relu_kan_lut_save_format(
    id: &str,
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    support_width: usize,
    grid_range: (f32, f32),
    sample_range: (f32, f32),
    num_samples: usize,
    quant_scale: i16,
) -> SavedFormat {
    let id = id.to_string();
    let max_val = 127.0 / quant_scale as f32;
    SavedFormat::empty()
        .transform(move |store, _| {
            let sw = store.get(&format!("{id}_sw"));
            let lut = sample_relu_kan_lut(
                &sw.values,
                in_features,
                out_features,
                grid_size,
                support_width,
                grid_range,
                sample_range,
                num_samples,
            );
            lut.into_iter().map(|v| v.clamp(-max_val, max_val)).collect()
        })
        .round()
        .quantise::<i8>(quant_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_zero_outside_support() {
        let (s_grid, e_grid) = make_relu_kan_grid(5, 3, (0.0, 1.0));
        for (i, (&s, &e)) in s_grid.iter().zip(&e_grid).enumerate() {
            // Just below s_i and just above e_i should be zero
            let just_below = eval_relu_kan_basis(s - 1e-3, &s_grid, &e_grid);
            let just_above = eval_relu_kan_basis(e + 1e-3, &s_grid, &e_grid);
            assert_eq!(just_below[i], 0.0, "basis {i} non-zero just below s={s}");
            assert_eq!(just_above[i], 0.0, "basis {i} non-zero just above e={e}");
        }
    }

    #[test]
    fn basis_peak_one_at_midpoint() {
        // With absorbed normalizer, R_i should equal exactly 1.0 at (s_i + e_i) / 2.
        let (s_grid, e_grid) = make_relu_kan_grid(5, 3, (0.0, 1.0));
        for (i, (&s, &e)) in s_grid.iter().zip(&e_grid).enumerate() {
            let mid = 0.5 * (s + e);
            let basis = eval_relu_kan_basis(mid, &s_grid, &e_grid);
            assert!(
                (basis[i] - 1.0).abs() < 1e-5,
                "basis {i} peak at midpoint {mid}: got {}, want 1.0",
                basis[i]
            );
        }
    }

    #[test]
    fn basis_non_negative() {
        let (s_grid, e_grid) = make_relu_kan_grid(5, 3, (0.0, 1.0));
        for &x in &[0.0, 0.1, 0.25, 0.4, 0.6, 0.9, 1.0] {
            let basis = eval_relu_kan_basis(x, &s_grid, &e_grid);
            for (i, &v) in basis.iter().enumerate() {
                assert!(v >= 0.0, "negative basis value: R[{i}]({x}) = {v}");
            }
        }
    }

    #[test]
    fn make_grid_correct_layout() {
        // For grid_size=5, support_width=3, grid_range=(0, 1):
        // num_basis = 8, span = 1, g = 5, k = 3
        // s_i = (i - 3) / 5
        // e_i = s_i + 4/5
        let (s_grid, e_grid) = make_relu_kan_grid(5, 3, (0.0, 1.0));
        assert_eq!(s_grid.len(), 8);
        assert_eq!(e_grid.len(), 8);
        for i in 0..8 {
            let s_expected = (i as f32 - 3.0) / 5.0;
            let e_expected = s_expected + 4.0 / 5.0;
            assert!((s_grid[i] - s_expected).abs() < 1e-6, "s[{i}] = {}", s_grid[i]);
            assert!((e_grid[i] - e_expected).abs() < 1e-6, "e[{i}] = {}", e_grid[i]);
        }
    }

    #[test]
    fn lut_matches_direct_evaluation() {
        // Set sw[0, 0, 0] = 1.0 (only the first basis weight of the first input is on).
        // The LUT for input 0 should equal R_0(x_k) at each sample point.
        let in_features = 1;
        let out_features = 1;
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;
        let num_samples = 32;

        let mut sw_vals = vec![0.0f32; out_features * in_features * num_basis];
        sw_vals[0] = 1.0;

        let lut = sample_relu_kan_lut(
            &sw_vals,
            in_features, out_features,
            grid_size, support_width,
            (0.0, 1.0), (0.0, 1.0),
            num_samples,
        );

        let (s_grid, e_grid) = make_relu_kan_grid(grid_size, support_width, (0.0, 1.0));
        for k in 0..num_samples {
            let x = k as f32 / (num_samples - 1) as f32;
            let expected = eval_relu_kan_basis(x, &s_grid, &e_grid)[0];
            let actual = lut[k];
            assert!(
                (actual - expected).abs() < 1e-6,
                "LUT mismatch at k={k}, x={x}: expected={expected}, actual={actual}"
            );
        }
    }

    #[test]
    fn lut_reconstructs_weighted_sum() {
        // Set every basis weight to 1.0 for one (in, out) edge.
        // The LUT at sample k should equal sum_b R_b(x_k).
        let in_features = 2;
        let out_features = 1;
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;
        let num_samples = 16;

        let mut sw_vals = vec![0.0f32; out_features * in_features * num_basis];
        // Fill weights for input feature 0 only, so input 1's slice stays zero.
        for b in 0..num_basis {
            let sw_idx = (0 * num_basis + b) * out_features + 0;
            sw_vals[sw_idx] = 1.0;
        }

        let lut = sample_relu_kan_lut(
            &sw_vals,
            in_features, out_features,
            grid_size, support_width,
            (0.0, 1.0), (0.0, 1.0),
            num_samples,
        );

        let (s_grid, e_grid) = make_relu_kan_grid(grid_size, support_width, (0.0, 1.0));
        for k in 0..num_samples {
            let x = k as f32 / (num_samples - 1) as f32;
            let expected: f32 = eval_relu_kan_basis(x, &s_grid, &e_grid).iter().sum();
            // input 0 → lut[0 * num_samples * out + k * out + 0]
            let actual = lut[k * out_features];
            assert!(
                (actual - expected).abs() < 1e-5,
                "weighted-sum LUT mismatch at k={k}, x={x}: expected={expected}, actual={actual}"
            );

            // input 1 (all weights zero) should be exactly 0
            let zero_actual = lut[1 * num_samples * out_features + k * out_features];
            assert_eq!(zero_actual, 0.0, "input-1 LUT non-zero at k={k}");
        }
    }

    #[test]
    fn lut_output_size() {
        let in_f = 4;
        let out_f = 2;
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;
        let num_samples = 64;

        let sw_vals = vec![0.0f32; out_f * in_f * num_basis];
        let lut = sample_relu_kan_lut(
            &sw_vals,
            in_f, out_f, grid_size, support_width,
            (0.0, 1.0), (0.0, 1.0), num_samples,
        );
        assert_eq!(lut.len(), in_f * num_samples * out_f);
    }

    #[test]
    fn negative_domain_grid() {
        // KAN2 in variant E uses grid_range (-1, 1). Verify s_grid spans the
        // domain symmetrically and the basis is non-degenerate at x=0.
        let (s_grid, e_grid) = make_relu_kan_grid(5, 3, (-1.0, 1.0));
        // Some basis must be non-zero at the middle of the domain.
        let basis = eval_relu_kan_basis(0.0, &s_grid, &e_grid);
        let total: f32 = basis.iter().sum();
        assert!(total > 0.0, "no basis active at x=0 for grid (-1, 1)");
    }
}
