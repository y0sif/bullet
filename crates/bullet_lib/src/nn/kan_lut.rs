//! KAN LUT (Lookup Table) sampling and export.
//!
//! At training time, KAN layers use B-spline basis evaluation (Cox-de Boor recursion).
//! At inference time in a chess engine, we replace this with lookup tables:
//!
//! 1. For each edge (i, j) in the KAN layer, sample the learned activation
//!    function at `num_samples` evenly-spaced points
//! 2. Quantize the sampled values to int8/int16
//! 3. Export via Bullet's `SavedFormat` system
//!
//! Engine inference becomes: `y[j] += lut[i][bucket(x[i])][j]` (input-major for cache locality)

use acyclib::graph::save::SavedFormat;

/// Evaluate B-spline basis functions at a single point.
///
/// Returns a vector of `num_basis = grid_size + spline_order` values.
/// This is a CPU-only reimplementation of the Cox-de Boor recursion
/// from `acyclib::device::cpu::bspline`, adapted for single-point evaluation.
fn eval_bspline_basis(x: f32, grid: &[f32], grid_size: usize, spline_order: usize) -> Vec<f32> {
    let num_basis = grid_size + spline_order;
    let num_knots = grid_size + 2 * spline_order + 1;
    let num_degree0 = grid_size + 2 * spline_order;
    assert_eq!(grid.len(), num_knots);

    let mut prev = vec![0.0f32; num_degree0];
    let mut curr = vec![0.0f32; num_degree0];

    // Degree 0: indicator functions
    for i in 0..num_degree0 {
        prev[i] = if x >= grid[i] && x < grid[i + 1] { 1.0 } else { 0.0 };
    }
    // Right boundary: include x == last knot
    if x >= grid[num_knots - 1] {
        for v in &mut prev[..num_degree0] {
            *v = 0.0;
        }
        prev[num_degree0 - 1] = 1.0;
    }

    // Cox-de Boor recursion
    for p in 1..=spline_order {
        let num_at_p = num_degree0 - p;
        for i in 0..num_at_p {
            let mut val = 0.0;

            let denom_left = grid[i + p] - grid[i];
            if denom_left != 0.0 {
                val += (x - grid[i]) / denom_left * prev[i];
            }

            let denom_right = grid[i + p + 1] - grid[i + 1];
            if denom_right != 0.0 {
                val += (grid[i + p + 1] - x) / denom_right * prev[i + 1];
            }

            curr[i] = val;
        }

        prev[..num_at_p].copy_from_slice(&curr[..num_at_p]);
    }

    prev[..num_basis].to_vec()
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Build the uniform knot vector for a B-spline grid.
///
/// Same logic as `kan_layer()` in `kan.rs`.
fn make_grid(grid_size: usize, spline_order: usize, grid_range: (f32, f32)) -> Vec<f32> {
    let num_knots = grid_size + 2 * spline_order + 1;
    let (grid_min, grid_max) = grid_range;
    let h = (grid_max - grid_min) / grid_size as f32;
    let mut grid = Vec::with_capacity(num_knots);

    for i in (1..=spline_order).rev() {
        grid.push(grid_min - i as f32 * h);
    }
    for i in 0..=grid_size {
        grid.push(grid_min + i as f32 * h);
    }
    for i in 1..=spline_order {
        grid.push(grid_max + i as f32 * h);
    }

    assert_eq!(grid.len(), num_knots);
    grid
}

/// Sample the full KAN LUT from raw weight data.
///
/// For each edge (i, j), evaluates the learned activation at `num_samples`
/// evenly-spaced points in `sample_range`:
///
/// ```text
/// lut[i][k][j] = sw[j, i*nb..(i+1)*nb] · basis(x_k) + bw[j, i] · silu(x_k)
/// ```
///
/// Returns a flat `Vec<f32>` with layout `[in_features][num_samples][out_features]`.
/// This layout is cache-friendly for inference: for a given input i with index k,
/// all output weights `lut[i][k][0..out]` are contiguous.
pub fn sample_kan_lut(
    sw_vals: &[f32],
    bw_vals: &[f32],
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    spline_order: usize,
    grid_range: (f32, f32),
    sample_range: (f32, f32),
    num_samples: usize,
) -> Vec<f32> {
    let num_basis = grid_size + spline_order;
    let grid = make_grid(grid_size, spline_order, grid_range);

    assert_eq!(sw_vals.len(), out_features * in_features * num_basis);
    assert_eq!(bw_vals.len(), out_features * in_features);

    // Precompute sample points
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

    // Precompute basis values at all sample points
    let basis_at_samples: Vec<Vec<f32>> = sample_points
        .iter()
        .map(|&x| eval_bspline_basis(x, &grid, grid_size, spline_order))
        .collect();

    // Precompute SiLU at all sample points
    let silu_at_samples: Vec<f32> = sample_points.iter().map(|&x| silu(x)).collect();

    // Build LUT: layout [in_features][num_samples][out_features]
    let total_size = in_features * num_samples * out_features;
    let mut lut = vec![0.0f32; total_size];

    for i in 0..in_features {
        for k in 0..num_samples {
            for j in 0..out_features {
                // Spline path: sw[j, i*nb..(i+1)*nb] · basis(x_k)
                let mut val = 0.0f32;
                for b in 0..num_basis {
                    // Column-major: sw[j, i*nb + b] = sw_vals[(i*nb + b) * out_features + j]
                    let sw_idx = (i * num_basis + b) * out_features + j;
                    val += sw_vals[sw_idx] * basis_at_samples[k][b];
                }

                // Base path: bw[j, i] · silu(x_k)
                // Column-major: bw[j, i] = bw_vals[i * out_features + j]
                let bw_idx = i * out_features + j;
                val += bw_vals[bw_idx] * silu_at_samples[k];

                lut[i * num_samples * out_features + k * out_features + j] = val;
            }
        }
    }

    lut
}

/// Generate `SavedFormat` entry for a KAN layer's LUT export (i8, cache-friendly layout).
///
/// # Binary layout
///
/// The LUT is stored as `[in_features][num_samples][out_features]` with i8 quantization.
/// This layout is cache-friendly: for a given input with index k, the output weights
/// `lut[i][k][0..out]` are contiguous in memory.
pub fn kan_lut_save_format(
    id: &str,
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    spline_order: usize,
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
            let bw = store.get(&format!("{id}_bw"));
            let lut = sample_kan_lut(
                &sw.values,
                &bw.values,
                in_features,
                out_features,
                grid_size,
                spline_order,
                grid_range,
                sample_range,
                num_samples,
            );
            // Clamp to representable i8 range to prevent quantization overflow
            lut.into_iter().map(|v| v.clamp(-max_val, max_val)).collect()
        })
        .round()
        .quantise::<i8>(quant_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bspline_partition_of_unity() {
        let grid_size = 5;
        let spline_order = 3;
        let grid = make_grid(grid_size, spline_order, (-1.0, 1.0));

        for &x in &[-0.9, -0.5, 0.0, 0.3, 0.7, 0.99] {
            let basis = eval_bspline_basis(x, &grid, grid_size, spline_order);
            let sum: f32 = basis.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Partition of unity violated: sum={sum} for x={x}"
            );
        }
    }

    #[test]
    fn bspline_non_negative() {
        let grid_size = 5;
        let spline_order = 3;
        let grid = make_grid(grid_size, spline_order, (-1.0, 1.0));

        for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let basis = eval_bspline_basis(x, &grid, grid_size, spline_order);
            for (i, &v) in basis.iter().enumerate() {
                assert!(v >= 0.0, "Negative basis value: B[{i}]({x}) = {v}");
            }
        }
    }

    #[test]
    fn silu_known_values() {
        // silu(0) = 0
        assert!((silu(0.0)).abs() < 1e-7);
        // silu(x) ≈ x for large x
        assert!((silu(10.0) - 10.0).abs() < 0.001);
        // silu(x) ≈ 0 for large negative x
        assert!(silu(-10.0).abs() < 0.001);
    }

    #[test]
    fn make_grid_correct_length() {
        let grid = make_grid(5, 3, (-1.0, 1.0));
        assert_eq!(grid.len(), 5 + 2 * 3 + 1); // 12
    }

    #[test]
    fn make_grid_matches_kan_layer() {
        // Verify our grid matches what kan_layer() would produce
        let grid = make_grid(5, 3, (-1.0, 1.0));
        let h = 2.0 / 5.0;

        // First extended knot
        assert!((grid[0] - (-1.0 - 3.0 * h)).abs() < 1e-6);
        // First interior knot
        assert!((grid[3] - (-1.0)).abs() < 1e-6);
        // Last interior knot
        assert!((grid[8] - 1.0).abs() < 1e-6);
        // Last extended knot
        assert!((grid[11] - (1.0 + 3.0 * h)).abs() < 1e-6);
    }

    #[test]
    fn lut_matches_direct_evaluation() {
        // Verify that LUT sampling reproduces the same values as direct evaluation
        // using known weights: set all spline weights to 0, base weight to 1.0
        // → LUT should equal silu(x) for each sample point
        let in_features = 2;
        let out_features = 1;
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order; // 8
        let num_samples = 16;

        // Column-major: sw[j, col] = sw_vals[col * out_features + j]
        let sw_vals = vec![0.0f32; out_features * in_features * num_basis];
        // bw[0, 0] = 1.0, bw[0, 1] = 1.0 (column-major: bw_vals[i * out + j])
        let bw_vals = vec![1.0f32; out_features * in_features];

        let lut = sample_kan_lut(
            &sw_vals, &bw_vals,
            in_features, out_features,
            grid_size, spline_order,
            (-1.0, 1.0), (0.0, 1.0),
            num_samples,
        );

        assert_eq!(lut.len(), out_features * in_features * num_samples);

        // For each input feature, LUT should be silu(x) at the sample points
        // Layout: [in][sample][out]
        for i in 0..in_features {
            for k in 0..num_samples {
                let x = k as f32 / (num_samples - 1) as f32; // sample in [0, 1]
                let expected = silu(x);
                let actual = lut[i * num_samples * out_features + k * out_features + 0];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "LUT mismatch at i={i}, k={k}: expected={expected}, actual={actual}"
                );
            }
        }
    }

    #[test]
    fn lut_spline_path_only() {
        // With base weights = 0, LUT should be purely from spline path
        // Set one spline weight to 1.0 and verify the LUT picks up basis values
        let in_features = 1;
        let out_features = 1;
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order; // 8
        let num_samples = 32;

        // Column-major: sw[0, b] = sw_vals[b * 1 + 0]
        // Set only the first basis weight to 1.0
        let mut sw_vals = vec![0.0f32; out_features * in_features * num_basis];
        sw_vals[0] = 1.0; // sw[0, 0] = 1.0

        let bw_vals = vec![0.0f32; out_features * in_features];

        let grid = make_grid(grid_size, spline_order, (-1.0, 1.0));
        let lut = sample_kan_lut(
            &sw_vals, &bw_vals,
            in_features, out_features,
            grid_size, spline_order,
            (-1.0, 1.0), (-1.0, 1.0),
            num_samples,
        );

        // LUT should equal B_0(x) at each sample point
        // Layout: [in][sample][out], with in=1, out=1 → flat [sample]
        for k in 0..num_samples {
            let x = -1.0 + 2.0 * k as f32 / (num_samples - 1) as f32;
            let basis = eval_bspline_basis(x, &grid, grid_size, spline_order);
            let expected = basis[0]; // first basis function
            let actual = lut[k]; // [0][k][0] = k for in=1, out=1
            assert!(
                (actual - expected).abs() < 1e-6,
                "Spline-only LUT mismatch at k={k}, x={x}: expected={expected}, actual={actual}"
            );
        }
    }

    #[test]
    fn lut_output_size() {
        let in_f = 4;
        let out_f = 2;
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order;
        let num_samples = 64;

        let sw_vals = vec![0.0f32; out_f * in_f * num_basis];
        let bw_vals = vec![0.0f32; out_f * in_f];

        let lut = sample_kan_lut(
            &sw_vals, &bw_vals,
            in_f, out_f, grid_size, spline_order,
            (-1.0, 1.0), (-1.0, 1.0), num_samples,
        );

        assert_eq!(lut.len(), out_f * in_f * num_samples);
    }

    #[test]
    fn sample_range_coverage() {
        // Verify sample points span the full range
        let num_samples = 256;
        let (s_min, s_max) = (0.0f32, 1.0f32);
        let points: Vec<f32> = (0..num_samples)
            .map(|k| s_min + (s_max - s_min) * k as f32 / (num_samples - 1) as f32)
            .collect();

        assert!((points[0] - s_min).abs() < 1e-7);
        assert!((points[num_samples - 1] - s_max).abs() < 1e-7);
        // Check uniform spacing
        let step = points[1] - points[0];
        for i in 1..num_samples {
            assert!((points[i] - points[i - 1] - step).abs() < 1e-5);
        }
    }
}
