use crate::device::{OperationResult, operation::relu_kan::ReluKanOps};

use super::{CpuBuffer, CpuThread};

impl ReluKanOps for CpuThread {
    fn relu_kan_basis_fwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        support_width: usize,
        input: &CpuBuffer<f32>,
        grid: &CpuBuffer<f32>,
        output: &mut CpuBuffer<f32>,
    ) -> OperationResult<Self::DeviceError> {
        let num_basis = grid_size + support_width;
        let s_grid = &grid.buf[..num_basis];
        let e_grid = &grid.buf[num_basis..2 * num_basis];

        for b in 0..batch_size {
            for j in 0..in_features {
                let x = input.buf[b * in_features + j];
                let out_offset = b * in_features * num_basis + j * num_basis;

                for i in 0..num_basis {
                    let u = (e_grid[i] - x).max(0.0);
                    let v = (x - s_grid[i]).max(0.0);
                    let width = e_grid[i] - s_grid[i];
                    let norm = if width > 0.0 { 16.0 / (width * width * width * width) } else { 0.0 };
                    let uv = u * v;
                    output.buf[out_offset + i] = norm * uv * uv;
                }
            }
        }

        Ok(())
    }

    fn relu_kan_basis_bwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        support_width: usize,
        input: &CpuBuffer<f32>,
        grid: &CpuBuffer<f32>,
        output_grad: &CpuBuffer<f32>,
        input_grad: &mut CpuBuffer<f32>,
    ) -> OperationResult<Self::DeviceError> {
        let num_basis = grid_size + support_width;
        let s_grid = &grid.buf[..num_basis];
        let e_grid = &grid.buf[num_basis..2 * num_basis];

        for b in 0..batch_size {
            for j in 0..in_features {
                let x = input.buf[b * in_features + j];
                let grad_offset = b * in_features * num_basis + j * num_basis;

                let mut grad_acc = 0.0f32;
                for i in 0..num_basis {
                    // Derivative is zero outside the support.
                    if x <= s_grid[i] || x >= e_grid[i] {
                        continue;
                    }

                    let u = e_grid[i] - x;
                    let v = x - s_grid[i];
                    let width = e_grid[i] - s_grid[i];
                    let norm = 16.0 / (width * width * width * width);

                    // d/dx (u*v)^2 = 2 * u * v * (d(u*v)/dx)
                    // d(u*v)/dx = -v + u  (inside support, both ReLUs active)
                    let deriv = 2.0 * norm * u * v * (u - v);
                    grad_acc += output_grad.buf[grad_offset + i] * deriv;
                }
                input_grad.buf[b * in_features + j] += grad_acc;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::device::DeviceBuffer;

    use super::*;

    fn make_buf(vals: &[f32]) -> CpuBuffer<f32> {
        let mut buf = CpuBuffer::<f32>::new(Arc::new(CpuThread), vals.len()).unwrap();
        buf.load_from_slice(vals).unwrap();
        buf
    }

    /// Build the ReLU-KAN (s, e) grid for domain [0, 1] per the paper.
    /// s_i = (i - k) / G, e_i = s_i + (k+1)/G for i in 0..G+k
    fn make_grid_01(grid_size: usize, support_width: usize) -> Vec<f32> {
        let g = grid_size as f32;
        let k = support_width as f32;
        let num_basis = grid_size + support_width;
        let mut grid = Vec::with_capacity(2 * num_basis);
        // s values
        for i in 0..num_basis {
            grid.push((i as f32 - k) / g);
        }
        // e values
        for i in 0..num_basis {
            grid.push((i as f32 - k) / g + (k + 1.0) / g);
        }
        grid
    }

    #[test]
    fn peak_at_support_midpoint() {
        // R_i is a symmetric bump centered at (s_i + e_i)/2 with peak value 1
        // (because the normalizer 16/(e-s)^4 makes (width/2 * width/2)^2 = width^4/16 → 1).
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;

        let grid_vals = make_grid_01(grid_size, support_width);
        let grid = make_buf(&grid_vals);

        for i in 0..num_basis {
            let s = grid_vals[i];
            let e = grid_vals[num_basis + i];
            let mid = 0.5 * (s + e);

            let input = make_buf(&[mid]);
            let mut output = make_buf(&vec![0.0; num_basis]);

            CpuThread::relu_kan_basis_fwd(
                1, 1, grid_size, support_width, &input, &grid, &mut output,
            )
            .unwrap();

            // R_i at its midpoint should be exactly 1.0
            assert!(
                (output.buf[i] - 1.0).abs() < 1e-5,
                "R_{i} at midpoint {mid}: expected 1.0, got {}",
                output.buf[i]
            );
        }
    }

    #[test]
    fn non_negative_and_zero_outside_support() {
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;

        let grid_vals = make_grid_01(grid_size, support_width);
        let grid = make_buf(&grid_vals);

        for &x in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let input = make_buf(&[x]);
            let mut output = make_buf(&vec![0.0; num_basis]);

            CpuThread::relu_kan_basis_fwd(
                1, 1, grid_size, support_width, &input, &grid, &mut output,
            )
            .unwrap();

            for i in 0..num_basis {
                let s = grid_vals[i];
                let e = grid_vals[num_basis + i];
                let v = output.buf[i];
                assert!(v >= 0.0, "Negative basis value at i={i}, x={x}: {v}");
                if x <= s || x >= e {
                    assert!(
                        v.abs() < 1e-6,
                        "Nonzero basis outside support at i={i}, x={x}, (s,e)=({s},{e}): {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn local_support() {
        // At any point, at most (support_width + 1) basis functions overlap
        // (since basis i covers [s_i, e_i] of width (k+1)/G and they're spaced 1/G apart).
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;

        let grid_vals = make_grid_01(grid_size, support_width);
        let grid = make_buf(&grid_vals);

        for &x in &[0.05, 0.2, 0.45, 0.6, 0.8, 0.95] {
            let input = make_buf(&[x]);
            let mut output = make_buf(&vec![0.0; num_basis]);

            CpuThread::relu_kan_basis_fwd(
                1, 1, grid_size, support_width, &input, &grid, &mut output,
            )
            .unwrap();

            let nonzero_count = output.buf[..num_basis].iter().filter(|&&v| v > 1e-10).count();
            assert!(
                nonzero_count <= support_width + 1,
                "Too many nonzero basis: {nonzero_count} > {} at x={x}",
                support_width + 1
            );
        }
    }

    #[test]
    fn batched_multi_feature() {
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;
        let in_features = 3;
        let batch_size = 2;

        let grid_vals = make_grid_01(grid_size, support_width);
        let grid = make_buf(&grid_vals);

        let input_vals = [0.0, 0.5, 0.2, 0.3, 0.8, 0.9];
        let input = make_buf(&input_vals);
        let mut output = make_buf(&vec![0.0; batch_size * in_features * num_basis]);

        CpuThread::relu_kan_basis_fwd(
            batch_size,
            in_features,
            grid_size,
            support_width,
            &input,
            &grid,
            &mut output,
        )
        .unwrap();

        // Spot-check: run a single-feature forward at input_vals[5] = 0.9 and compare the
        // last in-feature slice of the batched output.
        let probe_input = make_buf(&[0.9]);
        let mut probe_out = make_buf(&vec![0.0; num_basis]);
        CpuThread::relu_kan_basis_fwd(
            1,
            1,
            grid_size,
            support_width,
            &probe_input,
            &grid,
            &mut probe_out,
        )
        .unwrap();

        let last_offset = (batch_size - 1) * in_features * num_basis + (in_features - 1) * num_basis;
        for i in 0..num_basis {
            assert!(
                (output.buf[last_offset + i] - probe_out.buf[i]).abs() < 1e-6,
                "Batched output mismatch at i={i}: {} vs {}",
                output.buf[last_offset + i],
                probe_out.buf[i]
            );
        }
    }

    #[test]
    fn gradient_finite_difference() {
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;
        let eps = 1e-4;

        let grid_vals = make_grid_01(grid_size, support_width);
        let grid = make_buf(&grid_vals);

        // Sample interior points of several basis supports (avoid endpoints which
        // are non-differentiable in the ReLU sense).
        for &x in &[0.1, 0.25, 0.4, 0.55, 0.7, 0.85] {
            let input_plus = make_buf(&[x + eps]);
            let input_minus = make_buf(&[x - eps]);
            let mut out_plus = make_buf(&vec![0.0; num_basis]);
            let mut out_minus = make_buf(&vec![0.0; num_basis]);

            CpuThread::relu_kan_basis_fwd(
                1, 1, grid_size, support_width, &input_plus, &grid, &mut out_plus,
            )
            .unwrap();
            CpuThread::relu_kan_basis_fwd(
                1, 1, grid_size, support_width, &input_minus, &grid, &mut out_minus,
            )
            .unwrap();

            let mut numerical_grad = 0.0f32;
            let mut output_grad_vals = vec![0.0; num_basis];
            for i in 0..num_basis {
                output_grad_vals[i] = (i as f32 + 1.0) / num_basis as f32;
                numerical_grad +=
                    output_grad_vals[i] * (out_plus.buf[i] - out_minus.buf[i]) / (2.0 * eps);
            }

            let input = make_buf(&[x]);
            let output_grad = make_buf(&output_grad_vals);
            let mut input_grad = make_buf(&[0.0]);

            CpuThread::relu_kan_basis_bwd(
                1,
                1,
                grid_size,
                support_width,
                &input,
                &grid,
                &output_grad,
                &mut input_grad,
            )
            .unwrap();

            let analytic_grad = input_grad.buf[0];
            let rel_err = if numerical_grad.abs() > 1e-6 {
                (analytic_grad - numerical_grad).abs() / numerical_grad.abs()
            } else {
                (analytic_grad - numerical_grad).abs()
            };

            assert!(
                rel_err < 5e-3,
                "Gradient mismatch at x={x}: analytic={analytic_grad:.6}, numerical={numerical_grad:.6}, rel_err={rel_err:.6}"
            );
        }
    }

    #[test]
    fn zero_gradient_outside_all_supports() {
        let grid_size = 5;
        let support_width = 3;
        let num_basis = grid_size + support_width;

        let grid_vals = make_grid_01(grid_size, support_width);
        let grid = make_buf(&grid_vals);

        // Pick x way outside the grid; every basis is zero and so is the gradient.
        for &x in &[-5.0, 10.0] {
            let input = make_buf(&[x]);
            let output_grad = make_buf(&vec![1.0; num_basis]);
            let mut input_grad = make_buf(&[0.0]);

            CpuThread::relu_kan_basis_bwd(
                1,
                1,
                grid_size,
                support_width,
                &input,
                &grid,
                &output_grad,
                &mut input_grad,
            )
            .unwrap();

            assert!(
                input_grad.buf[0].abs() < 1e-6,
                "Nonzero gradient outside grid at x={x}: {}",
                input_grad.buf[0]
            );
        }
    }
}
