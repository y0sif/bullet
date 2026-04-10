use crate::device::{OperationResult, operation::bspline::BSplineOps};

use super::{CpuBuffer, CpuThread};

impl BSplineOps for CpuThread {
    fn bspline_basis_fwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        spline_order: usize,
        input: &CpuBuffer<f32>,
        grid: &CpuBuffer<f32>,
        output: &mut CpuBuffer<f32>,
    ) -> OperationResult<Self::DeviceError> {
        let num_basis = grid_size + spline_order;
        let num_knots = grid_size + 2 * spline_order + 1;
        let num_degree0 = grid_size + 2 * spline_order;
        let grid = &grid.buf[..num_knots];

        // Scratch space for Cox-de Boor recursion (reused across iterations)
        let mut prev = vec![0.0f32; num_degree0];
        let mut curr = vec![0.0f32; num_degree0];

        for b in 0..batch_size {
            for j in 0..in_features {
                let x = input.buf[b * in_features + j];
                let out_offset = b * in_features * num_basis + j * num_basis;

                // Degree 0: indicator functions
                for i in 0..num_degree0 {
                    prev[i] = if x >= grid[i] && x < grid[i + 1] {
                        1.0
                    } else {
                        0.0
                    };
                }
                // Right boundary: include x == last knot in the last interval
                if x >= grid[num_knots - 1] {
                    for v in &mut prev[..num_degree0] {
                        *v = 0.0;
                    }
                    prev[num_degree0 - 1] = 1.0;
                }

                // Cox-de Boor recursion for degrees 1..=spline_order
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

                // Write result: prev[0..num_basis] contains the final basis values
                output.buf[out_offset..out_offset + num_basis].copy_from_slice(&prev[..num_basis]);
            }
        }

        Ok(())
    }

    fn bspline_basis_bwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        spline_order: usize,
        input: &CpuBuffer<f32>,
        grid: &CpuBuffer<f32>,
        output_grad: &CpuBuffer<f32>,
        input_grad: &mut CpuBuffer<f32>,
    ) -> OperationResult<Self::DeviceError> {
        let k = spline_order;
        let num_basis = grid_size + k;
        let num_knots = grid_size + 2 * k + 1;
        let num_degree0 = grid_size + 2 * k;
        let grid = &grid.buf[..num_knots];

        // Scratch space
        let mut prev = vec![0.0f32; num_degree0];
        let mut curr = vec![0.0f32; num_degree0];
        let mut deriv = vec![0.0f32; num_basis];

        for b in 0..batch_size {
            for j in 0..in_features {
                let x = input.buf[b * in_features + j];
                let grad_offset = b * in_features * num_basis + j * num_basis;

                if k == 0 {
                    // Degree-0 B-splines: zero derivative
                    continue;
                }

                // Compute degree (k-1) basis values via Cox-de Boor
                for i in 0..num_degree0 {
                    prev[i] = if x >= grid[i] && x < grid[i + 1] {
                        1.0
                    } else {
                        0.0
                    };
                }
                if x >= grid[num_knots - 1] {
                    for v in &mut prev[..num_degree0] {
                        *v = 0.0;
                    }
                    prev[num_degree0 - 1] = 1.0;
                }

                for p in 1..k {
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

                // prev now has degree (k-1) basis: length = grid_size + k + 1
                let num_prev = grid_size + k + 1;

                // Derivative: dB_{i,k}/dx = k * (B_{i,k-1}/(t_{i+k}-t_i) - B_{i+1,k-1}/(t_{i+k+1}-t_{i+1}))
                for i in 0..num_basis {
                    let mut val = 0.0;

                    let denom_left = grid[i + k] - grid[i];
                    if denom_left != 0.0 && i < num_prev {
                        val += prev[i] / denom_left;
                    }

                    let denom_right = grid[i + k + 1] - grid[i + 1];
                    if denom_right != 0.0 && (i + 1) < num_prev {
                        val -= prev[i + 1] / denom_right;
                    }

                    deriv[i] = k as f32 * val;
                }

                // Chain rule: input_grad[b,j] += sum_k output_grad[b,j,k] * deriv[k]
                let mut grad_acc = 0.0f32;
                for ki in 0..num_basis {
                    grad_acc += output_grad.buf[grad_offset + ki] * deriv[ki];
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

    fn make_grid(grid_size: usize, spline_order: usize) -> Vec<f32> {
        let num_knots = grid_size + 2 * spline_order + 1;
        let h = 2.0 / grid_size as f32;
        let mut grid = Vec::with_capacity(num_knots);
        for i in (1..=spline_order).rev() {
            grid.push(-1.0 - i as f32 * h);
        }
        for i in 0..=grid_size {
            grid.push(-1.0 + i as f32 * h);
        }
        for i in 1..=spline_order {
            grid.push(1.0 + i as f32 * h);
        }
        grid
    }

    #[test]
    fn partition_of_unity() {
        // B-spline basis functions must sum to 1.0 for any x in the interior
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order; // 8

        let grid_vals = make_grid(grid_size, spline_order);
        let grid = make_buf(&grid_vals);

        for &x in &[-0.9, -0.5, -0.1, 0.0, 0.3, 0.7, 0.99] {
            let input = make_buf(&[x]);
            let mut output = make_buf(&vec![0.0; num_basis]);

            CpuThread::bspline_basis_fwd(1, 1, grid_size, spline_order, &input, &grid, &mut output)
                .unwrap();

            let sum: f32 = output.buf[..num_basis].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Partition of unity violated: sum={sum} for x={x}"
            );
        }
    }

    #[test]
    fn non_negative() {
        // B-spline basis values must be >= 0
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order;

        let grid_vals = make_grid(grid_size, spline_order);
        let grid = make_buf(&grid_vals);

        for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let input = make_buf(&[x]);
            let mut output = make_buf(&vec![0.0; num_basis]);

            CpuThread::bspline_basis_fwd(1, 1, grid_size, spline_order, &input, &grid, &mut output)
                .unwrap();

            for (i, &v) in output.buf[..num_basis].iter().enumerate() {
                assert!(v >= 0.0, "Negative basis value: B[{i}]({x}) = {v}");
            }
        }
    }

    #[test]
    fn local_support() {
        // Each basis function B_{i,k} is nonzero only on [t_i, t_{i+k+1})
        // So at most (spline_order + 1) basis functions are nonzero at any point
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order;

        let grid_vals = make_grid(grid_size, spline_order);
        let grid = make_buf(&grid_vals);

        for &x in &[-0.8, -0.3, 0.1, 0.6] {
            let input = make_buf(&[x]);
            let mut output = make_buf(&vec![0.0; num_basis]);

            CpuThread::bspline_basis_fwd(1, 1, grid_size, spline_order, &input, &grid, &mut output)
                .unwrap();

            let nonzero_count = output.buf[..num_basis]
                .iter()
                .filter(|&&v| v > 1e-10)
                .count();

            assert!(
                nonzero_count <= spline_order + 1,
                "Too many nonzero basis functions: {nonzero_count} > {} for x={x}",
                spline_order + 1
            );
        }
    }

    #[test]
    fn batched_multi_feature() {
        // Test with batch_size=2, in_features=3
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order;
        let in_features = 3;
        let batch_size = 2;

        let grid_vals = make_grid(grid_size, spline_order);
        let grid = make_buf(&grid_vals);

        let input_vals = [0.0, 0.5, -0.5, 0.3, -0.8, 0.9];
        let input = make_buf(&input_vals);
        let mut output = make_buf(&vec![0.0; batch_size * in_features * num_basis]);

        CpuThread::bspline_basis_fwd(
            batch_size,
            in_features,
            grid_size,
            spline_order,
            &input,
            &grid,
            &mut output,
        )
        .unwrap();

        // Verify partition of unity for each (batch, feature) pair
        for b in 0..batch_size {
            for j in 0..in_features {
                let offset = b * in_features * num_basis + j * num_basis;
                let sum: f32 = output.buf[offset..offset + num_basis].iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Partition of unity violated: batch={b}, feature={j}, sum={sum}"
                );
            }
        }
    }

    #[test]
    fn gradient_finite_difference() {
        // Verify backward pass via finite differences: d/dx basis(x) ≈ (basis(x+eps) - basis(x-eps)) / (2*eps)
        let grid_size = 5;
        let spline_order = 3;
        let num_basis = grid_size + spline_order;
        let eps = 1e-4;

        let grid_vals = make_grid(grid_size, spline_order);
        let grid = make_buf(&grid_vals);

        for &x in &[-0.7, -0.2, 0.0, 0.4, 0.8] {
            // Forward at x+eps and x-eps
            let input_plus = make_buf(&[x + eps]);
            let input_minus = make_buf(&[x - eps]);
            let mut out_plus = make_buf(&vec![0.0; num_basis]);
            let mut out_minus = make_buf(&vec![0.0; num_basis]);

            CpuThread::bspline_basis_fwd(1, 1, grid_size, spline_order, &input_plus, &grid, &mut out_plus).unwrap();
            CpuThread::bspline_basis_fwd(1, 1, grid_size, spline_order, &input_minus, &grid, &mut out_minus).unwrap();

            // Numerical gradient: for each basis function, compute derivative
            let mut numerical_grad = 0.0f32;
            let mut output_grad_vals = vec![0.0; num_basis];
            // Use a random-ish output_grad to test the chain rule
            for i in 0..num_basis {
                output_grad_vals[i] = (i as f32 + 1.0) / num_basis as f32;
                numerical_grad += output_grad_vals[i] * (out_plus.buf[i] - out_minus.buf[i]) / (2.0 * eps);
            }

            // Analytic gradient via backward pass
            let input = make_buf(&[x]);
            let output_grad = make_buf(&output_grad_vals);
            let mut input_grad = make_buf(&[0.0]);

            CpuThread::bspline_basis_bwd(
                1, 1, grid_size, spline_order, &input, &grid, &output_grad, &mut input_grad,
            )
            .unwrap();

            let analytic_grad = input_grad.buf[0];
            let rel_err = if numerical_grad.abs() > 1e-8 {
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
    fn linear_bspline_piecewise_linear() {
        // Order-1 (linear) B-splines should be triangular hat functions
        let grid_size = 4;
        let spline_order = 1;
        let num_basis = grid_size + spline_order; // 5

        let grid_vals = make_grid(grid_size, spline_order);
        let grid = make_buf(&grid_vals);

        // At x = midpoint of first interior interval, basis[1] should be 0.5 and basis[2] should be 0.5
        let h = 2.0 / grid_size as f32; // 0.5
        let x = -1.0 + 0.5 * h; // midpoint of first interior interval
        let input = make_buf(&[x]);
        let mut output = make_buf(&vec![0.0; num_basis]);

        CpuThread::bspline_basis_fwd(1, 1, grid_size, spline_order, &input, &grid, &mut output).unwrap();

        // For linear B-splines, at a midpoint between knots, exactly 2 basis functions should be active at 0.5 each
        let active: Vec<(usize, f32)> = output.buf[..num_basis]
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| *v > 1e-10)
            .collect();

        assert_eq!(active.len(), 2, "Expected 2 active basis functions, got {active:?}");
        for &(_, v) in &active {
            assert!(
                (v - 0.5f32).abs() < 1e-6,
                "Expected 0.5 at midpoint, got {active:?}"
            );
        }
    }
}
