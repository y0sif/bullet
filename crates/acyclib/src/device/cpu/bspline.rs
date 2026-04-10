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
