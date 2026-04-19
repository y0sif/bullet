use crate::device::{Device, OperationResult};

/// Device-level ReLU-KAN basis evaluation operations.
///
/// Evaluates the ReLU-KAN basis functions (arXiv 2406.02075):
/// `R_i(x) = [ReLU(e_i - x) * ReLU(x - s_i)]^2 * 16 / (e_i - s_i)^4`
///
/// Grid layout: a single constant buffer of length `2 * (grid_size + support_width)`
/// where the first half is the support-start array `s_0..s_{G+k-1}` and the
/// second half is the support-end array `e_0..e_{G+k-1}`.
pub trait ReluKanOps: Device {
    /// Forward: evaluate ReLU-KAN basis functions.
    ///
    /// - `input`:  (batch_size * in_features,) flat buffer
    /// - `grid`:   (2 * (grid_size + support_width),) — s values followed by e values
    /// - `output`: (batch_size * in_features * num_basis,) where num_basis = grid_size + support_width
    fn relu_kan_basis_fwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        support_width: usize,
        input: &Self::BufferF32,
        grid: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    /// Backward: gradient of loss w.r.t. input, given gradient w.r.t. basis output.
    ///
    /// Accumulates into `input_grad` (does not zero first).
    fn relu_kan_basis_bwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        support_width: usize,
        input: &Self::BufferF32,
        grid: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;
}
