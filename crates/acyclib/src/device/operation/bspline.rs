use crate::device::{Device, OperationResult};

/// Device-level B-spline basis evaluation operations.
///
/// Evaluates Cox-de Boor B-spline basis functions for KAN layers.
/// For each input feature, computes `grid_size + spline_order` basis values
/// using a shared knot vector (grid).
pub trait BSplineOps: Device {
    /// Forward: evaluate B-spline basis functions.
    ///
    /// - `input`:  (batch_size * in_features,) flat buffer
    /// - `grid`:   (grid_size + 2 * spline_order + 1,) knot vector
    /// - `output`: (batch_size * in_features * num_basis,) where num_basis = grid_size + spline_order
    fn bspline_basis_fwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        spline_order: usize,
        input: &Self::BufferF32,
        grid: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    /// Backward: gradient of loss w.r.t. input, given gradient w.r.t. basis output.
    ///
    /// Accumulates into `input_grad` (does not zero first).
    fn bspline_basis_bwd(
        batch_size: usize,
        in_features: usize,
        grid_size: usize,
        spline_order: usize,
        input: &Self::BufferF32,
        grid: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;
}
