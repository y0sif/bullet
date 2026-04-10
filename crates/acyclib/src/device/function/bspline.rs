use crate::device::{
    OperationError,
    function::DeviceOperation,
    operation::bspline::BSplineOps,
    tensor::TensorRef,
};

#[derive(Clone)]
pub struct BSplineBasisFwd<D: BSplineOps> {
    pub input: TensorRef<D>,
    pub grid: TensorRef<D>,
    pub output: TensorRef<D>,
    pub in_features: usize,
    pub grid_size: usize,
    pub spline_order: usize,
}

impl<D: BSplineOps> DeviceOperation<D> for BSplineBasisFwd<D> {
    fn opname(&self) -> String {
        format!(
            "BSplineBasisFwd(in={}, grid={}, order={})",
            self.in_features, self.grid_size, self.spline_order
        )
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let grid = self.grid.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let batch_size = input.batch_size().unwrap_or(1);

        D::bspline_basis_fwd(
            batch_size,
            self.in_features,
            self.grid_size,
            self.spline_order,
            &input.buf,
            &grid.buf,
            &mut output.buf,
        )
    }
}

#[derive(Clone)]
pub struct BSplineBasisBwd<D: BSplineOps> {
    pub input: TensorRef<D>,
    pub grid: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub input_grad: TensorRef<D>,
    pub in_features: usize,
    pub grid_size: usize,
    pub spline_order: usize,
}

impl<D: BSplineOps> DeviceOperation<D> for BSplineBasisBwd<D> {
    fn opname(&self) -> String {
        format!(
            "BSplineBasisBwd(in={}, grid={}, order={})",
            self.in_features, self.grid_size, self.spline_order
        )
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let grid = self.grid.dense();
        let output_grad = self.output_grad.dense();
        let mut input_grad = self.input_grad.dense_mut();

        if input.batch_size() != output_grad.batch_size()
            || input.batch_size() != input_grad.batch_size()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let batch_size = input.batch_size().unwrap_or(1);

        D::bspline_basis_bwd(
            batch_size,
            self.in_features,
            self.grid_size,
            self.spline_order,
            &input.buf,
            &grid.buf,
            &output_grad.buf,
            &mut input_grad.buf,
        )
    }
}
