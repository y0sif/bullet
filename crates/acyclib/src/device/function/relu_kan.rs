use crate::device::{
    OperationError,
    function::DeviceOperation,
    operation::relu_kan::ReluKanOps,
    tensor::TensorRef,
};

#[derive(Clone)]
pub struct ReluKanBasisFwd<D: ReluKanOps> {
    pub input: TensorRef<D>,
    pub grid: TensorRef<D>,
    pub output: TensorRef<D>,
    pub in_features: usize,
    pub grid_size: usize,
    pub support_width: usize,
}

impl<D: ReluKanOps> DeviceOperation<D> for ReluKanBasisFwd<D> {
    fn opname(&self) -> String {
        format!(
            "ReluKanBasisFwd(in={}, grid={}, k={})",
            self.in_features, self.grid_size, self.support_width
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

        D::relu_kan_basis_fwd(
            batch_size,
            self.in_features,
            self.grid_size,
            self.support_width,
            &input.buf,
            &grid.buf,
            &mut output.buf,
        )
    }
}

#[derive(Clone)]
pub struct ReluKanBasisBwd<D: ReluKanOps> {
    pub input: TensorRef<D>,
    pub grid: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub input_grad: TensorRef<D>,
    pub in_features: usize,
    pub grid_size: usize,
    pub support_width: usize,
}

impl<D: ReluKanOps> DeviceOperation<D> for ReluKanBasisBwd<D> {
    fn opname(&self) -> String {
        format!(
            "ReluKanBasisBwd(in={}, grid={}, k={})",
            self.in_features, self.grid_size, self.support_width
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

        D::relu_kan_basis_bwd(
            batch_size,
            self.in_features,
            self.grid_size,
            self.support_width,
            &input.buf,
            &grid.buf,
            &output_grad.buf,
            &mut input_grad.buf,
        )
    }
}
