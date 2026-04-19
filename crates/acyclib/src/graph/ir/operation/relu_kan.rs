use crate::{
    dag::NodeId,
    device::{
        function::{self, DeviceFunction},
        operation::relu_kan::ReluKanOps,
        tensor::Shape,
    },
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
        },
    },
};

/// ReLU-KAN basis evaluation IR operation (arXiv 2406.02075).
///
/// For each input feature, computes `num_basis = grid_size + support_width` basis values
/// `R_i(x) = [ReLU(e_i - x) * ReLU(x - s_i)]^2 * 16 / (e_i - s_i)^4`
///
/// - Input: (in_features, 1) batched
/// - Grid:  (2 * num_basis, 1) constant, containing s values followed by e values
/// - Output: (in_features * num_basis, 1) batched
#[derive(Clone, Debug)]
pub struct ReluKanBasis {
    pub input: AnnotatedNode,
    pub grid: AnnotatedNode,
    pub grid_size: usize,
    pub support_width: usize,
}

impl<B: BackendMarker> GraphIROperationBase<B> for ReluKanBasis {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.grid]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.grid, true)?;
        util::check_not_batched(ir, &self.grid)?;

        let in_shape = self.input.shape;
        if in_shape.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(in_shape)));
        }

        let num_basis = self.grid_size + self.support_width;
        let expected_grid = 2 * num_basis;
        if self.grid.shape.rows() != expected_grid || self.grid.shape.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![
                self.grid.shape,
                Shape::new(expected_grid, 1),
            ])));
        }

        let in_features = in_shape.rows();
        Ok(Shape::new(in_features * num_basis, 1))
    }

    fn shorthand(&self) -> String {
        format!("ReluKanBasis(grid={}, k={})", self.grid_size, self.support_width)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for ReluKanBasis
where
    B::Backend: ReluKanOps,
{
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let grid = graph.get_ref(self.grid.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize {
            input: input.clone(),
            output: output.clone(),
        });

        func.push(function::ReluKanBasisFwd {
            input,
            grid,
            output,
            in_features: self.input.shape.rows(),
            grid_size: self.grid_size,
            support_width: self.support_width,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(input_grad) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
            let grid = graph.get_ref(self.grid.idx, GraphNodeIdTy::Values);
            let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize {
                input: input.clone(),
                output: input_grad.clone(),
            });

            func.push(function::ReluKanBasisBwd {
                input,
                grid,
                output_grad,
                input_grad,
                in_features: self.input.shape.rows(),
                grid_size: self.grid_size,
                support_width: self.support_width,
            });
        }

        func
    }
}
