use crate::{
    dag::NodeId,
    device::{
        function::{self, DeviceFunction},
        operation::bspline::BSplineOps,
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

/// B-spline basis evaluation IR operation.
///
/// Takes an input tensor and a grid (knot vector) constant, evaluates all
/// B-spline basis functions for each input feature via Cox-de Boor recursion.
///
/// - Input:  (in_features, 1) batched
/// - Grid:   (num_knots, 1) constant, where num_knots = grid_size + 2 * spline_order + 1
/// - Output: (in_features * num_basis, 1) batched, where num_basis = grid_size + spline_order
#[derive(Clone, Debug)]
pub struct BSplineBasis {
    pub input: AnnotatedNode,
    pub grid: AnnotatedNode,
    pub grid_size: usize,
    pub spline_order: usize,
}

impl<B: BackendMarker> GraphIROperationBase<B> for BSplineBasis {
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

        let expected_knots = self.grid_size + 2 * self.spline_order + 1;
        if self.grid.shape.rows() != expected_knots || self.grid.shape.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![
                self.grid.shape,
                Shape::new(expected_knots, 1),
            ])));
        }

        let in_features = in_shape.rows();
        let num_basis = self.grid_size + self.spline_order;
        Ok(Shape::new(in_features * num_basis, 1))
    }

    fn shorthand(&self) -> String {
        format!("BSplineBasis(grid={}, order={})", self.grid_size, self.spline_order)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for BSplineBasis
where
    B::Backend: BSplineOps,
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

        func.push(function::BSplineBasisFwd {
            input,
            grid,
            output,
            in_features: self.input.shape.rows(),
            grid_size: self.grid_size,
            spline_order: self.spline_order,
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

            func.push(function::BSplineBasisBwd {
                input,
                grid,
                output_grad,
                input_grad,
                in_features: self.input.shape.rows(),
                grid_size: self.grid_size,
                spline_order: self.spline_order,
            });
        }

        func
    }
}
