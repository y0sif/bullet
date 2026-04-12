use std::num::NonZeroUsize;

use acyclib::{
    dag::NodeId,
    device::{
        function::{DeviceFunction, MatmulType, MaybeUpdateBatchSize, UnaryOp},
        operation::{DiffableFromOutput, GemmConfig},
        tensor::Shape,
    },
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            GraphIR, GraphIRError, GraphIRMethods,
            node::AnnotatedNode,
            operation::{
                GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError,
                affine::Matmul,
                binary::Concat,
                bspline::BSplineBasis,
                nary::LinearCombination,
                unary::{PairwiseMul, Unary},
                util,
            },
            passes::{GraphIRSimplePass, downcast},
        },
    },
};

use crate::{
    CudaDevice, CudaMarker,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

// ---------------------------------------------------------------------------
// Fusion pass: detect KAN layer pattern and replace with FusedKanLayer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct FuseKanLayer;

impl GraphIRSimplePass<CudaMarker> for FuseKanLayer {
    fn try_pass_on_node(&self, ir: &mut GraphIR<CudaMarker>, target: NodeId) -> Result<bool, GraphIRError> {
        let op = ir.get(target)?.op();

        if let Some(LinearCombination { items, shape: _ }) = downcast(op) {
            if let &[(a_idx, 1.0), (b_idx, 1.0)] = &items[..] {
                if try_fuse(ir, target, a_idx, b_idx)? {
                    return Ok(true);
                }
                if try_fuse(ir, target, b_idx, a_idx)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/// Try to match the spline+base KAN layer pattern and fuse it.
///
/// Expected graph rooted at `target` (LinearCombination):
/// ```text
///   LinearCombination(spline_matmul + base_matmul)
///     ├── Matmul(spline_weight, BSplineBasis(input, grid))
///     └── Matmul(base_weight, PairwiseMul(Concat(input, Sigmoid(input))))
/// ```
fn try_fuse(
    ir: &mut GraphIR<CudaMarker>,
    target: NodeId,
    spline_candidate: NodeId,
    base_candidate: NodeId,
) -> Result<bool, GraphIRError> {
    let spline_data = ir.get(spline_candidate)?;
    let base_data = ir.get(base_candidate)?;

    if spline_data.children() != 1 || base_data.children() != 1 {
        return Ok(false);
    }

    // --- Spline path: Matmul(spline_weight, BSplineBasis(input, grid)) ---
    let Some(Matmul { a: spline_weight, b: basis_node, transa: false, transb: false }) =
        downcast(spline_data.op())
    else {
        return Ok(false);
    };

    let basis_data = ir.get(basis_node.idx)?;
    if basis_data.children() != 1 {
        return Ok(false);
    }

    let Some(BSplineBasis { input, grid, grid_size, spline_order }) = downcast(basis_data.op()) else {
        return Ok(false);
    };

    // --- Base path: Matmul(base_weight, PairwiseMul(Concat(input, Sigmoid(input)))) ---
    let Some(Matmul { a: base_weight, b: silu_node, transa: false, transb: false }) =
        downcast(base_data.op())
    else {
        return Ok(false);
    };

    let silu_data = ir.get(silu_node.idx)?;
    if silu_data.children() != 1 {
        return Ok(false);
    }

    let Some(PairwiseMul { input: concat_node }) = downcast(silu_data.op()) else {
        return Ok(false);
    };

    let concat_data = ir.get(concat_node.idx)?;
    if concat_data.children() != 1 {
        return Ok(false);
    }

    let Some(Concat { a: concat_a, b: sig_node }) = downcast(concat_data.op()) else {
        return Ok(false);
    };

    if concat_a.idx != input.idx {
        return Ok(false);
    }

    let sig_data = ir.get(sig_node.idx)?;
    if sig_data.children() != 1 {
        return Ok(false);
    }

    let Some(Unary { input: sig_input, op: UnaryOp::DiffableFromOutput(DiffableFromOutput::Sigmoid) }) =
        downcast(sig_data.op())
    else {
        return Ok(false);
    };

    if sig_input.idx != input.idx {
        return Ok(false);
    }

    // Validate shapes
    let in_features = input.shape.rows();
    let num_basis = grid_size + spline_order;
    let out_features = spline_weight.shape.rows();

    if spline_weight.shape.cols() != in_features * num_basis {
        return Ok(false);
    }
    if base_weight.shape.rows() != out_features || base_weight.shape.cols() != in_features {
        return Ok(false);
    }

    ir.replace(
        target,
        FusedKanLayer { input, grid, spline_weight, base_weight, grid_size, spline_order },
    )?;

    Ok(true)
}

// ---------------------------------------------------------------------------
// FusedKanLayer IR operation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FusedKanLayer {
    pub input: AnnotatedNode,
    pub grid: AnnotatedNode,
    pub spline_weight: AnnotatedNode,
    pub base_weight: AnnotatedNode,
    pub grid_size: usize,
    pub spline_order: usize,
}

impl GraphIROperationBase<CudaMarker> for FusedKanLayer {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.grid, self.spline_weight, self.base_weight]
    }

    fn output_shape(&self, ir: &GraphIR<CudaMarker>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.grid, true)?;
        util::check_not_batched(ir, &self.grid)?;
        util::check_dense_eq(ir, &self.spline_weight, true)?;
        util::check_dense_eq(ir, &self.base_weight, true)?;
        util::check_not_batched(ir, &self.spline_weight)?;
        util::check_not_batched(ir, &self.base_weight)?;
        util::check_has_grad(ir, &[&self.spline_weight, &self.base_weight])?;

        let in_features = self.input.shape.rows();
        let num_basis = self.grid_size + self.spline_order;
        let out_features = self.spline_weight.shape.rows();

        if self.input.shape.cols() != 1 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.input.shape)));
        }
        if self.spline_weight.shape.cols() != in_features * num_basis {
            return Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![
                self.spline_weight.shape,
                Shape::new(out_features, in_features * num_basis),
            ])));
        }
        if self.base_weight.shape != Shape::new(out_features, in_features) {
            return Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![
                self.base_weight.shape,
                Shape::new(out_features, in_features),
            ])));
        }

        let expected_knots = self.grid_size + 2 * self.spline_order + 1;
        if self.grid.shape != Shape::new(expected_knots, 1) {
            return Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![
                self.grid.shape,
                Shape::new(expected_knots, 1),
            ])));
        }

        Ok(Shape::new(out_features, 1))
    }

    fn ancillary_buffers(
        &self,
        _ir: &GraphIR<CudaMarker>,
    ) -> Result<Vec<(Shape, Option<NonZeroUsize>, bool)>, GraphIRError> {
        let in_features = self.input.shape.rows();
        let num_basis = self.grid_size + self.spline_order;

        Ok(vec![
            // Ancillary 0: basis values (D_in * num_basis, 1) batched
            // Forward: stores basis. Backward: reused as basis_grad temp.
            (Shape::new(in_features * num_basis, 1), None, true),
            // Ancillary 1: SiLU values (D_in, 1) batched
            // Forward: stores silu. Backward: reused as silu_grad temp.
            (Shape::new(in_features, 1), None, true),
        ])
    }

    fn shorthand(&self) -> String {
        format!("FusedKanLayer(grid={}, order={})", self.grid_size, self.spline_order)
    }
}

impl GraphIROperationCompilable<CudaMarker> for FusedKanLayer {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();

        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let grid = graph.get_ref(self.grid.idx, GraphNodeIdTy::Values);
        let spline_weight = graph.get_ref(self.spline_weight.idx, GraphNodeIdTy::Values);
        let base_weight = graph.get_ref(self.base_weight.idx, GraphNodeIdTy::Values);
        let anc_basis = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(0));
        let anc_silu = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(1));
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let in_features = self.input.shape.rows();
        let out_features = self.spline_weight.shape.rows();
        let num_basis = self.grid_size + self.spline_order;
        let num_knots = self.grid_size + 2 * self.spline_order + 1;

        // Update batch sizes
        func.push(MaybeUpdateBatchSize { input: input.clone(), output: anc_basis.clone() });
        func.push(MaybeUpdateBatchSize { input: input.clone(), output: anc_silu.clone() });
        func.push(MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });

        // --- Step 1: Fused BSplineBasis + SiLU kernel ---
        // Replaces 4 separate ops: BSplineBasis + Sigmoid + Concat + PairwiseMul
        {
            let inputs = vec![
                KernelInput::Size(Expr::Var), // batch_size
                KernelInput::Slice { slice: input.clone(), layout: None, mutable: false, batched: true, shape: self.input.shape },
                KernelInput::Slice { slice: grid, layout: None, mutable: false, batched: false, shape: self.grid.shape },
                KernelInput::Slice { slice: anc_basis.clone(), layout: None, mutable: true, batched: true, shape: Shape::new(in_features * num_basis, 1) },
                KernelInput::Slice { slice: anc_silu.clone(), layout: None, mutable: true, batched: true, shape: Shape::new(in_features, 1) },
            ];

            let threads = 256i32;
            let total = Expr::Var * Expr::Const(in_features as i32);
            let blocks = (total + Expr::Const(threads - 1)) / Expr::Const(threads);
            let grid_dim = [blocks, Expr::Const(1), Expr::Const(1)];
            let block_dim = [Expr::Const(threads), Expr::Const(1), Expr::Const(1)];

            let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes: Expr::Const(0) };

            let code = include_str!("kan/fwd.cu")
                .lines()
                .skip(7)
                .map(|x| format!("{x}\n"))
                .collect::<String>()
                .replace("DECL_D_IN", &in_features.to_string())
                .replace("DECL_NUM_BASIS", &num_basis.to_string())
                .replace("DECL_GRID_SIZE", &self.grid_size.to_string())
                .replace("DECL_SPLINE_ORDER", &self.spline_order.to_string())
                .replace("DECL_NUM_KNOTS", &num_knots.to_string());

            let kernel = unsafe { Kernel::new("FusedBasisSiLU".to_string(), code, args) };
            func.push(kernel.unwrap());
        }

        // --- Step 2: output = spline_weight @ basis (cuBLAS GEMM) ---
        // spline_weight: (D_out, D_in*num_basis) not batched
        // basis: (D_in*num_basis, 1) batched → NobBat
        // Result: (D_out, 1) batched, beta=0 (overwrite)
        func.push(acyclib::device::function::Matmul {
            cfg: GemmConfig::new(
                1.0, 0.0,
                self.spline_weight.shape, false,
                Shape::new(in_features * num_basis, 1), false,
            ),
            input_a: spline_weight,
            input_b: anc_basis,
            output: output.clone(),
            ty: MatmulType::NobBat,
        });

        // --- Step 3: output += base_weight @ silu (cuBLAS GEMM, accumulate) ---
        // base_weight: (D_out, D_in) not batched
        // silu: (D_in, 1) batched → NobBat
        // Result: (D_out, 1) batched, beta=1 (accumulate into existing output)
        func.push(acyclib::device::function::Matmul {
            cfg: GemmConfig::new(
                1.0, 1.0,
                self.base_weight.shape, false,
                Shape::new(in_features, 1), false,
            ),
            input_a: base_weight,
            input_b: anc_silu,
            output,
            ty: MatmulType::NobBat,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();

        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let grid = graph.get_ref(self.grid.idx, GraphNodeIdTy::Values);
        let spline_weight = graph.get_ref(self.spline_weight.idx, GraphNodeIdTy::Values);
        let base_weight = graph.get_ref(self.base_weight.idx, GraphNodeIdTy::Values);
        let anc_basis = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(0));
        let anc_silu = graph.get_ref(output_node, GraphNodeIdTy::Ancillary(1));
        let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let in_features = self.input.shape.rows();
        let out_features = self.spline_weight.shape.rows();
        let num_basis = self.grid_size + self.spline_order;
        let num_knots = self.grid_size + 2 * self.spline_order + 1;

        // --- Weight gradients via GEMM (must run BEFORE input grad GEMMs reuse ancillary) ---

        // sw_grad += output_grad @ basis^T (BatBatRed)
        if let Some(sw_grad) = graph.maybe_get_ref(self.spline_weight.idx, GraphNodeIdTy::Gradients) {
            func.push(MaybeUpdateBatchSize { input: spline_weight.clone(), output: sw_grad.clone() });
            func.push(acyclib::device::function::Matmul {
                cfg: GemmConfig::new(
                    1.0, 1.0,
                    Shape::new(out_features, 1), false,
                    Shape::new(in_features * num_basis, 1), true,
                ),
                input_a: output_grad.clone(),
                input_b: anc_basis.clone(),
                output: sw_grad,
                ty: MatmulType::BatBatRed,
            });
        }

        // bw_grad += output_grad @ silu^T (BatBatRed)
        if let Some(bw_grad) = graph.maybe_get_ref(self.base_weight.idx, GraphNodeIdTy::Gradients) {
            func.push(MaybeUpdateBatchSize { input: base_weight.clone(), output: bw_grad.clone() });
            func.push(acyclib::device::function::Matmul {
                cfg: GemmConfig::new(
                    1.0, 1.0,
                    Shape::new(out_features, 1), false,
                    Shape::new(in_features, 1), true,
                ),
                input_a: output_grad.clone(),
                input_b: anc_silu.clone(),
                output: bw_grad,
                ty: MatmulType::BatBatRed,
            });
        }

        // --- Input gradient ---
        if let Some(input_grad) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            func.push(MaybeUpdateBatchSize { input: input.clone(), output: input_grad.clone() });

            // Overwrite ancillary 0 with basis_grad = spline_weight^T @ output_grad
            // (safe: weight grad GEMMs above already read ancillary 0)
            func.push(acyclib::device::function::Matmul {
                cfg: GemmConfig::new(
                    1.0, 0.0,
                    self.spline_weight.shape, true,  // (D_in*num_basis, D_out)
                    Shape::new(out_features, 1), false,
                ),
                input_a: spline_weight,
                input_b: output_grad.clone(),
                output: anc_basis.clone(),  // overwrite with basis_grad
                ty: MatmulType::NobBat,
            });

            // Overwrite ancillary 1 with silu_grad = base_weight^T @ output_grad
            func.push(acyclib::device::function::Matmul {
                cfg: GemmConfig::new(
                    1.0, 0.0,
                    self.base_weight.shape, true,  // (D_in, D_out)
                    Shape::new(out_features, 1), false,
                ),
                input_a: base_weight,
                input_b: output_grad,
                output: anc_silu.clone(),  // overwrite with silu_grad
                ty: MatmulType::NobBat,
            });

            // Fused bspline derivative + SiLU derivative kernel
            let inputs = vec![
                KernelInput::Size(Expr::Var),
                KernelInput::Slice { slice: input, layout: None, mutable: false, batched: true, shape: self.input.shape },
                KernelInput::Slice { slice: grid, layout: None, mutable: false, batched: false, shape: self.grid.shape },
                KernelInput::Slice { slice: anc_basis, layout: None, mutable: false, batched: true, shape: Shape::new(in_features * num_basis, 1) },
                KernelInput::Slice { slice: anc_silu, layout: None, mutable: false, batched: true, shape: Shape::new(in_features, 1) },
                KernelInput::Slice { slice: input_grad, layout: None, mutable: true, batched: true, shape: self.input.shape },
            ];

            let threads = 256i32;
            let total = Expr::Var * Expr::Const(in_features as i32);
            let blocks = (total + Expr::Const(threads - 1)) / Expr::Const(threads);
            let grid_dim = [blocks, Expr::Const(1), Expr::Const(1)];
            let block_dim = [Expr::Const(threads), Expr::Const(1), Expr::Const(1)];

            let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes: Expr::Const(0) };

            let code = include_str!("kan/bwd_input.cu")
                .lines()
                .skip(7)
                .map(|x| format!("{x}\n"))
                .collect::<String>()
                .replace("DECL_D_IN", &in_features.to_string())
                .replace("DECL_NUM_BASIS", &num_basis.to_string())
                .replace("DECL_GRID_SIZE", &self.grid_size.to_string())
                .replace("DECL_SPLINE_ORDER", &self.spline_order.to_string())
                .replace("DECL_NUM_KNOTS", &num_knots.to_string());

            let kernel = unsafe { Kernel::new("FusedBwdInputGrad".to_string(), code, args) };
            func.push(kernel.unwrap());
        }

        func
    }
}
