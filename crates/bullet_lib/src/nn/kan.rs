use acyclib::graph::{
    builder::{GraphBuilder, GraphBuilderNode, InitSettings},
    ir::{
        BackendMarker,
        operation::{GraphIROperationCompilable, bspline::BSplineBasis, sparse::SparseAffineActivate},
    },
};

use super::Shape;

/// A KAN (Kolmogorov-Arnold Network) layer.
///
/// Forward: `spline_weight @ bspline_basis(input, grid) + base_weight @ silu(input)`
///
/// This combines a learned B-spline activation (the "spline" path) with a
/// learned linear+SiLU activation (the "base" path), following the efficient-KAN
/// architecture that was validated in kanue (-22% loss over CReLU baseline).
pub struct KanLayer<'a, B: BackendMarker> {
    pub spline_weight: GraphBuilderNode<'a, B>,
    pub base_weight: GraphBuilderNode<'a, B>,
    pub grid: GraphBuilderNode<'a, B>,
    pub grid_size: usize,
    pub spline_order: usize,
}

impl<'a, B: BackendMarker> KanLayer<'a, B>
where
    BSplineBasis: GraphIROperationCompilable<B>,
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    pub fn forward(self, input: GraphBuilderNode<'a, B>) -> GraphBuilderNode<'a, B> {

        // Spline path: spline_weight @ bspline_basis(input, grid)
        let basis = input.bspline_basis(self.grid, self.grid_size, self.spline_order);
        let spline_out = self.spline_weight.matmul(basis);

        // Base path: base_weight @ silu(input)
        // SiLU(x) = x * sigmoid(x), composed from existing ops
        let silu_input = input * input.sigmoid();
        let base_out = self.base_weight.matmul(silu_input);

        // Combined output
        spline_out + base_out
    }
}

/// Create a new KAN layer with the given dimensions.
///
/// - `id`: weight name prefix (e.g. "kan1")
/// - `in_features`: input dimension
/// - `out_features`: output dimension
/// - `grid_size`: number of B-spline grid intervals (default: 5)
/// - `spline_order`: B-spline degree (default: 3 for cubic)
///
/// The grid spans [-1, 1] with uniform knot spacing, extended by
/// `spline_order` extra knots on each side.
pub fn kan_layer<'a, B: BackendMarker>(
    builder: &'a GraphBuilder<B>,
    id: &str,
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    spline_order: usize,
) -> KanLayer<'a, B>
where
    BSplineBasis: GraphIROperationCompilable<B>,
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    let num_basis = grid_size + spline_order;
    let num_knots = grid_size + 2 * spline_order + 1;

    // Spline weight: (out_features, in_features * num_basis)
    // Initialize small — spline path should start near zero and learn activations
    let spline_weight = builder.new_weights(
        &format!("{id}_sw"),
        Shape::new(out_features, in_features * num_basis),
        InitSettings::Normal {
            mean: 0.0,
            stdev: 0.1 / ((in_features * num_basis) as f32).sqrt(),
        },
    );

    // Base weight: (out_features, in_features)
    // Kaiming init for the linear+SiLU path
    let base_weight = builder.new_weights(
        &format!("{id}_bw"),
        Shape::new(out_features, in_features),
        InitSettings::Normal {
            mean: 0.0,
            stdev: (2.0 / in_features as f32).sqrt(),
        },
    );

    // Grid: uniform knots spanning [-1, 1] with extensions
    let h = 2.0 / grid_size as f32;
    let mut grid_vals = Vec::with_capacity(num_knots);

    // Left extension
    for i in (1..=spline_order).rev() {
        grid_vals.push(-1.0 - i as f32 * h);
    }
    // Interior knots
    for i in 0..=grid_size {
        grid_vals.push(-1.0 + i as f32 * h);
    }
    // Right extension
    for i in 1..=spline_order {
        grid_vals.push(1.0 + i as f32 * h);
    }
    assert_eq!(grid_vals.len(), num_knots);

    let grid = builder.new_constant(Shape::new(num_knots, 1), &grid_vals);

    KanLayer {
        spline_weight,
        base_weight,
        grid,
        grid_size,
        spline_order,
    }
}
