use acyclib::graph::{
    builder::{GraphBuilder, GraphBuilderNode, InitSettings},
    ir::{
        BackendMarker,
        operation::{GraphIROperationCompilable, relu_kan::ReluKanBasis, sparse::SparseAffineActivate},
    },
};

use super::Shape;

/// A ReLU-KAN layer (arXiv 2406.02075).
///
/// Forward: `spline_weight @ relu_kan_basis(input, grid)`
///
/// The ReLU-KAN paper uses a pure-basis layer (no base path). If a base path
/// is desired for training stability, the caller can wire one manually or
/// reuse [`crate::nn::kan::KanLayer`].
pub struct ReluKanLayer<'a, B: BackendMarker> {
    pub spline_weight: GraphBuilderNode<'a, B>,
    pub grid: GraphBuilderNode<'a, B>,
    pub grid_size: usize,
    pub support_width: usize,
}

impl<'a, B: BackendMarker> ReluKanLayer<'a, B>
where
    ReluKanBasis: GraphIROperationCompilable<B>,
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    pub fn forward(self, input: GraphBuilderNode<'a, B>) -> GraphBuilderNode<'a, B> {
        let basis = input.relu_kan_basis(self.grid, self.grid_size, self.support_width);
        self.spline_weight.matmul(basis)
    }
}

/// Create a new ReLU-KAN layer.
///
/// - `id`: weight name prefix (e.g. "kan1")
/// - `in_features`: input dimension
/// - `out_features`: output dimension
/// - `grid_size`: number of intervals (G in the paper)
/// - `support_width`: support width in grid units (k in the paper, fixed exponent is 2)
/// - `grid_range`: (min, max) — input domain. Paper uses (0.0, 1.0); must match the
///   expected input distribution (e.g. (0.0, 1.0) for CReLU output).
pub fn relu_kan_layer<'a, B: BackendMarker>(
    builder: &'a GraphBuilder<B>,
    id: &str,
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    support_width: usize,
    grid_range: (f32, f32),
) -> ReluKanLayer<'a, B>
where
    ReluKanBasis: GraphIROperationCompilable<B>,
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    let num_basis = grid_size + support_width;

    let spline_weight = builder.new_weights(
        &format!("{id}_sw"),
        Shape::new(out_features, in_features * num_basis),
        InitSettings::Normal {
            mean: 0.0,
            stdev: 0.1 / ((in_features * num_basis) as f32).sqrt(),
        },
    );

    // s_i = grid_min + (i - k) / G * span
    // e_i = grid_min + (i + 1) / G * span
    let (grid_min, grid_max) = grid_range;
    let span = grid_max - grid_min;
    let g = grid_size as f32;
    let k = support_width as f32;

    let mut grid_vals = Vec::with_capacity(2 * num_basis);
    for i in 0..num_basis {
        grid_vals.push(grid_min + (i as f32 - k) / g * span);
    }
    for i in 0..num_basis {
        grid_vals.push(grid_min + (i as f32 - k + k + 1.0) / g * span);
    }

    let grid = builder.new_constant(Shape::new(2 * num_basis, 1), &grid_vals);

    ReluKanLayer {
        spline_weight,
        grid,
        grid_size,
        support_width,
    }
}
