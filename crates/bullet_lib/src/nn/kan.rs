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
/// Spline path: `spline_weight @ bspline_basis(input, grid)`
/// Optional base path: `base_weight @ base_activation(input)`
///
/// The base activation is chosen at `forward()` time via a closure
/// (e.g. `|x| x * x.sigmoid()` for SiLU, `|x| x.screlu()` for SCReLU,
/// `|x| x` for linear). Pure-spline layers skip the base path entirely.
pub struct KanLayer<'a, B: BackendMarker> {
    pub spline_weight: GraphBuilderNode<'a, B>,
    /// None when the layer was constructed without a base path (pure spline).
    pub base_weight: Option<GraphBuilderNode<'a, B>>,
    pub grid: GraphBuilderNode<'a, B>,
    pub grid_size: usize,
    pub spline_order: usize,
}

impl<'a, B: BackendMarker> KanLayer<'a, B>
where
    BSplineBasis: GraphIROperationCompilable<B>,
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    /// Apply the KAN layer with a base path.
    ///
    /// `base_activation` is applied to `input` before the base-path matmul.
    /// Panics if the layer was built without a base path — use `forward_pure` instead.
    pub fn forward<F>(self, input: GraphBuilderNode<'a, B>, base_activation: F) -> GraphBuilderNode<'a, B>
    where
        F: FnOnce(GraphBuilderNode<'a, B>) -> GraphBuilderNode<'a, B>,
    {
        let basis = input.bspline_basis(self.grid, self.grid_size, self.spline_order);
        let spline_out = self.spline_weight.matmul(basis);
        let base_weight = self
            .base_weight
            .expect("KanLayer built without base path; call forward_pure() instead");
        let base_out = base_weight.matmul(base_activation(input));
        spline_out + base_out
    }

    /// Apply the KAN layer without a base path (pure spline).
    ///
    /// Panics if the layer was built with a base path.
    pub fn forward_pure(self, input: GraphBuilderNode<'a, B>) -> GraphBuilderNode<'a, B> {
        assert!(
            self.base_weight.is_none(),
            "KanLayer built with base path; call forward(input, base_activation) instead"
        );
        let basis = input.bspline_basis(self.grid, self.grid_size, self.spline_order);
        self.spline_weight.matmul(basis)
    }
}

/// Create a new KAN layer.
///
/// - `id`: weight name prefix (e.g. "kan1")
/// - `in_features`: input dimension
/// - `out_features`: output dimension
/// - `grid_size`: number of B-spline grid intervals (default: 5)
/// - `spline_order`: B-spline degree (default: 3 for cubic)
/// - `grid_range`: (min, max) range for the B-spline grid knots. Must match the
///   expected input distribution (e.g. `(0.0, 1.0)` for SCReLU/CReLU output).
/// - `with_base`: if true, allocates a `(out_features, in_features)` base weight;
///   if false, the layer is pure-spline (no base path). The base activation itself
///   is chosen later at `forward()` time.
pub fn kan_layer<'a, B: BackendMarker>(
    builder: &'a GraphBuilder<B>,
    id: &str,
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    spline_order: usize,
    grid_range: (f32, f32),
    with_base: bool,
) -> KanLayer<'a, B>
where
    BSplineBasis: GraphIROperationCompilable<B>,
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    let num_basis = grid_size + spline_order;
    let num_knots = grid_size + 2 * spline_order + 1;

    let spline_weight = builder.new_weights(
        &format!("{id}_sw"),
        Shape::new(out_features, in_features * num_basis),
        InitSettings::Normal {
            mean: 0.0,
            stdev: 0.1 / ((in_features * num_basis) as f32).sqrt(),
        },
    );

    let base_weight = with_base.then(|| {
        builder.new_weights(
            &format!("{id}_bw"),
            Shape::new(out_features, in_features),
            InitSettings::Normal {
                mean: 0.0,
                stdev: (2.0 / in_features as f32).sqrt(),
            },
        )
    });

    let (grid_min, grid_max) = grid_range;
    let h = (grid_max - grid_min) / grid_size as f32;
    let mut grid_vals = Vec::with_capacity(num_knots);

    for i in (1..=spline_order).rev() {
        grid_vals.push(grid_min - i as f32 * h);
    }
    for i in 0..=grid_size {
        grid_vals.push(grid_min + i as f32 * h);
    }
    for i in 1..=spline_order {
        grid_vals.push(grid_max + i as f32 * h);
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
