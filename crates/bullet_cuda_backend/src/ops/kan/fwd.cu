#ifndef STUFF
#define DECL_D_IN 256
#define DECL_NUM_BASIS 8
#define DECL_GRID_SIZE 5
#define DECL_SPLINE_ORDER 3
#define DECL_NUM_KNOTS 12
#endif

constexpr int D_IN = DECL_D_IN;
constexpr int NUM_BASIS = DECL_NUM_BASIS;
constexpr int GRID_SIZE = DECL_GRID_SIZE;
constexpr int SPLINE_ORDER = DECL_SPLINE_ORDER;
constexpr int NUM_KNOTS = DECL_NUM_KNOTS;
constexpr int NUM_DEGREE0 = GRID_SIZE + 2 * SPLINE_ORDER;

// Fused BSplineBasis + SiLU computation.
// Replaces: BSplineBasis + Sigmoid + Concat + PairwiseMul (4 separate kernels).
// Matmuls are handled by cuBLAS after this kernel.
// Thread layout: one thread per (batch, input_feature).
extern "C" __global__ void kernel(
    const int batch_size,
    const float* __restrict__ input,
    const float* __restrict__ grid_knots,
    float* __restrict__ out_basis,
    float* __restrict__ out_silu)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * D_IN;
    if (tid >= total) return;

    const int b = tid / D_IN;
    const int j = tid % D_IN;

    const float x = input[b * D_IN + j];

    // Cox-de Boor recursion
    float prev[32];
    float curr[32];

    for (int i = 0; i < NUM_DEGREE0; i++)
        prev[i] = (x >= grid_knots[i] && x < grid_knots[i + 1]) ? 1.0f : 0.0f;

    if (x >= grid_knots[NUM_KNOTS - 1]) {
        for (int i = 0; i < NUM_DEGREE0; i++) prev[i] = 0.0f;
        prev[NUM_DEGREE0 - 1] = 1.0f;
    }

    for (int p = 1; p <= SPLINE_ORDER; p++) {
        const int num_at_p = NUM_DEGREE0 - p;
        for (int i = 0; i < num_at_p; i++) {
            float val = 0.0f;
            float dl = grid_knots[i + p] - grid_knots[i];
            if (dl != 0.0f)
                val += (x - grid_knots[i]) / dl * prev[i];
            float dr = grid_knots[i + p + 1] - grid_knots[i + 1];
            if (dr != 0.0f)
                val += (grid_knots[i + p + 1] - x) / dr * prev[i + 1];
            curr[i] = val;
        }
        for (int i = 0; i < num_at_p; i++) prev[i] = curr[i];
    }

    // Write basis values
    const int basis_off = b * D_IN * NUM_BASIS + j * NUM_BASIS;
    for (int i = 0; i < NUM_BASIS; i++)
        out_basis[basis_off + i] = prev[i];

    // SiLU = x * sigmoid(x)
    float sig = 1.0f / (1.0f + expf(-x));
    out_silu[b * D_IN + j] = x * sig;
}
