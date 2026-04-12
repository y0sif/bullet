#ifndef STUFF
#define DECL_MAXY 32768
#define DECL_D_IN 256
#define DECL_D_OUT 128
#define DECL_NUM_BASIS 8
#define DECL_GRID_SIZE 5
#define DECL_SPLINE_ORDER 3
#define DECL_NUM_KNOTS 12
#endif

constexpr int MaximumBlocksY = DECL_MAXY;
constexpr int D_IN = DECL_D_IN;
constexpr int D_OUT = DECL_D_OUT;
constexpr int NUM_BASIS = DECL_NUM_BASIS;
constexpr int GRID_SIZE = DECL_GRID_SIZE;
constexpr int SPLINE_ORDER = DECL_SPLINE_ORDER;
constexpr int NUM_KNOTS = DECL_NUM_KNOTS;
constexpr int NUM_DEGREE0 = GRID_SIZE + 2 * SPLINE_ORDER;

// Fused forward: BSplineBasis + SiLU + spline_weight @ basis + base_weight @ silu
// One block per batch element. Shared memory holds grid, basis values, and SiLU values.
extern "C" __global__ void kernel(
    const int batch_size,
    const float* __restrict__ input,
    const float* __restrict__ grid_knots,
    const float* __restrict__ spline_weight,
    const float* __restrict__ base_weight,
    float* __restrict__ output,
    float* __restrict__ anc_basis,
    float* __restrict__ anc_silu)
{
    const int b = MaximumBlocksY * blockIdx.x + blockIdx.y;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;

    // Shared memory layout: grid[NUM_KNOTS] | basis[D_IN * NUM_BASIS] | silu[D_IN]
    extern __shared__ float smem[];
    float* s_grid = smem;
    float* s_basis = smem + NUM_KNOTS;
    float* s_silu = smem + NUM_KNOTS + D_IN * NUM_BASIS;

    // Load grid into shared memory
    for (int i = tid; i < NUM_KNOTS; i += blockDim.x)
        s_grid[i] = grid_knots[i];
    __syncthreads();

    // Phase 1: Compute B-spline basis and SiLU for each input feature
    for (int j = tid; j < D_IN; j += blockDim.x) {
        const float x = input[b * D_IN + j];

        // Cox-de Boor recursion
        float prev[32];
        float curr[32];

        for (int i = 0; i < NUM_DEGREE0; i++)
            prev[i] = (x >= s_grid[i] && x < s_grid[i + 1]) ? 1.0f : 0.0f;

        if (x >= s_grid[NUM_KNOTS - 1]) {
            for (int i = 0; i < NUM_DEGREE0; i++) prev[i] = 0.0f;
            prev[NUM_DEGREE0 - 1] = 1.0f;
        }

        for (int p = 1; p <= SPLINE_ORDER; p++) {
            const int num_at_p = NUM_DEGREE0 - p;
            for (int i = 0; i < num_at_p; i++) {
                float val = 0.0f;
                float dl = s_grid[i + p] - s_grid[i];
                if (dl != 0.0f)
                    val += (x - s_grid[i]) / dl * prev[i];
                float dr = s_grid[i + p + 1] - s_grid[i + 1];
                if (dr != 0.0f)
                    val += (s_grid[i + p + 1] - x) / dr * prev[i + 1];
                curr[i] = val;
            }
            for (int i = 0; i < num_at_p; i++) prev[i] = curr[i];
        }

        // Write basis to shared memory and ancillary buffer
        const int basis_off = j * NUM_BASIS;
        const int anc_off = b * D_IN * NUM_BASIS + basis_off;
        for (int i = 0; i < NUM_BASIS; i++) {
            s_basis[basis_off + i] = prev[i];
            anc_basis[anc_off + i] = prev[i];
        }

        // SiLU = x * sigmoid(x)
        float sig = 1.0f / (1.0f + expf(-x));
        float silu = x * sig;
        s_silu[j] = silu;
        anc_silu[b * D_IN + j] = silu;
    }
    __syncthreads();

    // Phase 2: Compute output = spline_weight @ basis + base_weight @ silu
    for (int i = tid; i < D_OUT; i += blockDim.x) {
        float result = 0.0f;

        // Spline path: dot(spline_weight[i, :], basis[:])
        const float* sw_row = spline_weight + i * (D_IN * NUM_BASIS);
        for (int k = 0; k < D_IN * NUM_BASIS; k++)
            result += sw_row[k] * s_basis[k];

        // Base path: dot(base_weight[i, :], silu[:])
        const float* bw_row = base_weight + i * D_IN;
        for (int k = 0; k < D_IN; k++)
            result += bw_row[k] * s_silu[k];

        output[b * D_OUT + i] = result;
    }
}
