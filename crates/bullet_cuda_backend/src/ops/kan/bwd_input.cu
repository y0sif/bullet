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

// Fused backward for input gradients.
// Combines bspline derivative (spline path) + SiLU derivative (base path).
// One block per batch element. Threads handle input features.
// Weight gradients are computed separately via GEMM.
extern "C" __global__ void kernel(
    const int batch_size,
    const float* __restrict__ input,
    const float* __restrict__ grid_knots,
    const float* __restrict__ spline_weight,
    const float* __restrict__ base_weight,
    const float* __restrict__ output_grad,
    float* __restrict__ input_grad)
{
    const int b = MaximumBlocksY * blockIdx.x + blockIdx.y;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;

    // Shared memory: grid[NUM_KNOTS] | output_grad[D_OUT]
    extern __shared__ float smem[];
    float* s_grid = smem;
    float* s_og = smem + NUM_KNOTS;

    // Load grid and output_grad into shared memory
    for (int i = tid; i < NUM_KNOTS; i += blockDim.x)
        s_grid[i] = grid_knots[i];
    for (int i = tid; i < D_OUT; i += blockDim.x)
        s_og[i] = output_grad[b * D_OUT + i];
    __syncthreads();

    for (int j = tid; j < D_IN; j += blockDim.x) {
        const float x = input[b * D_IN + j];
        float grad_acc = 0.0f;

        // === Spline path: basis derivative * (spline_weight^T @ output_grad) ===
        if (SPLINE_ORDER > 0) {
            const int k = SPLINE_ORDER;
            const int num_prev = GRID_SIZE + k + 1;

            // Compute degree-(k-1) basis via Cox-de Boor
            float prev[32];
            float curr[32];

            for (int i = 0; i < NUM_DEGREE0; i++)
                prev[i] = (x >= s_grid[i] && x < s_grid[i + 1]) ? 1.0f : 0.0f;
            if (x >= s_grid[NUM_KNOTS - 1]) {
                for (int i = 0; i < NUM_DEGREE0; i++) prev[i] = 0.0f;
                prev[NUM_DEGREE0 - 1] = 1.0f;
            }

            for (int p = 1; p < k; p++) {
                const int num_at_p = NUM_DEGREE0 - p;
                for (int i = 0; i < num_at_p; i++) {
                    float val = 0.0f;
                    float dl = s_grid[i + p] - s_grid[i];
                    if (dl != 0.0f) val += (x - s_grid[i]) / dl * prev[i];
                    float dr = s_grid[i + p + 1] - s_grid[i + 1];
                    if (dr != 0.0f) val += (s_grid[i + p + 1] - x) / dr * prev[i + 1];
                    curr[i] = val;
                }
                for (int i = 0; i < num_at_p; i++) prev[i] = curr[i];
            }

            // For each basis function: compute derivative and chain through weights
            for (int bi = 0; bi < NUM_BASIS; bi++) {
                // Basis derivative: dB_{bi,k}/dx
                float deriv = 0.0f;
                float dl = s_grid[bi + k] - s_grid[bi];
                if (dl != 0.0f && bi < num_prev)
                    deriv += prev[bi] / dl;
                float dr = s_grid[bi + k + 1] - s_grid[bi + 1];
                if (dr != 0.0f && (bi + 1) < num_prev)
                    deriv -= prev[bi + 1] / dr;
                deriv *= (float)k;

                // Chain rule: sum_i spline_weight[i, j*NUM_BASIS+bi] * output_grad[b, i]
                float basis_grad = 0.0f;
                const int sw_col = j * NUM_BASIS + bi;
                for (int oi = 0; oi < D_OUT; oi++)
                    basis_grad += spline_weight[oi * (D_IN * NUM_BASIS) + sw_col] * s_og[oi];

                grad_acc += basis_grad * deriv;
            }
        }

        // === Base path: SiLU derivative * (base_weight^T @ output_grad) ===
        {
            float sig = 1.0f / (1.0f + expf(-x));
            // d/dx[x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            float silu_deriv = sig * (1.0f + x * (1.0f - sig));

            float base_grad = 0.0f;
            for (int oi = 0; oi < D_OUT; oi++)
                base_grad += base_weight[oi * D_IN + j] * s_og[oi];

            grad_acc += base_grad * silu_deriv;
        }

        atomicAdd(&input_grad[b * D_IN + j], grad_acc);
    }
}
