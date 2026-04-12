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

// Fused backward for input gradients.
// Combines bspline derivative chain rule + SiLU derivative.
// basis_grad and silu_grad are pre-computed via GEMM before this kernel.
// Thread layout: one thread per (batch, input_feature).
extern "C" __global__ void kernel(
    const int batch_size,
    const float* __restrict__ input,
    const float* __restrict__ grid_knots,
    const float* __restrict__ basis_grad,
    const float* __restrict__ silu_grad,
    float* __restrict__ input_grad)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * D_IN;
    if (tid >= total) return;

    const int b = tid / D_IN;
    const int j = tid % D_IN;
    const float x = input[b * D_IN + j];
    float grad_acc = 0.0f;

    // === Spline path: basis derivative chain rule ===
    if (SPLINE_ORDER > 0) {
        const int k = SPLINE_ORDER;
        const int num_prev = GRID_SIZE + k + 1;

        // Compute degree-(k-1) basis via Cox-de Boor
        float prev[32];
        float curr[32];

        for (int i = 0; i < NUM_DEGREE0; i++)
            prev[i] = (x >= grid_knots[i] && x < grid_knots[i + 1]) ? 1.0f : 0.0f;
        if (x >= grid_knots[NUM_KNOTS - 1]) {
            for (int i = 0; i < NUM_DEGREE0; i++) prev[i] = 0.0f;
            prev[NUM_DEGREE0 - 1] = 1.0f;
        }

        for (int p = 1; p < k; p++) {
            const int num_at_p = NUM_DEGREE0 - p;
            for (int i = 0; i < num_at_p; i++) {
                float val = 0.0f;
                float dl = grid_knots[i + p] - grid_knots[i];
                if (dl != 0.0f) val += (x - grid_knots[i]) / dl * prev[i];
                float dr = grid_knots[i + p + 1] - grid_knots[i + 1];
                if (dr != 0.0f) val += (grid_knots[i + p + 1] - x) / dr * prev[i + 1];
                curr[i] = val;
            }
            for (int i = 0; i < num_at_p; i++) prev[i] = curr[i];
        }

        // Derivative formula + chain rule with pre-computed basis_grad
        const int bg_off = b * D_IN * NUM_BASIS + j * NUM_BASIS;
        for (int bi = 0; bi < NUM_BASIS; bi++) {
            float deriv = 0.0f;
            float dl = grid_knots[bi + k] - grid_knots[bi];
            if (dl != 0.0f && bi < num_prev)
                deriv += prev[bi] / dl;
            float dr = grid_knots[bi + k + 1] - grid_knots[bi + 1];
            if (dr != 0.0f && (bi + 1) < num_prev)
                deriv -= prev[bi + 1] / dr;
            deriv *= (float)k;

            grad_acc += basis_grad[bg_off + bi] * deriv;
        }
    }

    // === Base path: SiLU derivative ===
    {
        float sig = 1.0f / (1.0f + expf(-x));
        float silu_deriv = sig * (1.0f + x * (1.0f - sig));
        grad_acc += silu_grad[b * D_IN + j] * silu_deriv;
    }

    atomicAdd(&input_grad[b * D_IN + j], grad_acc);
}
