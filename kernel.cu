// kernel.cu — sample CUDA kernels with intentional memory-hierarchy issues,
// used to exercise the gpu-memory-profiler subagent.

#include <cuda_runtime.h>

#define N 4096
#define TILE 32

// Issue: naive matmul, no shared-memory tiling — every element of A and B
// is re-read from global memory once per output element instead of being
// cached and reused across the block.
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            // A[row][k] is coalesced across threadIdx.x? No — row is fixed
            // per warp along x here, so this is a broadcast, but B below
            // is the real problem:
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Issue: uncoalesced global write. Consecutive threads (threadIdx.x) write
// to addresses that stride by `n`, not by 1 — each warp touches n separate
// cache lines instead of one contiguous one.
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) {
        out[x * n + y] = in[y * n + x];
    }
}

// Issue: shared-memory bank conflicts. shared[][TILE] padded to TILE (not
// TILE+1), so column access below has all 32 threads in a warp hitting the
// same bank (stride of TILE = 32 words = 32 banks -> conflict).
__global__ void reduce_columns(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;

    tile[ty][tx] = (col < n) ? in[ty * n + col] : 0.0f;
    __syncthreads();

    // Column-major read: tile[k][tx] for varying k, fixed tx — each thread
    // reads a different row but same-ish column pattern; with stride TILE
    // and no padding this causes bank conflicts.
    float sum = 0.0f;
    for (int k = 0; k < TILE; ++k) {
        sum += tile[k][tx];
    }
    if (tx == 0) out[blockIdx.x * TILE + ty] = sum;
}

// Issue: read-only lookup table fetched via plain pointer dereference in a
// hot loop instead of __ldg / const __restrict__, missing the read-only
// data cache path.
__global__ void apply_lut(const float* data, const float* lut, float* out, int n, int lutSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = static_cast<int>(data[i]) % lutSize;
        out[i] = lut[idx] * data[i];
    }
}

// Issue: large per-thread local arrays likely to spill registers into
// local memory (which physically lives in global memory / L2).
__global__ void poly_eval_heavy(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float coeffs[64];
        float xi = x[i];
        float acc = 0.0f;
        for (int c = 0; c < 64; ++c) {
            coeffs[c] = xi * (c + 1) - c * 0.5f;
        }
        for (int c = 0; c < 64; ++c) {
            acc += coeffs[c] * coeffs[63 - c];
        }
        out[i] = acc;
    }
}
