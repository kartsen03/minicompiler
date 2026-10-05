#pragma once

#include "minicompiler/cuda/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace minicompiler::cuda {

// How many ways SplitK divides K for an m x n x k matmul on a GPU with
// `sm_count` SMs, from the number of 128x128 output tiles: none when the tiles
// cover at least 80% of the SMs; one block per SM when they cover at most half;
// up to two blocks per SM in between; and at least 64 of K per split.
int split_k_splits(int m, int n, int k, int sm_count);

// Device memory launch_matmul needs as `workspace` for `kernel` with `splits`:
// one m x n slice of partial products per split for SplitK, otherwise none.
std::size_t matmul_workspace_bytes(MatmulKernel kernel, int m, int n, int splits);

// C[m,n] = A[m,k] * B[k,n], row-major float32, enqueued on `stream`. Auto
// must first be resolved (resolve_matmul_kernel) to the kernel it runs;
// passed as is, it fails with cudaErrorInvalidValue. SplitK reads `splits`
// and, when it is above 1, `workspace`.
cudaError_t launch_matmul(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream, int splits = 1, float* workspace = nullptr);

// Whether a TensorCore* kernel runs on the current GPU: the GPU has its format
// (TF32 and BF16: compute capability 8.0; FP16: 7.0) and the build has code
// for it compiled for such an architecture. Launching one that does not run
// fails with cudaErrorNotSupported.
bool tensor_core_supported(MatmulKernel kernel);

// The TensorCore* kernels (matmul_tensor_core.cu); launch_matmul forwards to it.
cudaError_t launch_tensor_core(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                               cudaStream_t stream);

}
