#pragma once

#include "minicompiler/cuda/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace minicompiler::cuda {

// How many ways to divide K for an m x n x k matmul on a GPU with `sm_count`
// SMs, from the number of 128x128 output tiles and how many blocks of the
// kernel an SM holds at once: none when the tiles cover at least 80% of the
// SMs; one block per SM when they cover at most half, or always when an SM
// holds only one; up to two blocks per SM in between; and at least 64 of K per
// split. SplitK's kernel holds two (its launch bounds).
int split_k_splits(int m, int n, int k, int sm_count, int blocks_per_sm = 2);

// Whether `kernel` divides K into `splits` when launched with more than one:
// SplitK and the TensorCore* kernels.
bool uses_split_k(MatmulKernel kernel);

// The splits to launch `kernel` with: split_k_splits with the kernel's
// blocks per SM for a kernel that splits K, otherwise 1.
int matmul_splits(MatmulKernel kernel, int m, int n, int k, int sm_count);

// Device memory launch_matmul needs as `workspace` for `kernel` with `splits`:
// one m x n slice of partial products per split for a kernel that splits K,
// otherwise none.
std::size_t matmul_workspace_bytes(MatmulKernel kernel, int m, int n, int splits);

// C[m,n] = A[m,k] * B[k,n], row-major float32, enqueued on `stream`. Auto
// must first be resolved (resolve_matmul_kernel) to the kernel it runs;
// passed as is, it fails with cudaErrorInvalidValue. A kernel that splits K
// reads `splits` and, when it is above 1, `workspace`.
cudaError_t launch_matmul(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream, int splits = 1, float* workspace = nullptr);

// Whether a TensorCore* kernel runs on the current GPU: the GPU has its format
// (TF32 and BF16: compute capability 8.0; FP16: 7.0) and the build has code
// for it compiled for such an architecture. Launching one that does not run
// fails with cudaErrorNotSupported.
bool tensor_core_supported(MatmulKernel kernel);

// The TensorCore* kernels (matmul_tensor_core.cu); launch_matmul forwards to it.
cudaError_t launch_tensor_core(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                               cudaStream_t stream, int splits = 1, float* workspace = nullptr);

// How many blocks of a TensorCore* kernel an SM of the current GPU holds at
// once (the occupancy calculator; registers decide it), or 1 if unknown.
int tensor_core_blocks_per_sm(MatmulKernel kernel);

// C = the sum of `splits` partial products of `count` elements each, stored
// one after another in `partial`, added in the same order for every element,
// so the result does not depend on how the blocks were scheduled. `vec`:
// float4 access, for a count that is a multiple of 4 and 16-byte aligned C and
// partial.
cudaError_t sum_split_partials(const float* partial, float* c, std::size_t count, int splits, bool vec,
                               cudaStream_t stream);

}
