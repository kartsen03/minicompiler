#pragma once

#include "minicompiler/cuda/cuda_backend.hpp"

#include <cuda_runtime.h>

namespace minicompiler::cuda {

// C[m,n] = A[m,k] * B[k,n], row-major float32, enqueued on `stream`.
cudaError_t launch_matmul(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream);

}
