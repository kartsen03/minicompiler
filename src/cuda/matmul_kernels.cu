#include "cuda/matmul_kernels.hpp"

namespace minicompiler::cuda {
namespace {

// One thread per element of C. Neighbouring threads (threadIdx.x) take
// neighbouring columns, so their reads of B and writes of C are coalesced,
// and they all read the same element of A (a broadcast). Every operand comes
// from global memory: each element of A and B is read again by every thread
// that needs it.
__global__ void matmul_naive(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m,
                             int n, int k) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= m || col >= n) return;
	float acc = 0.0f;
	for (int t=0; t<k; ++t) acc += a[row * k + t] * b[t * n + col];
	c[row * n + col] = acc;
}

}

cudaError_t launch_matmul(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream) {
	switch (kernel) {
		case MatmulKernel::Naive: {
			const dim3 block(16, 16);
			const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
			matmul_naive<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
			return cudaGetLastError();
		}
	}
	return cudaErrorInvalidValue;
}

}
