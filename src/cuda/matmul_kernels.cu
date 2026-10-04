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

// Shared-memory tiling. A 32x32 block of threads computes a 32x32 tile of C,
// one element per thread. It walks along K in steps of 32: the block loads a
// 32x32 tile of A and one of B into shared memory (one element of each per
// thread, coalesced), synchronizes, and every thread accumulates 32 products
// from shared memory. Each element loaded from global memory is reused by 32
// threads instead of being read 32 times. Tiles that run past the edge of A or
// B are padded with zeros, so any M, N, K works.
constexpr int kTile = 32;

__global__ void matmul_tiled(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m,
                             int n, int k) {
	__shared__ float as[kTile][kTile];
	__shared__ float bs[kTile][kTile];
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int row = blockIdx.y * kTile + ty;
	const int col = blockIdx.x * kTile + tx;
	float acc = 0.0f;
	for (int t=0; t<k; t+=kTile) {
		as[ty][tx] = (row < m && t + tx < k) ? a[row * k + t + tx] : 0.0f;
		bs[ty][tx] = (t + ty < k && col < n) ? b[(t + ty) * n + col] : 0.0f;
		__syncthreads();
#pragma unroll
		for (int i=0; i<kTile; ++i) acc += as[ty][i] * bs[i][tx];
		__syncthreads();
	}
	if (row < m && col < n) c[row * n + col] = acc;
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
		case MatmulKernel::Tiled: {
			const dim3 block(kTile, kTile);
			const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
			matmul_tiled<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
			return cudaGetLastError();
		}
	}
	return cudaErrorInvalidValue;
}

}
