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

// Shared-memory tiling plus register blocking. A block computes a BM x BN tile
// of C, and each thread a TM x TN sub-tile kept in registers. Per step of BK
// along K the block stages a BM x BK tile of A and a BK x BN tile of B in
// shared memory; each thread then loads TM values of A and TN of B into
// registers and performs TM x TN multiply-adds, so every value read from
// shared memory feeds TN (or TM) FMAs instead of 1. Thread (r, c) owns rows
// r + R*i and columns c + C*j, where R x C is the grid of threads: across a
// warp the B reads hit consecutive addresses (no bank conflicts), the A reads
// are broadcasts, and the final stores to C are coalesced.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmul_register_tiled(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n,
                      int k) {
	constexpr int kRowsOfThreads = BM / TM;
	constexpr int kColsOfThreads = BN / TN;
	constexpr int kThreads = kRowsOfThreads * kColsOfThreads;
	__shared__ float as[BM][BK];
	__shared__ float bs[BK][BN];
	const int tid = threadIdx.x;
	const int thread_row = tid / kColsOfThreads;
	const int thread_col = tid % kColsOfThreads;
	const int block_row = blockIdx.y * BM;
	const int block_col = blockIdx.x * BN;

	float acc[TM][TN] = {};
	float a_frag[TM];
	float b_frag[TN];
	for (int k0=0; k0<k; k0+=BK) {
		// Stage the tiles: consecutive threads read consecutive addresses of a
		// row of A (8 floats, one 32-byte sector) or of B (fully coalesced) and
		// write consecutive shared-memory words.
		for (int i=tid; i<BM * BK; i+=kThreads) {
			const int r = i / BK;
			const int col_k = i % BK;
			const int gr = block_row + r;
			const int gk = k0 + col_k;
			as[r][col_k] = (gr < m && gk < k) ? a[gr * k + gk] : 0.0f;
		}
		for (int i=tid; i<BK * BN; i+=kThreads) {
			const int r = i / BN;
			const int col_n = i % BN;
			const int gk = k0 + r;
			const int gc = block_col + col_n;
			bs[r][col_n] = (gk < k && gc < n) ? b[gk * n + gc] : 0.0f;
		}
		__syncthreads();
#pragma unroll
		for (int t=0; t<BK; ++t) {
#pragma unroll
			for (int i=0; i<TM; ++i) a_frag[i] = as[thread_row + i * kRowsOfThreads][t];
#pragma unroll
			for (int j=0; j<TN; ++j) b_frag[j] = bs[t][thread_col + j * kColsOfThreads];
#pragma unroll
			for (int i=0; i<TM; ++i) {
#pragma unroll
				for (int j=0; j<TN; ++j) acc[i][j] += a_frag[i] * b_frag[j];
			}
		}
		__syncthreads();
	}
#pragma unroll
	for (int i=0; i<TM; ++i) {
		const int gr = block_row + thread_row + i * kRowsOfThreads;
		if (gr >= m) continue;
#pragma unroll
		for (int j=0; j<TN; ++j) {
			const int gc = block_col + thread_col + j * kColsOfThreads;
			if (gc < n) c[gr * n + gc] = acc[i][j];
		}
	}
}

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t launch_register_tiled(const float* a, const float* b, float* c, int m, int n, int k,
                                  cudaStream_t stream) {
	const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
	matmul_register_tiled<BM, BN, BK, TM, TN><<<grid, (BM / TM) * (BN / TN), 0, stream>>>(a, b, c, m, n, k);
	return cudaGetLastError();
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
		// 256 threads, 8x8 outputs each: 4 FMAs per shared-memory load, 128
		// registers per thread, so 2 blocks (16 warps) fit on an SM.
		case MatmulKernel::RegisterTiled128: return launch_register_tiled<128, 128, 8, 8, 8>(a, b, c, m, n, k, stream);
		// 256 threads, 4x4 outputs each: 2 FMAs per load, but 64 registers per
		// thread, so 4 blocks (32 warps) fit on an SM to hide memory latency, and
		// a small output still splits into enough blocks to occupy every SM.
		case MatmulKernel::RegisterTiled64: return launch_register_tiled<64, 64, 8, 4, 4>(a, b, c, m, n, k, stream);
		case MatmulKernel::RegisterTiled: break;  // must be resolved to a tile size first
	}
	return cudaErrorInvalidValue;
}

}
