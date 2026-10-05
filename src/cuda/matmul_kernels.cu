#include "cuda/matmul_kernels.hpp"

#include <cstddef>
#include <cstdint>

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

// 128-bit memory operations. Same work split as the 128x128 register-tiled
// kernel (256 threads, an 8x8 block of C per thread), but every load moves
// four floats:
//  - Global loads are float4 when K and N are multiples of 4, so every row
//    starts 16-byte aligned (otherwise, element by element). A warp's 32
//    loads of A cover 16 rows x 8 consecutive k: one full 32-byte sector each.
//  - The A tile is stored transposed (M contiguous) in shared memory, so a
//    thread reads its 8 values of A as two float4, like its 8 values of B: 4
//    shared loads per k step for 64 FMAs, where the scalar kernel needs 16.
//  - Threads are laid out per warp so those reads are conflict-free. The 8
//    warps tile the block 2 x 4, each owning a 64x32 region, and thread
//    (ty, tx) of a warp owns rows 4ty..4ty+3 and 32+4ty..32+4ty+3 and columns
//    4tx..4tx+3 and 16+4tx..16+4tx+3 of it: each float4 read of A by the warp
//    then spans 128 contiguous bytes, one per bank.
//  - The transposed tile's rows are padded by 4 floats, which keeps them
//    16-byte aligned and spreads the transposed stores of a warp (16 rows,
//    two groups of 4 k) over all 32 banks.
constexpr int kVecBM = 128;
constexpr int kVecBN = 128;
constexpr int kVecThreads = 256;
constexpr int kAPad = 4;

template <bool kVec>
__device__ __forceinline__ float4 load4(const float* __restrict__ p, int row, int col, int rows, int cols) {
	float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (row >= rows) return v;
	const float* src = p + static_cast<std::ptrdiff_t>(row) * cols + col;
	if constexpr (kVec) {
		// cols is a multiple of 4, so the 4 elements are all in range or all out.
		if (col < cols) v = *reinterpret_cast<const float4*>(src);
	} else {
		if (col + 0 < cols) v.x = src[0];
		if (col + 1 < cols) v.y = src[1];
		if (col + 2 < cols) v.z = src[2];
		if (col + 3 < cols) v.w = src[3];
	}
	return v;
}

// This thread's share of the A tile (BK/8 float4 of 4 consecutive k) and of
// the B tile (BK/8 float4 of 4 consecutive columns) starting at k0.
template <int BK, bool kVec>
__device__ __forceinline__ void load_tiles(const float* __restrict__ a, const float* __restrict__ b, int m, int n,
                                           int k, int block_row, int block_col, int k0, float4 (&ra)[BK / 8],
                                           float4 (&rb)[BK / 8]) {
	const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
	for (int l=0; l<BK / 8; ++l) {
		const int g = warp + 8 * l;  // which 16 rows (g % 8) and which 8 k (g / 8)
		ra[l] = load4<kVec>(a, block_row + (g % 8) * 16 + lane / 2, k0 + (g / 8) * 8 + (lane % 2) * 4, m, k);
		rb[l] = load4<kVec>(b, k0 + g, block_col + lane * 4, k, n);
	}
}

template <int BK>
__device__ __forceinline__ void store_tiles(float (&as)[BK][kVecBM + kAPad], float (&bs)[BK][kVecBN],
                                            const float4 (&ra)[BK / 8], const float4 (&rb)[BK / 8]) {
	const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
	for (int l=0; l<BK / 8; ++l) {
		const int g = warp + 8 * l;
		const int row = (g % 8) * 16 + lane / 2;
		const int kk = (g / 8) * 8 + (lane % 2) * 4;
		as[kk + 0][row] = ra[l].x;
		as[kk + 1][row] = ra[l].y;
		as[kk + 2][row] = ra[l].z;
		as[kk + 3][row] = ra[l].w;
		*reinterpret_cast<float4*>(&bs[g][lane * 4]) = rb[l];
	}
}

template <int BK>
__device__ __forceinline__ void compute_tile(const float (&as)[BK][kVecBM + kAPad], const float (&bs)[BK][kVecBN],
                                             float (&acc)[8][8], int a_col, int b_col) {
#pragma unroll
	for (int t=0; t<BK; ++t) {
		const float4 a0 = *reinterpret_cast<const float4*>(&as[t][a_col]);
		const float4 a1 = *reinterpret_cast<const float4*>(&as[t][a_col + 32]);
		const float4 b0 = *reinterpret_cast<const float4*>(&bs[t][b_col]);
		const float4 b1 = *reinterpret_cast<const float4*>(&bs[t][b_col + 16]);
		const float af[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
		const float bf[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
		for (int i=0; i<8; ++i) {
#pragma unroll
			for (int j=0; j<8; ++j) acc[i][j] += af[i] * bf[j];
		}
	}
}

// Writes the thread's 8x8 block of C, as float4 where the row allows.
template <bool kVec>
__device__ __forceinline__ void store_c(float* __restrict__ c, const float (&acc)[8][8], int m, int n, int row0,
                                        int col0) {
#pragma unroll
	for (int i=0; i<8; ++i) {
		const int row = row0 + (i < 4 ? i : 28 + i);  // rows 4ty..4ty+3, then 32+4ty..
		if (row >= m) continue;
#pragma unroll
		for (int half=0; half<2; ++half) {
			const int col = col0 + half * 16;
			float* dst = c + static_cast<std::ptrdiff_t>(row) * n + col;
			const float* v = &acc[i][half * 4];
			if constexpr (kVec) {
				if (col < n) *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
			} else {
#pragma unroll
				for (int q=0; q<4; ++q) {
					if (col + q < n) dst[q] = v[q];
				}
			}
		}
	}
}

// kDoubleBuffer: the next tile's global loads are issued before the current
// tile's math and written to a second shared-memory buffer after it, so their
// latency hides behind the FMAs instead of stalling every tile, and one barrier
// per tile is enough instead of two.
template <int BK, bool kVec, bool kDoubleBuffer>
__global__ void __launch_bounds__(kVecThreads, 2)
matmul_vectorized(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n,
                  int k) {
	constexpr int kStages = kDoubleBuffer ? 2 : 1;
	__shared__ __align__(16) float as[kStages][BK][kVecBM + kAPad];
	__shared__ __align__(16) float bs[kStages][BK][kVecBN];
	const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
	const int warp_row = (warp / 4) * 64, warp_col = (warp % 4) * 32;  // the warp's 64x32 region
	const int ty = lane / 4, tx = lane % 4;
	const int block_row = blockIdx.y * kVecBM, block_col = blockIdx.x * kVecBN;
	const int a_col = warp_row + ty * 4, b_col = warp_col + tx * 4;

	float acc[8][8] = {};
	float4 ra[BK / 8], rb[BK / 8];
	if constexpr (kDoubleBuffer) {
		const int tiles = (k + BK - 1) / BK;
		load_tiles<BK, kVec>(a, b, m, n, k, block_row, block_col, 0, ra, rb);
		store_tiles<BK>(as[0], bs[0], ra, rb);
		__syncthreads();
		for (int t=0; t<tiles; ++t) {
			const int cur = t & 1;
			const bool more = t + 1 < tiles;
			if (more) load_tiles<BK, kVec>(a, b, m, n, k, block_row, block_col, (t + 1) * BK, ra, rb);
			compute_tile<BK>(as[cur], bs[cur], acc, a_col, b_col);
			// The other buffer was last read in the previous iteration, before its
			// closing barrier, so it is free to overwrite.
			if (more) store_tiles<BK>(as[cur ^ 1], bs[cur ^ 1], ra, rb);
			__syncthreads();
		}
	} else {
		for (int k0=0; k0<k; k0+=BK) {
			load_tiles<BK, kVec>(a, b, m, n, k, block_row, block_col, k0, ra, rb);
			store_tiles<BK>(as[0], bs[0], ra, rb);
			__syncthreads();
			compute_tile<BK>(as[0], bs[0], acc, a_col, b_col);
			__syncthreads();
		}
	}
	store_c<kVec>(c, acc, m, n, block_row + a_col, block_col + b_col);
}

// float4 access needs every row of A, B and C to start 16-byte aligned.
bool rows_aligned(const float* a, const float* b, const float* c, int n, int k) {
	const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
	return n % 4 == 0 && k % 4 == 0 && aligned(a) && aligned(b) && aligned(c);
}

template <int BK, bool kDoubleBuffer>
cudaError_t launch_vectorized(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t stream) {
	const dim3 grid((n + kVecBN - 1) / kVecBN, (m + kVecBM - 1) / kVecBM);
	if (rows_aligned(a, b, c, n, k)) {
		matmul_vectorized<BK, true, kDoubleBuffer><<<grid, kVecThreads, 0, stream>>>(a, b, c, m, n, k);
	} else {
		matmul_vectorized<BK, false, kDoubleBuffer><<<grid, kVecThreads, 0, stream>>>(a, b, c, m, n, k);
	}
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
		case MatmulKernel::Vectorized: return launch_vectorized<8, false>(a, b, c, m, n, k, stream);
		case MatmulKernel::DoubleBuffered: return launch_vectorized<8, true>(a, b, c, m, n, k, stream);
		case MatmulKernel::RegisterTiled: break;  // must be resolved to a tile size first
	}
	return cudaErrorInvalidValue;
}

}
