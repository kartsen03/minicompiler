#include "cuda/matmul_kernels.hpp"

#include "cuda/matmul_common.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

// This thread's share of the A tile (BK/8 float4 of 4 consecutive k) and of
// the B tile (BK/8 float4 of 4 consecutive columns) starting at k0, with K
// ending at k_end (a split's end, or k).
template <int BK, bool kVec>
__device__ __forceinline__ void load_tiles(const float* __restrict__ a, const float* __restrict__ b, int m, int n,
                                           int k, int k_end, int block_row, int block_col, int k0,
                                           float4 (&ra)[BK / 8], float4 (&rb)[BK / 8]) {
	const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
	for (int l=0; l<BK / 8; ++l) {
		const int g = warp + 8 * l;  // which 16 rows (g % 8) and which 8 k (g / 8)
		ra[l] = load4<kVec>(a, block_row + (g % 8) * 16 + lane / 2, k0 + (g / 8) * 8 + (lane % 2) * 4, m, k_end, k);
		rb[l] = load4<kVec>(b, k0 + g, block_col + lane * 4, k_end, n, n);
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
//
// Split-K: block z of the grid's depth covers K from z * k_split to
// (z + 1) * k_split. With a depth of 1 it writes C; deeper, it writes its
// partial product to slice z of `partial`, and sum_splits adds the slices.
template <int BK, bool kVec, bool kDoubleBuffer>
__global__ void __launch_bounds__(kVecThreads, 2)
matmul_vectorized(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n,
                  int k, int k_split, float* __restrict__ partial) {
	constexpr int kStages = kDoubleBuffer ? 2 : 1;
	__shared__ __align__(16) float as[kStages][BK][kVecBM + kAPad];
	__shared__ __align__(16) float bs[kStages][BK][kVecBN];
	const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
	const int warp_row = (warp / 4) * 64, warp_col = (warp % 4) * 32;  // the warp's 64x32 region
	const int ty = lane / 4, tx = lane % 4;
	const int block_row = blockIdx.y * kVecBM, block_col = blockIdx.x * kVecBN;
	const int a_col = warp_row + ty * 4, b_col = warp_col + tx * 4;
	const int k_begin = blockIdx.z * k_split;
	const int k_end = min(k, k_begin + k_split);

	float acc[8][8] = {};
	float4 ra[BK / 8], rb[BK / 8];
	if constexpr (kDoubleBuffer) {
		const int tiles = (k_end - k_begin + BK - 1) / BK;
		load_tiles<BK, kVec>(a, b, m, n, k, k_end, block_row, block_col, k_begin, ra, rb);
		store_tiles<BK>(as[0], bs[0], ra, rb);
		__syncthreads();
		for (int t=0; t<tiles; ++t) {
			const int cur = t & 1;
			const bool more = t + 1 < tiles;
			if (more) load_tiles<BK, kVec>(a, b, m, n, k, k_end, block_row, block_col, k_begin + (t + 1) * BK, ra, rb);
			compute_tile<BK>(as[cur], bs[cur], acc, a_col, b_col);
			// The other buffer was last read in the previous iteration, before its
			// closing barrier, so it is free to overwrite.
			if (more) store_tiles<BK>(as[cur ^ 1], bs[cur ^ 1], ra, rb);
			__syncthreads();
		}
	} else {
		for (int k0=k_begin; k0<k_end; k0+=BK) {
			load_tiles<BK, kVec>(a, b, m, n, k, k_end, block_row, block_col, k0, ra, rb);
			store_tiles<BK>(as[0], bs[0], ra, rb);
			__syncthreads();
			compute_tile<BK>(as[0], bs[0], acc, a_col, b_col);
			__syncthreads();
		}
	}
	// Chosen after the loop, so the pointer does not hold registers during it.
	float* __restrict__ out = gridDim.z == 1 ? c : partial + static_cast<std::ptrdiff_t>(blockIdx.z) * m * n;
	store_c<kVec>(out, acc, m, n, block_row + a_col, block_col + b_col);
}

// C = the split-K partial products added slice by slice, always in the same
// order, so the result does not depend on how the blocks were scheduled.
template <bool kVec>
__global__ void __launch_bounds__(256)
sum_splits(const float* __restrict__ partial, float* __restrict__ c, std::size_t count, int splits) {
	using T = std::conditional_t<kVec, float4, float>;
	const T* in = reinterpret_cast<const T*>(partial);
	T* out = reinterpret_cast<T*>(c);
	const std::size_t n = kVec ? count / 4 : count;
	for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
	     i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
		T s = in[i];
		for (int z=1; z<splits; ++z) {
			const T v = in[z * n + i];
			if constexpr (kVec) {
				s.x += v.x;
				s.y += v.y;
				s.z += v.z;
				s.w += v.w;
			} else {
				s += v;
			}
		}
		out[i] = s;
	}
}

template <int BK, bool kDoubleBuffer>
cudaError_t launch_vectorized(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t stream,
                              int splits = 1, float* workspace = nullptr) {
	const int k_split = split_size(k, splits, BK);
	const int depth = split_depth(k, k_split);
	if (depth > 1 && !workspace) return cudaErrorInvalidValue;
	const dim3 grid((n + kVecBN - 1) / kVecBN, (m + kVecBM - 1) / kVecBM, depth);
	const bool vec = rows_aligned(a, b, c, n, k) && (depth == 1 || aligned16(workspace));
	if (vec) {
		matmul_vectorized<BK, true, kDoubleBuffer><<<grid, kVecThreads, 0, stream>>>(a, b, c, m, n, k, k_split,
		                                                                              workspace);
	} else {
		matmul_vectorized<BK, false, kDoubleBuffer><<<grid, kVecThreads, 0, stream>>>(a, b, c, m, n, k, k_split,
		                                                                               workspace);
	}
	if (depth == 1) return cudaGetLastError();
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) return err;
	return sum_split_partials(workspace, c, static_cast<std::size_t>(m) * n, depth, vec, stream);
}

}

cudaError_t sum_split_partials(const float* partial, float* c, std::size_t count, int splits, bool vec,
                               cudaStream_t stream) {
	const std::size_t items = vec ? count / 4 : count;
	const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((items + 255) / 256, 4096));
	if (vec) {
		sum_splits<true><<<blocks, 256, 0, stream>>>(partial, c, count, splits);
	} else {
		sum_splits<false><<<blocks, 256, 0, stream>>>(partial, c, count, splits);
	}
	return cudaGetLastError();
}

int split_k_splits(int m, int n, int k, int sm_count, int blocks_per_sm) {
	const std::int64_t tiles = ((static_cast<std::int64_t>(m) + kVecBM - 1) / kVecBM) *
	                           ((static_cast<std::int64_t>(n) + kVecBN - 1) / kVecBN);
	const std::int64_t sms = sm_count;
	// An output that nearly fills the GPU gains less from more blocks than the
	// partial products cost to write and add.
	if (5 * tiles >= 4 * sms) return 1;
	// Up to half the SMs' worth of tiles: one block per SM, since a second block
	// on only some SMs makes a tail that the rest wait for. Beyond that, two
	// splits would already overflow one block per SM, so split for up to two
	// where an SM holds two; where it holds one, a second wave would cost more
	// than the split saves.
	std::int64_t splits = 2 * tiles <= sms || blocks_per_sm < 2 ? sms / tiles : 2 * sms / tiles;
	splits = std::min(splits, static_cast<std::int64_t>(k / 64));  // at least 64 of K per split
	return static_cast<int>(std::max<std::int64_t>(splits, 1));
}

bool uses_split_k(MatmulKernel kernel) {
	return kernel == MatmulKernel::SplitK || kernel == MatmulKernel::TensorCoreTf32 ||
	       kernel == MatmulKernel::TensorCoreBf16 || kernel == MatmulKernel::TensorCoreF16;
}

int matmul_splits(MatmulKernel kernel, int m, int n, int k, int sm_count) {
	if (kernel == MatmulKernel::SplitK) return split_k_splits(m, n, k, sm_count);
	if (uses_split_k(kernel)) return split_k_splits(m, n, k, sm_count, tensor_core_blocks_per_sm(kernel));
	return 1;
}

std::size_t matmul_workspace_bytes(MatmulKernel kernel, int m, int n, int splits) {
	if (!uses_split_k(kernel) || splits <= 1) return 0;
	return static_cast<std::size_t>(splits) * m * n * sizeof(float);
}

cudaError_t launch_matmul(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream, int splits, float* workspace) {
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
		case MatmulKernel::SplitK: return launch_vectorized<8, true>(a, b, c, m, n, k, stream, splits, workspace);
		case MatmulKernel::TensorCoreTf32:
		case MatmulKernel::TensorCoreBf16:
		case MatmulKernel::TensorCoreF16:
			return launch_tensor_core(kernel, a, b, c, m, n, k, stream, splits, workspace);
		case MatmulKernel::Auto: break;  // must be resolved to a kernel first
	}
	return cudaErrorInvalidValue;
}

}
