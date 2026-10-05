// Tensor-core matmul kernels (WMMA): FP32 inputs rounded to TF32, BF16 or
// FP16 as they are staged into shared memory, products accumulated in FP32.
// Not exact FP32: see MatmulKernel::TensorCore* for the error each allows.

#include "cuda/matmul_kernels.hpp"

#include "cuda/matmul_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace minicompiler::cuda {

// mma.h declares WMMA only where it can be used: in the host pass, and in
// device code for compute capability 7.0 and up, with TF32 from 8.0. Device
// code for an architecture without a kernel's format compiles that kernel
// without a body (see matmul_tensor_core), and stand-ins take the place of
// what mma.h leaves out.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define MC_HAVE_WMMA 1
namespace wmma = nvcuda::wmma;
#endif

// The input precisions: how a float is rounded into the shared-memory tile,
// the element type of a WMMA fragment, the K of one 16x16xK mma, and the first
// compute capability with that tensor-core format (times 100). Not in the
// anonymous namespace, where a device pass that compiles a kernel without its
// body would warn that these members are never used.
namespace tensor_core {

struct Tf32 {
	using Smem = float;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
	using Frag = wmma::precision::tf32;
	__device__ static Smem from(float x) { return wmma::__float_to_tf32(x); }
#else
	using Frag = float;
	__device__ static Smem from(float x) { return x; }
#endif
	static constexpr int kK = 8;
	static constexpr int kMinArch = 800;
};
struct Bf16 {
	using Smem = __nv_bfloat16;
	using Frag = __nv_bfloat16;
	static constexpr int kK = 16;
	static constexpr int kMinArch = 800;
	__device__ static Smem from(float x) { return __float2bfloat16_rn(x); }
};
struct F16 {
	using Smem = __half;
	using Frag = __half;
	static constexpr int kK = 16;
	static constexpr int kMinArch = 700;
	__device__ static Smem from(float x) { return __float2half_rn(x); }
};

}

namespace {

using tensor_core::Bf16;
using tensor_core::F16;
using tensor_core::Tf32;

constexpr int kTcBM = 128;
constexpr int kTcBN = 128;
constexpr int kTcThreads = 256;
constexpr int kTcWarps = kTcThreads / 32;

// The architecture device code is being compiled for (0 in the host pass).
#if defined(__CUDA_ARCH__)
constexpr int kArch = __CUDA_ARCH__;
#else
constexpr int kArch = 0;
#endif

// A block of 256 threads computes a 128x128 tile of C. The 8 warps tile it 2
// x 4; each warp owns a 64x32 region as 4 x 2 WMMA accumulators of 16x16.
// Per step of BK = 2 mma depths along K the block stages a 128 x BK tile of
// A and a BK x 128 tile of B in shared memory, rounded to the input precision
// once there rather than at every fragment load, and double-buffered as in
// the SIMT kernel: the next tile's float4 global loads are issued before the
// current tile's mma and stored after them. Rows are padded by 16 bytes, the
// usual remedy for bank conflicts in fragment loads. Accumulators leave
// through a per-warp 16x16 scratch tile, so edge tiles store with bounds
// checks and C needs no alignment. The kernel has a body only in device code
// for an architecture with P's format.
template <typename P, bool kVec>
__global__ void __launch_bounds__(kTcThreads)
matmul_tensor_core(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n,
                   int k) {
#if defined(MC_HAVE_WMMA)
	if constexpr (kArch >= P::kMinArch) {
		using S = typename P::Smem;
		constexpr int kK = P::kK;
		constexpr int BK = 2 * kK;
		constexpr int kPad = 16 / static_cast<int>(sizeof(S));
		__shared__ __align__(32) S as[2][kTcBM][BK + kPad];
		__shared__ __align__(32) S bs[2][BK][kTcBN + kPad];
		__shared__ __align__(32) float scratch[kTcWarps][16][16];

		const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
		const int warp_row = (warp / 4) * 64, warp_col = (warp % 4) * 32;
		const int block_row = blockIdx.y * kTcBM, block_col = blockIdx.x * kTcBN;

		wmma::fragment<wmma::accumulator, 16, 16, kK, float> acc[4][2];
#pragma unroll
		for (int i=0; i<4; ++i) {
#pragma unroll
			for (int j=0; j<2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
		}

		// This thread's float4 of the A tile (BK/4 per row) and of the B tile.
		constexpr int kA4 = kTcBM * BK / 4 / kTcThreads;
		constexpr int kB4 = BK * kTcBN / 4 / kTcThreads;
		float4 ra[kA4], rb[kB4];
		auto load = [&](int k0) {
#pragma unroll
			for (int l=0; l<kA4; ++l) {
				const int f = threadIdx.x + l * kTcThreads;
				ra[l] = load4<kVec>(a, block_row + f / (BK / 4), k0 + (f % (BK / 4)) * 4, m, k, k);
			}
#pragma unroll
			for (int l=0; l<kB4; ++l) {
				const int f = threadIdx.x + l * kTcThreads;
				rb[l] = load4<kVec>(b, k0 + f / (kTcBN / 4), block_col + (f % (kTcBN / 4)) * 4, k, n, n);
			}
		};
		auto store = [&](int s) {
#pragma unroll
			for (int l=0; l<kA4; ++l) {
				const int f = threadIdx.x + l * kTcThreads;
				S* dst = &as[s][f / (BK / 4)][(f % (BK / 4)) * 4];
				dst[0] = P::from(ra[l].x);
				dst[1] = P::from(ra[l].y);
				dst[2] = P::from(ra[l].z);
				dst[3] = P::from(ra[l].w);
			}
#pragma unroll
			for (int l=0; l<kB4; ++l) {
				const int f = threadIdx.x + l * kTcThreads;
				S* dst = &bs[s][f / (kTcBN / 4)][(f % (kTcBN / 4)) * 4];
				dst[0] = P::from(rb[l].x);
				dst[1] = P::from(rb[l].y);
				dst[2] = P::from(rb[l].z);
				dst[3] = P::from(rb[l].w);
			}
		};

		const int tiles = (k + BK - 1) / BK;
		load(0);
		store(0);
		__syncthreads();
		for (int t=0; t<tiles; ++t) {
			const int cur = t & 1;
			const bool more = t + 1 < tiles;
			if (more) load((t + 1) * BK);
#pragma unroll
			for (int kk=0; kk<BK; kk+=kK) {
				wmma::fragment<wmma::matrix_a, 16, 16, kK, typename P::Frag, wmma::row_major> af[4];
				wmma::fragment<wmma::matrix_b, 16, 16, kK, typename P::Frag, wmma::row_major> bf[2];
#pragma unroll
				for (int i=0; i<4; ++i) wmma::load_matrix_sync(af[i], &as[cur][warp_row + i * 16][kk], BK + kPad);
#pragma unroll
				for (int j=0; j<2; ++j) wmma::load_matrix_sync(bf[j], &bs[cur][kk][warp_col + j * 16], kTcBN + kPad);
#pragma unroll
				for (int i=0; i<4; ++i) {
#pragma unroll
					for (int j=0; j<2; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
				}
			}
			// The other buffer was last read before the previous barrier.
			if (more) store(cur ^ 1);
			__syncthreads();
		}

		float (&sc)[16][16] = scratch[warp];
		const int r = lane / 2, c0 = (lane % 2) * 8;  // each lane copies 8 consecutive floats of a row
#pragma unroll
		for (int i=0; i<4; ++i) {
#pragma unroll
			for (int j=0; j<2; ++j) {
				wmma::store_matrix_sync(&sc[0][0], acc[i][j], 16, wmma::mem_row_major);
				__syncwarp();
				const int gr = block_row + warp_row + i * 16 + r;
				const int gc = block_col + warp_col + j * 16 + c0;
				if (gr < m) {
					float* dst = c + static_cast<std::ptrdiff_t>(gr) * n + gc;
#pragma unroll
					for (int q=0; q<8; ++q) {
						if (gc + q < n) dst[q] = sc[r][c0 + q];
					}
				}
				__syncwarp();
			}
		}
	}
#endif
}

// Whether the current GPU runs the kernel for P with its body: the body is
// compiled only for architectures that have the format, and the code a GPU
// runs can be PTX built for an older one (a build for 7.5 on an 8.0 GPU), in
// which the kernel would do nothing. That code's architecture is ptxVersion.
template <typename P>
bool supported() {
	cudaFuncAttributes attr;
	if (cudaFuncGetAttributes(&attr, matmul_tensor_core<P, true>) != cudaSuccess) {
		cudaGetLastError();  // no code for this GPU; clear the error so no later check reports it
		return false;
	}
	return attr.ptxVersion * 10 >= P::kMinArch;
}

template <typename P>
cudaError_t launch(const float* a, const float* b, float* c, int m, int n, int k, cudaStream_t stream) {
	if (!supported<P>()) return cudaErrorNotSupported;
	const dim3 grid((n + kTcBN - 1) / kTcBN, (m + kTcBM - 1) / kTcBM);
	if (rows_aligned(a, b, c, n, k)) {
		matmul_tensor_core<P, true><<<grid, kTcThreads, 0, stream>>>(a, b, c, m, n, k);
	} else {
		matmul_tensor_core<P, false><<<grid, kTcThreads, 0, stream>>>(a, b, c, m, n, k);
	}
	return cudaGetLastError();
}

}

bool tensor_core_supported(MatmulKernel kernel) {
	switch (kernel) {
		case MatmulKernel::TensorCoreTf32: return supported<Tf32>();
		case MatmulKernel::TensorCoreBf16: return supported<Bf16>();
		case MatmulKernel::TensorCoreF16: return supported<F16>();
		default: return false;
	}
}

cudaError_t launch_tensor_core(MatmulKernel kernel, const float* a, const float* b, float* c, int m, int n, int k,
                               cudaStream_t stream) {
	switch (kernel) {
		case MatmulKernel::TensorCoreTf32: return launch<Tf32>(a, b, c, m, n, k, stream);
		case MatmulKernel::TensorCoreBf16: return launch<Bf16>(a, b, c, m, n, k, stream);
		case MatmulKernel::TensorCoreF16: return launch<F16>(a, b, c, m, n, k, stream);
		default: return cudaErrorInvalidValue;
	}
}

}
