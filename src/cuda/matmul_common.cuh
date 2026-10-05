#pragma once

// Device helpers shared by the matmul kernels (CUDA sources only).

#include <cstddef>
#include <cstdint>

namespace minicompiler::cuda {

// Four consecutive elements of a row-major matrix with row stride `stride`,
// from (row, col); zero for rows at or past row_end and columns at or past
// col_end. kVec: one 16-byte load, for rows that start 16-byte aligned and a
// col_end that is a multiple of 4 (the 4 elements are then all in range or all
// out); otherwise element by element.
template <bool kVec>
__device__ __forceinline__ float4 load4(const float* __restrict__ p, int row, int col, int row_end, int col_end,
                                        int stride) {
	float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (row >= row_end) return v;
	const float* src = p + static_cast<std::ptrdiff_t>(row) * stride + col;
	if constexpr (kVec) {
		if (col < col_end) v = *reinterpret_cast<const float4*>(src);
	} else {
		if (col + 0 < col_end) v.x = src[0];
		if (col + 1 < col_end) v.y = src[1];
		if (col + 2 < col_end) v.z = src[2];
		if (col + 3 < col_end) v.w = src[3];
	}
	return v;
}

inline bool aligned16(const void* p) {
	return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

// float4 access needs every row of A, B and C to start 16-byte aligned.
inline bool rows_aligned(const float* a, const float* b, const float* c, int n, int k) {
	return n % 4 == 0 && k % 4 == 0 && aligned16(a) && aligned16(b) && aligned16(c);
}

}
