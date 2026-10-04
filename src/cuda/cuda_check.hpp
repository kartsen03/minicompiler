#pragma once

// Turns CUDA runtime, driver and NVRTC error codes into minicompiler Errors.
// Each macro returns from the enclosing function, which must return Status or
// a Result.

#include "minicompiler/result.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <string>

namespace minicompiler::cuda {

inline Error cuda_error(const char* what, cudaError_t e) {
	return Error{std::string(what) + ": " + cudaGetErrorString(e)};
}

inline Error driver_error(const char* what, CUresult e) {
	const char* msg = nullptr;
	cuGetErrorString(e, &msg);
	return Error{std::string(what) + ": " + (msg ? msg : "unknown CUDA driver error")};
}

inline Error nvrtc_error(const char* what, nvrtcResult e) {
	return Error{std::string(what) + ": " + nvrtcGetErrorString(e)};
}

}

#define MC_CUDA_RETURN(expr)                                                          \
	do {                                                                              \
		const cudaError_t mc_err_ = (expr);                                           \
		if (mc_err_ != cudaSuccess) return ::minicompiler::cuda::cuda_error(#expr, mc_err_); \
	} while (0)

#define MC_CU_RETURN(expr)                                                              \
	do {                                                                                \
		const CUresult mc_err_ = (expr);                                                \
		if (mc_err_ != CUDA_SUCCESS) return ::minicompiler::cuda::driver_error(#expr, mc_err_); \
	} while (0)

#define MC_NVRTC_RETURN(expr)                                                            \
	do {                                                                                 \
		const nvrtcResult mc_err_ = (expr);                                              \
		if (mc_err_ != NVRTC_SUCCESS) return ::minicompiler::cuda::nvrtc_error(#expr, mc_err_); \
	} while (0)
