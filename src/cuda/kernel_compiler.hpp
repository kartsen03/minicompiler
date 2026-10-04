#pragma once

#include "minicompiler/result.hpp"

#include <cuda.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace minicompiler::cuda {

// Compiles generated CUDA source with NVRTC straight to a cubin for the
// device's exact architecture (no PTX JIT at load time) and loads it with the
// driver API. Each distinct (architecture, source) pair is compiled once per
// process; later requests come from the cache. The calling thread must have
// a current CUDA context.
class KernelCompiler {
public:
	static KernelCompiler& instance();

	// The function `name` in `source`, built for sm_<cc_major><cc_minor>.
	// `compile_ms` receives the NVRTC time (0 for a cache hit).
	Result<CUfunction> get(const std::string& source, const char* name, int cc_major, int cc_minor,
	                       double* compile_ms = nullptr);

private:
	struct Entry {
		CUmodule module;
		CUfunction function;
	};
	std::mutex mutex_;
	std::unordered_map<std::string, Entry> cache_;
};

}
