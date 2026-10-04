#include "cuda/kernel_compiler.hpp"

#include "cuda/cuda_check.hpp"

#include <chrono>

namespace minicompiler::cuda {
namespace {

struct ProgramGuard {
	nvrtcProgram program = nullptr;
	~ProgramGuard() {
		if (program) nvrtcDestroyProgram(&program);
	}
};

}

KernelCompiler& KernelCompiler::instance() {
	static KernelCompiler compiler;
	return compiler;
}

Result<CUfunction> KernelCompiler::get(const std::string& source, const char* name, int cc_major, int cc_minor,
                                        double* compile_ms) {
	const std::string arch = "sm_" + std::to_string(cc_major) + std::to_string(cc_minor);
	const std::string key = arch + '\n' + source;
	std::lock_guard<std::mutex> lock(mutex_);
	if (compile_ms) *compile_ms = 0.0;
	if (auto it = cache_.find(key); it != cache_.end()) return it->second.function;

	const auto start = std::chrono::steady_clock::now();
	ProgramGuard guard;
	MC_NVRTC_RETURN(nvrtcCreateProgram(&guard.program, source.c_str(), "mc_generated.cu", 0, nullptr, nullptr));
	const std::string arch_option = "--gpu-architecture=" + arch;
	const char* options[] = {arch_option.c_str(), "--std=c++17", "-lineinfo"};
	if (nvrtcCompileProgram(guard.program, 3, options) != NVRTC_SUCCESS) {
		std::size_t log_size = 0;
		nvrtcGetProgramLogSize(guard.program, &log_size);
		std::string log(log_size, '\0');
		nvrtcGetProgramLog(guard.program, log.data());
		return Error{"NVRTC could not compile a generated kernel:\n" + log + "\n" + source};
	}
	std::size_t cubin_size = 0;
	MC_NVRTC_RETURN(nvrtcGetCUBINSize(guard.program, &cubin_size));
	std::string cubin(cubin_size, '\0');
	MC_NVRTC_RETURN(nvrtcGetCUBIN(guard.program, cubin.data()));

	Entry entry{};
	MC_CU_RETURN(cuModuleLoadData(&entry.module, cubin.data()));
	MC_CU_RETURN(cuModuleGetFunction(&entry.function, entry.module, name));
	cache_.emplace(key, entry);
	if (compile_ms) {
		*compile_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	}
	return entry.function;
}

}
