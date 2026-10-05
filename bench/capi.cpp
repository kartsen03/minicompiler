// A small C interface for timing minicompiler from Python with ctypes, in the
// same process as PyTorch (bench/torch_compare.py). It compiles a .mcg graph
// for the CPU or CUDA backend. On the CPU it runs on caller-owned buffers such
// as NumPy arrays; on the GPU the inputs are uploaded once and the kernels can
// be enqueued on a caller's stream (PyTorch's), so both are timed with the
// same CUDA events. It exists for benchmarking and is not a stable API.

#include "harness.hpp"

#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/runtime/backend.hpp"

#ifdef MINICOMPILER_HAVE_CUDA
#include "minicompiler/cuda/cuda_backend.hpp"
#endif

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define MC_EXPORT extern "C" __declspec(dllexport)
#else
#define MC_EXPORT extern "C" __attribute__((visibility("default")))
#endif

using namespace minicompiler;

namespace {

struct Compiled {
	Graph graph;
	std::unique_ptr<Executable> exe;
#ifdef MINICOMPILER_HAVE_CUDA
	cuda::CudaExecutable* gpu = nullptr;  // exe, when compiled for the CUDA backend
#endif
	std::vector<const float*> inputs;
	std::vector<float*> outputs;
};

void copy_string(char* out, int size, const std::string& text) {
	if (out && size > 0) std::snprintf(out, static_cast<std::size_t>(size), "%s", text.c_str());
}

#ifdef MINICOMPILER_HAVE_CUDA
cuda::CudaExecutable* gpu_of(void* handle) {
	return static_cast<Compiled*>(handle)->gpu;
}
#endif

}

// Parses `graph_path` with dim overrides `dims` ("M=64,N=4096" or ""), runs
// the pass pipeline `passes` ("default", "none" or a list such as
// "dne,fold"), and compiles for `backend` ("cpu" or "cuda"). Returns nullptr
// and writes a message to `error` on failure.
MC_EXPORT void* mc_compile(const char* graph_path, const char* dims, const char* passes, const char* backend,
                           char* error, int error_size) {
	ParseOptions options;
	std::stringstream list(dims ? dims : "");
	for (std::string item; std::getline(list, item, ',');) {
		const std::size_t eq = item.find('=');
		char* end = nullptr;
		errno = 0;
		const long long value = eq == std::string::npos ? 0 : std::strtoll(item.c_str() + eq + 1, &end, 10);
		if (eq == std::string::npos || *end != '\0' || errno == ERANGE) {
			copy_string(error, error_size, "bad dim '" + item + "', expected NAME=VALUE");
			return nullptr;
		}
		options.dims[item.substr(0, eq)] = value;
	}
	Result<Graph> graph = parse_graph_file(graph_path ? graph_path : "", options);
	if (!graph.ok()) {
		copy_string(error, error_size, graph.error().message);
		return nullptr;
	}
	Result<std::vector<std::string>> pipeline = parse_pipeline(passes ? passes : "default");
	if (!pipeline.ok()) {
		copy_string(error, error_size, pipeline.error().message);
		return nullptr;
	}
	Result<Graph> optimized = run_pipeline(graph.value(), pipeline.value());
	if (!optimized.ok()) {
		copy_string(error, error_size, optimized.error().message);
		return nullptr;
	}
	Result<std::unique_ptr<Backend>> chosen = create_backend(backend ? backend : "cpu");
	if (!chosen.ok()) {
		copy_string(error, error_size, chosen.error().message);
		return nullptr;
	}
	Result<std::unique_ptr<Executable>> exe = (*chosen)->compile(optimized.value());
	if (!exe.ok()) {
		copy_string(error, error_size, exe.error().message);
		return nullptr;
	}
	auto compiled = std::make_unique<Compiled>();
	compiled->graph = std::move(optimized).value();
	compiled->exe = std::move(exe).value();
#ifdef MINICOMPILER_HAVE_CUDA
	compiled->gpu = dynamic_cast<cuda::CudaExecutable*>(compiled->exe.get());
#endif
	return compiled.release();
}

MC_EXPORT int mc_num_inputs(const void* handle) {
	return static_cast<int>(static_cast<const Compiled*>(handle)->graph.inputs().size());
}

MC_EXPORT int mc_num_outputs(const void* handle) {
	return static_cast<int>(static_cast<const Compiled*>(handle)->graph.outputs().size());
}

// Number of elements in output k.
MC_EXPORT long long mc_output_elements(const void* handle, int k) {
	const Graph& g = static_cast<const Compiled*>(handle)->graph;
	return static_cast<long long>(g.node(g.outputs()[static_cast<std::size_t>(k)]).type.num_elements());
}

// Runs the graph: inputs[k] and outputs[k] point to float32 host buffers in
// graph order (on the GPU this includes the copies). Returns 0 on success.
MC_EXPORT int mc_run(void* handle, const float* const* inputs, float* const* outputs) {
	auto* c = static_cast<Compiled*>(handle);
	c->inputs.assign(inputs, inputs + c->graph.inputs().size());
	c->outputs.assign(outputs, outputs + c->graph.outputs().size());
	return c->exe->run_buffers(c->inputs, c->outputs).ok() ? 0 : 1;
}

// GPU only. Copies the host inputs into the executable's device buffers,
// where they stay for any number of mc_enqueue calls. Returns 0 on success.
MC_EXPORT int mc_upload(void* handle, const float* const* inputs) {
#ifdef MINICOMPILER_HAVE_CUDA
	auto* c = static_cast<Compiled*>(handle);
	if (!c->gpu) return 1;
	c->inputs.assign(inputs, inputs + c->graph.inputs().size());
	return c->gpu->upload_inputs(c->inputs).ok() && c->gpu->synchronize().ok() ? 0 : 1;
#else
	(void)handle, (void)inputs;
	return 1;
#endif
}

// GPU only. Launches the graph's kernels on `stream` (a cudaStream_t; null
// means the executable's own stream): no copies, no synchronization.
MC_EXPORT int mc_enqueue(void* handle, void* stream) {
#ifdef MINICOMPILER_HAVE_CUDA
	cuda::CudaExecutable* gpu = gpu_of(handle);
	return gpu && gpu->enqueue(stream).ok() ? 0 : 1;
#else
	(void)handle, (void)stream;
	return 1;
#endif
}

// GPU only. Waits for the device, then copies the outputs to host buffers.
MC_EXPORT int mc_download(void* handle, float* const* outputs) {
#ifdef MINICOMPILER_HAVE_CUDA
	auto* c = static_cast<Compiled*>(handle);
	if (!c->gpu) return 1;
	c->outputs.assign(outputs, outputs + c->graph.outputs().size());
	return c->gpu->download_outputs(c->outputs).ok() ? 0 : 1;
#else
	(void)handle, (void)outputs;
	return 1;
#endif
}

// GPU only: kernel launches per mc_enqueue (-1 for a CPU executable).
MC_EXPORT int mc_kernel_launches(void* handle) {
#ifdef MINICOMPILER_HAVE_CUDA
	cuda::CudaExecutable* gpu = gpu_of(handle);
	return gpu ? static_cast<int>(gpu->kernel_launches()) : -1;
#else
	(void)handle;
	return -1;
#endif
}

// GPU only: the matmul kernel each matmul runs, comma-separated, into `out`.
// Returns 0 on success.
MC_EXPORT int mc_matmul_kernels(void* handle, char* out, int size) {
#ifdef MINICOMPILER_HAVE_CUDA
	cuda::CudaExecutable* gpu = gpu_of(handle);
	if (!gpu) return 1;
	std::string names;
	for (cuda::MatmulKernel k : gpu->matmul_kernels()) {
		if (!names.empty()) names += ",";
		names += cuda::matmul_kernel_name(k);
	}
	copy_string(out, size, names);
	return 0;
#else
	(void)handle, (void)out, (void)size;
	return 1;
#endif
}

// The GPU's SM clock in MHz right now (NVML), or 0 if unavailable.
MC_EXPORT double mc_gpu_sm_clock_mhz() {
#ifdef MINICOMPILER_HAVE_CUDA
	return cuda::current_sm_clock_mhz(0);
#else
	return 0.0;
#endif
}

// A JSON object describing GPU 0 the way bench_cuda records it, with
// `clocks` (n samples of the SM clock, may be empty) summarized. Returns 0 on
// success, 1 without a GPU, 2 when `out` is too small.
MC_EXPORT int mc_gpu_device_json(const double* clocks, int n, char* out, int size) {
#ifdef MINICOMPILER_HAVE_CUDA
	Result<cuda::DeviceInfo> device = cuda::query_device(0);
	if (!device.ok()) return 1;
	bench::JsonWriter json;
	json.begin_object();
	bench::write_device(json, device.value(), std::vector<double>(clocks, clocks + (clocks ? n : 0)));
	json.end_object();
	const std::string text = json.str();
	if (static_cast<int>(text.size()) >= size) return 2;
	copy_string(out, size, text);
	return 0;
#else
	(void)clocks, (void)n, (void)out, (void)size;
	return 1;
#endif
}

MC_EXPORT void mc_release(void* handle) {
	delete static_cast<Compiled*>(handle);
}
