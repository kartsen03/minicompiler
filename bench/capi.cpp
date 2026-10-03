// A small C interface for timing minicompiler from Python with ctypes, in the
// same process as PyTorch (bench/torch_compare.py). It compiles a .mcg graph
// for the CPU backend and runs it on caller-owned buffers such as NumPy
// arrays. It exists for benchmarking and is not a stable API.

#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/runtime/backend.hpp"

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
	std::vector<const float*> inputs;
	std::vector<float*> outputs;
};

void set_error(char* error, int size, const std::string& message) {
	if (error && size > 0) std::snprintf(error, static_cast<std::size_t>(size), "%s", message.c_str());
}

}

// Parses `graph_path` with dim overrides `dims` ("M=64,N=4096" or ""), runs
// the pass pipeline `passes` ("default", "none" or a list such as
// "dne,fold"), and compiles for the CPU backend. Returns nullptr and writes a
// message to `error` on failure.
MC_EXPORT void* mc_compile(const char* graph_path, const char* dims, const char* passes, char* error,
                           int error_size) {
	ParseOptions options;
	std::stringstream list(dims ? dims : "");
	for (std::string item; std::getline(list, item, ',');) {
		const std::size_t eq = item.find('=');
		char* end = nullptr;
		errno = 0;
		const long long value = eq == std::string::npos ? 0 : std::strtoll(item.c_str() + eq + 1, &end, 10);
		if (eq == std::string::npos || *end != '\0' || errno == ERANGE) {
			set_error(error, error_size, "bad dim '" + item + "', expected NAME=VALUE");
			return nullptr;
		}
		options.dims[item.substr(0, eq)] = value;
	}
	Result<Graph> graph = parse_graph_file(graph_path ? graph_path : "", options);
	if (!graph.ok()) {
		set_error(error, error_size, graph.error().message);
		return nullptr;
	}
	Result<std::vector<std::string>> pipeline = parse_pipeline(passes ? passes : "default");
	if (!pipeline.ok()) {
		set_error(error, error_size, pipeline.error().message);
		return nullptr;
	}
	Result<Graph> optimized = run_pipeline(graph.value(), pipeline.value());
	if (!optimized.ok()) {
		set_error(error, error_size, optimized.error().message);
		return nullptr;
	}
	Result<std::unique_ptr<Backend>> backend = create_backend("cpu");
	Result<std::unique_ptr<Executable>> exe = (*backend)->compile(optimized.value());
	if (!exe.ok()) {
		set_error(error, error_size, exe.error().message);
		return nullptr;
	}
	auto compiled = std::make_unique<Compiled>();
	compiled->graph = std::move(optimized).value();
	compiled->exe = std::move(exe).value();
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

// Runs the graph: inputs[k] and outputs[k] point to float32 buffers in graph
// order. Returns 0 on success.
MC_EXPORT int mc_run(void* handle, const float* const* inputs, float* const* outputs) {
	auto* c = static_cast<Compiled*>(handle);
	c->inputs.assign(inputs, inputs + c->graph.inputs().size());
	c->outputs.assign(outputs, outputs + c->graph.outputs().size());
	return c->exe->run_buffers(c->inputs, c->outputs).ok() ? 0 : 1;
}

MC_EXPORT void mc_release(void* handle) {
	delete static_cast<Compiled*>(handle);
}
