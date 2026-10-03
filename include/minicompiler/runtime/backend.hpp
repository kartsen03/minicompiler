#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/result.hpp"
#include "minicompiler/runtime/host_tensor.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace minicompiler {

// A graph compiled for one backend, ready to run repeatedly. Buffers are
// allocated once at compile time and reused by every run, so an Executable
// is not safe to run from two threads at once.
class Executable {
public:
	virtual ~Executable() = default;

	// `inputs` follow Graph::inputs() in order and type. `outputs` is resized
	// to match Graph::outputs(); passing the same vector again reuses its
	// storage. `outputs` must be a different vector from `inputs`.
	virtual Status run(const std::vector<HostTensor>& inputs, std::vector<HostTensor>& outputs) = 0;

	// Runs on caller-owned host memory, with no copies and no allocation:
	// inputs[k] holds the elements of Graph::inputs()[k] and outputs[k] has
	// room for those of Graph::outputs()[k]. An output may not overlap an input.
	// The PyTorch benchmark uses this to run on NumPy arrays in place.
	virtual Status run_buffers(const std::vector<const float*>& inputs, const std::vector<float*>& outputs) = 0;

	// Bytes of intermediate storage held after buffer reuse.
	virtual std::size_t intermediate_bytes() const = 0;
};

class Backend {
public:
	virtual ~Backend() = default;
	virtual const char* name() const = 0;
	virtual Result<std::unique_ptr<Executable>> compile(const Graph& graph) = 0;
};

// Backends compiled into this build: always "cpu", plus "cuda" when the
// project was configured with a CUDA toolkit.
std::vector<std::string> available_backends();

// Selects a backend at runtime by name.
Result<std::unique_ptr<Backend>> create_backend(std::string_view name);

}
