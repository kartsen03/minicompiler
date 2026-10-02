#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/tensor.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace minicompiler {

// A tensor in host memory: a type and its values, row-major.
struct HostTensor {
	TensorType type;
	std::vector<float> data;

	HostTensor() = default;
	explicit HostTensor(TensorType t) : type(std::move(t)), data(type.num_elements()) {}
	HostTensor(TensorType t, std::vector<float> values) : type(std::move(t)), data(std::move(values)) {}
};

// Seeded inputs for `graph`: input k is filled with uniform_values(n, seed + k,
// lo, hi). bench/mcgraph.py generates the same tensors for PyTorch.
std::vector<HostTensor> make_random_inputs(const Graph& graph, std::uint64_t seed, float lo = -1.0f,
                                           float hi = 1.0f);

}
