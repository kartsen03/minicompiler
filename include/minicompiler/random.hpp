#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minicompiler {

// Deterministic initialization for weights and benchmark inputs.
//
// bench/mcgraph.py reimplements this generator in NumPy so that minicompiler
// and PyTorch see bit-identical tensors. The generator is splitmix64; each
// value takes the top 24 bits of one draw as an exact float u in [0, 1) and
// returns lo + (hi - lo) * u, rounded after the multiply and after the add.
void fill_uniform(float* out, std::size_t count, std::uint64_t seed, float lo, float hi);
std::vector<float> uniform_values(std::size_t count, std::uint64_t seed, float lo, float hi);

}
