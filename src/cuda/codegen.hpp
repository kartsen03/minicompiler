#pragma once

// Generates CUDA C++ source for elementwise kernels. Plain C++ with no CUDA
// headers, so it is built and unit-tested even where no GPU or toolkit exists.

#include "minicompiler/fused_program.hpp"
#include "minicompiler/tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace minicompiler::cuda {

inline constexpr const char* kElementwiseKernelName = "mc_elementwise";

struct ElementwiseKernel {
	std::string source;
	// Each thread handles 4 consecutive elements with float4 loads and
	// stores; the kernel's `n` argument then counts groups of 4.
	bool vectorized = false;
};

// One kernel that evaluates `program` for every element of an output of shape
// `out_shape`, reading inputs of shapes `input_shapes` with NumPy broadcasting:
//
//   extern "C" __global__ void mc_elementwise(float* out, const float* in0, ..., unsigned n)
//
// Shapes are known when the graph is compiled, so broadcast index math is
// emitted with constant divisors, scalar inputs and immediates are hoisted out
// of the loop, and the kernel uses float4 accesses when every input is either
// the output's layout, a scalar, or a suffix of the output's shape whose size
// is a multiple of 4 (a bias row, say).
ElementwiseKernel generate_elementwise_kernel(const FusedProgram& program, const std::vector<Shape>& input_shapes,
                                              const Shape& out_shape, bool allow_vectorize = true);

// A single unary or binary op as a program: load the operands, apply the op.
// Unfused graphs compile every elementwise node through the same generator,
// so a fused-vs-unfused comparison differs only in fusion.
FusedProgram single_op_program(OpKind op);

}
