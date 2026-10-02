#pragma once

// Internal: the Eigen implementations behind the CPU backend. Constant
// folding calls them too, so a folded constant is computed by exactly the
// code that would otherwise have computed it at runtime.

#include "minicompiler/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minicompiler::cpu {

// Elements per block when evaluating a fused program. With one buffer per
// instruction, a block of 512 floats (2 KB) keeps a typical fused kernel's
// working set inside a 48 KB L1 data cache.
inline constexpr std::size_t kFusedBlock = 512;

void unary(OpKind op, const float* x, float* y, std::size_t n);

void binary(OpKind op, const float* a, const Shape& a_shape, const float* b, const Shape& b_shape, float* out,
            const Shape& out_shape);

// Row-major C[m,n] = A[m,k] * B[k,n] (Eigen GEMM, single-threaded).
void matmul(const float* a, const float* b, float* c, std::int64_t m, std::int64_t k, std::int64_t n);

// Evaluates a fused program kFusedBlock elements at a time: every instruction
// runs over one block before the next block starts, so intermediates stay in
// cache-resident block buffers instead of making a full pass over memory per
// op. `scratch` holds those buffers and is reused across calls.
void fused(const FusedProgram& program, const std::vector<const float*>& inputs,
           const std::vector<Shape>& input_shapes, float* out, const Shape& out_shape, std::vector<float>& scratch);

// Runs one compute node on host buffers, operands in node order.
void execute_node(const Node& node, const std::vector<const float*>& operands,
                  const std::vector<Shape>& operand_shapes, float* out, std::vector<float>& scratch);

}
