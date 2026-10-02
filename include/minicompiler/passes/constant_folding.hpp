#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/passes/pass.hpp"
#include "minicompiler/result.hpp"

#include <cstddef>

namespace minicompiler {

struct ConstantFoldingOptions {
	// Results with more elements than this are left for runtime, so folding
	// cannot blow up the size of the compiled graph.
	std::size_t max_elements = std::size_t{1} << 22;
};

// Replaces every compute node whose operands are all constants with a
// Constant holding its value. Nodes are visited in topological order, so a
// whole constant subexpression folds in one pass (sqrt(2 / pi) becomes one
// constant). Values are computed by the CPU backend's own kernels, so a
// folded constant matches what the node computes at runtime to within an ulp
// or two (Eigen runs the elements before the first aligned address through
// scalar code and the rest through SIMD approximations, so which elements
// take which path depends on the buffer's address). The folded node's
// original operands are left in place; dead-node elimination removes the
// ones nothing else uses.
Result<Graph> fold_constants(const Graph& graph, PassReport* report = nullptr,
                             const ConstantFoldingOptions& options = {});

}
