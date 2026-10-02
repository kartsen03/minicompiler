#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/result.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace minicompiler {

// The .mcg text format: one statement per line, `#` starts a comment.
//
//   graph mlp                                    name of the graph (optional)
//   dim B = 64                                   named dimension, overridable
//   input x : f32[B, 256]                        runtime argument
//   const w : f32[256, 512] = uniform(seed=1, lo=-0.05, hi=0.05)
//   const b : f32[512] = 0.1                     every element 0.1
//   const v : f32[3] = [1, 2, 3]                 explicit values, row-major
//   h = matmul x, w                              name = op operand, operand
//   y = relu h
//   output y                                     one or more, comma-separated
//
// Ops: neg exp log sqrt relu sigmoid tanh add sub mul div matmul.
struct ParseOptions {
	// Values that replace the defaults of `dim` declarations, e.g. {"B", 1}.
	std::map<std::string, std::int64_t> dims;
};

Result<Graph> parse_graph(std::string_view text, const ParseOptions& options = {});
Result<Graph> parse_graph_file(const std::string& path, const ParseOptions& options = {});

// Human-readable listing, one node per line (used by `mcc --print`).
std::string print_graph(const Graph& graph);

// Short description of a constant's value: "0.5", "[1, 2, 3]" or "<4096 values>".
std::string describe_constant(const Node& constant);

}
