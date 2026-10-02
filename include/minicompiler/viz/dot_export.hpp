#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/result.hpp"

#include <string>

namespace minicompiler {

struct DotOptions {
	// Caption drawn above the graph. Defaults to the graph name.
	std::string title;
	// List the instructions of each fused kernel inside its node.
	bool show_fused_bodies = true;
};

// Renders the graph in Graphviz DOT. Inputs are ellipses, constants grey,
// elementwise ops yellow, matmuls orange, fused kernels green; graph outputs
// get a double border. Render with e.g. `dot -Tsvg graph.dot -o graph.svg`.
std::string to_dot(const Graph& graph, const DotOptions& options = {});

Status write_dot(const Graph& graph, const std::string& path, const DotOptions& options = {});

}
