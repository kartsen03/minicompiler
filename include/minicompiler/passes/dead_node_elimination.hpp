#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/passes/pass.hpp"
#include "minicompiler/result.hpp"

namespace minicompiler {

// Removes every node that no graph output depends on. Graph inputs are kept
// even when unused, since they are part of the graph's calling convention.
//
// Mark and sweep: mark the outputs, then walk the nodes from last to first
// and mark the operands of every marked node. A node's users always come
// after it, so one backward walk reaches everything the outputs depend on.
// The unmarked nodes are dropped by copying the marked ones into a new graph.
Result<Graph> eliminate_dead_nodes(const Graph& graph, PassReport* report = nullptr);

}
