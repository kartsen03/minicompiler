#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/passes/pass.hpp"
#include "minicompiler/result.hpp"

#include <vector>

namespace minicompiler {

// Finds the groups of elementwise nodes that can run as one kernel.
//
// Groups grow from a root (visited last-to-first) toward its producers. A
// producer p joins the root's group when:
//   - p is a unary or binary elementwise op,
//   - p has the root's shape (a smaller, broadcast producer would be
//     recomputed for every output element),
//   - p is not a graph output (outputs must be written to memory), and
//   - every user of p is already in the group (otherwise p's value is needed
//     outside the kernel and must be written to memory anyway).
// The last rule also keeps groups convex: no path leaves the group and comes
// back, so replacing a group by one node cannot create a cycle. Candidates are
// examined from the highest id down; since users follow their producers,
// every user that could join a group has been decided before the producer is
// examined.
//
// Returns every group with two or more nodes, nodes in topological order (the
// root last).
std::vector<std::vector<NodeId>> find_fusion_groups(const Graph& graph);

// Replaces each group from find_fusion_groups() with one FusedElementwise
// node placed at the root's position and named after it. The node's program
// loads each distinct external operand once and bakes scalar constants in as
// immediates; constants that become unused are left for dead-node
// elimination.
Result<Graph> fuse_elementwise(const Graph& graph, PassReport* report = nullptr);

}
