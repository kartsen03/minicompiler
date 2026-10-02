#pragma once

#include "minicompiler/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minicompiler {

// Assigns the value of every compute node to a reusable buffer.
//
// Nodes execute in id order, so a node id doubles as a time step. A value is
// live from the step that defines it to the last step that reads it; graph
// outputs stay live to the end. Values whose live ranges do not overlap can
// share a buffer. The planner walks the nodes in order, gives each new value
// a free buffer (best fit, growing one if none is big enough, a new buffer
// if none is free), and frees a buffer after its value's last use. A node's
// output is placed before its operands are freed, so no op ever writes into
// a buffer it is still reading.
//
// Graph inputs and constants are not pooled: inputs come from the caller
// (the CUDA backend copies them in once per run) and constants are uploaded
// once at compile time.
struct PlanOptions {
	// Give graph outputs pooled buffers. The CPU backend turns this off and
	// writes outputs straight into the caller's tensors; the CUDA backend
	// needs device buffers for them before copying back to the host.
	bool pool_outputs = true;
};

struct MemoryPlan {
	static constexpr std::int32_t kNotPooled = -1;

	std::vector<std::int32_t> buffer_of;    // per node: buffer index, or kNotPooled
	std::vector<std::size_t> buffer_bytes;  // size of each buffer
	// Per node: the last step that reads it, num_nodes() for graph outputs,
	// kNoNode if nothing reads it.
	std::vector<NodeId> last_use;

	std::size_t bytes_without_reuse = 0;  // what one buffer per pooled value would need
	std::size_t bytes_with_reuse() const;
};

MemoryPlan plan_memory(const Graph& graph, const PlanOptions& options = {});

}
