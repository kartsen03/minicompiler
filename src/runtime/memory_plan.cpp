#include "minicompiler/runtime/memory_plan.hpp"

#include <algorithm>
#include <numeric>

namespace minicompiler {

std::size_t MemoryPlan::bytes_with_reuse() const {
	return std::accumulate(buffer_bytes.begin(), buffer_bytes.end(), std::size_t{0});
}

MemoryPlan plan_memory(const Graph& graph, const PlanOptions& options) {
	const std::size_t n = graph.num_nodes();
	MemoryPlan plan;
	plan.buffer_of.assign(n, MemoryPlan::kNotPooled);
	plan.last_use.assign(n, kNoNode);

	for (std::size_t i=0; i<n; ++i) {
		for (NodeId in : graph.node(static_cast<NodeId>(i)).inputs) plan.last_use[in] = static_cast<NodeId>(i);
	}
	for (NodeId out : graph.outputs()) plan.last_use[out] = static_cast<NodeId>(n);

	std::vector<std::int32_t> free_buffers;
	for (std::size_t i=0; i<n; ++i) {
		const NodeId id = static_cast<NodeId>(i);
		const Node& node = graph.node(id);
		if (!is_compute(node.op)) continue;

		const bool pooled = options.pool_outputs || !graph.is_output(id);
		if (pooled) {
			const std::size_t bytes = node.type.size_bytes();
			plan.bytes_without_reuse += bytes;

			// Best fit among the free buffers; otherwise grow the largest free
			// one; otherwise open a new buffer.
			auto best = free_buffers.end();
			for (auto it = free_buffers.begin(); it != free_buffers.end(); ++it) {
				const std::size_t size = plan.buffer_bytes[*it];
				if (size >= bytes && (best == free_buffers.end() || size < plan.buffer_bytes[*best])) best = it;
			}
			if (best == free_buffers.end() && !free_buffers.empty()) {
				best = std::max_element(free_buffers.begin(), free_buffers.end(), [&](std::int32_t a, std::int32_t b) {
					return plan.buffer_bytes[a] < plan.buffer_bytes[b];
				});
				plan.buffer_bytes[*best] = bytes;
			}
			if (best != free_buffers.end()) {
				plan.buffer_of[id] = *best;
				free_buffers.erase(best);
			} else {
				plan.buffer_of[id] = static_cast<std::int32_t>(plan.buffer_bytes.size());
				plan.buffer_bytes.push_back(bytes);
			}
		}

		// Only now free the operands whose last use is this node. This happens
		// for unpooled outputs too: their operands are still pooled values.
		for (std::size_t k=0; k<node.inputs.size(); ++k) {
			const NodeId in = node.inputs[k];
			const bool repeated = std::find(node.inputs.begin(), node.inputs.begin() + static_cast<std::ptrdiff_t>(k), in) !=
			                      node.inputs.begin() + static_cast<std::ptrdiff_t>(k);
			if (!repeated && plan.buffer_of[in] != MemoryPlan::kNotPooled && plan.last_use[in] == id) {
				free_buffers.push_back(plan.buffer_of[in]);
			}
		}
		// Nothing reads this value: its buffer is free again right away.
		if (pooled && plan.last_use[id] == kNoNode) free_buffers.push_back(plan.buffer_of[id]);
	}
	return plan;
}

}
