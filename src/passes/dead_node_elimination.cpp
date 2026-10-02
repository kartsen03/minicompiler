#include "minicompiler/passes/dead_node_elimination.hpp"

#include <vector>

namespace minicompiler {

Result<Graph> eliminate_dead_nodes(const Graph& graph, PassReport* report) {
	const std::size_t n = graph.num_nodes();

	// Mark.
	std::vector<bool> live(n, false);
	for (NodeId out : graph.outputs()) live[out] = true;
	for (std::size_t i=n; i-- > 0;) {
		if (!live[i]) continue;
		for (NodeId in : graph.node(static_cast<NodeId>(i)).inputs) live[in] = true;
	}
	for (NodeId in : graph.inputs()) live[in] = true;

	// Sweep: copy the live nodes into a new graph, renumbering operands.
	Graph result(graph.name());
	std::vector<NodeId> remap(n, kNoNode);
	std::size_t removed = 0;
	for (std::size_t i=0; i<n; ++i) {
		if (!live[i]) {
			++removed;
			continue;
		}
		const Node& node = graph.node(static_cast<NodeId>(i));
		std::vector<NodeId> inputs;
		for (NodeId in : node.inputs) inputs.push_back(remap[in]);
		Result<NodeId> id = result.copy_node(node, std::move(inputs));
		if (!id.ok()) return id.error();
		remap[i] = id.value();
	}
	for (NodeId out : graph.outputs()) {
		Status st = result.add_output(remap[out]);
		if (!st.ok()) return st.error();
	}

	fill_report(report, "dne", graph, result, "removed " + std::to_string(removed) + " dead node(s)");
	return result;
}

}
