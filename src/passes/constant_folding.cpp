#include "minicompiler/passes/constant_folding.hpp"

#include "cpu/kernels.hpp"

#include <algorithm>
#include <vector>

namespace minicompiler {

Result<Graph> fold_constants(const Graph& graph, PassReport* report, const ConstantFoldingOptions& options) {
	Graph result(graph.name());
	std::vector<NodeId> remap(graph.num_nodes(), kNoNode);
	std::vector<float> scratch;
	std::size_t folded = 0;

	for (std::size_t i=0; i<graph.num_nodes(); ++i) {
		const Node& node = graph.node(static_cast<NodeId>(i));
		std::vector<NodeId> inputs;
		for (NodeId in : node.inputs) inputs.push_back(remap[in]);

		const bool all_constant = !inputs.empty() && std::all_of(inputs.begin(), inputs.end(), [&](NodeId in) {
			return result.node(in).op == OpKind::Constant;
		});
		if (is_compute(node.op) && all_constant && node.type.num_elements() <= options.max_elements) {
			std::vector<const float*> operands;
			std::vector<Shape> shapes;
			for (NodeId in : inputs) {
				operands.push_back(result.node(in).constant->data());
				shapes.push_back(result.node(in).type.shape);
			}
			std::vector<float> values(node.type.num_elements());
			cpu::execute_node(node, operands, shapes, values.data(), scratch);
			Result<NodeId> id = result.add_constant(node.name, node.type, std::move(values));
			if (!id.ok()) return id.error();
			remap[i] = id.value();
			++folded;
			continue;
		}

		Result<NodeId> id = result.copy_node(node, std::move(inputs));
		if (!id.ok()) return id.error();
		remap[i] = id.value();
	}
	for (NodeId out : graph.outputs()) {
		Status st = result.add_output(remap[out]);
		if (!st.ok()) return st.error();
	}

	fill_report(report, "fold", graph, result, "folded " + std::to_string(folded) + " node(s) into constants");
	return result;
}

}
