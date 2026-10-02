#include "minicompiler/passes/operator_fusion.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>

namespace minicompiler {

std::vector<std::vector<NodeId>> find_fusion_groups(const Graph& graph) {
	const std::size_t n = graph.num_nodes();
	const std::vector<std::vector<NodeId>> users = graph.compute_users();
	std::vector<std::int32_t> group_of(n, -1);
	std::vector<std::vector<NodeId>> groups;

	for (std::size_t i=n; i-- > 0;) {
		const NodeId root = static_cast<NodeId>(i);
		const Node& root_node = graph.node(root);
		if (!is_elementwise(root_node.op) || group_of[i] != -1) continue;

		const auto gid = static_cast<std::int32_t>(groups.size());
		group_of[i] = gid;
		std::vector<NodeId> members = {root};
		std::priority_queue<NodeId> candidates;  // highest id first
		for (NodeId in : root_node.inputs) candidates.push(in);

		while (!candidates.empty()) {
			const NodeId p = candidates.top();
			candidates.pop();
			if (group_of[p] != -1) continue;  // already in the group (pushed twice)
			const Node& pn = graph.node(p);
			if (!is_elementwise(pn.op)) continue;
			if (pn.type.shape != root_node.type.shape) continue;
			if (graph.is_output(p)) continue;
			const bool all_users_inside = std::all_of(users[p].begin(), users[p].end(),
			                                          [&](NodeId u) { return group_of[u] == gid; });
			if (!all_users_inside) continue;
			group_of[p] = gid;
			members.push_back(p);
			for (NodeId in : pn.inputs) candidates.push(in);
		}
		std::sort(members.begin(), members.end());
		groups.push_back(std::move(members));
	}

	std::vector<std::vector<NodeId>> fusable;
	for (auto it=groups.rbegin(); it!=groups.rend(); ++it) {  // report in topological order of roots
		if (it->size() >= 2) fusable.push_back(std::move(*it));
	}
	return fusable;
}

namespace {

// Translates one group into a FusedProgram over its external operands.
class ProgramBuilder {
public:
	ProgramBuilder(const Graph& graph, const std::vector<NodeId>& remap) : graph_(graph), remap_(remap) {}

	void add_member(NodeId id) {
		const Node& node = graph_.node(id);
		const std::int32_t a = operand(node.inputs[0]);
		if (is_unary_elementwise(node.op)) {
			emit(id, FusedInstr::unary(node.op, a));
		} else {
			const std::int32_t b = operand(node.inputs[1]);  // after `a`: keep instruction order deterministic
			emit(id, FusedInstr::binary(node.op, a, b));
		}
		program_->source_nodes.push_back(node.name);
	}

	std::shared_ptr<FusedProgram> program() { return program_; }
	const std::vector<NodeId>& external_inputs() const { return external_inputs_; }

private:
	// The register holding `id`: a member computed earlier, or a load or
	// immediate created the first time an external operand is used.
	std::int32_t operand(NodeId id) {
		auto it = reg_of_.find(id);
		if (it != reg_of_.end()) return it->second;
		const Node& node = graph_.node(id);
		if (node.op == OpKind::Constant && node.type.num_elements() == 1) {
			return emit(id, FusedInstr::imm(node.constant->at(0)));
		}
		external_inputs_.push_back(remap_[id]);
		return emit(id, FusedInstr::load(static_cast<std::int32_t>(external_inputs_.size() - 1)));
	}

	std::int32_t emit(NodeId id, FusedInstr ins) {
		program_->instrs.push_back(ins);
		const auto reg = static_cast<std::int32_t>(program_->instrs.size() - 1);
		reg_of_[id] = reg;
		return reg;
	}

	const Graph& graph_;
	const std::vector<NodeId>& remap_;
	std::shared_ptr<FusedProgram> program_ = std::make_shared<FusedProgram>();
	std::vector<NodeId> external_inputs_;  // ids in the new graph
	std::map<NodeId, std::int32_t> reg_of_;
};

}

Result<Graph> fuse_elementwise(const Graph& graph, PassReport* report) {
	const std::vector<std::vector<NodeId>> groups = find_fusion_groups(graph);
	std::vector<std::int32_t> group_of(graph.num_nodes(), -1);
	for (std::size_t g=0; g<groups.size(); ++g) {
		for (NodeId m : groups[g]) group_of[m] = static_cast<std::int32_t>(g);
	}

	Graph result(graph.name());
	std::vector<NodeId> remap(graph.num_nodes(), kNoNode);
	std::size_t fused_ops = 0;
	for (std::size_t i=0; i<graph.num_nodes(); ++i) {
		const NodeId id = static_cast<NodeId>(i);
		const Node& node = graph.node(id);
		if (group_of[i] < 0) {
			std::vector<NodeId> inputs;
			for (NodeId in : node.inputs) inputs.push_back(remap[in]);
			Result<NodeId> copied = result.copy_node(node, std::move(inputs));
			if (!copied.ok()) return copied.error();
			remap[i] = copied.value();
			continue;
		}
		const std::vector<NodeId>& members = groups[static_cast<std::size_t>(group_of[i])];
		if (id != members.back()) continue;  // emitted together with the group's root

		ProgramBuilder builder(graph, remap);
		for (NodeId m : members) builder.add_member(m);
		Result<NodeId> fused = result.add_fused(builder.external_inputs(), builder.program(), node.type, node.name);
		if (!fused.ok()) return fused.error();
		remap[i] = fused.value();
		fused_ops += members.size();
	}
	for (NodeId out : graph.outputs()) {
		Status st = result.add_output(remap[out]);
		if (!st.ok()) return st.error();
	}

	fill_report(report, "fuse", graph, result,
	            "fused " + std::to_string(fused_ops) + " op(s) into " + std::to_string(groups.size()) + " kernel(s)");
	return result;
}

}
