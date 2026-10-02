#include "minicompiler/passes/pipeline.hpp"

#include "minicompiler/passes/constant_folding.hpp"
#include "minicompiler/passes/dead_node_elimination.hpp"
#include "minicompiler/passes/operator_fusion.hpp"

namespace minicompiler {

const std::vector<PassInfo>& registered_passes() {
	static const std::vector<PassInfo> passes = {
		{"dne", "dead-node elimination: remove nodes no output depends on", &eliminate_dead_nodes},
		{"fold", "constant folding: evaluate nodes whose operands are all constants",
		 [](const Graph& g, PassReport* r) { return fold_constants(g, r); }},
		{"fuse", "operator fusion: merge groups of elementwise ops into single kernels", &fuse_elementwise},
	};
	return passes;
}

const std::vector<std::string>& default_pipeline() {
	static const std::vector<std::string> pipeline = {"dne", "fold", "dne", "fuse", "dne"};
	return pipeline;
}

namespace {

const PassInfo* find_pass(std::string_view name) {
	for (const PassInfo& p : registered_passes()) {
		if (name == p.name) return &p;
	}
	return nullptr;
}

}

Result<std::vector<std::string>> parse_pipeline(std::string_view spec) {
	if (spec == "default") return default_pipeline();
	if (spec == "none" || spec.empty()) return std::vector<std::string>{};
	std::vector<std::string> passes;
	std::size_t start = 0;
	while (start <= spec.size()) {
		std::size_t end = spec.find(',', start);
		if (end == std::string_view::npos) end = spec.size();
		const std::string_view name = spec.substr(start, end - start);
		if (!find_pass(name)) {
			std::string known;
			for (const PassInfo& p : registered_passes()) known += std::string(known.empty() ? "" : ", ") + p.name;
			return Error{"unknown pass '" + std::string(name) + "' (passes: " + known + ")"};
		}
		passes.emplace_back(name);
		start = end + 1;
	}
	return passes;
}

Result<Graph> run_pipeline(const Graph& graph, const std::vector<std::string>& passes,
                           std::vector<PassReport>* reports, const PassObserver& observer) {
	Status st = graph.verify();
	if (!st.ok()) return Error{"input graph is invalid: " + st.message()};
	Graph current = graph;
	if (observer) observer(0, "input", current);
	for (std::size_t i=0; i<passes.size(); ++i) {
		const PassInfo* pass = find_pass(passes[i]);
		if (!pass) return Error{"unknown pass '" + passes[i] + "'"};
		PassReport report;
		Result<Graph> next = pass->run(current, &report);
		if (!next.ok()) return Error{"pass '" + passes[i] + "' failed: " + next.error().message};
		st = next->verify();
		if (!st.ok()) return Error{"pass '" + passes[i] + "' produced an invalid graph: " + st.message()};
		current = std::move(next).value();
		if (reports) reports->push_back(std::move(report));
		if (observer) observer(i + 1, passes[i], current);
	}
	return current;
}

}
