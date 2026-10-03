#include "cpu/cpu_backend.hpp"

#include "cpu/kernels.hpp"
#include "minicompiler/runtime/memory_plan.hpp"

#include <algorithm>

namespace minicompiler {
namespace {

// Executes the graph node by node with Eigen. Intermediates live in buffers
// assigned by the memory planner; inputs are read in place from the caller's
// tensors, constants from the graph, and graph outputs are written straight
// into the caller's output tensors (no copy at the end of a run).
class CpuExecutable final : public Executable {
public:
	explicit CpuExecutable(Graph graph) : graph_(std::move(graph)), plan_(plan_memory(graph_, {false})) {
		for (std::size_t bytes : plan_.buffer_bytes) buffers_.emplace_back(bytes / sizeof(float));
		values_.assign(graph_.num_nodes(), nullptr);
		operand_shapes_.resize(graph_.num_nodes());
		for (std::size_t i=0; i<graph_.num_nodes(); ++i) {
			for (NodeId in : graph_.node(static_cast<NodeId>(i)).inputs) {
				operand_shapes_[i].push_back(graph_.node(in).type.shape);
			}
		}
	}

	Status run(const std::vector<HostTensor>& inputs, std::vector<HostTensor>& outputs) override {
		// Outputs are written while inputs are still being read, so the two
		// must not share storage (distinct vectors never do).
		if (&inputs == &outputs) return Error{"run() needs separate input and output vectors"};
		if (inputs.size() != graph_.inputs().size()) {
			return Error{"expected " + std::to_string(graph_.inputs().size()) + " inputs, got " +
			             std::to_string(inputs.size())};
		}
		for (std::size_t k=0; k<inputs.size(); ++k) {
			const Node& node = graph_.node(graph_.inputs()[k]);
			if (inputs[k].type != node.type || inputs[k].data.size() != node.type.num_elements()) {
				return Error{"input '" + node.name + "' should be " + to_string(node.type) + ", got " +
				             to_string(inputs[k].type)};
			}
			values_[graph_.inputs()[k]] = inputs[k].data.data();
		}

		// Outputs computed by a node are written in place; resizing to the
		// same size keeps the caller's storage, so repeated runs allocate nothing.
		outputs.resize(graph_.outputs().size());
		destinations_.assign(graph_.num_nodes(), nullptr);
		for (std::size_t k=0; k<graph_.outputs().size(); ++k) {
			const Node& node = graph_.node(graph_.outputs()[k]);
			outputs[k].type = node.type;
			outputs[k].data.resize(node.type.num_elements());
			if (is_compute(node.op)) destinations_[graph_.outputs()[k]] = outputs[k].data.data();
		}

		for (std::size_t i=0; i<graph_.num_nodes(); ++i) {
			const Node& node = graph_.node(static_cast<NodeId>(i));
			if (node.op == OpKind::Input) continue;
			if (node.op == OpKind::Constant) {
				values_[i] = node.constant->data();
				continue;
			}
			float* out = destinations_[i] ? destinations_[i] : buffers_[static_cast<std::size_t>(plan_.buffer_of[i])].data();
			operands_.clear();
			for (NodeId in : node.inputs) operands_.push_back(values_[in]);
			cpu::execute_node(node, operands_, operand_shapes_[i], out, scratch_);
			values_[i] = out;
		}

		// An output that is a graph input or a constant is still copied.
		for (std::size_t k=0; k<graph_.outputs().size(); ++k) {
			const NodeId id = graph_.outputs()[k];
			if (!is_compute(graph_.node(id).op)) {
				std::copy(values_[id], values_[id] + outputs[k].data.size(), outputs[k].data.begin());
			}
		}
		return Status();
	}

	std::size_t intermediate_bytes() const override { return plan_.bytes_with_reuse(); }

private:
	Graph graph_;
	MemoryPlan plan_;
	std::vector<std::vector<float>> buffers_;
	std::vector<const float*> values_;               // where each node's value lives during a run
	std::vector<float*> destinations_;               // per node: caller memory for graph outputs
	std::vector<std::vector<Shape>> operand_shapes_;  // per node
	std::vector<const float*> operands_;             // reused per node
	std::vector<float> scratch_;                     // fused-kernel block buffers
};

class CpuBackend final : public Backend {
public:
	const char* name() const override { return "cpu"; }

	Result<std::unique_ptr<Executable>> compile(const Graph& graph) override {
		Status st = graph.verify();
		if (!st.ok()) return st.error();
		return std::unique_ptr<Executable>(std::make_unique<CpuExecutable>(graph));
	}
};

}

std::unique_ptr<Backend> make_cpu_backend() {
	return std::make_unique<CpuBackend>();
}

}
