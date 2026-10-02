#pragma once

#include "minicompiler/fused_program.hpp"
#include "minicompiler/ops.hpp"
#include "minicompiler/result.hpp"
#include "minicompiler/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace minicompiler {

using NodeId = std::int32_t;
inline constexpr NodeId kNoNode = -1;

// One operation in the graph. Every node defines exactly one tensor value,
// so "node" and "value" are used interchangeably.
struct Node {
	OpKind op = OpKind::Input;
	std::string name;
	std::vector<NodeId> inputs;  // operands, by node id
	TensorType type;             // type of the value this node defines

	// OpKind::Constant: the values, row-major. Shared, so copying a graph
	// (every pass builds a new one) never copies weight data.
	std::shared_ptr<const std::vector<float>> constant;

	// OpKind::FusedElementwise: what the fused kernel computes.
	std::shared_ptr<const FusedProgram> fused;
};

// A computation graph in SSA form: a DAG whose nodes are ops and whose edges
// run from each operand to the node that uses it.
//
// Nodes live in a vector and refer to their operands by index. Operands must
// already exist when a node is added, so the vector order is always a valid
// topological order and a cycle cannot be built. Passes never mutate a graph
// in place; each one builds a new Graph, which keeps that invariant, and
// verify() re-checks it after every pass.
class Graph {
public:
	explicit Graph(std::string name = "graph") : name_(std::move(name)) {}

	const std::string& name() const { return name_; }

	// An empty `name` gets a generated one, e.g. "mul_7".
	Result<NodeId> add_input(std::string name, TensorType type);
	Result<NodeId> add_constant(std::string name, TensorType type, std::vector<float> values);
	Result<NodeId> add_constant(std::string name, TensorType type,
	                            std::shared_ptr<const std::vector<float>> values);
	Result<NodeId> add_op(OpKind op, std::vector<NodeId> inputs, std::string name = {});
	Result<NodeId> add_fused(std::vector<NodeId> inputs, std::shared_ptr<const FusedProgram> program,
	                         TensorType type, std::string name = {});
	Status add_output(NodeId id);

	// Adds a node of the same kind, name and payload as `prototype` with new
	// operands. Passes use it to carry nodes over into the graph they build.
	Result<NodeId> copy_node(const Node& prototype, std::vector<NodeId> inputs);

	std::size_t num_nodes() const { return nodes_.size(); }
	const Node& node(NodeId id) const { return nodes_[static_cast<std::size_t>(id)]; }
	const std::vector<Node>& nodes() const { return nodes_; }
	const std::vector<NodeId>& inputs() const { return inputs_; }
	const std::vector<NodeId>& outputs() const { return outputs_; }
	bool is_output(NodeId id) const;

	// First node with this name.
	std::optional<NodeId> find(std::string_view name) const;

	// users[v] lists the nodes that read value v (each user once, ascending).
	std::vector<std::vector<NodeId>> compute_users() const;

	// Checks every structural invariant: operands precede their users (hence
	// acyclic), arities, inferred types, constant sizes, fused programs and
	// the input/output lists.
	Status verify() const;

private:
	Status check_operands(const std::vector<NodeId>& inputs) const;
	NodeId append(Node node);

	std::string name_;
	std::vector<Node> nodes_;
	std::vector<NodeId> inputs_;
	std::vector<NodeId> outputs_;
};

// Node and op counts, used by the pass reports and the benchmarks.
struct GraphStats {
	std::size_t nodes = 0;
	std::size_t inputs = 0;
	std::size_t constants = 0;
	// Nodes that run at runtime. Each is one step on the CPU backend and one
	// kernel launch on the CUDA backend.
	std::size_t compute_nodes = 0;
	std::size_t fused_nodes = 0;
	// Primitive elementwise ops executed, including those inside fused nodes.
	std::size_t elementwise_ops = 0;
	std::size_t matmuls = 0;
	std::map<std::string, std::size_t> op_counts;  // compute nodes by op name
};

GraphStats compute_stats(const Graph& graph);

}
