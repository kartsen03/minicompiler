#include "minicompiler/graph.hpp"

#include <algorithm>

namespace minicompiler {
namespace {

Status check_type(const TensorType& type, const std::string& what) {
	if (type.dtype != DType::Float32) {
		return Error{what + ": only f32 is supported, got " + to_string(type)};
	}
	for (std::int64_t d : type.shape) {
		if (d < 1) return Error{what + ": dimensions must be >= 1, got " + to_string(type)};
	}
	return Status();
}

// Each operand must broadcast to the fused node's output shape.
Status check_fused(const FusedProgram& program, const std::vector<TensorType>& operand_types,
                   const TensorType& type, const std::string& what) {
	Status st = validate(program, operand_types.size());
	if (!st.ok()) return Error{what + ": " + st.message()};
	for (const TensorType& t : operand_types) {
		Result<Shape> s = broadcast_shapes(t.shape, type.shape);
		if (!s.ok() || s.value() != type.shape) {
			return Error{what + ": input " + to_string(t) + " does not broadcast to " + to_string(type)};
		}
	}
	return Status();
}

}

NodeId Graph::append(Node node) {
	const NodeId id = static_cast<NodeId>(nodes_.size());
	if (node.name.empty()) node.name = std::string(op_name(node.op)) + "_" + std::to_string(id);
	nodes_.push_back(std::move(node));
	return id;
}

Status Graph::check_operands(const std::vector<NodeId>& inputs) const {
	for (NodeId in : inputs) {
		if (in < 0 || static_cast<std::size_t>(in) >= nodes_.size()) {
			return Error{"operand %" + std::to_string(in) + " does not exist"};
		}
	}
	return Status();
}

Result<NodeId> Graph::add_input(std::string name, TensorType type) {
	Status st = check_type(type, "input '" + name + "'");
	if (!st.ok()) return st.error();
	Node n;
	n.op = OpKind::Input;
	n.name = std::move(name);
	n.type = std::move(type);
	const NodeId id = append(std::move(n));
	inputs_.push_back(id);
	return id;
}

Result<NodeId> Graph::add_constant(std::string name, TensorType type, std::vector<float> values) {
	return add_constant(std::move(name), std::move(type),
	                    std::make_shared<const std::vector<float>>(std::move(values)));
}

Result<NodeId> Graph::add_constant(std::string name, TensorType type,
                                   std::shared_ptr<const std::vector<float>> values) {
	const std::string what = "constant '" + name + "'";
	Status st = check_type(type, what);
	if (!st.ok()) return st.error();
	if (!values || values->size() != type.num_elements()) {
		return Error{what + ": expected " + std::to_string(type.num_elements()) + " values for " +
		             to_string(type) + ", got " + std::to_string(values ? values->size() : 0)};
	}
	Node n;
	n.op = OpKind::Constant;
	n.name = std::move(name);
	n.type = std::move(type);
	n.constant = std::move(values);
	return append(std::move(n));
}

Result<NodeId> Graph::add_op(OpKind op, std::vector<NodeId> inputs, std::string name) {
	if (!is_elementwise(op) && op != OpKind::MatMul) {
		return Error{std::string("add_op cannot create '") + op_name(op) + "' nodes"};
	}
	Status st = check_operands(inputs);
	if (!st.ok()) return st.error();
	std::vector<TensorType> operand_types;
	for (NodeId in : inputs) operand_types.push_back(node(in).type);
	Result<TensorType> type = infer_result_type(op, operand_types);
	if (!type.ok()) {
		return Error{(name.empty() ? std::string(op_name(op)) : name) + ": " + type.error().message};
	}
	Node n;
	n.op = op;
	n.name = std::move(name);
	n.inputs = std::move(inputs);
	n.type = std::move(type).value();
	return append(std::move(n));
}

Result<NodeId> Graph::add_fused(std::vector<NodeId> inputs, std::shared_ptr<const FusedProgram> program,
                                TensorType type, std::string name) {
	const std::string what = "fused node '" + name + "'";
	if (!program) return Error{what + ": missing program"};
	Status st = check_operands(inputs);
	if (!st.ok()) return st.error();
	st = check_type(type, what);
	if (!st.ok()) return st.error();
	std::vector<TensorType> operand_types;
	for (NodeId in : inputs) operand_types.push_back(node(in).type);
	st = check_fused(*program, operand_types, type, what);
	if (!st.ok()) return st.error();
	Node n;
	n.op = OpKind::FusedElementwise;
	n.name = std::move(name);
	n.inputs = std::move(inputs);
	n.type = std::move(type);
	n.fused = std::move(program);
	return append(std::move(n));
}

Status Graph::add_output(NodeId id) {
	Status st = check_operands({id});
	if (!st.ok()) return st;
	if (is_output(id)) return Error{"'" + node(id).name + "' is already an output"};
	outputs_.push_back(id);
	return Status();
}

Result<NodeId> Graph::copy_node(const Node& prototype, std::vector<NodeId> inputs) {
	switch (prototype.op) {
		case OpKind::Input:
			return add_input(prototype.name, prototype.type);
		case OpKind::Constant:
			return add_constant(prototype.name, prototype.type, prototype.constant);
		case OpKind::FusedElementwise:
			return add_fused(std::move(inputs), prototype.fused, prototype.type, prototype.name);
		default:
			return add_op(prototype.op, std::move(inputs), prototype.name);
	}
}

bool Graph::is_output(NodeId id) const {
	return std::find(outputs_.begin(), outputs_.end(), id) != outputs_.end();
}

std::optional<NodeId> Graph::find(std::string_view name) const {
	for (std::size_t i=0; i<nodes_.size(); ++i) {
		if (nodes_[i].name == name) return static_cast<NodeId>(i);
	}
	return std::nullopt;
}

std::vector<std::vector<NodeId>> Graph::compute_users() const {
	std::vector<std::vector<NodeId>> users(nodes_.size());
	for (std::size_t i=0; i<nodes_.size(); ++i) {
		const NodeId user = static_cast<NodeId>(i);
		for (NodeId in : nodes_[i].inputs) {
			// x*x reads x twice but is still one user.
			if (users[in].empty() || users[in].back() != user) users[in].push_back(user);
		}
	}
	return users;
}

Status Graph::verify() const {
	std::size_t input_nodes = 0;
	for (std::size_t i=0; i<nodes_.size(); ++i) {
		const Node& n = nodes_[i];
		const std::string what = "node %" + std::to_string(i) + " '" + n.name + "'";
		for (NodeId in : n.inputs) {
			// Operands must come strictly earlier: this is what makes the
			// node order topological and rules out cycles.
			if (in < 0 || static_cast<std::size_t>(in) >= i) {
				return Error{what + ": operand %" + std::to_string(in) + " does not precede it"};
			}
		}
		const int arity = op_arity(n.op);
		if (arity >= 0 && n.inputs.size() != static_cast<std::size_t>(arity)) {
			return Error{what + ": " + op_name(n.op) + " has " + std::to_string(n.inputs.size()) + " operands"};
		}
		std::vector<TensorType> operand_types;
		for (NodeId in : n.inputs) operand_types.push_back(node(in).type);

		switch (n.op) {
			case OpKind::Input: {
				++input_nodes;
				if (std::find(inputs_.begin(), inputs_.end(), static_cast<NodeId>(i)) == inputs_.end()) {
					return Error{what + ": input node missing from the input list"};
				}
				Status st = check_type(n.type, what);
				if (!st.ok()) return st;
				break;
			}
			case OpKind::Constant: {
				Status st = check_type(n.type, what);
				if (!st.ok()) return st;
				if (!n.constant || n.constant->size() != n.type.num_elements()) {
					return Error{what + ": constant payload does not match " + to_string(n.type)};
				}
				break;
			}
			case OpKind::FusedElementwise: {
				if (!n.fused) return Error{what + ": missing fused program"};
				Status st = check_fused(*n.fused, operand_types, n.type, what);
				if (!st.ok()) return st;
				break;
			}
			default: {
				Result<TensorType> t = infer_result_type(n.op, operand_types);
				if (!t.ok()) return Error{what + ": " + t.error().message};
				if (t.value() != n.type) {
					return Error{what + ": type " + to_string(n.type) + " should be " + to_string(t.value())};
				}
				break;
			}
		}
	}
	if (input_nodes != inputs_.size()) return Error{"input list does not match the input nodes"};
	for (NodeId in : inputs_) {
		if (in < 0 || static_cast<std::size_t>(in) >= nodes_.size() || node(in).op != OpKind::Input) {
			return Error{"input list refers to a non-input node"};
		}
	}
	if (outputs_.empty()) return Error{"graph has no outputs"};
	for (std::size_t i=0; i<outputs_.size(); ++i) {
		const NodeId out = outputs_[i];
		if (out < 0 || static_cast<std::size_t>(out) >= nodes_.size()) return Error{"output refers to a missing node"};
		if (std::find(outputs_.begin(), outputs_.begin() + static_cast<std::ptrdiff_t>(i), out) !=
		    outputs_.begin() + static_cast<std::ptrdiff_t>(i)) {
			return Error{"'" + node(out).name + "' is listed as an output twice"};
		}
	}
	return Status();
}

GraphStats compute_stats(const Graph& graph) {
	GraphStats s;
	s.nodes = graph.num_nodes();
	for (const Node& n : graph.nodes()) {
		if (n.op == OpKind::Input) {
			++s.inputs;
			continue;
		}
		if (n.op == OpKind::Constant) {
			++s.constants;
			continue;
		}
		++s.compute_nodes;
		++s.op_counts[op_name(n.op)];
		if (n.op == OpKind::FusedElementwise) {
			++s.fused_nodes;
			s.elementwise_ops += n.fused->num_ops();
		} else if (n.op == OpKind::MatMul) {
			++s.matmuls;
		} else {
			++s.elementwise_ops;
		}
	}
	return s;
}

}
