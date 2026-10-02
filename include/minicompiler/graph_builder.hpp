#pragma once

#include "minicompiler/graph.hpp"

#include <optional>
#include <string>
#include <vector>

namespace minicompiler {

// Convenience front end for building graphs in C++.
//
// Errors are sticky: the first failure is recorded, later calls do nothing
// and return kNoNode, and build() reports that first error. Graph-building
// code therefore reads straight through without a check after every line.
//
//   GraphBuilder b("example");
//   NodeId x = b.input("x", {4, 8});
//   NodeId y = b.relu(b.mul(x, b.scalar("two", 2.0f)));
//   b.output(y);
//   Result<Graph> g = std::move(b).build();
class GraphBuilder {
public:
	explicit GraphBuilder(std::string graph_name) : graph_(std::move(graph_name)) {}

	NodeId input(std::string name, Shape shape);
	NodeId constant(std::string name, Shape shape, std::vector<float> values);
	NodeId scalar(std::string name, float value);
	NodeId op(OpKind op, std::vector<NodeId> inputs, std::string name = {});

	NodeId neg(NodeId x, std::string name = {}) { return op(OpKind::Neg, {x}, std::move(name)); }
	NodeId exp(NodeId x, std::string name = {}) { return op(OpKind::Exp, {x}, std::move(name)); }
	NodeId log(NodeId x, std::string name = {}) { return op(OpKind::Log, {x}, std::move(name)); }
	NodeId sqrt(NodeId x, std::string name = {}) { return op(OpKind::Sqrt, {x}, std::move(name)); }
	NodeId relu(NodeId x, std::string name = {}) { return op(OpKind::Relu, {x}, std::move(name)); }
	NodeId sigmoid(NodeId x, std::string name = {}) { return op(OpKind::Sigmoid, {x}, std::move(name)); }
	NodeId tanh(NodeId x, std::string name = {}) { return op(OpKind::Tanh, {x}, std::move(name)); }
	NodeId add(NodeId a, NodeId b, std::string name = {}) { return op(OpKind::Add, {a, b}, std::move(name)); }
	NodeId sub(NodeId a, NodeId b, std::string name = {}) { return op(OpKind::Sub, {a, b}, std::move(name)); }
	NodeId mul(NodeId a, NodeId b, std::string name = {}) { return op(OpKind::Mul, {a, b}, std::move(name)); }
	NodeId div(NodeId a, NodeId b, std::string name = {}) { return op(OpKind::Div, {a, b}, std::move(name)); }
	NodeId matmul(NodeId a, NodeId b, std::string name = {}) { return op(OpKind::MatMul, {a, b}, std::move(name)); }

	void output(NodeId id);

	// Returns the first recorded error, or the finished graph after verify().
	Result<Graph> build() &&;

	const Graph& graph() const { return graph_; }

private:
	NodeId record(Result<NodeId> r);
	bool failed() const { return error_.has_value(); }

	Graph graph_;
	std::optional<Error> error_;
};

}
