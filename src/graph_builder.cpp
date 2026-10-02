#include "minicompiler/graph_builder.hpp"

namespace minicompiler {

NodeId GraphBuilder::record(Result<NodeId> r) {
	if (r.ok()) return r.value();
	error_ = r.error();
	return kNoNode;
}

NodeId GraphBuilder::input(std::string name, Shape shape) {
	if (failed()) return kNoNode;
	return record(graph_.add_input(std::move(name), TensorType(std::move(shape))));
}

NodeId GraphBuilder::constant(std::string name, Shape shape, std::vector<float> values) {
	if (failed()) return kNoNode;
	return record(graph_.add_constant(std::move(name), TensorType(std::move(shape)), std::move(values)));
}

NodeId GraphBuilder::scalar(std::string name, float value) {
	return constant(std::move(name), {}, {value});
}

NodeId GraphBuilder::op(OpKind kind, std::vector<NodeId> inputs, std::string name) {
	if (failed()) return kNoNode;
	return record(graph_.add_op(kind, std::move(inputs), std::move(name)));
}

void GraphBuilder::output(NodeId id) {
	if (failed()) return;
	Status st = graph_.add_output(id);
	if (!st.ok()) error_ = st.error();
}

Result<Graph> GraphBuilder::build() && {
	if (failed()) return *error_;
	Status st = graph_.verify();
	if (!st.ok()) return st.error();
	return std::move(graph_);
}

}
