#include "minicompiler/ops.hpp"

#include <algorithm>
#include <string>

namespace minicompiler {

const char* op_name(OpKind op) {
	switch (op) {
		case OpKind::Input: return "input";
		case OpKind::Constant: return "const";
		case OpKind::Neg: return "neg";
		case OpKind::Exp: return "exp";
		case OpKind::Log: return "log";
		case OpKind::Sqrt: return "sqrt";
		case OpKind::Relu: return "relu";
		case OpKind::Sigmoid: return "sigmoid";
		case OpKind::Tanh: return "tanh";
		case OpKind::Add: return "add";
		case OpKind::Sub: return "sub";
		case OpKind::Mul: return "mul";
		case OpKind::Div: return "div";
		case OpKind::MatMul: return "matmul";
		case OpKind::FusedElementwise: return "fused";
	}
	return "?";
}

std::optional<OpKind> op_from_name(std::string_view name) {
	static constexpr OpKind kWritableOps[] = {
		OpKind::Neg, OpKind::Exp, OpKind::Log, OpKind::Sqrt, OpKind::Relu, OpKind::Sigmoid,
		OpKind::Tanh, OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div, OpKind::MatMul,
	};
	for (OpKind op : kWritableOps) {
		if (name == op_name(op)) return op;
	}
	return std::nullopt;
}

bool is_unary_elementwise(OpKind op) {
	switch (op) {
		case OpKind::Neg:
		case OpKind::Exp:
		case OpKind::Log:
		case OpKind::Sqrt:
		case OpKind::Relu:
		case OpKind::Sigmoid:
		case OpKind::Tanh:
			return true;
		default:
			return false;
	}
}

bool is_binary_elementwise(OpKind op) {
	switch (op) {
		case OpKind::Add:
		case OpKind::Sub:
		case OpKind::Mul:
		case OpKind::Div:
			return true;
		default:
			return false;
	}
}

int op_arity(OpKind op) {
	if (op == OpKind::Input || op == OpKind::Constant) return 0;
	if (is_unary_elementwise(op)) return 1;
	if (is_binary_elementwise(op) || op == OpKind::MatMul) return 2;
	return -1;
}

Result<Shape> broadcast_shapes(const Shape& a, const Shape& b) {
	const std::size_t rank = std::max(a.size(), b.size());
	Shape out(rank);
	for (std::size_t i=0; i<rank; ++i) {
		// i counts dimensions from the right; missing leading dims act as 1.
		const std::int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
		const std::int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
		if (da != db && da != 1 && db != 1) {
			return Error{"cannot broadcast " + to_string(a) + " with " + to_string(b)};
		}
		out[rank - 1 - i] = da == 1 ? db : da;
	}
	return out;
}

std::vector<std::int64_t> broadcast_strides(const Shape& in, const Shape& out) {
	std::vector<std::int64_t> strides(out.size(), 0);
	std::int64_t stride = 1;
	for (std::size_t i=0; i<in.size(); ++i) {
		// i counts dimensions from the right.
		const std::int64_t dim = in[in.size() - 1 - i];
		strides[out.size() - 1 - i] = dim == 1 ? 0 : stride;
		stride *= dim;
	}
	return strides;
}

Result<TensorType> infer_result_type(OpKind op, const std::vector<TensorType>& operands) {
	const int arity = op_arity(op);
	if (arity <= 0) {
		return Error{std::string("'") + op_name(op) + "' has no inferred result type"};
	}
	if (operands.size() != static_cast<std::size_t>(arity)) {
		return Error{std::string(op_name(op)) + " expects " + std::to_string(arity) +
		             " operand(s), got " + std::to_string(operands.size())};
	}
	for (const TensorType& t : operands) {
		if (t.dtype != DType::Float32) {
			return Error{std::string(op_name(op)) + ": only f32 is supported, got " + to_string(t)};
		}
	}

	if (is_unary_elementwise(op)) return operands[0];

	if (is_binary_elementwise(op)) {
		Result<Shape> shape = broadcast_shapes(operands[0].shape, operands[1].shape);
		if (!shape.ok()) return Error{std::string(op_name(op)) + ": " + shape.error().message};
		return TensorType(std::move(shape).value());
	}

	// MatMul
	const Shape& a = operands[0].shape;
	const Shape& b = operands[1].shape;
	if (a.size() != 2 || b.size() != 2) {
		return Error{"matmul expects rank-2 operands, got " + to_string(a) + " and " + to_string(b)};
	}
	if (a[1] != b[0]) {
		return Error{"matmul inner dimensions differ: " + to_string(a) + " x " + to_string(b)};
	}
	return TensorType({a[0], b[1]});
}

}
