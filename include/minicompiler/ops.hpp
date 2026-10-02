#pragma once

#include "minicompiler/result.hpp"
#include "minicompiler/tensor.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace minicompiler {

enum class OpKind : std::uint8_t {
	// Leaves
	Input,     // a runtime argument of the graph
	Constant,  // a tensor whose value is known at compile time

	// Unary elementwise
	Neg,
	Exp,
	Log,
	Sqrt,
	Relu,
	Sigmoid,
	Tanh,

	// Binary elementwise, with NumPy-style broadcasting
	Add,
	Sub,
	Mul,
	Div,

	// [M,K] x [K,N] -> [M,N]
	MatMul,

	// Created by the operator-fusion pass: several elementwise ops executed as
	// one kernel. The node's FusedProgram says what the kernel computes.
	FusedElementwise,
};

// Lowercase mnemonic used by the text format, the printer and DOT labels.
const char* op_name(OpKind op);

// Looks up an op by mnemonic. Only ops that appear as `name = op args` in a
// graph file are found; input, const and fused have their own syntax.
std::optional<OpKind> op_from_name(std::string_view name);

bool is_unary_elementwise(OpKind op);
bool is_binary_elementwise(OpKind op);
inline bool is_elementwise(OpKind op) {
	return is_unary_elementwise(op) || is_binary_elementwise(op);
}

// True for nodes that do work at runtime: everything but Input and Constant.
inline bool is_compute(OpKind op) {
	return op != OpKind::Input && op != OpKind::Constant;
}

// Operand count, or -1 when it varies (FusedElementwise).
int op_arity(OpKind op);

// NumPy broadcasting: shapes are aligned on the right, and each pair of
// dimensions must be equal or contain a 1.
Result<Shape> broadcast_shapes(const Shape& a, const Shape& b);

// Element strides for reading a row-major tensor of shape `in` at the
// coordinates of a shape `out` that it broadcasts to: one stride per output
// dimension, 0 along dimensions where `in` is broadcast.
std::vector<std::int64_t> broadcast_strides(const Shape& in, const Shape& out);

// Result type of an elementwise op or MatMul applied to `operands`.
Result<TensorType> infer_result_type(OpKind op, const std::vector<TensorType>& operands);

}
