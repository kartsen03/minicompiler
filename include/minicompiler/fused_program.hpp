#pragma once

#include "minicompiler/ops.hpp"
#include "minicompiler/result.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace minicompiler {

// One instruction of a fused elementwise kernel. The program runs once per
// output element: every instruction produces one float, and operands name
// earlier instructions by index (a tiny SSA form, one "register" each).
struct FusedInstr {
	enum class Kind : std::uint8_t {
		Load,       // element of the fused node's input number `a`, broadcast to the output shape
		Immediate,  // the scalar `immediate`, baked into the kernel
		Unary,      // op(r[a])
		Binary,     // op(r[a], r[b])
	};

	Kind kind = Kind::Load;
	OpKind op = OpKind::Input;
	std::int32_t a = -1;
	std::int32_t b = -1;
	float immediate = 0.0f;

	static FusedInstr load(std::int32_t input) { return {Kind::Load, OpKind::Input, input, -1, 0.0f}; }
	static FusedInstr imm(float value) { return {Kind::Immediate, OpKind::Constant, -1, -1, value}; }
	static FusedInstr unary(OpKind op, std::int32_t x) { return {Kind::Unary, op, x, -1, 0.0f}; }
	static FusedInstr binary(OpKind op, std::int32_t x, std::int32_t y) { return {Kind::Binary, op, x, y, 0.0f}; }
};

// The body of a FusedElementwise node. The last instruction is the result.
struct FusedProgram {
	std::vector<FusedInstr> instrs;
	// Names of the original graph nodes this kernel replaces, in program
	// order. Used for DOT labels and to check fusion groups in tests.
	std::vector<std::string> source_nodes;

	// Number of elementwise ops (Unary and Binary instructions).
	std::size_t num_ops() const;
};

// Checks that operands refer to earlier instructions, loads refer to one of
// the node's `num_inputs` inputs, and ops have the right arity.
Status validate(const FusedProgram& program, std::size_t num_inputs);

// One instruction per line, e.g. "r2 = mul r0, r1".
std::string to_string(const FusedProgram& program);

}
