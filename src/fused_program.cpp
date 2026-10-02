#include "minicompiler/fused_program.hpp"

#include <sstream>

namespace minicompiler {

std::size_t FusedProgram::num_ops() const {
	std::size_t n = 0;
	for (const FusedInstr& ins : instrs) {
		if (ins.kind == FusedInstr::Kind::Unary || ins.kind == FusedInstr::Kind::Binary) ++n;
	}
	return n;
}

Status validate(const FusedProgram& program, std::size_t num_inputs) {
	if (program.instrs.empty()) return Error{"fused program is empty"};
	for (std::size_t i=0; i<program.instrs.size(); ++i) {
		const FusedInstr& ins = program.instrs[i];
		const std::string where = "fused instruction r" + std::to_string(i) + ": ";
		auto is_earlier = [&](std::int32_t r) { return r >= 0 && static_cast<std::size_t>(r) < i; };
		switch (ins.kind) {
			case FusedInstr::Kind::Load:
				if (ins.a < 0 || static_cast<std::size_t>(ins.a) >= num_inputs) {
					return Error{where + "loads input " + std::to_string(ins.a) + " of " + std::to_string(num_inputs)};
				}
				break;
			case FusedInstr::Kind::Immediate:
				break;
			case FusedInstr::Kind::Unary:
				if (!is_unary_elementwise(ins.op)) return Error{where + op_name(ins.op) + " is not a unary elementwise op"};
				if (!is_earlier(ins.a)) return Error{where + "operand must be an earlier instruction"};
				break;
			case FusedInstr::Kind::Binary:
				if (!is_binary_elementwise(ins.op)) return Error{where + op_name(ins.op) + " is not a binary elementwise op"};
				if (!is_earlier(ins.a) || !is_earlier(ins.b)) return Error{where + "operands must be earlier instructions"};
				break;
		}
	}
	return Status();
}

std::string to_string(const FusedProgram& program) {
	std::ostringstream os;
	for (std::size_t i=0; i<program.instrs.size(); ++i) {
		const FusedInstr& ins = program.instrs[i];
		os << "r" << i << " = ";
		switch (ins.kind) {
			case FusedInstr::Kind::Load: os << "load in" << ins.a; break;
			case FusedInstr::Kind::Immediate: os << "imm " << ins.immediate; break;
			case FusedInstr::Kind::Unary: os << op_name(ins.op) << " r" << ins.a; break;
			case FusedInstr::Kind::Binary: os << op_name(ins.op) << " r" << ins.a << ", r" << ins.b; break;
		}
		os << "\n";
	}
	return os.str();
}

}
