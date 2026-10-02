#include "cpu/kernels.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>

namespace minicompiler::cpu {
namespace {

using ArrayMap = Eigen::Map<Eigen::ArrayXf>;
using ConstArrayMap = Eigen::Map<const Eigen::ArrayXf>;
using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

ConstArrayMap view(const float* p, std::size_t n) {
	return ConstArrayMap(p, static_cast<Eigen::Index>(n));
}

template <typename In>
void apply_unary(OpKind op, const In& x, ArrayMap& y) {
	switch (op) {
		case OpKind::Neg: y = -x; break;
		case OpKind::Exp: y = x.exp(); break;
		case OpKind::Log: y = x.log(); break;
		case OpKind::Sqrt: y = x.sqrt(); break;
		case OpKind::Relu: y = x.max(0.0f); break;
		case OpKind::Sigmoid: y = (1.0f + (-x).exp()).inverse(); break;
		case OpKind::Tanh: y = x.tanh(); break;
		default: assert(false && "not a unary elementwise op");
	}
}

// X and Y are each an Eigen array expression or a float scalar.
template <typename X, typename Y>
void apply_binary(OpKind op, const X& x, const Y& y, ArrayMap& out) {
	switch (op) {
		case OpKind::Add: out = x + y; break;
		case OpKind::Sub: out = x - y; break;
		case OpKind::Mul: out = x * y; break;
		case OpKind::Div: out = x / y; break;
		default: assert(false && "not a binary elementwise op");
	}
}

// Scalar versions, used when every operand of a fused instruction is a
// scalar. Unary ops go through a one-element array so they use the same
// Eigen math as the vector path.
float unary_scalar(OpKind op, float x) {
	float y;
	ArrayMap out(&y, 1);
	apply_unary(op, view(&x, 1), out);
	return y;
}

float binary_scalar(OpKind op, float a, float b) {
	switch (op) {
		case OpKind::Add: return a + b;
		case OpKind::Sub: return a - b;
		case OpKind::Mul: return a * b;
		case OpKind::Div: return a / b;
		default: assert(false && "not a binary elementwise op"); return 0.0f;
	}
}

std::int64_t offset_at(std::size_t i, const Shape& out, const std::vector<std::int64_t>& strides) {
	std::int64_t off = 0;
	for (std::size_t d=out.size(); d-- > 0;) {
		const auto dim = static_cast<std::size_t>(out[d]);
		off += static_cast<std::int64_t>(i % dim) * strides[d];
		i /= dim;
	}
	return off;
}

// How a fused kernel reads one of its inputs.
struct LoadPlan {
	enum class Kind {
		Contiguous,  // same layout as the output: read in place
		Scalar,      // one element, broadcast everywhere
		Periodic,    // the trailing dims of the output (e.g. a bias row): repeats every `period` elements
		General,     // anything else: per-element index math
	};
	Kind kind = Kind::Contiguous;
	std::size_t period = 0;
	std::vector<std::int64_t> strides;
};

LoadPlan plan_load(const Shape& in, const Shape& out) {
	LoadPlan lp;
	const std::size_t n_in = num_elements(in);
	if (n_in == 1) {
		lp.kind = LoadPlan::Kind::Scalar;
		return lp;
	}
	// A broadcastable input with as many elements as the output has the same layout.
	if (n_in == num_elements(out)) return lp;
	std::size_t lead = 0;
	while (lead < in.size() && in[lead] == 1) ++lead;
	const std::size_t rest = in.size() - lead;
	if (std::equal(in.begin() + static_cast<std::ptrdiff_t>(lead), in.end(), out.end() - static_cast<std::ptrdiff_t>(rest))) {
		lp.kind = LoadPlan::Kind::Periodic;
		lp.period = n_in;
		return lp;
	}
	lp.kind = LoadPlan::Kind::General;
	lp.strides = broadcast_strides(in, out);
	return lp;
}

// A fused-program register for the current block: either a pointer to `len`
// floats or a single scalar.
struct Reg {
	const float* vec = nullptr;
	float scalar = 0.0f;
};

Reg load_block(const LoadPlan& lp, const float* in, std::size_t start, std::size_t len, float* dst,
               const Shape& out_shape) {
	switch (lp.kind) {
		case LoadPlan::Kind::Contiguous:
			return {in + start, 0.0f};
		case LoadPlan::Kind::Scalar:
			return {nullptr, in[0]};
		case LoadPlan::Kind::Periodic: {
			std::size_t pos = start % lp.period;
			for (std::size_t j=0; j<len;) {
				const std::size_t chunk = std::min(len - j, lp.period - pos);
				std::copy(in + pos, in + pos + chunk, dst + j);
				j += chunk;
				pos = 0;
			}
			return {dst, 0.0f};
		}
		case LoadPlan::Kind::General:
			for (std::size_t j=0; j<len; ++j) dst[j] = in[offset_at(start + j, out_shape, lp.strides)];
			return {dst, 0.0f};
	}
	return {};
}

}

void unary(OpKind op, const float* x, float* y, std::size_t n) {
	ArrayMap out(y, static_cast<Eigen::Index>(n));
	apply_unary(op, view(x, n), out);
}

void binary(OpKind op, const float* a, const Shape& a_shape, const float* b, const Shape& b_shape, float* out,
            const Shape& out_shape) {
	const std::size_t n = num_elements(out_shape);
	ArrayMap o(out, static_cast<Eigen::Index>(n));
	const bool a_full = num_elements(a_shape) == n;
	const bool b_full = num_elements(b_shape) == n;
	if (a_full && b_full) return apply_binary(op, view(a, n), view(b, n), o);
	if (a_full && num_elements(b_shape) == 1) return apply_binary(op, view(a, n), b[0], o);
	if (num_elements(a_shape) == 1 && b_full) return apply_binary(op, a[0], view(b, n), o);

	// General broadcasting, one output row (last dimension) at a time. Along
	// the last dimension each operand either advances (stride 1) or repeats a
	// single element (stride 0); the leading dimensions select the row.
	const std::size_t rank = out_shape.size();
	const std::vector<std::int64_t> sa = broadcast_strides(a_shape, out_shape);
	const std::vector<std::int64_t> sb = broadcast_strides(b_shape, out_shape);
	const auto inner = static_cast<std::size_t>(out_shape[rank - 1]);
	const bool a_moves = sa[rank - 1] != 0;
	const bool b_moves = sb[rank - 1] != 0;
	std::vector<std::int64_t> coord(rank, 0);
	for (std::size_t row=0; row<n; row+=inner) {
		std::int64_t oa = 0;
		std::int64_t ob = 0;
		for (std::size_t d=0; d+1<rank; ++d) {
			oa += coord[d] * sa[d];
			ob += coord[d] * sb[d];
		}
		ArrayMap o_row(out + row, static_cast<Eigen::Index>(inner));
		if (a_moves && b_moves) {
			apply_binary(op, view(a + oa, inner), view(b + ob, inner), o_row);
		} else if (a_moves) {
			apply_binary(op, view(a + oa, inner), b[ob], o_row);
		} else if (b_moves) {
			apply_binary(op, a[oa], view(b + ob, inner), o_row);
		} else {
			o_row.setConstant(binary_scalar(op, a[oa], b[ob]));
		}
		for (std::size_t d=rank-1; d-- > 0;) {  // advance the row coordinate
			if (++coord[d] < out_shape[d]) break;
			coord[d] = 0;
		}
	}
}

void matmul(const float* a, const float* b, float* c, std::int64_t m, std::int64_t k, std::int64_t n) {
	Eigen::Map<const RowMatrix> A(a, m, k);
	Eigen::Map<const RowMatrix> B(b, k, n);
	Eigen::Map<RowMatrix> C(c, m, n);
	C.noalias() = A * B;
}

void fused(const FusedProgram& program, const std::vector<const float*>& inputs,
           const std::vector<Shape>& input_shapes, float* out, const Shape& out_shape, std::vector<float>& scratch) {
	const std::size_t n = num_elements(out_shape);
	const std::size_t num_regs = program.instrs.size();
	scratch.resize(num_regs * kFusedBlock);

	std::vector<LoadPlan> loads;
	loads.reserve(inputs.size());
	for (const Shape& s : input_shapes) loads.push_back(plan_load(s, out_shape));
	std::vector<Reg> regs(num_regs);

	for (std::size_t start=0; start<n; start+=kFusedBlock) {
		const std::size_t len = std::min(kFusedBlock, n - start);
		for (std::size_t r=0; r<num_regs; ++r) {
			const FusedInstr& ins = program.instrs[r];
			// The last instruction writes straight into the output.
			float* dst = r + 1 == num_regs ? out + start : scratch.data() + r * kFusedBlock;
			ArrayMap block(dst, static_cast<Eigen::Index>(len));
			switch (ins.kind) {
				case FusedInstr::Kind::Load:
					regs[r] = load_block(loads[ins.a], inputs[ins.a], start, len, dst, out_shape);
					break;
				case FusedInstr::Kind::Immediate:
					regs[r] = {nullptr, ins.immediate};
					break;
				case FusedInstr::Kind::Unary: {
					const Reg x = regs[ins.a];
					if (!x.vec) {
						regs[r] = {nullptr, unary_scalar(ins.op, x.scalar)};
						break;
					}
					apply_unary(ins.op, view(x.vec, len), block);
					regs[r] = {dst, 0.0f};
					break;
				}
				case FusedInstr::Kind::Binary: {
					const Reg x = regs[ins.a];
					const Reg y = regs[ins.b];
					if (!x.vec && !y.vec) {
						regs[r] = {nullptr, binary_scalar(ins.op, x.scalar, y.scalar)};
						break;
					}
					if (x.vec && y.vec) {
						apply_binary(ins.op, view(x.vec, len), view(y.vec, len), block);
					} else if (x.vec) {
						apply_binary(ins.op, view(x.vec, len), y.scalar, block);
					} else {
						apply_binary(ins.op, x.scalar, view(y.vec, len), block);
					}
					regs[r] = {dst, 0.0f};
					break;
				}
			}
		}
		// The result is normally already in place; it is not when the program
		// ends in a scalar or in a load read in place.
		const Reg result = regs[num_regs - 1];
		if (!result.vec) {
			std::fill(out + start, out + start + len, result.scalar);
		} else if (result.vec != out + start) {
			std::copy(result.vec, result.vec + len, out + start);
		}
	}
}

void execute_node(const Node& node, const std::vector<const float*>& operands,
                  const std::vector<Shape>& operand_shapes, float* out, std::vector<float>& scratch) {
	if (is_unary_elementwise(node.op)) {
		unary(node.op, operands[0], out, node.type.num_elements());
	} else if (is_binary_elementwise(node.op)) {
		binary(node.op, operands[0], operand_shapes[0], operands[1], operand_shapes[1], out, node.type.shape);
	} else if (node.op == OpKind::MatMul) {
		matmul(operands[0], operands[1], out, operand_shapes[0][0], operand_shapes[0][1], operand_shapes[1][1]);
	} else if (node.op == OpKind::FusedElementwise) {
		fused(*node.fused, operands, operand_shapes, out, node.type.shape, scratch);
	} else {
		assert(false && "not a compute node");
	}
}

}
