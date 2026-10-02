#include "test_util.hpp"

#include "minicompiler/graph_builder.hpp"
#include "minicompiler/random.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>

namespace minicompiler::testutil {

::testing::AssertionResult all_close(const std::vector<float>& actual, const std::vector<float>& expected,
                                     double rtol, double atol) {
	if (actual.size() != expected.size()) {
		return ::testing::AssertionFailure() << "size " << actual.size() << " vs expected " << expected.size();
	}
	std::size_t bad = 0;
	std::size_t worst = 0;
	double worst_ratio = 0.0;
	for (std::size_t i=0; i<actual.size(); ++i) {
		const double a = actual[i];
		const double e = expected[i];
		if (a == e) continue;  // also covers equal infinities
		if (std::isnan(a) && std::isnan(e)) continue;
		const double err = std::fabs(a - e);
		const double allowed = atol + rtol * std::fabs(e);
		if (!(err <= allowed)) {
			++bad;
			const double ratio = allowed > 0 ? err / allowed : HUGE_VAL;
			if (!(ratio <= worst_ratio)) {
				worst_ratio = ratio;
				worst = i;
			}
		}
	}
	if (bad == 0) return ::testing::AssertionSuccess();
	return ::testing::AssertionFailure() << bad << " of " << actual.size() << " elements differ; worst at ["
	                                     << worst << "]: actual " << actual[worst] << ", expected "
	                                     << expected[worst] << " (rtol " << rtol << ", atol " << atol << ")";
}

double max_abs(const std::vector<float>& v) {
	double m = 0.0;
	for (float x : v) m = std::max(m, static_cast<double>(std::fabs(x)));
	return m;
}

namespace {

double ref_unary(OpKind op, double x) {
	switch (op) {
		case OpKind::Neg: return -x;
		case OpKind::Exp: return std::exp(x);
		case OpKind::Log: return std::log(x);
		case OpKind::Sqrt: return std::sqrt(x);
		case OpKind::Relu: return x > 0.0 ? x : 0.0;
		case OpKind::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
		case OpKind::Tanh: return std::tanh(x);
		default: ADD_FAILURE() << "not unary: " << op_name(op); return 0.0;
	}
}

double ref_binary(OpKind op, double a, double b) {
	switch (op) {
		case OpKind::Add: return a + b;
		case OpKind::Sub: return a - b;
		case OpKind::Mul: return a * b;
		case OpKind::Div: return a / b;
		default: ADD_FAILURE() << "not binary: " << op_name(op); return 0.0;
	}
}

// Offset into a tensor read with `strides` at output linear index i.
std::int64_t offset_at(std::size_t i, const Shape& out, const std::vector<std::int64_t>& strides) {
	std::int64_t off = 0;
	for (std::size_t d=out.size(); d-- > 0;) {
		const std::int64_t coord = static_cast<std::int64_t>(i % static_cast<std::size_t>(out[d]));
		i /= static_cast<std::size_t>(out[d]);
		off += coord * strides[d];
	}
	return off;
}

}

std::vector<HostTensor> evaluate_reference(const Graph& graph, const std::vector<HostTensor>& inputs) {
	std::vector<std::vector<float>> values(graph.num_nodes());
	for (std::size_t k=0; k<graph.inputs().size(); ++k) values[graph.inputs()[k]] = inputs.at(k).data;

	for (std::size_t i=0; i<graph.num_nodes(); ++i) {
		const Node& n = graph.node(static_cast<NodeId>(i));
		const Shape& out = n.type.shape;
		const std::size_t count = n.type.num_elements();
		std::vector<float>& v = values[i];
		switch (n.op) {
			case OpKind::Input:
				break;
			case OpKind::Constant:
				v = *n.constant;
				break;
			case OpKind::MatMul: {
				const Shape& as = graph.node(n.inputs[0]).type.shape;
				const std::int64_t m = as[0], k = as[1], cols = out[1];
				const std::vector<float>& a = values[n.inputs[0]];
				const std::vector<float>& b = values[n.inputs[1]];
				v.assign(count, 0.0f);
				for (std::int64_t r=0; r<m; ++r) {
					for (std::int64_t c=0; c<cols; ++c) {
						double acc = 0.0;
						for (std::int64_t t=0; t<k; ++t) acc += static_cast<double>(a[r * k + t]) * b[t * cols + c];
						v[r * cols + c] = static_cast<float>(acc);
					}
				}
				break;
			}
			case OpKind::FusedElementwise: {
				const FusedProgram& p = *n.fused;
				std::vector<std::vector<std::int64_t>> strides;
				for (NodeId in : n.inputs) strides.push_back(broadcast_strides(graph.node(in).type.shape, out));
				std::vector<double> regs(p.instrs.size());
				v.resize(count);
				for (std::size_t e=0; e<count; ++e) {
					for (std::size_t r=0; r<p.instrs.size(); ++r) {
						const FusedInstr& ins = p.instrs[r];
						double x = 0.0;
						switch (ins.kind) {
							case FusedInstr::Kind::Load:
								x = values[n.inputs[ins.a]][offset_at(e, out, strides[ins.a])];
								break;
							case FusedInstr::Kind::Immediate: x = ins.immediate; break;
							case FusedInstr::Kind::Unary: x = ref_unary(ins.op, regs[ins.a]); break;
							case FusedInstr::Kind::Binary: x = ref_binary(ins.op, regs[ins.a], regs[ins.b]); break;
						}
						regs[r] = static_cast<float>(x);  // registers hold floats in the real kernels too
					}
					v[e] = static_cast<float>(regs.back());
				}
				break;
			}
			default: {
				std::vector<std::vector<std::int64_t>> strides;
				for (NodeId in : n.inputs) strides.push_back(broadcast_strides(graph.node(in).type.shape, out));
				v.resize(count);
				for (std::size_t e=0; e<count; ++e) {
					const double a = values[n.inputs[0]][offset_at(e, out, strides[0])];
					v[e] = static_cast<float>(is_unary_elementwise(n.op)
					                              ? ref_unary(n.op, a)
					                              : ref_binary(n.op, a, values[n.inputs[1]][offset_at(e, out, strides[1])]));
				}
				break;
			}
		}
	}

	std::vector<HostTensor> outputs;
	for (NodeId out : graph.outputs()) outputs.emplace_back(graph.node(out).type, values[out]);
	return outputs;
}

namespace {

struct Interval {
	double lo;
	double hi;
	double magnitude() const { return std::max(std::fabs(lo), std::fabs(hi)); }
};

struct Value {
	NodeId id;
	Shape shape;
	Interval range;
	bool is_constant;
};

// Grows a random graph while keeping every value inside a known interval.
class RandomGraphGenerator {
public:
	RandomGraphGenerator(std::uint64_t seed, const RandomGraphOptions& options)
	    : rng_(seed), options_(options), builder_("random_" + std::to_string(seed)), const_seed_(seed * 7919) {}

	Graph generate() {
		const std::int64_t m = pick<std::int64_t>({1, 2, 3, 7, 16, 33});
		const std::int64_t n = pick<std::int64_t>({1, 4, 5, 8, 17, 64});
		values_.push_back({builder_.input("x", {m, n}), {m, n}, {-1, 1}, false});
		if (chance(0.7)) values_.push_back({builder_.input("y", {m, n}), {m, n}, {-1, 1}, false});
		if (chance(0.5)) values_.push_back({builder_.input("row", {n}), {n}, {-1, 1}, false});
		if (chance(0.5)) values_.push_back({builder_.input("col", {m, 1}), {m, 1}, {-1, 1}, false});
		add_constant("s", {}, 0.5, 2.0);
		add_constant("bias", {n}, -1.0, 1.0);
		if (chance(0.5)) add_constant("pos", {m, n}, 0.5, 1.5);
		const std::size_t leaves = values_.size();

		int created = 0;
		for (int attempt=0; created < options_.num_ops && attempt < options_.num_ops * 50; ++attempt) {
			const double r = uniform();
			std::optional<Value> v;
			if (r < 0.40) {
				v = try_unary();
			} else if (r < 0.88 || !options_.allow_matmul) {
				v = try_binary();
			} else {
				v = try_matmul();
			}
			if (v) {
				values_.push_back(*v);
				++created;
			}
		}

		std::vector<NodeId> outputs;
		if (values_.size() > leaves) {
			outputs.push_back(values_.back().id);
		} else {
			outputs.push_back(builder_.neg(values_[0].id));
		}
		const int extra = static_cast<int>(rng_() % 3);
		for (int i=0; i<extra && values_.size() > leaves; ++i) {
			const NodeId id = values_[leaves + rng_() % (values_.size() - leaves)].id;
			if (std::find(outputs.begin(), outputs.end(), id) == outputs.end()) outputs.push_back(id);
		}
		for (NodeId id : outputs) builder_.output(id);

		Result<Graph> g = std::move(builder_).build();
		if (!g.ok()) {
			ADD_FAILURE() << "random graph generator built an invalid graph: " << g.error().message;
			return Graph();
		}
		return std::move(g).value();
	}

private:
	template <typename T>
	T pick(const std::vector<T>& choices) { return choices[rng_() % choices.size()]; }
	double uniform() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }
	bool chance(double p) { return uniform() < p; }
	const Value& any_value() { return values_[rng_() % values_.size()]; }

	void add_constant(const std::string& name, Shape shape, double lo, double hi) {
		const std::size_t count = num_elements(shape);
		const NodeId id = builder_.constant(name, shape, uniform_values(count, ++const_seed_, static_cast<float>(lo),
		                                                                static_cast<float>(hi)));
		values_.push_back({id, std::move(shape), {lo, hi}, true});
	}

	std::optional<Value> finish(OpKind op, std::vector<NodeId> operands, Shape shape, Interval range,
	                            bool is_constant) {
		if (!(range.magnitude() <= 1e3)) return std::nullopt;
		const NodeId id = builder_.op(op, std::move(operands));
		if (id == kNoNode) return std::nullopt;
		return Value{id, std::move(shape), range, is_constant};
	}

	std::optional<Value> try_unary() {
		const Value a = any_value();
		const Interval x = a.range;
		const OpKind op = pick<OpKind>({OpKind::Neg, OpKind::Exp, OpKind::Log, OpKind::Sqrt, OpKind::Relu,
		                                OpKind::Sigmoid, OpKind::Tanh});
		Interval out{};
		switch (op) {
			case OpKind::Neg: out = {-x.hi, -x.lo}; break;
			case OpKind::Exp:
				if (x.hi > 8.0) return std::nullopt;
				out = {std::exp(x.lo), std::exp(x.hi)};
				break;
			case OpKind::Log:
				if (x.lo < 0.05) return std::nullopt;
				out = {std::log(x.lo), std::log(x.hi)};
				break;
			case OpKind::Sqrt:
				if (x.lo < 0.0) return std::nullopt;
				out = {std::sqrt(x.lo), std::sqrt(x.hi)};
				break;
			case OpKind::Relu: out = {std::max(x.lo, 0.0), std::max(x.hi, 0.0)}; break;
			case OpKind::Sigmoid: out = {1.0 / (1.0 + std::exp(-x.lo)), 1.0 / (1.0 + std::exp(-x.hi))}; break;
			case OpKind::Tanh: out = {std::tanh(x.lo), std::tanh(x.hi)}; break;
			default: return std::nullopt;
		}
		return finish(op, {a.id}, a.shape, out, a.is_constant);
	}

	std::optional<Value> try_binary() {
		// Sometimes combine two constants, so constant folding has work to do.
		Value a = any_value();
		Value b = any_value();
		if (chance(0.15)) {
			std::vector<Value> constants;
			for (const Value& v : values_) {
				if (v.is_constant) constants.push_back(v);
			}
			a = constants[rng_() % constants.size()];
			b = constants[rng_() % constants.size()];
		}
		Result<Shape> shape = broadcast_shapes(a.shape, b.shape);
		if (!shape.ok()) return std::nullopt;
		const Interval x = a.range;
		const Interval y = b.range;
		const OpKind op = pick<OpKind>({OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div});
		Interval out{};
		switch (op) {
			case OpKind::Add: out = {x.lo + y.lo, x.hi + y.hi}; break;
			case OpKind::Sub: out = {x.lo - y.hi, x.hi - y.lo}; break;
			case OpKind::Mul: out = product(x, y); break;
			case OpKind::Div:
				if (y.lo < 0.1 && y.hi > -0.1) return std::nullopt;  // divisor must stay away from zero
				out = product(x, {1.0 / y.hi, 1.0 / y.lo});
				break;
			default: return std::nullopt;
		}
		return finish(op, {a.id, b.id}, std::move(shape).value(), out, a.is_constant && b.is_constant);
	}

	std::optional<Value> try_matmul() {
		const Value a = any_value();
		if (a.shape.size() != 2) return std::nullopt;
		const std::int64_t k = a.shape[1];
		const std::int64_t cols = pick<std::int64_t>({1, 3, 8, a.shape[1]});
		const double w = 1.0 / std::sqrt(static_cast<double>(k));
		add_constant("w" + std::to_string(const_seed_), {k, cols}, -w, w);
		const Value weight = values_.back();
		const double bound = static_cast<double>(k) * a.range.magnitude() * w;
		return finish(OpKind::MatMul, {a.id, weight.id}, {a.shape[0], cols}, {-bound, bound}, a.is_constant);
	}

	static Interval product(Interval x, Interval y) {
		const double c[4] = {x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi};
		return {*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
	}

	std::mt19937_64 rng_;
	RandomGraphOptions options_;
	GraphBuilder builder_;
	std::uint64_t const_seed_;
	std::vector<Value> values_;
};

}

Graph make_random_graph(std::uint64_t seed, const RandomGraphOptions& options) {
	return RandomGraphGenerator(seed, options).generate();
}

}
