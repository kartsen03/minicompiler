#include "minicompiler/graph_builder.hpp"
#include "minicompiler/random.hpp"
#include "minicompiler/runtime/backend.hpp"

#include "test_util.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace minicompiler;
using minicompiler::testutil::all_close;
using minicompiler::testutil::evaluate_reference;

namespace {

std::unique_ptr<Executable> compile_cpu(const Graph& g) {
	auto backend = create_backend("cpu");
	EXPECT_TRUE(backend.ok());
	auto exe = (*backend)->compile(g);
	EXPECT_TRUE(exe.ok()) << exe.error().message;
	return std::move(exe).value();
}

std::vector<HostTensor> run_cpu(const Graph& g, const std::vector<HostTensor>& inputs) {
	std::vector<HostTensor> outputs;
	Status st = compile_cpu(g)->run(inputs, outputs);
	EXPECT_TRUE(st.ok()) << st.message();
	return outputs;
}

Graph unary_graph(OpKind op, Shape shape) {
	GraphBuilder b("unary");
	b.output(b.op(op, {b.input("x", shape)}));
	return std::move(b).build().value();
}

Graph binary_graph(OpKind op, Shape a, Shape b_shape) {
	GraphBuilder b("binary");
	NodeId a_id = b.input("a", a);
	NodeId b_id = b.input("b", b_shape);
	b.output(b.op(op, {a_id, b_id}));
	return std::move(b).build().value();
}

}

// +, -, *, /, neg and relu are exact or correctly rounded in IEEE arithmetic,
// so Eigen must match the double-precision reference bit for bit.
//
// sqrt is not: with EIGEN_FAST_MATH (Eigen's default, which also provides the
// vectorized tanh) the AVX path computes x * rsqrt(x) refined by one Newton
// step, so allow 2 ulp. exp, log, tanh and sigmoid use polynomial
// approximations accurate to a few ulp: allow 8 ulp (rtol 1e-6) plus 1 ulp at
// 1.0 for results near zero.
constexpr double kSqrtRtol = 2.4e-7;
constexpr double kTranscendentalRtol = 1e-6;
constexpr double kTranscendentalAtol = 1.2e-7;

TEST(CpuKernels, UnaryOpsMatchReference) {
	const Shape shape = {37, 29};  // not a multiple of the SIMD width
	for (OpKind op : {OpKind::Neg, OpKind::Exp, OpKind::Log, OpKind::Sqrt, OpKind::Relu, OpKind::Sigmoid,
	                  OpKind::Tanh}) {
		Graph g = unary_graph(op, shape);
		const bool positive_domain = op == OpKind::Log || op == OpKind::Sqrt;
		auto inputs = make_random_inputs(g, 11, positive_domain ? 0.01f : -6.0f, 6.0f);
		const auto expected = evaluate_reference(g, inputs);
		const auto actual = run_cpu(g, inputs);
		double rtol = kTranscendentalRtol;
		double atol = kTranscendentalAtol;
		if (op == OpKind::Neg || op == OpKind::Relu) rtol = atol = 0.0;
		if (op == OpKind::Sqrt) {
			rtol = kSqrtRtol;
			atol = 0.0;
		}
		EXPECT_TRUE(all_close(actual[0].data, expected[0].data, rtol, atol)) << op_name(op);
	}
}

TEST(CpuKernels, BinaryOpsMatchReferenceForEveryBroadcastPattern) {
	const std::vector<std::pair<Shape, Shape>> patterns = {
		{{17, 33}, {17, 33}},     // same shape
		{{17, 33}, {}},           // scalar on the right
		{{}, {17, 33}},           // scalar on the left
		{{17, 33}, {33}},         // row vector (bias)
		{{17, 33}, {17, 1}},      // column vector
		{{17, 1}, {1, 33}},       // outer broadcast: neither operand has the output shape
		{{2, 3, 4}, {3, 1}},      // rank 3
		{{5, 1, 6}, {1, 4, 1}},
	};
	for (const auto& [sa, sb] : patterns) {
		for (OpKind op : {OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div}) {
			Graph g = binary_graph(op, sa, sb);
			auto inputs = make_random_inputs(g, 5, 0.5f, 2.0f);  // keep divisors away from zero
			const auto expected = evaluate_reference(g, inputs);
			const auto actual = run_cpu(g, inputs);
			EXPECT_TRUE(all_close(actual[0].data, expected[0].data, 0.0, 0.0))
			    << op_name(op) << " " << to_string(sa) << " with " << to_string(sb);
		}
	}
}

TEST(CpuKernels, MatmulIsWithinTheDotProductErrorBound) {
	// Any summation order of a length-k float dot product is within
	// gamma_k * sum|a_i b_i| of the exact value, gamma_k = k u / (1 - k u),
	// u = 2^-24. The reference is exact up to its final rounding to float.
	for (auto [m, k, n] : std::vector<std::array<std::int64_t, 3>>{{1, 1, 1}, {3, 5, 7}, {64, 64, 64}, {33, 257, 65}}) {
		GraphBuilder b("mm");
		NodeId a = b.input("a", {m, k});
		NodeId w = b.input("w", {k, n});
		b.output(b.matmul(a, w));
		Graph g = std::move(b).build().value();
		auto inputs = make_random_inputs(g, 3);
		const auto expected = evaluate_reference(g, inputs);
		const auto actual = run_cpu(g, inputs);

		const double u = std::ldexp(1.0, -24);
		const double gamma = static_cast<double>(k) * u / (1.0 - static_cast<double>(k) * u);
		for (std::int64_t r=0; r<m; ++r) {
			for (std::int64_t c=0; c<n; ++c) {
				double abs_dot = 0.0;
				for (std::int64_t t=0; t<k; ++t) {
					abs_dot += std::fabs(inputs[0].data[r * k + t] * static_cast<double>(inputs[1].data[t * n + c]));
				}
				const double e = expected[0].data[r * n + c];
				const double bound = gamma * abs_dot + std::fabs(e) * u;
				ASSERT_LE(std::fabs(actual[0].data[r * n + c] - e), bound) << m << "x" << k << "x" << n;
			}
		}
	}
}

TEST(CpuKernels, FusedProgramMatchesReferenceForEveryLoadKind) {
	// out = tanh(x * s + row) - col / 2 over [37, 41]: x is read in place,
	// s is a scalar, row repeats every 41 elements, col needs index math.
	// 37 * 41 = 1517 elements, so the last block is partial.
	Graph g("fused");
	NodeId x = g.add_input("x", TensorType({37, 41})).value();
	NodeId s = g.add_input("s", TensorType(Shape{})).value();
	NodeId row = g.add_input("row", TensorType({41})).value();
	NodeId col = g.add_input("col", TensorType({37, 1})).value();
	auto p = std::make_shared<FusedProgram>();
	p->instrs = {
		FusedInstr::load(0), FusedInstr::load(1), FusedInstr::binary(OpKind::Mul, 0, 1),
		FusedInstr::load(2), FusedInstr::binary(OpKind::Add, 2, 3), FusedInstr::unary(OpKind::Tanh, 4),
		FusedInstr::load(3), FusedInstr::imm(2.0f), FusedInstr::binary(OpKind::Div, 6, 7),
		FusedInstr::binary(OpKind::Sub, 5, 8),
	};
	NodeId f = g.add_fused({x, s, row, col}, p, TensorType({37, 41}), "f").value();
	ASSERT_TRUE(g.add_output(f).ok());

	auto inputs = make_random_inputs(g, 21);
	EXPECT_TRUE(all_close(run_cpu(g, inputs)[0].data, evaluate_reference(g, inputs)[0].data, kTranscendentalRtol,
	                      kTranscendentalAtol));
}

TEST(CpuKernels, FusedProgramHandlesScalarAndPassThroughResults) {
	Graph g("edge");
	NodeId x = g.add_input("x", TensorType({1000})).value();
	NodeId s = g.add_input("s", TensorType(Shape{})).value();
	auto copy = std::make_shared<FusedProgram>();
	copy->instrs = {FusedInstr::load(0)};
	auto splat = std::make_shared<FusedProgram>();
	splat->instrs = {FusedInstr::load(1), FusedInstr::imm(3.0f), FusedInstr::binary(OpKind::Mul, 0, 1)};
	NodeId a = g.add_fused({x}, copy, TensorType({1000}), "copy").value();
	NodeId b2 = g.add_fused({x, s}, splat, TensorType({1000}), "splat").value();
	ASSERT_TRUE(g.add_output(a).ok());
	ASSERT_TRUE(g.add_output(b2).ok());

	auto inputs = make_random_inputs(g, 9);
	const auto out = run_cpu(g, inputs);
	EXPECT_EQ(out[0].data, inputs[0].data);
	EXPECT_EQ(out[1].data, std::vector<float>(1000, inputs[1].data[0] * 3.0f));
}

TEST(CpuBackend, RandomGraphsMatchReference) {
	// Up to 16 ops deep, so allow for error growth: 1e-4 relative to each
	// output's largest magnitude.
	for (std::uint64_t seed=1; seed<=200; ++seed) {
		Graph g = testutil::make_random_graph(seed);
		auto inputs = make_random_inputs(g, seed);
		const auto expected = evaluate_reference(g, inputs);
		const auto actual = run_cpu(g, inputs);
		ASSERT_EQ(actual.size(), expected.size());
		for (std::size_t k=0; k<actual.size(); ++k) {
			const double scale = std::max(1.0, testutil::max_abs(expected[k].data));
			EXPECT_TRUE(all_close(actual[k].data, expected[k].data, 1e-4, 1e-4 * scale))
			    << "seed " << seed << ", output " << k;
		}
	}
}

TEST(CpuBackend, ReusesBuffersAcrossRuns) {
	Graph g = testutil::make_random_graph(7);
	auto exe = compile_cpu(g);
	std::vector<HostTensor> first, second, again;
	ASSERT_TRUE(exe->run(make_random_inputs(g, 1), first).ok());
	ASSERT_TRUE(exe->run(make_random_inputs(g, 2), second).ok());
	ASSERT_TRUE(exe->run(make_random_inputs(g, 1), again).ok());
	for (std::size_t k=0; k<first.size(); ++k) EXPECT_EQ(first[k].data, again[k].data);
	EXPECT_LE(exe->intermediate_bytes(), 1u << 20);
}

TEST(CpuBackend, AnOutputCanFeedLaterNodes) {
	// y1 is written into the caller's tensor and then read by y2.
	GraphBuilder b("chained_outputs");
	NodeId x = b.input("x", {300});
	NodeId y1 = b.exp(x, "y1");
	b.output(y1);
	b.output(b.tanh(b.neg(y1), "y2"));
	Graph g = std::move(b).build().value();
	auto inputs = make_random_inputs(g, 4);
	const auto expected = evaluate_reference(g, inputs);
	const auto actual = run_cpu(g, inputs);
	EXPECT_TRUE(all_close(actual[0].data, expected[0].data, kTranscendentalRtol, kTranscendentalAtol));
	EXPECT_TRUE(all_close(actual[1].data, expected[1].data, kTranscendentalRtol, kTranscendentalAtol));
}

TEST(CpuBackend, OutputsCanBeInputsOrConstants) {
	GraphBuilder b("passthrough");
	NodeId x = b.input("x", {3});
	NodeId c = b.constant("c", {2}, {4, 5});
	b.output(x);
	b.output(c);
	Graph g = std::move(b).build().value();
	auto inputs = make_random_inputs(g, 1);
	const auto out = run_cpu(g, inputs);
	EXPECT_EQ(out[0].data, inputs[0].data);
	EXPECT_EQ(out[1].data, (std::vector<float>{4, 5}));
}

TEST(CpuBackend, RejectsTheSameVectorForInputsAndOutputs) {
	// Outputs are written while inputs are still read, so aliasing would
	// silently corrupt results.
	Graph g = unary_graph(OpKind::Neg, {4});
	auto exe = compile_cpu(g);
	std::vector<HostTensor> v = make_random_inputs(g, 1);
	Status st = exe->run(v, v);
	ASSERT_FALSE(st.ok());
	EXPECT_EQ(st.message(), "run() needs separate input and output vectors");
}

TEST(CpuBackend, RunsOnCallerOwnedBuffers) {
	Graph g = testutil::make_random_graph(11);
	auto exe = compile_cpu(g);
	const auto inputs = make_random_inputs(g, 3);
	std::vector<HostTensor> expected;
	ASSERT_TRUE(exe->run(inputs, expected).ok());

	std::vector<const float*> in;
	for (const HostTensor& t : inputs) in.push_back(t.data.data());
	std::vector<std::vector<float>> storage;
	for (const HostTensor& t : expected) storage.emplace_back(t.data.size(), -1.0f);
	std::vector<float*> out;
	for (std::vector<float>& s : storage) out.push_back(s.data());
	ASSERT_TRUE(exe->run_buffers(in, out).ok());
	for (std::size_t k=0; k<expected.size(); ++k) EXPECT_EQ(storage[k], expected[k].data);
}

TEST(CpuBackend, RunBuffersRejectsOverlapAndWrongCounts) {
	Graph g = unary_graph(OpKind::Neg, {8});
	auto exe = compile_cpu(g);
	std::vector<float> buf(16, 1.0f);
	EXPECT_FALSE(exe->run_buffers({buf.data()}, {buf.data() + 4}).ok());  // [4,12) overlaps [0,8)
	EXPECT_TRUE(exe->run_buffers({buf.data()}, {buf.data() + 8}).ok());   // adjacent is fine
	EXPECT_EQ(buf[8], -1.0f);
	EXPECT_FALSE(exe->run_buffers({}, {buf.data()}).ok());
}

TEST(CpuBackend, RejectsMismatchedInputs) {
	Graph g = unary_graph(OpKind::Exp, {4});
	auto exe = compile_cpu(g);
	std::vector<HostTensor> outputs;
	EXPECT_FALSE(exe->run({}, outputs).ok());
	EXPECT_FALSE(exe->run({HostTensor(TensorType({5}))}, outputs).ok());
	EXPECT_TRUE(exe->run({HostTensor(TensorType({4}))}, outputs).ok());
}

TEST(Backends, SelectedByNameAtRuntime) {
	auto cpu = create_backend("cpu");
	ASSERT_TRUE(cpu.ok());
	EXPECT_STREQ((*cpu)->name(), "cpu");
	EXPECT_FALSE(create_backend("tpu").ok());
#ifndef MINICOMPILER_HAVE_CUDA
	auto cuda = create_backend("cuda");
	ASSERT_FALSE(cuda.ok());
	EXPECT_NE(cuda.error().message.find("MINICOMPILER_ENABLE_CUDA"), std::string::npos);
#endif
}
