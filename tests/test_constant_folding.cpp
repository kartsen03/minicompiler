#include "minicompiler/graph_builder.hpp"
#include "minicompiler/passes/constant_folding.hpp"
#include "minicompiler/passes/dead_node_elimination.hpp"
#include "minicompiler/runtime/backend.hpp"

#include "test_util.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string_view>

using namespace minicompiler;

namespace {

const Node& node_named(const Graph& g, std::string_view name) {
	return g.node(*g.find(name));
}

std::vector<HostTensor> run_cpu(const Graph& g, const std::vector<HostTensor>& inputs) {
	auto exe = create_backend("cpu").value()->compile(g);
	std::vector<HostTensor> out;
	EXPECT_TRUE(exe.ok() && (*exe)->run(inputs, out).ok());
	return out;
}

}

TEST(ConstantFolding, ComputesTheRightScalar) {
	GraphBuilder b("gelu_coeff");
	NodeId x = b.input("x", {4});
	NodeId two = b.scalar("two", 2.0f);
	NodeId pi = b.scalar("pi", 3.14159265f);
	NodeId c = b.sqrt(b.div(two, pi, "ratio"), "c");  // sqrt(2/pi), the GELU constant
	b.output(b.mul(x, c, "y"));
	Graph g = std::move(b).build().value();

	PassReport report;
	Graph folded = fold_constants(g, &report).value();
	const Node& cn = node_named(folded, "c");
	ASSERT_EQ(cn.op, OpKind::Constant);
	EXPECT_NEAR(cn.constant->at(0), std::sqrt(2.0 / 3.14159265), 2e-7);
	EXPECT_EQ(node_named(folded, "ratio").op, OpKind::Constant);
	EXPECT_EQ(node_named(folded, "y").op, OpKind::Mul);  // depends on x: stays
	EXPECT_EQ(report.summary, "folded 2 node(s) into constants");
	EXPECT_EQ(report.before.compute_nodes, 3u);
	EXPECT_EQ(report.after.compute_nodes, 1u);
}

TEST(ConstantFolding, FoldsTensorsWithBroadcastingAndMatmul) {
	// BatchNorm in inference mode: scale = gamma / sqrt(var + eps) is all
	// constants, and so is W' = W * scale.
	GraphBuilder b("bn");
	NodeId x = b.input("x", {2, 3});
	NodeId var = b.constant("var", {3}, {0.25f, 1.0f, 4.0f});
	NodeId eps = b.scalar("eps", 0.0f);
	NodeId gamma = b.constant("gamma", {3}, {1.0f, 2.0f, 3.0f});
	NodeId scale = b.div(gamma, b.sqrt(b.add(var, eps)), "scale");
	NodeId w = b.constant("w", {2, 3}, {1, 2, 3, 4, 5, 6});
	NodeId ws = b.mul(w, scale, "ws");                               // [2,3] * [3]
	NodeId sq = b.matmul(b.constant("m", {2, 2}, {1, 2, 3, 4}), ws, "sq");  // [2,2] x [2,3]
	b.output(b.add(x, sq, "y"));
	Graph folded = fold_constants(std::move(b).build().value()).value();

	EXPECT_EQ(*node_named(folded, "scale").constant, (std::vector<float>{2.0f, 2.0f, 1.5f}));
	EXPECT_EQ(*node_named(folded, "ws").constant, (std::vector<float>{2, 4, 4.5f, 8, 10, 9}));
	// [[1,2],[3,4]] x [[2,4,4.5],[8,10,9]]
	EXPECT_EQ(*node_named(folded, "sq").constant, (std::vector<float>{18, 24, 22.5f, 38, 52, 49.5f}));
	EXPECT_EQ(node_named(folded, "y").op, OpKind::Add);
}

TEST(ConstantFolding, LeavesAnythingThatDependsOnAnInput) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {4});
	NodeId c = b.scalar("c", 1.0f);
	b.output(b.exp(b.add(x, c)));
	Graph g = std::move(b).build().value();
	PassReport report;
	Graph folded = fold_constants(g, &report).value();
	EXPECT_EQ(compute_stats(folded).compute_nodes, 2u);
	EXPECT_EQ(report.summary, "folded 0 node(s) into constants");
}

TEST(ConstantFolding, RespectsTheSizeLimit) {
	GraphBuilder b("big");
	NodeId a = b.constant("a", {64, 64}, std::vector<float>(4096, 1.0f));
	b.output(b.neg(a, "n"));
	Graph g = std::move(b).build().value();
	ConstantFoldingOptions small;
	small.max_elements = 1000;
	EXPECT_EQ(node_named(fold_constants(g, nullptr, small).value(), "n").op, OpKind::Neg);
	EXPECT_EQ(node_named(fold_constants(g).value(), "n").op, OpKind::Constant);
}

TEST(ConstantFolding, FoldsAnOutputThatIsEntirelyConstant) {
	GraphBuilder b("all_const");
	b.input("x", {2});
	NodeId c = b.constant("c", {2}, {1.0f, 4.0f});
	b.output(b.sqrt(c, "root"));
	Graph folded = fold_constants(std::move(b).build().value()).value();
	const Node& out = folded.node(folded.outputs()[0]);
	EXPECT_EQ(out.op, OpKind::Constant);
	EXPECT_EQ(*out.constant, (std::vector<float>{1.0f, 2.0f}));
}

TEST(ConstantFolding, PreservesResultsOnRandomGraphs) {
	// Folding evaluates with the backend's own kernels, but not bit-for-bit
	// identically: for an unaligned destination Eigen computes the leading
	// elements (up to the first aligned address) with scalar code such as
	// std::exp and the rest with SIMD polynomials, and the two can differ by
	// an ulp. The folded constant and the runtime buffer sit at different
	// addresses, so allow a few ulp, relative to each output's magnitude.
	std::size_t total_folded = 0;
	for (std::uint64_t seed=1; seed<=200; ++seed) {
		Graph g = testutil::make_random_graph(seed);
		PassReport report;
		Graph folded = eliminate_dead_nodes(fold_constants(g, &report).value()).value();
		total_folded += report.before.compute_nodes - report.after.compute_nodes;
		auto inputs = make_random_inputs(g, seed);
		const auto expected = run_cpu(g, inputs);
		const auto actual = run_cpu(folded, inputs);
		ASSERT_EQ(actual.size(), expected.size());
		for (std::size_t k=0; k<actual.size(); ++k) {
			const double scale = std::max(1.0, testutil::max_abs(expected[k].data));
			EXPECT_TRUE(testutil::all_close(actual[k].data, expected[k].data, 1e-6, 1e-6 * scale))
			    << "seed " << seed << ", output " << k;
		}
	}
	EXPECT_GT(total_folded, 50u) << "the random graphs should exercise folding";
}
