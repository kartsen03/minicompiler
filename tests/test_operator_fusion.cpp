#include "minicompiler/graph_builder.hpp"
#include "minicompiler/passes/dead_node_elimination.hpp"
#include "minicompiler/passes/operator_fusion.hpp"
#include "minicompiler/runtime/backend.hpp"

#include "test_util.hpp"

#include <gtest/gtest.h>

#include <algorithm>

using namespace minicompiler;

namespace {

// Names of the nodes in each fusion group.
std::vector<std::vector<std::string>> group_names(const Graph& g) {
	std::vector<std::vector<std::string>> out;
	for (const auto& group : find_fusion_groups(g)) {
		std::vector<std::string> names;
		for (NodeId id : group) names.push_back(g.node(id).name);
		out.push_back(names);
	}
	return out;
}

std::vector<HostTensor> run_cpu(const Graph& g, const std::vector<HostTensor>& inputs) {
	auto exe = create_backend("cpu").value()->compile(g);
	std::vector<HostTensor> out;
	EXPECT_TRUE(exe.ok() && (*exe)->run(inputs, out).ok());
	return out;
}

// GELU, tanh approximation: 0.5 x (1 + tanh(c (x + 0.044715 x^3))).
Graph gelu_graph() {
	GraphBuilder b("gelu");
	NodeId x = b.input("x", {8, 64});
	NodeId x2 = b.mul(x, x, "x2");
	NodeId x3 = b.mul(x2, x, "x3");
	NodeId t = b.mul(x3, b.scalar("coef", 0.044715f), "t");
	NodeId inner = b.add(x, t, "inner");
	NodeId u = b.mul(inner, b.scalar("c", 0.7978846f), "u");
	NodeId th = b.tanh(u, "th");
	NodeId p = b.add(th, b.scalar("one", 1.0f), "p");
	NodeId hx = b.mul(x, b.scalar("half", 0.5f), "hx");
	b.output(b.mul(hx, p, "y"));
	return std::move(b).build().value();
}

}

TEST(OperatorFusion, FusesAnElementwiseChainIntoOneKernel) {
	Graph g = gelu_graph();
	EXPECT_EQ(group_names(g), (std::vector<std::vector<std::string>>{
	                              {"x2", "x3", "t", "inner", "u", "th", "p", "hx", "y"}}));

	PassReport report;
	Graph fused = fuse_elementwise(g, &report).value();
	EXPECT_EQ(report.summary, "fused 9 op(s) into 1 kernel(s)");
	EXPECT_EQ(report.before.compute_nodes, 9u);
	EXPECT_EQ(report.after.compute_nodes, 1u);
	EXPECT_EQ(report.after.elementwise_ops, 9u);  // same work, one kernel

	const Node& y = fused.node(fused.outputs()[0]);
	ASSERT_EQ(y.op, OpKind::FusedElementwise);
	EXPECT_EQ(y.name, "y");
	EXPECT_EQ(y.inputs.size(), 1u);  // only x is loaded; x is read once
	const FusedProgram& p = *y.fused;
	EXPECT_EQ(p.num_ops(), 9u);
	EXPECT_EQ(std::count_if(p.instrs.begin(), p.instrs.end(),
	                        [](const FusedInstr& i) { return i.kind == FusedInstr::Kind::Load; }), 1);
	EXPECT_EQ(std::count_if(p.instrs.begin(), p.instrs.end(),
	                        [](const FusedInstr& i) { return i.kind == FusedInstr::Kind::Immediate; }), 4);

	// The four scalar constants were inlined, so DNE can remove them.
	Graph clean = eliminate_dead_nodes(fused).value();
	EXPECT_EQ(clean.num_nodes(), 2u);  // x and the fused kernel
}

TEST(OperatorFusion, MatmulSplitsGroups) {
	GraphBuilder b("mbr");
	NodeId x = b.input("x", {4, 8});
	NodeId w = b.constant("w", {8, 16}, std::vector<float>(128, 0.1f));
	NodeId bias = b.constant("b", {16}, std::vector<float>(16, 0.2f));
	NodeId h = b.matmul(x, w, "h");
	b.output(b.relu(b.add(h, bias, "hb"), "y"));
	Graph g = std::move(b).build().value();
	EXPECT_EQ(group_names(g), (std::vector<std::vector<std::string>>{{"hb", "y"}}));

	Graph fused = fuse_elementwise(g).value();
	const Node& y = fused.node(fused.outputs()[0]);
	ASSERT_EQ(y.op, OpKind::FusedElementwise);
	ASSERT_EQ(y.inputs.size(), 2u);
	EXPECT_EQ(fused.node(y.inputs[0]).op, OpKind::MatMul);  // h, read from memory
	EXPECT_EQ(fused.node(y.inputs[1]).name, "b");           // non-scalar constant: loaded
	EXPECT_EQ(compute_stats(fused).compute_nodes, 2u);
}

TEST(OperatorFusion, KeepsAProducerThatIsNeededOutsideTheGroup) {
	GraphBuilder b("shared");
	NodeId x = b.input("x", {4, 4});
	NodeId a = b.exp(x, "a");
	NodeId t = b.tanh(a, "t");
	NodeId y = b.add(t, x, "y");
	NodeId z = b.matmul(a, b.constant("w", {4, 4}, std::vector<float>(16, 1.0f)), "z");  // also reads a
	b.output(y);
	b.output(z);
	EXPECT_EQ(group_names(std::move(b).build().value()), (std::vector<std::vector<std::string>>{{"t", "y"}}));
}

TEST(OperatorFusion, DoesNotFuseAwayAGraphOutput) {
	GraphBuilder b("outputs");
	NodeId x = b.input("x", {4});
	NodeId a = b.exp(x, "a");
	b.output(a);
	b.output(b.neg(a, "n"));
	EXPECT_TRUE(group_names(std::move(b).build().value()).empty());
}

TEST(OperatorFusion, FusesADiamondAndComputesTheSharedValueOnce) {
	GraphBuilder b("diamond");
	NodeId x = b.input("x", {16});
	NodeId a = b.tanh(x, "a");
	NodeId e = b.exp(a, "e");
	NodeId n = b.neg(a, "n");
	b.output(b.add(e, n, "y"));
	Graph g = std::move(b).build().value();
	EXPECT_EQ(group_names(g), (std::vector<std::vector<std::string>>{{"a", "e", "n", "y"}}));
	const Graph fused = fuse_elementwise(g).value();
	const FusedProgram& p = *fused.node(1).fused;
	EXPECT_EQ(p.num_ops(), 4u);  // tanh appears once even though two ops read it
}

TEST(OperatorFusion, DoesNotPullInABroadcastProducer) {
	// v = exp(row) has shape [8]; fusing it into the [4,8] group would compute
	// exp four times per element of row.
	GraphBuilder b("bcast");
	NodeId x = b.input("x", {4, 8});
	NodeId v = b.exp(b.input("row", {8}), "v");
	b.output(b.tanh(b.add(x, v, "s"), "y"));
	EXPECT_EQ(group_names(std::move(b).build().value()), (std::vector<std::vector<std::string>>{{"s", "y"}}));
}

TEST(OperatorFusion, NeverCreatesACycleThroughAnOutsideNode) {
	// a feeds both the matmul m and y = a * m. Fusing {a, y} would need m
	// inside the kernel while m needs the kernel's a: a cycle. a has a user
	// outside the group, so it stays out.
	GraphBuilder b("cycle");
	NodeId x = b.input("x", {4, 4});
	NodeId a = b.exp(x, "a");
	NodeId m = b.matmul(a, b.constant("w", {4, 4}, std::vector<float>(16, 0.5f)), "m");
	b.output(b.mul(a, m, "y"));
	Graph g = std::move(b).build().value();
	EXPECT_TRUE(group_names(g).empty());
	Graph fused = fuse_elementwise(g).value();
	EXPECT_TRUE(fused.verify().ok());
}

TEST(OperatorFusion, PreservesResultsOnRandomGraphs) {
	// Same kernels, evaluated block by block; differences come only from
	// Eigen's alignment-dependent scalar/SIMD split (see constant folding).
	std::size_t fused_nodes = 0;
	for (std::uint64_t seed=1; seed<=200; ++seed) {
		Graph g = testutil::make_random_graph(seed, {24, true});
		Graph fused = fuse_elementwise(g).value();
		ASSERT_TRUE(fused.verify().ok()) << "seed " << seed;
		fused_nodes += compute_stats(fused).fused_nodes;
		EXPECT_EQ(compute_stats(fused).elementwise_ops, compute_stats(g).elementwise_ops) << "seed " << seed;

		auto inputs = make_random_inputs(g, seed);
		const auto expected = run_cpu(g, inputs);
		const auto actual = run_cpu(fused, inputs);
		for (std::size_t k=0; k<actual.size(); ++k) {
			const double scale = std::max(1.0, testutil::max_abs(expected[k].data));
			EXPECT_TRUE(testutil::all_close(actual[k].data, expected[k].data, 1e-6, 1e-6 * scale))
			    << "seed " << seed << ", output " << k;
		}
	}
	EXPECT_GT(fused_nodes, 100u) << "the random graphs should exercise fusion";
}
