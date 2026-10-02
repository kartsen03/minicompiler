#include "minicompiler/graph_builder.hpp"
#include "minicompiler/passes/dead_node_elimination.hpp"

#include "test_util.hpp"

#include <gtest/gtest.h>

#include <set>

using namespace minicompiler;

namespace {

std::set<std::string> names(const Graph& g) {
	std::set<std::string> s;
	for (const Node& n : g.nodes()) s.insert(n.name);
	return s;
}

}

TEST(DeadNodeElimination, RemovesNodesNoOutputDependsOn) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {4, 8});
	NodeId w = b.constant("w", {8, 8}, std::vector<float>(64, 0.1f));
	NodeId h = b.matmul(x, w, "h");
	NodeId y = b.relu(h, "y");
	// An unused branch, as left by a training-only head.
	NodeId aux = b.tanh(b.matmul(h, b.constant("w_aux", {8, 8}, std::vector<float>(64, 0.2f)), "aux_mm"), "aux");
	(void)aux;
	b.output(y);
	Graph g = std::move(b).build().value();

	PassReport report;
	Graph out = eliminate_dead_nodes(g, &report).value();
	EXPECT_EQ(names(out), (std::set<std::string>{"x", "w", "h", "y"}));
	EXPECT_TRUE(out.verify().ok());
	EXPECT_EQ(report.before.nodes, 7u);
	EXPECT_EQ(report.after.nodes, 4u);
	EXPECT_EQ(report.summary, "removed 3 dead node(s)");
	// The surviving output keeps its name and type.
	ASSERT_EQ(out.outputs().size(), 1u);
	EXPECT_EQ(out.node(out.outputs()[0]).name, "y");
}

TEST(DeadNodeElimination, KeepsUnusedInputsInOrder) {
	GraphBuilder b("g");
	b.input("a", {2});
	NodeId used = b.input("b", {2});
	b.input("c", {2});
	b.output(b.neg(used));
	Graph out = eliminate_dead_nodes(std::move(b).build().value()).value();
	ASSERT_EQ(out.inputs().size(), 3u);
	EXPECT_EQ(out.node(out.inputs()[0]).name, "a");
	EXPECT_EQ(out.node(out.inputs()[2]).name, "c");
}

TEST(DeadNodeElimination, KeepsEverythingWhenNothingIsDead) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {3});
	NodeId y = b.exp(b.neg(x));
	b.output(y);
	Graph g = std::move(b).build().value();
	PassReport report;
	Graph out = eliminate_dead_nodes(g, &report).value();
	EXPECT_EQ(out.num_nodes(), g.num_nodes());
	EXPECT_EQ(report.summary, "removed 0 dead node(s)");
}

TEST(DeadNodeElimination, KeepsWhatAnyOfSeveralOutputsNeeds) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {3});
	NodeId a = b.exp(x, "a");
	NodeId c = b.neg(a, "c");
	b.tanh(x, "dead");
	b.output(c);
	b.output(a);  // a is an output and an operand of c
	Graph out = eliminate_dead_nodes(std::move(b).build().value()).value();
	EXPECT_EQ(names(out), (std::set<std::string>{"x", "a", "c"}));
	ASSERT_EQ(out.outputs().size(), 2u);
	EXPECT_EQ(out.node(out.outputs()[0]).name, "c");
	EXPECT_EQ(out.node(out.outputs()[1]).name, "a");
}

TEST(DeadNodeElimination, IsIdempotentAndPreservesLiveNodesOnRandomGraphs) {
	for (std::uint64_t seed=1; seed<=100; ++seed) {
		Graph g = testutil::make_random_graph(seed);
		Graph once = eliminate_dead_nodes(g).value();
		Graph twice = eliminate_dead_nodes(once).value();
		ASSERT_TRUE(once.verify().ok()) << "seed " << seed;
		EXPECT_EQ(once.num_nodes(), twice.num_nodes()) << "seed " << seed;
		// Every remaining non-input node feeds an output.
		const auto users = once.compute_users();
		for (NodeId id=0; id<static_cast<NodeId>(once.num_nodes()); ++id) {
			if (once.node(id).op == OpKind::Input) continue;
			EXPECT_TRUE(once.is_output(id) || !users[id].empty()) << "seed " << seed << ": " << once.node(id).name;
		}
	}
}
