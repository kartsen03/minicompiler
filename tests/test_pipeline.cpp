#include "minicompiler/graph_builder.hpp"
#include "minicompiler/passes/pipeline.hpp"

#include <gtest/gtest.h>

using namespace minicompiler;

TEST(Pipeline, ParsesPassLists) {
	EXPECT_EQ(parse_pipeline("default").value(), default_pipeline());
	EXPECT_EQ(default_pipeline(), (std::vector<std::string>{"dne", "fold", "dne", "fuse", "dne"}));
	EXPECT_TRUE(parse_pipeline("none").value().empty());
	EXPECT_EQ(parse_pipeline("fold,fuse").value(), (std::vector<std::string>{"fold", "fuse"}));
	Result<std::vector<std::string>> bad = parse_pipeline("dne,inline");
	ASSERT_FALSE(bad.ok());
	EXPECT_EQ(bad.error().message, "unknown pass 'inline' (passes: dne, fold, fuse)");
}

TEST(Pipeline, ReportsEveryStageToTheObserver) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {8});
	NodeId c = b.sqrt(b.scalar("four", 4.0f));
	b.output(b.tanh(b.mul(x, c)));
	Graph g = std::move(b).build().value();

	std::vector<std::string> stages;
	std::vector<std::size_t> node_counts;
	std::vector<PassReport> reports;
	Graph out = run_pipeline(g, default_pipeline(), &reports,
	                         [&](std::size_t stage, const std::string& pass, const Graph& graph) {
		                         EXPECT_EQ(stage, stages.size());
		                         stages.push_back(pass);
		                         node_counts.push_back(graph.num_nodes());
	                         }).value();

	EXPECT_EQ(stages, (std::vector<std::string>{"input", "dne", "fold", "dne", "fuse", "dne"}));
	EXPECT_EQ(node_counts, (std::vector<std::size_t>{5, 5, 5, 4, 3, 2}));
	ASSERT_EQ(reports.size(), 5u);
	EXPECT_EQ(reports[1].summary, "folded 1 node(s) into constants");
	EXPECT_EQ(reports[3].summary, "fused 2 op(s) into 1 kernel(s)");
	EXPECT_EQ(compute_stats(out).compute_nodes, 1u);
}

TEST(Pipeline, RejectsAnInvalidInputGraph) {
	Graph g;  // no outputs
	(void)g.add_input("x", TensorType({2}));
	Result<Graph> r = run_pipeline(g, default_pipeline());
	ASSERT_FALSE(r.ok());
	EXPECT_EQ(r.error().message.rfind("input graph is invalid", 0), 0u);
}
