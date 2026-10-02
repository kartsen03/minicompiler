#include "minicompiler/graph_builder.hpp"
#include "minicompiler/viz/dot_export.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

using namespace minicompiler;

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
	return haystack.find(needle) != std::string::npos;
}

}

TEST(DotExport, EmitsEveryNodeAndEdge) {
	GraphBuilder b("tiny");
	NodeId x = b.input("x", {4});
	NodeId c = b.scalar("two", 2.0f);
	NodeId y = b.mul(x, c, "y");
	b.output(b.exp(y, "z"));
	Graph g = std::move(b).build().value();

	const std::string dot = to_dot(g);
	EXPECT_EQ(dot.rfind("digraph \"tiny\" {", 0), 0u);
	EXPECT_TRUE(contains(dot, "n0 [label=\"x\\ninput\\nf32[4]\", shape=ellipse"));
	EXPECT_TRUE(contains(dot, "n1 [label=\"two\\nconst 2\\nf32[]\""));
	EXPECT_TRUE(contains(dot, "n0 -> n2;"));
	EXPECT_TRUE(contains(dot, "n1 -> n2;"));
	EXPECT_TRUE(contains(dot, "n2 -> n3;"));
	EXPECT_TRUE(contains(dot, "peripheries=2"));  // the output
	EXPECT_TRUE(contains(dot, "4 nodes, 2 compute (2 elementwise ops, 0 matmul)"));
	EXPECT_EQ(dot.back(), '\n');
}

TEST(DotExport, DrawsOneEdgeForARepeatedOperand) {
	GraphBuilder b("sq");
	NodeId x = b.input("x", {4});
	b.output(b.mul(x, x));
	const std::string dot = to_dot(std::move(b).build().value());
	const std::size_t first = dot.find("n0 -> n1;");
	ASSERT_NE(first, std::string::npos);
	EXPECT_EQ(dot.find("n0 -> n1;", first + 1), std::string::npos);
}

TEST(DotExport, ShowsFusedKernelBodies) {
	Graph g("f");
	NodeId x = g.add_input("x", TensorType({8})).value();
	auto program = std::make_shared<FusedProgram>();
	program->instrs = {FusedInstr::load(0), FusedInstr::imm(0.5f), FusedInstr::binary(OpKind::Mul, 0, 1),
	                   FusedInstr::unary(OpKind::Tanh, 2)};
	NodeId f = g.add_fused({x}, program, TensorType({8}), "act").value();
	ASSERT_TRUE(g.add_output(f).ok());

	const std::string dot = to_dot(g);
	EXPECT_TRUE(contains(dot, "<B>act</B>"));
	EXPECT_TRUE(contains(dot, "fused: 2 ops"));
	EXPECT_TRUE(contains(dot, "r0 = load x"));
	EXPECT_TRUE(contains(dot, "r3 = tanh r2"));

	DotOptions brief;
	brief.show_fused_bodies = false;
	EXPECT_FALSE(contains(to_dot(g, brief), "r0 = load x"));
}

TEST(DotExport, EscapesNamesAndUsesTheTitle) {
	GraphBuilder b("g");
	NodeId x = b.input("say \"hi\"", {2});
	b.output(b.neg(x));
	DotOptions options;
	options.title = "after dne";
	const std::string dot = to_dot(std::move(b).build().value(), options);
	EXPECT_TRUE(contains(dot, "label=\"say \\\"hi\\\"\\ninput"));
	EXPECT_TRUE(contains(dot, "label=\"after dne\\n"));
}

TEST(DotExport, OutputRendersWithGraphviz) {
	if (std::system("dot -V > /dev/null 2>&1") != 0) GTEST_SKIP() << "Graphviz 'dot' is not installed";

	Graph g("render_check");
	NodeId x = g.add_input("x", TensorType({8})).value();
	NodeId c = g.add_constant("c", TensorType({8}), std::vector<float>(8, 1.0f)).value();
	auto program = std::make_shared<FusedProgram>();
	program->instrs = {FusedInstr::load(0), FusedInstr::load(1), FusedInstr::binary(OpKind::Sub, 0, 1),
	                   FusedInstr::imm(0.25f), FusedInstr::binary(OpKind::Mul, 2, 3)};
	NodeId f = g.add_fused({x, c}, program, TensorType({8}), "a<b & \"c\"").value();
	ASSERT_TRUE(g.add_output(f).ok());

	const std::filesystem::path dir = std::filesystem::temp_directory_path();
	const std::string dot_path = (dir / "minicompiler_render_check.dot").string();
	const std::string svg_path = (dir / "minicompiler_render_check.svg").string();
	ASSERT_TRUE(write_dot(g, dot_path).ok());
	const std::string cmd = "dot -Tsvg \"" + dot_path + "\" -o \"" + svg_path + "\"";
	EXPECT_EQ(std::system(cmd.c_str()), 0) << "Graphviz rejected the DOT output";
	EXPECT_GT(std::filesystem::file_size(svg_path), 0u);
	std::filesystem::remove(dot_path);
	std::filesystem::remove(svg_path);
}
