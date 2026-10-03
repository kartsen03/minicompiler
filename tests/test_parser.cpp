#include "minicompiler/parser.hpp"
#include "minicompiler/random.hpp"

#include <gtest/gtest.h>

#include <cstring>

using namespace minicompiler;

namespace {

constexpr const char* kExample = R"(
# A small MLP layer.
graph layer
dim B = 2
input x : f32[B, 3]
const w : f32[3, 4] = uniform(seed=7, lo=-0.5, hi=0.5)
const b : f32[4] = [1, 2, 3, 4]
const half : f32[] = 0.5
h = matmul x, w
hb = add h, b      # bias broadcast over rows
y = mul hb, half
output y, h
)";

std::string parse_error(std::string_view text, const ParseOptions& options = {}) {
	Result<Graph> g = parse_graph(text, options);
	return g.ok() ? std::string("<parsed>") : g.error().message;
}

}

TEST(Parser, ParsesEveryStatementKind) {
	Result<Graph> g = parse_graph(kExample);
	ASSERT_TRUE(g.ok()) << g.error().message;
	EXPECT_EQ(g->name(), "layer");
	EXPECT_EQ(g->num_nodes(), 7u);
	EXPECT_EQ(g->inputs().size(), 1u);
	ASSERT_EQ(g->outputs().size(), 2u);
	EXPECT_EQ(g->node(g->outputs()[0]).name, "y");

	const Node& x = g->node(*g->find("x"));
	EXPECT_EQ(x.op, OpKind::Input);
	EXPECT_EQ(x.type, TensorType({2, 3}));

	const Node& b = g->node(*g->find("b"));
	EXPECT_EQ(*b.constant, (std::vector<float>{1, 2, 3, 4}));
	EXPECT_EQ(g->node(*g->find("half")).constant->at(0), 0.5f);

	const Node& w = g->node(*g->find("w"));
	EXPECT_EQ(*w.constant, uniform_values(12, 7, -0.5f, 0.5f));

	const Node& y = g->node(*g->find("y"));
	EXPECT_EQ(y.op, OpKind::Mul);
	EXPECT_EQ(y.type, TensorType({2, 4}));
}

TEST(Parser, DimOverridesChangeShapes) {
	ParseOptions options;
	options.dims["B"] = 64;
	Result<Graph> g = parse_graph(kExample, options);
	ASSERT_TRUE(g.ok()) << g.error().message;
	EXPECT_EQ(g->node(*g->find("y")).type, TensorType({64, 4}));

	options.dims["Q"] = 1;
	EXPECT_NE(parse_error(kExample, options).find("no dim 'Q'"), std::string::npos);
}

TEST(Parser, ReportsErrorsWithLineNumbers) {
	EXPECT_EQ(parse_error("input x : f32[4]\ny = frobnicate x\noutput y"), "line 2: unknown op 'frobnicate'");
	EXPECT_EQ(parse_error("input x : f32[4]\ny = relu z\noutput y"), "line 2: unknown value 'z'");
	EXPECT_EQ(parse_error("input x : f32[4]\nx = relu x\noutput x"), "line 2: 'x' is defined twice");
	EXPECT_EQ(parse_error("input x : i32[4]\noutput x"), "line 1: unsupported element type 'i32' (only f32)");
	EXPECT_EQ(parse_error("input x : f32[N]\noutput x"), "line 1: unknown dim 'N'");
	EXPECT_EQ(parse_error("const c : f32[3] = [1, 2]\noutput c"), "line 1: expected 3 values for f32[3], got 2");
	EXPECT_EQ(parse_error("input x : f32[4]\n"), "the graph declares no outputs");
	EXPECT_EQ(parse_error("input x : f32[4] ;"), "line 1: unexpected character ';'");
	EXPECT_EQ(parse_error("input x : f32[4]\ny = relu x x"), "line 2: unexpected 'x'");

	// Shape errors from the IR carry the value's name.
	const std::string err = parse_error("input a : f32[4, 8]\ninput b : f32[4]\nc = add a, b\noutput c");
	EXPECT_EQ(err.rfind("line 3: c: add: cannot broadcast", 0), 0u) << err;
}

TEST(Parser, RejectsInvalidShapesBeforeAllocating) {
	EXPECT_EQ(parse_error("const c : f32[-1] = 0\noutput c"),
	          "line 1: invalid shape [-1]: dimensions must be >= 1 and the total at most 2^31 - 1 elements");
	for (const char* text : {"const c : f32[-1] = uniform(seed=1, lo=0, hi=1)\noutput c",
	                         "input x : f32[7, 7905747460161236407]\noutput x",
	                         "const c : f32[4294967296, 4294967296] = 1\noutput c",
	                         "input x : f32[4, 0]\noutput x"}) {
		EXPECT_EQ(parse_error(text).rfind("line 1: invalid shape", 0), 0u) << text;
	}
}

TEST(Parser, KeepsLargeSeedsExact) {
	// 2^53 and 2^53 + 1 are the same double; the seed must stay an integer.
	Graph a = parse_graph("const c : f32[8] = uniform(seed=9007199254740992, lo=0, hi=1)\noutput c").value();
	Graph b = parse_graph("const c : f32[8] = uniform(seed=9007199254740993, lo=0, hi=1)\noutput c").value();
	EXPECT_NE(*a.node(0).constant, *b.node(0).constant);
	EXPECT_EQ(*b.node(0).constant, uniform_values(8, 9007199254740993ULL, 0.0f, 1.0f));
}

TEST(Parser, IgnoresCommentsBlankLinesAndCarriageReturns) {
	Result<Graph> g = parse_graph("# header\r\n\r\ninput x : f32[2]   # trailing\r\ny = exp x\r\noutput y\r\n");
	ASSERT_TRUE(g.ok()) << g.error().message;
	EXPECT_EQ(g->num_nodes(), 2u);
}

TEST(Parser, AcceptsScalarsAndSignedNumbers) {
	Result<Graph> g = parse_graph("const c : f32[] = -1.5e-1\nconst d : f32[2] = [-2, +.5]\ny = mul c, d\noutput y");
	ASSERT_TRUE(g.ok()) << g.error().message;
	EXPECT_EQ(g->node(*g->find("c")).constant->at(0), -0.15f);
	EXPECT_EQ(*g->node(*g->find("d")).constant, (std::vector<float>{-2.0f, 0.5f}));
}

TEST(Printer, ListsNodesWithTypes) {
	Graph g = parse_graph(kExample).value();
	const std::string text = print_graph(g);
	EXPECT_NE(text.find("graph layer\n"), std::string::npos);
	EXPECT_NE(text.find("%0 x = input : f32[2,3]"), std::string::npos) << text;
	EXPECT_NE(text.find("%2 b = const [1, 2, 3, 4] : f32[4]"), std::string::npos) << text;
	EXPECT_NE(text.find("%6 y = mul %5, %3 : f32[2,4]  (output)"), std::string::npos) << text;
}

TEST(Random, IsDeterministicAndInRange) {
	const std::vector<float> a = uniform_values(10000, 123, -2.0f, 3.0f);
	EXPECT_EQ(a, uniform_values(10000, 123, -2.0f, 3.0f));
	EXPECT_NE(a, uniform_values(10000, 124, -2.0f, 3.0f));
	for (float v : a) {
		EXPECT_GE(v, -2.0f);
		EXPECT_LT(v, 3.0f);
	}
}

TEST(Random, MatchesTheValuesPinnedForPython) {
	// Expected bits were computed by an independent Python implementation;
	// bench/mcgraph.py checks its NumPy port against the same values, so both
	// languages generate identical weights and inputs.
	const std::vector<float> v = uniform_values(4, 42, -0.3f, 0.7f);
	const std::uint32_t expected[4] = {0x3EE214CAu, 0xBE0F73A8u, 0xBCAF4CC0u, 0x3D350140u};
	for (int i=0; i<4; ++i) {
		std::uint32_t bits;
		std::memcpy(&bits, &v[i], 4);
		EXPECT_EQ(bits, expected[i]) << "value " << i << " = " << v[i];
	}
}
