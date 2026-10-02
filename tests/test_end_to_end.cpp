// End-to-end checks: the optimized graph computes what the unoptimized graph
// computes, on the benchmark graphs and on random graphs.

#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/runtime/backend.hpp"

#include "test_util.hpp"

#include <gtest/gtest.h>

#include <algorithm>

using namespace minicompiler;

namespace {

// The optimized graph runs the same Eigen kernels on the same values, in the
// same per-element order: folding evaluates constant ops early, fusion runs
// ops block by block, and no pass reassociates arithmetic. The only source of
// difference is Eigen's alignment-dependent split between scalar and SIMD
// code for transcendental functions, an ulp or two per op. The tolerance
// below allows two orders of magnitude more than that:
//   |optimized - unoptimized| <= 1e-5 * |unoptimized| + 1e-5 * max|unoptimized|
constexpr double kRtol = 1e-5;
constexpr double kAtolScale = 1e-5;

std::string graph_path(const std::string& name) {
	return std::string(MINICOMPILER_SOURCE_DIR) + "/bench/graphs/" + name + ".mcg";
}

std::vector<HostTensor> run_cpu(const Graph& g, const std::vector<HostTensor>& inputs) {
	auto exe = create_backend("cpu").value()->compile(g);
	EXPECT_TRUE(exe.ok());
	std::vector<HostTensor> out;
	Status st = (*exe)->run(inputs, out);
	EXPECT_TRUE(st.ok()) << st.message();
	return out;
}

void expect_equivalent(const Graph& reference, const Graph& optimized, std::uint64_t seed, const std::string& what) {
	auto inputs = make_random_inputs(reference, seed);
	const auto expected = run_cpu(reference, inputs);
	const auto actual = run_cpu(optimized, inputs);
	ASSERT_EQ(actual.size(), expected.size()) << what;
	for (std::size_t k=0; k<actual.size(); ++k) {
		ASSERT_EQ(actual[k].type, expected[k].type) << what;
		const double atol = kAtolScale * std::max(1.0, testutil::max_abs(expected[k].data));
		EXPECT_TRUE(testutil::all_close(actual[k].data, expected[k].data, kRtol, atol))
		    << what << ", output " << k << ", seed " << seed;
	}
}

struct StageCounts {
	std::size_t nodes;
	std::size_t compute_nodes;
};

std::vector<StageCounts> stage_counts(const Graph& g, Graph* optimized) {
	std::vector<StageCounts> counts;
	*optimized = run_pipeline(g, default_pipeline(), nullptr, [&](std::size_t, const std::string&, const Graph& s) {
		             const GraphStats st = compute_stats(s);
		             counts.push_back({st.nodes, st.compute_nodes});
	             }).value();
	return counts;
}

bool operator==(const StageCounts& a, const StageCounts& b) {
	return a.nodes == b.nodes && a.compute_nodes == b.compute_nodes;
}

std::ostream& operator<<(std::ostream& os, const StageCounts& c) {
	return os << "{" << c.nodes << " nodes, " << c.compute_nodes << " compute}";
}

}

// Expected counts at each stage: input, dne, fold, dne, fuse, dne.

TEST(EndToEnd, GeluChain) {
	ParseOptions options;
	options.dims = {{"M", 64}, {"N", 300}};
	Graph g = parse_graph_file(graph_path("gelu_chain"), options).value();
	Graph opt;
	const auto counts = stage_counts(g, &opt);
	EXPECT_EQ(counts, (std::vector<StageCounts>{{17, 11}, {17, 11}, {17, 9}, {14, 9}, {6, 1}, {2, 1}}));
	EXPECT_EQ(compute_stats(opt).fused_nodes, 1u);
	EXPECT_EQ(compute_stats(opt).elementwise_ops, 9u);
	for (std::uint64_t seed : {1, 2, 3}) expect_equivalent(g, opt, seed, "gelu_chain");
}

TEST(EndToEnd, MatmulBiasRelu) {
	ParseOptions options;
	options.dims = {{"M", 33}, {"K", 70}, {"N", 129}};
	Graph g = parse_graph_file(graph_path("matmul_bias_relu"), options).value();
	Graph opt;
	const auto counts = stage_counts(g, &opt);
	EXPECT_EQ(counts, (std::vector<StageCounts>{{6, 3}, {6, 3}, {6, 3}, {6, 3}, {5, 2}, {5, 2}}));
	for (std::uint64_t seed : {1, 2, 3}) expect_equivalent(g, opt, seed, "matmul_bias_relu");
}

TEST(EndToEnd, MlpBlock) {
	ParseOptions options;
	options.dims = {{"B", 16}, {"D", 64}, {"H", 256}};
	Graph g = parse_graph_file(graph_path("mlp_block"), options).value();
	Graph opt;
	const auto counts = stage_counts(g, &opt);
	// fold turns var+eps, its sqrt, 2/pi and its sqrt into constants (22 -> 18
	// compute); dne then drops var, eps, two, pi and the two intermediate
	// constants (37 -> 31); fuse merges 14 + 2 ops into two kernels (-> 17);
	// dne drops the four scalars baked into the kernels (-> 13).
	EXPECT_EQ(counts, (std::vector<StageCounts>{{37, 22}, {37, 22}, {37, 18}, {31, 18}, {17, 4}, {13, 4}}));
	const GraphStats s = compute_stats(opt);
	EXPECT_EQ(s.matmuls, 2u);
	EXPECT_EQ(s.fused_nodes, 2u);
	EXPECT_EQ(s.elementwise_ops, 16u);  // 14 in the BatchNorm+GELU kernel, 2 in bias+residual
	for (std::uint64_t seed : {1, 2, 3}) expect_equivalent(g, opt, seed, "mlp_block");
}

TEST(EndToEnd, BenchmarkGraphsParseAtTheirDefaultSizes) {
	for (const char* name : {"gelu_chain", "matmul_bias_relu", "mlp_block"}) {
		Result<Graph> g = parse_graph_file(graph_path(name));
		EXPECT_TRUE(g.ok()) << name << ": " << (g.ok() ? "" : g.error().message);
	}
}

TEST(EndToEnd, RandomGraphsThroughTheDefaultPipeline) {
	std::size_t removed = 0;
	for (std::uint64_t seed=1; seed<=300; ++seed) {
		Graph g = testutil::make_random_graph(seed, {24, true});
		Graph opt = run_pipeline(g, default_pipeline()).value();
		removed += g.num_nodes() - opt.num_nodes();
		expect_equivalent(g, opt, seed, "random graph " + std::to_string(seed));
	}
	EXPECT_GT(removed, 1000u);
}
