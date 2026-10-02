// What the optimization passes buy on the Eigen CPU backend.
//
// For each benchmark graph and size, the same graph is compiled four ways:
// no passes, dead-node elimination only, DNE + constant folding, and the
// full pipeline (with fusion). Each variant is checked against the
// unoptimized outputs, then all four are timed interleaved (one run of each
// per round, rotating the order) so slow drift such as thermal throttling
// affects them equally. Results go to results/cpu/passes.json.

#include "harness.hpp"

#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/runtime/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace minicompiler;

namespace {

struct Config {
	const char* graph;
	std::map<std::string, std::int64_t> dims;
	const char* size;
};

struct Variant {
	const char* name;
	const char* passes;
};

const std::vector<Config> kConfigs = {
	{"gelu_chain", {{"M", 1}, {"N", 4096}}, "1x4096 (16 KB)"},
	{"gelu_chain", {{"M", 64}, {"N", 4096}}, "64x4096 (1 MB)"},
	{"gelu_chain", {{"M", 256}, {"N", 4096}}, "256x4096 (4 MB)"},
	{"gelu_chain", {{"M", 2048}, {"N", 4096}}, "2048x4096 (32 MB)"},
	{"matmul_bias_relu", {{"M", 64}, {"K", 1024}, {"N", 1024}}, "64x1024 @ 1024x1024"},
	{"matmul_bias_relu", {{"M", 256}, {"K", 1024}, {"N", 1024}}, "256x1024 @ 1024x1024"},
	{"mlp_block", {{"B", 32}, {"D", 512}, {"H", 2048}}, "B=32, 512->2048->512"},
	{"mlp_block", {{"B", 128}, {"D", 512}, {"H", 2048}}, "B=128, 512->2048->512"},
	{"mlp_block", {{"B", 512}, {"D", 512}, {"H", 2048}}, "B=512, 512->2048->512"},
};

const std::vector<Variant> kVariants = {
	{"none", "none"},
	{"dne", "dne"},
	{"dne+fold", "dne,fold,dne"},
	{"all", "default"},
};

// Normwise difference: max |a - b| / max(1, max |b|).
double normwise_diff(const std::vector<HostTensor>& a, const std::vector<HostTensor>& b) {
	double worst = 0.0;
	for (std::size_t k=0; k<a.size(); ++k) {
		double scale = 1.0, diff = 0.0;
		for (std::size_t i=0; i<b[k].data.size(); ++i) {
			scale = std::max(scale, static_cast<double>(std::fabs(b[k].data[i])));
			diff = std::max(diff, static_cast<double>(std::fabs(a[k].data[i] - b[k].data[i])));
		}
		worst = std::max(worst, diff / scale);
	}
	return worst;
}

struct Compiled {
	Graph graph;
	std::unique_ptr<Executable> exe;
	std::vector<HostTensor> outputs;
	std::vector<double> samples_ms;
};

}

int main(int argc, char** argv) {
	std::string out_path = "results/cpu/passes.json";
	std::string graph_dir = std::string(MINICOMPILER_SOURCE_DIR) + "/bench/graphs";
	double seconds_per_config = 3.0;
	for (int i=1; i<argc; ++i) {
		const std::string a = argv[i];
		if (a == "--out" && i + 1 < argc) {
			out_path = argv[++i];
		} else if (a == "--graphs" && i + 1 < argc) {
			graph_dir = argv[++i];
		} else if (a == "--quick") {
			seconds_per_config = 0.2;
		} else {
			std::cerr << "usage: bench_cpu [--out FILE] [--graphs DIR] [--quick]\n";
			return 1;
		}
	}

	auto backend = create_backend("cpu");
	bench::JsonWriter json;
	json.begin_object();
	json.field("benchmark", "passes off vs on, Eigen CPU backend");
	json.field("method", "interleaved rounds (one run per variant per round, rotating order) after 5 warmup runs "
	                     "each; median of the rounds");
	bench::write_environment(json);
	json.key("results").begin_array();

	std::printf("%-17s %-24s %-9s %7s %8s %11s %9s\n", "graph", "size", "variant", "nodes", "compute", "median ms",
	            "speedup");
	for (const Config& cfg : kConfigs) {
		ParseOptions parse;
		parse.dims = cfg.dims;
		Result<Graph> input = parse_graph_file(graph_dir + "/" + cfg.graph + ".mcg", parse);
		if (!input.ok()) {
			std::cerr << input.error().message << "\n";
			return 1;
		}
		const std::vector<HostTensor> inputs = make_random_inputs(input.value(), 1);

		std::vector<Compiled> variants;
		for (const Variant& v : kVariants) {
			Result<Graph> g = run_pipeline(input.value(), parse_pipeline(v.passes).value());
			if (!g.ok()) {
				std::cerr << g.error().message << "\n";
				return 1;
			}
			Compiled c{std::move(g).value(), nullptr, {}, {}};
			c.exe = std::move((*backend)->compile(c.graph)).value();
			Status st = c.exe->run(inputs, c.outputs);
			if (!st.ok()) {
				std::cerr << st.message() << "\n";
				return 1;
			}
			variants.push_back(std::move(c));
		}

		// Warm up, then size the number of rounds from the unoptimized run time.
		for (Compiled& c : variants) {
			for (int i=0; i<5; ++i) (void)c.exe->run(inputs, c.outputs);
		}
		const bench::TimingStats probe = bench::time_cpu([&] { (void)variants[0].exe->run(inputs, variants[0].outputs); }, 0, 5);
		const double per_round_ms = std::max(probe.median_ms * static_cast<double>(variants.size()), 1e-3);
		const int rounds = std::clamp(static_cast<int>(seconds_per_config * 1000.0 / per_round_ms), 20, 2000);
		for (int r=0; r<rounds; ++r) {
			for (std::size_t j=0; j<variants.size(); ++j) {
				Compiled& c = variants[(j + static_cast<std::size_t>(r)) % variants.size()];
				const auto t0 = std::chrono::steady_clock::now();
				(void)c.exe->run(inputs, c.outputs);
				const auto t1 = std::chrono::steady_clock::now();
				c.samples_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
			}
		}

		json.begin_object();
		json.field("graph", cfg.graph).field("size", cfg.size);
		json.key("dims").begin_object();
		for (const auto& [name, value] : cfg.dims) json.field(name, value);
		json.end_object();
		json.key("variants").begin_array();
		const double base = bench::summarize(variants[0].samples_ms, 5).median_ms;
		for (std::size_t j=0; j<variants.size(); ++j) {
			Compiled& c = variants[j];
			const bench::TimingStats t = bench::summarize(c.samples_ms, 5);
			const GraphStats s = compute_stats(c.graph);
			const double diff = normwise_diff(c.outputs, variants[0].outputs);
			json.begin_object();
			json.field("variant", kVariants[j].name).field("passes", kVariants[j].passes);
			json.field("nodes", s.nodes).field("compute_nodes", s.compute_nodes);
			json.field("elementwise_ops", s.elementwise_ops).field("matmuls", s.matmuls).field("fused_nodes", s.fused_nodes);
			json.field("intermediate_bytes", c.exe->intermediate_bytes());
			json.field("max_normwise_diff_vs_none", diff);
			json.timing("timing", t);
			json.field("speedup_vs_none", base / t.median_ms);
			json.end_object();
			std::printf("%-17s %-24s %-9s %7zu %8zu %11.4f %8.2fx%s\n", cfg.graph, cfg.size, kVariants[j].name, s.nodes,
			            s.compute_nodes, t.median_ms, base / t.median_ms, diff > 1e-5 ? "  OUTPUT MISMATCH" : "");
			if (diff > 1e-5) {
				std::cerr << "optimized outputs differ from the unoptimized ones by " << diff << "\n";
				return 1;
			}
		}
		json.end_array();
		json.end_object();
	}
	json.end_array();
	json.end_object();

	std::ofstream out(out_path, std::ios::binary);
	out << json.str();
	if (!out) {
		std::cerr << "cannot write " << out_path << "\n";
		return 1;
	}
	std::cout << "wrote " << out_path << "\n";
	return 0;
}
