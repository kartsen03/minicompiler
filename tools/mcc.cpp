// mcc: the minicompiler driver. Parses a .mcg graph, runs the pass pipeline,
// and optionally dumps DOT per stage, runs the graph, or benchmarks it.

#include "harness.hpp"

#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/runtime/backend.hpp"
#include "minicompiler/viz/dot_export.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace minicompiler;

namespace {

constexpr const char* kUsage = R"(usage: mcc GRAPH.mcg [options]

compile:
  --dim NAME=VALUE      override a dim declared in the graph (repeatable)
  --passes LIST         "default" (dne,fold,dne,fuse,dne), "none", or a
                        comma-separated list of: dne, fold, fuse
  --print               print the optimized graph
  --stats               print node counts after each pass
  --dump-dot DIR        write DIR/NN_STAGE.dot for the input and after each pass

run:
  --backend NAME        cpu (default) or cuda
  --run                 run once on seeded random inputs; print output checksums
  --seed N              input seed (default 1)
  --dump-outputs FILE   write the outputs as raw float32, concatenated
  --bench               time repeated runs and print JSON
  --warmup N            untimed runs before timing (default 10)
  --reps N              timed runs (default 50)

info:
  --list-backends       print the backends compiled into this build
  --list-passes         print the available passes
)";

struct Options {
	std::string graph_path;
	ParseOptions parse;
	std::string passes = "default";
	bool print = false;
	bool stats = false;
	std::string dot_dir;
	std::string backend = "cpu";
	bool run = false;
	std::uint64_t seed = 1;
	std::string dump_outputs;
	bool bench = false;
	int warmup = 10;
	int reps = 50;
};

int fail(const std::string& message) {
	std::cerr << "mcc: " << message << "\n";
	return 1;
}

std::optional<long long> parse_int(const std::string& s) {
	char* end = nullptr;
	errno = 0;
	const long long v = std::strtoll(s.c_str(), &end, 10);
	if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE) return std::nullopt;
	return v;
}

std::string stats_line(const GraphStats& s) {
	return std::to_string(s.nodes) + " nodes, " + std::to_string(s.compute_nodes) + " compute (" +
	       std::to_string(s.elementwise_ops) + " elementwise ops, " + std::to_string(s.matmuls) + " matmul, " +
	       std::to_string(s.fused_nodes) + " fused kernels)";
}

void write_stats(bench::JsonWriter& json, std::string_view key, const GraphStats& s) {
	json.key(key).begin_object();
	json.field("nodes", s.nodes).field("compute_nodes", s.compute_nodes).field("constants", s.constants);
	json.field("elementwise_ops", s.elementwise_ops).field("matmuls", s.matmuls).field("fused_nodes", s.fused_nodes);
	json.end_object();
}

}

int main(int argc, char** argv) {
	Options opt;
	std::vector<std::string> args(argv + 1, argv + argc);
	for (std::size_t i=0; i<args.size(); ++i) {
		const std::string& a = args[i];
		auto next = [&]() -> std::optional<std::string> {
			if (i + 1 >= args.size()) return std::nullopt;
			return args[++i];
		};
		if (a == "-h" || a == "--help") {
			std::cout << kUsage;
			return 0;
		} else if (a == "--list-backends") {
			for (const std::string& b : available_backends()) std::cout << b << "\n";
			return 0;
		} else if (a == "--list-passes") {
			for (const PassInfo& p : registered_passes()) std::cout << p.name << "\t" << p.description << "\n";
			return 0;
		} else if (a == "--print") {
			opt.print = true;
		} else if (a == "--stats") {
			opt.stats = true;
		} else if (a == "--run") {
			opt.run = true;
		} else if (a == "--bench") {
			opt.bench = true;
		} else if (a == "--dim" || a == "--passes" || a == "--dump-dot" || a == "--backend" || a == "--seed" ||
		           a == "--dump-outputs" || a == "--warmup" || a == "--reps") {
			std::optional<std::string> v = next();
			if (!v) return fail(a + " needs a value");
			if (a == "--dim") {
				const std::size_t eq = v->find('=');
				std::optional<long long> n = eq == std::string::npos ? std::nullopt : parse_int(v->substr(eq + 1));
				if (!n) return fail("--dim expects NAME=VALUE, got '" + *v + "'");
				opt.parse.dims[v->substr(0, eq)] = *n;
			} else if (a == "--passes") {
				opt.passes = *v;
			} else if (a == "--dump-dot") {
				opt.dot_dir = *v;
			} else if (a == "--backend") {
				opt.backend = *v;
			} else if (a == "--dump-outputs") {
				opt.dump_outputs = *v;
			} else {
				std::optional<long long> n = parse_int(*v);
				if (!n || *n < 0) return fail(a + " expects a non-negative integer, got '" + *v + "'");
				if (a != "--seed" && *n > 1000000) return fail(a + " must be at most 1000000");
				if (a == "--seed") opt.seed = static_cast<std::uint64_t>(*n);
				if (a == "--warmup") opt.warmup = static_cast<int>(*n);
				if (a == "--reps") opt.reps = static_cast<int>(*n);
			}
		} else if (!a.empty() && a[0] == '-') {
			return fail("unknown option '" + a + "' (see --help)");
		} else if (opt.graph_path.empty()) {
			opt.graph_path = a;
		} else {
			return fail("more than one graph file given");
		}
	}
	if (opt.graph_path.empty()) {
		std::cerr << kUsage;
		return 1;
	}
	if (opt.bench && opt.reps < 1) return fail("--reps must be at least 1");

	Result<Graph> parsed = parse_graph_file(opt.graph_path, opt.parse);
	if (!parsed.ok()) return fail(parsed.error().message);
	Result<std::vector<std::string>> passes = parse_pipeline(opt.passes);
	if (!passes.ok()) return fail(passes.error().message);

	if (!opt.dot_dir.empty()) {
		std::error_code ec;
		std::filesystem::create_directories(opt.dot_dir, ec);
		if (ec) return fail("cannot create '" + opt.dot_dir + "': " + ec.message());
	}
	std::vector<PassReport> reports;
	std::string dot_error;
	Result<Graph> optimized = run_pipeline(
	    parsed.value(), passes.value(), &reports, [&](std::size_t stage, const std::string& pass, const Graph& g) {
		    if (opt.dot_dir.empty()) return;
		    char prefix[8];
		    std::snprintf(prefix, sizeof(prefix), "%02zu", stage);
		    DotOptions dot;
		    dot.title = g.name() + (stage == 0 ? ": input" : ": after pass " + std::to_string(stage) + " (" + pass + ")");
		    const std::string path = opt.dot_dir + "/" + prefix + "_" + pass + ".dot";
		    Status st = write_dot(g, path, dot);
		    if (!st.ok() && dot_error.empty()) dot_error = st.message();
	    });
	if (!optimized.ok()) return fail(optimized.error().message);
	if (!dot_error.empty()) return fail(dot_error);
	const Graph& graph = optimized.value();

	if (opt.stats) {
		std::cout << "input:       " << stats_line(compute_stats(parsed.value())) << "\n";
		for (const PassReport& r : reports) {
			std::cout << "after " << r.pass << std::string(6 - std::min<std::size_t>(r.pass.size(), 5), ' ')
			          << stats_line(r.after) << "  [" << r.summary << "]\n";
		}
	}
	if (opt.print) std::cout << print_graph(graph);
	if (!opt.run && !opt.bench && opt.dump_outputs.empty()) return 0;

	Result<std::unique_ptr<Backend>> backend = create_backend(opt.backend);
	if (!backend.ok()) return fail(backend.error().message);
	Result<std::unique_ptr<Executable>> exe = (*backend)->compile(graph);
	if (!exe.ok()) return fail(exe.error().message);

	const std::vector<HostTensor> inputs = make_random_inputs(graph, opt.seed);
	std::vector<HostTensor> outputs;
	Status st = (*exe)->run(inputs, outputs);
	if (!st.ok()) return fail(st.message());

	if (opt.run) {
		for (std::size_t k=0; k<outputs.size(); ++k) {
			double sum = 0.0, abs_sum = 0.0;
			for (float v : outputs[k].data) {
				sum += v;
				abs_sum += std::fabs(v);
			}
			std::cout << graph.node(graph.outputs()[k]).name << " : " << to_string(outputs[k].type) << "  sum "
			          << sum << "  abs-sum " << abs_sum << "\n";
		}
	}
	if (!opt.dump_outputs.empty()) {
		std::ofstream out(opt.dump_outputs, std::ios::binary);
		for (const HostTensor& t : outputs) {
			out.write(reinterpret_cast<const char*>(t.data.data()),
			          static_cast<std::streamsize>(t.data.size() * sizeof(float)));
		}
		if (!out) return fail("failed writing '" + opt.dump_outputs + "'");
	}
	if (opt.bench) {
		Executable& e = *exe.value();
		const bench::TimingStats t = bench::time_cpu([&] { (void)e.run(inputs, outputs); }, opt.warmup, opt.reps);
		bench::JsonWriter json;
		json.begin_object();
		json.field("graph", graph.name()).field("backend", opt.backend).field("passes", opt.passes);
		json.key("inputs").begin_object();
		for (NodeId id : graph.inputs()) json.field(graph.node(id).name, to_string(graph.node(id).type));
		json.end_object();
		write_stats(json, "input_stats", compute_stats(parsed.value()));
		write_stats(json, "optimized_stats", compute_stats(graph));
		json.field("intermediate_bytes", e.intermediate_bytes());
		json.timing("timing", t);
		bench::write_environment(json);
		json.end_object();
		std::cout << json.str();
	}
	return 0;
}
