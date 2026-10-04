// GPU benchmarks for the CUDA backend, timed with CUDA events.
//
//   elementwise  the GELU chain from 16 KB to 256 MB, unfused (one kernel per
//                op) against fused (one kernel), with achieved memory
//                bandwidth as a percentage of the device's peak
//
// Every variant is compiled once, its inputs are uploaded once, and then the
// variants are timed in interleaved rounds (one enqueue of each per round,
// rotating order): each enqueue is bracketed by CUDA events on one stream.
// Results go to results/gpu/<suite>.json with the device, clocks and commit.

#include "harness.hpp"

#include "minicompiler/cuda/cuda_backend.hpp"
#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/runtime/backend.hpp"

#include <cuda_runtime.h>

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

#define CHECK_CUDA(expr)                                                                  \
	do {                                                                                  \
		const cudaError_t err_ = (expr);                                                  \
		if (err_ != cudaSuccess) {                                                        \
			std::fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(err_));     \
			std::exit(1);                                                                 \
		}                                                                                 \
	} while (0)

template <typename T>
T must(Result<T> r, const char* what) {
	if (!r.ok()) {
		std::fprintf(stderr, "%s: %s\n", what, r.error().message.c_str());
		std::exit(1);
	}
	return std::move(r).value();
}

void must(const Status& st, const char* what) {
	if (!st.ok()) {
		std::fprintf(stderr, "%s: %s\n", what, st.message().c_str());
		std::exit(1);
	}
}

// Bytes each kernel reads and writes, summed over the compiled program:
// every non-scalar operand once and the output once. Scalars stay in cache.
double bytes_moved(const Graph& g) {
	double bytes = 0;
	for (const Node& n : g.nodes()) {
		if (!is_compute(n.op)) continue;
		bytes += static_cast<double>(n.type.size_bytes());
		for (NodeId in : n.inputs) {
			const TensorType& t = g.node(in).type;
			if (t.num_elements() > 1) bytes += static_cast<double>(t.size_bytes());
		}
	}
	return bytes;
}

struct Variant {
	std::string name;
	Graph graph;
	std::unique_ptr<cuda::CudaExecutable> exe;
	std::vector<double> samples_ms;
	double bytes = 0;
};

// Times the variants in interleaved rounds with CUDA events on `stream`.
void time_interleaved(std::vector<Variant>& variants, cudaStream_t stream, int warmup, double seconds) {
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));
	auto once = [&](Variant& v) {
		CHECK_CUDA(cudaEventRecord(start, stream));
		must(v.exe->enqueue(stream), "enqueue");
		CHECK_CUDA(cudaEventRecord(stop, stream));
		CHECK_CUDA(cudaEventSynchronize(stop));
		float ms = 0;
		CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
		return static_cast<double>(ms);
	};
	for (Variant& v : variants) {
		for (int i=0; i<warmup; ++i) once(v);
	}
	double round_ms = 0;
	for (Variant& v : variants) round_ms += once(v);
	const int rounds = std::clamp(static_cast<int>(seconds * 1000.0 / std::max(round_ms, 1e-3)), 30, 5000);
	for (int r=0; r<rounds; ++r) {
		for (std::size_t j=0; j<variants.size(); ++j) {
			Variant& v = variants[(j + static_cast<std::size_t>(r)) % variants.size()];
			v.samples_ms.push_back(once(v));
		}
	}
	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));
}

double max_normwise_diff(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
	double worst = 0;
	for (std::size_t k=0; k<a.size(); ++k) {
		double scale = 1.0, diff = 0.0;
		for (std::size_t i=0; i<a[k].size(); ++i) {
			scale = std::max(scale, static_cast<double>(std::fabs(b[k][i])));
			diff = std::max(diff, static_cast<double>(std::fabs(a[k][i] - b[k][i])));
		}
		worst = std::max(worst, diff / scale);
	}
	return worst;
}

void write_device(bench::JsonWriter& json, const cuda::DeviceInfo& d, const std::vector<double>& clocks) {
	json.key("device").begin_object();
	json.field("name", d.name).field("compute_capability", std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor));
	json.field("sm_count", d.sm_count).field("fp32_lanes_per_sm", d.fp32_lanes_per_sm);
	json.field("sm_clock_mhz_cuda_attribute", d.sm_clock_mhz).field("max_sm_clock_mhz_nvml", d.max_sm_clock_mhz);
	json.field("mem_clock_mhz", d.mem_clock_mhz).field("bus_width_bits", d.bus_width_bits);
	json.field("peak_bandwidth_gbs", d.peak_bandwidth_gbs());
	json.field("cuda_runtime", d.runtime_version).field("cuda_driver", d.driver_version);
	if (!clocks.empty()) {
		std::vector<double> c = clocks;
		std::sort(c.begin(), c.end());
		json.key("sm_clock_mhz_during_run").begin_object();
		json.field("min", c.front()).field("median", c[c.size() / 2]).field("max", c.back()).end_object();
	}
	json.end_object();
}

int run_elementwise(const std::string& graph_dir, const std::string& out_path, double seconds) {
	const cuda::DeviceInfo device = must(cuda::query_device(0), "query_device");
	CHECK_CUDA(cudaSetDevice(0));
	cudaStream_t stream;
	CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	const double peak = device.peak_bandwidth_gbs();
	std::vector<double> clocks;

	struct Size {
		std::int64_t m;
		const char* label;
	};
	const std::vector<Size> sizes = {{1, "1x4096 (16 KB)"},       {16, "16x4096 (256 KB)"},
	                                 {256, "256x4096 (4 MB)"},    {1024, "1024x4096 (16 MB)"},
	                                 {4096, "4096x4096 (64 MB)"}, {16384, "16384x4096 (256 MB)"}};
	const std::vector<std::pair<const char*, const char*>> pipelines = {{"unfused", "dne,fold,dne"},
	                                                                    {"fused", "default"}};

	bench::JsonWriter json;
	json.begin_object();
	json.field("benchmark", "GELU chain on the GPU: one kernel per op (unfused) vs one fused kernel");
	json.field("method", "CUDA events around each enqueue of the whole graph on one stream; inputs already on the "
	                     "device; 10 warmup enqueues per variant, then interleaved rounds (rotating order); "
	                     "time = median over rounds; speedup = median over rounds of unfused/fused in the same round; "
	                     "bandwidth = bytes each kernel reads and writes / median time");
	bench::write_environment(json);
	json.key("results").begin_array();
	std::printf("%-22s %-8s %8s %12s %12s %10s %9s\n", "size", "variant", "kernels", "median ms", "GB/s", "% of peak",
	            "speedup");
	for (const Size& size : sizes) {
		ParseOptions options;
		options.dims = {{"M", size.m}, {"N", 4096}};
		const Graph input = must(parse_graph_file(graph_dir + "/gelu_chain.mcg", options), "parse");
		const std::vector<HostTensor> inputs = make_random_inputs(input, 1);
		std::vector<Variant> variants;
		for (const auto& [name, passes] : pipelines) {
			Variant v{name, must(run_pipeline(input, must(parse_pipeline(passes), "pipeline")), "passes"), nullptr, {}, 0};
			v.exe = must(cuda::compile_for_cuda(v.graph), "compile");
			v.bytes = bytes_moved(v.graph);
			std::vector<const float*> in;
			for (const HostTensor& t : inputs) in.push_back(t.data.data());
			must(v.exe->upload_inputs(in), "upload");
			variants.push_back(std::move(v));
		}
		clocks.push_back(cuda::current_sm_clock_mhz(0));
		time_interleaved(variants, stream, 10, seconds);
		clocks.push_back(cuda::current_sm_clock_mhz(0));

		// The two variants must agree before their times mean anything.
		std::vector<std::vector<std::vector<float>>> outs;
		for (Variant& v : variants) {
			std::vector<std::vector<float>> o(1, std::vector<float>(static_cast<std::size_t>(size.m) * 4096));
			std::vector<float*> ptrs = {o[0].data()};
			must(v.exe->download_outputs(ptrs), "download");
			outs.push_back(std::move(o));
		}
		const double diff = max_normwise_diff(outs[1], outs[0]);

		json.begin_object();
		json.field("size", size.label).field("elements", static_cast<std::int64_t>(size.m * 4096));
		json.field("max_normwise_diff_fused_vs_unfused", diff);
		json.key("variants").begin_array();
		std::vector<double> paired;
		for (std::size_t r=0; r<variants[0].samples_ms.size(); ++r) {
			paired.push_back(variants[0].samples_ms[r] / variants[1].samples_ms[r]);
		}
		std::sort(paired.begin(), paired.end());
		for (std::size_t j=0; j<variants.size(); ++j) {
			Variant& v = variants[j];
			const bench::TimingStats t = bench::summarize(v.samples_ms, 10);
			const double gbs = v.bytes / (t.median_ms * 1e-3) / 1e9;
			json.begin_object();
			json.field("variant", v.name).field("passes", pipelines[j].second);
			json.field("kernel_launches", v.exe->kernel_launches());
			json.field("bytes_moved", v.bytes);
			json.timing("timing", t);
			json.field("achieved_bandwidth_gbs", gbs).field("percent_of_peak_bandwidth", 100.0 * gbs / peak);
			if (j == 1) {
				json.key("speedup_paired").begin_object();
				json.field("median", bench::percentile(paired, 0.5)).field("p10", bench::percentile(paired, 0.1));
				json.field("p90", bench::percentile(paired, 0.9)).end_object();
			}
			json.end_object();
			std::printf("%-22s %-8s %8zu %12.4f %12.1f %9.1f%% %8s\n", size.label, v.name.c_str(),
			            v.exe->kernel_launches(), t.median_ms, gbs, 100.0 * gbs / peak,
			            j == 1 ? (std::to_string(bench::percentile(paired, 0.5)).substr(0, 5) + "x").c_str() : "");
		}
		json.end_array();
		json.end_object();
		if (diff > 1e-5) {
			std::fprintf(stderr, "fused and unfused outputs differ by %g\n", diff);
			return 1;
		}
	}
	json.end_array();
	write_device(json, device, clocks);
	json.end_object();
	CHECK_CUDA(cudaStreamDestroy(stream));

	std::ofstream out(out_path, std::ios::binary);
	out << json.str();
	std::printf("peak bandwidth %.1f GB/s (%s, %.0f MHz memory clock, %d-bit bus); wrote %s\n", peak,
	            device.name.c_str(), device.mem_clock_mhz, device.bus_width_bits, out_path.c_str());
	return out ? 0 : 1;
}

}

int main(int argc, char** argv) {
	std::string suite = "elementwise";
	std::string out_path;
	std::string graph_dir = std::string(MINICOMPILER_SOURCE_DIR) + "/bench/graphs";
	double seconds = 2.0;
	for (int i=1; i<argc; ++i) {
		const std::string a = argv[i];
		if (a == "--suite" && i + 1 < argc) {
			suite = argv[++i];
		} else if (a == "--out" && i + 1 < argc) {
			out_path = argv[++i];
		} else if (a == "--graphs" && i + 1 < argc) {
			graph_dir = argv[++i];
		} else if (a == "--quick") {
			seconds = 0.2;
		} else {
			std::cerr << "usage: bench_cuda [--suite elementwise] [--out FILE] [--graphs DIR] [--quick]\n";
			return 1;
		}
	}
	if (out_path.empty()) out_path = "results/gpu/" + suite + ".json";
	if (suite == "elementwise") return run_elementwise(graph_dir, out_path, seconds);
	std::cerr << "unknown suite '" << suite << "'\n";
	return 1;
}
