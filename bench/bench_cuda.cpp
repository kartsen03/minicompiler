// GPU benchmarks for the CUDA backend, timed with CUDA events.
//
//   elementwise  the GELU chain from 16 KB to 256 MB, unfused (one kernel per
//                op) against fused (one kernel), with achieved memory
//                bandwidth as a percentage of the device's peak
//   matmul       naive, shared-memory tiled and register-tiled kernels against
//                cuBLAS, with GFLOP/s as a percentage of the device's peak
//
// Every variant is compiled once, its inputs are uploaded once, and then the
// variants are timed in interleaved rounds (one enqueue of each per round,
// rotating order): each enqueue is bracketed by CUDA events on one stream.
// Results go to results/gpu/<suite>.json with the device, clocks and commit.

#include "harness.hpp"

#include "minicompiler/cuda/cuda_backend.hpp"
#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/random.hpp"
#include "minicompiler/runtime/backend.hpp"

#include "cuda/matmul_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
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

// A laptop GPU idles at a few hundred MHz and needs about half a second of
// sustained work to reach its boost clock, so warm-up is measured in time:
// keep calling `round` until `seconds` have passed.
template <typename Fn>
void warm_up_for(double seconds, Fn&& round) {
	const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
	while (std::chrono::steady_clock::now() < until) round();
}

// Times the variants in interleaved rounds with CUDA events on `stream`,
// sampling the SM clock after every round (while the GPU is under load).
void time_interleaved(std::vector<Variant>& variants, cudaStream_t stream, int warmup, double warmup_seconds,
                      double seconds, std::vector<double>& clocks) {
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
	warm_up_for(warmup_seconds, [&] {
		for (Variant& v : variants) once(v);
	});
	double round_ms = 0;
	for (Variant& v : variants) round_ms += once(v);
	const int rounds = std::clamp(static_cast<int>(seconds * 1000.0 / std::max(round_ms, 1e-3)), 30, 5000);
	for (int r=0; r<rounds; ++r) {
		for (std::size_t j=0; j<variants.size(); ++j) {
			Variant& v = variants[(j + static_cast<std::size_t>(r)) % variants.size()];
			v.samples_ms.push_back(once(v));
		}
		clocks.push_back(cuda::current_sm_clock_mhz(0));
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

int run_elementwise(const std::string& graph_dir, const std::string& out_path, double warmup_seconds,
                    double seconds) {
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
	                     "device; 10 warmup enqueues per variant and warm-up rounds until the GPU has been busy for "
	                     "warmup_seconds (it boosts only under sustained load), then interleaved rounds (rotating "
	                     "order); "
	                     "time = median over rounds; speedup = median over rounds of unfused/fused in the same round; "
	                     "bandwidth = bytes each kernel reads and writes / median time");
	json.field("warmup_seconds", warmup_seconds).field("seconds_per_config", seconds);
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
		time_interleaved(variants, stream, 10, warmup_seconds, seconds, clocks);

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
	bench::write_device(json, device, clocks);
	json.end_object();
	CHECK_CUDA(cudaStreamDestroy(stream));

	std::ofstream out(out_path, std::ios::binary);
	out << json.str();
	std::printf("peak bandwidth %.1f GB/s (%s, %.0f MHz memory clock, %d-bit bus); wrote %s\n", peak,
	            device.name.c_str(), device.mem_clock_mhz, device.bus_width_bits, out_path.c_str());
	return out ? 0 : 1;
}

#define CHECK_CUBLAS(expr)                                                    \
	do {                                                                      \
		const cublasStatus_t st_ = (expr);                                    \
		if (st_ != CUBLAS_STATUS_SUCCESS) {                                   \
			std::fprintf(stderr, "%s failed: status %d\n", #expr, int(st_));  \
			std::exit(1);                                                     \
		}                                                                     \
	} while (0)

// One way of computing C = A B, enqueued on a stream.
struct MatmulVariant {
	std::string name;
	std::function<void(cudaStream_t)> run;
	std::vector<double> samples_ms;
};

// Checks `c` against a float64 dot product on `samples` random elements: a
// float32 result must lie within gamma_k * sum|a b| of the exact value (the
// standard bound for any summation order), plus its own rounding.
bool within_error_bound(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int m,
                        int n, int k, int samples) {
	const double u = std::ldexp(1.0, -24);
	const double gamma = k * u / (1.0 - k * u);
	std::uint64_t state = 12345;
	for (int s=0; s<samples; ++s) {
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		const int r = static_cast<int>((state >> 33) % static_cast<std::uint64_t>(m));
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		const int col = static_cast<int>((state >> 33) % static_cast<std::uint64_t>(n));
		double exact = 0.0, abs_sum = 0.0;
		for (int t=0; t<k; ++t) {
			const double p = static_cast<double>(a[static_cast<std::size_t>(r) * k + t]) * b[static_cast<std::size_t>(t) * n + col];
			exact += p;
			abs_sum += std::fabs(p);
		}
		const double got = c[static_cast<std::size_t>(r) * n + col];
		if (std::fabs(got - exact) > gamma * abs_sum + u * std::fabs(exact)) return false;
	}
	return true;
}

int run_matmul(const std::string& out_path, double warmup_seconds, double seconds) {
	const cuda::DeviceInfo device = must(cuda::query_device(0), "query_device");
	CHECK_CUDA(cudaSetDevice(0));
	cudaStream_t stream;
	CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	cublasHandle_t handle;
	CHECK_CUBLAS(cublasCreate(&handle));
	CHECK_CUBLAS(cublasSetStream(handle, stream));
	// Plain FP32: no TF32 tensor cores (NVIDIA_TF32_OVERRIDE=0 is also set in
	// main), so cuBLAS does the same arithmetic as the hand-written kernels.
	CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

	const double max_clock = device.max_sm_clock_mhz > 0 ? device.max_sm_clock_mhz : device.sm_clock_mhz;
	const double peak_at_max_clock = device.peak_fp32_gflops(max_clock);
	std::vector<double> clocks;

	struct Shape3 {
		int m, k, n;
		const char* label;
	};
	const std::vector<Shape3> shapes = {
		{512, 512, 512, "512^3"},
		{640, 640, 640, "640^3"},
		{1024, 1024, 1024, "1024^3"},
		{2048, 2048, 2048, "2048^3"},
		{4096, 4096, 4096, "4096^3"},
		{1000, 1000, 1000, "1000^3 (not a tile multiple)"},
		{1023, 1029, 1031, "1023x1029x1031 (odd)"},
		{256, 1024, 1024, "256x1024x1024 (matmul_bias_relu)"},
		{128, 512, 2048, "128x512x2048 (MLP layer 1, B=128)"},
		{128, 2048, 512, "128x2048x512 (MLP layer 2, B=128)"},
		{512, 2048, 512, "512x2048x512 (MLP layer 2, B=512)"},
	};

	bench::JsonWriter json;
	json.begin_object();
	json.field("benchmark", "FP32 matmul C[m,n] = A[m,k] B[k,n] (row-major): naive, shared-memory tiled and "
	                        "register-tiled (128x128 and 64x64 tiles) kernels against cuBLAS");
	json.field("method", "CUDA events around each kernel on one stream; 3 warmup runs per variant and warm-up "
	                     "rounds until the GPU has been busy for warmup_seconds, then interleaved rounds (rotating "
	                     "order); time = median over rounds; GFLOP/s = 2mnk / time; ratios are "
	                     "medians over rounds of per-round ratios. Each kernel's result is checked on 1000 sampled "
	                     "elements against a float64 dot product within the FP32 error bound. cuBLAS runs in "
	                     "CUBLAS_DEFAULT_MATH with NVIDIA_TF32_OVERRIDE=0 (no TF32 tensor cores). "
	                     "register_tiled_picks is the tile size the backend's register_tiled kernel uses for the "
	                     "shape: 64x64 when the output has fewer 128x128 tiles than the GPU has SMs");
	json.field("warmup_seconds", warmup_seconds).field("seconds_per_config", seconds);
	bench::write_environment(json);
	json.key("peak_fp32_gflops").begin_object();
	json.field("at_max_sm_clock", peak_at_max_clock).field("max_sm_clock_mhz", max_clock);
	json.field("at_cuda_reported_clock", device.peak_fp32_gflops(device.sm_clock_mhz));
	json.field("cuda_reported_clock_mhz", device.sm_clock_mhz);
	json.field("formula", "2 x SMs x FP32 lanes per SM x clock");
	json.end_object();
	json.key("results").begin_array();
	std::printf("%-36s %-19s %10s %10s %11s %10s\n", "shape (m x k x n)", "kernel", "median ms", "GFLOP/s",
	            "% of peak", "% cuBLAS");
	for (const Shape3& s : shapes) {
		const std::size_t a_n = static_cast<std::size_t>(s.m) * s.k;
		const std::size_t b_n = static_cast<std::size_t>(s.k) * s.n;
		const std::size_t c_n = static_cast<std::size_t>(s.m) * s.n;
		const std::vector<float> a = uniform_values(a_n, 1, -1.0f, 1.0f);
		const std::vector<float> b = uniform_values(b_n, 2, -1.0f, 1.0f);
		float *da, *db, *dc;
		CHECK_CUDA(cudaMalloc(&da, a_n * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&db, b_n * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&dc, c_n * sizeof(float)));
		CHECK_CUDA(cudaMemcpy(da, a.data(), a_n * sizeof(float), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(db, b.data(), b_n * sizeof(float), cudaMemcpyHostToDevice));

		std::vector<MatmulVariant> variants;
		for (cuda::MatmulKernel kernel : {cuda::MatmulKernel::Naive, cuda::MatmulKernel::Tiled,
		                                  cuda::MatmulKernel::RegisterTiled128, cuda::MatmulKernel::RegisterTiled64}) {
			variants.push_back({cuda::matmul_kernel_name(kernel), [=](cudaStream_t st) {
				                    CHECK_CUDA(cuda::launch_matmul(kernel, da, db, dc, s.m, s.n, s.k, st));
			                    }, {}});
		}
		variants.push_back({"cublas", [=](cudaStream_t) {
			                    const float alpha = 1.0f, beta = 0.0f;
			                    // Row-major C = A B is column-major C^T = B^T A^T.
			                    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.n, s.m, s.k, &alpha, db, s.n,
			                                             da, s.k, &beta, dc, s.n));
		                    }, {}});

		// Correctness first: every variant, cuBLAS included, against float64.
		std::vector<float> c(c_n);
		for (MatmulVariant& v : variants) {
			CHECK_CUDA(cudaMemsetAsync(dc, 0xff, c_n * sizeof(float), stream));  // NaN, so a no-op kernel fails
			v.run(stream);
			CHECK_CUDA(cudaStreamSynchronize(stream));
			CHECK_CUDA(cudaMemcpy(c.data(), dc, c_n * sizeof(float), cudaMemcpyDeviceToHost));
			if (!within_error_bound(a, b, c, s.m, s.n, s.k, 1000)) {
				std::fprintf(stderr, "%s gives wrong results for %s\n", v.name.c_str(), s.label);
				return 1;
			}
		}

		cudaEvent_t start, stop;
		CHECK_CUDA(cudaEventCreate(&start));
		CHECK_CUDA(cudaEventCreate(&stop));
		auto once = [&](MatmulVariant& v) {
			CHECK_CUDA(cudaEventRecord(start, stream));
			v.run(stream);
			CHECK_CUDA(cudaEventRecord(stop, stream));
			CHECK_CUDA(cudaEventSynchronize(stop));
			float ms = 0;
			CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
			return static_cast<double>(ms);
		};
		for (MatmulVariant& v : variants) {
			for (int i=0; i<3; ++i) once(v);
		}
		warm_up_for(warmup_seconds, [&] {
			for (MatmulVariant& v : variants) once(v);
		});
		double round_ms = 0;
		for (MatmulVariant& v : variants) round_ms += once(v);
		const int rounds = std::clamp(static_cast<int>(seconds * 1000.0 / std::max(round_ms, 1e-3)), 10, 2000);
		const std::size_t first_clock = clocks.size();
		for (int r=0; r<rounds; ++r) {
			for (std::size_t j=0; j<variants.size(); ++j) {
				MatmulVariant& v = variants[(j + static_cast<std::size_t>(r)) % variants.size()];
				v.samples_ms.push_back(once(v));
			}
			clocks.push_back(cuda::current_sm_clock_mhz(0));  // sampled under load
		}
		CHECK_CUDA(cudaEventDestroy(start));
		CHECK_CUDA(cudaEventDestroy(stop));

		auto paired = [&](const MatmulVariant& num, const MatmulVariant& den) {
			std::vector<double> r;
			for (std::size_t i=0; i<num.samples_ms.size(); ++i) r.push_back(num.samples_ms[i] / den.samples_ms[i]);
			std::sort(r.begin(), r.end());
			return bench::percentile(r, 0.5);
		};
		std::vector<double> shape_clocks(clocks.begin() + static_cast<std::ptrdiff_t>(first_clock), clocks.end());
		std::sort(shape_clocks.begin(), shape_clocks.end());
		const double clock_now = shape_clocks.empty() ? 0.0 : shape_clocks[shape_clocks.size() / 2];
		const double peak_now = clock_now > 0 ? device.peak_fp32_gflops(clock_now) : 0.0;
		const double flops = 2.0 * s.m * s.n * s.k;
		const MatmulVariant& naive = variants[0];
		const MatmulVariant& cublas = variants.back();
		const std::string picks = cuda::matmul_kernel_name(
		    cuda::resolve_matmul_kernel(cuda::MatmulKernel::RegisterTiled, s.m, s.n, device.sm_count));
		json.begin_object();
		json.field("shape", s.label).field("m", s.m).field("k", s.k).field("n", s.n);
		json.field("tiles_128x128", ((s.m + 127) / 128) * ((s.n + 127) / 128));
		json.field("register_tiled_picks", picks);
		json.field("sm_clock_mhz_median", clock_now);
		json.key("kernels").begin_array();
		for (const MatmulVariant& v : variants) {
			const bench::TimingStats t = bench::summarize(v.samples_ms, 3);
			const double gflops = flops / (t.median_ms * 1e6);
			const double of_cublas = 100.0 * paired(cublas, v);  // cuBLAS time / this time
			json.begin_object();
			json.field("kernel", v.name);
			json.timing("timing", t);
			json.field("gflops", gflops).field("percent_of_peak_at_max_clock", 100.0 * gflops / peak_at_max_clock);
			if (peak_now > 0) json.field("percent_of_peak_at_measured_clock", 100.0 * gflops / peak_now);
			json.field("percent_of_cublas", of_cublas).field("speedup_over_naive", paired(naive, v));
			json.end_object();
			std::printf("%-36s %-19s %10.3f %10.0f %10.1f%% %9.1f%%%s\n", s.label, v.name.c_str(), t.median_ms,
			            gflops, 100.0 * gflops / peak_at_max_clock, of_cublas, v.name == picks ? "  <- picked" : "");
		}
		json.end_array();
		json.end_object();
		CHECK_CUDA(cudaFree(da));
		CHECK_CUDA(cudaFree(db));
		CHECK_CUDA(cudaFree(dc));
	}
	json.end_array();
	bench::write_device(json, device, clocks);
	json.end_object();
	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK_CUDA(cudaStreamDestroy(stream));

	std::ofstream out(out_path, std::ios::binary);
	out << json.str();
	std::printf("peak FP32 %.0f GFLOP/s at the %.0f MHz maximum SM clock (%d SMs x %d lanes x 2); wrote %s\n",
	            peak_at_max_clock, max_clock, device.sm_count, device.fp32_lanes_per_sm, out_path.c_str());
	return out ? 0 : 1;
}

}

int main(int argc, char** argv) {
	std::string suite = "elementwise";
	std::string out_path;
	std::string graph_dir = std::string(MINICOMPILER_SOURCE_DIR) + "/bench/graphs";
	double seconds = 2.0;
	double warmup_seconds = 1.0;
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
			warmup_seconds = 0.2;
		} else {
			std::cerr << "usage: bench_cuda [--suite elementwise|matmul] [--out FILE] [--graphs DIR] [--quick]\n";
			return 1;
		}
	}
	if (out_path.empty()) out_path = "results/gpu/" + suite + ".json";
	// Before CUDA starts: keep every library on plain FP32 arithmetic.
	setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
	if (suite == "elementwise") return run_elementwise(graph_dir, out_path, warmup_seconds, seconds);
	if (suite == "matmul") return run_matmul(out_path, warmup_seconds, seconds);
	std::cerr << "unknown suite '" << suite << "'\n";
	return 1;
}
