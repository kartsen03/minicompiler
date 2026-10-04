// The CUDA backend against the Eigen CPU backend. Built only when the project
// has a CUDA toolkit; every test skips when no GPU is present.

#include "minicompiler/cuda/cuda_backend.hpp"
#include "minicompiler/graph_builder.hpp"
#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"

#include "test_util.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>

using namespace minicompiler;
using minicompiler::testutil::all_close;

namespace {

bool have_gpu() {
	static const bool present = cuda::query_device(0).ok();
	return present;
}

#define REQUIRE_GPU() \
	if (!have_gpu()) GTEST_SKIP() << "no CUDA device"

std::vector<HostTensor> run_on(const char* backend, const Graph& g, const std::vector<HostTensor>& inputs) {
	Result<std::unique_ptr<Backend>> b = create_backend(backend);
	EXPECT_TRUE(b.ok()) << (b.ok() ? "" : b.error().message);
	Result<std::unique_ptr<Executable>> exe = (*b)->compile(g);
	EXPECT_TRUE(exe.ok()) << (exe.ok() ? "" : exe.error().message);
	std::vector<HostTensor> out;
	if (!exe.ok()) return out;
	Status st = (*exe)->run(inputs, out);
	EXPECT_TRUE(st.ok()) << (st.ok() ? "" : st.message());
	return out;
}

Graph unary_graph(OpKind op, Shape shape) {
	GraphBuilder b("unary");
	b.output(b.op(op, {b.input("x", shape)}));
	return std::move(b).build().value();
}

Graph binary_graph(OpKind op, Shape a, Shape b_shape) {
	GraphBuilder b("binary");
	NodeId x = b.input("a", a);
	NodeId y = b.input("b", b_shape);
	b.output(b.op(op, {x, y}));
	return std::move(b).build().value();
}

std::string graph_path(const std::string& name) {
	return std::string(MINICOMPILER_SOURCE_DIR) + "/bench/graphs/" + name + ".mcg";
}

}

// Tolerances. +, -, *, /, neg and relu are IEEE-exact on both sides (CUDA's
// division and sqrtf are correctly rounded by default), so single ops must
// match bit for bit. CUDA documents expf and tanhf to 2 ulp and logf to 1 ulp;
// Eigen's tanh measured 4.75 ulp and its sqrt is 2 ulp. Summing the two sides
// stays under 8 ulp: rtol 1e-6, plus 1 ulp at 1.0 for results near zero.
constexpr double kTranscendentalRtol = 1e-6;
constexpr double kTranscendentalAtol = 1.2e-7;

TEST(CudaBackend, IsSelectedByName) {
	REQUIRE_GPU();
	Result<std::unique_ptr<Backend>> b = create_backend("cuda");
	ASSERT_TRUE(b.ok());
	EXPECT_STREQ((*b)->name(), "cuda");
	const cuda::DeviceInfo info = cuda::query_device(0).value();
	EXPECT_GT(info.sm_count, 0);
	EXPECT_GT(info.peak_bandwidth_gbs(), 0.0);
	EXPECT_GT(info.peak_fp32_gflops(info.sm_clock_mhz), 0.0);
}

TEST(CudaBackend, UnaryOpsMatchTheCpuBackend) {
	REQUIRE_GPU();
	for (const Shape& shape : {Shape{37, 29}, Shape{64, 64}}) {  // scalar path, then float4 path
		for (OpKind op : {OpKind::Neg, OpKind::Exp, OpKind::Log, OpKind::Sqrt, OpKind::Relu, OpKind::Sigmoid,
		                  OpKind::Tanh}) {
			Graph g = unary_graph(op, shape);
			const bool positive = op == OpKind::Log || op == OpKind::Sqrt;
			auto inputs = make_random_inputs(g, 11, positive ? 0.01f : -6.0f, 6.0f);
			const auto cpu = run_on("cpu", g, inputs);
			const auto gpu = run_on("cuda", g, inputs);
			const bool exact = op == OpKind::Neg || op == OpKind::Relu;
			EXPECT_TRUE(all_close(gpu[0].data, cpu[0].data, exact ? 0.0 : kTranscendentalRtol,
			                      exact ? 0.0 : kTranscendentalAtol))
			    << op_name(op) << " " << to_string(shape);
		}
	}
}

TEST(CudaBackend, BinaryOpsMatchBitForBitForEveryBroadcastPattern) {
	REQUIRE_GPU();
	const std::vector<std::pair<Shape, Shape>> patterns = {
		{{17, 33}, {17, 33}}, {{17, 32}, {}},   {{}, {17, 33}},     {{17, 32}, {32}},
		{{17, 33}, {17, 1}},  {{17, 1}, {1, 33}}, {{2, 3, 4}, {3, 1}}, {{5, 1, 6}, {1, 4, 1}},
	};
	for (const auto& [sa, sb] : patterns) {
		for (OpKind op : {OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div}) {
			Graph g = binary_graph(op, sa, sb);
			auto inputs = make_random_inputs(g, 5, 0.5f, 2.0f);
			EXPECT_TRUE(all_close(run_on("cuda", g, inputs)[0].data, run_on("cpu", g, inputs)[0].data, 0.0, 0.0))
			    << op_name(op) << " " << to_string(sa) << " with " << to_string(sb);
		}
	}
}

TEST(CudaBackend, EveryMatmulKernelIsWithinTheDotProductErrorBound) {
	REQUIRE_GPU();
	// Each side is within gamma_k * sum|a b| of the exact dot product, so they
	// differ by at most twice that (plus each side's final rounding). The
	// shapes straddle the 32 and 128 tile sizes and the K step of 8, so the
	// zero-padded edge tiles are exercised.
	const std::vector<std::array<std::int64_t, 3>> shapes = {
		{1, 1, 1},       {3, 5, 7},       {1, 1000, 1},     {33, 257, 65},  {127, 129, 131},
		{128, 128, 128}, {129, 13, 129},  {256, 256, 256},  {16, 4099, 16}, {513, 77, 259}};
	const double u = std::ldexp(1.0, -24);
	for (const cuda::MatmulKernel kernel :
	     {cuda::MatmulKernel::Naive, cuda::MatmulKernel::Tiled}) {
		cuda::CudaOptions options;
		options.matmul = kernel;
		for (const auto& [m, k, n] : shapes) {
			GraphBuilder b("mm");
			NodeId a = b.input("a", {m, k});
			NodeId w = b.input("w", {k, n});
			b.output(b.matmul(a, w));
			Graph g = std::move(b).build().value();
			auto inputs = make_random_inputs(g, 3);
			const auto cpu = run_on("cpu", g, inputs);
			std::vector<HostTensor> gpu;
			ASSERT_TRUE(cuda::compile_for_cuda(g, options).value()->run(inputs, gpu).ok());
			const double gamma = static_cast<double>(k) * u / (1.0 - static_cast<double>(k) * u);
			for (std::int64_t r=0; r<m; ++r) {
				for (std::int64_t c=0; c<n; ++c) {
					double abs_dot = 0.0;
					for (std::int64_t t=0; t<k; ++t) {
						abs_dot += std::fabs(inputs[0].data[r * k + t] * static_cast<double>(inputs[1].data[t * n + c]));
					}
					const double e = cpu[0].data[r * n + c];
					ASSERT_LE(std::fabs(gpu[0].data[r * n + c] - e), 2.0 * gamma * abs_dot + 2.0 * u * std::fabs(e))
					    << cuda::matmul_kernel_name(kernel) << " " << m << "x" << k << "x" << n << " at (" << r
					    << ", " << c << ")";
				}
			}
		}
	}
}

TEST(CudaBackend, BenchmarkGraphsMatchTheCpuBackendFusedAndUnfused) {
	REQUIRE_GPU();
	const std::vector<std::pair<std::string, std::map<std::string, std::int64_t>>> graphs = {
		{"gelu_chain", {{"M", 64}, {"N", 300}}},
		{"matmul_bias_relu", {{"M", 33}, {"K", 70}, {"N", 129}}},
		{"mlp_block", {{"B", 16}, {"D", 64}, {"H", 256}}},
	};
	for (const auto& [name, dims] : graphs) {
		ParseOptions options;
		options.dims = dims;
		Graph g = parse_graph_file(graph_path(name), options).value();
		for (const char* passes : {"none", "default"}) {
			Graph compiled = run_pipeline(g, parse_pipeline(passes).value()).value();
			auto inputs = make_random_inputs(compiled, 1);
			const auto cpu = run_on("cpu", compiled, inputs);
			const auto gpu = run_on("cuda", compiled, inputs);
			ASSERT_EQ(gpu.size(), cpu.size());
			const double scale = std::max(1.0, testutil::max_abs(cpu[0].data));
			EXPECT_TRUE(all_close(gpu[0].data, cpu[0].data, 1e-5, 1e-5 * scale)) << name << ", passes " << passes;
		}
	}
}

TEST(CudaBackend, RandomGraphsMatchTheCpuBackend) {
	REQUIRE_GPU();
	// Up to 24 ops deep with transcendental functions on both sides that
	// differ by a few ulp each: allow 1e-4 relative to each output's magnitude.
	for (std::uint64_t seed=1; seed<=200; ++seed) {
		Graph g = testutil::make_random_graph(seed, {24, true});
		for (const char* passes : {"none", "default"}) {
			Graph compiled = run_pipeline(g, parse_pipeline(passes).value()).value();
			auto inputs = make_random_inputs(compiled, seed);
			const auto cpu = run_on("cpu", compiled, inputs);
			const auto gpu = run_on("cuda", compiled, inputs);
			ASSERT_EQ(gpu.size(), cpu.size()) << "seed " << seed;
			for (std::size_t k=0; k<cpu.size(); ++k) {
				const double scale = std::max(1.0, testutil::max_abs(cpu[k].data));
				EXPECT_TRUE(all_close(gpu[k].data, cpu[k].data, 1e-4, 1e-4 * scale))
				    << "seed " << seed << ", passes " << passes << ", output " << k;
			}
		}
	}
}

TEST(CudaBackend, FusionCutsKernelLaunches) {
	REQUIRE_GPU();
	ParseOptions options;
	options.dims = {{"M", 8}, {"N", 128}};
	Graph g = parse_graph_file(graph_path("gelu_chain"), options).value();
	auto launches = [&](const char* passes) {
		Graph compiled = run_pipeline(g, parse_pipeline(passes).value()).value();
		return cuda::compile_for_cuda(compiled).value()->kernel_launches();
	};
	EXPECT_EQ(launches("none"), 11u);          // 9 elementwise ops on x, 2 on scalar constants
	EXPECT_EQ(launches("dne,fold,dne"), 9u);   // the scalar ops folded away
	EXPECT_EQ(launches("default"), 1u);        // one fused kernel
}

TEST(CudaBackend, ReusesIntermediateBuffersOnTheDevice) {
	REQUIRE_GPU();
	GraphBuilder b("chain");
	NodeId v = b.input("x", {1024});
	for (int i=0; i<8; ++i) v = b.tanh(v);
	b.output(v);
	auto exe = cuda::compile_for_cuda(std::move(b).build().value()).value();
	EXPECT_EQ(exe->intermediate_bytes(), 2u * 4096u);  // two ping-pong buffers instead of eight
}

TEST(CudaBackend, RunsRepeatedlyOnDeviceResidentInputs) {
	REQUIRE_GPU();
	Graph g = testutil::make_random_graph(17);
	auto exe = cuda::compile_for_cuda(g).value();
	const auto inputs = make_random_inputs(g, 2);
	std::vector<HostTensor> expected;
	ASSERT_TRUE(exe->run(inputs, expected).ok());

	std::vector<const float*> in;
	for (const HostTensor& t : inputs) in.push_back(t.data.data());
	ASSERT_TRUE(exe->upload_inputs(in).ok());
	ASSERT_TRUE(exe->enqueue().ok());
	ASSERT_TRUE(exe->enqueue().ok());  // the inputs stay on the device between launches
	std::vector<std::vector<float>> storage;
	std::vector<float*> out;
	for (const HostTensor& t : expected) storage.emplace_back(t.data.size());
	for (std::vector<float>& s : storage) out.push_back(s.data());
	ASSERT_TRUE(exe->download_outputs(out).ok());
	for (std::size_t k=0; k<expected.size(); ++k) EXPECT_EQ(storage[k], expected[k].data);
}

TEST(CudaBackend, RejectsMismatchedInputs) {
	REQUIRE_GPU();
	auto exe = cuda::compile_for_cuda(unary_graph(OpKind::Exp, {4})).value();
	std::vector<HostTensor> outputs;
	EXPECT_FALSE(exe->run({}, outputs).ok());
	EXPECT_FALSE(exe->run({HostTensor(TensorType({5}))}, outputs).ok());
	EXPECT_TRUE(exe->run({HostTensor(TensorType({4}))}, outputs).ok());
}
