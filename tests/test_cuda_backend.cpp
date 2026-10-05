// The CUDA backend against the Eigen CPU backend. Built only when the project
// has a CUDA toolkit; every test skips when no GPU is present.

#include "minicompiler/cuda/cuda_backend.hpp"
#include "minicompiler/graph_builder.hpp"
#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/random.hpp"

#include "cuda/matmul_kernels.hpp"
#include "test_util.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

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
	// shapes straddle the 32, 64 and 128 tile sizes and the K step of 8, so
	// the zero-padded edge tiles are exercised. Shapes with K and N multiples
	// of 4 take the float4 paths, the rest the element-by-element ones.
	const std::vector<std::array<std::int64_t, 3>> shapes = {
		{1, 1, 1},       {3, 5, 7},      {1, 1000, 1},    {33, 257, 65},  {65, 9, 63},       {127, 129, 131},
		{128, 128, 128}, {129, 13, 129}, {256, 256, 256}, {16, 4099, 16}, {513, 77, 259},    {129, 132, 260},
		{300, 1004, 36}, {2, 12, 4}};
	const double u = std::ldexp(1.0, -24);
	for (const cuda::MatmulKernel kernel :
	     {cuda::MatmulKernel::Auto, cuda::MatmulKernel::Naive, cuda::MatmulKernel::Tiled,
	      cuda::MatmulKernel::RegisterTiled128, cuda::MatmulKernel::RegisterTiled64, cuda::MatmulKernel::Vectorized,
	      cuda::MatmulKernel::DoubleBuffered, cuda::MatmulKernel::SplitK}) {
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

TEST(CudaBackend, TensorCoreMatmulsAreWithinTheirPrecisionBounds) {
	REQUIRE_GPU();
	// Rounding an input to the format costs at most its unit roundoff u, so
	// each product is within (2u + u^2)|a b| of the exact one. Accumulating k
	// of them in FP32 adds up to gamma_k sum|a b|; doubled here, because tensor
	// cores have been measured to round toward zero, not to nearest, when they
	// accumulate (Fasi et al., "Numerical behavior of NVIDIA tensor cores",
	// PeerJ Computer Science, 2021). FP16 loses absolute accuracy below its
	// smallest normal (2^-14), at most 2^-25 per rounded input.
	struct Case {
		cuda::MatmulKernel kernel;
		double u;
		bool subnormals;
	};
	const Case cases[] = {{cuda::MatmulKernel::TensorCoreTf32, std::ldexp(1.0, -11), false},
	                      {cuda::MatmulKernel::TensorCoreBf16, std::ldexp(1.0, -8), false},
	                      {cuda::MatmulKernel::TensorCoreF16, std::ldexp(1.0, -11), true}};
	const std::vector<std::array<std::int64_t, 3>> shapes = {
		{1, 1, 1},       {3, 5, 7},       {33, 257, 65},  {127, 129, 131}, {128, 128, 128},
		{129, 33, 129},  {16, 4099, 16},  {300, 1004, 36}, {2, 12, 4},     {256, 256, 256}};
	const double u32 = std::ldexp(1.0, -24);
	int tested = 0;
	for (const Case& tc : cases) {
		if (!cuda::tensor_core_supported(tc.kernel)) continue;
		++tested;
		cuda::CudaOptions options;
		options.matmul = tc.kernel;
		for (const auto& [m, k, n] : shapes) {
			GraphBuilder b("mm");
			NodeId a = b.input("a", {m, k});
			NodeId w = b.input("w", {k, n});
			b.output(b.matmul(a, w));
			Graph g = std::move(b).build().value();
			auto inputs = make_random_inputs(g, 3);
			std::vector<HostTensor> gpu;
			ASSERT_TRUE(cuda::compile_for_cuda(g, options).value()->run(inputs, gpu).ok());
			const double gamma = static_cast<double>(k) * u32 / (1.0 - static_cast<double>(k) * u32);
			const double tiny = tc.subnormals ? 2.0 * static_cast<double>(k) * std::ldexp(1.0, -25) : 0.0;
			for (std::int64_t r=0; r<m; ++r) {
				for (std::int64_t c=0; c<n; ++c) {
					double exact = 0.0, abs_dot = 0.0;
					for (std::int64_t t=0; t<k; ++t) {
						const double p = inputs[0].data[r * k + t] * static_cast<double>(inputs[1].data[t * n + c]);
						exact += p;
						abs_dot += std::fabs(p);
					}
					const double bound = (2.0 * tc.u + tc.u * tc.u + 2.0 * gamma) * abs_dot + 2.0 * u32 * std::fabs(exact) + tiny;
					ASSERT_LE(std::fabs(gpu[0].data[r * n + c] - exact), bound)
					    << cuda::matmul_kernel_name(tc.kernel) << " " << m << "x" << k << "x" << n << " at (" << r << ", "
					    << c << ")";
				}
			}
		}
	}
	if (tested == 0) GTEST_SKIP() << "no tensor-core format on this GPU";
}

TEST(CudaBackend, TensorCoreKernelsThatCannotRunFailToCompile) {
	REQUIRE_GPU();
	// A tensor-core kernel the GPU lacks, or that this build compiled only for
	// older architectures (where its body is empty), is an error at compile
	// time rather than a kernel that leaves C unwritten.
	GraphBuilder b("mm");
	b.output(b.matmul(b.input("a", {64, 64}), b.input("w", {64, 64})));
	Graph g = std::move(b).build().value();
	for (const cuda::MatmulKernel kernel :
	     {cuda::MatmulKernel::TensorCoreTf32, cuda::MatmulKernel::TensorCoreBf16, cuda::MatmulKernel::TensorCoreF16}) {
		cuda::CudaOptions options;
		options.matmul = kernel;
		Result<std::unique_ptr<cuda::CudaExecutable>> exe = cuda::compile_for_cuda(g, options);
		EXPECT_EQ(exe.ok(), cuda::tensor_core_supported(kernel)) << cuda::matmul_kernel_name(kernel);
		if (!exe.ok()) {
			EXPECT_NE(exe.error().message.find(cuda::matmul_kernel_name(kernel)), std::string::npos);
		}
	}
}

TEST(CudaBackend, SplitKIsWithinTheErrorBoundForAnySplitCount) {
	REQUIRE_GPU();
	// Forced split counts, where the rule would pick one per shape: more splits
	// than K has tiles, a last split shorter than the others, K below one tile,
	// and shapes that take the element-by-element path (K or N not a multiple
	// of 4). Splitting K only reorders the sum, so each kernel's usual bound
	// holds: FP32's, or the tensor-core bound of its input format (see
	// TensorCoreMatmulsAreWithinTheirPrecisionBounds).
	struct Kernel {
		cuda::MatmulKernel kernel;
		double u;  // the inputs' unit roundoff, 0 for FP32
		bool subnormals;
	};
	const Kernel kernels[] = {{cuda::MatmulKernel::SplitK, 0.0, false},
	                          {cuda::MatmulKernel::TensorCoreTf32, std::ldexp(1.0, -11), false},
	                          {cuda::MatmulKernel::TensorCoreBf16, std::ldexp(1.0, -8), false},
	                          {cuda::MatmulKernel::TensorCoreF16, std::ldexp(1.0, -11), true}};
	const std::vector<std::array<int, 3>> shapes = {
		{33, 1000, 65}, {128, 2048, 512}, {16, 4099, 16}, {2, 12, 4}, {129, 260, 132}, {64, 5, 8}};
	const double u = std::ldexp(1.0, -24);
	for (const auto& [m, k, n] : shapes) {
		const std::size_t a_n = static_cast<std::size_t>(m) * k, b_n = static_cast<std::size_t>(k) * n;
		const std::size_t c_n = static_cast<std::size_t>(m) * n;
		const std::vector<float> a = uniform_values(a_n, 7, -1.0f, 1.0f);
		const std::vector<float> b = uniform_values(b_n, 8, -1.0f, 1.0f);
		std::vector<double> exact(c_n, 0.0), abs_dot(c_n, 0.0);
		for (int r=0; r<m; ++r) {
			for (int t=0; t<k; ++t) {
				const double av = a[static_cast<std::size_t>(r) * k + t];
				for (int col=0; col<n; ++col) {
					const double p = av * b[static_cast<std::size_t>(t) * n + col];
					exact[static_cast<std::size_t>(r) * n + col] += p;
					abs_dot[static_cast<std::size_t>(r) * n + col] += std::fabs(p);
				}
			}
		}
		float *da, *db, *dc;
		ASSERT_EQ(cudaMalloc(&da, a_n * sizeof(float)), cudaSuccess);
		ASSERT_EQ(cudaMalloc(&db, b_n * sizeof(float)), cudaSuccess);
		ASSERT_EQ(cudaMalloc(&dc, c_n * sizeof(float)), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(da, a.data(), a_n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
		ASSERT_EQ(cudaMemcpy(db, b.data(), b_n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
		const double gamma = k * u / (1.0 - k * u);
		for (const Kernel& kn : kernels) {
			if (kn.u > 0 && !cuda::tensor_core_supported(kn.kernel)) continue;
			auto bound = [&](std::size_t i) {
				if (kn.u == 0) return gamma * abs_dot[i] + u * std::fabs(exact[i]);
				const double tiny = kn.subnormals ? 2.0 * k * std::ldexp(1.0, -25) : 0.0;
				return (2.0 * kn.u + kn.u * kn.u + 2.0 * gamma) * abs_dot[i] + 2.0 * u * std::fabs(exact[i]) + tiny;
			};
			for (int splits : {1, 2, 3, 7, 16, 64}) {
				float* workspace = nullptr;
				const std::size_t bytes = cuda::matmul_workspace_bytes(kn.kernel, m, n, splits);
				if (bytes > 0) {
					ASSERT_EQ(cudaMalloc(&workspace, bytes), cudaSuccess);
				}
				ASSERT_EQ(cudaMemset(dc, 0xff, c_n * sizeof(float)), cudaSuccess);  // NaN, so unwritten outputs fail
				ASSERT_EQ(cuda::launch_matmul(kn.kernel, da, db, dc, m, n, k, nullptr, splits, workspace), cudaSuccess);
				std::vector<float> c(c_n);
				ASSERT_EQ(cudaMemcpy(c.data(), dc, c_n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
				for (std::size_t i=0; i<c_n; ++i) {
					ASSERT_LE(std::fabs(c[i] - exact[i]), bound(i))
					    << cuda::matmul_kernel_name(kn.kernel) << " " << m << "x" << k << "x" << n << " with " << splits
					    << " splits, element " << i;
				}
				if (workspace) cudaFree(workspace);
			}
		}
		cudaFree(da);
		cudaFree(db);
		cudaFree(dc);
	}
}

TEST(CudaBackend, SplitKSplitsSmallOutputsAcrossTheSms) {
	// On 30 SMs (arguments are m, n, k):
	EXPECT_EQ(cuda::split_k_splits(128, 512, 2048, 30), 7);    // 4 tiles: one block per SM, 28 blocks
	EXPECT_EQ(cuda::split_k_splits(128, 1920, 2048, 30), 2);   // 15 tiles, the last count with one per SM
	EXPECT_EQ(cuda::split_k_splits(128, 2048, 2048, 30), 3);   // 16 tiles: up to two per SM, 48 blocks
	EXPECT_EQ(cuda::split_k_splits(128, 2944, 2048, 30), 2);   // 23 tiles, just below 80% of the SMs
	EXPECT_EQ(cuda::split_k_splits(128, 3072, 2048, 30), 1);   // 24 tiles: no split
	EXPECT_EQ(cuda::split_k_splits(640, 640, 640, 30), 1);     // 25 tiles
	EXPECT_EQ(cuda::split_k_splits(1024, 1024, 1024, 30), 1);  // 64 tiles fill the GPU
	EXPECT_EQ(cuda::split_k_splits(128, 128, 200, 30), 3);     // 1 tile, but at least 64 of K per split
	EXPECT_EQ(cuda::split_k_splits(128, 128, 100, 30), 1);     // K too short to split
	// A kernel of which an SM holds one block: never a second wave.
	EXPECT_EQ(cuda::split_k_splits(128, 512, 2048, 30, 1), 7);   // 4 tiles: as before, one block per SM
	EXPECT_EQ(cuda::split_k_splits(128, 2048, 2048, 30, 1), 1);  // 16 tiles: a split would need two waves
	EXPECT_EQ(cuda::split_k_splits(128, 1792, 2048, 30, 1), 2);  // 14 tiles, 28 blocks
	for (int blocks_per_sm : {1, 2}) {
		for (int tiles=1; tiles<=40; ++tiles) {
			const int splits = cuda::split_k_splits(128, 128 * tiles, 1 << 16, 30, blocks_per_sm);
			if (splits > 1) {
				EXPECT_LE(tiles * splits, blocks_per_sm * 30) << tiles << " tiles, " << blocks_per_sm << " per SM";
			}
		}
	}
	EXPECT_EQ(cuda::matmul_workspace_bytes(cuda::MatmulKernel::SplitK, 128, 512, 15), 15u * 128 * 512 * 4);
	EXPECT_EQ(cuda::matmul_workspace_bytes(cuda::MatmulKernel::SplitK, 128, 512, 1), 0u);
	EXPECT_EQ(cuda::matmul_workspace_bytes(cuda::MatmulKernel::DoubleBuffered, 128, 512, 15), 0u);
	EXPECT_EQ(cuda::matmul_workspace_bytes(cuda::MatmulKernel::TensorCoreF16, 128, 512, 7), 7u * 128 * 512 * 4);
	EXPECT_EQ(cuda::matmul_workspace_bytes(cuda::MatmulKernel::Auto, 128, 512, 7), 0u);  // resolved first
}

TEST(CudaBackend, MatmulKernelNamesRoundTrip) {
	for (const cuda::MatmulKernel kernel :
	     {cuda::MatmulKernel::Auto, cuda::MatmulKernel::Naive, cuda::MatmulKernel::Tiled,
	      cuda::MatmulKernel::RegisterTiled128, cuda::MatmulKernel::RegisterTiled64, cuda::MatmulKernel::Vectorized,
	      cuda::MatmulKernel::DoubleBuffered, cuda::MatmulKernel::SplitK, cuda::MatmulKernel::TensorCoreTf32,
	      cuda::MatmulKernel::TensorCoreBf16, cuda::MatmulKernel::TensorCoreF16}) {
		Result<cuda::MatmulKernel> parsed = cuda::matmul_kernel_from_name(cuda::matmul_kernel_name(kernel));
		ASSERT_TRUE(parsed.ok());
		EXPECT_EQ(parsed.value(), kernel);
	}
	EXPECT_FALSE(cuda::matmul_kernel_from_name("register_tiled_32").ok());
}

TEST(CudaBackend, AutoSplitsKOnlyWhenTheOutputCannotFillTheGpu) {
	using cuda::MatmulKernel;
	auto resolve = [](int m, int n, int k, int sms) { return cuda::resolve_matmul_kernel(MatmulKernel::Auto, m, n, k, sms); };
	EXPECT_EQ(resolve(128, 512, 2048, 30), MatmulKernel::SplitK);           // 4 tiles of 128x128 for 30 SMs
	EXPECT_EQ(resolve(512, 512, 512, 30), MatmulKernel::SplitK);            // 16 tiles
	EXPECT_EQ(resolve(640, 640, 640, 30), MatmulKernel::DoubleBuffered);    // 25 tiles: no split pays
	EXPECT_EQ(resolve(2048, 2048, 2048, 30), MatmulKernel::DoubleBuffered); // 256 tiles
	EXPECT_EQ(resolve(128, 128, 100, 30), MatmulKernel::DoubleBuffered);    // too little K to split
	EXPECT_EQ(resolve(640, 640, 640, 40), MatmulKernel::SplitK);            // the same shape on 40 SMs
	EXPECT_EQ(resolve(std::numeric_limits<int>::max(), 1, 1, 30), MatmulKernel::DoubleBuffered);  // no int overflow
	// Any other kernel is left alone.
	EXPECT_EQ(cuda::resolve_matmul_kernel(MatmulKernel::RegisterTiled64, 4096, 4096, 4096, 30),
	          MatmulKernel::RegisterTiled64);
	EXPECT_EQ(cuda::resolve_matmul_kernel(MatmulKernel::SplitK, 4096, 4096, 4096, 30), MatmulKernel::SplitK);
	EXPECT_EQ(cuda::resolve_matmul_kernel(MatmulKernel::Naive, 1, 1, 1, 30), MatmulKernel::Naive);
}

TEST(CudaBackend, CompiledGraphsReportTheMatmulKernelForEachShape) {
	REQUIRE_GPU();
	const int sms = cuda::query_device(0).value().sm_count;
	GraphBuilder b("two_layers");
	NodeId x = b.input("x", {128, 512});
	NodeId h = b.matmul(x, b.input("w1", {512, 2048}));
	b.output(b.matmul(h, b.input("w2", {2048, 4096})));
	auto exe = cuda::compile_for_cuda(std::move(b).build().value()).value();
	const std::vector<cuda::MatmulKernel> expected = {
		cuda::resolve_matmul_kernel(cuda::MatmulKernel::Auto, 128, 2048, 512, sms),
		cuda::resolve_matmul_kernel(cuda::MatmulKernel::Auto, 128, 4096, 2048, sms)};
	EXPECT_EQ(exe->matmul_kernels(), expected);
	cuda::CudaOptions options;
	options.matmul = cuda::MatmulKernel::Tiled;
	GraphBuilder b2("one_layer");
	b2.output(b2.matmul(b2.input("x", {8, 8}), b2.input("w", {8, 8})));
	EXPECT_EQ(cuda::compile_for_cuda(std::move(b2).build().value(), options).value()->matmul_kernels(),
	          std::vector<cuda::MatmulKernel>{cuda::MatmulKernel::Tiled});
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
