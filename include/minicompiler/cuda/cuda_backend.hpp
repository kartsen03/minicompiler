#pragma once

// The CUDA backend. Available when the project was configured with a CUDA
// toolkit (MINICOMPILER_HAVE_CUDA is defined); otherwise create_backend("cuda")
// reports that the build has no CUDA backend. This header needs no CUDA headers.

#include "minicompiler/graph.hpp"
#include "minicompiler/result.hpp"
#include "minicompiler/runtime/backend.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace minicompiler::cuda {

// The matmul kernels, in the order they were written: each step of the list
// is one optimization over the one before, and all of them stay selectable.
enum class MatmulKernel {
	Auto,              // the default: SplitK where it splits K, otherwise DoubleBuffered (see below)
	Naive,             // one thread per output element, operands read from global memory
	Tiled,             // 32x32 tiles of A and B staged in shared memory, one output per thread
	RegisterTiled128,  // 128x128 tiles in shared memory, an 8x8 block of outputs per thread in registers
	RegisterTiled64,   // 64x64 tiles, a 4x4 block of outputs per thread
	Vectorized,        // RegisterTiled128 with 128-bit global and shared-memory loads and a transposed A tile
	DoubleBuffered,    // Vectorized, loading the next tile while computing the current one
	SplitK,            // DoubleBuffered with K split across blocks when the output has too few tiles to fill the GPU
	// Tensor cores (WMMA), opt-in because they are not FP32: the inputs are
	// rounded to TF32 or FP16 (unit roundoff 2^-11) or BF16 (2^-8) and the
	// products accumulated in FP32, so each output is within about
	// (2u + k 2^-24) sum|a b| of the exact dot product. TF32 and BF16 need
	// compute capability 8.0, FP16 7.0. They split K where SplitK does.
	TensorCoreTf32,
	TensorCoreBf16,
	TensorCoreF16,
};

const char* matmul_kernel_name(MatmulKernel kernel);

// "auto", "naive", "tiled", "register_tiled_128", "register_tiled_64", "vectorized", "double_buffered",
// "split_k", "tensor_core_tf32", "tensor_core_bf16" or "tensor_core_f16".
Result<MatmulKernel> matmul_kernel_from_name(const std::string& name);

// The kernel that computes an m x n x k matmul on a GPU with `sm_count` SMs.
// Auto becomes SplitK when the output has too few 128x128 tiles to fill the
// GPU and K is long enough to split, and DoubleBuffered otherwise. Every other
// kernel is returned unchanged.
MatmulKernel resolve_matmul_kernel(MatmulKernel kernel, int m, int n, int k, int sm_count);

struct CudaOptions {
	int device = 0;
	MatmulKernel matmul = MatmulKernel::Auto;
	// Let elementwise kernels use float4 loads and stores when the layout allows.
	bool vectorize = true;
};

// What the hardware can do, queried at runtime.
struct DeviceInfo {
	std::string name;
	int cc_major = 0;
	int cc_minor = 0;
	int sm_count = 0;
	int fp32_lanes_per_sm = 0;   // from the compute capability (CUDA Programming Guide)
	double sm_clock_mhz = 0;     // the clock CUDA reports (cudaDevAttrClockRate)
	double max_sm_clock_mhz = 0; // the maximum boost clock, from NVML (0 if unavailable)
	double mem_clock_mhz = 0;
	int bus_width_bits = 0;
	std::size_t total_memory = 0;
	int runtime_version = 0;
	int driver_version = 0;

	// 2 transfers per clock (DDR) x clock x bus width.
	double peak_bandwidth_gbs() const { return 2.0 * mem_clock_mhz * 1e6 * bus_width_bits / 8.0 / 1e9; }
	// One fused multiply-add (2 FLOPs) per FP32 lane per clock.
	double peak_fp32_gflops(double clock_mhz) const { return 2.0 * sm_count * fp32_lanes_per_sm * clock_mhz * 1e6 / 1e9; }
};

Result<DeviceInfo> query_device(int device = 0);

// SM clock in MHz right now (NVML), or 0 if unavailable.
double current_sm_clock_mhz(int device = 0);

// A graph compiled for the GPU. Constants are uploaded once at compile time,
// intermediates live in reused device buffers, and run() copies only the
// inputs in and the outputs out. To time the kernels alone, upload the inputs
// once and call enqueue().
class CudaExecutable : public Executable {
public:
	// Copies the inputs to the device; returns once they are there, so kernels
	// enqueued afterwards on any stream see them.
	virtual Status upload_inputs(const std::vector<const float*>& host_inputs) = 0;
	// Launches every kernel in order on `stream` (a cudaStream_t; nullptr means
	// the executable's own stream) with no copies and no synchronization.
	virtual Status enqueue(void* stream = nullptr) = 0;
	virtual Status synchronize() = 0;
	// Waits for the executable's stream, then copies the outputs to the host.
	virtual Status download_outputs(const std::vector<float*>& host_outputs) = 0;

	virtual std::size_t kernel_launches() const = 0;              // per enqueue()
	virtual std::vector<std::string> kernel_sources() const = 0;  // generated elementwise kernels
	virtual std::vector<MatmulKernel> matmul_kernels() const = 0; // the kernel each matmul runs, in order
	virtual double jit_compile_ms() const = 0;                    // NVRTC time spent compiling this graph
};

Result<std::unique_ptr<CudaExecutable>> compile_for_cuda(const Graph& graph, const CudaOptions& options = {});
Result<std::unique_ptr<Backend>> make_cuda_backend(const CudaOptions& options = {});

}
