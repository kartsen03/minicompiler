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

enum class MatmulKernel {
	Naive,             // one thread per output element, operands read from global memory
	Tiled,             // 32x32 tiles of A and B staged in shared memory, one output per thread
	RegisterTiled,     // RegisterTiled128 or RegisterTiled64, whichever suits the shape (see below)
	RegisterTiled128,  // 128x128 tiles in shared memory, an 8x8 block of outputs per thread in registers
	RegisterTiled64,   // 64x64 tiles, a 4x4 block of outputs per thread
};

const char* matmul_kernel_name(MatmulKernel kernel);

// "naive", "tiled", "register_tiled", "register_tiled_128" or "register_tiled_64".
Result<MatmulKernel> matmul_kernel_from_name(const std::string& name);

// The kernel that computes an m x n output on a GPU with `sm_count` SMs.
// RegisterTiled becomes RegisterTiled64 when the output's 128x128 tiles
// would keep at most three quarters of the SMs busy, so that the output is
// spread over every SM, and RegisterTiled128 otherwise. Every other kernel is
// returned unchanged.
MatmulKernel resolve_matmul_kernel(MatmulKernel kernel, int m, int n, int sm_count);

struct CudaOptions {
	int device = 0;
	MatmulKernel matmul = MatmulKernel::RegisterTiled;
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
