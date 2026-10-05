#include "minicompiler/cuda/cuda_backend.hpp"

#include "cuda/codegen.hpp"
#include "cuda/cuda_check.hpp"
#include "cuda/kernel_compiler.hpp"
#include "cuda/matmul_kernels.hpp"
#include "minicompiler/runtime/memory_plan.hpp"

#include <algorithm>
#include <utility>

namespace minicompiler::cuda {

const char* matmul_kernel_name(MatmulKernel kernel) {
	switch (kernel) {
		case MatmulKernel::Auto: return "auto";
		case MatmulKernel::Naive: return "naive";
		case MatmulKernel::Tiled: return "tiled";
		case MatmulKernel::RegisterTiled128: return "register_tiled_128";
		case MatmulKernel::RegisterTiled64: return "register_tiled_64";
		case MatmulKernel::Vectorized: return "vectorized";
		case MatmulKernel::DoubleBuffered: return "double_buffered";
		case MatmulKernel::SplitK: return "split_k";
		case MatmulKernel::TensorCoreTf32: return "tensor_core_tf32";
		case MatmulKernel::TensorCoreBf16: return "tensor_core_bf16";
		case MatmulKernel::TensorCoreF16: return "tensor_core_f16";
	}
	return "?";
}

Result<MatmulKernel> matmul_kernel_from_name(const std::string& name) {
	for (MatmulKernel k : {MatmulKernel::Auto, MatmulKernel::Naive, MatmulKernel::Tiled,
	                       MatmulKernel::RegisterTiled128, MatmulKernel::RegisterTiled64, MatmulKernel::Vectorized,
	                       MatmulKernel::DoubleBuffered, MatmulKernel::SplitK, MatmulKernel::TensorCoreTf32,
	                       MatmulKernel::TensorCoreBf16, MatmulKernel::TensorCoreF16}) {
		if (name == matmul_kernel_name(k)) return k;
	}
	return Error{"unknown matmul kernel '" + name +
	             "' (auto, naive, tiled, register_tiled_128, register_tiled_64, vectorized, double_buffered, "
	             "split_k, tensor_core_tf32, tensor_core_bf16, tensor_core_f16)"};
}

MatmulKernel resolve_matmul_kernel(MatmulKernel kernel, int m, int n, int k, int sm_count) {
	if (kernel != MatmulKernel::Auto) return kernel;
	// The double-buffered kernel beats both register-tiled ones on every
	// measured shape; splitting K adds a reduction, so it is used only where
	// the output alone cannot fill the GPU.
	return split_k_splits(m, n, k, sm_count) > 1 ? MatmulKernel::SplitK : MatmulKernel::DoubleBuffered;
}

namespace {

constexpr unsigned kBlockSize = 256;  // matches __launch_bounds__ in the generated kernels

// One cudaMalloc allocation, freed with the object.
class DeviceBuffer {
public:
	DeviceBuffer() = default;
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;
	DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
	DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
		std::swap(ptr_, other.ptr_);
		return *this;
	}
	~DeviceBuffer() {
		if (ptr_) cudaFree(ptr_);
	}

	Status allocate(std::size_t bytes) {
		MC_CUDA_RETURN(cudaMalloc(&ptr_, std::max<std::size_t>(bytes, sizeof(float))));
		return Status();
	}
	float* get() const { return static_cast<float*>(ptr_); }

private:
	void* ptr_ = nullptr;
};

// One kernel launch in the compiled program.
struct Launch {
	bool is_matmul = false;
	// Elementwise (an NVRTC-compiled kernel): out, in0, in1, ..., count.
	CUfunction function = nullptr;
	unsigned grid = 0;
	unsigned count = 0;                 // elements, or groups of 4 when vectorized
	std::vector<CUdeviceptr> pointers;  // out first, then the inputs
	std::vector<void*> params;          // addresses of the kernel arguments, for cuLaunchKernel
	// MatMul: C[m,n] = A[m,k] B[k,n], with the kernel chosen for this shape,
	// and for SplitK the number of splits and their workspace.
	MatmulKernel matmul = MatmulKernel::RegisterTiled128;
	int splits = 1;
	float* workspace = nullptr;
	const float* a = nullptr;
	const float* b = nullptr;
	float* c = nullptr;
	int m = 0;
	int n = 0;
	int k = 0;
};

class CudaExecutableImpl final : public CudaExecutable {
public:
	static Result<std::unique_ptr<CudaExecutableImpl>> create(const Graph& graph, const CudaOptions& options) {
		std::unique_ptr<CudaExecutableImpl> exe(new CudaExecutableImpl(graph, options));
		Status st = exe->init();
		if (!st.ok()) return st.error();
		return exe;
	}

	~CudaExecutableImpl() override {
		if (stream_) cudaStreamDestroy(stream_);
	}

	Status run(const std::vector<HostTensor>& inputs, std::vector<HostTensor>& outputs) override {
		if (&inputs == &outputs) return Error{"run() needs separate input and output vectors"};
		if (inputs.size() != graph_.inputs().size()) {
			return Error{"expected " + std::to_string(graph_.inputs().size()) + " inputs, got " +
			             std::to_string(inputs.size())};
		}
		std::vector<const float*> in;
		for (std::size_t k=0; k<inputs.size(); ++k) {
			const Node& node = graph_.node(graph_.inputs()[k]);
			if (inputs[k].type != node.type || inputs[k].data.size() != node.type.num_elements()) {
				return Error{"input '" + node.name + "' should be " + to_string(node.type) + ", got " +
				             to_string(inputs[k].type)};
			}
			in.push_back(inputs[k].data.data());
		}
		outputs.resize(graph_.outputs().size());
		std::vector<float*> out;
		for (std::size_t k=0; k<outputs.size(); ++k) {
			const TensorType& type = graph_.node(graph_.outputs()[k]).type;
			outputs[k].type = type;
			outputs[k].data.resize(type.num_elements());
			out.push_back(outputs[k].data.data());
		}
		return run_buffers(in, out);
	}

	Status run_buffers(const std::vector<const float*>& inputs, const std::vector<float*>& outputs) override {
		if (inputs.size() != graph_.inputs().size() || outputs.size() != graph_.outputs().size()) {
			return Error{"expected " + std::to_string(graph_.inputs().size()) + " input and " +
			             std::to_string(graph_.outputs().size()) + " output buffers"};
		}
		Status st = upload_inputs(inputs);
		if (st.ok()) st = enqueue(nullptr);
		if (st.ok()) st = download_outputs(outputs);
		return st;
	}

	std::size_t intermediate_bytes() const override { return plan_.bytes_with_reuse(); }

	Status upload_inputs(const std::vector<const float*>& host_inputs) override {
		if (host_inputs.size() != graph_.inputs().size()) return Error{"wrong number of inputs"};
		for (std::size_t k=0; k<host_inputs.size(); ++k) {
			const NodeId id = graph_.inputs()[k];
			MC_CUDA_RETURN(cudaMemcpyAsync(value_[id], host_inputs[k], graph_.node(id).type.size_bytes(),
			                               cudaMemcpyHostToDevice, stream_));
		}
		return Status();
	}

	Status enqueue(void* stream) override {
		const cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : stream_;
		for (Launch& launch : launches_) {
			if (launch.is_matmul) {
				MC_CUDA_RETURN(launch_matmul(launch.matmul, launch.a, launch.b, launch.c, launch.m, launch.n, launch.k,
				                             s, launch.splits, launch.workspace));
			} else {
				MC_CU_RETURN(cuLaunchKernel(launch.function, launch.grid, 1, 1, kBlockSize, 1, 1, 0, s,
				                            launch.params.data(), nullptr));
			}
		}
		return Status();
	}

	Status synchronize() override {
		MC_CUDA_RETURN(cudaDeviceSynchronize());
		return Status();
	}

	Status download_outputs(const std::vector<float*>& host_outputs) override {
		if (host_outputs.size() != graph_.outputs().size()) return Error{"wrong number of outputs"};
		// Kernels may have been enqueued on a caller's stream, so wait for the device.
		MC_CUDA_RETURN(cudaDeviceSynchronize());
		for (std::size_t k=0; k<host_outputs.size(); ++k) {
			const NodeId id = graph_.outputs()[k];
			MC_CUDA_RETURN(cudaMemcpy(host_outputs[k], value_[id], graph_.node(id).type.size_bytes(),
			                          cudaMemcpyDeviceToHost));
		}
		return Status();
	}

	std::size_t kernel_launches() const override { return launches_.size(); }
	std::vector<std::string> kernel_sources() const override { return sources_; }
	std::vector<MatmulKernel> matmul_kernels() const override {
		std::vector<MatmulKernel> kernels;
		for (const Launch& launch : launches_) {
			if (launch.is_matmul) kernels.push_back(launch.matmul);
		}
		return kernels;
	}
	double jit_compile_ms() const override { return jit_ms_; }

private:
	CudaExecutableImpl(const Graph& graph, const CudaOptions& options)
	    : graph_(graph), options_(options), plan_(plan_memory(graph_)) {}

	Status init() {
		Result<DeviceInfo> device = query_device(options_.device);
		if (!device.ok()) return device.error();
		device_ = std::move(device).value();
		MC_CUDA_RETURN(cudaSetDevice(options_.device));
		MC_CUDA_RETURN(cudaFree(nullptr));  // creates the primary context that NVRTC modules load into
		MC_CUDA_RETURN(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

		// Intermediates and outputs: one allocation per buffer of the memory plan.
		pool_.resize(plan_.buffer_bytes.size());
		for (std::size_t b=0; b<pool_.size(); ++b) {
			Status st = pool_[b].allocate(plan_.buffer_bytes[b]);
			if (!st.ok()) return st;
		}

		value_.assign(graph_.num_nodes(), nullptr);
		for (std::size_t i=0; i<graph_.num_nodes(); ++i) {
			const NodeId id = static_cast<NodeId>(i);
			const Node& node = graph_.node(id);
			if (node.op == OpKind::Input || node.op == OpKind::Constant) {
				// Graph inputs get a buffer of their own; constants are uploaded once, here.
				DeviceBuffer buffer;
				Status st = buffer.allocate(node.type.size_bytes());
				if (!st.ok()) return st;
				if (node.op == OpKind::Constant) {
					MC_CUDA_RETURN(cudaMemcpy(buffer.get(), node.constant->data(), node.type.size_bytes(),
					                          cudaMemcpyHostToDevice));
				}
				value_[i] = buffer.get();
				owned_.push_back(std::move(buffer));
				continue;
			}
			value_[i] = pool_[static_cast<std::size_t>(plan_.buffer_of[i])].get();
			Status st = add_launch(id);
			if (!st.ok()) return st;
		}

		// The launch list is complete, so addresses into it are now stable.
		for (Launch& launch : launches_) {
			if (launch.is_matmul) continue;
			for (CUdeviceptr& p : launch.pointers) launch.params.push_back(&p);
			launch.params.push_back(&launch.count);
		}
		return Status();
	}

	Status add_launch(NodeId id) {
		const Node& node = graph_.node(id);
		Launch launch;
		if (node.op == OpKind::MatMul) {
			const Shape& a = graph_.node(node.inputs[0]).type.shape;
			launch.is_matmul = true;
			launch.a = value_[node.inputs[0]];
			launch.b = value_[node.inputs[1]];
			launch.c = value_[id];
			launch.m = static_cast<int>(a[0]);
			launch.k = static_cast<int>(a[1]);
			launch.n = static_cast<int>(node.type.shape[1]);
			launch.matmul = resolve_matmul_kernel(options_.matmul, launch.m, launch.n, launch.k, device_.sm_count);
			const bool tensor_core = launch.matmul == MatmulKernel::TensorCoreTf32 ||
			                         launch.matmul == MatmulKernel::TensorCoreBf16 ||
			                         launch.matmul == MatmulKernel::TensorCoreF16;
			if (tensor_core && !tensor_core_supported(launch.matmul)) {
				return Error{std::string("matmul kernel '") + matmul_kernel_name(launch.matmul) + "' does not run on " +
				             device_.name + " with this build: TF32 and BF16 need compute capability 8.0 and FP16 "
				             "7.0, in the GPU and in CMAKE_CUDA_ARCHITECTURES"};
			}
			if (uses_split_k(launch.matmul)) {
				launch.splits = matmul_splits(launch.matmul, launch.m, launch.n, launch.k, device_.sm_count);
				const std::size_t bytes = matmul_workspace_bytes(launch.matmul, launch.m, launch.n, launch.splits);
				if (bytes > 0) {
					DeviceBuffer workspace;
					Status st = workspace.allocate(bytes);
					if (!st.ok()) return st;
					launch.workspace = workspace.get();
					owned_.push_back(std::move(workspace));
				}
			}
			launches_.push_back(std::move(launch));
			return Status();
		}

		// Elementwise: a fused group's program, or a single op as a one-op
		// program, through the same generator.
		const FusedProgram program = node.op == OpKind::FusedElementwise ? *node.fused : single_op_program(node.op);
		std::vector<Shape> shapes;
		for (NodeId in : node.inputs) shapes.push_back(graph_.node(in).type.shape);
		const ElementwiseKernel kernel = generate_elementwise_kernel(program, shapes, node.type.shape, options_.vectorize);
		double compile_ms = 0.0;
		Result<CUfunction> function = KernelCompiler::instance().get(kernel.source, kElementwiseKernelName,
		                                                             device_.cc_major, device_.cc_minor, &compile_ms);
		if (!function.ok()) return function.error();
		jit_ms_ += compile_ms;
		sources_.push_back(kernel.source);

		const std::size_t n = node.type.num_elements();
		launch.function = function.value();
		launch.count = static_cast<unsigned>(kernel.vectorized ? n / 4 : n);
		// At most one full wave of resident blocks; the grid-stride loop covers the rest.
		int blocks_per_sm = 0;
		MC_CU_RETURN(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, launch.function,
		                                                         static_cast<int>(kBlockSize), 0));
		const std::size_t needed = (launch.count + kBlockSize - 1) / kBlockSize;
		const std::size_t resident = std::max<std::size_t>(1, static_cast<std::size_t>(blocks_per_sm) * device_.sm_count);
		launch.grid = static_cast<unsigned>(std::min(needed, resident));
		launch.pointers.push_back(reinterpret_cast<CUdeviceptr>(value_[id]));
		for (NodeId in : node.inputs) launch.pointers.push_back(reinterpret_cast<CUdeviceptr>(value_[in]));
		launches_.push_back(std::move(launch));
		return Status();
	}

	Graph graph_;
	CudaOptions options_;
	MemoryPlan plan_;
	DeviceInfo device_;
	cudaStream_t stream_ = nullptr;
	std::vector<DeviceBuffer> pool_;   // intermediates and outputs, reused by lifetime
	std::vector<DeviceBuffer> owned_;  // graph inputs and constants
	std::vector<float*> value_;        // per node: where its value lives on the device
	std::vector<Launch> launches_;
	std::vector<std::string> sources_;
	double jit_ms_ = 0.0;
};

class CudaBackend final : public Backend {
public:
	explicit CudaBackend(const CudaOptions& options) : options_(options) {}
	const char* name() const override { return "cuda"; }

	Result<std::unique_ptr<Executable>> compile(const Graph& graph) override {
		Result<std::unique_ptr<CudaExecutable>> exe = compile_for_cuda(graph, options_);
		if (!exe.ok()) return exe.error();
		return std::unique_ptr<Executable>(std::move(exe).value());
	}

private:
	CudaOptions options_;
};

}

Result<std::unique_ptr<CudaExecutable>> compile_for_cuda(const Graph& graph, const CudaOptions& options) {
	Status st = graph.verify();
	if (!st.ok()) return st.error();
	Result<std::unique_ptr<CudaExecutableImpl>> exe = CudaExecutableImpl::create(graph, options);
	if (!exe.ok()) return exe.error();
	return std::unique_ptr<CudaExecutable>(std::move(exe).value());
}

Result<std::unique_ptr<Backend>> make_cuda_backend(const CudaOptions& options) {
	Result<DeviceInfo> device = query_device(options.device);
	if (!device.ok()) return Error{"no usable CUDA device: " + device.error().message};
	return std::unique_ptr<Backend>(std::make_unique<CudaBackend>(options));
}

}
