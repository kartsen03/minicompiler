#include "minicompiler/runtime/backend.hpp"

#include "cpu/cpu_backend.hpp"

#ifdef MINICOMPILER_HAVE_CUDA
#include "minicompiler/cuda/cuda_backend.hpp"
#endif

namespace minicompiler {

std::vector<std::string> available_backends() {
	std::vector<std::string> names = {"cpu"};
#ifdef MINICOMPILER_HAVE_CUDA
	names.push_back("cuda");
#endif
	return names;
}

Result<std::unique_ptr<Backend>> create_backend(std::string_view name) {
	if (name == "cpu") return make_cpu_backend();
	if (name == "cuda") {
#ifdef MINICOMPILER_HAVE_CUDA
		return cuda::make_cuda_backend();
#else
		return Error{"this build has no CUDA backend: configure with a CUDA toolkit installed and "
		             "MINICOMPILER_ENABLE_CUDA=ON"};
#endif
	}
	std::string known;
	for (const std::string& b : available_backends()) known += (known.empty() ? "" : ", ") + b;
	return Error{"unknown backend '" + std::string(name) + "' (available: " + known + ")"};
}

}
