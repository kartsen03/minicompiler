#include "minicompiler/runtime/host_tensor.hpp"

#include "minicompiler/random.hpp"

namespace minicompiler {

std::vector<HostTensor> make_random_inputs(const Graph& graph, std::uint64_t seed, float lo, float hi) {
	std::vector<HostTensor> inputs;
	for (std::size_t k=0; k<graph.inputs().size(); ++k) {
		const TensorType& type = graph.node(graph.inputs()[k]).type;
		inputs.emplace_back(type, uniform_values(type.num_elements(), seed + k, lo, hi));
	}
	return inputs;
}

}
