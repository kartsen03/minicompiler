// Builds a small graph with the C++ API, optimizes it, and runs it on the CPU
// backend: y = relu(x W + b) * sigmoid(x W + b), with a constant scale that
// constant folding computes at compile time.

#include "minicompiler/graph_builder.hpp"
#include "minicompiler/parser.hpp"
#include "minicompiler/passes/pipeline.hpp"
#include "minicompiler/random.hpp"
#include "minicompiler/runtime/backend.hpp"

#include <iostream>

using namespace minicompiler;

int main() {
	GraphBuilder b("quickstart");
	NodeId x = b.input("x", {2, 4});
	NodeId w = b.constant("w", {4, 3}, uniform_values(12, 1, -0.5f, 0.5f));
	NodeId bias = b.constant("b", {3}, {0.1f, 0.2f, 0.3f});
	NodeId scale = b.sqrt(b.scalar("four", 4.0f), "scale");  // folds to 2
	NodeId h = b.add(b.matmul(x, w, "xw"), bias, "h");
	// Separate statements fix the node order; C++ leaves the evaluation order
	// of function arguments unspecified.
	NodeId r = b.relu(h, "r");
	NodeId s = b.sigmoid(h, "s");
	NodeId y = b.mul(b.mul(r, s, "gated"), scale, "y");
	b.output(y);

	Result<Graph> graph = std::move(b).build();
	if (!graph.ok()) {
		std::cerr << graph.error().message << "\n";
		return 1;
	}

	std::vector<PassReport> reports;
	Result<Graph> optimized = run_pipeline(graph.value(), default_pipeline(), &reports);
	if (!optimized.ok()) {
		std::cerr << optimized.error().message << "\n";
		return 1;
	}
	for (const PassReport& r : reports) std::cout << r.pass << ": " << r.summary << "\n";
	std::cout << "\n" << print_graph(optimized.value()) << "\n";

	auto backend = create_backend("cpu");
	auto exe = (*backend)->compile(optimized.value());
	if (!exe.ok()) {
		std::cerr << exe.error().message << "\n";
		return 1;
	}
	std::vector<HostTensor> outputs;
	Status st = (*exe)->run(make_random_inputs(optimized.value(), 7), outputs);
	if (!st.ok()) {
		std::cerr << st.message() << "\n";
		return 1;
	}
	std::cout << "y =";
	for (float v : outputs[0].data) std::cout << " " << v;
	std::cout << "\n";
	return 0;
}
