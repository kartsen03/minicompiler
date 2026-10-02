#pragma once

#include "minicompiler/graph.hpp"
#include "minicompiler/passes/pass.hpp"
#include "minicompiler/result.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace minicompiler {

using PassFn = Result<Graph> (*)(const Graph&, PassReport*);

struct PassInfo {
	const char* name;
	const char* description;
	PassFn run;
};

// "dne", "fold" and "fuse".
const std::vector<PassInfo>& registered_passes();

// dne, fold, dne, fuse, dne: remove dead code first so nothing dead gets
// folded, clean up the operands that folding orphaned, fuse, then remove
// the scalar constants that fusion baked into kernels.
const std::vector<std::string>& default_pipeline();

// "default", "none", or a comma-separated list of pass names.
Result<std::vector<std::string>> parse_pipeline(std::string_view spec);

// Called with the input graph (stage 0, pass "input") and after each pass
// (stage i, the i-th pass's name). Used to dump DOT files per stage.
using PassObserver = std::function<void(std::size_t stage, const std::string& pass, const Graph& graph)>;

// Runs the passes in order. The graph is verified before the first pass and
// after every pass, so a pass that breaks an invariant is caught at once.
Result<Graph> run_pipeline(const Graph& graph, const std::vector<std::string>& passes,
                           std::vector<PassReport>* reports = nullptr, const PassObserver& observer = {});

}
