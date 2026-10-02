#pragma once

#include "minicompiler/graph.hpp"

#include <string>

namespace minicompiler {

// What one pass did: node counts before and after plus a one-line summary.
struct PassReport {
	std::string pass;
	GraphStats before;
	GraphStats after;
	std::string summary;
};

inline void fill_report(PassReport* report, const char* pass, const Graph& before, const Graph& after,
                        std::string summary) {
	if (!report) return;
	report->pass = pass;
	report->before = compute_stats(before);
	report->after = compute_stats(after);
	report->summary = std::move(summary);
}

}
