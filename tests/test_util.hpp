#pragma once

// Shared helpers for the test suites: tolerance checks, a double-precision
// reference evaluator, and a seeded generator of random valid graphs.

#include "minicompiler/graph.hpp"
#include "minicompiler/runtime/host_tensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace minicompiler::testutil {

// Elementwise |actual - expected| <= atol + rtol * |expected|. NaNs must
// match NaNs. On failure the message names the worst element.
::testing::AssertionResult all_close(const std::vector<float>& actual, const std::vector<float>& expected,
                                     double rtol, double atol);

double max_abs(const std::vector<float>& v);

// Evaluates the graph naively in double precision (scalar loops, explicit
// broadcasting index math, triple-loop matmul), rounding each node's result
// to float. This is the oracle the backends are tested against.
std::vector<HostTensor> evaluate_reference(const Graph& graph, const std::vector<HostTensor>& inputs);

struct RandomGraphOptions {
	int num_ops = 16;
	bool allow_matmul = true;
};

// A random graph built from unary, binary (with every broadcasting pattern)
// and matmul nodes over inputs and constants. The generator tracks an
// interval bound for every value and only emits an op when it is
// numerically safe (log/sqrt of positive values, division by values bounded
// away from zero, no overflow), so outputs are finite and meaningful. Values
// nothing reads are left in place, which gives dead-node elimination work;
// some ops combine two constants, which gives constant folding work.
Graph make_random_graph(std::uint64_t seed, const RandomGraphOptions& options = {});

}
