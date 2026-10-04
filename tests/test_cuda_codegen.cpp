// The CUDA kernel generator is plain C++, so these tests run without a GPU.

#include "cuda/codegen.hpp"
#include "minicompiler/graph_builder.hpp"
#include "minicompiler/passes/operator_fusion.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace minicompiler;
using cuda::generate_elementwise_kernel;
using cuda::single_op_program;

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
	return haystack.find(needle) != std::string::npos;
}

std::size_t count(const std::string& haystack, const std::string& needle) {
	std::size_t n = 0;
	for (std::size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + 1)) ++n;
	return n;
}

}

TEST(CudaCodegen, FusedGeluBecomesOneVectorizedKernel) {
	GraphBuilder b("gelu");
	NodeId x = b.input("x", {8, 64});
	NodeId x3 = b.mul(b.mul(x, x), x);
	NodeId inner = b.add(x, b.mul(x3, b.scalar("coef", 0.044715f)));
	NodeId th = b.tanh(b.mul(inner, b.scalar("c", 0.7978846f)));
	b.output(b.mul(b.mul(x, b.scalar("half", 0.5f)), b.add(th, b.scalar("one", 1.0f))));
	const Graph fused = fuse_elementwise(std::move(b).build().value()).value();
	const Node& y = fused.node(fused.outputs()[0]);
	ASSERT_EQ(y.op, OpKind::FusedElementwise);

	const cuda::ElementwiseKernel k = generate_elementwise_kernel(*y.fused, {{8, 64}}, y.type.shape);
	EXPECT_TRUE(k.vectorized);
	EXPECT_TRUE(contains(k.source, "mc_elementwise(float* __restrict__ out, const float* __restrict__ in0, unsigned n)"));
	EXPECT_TRUE(contains(k.source, "reinterpret_cast<const float4*>(in0)[i]"));
	EXPECT_TRUE(contains(k.source, "reinterpret_cast<float4*>(out)[i] = o;"));
	// Constants are baked in as exact bit patterns, once, before the loop.
	EXPECT_EQ(count(k.source, "__uint_as_float(0x3f000000u);  // 0.5"), 1u);
	EXPECT_LT(k.source.find("0x3f000000u"), k.source.find("for (unsigned i"));
	EXPECT_EQ(count(k.source, "tanhf("), 4u);  // one per float4 lane
}

TEST(CudaCodegen, ReadsABiasRowPeriodically) {
	const FusedProgram add = single_op_program(OpKind::Add);
	const auto vec = generate_elementwise_kernel(add, {{16, 64}, {64}}, {16, 64});
	EXPECT_TRUE(vec.vectorized);
	EXPECT_TRUE(contains(vec.source, "reinterpret_cast<const float4*>(in1)[i % 16u]")) << vec.source;

	const auto scalar = generate_elementwise_kernel(add, {{16, 64}, {64}}, {16, 64}, false);
	EXPECT_FALSE(scalar.vectorized);
	EXPECT_TRUE(contains(scalar.source, "in1[i % 64u]")) << scalar.source;

	// A period that is not a multiple of 4 would let a float4 straddle two repetitions.
	const auto odd = generate_elementwise_kernel(add, {{16, 6}, {6}}, {16, 6});
	EXPECT_FALSE(odd.vectorized);
	EXPECT_TRUE(contains(odd.source, "in1[i % 6u]"));
}

TEST(CudaCodegen, UsesConstantIndexMathForGeneralBroadcasts) {
	const FusedProgram mul = single_op_program(OpKind::Mul);
	const auto k = generate_elementwise_kernel(mul, {{5, 4, 6}, {5, 1, 6}}, {5, 4, 6});
	EXPECT_FALSE(k.vectorized);
	EXPECT_TRUE(contains(k.source, "in0[i]"));
	EXPECT_TRUE(contains(k.source, "in1[(i / 24u) * 6u + (i % 6u)]")) << k.source;

	const auto column = generate_elementwise_kernel(mul, {{7, 9}, {7, 1}}, {7, 9});
	EXPECT_TRUE(contains(column.source, "in1[(i / 9u)]")) << column.source;
}

TEST(CudaCodegen, LoadsScalarInputsOnceBeforeTheLoop) {
	const auto k = generate_elementwise_kernel(single_op_program(OpKind::Div), {{1000}, {}}, {1000});
	const std::size_t load = k.source.find("const float r1 = in1[0];");
	ASSERT_NE(load, std::string::npos) << k.source;
	EXPECT_LT(load, k.source.find("for (unsigned i"));
	EXPECT_TRUE(contains(k.source, "const float r2 = r0 / r1;"));
	EXPECT_TRUE(k.vectorized);  // 1000 is a multiple of 4
	EXPECT_FALSE(generate_elementwise_kernel(single_op_program(OpKind::Div), {{1001}, {}}, {1001}).vectorized);
}

TEST(CudaCodegen, WritesImmediatesAsExactBitPatterns) {
	FusedProgram p;
	p.instrs = {FusedInstr::load(0), FusedInstr::imm(-0.1f), FusedInstr::binary(OpKind::Mul, 0, 1),
	            FusedInstr::imm(INFINITY), FusedInstr::binary(OpKind::Add, 2, 3)};
	const auto k = generate_elementwise_kernel(p, {{8}}, {8});
	EXPECT_TRUE(contains(k.source, "__uint_as_float(0xbdcccccdu)")) << k.source;  // -0.1f
	EXPECT_TRUE(contains(k.source, "__uint_as_float(0x7f800000u)")) << k.source;  // +inf
}

TEST(CudaCodegen, EmitsEveryOp) {
	for (OpKind op : {OpKind::Neg, OpKind::Exp, OpKind::Log, OpKind::Sqrt, OpKind::Relu, OpKind::Sigmoid,
	                  OpKind::Tanh}) {
		const FusedProgram p = single_op_program(op);
		ASSERT_EQ(p.instrs.size(), 2u);
		EXPECT_FALSE(contains(generate_elementwise_kernel(p, {{4}}, {4}).source, "unsupported")) << op_name(op);
	}
	for (OpKind op : {OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Div}) {
		const FusedProgram p = single_op_program(op);
		ASSERT_EQ(p.instrs.size(), 3u);
		EXPECT_FALSE(contains(generate_elementwise_kernel(p, {{4}, {4}}, {4}).source, "unsupported")) << op_name(op);
	}
}
