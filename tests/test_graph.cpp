#include "minicompiler/graph.hpp"
#include "minicompiler/graph_builder.hpp"

#include <gtest/gtest.h>

using namespace minicompiler;

TEST(Broadcast, FollowsNumpyRules) {
	EXPECT_EQ(broadcast_shapes({4, 8}, {4, 8}).value(), (Shape{4, 8}));
	EXPECT_EQ(broadcast_shapes({4, 8}, {8}).value(), (Shape{4, 8}));      // row vector (bias)
	EXPECT_EQ(broadcast_shapes({4, 1}, {1, 8}).value(), (Shape{4, 8}));   // outer product shape
	EXPECT_EQ(broadcast_shapes({}, {3, 5}).value(), (Shape{3, 5}));       // scalar
	EXPECT_EQ(broadcast_shapes({2, 1, 6}, {3, 1}).value(), (Shape{2, 3, 6}));
	EXPECT_FALSE(broadcast_shapes({4, 8}, {4}).ok());
	EXPECT_FALSE(broadcast_shapes({2, 3}, {3, 2}).ok());
}

TEST(ShapeInference, ElementwiseAndMatmul) {
	const TensorType a({4, 8});
	EXPECT_EQ(infer_result_type(OpKind::Tanh, {a}).value(), a);
	EXPECT_EQ(infer_result_type(OpKind::Add, {a, TensorType({8})}).value(), a);
	EXPECT_EQ(infer_result_type(OpKind::MatMul, {a, TensorType({8, 3})}).value(), TensorType({4, 3}));

	EXPECT_FALSE(infer_result_type(OpKind::MatMul, {a, TensorType({4, 3})}).ok());   // inner dims differ
	EXPECT_FALSE(infer_result_type(OpKind::MatMul, {a, TensorType({8})}).ok());      // rank 1
	EXPECT_FALSE(infer_result_type(OpKind::Add, {a}).ok());                          // arity
	EXPECT_FALSE(infer_result_type(OpKind::Neg, {TensorType({4}, DType::Int32)}).ok());
}

TEST(Graph, BuilderProducesTopologicallyOrderedNodes) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {2, 3});
	NodeId w = b.constant("w", {3, 4}, std::vector<float>(12, 0.5f));
	NodeId y = b.relu(b.matmul(x, w, "xw"), "y");
	b.output(y);
	Result<Graph> g = std::move(b).build();
	ASSERT_TRUE(g.ok()) << g.error().message;

	ASSERT_EQ(g->num_nodes(), 4u);
	for (NodeId id=0; id<static_cast<NodeId>(g->num_nodes()); ++id) {
		for (NodeId in : g->node(id).inputs) EXPECT_LT(in, id);
	}
	EXPECT_EQ(g->node(*g->find("xw")).type, TensorType({2, 4}));
	EXPECT_EQ(g->inputs(), std::vector<NodeId>{x});
	EXPECT_EQ(g->outputs(), std::vector<NodeId>{y});
	EXPECT_TRUE(g->is_output(y));
	EXPECT_FALSE(g->is_output(x));
}

TEST(Graph, GeneratesNamesForUnnamedNodes) {
	Graph g;
	NodeId x = g.add_input("x", TensorType({4})).value();
	NodeId y = g.add_op(OpKind::Exp, {x}).value();
	EXPECT_EQ(g.node(y).name, "exp_1");
}

TEST(Graph, RejectsInvalidNodes) {
	Graph g;
	NodeId x = g.add_input("x", TensorType({4, 8})).value();
	EXPECT_FALSE(g.add_op(OpKind::Add, {x, 7}).ok());                                      // missing operand
	EXPECT_FALSE(g.add_op(OpKind::Add, {x, g.add_input("v", TensorType({4})).value()}).ok());  // broadcast
	EXPECT_FALSE(g.add_op(OpKind::Input, {}).ok());                                         // not an op
	EXPECT_FALSE(g.add_input("i", TensorType({4}, DType::Int32)).ok());
	EXPECT_FALSE(g.add_input("z", TensorType({0, 4})).ok());
	EXPECT_FALSE(g.add_constant("c", TensorType({2, 2}), std::vector<float>{1, 2, 3}).ok());
	EXPECT_FALSE(g.add_output(42).ok());
	ASSERT_TRUE(g.add_output(x).ok());
	EXPECT_FALSE(g.add_output(x).ok());  // duplicate
}

TEST(Graph, RejectsTensorsWithTooManyElements) {
	Graph g;
	// 7 * 7905747460161236407 wraps around to 1 in 64-bit arithmetic.
	EXPECT_FALSE(g.add_input("x", TensorType({7, 7905747460161236407})).ok());
	EXPECT_FALSE(g.add_constant("c", TensorType({4294967296, 4294967296}), std::vector<float>{}).ok());
	// Operands within the limit can still produce a result beyond it.
	NodeId col = g.add_input("col", TensorType({65536, 1})).value();
	NodeId row = g.add_input("row", TensorType({1, 65536})).value();
	EXPECT_FALSE(g.add_op(OpKind::Add, {col, row}).ok());     // [65536,65536] is 2^32 elements
	EXPECT_FALSE(g.add_op(OpKind::MatMul, {col, row}).ok());
	EXPECT_TRUE(g.add_input("largest", TensorType({46340, 46340})).ok());  // 2147395600 < 2^31 - 1
}

TEST(Graph, BuilderErrorsAreSticky) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {4, 8});
	NodeId bad = b.matmul(x, x, "bad");  // [4,8] x [4,8]
	EXPECT_EQ(bad, kNoNode);
	EXPECT_EQ(b.relu(x), kNoNode);  // ignored after the first error
	Result<Graph> g = std::move(b).build();
	ASSERT_FALSE(g.ok());
	EXPECT_NE(g.error().message.find("bad"), std::string::npos) << g.error().message;
}

TEST(Graph, VerifyRequiresOutputs) {
	Graph g;
	(void)g.add_input("x", TensorType({4}));
	EXPECT_FALSE(g.verify().ok());
}

TEST(Graph, ComputesUsersOncePerUser) {
	GraphBuilder b("diamond");
	NodeId x = b.input("x", {8});
	NodeId sq = b.mul(x, x, "sq");  // reads x twice
	NodeId e = b.exp(x, "e");
	NodeId y = b.add(sq, e, "y");
	b.output(y);
	Graph g = std::move(b).build().value();

	const auto users = g.compute_users();
	EXPECT_EQ(users[x], (std::vector<NodeId>{sq, e}));
	EXPECT_EQ(users[sq], std::vector<NodeId>{y});
	EXPECT_TRUE(users[y].empty());
}

TEST(Graph, CopyNodeSharesConstantData) {
	GraphBuilder b("g");
	NodeId c = b.constant("c", {3}, {1, 2, 3});
	b.output(b.neg(c));
	Graph src = std::move(b).build().value();

	Graph dst("copy");
	NodeId c2 = dst.copy_node(src.node(c), {}).value();
	EXPECT_EQ(dst.node(c2).constant.get(), src.node(c).constant.get());
	EXPECT_EQ(dst.node(c2).name, "c");
}

TEST(Graph, ValidatesFusedNodes) {
	Graph g;
	NodeId x = g.add_input("x", TensorType({4, 8})).value();
	NodeId v = g.add_input("v", TensorType({8})).value();

	auto program = std::make_shared<FusedProgram>();
	program->instrs = {FusedInstr::load(0), FusedInstr::load(1), FusedInstr::binary(OpKind::Mul, 0, 1),
	                   FusedInstr::unary(OpKind::Tanh, 2)};
	EXPECT_TRUE(g.add_fused({x, v}, program, TensorType({4, 8}), "f").ok());
	EXPECT_FALSE(g.add_fused({x}, program, TensorType({4, 8}), "f").ok());         // loads input 1 of 1
	EXPECT_FALSE(g.add_fused({x, v}, program, TensorType({8}), "f").ok());         // x doesn't broadcast to [8]

	auto forward_ref = std::make_shared<FusedProgram>();
	forward_ref->instrs = {FusedInstr::load(0), FusedInstr::unary(OpKind::Exp, 1)};
	EXPECT_FALSE(g.add_fused({x}, forward_ref, TensorType({4, 8}), "f").ok());     // operand not earlier
}

TEST(GraphStats, CountsComputeNodesAndOps) {
	GraphBuilder b("g");
	NodeId x = b.input("x", {4, 8});
	NodeId w = b.constant("w", {8, 8}, std::vector<float>(64, 1.0f));
	NodeId h = b.matmul(x, w);
	NodeId y = b.tanh(b.add(h, x));
	b.output(y);
	Graph g = std::move(b).build().value();

	GraphStats s = compute_stats(g);
	EXPECT_EQ(s.nodes, 5u);
	EXPECT_EQ(s.inputs, 1u);
	EXPECT_EQ(s.constants, 1u);
	EXPECT_EQ(s.compute_nodes, 3u);
	EXPECT_EQ(s.matmuls, 1u);
	EXPECT_EQ(s.elementwise_ops, 2u);
	EXPECT_EQ(s.op_counts.at("tanh"), 1u);
}
