#include "minicompiler/graph_builder.hpp"
#include "minicompiler/runtime/memory_plan.hpp"

#include "test_util.hpp"

#include <gtest/gtest.h>

using namespace minicompiler;

TEST(MemoryPlan, ChainPingPongsBetweenTwoBuffers) {
	GraphBuilder b("chain");
	NodeId v = b.input("x", {256});
	for (int i=0; i<6; ++i) v = b.tanh(v);
	b.output(v);
	Graph g = std::move(b).build().value();

	MemoryPlan plan = plan_memory(g);
	EXPECT_EQ(plan.buffer_bytes.size(), 2u);
	EXPECT_EQ(plan.bytes_without_reuse, 6u * 1024u);
	EXPECT_EQ(plan.bytes_with_reuse(), 2u * 1024u);
	EXPECT_EQ(plan.buffer_of[g.inputs()[0]], MemoryPlan::kNotPooled);
}

TEST(MemoryPlan, KeepsOperandsUntilTheirLastUse) {
	GraphBuilder b("diamond");
	NodeId x = b.input("x", {8});
	NodeId a = b.exp(x, "a");
	NodeId n = b.neg(a, "n");
	NodeId c = b.add(a, n, "c");  // a is still needed here
	b.output(c);
	Graph g = std::move(b).build().value();

	MemoryPlan plan = plan_memory(g);
	EXPECT_EQ(plan.last_use[a], c);
	EXPECT_NE(plan.buffer_of[a], plan.buffer_of[n]);
	// c is allocated while a and n are still live, so it cannot alias them.
	EXPECT_NE(plan.buffer_of[c], plan.buffer_of[a]);
	EXPECT_NE(plan.buffer_of[c], plan.buffer_of[n]);
}

TEST(MemoryPlan, NeverReusesAnOutputBuffer) {
	GraphBuilder b("outputs");
	NodeId x = b.input("x", {8});
	NodeId y1 = b.exp(x, "y1");
	NodeId t = b.neg(x, "t");
	NodeId y2 = b.tanh(t, "y2");
	b.output(y1);
	b.output(y2);
	Graph g = std::move(b).build().value();

	MemoryPlan plan = plan_memory(g);
	EXPECT_EQ(plan.last_use[y1], static_cast<NodeId>(g.num_nodes()));
	EXPECT_NE(plan.buffer_of[y2], plan.buffer_of[y1]);
	EXPECT_NE(plan.buffer_of[t], plan.buffer_of[y1]);
}

TEST(MemoryPlan, DeadValueFreesItsBufferImmediately) {
	GraphBuilder b("dead");
	NodeId x = b.input("x", {8});
	NodeId dead = b.exp(x, "dead");
	NodeId y = b.neg(x, "y");
	b.output(y);
	Graph g = std::move(b).build().value();

	MemoryPlan plan = plan_memory(g);
	EXPECT_EQ(plan.last_use[dead], kNoNode);
	EXPECT_EQ(plan.buffer_of[y], plan.buffer_of[dead]);
	EXPECT_EQ(plan.buffer_bytes.size(), 1u);
}

TEST(MemoryPlan, GrowsAFreeBufferWhenNoneIsLargeEnough) {
	GraphBuilder b("grow");
	NodeId v = b.input("v", {8});
	NodeId m = b.input("m", {4, 8});
	NodeId a = b.exp(v, "a");       // 32 bytes
	NodeId n = b.neg(a, "n");       // 32 bytes; frees a
	NodeId c = b.add(m, n, "c");    // 128 bytes: takes a's buffer and grows it
	b.output(c);
	Graph g = std::move(b).build().value();

	MemoryPlan plan = plan_memory(g);
	EXPECT_EQ(plan.buffer_bytes.size(), 2u);
	EXPECT_EQ(plan.buffer_of[c], plan.buffer_of[a]);
	EXPECT_EQ(plan.buffer_bytes[static_cast<std::size_t>(plan.buffer_of[c])], 128u);
}

TEST(MemoryPlan, SharedBuffersNeverHoldTwoLiveValuesOnRandomGraphs) {
	for (std::uint64_t seed=1; seed<=200; ++seed) {
		Graph g = testutil::make_random_graph(seed, {24, true});
		MemoryPlan plan = plan_memory(g);
		ASSERT_LE(plan.bytes_with_reuse(), plan.bytes_without_reuse) << "seed " << seed;
		for (NodeId u=0; u<static_cast<NodeId>(g.num_nodes()); ++u) {
			const std::int32_t buf = plan.buffer_of[u];
			if (!is_compute(g.node(u).op)) {
				EXPECT_EQ(buf, MemoryPlan::kNotPooled);
				continue;
			}
			ASSERT_GE(plan.buffer_bytes[static_cast<std::size_t>(buf)], g.node(u).type.size_bytes()) << "seed " << seed;
			for (NodeId v=u+1; v<static_cast<NodeId>(g.num_nodes()); ++v) {
				if (plan.buffer_of[v] != buf) continue;
				// u must be dead (last read strictly before v is written).
				EXPECT_LT(plan.last_use[u], v) << "seed " << seed << ": %" << u << " and %" << v << " overlap";
			}
		}
	}
}
