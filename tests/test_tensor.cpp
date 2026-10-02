#include "minicompiler/tensor.hpp"

#include <gtest/gtest.h>

#include <sstream>

using namespace minicompiler;

TEST(TensorType, CountsElementsAndBytes) {
	TensorType t({4, 8, 2});
	EXPECT_EQ(t.rank(), 3u);
	EXPECT_EQ(t.num_elements(), 64u);
	EXPECT_EQ(t.size_bytes(), 256u);

	TensorType d({3}, DType::Float64);
	EXPECT_EQ(d.size_bytes(), 24u);
}

TEST(TensorType, ScalarHasOneElement) {
	TensorType s(Shape{});
	EXPECT_EQ(s.rank(), 0u);
	EXPECT_EQ(s.num_elements(), 1u);
}

TEST(TensorType, EqualityComparesShapeAndDtype) {
	EXPECT_EQ(TensorType({2, 3}), TensorType({2, 3}));
	EXPECT_NE(TensorType({2, 3}), TensorType({3, 2}));
	EXPECT_NE(TensorType({2, 3}), TensorType({2, 3}, DType::Int32));
}

TEST(TensorType, PrintsCompactForm) {
	EXPECT_EQ(to_string(TensorType({4, 8})), "f32[4,8]");
	EXPECT_EQ(to_string(TensorType(Shape{})), "f32[]");
	std::ostringstream os;
	os << TensorType({1}, DType::Bool);
	EXPECT_EQ(os.str(), "bool[1]");
}
