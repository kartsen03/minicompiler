#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace minicompiler {

enum class DType : std::uint8_t {
	Float32,
	Float64,
	Int32,
	Int64,
	Bool,
};

inline std::size_t dtype_size_bytes(DType dt) {
	switch (dt) {
		case DType::Float32: return 4;
		case DType::Float64: return 8;
		case DType::Int32: return 4;
		case DType::Int64: return 8;
		case DType::Bool: return 1;
	}
	return 0;
}

inline const char* dtype_name(DType dt) {
	switch (dt) {
		case DType::Float32: return "f32";
		case DType::Float64: return "f64";
		case DType::Int32: return "i32";
		case DType::Int64: return "i64";
		case DType::Bool: return "bool";
	}
	return "?";
}

using Shape = std::vector<std::int64_t>;

inline std::size_t num_elements(const Shape& shape) {
	std::size_t n = 1;
	for (std::int64_t d : shape) {
		n *= static_cast<std::size_t>(d);
	}
	return n;
}

// The static type of a value in the IR: its shape and element type. It holds
// no data; constant payloads live on Constant nodes and runtime buffers belong
// to the backends.
struct TensorType {
	Shape shape;
	DType dtype = DType::Float32;

	TensorType() = default;
	explicit TensorType(Shape s, DType d = DType::Float32) : shape(std::move(s)), dtype(d) {}

	std::size_t rank() const { return shape.size(); }
	std::size_t num_elements() const { return minicompiler::num_elements(shape); }
	std::size_t size_bytes() const { return num_elements() * dtype_size_bytes(dtype); }

	bool operator==(const TensorType& other) const {
		return shape == other.shape && dtype == other.dtype;
	}
	bool operator!=(const TensorType& other) const { return !(*this == other); }
};

// "[4,8]"; a scalar is "[]".
inline std::string to_string(const Shape& shape) {
	std::string s = "[";
	for (std::size_t i=0; i<shape.size(); ++i) {
		if (i>0) s += ",";
		s += std::to_string(shape[i]);
	}
	return s + "]";
}

// "f32[4,8]"
inline std::string to_string(const TensorType& t) {
	return dtype_name(t.dtype) + to_string(t.shape);
}

inline std::ostream& operator<<(std::ostream& os, const TensorType& t) {
	return os << to_string(t);
}

}
