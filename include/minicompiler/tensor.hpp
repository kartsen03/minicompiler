#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace minicompiler {

enum class DType : std::uint8_t {
	Float32,
	Float64,
	Int32,
	Int64,
	Bool,
};

inline std::size_t DType_size_bytes(DType dt) {
	switch (dt) {
		case DType::Float32: return 4;
		case DType::Float64: return 8;
		case DType::Int32: return 4;
		case DType::Int64: return 8;
		case DType::Bool: return 1;
	}
	return 0;
}

inline const char* DType_name(DType dt) {
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

struct Tensor {
	Shape shape;
	DType dtype = DType::Float32;

	Tensor() = default;
	Tensor(Shape s, DType d) : shape(std::move(s)), dtype(d) {}

	std::size_t rank() const { return shape.size(); }

	std::size_t num_elements() const {
		std::size_t n = 1;
		for (std::int64_t d : shape) {
			n *= static_cast<std::size_t>(d);
		}
		return n;
	}

	std::size_t size_bytes() const {
		return num_elements() * DType_size_bytes(dtype);
	}

	bool operator==(const Tensor& other) const {
		return shape == other.shape && dtype == other.dtype;
	}
	bool operator!=(const Tensor& other) const { return !(*this == other); }
};

inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
	os << DType_name(t.dtype) << "[";
	for (std::size_t i=0; i<t.shape.size(); ++i) {
		if (i>0) os << ",";
		os << t.shape[i];
	}
	os << "]";
	return os;
}

inline Tensor make_tensor(Shape s, DType d = DType::Float32) {
	return Tensor(std::move(s), d);
}

}
