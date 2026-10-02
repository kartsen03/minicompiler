#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace minicompiler {

// minicompiler does not use exceptions for control flow. A fallible function
// returns Status (success or an Error) or Result<T> (a T or an Error), and
// the caller checks ok() before using the value.
struct Error {
	std::string message;
};

class [[nodiscard]] Status {
public:
	Status() = default;
	Status(Error error) : error_(std::move(error)) {}

	bool ok() const { return !error_.has_value(); }
	const Error& error() const {
		assert(!ok());
		return *error_;
	}
	const std::string& message() const { return error().message; }

private:
	std::optional<Error> error_;
};

// A minimal stand-in for C++23 std::expected<T, Error>.
template <typename T>
class [[nodiscard]] Result {
public:
	Result(T value) : storage_(std::in_place_index<0>, std::move(value)) {}
	Result(Error error) : storage_(std::in_place_index<1>, std::move(error)) {}

	bool ok() const { return storage_.index() == 0; }

	T& value() & {
		assert(ok());
		return std::get<0>(storage_);
	}
	const T& value() const& {
		assert(ok());
		return std::get<0>(storage_);
	}
	T&& value() && {
		assert(ok());
		return std::get<0>(std::move(storage_));
	}
	const Error& error() const {
		assert(!ok());
		return std::get<1>(storage_);
	}

	T* operator->() { return &value(); }
	const T* operator->() const { return &value(); }
	T& operator*() & { return value(); }
	const T& operator*() const& { return value(); }

	Status status() const { return ok() ? Status() : Status(error()); }

private:
	std::variant<T, Error> storage_;
};

}
