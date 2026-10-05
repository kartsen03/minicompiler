#pragma once

// Timing, statistics and JSON output shared by mcc and the benchmark
// programs, so every number is measured the same way: `warmup` untimed runs,
// then `reps` individually timed runs, summarized by their median.

#include <Eigen/Core>

#ifdef MINICOMPILER_HAVE_CUDA
#include "minicompiler/cuda/cuda_backend.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace minicompiler::bench {

struct TimingStats {
	int warmup = 0;
	int reps = 0;
	double median_ms = 0;
	double min_ms = 0;
	double p10_ms = 0;
	double p90_ms = 0;
	double mean_ms = 0;
};

// Linear-interpolated percentile of sorted samples, q in [0, 1].
inline double percentile(const std::vector<double>& sorted, double q) {
	if (sorted.empty()) return 0.0;
	const double pos = q * static_cast<double>(sorted.size() - 1);
	const auto lo = static_cast<std::size_t>(std::floor(pos));
	const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
	return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

inline TimingStats summarize(std::vector<double> samples_ms, int warmup) {
	std::sort(samples_ms.begin(), samples_ms.end());
	TimingStats s;
	s.warmup = warmup;
	s.reps = static_cast<int>(samples_ms.size());
	s.median_ms = percentile(samples_ms, 0.5);
	s.min_ms = samples_ms.empty() ? 0.0 : samples_ms.front();
	s.p10_ms = percentile(samples_ms, 0.1);
	s.p90_ms = percentile(samples_ms, 0.9);
	double sum = 0.0;
	for (double v : samples_ms) sum += v;
	s.mean_ms = samples_ms.empty() ? 0.0 : sum / static_cast<double>(samples_ms.size());
	return s;
}

// Times fn() with a monotonic wall clock: `warmup` untimed calls, then `reps`
// timed calls.
template <typename Fn>
TimingStats time_cpu(Fn&& fn, int warmup, int reps) {
	for (int i=0; i<warmup; ++i) fn();
	std::vector<double> samples;
	samples.reserve(static_cast<std::size_t>(reps));
	for (int i=0; i<reps; ++i) {
		const auto t0 = std::chrono::steady_clock::now();
		fn();
		const auto t1 = std::chrono::steady_clock::now();
		samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
	}
	return summarize(std::move(samples), warmup);
}

// A small streaming JSON writer (objects, arrays, strings, numbers, bools).
class JsonWriter {
public:
	JsonWriter& begin_object() { return open('{'); }
	JsonWriter& end_object() { return close('}'); }
	JsonWriter& begin_array() { return open('['); }
	JsonWriter& end_array() { return close(']'); }

	JsonWriter& key(std::string_view k) {
		separate();
		write_string(k);
		os_ << ": ";
		after_key_ = true;
		return *this;
	}
	JsonWriter& value(std::string_view s) {
		separate();
		write_string(s);
		return *this;
	}
	JsonWriter& value(const char* s) { return value(std::string_view(s)); }
	JsonWriter& value(const std::string& s) { return value(std::string_view(s)); }
	JsonWriter& value(bool b) {
		separate();
		os_ << (b ? "true" : "false");
		return *this;
	}
	JsonWriter& value(double d) {
		separate();
		if (std::isfinite(d)) {
			std::ostringstream tmp;
			tmp.precision(6);
			tmp << d;
			os_ << tmp.str();
		} else {
			os_ << "null";
		}
		return *this;
	}
	JsonWriter& value(std::int64_t v) {
		separate();
		os_ << v;
		return *this;
	}
	JsonWriter& value(int v) { return value(static_cast<std::int64_t>(v)); }
	JsonWriter& value(std::size_t v) { return value(static_cast<std::int64_t>(v)); }

	template <typename T>
	JsonWriter& field(std::string_view k, const T& v) {
		key(k);
		return value(v);
	}

	JsonWriter& timing(std::string_view k, const TimingStats& t) {
		key(k).begin_object();
		field("warmup", t.warmup).field("reps", t.reps).field("median_ms", t.median_ms).field("min_ms", t.min_ms);
		field("p10_ms", t.p10_ms).field("p90_ms", t.p90_ms).field("mean_ms", t.mean_ms);
		return end_object();
	}

	std::string str() const { return os_.str() + "\n"; }

private:
	JsonWriter& open(char c) {
		separate();
		os_ << c;
		first_.push_back(true);
		return *this;
	}
	JsonWriter& close(char c) {
		first_.pop_back();
		os_ << c;
		return *this;
	}
	void separate() {
		if (after_key_) {
			after_key_ = false;
			return;
		}
		if (!first_.empty()) {
			if (!first_.back()) os_ << ", ";
			first_.back() = false;
		}
	}
	void write_string(std::string_view s) {
		os_ << '"';
		for (char c : s) {
			switch (c) {
				case '"': os_ << "\\\""; break;
				case '\\': os_ << "\\\\"; break;
				case '\n': os_ << "\\n"; break;
				case '\t': os_ << "\\t"; break;
				default:
					if (static_cast<unsigned char>(c) < 0x20) {
						os_ << "?";
					} else {
						os_ << c;
					}
			}
		}
		os_ << '"';
	}

	std::ostringstream os_;
	std::vector<bool> first_;
	bool after_key_ = false;
};

inline std::string env_or(const char* name, const char* fallback) {
	const char* v = std::getenv(name);
	return v && *v ? v : fallback;
}

inline std::string cpu_model() {
#if defined(__linux__)
	std::ifstream in("/proc/cpuinfo");
	for (std::string line; std::getline(in, line);) {
		if (line.rfind("model name", 0) == 0) {
			const std::size_t colon = line.find(':');
			if (colon != std::string::npos) return line.substr(colon + 2);
		}
	}
#endif
	return "unknown";
}

inline std::string os_description() {
#if defined(__linux__)
	std::string pretty = "Linux";
	std::ifstream in("/etc/os-release");
	for (std::string line; std::getline(in, line);) {
		if (line.rfind("PRETTY_NAME=", 0) == 0) {
			pretty = line.substr(12);
			pretty.erase(std::remove(pretty.begin(), pretty.end(), '"'), pretty.end());
		}
	}
	utsname u{};
	if (uname(&u) == 0) pretty += std::string(", kernel ") + u.release;
	return pretty;
#elif defined(_WIN32)
	return "Windows";
#else
	return "unknown";
#endif
}

inline std::string compiler_description() {
#if defined(__clang__)
	return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
	return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
	return "msvc " + std::to_string(_MSC_VER);
#else
	return "unknown";
#endif
}

// The machine and build, recorded next to every result. The commit comes from
// MINICOMPILER_COMMIT (set by the benchmark scripts, with "-dirty" when the
// tree has uncommitted changes).
inline void write_environment(JsonWriter& json) {
	json.key("environment").begin_object();
	json.field("commit", env_or("MINICOMPILER_COMMIT", "unknown"));
	json.field("cpu", cpu_model());
	json.field("hardware_threads", static_cast<int>(std::thread::hardware_concurrency()));
	json.field("os", os_description());
	json.field("compiler", compiler_description());
	json.field("eigen", std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
	                        std::to_string(EIGEN_MINOR_VERSION));
#ifdef MINICOMPILER_BUILD_FLAGS
	json.field("build_flags", MINICOMPILER_BUILD_FLAGS);
#endif
	json.field("threads", "1 (Eigen without OpenMP)");
	// Laptop power modes cap clocks, so record which one the host was in (set
	// by the benchmark scripts; WSL cannot query Windows directly).
	json.field("host_power_plan", env_or("MINICOMPILER_HOST_POWER_PLAN", "unknown"));
	json.end_object();
}

#ifdef MINICOMPILER_HAVE_CUDA
// The GPU as queried at runtime, plus the range of SM clocks sampled while
// the benchmark ran (laptop GPUs boost and throttle with power and heat).
inline void write_device(JsonWriter& json, const cuda::DeviceInfo& d, std::vector<double> clocks) {
	json.key("device").begin_object();
	json.field("name", d.name).field("compute_capability", std::to_string(d.cc_major) + "." + std::to_string(d.cc_minor));
	json.field("sm_count", d.sm_count).field("fp32_lanes_per_sm", d.fp32_lanes_per_sm);
	json.field("sm_clock_mhz_cuda_attribute", d.sm_clock_mhz).field("max_sm_clock_mhz_nvml", d.max_sm_clock_mhz);
	json.field("mem_clock_mhz", d.mem_clock_mhz).field("bus_width_bits", d.bus_width_bits);
	json.field("peak_bandwidth_gbs", d.peak_bandwidth_gbs());
	json.field("cuda_runtime", d.runtime_version).field("cuda_driver", d.driver_version);
	if (!clocks.empty()) {
		std::sort(clocks.begin(), clocks.end());
		json.key("sm_clock_mhz_during_run").begin_object();
		json.field("min", clocks.front()).field("median", clocks[clocks.size() / 2]).field("max", clocks.back());
		json.end_object();
	}
	json.end_object();
}
#endif

}
