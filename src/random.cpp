#include "minicompiler/random.hpp"

namespace minicompiler {
namespace {

std::uint64_t splitmix64(std::uint64_t& state) {
	std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

}

void fill_uniform(float* out, std::size_t count, std::uint64_t seed, float lo, float hi) {
	std::uint64_t state = seed;
	const float range = hi - lo;
	for (std::size_t i=0; i<count; ++i) {
		const float u = static_cast<float>(splitmix64(state) >> 40) * 0x1p-24f;  // exact
		// volatile keeps the multiply and the add separately rounded. Without
		// it the compiler may contract them into one FMA, and NumPy, which
		// rounds after each step, would disagree in the last bit.
		volatile float scaled = range * u;
		out[i] = lo + scaled;
	}
}

std::vector<float> uniform_values(std::size_t count, std::uint64_t seed, float lo, float hi) {
	std::vector<float> v(count);
	fill_uniform(v.data(), count, seed, lo, hi);
	return v;
}

}
