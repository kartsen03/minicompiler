#include "minicompiler/cuda/cuda_backend.hpp"

#include "cuda/cuda_check.hpp"

#include <nvml.h>

namespace minicompiler::cuda {
namespace {

// FP32 multiply-add lanes per SM, by compute capability (CUDA C++ Programming
// Guide, "Arithmetic Instructions": 32-bit floating-point add, multiply and
// multiply-add results per clock cycle per multiprocessor). This is the one
// architectural constant; SM count and clocks are queried from the device.
int fp32_lanes_per_sm(int major, int minor) {
	if (major == 6) return minor == 0 ? 64 : 128;
	if (major == 7) return 64;                     // Volta, Turing (Tesla T4: 7.5)
	if (major == 8) return minor == 0 ? 64 : 128;  // A100: 8.0; GA10x (RTX 30: 8.6) and Ada (8.9): 128
	return 128;                                    // Hopper and later
}

// The NVML handle for a CUDA device, matched by PCI bus id because NVML and
// CUDA may number devices differently.
bool nvml_handle(int device, nvmlDevice_t* handle) {
	static const bool initialized = nvmlInit_v2() == NVML_SUCCESS;
	if (!initialized) return false;
	char bus_id[64];
	if (cudaDeviceGetPCIBusId(bus_id, sizeof bus_id, device) != cudaSuccess) return false;
	return nvmlDeviceGetHandleByPciBusId_v2(bus_id, handle) == NVML_SUCCESS;
}

}

Result<DeviceInfo> query_device(int device) {
	int count = 0;
	MC_CUDA_RETURN(cudaGetDeviceCount(&count));
	if (device < 0 || device >= count) {
		return Error{"no CUDA device " + std::to_string(device) + " (" + std::to_string(count) + " present)"};
	}
	cudaDeviceProp prop{};
	MC_CUDA_RETURN(cudaGetDeviceProperties(&prop, device));
	int clock_khz = 0;
	int mem_clock_khz = 0;
	int bus_width = 0;
	MC_CUDA_RETURN(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
	MC_CUDA_RETURN(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
	MC_CUDA_RETURN(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device));

	DeviceInfo info;
	info.name = prop.name;
	info.cc_major = prop.major;
	info.cc_minor = prop.minor;
	info.sm_count = prop.multiProcessorCount;
	info.fp32_lanes_per_sm = fp32_lanes_per_sm(prop.major, prop.minor);
	info.sm_clock_mhz = clock_khz / 1000.0;
	info.mem_clock_mhz = mem_clock_khz / 1000.0;
	info.bus_width_bits = bus_width;
	info.total_memory = prop.totalGlobalMem;
	MC_CUDA_RETURN(cudaRuntimeGetVersion(&info.runtime_version));
	MC_CUDA_RETURN(cudaDriverGetVersion(&info.driver_version));
	nvmlDevice_t handle;
	unsigned int mhz = 0;
	if (nvml_handle(device, &handle) && nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) {
		info.max_sm_clock_mhz = mhz;
	}
	return info;
}

double current_sm_clock_mhz(int device) {
	nvmlDevice_t handle;
	unsigned int mhz = 0;
	if (nvml_handle(device, &handle) && nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) return mhz;
	return 0.0;
}

}
