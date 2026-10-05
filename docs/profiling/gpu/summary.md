# Nsight Systems summary

One NVTX range per synchronized call (see `bench/profile_gpu.py`). *Wall* is the range's median duration on the host: Python, launches and the wait for the GPU. *GPU span* is the median time from the start of the call's first GPU operation to the end of its last (Nsight's projection of the range onto the GPU), gaps included. *Kernel time* is the sum over the call's kernels of their average durations. Wall minus kernel time is the time the call spent outside kernels.

## gelu_32mb

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 279.1 µs | 211.2 µs | 213.0 µs | 1 | `mc_elementwise` 1 × 213.0 µs |
| eager | 50 | 2420.5 µs | 2256.4 µs | 2373.5 µs | 11 | `vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, binary_in...` 3 × 312.4 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 3 × 220.5 µs; `vectorized_elementwise_kernel<4, CUDAFunctor_add<float>, array<char *, 3>, 0>` 1 × 330.5 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<CUDAFunctor_add<float>...` 1 × 227.1 µs; 3 more |
| eager_idiomatic | 50 | 280.6 µs | 210.1 µs | 211.4 µs | 1 | `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 211.4 µs |
| compile | 50 | 363.2 µs | 211.0 µs | 212.2 µs | 1 | `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 212.2 µs |
| compile_cudagraphs | 50 | 631.5 µs | 426.7 µs | 482.7 µs | 2 | `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 267.9 µs; `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 214.8 µs |

## mlp_b512

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 475.7 µs | 404.4 µs | 397.9 µs | 5 | `matmul_vectorized<8, 1, 1>` 2 × 178.1 µs; `mc_elementwise` 2 × 17.0 µs; `sum_splits<1>` 1 × 7.6 µs |
| eager | 50 | 859.5 µs | 772.1 µs | 736.2 µs | 22 | `ampere_sgemm_128x128_nn` 1 × 148.1 µs; `ampere_sgemm_128x64_nn` 1 × 140.4 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<CUDAFunctor_add<float>...` 6 × 19.8 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 4 × 27.9 µs; 6 more |
| eager_idiomatic | 50 | 420.1 µs | 370.2 µs | 350.5 µs | 6 | `ampere_sgemm_128x128_nn` 1 × 148.7 µs; `ampere_sgemm_128x64_nn` 1 × 141.4 µs; `batch_norm_transform_input_channels_last_kernel<float, float, float, 4>` 1 × 26.9 µs; `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 25.9 µs; 2 more |
| compile | 50 | 502.8 µs | 342.0 µs | 327.3 µs | 4 | `ampere_sgemm_128x128_nn` 1 × 152.6 µs; `ampere_sgemm_128x64_nn` 1 × 144.3 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 25.5 µs; `triton_poi_fused_add_1` 1 × 4.9 µs |
| compile_cudagraphs | 50 | 601.4 µs | 397.3 µs | 387.7 µs | 5 | `ampere_sgemm_128x128_nn` 1 × 151.4 µs; `ampere_sgemm_128x64_nn` 1 × 144.3 µs; `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 62.1 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 25.3 µs; 1 more |

## mlp_b128

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 160.6 µs | 123.4 µs | 115.5 µs | 6 | `matmul_vectorized<8, 1, 1>` 2 × 48.3 µs; `sum_splits<1>` 2 × 5.5 µs; `mc_elementwise` 2 × 4.0 µs |
| eager | 50 | 590.8 µs | 506.1 µs | 203.8 µs | 22 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 68.4 µs; `ampere_sgemm_128x32_nn` 1 × 64.3 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<CUDAFunctor_add<float>...` 6 × 3.8 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 4 × 5.0 µs; 6 more |
| eager_idiomatic | 50 | 276.5 µs | 219.0 µs | 144.7 µs | 6 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 68.6 µs; `ampere_sgemm_128x32_nn` 1 × 64.7 µs; `batch_norm_transform_input_channels_last_kernel<float, float, float, 4>` 1 × 4.9 µs; `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 2.8 µs; 2 more |
| compile | 50 | 382.7 µs | 187.9 µs | 141.2 µs | 4 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 68.8 µs; `ampere_sgemm_128x32_nn` 1 × 64.4 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 5.7 µs; `triton_poi_fused_add_1` 1 × 2.4 µs |
| compile_cudagraphs | 50 | 398.7 µs | 205.1 µs | 195.8 µs | 5 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 69.5 µs; `ampere_sgemm_128x32_nn` 1 × 65.0 µs; `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 53.8 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 5.4 µs; 1 more |

## gelu_16kb

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 44.6 µs | 7.3 µs | 7.3 µs | 1 | `mc_elementwise` 1 × 7.3 µs |
| eager | 50 | 226.6 µs | 179.2 µs | 87.0 µs | 11 | `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 3 × 8.6 µs; `vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, binary_in...` 3 × 7.0 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<CUDAFunctor_add<float>...` 1 × 8.7 µs; `vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, binary_in...` 1 × 8.4 µs; 3 more |
| eager_idiomatic | 50 | 47.6 µs | 7.3 µs | 7.3 µs | 1 | `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 7.3 µs |
| compile | 50 | 102.0 µs | 7.5 µs | 7.5 µs | 1 | `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 7.5 µs |
| compile_cudagraphs | 50 | 198.4 µs | 22.6 µs | 18.9 µs | 2 | `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 11.4 µs; `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 7.5 µs |
