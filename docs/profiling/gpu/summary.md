# Nsight Systems summary

One NVTX range per synchronized call (see `bench/profile_gpu.py`). *Wall* is the range's median duration on the host: Python, launches and the wait for the GPU. *GPU span* is the median time from the start of the call's first GPU operation to the end of its last (Nsight's projection of the range onto the GPU), gaps included. *Kernel time* is the sum over the call's kernels of their average durations. Wall minus kernel time is the time the call spent outside kernels.

## gelu_32mb

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 249.7 µs | 213.4 µs | 215.4 µs | 1 | `mc_elementwise` 1 × 215.4 µs |
| eager | 50 | 2379.6 µs | 2246.0 µs | 2274.7 µs | 11 | `vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, binary_in...` 3 × 293.9 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 3 × 216.3 µs; `vectorized_elementwise_kernel<4, CUDAFunctor_add<float>, array<char *, 3>, 0>` 1 × 316.6 µs; `vectorized_elementwise_kernel<4, tanh_kernel_cuda(at::TensorIteratorBase &)::...` 1 × 213.5 µs; 3 more |
| eager_idiomatic | 50 | 248.3 µs | 212.5 µs | 214.3 µs | 1 | `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 214.3 µs |
| compile | 50 | 378.4 µs | 212.8 µs | 214.4 µs | 1 | `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 214.4 µs |
| compile_cudagraphs | 50 | 647.0 µs | 431.5 µs | 432.3 µs | 2 | `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 217.0 µs; `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 215.3 µs |

## mlp_b512

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 958.5 µs | 850.8 µs | 845.3 µs | 4 | `matmul_register_tiled<64, 64, 8, 4, 4>` 1 × 495.7 µs; `matmul_register_tiled<128, 128, 8, 8, 8>` 1 × 314.8 µs; `mc_elementwise` 2 × 17.4 µs |
| eager | 50 | 874.2 µs | 790.0 µs | 749.8 µs | 22 | `ampere_sgemm_128x64_nn` 1 × 149.6 µs; `ampere_sgemm_128x128_nn` 1 × 148.6 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<CUDAFunctor_add<float>...` 6 × 20.0 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 4 × 28.2 µs; 6 more |
| eager_idiomatic | 50 | 424.6 µs | 370.6 µs | 353.1 µs | 6 | `ampere_sgemm_128x128_nn` 1 × 149.8 µs; `ampere_sgemm_128x64_nn` 1 × 141.9 µs; `batch_norm_transform_input_channels_last_kernel<float, float, float, 4>` 1 × 27.5 µs; `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 26.2 µs; 2 more |
| compile | 50 | 530.1 µs | 341.8 µs | 327.9 µs | 4 | `ampere_sgemm_128x128_nn` 1 × 153.7 µs; `ampere_sgemm_128x64_nn` 1 × 144.7 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 24.6 µs; `triton_poi_fused_add_1` 1 × 4.9 µs |
| compile_cudagraphs | 50 | 621.9 µs | 397.8 µs | 389.4 µs | 5 | `ampere_sgemm_128x128_nn` 1 × 152.7 µs; `ampere_sgemm_128x64_nn` 1 × 144.4 µs; `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 62.8 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 24.8 µs; 1 more |

## mlp_b128

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 629.9 µs | 524.3 µs | 520.1 µs | 4 | `matmul_register_tiled<64, 64, 8, 4, 4>` 2 × 256.1 µs; `mc_elementwise` 2 × 4.0 µs |
| eager | 50 | 656.6 µs | 564.7 µs | 197.4 µs | 22 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 65.5 µs; `ampere_sgemm_128x32_nn` 1 × 60.9 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<CUDAFunctor_add<float>...` 6 × 3.7 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 4 × 4.9 µs; 6 more |
| eager_idiomatic | 50 | 330.4 µs | 249.2 µs | 136.2 µs | 6 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 64.3 µs; `ampere_sgemm_128x32_nn` 1 × 60.4 µs; `batch_norm_transform_input_channels_last_kernel<float, float, float, 4>` 1 × 5.1 µs; `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 2.7 µs; 2 more |
| compile | 50 | 327.8 µs | 160.8 µs | 131.7 µs | 4 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 63.8 µs; `ampere_sgemm_128x32_nn` 1 × 60.0 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 5.4 µs; `triton_poi_fused_add_1` 1 × 2.4 µs |
| compile_cudagraphs | 50 | 401.2 µs | 196.0 µs | 186.8 µs | 5 | `ampere_sgemm_64x32_sliced1x4_nn` 1 × 64.0 µs; `ampere_sgemm_128x32_nn` 1 × 60.8 µs; `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 54.7 µs; `triton_poi_fused_add_div_mul_sqrt_sub_tanh_0` 1 × 5.2 µs; 1 more |

## gelu_16kb

| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |
|---|---:|---:|---:|---:|---:|---|
| minicompiler | 50 | 38.0 µs | 7.4 µs | 7.4 µs | 1 | `mc_elementwise` 1 × 7.4 µs |
| eager | 50 | 220.1 µs | 174.4 µs | 88.5 µs | 11 | `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<BinaryFunctor<float, f...` 3 × 8.8 µs; `vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, binary_in...` 3 × 7.2 µs; `elementwise_kernel<128, 2, void gpu_kernel_impl_nocast<CUDAFunctor_add<float>...` 1 × 8.9 µs; `vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, binary_in...` 1 × 8.5 µs; 3 more |
| eager_idiomatic | 50 | 46.5 µs | 7.5 µs | 7.5 µs | 1 | `vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(at::TensorIteratorBase &,...` 1 × 7.5 µs |
| compile | 50 | 101.3 µs | 7.6 µs | 7.5 µs | 1 | `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 7.5 µs |
| compile_cudagraphs | 50 | 175.9 µs | 22.5 µs | 18.9 µs | 2 | `multi_tensor_apply_kernel<TensorListMetadata<2>, UnaryOpFunctor<float, 2, 1, ...` 1 × 11.4 µs; `triton_poi_fused_add_div_mul_sqrt_tanh_0` 1 × 7.4 µs |
