# Nsight Compute summary

One launch of each kernel, full metric set, the GPU held at 899 MHz by Nsight Compute (FP32 peak there: 6,906 GFLOP/s). See `README.md` here for what the numbers show.

## GELU chain at 64 MB

| Variant | Kernels | Time | DRAM read | DRAM written | Traffic ÷ input size | DRAM throughput |
|---|---:|---:|---:|---:|---:|---:|
| unfused | 9 | 4,902.4 µs | 770 MiB | 573 MiB | 21.0 | 92% |
| fused | 1 | 484.6 µs | 66 MiB | 64 MiB | 2.0 | 90% |

## Matmul 128x2048x512 (m × k × n)

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | FMA pipe | Shared-load bank conflicts | L1 hit | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| naive | 712.4 µs | 377 | 40 | 256 × 256 | 76% / 100% | 18% | no shared loads | 88% | 66% | 10% | lg_throttle 19.0, long_scoreboard 6.3, not_selected 3.3 |
| tiled | 718.6 µs | 374 | 38 | 1024 × 64 | 67% / 67% | 8% | 0% of wavefronts | 0% | 69% | 5% | mio_throttle 21.0, long_scoreboard 4.5, not_selected 3.6 |
| register_tiled_128 | 1,053.7 µs | 255 | 128 | 256 × 4 | 17% / 33% | 29% | 0% of wavefronts | 2% | 40% | 2% | long_scoreboard 1.1, not_selected 0.7, wait 0.5 |
| register_tiled_64 | 794.0 µs | 338 | 64 | 256 × 16 | 17% / 67% | 11% | 0% of wavefronts | 1% | 70% | 2% | long_scoreboard 5.5, wait 1.4, short_scoreboard 0.9 |
| vectorized | 624.4 µs | 430 | 115 | 256 × 4 | 17% / 33% | 47% | 0% of wavefronts | 0% | 40% | 3% | long_scoreboard 0.8, not_selected 0.7, barrier 0.3 |
| double_buffered | 529.7 µs | 507 | 125 | 256 × 4 | 17% / 33% | 56% | 0% of wavefronts | 0% | 40% | 4% | not_selected 0.7, dispatch_stall 0.4, short_scoreboard 0.3 |
| split_k | 94.2 µs | 2,850 | 125 | 256 × 28 | 17% / 33% | 53% | 0% of wavefronts | 0% | 49% | 29% | not_selected 0.7, dispatch_stall 0.4, short_scoreboard 0.3 |
| cublas (`ampere_sgemm_64x32_sliced1x4_nn`) | 119.4 µs | 2,248 | 82 | 256 × 64 | 29% / 33% | 46% | 0% of wavefronts | 0% | 78% | 18% | not_selected 1.2, wait 0.8, long_scoreboard 0.7 |

Tensor cores on 128x2048x512: the inputs rounded to each format, FP32 accumulation. Tensor throughput is the tensor ops executed against the peak of the path used, as Nsight Compute reports both.

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | Tensor throughput (path) | Shared-load bank conflicts | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tensor_core_tf32 | 90.4 µs | 2,969 | 128 | 256 × 28 | 17% / 33% | 48% (TF32 → FP32) | 0% of wavefronts | 49% | 30% | math_pipe_throttle 3.7, wait 2.2, long_scoreboard 0.9 |
| cublas_tf32 (`Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4>`) | 67.8 µs | 3,959 | 136 | 128 × 24 | 8% / 8% | 57% (TF32 → FP32) | 0% of wavefronts | 66% | 29% | math_pipe_throttle 3.0, wait 0.6, barrier 0.4 |
| tensor_core_bf16 | 60.3 µs | 4,453 | 165 | 256 × 28 | 17% / 17% | 39% (BF16 → FP32) | 0% of wavefronts | 49% | 45% | math_pipe_throttle 3.8, wait 2.3, barrier 0.9 |
| cublas_bf16 (`cutlass::Kernel2<cutlass_80_tensorop_s1688bf16gemm_64x128...`) | 58.2 µs | 4,609 | 152 | 128 × 16 | 8% / 8% | 38% (BF16 → FP32) | 0% of wavefronts | 60% | 37% | math_pipe_throttle 2.7, wait 0.8, short_scoreboard 0.2 |
| tensor_core_f16 | 54.1 µs | 4,961 | 162 | 256 × 28 | 17% / 17% | 44% (FP16 → FP32) | 0% of wavefronts | 49% | 50% | math_pipe_throttle 3.9, wait 2.1, barrier 1.5 |
| cublas_f16 (`cutlass::Kernel2<cutlass_80_tensorop_s1688f16gemm_128x128...`) | 61.2 µs | 4,390 | 235 | 128 × 16 | 8% / 8% | 38% (FP16 → FP32) | 0% of wavefronts | 50% | 33% | math_pipe_throttle 3.2, wait 0.9, short_scoreboard 0.3 |

## Matmul 2048x2048x2048 (m × k × n)

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | FMA pipe | Shared-load bank conflicts | L1 hit | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| naive | 40,332.6 µs | 426 | 40 | 256 × 16384 | 100% / 100% | 19% | no shared loads | 87% | 50% | 17% | lg_throttle 25.5, long_scoreboard 6.6, not_selected 4.5 |
| tiled | 32,942.4 µs | 522 | 38 | 1024 × 4096 | 67% / 67% | 8% | 0% of wavefronts | 0% | 47% | 11% | mio_throttle 21.0, long_scoreboard 4.5, not_selected 3.6 |
| register_tiled_128 | 6,819.2 µs | 2,519 | 128 | 256 × 256 | 32% / 33% | 40% | 0% of wavefronts | 2% | 68% | 9% | not_selected 1.9, dispatch_stall 0.9, long_scoreboard 0.6 |
| register_tiled_64 | 9,638.5 µs | 1,782 | 64 | 256 × 1024 | 64% / 67% | 31% | 0% of wavefronts | 3% | 51% | 18% | long_scoreboard 4.6, not_selected 2.0, mio_throttle 1.8 |
| vectorized | 4,462.8 µs | 3,850 | 115 | 256 × 256 | 32% / 33% | 59% | 0% of wavefronts | 1% | 71% | 13% | not_selected 2.0, long_scoreboard 0.8, barrier 0.6 |
| double_buffered | 4,411.7 µs | 3,894 | 125 | 256 × 256 | 32% / 33% | 61% | 0% of wavefronts | 5% | 77% | 10% | not_selected 2.2, dispatch_stall 0.7, mio_throttle 0.4 |
| split_k | 4,414.8 µs | 3,891 | 125 | 256 × 256 | 32% / 33% | 61% | 0% of wavefronts | 5% | 77% | 10% | not_selected 2.2, dispatch_stall 0.7, mio_throttle 0.4 |
| cublas (`ampere_sgemm_128x64_nn`) | 4,112.9 µs | 4,177 | 122 | 128 × 512 | 31% / 33% | 64% | 0% of wavefronts | 0% | 90% | 16% | not_selected 1.8, wait 1.0, dispatch_stall 0.9 |

Tensor cores on 2048x2048x2048: the inputs rounded to each format, FP32 accumulation. Tensor throughput is the tensor ops executed against the peak of the path used, as Nsight Compute reports both.

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | Tensor throughput (path) | Shared-load bank conflicts | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tensor_core_tf32 | 3,656.5 µs | 4,698 | 128 | 256 × 256 | 32% / 33% | 68% (TF32 → FP32) | 0% of wavefronts | 71% | 16% | math_pipe_throttle 7.8, wait 2.4, barrier 1.3 |
| cublas_tf32 (`Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_ali...`) | 2,717.1 µs | 6,323 | 224 | 128 × 256 | 8% / 8% | 92% (TF32 → FP32) | 0% of wavefronts | 81% | 16% | math_pipe_throttle 5.2, wait 0.9, barrier 0.1 |
| tensor_core_bf16 | 2,344.5 µs | 7,328 | 165 | 256 × 256 | 17% / 17% | 53% (BF16 → FP32) | 0% of wavefronts | 68% | 26% | math_pipe_throttle 4.2, wait 2.4, barrier 0.9 |
| cublas_bf16 (`Kernel2<cutlass_80_tensorop_s1688bf16gemm_128x256_16x3_nn...`) | 1,506.1 µs | 11,407 | 235 | 256 × 256 | 17% / 17% | 83% (BF16 → FP32) | 0% of wavefronts | 81% | 29% | math_pipe_throttle 7.1, barrier 1.2, wait 0.9 |
| tensor_core_f16 | 2,004.7 µs | 8,570 | 162 | 256 × 256 | 17% / 17% | 62% (FP16 → FP32) | 0% of wavefronts | 68% | 30% | math_pipe_throttle 4.3, wait 2.2, barrier 1.6 |
| cublas_f16 (`Kernel2<cutlass_80_tensorop_s1688f16gemm_256x128_16x3_nn_...`) | 1,592.9 µs | 10,785 | 230 | 256 × 128 | 17% / 17% | 78% (FP16 → FP32) | 0% of wavefronts | 77% | 24% | math_pipe_throttle 7.7, barrier 1.2, wait 0.9 |

## Matmul 512x512x512 (m × k × n)

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | FMA pipe | Shared-load bank conflicts | L1 hit | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| naive | 656.7 µs | 409 | 40 | 256 × 1024 | 93% / 100% | 19% | no shared loads | 89% | 96% | 2% | lg_throttle 25.9, not_selected 4.6, long_scoreboard 3.1 |
| tiled | 537.8 µs | 499 | 38 | 1024 × 256 | 67% / 67% | 9% | 0% of wavefronts | 1% | 95% | 2% | mio_throttle 20.7, not_selected 3.6, long_scoreboard 3.4 |
| register_tiled_128 | 271.6 µs | 988 | 128 | 256 × 16 | 17% / 33% | 29% | 0% of wavefronts | 6% | 78% | 3% | long_scoreboard 1.1, not_selected 0.7, wait 0.5 |
| register_tiled_64 | 247.7 µs | 1,084 | 64 | 256 × 64 | 35% / 67% | 20% | 0% of wavefronts | 3% | 88% | 4% | long_scoreboard 5.0, wait 1.5, mio_throttle 1.1 |
| vectorized | 167.8 µs | 1,600 | 115 | 256 × 16 | 17% / 33% | 45% | 0% of wavefronts | 0% | 78% | 6% | long_scoreboard 0.9, not_selected 0.7, barrier 0.3 |
| double_buffered | 139.0 µs | 1,931 | 125 | 256 × 16 | 17% / 33% | 55% | 0% of wavefronts | 1% | 78% | 7% | not_selected 0.7, dispatch_stall 0.4, short_scoreboard 0.3 |
| split_k | 109.0 µs | 2,462 | 125 | 256 × 48 | 28% / 33% | 55% | 0% of wavefronts | 4% | 82% | 25% | not_selected 1.9, dispatch_stall 0.6, short_scoreboard 0.3 |
| cublas (`ampere_sgemm_128x64_nn`) | 91.3 µs | 2,941 | 122 | 128 × 128 | 28% / 33% | 50% | 0% of wavefronts | 7% | 93% | 11% | not_selected 1.4, wait 1.2, dispatch_stall 0.7 |

Tensor cores on 512x512x512: the inputs rounded to each format, FP32 accumulation. Tensor throughput is the tensor ops executed against the peak of the path used, as Nsight Compute reports both.

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | Tensor throughput (path) | Shared-load bank conflicts | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| tensor_core_tf32 | 101.3 µs | 2,649 | 128 | 256 × 48 | 26% / 33% | 47% (TF32 → FP32) | 1% of wavefronts | 82% | 27% | math_pipe_throttle 6.0, wait 2.3, long_scoreboard 1.1 |
| cublas_tf32 (`Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4>`) | 77.3 µs | 3,474 | 136 | 128 × 96 | 8% / 8% | 51% (TF32 → FP32) | 0% of wavefronts | 91% | 13% | math_pipe_throttle 2.4, wait 0.6, long_scoreboard 0.4 |
| tensor_core_bf16 | 73.4 µs | 3,655 | 165 | 256 × 16 | 17% / 17% | 27% (BF16 → FP32) | 0% of wavefronts | 76% | 14% | math_pipe_throttle 4.0, wait 2.3, barrier 0.9 |
| cublas_bf16 (`Kernel2<cutlass_80_tensorop_s1688bf16gemm_128x128_16x6_nn...`) | 52.2 µs | 5,140 | 235 | 128 × 16 | 8% / 8% | 37% (BF16 → FP32) | 0% of wavefronts | 83% | 17% | math_pipe_throttle 3.2, wait 0.9, short_scoreboard 0.3 |
| tensor_core_f16 | 63.7 µs | 4,215 | 162 | 256 × 16 | 17% / 17% | 31% (FP16 → FP32) | 0% of wavefronts | 78% | 15% | math_pipe_throttle 4.0, wait 2.2, barrier 1.6 |
| cublas_f16 (`Kernel2<cutlass_80_tensorop_s1688f16gemm_128x128_16x5_nn_...`) | 52.6 µs | 5,099 | 235 | 128 × 16 | 8% / 8% | 37% (FP16 → FP32) | 0% of wavefronts | 83% | 13% | math_pipe_throttle 3.2, wait 0.9, short_scoreboard 0.3 |
