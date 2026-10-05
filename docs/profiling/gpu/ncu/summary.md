# Nsight Compute summary

One launch of each kernel, full metric set, the GPU held at 900 MHz by Nsight Compute (FP32 peak there: 6,908 GFLOP/s). See `README.md` here for what the numbers show.

## GELU chain at 64 MB

| Variant | Kernels | Time | DRAM read | DRAM written | Traffic ÷ input size | DRAM throughput |
|---|---:|---:|---:|---:|---:|---:|
| unfused | 9 | 4,890 µs | 768 MiB | 573 MiB | 21.0 | 92% |
| fused | 1 | 468 µs | 64 MiB | 64 MiB | 2.0 | 91% |

## Matmul 128x2048x512 (m × k × n)

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | FMA pipe | Shared-load bank conflicts | L1 hit | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| naive | 706.3 µs | 380 | 40 | 256 × 256 | 76% / 100% | 18% | no shared loads | 88% | 67% | 10% | lg_throttle 19.0, long_scoreboard 6.3, not_selected 3.3 |
| tiled | 718.3 µs | 374 | 38 | 1024 × 64 | 67% / 67% | 8% | 0% of wavefronts | 0% | 69% | 5% | mio_throttle 21.0, long_scoreboard 4.5, not_selected 3.6 |
| register_tiled_128 | 1,057.5 µs | 254 | 128 | 256 × 4 | 17% / 33% | 29% | 0% of wavefronts | 2% | 40% | 2% | long_scoreboard 1.1, not_selected 0.7, wait 0.5 |
| register_tiled_64 | 795.3 µs | 338 | 64 | 256 × 16 | 17% / 67% | 11% | 0% of wavefronts | 1% | 69% | 2% | long_scoreboard 5.5, wait 1.4, short_scoreboard 0.9 |
| cublas (`ampere_sgemm_64x32_sliced1x4_nn`) | 119.6 µs | 2,244 | 82 | 256 × 64 | 30% / 33% | 46% | 0% of wavefronts | 0% | 78% | 18% | not_selected 1.2, wait 0.8, long_scoreboard 0.7 |

## Matmul 2048x2048x2048 (m × k × n)

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | FMA pipe | Shared-load bank conflicts | L1 hit | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| naive | 40,324.1 µs | 426 | 40 | 256 × 16384 | 100% / 100% | 19% | no shared loads | 87% | 50% | 17% | lg_throttle 25.5, long_scoreboard 6.6, not_selected 4.5 |
| tiled | 32,952.4 µs | 521 | 38 | 1024 × 4096 | 67% / 67% | 8% | 0% of wavefronts | 0% | 48% | 11% | mio_throttle 21.0, long_scoreboard 4.5, not_selected 3.6 |
| register_tiled_128 | 6,820.0 µs | 2,519 | 128 | 256 × 256 | 32% / 33% | 40% | 0% of wavefronts | 2% | 71% | 8% | not_selected 1.9, dispatch_stall 0.9, long_scoreboard 0.6 |
| register_tiled_64 | 9,651.7 µs | 1,780 | 64 | 256 × 1024 | 64% / 67% | 31% | 0% of wavefronts | 3% | 53% | 17% | long_scoreboard 4.6, not_selected 2.0, mio_throttle 1.8 |
| cublas (`ampere_sgemm_128x64_nn`) | 4,111.2 µs | 4,179 | 122 | 128 × 512 | 31% / 33% | 64% | 0% of wavefronts | 0% | 90% | 16% | not_selected 1.8, wait 1.0, dispatch_stall 0.9 |

## Matmul 512x512x512 (m × k × n)

| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | FMA pipe | Shared-load bank conflicts | L1 hit | L2 hit | DRAM | Top stall reasons (cycles per issue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| naive | 656.9 µs | 409 | 40 | 256 × 1024 | 93% / 100% | 19% | no shared loads | 89% | 97% | 2% | lg_throttle 25.9, not_selected 4.6, long_scoreboard 3.2 |
| tiled | 536.6 µs | 500 | 38 | 1024 × 256 | 67% / 67% | 9% | 0% of wavefronts | 1% | 94% | 2% | mio_throttle 20.7, not_selected 3.6, long_scoreboard 3.5 |
| register_tiled_128 | 270.6 µs | 992 | 128 | 256 × 16 | 17% / 33% | 29% | 0% of wavefronts | 6% | 78% | 3% | long_scoreboard 1.1, not_selected 0.7, wait 0.5 |
| register_tiled_64 | 244.9 µs | 1,096 | 64 | 256 × 64 | 35% / 67% | 20% | 0% of wavefronts | 3% | 88% | 4% | long_scoreboard 4.9, wait 1.5, mio_throttle 1.1 |
| cublas (`ampere_sgemm_128x64_nn`) | 91.3 µs | 2,940 | 122 | 128 × 128 | 28% / 33% | 51% | 0% of wavefronts | 7% | 93% | 11% | not_selected 1.4, wait 1.2, dispatch_stall 0.7 |
