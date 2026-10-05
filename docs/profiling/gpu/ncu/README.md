# Kernel counters (Nsight Compute)

[`scripts/profile_kernels_ncu.sh`](../../../../scripts/profile_kernels_ncu.sh)
profiles one launch of every GPU kernel with Nsight Compute's full metric set,
each workload in its own run filtered to its NVTX range
(`bench_cuda --suite profile`): the GELU chain at 64 MB unfused and fused,
and the naive, tiled and both register-tiled matmul kernels plus cuBLAS on
2048³, 512³ and the 128×2048×512 layer.
[`summary.md`](summary.md) has the tables, [`metrics.csv`](metrics.csv) the
numbers per workload, and [`roofline.svg`](roofline.svg) the chart below.
Versions and commit: [`environment.txt`](environment.txt).

Nsight Compute held the SM clock at its 900 MHz base while profiling, so the
durations here are not the benchmark's, and every percentage is relative to
that clock (FP32 peak there: 2 × 30 SMs × 128 lanes × 0.9 GHz = 6,908
GFLOP/s). The memory clock stayed at its full rate.

![Roofline: FLOP per DRAM byte against GFLOP/s](roofline.svg)

## What the counters show

**Fusion cuts DRAM traffic from 21 passes over the input to 2.** The unfused
GELU chain reads 768 MiB and writes 573 MiB, 21.0 times the 64 MiB input; the
fused kernel reads 64 MiB and writes 64 MiB, 2.0 times. Both stream at 91–92%
of the DRAM's peak throughput, so the fused kernel's 10.4x lower time here
(468 µs against 4,890 µs) is the traffic it removes. On the roofline both sit
on the bandwidth roof.

**The naive matmul is not as naive as it looks.** 87–89% of its global loads
hit in L1: neighbouring threads of a block read the same rows of A and columns
of B, and the cache catches that reuse. What limits it is the queue of
outstanding global loads (its top stall, `lg_throttle`, 19–26 cycles per
issued instruction).

**The 32×32 shared-memory kernel is limited by shared memory.** Its FMA pipe is
busy 8–9% of the time, and its top stall is `mio_throttle` at 21 cycles per
issued instruction: the queue for shared-memory instructions is full, because
every multiply-add needs two shared-memory loads. That is why tiling alone gains
only about 1.2x over the naive kernel on the square shapes here.

**Register tiling is conflict-free and issue-bound.** No kernel has any
shared-memory bank conflicts (0% of wavefronts, loads and stores). The
128×128 kernel runs at 32% occupancy of a 33% maximum (16 of 48 warps per SM,
set by its 128 registers per thread), and its stalls are short: the top one is
`not_selected`, meaning warps were ready to issue. Its FMA pipe is busy 40% of
the time at 2048³ (2,519 GFLOP/s, 36% of peak). cuBLAS's
`ampere_sgemm_128x64_nn`, at the same occupancy with 122 registers, keeps the
FMA pipe 64% busy (4,179 GFLOP/s, 60% of peak). The difference is issue
slots: for its 64 FMAs per k step, each thread of the register-tiled kernel
also issues 16 32-bit shared-memory loads and the address arithmetic for them.
128-bit loads cut those 16 to 4.

**64×64 tiles trade reuse for occupancy.** Their theoretical occupancy is
twice the 128×128 kernel's (67%, 32 warps), which wins on outputs too small to
fill the GPU (1,096 against 992 GFLOP/s at 512³) and loses on large ones (FMA
pipe 31% against 40%, 1,780 against 2,519 GFLOP/s at 2048³).

**The 128×2048×512 layer starves the GPU.** Its 128×128 tiles make 4 blocks
for 30 SMs (17% achieved occupancy, 254 GFLOP/s), and its 64×64 tiles only 16
(338 GFLOP/s); even the naive kernel, with 256 blocks, is faster (380). cuBLAS
runs it with `ampere_sgemm_64x32_sliced1x4_nn`: 64×32 tiles and, by its name,
K sliced four ways within each block, launching 64 blocks of 256 threads for
2,244 GFLOP/s.
