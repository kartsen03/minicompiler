# Kernel counters (Nsight Compute)

[`scripts/profile_kernels_ncu.sh`](../../../../scripts/profile_kernels_ncu.sh)
profiles one launch of every GPU kernel with Nsight Compute's full metric set,
each workload in its own run filtered to its NVTX range
(`bench_cuda --suite profile`): the GELU chain at 64 MB unfused and fused,
and on 2048³, 512³ and the 128×2048×512 layer every FP32 matmul kernel, the
three tensor-core kernels, cuBLAS's FP32 SGEMM and cuBLAS's tensor-core GEMM
in each input format. [`summary.md`](summary.md) has the tables,
[`metrics.csv`](metrics.csv) the numbers per workload, and
[`roofline.svg`](roofline.svg) the chart below. Versions and commit:
[`environment.txt`](environment.txt).

Nsight Compute held the SM clock at its 900 MHz base while profiling, so the
durations here are not the benchmark's, and every percentage is relative to
that clock (FP32 peak there: 2 × 30 SMs × 128 lanes × 0.9 GHz = 6,906
GFLOP/s). The memory clock stayed at its full rate. A hot GPU can still slow
below the locked clock; the summary script flags any workload that did, and
none did in this profile.

![Roofline: FLOP per DRAM byte against GFLOP/s](roofline.svg)

## What the counters show

**Fusion cuts DRAM traffic from 21 passes over the input to 2.** The unfused
GELU chain reads 770 MiB and writes 573 MiB, 21.0 times the 64 MiB input; the
fused kernel reads 66 MiB and writes 64 MiB, 2.0 times. Both stream at
90–92% of the DRAM's peak throughput, so the fused kernel's 10.1x lower time
here (485 µs against 4,902 µs) is the traffic it removes. On the roofline both
sit on the bandwidth roof.

**The naive matmul is not as naive as it looks.** 87–89% of its global loads
hit in L1: neighbouring threads of a block read the same rows of A and columns
of B, and the cache catches that reuse. What limits it is the queue of
outstanding global loads (its top stall, `lg_throttle`, 19–26 cycles per
issued instruction).

**The 32×32 shared-memory kernel is limited by shared memory.** Its FMA pipe
is busy 8–9% of the time, and its top stall is `mio_throttle` at 21 cycles
per issued instruction: the queue for shared-memory instructions is full,
because every multiply-add needs two shared-memory loads.

**Register tiling is conflict-free and issue-bound.** No FP32 kernel has any
shared-memory bank conflicts. The 128×128 kernel runs at 32% occupancy of a
33% maximum (16 of 48 warps per SM, set by its 128 registers per thread), and
its stalls are short: the top one is `not_selected`, meaning warps were ready
to issue. Its FMA pipe is busy 40% of the time at 2048³ (2,519 GFLOP/s);
cuBLAS's `ampere_sgemm_128x64_nn`, at the same occupancy, 64% (4,177 GFLOP/s).
The difference is issue slots: for its 64 FMAs per k step, each thread also
issues 16 32-bit shared-memory loads and their address arithmetic. The 64×64
version has twice the occupancy (67%) but half the reuse: it wins only where
the 128×128 tiles cannot fill the GPU (1,084 against 988 GFLOP/s at 512³).

**128-bit loads close most of that gap.** The vectorized kernel loads A
(stored transposed) and B with `LDS.128`, 4 loads per 64 FMAs instead of 16,
with no bank conflicts: its FMA pipe is busy 59% at 2048³ (3,850 GFLOP/s,
1.53x the 128×128 kernel).

**Double buffering hides load latency where occupancy cannot.** At 2048³
every SM holds two blocks, and the double-buffered kernel gains little (61%
FMA, 3,894 GFLOP/s). At 512³ the output has 16 tiles, so 16 SMs hold one
block each; there the vectorized kernel's top stall is `long_scoreboard`
(waiting on global loads), which double buffering removes from the top three,
and the FMA pipe goes from 45% to 55% busy (1,600 to 1,931 GFLOP/s).

**Split-K fills the GPU on small outputs.** On the 128×2048×512 layer the
double-buffered kernel launches 4 blocks for 30 SMs (17% achieved occupancy,
507 GFLOP/s). Split into 7, it launches 28 blocks of K slices plus a short
kernel that adds the 7 partial products: 94 µs against 530 µs, 2,850
GFLOP/s, ahead of cuBLAS's `ampere_sgemm_64x32_sliced1x4_nn` (119 µs, 2,248
GFLOP/s), which by its name slices K four ways inside each of its 64 blocks
instead. At 512³, 3
splits make 48 blocks: 109 µs against 139 µs, where cuBLAS takes 91 µs.

**Tensor cores: the peaks Nsight Compute reports for this GPU.** 512 tensor
ops per SM per clock for FP16 and BF16 inputs with FP32 accumulation (1,024
with FP16 accumulation) and 256 for TF32, the same as the FP32 FMA units'
128 lanes × 2. So TF32 cannot beat FP32 here on peak, only on efficiency.

**Our tensor-core kernels keep their tensor pipe less busy than cuBLAS's.**
At 2048³ the FP16 kernel reaches 62% of the FP16 → FP32 peak (8,570
GFLOP/s), cuBLAS's `cutlass_80_tensorop_s1688f16gemm_256x128_16x3` 78%
(10,785 GFLOP/s); BF16 53% against 83%, TF32 68% against 92%. Occupancy is
not the difference: the 16-bit kernels on both sides run 8 warps per SM (our
162–165 registers, cuBLAS's 230–235), and cuBLAS's TF32 kernel only 4.
cuBLAS's warps mostly wait on the busy tensor pipe (`math_pipe_throttle` 7.7
cycles per issue against our 4.3), ours more on fixed-latency dependencies
(`wait`) and the per-tile barrier. Its kernels pipeline three to six stages
of tiles (the `16x3` to `16x6` in their names), and the FP16 one's 72 KB of
shared memory is exactly three stages of 256×16 and 16×128 FP32 tiles, so it
stages the FP32 inputs and converts them after the shared-memory load. Ours
convert while staging, through WMMA, with two stages. On the 128×2048×512 layer,
where ours splits K seven ways, its FP16 kernel is ahead (54 µs against
61 µs) and BF16 level.

**The first tensor-core epilogue was conflict-bound.** Every accumulator
tile went through a 16×16 shared-memory scratch tile, read back with an
8-way bank conflict on every load (917,504 conflicts at 2048³), and out with
scalar stores. Storing whole tiles straight to C removed them; only edge
tiles still use the scratch tile. TF32's B tile then showed 25% of its
shared-load wavefronts conflicted, from rows padded by 4 floats (4 banks
apart); padding by 8 floats removed that too (0–1% now).
