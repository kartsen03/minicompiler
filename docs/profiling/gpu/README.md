# GPU profiles (Nsight Systems)

`scripts/profile_gpu.sh` runs [`bench/profile_gpu.py`](../../../bench/profile_gpu.py)
under Nsight Systems: every variant of the GPU PyTorch comparison, warmed up
as in the benchmark, then 50 synchronized calls each, every call in its own
NVTX range. [`summary.md`](summary.md) splits each call into its wall time,
the GPU span from its first GPU operation to the end of its last, and the
time its kernels actually ran, and lists the kernels. The three CSV files are
Nsight's `nvtx_sum`, `nvtx_gpu_proj_sum` and `nvtx_kern_sum` reports they come
from. Machine and versions: [`environment.txt`](environment.txt). Tracing adds
a little host overhead, so the wall times here run slightly above the
benchmark's.

Nsight Compute, which reads per-kernel hardware counters (occupancy, DRAM
throughput, bank conflicts), fails here with `ERR_NVGPUCTRPERM`: GPU
performance counters are restricted to administrators on this machine. There
is no kernel-level counter analysis in this repository.

## What the profiles show

**On the GELU chain the kernels are equally fast; the difference is the host.**
At 32 MB, minicompiler's generated kernel, PyTorch's hand-written `F.gelu`
kernel and Inductor's Triton kernel each take 214–215 µs, the time to stream
64 MB through DRAM. Per call, minicompiler spends 250 µs, `F.gelu` 248 µs and
the `torch.compile` function 378 µs: compiled-function overhead on the host,
not a slower kernel, is why the benchmark measures minicompiler 1.68x ahead
of `torch.compile` there. With CUDA graphs (`mode="reduce-overhead"`), each
call first copies the input into the graph's static buffer, a 217 µs kernel
as long as the GELU itself.

**On the MLP block the matmuls decide.** At B=512 minicompiler's two matmul
kernels take 315 µs (128×128 tiles, first layer) and 496 µs (64×64 tiles,
second layer, K = 2048), where cuBLAS takes 142–150 µs per layer. minicompiler's
two fused elementwise kernels take 35 µs together, Inductor's two Triton
kernels 30 µs, and eager's 20 separate elementwise kernels about 450 µs. At
B=128 cuBLAS runs the second layer with `ampere_sgemm_64x32_sliced1x4`:
smaller tiles, with K split four ways inside each block, which is the
split-K idea minicompiler does not have yet.

**At 16 KB a call is mostly host time.** minicompiler's call takes 38 µs for
a 7.4 µs kernel (the GPU stays near idle clocks at this size); `F.gelu` 47 µs
for 7.5 µs, `torch.compile` 101 µs for 7.5 µs, and eager's 11 kernels run for
89 µs inside a 220 µs call.
