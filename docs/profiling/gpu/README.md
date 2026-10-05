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

For per-kernel hardware counters (occupancy, DRAM traffic, bank conflicts,
stall reasons, a roofline), see the Nsight Compute profiles in
[`ncu/`](ncu/README.md).

## What the profiles show

**On the GELU chain the kernels are equally fast; the difference is the host.**
At 32 MB, minicompiler's generated kernel, PyTorch's hand-written `F.gelu`
kernel and Inductor's Triton kernel each take 211–213 µs, the time to stream
64 MB through DRAM. Per call, minicompiler spends 279 µs, `F.gelu` 281 µs and
the `torch.compile` function 363 µs: compiled-function overhead on the host,
not a slower kernel, is why the benchmark measures minicompiler 1.53x ahead
of `torch.compile` there. With CUDA graphs (`mode="reduce-overhead"`), each
call first copies the input into the graph's static buffer, a 215 µs kernel
as long as the GELU itself.

**On the MLP block the matmuls decide, and the batch size decides who wins
them.** At B=512 minicompiler's two matmuls take 178 µs each on average (the
second split three ways, plus an 8 µs kernel that adds the partial products),
where cuBLAS takes 140–148 µs per layer. Its two fused elementwise kernels
take 34 µs together, Inductor's two Triton kernels 30 µs, and eager's 20
separate elementwise kernels about 450 µs. At B=128 minicompiler splits both
layers (3 and 7 ways): its matmuls take 48 µs each on average plus 11 µs of
partial sums, against cuBLAS's 65 µs (`ampere_sgemm_128x32_nn`) and 69 µs
(`ampere_sgemm_64x32_sliced1x4_nn`). With less host time per call as well,
minicompiler's call takes 161 µs, idiomatic PyTorch's 277 µs.

**At 16 KB a call is mostly host time.** minicompiler's call takes 45 µs for
a 7.3 µs kernel (the GPU stays near idle clocks at this size); `F.gelu` 48 µs
for 7.3 µs, `torch.compile` 102 µs for 7.5 µs, and eager's 11 kernels run for
87 µs inside a 227 µs call.
