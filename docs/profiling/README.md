# CPU profiling with perf and flame graphs

`scripts/profile_cpu.sh` builds a Release tree with frame pointers and debug
info, samples `mcc` with `perf record -e cpu-clock -F 1999 -g`, and folds the
stacks with Brendan Gregg's [FlameGraph](https://github.com/brendangregg/FlameGraph)
scripts. Each workload gets an interactive SVG flame graph (open it in a
browser) and a text summary of self time by shared object and symbol.

**Setup** ([environment.txt](environment.txt); the "before" profile ran at `cbcafa7`, see [environment_before_output_fix.txt](environment_before_output_fix.txt)): perf 6.8.12 on WSL2 (kernel
6.6.114.1-microsoft-standard-WSL2), Intel Core i7-12700H, one thread pinned
with `taskset -c 2`. WSL2 exposes no hardware PMU (`perf stat` reports
`cycles` and `instructions` as "not supported"), so sampling uses the
software `cpu-clock` timer event. That gives time-based hotspots, not cycle,
cache-miss or instruction counts.

| Workload | Flame graph | Self-time summary |
|---|---|---|
| GELU chain, 2048x4096, no passes | [gelu_unfused.svg](gelu_unfused.svg) | [gelu_unfused.txt](gelu_unfused.txt) |
| GELU chain, 2048x4096, all passes (one fused kernel) | [gelu_fused.svg](gelu_fused.svg) | [gelu_fused.txt](gelu_fused.txt) |
| Same, before the output-copy fix | [gelu_fused_before_output_fix.svg](gelu_fused_before_output_fix.svg) | [gelu_fused_before_output_fix.txt](gelu_fused_before_output_fix.txt) |
| MLP block, B=512, all passes | [mlp_block_fused.svg](mlp_block_fused.svg) | [mlp_block_fused.txt](mlp_block_fused.txt) |

## What the profiles showed

**1. The unfused GELU chain is memory-bound.** Without fusion, 87% of samples
land in the eight cheap binary kernels (48% vector-vector, 39%
vector-scalar), and only 12% in `tanh`, the one op with real arithmetic. Each
of those ops streams a 32 MB tensor through DRAM to do one multiply or add per
element. That is the traffic fusion removes: the fused kernel reads `x` once
and writes `y` once.

**2. A copy of every output cost 16.8% of the fused kernel's time, and was
removed.** In the first fused profile ([before](gelu_fused_before_output_fix.txt)),
libc accounted for 16.8% of samples, spread over addresses inside `memmove`.
The frame-pointer call graph credited them to `main`, because glibc's
`memmove` keeps no frame pointer, so its real caller is skipped. Re-recording
with `--call-graph dwarf` showed the caller was `std::copy` in
`CpuExecutable::run`: each run computed the output into a pooled buffer, then
copied all 32 MB into the caller's tensor. Commit `bd2306a` makes the
executable write graph outputs straight into the caller's tensors. On the
same benchmark (`bench_cpu --quick`, 2048x4096) the fused GELU went from
12.97 ms to 10.64 ms, and libc disappears from the [new profile](gelu_fused.txt).

**3. In the fused kernel, the remaining time is the compulsory traffic and
`tanh`.** With fusion, `tanh` rises from 12% to 28% of samples. The
vector-vector kernels still take 49%, because the first instruction that
reads `x` and the last one that writes `y` absorb the unavoidable DRAM
read of the input and write of the output; the intermediates in between stay
in the 2 KB block buffers. `cpu::fused` itself, which dispatches the
instructions block by block, is 8%.

**4. The MLP block is GEMM-bound, so fusion has little to work with.** About
90% of samples are in Eigen's matrix multiply: 84.7% in the `gebp_kernel`
micro-kernel and 5.2% in operand packing (`gemm_pack_lhs`, `gemm_pack_rhs`).
The fused elementwise kernels are about 7%. This is why fusion speeds the MLP
block up by only about 1.1x (see the README results), while the GELU chain
gains about 3x. Making this graph faster would take a faster or multithreaded
GEMM, not more fusion.
