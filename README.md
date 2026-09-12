# gpu-memory-profiler

A [Claude Code](https://claude.com/claude-code) subagent that reviews CUDA kernels for GPU memory-hierarchy issues — uncoalesced global memory access, register spilling, shared-memory bank conflicts, low occupancy, and missed opportunities to cache reused data in shared memory or registers.

## Contents

- **[.claude/agents/gpu-memory-profiler.md](.claude/agents/gpu-memory-profiler.md)** — the subagent definition.
- **[kernel.cu](kernel.cu)** — a sample CUDA file with five kernels, each seeded with one specific memory-hierarchy issue, for exercising the subagent:
  - `matmul_naive` — no shared-memory tiling, redundant global reads
  - `transpose_naive` — uncoalesced (strided) global write
  - `reduce_columns` — unpadded shared memory causing bank conflicts
  - `apply_lut` — read-only lookup table missing `__ldg` / `__restrict__`
  - `poly_eval_heavy` — large per-thread local array, likely register spill

## Usage

1. Copy `.claude/agents/gpu-memory-profiler.md` into the `.claude/agents/` directory of your project.
2. Restart Claude Code so it picks up the new subagent definition.
3. Ask Claude to use it, e.g.:
   ```
   Use the gpu-memory-profiler subagent on kernel.cu
   ```

The subagent will analyze the kernel (and any profiler output from `ncu`/`nvprof`/Nsight you provide) and report, for each issue found: where it is, why it matters for the specific part of the memory hierarchy involved, and a concrete code-level fix — clearly separating findings backed by profiler numbers from those that are speculative based on reading the code alone.
