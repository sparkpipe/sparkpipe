# Hardware independence — measured status (2026-08-27)

Where the stack stands, layer by layer, and what the lane agents are
bound to while the Phase 1B device API (the HAL) has not landed. Every
claim below is checkable in-tree; the numbers are from today's main.

## Layer 1 — kernel tree (inference/kernels, inference/llms): INDEPENDENT

The house kernel style IS the portability mechanism. Every kernel writes
its work loop as `for (i = threadIdx.x; i < N; i += THREADS)` and its
reductions as `stride = THREADS/2; ...`, which makes THREADS==1 a
single-threaded CPU execution by construction. The shim that proves it:
`tests/host_cuda/lm_host_cuda.cuh` (205 lines; `__global__`/`__device__`
become no-ops, shared memory becomes an array, syncthreads vanishes).
`tests/host_cuda/cuda_runtime.h` is deliberately 7 lines of redirect —
anything the kernel tree needs beyond the shim FAILS LOUDLY rather than
being silently emulated.

What this buys: per-family CPU oracle tests (k3_layer, kda, glm52_layer,
gqa, head, k3_slice, k3_engine, kv_failure, dsv4 module syntax) run in CI
with no GPU and no CUDA toolkit. Arithmetic, indexing, and argument order
are verified; the shim itself documents what it CANNOT catch (races,
warp assumptions, bank conflicts, occupancy, THREADS>1-only bugs).
TMA/async-pipeline intrinsics have stub declarations only — no host
meaning, not emulated.

## Layer 2 — module layer (modules/*): CUDA-BOUND (by design, for now)

15 source files across all 7 families include cuda_runtime directly.
Runner ABIs pass `cudaStream_t` (inside family-local headers — fine).
Published artifacts link cudart. This is the surface Phase 1B extracts
behind `spark_device.h` (memory/streams/events/launch/math-types), which
does NOT exist yet — no stub, no skeleton.

The naming hook already exists: `MODULE_TARGET` is a namespaced tuple,
`cuda.sm121.<family>.<stage>.<codecs>` (see every module Makefile), so a
future `host.*` / `metal.*` / `rocm.gfx*` target slots in without
renaming. Backends today: one (cuda), one arch (sm121/sm121a).

## Layer 3 — shared layer (runtime/, node/, cache/, ring/): CUDA-CLEAN

`node/model_residentd.c` and `runtime/stage_module_common.c` are the only
shared files touching CUDA, both at the dlopen'd module boundary. The
serving adapter ABI is device-opaque (`void *execution_stream` — no raw
CUDA types cross it). Zero family references below the module boundary.
The collective layer has a real two-backend switch (host TCP tier; device
tier with nccl / hidden_transport selectable per deployment) — library
independence adjacent to, but not the same as, hardware independence.

## What Phase 1B adds (not started)

`spark_device.h` + modules stop calling cuda_runtime; backend+arch
becomes a build-time target tuple; host backend promoted from test shim
to first-class CI oracle; metal backend on the controller Mac; rocm
skeleton. Sequenced AFTER Phase 1A consolidation deliberately: the DRY
adapter template lands first so the HAL wraps ONE lifecycle, not seven
copies of it.

## The line agents must hold (binding — see AGENT_LANE_BRIEFS/README.md)

Kernel tree stays host-compilable; modules stay CUDA-direct but must not
widen the extraction surface silently; ABIs stay device-opaque outside
family-local headers.
