# KV architecture: what already exists, before you build anything

The previous version of this document described `work_control.c`'s KV manager in
detail, concluded "the only gap is the directory key", and specified prefix
sharing to close it. That specification was implemented (PR #509, ~983 lines)
before anyone searched for the words "prefix cache". This repository already
contains a complete prefix-reuse implementation. Read this section before
writing a line.

## Three subsystems, two of them overlapping

| | `KvCacheArena` | `PrefixCache` | `WorkControlKvState` |
| --- | --- | --- | --- |
| file | `glm52/src/spark_glm52_kv_cache.c` | `glm52/src/spark_glm52_prefix_cache.c` | `glm52/src/spark_glm52_pp13_work_control.c` |
| lines | 2401 | 2713 | 2141 |
| allocate | `ArenaAcquireBlock` | via arena | `KvAcquirePhysicalBlock` |
| refcount | `ArenaRetainBlock` / `ReleaseBlockReference` | entry `reference_count` | `KvBlockResolve` / `KvBlockDeref` |
| residency | `ArenaMarkBlockResident` / `MarkBlockNonResident` | `ProbeReusablePrefixResidency` | `residency_state` |
| eviction | `ArenaTrimResidentBlocks`, `EvictResidentBlocksToLimit` | `TrimResidentBlocksByReuseScore` | clock sweep |
| prefetch | `ArenaBuildPrefetchPlan` | `BuildSequencePrefetchSources` | `CollectKvPrefetchEntries` |
| identity | - | `HashPromptTokens`, rolling per-block chain over token ids | 128-bit block key (added by #509) |
| block table | - | `BuildPhysicalBlockTable` -> lane-major | `BuildHostKvBlockTable` -> lane-major |

The last two rows are the point. `PrefixCache` and `WorkControlKvState` both hash
identity and both emit a lane-major physical block table with per-lane counts.
They are two implementations of one thing.

## The two block tables never meet

- `pp13_service_backend.c:2226` allocates `host_physical_block_indices` and hands
  it to the serving configuration at `:2535`. The scheduler fills it through
  `PrefixCacheBuildPhysicalBlockTable`.
- `spark_glm52_pp13_node_context_builder_cuda.cu:5738` allocates its **own**
  `host_physical_block_indices` and fills it through
  `WorkControlBuildHostKvBlockTable`. That is the table the device uses; see the
  dirty-range copies at `.cu:6570-6684`.

The prefix cache's table does not reach the device.

## Cross-sequence prefix reuse is implemented and switched off

```c
// pp13_service_backend.c:2352
scheduler_configuration.configuration_flags =
    SPARK_GLM52_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS &
    ~SPARK_GLM52_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE;
```

`..._FLAG_CROSS_SEQUENCE_PREFIX_REUSE` (`0x40`) is part of `DEFAULT_FLAGS`.
`scheduler.c:147` and `request_api.c:197` both honour it. The standalone serving
path gets cross-sequence prefix reuse by default. The PP13 backend is the only
caller that masks it off, with no comment.

`git log -S` dates that mask to `120c171 Import audited SparkPipe architecture
proposal` - the initial import. Confirmed by the owner: it was masked off because
the path had not been validated, not because anything broke. It has simply never
been switched on for the ring.

That makes it a candidate to validate on hardware, not a landmine to route
around. The host path it feeds is unvalidated, and so is the device path that
ignores it - nothing in this subsystem has run on silicon.

## What this means for the B8 duplication

The B8 shared-prefix problem is not a missing capability. It is a capability
that exists, defaults on, is honoured by two call paths, and is masked off for
the one path that needs it - feeding a block table the device never reads.

Before adding sharing anywhere, answer this, and prefer validating what exists
over building more:

1. Does enabling the flag on hardware deliver reuse, now that we know it was
   masked only for want of validation? That is one line and a ring run.
2. Should the device consume the scheduler's table instead of building its own?
   If yes, `WorkControlKvState`'s directory, pool, residency, clock sweep,
   prefetch and NVMe paging - roughly 1300 lines - are all deletable, because
   `KvCacheArena` and `PrefixCache` already do every one of those things.

Both answers are subtractive. #509 was additive and has merged, so the sharing
mechanism now exists twice: once in `PrefixCache`, once in `WorkControlKvState`.
Settle question 1 on the ring at the first opportunity. If the existing path
works once validated, the merged mechanism is redundant and roughly 1300 lines
of `WorkControlKvState` go with it. That deletion is the win; the merge did not
close this question, it enlarged it.

## Method, restated because it failed twice

The previous handoff's first lesson was "search the repo before implementing",
written after content-hash sharing was rewritten on top of an existing
`spark_glm52_kv_dedup.c`. The same failure then happened one layer up, at larger
scale, by an author who had read that lesson and quoted it.

Grepping for callers of the component you intend to modify is not sufficient.
`work_control.c` has a genuine production caller; that check passes and tells you
nothing. **Search for the capability by name before you build it** - "prefix",
"reuse", "dedup", "share", "hash" - across `model-families/` and `modules/`.

## Environment (verified this session)

- No GPU and **no nvcc** in the working container. Host C builds and runs;
  `make -j4 test` exits 0. Anything `.cu` or `.cuh` is diff-review only, with no
  compile gate at all.
- `tools/length_gate.py` enforces 50 lines per function and exits nonzero.
- Worst offenders in `work_control.c`, both pre-existing: `ValidatePacket` 325
  lines, `BuildPrefillPacket` 124.
- `model-families/glm52/src/spark_glm52_request_api.c` was 7350 lines and is now
  7178 after #511. Remaining known duplication in it: `CompleteDispatch` (195)
  and `CancelDispatch` (157) share 121 lines, but the shared part is scaffolding
  and the variation is at every leaf, so a merge would need several flags and
  callbacks and would add more complexity than it removes. Deliberately left.
- Not yet audited: `scheduler.c` (2463) and `serving_engine.c` (2500), which sit
  on the same prefix cache and arena as `request_api.c`. The pair-similarity
  scan has only been run within single files, never across them.

## Outstanding work

- **`a3a10fe` never landed.** It was skipped when #507 was rebased: the FP8
  per-K-tile activation scale conflicts with main's corrected
  `SparkLmFp8LoadFragA` fragment mapping, and a wrong mma fragment mapping
  assembles cleanly and renders silently wrong. It needs re-deriving on top of
  main's structure somewhere with a compiler. Nothing in CI will catch a mistake
  here, so do not merge it on review alone.
- **The ring run described above.** One flag, one deployment, and the answer
  decides whether ~1300 lines come out.
- **#507, #509, #510, #511 are merged.** #512 was closed as an older draft of
  this document.

## Where the code actually is, measured

The first-party tree is ~197k lines. `third_party/` is not included in that and is
the larger number: **2048 of the repo's 5045 tracked files, 41%**, vendored
flashinfer with CUTLASS inside it.

Our own code consumes exactly four headers from it:

```
flashinfer/gemm/gemm_groupwise_sm120.cuh
flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh
<cutlass/bfloat16.h>
<cutlass/float8.h>
```

Their transitive closure is 474 files, so **1574 vendored files - 77% - are
outside the dependency closure**, and the whole tree is consumed by one Makefile
target, `glm52_fp8_scaled_gemm_cuda_gate`, as include paths. Submodule or fetched
dependency removes 2048 files without touching a line of logic. This is the
largest single codesize lever in the repository and it is a packaging decision,
not a refactor.

## glm52's 27k file is not slop, and the shared library is not a toy

Both were suspected; both were measured.

- Shared library, 3113 lines: `mma.sync` x9, `wmma` x32, `__shared__` x33,
  `__shfl` x11, `__ldg` x17. Real tensor-core code with an FP8 tile that does
  proper fragment mapping and producer/consumer double buffering.
- glm52's 27308-line stage: 1% of lines behind disabled feature flags, 10 of 300
  static definitions unreferenced, ~7% internal duplication. Dense and live.
- Structural overlap between the two: **zero**. glm52 is not a copy of the shared
  library; its kernels solve the same problems differently, because MLA absorbed
  attention is not standard attention and FP8 block-scaled MoE is not the shared
  MoE.

The 8.8x size gap is scope: DSA sparse attention (4681 lines, 70 functions), MTP
tree speculation (798), three quantization backends, cublasLt integration.

## The real duplication is algorithmic, across families

| algorithm | glm52 | dsv4 | qwen36 | k3 | mimo25 |
| --- | --- | --- | --- | --- | --- |
| top-k | yes | yes | - | yes | yes |
| sparse attention | yes | yes | - | - | - |
| MLA | yes | - | - | yes | - |
| MTP / draft / speculative | yes | - | - | - | - |
| dspark | yes | - | - | - | - |

**Correction to an earlier claim in this document: top-k is NOT implemented four
times.** That figure came from counting keyword occurrences per family, which is
the wrong instrument and was reported as a finding. Checked properly:
`SparkLmOrderedTopKKey` and `SparkLmBitonicSortKeysAscending` are already in the
shared library, and dsv4, k3 and mimo25 call them - `SparkDsv4OrderedTopKKey` is
a one-line wrapper, not a reimplementation. Genuinely independent selection
kernels: **two**, glm52's and `SparkDsv4TopKKernel`, whose body uses no shared
primitive.

So the shared primitives exist and are partially adopted. The remaining
duplication is the selection kernels on top of them, which is a smaller and
different job than promoting a primitive that does not exist yet.

Sparse attention is implemented twice, MLA twice - those were counted the same
crude way and should be re-checked against actual function bodies before anyone
plans work around them.

MTP, DSA and dspark are single-implementation today but universally applicable -
any model can do speculative decoding or sparse attention. They are in glm52's
tree because glm52 was written first, not because they are glm52-specific.
`model-families/glm52/src/spark_glm52_dspark.c` is 877 lines of host C that
references exactly **one** glm52 model constant. It is already almost entirely
generic and should live in `model-families/common/src/`.

Order of work, revised after the correction above: `dspark` first, because it is
877 lines of host C with one glm52 constant in it, it compiles and tests without
a GPU, and its value is preventing a duplicate rather than removing one. Then
re-measure sparse attention and MLA by comparing function bodies rather than
keyword counts, and let that decide what follows. Do not plan around the counts
in the table above without checking them.

The general lesson, which cost four wrong conclusions in one session: a keyword
count tells you where to look, never what is there. Every conclusion drawn from
one and not checked against the code was reversed by checking.

## Not attempted, and why

**TMA and async copy.** Neither the shared library nor glm52 uses `cp.async`,
`cp.async.bulk` or thread-block clusters - zero occurrences in either - while the
handoff records that TMA and `__cluster_dims__` ARE available on `sm_121a`. The
async-copy pipeline is unexploited across the entire codebase and is the clearest
performance lever that is genuinely missing rather than merely large.

It was not implemented here because this container has no CUDA compiler. The
distribution toolkit 404s on a driver dependency, and `nvidia-cuda-nvcc-cu12`
ships `ptxas` only, which assembles PTX and cannot compile CUDA C++. There is no
clang. Writing TMA descriptors that cannot be syntax-checked, into kernels whose
own comments note that a wrong fragment mapping assembles cleanly and renders
silently wrong, would be the least defensible change available.

But `ptxas` alone answers the question underneath it, and
`tests/test_ptx_capability_gate.py` now does. The capability claims in this
document were prose; they are now assembled against `sm_121a` on every
`make test`:

| form | sm_121a |
| --- | --- |
| `cp.async.ca`, `cp.async.cg` | available |
| `cp.async.bulk` | available |
| `cp.async.bulk.tensor` 1D / 2D / 3D | available |
| `mbarrier.expect_tx` | available |
| `barrier.cluster.*`, `mapa.shared::cluster` | available |
| `mma.sync.m16n8k16.bf16` | available |
| `tcgen05.*` | NOT available |
| `wgmma.*` | NOT available |

The gate fails if a required form stops assembling, and equally if `tcgen05` or
`wgmma` starts, because that means the target changed and the kernel strategy
should be revisited. Where there is no `ptxas` it skips rather than failing.

So the async-copy work is now specified against a verified target rather than an
assumed one. `cp.async.ca` is the cheapest first step - it is available, the
codebase uses it zero times, and it needs no tensor-map descriptor. Full TMA needs
host-side `CUtensorMap` setup and is the second step. Both still need a machine
with a compiler to land; what has changed is that the target's capabilities are no
longer a matter of belief.
