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
proposal` - the initial import. It was not disabled because something broke. It
has simply never been enabled on the ring path.

## What this means for the B8 duplication

The B8 shared-prefix problem is not a missing capability. It is a capability
that exists, defaults on, is honoured by two call paths, and is masked off for
the one path that needs it - feeding a block table the device never reads.

Before adding sharing anywhere, answer these two, in this order:

1. Why is the mask there? A deliberate disable with no comment usually means
   somebody hit something. `git log -S` says it predates all development, so the
   answer may be "nobody ever tried".
2. Should the device consume the scheduler's table instead of building its own?
   If yes, `WorkControlKvState`'s directory, pool, residency, clock sweep,
   prefetch and NVMe paging - roughly 1300 lines - are all deletable, because
   `KvCacheArena` and `PrefixCache` already do every one of those things.

Both answers are subtractive. PR #509 is additive. That is the wrong direction
and #509 should not merge until question 2 is settled.

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
- `model-families/glm52/src/spark_glm52_request_api.c` is 7350 lines and has not
  been audited for the duplication described above.

## PR state at handoff

- **#507** rebased onto current `main`. Two kernel-dispatch conflicts resolved by
  keeping both guards. One commit skipped: `a3a10fe`, the FP8 per-K-tile
  activation scale, which conflicts with main's corrected `SparkLmFp8LoadFragA`
  fragment mapping. It needs re-deriving on top of main's structure somewhere
  with a compiler; a wrong mma fragment mapping assembles cleanly and renders
  silently wrong.
- **#509** block identity and prefix sharing in `work_control.c`. Compiles,
  tested, `make -j4 test` exits 0, B8 collapses 40 sequence slots onto 12
  physical blocks. Deletes `kv_dedup.c`, `jit_kv_pool.c` and the batch-plane
  simulator. Net +983/-1616. **Hold it** - see question 2 above.
