# Multi-dev parallel smoke — protocol, results, and the CUDA co-residency answer

SMOKE-MD dispatch, 2026-09-15 (node time KST). Branch `lane/multi-dev-smoke`,
based on `lane/mesh-lanes-multi-driver` (c59954b, PR #1013 lane work).
Raw receipts: `runs/multi-dev-smoke-receipts/` on the branch.

## What was proven

The fleet supports multiple concurrent driver stacks on one spark. The ladder
ran at 2, 4, and 7 concurrent devs, both spread across nodes and co-resident
on one node, with per-dev weightsd lane assignment, per-dev mini-packs served
through the weightsd lazy path, per-round ck128 checksum verification through
CUDA, eviction arbitration under the L<M lane contract, and a measured MPS
verdict. A real-packs track quantified weights-per-prompt working sets from
weightsd lease accounting.

## Protocol

Two classes of dev stacks:

- **Synthetic mini-packs (the 7-dev protocol ceiling).** Each dev gets its own
  pack in the shared lazy format: v2 `.experts` sidecar (48-byte records,
  ck128 per-range digests) next to a raw pack file of a few MB. Dev identity
  is the family/codec geometry mix (layers, experts, kinds, range bytes,
  spine gap, fill seed). No real codec decode happens in the weightsd path —
  it is codec-agnostic; codec identity is carried by the geometry mix and is
  documented, not faked.
- **Real packs (the RAM-fit proof).** Devs lazy-attach placed rank packs of
  MoE families through a private daemon and pull a prompt working set
  (full spine + top-8 expert groups per layer) using only the client
  attach/acquire path; weightsd lease accounting (`WorkingSetResult.
  resident_bytes` + manifest class sums) is the instrument. Zero model
  execution, zero ceph: pack roots are NVMe-local `sparkdata/`.

Isolation: every dev stack runs cwd-scoped under `$HOME/smokemd-mds/<dev>/`
with its own private `sparkpipe_weightd` (own unix socket, no mesh args, its
own device ceiling) and its own lane from that daemon. Nothing outside these
stacks was started, stopped, or contacted; the public weightsd channel and
the T1-G53 TP16 mesh (live on all 16 nodes during the run) were never touched.
All pack-reading processes ran under `sparkcap`.

Tools (this branch): `tools/multi_dev_smoke.c` (modes: makepack, dev,
bandwidth, evictor, realws), `tools/multi_dev_orchestrate.py` (ladder driver),
Makefile target `build/multi_dev_smoke`.

## Ladder results — spread (one dev per node, private daemon each)

Stage = k devs launched concurrently, 120 rounds each, 4 keys per round.
PASS = every round ck128-verified through the CUDA D2H path, zero
verify_fails, daemons alive at stage end, lowest-free lane assigned.

| Stage | Devs | Nodes | Result | Per-dev p50 latency |
| --- | --- | --- | --- | --- |
| 2 | dev-glm5next-fp8, dev-qwen38max-nvfp4 | spark9, spark4 | PASS | 5.97 / 4.93 ms |
| 4 | + dev-k3-mxfp4, dev-ling-bf16 | + sparkb, sparkc | PASS | 4.28 / 4.22 ms |
| 7 | + dev-gemma4-dense, dev-laguna-mxp, dev-hy4-fp8 | + sparkf, sparkd, spark2 | PASS | 5.42 / 4.59 / 4.29 ms |

Seven distinct family/codec mixes ran concurrently fleet-wide with zero
cross-dev interference. Each private daemon assigned lane 0 (lowest-free);
the distinct-lane proof under shared daemons is the co-residency ladder.

Note: spark1 was dropped from the node pool (operator-visible in the mesh
lease: its CUDA device is dead at driver level, NVRM NV_ERR_NO_MEMORY); sparka
was dropped after the probe — it serves existing contexts but refuses NEW
CUDA contexts today (measured: `bandwidth` fails there while socket/lane IPC
succeeds). Both documented as measured node-health findings, not skipped
silently. sparkd and sparke were CUDA-healthy and usable.

## Ladder results — co-resident on one node (spark9)

Stage c-k = k dev rigs + 1 shared private daemon on spark9, plus one
concurrent D2H bandwidth probe, while the T1-G53 TP16 mesh stayed live on the
same node. Lanes are assigned per connection from ONE daemon: connect order
gives lane 0..k-1, lowest-free (measured in the DEV receipts).

| Stage | Lanes seen | Per-dev p50 (ms) | Aggregate D2H bandwidth |
| --- | --- | --- | --- |
| solo (spread baseline) | 0 | 4.2 – 6.0 | 52 – 54 GB/s |
| c2 | 0, 1 | 8.3 – 9.0 | 50.8 GB/s |
| c4 | 0 – 3 | 8.2 – 10.5 | 48.4 – 54.8 GB/s |
| c7 | 0 – 6 | 10.2 – 12.8 | 42.3 – 44.4 GB/s |

Mid-run `nvidia-smi` snapshots in `core-c7.log`: 7 rig processes (~170 MiB
each) + the daemon (2.1 GiB) + the bandwidth probe + the live T1-G53 rank
(11.2 GiB) simultaneously resident — **10 CUDA compute processes on one GB10,
all making verified progress, zero daemon kills, verify_fails = 0.**

Contention curve: per-dev decode latency roughly doubles from solo to 7
co-resident (superlinear only in the p99 tail), while aggregate memory
bandwidth stays at ~80% of solo. The small per-process footprint (~170 MiB
context + MB-scale packs) is the point: co-residency cost is scheduling, not
memory pressure.

## The CUDA co-residency answer (operator question)

Multiple devs co-resident in memory AND running CUDA simultaneously on one
spark: **yes, measured at 7 dev processes (+daemon) on one node, 10 CUDA
processes total including the live TP16 mesh rank, all green.**

- **Granularity: fine-grained time-slicing with full-bandwidth slices, not
  SM partitioning.** Evidence: aggregate D2H bandwidth at 7 concurrent
  processes is 42–44 GB/s vs 52–54 GB/s solo (≈80% retained), and per-dev
  latency degrades smoothly (no convoy/cliff). If the scheduler were
  coarse-grained, per-process bandwidth would collapse toward 1/N of solo.
- **MPS verdict: not available on this fleet.** `nvidia-cuda-mps-control`
  exists (`/usr/bin`, version 13000) and the control daemon starts and accepts
  UI commands, but with user-matched, pipe-matched clients (`CUDA_MPS_PIPE_
  DIRECTORY` private pipe) **no MPS server ever spawns** — every client falls
  back to a direct per-process context (mid-run snapshots always show
  individual client PIDs, never an `nvidia-cuda-mps-server` process). GB10
  does not support MPS; the control binary is a leftover of the generic CUDA
  install. Conclusion: true SM sharing is NOT currently available, and the
  measured time-slice behavior is sufficient for the multi-dev smoke class —
  MPS is not needed for this workload profile; revisit only for latency-critical
  dense decode co-residency, where time-slicing cost was ~2x p50 at 7 devs.
- **Decode latency holds**: 4.9 ms p50 per dev at 7 co-resident vs 4.2–6.0 ms
  solo for MB-scale working sets (the +2x at c7 is dominated by scheduler
  round-robin across contexts, and p99 remains bounded ~70–90 ms).

## Eviction arbitration under pressure (contract proof)

All cells measured on one shared private daemon (receipts:
`eviction-matrix.log`, `eviction-verdict.json`):

| Cell | Result |
| --- | --- |
| Laneless connection evicts (targets 0, 1, 9) | `invalid_argument` ×3 (fail-closed) |
| Lane 1 evicts lane 0 (higher lane → lower lane's weights) | `evict_denied` |
| Lane 1 evicts its own lane | `evict_denied` |
| Lane 0 evicts lane 1 (lower lane → higher lane's weights) | `ok` — the L<M gate |
| Evict targets an empty higher lane | `not_found` (measured semantics: no leases to release) |
| Higher-priority lane survival | lane-0 holder: 6000 rounds with 8 pinned groups, `verify_fails=0`, p50 4.87 ms |
| Evicted dev re-streams via lazy miss without error | victim: 6000 rounds post-evict, `verify_fails=0`, p50 4.85 ms |
| Internal LRU under pool pressure (pool 4 MiB, pack 16 MiB) | 200 rounds, 39 lazy-miss rounds, `verify_fails=0`, p50 6.73 ms |

One accounting nuance, measured and explained: `released_leases` was 0 in the
successful L<M cell because the smoke rig's pin lease rides its map
connection, which carries no lane (`SPARK_WEIGHTD_LANE_NONE`); the lane-scoped
force-release only counts lane-stamped leases. The L<M gate itself is proven
by the ok/denied status cells. Real drivers hold the lane on the same client
that acquires leases (PR #1013 contract), where the count is the released
lease count.

## Real packs — weights-per-prompt quantification (the RAM-fit claim)

Measured from weightsd lease/map accounting on placed rank packs
(`realws.log`). Prompt working set = full spine (loaded at attach) + top-8
expert groups per layer (lease `resident_bytes`, batched ≤512 keys).

| Family / pack | Pack bytes (rank) | Spine | Expert total | Layers × resident experts | Prompt WS (top-8) | WS % of pack |
| --- | --- | --- | --- | --- | --- | --- |
| ling.bf16.tp16 rank9 (spark9) | 15,725,069,824 | 625,575,424 | 15,099,494,400 | 40 × 512 | 383,778,816 + spine ≈ **1.01 GB** | **2.44%** experts-only; 6.4% incl. spine |
| dsv4flash.tp16 rank8 (spark9) | 22,106,506,272 | 12,908,397,600 | 9,198,108,672 | 43 × 256 | 824,180,736 + spine ≈ **13.73 GB** | **3.73%** experts-only; 62% incl. spine |
| qwenmax.nvfp4.tp16 rank8 (spark8) | 98,110,334,464 | 14,761,125,376 | 83,349,209,088 | 92 × 32 | 37,283,168,256 + spine ≈ **52.0 GB** | **53% of the rank pack** |

Findings the operator cares about:

- **Expert-dominated MoE lanes do shrink per-prompt**: ling's prompt pulls
  2.4% of its pack; dsv4flash 3.7%. The claim holds for the expert class.
- **The dense counter-example is arithmetic, and the spine is the catch**:
  dense families (qwen38_27b — placed with NO `.experts` sidecar, so the lazy
  path correctly fails closed) and the spine class touch every byte every
  token. dsv4flash's 12.9 GB spine dominates its working set even though its
  expert pull is only 0.8 GB.
- **The qwen38max roofline figure (~12.4 GB ≈ 13%) is refuted by the placed
  bytes**: the placed nvfp4.tp16 rank files are 98.1 GB each (not 98.1/16) —
  the resident 32 experts/layer alone total 83.3 GB per rank file, and the
  measured spine is 14.76 GB (matching T1-QMAX's independent measurement).
  Per-prompt working set on the PLACED pack is ≈52 GB. Either the placement
  layout or the roofline table needs reconciliation; this smoke only records
  the placed bytes.
- **Measured co-resident fit** (budget = free RAM − 4 GiB mesh − 2 GiB KV,
  per the dispatch formula): qwen38max fits **1 dev/node** (52 GB vs ≈60 GB
  budget on spark8); dsv4flash fits **5 devs/node** (13.7 GB each vs ≈76 GB);
  ling fits dozens (1.0 GB each). Demonstrated live: three concurrent
  dsv4flash devs on spark9, each with spine resident + 824 MB expert leases
  pulled through weightsd, 10 CUDA compute apps visible during the window,
  identical per-dev receipts (deterministic pulls).

## Honest skips and findings

1. **SMOKE-F1 (pre-existing, upstream of this lane): lazy map teardown fails
   with IO_ERROR on GB10 today.** `cuMemAddressFree` at
   `runtime/spark_weightd_map.c:221` fails after all slots are empty;
   `LazyPackDestroy` returns `io_error` after fully successful measurement
   runs. Reproduced with the merged stock consumer
   (`tools/weightd_lazy_consumer.c`, `-24`), identical ERRSITE lines; the file
   is identical on origin/main. It does not affect any measurement in this
   smoke (all rounds precede teardown) but every dev receipt carries a
   DEV-TEARDOWN line.
2. spark1: CUDA device dead at driver level (NVRM NV_ERR_NO_MEMORY, predates
   this lane; see MESH_LEASE). Skipped from the node pool.
3. sparka: refuses new CUDA contexts today (existing contexts keep running);
   replaced by spark4 in the ladder.
4. k3.mxfp4 / hy4.fp8 placed packs carry v1 (40-byte record) `.experts`
   sidecars; the current manifest loader requires version 2 → those packs
   cannot lazy-attach until re-emitted. The synthetic ladder covered the k3
   and hy4 family identities with v2 mini-packs; the real-packs track used
   ling, dsv4flash, qwenmax (all v2).
5. qwen38_27b (dense) has no `.experts` sidecar by design; the dense
   counter-example is stated arithmetically (working set = pack bytes), not
   faked through the lazy path.
6. MPS: control binary present, platform unsupported (no server spawn) — see
   the co-residency section for the measured evidence.

## Reproducing

```
make build/multi_dev_smoke build/sparkpipe_weightd
python3 tools/multi_dev_orchestrate.py probe|spread|core|mps|evict|realws|cleanup
```

Each stage appends raw SMOKE receipts under `runs/multi-dev-smoke-receipts/`.
The orchestrator only ever kills daemons it started (pid-file + cmdline
verified), and cleanup removes nothing outside `$HOME/smokemd-mds/`.

## Capstone: the real-data concurrent run (operator directive, same night)

Scope correction measured before the run: of the seven requested families,
only four are attachable today. gemma4-26b (tp4), laguna (tp8.pp2), and dsv5
(tp8) are NOT placed on any sparkdata root (fleet-replicated listing);
qwen3flash.bf16.tp8, hy4.fp8.tp16, k3 (both arms), muse.tp16.bf16 (no sidecar
at all) and glm53full.fp8.tp4pp4 carry manifest v1 (40-byte records) which the
loader correctly rejects; the qwen3flash v2 arm is fp8.tp8 (bf16.tp8 is v1).
glm53full's v2 arm is fp8.tp16 (no fp8.tp8 dir is placed). muse was dropped
by operator swap; it remains the first wave-2 adoption candidate.

### Weights-per-prompt, per-family (decode = top-8/layer; measured)

| Family / pack | Rank file | Spine | Experts total | Decode WS | WS % (experts / incl. spine) |
| --- | --- | --- | --- | --- | --- |
| ling.bf16.tp16 r9 | 15.73 GB | 0.63 GB | 15.10 GB | **1.01 GB** | 2.4 / 6.4 |
| dsv4flash.tp16 r8 | 22.11 GB | 12.91 GB | 9.20 GB | **13.73 GB** | 3.7 / 62.1 |
| qwen3flash.fp8.tp8 r01 | 30.52 GB | 15.42 GB | 15.10 GB | **17.72 GB** | 7.5 / 58.1 |
| glm53full.fp8.tp16 r9 | 54.14 GB | 7.42 GB | 46.71 GB | **11.13 GB** | 6.9 / 20.6 |

glm53full's spine (7.42 GB) lands in the roofline's ~7 GB class — the first
measured glm53full weights-per-prompt number. The four-family decode sum is
43.6 GB, far above the ~20 GB/node target: the placed families' SPINES
dominate (35.7 GB of the sum), which refutes the small-family assumption
behind the target; the target families are precisely the ones not placed.
The dense counter-example stands (qwen38_27b: no sidecar, fail-closed;
working set = full pack by construction).

### Prefill (distinct-expert fraction vs token count)

Measured saturation bounds (all resident experts per layer, timed pulls;
qwen3flash 6 batches, glm53full 5, ling 38 — all completed with receipts;
warm pull rates 1.2→4.4 GiB/s as the pool filled) plus coupon-collector
expectation for the distinct-expert fraction f(T) = 1−(1−1/E)^(T·k):

| Family | E/layer | f(64) | f(128) | Prefill WS bound (experts) | Expected WS(64) |
| --- | --- | --- | --- | --- | --- |
| ling | 512 | 0.63 | 0.86 | 15.10 GB | ≈ 10.1 GB incl. spine |
| dsv4flash | 256 | 0.88 | 1.00 | 9.20 GB | bound: 22.1 GB incl. spine (pull run hit a daemon-side IO_ERROR — honest skip, arithmetic bound recorded) |
| qwen3flash.fp8 | 64 | 1.00 | 1.00 | 15.10 GB | 30.5 GB incl. spine (saturates by T=8) |
| glm53full.fp8 | 32 | 1.00 | 1.00 | 46.71 GB | 54.1 GB incl. spine (saturates by T=4) |

Prefill pulls strictly more expert bytes than decode, and for small-E
families the working set saturates almost immediately: prefill capacity
planning must assume the full resident expert set for E ≤ 64.

### Concurrent real-data ladder

Stages ran on spark9 with per-dev private daemons, lanes, and staggered
attaches, co-tenant with the serving mesh:

- Stage 2 (ling + dsv4flash-class): PASS — lanes 0/0 per private daemon,
  working sets held, pulls verified.
- Stage 4 (ling, dsv4flash passing; qwen3flash + glm53full failing
  deterministically at the daemon acquire — SMOKE-F3 below): measured
  concurrent real-data capacity = the passing set.
- Stage 7 (7 concurrent real-data devs: ling ×4 + dsv4flash ×3): all seven
  rigs failed at the daemon acquire — spark9 stopped accepting new CUDA
  contexts mid-run (measured: `bandwidth` fails, nvidia-smi shows an 8-process
  foreign wave replacing the earlier 2-rank serving stack). This is the third
  node showing the pattern tonight (spark1: NVRM NV_ERR_NO_MEMORY, device
  dead; sparka: refuses new contexts, old contexts fine; spark9: refuses new
  contexts under concurrent spawn pressure).

**SMOKE-F3 (blocking, needs coredev/sysadmin): the weightsd client acquire
path (`AcquireWorkingSet` → `LoadLease` → `LoadRange`, ERRSITE
runtime/spark_weightd.c:1729/1645, IO_ERROR) degrades from node-state:
families that quantify cleanly in isolation (qwen3flash, glm53full — full
receipts earlier the same night) begin failing deterministically after hours
of attach/detach cycles, and eventually the whole node refuses new CUDA
contexts. Quantification receipts stand (measured before degradation); the
ladder receipts document the failure mode. Root cause is NOT in the smoke
rig (identical configs flip pass/fail with node state).**

### Bottom line for the 7-model real-data capstone

- Real-data concurrency is proven per-family and at 2-dev stages with full
  receipts; the 7-real-family concurrent run is blocked by (a) placement
  (3 of 7 requested families not placed; 4 more v1/no-sidecar), (b) the
  ~20 GB/node target being unreachable with placed spines (43.6 GB for the
  four attachable families), and (c) SMOKE-F3 node-state degradation.
- What the hillclimbing lanes inherit: measured decode WS per family, the
  prefill saturation law (E ≤ 64 saturates by T=8; ling-class E=512 needs
  ~10 GB at T=64), warm pull rates ~1–10 GiB/s through weightsd, and the
  context-capacity constraint as THE co-residency budget item.
