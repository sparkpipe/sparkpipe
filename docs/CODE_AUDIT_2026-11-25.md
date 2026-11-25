# Code audit: the 24-25 November merges

Date: 2026-11-25. Scope: 114 commits on main from 2026-11-24 00:11Z to 2026-11-25 06:54Z (#2129 to #2231, plus
twelve older PRs, ten of them drafts), about 86K lines added and 11K removed in about 500 files. The risky parts were
read closely: kernels, runtime, network, fleet settings, the build gate and the site. The rest was skimmed. Nothing was
run on a GPU for this audit; it reads code, diffs, PR text and the fleet's own logs.

## Summary

The merges brought real speed, but main is not safe to keep building on as it stands. Main has been red for about 2.5
days, and changes reached the fleet without the checks that catch silent breakage.

What holds up: the no-comments, model-neutral and device-neutral rules pass, and 24 test files were added. The RDMA
doorbell ordering is correct, and the new wire compression is off by default. The gains are real where speculation and
the rows16 lane apply: K3 at 38.5 tok/s with speculation, Qwen 27B at 171, DeepSeek V4.1 Flash prefill at 11K to 14K
tok/s against about 2.2K before.

Already found from the fleet's logs: the K3.1 Pro multi-row slowdown came from #2198 and #2201 (restored by #2209); the
sixteen ranks stopped computing in lock-step during node out-of-memory bursts; the nvrm build broke on defaulted and
overloaded kernels (fixed by #2208); and each model's temperature-0 replies changed two or three times across the
day's builds.

## Fix first

Get main green and gate every rollout on exactness before more speed work lands. Ranked by risk:

1. **Green main, kept green.** Require all four checks to pass, merge through a queue, and merge nothing on red except
   the fix. Make the compile gate print the real compiler error.
2. **Exactness before rollout.** Every kernel PR shows a digest A/B or the publisher-reference gate, and refactors are
   bit-identical. Every fleet release runs the kit's old/new capture, the golden compare and Gate GX.
3. **A memory plan for the lanes.** Refuse any `deployment/lanes.json` setting that does not fit a node's memory. Cap
   the stage runner's graph cache and fall back to running without graphs instead of aborting.
4. **Hold the site deploy** until the owner has reviewed the new public copy.
5. **Move sklc off main** until the source of its instruction encodings is documented.
6. **Re-qualify the engine drafter.** Run the soak's gates for the three switched lanes and the five new drafters
   before keeping the switch.
7. **Verify the nvrm launch-chain change (#2167)** with the stale-read test, the strict gate and a CUDA-vs-nvrm
   identity run.
8. **Guard the RDMA send queue** against overwriting requests still in flight.
9. **Record the source, revision and licence** of every third-party drafter and checkpoint.
10. **Keep wire compression off** until each lane passes a quality measurement and Gate GX with it on.

## Kernels and runtime

The kernel work changed outputs without proof that each change was intended, and one runtime change reopens a bug
class already measured once.

**Exactness evidence (#2144, #2146, #2147, #2148, #2204).** The shared-kernel refactors removed about 5K lines across
every model family. None of their PRs shows a bit-identity or reference result. Two kernel bugs ran on the fleet for
hours (#2182 and #2184, fixed by #2185 and #2186). #2201 dropped the `k_tiles < 32` guard behind the K3.1 Pro slowdown
(restored in #2209, measured on one GPU only).

- Suggest: a refactor must be byte-identical on the replay tools for every family it touches. A numerics change names
  its reference gate and result.
- Suggest: confirm #2209 on the fleet at the batch sizes that regressed (2048-row prefill, batch decode).

**nvrm launch chains (#2167).** Plain launches now chain in hardware (QMD-to-QMD) or take a compute wait-for-idle
instead of the semaphore acquire. The end-of-grid memory barrier drops from system scope (`L1_SYSMEMBAR`) to GPU scope
(`L1_MEMBAR`). On 2026-11-22, plain kernels chained by latches were measured reading stale cached data (282 of 153,600
reads; see #2110). The PR claims zero stale reads, but CI's nvrm phases have not run since #2181 and no GPU run is
linked.

- Suggest: rerun `nvrm-latch-segments` in stale and trigger modes, the strict gate, and the 27B and TP4 CUDA-vs-nvrm
  identity runs on current main.
- Suggest: add a test where one kernel writes host-visible memory and a later chained kernel publishes it to the
  network card or CPU. With a GPU-scope barrier, nothing guarantees the first kernel's writes reached system scope.
- Suggest: extend the dependent-wait audit to reads of the parent's outputs before `griddepcontrol.wait`. It checks
  writes only, and #2167 now lets dependent kernels overlap their parent.

**Graph cache (`inference/runner/stage_runner.cu`).** The runner keeps one captured graph per (key, rows) with no
eviction, up to 4 x (2 + verify lanes) x rows slots. Each is about 80 MiB on K3.1 Pro, and a capture failure aborts.
That memory sits outside the lane's state budget and lines up with the node out-of-memory bursts.

- Suggest: an LRU under a byte budget counted in the lane's memory plan; run without the graph on capture failure,
  with a log line; report graph bytes in `STAGE-MEMORY`.

**Wire compression (#2231).** It is opt-in through `SPARK_MESH_WIRE_CODEC`, the default stays exact BF16, and no
deployment sets it. Turning it on changes outputs.

- Suggest: keep it out of lanes.json until a per-lane quality measurement and Gate GX pass with it on.

## Network

The RDMA send path is ordered correctly, but nothing stops it from overwriting requests the network card has not yet
sent.

**What is right (#2225, #2164, `node/weightd_mesh.c`).** Work requests are written, then a store barrier, the doorbell
record, another barrier, then an 8-byte BlueFlame ring with alternating combining buffers. Inline requests stay inside
one 64-byte slot (inline limit 28 bytes, at most four 16-byte units).

**The gap.** `SparkWeightdMeshSqWrite` writes slot `produced & mask` without checking that the slot was retired. Data
writes are unsignaled and only each publish's tail write asks for a completion. A large burst or a slow completion
drain can therefore wrap the 512-slot ring and overwrite requests still in flight: silent corruption between nodes, or
a hang.

- Suggest: an explicit occupancy guard that waits, or refuses, when `produced - retired` reaches the queue depth, with
  a counter in the `WD-MESH` stats.
- Suggest: a mock test that stalls completions and drives the queue past its depth.

The GPU-side collective changes (#2171, #2199, #2224) were not reviewed line by line.

## Fleet configuration

Every lane's concurrency rose four- to eightfold with no memory plan, and the engine drafter reached production
without the soak that was meant to qualify it.

**Lane settings (`deployment/lanes.json`; #2170, #2176, #2178, #2197).** All 21 lanes changed. Weightd's device limit
(110 GiB) and KV reserve (8 GiB) did not move, and state budgets stayed the same except DeepSeek V4.1 Flash. Nodes
logged out-of-memory kills during these two days.

| Lane | Sequences | KV logical pages | Other changes |
| --- | --- | --- | --- |
| deepseek-v4.1-flash | 8 -> 64 | 32,768 -> 262,144 | rows 2,048 -> 16,384; state 5 -> 32 GiB; rows16 model |
| glm-5.3 | 8 -> 64 | 131,072 -> 1,048,576 | none |
| qwen3.8-27b | 8 -> 64 | 32,768 -> 262,144 | none |
| qwen3.8-max | 16 -> 64 | 8,192 -> 32,768 | none |
| kimi-k3.1-pro | 8 -> 32 | 131,072 -> 524,288 | none |

- Suggest: a per-node memory plan as a test over lanes.json: weights, physical KV, state times sequences, graph cache,
  scratch and headroom. It refuses any lane that does not fit.
- Suggest: measure each lane at its configured concurrency before a release, not only at one request.

**Engine drafter switch (#2160).** kimi-k3.1-pro, qwen3.8-27b and deepseek-v4.1-flash moved from `block_draftd` to the
engine draft server. The soak that was to qualify this stopped after the K3.1 Pro Python arm, so its gates never ran:
accepted drafts per round within 2%, unavailable drafts no higher, no restarts. Later PRs changed the drafting path
again (#2180, #2190, #2216, #2221: batch-8 drafting and five new drafters).

- Suggest: rerun the soak's compare per lane, including the five new drafters, before keeping the engine in
  production.

## Tooling and provenance

**sklc (`tools/sklc`, `inference/kernels/skl`; #2159, #2207).** It is a second kernel language with 13 kernels that
duplicate existing CUDA kernels, and a compiler that writes sm_121 machine code from hand-coded instruction encodings
with its own cubin writer. Nothing in a production build uses it; only a benchmark and `tests/test_sklc.py` do.

- The encodings have no documented source, so nobody else can review or extend them.
- Thirteen kernels in a language only this tool reads double the maintenance and review load, while every shipped
  kernel is already built with the CUDA toolchain.
- Suggest: move sklc off main until the source of its encodings is documented, and keep `.skl` kernels out of any
  shipped path.

**Third-party drafter weights (#2215 to #2220, `tools/drafter_pack.py`).** The new drafters are packed from other
parties' releases (DSpark drafters for GLM-5.3, Qwen3.8-Max, K-EXAONE 2.0 and A.X-K2; DFlash2 for GLM-5.3-Flash). The
code is ours; the weights are not, and nothing records their source revision or licence.

- Suggest: a provenance manifest per pack (source, revision, licence) checked by a test, confirmed before a pack is
  served; the same for third-party model checkpoints.

## Public site copy

Main's site copy changes the public claims, and the next deploy from main would publish it. The RAIS page, the
home-page banner and `site/rais-data.json` (#2174, #2189, #2206, and the Wave-4B commit) add backend comparisons on one
GB10, "Certified" scores from RAIS, which is our own benchmark, Apple M5 Ultra results, and the paths of internal
receipt files.

- Suggest: no site deploy until the owner has reviewed this copy.
- Suggest: present RAIS as our benchmark with our numbers, without "certified", and keep internal hardware and file
  paths off public pages.

## Process and CI

The merges came faster than CI could check them, on a main that was already red, so CI has not judged any of it.

- **Merge speed.** 68 of the first 89 PRs merged under five minutes after opening (median one minute), none reviewed.
- **Red main.** The last green run was 867895e32 (2026-11-22 20:30Z). Main went red at cadb695d9, the merge of #2127,
  whose merge result was not compiled before merging (an `nvrm_gpu.c` argument mismatch). Merges then continued on red
  for about 2.5 days. Each merge cancels the run before it, and the last completed run (70d5a7cab) failed the compile,
  host and light jobs.
- **Gate phases skipped.** Since #2181 the compile gate stops at "stage-runner binaries", so the nvrm build, the
  dependent-wait audit and the hosted-root checks have not run. The gate's error filter hides the real compiler
  message, which is only in the `cuda13-sm121a-receipts` artifact.
- **Code-size budget.** 42 new grants added 23,281 lines of allowance, so the budget no longer limits anything.
- **Releases.** Five release roots reached the fleet within 8 hours (98386babe, 07a6e6bcd, d99d5cc81, 99fc2a2c1,
  454edc951), each qualified only by RAIS's text-level checks.
- **Drafts.** Ten drafts were merged: #1916, #1943, #1944, #1947, #1950, #1951, #1958, #1970, #2055 and #2128. #2128,
  the nvrm mesh path, was still waiting on its two-node benchmark and four-node test. #2055 went in without its
  before/after measurement, and #1972 now conflicts with main.

Suggestions:

- Required green checks on main and a merge queue. Nothing merges on red except the fix, and the merge result compiles
  before it lands.
- The gate prints the first real compiler error and runs every phase, even after one fails.
- At most one fleet release a day, each with the kit's old/new capture, the golden compare and Gate GX.
- Grants need a reviewer and a measured reason, or the ceiling is raised once, openly.
- Nobody merges another person's draft.

## Not covered

These parts were skimmed only and still need a reviewer:

- Metal kernel work for M5 Ultra (#2165, #2177, #2192, about 4K lines) and the Metal stages merged with draft #1916.
- New model stages and packers: HY4-FP8, Ling 3.1-Flash and 3.2-Air, sage-s2.1-flash (#2223, #2229), checked against
  their publishers' reference code.
- The kernels inside the five new drafter modules, and the tree-branch drafting changes (#2145, #2152, #2216).
- GPU-side collective kernels (#2171, #2199, #2224) and the per-family performance PRs #2133 to #2142 and #2193 to
  #2213.
- The RAIS tools themselves (`tools/rais_score.py`, `tools/rais_bench.py`), beyond what the site claims.
- Each of the 24 new test files. One known flaky one: `tests/test_model_batch_engine_mock.c` line 3981 depends on
  timing.
- #2210, which saves a recurrent model's prefix state before the chat template's suffix, and #2055's start-cost
  change, never measured before and after.
