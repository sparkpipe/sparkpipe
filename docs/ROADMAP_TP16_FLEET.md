# Roadmap: TP16-everything fleet → weightd-managed serving → per-model sessions

Operator directive 2026-08-30. Six phases, each with a verification
gate; no phase stacks on an unverified one (bottom-up law).

## Current state, honestly separated

WORKS: parallel 16× daemon launch (proven tonight — simultaneous,
correct ranks); config generation (generator + sane limits); single-
node serving end-to-end (27B at 96% telemetry-confirmed); queue/
dispatcher layer; glm5_next TP16 packs (served all night); glm5.3-full
ALL THREE resolutions placed fleet-wide (nvfp4 526G, fp8 866G, bf16
1.568T — receipted tonight).

DOESN'T: dsv4-flash on-node packs are Aug-11 era vs current adapter
(tensor_count drift — the 16× failure's root cause); the other models'
TP16 sets don't exist yet (27B is TP1-only, k3 is TP4PP4, qwen-flash
TP4, dsv4-pro TP4PP4, qwen-max never packed); weightd is built and
hardware-verified but NOT the serving path; code-update-without-weight-
reload is UNTESTED (weightd's whole point).

## Phase 1 — canonical TP16 stagepacks (OPERATOR: model devs build their own unless done/in-progress)

Coordinator retains: qwen-max nvfp4 (in flight, 15/16) + the done sets
(glm5.3-flash, glm5.3-full). DEVOLVED to model-dev sessions with the
guide's stagepack section as the runbook: dsv4-flash canonical rebuild,
27B TP16, k3 TP16, qwen-flash BF16 (per the corrected policy), dsv4-pro
TP16. The 5-min cycle verifies/places against receipts as they land.


Per-node NVMe is ~3.7TB; fleet-wide storage need is modest (below).
Per-rank sizes (total/16): 27B-fp8 2G · dsv4-flash 8.4G · qwen-flash
~10.5G · glm5.3-flash ~25G · glm5.3-full-fp8 54G · dsv4-pro ~52G ·
qwen-max ~144G · k3 ~96G.

Build matrix (one spark per model, all in parallel — source reads hit
warm ceph concurrently):

| model | source (warm) | build | node |
|---|---|---|---|
| dsv4-flash | deepseek-v4-flash-0731 (156G) | base packer → tp16 splice (NOT the merge-format v3_full) | spark0 |
| 27B | incoai fp8 (30G) | qwen38 packer TP16 | spark1 |
| qwen-flash | 336G bf16 source (the ONLY official form held) | SERVE BF16 at TP16 (~21G/rank) — quantizing it ourselves VIOLATES the standing policy (operator, re-confirmed 2026-08-30: no self-quantization, ever; official/vetted sources only). IF an FP8/NVFP4 serving arm is wanted: hunt an official or vetted community release first (same as qwen-max-4bit); none exists on warm today | spark2 |
| dsv4-pro | dsv4-pro-0813-ga (832G) | pro packer TP16 (fix verifier pins first) | spark3 |
| k3 | kimi-k3-nvfp4 (1.5T) | reslice TP4PP4→TP16 (expert_tile_k=32 path, lane-documented) | spark4 |
| qwen-max (4-BIT per operator) | PENDING SOURCE: no official/vetted
  4-bit checkpoint on warm yet (only the 2.3T fp8). Phase 1
  prerequisite: hunt official/community NVFP4/MXFP4 qwen-max (the
  bulk-packs2 NVFP4 portfolio gap list first), fetch + verify per the
  standing quantization policy (NEVER self-quantize). IF none exists
  anywhere: operator decision (exception vs fp8 fallback). ~72-75G/rank
  at 4-bit | spark5 |
| glm5.3-flash | DONE (serving packs current) | — | — |
| glm5.3-full | DONE (fp8 = serving arm) | — | — |

HOUSEKEEPING (AMENDED per operator: 16-bit glm 5.3 is a test target):
retire on-node nvfp4 glm53full copies (33G/node — the non-serving
arm); KEEP the bf16 (98G/node) on-node through its Phase 2 test
(operator wants 16-bit glm 5.3 verified), retire after that gate
passes; retire the stale Aug-11 tp16.b1 and tp4_pp4 dsv4 copies;
retire k3's TP4PP4 node copies after the TP16 set lands and loads.

GATE: every pack passes glm52_validate_pack-equivalent per family +
placement two-pass proof ("already placed" 16/16).

## Phase 2 — every stagepack loads and does inference

Per model: isolation load on ONE node (the bottom-up law) → simple
inference check (8 tokens; accuracy NOT required — weights-loaded-
properly is the bar) → then the 16× parallel load (the proven launch
path) → same inference through the fleet. Record load time per model.
GATE: all 8 models: 16/16 ready + tokens out + load_seconds logged.
ADDED (operator): the glm5.3-full BF16 arm is a Phase 2 test target —
16-bit glm 5.3, packs already on-node, DEPENDENCY: the codec-1 module
acceptance (ruled, queued behind the coherence chain).

## Phase 3 — weightd becomes the serving path

3a. Per-node weightd RUNNING PERSISTENTLY (systemd or respawn-on-
death): the hardware-verified daemon (identity arenas, NO-2× law, fd
sharing). Load/unload scripts against it — MANUAL first, one node,
then all 16 (the operator's debug-one-step law).
3b. THE CENTRAL COORDINATOR (operator design): one process holding
the fleet memory table (per-node 110GiB budgets, live arenas,
refcounts); LOAD(model)/UNLOAD(model) messages fan to node weightds;
all-16-ack or rollback; fleet-wide LRU eviction to make room.
GATE: manual load/unload of every model via the central daemon, from
scripts, with per-node memory telemetry confirming fill/drain.

## Phase 4 — NVMe-speed cycling

Verify the loop: load model → inference → unload → next, for all 8,
measuring per-model load (target: minutes, bounded by ~rank-size/
NVMe-read ≈ 8-25G at 2-3GB/s ≈ 5-10s/rank for flashes; qwen-max's
144G ≈ 60-90s). Weightd residency: second load of a model = warm hit
(seconds). GATE: cycle time table for all 8, telemetry-confirmed.

## Phase 5 — the queue rides weightd; code-update-without-reload

Dispatcher becomes weightd's client: task = "make model X ready → run
test Y". THE UNTESTED PATH: update code (.so) while weights stay
resident — weightd's stop-attach-start (the design's core promise).
GATE: a code-only update cycles in seconds with weights untouched
(memory telemetry flat through the update).
Then co-residency: 27B + both flash models resident SIMULTANEOUSLY
(~2+8.4+10.5+25G/rank ≈ 46G — well under the 110GiB law (and the 4-bit qwen-max at ~72G/rank leaves headroom for larger pairs); the
operator's 1.8TB fleet-wide budget also admits larger pairs).
GATE: interleaved inference on all three, zero weight swaps,
telemetry shows all three serving.

## Phase 6 — per-model development sessions (operator-created)

Each model gets a dedicated session; this coordinator arbitrates via
the queue + central weightd (no more fleet squatters — every access
is a task). Model sessions submit PRs; integration = this session's
merge discipline (gates by exit code, ratchet exact, telemetry as the
acceptance gate).

## Known per-model bring-up risks (carried, not new)

glm5_next: coherence (MoE-body oracle — the last unverified family);
its packs are DONE so Phase 1 doesn't block its debug. dsv4-pro:
historic verifier pin drift. qwen-max: first-ever pack (geometry
unknowns). k3: the reslice is documented but unrun. None change the
phase order; each surfaces at its Phase 2 gate if at all.

---

## PHASE E (added per operator): MULTI-TOPOLOGY COMPLETENESS

Every model gets stagepacks for EVERY topology it will serve:
TP16, TP8, TP4PP4, TP4, TP8PP2 etc. — not just TP16.

| model | TP16 | TP4PP4 | TP8 | TP4 | other |
|---|---|---|---|---|---|
| glm5.3-full | ✓ ×3 arms | — | — | — | — |
| glm5.3-flash | ✓ | — | — | — | — |
| dsv4-flash | ✓ | — | — | — | — |
| dsv4-pro | — | ✓ 10/16 placed | — | — | TP4PP4 natural |
| qwen-max | ✓ PP16 | — | — | — | — |
| qwen-flash | — | — | ✓ 8/8 placed | — | TP8 (24 heads) |
| qwen 27B | — | — | — | ✓ 4/4 placed | TP4 (68 blocks) |
| kimi-k3 | base building | ✓ TP4PP4 placed | — | — | TP16 + TP4PP4 both |

GAPS to fill (per model, per topology):
- dsv4-pro TP4PP4: place the remaining 6 nodes (spark0-5) — dirs premade
- kimi-k3: TP16 building on warm; the existing TP4PP4 packs stay kept
- 27B: TP4 placed; TP16 blocked on the 68-block grid (variable-width port or TP4 serves)
- glm5.3-flash: TP16 done; TP8/other topologies as demand arises
- OLD-topology cleanup: qwen-flash TP4 packs on spark4-7, qwen-max FP8 TP4 on spark7, glm52 sets — the unambiguous stale items. TP4PP4 sets for k3/pro are KEPT (they are the pro serving topology)
