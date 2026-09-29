# Accuracy A/B on the fleet: runbook

This runbook covers the fleet side of an accuracy A/B campaign. A campaign
compares **arms**. Two arms differ on exactly one axis: the routed-expert codec
(E), the KV codec (K) or the drafter (D). Everything else, the **spine**, is
byte-identical. The statistics, the arm descriptor, the score dump and the
receipts schema are described in `qualification/ab/`. This document says how
arms are placed on the Sparks, how they are started, run and stopped, and
which guards keep them away from production.

The first campaign is GLM-5.3 Flash TP16 (`qualification/ab/fleet/glm53flash_w1.json`):

| Arm | Spine | Experts | Role |
|---|---|---|---|
| F0 | production (FP8 release dequantized) | publisher FP8 | bridge to production |
| F1 | publisher BF16 release | publisher BF16 | E ceiling, campaign reference |
| F2 | publisher BF16 release | publisher FP8 | E anchor |
| F3 | publisher BF16 release | NVIDIA NVFP4 | 4-bit arm |
| F1AA | as F1, same arena | as F1 | second engine for the A/A gate |

Arms built by our own quantization (F4-F8) need owner decision O1 and are not
part of this runbook until it is granted.

## 1. Rules that never bend

- **Production is the lead's.** Only the lead touches the production root
  (`~/sparkdata/glm53flash.fp8.tp16`), `fleet-agent`, weightd lane 0, the
  `g53-api` and `~/release`. Arm runs happen in lead-scheduled windows.
- **No arm attaches production's pack outside a lead window.** weightd keys an
  arena by pack identity (pack SHA, revision, pool, manifest), not by lane.
  A root that points at production's rank pack with production's pool therefore
  maps production's arena, and shares its crash domain. F0 does this on
  purpose, and only in a window with production held.
- **No model API on any Spark.** Arms are driven with `sparkpipe_model_batch`,
  which needs no API.
- **Headroom.** Every node keeps at least 20 GiB MemAvailable. The rule and
  the number come from `tools/fleet_node_agent.sh` (`FLEET_AGENT_HEADROOM_GIB`,
  default 20). Lowering it for a window is a lead decision, never a plan
  assumption.
- **Perf windows.** Accuracy runs are not speed runs. A GPU job longer than a
  minute on a GLM node needs `lanes/PERF_HOLDER` to be free or held by the
  lead for this window. Speed numbers come only from exclusive perf windows and
  always carry a roofline line.
- **Disk.** Every arm root and pack directory is registered in the node's
  `~/KEEP`. Non-pack files stay under 20 GB per lane per node.

## 2. Arm roots

An arm root is a fleet-agent layout root at `~/sparkdata/<root>` on all 16
nodes:

```
~/sparkdata/qab-f2/
  agent.env                 rendered by tools/ab_fleet.py env
  model_resident.json       runtime_root, control ports, kv_backing_directory = <root>/run/kv_backing
  config/stage_NN.json      one per rank; stage.json -> stage_<rank>.json
  bin/ lib/ stages/         the arm firmware (identical SHA256SUMS on every node)
  packs/                    symlinks to the arm's pack directory (read-only)
  inputs/                   probe file, Tier-2 row list (same SHA on every node)
  run/                      the current run's private directories; created empty per run
  runs/<run_id>/            finished runs (dumps, logs), moved out of run/
```

### 2.1 agent.env

`python3 tools/ab_fleet.py env <spec> <arm>` renders it. The keys follow
`docs/FLEET_AGENT_ROOTS.md`:

| Key | Value |
|---|---|
| `AGENT_ROLE` | `dev` |
| `AGENT_SYNC` | `local` (never synced from the hub release) |
| `AGENT_MEMORY_MAX` | residentd cgroup limit, `20G` |
| `AGENT_MEMORY_NEED_GIB` | residentd plus the arena if this arm is the first to map it in its slot (F1 50, F1AA 12, F2 33, F3 24, F0 12) |
| `SPARK_WEIGHTD_LANE` | the arm's own mesh lane |
| `SPARK_WEIGHTD_EXPERT_POOL_BYTES` | per arm; arms that share an arena must use the same pool, or weightd refuses the attach |
| `SPARK_WEIGHTD_ATTACH`, `SPARK_WEIGHTD_SOCKET`, `SPARK_TP_MESH_RANKS`, `SPARK_TP_WAIT_MODE`, pin and graph switches | from the spec's `common_env`, equal to production's execution mode |

The window scripts start the arm residentds as lane-owned transient units
(`sp-qab-<arm>`) with the same env, `MemoryMax` and gate as the fleet agent
would use. The arm roots are **not** listed in `~/.fleet_agent_roots`, so the
agent never restarts or yields them in the middle of a run. A lead who prefers
agent supervision can list them; the agent then enforces the same gate.

### 2.2 Private directories

`SparkKvSnapshotPrune` deletes every snapshot file whose layout digest differs
from the running engine's. An arm pointed at production's snapshot directory
would delete production's snapshots at startup. Score dumps contain prompt
logits and must never land in a production root. So:

- `kv_snapshot_directory`, `score_dump_directory` and `kv_backing_directory`
  resolve under the arm root, after symlinks. Stage members are relative paths
  (the runtime joins them to the runtime root).
- No private directory may lie inside a protected path (production's root, its
  KV backing directory, `~/release`, `~/sparkdata/core`, `~/sparkdata/weightd`)
  or overlap another arm's.
- Every run starts with them empty.

`python3 tools/ab_fleet.py root-check <spec> ARM=ROOT ... [--fresh]` enforces
all three. The window scripts run it on every node before every start, and
refuse the run if it fails. `tests/test_ab_fleet.py` covers the refusals:
a snapshot directory symlinked into production, a `..` path, an absolute path,
production's KV backing directory, a root that aliases production, two arms
sharing a directory, and a non-empty directory under `--fresh`.

### 2.3 Lanes and ports

Arms take mesh lanes from the free set. A lane is free when no residentd
holds it on any node (`SPARK_WEIGHTD_LANE` in the process environment;
production without the variable takes the lowest free lane, 0) and no other
lane's notes reserve it. On 2026-09-29 the free set was 1, 2, 11, 13, 14 and 15;
lane 11 is left out because the dormant laguna plan reserves it with the same
port formula.

| Arm | Lane | Control | Collective | Transport base | Session blocks |
|---|---|---|---|---|---|
| F1 | 1 | 23016+r | 53016+r | 64016 | 64256, 64512 (group 0) |
| F1AA | 2 | 23032+r | 53032+r | 64032 | 64768, 65024 (group 1) |
| F2 | 15 | 23240+r | 53240+r | 64240 | group 0 |
| F3 | 13 | 23208+r | 53208+r | 64208 | group 1 |
| F0 | 14 | 23224+r | 53224+r | 64224 | group 1 |

Concurrent arms never share a port group. The collective identifier is
`0x514142000000 + lane`.

## 3. Memory and placement

### 3.1 What an arm costs

- A residentd: about 12 GiB with production's runtime limits (16 sequences,
  8192 KV pages). Arms run CT-short with 2048 pages, which is expected to cost
  less. The receipts measure it.
- An arena: about the pack size (spine plus pinned experts). F1 37.4 GiB,
  F2 20.4 GiB, F3 about 12 GiB. An arena is paid once per node, by the first
  arm that maps it. F1AA rides on F1's arena and F0 on production's.

### 3.2 The rule

The fleet agent starts a dev root only if
`floor(MemAvailable) - AGENT_MEMORY_NEED_GIB >= 20`, checked against the
MemAvailable that the roots already running leave. So the arms that run
together must satisfy, on **every** node:

```
floor(MemAvailable before the slot) - sum(AGENT_MEMORY_NEED_GIB) >= 20
```

A TP16 arm runs on all 16 nodes or on none. An optional arm (F2 kept beside
F0) is kept only if every node fits.

### 3.3 Dry run

```
python3 tools/ab_fleet.py place <spec> <memory.json> [--basis live|window] [--evicted production]
```

`memory.json` maps each node to `mem_available_gib`, `window_available_gib` and
`lanes_in_use`. The lane probe writes it from live readings:

- `mem_available_gib` is MemAvailable now.
- `window_available_gib` projects a full-fleet window: MemAvailable, plus the
  GPU memory of every residentd (all stopped or held), plus weightd's GPU
  memory above production's 20.4 GiB arena (the other lanes' arenas
  reclaimed, production's kept).

The command exits 0 only if every slot fits and no arm lane is in use. The
window scripts repeat the check with live MemAvailable right before each slot,
and gate each unit start the same way.

## 4. Reclaim

Stopping a residentd leaves its arena resident (cold). Two ways to free it:

- `weightd_warm /tmp/spark_weightd.sock --reclaim-pack <pack>` (#1345, once
  merged and rolled out to the fleet weightd) frees only that pack's cold
  arenas. This is the scoped reclaim the window scripts use when the fleet
  weightd has it.
- `weightd_warm /tmp/spark_weightd.sock --reclaim` frees **every** cold arena on
  the node, including a held production arena and any stopped lane's. The
  scripts use it only when `RECLAIM_NODE_GLOBAL_APPROVED=yes` is set by the
  lead, and only after checking that no residentd other than the arms is
  running. F0 then reloads production's arena (need 33 GiB instead of 12, use
  `--evicted production` in the dry run). Production restarts on the arena
  that F0 left warm.

A node-global reclaim between a production stop and its restart drops
production's arena. Before a window, run it while production still serves
(its arena is busy and stays); after a window, run it once production is
ready again.

## 5. The W1 window

W1 is the first result: F1, F2, F3 and the F0 bridge on CT-short, with the A/A
gate. It needs a full-fleet window of about 3.5 h: production held, every
other lane stopped, their arenas reclaimed, production's arena kept.

Slots (each is one runnable script with receipts; see the lane notes for the
exact commands):

| Slot | Arms (GiB/node) | Runs |
|---|---|---|
| A | F1 (50), then F1AA on F1's arena (12) | F1 reference CT-short; merge and probe file; F1AA A/A in permuted order concurrently with F1's COMPSEC-17; stop; reclaim F1's pack |
| B | F2 (33) + F3 (24) | F2 and F3 probe runs concurrently; F2 second run (fresh engine); COMPSEC-17 on both, F2 with the dump on and off; stop F3 (and F2 unless it is kept for C) |
| C | F0 (12 on production's arena), F2 kept only if every node fits | F0 probe run; stop; production's arena stays for production's restart |

Every run:

1. stops the arm's units if running, moves `run/` to `runs/<previous id>/` and
   creates an empty `run/`;
2. runs `root-check --fresh` on every node;
3. checks the gate with live MemAvailable, then starts 16 units and waits for
   16/16 `model_residentd ready` (abort on any unit failure or MemAvailable
   below 20 GiB on any node);
4. drives the batch files with `sparkpipe_model_batch` on the coordinator node,
   `SPARK_MODEL_BATCH_SEQUENTIAL=1`, `output_token_budget=1`, at most 16
   requests per file, and refuses the run if any event reports
   `cached_prompt_tokens != 0` or any request does not complete;
5. writes the receipt: arm, run id, source commit and firmware SHA256SUMS,
   pack SHAs, lane and pool, env and config SHAs, the `ready` event, request
   and event counts, per-rank dump SHAs, MemAvailable before and after and unit
   MemoryCurrent per node, the weightd map lines, co-tenants, UTC start and end.

The A/A gate is hard: the F1 and F1AA merged dumps and generated token ids
must be bit-identical, and so must F2's first and second runs. If they are
not, the campaign stops and a determinism bug is filed. If a node cannot fit
F1AA beside F1 when the window opens, slot A runs without F1AA and the gate is
F2's repeat alone (F2's second run is concurrent with F3).

Served tokens must also be identical with the score dump on and off (F2's
COMPSEC-17 runs); otherwise the dump is not output-neutral and the campaign
stops.

## 6. After a window

- Stop the arm units, reclaim only the arm packs, leave production's arena.
- Flip the arm roots' `~/KEEP` lines to `DELETE-OK` unless the next window
  needs them. Packs that later windows need stay `KEEP`.
- Copy the Tier-1 dumps and receipts off the nodes. Delete an arm's Tier-2 rows
  after its partials are computed; keep the reference's.
