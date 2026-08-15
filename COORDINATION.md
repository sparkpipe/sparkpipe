# SparkPipe multi-session development coordination

Scope: the six parallel model sessions sharing the 16-Spark fleet
(spark0-9, sparka-f) and the shared `sparkpipe` repository.

## Fleet facts

- Every Spark: 128 GB unified memory (119 GB visible), two rails
  (100 GbE switch + direct pair), CUDA 13 on `/usr/local/cuda-13.0`.
- A residentd deployment claims, per rank: GPU memory, a TCP control port,
  collective listen/peer ports, a transport control port, an RDMA fabric
  share, and a KV backing directory. Nothing else conflicts.
- Measured resident footprints: DSV4 Flash TP4 ~38.5 GB device + 11 GB pack
  per rank; Qwen 27B PP16 ~2.9 GB pack per rank. Two or even three models
  fit in one node's memory. **Memory is not the constraint; measurement
  cleanliness is.**

## Port registry (the coexistence mechanism)

Two deployments coexist iff they never share a host AND a port. The registry
allocates per-model blocks so sessions never negotiate ad hoc:

| Model / slot | Hosts | Control port (per rank) | Collective listen/peer | Transport ctrl base |
| --- | --- | ---: | ---: | ---: |
| DSV4 Flash TP4 (always-on) | spark4-7 | 18480 | 62620-62623 | 59700 |
| Qwen 27B PP16 (always-on) | spark0-3 | 17480 | 61620-61623 | 58700 |
| Big-model slot A | spark8-f | 19480 | 63620-63623 | 60700 |
| Big-model slot B (full-16 windows) | all 16 | 20480 | 64620-64623 | 61700 |

Rules:

1. A session must use exactly its registered block on its registered hosts.
   Control port = block base (one per rank is fine: the same port on
   different hosts never collides).
2. Adding a model = adding a row here FIRST, via PR, before any deploy.
3. The KV backing directory lives under the runtime root, so runtime dirs
   are already isolated. Never point two deployments at one KV dir.

## Coexistence vs measurement

- **Dev-active (always OK):** residentd up, kernels idle. Two or three
  models may be dev-active on the same node; the idle residentd consumes
  memory only.
- **Measured runs (exclusive):** B1 decode receipts are latency-critical;
  any co-resident GPU work, L2 traffic, or RDMA traffic on the measured
  hosts invalidates them. Before a measured window, stop the other models'
  residentds ON THOSE HOSTS ONLY. Model promotion is <60 s, so swapping is
  cheap. Restore them after the window.

## Time slices

30-minute exclusive measurement windows, scheduled on the shared hosts
(spark8-f for four-host topologies; all 16 for TP16/PP16 topologies, in
which case the always-on models pause on their hosts for the window).

- Priority order decides the queue. DSV4 Flash is currently leading-edge:
  half the slots.
- A window that starts late ends on time; no overruns into the next slot.
- A session that only needs its always-on hosts (DSV4 Flash, Qwen 27B) does
  not consume a slot.
- Every receipt records the fleet state at run time (see probe below).

## Repository conventions

- One worktree per session under `/Users/cem/Documents/dsh/`:
  `sparkpipe-dsv4`, `sparkpipe-glm52`, ... The main checkout is the Qwen
  session's; nobody else switches its branch.
- Branch prefix per model: `dsv4-*`, `glm52-*`, `qwen38-*`, ...
- Common code (`model-families/common/`, `inference/kernels/`,
  `ring/transport/`, `include/sparkpipe/`, `runtime/`): changes stay on
  candidate branches. Nothing common is pushed to main until a measured win
  plus cross-session review. Prefer additive changes (new kernels/functions)
  over edits to shared paths.
- Everything is PR'd: no experiment may live only on a Spark or only in a
  local tree. Push via `tools/sparkpipe_github_pat.sh` (never plain git
  push with a cached credential).

## Fleet probe

`tools/devcycle/fleet_status.sh` prints one line per host: which residentd
(rank + cwd) is running, or free. Run it before every measured window and
include its output with receipts.
