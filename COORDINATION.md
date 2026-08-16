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
  per rank; Qwen 27B PP16 ~2.9 GB pack per rank. Two or three models fit in
  one node's memory. **Memory is not the constraint; measurement cleanliness
  and big-model mutual exclusion are.**

## The isolation model: tiers, scopes, and one current big model

Every model is registered in
[`deployment/fleet_registry.json`](deployment/fleet_registry.json) with a
**tier** and a **scope**:

| Model | Tier | Scope | Hosts |
| --- | --- | --- | --- |
| Qwen 27B | always-on | band | spark0-3 |
| DSV4 Flash | always-on | band | spark4-7 |
| GLM 5.2 | big | band | spark8-f |
| DSV4 Pro | big | fleet | spark0-f |
| K3 | big | fleet | spark0-f |

- **Always-on models** (Qwen 27B, DSV4 Flash) each own a fixed four-host
  band and run concurrently with each other and with one big model.
- **Band big models** (GLM 5.2, ...) share the spark8-f band and are
  mutually exclusive: exactly one of them is the **current big model**.
- **Fleet big models** (DSV4 Pro, K3) take all 16 sparks: swapping one in
  evicts *everything*, including the always-on tier. K3 is the heaviest
  case and is expected to need this; the swap mechanism handles it the same
  way as DSV4 Pro.
- The big data stays on each Spark's NVMe (`sparkdata/` dirs); a swap is
  stop-old + start-new, no data copy. Promotion is <60 s.

## Designating the current big model

`tools/fleet_swap.sh MODEL` is the single mechanism every session uses:

- `fleet_swap.sh glm52` — stops the current band model on spark8-f,
  starts GLM 5.2 there. Qwen 27B and DSV4 Flash are untouched.
- `fleet_swap.sh k3` — snapshots which models were running, stops ALL
  residentds on all 16 sparks, starts K3. Swapping it back out restores
  the snapshot.
- `fleet_swap.sh status` — prints the current big model and the running
  set (reads the state file).
- State lives at `/tmp/sparkpipe_fleet_state.json` on every spark, with
  the authoritative copy on spark0. **Never start or stop a big model
  residentd by hand** — always through the swap script, so the state file
  stays true.

Adding a model = adding a registry entry (tier, scope, hosts, ports,
runtime root, pack dir) via PR, then swapping.

## Port registry (inside the model registry)

Ports are assigned per model in `fleet_registry.json` so two deployments
never share a host AND a port. Current blocks: Qwen 27B control 17480 /
collective 61620 / transport 58700; DSV4 Flash 18480 / 62620 / 59700;
big band 19480 / 63620 / 60700; fleet slot 20480 / 64620 / 61700; K3
21480 / 61620 / 62700.

## Coexistence vs measurement

- **Dev-active (always OK):** residentd up, kernels idle. The always-on
  models plus the current big model may be dev-active together; idle
  residentds consume memory only.
- **Measured runs (exclusive):** B1 decode receipts are latency-critical;
  any co-resident GPU work, L2 traffic, or RDMA traffic on the measured
  hosts invalidates them. Before a measured window, stop the other models'
  residentds ON THOSE HOSTS ONLY (via the swap script if it is the big
  model, or a targeted stop for the sibling always-on model). Model
  promotion is <60 s, so swapping is cheap. Restore after the window.
- Always-on bands (DSV4 Flash on spark4-7, Qwen 27B on spark0-3) measure
  without consuming a time slice whenever their band is otherwise idle.

## Time slices

30-minute exclusive measurement windows on the big band (spark8-f), and
whole-fleet windows for fleet-scope models (during which the always-on
models pause). Priority order decides the queue; a window that starts late
ends on time. Every receipt records the fleet state at run time
(`tools/devcycle/fleet_status.sh` output).

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
