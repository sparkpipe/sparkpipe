# Shared-socket family wrapper templates (multidev lanes 0-7)

These templates are the extraction of the VALIDATED GLM5.3-flash multidev
deployment wrapper (PR #1082 receipts: eight concurrent TP16 residents per
fleet run, 13.3-13.5 aggregate tok/s, 48,346 MiB device per Spark, 256/256
tokens exact, verified clean shutdown) into the parts every driver family
reuses. Lane 0 (glm5_next) owns this pattern; the other seven lanes port it
rather than deriving private variants.

Files:

- `family_deployment.template.json` - deployment layout skeleton (schema 2,
  one `nodes[]` entry per rank, `rank_index` ordered from 0).
- `run-family-job.sh.template` - the queue command file: queue-contract
  validation, private runtime preparation, weightd mode selection, optional
  smoke-set preload, resident launch.

## The pattern in one paragraph

Each participating node runs the same wrapper through the authoritative
queue (`spark_queue.py add --per-node --cmd-file`). The wrapper checks the
queue contract (`SPARK_QUEUE_ATTEMPT/RANK/SIZE/PORTS/RUNTIME_ROOT`), rewrites
the committed family deployment and stage config into a private deployment
under `$SPARK_QUEUE_RUNTIME_ROOT` (private runtime root, private KV
directory, all listeners remapped into the lane's queue-reserved port
ranges), exposes exactly one verified `packs/*.sha256` digest sidecar for
weightd attach, then `exec`s the verified `sparkpipe_model_residentd` with
`SPARK_WEIGHTD_ATTACH=1`, the socket path, the lane id and the physical mesh
rank map. Nothing here starts a daemon in shared-socket mode.

## Porting checklist (per family)

1. Commit the family deployment from `family_deployment.template.json` into
   `deployment/<family>_<topology>/` with one node entry per rank and the
   on-NVMe runtime roots (precedent: `deployment/glm5_next_tp16/`).
2. Fill `FAMILY PARAMETERS` in the wrapper copy: family, lane, hosts,
   `MESH_RANKS` (see the topology table in `docs/MULTIDEV_QUICKSTART.md`),
   build prefix, deployment source path, family env flags.
3. Generate the pack sidecars with the family manifest tooling
   (`tools/<family>_experts_manifest.c` precedent): `<pack>.experts` for
   routed-expert identity and the exact `<pack>.sha256`. The release shared
   weightd additionally requires the whole-pack `<pack>.ck128` sidecar
   (stamp once per node; glm5_next precedent:
   `tools/glm5_next_ck128_stamp.sh`). Loading fails closed without them -
   generic byte segments do not substitute for routed-expert IDs.
4. Reserve every listener with `--ports` at submission: control
   `23000+16L..+15`, collective `53000+16L..+15`, transport
   `64000+16L..+15` (from `tools/devcycle/lane_assignments.json`,
   collective block amended by PR #1094 — the original 67000+16L block
   exceeded the 65535 TCP port bound and was rejected by the queue), plus
   the session block `23168+64L..+63` when the stage config carries session
   matrices or a draft bridge.
5. Budget the lane from the measured working set with
   `tools/devcycle/lane_budget_calc.py` against the family
   `smoke_experts.json`; defaults are 9792 total / 6400 device MiB per node.
6. Numerical validation runs on the qualified harness, not this wrapper:
   `python3 tools/inference_smoke.py --spec <family spec>` (exact-token
   reference, receipts, shutdown checks). The wrapper is the resident launch
   path.

## Weightd modes

- `shared-socket` (default): attaches to the operator-tracked shared daemon
  (`FAMILY_WEIGHTD_SOCKET`); fails closed if the socket is absent. Daemon
  budgets are the operator's declarations; the wrapper exports none.
- `private`: the qualified single-node smoke shape - starts a job-cgroup
  weightd from the lane device budget, single node only. Multi-node private
  smoke belongs to `tools/inference_smoke.py`.

## Hard rules inherited from the validated wrapper

- Fail closed on: malformed pack identity, unknown listener configuration
  key, unreserved listener port, missing shared socket, unbounded KV
  backing, pack path escaping the runtime root.
- `packs/` must contain exactly one `*.sha256` digest for shared weightd
  attachment; pack bytes are shared read-only via symlinks.
- Every child stays in the queue cgroup; no nested SSH GPU launches; no
  private weightd on multi-node jobs.
- Shared-lane smoke keeps B1/one-active-sequence with the pinned CUDA
  environment (`CUDA_MODULE_LOADING=LAZY`,
  `CUDA_DEVICE_DATA_LOADING=LAZY`, `CUDA_DEVICE_MAX_CONNECTIONS=32`).
- GLM whole-chain graphs additionally require
  `SPARK_GLM5_NEXT_GRAPH_PATH=1`, `SPARK_GLM5_NEXT_PIN_EXPERTS=1` and the
  full expert pool; partial-pool eager mode is not GPU-qualified. Substitute
  your family's qualified flags - never invent new ones.

## Cold-launch preload hook

`FAMILY_WORKING_SET` (a `.wset` of deduplicated routed-expert key pairs)
warms the smoke expert set through the same socket the resident attaches to
before launch - the milestone-3 `< 5 s` cold-launch path. The set itself is
recorded in `model-families/<family>/smoke_experts.json` (lane 0 lands the
first instance). On a shared socket this materializes pages in the
operator's daemon: keep the working set inside its declared expert-pool
budget.

## Local compile-gate fallback (Actions stalls)

When Actions does not queue a run for a pushed head, gate the exact head on
sparkb through the queue (sync the ref, then a CPU job):

```sh
export PATH=/usr/local/cuda/bin:$PATH      # cuobjdump is not on sparkb's default PATH
export NVCC=/usr/local/cuda/bin/nvcc
export CUDA_ARCH=sm_121a
bash tools/cuda13_sm121a_compile_gate.sh
```

The run must end with `PASS CUDA 13 exact sm_121a compile gate`; retain the
attempt id and log path in the PR (receipt precedent: PR #1091's
ebc7cd31 fallback, PR #1085's r5).

## Job-submission rules (each paid for with an incident)

- The queue provides `SPARK_QUEUE_RUNTIME_ROOT` as a PATH, not a
  directory - wrappers `mkdir -p` their own runtime root before the first
  write (PR #1101 r1 postmortem).
- Queue `--cmd` / cmd-files carry NO shell syntax at all: `systemd-run`
  pre-expands `$VAR` in cmd-file text (lane 5 probe: `[/home/${X}]` became
  `[/home/]`). Wrappers are bare repo scripts; environment comes from
  `ARM=A bash tools/<family>_job.sh` style invocations only.
- `weightd_warm` pins the GLM pack identity (`glm5_next_stage`) in its
  attach slice; non-GLM families need the identity parameterization before
  routing a warm path through it (lane 5, additive).

## M3 receipt template (the three-number preload convention)

Per the operator preload model (PR #1103): expert tracing is one-time, the
smoke-expert manifest IS the permanent working set, new instances
batch-preload the entire traced set at launch, the shared weightd holds it
resident for the daemon lifetime, and only a weightd restart re-triggers
the slow load. Cold-launch receipts therefore report THREE numbers:

1. `daemon_cold_warm_seconds` - one-time warm of the traced set into a
   cold daemon (per daemon lifetime, NOT the launch target).
2. `instance_ready_seconds` - a new instance's time-to-ready against the
   warm daemon. THE `< 5 s` TARGET IS THIS NUMBER, never (1).
3. `steady_state_decode_tokens_per_second` (+ `ttft_seconds`) - serving
   from resident RAM.

The A/B contrast arm (no preload) additionally reports the same axes with
the daemon cold, to show what the preload removes. The receipt is plain
KEY=VALUE props (`receipt.props`) merged into JSON at finalize - digits
become numbers, everything else stays a string - plus fixed identity
fields:

```
arm, rank, attempt, status
residentd_sha256, weightd_warm_sha256, model_batch_sha256, wset_sha256
daemon_census_before, daemon_census_after
time_to_ready_seconds, ready_line
warm_seconds, warm_log            (preload arm only)
request_seconds
bench: ttft_seconds, decode_tokens_per_second, token_count, valid
teardown (term|killed), error (if any)
```

Every number is smoke-relative (shared-lane stdout timing, stated mode)
unless produced by an isolated fleet window.

