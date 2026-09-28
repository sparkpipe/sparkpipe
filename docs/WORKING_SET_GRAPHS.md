# Working-set graphs: poison, rollback and replay

A CUDA graph that routes MoE experts can run while only a working set of
experts is resident. A step that routes to an expert the rank does not hold is
detected inside the graph. Every rank then agrees that the step missed, and the
step is discarded, rolled back and replayed after the missing experts are
loaded. The committed output is bit-identical to the output of the graph with
every expert pinned.

All of the pieces below are model-neutral. A family provides its layer count,
expert count, miss-pack stride and recurrent-state spans.

## Pieces

| Piece | File | Role |
|---|---|---|
| Cover kernel | `inference/kernels/expert_cover.cuh` `LmExpertCoverKernel` | After the router. A covered route writes nothing. A miss substitutes the first covered expert of its layer (memory safety only), appends `layer * pack_stride + expert` to the slot's miss ring (the ring never wraps; `miss[1]` keeps counting) and sets `miss[0]`. A layer with no covered expert traps. |
| Poison | `LmHeadMissPoisonKernel` | Before the head MAX all-reduce. On a local miss it sets every row's maxloc to `UINT64_MAX`, a value no ordered score can produce. |
| Unpack | `LmHeadMaxlocUnpackPoisonKernel` | Maps `UINT64_MAX` to token `SPARK_STEP_POISON_TOKEN`. |
| Verdict | `include/sparkpipe/spark_step_verdict.h` `SparkStepVerdictClassify` | Takes the reduced tokens plus the local miss flag and returns commit, rollback-local, rollback-remote, mixed or poison-lost. |
| Replay policy | `SparkStepReplayNext` | Commit, replay (at most `limit` times), or exhausted (the caller runs that step on its exact eager path). Mixed and poison-lost fail. |
| Snapshot | `inference/kernels/state_snapshot.cuh` `LmStateSpansCopy` (spans: `include/sparkpipe/spark_state_span.h`) | Gathers the wave's recurrent-state rows (through the same `state_index` the layer kernels use) into a snapshot at graph start. Rollback scatters it back. Rows outside the wave are never touched. |
| Working set | `runtime/spark_expert_working_set.c` | See below. |

The working-set manager keeps the host cover bitmap (`layers x ceil(experts/32)`
words, the layout the cover kernel reads). It enforces these rules:

- **W1, cover implies held.** A bit is set only after the acquire hook
  succeeds for that key. `Add` is all-or-nothing, so an invalid key, a denied
  acquire or a key count past `cap_keys` leaves the cover unchanged and is
  counted in `grow_denied`.
- **W2, anchors.** `CheckAnchors` fails with UNSUPPORTED and names the first
  routed layer that has no held expert.
- **Harvest.** Only entries of the first missed layer L\* are acted on. Later
  layers ran on substituted, and therefore wrong, activations. Ring overflow is
  accepted when the ring still holds a later layer, which proves that L\* is
  complete. Otherwise harvest returns CAPACITY_EXCEEDED and the caller must
  take the eager path. An out-of-range route at L\* is VALIDATION_FAILED.

## Step protocol (host-stepped chains)

1. The PARTIAL graph starts with a ring reset (memset node) and the snapshot
   gather, runs the cover kernel after every router, and poisons before the
   head reduce.
2. After the step's stream sync, each rank classifies its step. All ranks see
   the same reduced tokens, so the verdicts agree up to local versus remote.
3. On rollback each rank does the following. None of it runs `FeedStep`.
   - It restores the snapshot.
   - It harvests its own ring and calls `Add`.
   - It uploads the cover.
   - It relaunches the same graph for the same step.

   Every layer before L\* hit on every rank, and every miss at L\* is loaded
   before the replay. So L\* strictly increases across attempts, and at most
   one replay per routed layer is needed.

The graphs capture only data-dependent behaviour. A miss never changes control
flow or collective ordinals, so hardware-wait collectives cannot deadlock.

## Tests

- `make test-working-set-rollback` (GPU, sparkf) runs a 16-rank toy MoE.
  It has 6 layers, 40 experts, top-2, a recurrent state per layer and 2 rows
  mapped to non-contiguous state slots. Each step's per-rank MoE partials are
  sum-reduced, and the head is a sharded argmax with a MAX reduce. It checks:
  - The all-pinned graph is deterministic.
  - A cold working set starts with one anchor expert per layer (NaN in every
    unheld expert slot). It rolls back and replays until it is warm, and its
    tokens and final recurrent state are bit-identical to the all-pinned graph.
  - The warm rerun has zero rollbacks.
  - When exactly one rank lacks experts, every rollback is local on that rank
    and remote on the other 15.
  - State slots outside the wave are untouched.
  - With a replay limit of 2 on a cold working set, steps that miss at more
    than two layers exhaust their replays. All 16 ranks agree on
    `SPARK_STEP_ACTION_EXHAUSTED`, restore the snapshot and run that step
    eager (per-layer router readback and expert load, no cover, snapshot or
    poison). On sparkf 2 of 32 steps ran eager, and the tokens and final
    recurrent state are still bit-identical to the all-pinned graph.
- `build/test_expert_working_set` (host, part of `make test`) checks:
  - `Add` atomicity, with W1 checked from inside the acquire hook;
  - anchors;
  - harvest rules, including overflow and corrupt rings;
  - 10,000 random growth rounds with injected denials;
  - the replay policy.

## GLM 5.3 Flash stage module (A3)

`SPARK_GLM5_NEXT_EXPERT_WSET=<path>` selects the working-set residency mode.
`SPARK_GLM5_NEXT_EXPERT_WSET_SHA256=<hex>` is required with it. The file is a
`.wset`: little-endian `(layer u32, expert u32)` pairs, as written by
`tools/glm5_next_wset_from_trace.py`.

**Refusals at attach.** The mode fails at attach, and never falls back, when:
- `SPARK_GLM5_NEXT_PIN_EXPERTS=1` is also set;
- the stage is not a single stage that owns both the embedding and the final
  head;
- there is no lazy expert pack;
- the digest does not match;
- a key is outside the stage's routed layers;
- a routed layer has no held expert.

**Leases.** Keys are leased through weightd in groups of at most 512
(`SparkWeightdMapAcquire` + `BeginUse`, sharing the pinned-lease table of 32).
Attach prints
`EXPERT-RESIDENCY mode=working-set wset= keys= of N leases= rows_max=8 replays=2`.

**Where the graph path runs.** Waves of up to 8 rows take the PARTIAL graph
or linear chain; wider waves take eager. Every captured PARTIAL step:
- resets `miss[0..1]`;
- snapshots every KDA state and Q/K/V conv-window row of the wave after
  `WaveBegin`, using `4 x kda_layers` spans and the layer kernels' own
  `state_index`;
- runs the cover kernel after every router;
- poisons the head maxloc before the head MAX reduce.

FULL graphs bind no cover and capture none of these nodes.

**Rollback.** On a rollback verdict, whether from `GraphStep`, `SettleStep`
or the final step of a working-set chain, `SparkGlm5NextWsRetry` does the
following:
1. Disarms capture.
2. Restores the snapshot and syncs.
3. Harvests the first missed layer and grows the set. Denied growth is counted
   and leaves the result correct.
4. Replays the same step. `FeedStep` does not run.

After 2 replays the step runs eager (`ws_force_eager`, cleared when the step
commits). Every input to the replay/eager decision is the agreed verdict
history, so all ranks decide the same way; a ring overflow does not change the
decision. `GRAPH-WS-RECOVER` logs each recovery with running local, remote,
replay and eager counts.

**Known limits (follow-ups).**
- Growth leases are not compacted. After the 32-slot lease table fills,
  growth is denied and misses resolve through replay then eager.
- There is no degrade gate: a lane that keeps thrashing pays two replays plus
  an eager step on every miss.
- Keys seen on eager steps are not fed back into growth.
- A corrupt ring on one rank fails that rank's chain, and the other ranks find
  out through the collective timeout. There is no cancel broadcast yet.

The fleet proof is not done yet. It needs a dev weightd lane (lane 5) running
this build with a trace-built `.wset`, compared against the same build in FULL
mode.
