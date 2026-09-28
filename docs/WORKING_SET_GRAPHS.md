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
- `build/test_expert_working_set` (host, part of `make test`) checks:
  - `Add` atomicity, with W1 checked from inside the acquire hook;
  - anchors;
  - harvest rules, including overflow and corrupt rings;
  - 10,000 random growth rounds with injected denials;
  - the replay policy.

The fleet proof is not done yet. It needs the GLM stage module wired to these
pieces (snapshot spans, PARTIAL capture, recovery in `TpChainAdvance`), running
in a dev weightd lane with a `.wset`.
