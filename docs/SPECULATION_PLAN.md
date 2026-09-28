# Speculation plan

Status: plan of record for the speculation lane, 2026-09-28. The first PR
(S1, branch `spec/verify-hooks-1`) is implemented; everything after it is
planned work. Numbers marked *model* are estimates from the fleet-calibrated
cost model in the 2026-09-28 speculation research note, not measurements.
Speculative and non-speculative results are always reported separately
(invariant I42), and speculative results are also reported per content
class (code, prose, chat, structured) because acceptance depends on content.

## 1. Owner rulings this plan implements

- **Best method per model, chosen by measurement.** MTP, DFlash2, DSpark,
  TensorFold-style trees and suffix/n-gram drafting are all candidates. No
  method is assumed; each model keeps the method that wins on the ledger
  corpora, per content class.
- **The drafter runs on the rtx5090.** Drafts reach the fleet through a
  busy-polled relay process on sparkf (rank 15), whose private link to the
  rtx5090 measured 22-45 us round trip. Rank 15 fans the draft ids out to all
  16 ranks over the mesh. This supersedes the fleet-MTP recommendation of the
  research note; its verify-path, controller and exactness work carries over
  unchanged because it is drafter-agnostic.
- **Every round is deadline-bounded.** If a draft is not at rank 15 by the
  round deadline, the round runs with k=0 (a plain decode step). A k=0 round
  is a counted, reported outcome of the round policy, not a hidden fallback:
  output is identical either way, and the miss rate is a first-class metric.
- **Multi-drafter trees.** Where drafters agree, or one is confident, a
  branch is extended; where they disagree, it is truncated or split. Trees
  are built on the rtx5090 and verified on the fleet in one target pass.
- **Target about 4x at B8 and B64.** Section 6 states what has to be true for
  that, because today's row costs do not allow it yet.

## 2. Architecture

```
 rtx5090 (sm_120, 32 GB)                 sparkf = rank 15              ranks 0..15
 +---------------------------+  /30 link  +------------------------+  mesh  +----------------------+
 | draftd                    |<---------->| relay (busy poll,      |------->| verify graph         |
 |  drafter set per model    |  22-45 us  |  pinned core)          | bcast  |  (rows, regime) key  |
 |  per-sequence drafter KV  |    RTT     |  deadline owner        | root 15|  tree mask, target   |
 |  tree builder             |            |  tap ring reader       |        |  argmax per node     |
 |  pre-drafting             |            +------------------------+        |  device accept       |
 +---------------------------+                       ^                      |  state fold          |
              ^                                      | hidden rows,         +----------+-----------+
              |          taps: committed ids +       | committed ids                   |
              +-------- hidden rows per round -------+---------------------------------+
```

### 2.1 draftd on the rtx5090

- One process, `draftd`, owns every drafter for the model being served:
  the model's MTP layer where one exists (run TP1 on the rtx5090, reusing the
  module's ops=0 MTP path compiled for sm_120), DSpark or DFlash2 block
  drafters where licensed, and the tap-free suffix/n-gram drafter.
- Per-sequence drafter state lives on the rtx5090: MTP layer KV and index
  KV, the last hidden row, and the suffix index over the sequence's tokens.
  Prefill streams the prompt's hidden rows once; each round appends the
  committed rows.
- **Pre-drafting.** While the fleet verifies round r, draftd drafts round
  r+1 for the most likely outcomes of r (all drafts accepted; first
  rejection at the most likely depth). When the real outcome matches, the
  round r+1 draft is already at rank 15 when round r ends, which removes the
  drafter from the critical path.
- **Tree builder.** Section 3.
- draftd shares the rtx5090 with the hub, `g53-api` and training jobs. Its
  VRAM budget is declared at start and checked (fail loudly if it cannot be
  reserved); it never evicts other users.

### 2.2 Relay protocol (rank 15)

The relay is a thread in the rank-15 engine, pinned to one core, busy
polling both the private link socket and the engine's tap ring. All frames
are fixed-size, little-endian, and carry `(engine generation, round id)` so
that a stale frame from an old engine or an old round is dropped and
counted, never applied.

| Frame | Direction | Content |
| --- | --- | --- |
| `TAP` | rank 15 -> draftd | round id; per sequence: sequence id, generation, first position, committed count, committed ids, and the pre-head hidden row of each committed position (GLM: 4096 x bf16 = 8 KB per token) |
| `DRAFT` | draftd -> rank 15 | round id; per sequence: node count (0..32), token id and parent index per node, drafter-agreement mask and calibrated confidence per node |
| `CANCEL` | rank 15 -> draftd | sequence id and generation: release drafter state on completion, cancellation or reset |
| `HELLO` | both | model id, drafter set, geometry and cache-layout fingerprints; a mismatch fails the connection with a named status |

- **Hidden tap.** A graph memcpy node copies each committed row's pre-head
  hidden state into a pinned ring on rank 15 (rank 15 holds the full row
  after the final all-reduce). The relay drains the ring into `TAP` frames.
  Tap bandwidth is small: 8 KB per token is about 8 MB/s at 1000 tok/s on a
  10 Gb/s link.
- **Fan-out.** The relay turns the `DRAFT` frame for the next round into one
  mesh broadcast from root 15 (`round id, rows per sequence, token ids,
  parent indices`), at most a few hundred bytes at B1 and a few KB at B64.
  Every rank launches the verify graph that message names, so all 16 ranks
  always agree on the round shape, including k=0.
- **Deadline.** The controller sets the deadline for each round: the
  expected end of the current verify plus a slack. Rank 15 alone decides at
  the deadline. If the draft has not arrived it broadcasts a k=0 round; all
  ranks follow the broadcast, never a local timer. Misses, late drafts and
  stale frames are counted per drafter and exported in the heartbeat.
- **Failure.** A dead link or a crashed draftd turns every round into k=0
  and raises the heartbeat flag. With `speculation.required=true` the engine
  instead fails requests with a named status; the default is decided by the
  owner (open question 3).
- Phase 1 launches the verify graph from each rank's host after the
  broadcast lands. Phase 2 (with the L2 device-fed chains) has the graph's
  first node wait on the broadcast flag on the device, removing one host
  submit per round.

### 2.3 Verify on the fleet

- A verify wave is its own graph regime, keyed by row count and attention
  regime (UNSPLIT, SPLIT, SELECTED), separate from the production decode
  graphs, so production graphs never change shape because speculation is
  on. S1 lands this for single-sequence chains (section 5).
- Every verify row must be bitwise equal to the B1 step at the same position
  (G-ROWEQ). That needs attention decisions keyed by each row's own
  position, not the wave's longest row: the split and DSA-selection choices
  are regime-level today, and S1 admits only waves whose rows all share one
  regime; S3 moves tiling to absolute key position (the batch-invariance
  lane L1 rule).
- Trees add an ancestor mask to attention, per-node DSA selection along the
  node's path, and a tree scan for the KDA recurrent state and the causal
  conv window: a node's state is its parent's state updated by the node's
  own key and value. The fold keeps the accepted leaf's state.
- Rejected positions leave KV and index-pool slots that are overwritten by
  the next round. The index pool (KPOOL=4) recomputes a pool whose last
  member was a rejected draft.

### 2.4 Acceptance engine

- One model-neutral device kernel resolves a verified tree: the longest
  root path whose every node equals its parent's target argmax, plus one
  bonus token (the argmax at the last accepted node), capped at the block
  end, at `max_tokens` and at the first EOS. It writes the accepted count
  and path to device memory.
- The fold kernels (KDA replay or tree state select, conv re-commit,
  index-pool fix-up, hidden-row copy for the tap, bonus-token feed for the
  next round) read that count from device memory, so the whole round is
  one graph with no host round trip.
- The host twin is `SparkSpeculationPolicyResolveVerifierTree`; the device
  kernel is pinned equal to it on random trees.
- Greedy speculation is exact by construction. Sampled speculation uses
  Gumbel-argmax noise keyed by (seed, absolute position, token id), applied
  identically by drafter and verifier, so seeded sampled streams are also
  byte-identical with and without speculation (S9).

### 2.5 Controller

A model-neutral controller in common code chooses the round shape:

- inputs: a measured table of verify cost T(rows) per regime and occupancy
  (EMA from `G5N-WAVE-TIMING` and its equivalents), per-drafter survival
  probability per depth and content class, draft latency and deadline-miss
  rate, and the distinct-expert count from the decode cover bitmap;
- output: rows per sequence for the next round (0 allowed), maximizing
  expected committed tokens per millisecond across the batch; at B>1 it
  hands rows to the sequences with the highest marginal gain per row;
- a small probe round every 32 rounds keeps the survival estimates live, and
  hysteresis stops k from oscillating;
- it never changes output, only throughput. Its decisions and counters
  (rounds, k histogram, proposed and accepted per depth per drafter, verify
  and draft ms, misses) go to the heartbeat and the completion.

## 3. Tree construction on the rtx5090

1. Each drafter proposes a chain, or a top-2 chain where it is calibrated.
2. The chains merge into a token trie; shared prefixes are verified once.
3. A node's score estimates its acceptance probability: the product of its
   path's calibrated confidences, raised where independent drafters agree
   (agreement is the strongest evidence) and lowered where they disagree.
4. Extension: a branch keeps growing while its score clears the threshold
   and the round's row budget allows. Truncation: at the first disagreement
   a branch stops unless the alternative's score also clears the threshold,
   in which case the tree splits there.
5. The row budget comes from the controller's marginal cost of a row. On
   this 288-expert top-8 MoE a row costs expert bytes at B1 but almost none
   at B64, where the expert union is already near saturation (171-240 of
   288), so trees stay narrow at B1 and widen at B64.

## 4. Drafters per model

"Candidates" are measured against each other on the ledger corpora; the
winner per model and content class is recorded in the model's hill-climb
log. Licence status comes from the speculator licence admission record and
must be checked before any serving use.

| Model (module) | Drafters available | Notes and first measurement |
| --- | --- | --- |
| GLM-5.3 Flash (`glm5_next`) | MTP layer 45 (in checkpoint); suffix/n-gram; DFlash2 on the rtx5090 | DFlash2 is CC BY-NC-ND and rejected for serving (evaluation only if the owner allows). First: suffix, then MTP on draftd. Published MTP tau 3.7-5.1 at 7 drafts |
| DeepSeek V4.1-Flash (`dsv41_flash`) | DSpark block in the pack (3 blocks x 5, Markov rank 256); suffix | DFlash on the rtx5090 targets V4 Flash, not V4.1: check before use |
| MiMo-V2.6 Flash / Pro (`mimo26`) | MTP head (3 layers, F32 storage); `dflash/` 5.5 GB draft model in the checkpoint; suffix | measure MTP depth 1-3 against the bundled DFlash |
| Kimi K3 (`k3`) | DSpark drafter (format in module); DFlash (modal) and DFlash2 (lightseek) on the rtx5090; suffix | K3 at 100+ tok/s B1 needs speculation: the no-spec ceiling is about 65 tok/s *model* |
| GLM-5.3 Full (`glm52` module) | DSpark draft backend module; MTP tree path; suffix | re-qualify on the GLM-5.3 Full checkpoint |
| Hy4 (`hy4`) | suffix only: the model header packs no MTP layer | check the checkpoint for an MTP head before training a drafter |
| Ling-3.0-flash and finance fine-tune (`ling`) | MTP layer 42 (MLA + MoE); suffix | the fine-tune needs its own acceptance numbers |
| Laguna S 2.1 (`laguna`) | suffix only: `mtp_layer` is null | trained drafter only if suffix leaves a gap worth it |
| DeepSeek V4-Pro-0813 (`dsv4`) | 3 MTP/DSpark layers; suffix | last in the roster |
| Qwen3.8-27B (`qwen38_27b`, internal only) | MTP; DFlash2 (incoai) on the rtx5090; DSpark format; suffix | internal use only, never on the external API |
| Gemma 4, Muse Glimmer (extras) | suffix only (Muse Glimmer lists no MTP layers) | when a lane frees |

## 5. PR sequence

Each PR lists its gate. Fleet gates run on pinned PR builds in parallel
with CI (I33, I34); a merge is not the gate.

- **S1, verify-regime hooks (this PR).**
  - `model-families/glm5_next/include/sparkpipe/spark_glm5_next_verify_regime.h`:
    the single-sequence verify regime keyed by row count (2..8) and attention
    regime; admission is exactly "one lane, consecutive positions, greedy,
    first row, all rows in one attention regime, inside the sequence
    capacity"; capture bounds per regime.
  - Module: `SPARK_GLM5_NEXT_VERIFY_ROWS` (absent or 0 is off; 2..8 sets the
    largest verify wave; anything else fails startup). A nonzero value needs
    the graph path, TP>=2 single-stage lazy pack, pinned experts, no MTP and
    `execution_row_capacity >= 8`, or startup fails with a named status.
  - Capture at engine start: on each slot's first chain after the experts are
    warm (the earliest point a graph can be captured), the UNSPLIT and SPLIT
    verify graphs for every configured row count are captured in one pass
    into a table separate from the production graphs; the SELECTED row set
    is captured together when a verify wave first needs a larger bound, as
    the production SELECTED graphs are. A capture failure is a terminal
    engine failure, never an eager fallback.
  - Verify routing: a verify wave with the regime on replays its table entry;
    an inadmissible wave fails the request with a named status. The device
    accept and state fold are S2: until then a replayed verify wave fails
    with `UNSUPPORTED` (`GRAPH-VERIFY-COMMIT-MISSING`). No production code
    produces verify waves at TP>=2 yet, so serving is unchanged with the
    regime on or off.
  - Common test drafters: `SparkSpeculationReferenceDraftTokens` with an
    ORACLE mode (replays a recorded greedy continuation) and an ADVERSARY
    mode (every draft wrong at depth 1), usable as any policy draft function.
  - Gate: host tests (`tests/test_glm5_next_verify_regime.py`,
    `build/test_speculation_reference_draft`); on the fleet, the regime off is
    byte-identical and time-identical to main, and the regime on captures 14
    graphs per slot at the first warm chain with production B1/B8 unchanged.
- **S2, exact verify round.** Device accept kernel; KDA replay layout sized
  for 8 rows without MTP; fold, conv re-commit and index-pool fix-up reading
  the accepted count from device memory; remove the uncapturable host-stack
  memcpy of `committed_steps`; oracle and adversary wired through the
  adapter from a recorded continuation file. Gates: G-ROWEQ (per-layer
  hidden hashes and logits of an n-row verify bitwise equal to n B1 steps,
  n=2..8, contexts 63, 64, 2048, 2049, 4099, 32K, TP16); G-ORACLE,
  G-ADVERSARY, G-RANDOM and G-POISON byte-identical on COMPSEC-17 x 512
  tokens x 3 seeds; oracle k=3 at least 100 tok/s *model 114*.
- **S3, absolute-position attention tiling** (shared with lane L1).
  Split-KV and the 2048-entry selection decided per key position, which lets
  verify waves straddle regime boundaries. Gate: G-ROWEQ across boundaries.
- **S4, controller and frame ABI bump 1.** `src/spark_speculation_controller.c`,
  counters in the wave timing, heartbeat and completion,
  `MAX_TOKENS_PER_SEQUENCE` 8 -> 32 in one coordinated PR for all adapters.
  Gate: table-driven controller cases; k=0 whenever a row costs more than
  it commits; every adapter builds and loads.
- **S5, relay and suffix drafter.** Rank-15 busy-poll relay, fixed frames,
  mesh fan-out from root 15, deadline and counted k=0 rounds, `draftd` on
  the rtx5090 with the suffix/n-gram drafter first (no ruling or licence
  needed). Gates: byte-identical output with the relay on, off and with
  draftd killed mid-run; at least +10% on a code/agentic corpus and at
  least 0.98x on chat; RTT p50/p99 and miss rate in the ledger.
- **S6, MTP on draftd.** GLM layer 45 on the rtx5090 (TP1, sm_120), hidden
  tap ring on rank 15, per-sequence MTP KV filled at prefill, pre-drafting.
  Gates: draft top-1 agrees with the reference MTP on at least 99% of 1000
  positions; B1 at least 65 tok/s in-chain *model 77*; byte-identical.
- **S7, trees.** Tree verify graphs (ancestor mask, per-node DSA selection,
  KDA tree scan), multi-drafter trie with extension and truncation, device
  tree accept. Gate: beats the S6 chain by at least 3% on the ledger
  corpora with byte-identical output.
- **S8, B>1 (frame ABI bump 2).** Per-lane committed counts, multi-sequence
  verify waves keyed by (batch bucket, rows), controller row allocation
  across sequences. Gates: at least 0.98x of k=0 at every B in {2, 4, 8, 16,
  32, 64}; byte-identical at every B; the section 6 targets.
- **S9, exact sampled speculation.** Keyed Gumbel noise in drafter and
  verifier. Gate: seeded sampled streams byte-identical with and without
  speculation, 17 prompts x 3 seeds.
- **S10, per-model enablement.** Each driver agent adds its drafters from
  the table in section 4 through the same draftd, relay and verify regime;
  only the geometry, draft function and state fold are model-specific.

## 6. What 4x at B8 and B64 requires

Aggregate gain at batch B with k drafts per sequence is
`E[committed per sequence] x T(B) / T(B x (k+1))`.

- With tau = E[committed] of 4-5 (MTP chains at alpha 0.8-0.86 *model*, more
  with agreeing trees), 4x needs `T(B(k+1)) / T(B)` near 1.0-1.25.
- Today it is not: B8 at 51.7 ms against B1 at 25.5 ms, and B32 at 203 ms
  before the skinny grouped experts, put B8 at 0 to +20% *model* and B16-64
  at k=0.
- It becomes reachable when the grouped-expert and dense kernels are memory
  bound at 64-512 rows (the expert union is already saturated there, so extra
  rows add almost no bytes) and the collectives are GPU-initiated (lane L2).
  The controller then picks wide trees at B64 on its own; nothing in this
  plan hard-codes a batch policy.
- Until then the speculation lane reports the measured multiple at B1, B8
  and B64 per content class next to the non-speculative number, and the gap
  to 4x goes to the kernel and collective lanes as a row-cost target.

## 7. Open questions for the owner

1. VRAM budget for draftd on the rtx5090 next to the hub, `g53-api` and
   training (the GLM layer-45 MTP is about 9 GB at FP8 *model*).
2. DFlash2 for GLM-5.3 Flash: evaluation-only use in the lab, or not at all?
3. `speculation.required` default for releases: named failure, or counted
   k=0 rounds with a heartbeat flag?
4. Timing of the two frame ABI bumps relative to the driver agents' rebuilds.

## 8. Fleet steps for S1

1. Build the S1 branch at a pinned SHA into a separate test root on all 16
   Sparks (I33) and deploy it with the current serving settings and without
   `SPARK_GLM5_NEXT_VERIFY_ROWS`. Check `GLM verify regime rows=0` in every
   rank log, then B1 128/512 tokens, the B8 8-stream sweep and COMPSEC-17:
   byte-identical text and ms/token within noise of main.
2. Redeploy the same build with `SPARK_GLM5_NEXT_VERIFY_ROWS=8`. Check 14
   `GRAPH-VERIFY-CAPTURE ... status=0` lines and one `GRAPH-VERIFY-TABLE ...
   status=0` line per slot per rank after the first warm chain, record
   `capture_ms` and the device memory delta, and rerun step 1's checks:
   identical output and timing after the one-time capture.
3. Start one engine with `SPARK_GLM5_NEXT_VERIFY_ROWS=9` and one with the
   regime on and `SPARK_GLM5_NEXT_GRAPH_PATH=0`: both must fail startup with
   the named message.
