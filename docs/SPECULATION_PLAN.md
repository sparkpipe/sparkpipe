# Speculation plan

Status: plan of record for the speculation lane, 2026-09-28. S1
(`spec/verify-hooks-1`) and the first half of S2 (host accept and eager fold
after the verify graph, `spec/verify-fold-2`, `spec/lookup-draft-3`,
`spec/verify-commit-4`) are implemented; section 9 says what runs today and
how to measure it. S6a, the checkpoint's MTP layer as an in-engine drafter
at TP16 (`spec/glm-mtp-drafter`), is implemented and exact at TP1 on the
GPU; section 10 is its production A/B plan and section 11 the path from
MTP-only to about 200 tok/s. Everything else is planned work. Numbers marked *model* are estimates from the fleet-calibrated
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
- Greedy speculation is exact only when every verify row reproduces the
  B1 step bit for bit, head included (G-ROWEQ). The acceptance rule does
  not make that true on its own; section 9.2 lists what is proven. Sampled
  speculation uses
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
| GLM-5.3 Flash (`glm5_next`) | MTP layer 45 (in the checkpoint, not in the production pack: served from a per-rank sidecar pack, section 9.5); suffix/n-gram; DFlash2 on the rtx5090 | DFlash2 is CC BY-NC-ND and rejected for serving (evaluation only if the owner allows). GB10 measurements chose MTP over DFlash2. First: MTP in-engine (S6a) and lookup; then MTP on draftd. Published MTP tau 3.7-5.1 at 7 drafts |
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

## 9. What runs today (S2a) and how to measure it

### 9.1 Behaviour

- `SPARK_GLM5_NEXT_VERIFY_ROWS=2..8` needs `SPARK_GLM5_NEXT_VERIFY_DRAFTER`:
  `lookup` (the prompt-lookup drafter), `oracle:PATH` or `adversary:PATH`
  (a recorded greedy sequence, little-endian uint32 token ids from position
  0). A drafter without the regime, the regime without a drafter, or any
  other value fails startup with a named message.
- Drafter interface: any `SparkSpeculationDraftFunction` (context, request
  with lane, sequence id and anchor position, result with up to k token ids).
  The module holds one; `lookup`, `oracle` and `adversary` are the three
  built in. The rtx5090 relay drafter (S5) plugs in behind the same call.
- A B1 greedy decode frame of S >= 2 tokens (the runtime's resident chain
  frames, S <= 8, never crossing a 64-token block) asks the drafter for
  k = min(rows - 1, S - 1) tokens, reduced so the wave stays in one
  attention regime. No draft: the frame runs as the ordinary S-step chain.
  Otherwise the frame produces exactly S tokens from verify rounds and,
  where the drafter has nothing (or one token is left), plain B1 steps on
  the production B1 graph; after each plain step the drafter is asked
  again. A verify round:
  1. replay the captured verify graph for 1 + k rows with `commit=0`
     (KDA replay record on, conv windows and recurrent state untouched);
  2. sync, resolve the longest matching prefix on the host
     (`SparkSpeculationPolicyResolveVerifierTokens`): a accepted drafts plus
     the bonus token;
  3. `SparkGlm5NextLaunchCudaReplayFold(wave, a + 1)` folds exactly the
     committed rows into the KDA state and re-commits the conv windows;
     rejected rows' KV and index slots are overwritten by the next round;
  4. the drafter observes the committed tokens and the next round starts
     from the new anchor.
  The depth of each round is also capped per lane by
  `SparkSpeculationDepthCapNext`: a round whose drafts are all accepted
  doubles the cap (up to rows - 1), a rejection sets it to the accepted
  length plus one, and a new sequence on the lane starts at rows - 1. A
  drafter that keeps missing therefore costs 2-row waves, not 8-row ones,
  until it hits again. The cap only reads committed counts, so it is
  identical on every rank. The goodput controller of S4 replaces it.
  The completion carries the S tokens like a chain frame, so spec-on and
  spec-off runs have the same frame count. Every frame prints
  `VERIFY-FRAME slot position budget produced rounds accepted steps |
  cumulative` (steps are the plain B1 steps inside the frame).
- Rank agreement: the frame, the drafter history (built from frames and
  all-reduced outputs), the experts-warm flag and the verify-table flag
  flip on the same frame on every rank. Graph state is not like that: a
  rank can lose the graph path on its own (a stuck or failed graph, a
  failed arm, a failed B1 capture). Plain decoding tolerates that because
  graph and eager steps issue the same collectives, but a verify wave and
  S plain steps do not. So a frame is classified by
  `SparkGlm5NextVerifyFrameClass`: frame-level reasons (shape, sampling,
  cold, no draft) give a plain frame on every rank, counted per class and
  logged as `VERIFY-PLAIN-FRAME` at powers of two; rank-local graph state
  on a frame that would otherwise verify is `VERIFY-RANK-LOCAL-INELIGIBLE`
  and a terminal engine failure on that rank (the others then fail at the
  collective timeout), never a silent plain frame. A remote drafter will
  need the root-15 broadcast of section 2.2 instead.
- Head: B1 greedy uses the certified FP8 head (screen, exact BF16 rescore,
  lowest id on ties). Verify rows with a certified head loaded run the same
  certified B1 head once per row, so the head of a verify row is the B1
  head by construction. Cost: one certified head per row (about 0.38 ms on
  the head-owning rank per row at TP16's vocabulary shard). The exact rows
  kernel of #1293 replaces this with one launch (about 0.42 ms for 8 rows).
- Not yet: the device-side accept and in-graph fold (S2b, one graph per
  round instead of graph + host resolve + eager fold), a cost-aware k
  controller (S4), rounds across a 64-token block or frame, B > 1, sampled
  rows.

### 9.2 Exactness evidence

- `validate_mtp_parity` (sm_121a, synthetic 4-layer stack at full
  geometry, TP1): besides the MTP chain it runs verify rounds of 2..8 rows
  with the reference drafter in oracle, adversary and fault-at-depth modes
  on a short walk (28 rounds) and on a walk that prefills 2040 positions and
  decodes across context 2048 into the selected-attention regime (14
  rounds, waves trimmed to one regime as the engine admits them), each with
  the wave's natural maximum context and with the captured-graph bound. It
  checks, every round, the accepted count, the committed tokens against the
  serial greedy stream, and the KDA state, conv windows, KV and index bytes
  against the serial decode. PASS on sparkf 2026-09-28.
- Since the review fixes the harness runs what production runs: the
  certified FP8 head is loaded, so B1 baseline steps take the certified
  B1 head and verify rows take it once per row; the split threshold is the
  deployed 64, so contexts 64..2048 run the SPLIT regime; and a final pass
  twins every lm_head row, so every argmax is an exact tie and a row that
  resolved ties differently from B1 fails. All of this is TP1 with
  index_cp_degree 1. It does not prove TP16.
- `tests/test_glm5_next_stage_context.py` drives the real module verify
  loop (plan, resolve, fold count, observe, completion fields) with the
  three drafters against a toy target: output equals the greedy stream,
  and every fold receives exactly accepted + 1 rows (the count is
  recomputed by the test from the drafts and the verifier outputs, so a
  fold of all wave rows fails). A two-rank check gives one rank a failed B1
  capture, a disabled graph or a lost graph path and requires that rank to
  fail terminally while the healthy rank verifies.
- Missing and required before a speed claim: G-ROWEQ at TP16 on the graph
  path (a verify row must equal the B1 step bit for bit) and the fleet
  oracle/adversary runs of 9.3.

### 9.3 Fleet steps (own root, weightd lane 8, after RELEASE_DONE)

`tools/spec_verify_bench.py` has no default endpoint (production g53-api
is :8433; the dev API must be named explicitly), requires token ids in
every response and stream event (text alone never counts as exact), and
reports decode tok/s from the first streamed token to the last, with TTFT
separate.

1. Spec off, same build: `SPARK_GLM5_NEXT_VERIFY_ROWS` unset. Record the
   baseline: `tools/spec_verify_bench.py run --endpoint DEV --label off
   --out off.json` (prose, code, repetitive; 512 tokens, temperature 0) and
   COMPSEC-17 with `tools/glm5_next_compsec17.py`.
2. Record an oracle sequence: `tools/spec_verify_bench.py record
   --endpoint DEV --prompt-ids p.json --max-tokens 512 --out oracle.u32`.
3. Restart with `SPARK_GLM5_NEXT_VERIFY_ROWS=8
   SPARK_GLM5_NEXT_VERIFY_DRAFTER=oracle:oracle.u32` and replay:
   `tools/spec_verify_bench.py replay --endpoint DEV --prompt-ids p.json
   --expect oracle.u32` must print `"exact": true` (exit 0), and the rank
   logs must show `produced=8 rounds=1 accepted=7` on (almost) every
   frame. This is the upper bound of the round machinery (tau = 8 per
   frame) and the TP16 G-ROWEQ check.
4. Same with `adversary:oracle.u32`: `replay` exact, `accepted=0`.
5. Restart with `SPARK_GLM5_NEXT_VERIFY_DRAFTER=lookup` and run step 1's
   corpus: `run --endpoint DEV --label lookup --out lookup.json`, then
   `compare off.json lookup.json` (must print `"exact": true`) and `log
   <rank log>` for tokens per round and frame fill. Report decode tok/s
   spec-off and spec-on per content class separately, with the method
   (`lookup`, rows=8) and the acceptance length.

### 9.4 rtx5090 drafter over the sparkf relay: stub

Status: not wired. No engine or module code calls the relay drafter and
`SPARK_GLM5_NEXT_VERIFY_DRAFTER` cannot select it; it is a host-tested
building block (invariant I01: a stub does not establish the feature).

`include/sparkpipe/spark_speculation_relay_draft.h` is the rank-15 end of
section 2.2, reduced to what can be tested on a host:

- **Frames.** One fixed 112-byte little-endian frame for both directions:
  magic `SPR1`, version and kind (REQUEST or DRAFT), engine generation,
  round id, sequence id, anchor position, anchor token, token count,
  requested count, 16 token slots (unused slots must be zero). A REQUEST
  carries the tokens committed since the previous round (ending at the
  anchor), so draftd keeps its per-sequence history without a separate TAP
  frame for tap-free drafters; the prompt reaches draftd once at prefill.
  A DRAFT carries up to the requested number of draft tokens.
- **Mailbox drafter.** `SparkSpeculationRelayDraftIssue` opens round r and
  encodes its REQUEST; the relay thread sends it and hands every received
  frame to `SparkSpeculationRelayDraftDeliver`, which keeps only a DRAFT for
  the current engine generation, round, sequence and anchor (anything else
  is counted `stale` and dropped). `SparkSpeculationRelayDraftTokens` is a
  `SparkSpeculationDraftFunction`: called at the round deadline it returns
  the delivered draft once (`used`) or nothing (`misses`, a k=0 round).
- **Test.** `build/test_speculation_relay_draft` pins the byte layout,
  rejects malformed frames, checks stale/late/foreign drafts, and runs a
  400-token greedy loop where an in-process draftd (the lookup drafter fed
  only by REQUEST frames) answers through encoded frames and every fifth
  draft arrives after the deadline: output equals the greedy stream and the
  late rounds are counted misses.
- **Still to build (S5).** The rank-15 relay thread (busy-polled socket on
  the sparkf-rtx5090 link, pinned core), the root-15 mesh broadcast of each
  round's draft ids so all 16 ranks run the same verify shape (the module's
  draft function on ranks 0-14 then reads the broadcast, not the socket),
  the prefill history upload, and the draftd process on the rtx5090 (lookup
  first, then GLM layer-45 MTP at TP1). Until the broadcast exists this
  drafter must not be selected at TP>1: a rank-local deadline would let
  ranks disagree on k.

### 9.5 MTP drafter in the engine (S6a)

Status: implemented on `spec/glm-mtp-drafter`, exact at TP1 on the GPU,
not run at TP16. Off by default.

**Selection.** `SPARK_GLM5_NEXT_VERIFY_ROWS=2..8` with
`SPARK_GLM5_NEXT_VERIFY_DRAFTER=mtp` or `mtp+lookup`, and
`SPARK_GLM5_NEXT_VERIFY_MTP_DIR=<directory>` (absolute, or relative to the
root, which is the engine's working directory). Startup fails with a named
message when the directory is unset, the rank's pack is missing or invalid,
or the main pack itself carries layer-45 tensors.

**Weights.** The production pack has no MTP layer: rank 15's
`glm53flash.fp8.tp16.rankf.sp` has header flags 0, 1160 tensors for layers
0-44 and receipt `"mtp": "none"` (read on sparkf, 2026-09-28). Rebuilding the
21.7 GB rank packs and re-registering weightd lane 0 is not needed. The
drafter loads a per-rank sidecar pack instead:

- name `glm5_next_mtp.tp16.rank<R>.g5nsp`, header flags `MTP`, layer span
  45..45, TP degree and rank in the header;
- 27 tensors: DSA attention and indexer, router and correction bias, 288
  FP8 experts, shared expert, `eh_proj`, `enorm`, `hnorm` and
  `shared_head.norm`; embedding and `lm_head` are the main pack's;
- 616,611,456 payload bytes per rank at TP16 (7.8 GB at TP1);
- built by `tools/glm5_next_resident_stagepack.py --mtp-only --tp-all 16`
  (the dry plan against `/mnt/model-warm/glm-5.3-flash` plans all 16 ranks);
- loaded at init into one device allocation of 616,611,584 bytes, not through
  weightd, so the lane 0 arena and its pack sha stay as they are.

**Hidden tap.** The MTP layer reads the target's final hidden row (the HC
mean before the final norm) of the position whose argmax is the anchor. The
engine copies it into a per-lane buffer after every verify round (row
`committed - 1`), after every plain step inside a verify frame (row 0) and at
the end of every other frame (the last row of a B1 decode frame, or of a
single-wave, single-sequence prefill). Each copy is tagged with the sequence
id and the next position. The drafter drafts only when the tag equals the
round's anchor; otherwise it returns no draft and counts it as `cold`, which
happens once per sequence when the prefill spans several waves.

**Draft.** A chain of up to 7 tokens (`MTP_CHAIN_MAX`) through layer 45:
vocab-sharded embedding, `enorm`/`hnorm`, the replicated `eh_proj`, DSA
attention on this rank's heads, the MoE on this rank's slice of every expert,
and the shared head on this rank's vocabulary shard. Per drafted token that
is three bf16 all-reduces and one max all-reduce, issued as stream-ordered
collectives (the `hardware` wait mode production runs; anything else fails
the draft with `VERIFY-MTP-UNSUPPORTED`). Tokens feed the next step on the
device; the host syncs once per draft and checks the deferred collective
rounds. Every rank computes the same chain from the same all-reduced rows, so
the verify shape agrees across ranks without a broadcast. A token outside
the vocabulary truncates the draft (`truncated`).

**Mix (`mtp+lookup`).** The model-neutral
`SparkSpeculationDrafterMix` asks the lookup drafter first and uses it when it
proposes at least 2 tokens (or the whole requested depth when that is
smaller); otherwise it asks MTP. Each source keeps its own per-lane depth cap
(`SparkSpeculationDepthCapNext`), the lookup cap never below 2, so a lookup
miss does not shrink MTP rounds.

**Counters.** After every `VERIFY-FRAME` line the engine prints
`VERIFY-MTP drafts= tokens= cold= truncated= taps= draft_us= | lookup rounds= proposed= accepted= declined= | mtp rounds= proposed= accepted=`
(cumulative). `tools/spec_verify_bench.py log` turns the last one into
acceptance, accepted length and tokens per round per drafter.

**Limits of this step.**

1. The MTP attention is context-free. Its KV is one page per slot holding
   only the chain's own positions, not the MTP-layer KV of the whole
   sequence, so acceptance will be below published MTP numbers. Per-sequence
   MTP KV, filled from the prompt's hidden rows at prefill and from every
   committed row, is S6b.
2. The draft head is the bf16 `lm_head` shard, not the FP8 certified screen.
3. Drafting is on the critical path. The cost is not measured; the estimate
   in section 11 uses 1.3 ms per drafted token plus 0.3 ms per draft at TP16.
4. Nothing is measured at TP16 yet; G-ROWEQ at TP16 is still the gate.

**Exactness evidence.**

- `validate_mtp_parity` on sparkf (sm_121a, TP1, synthetic full-geometry
  stack), PASS 2026-09-28 on `5d1308b`: verify rounds of 2..8 rows drafted by
  the MTP chain at depth `rows - 1`, organically, with a reference prefix
  spliced in (at least the splice must be accepted), and drafted twice with
  counting TP reductions (identical chain, 3 row reductions and 1 max
  reduction per token). Short walk with and without the capture bound, and
  the 2040-position walk across context 2048. Every round: tokens, KDA state,
  conv windows, KV and index byte-equal to the serial greedy decode. The
  earlier phases (MTP chain frames, reference rounds, tied head) pass with
  the device-fed chain. Organic MTP agreement is 0 of 44 positions on
  synthetic weights, as expected.
- `tests/test_glm5_next_stage_context.py` (also under ASan/UBSan) drives the
  real module loop with `mtp` at TP1 and through the TP op path at TP2, and
  `mtp+lookup` at TP1 and TP16: output equals the greedy stream; the tapped
  row is `committed - 1` after a round and 0 after a plain step; the tag
  equals the next anchor; `cold` happens only on the first frame; an invalid
  token truncates the draft; four reductions per drafted token and one
  deferred check per draft. It also checks the sidecar validation (27 kinds
  at TP16; flags, rank, layer, codec, duplicate, geometry and overlap
  rejections) and a full load from a sparse 617 MB file.
- `build/test_speculation_drafter_mix`: routing, per-source caps and a
  900-token greedy loop where the mix equals the greedy stream and every
  round is attributed to one source.

## 10. Production A/B plan at TP16 (after the lead releases main)

This runs on the production root, so every step below is the lead's. Each
arm is one environment change, one engine restart on all 16 nodes (weightd
stays warm) and one perf window (`perf_window.py <lane> 15 -- ...`). Spec-off
and spec-on results are reported separately, per content class.

### 10.1 Once, before the first arm

1. Release the merged main through the normal path (FLEET_RELEASE_RUNBOOK
   section 4) and rebuild the hub API channel from the same SHA (section 6).
   With no `SPARK_GLM5_NEXT_VERIFY_*` variable the MTP and mix code is inert.
2. Build the 16 MTP sidecar packs in one Ceph window (about 10 GB written on
   sparkf; the checkpoint's layer 45 lives in shards 1 and 2):

   ```sh
   python3 /Users/mac/sparkpipe-coord/tools/ceph_window.py glm-spec 60 -- \
     ssh sparkf 'cd ~/build-spec/glmspec && python3 tools/glm5_next_resident_stagepack.py \
       --source /mnt/model-warm/glm-5.3-flash --output-dir $HOME/sparkdata/glm53flash-mtp.tp16 \
       --mtp-only --tp-all 16 && cd $HOME/sparkdata/glm53flash-mtp.tp16 && sha256sum *.g5nsp > SHA256SUMS'
   ```

3. Place rank R's file on node R and register it:

   ```sh
   hosts=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
   for r in $(seq 0 15); do h=${hosts[$r]}; f=glm5_next_mtp.tp16.rank$r.g5nsp
     ssh $h 'mkdir -p ~/sparkdata/glm53flash.fp8.tp16/packs/mtp'
     ssh sparkf "scp ~/sparkdata/glm53flash-mtp.tp16/$f $h:sparkdata/glm53flash.fp8.tp16/packs/mtp/$f"
     want=$(ssh sparkf "grep ' $f\$' ~/sparkdata/glm53flash-mtp.tp16/SHA256SUMS | cut -c1-64")
     ssh $h "cd ~/sparkdata/glm53flash.fp8.tp16/packs/mtp && [ \"\$(sha256sum < $f | cut -c1-64)\" = $want ] && echo $h OK &&
       printf '%s\t%s\t%s\t%s\n' \$HOME/sparkdata/glm53flash.fp8.tp16/packs/mtp glm-spec KEEP 'GLM MTP sidecar pack (617 MB), speculation A/B' >> ~/KEEP"
   done
   ```

4. Move the production root's engine environment into `agent.env`. The GLM
   root is a legacy root today: its engine gets `SPARK_GLM5_NEXT_*` only from
   the agent's `G5_*` knobs, and changing the unit drop-in restarts
   fleet-agent, weightd and the engine. A root with `agent.env` gets exactly
   the file's keys (plus the agent's inherited environment), and a release
   that changes `agent.env` restarts only the engine. The spec-off file
   reproduces today's engine environment:

   ```sh
   R=~/release/glm53flash.fp8.tp16
   cat > $R/agent.env.new <<'ENV'
   AGENT_ROLE=production
   SPARK_WEIGHTD_EXPERT_POOL_BYTES=34359738368
   SPARK_GLM5_NEXT_PIN_EXPERTS=1
   SPARK_GLM5_NEXT_GRAPH_PATH=1
   SPARK_TP_WAIT_MODE=hardware
   ENV
   mv -f $R/agent.env.new $R/agent.env
   cd $R && find lib bin stages config model_resident.json agent.env -type f ! -name stage.json ! -name MANIFEST |
       sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST
   ```

   `agent.env` must be in the MANIFEST file list from now on; the runbook's
   `find` omits it. Check that all 16 engines restart and report ready, that
   the heartbeat `env` sha matches on all 16, and that the engine environment
   matches the table in runbook section 2.1:

   ```sh
   for h in "${hosts[@]}"; do ssh $h 'rr=$HOME/sparkdata/glm53flash.fp8.tp16; p=$(pgrep -f "bin/sparkpipe_model_[r]esidentd" | while read p; do [ "$(readlink /proc/$p/cwd)" = "$rr" ] && echo $p; done)
     echo $(hostname) ready=$(grep -c "model_residentd ready" $rr/residentd.log) $(tr "\0" "\n" < /proc/$p/environ | grep -E "^SPARK_(GLM5_NEXT|TP_WAIT|WEIGHTD_EXPERT)" | sort | tr "\n" " ")'; done
   ```

5. Tokenize one code prompt for the oracle arm and record the oracle on the
   hub, in the build checkout of the released SHA:

   ```sh
   python3 - <<'PY'
   import json, importlib.util
   from tokenizers import Tokenizer
   spec = importlib.util.spec_from_file_location("b", "tools/spec_verify_bench.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
   t = Tokenizer.from_file("/home/spec/g53-api-channel/runtime/tokenizer/tokenizer.json")
   json.dump(t.encode(b.GLM_PREFIX + b.PROMPTS["code"][0] + b.GLM_ASSISTANT, add_special_tokens=False).ids, open("spec_p.json", "w"))
   PY
   ```

### 10.2 Arms

Every arm: write `$R/agent.env` (spec-off lines of 10.1 step 4 plus the arm's
lines), rewrite the MANIFEST as in 10.1 step 4, wait for 16 × `model_residentd
ready`, then run the arm inside one perf window from the hub build checkout.
`OUT=~/spec-ab-<sha7>`.

| Arm | Extra `agent.env` lines | Run in the window | Pass condition |
| --- | --- | --- | --- |
| off | none | `tools/glm5_next_spec_ab.sh off $OUT`; `tools/spec_verify_bench.py record --endpoint http://127.0.0.1:8433 --prompt-ids spec_p.json --max-tokens 512 --out spec_oracle.u32`; COMPSEC-17 to `$OUT/compsec-off` | 36 tok/s class, COMPSEC 14/17 |
| oracle | `SPARK_GLM5_NEXT_VERIFY_ROWS=8`, `SPARK_GLM5_NEXT_VERIFY_DRAFTER=oracle:config/spec_oracle.u32` | `tools/spec_verify_bench.py replay --endpoint http://127.0.0.1:8433 --prompt-ids spec_p.json --expect spec_oracle.u32` | `"exact": true`; rank log `produced=8 rounds=1 accepted=7` on nearly every frame (G-ROWEQ at TP16 and the round-machinery ceiling) |
| adversary | as oracle with `adversary:config/spec_oracle.u32` | the same replay | `"exact": true`, `accepted=0` |
| mtp8 | `SPARK_GLM5_NEXT_VERIFY_ROWS=8`, `SPARK_GLM5_NEXT_VERIFY_DRAFTER=mtp`, `SPARK_GLM5_NEXT_VERIFY_MTP_DIR=packs/mtp` | `tools/glm5_next_spec_ab.sh mtp8 $OUT $OUT/off.json`; COMPSEC-17 to `$OUT/compsec-mtp8`, then `tools/glm5_next_compsec17.py --compare $OUT/compsec-off $OUT/compsec-mtp8` | script exit 0 (token ids equal to off per class); COMPSEC compare exit 0 |
| mtp4 | as mtp8 with `SPARK_GLM5_NEXT_VERIFY_ROWS=4` | `tools/glm5_next_spec_ab.sh mtp4 $OUT $OUT/off.json` | exit 0 |
| mix8 | as mtp8 with `SPARK_GLM5_NEXT_VERIFY_DRAFTER=mtp+lookup` | `tools/glm5_next_spec_ab.sh mix8 $OUT $OUT/off.json` | exit 0 |

The oracle file reaches every node through the release: copy
`spec_oracle.u32` to `$R/config/spec_oracle.u32` before the oracle arm's
MANIFEST rewrite (a `config/` file that is not a stage, env or rank file does
not restart anything by itself).

Expected ready lines per node, in `residentd.log`:

```
GLM execution mode=graph
GLM verify regime rows=8 drafter=4                       (drafter 1 lookup, 2 oracle, 3 adversary, 4 mtp, 5 mtp+lookup)
GLM verify MTP pack packs/mtp/glm5_next_mtp.tp16.rank<R>.g5nsp tensors=27 device_bytes=616611584 tp=16 rank=<R>
model_residentd ready
GRAPH-VERIFY-TABLE slot=<s> rows_max=8 status=0 capture_ms=<t>     (after the first warm chain)
VERIFY-FRAME ... and VERIFY-MTP ...                                  (per frame, once requests run)
```

Stop the arm and restore the spec-off `agent.env` on any of: a replay or
compare that is not exact, `VERIFY-RANK-LOCAL-INELIGIBLE`,
`VERIFY-MTP-DRAFT-FAILED`, `VERIFY-MTP-UNSUPPORTED`, `GRAPH-VERIFY-TABLE ...
status!=0`, `GRAPH-VERIFY-REJECTED`, or an engine that does not reach ready.

Record per arm: decode tok/s per class (`$OUT/<arm>.summary.json`), tokens
per round and acceptance per drafter (`$OUT/<arm>.acceptance.json`), mean
draft time (`draft_us`), plain frames by class (`VERIFY-PLAIN-FRAME`), and the
COMPSEC score. The table goes to the GLM hill-climb log with the method
(`mtp`, `mtp+lookup`, rows) next to the spec-off number.

### 10.3 Rollback

Publish the spec-off `agent.env` (engine restart, weightd stays warm). To
return the root to the legacy environment as well, remove `agent.env` from
`$R` and from the MANIFEST, delete `~/sparkdata/glm53flash.fp8.tp16/agent.env`
on all 16 nodes, and drain the root with the cwd-scoped TERM of runbook
section 4.3; the agent restarts it with the unit's `G5_*` knobs. The MTP packs
are inert without the drafter env; delete them and flip their `~/KEEP` lines
to `DELETE-OK` when the lane ends.

## 11. From MTP-only to about 200 tok/s

### 11.1 What the real streams say

`tools/glm5_next_spec_replay.c` replays recorded greedy streams through the
module's own lookup drafter, depth cap, drafter mix, verify-depth rules and
runtime frame sizing (at most 8 tokens, never across a 64-token block), with
a seeded synthetic drafter of per-token acceptance p standing in for MTP.
`tools/glm5_next_spec_estimate.py` prices the rounds: a verify of r rows
costs `B1 + (r - 1) x row`, plus a per-round host cost and the MTP draft
cost. The streams are the 9 prompts of `spec_verify_bench.py` (prose, code,
repetitive; 512 tokens, temperature 0) recorded from production g53-api
(`09fdad6`) on 2026-09-28 at 17:20Z, with the prompts tokenized by the
channel's tokenizer (11 tokens for the smoke prompt, as the API reports).
The decode rates of that run (9-32 tok/s) are not a baseline: mimo, ling and
hy4 lanes were using fleet GPUs at the time. Only the token ids are used.

Lookup is measured; MTP is a parameter (p = 0.5, 0.65, 0.8) until the fleet
logs `VERIFY-MTP`. Today's costs: B1 25.2 ms, 2.7 ms per extra verify row
(2.35 ms kernel from the single-GPU bench, B1 15.39 ms to B8 31.87 ms, plus
the 0.38 ms certified head per row), 1.0 ms per round, MTP 1.3 ms per drafted
token plus 0.3 ms per draft, frames of 8. Spec-off prices at 39.7 tok/s.

| Drafter, rows | code | prose | repetitive |
| --- | --- | --- | --- |
| lookup, 8 | 39.9 (1.01x, 1.45 tokens/round) | 39.7 (1.00x) | 39.6 (1.00x) |
| MTP p=0.5, 4 | 48.3 (1.22x) | 48.3 (1.22x) | 48.9 (1.23x) |
| MTP p=0.65, 4 | 56.0 (1.41x) | 56.3 (1.42x) | 56.3 (1.42x) |
| MTP p=0.8, 4 | 69.6 (1.75x) | 69.7 (1.76x) | 69.7 (1.76x) |
| MTP p=0.8, 8 | 69.3 (1.75x, 2.75 tokens/round) | 69.1 (1.74x) | 70.0 (1.76x) |
| MTP+lookup p=0.8, 8 | 67.5 (1.70x) | 68.5 (1.73x) | 67.7 (1.71x) |

Findings:

- Lookup gives nothing on these streams. The "repetitive" prompts produce
  number sequences (squares, `user_<n>`) whose next token never repeats a
  3-token context, and a frame whose first round has no draft runs all its
  steps plain (62 of 63 frames for prose[0]). Lookup stays in the mix for
  copy-heavy agentic and editing traffic, where it is free.
- With MTP at p = 0.8 the estimate is about 70 tok/s at today's costs, and
  rows 4 and 8 are equal: the extra rows cost as much as they commit.
- On today's row costs no drafter in this model passes about 75 tok/s at
  TP16; about 100 tok/s (what the owner saw at TP4 with speculation) needs
  the cheaper rows and faster B1 step of 11.2.

### 11.2 What moves the number

Same streams and drafters with frames of 32 (S4), 8 rows, and either the
acceptance depth cap of today or a fixed full depth (what a cost-aware
controller picks when rows are cheap). Tokens per round in brackets; the
three classes agree to within 2 tok/s, so one number is shown.

| Cost model | MTP p=0.65, capped / fixed | MTP p=0.8, capped / fixed | p=0.85, fixed | p=0.9, fixed |
| --- | --- | --- | --- | --- |
| today: B1 25.2 ms, row 2.7 ms, round 1.0 ms, MTP 1.3 ms/token + 0.3 ms/draft | 59 / 51 | 71 / 75 (3.95) | | |
| B1 22 ms (merged kernel stack), row 1.0 ms, round 0.3 ms, MTP 0.5 ms/token | 82 / 82 | 102 / 121 | 135 | 153 |
| B1 20 ms, row 0.3 ms, round 0.2 ms, drafts pre-drafted off the critical path | 101 / 118 | 131 / 173 | 195 (4.42) | 220 (4.97) |

About 200 tok/s therefore needs all of:

1. **Verify rows nearly free.** A verify of 8 rows must cost close to one B1
   step. Today a row costs about 2.7 ms. The grouped-expert and dense skinny
   kernels have to stay memory-bound across 8 rows (the expert union of 8
   consecutive rows is about 58 of 288), and the certified head must run as
   one rows launch (#1293) instead of once per row.
2. **A faster B1 step.** 25.2 ms today; the merged kernel stack took the
   single-GPU step from 15.7 to 12.85 ms. 20 ms at TP16 needs that plus
   collective work (lane L2).
3. **Higher MTP acceptance.** p near 0.85-0.9 per token; what the
   context-free chain reaches is unmeasured until the mtp8 arm. S6b gives layer 45 its per-sequence
   KV; S7 trees add independent drafters where MTP is unsure.
4. **Drafting off the critical path.** In-engine MTP costs k x (about 1.3
   ms) per round on the same GPUs. Pre-drafting on the rtx5090 (section 2.1)
   or an in-graph MTP chain removes most of it.
5. **Bigger frames and a cost-aware depth.** Frames of 8 cut rounds short;
   S4 raises `MAX_TOKENS_PER_SEQUENCE` to 32 and replaces the doubling depth
   cap with the goodput controller (p = 0.8: 2.8 tokens per round capped, 3.95
   fixed).
6. **Device accept and fold in the verify graph (S2b).** About 1 ms of host
   work per round today.

### 11.3 rtx5090 drafter over the sparkf relay: design and stub status

What exists (#1284, host-tested only, not selectable): fixed 112-byte `SPR1`
frames (REQUEST with the committed tokens since the last round, DRAFT with up
to 16 tokens), the mailbox drafter that keeps only a draft for the current
engine generation, round, sequence and anchor, and a 400-token greedy loop
against an in-process draftd with late drafts counted as misses.

What the rtx5090 drafter still needs, in order:

1. **Root-15 broadcast.** The relay thread on rank 15 receives the draft and
   broadcasts `(round, k, tokens)` over the mesh before every verify, so all
   16 ranks run the same shape; ranks 0-14 read the broadcast, never the
   socket. S6a does not need this because its drafts come out of
   all-reduces. This is the piece that makes any remote drafter
   rank-consistent, and it is the first S5 PR.
2. **Relay thread and deadline.** Busy-polled socket on the sparkf-rtx5090
   link (22-45 us RTT), deadline set by the controller, k=0 rounds counted.
3. **draftd process.** Lookup first (no tap), then MTP: layer 45 at TP1 on the
   rtx5090 (7.8 GB, fits next to `g53-api`), per-sequence MTP KV filled from
   `TAP` frames. A `TAP` frame carries 8 KB of hidden row per committed token,
   drained from a pinned ring written by a graph memcpy node on rank 15.
4. **Pre-drafting.** Draft round r+1 for the likely outcomes of round r while
   the fleet verifies; this is what takes drafting off the critical path.
5. **Trees (S7).** The rtx5090 builds a token trie from MTP, lookup/suffix and
   any licensed drafter; the fleet verifies it in one pass. The fleet side
   needs tree verify graphs (ancestor mask, per-node DSA selection, KDA tree
   scan and tree fold), the largest single item on this list.

The in-engine MTP drafter of S6a and the rtx5090 path are not exclusive: S6a
gives the first measured MTP acceptance and speed on the fleet with no new
transport, and its hidden tap, tags and draft interface are what draftd's MTP
reuses. Section 1 rules that drafters run on the rtx5090; running MTP on the
fleet first was the lead's direction on 2026-09-28 and is recorded here for
the owner.
