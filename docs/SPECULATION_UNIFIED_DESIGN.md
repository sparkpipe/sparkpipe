# Speculation subsystem

The one authority for how speculation is built and where it is going.
[README.md](../README.md#speculative-decoding) states the end state;
[TECHDEBT.md](../TECHDEBT.md#speculation) lists the open gaps. Code facts
were checked on 2026-09-28 against main `bbf5432`; line numbers are at that
commit. Superseded designs are in `docs/archive/`
(`SPECULATION_PROVIDER_DESIGN.md`, `SPECULATION_TREE_COMPOSITION_DESIGN.md`,
`SPECULATION_SUBSYSTEM_BOUNDARY.md`); their still-valid parts are below.

## Status

- The served GLM 5.3 Flash (glm5_next, TP16) does not speculate (invariant
  I42: current GLM targets are non-speculative). Its MTP path is TP1 only
  ([below](#glm5_next-mtp-research-path)).
- The RTX 5090 draft farm drafts for no served model. The DFT3 client
  exists; only qwen38_27b calls it. No farm process was running on the hub
  at 2026-09-28 09:00 UTC (`pgrep`).
- No verify frame checks a tree. Every served acceptance is a chain.

## Architecture as built

### Seam: the adapter's only speculation surface

`include/sparkpipe/spark_speculation_seam.h`, `src/spark_speculation_seam.c`
(#788, `d93374b`). One seam per adapter. It owns one policy-engine instance
and, when a remote source is enabled, a draft bridge
(`SparkSpeculationSeamInitialize`, seam.c:345-471).

Sources are bits of a mask (seam.h:14-29):

| Bit | Source | Kind |
| --- | --- | --- |
| `0x1` | MTP | local |
| `0x2` | DSPARK | local |
| `0x4` | DFLASH2 | remote, needs hidden-tap rows |
| `0x8` | NGRAM | remote, tap-free |
| `0x10` | SUFFIX | remote, tap-free |
| `0x20` | NGRAM3 | remote, tap-free |

Control is one env per family, `SPARK_<FAMILY>_SPECULATORS`:

- unset or `1` selects the family default (`default_source_mask`,
  seam.c:116-122);
- `0` selects nothing;
- anything else is parsed with `strtoull` base 0 (hex works). A bit outside
  the family's available mask is `SCHEMA_ERROR` (seam.c:43-81).

A remote bit requires `draft_bridge_host`/`draft_bridge_port` in the stage
config; adapters that accept a bridge drop the remote bits from their
available mask when the host is absent. DFLASH2 also requires
`max_tap_row_count > 0` and aux layers in the model contract
(seam.c:161-173).

Retired and rejected with `SCHEMA_ERROR`: `SPARK_DSV4_DSPARK` (dsv4
adapter.c:851-855), `SPARK_QWEN38_27B_SERVING_SPECULATE`,
`SPARK_QWEN38_27B_SERVING_SPEC_METHOD` and
`SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_COUNT` (qwen38_27b
adapter.c:502-520), `SPARK_K3_SERVING_SPECULATE` (k3 adapter.c:429-434).
`SPARK_GLM5_NEXT_MTP=0|1` remains as an alias for the MTP bit; setting it
together with `SPARK_GLM5_NEXT_SPECULATORS` is an error (glm5_next
adapter.c:72-116).

Lifecycle: `SparkSpeculationSeamStageLocalDraft` (a draft the model made),
`SparkSpeculationSeamDraftRemoteChain` (ask the bridge),
`SparkSpeculationSeamAcceptChain` (resolve the verifier's tokens, complete
the round), `SparkSpeculationSeamCancelSequence`. A lane is one sequence
slot (`lane_count` = the adapter's active-sequence capacity).

### Policy engine: the one acceptance accounting

`include/sparkpipe/spark_speculation_policy.h`,
`src/spark_speculation_policy.c`, model-neutral. Greedy only: accept the
longest matching path from the root, commit it plus one bonus token from
the verifier.

- `SparkSpeculationPolicyResolveVerifierTree` (policy.c:721) takes a parent
  array: at most 32 nodes, one verifier row per node plus the root row.
  `SparkSpeculationPolicyResolveVerifierTokens` (:834) is the chain case
  (no parent array).
- Production calls only the chain form: the seam's `AcceptChain`, dsv4
  module.c:3366 and glm5_next module.c:2370. The tree form is exercised
  only by `tests/test_speculation_tree_resolve.c`, which the Makefile does
  not build (it passed when compiled by hand on the rtx5090, 2026-09-28).
- Per-sequence state (`SparkSpeculationSequenceState`, policy.h:131-148)
  stores draft ids and confidences only, no parents.
- `SparkSpeculationModelContract` carries geometry, draft limits and the
  hidden-tap layer ids (`aux_layer_ids`); validation is structural, with no
  single-model constants (policy.c:6-52).

### Draft bridge: DFT3 client to a remote drafter

`include/sparkpipe/spark_draft_bridge.h`, `ring/transport/draft_bridge.c`
(#786, `16fb012`). TCP (`SOCK_STREAM`, `TCP_NODELAY`; draft_bridge.c:201,
256), magic `DFT3` (:20). `SparkDraftBridgeProposeTree` sends the committed
tokens, optional tap rows, a source mask, a depth of at most 8 per request
and a time budget. It returns scored nodes (`token_id`, `parent_index`,
`depth`, `source_bit`, `score`); a tree may be up to 32 deep and 4096 nodes
(bridge.h:26-31, 56-63). The seam keeps one path: the deepest node, ties
broken by score (`SparkSpeculationSeamExtractChain`, seam.c:202-270).

The server is not in this repository. It lives on the hub in
`~/draft_service` (`farm_server.py`, DFT3 wire format in
`draft_protocol.py`); see [RTX5090_SPECULATION_NODE.md](RTX5090_SPECULATION_NODE.md).

### Model side: verify and commit

The model's multi-row frame computes the verifier's tokens and the model
commits or rolls back its own state. glm5_next KDA state rolls back through
the ReplaySSM fold `LmReplayFoldKernel` (`inference/kernels/linear_attn.cuh:93`,
launched from `spark_glm5_next_resident_decode_stage_cuda.cu:803`). A model
supplies three things: a model contract, a draft (local) or tap rows
(remote), and its own state fold/rollback.

### Legacy pieces still in the tree

- Provider slot (`include/sparkpipe/spark_speculation_provider.h`,
  `runtime/speculation_provider.c`): only user is qwen38_max's MTP shim
  (qwen38_max adapter.c:244-279), which binds only when the MTP layer count
  is nonzero (:358). Its `SPEC_METHOD`/`DRAFT_COUNT` schema is descriptor
  data nothing reads.
- `include/sparkpipe/spark_speculation_tree.h` (static tree shapes): reached
  only through `spark_glm52_mtp_tree.h`, which only
  `tests/test_glm52_mtp_tree.c` and `tests/test_speculation_tree_pin.c`
  include.
- `inference/kernels/speculate.cuh` (greedy and sampled verify kernels):
  included by four unity files (common_glm_cuda_tree, glm5_next, laguna,
  ling), launched nowhere.
- `modules/glm52_dspark_draft_backend` (GLM 5.2 DSpark backend): no serving
  path drives it; the glm52 adapter offers no source and declares
  `max_speculative_token_count = 0` (glm52 adapter.c:179). GLM 5.2 weights
  are deprecated.
- The `SPARK_DSPARK_TARGET_*` switch in
  `model-families/common/include/sparkpipe/spark_dspark_drafter.h:7-65`
  (glm52, k3, dsv4-pro-0813). `Makefile:791` still passes
  `-DSPARK_DSPARK_TARGET_GLM52=1` to `spark_speculation_policy.o`, which no
  longer includes that header.

## Per-family state

| Family | Env | Available sources | Default | What runs |
| --- | --- | --- | --- | --- |
| dsv4 | `SPARK_DSV4_SPECULATORS` | DSPARK; tap-free remote with a bridge (adapter.c:196-200, 859-864) | all available | DSpark sets the module's `NODE_CONTEXT_FLAG_DSPARK` (:917); acceptance through the policy resolver (module.c:3366); no remote call site |
| glm5_next | `SPARK_GLM5_NEXT_SPECULATORS` / `SPARK_GLM5_NEXT_MTP` | MTP; tap-free remote with a bridge (adapter.c:62-64, 427-429) | none (:104, :440) | MTP at TP1 only; no remote call site; no DFLASH2 source since `f301e1c` |
| glm52 | `SPARK_GLM52_SPECULATORS` | none (adapter.c:351) | none | seam initialized, no source |
| k3 | `SPARK_K3_SPECULATORS` | none (adapter.c:438) | none | engine verify commit and `K3AdaptiveDepth` (`inference/llms/kimi_k3/spec_verify.h`) run only in `tests/host_cuda/k3_*` |
| qwen38_27b | `SPARK_QWEN38_27B_SPECULATORS` | MTP, DSPARK, DFLASH2; tap-free remote with a bridge (adapter.c:88-94, 564-566) | MTP (:568-569, :591) | one local method, or remote tap-free only; mixing or two local methods refused (:531, :542); the only remote consumer (:1549) |
| qwen38_max | `SPARK_QWEN38_MAX_SPECULATORS` | none (adapter.c:72) | none | provider-slot shim, above |
| qwen4_flash | `SPARK_QWEN4_FLASH_SPECULATORS` | none (adapter.c:58) | none | seam initialized, empty |
| gemma4, laguna, ling, minimax, muse_glimmer | none | none | none | no seam |

qwen38_27b DFlash2 cannot start. The adapter treats DFLASH2 as a local
method, but the seam classifies it as a remote tap source. Without a bridge
the adapter drops the bit (adapter.c:565-566) and `0x4` fails the parse
(seam.c:77); with a bridge the seam requires tap rows that the adapter
declares as `max_tap_row_count = 0` (seam.c:164-167). A probe on the rtx5090
(2026-09-28) that replayed the adapter's masks against the real seam got
`SCHEMA_ERROR` in both cases, while `0x1` and `0x2` initialized. No test
covers `0x4`.

## glm5_next MTP: research path

MTP chain speculation (#815, `a4f1263`) drafts through layer 45, draft depth
2 (`SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH`), gated by B1
greedy token parity (`validation/spark_glm5_next_resident_decode_stage_mtp_parity.cu`).
It is off by default and cannot serve the TP16 engine:

- TP1 only: init returns `UNSUPPORTED` when MTP is on and `tp_degree != 1`
  (`SparkGlm5NextInitializeState`, module.c:4842-4846), or `SCHEMA_ERROR`
  earlier if the pack has no layer-45 tensors (:4837-4841). The #815 commit
  message puts full-model TP1 at 271-316 GiB, which does not fit one Spark.
- MTP or speculative verify turns off the linear chain
  (`SparkGlm5NextLinearEligible`, :2855-2858) and graph replay
  (`chain->spec_verify == 0` in `SparkGlm5NextTpChainAdvance`, :3378), the
  paths behind the measured 36 tok/s B1 (lead-dev measurement, 2026-09-28).
- `SparkGlm5NextMtpDriveDraft` returns early unless the frame has one step
  and one active sequence (:2216-2220).
- The MTP layer's draft KV is one page per execution slot (TECHDEBT).
- `tools/fleet_serve.sh:96` still defaults `SPARK_GLM5_NEXT_MTP` to `1`,
  which fails init at TP16; the fleet-agent drop-in does not set it.

TP16 serving would need a TP-sharded MTP head, draft and verify rows inside
the graph chains, per-sequence draft KV, and a tree or chain verify that
keeps the graph path.

## Remote drafting on the RTX 5090

As built, only qwen38_27b drafts remotely, and synchronously: decode frame,
then a blocking `SparkSpeculationSeamDraftRemoteChain` with a 20 ms budget
(`SPARK_QWEN38_27B_SERVING_SEAM_DRAFT_TIME_BUDGET_MS`), then verify of the
returned chain. It sends no tap rows (adapter.c:1549-1551), so only NGRAM,
SUFFIX and NGRAM3 can answer. `tests/test_qwen38_27b_remote_spec.c` covers
this path against a stub server; the Makefile does not build it.

Latency budget: busy-polled 64 B UDP round trip sparkf to rtx5090 is 22.6 us
p50 / 25.2 us p99, and about 300 us p50 from other Sparks through sparkf
(lead-dev measurements, 2026-09-28). The bridge is TCP, so these are lower
bounds for DFT3. Arithmetic: a synchronous draft that uses its full 20 ms on
a 23-24.5 ms GLM B1 step makes the step 43-44.5 ms, so it must commit at
least 1.82-1.87 tokens per step just to break even, before verify-row cost.

Before the 5090 can draft for GLM: a `SparkSpeculationSeamDraftRemoteChain`
call site in the glm5_next adapter, tap capture inside the graph engine if a
DFlash2 drafter is used (TECHDEBT: glm5_next has no DFlash2 source), and
drafting pipelined one round ahead so the round trip hides behind the
forward.

## Guarantee and its limit

Verification pins every emitted token to the target model's verify-frame
output. That frame is multi-row, and batched serving is not batch-invariant:
dense projections switch kernels above eight rows and attention depends on
the wave (TECHDEBT, batch invariance items); COMPSEC concurrent runs differ
(lead-dev, 2026-09-28). A speculative run can therefore differ from a B1
non-speculative run, though not from the target's own multi-row output.
TensorFold's rule applies: compare multi-row with one-row output at load and
disable drafting when they differ (TECHDEBT). Adopt it as the gate for any
speculation that claims unchanged output.

## Direction: multi-drafter trees (not built)

The deliverable is tokens per second and latency on whichever topology wins;
TP16, PP16 and TP4xPP4 are to be hill-climbed (user direction, 2026-09-28).
Speculation therefore stays topology-neutral: verify rows are rows in
whatever wave layout runs.

Premise (user, 2026-09-28): at small batch, decode is bandwidth-bound, so
extra verify rows are nearly free until the batch reaches a compute ridge
B*. A multi-drafter tree with conditional truncation and extension spends
that spare compute. Target (user, 2026-09-28): approach the roughly 4x
reported for TensorFold, at B8 and if possible B64. That figure is
unverified; per I42, cite a TensorFold receipt before using it as a target.
B* is unmeasured on GB10 for every served model and topology: measure
verify-row cost against row count per topology, which sets the largest
batch at which speculation pays.

Why several drafters (from the 2026-08-30 addendum): a speculator cannot
improve quality, only acceptance. Naive best-of-N costs N verifies; a tree
built from the drafters' divergences verifies in one target pass with a
tree-attention mask. Drafters approximate the same target, so their errors
correlate and diversity beats count: the cross-drafter agreement matrix,
free to measure during serving, picks the best diverse pair, often not the
two individually best drafters. MTP is the natural trunk: near-free and
differently correlated from external drafters.

Gain model (arithmetic, i.i.d. per-position acceptance p, depth k, verify
rows free): expected accepted draft tokens E = p(1-p^k)/(1-p), plus one
bonus.

| p | E at k=4 | E at k=6 | E at k=8 |
| --- | --- | --- | --- |
| 0.70 | 1.77 | 2.06 | 2.20 |
| 0.80 | 2.36 | 2.95 | 3.33 |
| 0.85 | 2.71 | 3.53 | 4.12 |

Raising p from 0.7 to 0.8-0.85 adds 0.59 (k=4) to 1.47 (k=6) tokens per
step; the addendum's p range is an assumption, not a measurement. 4 tokens
per step (E = 3) needs p ≥ 0.805 at k=6 or p ≥ 0.776 at k=8.

Composition rules (carried over from the tree-composition design, re-based
on the seam and bridge):

1. The verifier row budget is hard. Composition chooses which nodes fill it,
   never grows it; the budget per step comes from B and the measured ridge.
2. Verify the tree instead of collapsing it: parents in the sequence state,
   a tree `AcceptChain`, and a tree-attention verify frame
   (`ResolveVerifierTree` already does the resolution).
3. Local and remote sources together (qwen38_27b refuses the mix today),
   merged by source priority and score into the row budget.
4. Acceptance attributed per `source_bit`, so each source has a measured
   rate that drives its budget and retirement.
5. Conditional truncation and extension: depth follows recent acceptance and
   draft confidence. Precedents: `K3AdaptiveDepth` (window 8, floor 4,
   ceiling 7; `spec_verify.h:9-13`) and the DFT3 server's time-budgeted tree
   growth.
6. Draft one round ahead; a late remote round contributes no nodes and never
   stalls the step.
7. Host pins before any cell runs, including: a composed tree resolves
   identically to the same tree resolved from one source.

| Phase | Work | State |
| --- | --- | --- |
| P1 | Single-drafter acceptance and the agreement matrix on real inputs | not started (`experiments/` holds only a README) |
| P2 | Offline replay: best diverse pair or triple | not started |
| P3 | Tree verification in the target (rules 1-2) | resolver only |
| P4 | Contextual router: per-position allocation from recent agreement and token entropy; retire drafters that stop paying | not started |

## Cleanup still open

Delete or adopt the provider slot and the qwen38_max shim,
`spark_speculation_tree.h` and `speculate.cuh`; drop the `Makefile:791`
flag; add `test_speculation_tree_resolve.c`,
`test_speculation_headers_coexist.c` and `test_qwen38_27b_remote_spec.c` to
the Makefile; fix the qwen38_27b DFLASH2 classification; drop the MTP
default in `tools/fleet_serve.sh`.
