# PROGRESS — gemma4 driver lane

Branch `lane/gemma4-driver`, rebased onto origin/main **50bd0d3** (PR #913,
the E2E-proven platform). NEVER pushed (manager owns GitHub). Working dir
/Users/mac/batch-gemma4. DESIGN.md is now tracked (02a5264) with the
publisher-corrected rope constants — round 1 had kept it untracked; the
corrections made it worth freezing in-tree.

## REBASE RECORD (round 2)

Two rebases this session, both resolving toward main, family files kept:

1. **14df85a** (intermediate, now obsolete): the 9 round-1 family commits
   replayed clean; the manager's 20 mesh/transport commits
   (origin/lane/gemma4-driver ec378a2, preserved locally on ref
   `lane/gemma4-manager-mesh`) were DROPPED — they cherry-pick the glm53-p0
   weightd-mesh experiment that main's own 5942464 then reverted, and
   84edf57's gemma4 credit-binding strip predated the strip landing on main.
2. **50bd0d3** (current base): all lane commits replayed clean (no
   conflicts). Post-#913 the mesh dataflow IS the platform, so the manager's
   direction landed after all — via glm53-p0's own rebase wave.

**LIVE-SESSION HAZARD (standing rule):** a concurrent session works in the
/Users/mac/lane-* worktrees; this lane never reads or writes those. Re-fetch
before any future rebase; the manager's force-with-lease protects pushes.

## Platform alignment to 50bd0d3 (4947ba6)

The #913 engine made the module-side credit-binding machinery dead
(RouteCount is a stub returning 1, ApplyTopology copies rank_count, Create
opens the weightd mesh client and configuration.credit_bindings is never
read). gemma4 module now matches the glm5_next E2E template: credit state
fields, ProbeMemoryMode/RouteCount/CreditBytes block, device+mapped-host
credit allocations, binding population loop, and the wiring are DELETED
(module.c 1380 -> 1296 then 1333 with the oracle-era work; net -84); the
glm5_next fail-closed validation (nonzero timeouts/identifier,
hidden_transport backend) replaces them at Create. Both arm Makefiles drop
the deleted tp_device_collective_nccl.c link and take the weightd source
list via runtime/weightd_sources.mk (the new transport calls
SparkWeightdClientConnect at Create).

**Known gap (driver feature, unchanged):** mesh dataflow participation needs
the weightd lazy attach (PrepareReceiveBf16 with attached.mesh_send_buffer_addr
after Create). The gemma4 module has no lazy_pack yet, matching every
non-glm5_next module at this tip; Tp submissions fail at the mesh_buffer==0
guard until that lands. The eager pack load via spark_pack_load_common.h
remains the majority template (qwen38_max uses it on 50bd0d3).

## Deployment generator / session-port tables (briefing question, settled)

KEPT, not obsolete: post-#913 main still carries session_ports in
spark_tp_device_collective.h (topology + config), main's glm5_next module
still memcpys them, and main's glm5_next deployment generator still emits
explicit tables (env-driven bases). My env-driven fail-closed table
(SPARK_GEMMA4_STAGE_TP_SESSION_PORTS) matches the current main shape. When
the deeper strip (session matrices -> shared-memory coordination) lands on
main, the gemma4 table dies with it — one commit.

## Contract freeze (e148184)

Warm checkpoints landed (/mnt/model-warm/gemma-4-{31b-it,26b-a4b-it},
59G/49G, PUBLISHED). Warm config.json + model.safetensors.index.json are
byte-identical to the anchor kit's publisher snapshots, so the HF commit
pins carry over:

- 31B: `842da3794eaa0b77d5f08bae87a17459d91ff475`
- 26B: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`

model_contracts/gemma4_31b_authoritative.json +
gemma4_26b_a4b_authoritative.json: full geometry, rope (64 rotated pairs),
eos {1,106,50}, topology (TP16xPP1 / TP4xPP4 stage lists {8,8,7,7}), pack
folds, kv replication law, tensor census derived from the warm indexes
(832/657 text-stack entries), digest_freeze (small files hashed twice,
workstation + sparka, agreement; shards = HF LFS oid at the pinned revision
with byte-exact warm size equality; the sparka local full-shard hash
confirmation is running detached at /tmp/gemma4_shas.txt and lands with the
lane report). Both Makefiles pin GEMMA4_MODEL_REVISION to the HF commits —
pending-warm-download is GONE; the adapter's fail-closed #error stays for
unset macros only.

**Publisher finding folded in:** generation_config eos_token_id is
[1,106,**50**] — headers bound only {1,106}. All three now bind
(EOS_ALTERNATE_2_TOKEN_ID 50); the constants had no consumers yet, so this
fixes a future stop list, not behavior.

## Round 3 state (AC6/AC8)

**AC6 CUDA tier: RECEIPTED.** Both arms `make validate` (retained-receipt
script, nvcc CUDA 13 sm_121a, sparka GB10, sparkcap-wrapped): dense arm
`gemma4_validation PASS ... h5376.l60.v1 sites=23`, MoE arm
`... h2816.l30.e128k8.v1 sites=24` (router tier extra). The validator
(modules/gemma4_resident_decode_stage/validation/) drives every carved
SparkGemma4Launch* entry against a host mirror of the anchor-oracle math:
embed gather+scale bitwise (73.5/53.0 per arm), rms/fused-residual norms,
weighted + scale-free per-head norms, bf16 linear (scalar/tile paths),
residual/branch adds bitwise, gated gelu (bitwise vs mirror), sliding theta
rope + full inv_freq-table rope with the 64-pair identity region bitwise,
the 1024-window boundary matrix (5/1024/1030/2048) bitwise, KV store
bitwise readback + window decode (both sliding geometries), the k_eq_v
full-layer chain (v_norm from raw k before in-place k_norm, full-rope
table, store post-rope, 1200-token full-context decode), router softmax +
renormalised top-8 lowest-index ties + zero-residual uniform 1/128 (MoE),
and a bit-exact decode determinism rerun. OPEN ITEM: the composed per-layer
chain tier (norm->q->rope->kv->store->window->decode->o->FFN) segfaults in
libcuda on device at chain entry on the shared sparka GPU (gdb: fault
inside cuMemcpyDtoH; ASAN flags only device-dst memcpys = false positives
on unified memory); gated behind SPARK_GEMMA4_VALIDATION_CHAIN for the
receipts, diagnostics in the round-3 commits — needs a quiet GPU session.

**Platform bugs found by this round (manager flags):**
- `SparkLmExpertTileMloopKernel` (spark_lm_kernels.cuh, bf16 rows>=32
  path) loses ~1/4 of the K accumulation — probe-verified exact 0.75
  checksum ratio at in=512/5376/21504. Latent platform-wide: E2E decode
  batches stayed < 32 rows. gemma4's LaunchLinear routes all rows through
  SparkLmHostLaunchBatchedLinear until the platform fix lands.
- GB10 (sm_121a) MaxSharedMemoryPerBlockOptin = 101376 B: configure
  requests must be sized to the real max linear input; also plain
  cudaMalloc memory is NOT host-writable — the launchers reset the KV
  access_error host-side, so the error slot must come from the ledger's
  host-mapped allocations (validator uses cudaMallocManaged).
- The two arms write the same build/modules/.../ archive path: a clean is
  required between dense and MoE validate runs (workflow gap).

**AC8 real-pack: packer RECEIPTED against the warm checkpoints.**
tools/gemma4_stagepack.py (donor pattern per DESIGN section 6):
census-locked inventories (31B TP16 full-model 723 tensors incl.
layer_scalar; 26B stages 134/133/115/116), frozen TP16/TP4 shard maps —
full-kv x4 (31B, rank r reads head r/4) and x2 (26B, r/2) replication,
sliding k|v row fusion, router.scale x H**-0.5 folded into proj columns,
per_expert_scale folded into expert down rows (slot emits raw f32 for the
seam), f32 inv_freq table, two-pass placement proof (directory sha256 +
verify walk), .experts v2 manifests (48B records, ck128 ported bit-exact
from src/spark_ck128.c and cross-verified against the C reference),
spine/expert byte split + boundary-rank (0/last) checks in the receipt.
Receipt packs on sparka: 26B stage 2 (layers 16-22) ranks 0 and 3
(boundary, full-kv source head 1 on rank 3), 31B layers 0-7 rank 15
(boundary, full-kv source head 3 = r/4, embed base 245760).

**CONTRACT CORRECTION (warm-payload falsification):** the freeze-time
"layer_scalar checkpoint all-ones" assertion is FALSE on the warm
checkpoints — real learned per-layer scalars (26B layers 0/16/29 =
0.0703125/0.5546875/0.1953125 bf16). The packer's fail-closed check fired
exactly as designed on the first real-pack attempt. Fix landed this
round: LAYER_SCALAR tensor kind (census +1/layer: 31B 723/60L, 26B
573/30L full-model), module applies the scalar at the layer output
(SparkGemma4LaunchLayerScale), packer emits + records the values,
contracts updated. The anchor kit's all-ones fixture presumably reflected
an earlier snapshot — anchors and warm payload disagree, warm wins
(never-quantize weights law: the checkpoint is the source of truth).

**S4/S4' audit items:** norm.cuh tail block is byte-identical to the
muse-approved landing (verified) — merges as one copy; SPARK_RETURN/
lazy_pack: zero code references on the lane (PROGRESS prose only).
Rebase surface measured vs origin/main tip: ONE file (top-level
Makefile) + the shared norm.cuh tail; A-0057 rule recorded: take main's
side on every non-family file at the rebase hop. Manifest+sums regen
lands as the LAST commit of this round per the ledger rule.

## Acceptance criteria status

| AC | status | evidence |
|---|---|---|
| 1 port + deletion | DONE (round 1) | commits 6cb165d..e400708; grep gate clean |
| 2 both arms compile | DONE | host syntax-check green in BOTH arms on 50bd0d3 (module.c, serving adapter, stagepack format, synth tool; -Wall -Wextra -Werror vs cuda_stub); .cu compile is AC6's nvcc item |
| 3 header bindings test | DONE | tests/test_gemma4_model_header.py: 58 bindings over both contracts — geometry, rope thetas, embed scales, eos set, rotated-pairs invariant = 64, digest structure, 40-hex revision pins. PASS locally + on sparka. Wired into Makefile PYTHON_TESTS (9390cd1). The test caught the table-elements(256)-vs-rotated-pairs(64) distinction before freeze |
| 4 stagepack + synth | DONE (round 1, re-verified) | format header + synth compile both arms on 50bd0d3 |
| 5 oracle | DONE | validation/spark_gemma4_reference.c (944754c): plain C11, zero driver imports; consumes raw-binary exports of the anchor fixtures (validation/anchors_export.py, 270 arrays). **121 check sites, ALL PASS, both models**: bf16 RNE self-vectors, embed 73.5/53.0, inv_freq tables BITWISE (64 nonzero 1e6**(-i/256) + 192 zeros — the ANCHORS finding-1 curve), identity region exact, cos/sin at 7 positions (6 counted 1-ulp libm entries), v_raw==k_raw BITWISE, weighted/scale-free norms, norm-then-rope exact, 1024-window leak-direction exact, softmax + out recompute, eager attention with probs rounded bf16 before p@v (anchor finding 4) worst rel 0.0000, KV cache stores BITWISE, layer_out identity, MoE top-8 lowest-index ties + zero-residual uniform 1/128 + branch sums bf16(b1+b2) |
| 6 CUDA validation tier | NOT STARTED (harness rewrite needed) | the deleted donor validator (.cu, 1682 lines) externed the deleted kernel set; the gemma harness must be written against the carved cuda.cu entries, then queued on sparka GPU via the v2 queue tool (budgets: GPU <=10GB); V0 = synth-pack stage run vs the oracle. The oracle side of the comparison now EXISTS and is green, so the validator lands against a fixed reference |
| 7 offline gates | DONE (sparka cpu-class) | on sparka under sparkcap (receipt GEMMA4-AC7-SPARKA-ALL-GREEN): dry-law PASS (196 files, model-neutral), code-size ratchet PASS at 236409 exact, header bindings 58 PASS, oracle build -Werror + 121-site run PASS. Locally: same green. **Complexity ceiling is RED on pristine 50bd0d3 itself** (qwen38_27b serving adapter CCN 88 > 75 — not this lane's code, verified on a pristine main tree); flagged for the manager, NOT absorbed here. Package manifest regen deferred to AC8/landing (LAST rule) |
| 8 real-pack | UNBLOCKED, pending AC6 | revisions pinned, shard digests recorded, tools/gemma4_stagepack.py (real-weight packer: router-scale fold, per-expert fold, sliding k|v fusion, full-layer kv replication, layer_scalar assert-all-ones) is the next artifact; then HF-reference numerical comparison on a spark |

## Code-size ceiling

Re-pinned EXACT twice this session (ratchet law: justification in-commit):
241052 (2a46b8c) -> 244667 (14df85a-based tip) -> **236409** at the current
tip after #913 deleted the old engine from main (main 50bd0d3 = 232148
exact, -9179; gemma4 family = +4261 net incl. oracle/exporter/alignment).

## Blockers / flags for the manager

- **Complexity ceiling red on main 50bd0d3 itself**: qwen38_27b
  SparkQwen38_27bServingSubmitSpeculativeDecode CCN 88 vs ceiling 75 —
  tests/test_complexity_ceiling.py fails on a pristine main tree. Owner is
  the qwen38-27b lane (or a ceiling ledger entry), not gemma4.
- **AC6 needs a GPU session**: validator .cu rewrite against the carved
  kernel set + v2-queue GPU job. The oracle reference side is green and
  fixture-backed, so the harness has a fixed target.
- **Shard sha confirmation still hashing on sparka** (/tmp/gemma4_shas.txt,
  slow warm mount); contracts already pin shards via HF LFS oid + byte-exact
  size equality. Cross-check when the job completes.
- **lazy_pack/mesh participation** is the platform-aligned feature gap
  before bring-up (see above).
- The manager's 20 dropped mesh commits remain on origin/lane/gemma4-driver
  (ec378a2) and local ref lane/gemma4-manager-mesh — nothing lost.
- Queue note: the v2 queue daemon did not claim a sparka job this session
  ("claimed 0"); the AC7 receipt came from a direct sparkcap-wrapped ssh
  run. Watch the dispatcher if queueing more sparka work.

## muse: serving-adapter contract completion (post-merge follow-up, d7ebb16)

Audit finding A-0074 class: the qwen38-pp serving template exported an
interface missing the mandatory prefetch / resolve_prefetch / reset members,
so `SparkModelServingAdapterValidateInterface` refused the muse module at
load (runtime/model_serving_adapter.c:251 via the SPARK_REQUIRE_SERVING_OPERATION
table; the ABI_MISMATCH paths are :526/:543). Fixed in the same shape the
glm5_next/laguna references use (~10 lines per member over the shared
spark_serving_cache_admission.h helper):

- prefetch: CACHE_PREPARE admission over the muse state (program_id from the
  loaded driver program, 512-lane thread-local scratch, full validate first).
- resolve_prefetch: COMMIT/ABORT mapped to CACHE_COMMIT/CACHE_ABORT on one
  submission; invalid resolution → INVALID_ARGUMENT.
- reset: generation-monotonic (0 refused, `<= current` refused), single-flight
  atomic CAS (concurrent reset → BUSY), quiesce + driver RESET admission, and
  the applied generation makes stale submissions VALIDATION_FAILED on
  validate/prefetch via a guarded template hook (SPARK_QWEN38_SERVING_ADAPTER_SUBMISSION_STALE).

Shared-template seams are #ifdef-guarded; siblings that do not opt in
preprocess to the identical table (verified: qwen38_max preprocessed
interface table diff = trailing comma only; `-fsyntax-only` green). Their
three-member completion stays with their own follow-ups.

Two adjacent load/initialize blockers found and fixed in the same pass:
- descriptor was refused by ValidateDescriptor: added cache_block_token_count
  (64) and stage_layer_counts[0] (52), and dropped CAPABILITY_SPECULATION —
  muse is dense with MTP_LAYER_COUNT 0, so the bit contradicted
  max_speculative_token_count 0 (SpeculationPairing refusal).
- the speculation family bind could never succeed: provider validate refuses
  default_draft_token_count 0 and the seam contract refuses draft_layer_count
  0, so initialize always failed. Deleted the dead bind (DFlash2 sidecar
  bind returns when the lane has a real draft contract). The muse lane's
  example JSON `examples/model_descriptions/muse_glimmer_resident_decode_stage_firmware.json`
  still carries qwen38-derived text (purpose/revision/mtp fields) — flagged,
  not absorbed here.

New host gate: `build/test_muse_glimmer_serving_adapter` (Makefile TEST_NAMES)
builds the real adapter .c into a dylib against the cuda stub plus a fixture
driver (tests/fixtures/muse_glimmer_serving_adapter_driver.c + config json)
and asserts, in order: load through
`SparkModelServingAdapterLoadInterfaceFromSharedObject` with
CAPABILITY_HIDDEN_TRANSPORT required (the runtime check path); descriptor
identity/geometry; interface-table completeness — all 10 mandatory members
non-null and each nulled member refused INVALID_ARGUMENT (A-0074 made
unrepresentable); initialize; prefetch null/count guards, prepare,
commit, abort; decode submit with 4200/4201 token receipt + snapshot
counts; reset generation semantics (0 refused, 1 accepted, stale
validate+prefetch → VALIDATION_FAILED, `<=` refused, quiesce→reset revival
at generation 3, post-reset submit receipt); destroy.

Mac gate receipts (all run to exit 0): test_model_serving_adapter,
test_serving_cache_admission, test_qwen38_27b_serving_adapter,
test_muse_glimmer_serving_adapter, test_code_size (268756/268809, no
growth), test_dry_law PASS. test_complexity_ceiling remains red on the
pre-existing main offender (qwen38_27b SubmitSpeculativeDecode CCN 88 >
75) — inherited, not this lane's code, same flag the gemma4 lane raised.

dsv4 note: modules/dsv4_resident_decode_stage still exports an interface
without `.reset` (prefetch/resolve_prefetch present) — its own load test
will refuse it under the current ValidateInterface; that is the dsv4
lane's follow-up, same class as this fix.
