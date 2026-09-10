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
