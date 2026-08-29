# draft-formats lane report — 2026-08-29 (attempt 1)

Lane: the speculator bake-off's format gate. Branch `lane/draft-formats`,
worktree `/tmp/lane-draftfmt`, base = main `7af7de4` (taken when the worktree
was created; main is moving under the parallel lanes — rebase conflicts are
the coordinator's to arbitrate at merge). PR: **#741** (open, never merged by
this lane).

Scoreboard tie-in: bulk-packs2 found 10/11 admitted speculator sources
format-blocked. This lane makes the fleet able to LOAD and SERVE those
drafters. G2+G3 (the K3 wire path + the provider slot) are LANDED on the
branch; G1 (the 27B DFlash2 bring-up) is fully staged and fires the moment a
spark9/a window opens — see the blocker section.

## G2 — K3 DSpark drafter wire path (LANDED)

### Source verification (pinned)

`/mnt/model-warm/kimi-k3-dspark-redhatai`, config.json +
safetensors-header dump (both run on spark5):

- DSparkDraftModel; block 8; aux taps {24,48,72,88,92}; mask 163837;
  draft_vocab 163840; markov_rank 256, markov_head_type vanilla;
  confidence_head_with_markov true; rope theta 10000 default; sliding_window
  2048; BF16.
- Transformer: hidden 7168, 5 layers, **96 Q / 16 KV x 64**, FFN 14336.
- **64 tensors, all BF16**: 5 layers x 11 per-layer kinds + 9 globals
  (`fc.weight` [7168, 35840] = 5 taps x 7168; `markov_head.markov_w1/w2`
  [163840, 256]; `confidence_head.proj.{weight,bias}` = [1, 7424]/[1] —
  7424 = hidden 7168 + markov 256; the drafter SHIPS its own
  `embed_tokens` and `lm_head` [163840, 7168] — unlike DFlash2, which
  shares the target's).

Cross-check against the tree: `inference/llms/kimi_k3/dspark.h` pins the
ORIGINAL 2026-07-27 release (block 7, taps {7,23,51,67,83}, mask 163824,
64 Q heads). The admitted redhatai source is a DIFFERENT release of the same
method. The two pins deliberately do not share identifiers; the format header
documents both. RadixArk (block 7) and inferact (different config schema —
no `transformer_layer_config` block) are NOT covered; the packer refuses them
naming why until their geometry is source-verified the same way (evidence:
the packer run against a stripped config fails with `config.json carries no
transformer_layer_config block`).

### Wire format (K3DS v1, magic 0x5344334B 'K3DS')

The qwen38_27b drafter pack's Q6SP v3 discipline — fixed little-endian
header, 56-byte entries (`6I4Q`), 256-aligned payloads — with the header
EXTENDED by 16 U32 fields (120 -> 184 bytes) because the DSpark releases
vary per source and the pack must be self-describing: taps[5], markov_rank,
mask_token_id, sliding_window, head-flags (embed|lm_head|confidence|
confidence+markov|sliding), rope_theta_milli, confidence_input_dim.

Kind numbering 0..16 mirrors `SparkQwen38_27bDsparkTensorKind`: 11 is the
target-tap projector in both; slots 12/13 — which DFlash2 repurposed for its
selector codebooks — carry the Markov W1/W2 weights whose [vocab, rank]
shapes are exactly what DFlash2 repurposed them FOR; 14 reserved; 17..20 are
the DSpark-only tensors (own embed/lm_head, confidence head).

### Packer: `tools/k3_dspark_stagepack.py` (committed a0a30f8 + 0f6c580)

- Geometry-driven from the source's `config.json`; every planned shape
  checked against the safetensors header (element-count equality — the 1-D
  norm vectors [64]/[1] land as [1,64]/[1,1]); dtype other than BF16
  refused (repackage-never-quantize).
- verify(): header sanity + per-entry bounds/alignment/overlap + ACTUAL file
  size vs header file_bytes (a truncated pack is rejected — caught live
  during development).
- Byte-for-byte round-trip PASS is the tool's exit gate.
- Performance: v1 did 64 random seeks + 64 re-opens of the ceph-backed
  source and measured **~3 MB/s**; now one sequential file-offset-order
  pass with pwrite into the planned slots.

**FULL-SIZE PACK BUILD: IN FLIGHT on spark5**
(`/tmp/draftfmt-build/kimi-k3-dspark-redhatai.k3dsp`, spawned pid recorded,
nohup): the header/entries write completed and the payload stream is filling
under heavy storage contention — the closeout lane's 15-rank repack is
reading the same warm storage concurrently, and measured throughput fell to
~1-2 MB/s (first attempt: ~3 MB/s under the same contention; a clean-storage
run is minutes). Attempt history, honestly: build #1 TERM'd BY ME (captured
pid) when the random-seek stall was diagnosed; build #2 TERM'd BY ME
(captured pid) when the VERIFY phase was found to have the same
random-seek stall; build #3 (current) has both fixes. The tool itself is
proven: mini-source round-trip PASS plus every gate; the full-size
round_trip=ok line lands in `/tmp/draftfmt-build/build.log` when storage
calms. Placement of a placement-worthy pack belongs to bulk-packs after
merge; nobody should serve this scratch build without its receipt.

### Module bind + validator (b8769a4, 95eead9)

- `modules/k3_resident_decode_stage/source/spark_k3_dspark_format.h` —
  constants, kind enum, kind-shape table (mirrors the 27B header's style).
- `SparkK3DsparkPackBind()` in `spark_k3_pack_load.c` (+ public header
  `spark_k3_dspark_pack.h`): file-backed mmap (the family loader's
  sanctioned path), header validation, EVERY pinned redhatai field checked
  with the field NAMED in the refusal (the supports()->WHY rule), entries
  shape-checked against the kind table, payload resolution by (kind, layer).
- Validator PASS (CPU, on spark5 AND via the new Makefile target):
  `build/test_k3_dspark_pack` → `k3 dspark drafter pack: bind + refusal
  paths all pass` — good bind + geometry round-trip + payload resolution +
  five refusal paths (radixark-class block 7 → names `block_size`; foreign
  tap → `tap_layer_0`; missing flags → `flags`; bad magic; truncation →
  `file_bytes`). Test trick: full-geometry pack in a SPARSE file (correct
  header/entries, holes for payloads — the bind never reads payload bytes).
- The K3 adapter translation unit compile-gates clean under nvcc
  `-Wall -Wextra -Werror` (sm_121a) with the new code.
- **END-TO-END ON REAL BYTES**: the in-flight pack's written header+entries
  (first 64KB, sparse-extended to the declared file_bytes) were bound by
  the probe (`/tmp/bindprobe` on spark5):
  ```
  BIND OK tensors=64 hidden=7168 layers=5 Q=96 KV=16 hd=64 ffn=14336
  vocab=163840 block=8 depth=7 taps=[24,48,72,88,92] markov=256
  mask=163837 window=2048 flags=0x1f conf_in=7424 file_bytes=9489806336
  ```
  The raw 64KB copy was REFUSED first (`actual size != header file_bytes`)
  — the truncation gate firing on real input — and the probe caught one
  real bind bug (required-flags must be a bitwise check; the pack's
  sliding bit is informational), fixed in 95eead9. Real packer output and
  real bind agree on every pinned field.

### What G2 does NOT include (honest)

The DSpark draft FORWARD (5-layer backbone walk + markov bias + confidence
gate on device) is not landed — that is the family's first real kernel
project and it is recorded as the follow-up. The provider's draft ops fail
closed with that reason. What lands today is everything the drafter kernels
will stand on: packs can be built, bound, and proven before a single draft
kernel exists.

## G3 — the drafter slot's first two real users (LANDED, f9bb4d7)

`docs/SPECULATION_PROVIDER_DESIGN.md`'s slot
(`spark_speculation_provider.h` + `runtime/speculation_provider.c`) existed
with only a test proving both binding shapes. Now:

- **K3 adapter** (embedded-provider shape): a DSPARK provider whose state IS
  the bound drafter pack. Initialize binds when
  `SPARK_K3_SERVING_SPECULATE=1` + `SPARK_K3_DSPARK_PACK_PATH` are set —
  fail loudly on an arm without a pack or a refused pack (the 27B's
  never-draft-silently rule); unarmed = cell-unchanged serving. The bound
  line prints geometry + `draft_forward=not_landed_fail_closed`.
- **MAX adapter** (same shape): the in-checkpoint MTP head (kind MTP, depth
  1 = `SPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT`), provider validated at
  initialize.
- Both providers carry: the ONE verify accounting (anchor-first,
  accepted = verified − 1, chain width + tokens-per-sequence reported once —
  the lease-advance bug class dies here), a capability query that names its
  WHY, a KV contract (K3: scratch+tail, no block history yet; the 27B's
  BLOCK_KV is the precedent to grow into), and draft ops that fail closed
  with the reason until the family draft kernels land.
- Descriptors: `SPECULATION` capability + the provider's depth envelope
  (K3 = block−1 = 7; MAX = 1). The shared validator's XOR rule
  (runtime/model_serving_adapter.c:55) is the only gate touched; the other
  two uses of the count (model_serving_adapter.c:599, model_residentd.c:1067)
  are envelope bounds — cells unchanged. The K3 fleet bring-up (other lane)
  is unaffected until merge: their staged .so predates this branch.
- Code-size ratchet RUN after the changes: `non-test authored lines: 215593
  (ceiling 216956) — the authored codebase did not grow`.

## G1 — 27B + DFlash2 on spark9 (STAGED; node blocked)

Value: first spec-mode serving outside the old bench; the DFlash2-vs-
incumbent pair's first data row (accepted/step telemetry from the
`qwen38_27b_spec accepted=` log lines + the `qwen38_27b_spec_diag`
per-round dumps).

**Blocker (exact):** spark9 AND sparka fell inside lane-glm5-closeout's
full-fleet reservation, acquired 05:45:33Z TODAY (one minute before my first
reserve attempt), TTL 360min → ~11:45Z. The queue refuses a second holder
(`spark9 already reserved: {'holder': 'lane-glm5-closeout', ...}`). Per the
one-wave-owner rule I did not co-launch GPU work inside their window; the
closeout lane's 15-rank glm5_next repack/redeploy cycle IS running (their
residentd ranks 9/10 are live on spark9/sparka). Retried the reserve
repeatedly through the session — still held at last attempt.

**Staged and ready (committed 0f6c580):
`tools/qwen38_27b_dflash2_serve.sh`** (SPARK_HOST-parameterized):

- `stage` — builds `~/sparkdata/qwen38.fp8.tp1.dflash2` beside the incumbent
  (binaries copied; the 29.9G target pack referenced by ABSOLUTE path; host
  fields fixed to the target node; **max_sequence_positions 4096** — the
  KV pool halved so 27B (~52G at halved KV per the measured 71.1G@8192) +
  drafter 3.6G + glm5_next rank 21.7G ≈ 77 GiB, comfortably under the
  110 GiB ceiling; the 8192-block config measured 114G/119G in the
  coordinator's co-resident test and is exactly what OOM'd sparka).
- `launch` — THE MANDATORY DFLASH2 LAUNCH ENV, verbatim:
  `SPARK_QWEN38_27B_SERVING_SPECULATE=1 SPARK_QWEN38_27B_SERVING_SPEC_METHOD=dflash2
  SPARK_QWEN38_27B_SERVING_SPECULATIVE_DRAFT_COUNT=8 SPARK_QWEN38_27B_DSPARK_PACK_PATH=<drafter>
  SPARK_QWEN38_27B_DFLASH2_STATE_SELECT=1 SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD=2
  SPARK_QWEN38_27B_DFLASH2_BLOCK_KV=0 SPARK_QWEN38_27B_DFLASH2_WINDOW=2048
  SPARK_QWEN38_27B_DFLASH2_CTX_CACHE=1` with `DSPARK_PACK_PATH` =
  `~/sparkdata/qwen38-dflash2-drafter-incoai.qwen38_27bsp` (sha verified
  `7bd3a840…` on-node). Spawn-captured pids, pre-truncated log, waits for
  the `model_residentd ready` line.
- `api` / `smoke` / `stop` — api on 17490 (27B port block is free on
  spark9; glm5_next holds only 19569), one greedy completion, then the
  acceptance lines; stop TERMs ONLY the recorded pids.
- Pre-flight verified on spark9: drafter pack present + sha `7bd3a840…`;
  incumbent deployment + pack present; DFLASH2 strings in the staged
  adapter .so and driver .so; port 17480/58700/17490 free; co-resident
  memory envelope computed above.

FIRE CONDITION: `spark_queue.py reserve --node spark9 --holder
lane-draft-formats ...` succeeds → run stage/launch/api/smoke → record
accepted/step → `stop` + release. Anyone (coordinator included) can run
this; the script is in the branch.

## INTEGRATION REQUEST (coordinator)

Carried in-branch (Makefile + link lines), to be integrated at merge:

1. `K3_SERVING_ADAPTER` link line (Makefile ~L785): += 
   `runtime/speculation_provider.c` — the adapter now calls
   `SparkSpeculationProviderValidate` (the ONE validation point; a second
   copy of the validator would be a DRY violation).
2. Test list (~L223) + rule (~L920): `test_k3_dspark_pack` (CPU-only;
   command verified: `make build/test_k3_dspark_pack && ./build/test_k3_dspark_pack`).
3. The qwen38_max serving ADAPTER has no main-Makefile target (the module
   Makefile builds only the module; wherever the deployment tooling links
   `model_serving_adapter.so` for max, it needs
   `runtime/speculation_provider.c` too).
4. DFlash2-vs-incumbent pair completion: the z-lab incumbent drafter pack
   (`qwen38-dflash2-drafter.qwen36sp`, spark2 only) needs a spark9/a copy
   (3.6G) so the pair flips on `DSPARK_PACK_PATH` alone — either
   bulk-packs or this lane, once a window opens.
5. Follow-up kernel work (NOT small; own lane): the K3 DSpark draft forward
   (backbone walk + markov bias + confidence gate + block KV), and the MAX
   MTP draft forward (module fails closed today). The wire paths land
   first so both start from proven packs + bound providers.

## Lane incidents / deviations

1. **Full-fleet reservation by lane-glm5-closeout (all nodes, 05:45:33Z,
   TTL 360min)** — blocked every GPU action this lane had (G1 bring-up, and
   any small-tier module GPU validation). CPU-only work proceeded per the
   bulk-packs2 precedent (packer + builds + tests on spark5; no daemon
   contact, no GPU). G1 is one command away, fully staged.
2. **First redhatai pack build was TERM'd by me (pid captured at spawn,
   TERM only)** when its throughput measured ~3 MB/s (random ceph seeks);
   replaced by the sequential packer and re-run. No fuzzy matching; the
   killed pid was the one this session spawned.
3. The scratch tree on spark5 (`/tmp/draftfmt-wt`, `/tmp/draftfmt-build`)
   holds the branch checkout + the verification pack; reproducible from the
   branch (sources.mk-free, plain cc/nvcc lines in the report above).

## Next experiments

1. Fire G1 when spark9/a clears (script ready; acceptance telemetry =
   `grep qwen38_27b_spec accepted= /tmp/qwen38_dflash2_spark9.log` →
   accepted/step per round, the bake-off's first data row).
2. RadixArk (block 7) + inferact config schemas source-verified and pinned
   as K3DS variants (the format already carries per-source taps/block; only
   the pinned constants + bind table grow).
3. The K3 DSpark draft-forward kernels behind `K3DsparkProviderDraftBegin`
   (the provider slot's first real inner loop), then a K3 serving smoke on
   the k3-fleet lane's packs when stages 0-2 land.
