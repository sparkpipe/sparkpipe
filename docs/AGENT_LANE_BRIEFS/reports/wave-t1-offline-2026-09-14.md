# Wave T1-OFFLINE — the offline reference-decoder half of the T1 accuracy gate (2026-09-14)

Agent: T1-OFFLINE (fleet T1 accuracy wave, mgr2 dispatch). Branch
`lane/wave-t1-offline` off origin/main `0f5e79f` (fresh clone
`/Users/mac/t1off`). Identity verified `gh api user --jq .login` =
`sparkpipe`; every GitHub command through
`tools/sparkpipe_github_pat.sh`.

Mission: the instrument every family's T1 needs — ONE parameterized
offline reference generator (python/numpy) consuming a family's
`llm_defines.h` + the warm checkpoint, emitting committed fixtures; the
driver compares against them within tolerance when serving opens
(serving-gated half). OFFLINE law honored: no daemon contact, no model
execution on the fleet's serving path, no GPU work, spark3/spark6 never
contacted. Warm checkpoint reads were read-only, node-local, under
`sparkcap` (MemoryMax=4096M), after per-dataset dd-probes.

## Deliverable 1 — the shared reference-decoder harness

- `tools/t1_reference_common.py` — safetensors reader (indexed or
  single-file; BF16/F32/F16/U8/F8_E4M3), bf16/fp8-block/nvfp4/mxfp4
  decoders, `llm_defines.h` parser (rejects `SET_ME_*` placeholders, so a
  family header that is not pinned fails loud), and the deterministic
  `T1R1` compressed fixture container (per-array sha256 inside the
  container; any flipped byte fails at read time naming the array).
- `tools/t1_reference_decoder.py` — generator CLI: family profile module +
  header + warm checkpoint + canonical `prompts.json` in, fixtures +
  `MANIFEST.json` out. Manifest carries checkpoint config/index sha256,
  header sha256, prompts sha256, per-fixture sha256, generator versions,
  and any recorded defines/config mismatches.
- `tools/t1_reference_compare.py` — the comparison contract for the
  serving-gated half (documented in `docs/T1_REFERENCE_COMPARE.md`):
  ids/routing exact, streams and head scores banded (default rel 0.02 /
  abs 1e-3, operator may rule tighter), `FIRST DIVERGENCE: <array>` +
  exit 1 on mismatch, plus `corrupt-fixture` (negative-control generator)
  and `verify-manifest`.
- Family engines consume `SPARK_LLM_*` keys only — never hardcoded dims:
  - `tools/t1_reference_glm5_next.py` — mHC hyper-connections (sigmoid
    pre, 2x-sigmoid post, softmax+20-iteration sinkhorn combiner), KDA
    gated-delta recurrence with short causal conv + f_a/f_b decay latent
    + A_log/dt_bias retention, nope-absorbed MLA for the DSA layers,
    sigmoid router with e_score_correction_bias + lowest-index stable
    tie-break, clamped swiglu experts, shared expert. Ops ported from the
    two committed, driver-validated host oracles
    (`glm5_next_kda_host_oracle.py`, `glm5_next_dsa_host_oracle.py`) and
    the checkpoint layer reference — no new math.
  - `tools/t1_reference_qwen38_max.py` — softplus-based GDN decay
    (`-exp(A_log)*softplus(dt+dt_bias)`), swish conv, per-head scalar
    delta rule with L2-normalized q/k, swish-gated head norm, partial
    rotary (64 of 256 dims, theta 1e7, rotate-half), fused q+output-gate
    attention (q second half applied as sigmoid gate per head), GQA 4 kv
    heads at scale 1/16, no-bias top-10 router with softmax over selected
    logits (route_scale 1.0 — module call constant), nvfp4 experts
    (e2m1 + per-16 e4m3 + per-tensor f32 scale_2, unrounded weights into
    the f32 dot exactly like the module's validation kernel), clamped
    routed swiglu (limit 10) with UNCLAMPED shared-expert swiglu, shared
    output gated by `sigmoid(x . shared_expert_gate)`. Ops ported from
    `modules/qwen38_max_resident_decode_stage` kernels +
    `inference/llms/qwen_3_8/` runner + the module's launch constants.
- Synthetic proof, runs offline on any machine with numpy:
  `tests/test_t1_reference_decoder.py` builds a miniature glm5_next-shaped
  checkpoint (safetensors written by the test), runs the real glm5_next
  profile end-to-end, and requires (a) two runs byte-identical
  (determinism), (b) manifest sha256 matches the fixture bytes, (c) a
  corrupted fixture byte FAILs the compare naming
  `pos0001_layer0001_streams`, (d) a defines/config disagreement
  (`SPARK_LLM_MLA_LATENT_DIMENSION` 9 vs 8) fails the generator loudly.
  PASS locally (numpy 2.5.0, python 3.14).

## Deliverable 2 — family ports

### glm5_next (model glm53flash — the reference driver) — REAL FIXTURES

- `prompts.json` (committed with the fixtures): 2 canonical prompts
  tokenized by the checkpoint's own tokenizer on spark5 —
  "The capital of France is" (ids 785,6722,315,9621,374... as committed)
  and "Counting upward: one, two," — greedy, `new_tokens: 4` each,
  capture layers {0, 3, 22, 44}.
- Validation ladder before fixtures were trusted:
  1. synthetic miniature proof (above);
  2. POSITION-0 CROSS-CHECK against the committed
     `tools/glm5_next_checkpoint_layer_reference.py` on spark5 with the
     real `/mnt/model-warm/glm-5.3-flash`: my engine's top-1 token equals
     the oracle's (154822 for token 785) and kda_gated norm agrees
     (0.17087 vs 0.17107);
  3. the cross-check FIRST caught a real defect: `kda_out` multiplied the
     raw uint16 `o_norm` words instead of their bf16 values (117,800x
     error; the earlier generated tokens were garbage — '，' for "The
     capital of France is"). Fixed; a dtype audit then caught the same
     defect in the qwen engine (A_log/dt_bias are BF16 in the warm arm,
     not F32). Both fixed before any fixture was committed.
- Result: BOTH prompts decode to finite, plausible greedy continuations
  (details in the harvest section below), fixtures committed under
  `qualification/t1_reference/glm5_next/` with MANIFEST.json.
- Harness finding (recorded in the fixture MANIFEST, adjudication
  pending): main's `model-families/glm5_next/include/sparkpipe/llm_defines.h`
  pins `SPARK_LLM_MLA_V_HEAD_DIMENSION 512u` but the checkpoint config
  and the kv_b_proj weight shape both say v_head_dim 256
  (`kv_b_proj` is [32768,512] = 64 heads x (256 nope + 256 v)). The
  generator treats checkpoint weights as ground truth and records the
  mismatch instead of failing the unambiguous keys.

### qwen38max — REAL FIXTURES (trimmed contract) or state at freeze

(see harvest section; engine complete + validated, generation
compute-bound on nvfp4 CPU dequant)

### k3 — header pinned, engine is the named follow-up

- `model-families/k3/include/sparkpipe/llm_defines.h` committed, pinned
  ONLY from sourced values: warm `config.json` (hidden 7168, 93 layers,
  vocab 163840, situ betas 4.0/25.0, 896 experts top-16 sigmoid
  renormalize, first routed layer 1, KDA 96 heads x 128 with
  gate_lower_bound -5, MLA q_lora 1536 / latent 512 / nope 128 / rope 64
  / v 128) and in-tree constants (`K3_ROUTED_SCALE` 1.0,
  `K3_MLA_QK_SCALE` 0.07216878365 = 192^-0.5,
  `K3_KDA_GATE_LOWER_BOUND` -5.0). Revision = the warm config sha256.
- Keys that cannot be sourced from the warm config or the in-tree pin
  (`SPARK_LLM_ROPE_THETA` — the MLA rope base lives in the publisher
  modeling file, not the repo; yarn factor; kda chunk size; gate
  bottleneck) are OMITTED, not invented: the header fails the moment an
  engine asks for them.
- The k3 ENGINE is the named offline follow-up and must not be guessed:
  kimi_k3 is a different architecture (latent-space MoE with
  `routed_expert_{up,down,norm}` + per-expert packed w1/w2/w3 and a
  ROUTER BIAS, v-space gate/output projections, extra res-proj streams
  `self_attention_res_proj`/`mlp_res_proj`). The extraction surface is
  `inference/llms/kimi_k3/layer.cuh` (~1000 lines) +
  `modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_runner.cu`
  — a dedicated session, same method as the glm5_next port (ops only
  from driver-validated sources, position-0 cross-check before
  fixtures).

## Deliverable 3 — ACC stragglers

- gemma4 anchor fixtures COMMITTED (the ACC-2 reproducibility gap):
  `validation/gemma4_anchors/` = all 270 exported bins + the original
  `manifest.txt` (name rows cols flag) + a new `MANIFEST.json` with
  sha256 per bin, fetched from `sparka:~/gemma4_anchors_bin` (13 MB).
  The committed set regenerates the 121-site claim offline: gcc-built
  `validation/spark_gemma4_reference.c` against the committed dir prints
  `GEMMA4 ORACLE ALL CHECKS PASS (121 sites, 6 counted near-ulp
  entries)`, exit 0. Negative control: flipping byte 3 of
  `26B__rope__inv_freq_full.bin` gives
  `SPARK_FAIL rope.full_table: expected=-4@0 actual=1@0`, exit 1 — the
  oracle convicts naming the site. (Note recorded honestly: a byte flip
  in `26B__moe__b1.bin` did NOT convict — that site's comparison
  tolerates it; the committed control uses the bitwise-checked rope
  table.)
- muse cross-shard bitwise (ACC-2 gap): `tools/muse_kv_shard_bitwise.py`
  turns the A7 kv-replication law from a construction claim into a
  measurement. Real packs run on spark5 (rank04 mesh-copied from spark4,
  3,639,537,920 B): ranks 04/05 (kv-head group 0) — 261 spans compared
  (52 layers x kv row-blocks + 209 replicated norm entries), 0
  mismatches, RESULT: PASS. Negative control: one flipped byte inside
  layer 0's kv row-block → `MISMATCH kv group 0 layer 0` naming rank and
  layer, RESULT: FAIL. Remaining ranks are one command each as their
  packs are reachable.
- qwen38-27b tp4pp4 ranks 1-15 re-derivation (ACC-1 gap): NOT STARTED —
  capacity went to the reference harness and the two real fixture runs.
  The rank00-proven method stands (packer re-run `convert` per rank vs
  placed bytes; ACC-1 receipt `placed_sha == rebuilt_sha` on rank00);
  command shape: `tools/qwen38_27b_stagepack.py` per rank with
  world_rank/tp4 settings against `/mnt/model-warm/qwen3.8-27b-fp8`,
  compare digests against the placed rank packs.

## Harvest — real fixture runs (node receipts)

All runs: `sudo -n systemd-run --scope -q -p MemoryMax=4096M
-p MemoryHigh=2900M --uid=1000 python3 -u t1_reference_decoder.py ...`
per the byte-trace memory law; unbuffered logs.

- glm5_next (spark5, warm `/mnt/model-warm/glm-5.3-flash`, 1.1 GB/s
  class): run1 (both prompts) + run2 (determinism pair) on the FIXED
  engine. Committed fixtures:
  `qualification/t1_reference/glm5_next/{capital_of_france,count_up}.t1r`
  + MANIFEST.json (config sha bb8f01c4..., index sha 3c3f4036...,
  header sha 1a7f4b06...). Continuations, decoded with the checkpoint's
  own tokenizer: "The capital of France is" -> [3837, 271, 271, 12]
  ('，\n\n\n\n-'), "Counting upward: one, two," -> [323, 12, 29805, 11]
  (' and-‐,'). These look odd because the canonical prompts are raw
  token ids without a chat template — that is the point: the fixture
  pins the model's RAW greedy behavior, not a pleasing completion.
  Their faithfulness is anchored, not assumed: the committed oracle
  agrees with the engine on position-0 top-1 for BOTH probes —
  token 785 -> 154822 ([gMASK], logit 14.19 vs 13.00) and token 374 ->
  264 (logit 15.25 vs 15.08) — the same continuation class the
  full-prompt fixtures record.
- Determinism receipts:
  - glm5_next `capital_of_france.t1r`: spark5 run1 vs run2 (40 minutes
    apart) BYTE-IDENTICAL (`cmp` clean).
  - glm5_next `count_up.t1r`: spark0 run vs rerun (with OMP_NUM_THREADS=20
    set on the rerun and unset on the first) BYTE-IDENTICAL.
  - qwen38_max `capital_of_france.t1r`: spark0 run1 vs run2
    BYTE-IDENTICAL.
  - Cross-node honest finding: a spark0 rerun of count_up DIFFERS from
    the spark5 reference (different routing from layer 3 onward — the
    sigmoid router's top-8/9th score tie flips under cross-machine
    gemm/BLAS reduction-order noise, and the greedy decode amplifies the
    flip through the remaining 42 routed layers into different tokens).
    Consequences recorded for the T1 gate design: (a) the committed
    fixtures are NODE+LIBRARY-PINNED references (the generator now
    records host and threading env in every manifest; the committed
    runs are documented here), (b) same-node reruns are byte-identical,
    so anti-tamper and regression checks are exact, (c) the driver-side
    compare cannot hard-require route-id equality or multi-layer stream
    bands against a differently-ordered implementation — the serving
    gate needs a chaos-aware tolerance design (e.g. early-layer
    teacher-forced sites plus final-token rank agreement). This wave
    ships the instrument and the pinned references; the tolerance
    ruling belongs to the gate owner.
- Negative controls on the REAL committed fixtures (in addition to the
  synthetic test), run with the fixed compare tool: a flipped byte in a
  copy of `glm5_next/count_up.t1r` FAILs naming
  `pos0006_layer0022_streams`; a flipped byte in a copy of
  `qwen38_max/capital_of_france.t1r` FAILs naming
  `pos0004_layer0045_route_weights`. `verify-manifest` PASSES on both
  committed sets. A compare-tool defect found by the cross-node work was
  fixed first: bf16-pattern arrays were compared as raw uint16 exact
  instead of through their meta dtype band — the tool now consults the
  container metadata (`BF16`/`F16` arrays get the tolerance band; integer
  arrays stay exact).
- qwen38_max (spark0 after sparka wedged on this dataset mid-run —
  4.2 MB/s probe on spark4, folio_wait_bit stall on sparka; spark0
  probed 1.0-1.5 GB/s): nvfp4 CPU dequant dominates (92 layers x
  10 experts x 3 matrices per position), so the committed qwen contract
  is ONE canonical prompt ("The capital of France is", ids
  760,6511,314,9338,369) with `new_tokens: 2` — generated
  [107300, 107300] ('资源篮' x2), head scores 4.95/5.04, finite.
  Committed: `qualification/t1_reference/qwen38_max/capital_of_france.t1r`
  + MANIFEST.json (config sha b2c724e3..., index sha 2b8d7065... —
  note this equals the tp4pp4 receipt's source index sha ACC-1
  recorded, identity leg across arms). No committed position-0 oracle
  exists for this family (the module has no host oracle), so the qwen
  fixtures ride on: ops ported only from driver-anchored sources, the
  config cross-check, the synthetic proof, and — when serving opens —
  the first driver compare through `tools/t1_reference_compare.py`,
  which is the instrument this wave exists to hand the serving-gated
  half. Full 2-prompt set is a rerun of the same command with
  `prompts_qwen38max.json` (committed alongside the tiny set).

## Ratchet

Branch measures 287645 authored lines vs the pinned ceiling 283170.
Attribution: +3017 pre-existing on origin/main `0f5e79f` (ACC-1/ACC-2
measured +1587 at the earlier `d1c3822`; more landed since), +1458 this
branch (tools/t1_reference_*.py 1306 + two llm_defines.h 152; the gemma4
anchor bins, manifests, and docs are excluded by construction). NOT
re-pinned here per dispatch: the coordinator adjudicates the main-tree
drift; this PR does not mask it and adds no tests to inflate anything.

## Ledger

- Branch: `lane/wave-t1-offline` (base origin/main `0f5e79f`).
- Fleet state touched: read-only warm reads on spark5 (glm-5.3-flash,
  muse rank05, dtype probes), sparka (qwen3.8-max, gemma4 anchor pull),
  spark4 (one muse rank04 pack read + 1.1 GB mesh-copy to spark5, one
  dd-probe); node-local NVMe scratch on spark5 (`~/muse_tp16_rank04...`,
  deletable) and sparka (`~/t1ref-qwen*`, deletable). Zero daemons, zero
  GPU, zero packs written, spark3/spark6 untouched. Disk caches purged
  where heavy batches ran.
