# DFlash2 Speculative Decoding — Handoff Document

**PR:** #675, branch `qwen38-dflash2`, HEAD `506770e` + the argmax-selection session
**Model:** Qwen/Qwen3.8-27B (FP8 target), DFlash2 block-diffusion drafter (5-layer, hidden 5120, block 8)
**Hardware:** GB10 Spark (sm_121a), single-GPU B1 serving
**Date:** 2026-08-22 (argmax-selection update)

---

## 0. THE ARGMAX UNLOCK (2026-08-22 session — read this first)

**Problem A is SOLVED.** The acceptance gap was NOT in the forward — it was the draft
selection rule. The SOTA reference's SERVING path (vLLM `vllm serve`) selects drafts as
**per-mask-row full-vocab ARGMAX**: the worker loads `DFlashSpeculator` (v1,
`dflash/speculator.py`), whose `_generate_draft` calls `sample_draft ->
gumbel_sample(compute_logits(mask-row hiddens))` — greedy == argmax. The codebook lattice
walk (`_score_edges` + `_selector_walk_kernel` in `dflash2/speculator.py`) is **dead code in
the serving path** — that module is never imported by the server (verified: its .pyc only
regenerated on manual import). The prior session ported the walk from that dead code.

Evidence: on the reference's own dumped inputs, argmax-per-row agrees with the reference's
drafts at **96–100% per position** (37/39, 37/39, 39/39, 39/39, 39/39, 39/39, 38/39) vs
87%-at-best for any walk variant (`tools/qwen36_dflash2_vllm_input_parity.py`,
SELECT_MODE=argmax default).

Engine: the host walk in `SparkQwen36ModuleRunDsparkBlockForward` is replaced by rank-0
selection (the device top-16 is value-desc/index-asc, so rank 0 IS the argmax).

Measured on O128 (spark2, k=8, `SPARK_QWEN36_DFLASH2_WINDOW=256`): **E 1.10 → 2.36**,
~3.36 committed/round, bit-exact vs the no-spec baseline (same token stream).
Reference on the SAME prompt at temperature 0: **E = 2.90** (streak 81/74/72/78/86/83/90).
Our pos-0 rate ≈ 76–81% (parity); the depth decay is still steeper than theirs — the
residual is draft quality at depth 2–3, not selection.

Other session findings:
- The specforge layer-bisect was INVALID: specforge's `Qwen3DFlashDecoderLayer` has no conv
  (DFlash1); DFlash2 weights loaded `strict=False` — the 0.92–0.98 cosines compared
  different architectures. Treat the conv path as validated only via the argmax agreement.
- The reference's context-KV is INCREMENTAL: each round `precompute_and_store_context_kv`
  processes only the NEW frame rows into the persistent cache (decode-round dumps are 8-row
  deltas). Our engine rebuilds the full window per round — measured cost is small (W=256 vs
  2048: +1.5% e2e), so this is NOT the bottleneck people feared.
- BLOCK_KV stale-history rows barely matter under argmax (1.54 vs 1.60 mean accepted) —
  the walk-era "unlock" attribution was confounded with the wrong selection rule.
- **Where the time actually goes** (`SPARK_QWEN36_PROFILE=1`, k=8, per frame ~229ms):
  **FFN 155ms (68%!)**, GDN 44ms (19%), head 16ms, attn 12ms. The old "28ms/row GDN
  serialization" attribution was wrong — that slope was FFN-per-row. The FFN runs at ~6%
  of memory bandwidth; and the engine runs ~1.84 frames per spec round (~128 wasted
  full-weight frames per 512 tokens — client submission granularity + fold arming;
  raising `max_prefill_rows_per_submission` makes it WORSE, 5.2 tps — the fold logic
  depends on 1-row submissions). These two are the next throughput levers, not the GDN.

---

## 1. Where We Are

| Metric | O128 | GSM | Notes |
|--------|------|-----|-------|
| No-spec baseline | ~7.7 | 8.13 | Bit-exact golden hashes |
| vLLM reference (SOTA) | 18–20 | 18–20 | E=4.30 accepted/round, CUDA-graphed |
| **Us now (k=5 unlock)** | **10.34** | **6.44** | E=1.10, bit-lossless |

**We are at ~53% of reference on O128, ~35% on GSM.** GSM at k=5 is below the no-spec baseline; the best GSM config is the 2-frame fold at k=3 (8.76 tok/s, commit `acf397c` minus BLOCK_KV).

### Golden hashes (must match after any change)
- O128 (128 tokens): `02740eb47b01c606`
- GSM (160 tokens): `dc9bea4ee644b324`

### Current best serving config (spark2)
```bash
cd /home/spark2/sparkdata/qwen38.fp8.tp1
env LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH \
  SPARK_QWEN36_SERVING_SPECULATE=1 \
  SPARK_QWEN36_SERVING_SPEC_METHOD=dflash2 \
  SPARK_QWEN36_SERVING_SPECULATIVE_DRAFT_COUNT=5 \
  SPARK_QWEN36_DSPARK_PACK_PATH=/home/spark2/sparkdata/qwen38-dflash2-drafter.qwen36sp \
  SPARK_QWEN36_DFLASH2_STATE_SELECT=1 \
  SPARK_QWEN36_DFLASH2_BONUS_FOLD=2 \
  SPARK_QWEN36_DFLASH2_BLOCK_KV=1 \
  setsid nohup bin/sparkpipe_model_residentd \
    --deployment config/model_resident.json --rank-index 0 \
    > /tmp/qwen38.log 2>&1 < /dev/null &
```

---

## 2. Architecture (What's Implemented)

### The one-frame round (`BONUS_FOLD=2`)
Each decode submission becomes ONE verify frame (the vLLM shape):
- Row 0 walks the previous round's accepted emission `e_m`, restoring GDN checkpoint slot `m` via `GDN_RESTORE_VERIFY_ROW`
- Rows 1..k walk the selected draft block's tokens
- The verify tail runs the multi-block padding drafter (`dspark` view ABI v2: `multi_block_count`)
- **Padding-select:** the module computes the verify's own accept depth (emissions vs walked rows, one tiny D2H) and drafts ONLY block m — drafting all k blocks was pure host-bound overhead (6.55 → 9.15 tok/s)
- Commits `m+1` tokens, arms restore slot `m` for the next round
- No correction frame; bootstrap round is 2-frame, round 2 walks from live state

### The persistent draft KV history (`BLOCK_KV=1`)
Per-layer raw k/v rows of every block the drafter ever ran, keyed by position (re-walked positions overwrite), stored in `dflash_block_hist_k/v` [4096 rows/layer], copied in after the current block rows and attended at their own positions. This is the HF DynamicCache semantics that specforge trains with and vLLM serves with.

### The device-side selector
Fused kernel: per-slot top-16 (value desc, index asc — the host pass's exact two-key order via sortable-packed atomicMax merge) + the 256-wide hidden projection (strided threads, `__fmul_rn`/`__fadd_rn` to bit-match scalar rounding). Replaces ~4MB logits D2H + ~35ms host pass with one compact ids/scores/hproj copy. `SPARK_QWEN36_DSPARK_SEL_CHECK=1` keeps the original host pass as a live oracle.

### The NeoX-128 rope
The drafter ropes HF-style `rotate_half` over the FULL 128-dim head (dim d pairs with d+64, frequency θ^(-2d/128)). This is NOT the target's convention (interleaved-64). Implemented in `SparkQwen36DsparkRopeFrequencyNeoX()` + the cache-attn kernel's q and k loops.

---

## 3. The Two Remaining Problems

### Problem A: Acceptance E=1.10 vs reference 4.30 (THE dominant gap)

The reference accepts 4.3 of 7 drafts per round; we accept 1.1. Same weights, same target, same prompts. This is a 4x gap that converts directly to throughput at fixed round cost.

**What's been proven equivalent (don't re-check these):**
- Input features: our taps = reference aux at cosine 1.000 per layer over the shared GSM prompt (tool: `tools/qwen36_dflash2_fp8_bf16_tapdiff.py`)
- Forward math: three implementations (CUDA, numpy, torch) agree bit-for-bit; specforge's `DFlashDraftModel` per-layer cosine 0.92–0.98 (tool: `tools/qwen36_dflash2_specforge_bisect.py`)
- bf16 rounding: fp32 forward gives identical 34/39 agreement (eliminated)
- Sliding window: all 5 layers are `sliding_attention` with window 2048 — never binds at our prompt lengths
- Context window size: sweeps at 2048/512/32 all flat
- Rejected-row masking: flat
- Rope convention: fixed (NeoX-128), validated on reference dumps
- Draft KV persistence: implemented, transforms the accept distribution (capped at 3 → runs through 7)

**What remains — the 13% draft divergence:**
Running our forward on the reference's own dumped inputs, we agree with their drafts at 87% (34/39 at position 0). The 13% disagreement shows:
- Confident score gaps (+0.08 to +0.79 under our math), NOT near-ties
- Their token always in our top-16 (candidate generation agrees; scoring differs)
- Concentrated at position 0 of EARLY rounds (first 2-3 rounds)
- Persist in fp32 (not a rounding artifact)

**The characterization:** the divergence is in the runtime state at round transitions — something the dump-and-replay harness can't fully replicate. The dumps capture `set_inputs_first_pass`'s parameters, but the reference's actual draft forward may see a slightly different context (e.g., the KV cache state accumulated through CUDA graph replays, or the bonus row's context KV from a different step in the pipeline).

**How to proceed:**
1. **Direct comparison in-process:** Load specforge's `DFlashDraftModel` AND our forward in the same Python process. Feed both the exact dump inputs AND build the KV cache identically (DynamicCache on their side, our persistent store on ours). Hook per-layer, compare block hidden states row-by-row.
2. **The cache warm-up hypothesis:** Our engine's block-KV history starts empty at bootstrap. The reference's DynamicCache accumulates from the FIRST draft forward (including the prefill precompute). The first 2-3 rounds see different cache depths. Test: pre-warm our history with the bootstrap round's block rows before the first one-frame verify.
3. **Round-start position-0 focus:** The disagreement is concentrated there. In the walk, slot 0's predecessor = the anchor (bonus). If the ANCHOR's context differs (e.g., the reference includes the bonus token's own KV row in the context), the entire first slot's edge scoring shifts. Check: does the reference's `precompute_and_store_context_kv` include a row for the bonus position?

### Problem B: GDN verify serialization (~28ms/row)

Our `SparkQwen36GdnStepKernel` serializes verify rows in-kernel (necessary for state chaining). The reference uses a chunked parallel mamba scan. Result: our verify cost scales linearly with k, theirs doesn't.

**Measured:** k=3→k=7 adds ~110ms (4 rows × 28ms) to the round. The reference's CUDA-graphed batched verify costs the same for any k.

**How to proceed:**
- The GDN recurrence is a linear state-space scan: `s_t = A·s_{t-1} + B·x_t`. For a block of k verify rows, this can be parallelized as a chunked scan (compute per-row transitions in parallel, then prefix-scan the state). vLLM's `mamba_ssm` kernel does exactly this.
- The kernel lives in `spark_qwen36_resident_decode_stage_cuda.cu`, function `SparkQwen36GdnStepKernel`. The checkpoint writes (per-row state snapshots to slots 8+row) must be preserved.
- After parallelizing, k=7 becomes viable → E=1.25 × lower round cost → ~12-14 tok/s even at current acceptance.

---

## 4. Build & Deploy (spark2)

```bash
# 1. rsync sources
rsync modules/qwen36_resident_decode_stage/source/*.c \
       modules/qwen36_resident_decode_stage/source/*.cu \
       modules/qwen36_resident_decode_stage/source/*.cuh \
       spark2:/home/spark2/sparkpipe/modules/qwen36_resident_decode_stage/source/
rsync modules/qwen36_resident_decode_stage/include/sparkpipe/*.h \
       spark2:/home/spark2/sparkpipe/modules/qwen36_resident_decode_stage/include/sparkpipe/

# 2. module (GPU validator — MUST pass)
ssh spark2 'pgrep -f "[s]parkpipe_model_residentd" | xargs -r kill; cd /home/spark2/sparkpipe && \
  make -C modules/qwen36_resident_decode_stage -j8 publish \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH=/home/spark2/sparkdata/qwen38.fp8.tp1/packs/qwen38-fp8.tp1.qwen36sp \
  STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=64 \
  TP_DEGREE=1 TP_RANK=0 TP_STANDALONE=1 MTP_LAYER_COUNT=1 \
  GDN_SNAPSHOT_SLOT_COUNT=16 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 \
  ALLOW_UNQUALIFIED_EXECUTION=1 2>&1 | tail -4'

# 3. adapter (TP1) — rebuild BOTH or you get the stale-adapter bug
ssh spark2 'cd /home/spark2/sparkpipe && \
  rm -f build/libqwen36_serving_adapter.so && \
  make build/libqwen36_serving_adapter.so \
  CC="cc -DSPARK_QWEN36_SERVING_TP_DEGREE=1u" -j8 2>&1 | grep -c error'

# 4. driver compile + deploy BOTH
ssh spark2 'cd /home/spark2/sparkpipe && \
  rm -rf /tmp/qwen38-driver-new && mkdir -p /tmp/qwen38-driver-new && \
  env SPARK_QWEN36_STAGE_COUNT=1 SPARK_QWEN36_STAGE_INDEX=0 \
  SPARK_QWEN36_STAGE_FIRST_LAYER=0 SPARK_QWEN36_STAGE_LAYER_COUNT=64 \
  SPARK_QWEN36_TP_DEGREE=1 SPARK_QWEN36_TP_RANK=0 SPARK_QWEN36_TP_STANDALONE=1 \
  SPARK_QWEN36_STAGE_MTP=1 SPARK_QWEN36_STAGE_GDN_SNAPSHOT_SLOTS=16 \
  SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES=8 SPARK_QWEN36_STAGE_KV_BLOCKS=8 \
  SPARK_QWEN36_STAGE_KV_STORE=none SPARK_QWEN36_STAGE_KV_SERVICE=none \
  SPARK_QWEN36_STAGE_KV_SOCKET=none SPARK_QWEN36_STAGE_KV_POOL_BYTES=0 \
  SPARK_QWEN36_STAGE_KV_WORKER_COUNT=0 SPARK_QWEN36_ALLOW_UNQUALIFIED_EXECUTION=1 \
  build/sparkpipe_model_compile \
  --model examples/model_descriptions/qwen36_resident_decode_stage_firmware.json \
  --library build/module_library --output /tmp/qwen38-driver-new \
  --include include --cc-arg -L/usr/local/cuda/lib64 --cc-arg -lcudart --cc-arg -lstdc++ \
  2>&1 | tail -1 && \
  cp /tmp/qwen38-driver-new/stages/stage_000/model_driver.so \
     /home/spark2/sparkdata/qwen38.fp8.tp1/lib/model_driver.so && \
  cp build/libqwen36_serving_adapter.so \
     /home/spark2/sparkdata/qwen38.fp8.tp1/lib/model_serving_adapter.so'

# 5. code-size gate (local)
python3 tests/test_code_size.py  # must pass; ratchet ceiling with justification
```

### Benchmark
```bash
ssh spark2 'cd /home/spark2/sparkdata/qwen38.fp8.tp1 && \
  export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH && \
  bin/sparkpipe_model_batch --deployment config/model_resident.json \
  --runtime-root $PWD --batch /tmp/o128_batch.json'
# Parse token events from output (JSON lines); compute tok/s and sha16
```

---

## 5. The Reference (spark3)

### Server (should still be running)
```bash
# Model staged on internal NVMe (4.7x faster than extnvme)
# /home/spark3/sparkdata/Qwen3.8-27B-local/
ssh spark3 'pgrep -f "[v]llm-venv/bin/vllm" && echo ALIVE || \
  cd /home/spark3 && \
  VLLM_DFLASH2_INPUT_DUMP=40 LIBRARY_PATH=/usr/local/cuda/lib64 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  nohup /home/spark3/vllm-venv/bin/vllm serve \
  /home/spark3/sparkdata/Qwen3.8-27B-local \
  --speculative-config "{\"method\": \"dflash\", \"model\": \"/home/spark3/sparkdata/qwen38-dflash2-drafter\", \"num_speculative_tokens\": 7}" \
  --max-model-len 2048 --gpu-memory-utilization 0.85 --port 8123 \
  > /tmp/vllm_dflash.log 2>&1 < /dev/null &'
```

### Key patched files (in spark3's venv)
- `vllm/v1/spec_decode/dflash.py` — input dump hook (`set_inputs_first_pass`, env `VLLM_DFLASH2_INPUT_DUMP=N` → `/tmp/vllmin_<n>.pt`)
- `vllm/model_executor/models/qwen3_dflash.py` — aux dump hook (`combine_hidden_states`, env `VLLM_DFLASH2_AUX_DUMP=1` → `/tmp/vllmaux_0.pt`)
- `vllm/model_executor/models/qwen3_dflash2.py`, `registry.py`, `spec_decode/__init__.py`, `dflash2/` — the DFlash2 PR files (copied from spark0)

### Dumps on spark3
- `/tmp/vllmin_0.pt` .. `/tmp/vllmin_39.pt` — 40 rounds of drafter inputs (target_hidden_states, positions, bonus, walked tokens, num_rejected)
- `/tmp/vllmaux_0.pt` — raw aux features (5-layer concatenation) for the first call

### Measuring acceptance from the reference
```bash
curl -s http://spark3:8123/metrics | grep spec_decode
# vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0".."6"}
# vllm:spec_decode_num_drafts_total, num_accepted_tokens_total
```

---

## 6. Tooling (all in-tree on branch `qwen38-dflash2`)

| Tool | Purpose |
|------|---------|
| `tools/qwen36_dflash2_vllm_input_parity.py` | Run OUR forward on THEIR dumped inputs; score draft agreement. Env: `KV_CACHE=1`, `ROPE_MODE=neox`, `DTYPE=fp32`. Run on spark3. |
| `tools/qwen36_dflash2_specforge_bisect.py` | Layer-by-layer compare vs specforge's DFlashDraftModel on the same inputs. Run on spark3. |
| `tools/qwen36_dflash2_fp8_bf16_tapdiff.py` | Prove our FP8 taps = their BF16 aux features per layer. |
| `tools/qwen36_dflash2_deep_parity.py` | Replay the numpy reference on one captured dump. |
| `tools/qwen36_dflash2_round_parity.py` | Statistical scorer across all captured rounds. |
| `tools/qwen36_dflash2_conv_sweep.py` | 16-combo convention sweep (rope × ctx × block). |
| `tools/qwen36_dspark_reference.py` | The numpy oracle (bf16-exact forward + selector). |

---

## 7. Root Causes Found & Fixed (Session Ledger)

| # | Bug | Fix | Commit | Impact |
|---|-----|-----|--------|--------|
| 1 | Tap-store grid wrote 256/5120 channels | `dim3 grid(rows, (H+255)/256)` | pre-session | taps were 95% unwritten |
| 2 | Drafter context dropped the walked row's tap (g_P pair) | `window_base = base - window` (include g_P) | `170b109` | p0 41%→70% |
| 3 | Interleaved-64 rope (wrong convention) | NeoX-128 in kernel | `acf397c` | trained convention |
| 4 | No draft KV memory (blocks don't attend own history) | `dflash_block_hist_k/v` persistent store | `acf397c` | distribution runs to 7 (was 3) |
| 5 | Three full-model passes per round | One-frame round + padding-select | `216dd39` | 2→1 passes |
| 6 | Host selector: 4MB D2H + 35ms scalar pass | Fused device kernel | `f8eac1a` | bit-identical, ~1-2% |
| 7 | Lane continuity rejected one-frame rounds at branch points | VERIFY_ROW restore re-establishes lane | `216dd39` | fixed chain-break crash |
| 8 | Selector hproj: threads 128..255 never written (rank 256) | Strided `for (rr = tid; rr < rank; rr += nt)` | `f8eac1a` | -6% O128 → fixed |

---

## 8. Known Gotchas

- **Rebuild BOTH binaries** (driver + adapter) after any change — a stale adapter with a new driver caused the one-frame crash that took hours to diagnose.
- **Kill residentd before building** — the GPU validator OOMs if the daemon holds memory.
- **The daemon leaves zombie GPU allocations** after kill -9; run `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9` then wait.
- **Prompt retokenization:** the reference's /v1/completions endpoint doesn't accept `prompt_token_ids`; decode→re-encode→send→verify `usage.prompt_tokens == expected`. O128 re-encodes to 130 (not 128).
- **Code-size gate:** `python3 tests/test_code_size.py` — ratchet the ceiling in the same commit with a justification comment.
- **spark3's extnvme is ~50MB/s effective** — the model is now staged on internal NVMe at `/home/spark3/sparkdata/Qwen3.8-27B-local` (loads in ~3 min vs ~20).

---

## 9. Recommended Next Steps (Priority Order — revised after the argmax unlock)

1. **Kill the double-frames** (~1.84 frames/round): the adapter runs a full-weight frame
   for ~128 of 280 frames that commit nothing — submission-granularity + fold-arming
   interaction in `spark_qwen36_serving_adapter.c`. Trace `SparkQwen36ModuleRunFrame`
   callers vs `spec_diag` rounds. Potential ~1.8x.

2. **FFN efficiency**: 155ms/frame at ~6% of the GB10's memory bandwidth for a full
   27GB weight sweep (floor ≈ 100ms/frame total). Check the fp8 GEMM path's batch
   efficiency at 9-row frames (kernel selection, split-k, layout). Potential ~1.5–2x.

3. **Depth-2+ draft quality** (E 2.36 vs reference 2.90 at T=0): pos-0 is at parity;
   the decay steepens after. Compare per-layer hiddens against the live reference via
   the stage dumps (`tools/qwen36_dflash2_stage_diff.py`; needs the vllmsel dump patch
   moved into the v1 speculator's `_generate_draft`, since dflash2/speculator.py is
   dead code — the /tmp/stage_patch.py on spark3 has the working pattern).

4. **GDN scan parallelization**: 44ms/frame (19%) — chunked-scan the verify rows.
   Worth doing only after 1–2 land.

5. **CUDA graphs** for the verify+drafter frame — the reference's flat-round-cost trick.

At E=2.36 with frames 1.84x→1.0 and FFN at bandwidth: ~15-17 tok/s. At E=2.9: SOTA 18-20.

### Measurement gotchas (this session)
- The reference at temperature 1.0 SAMPLES drafts (Gumbel) — comparing our greedy chains
  against temperature-1.0 dumps is meaningless. Drive it with `"temperature": 0`.
- `sparkpipe_model_batch` rejects a re-run of the same request_id; use fresh ids per run.
- Daemon relaunch over ssh drops the session (exit 255) — use launcher scripts
  (`/tmp/launch_spec2.sh <bkv> <k>` with `W=` env on spark2; pre-truncate the log and
  wait for "model_residentd ready", the stale-log grep race bites).
- GPU zombies after kill -9 need a wait before relaunch (~5s) or the stage load hangs.
- Benchmark wall-time includes ~10-20s fixed overhead — use `SPARK_QWEN36_PROFILE=1`
  frame spins for GPU-side comparisons; the profile counters are CUMULATIVE across the
  daemon lifetime (relaunch between measurements).
