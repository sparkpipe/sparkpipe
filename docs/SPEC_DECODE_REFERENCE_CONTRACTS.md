# Speculative Decoding — Reference Contracts (authority doc)

The playbook mined from working code, contract by contract, mapped to **our** DSV4
Flash DSpark resident stage, with the fix recipe for each of our four known gates
(CSA compressor bug, lease accounting, 8-row verify structure, draft-weight rung).

Sources in priority order (closest shape first):

1. **Community GB10 repos** (the dsv4-flash survey): tonyd2wild + Weschera
   (DeepSeek-V4-Flash DSpark, 2× DGX Spark — our exact hardware), 0xBakeer
   (Qwen3.8-27B DSpark+MTP, single DGX Spark — our exact hardware), plus
   joesinvestments (4× DGX Spark, the 123 tok/s SOTA) and the HF reference
   `inference/model.py`. Their kernel/recipe content is already mirrored locally
   in `docs/research-dspark/` (tony-README, DSPARK_VERIFY_LOOP_REPORT, joe-RECIPE,
   joe-RESULTS) and `docs/SURVEY_DSV4-FLASH.md`.
2. **vLLM mainline** `vllm/v1/spec_decode` + `vllm/v1/worker/gpu/spec_decode`
   (DFlash/DSpark/EAGLE + rejection sampler).
3. **SGLang** `python/sglang/srt/speculative/` (`dspark_components/` + EAGLE).
4. **TRT-LLM** draft-target/EAGLE/Medusa (`cpp/include/tensorrt_llm/runtime/`).
5. **llama.cpp** self-speculative/MTP/DSpark (`common/speculative.cpp`).

Canonical semantics for the DSV4-Flash DSpark verify/accept loop are in
`docs/research-dspark/DSPARK_VERIFY_LOOP_REPORT.md` (exact acceptance algorithm,
KV/markov state, draft-quantization map, engine status + the CSA gap). This doc is
the authority for the six contracts below.

---

## Contract index

| # | Contract | Our status | Our file |
|---|---|---|---|
| (a) | Draft-KV isolation from committed cache | ✅ match | `spark_dsv4_resident_decode_stage_module.c:4474-4507` |
| (b) | Multi-row verify causality | ✅ match | `:4411` (draft) · `:2393-2485` (verify) |
| (c) | Compressed-KV/compressor state under speculation | ❌ **gap** (CSA emission rollback) | `:2642-2678`, `:2316-2317` |
| (d) | Partial-accept completion/continuation accounting | ✅ match (lease bug was call-site) | `:3577-3597` |
| (e) | 8-row verify batch structure | ✅ match (staging fixed acc6783a) | `:2393-2485` |
| (f) | Draft-weight sourcing | ⚠️ rung-3 gate to verify | `:980-981`, `:1238-1240` |

---

## (a) Draft-KV isolation from the committed cache

**Contract:** the drafter's speculative KV must never enter the target's committed
cache; only the accepted prefix is committed. Every engine does this with a
*draft-private* buffer.

- **Community / vLLM (DSpark):** the draft layers keep **their own** sliding-window KV
  cache (window=128, *uncompressed* path), populated from the MAIN-branch KV via
  `precompute_and_store_context_kv`; block-position KV is transient (never cached).
  (`DSPARK_VERIFY_LOOP_REPORT.md:112-127`; vLLM
  `dflash/speculator.py:101` `draft_kv_cache_group_id`, `:181-193` separate context
  slots, `:434-438` `precompute_and_store_context_kv`.)
- **SGLang:** the drafter owns a separate `token_to_kv_pool`; the target's hidden
  states are injected *into* it (`dspark_kv_inject.py:72`, `:86-94`), never the
  target cache.
- **TRT-LLM:** draft tokens live in a separate `draftTokens`/`draftIndices` buffer,
  not the KV cache (`explicitDraftTokensBuffers.h:55-57`).
- **llama.cpp:** a separate draft `llama_context` with its own `mem_dft`; the
  speculative KV is removed with `llama_memory_seq_rm(mem_dft, seq, pos, -1)`
  (`common/speculative.cpp:1509`).
- **Ours:** draft runs replicated full-width with its own 128-slot
  `dspark_ring_bf16`; the anchor KV is scattered into that ring
  (`spark_dsv4_resident_decode_stage_module.c:4474-4507`, `:359-368`, `:1647`), and the
  draft attention reads `dspark_ring + committed block KV` non-causally (`:4411-4422`).

**Fix recipe:** none needed — our direction is correct. Do not let any "KV-reuse
optimization" write draft rows into the paged cache; the draft ring stays private,
and the committed cache is advanced only through the lane/lease advance in (d).

---

## (b) Multi-row verify causality

**Contract:** the draft block is ONE *non-causal* parallel pass; the verify is ONE
*causal* target pass over `1+k` positions, where each verify row's attention window
ends at its own position (per-row causal). There is no per-row recurrence inside the
verify.

- **Community / vLLM:** draft attention non-causal (`dflash/speculator.py:287`
  `causal=False`, `:104-112` `use_non_causal`); verify = `num_sampled_tokens =
  num_draft_tokens + 1` causal positions (`gpu_model_runner.py:2333`);
  `draft_seq_lens = seq_len + step` (`speculator.py:227-294`).
- **SGLang:** `seq_lens + verify_w` batch expansion, one `is_verify=True` target
  forward (`dspark_verify.py:253-261`).
- **TRT-LLM:** a *tree* of paths with a `packedMask` per-node causal mask
  (`explicitDraftTokensBuffers.h:63`, `speculativeChoicesUtils.h:35-39`) — the
  tree generalizes the linear block; the causal principle is identical.
- **llama.cpp:** anchor-first block (`speculative.cpp:926-930`), causal verify.
- **Ours:** 8 verify rows (anchor + SPEC_STEP drafts); "CacheScatter writes all 8
  rows' KV before SparseAttn reads, and the position-causal mask gives the
  block-causal semantics" (`DSPARK_VERIFY_LOOP_REPORT.md:277-279`).

**Fix recipe:** none needed — the position-causal mask over the 8 staged rows is the
reference semantics. Keep the draft non-causal and the verify causal; do not add
recurrence inside the verify rows.

---

## (c) Compressed-KV / compressor STATE under speculation — **our CSA bug**

**Contract:** the compressed-KV (CSA ratio 4 / HCA ratio 128) running state is
*position-keyed* (`state[position % ratio]`), so a block of `1+k < ratio` rows
never races — but the **boundary emission** (when a row lands on a compressed-group
boundary, `(position+1) % ratio == 0`) writes a compressed slot that *includes the
rejected rows' contributions* and must be rolled back.

- **Community / vLLM:** the draft never re-derives the prefix — it
  `precompute_and_store_context_kv` from the target's *hidden states* into the
  draft's separate sliding-window cache (`dflash/speculator.py:421-438`,
  `DSPARK_VERIFY_LOOP_REPORT.md:112-127`). The compressor emits a boundary slot at
  `position/ratio` behind the window.
- **SGLang:** the compressed state is an SWA ring with a **commit gate** — only
  `col < commit_lens` tokens get injected; uncommitted verify rows are dropped
  (`dspark_kv_inject.py:137-172`, esp. `:167-171`).
- **TRT-LLM / llama.cpp:** no compressor (EAGLE tree / self-spec MTP use the full KV);
  N/A.
- **Ours:** the CSA compressor (`SparkDsv4LaunchCompressStep` +
  `SparkDsv4LaunchKvEmission`, `spark_dsv4_resident_decode_stage_module.c:2642-2678`)
  accumulates a running per-lane state and emits the boundary slot at
  `row_emit_positions = position + 1 - ratio` (`:2316-2317`, `:2470-2471`). The
  running state is position-keyed so the 8 verify rows don't race — but **the
  compressed-emission boundary is not rolled back** when a boundary row is rejected.

**Fix recipe (the CSA bug):** before the verify, snapshot the compressed-emission
target region; if a rejected row lands on the emission boundary
(`(position+1) % ratio == 0`), restore that region and do not re-emit. The
accepted-prefix state is position-keyed, so a restore + no-re-emit is exact.
Probability ~`ratio`-dependent (~8/21 per frame at ratio 21; CSA ratio 4 → higher).
Recorded at `DSPARK_VERIFY_LOOP_REPORT.md:301-313`. The SGLang commit-gate pattern
(`col < commit_lens`) is the reference: only committed rows may update the
compressed state.

---

## (d) Partial-accept completion / continuation accounting — **our lease bugs**

**Contract:** acceptance is a *prefix* match (stop at the first reject); the emitted
burst is `1 + accepted` (bonus token + matched prefix), and the lane/continuation
advances by exactly `1 + accepted`, never 0 and never the full `1+k`.

- **Community:** `num_sampled = accepted + 1`; next anchor = last emitted token;
  rejected suffix re-drafted next block (`DSPARK_VERIFY_LOOP_REPORT.md:59-75`).
  joe (TP4 k=7): 6.019 tok/step = 1 + 7×0.717 (71.7% accept). 0xBakeer RESULTS.md
  defines the metric as `1 + accepted/k`.
- **vLLM:** `accepted_length` prefix stop + `num_sampled = accepted + 1`;
  acceptance noise keyed by position (`rejection_sampler_utils.py:541-667`, `:569`,
  `:1087-1188`); `get_num_sampled_and_rejected` (`rejection_sampler.py:276-282`).
- **SGLang:** `correct_len` + `bonus`; `commit_lens = correct_len + 1`
  (`dspark_verify.py:623-658`).
- **TRT-LLM:** `nextGenerationLengths`/`nextPositionOffsets`/`bestPathLengths`
  (`explicitDraftTokensBuffers.h:89-108`).
- **llama.cpp:** `accept(seq_id, n_accepted)` (`speculative.cpp:170`,
  `n_acc_tokens`/`n_acc_tokens_per_pos` `:149-151`).
- **Ours:** greedy Leviathan prefix match (`accepted` = first index where
  `output[i] != draft[i]`), `accepted_token_count = tokens_per_sequence =
  emitted_token_count = 1 + accepted`, `lane_next_positions += 1 + accepted`
  (`spark_dsv4_resident_decode_stage_module.c:3577-3597`).

**Fix recipe:** the module-side emit is correct. The historical CLIENT_LEASE_DISCONNECT
bugs were in the *lease call-site* advancing by `new_token_count` (total across
lanes) instead of the per-sequence `1+accepted` — the residentd fence
(`node/model_residentd.c`, `runtime->client.generation`) and the
completion-emitted-count advance (commit c8f76e5) already fix this. The authority
rule: the continuation lease advances by `completion.tokens_per_sequence` (the
`1+accepted` emit), keyed by `step_generation`.

---

## (e) 8-row verify batch structure — **our row-structure divergence**

**Contract:** k drafts + 1 anchor = `k+1` verify rows per request; row 0 is the
anchor at its real position P, rows 1..k are the drafts at P+1..P+k; row i's logits
predict position P+i+1; draft queries are `noise_token_id` (128799) at positions
P+1..P+k-1.

- **Community:** `DSPARK_VERIFY_LOOP_REPORT.md:43-50` (query 0 = anchor, queries
  1..k-1 = noise 128799; sample_pos = query_pos+1) and `:269-273` (rows
  anchor_position+1 .. +8; row i predicts the token after its input; all-7-accept
  adds the bonus output[7]).
- **vLLM:** DSpark `num_query_per_req = num_speculative_steps` (anchor-first,
  `sample_from_anchor=True`), vs DFlash `1+N` (anchor = bonus)
  (`dspark/speculator.py:40-55`, `dflash/speculator.py:46-47`).
- **SGLang:** `stride = 1 + gamma` (bonus + gamma drafts).
- **llama.cpp:** `draft-dspark` = "anchor-first block layout" + Markov head
  (`speculative.cpp:926-930`).
- **Ours:** 8 rows (1 + SPEC_STEP) at KV slots anchor_position+1..+8; the staging
  bugs (physical-page-per-row, 3×u32+3×u64 layout, page init, input-tokens-before-gather)
  were fixed in commit acc6783a (`DSPARK_VERIFY_LOOP_REPORT.md:283-299`).

**Fix recipe:** already fixed. The anchor-first (not 1+N) layout is the DSpark
convention; do not regress to DFlash's `1+N` anchor-as-bonus layout — the DSpark
anchor predicts the *first* draft token.

---

## (f) Draft-weight sourcing — **our rung-3 gate**

**Contract:** the draft weights ship **in the checkpoint** under `mtp.{0,1,2}.*`
(draft layers = stages 0/1/2, i.e. model layers 43/44/45), and the loader must map
the *shared expert* shards `gate_up_proj ← w1/w3` **and** `down_proj ← w2`, plus
`markov_head.markov_w1/w2` and `confidence_head.proj`. Dropping any always-on
shared-expert shard silently halves acceptance (output stays correct).

- **tonyd2wild (the canonical case):** `DSPARK-SHARED-EXPERT-FIX.md:40-132` — the
  draft loader dropped `shared_experts.w1/w3` (only `w2` was mapped), 12 tensors
  uninitialized → acceptance 25.7% → 60.2% after the fix
  (`_STACKED_PARAM_NAME_MAPPING` gains the `shared_experts.gate_up_proj ← w1/w3`
  rows). Their policy: **keep draft weights STOCK** — "Edited draft weights would
  land you back in acceptance collapse".
- **ds4/Metal:** the only full 4-bit draft in the wild (Q4_K GGUF of all 3 layers);
  Q4_K vs F16 acceptance Δ < 0.4%; F32 accumulation over Q4_K recovered 1.53 → 2.2.
- **vLLM:** loads the draft with the *target's* quant_config (linears fp8, experts
  MXFP4); no 4-bit draft in mainline (`DSPARK_VERIFY_LOOP_REPORT.md:187`).
- **SGLang / llama.cpp:** same checkpoint source; llama.cpp self-spec MTP reads
  `mtp.*` from the GGUF.
- **Ours:** the pack carries the 3 full draft layers replicated on every rank; a
  missing draft tensor is a refused pack (`spark_dsv4_resident_decode_stage_module.c:980-981`, `:1238-1240`). Experts MXFP4, linears FP8, markov/confidence heads
  bf16/fp32 (per the checkpoint layout, `DSPARK_VERIFY_LOOP_REPORT.md:183-186`).

**Fix recipe (rung-3 gate):** gate the pack on **all** draft tensors present —
18 shared-expert tensors (`w1/w3/w2` × 3 stages × (weight + scale)), the routed
experts, `markov_head.markov_w1/w2`, `confidence_head.proj`, and the draft
attn/ffn projections — and assert none are dropped (a dropped always-on shared
expert is the tonyd2wild failure mode: silent, output-correct, ~2.3× acceptance
loss). Do not re-quantize draft experts (keep MXFP4); quantizing draft *linears* to
4-bit is the optional bandwidth win (`DSPARK_VERIFY_LOOP_REPORT.md:213-219`).

---

## Authority summary — remaining fix order

1. **(c) CSA boundary-emission rollback** — the only correctness gap in the
   speculation path (`DSPARK_VERIFY_LOOP_REPORT.md:301-313`); snapshot/restore the
   compressed-emission region when a boundary row is rejected. Reference pattern:
   SGLang's `col < commit_lens` commit gate.
2. **(f) draft-weight rung-3 gate** — assert all 18 shared-expert + markov +
   confidence + projection tensors load (tonyd2wild's silent-drop failure mode).
3. **(d) lease audit** — confirm the residentd/pipeline lease advances by the
   `1+accepted` emit (already landed); never by `new_token_count`.
4. **(a/b/e)** verified correct — do not regress the private draft ring, the
   non-causal-draft/causal-verify split, or the anchor-first 8-row layout.

The exact-token gate protects the target in all cases: a degraded or mis-sourced
draft costs speed, never output correctness (tonyd2wild: "output quality was
perfect … only speed changed"; ds4: token-identical at temp 0 across draft quants).
