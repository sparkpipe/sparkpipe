# vLLM & SGLang speculative-decode contracts vs our DSV4 DSpark

Source-read comparison (2026-08-18). External sources: vLLM `vllm/v1/spec_decode` +
`vllm/v1/worker/gpu/spec_decode` (the v1 DFlash/DSpark/EAGLE path — the old
`vllm/spec_decode/draft_model_runner.py` is gone in v1) and SGLang
`python/sglang/srt/speculative/` (EAGLE + `dspark_components/`). Our side is the
DSV4 Flash DSpark resident stage under `modules/dsv4_resident_decode_stage/`.

Four contracts, then the CSA-write-direction verdict.

---

## 1. Draft KV isolation

**vLLM** — the drafter owns a *separate* KV-cache group and never writes the
target's committed cache.
- `dflash/speculator.py:101` `draft_kv_cache_group_id = -1`; `:181-185` resolves the
  draft's own `draft_kv_cache_group_ids`.
- `:434-438` the draft's "context KV" is stored into *separate* context slots
  (`_context_slot_mappings`, `:188-193`), not the target's slots.

**SGLang** — the drafter owns a separate pool; the target's hidden states are
injected *into the draft pool*, never the target's committed cache.
- `dspark_kv_inject.py:72` `pool = self.draft_model_runner.token_to_kv_pool`.
- `:86-94` `write_target_hidden_kv(...)` writes into that draft pool.
- `eagle_worker_v2.py:194-206` `alloc_memory_pool` allocates *draft* KV pools
  (separate from the target worker's).

**Ours** — the draft runs replicated full-width with its *own* sliding-window ring,
and its KV is scattered into that ring, never the committed cache.
- `spark_dsv4_resident_decode_stage_module.c:359-368` "the replicated draft ring (one
  sliding window per draft layer)"; `:1647` "one 128-slot sliding window".
- `:4474-4507` "the target position's kv enters this draft layer's ring at
  anchor_position % window" — `SparkDsv4LaunchCacheScatter` into
  `state->dspark_ring_bf16` (the draft ring), not the committed cache.

**Verdict:** same isolation direction — speculative KV lands in a *draft-private*
buffer; only the accepted tokens are committed to the target cache.

---

## 2. Multi-row verify causality

**vLLM** — the draft attention is *non-causal* (one parallel block), the verify is
the target's *causal* pass over the expanded batch.
- `dflash/speculator.py:46-47` `num_query_per_req = 1 + num_speculative_steps`
  (bonus + N drafts in one query block).
- `:104-112` `attn_vllm_config` sets `use_non_causal`; `:287` `causal=False`.
- `speculator.py:227-294` draft metadata sets
  `draft_seq_lens = seq_len + step` (clamped to `max_model_len`).
- Verify: `gpu_model_runner.py:2333` `num_sampled_tokens = num_draft_tokens + 1` —
  the target runs `1+N` logits per request, each row attending causally to the
  prefix + preceding verify rows.

**SGLang** — same shape: the verify *extends* the sequence and runs one causal
target forward.
- `dspark_verify.py:253-261` `batch.seq_lens_cpu = seq_lens_cpu_backup + verify_w`
  then one `forward_batch_generation(..., is_verify=True)`.

**Ours** — the verify is a single batched frame of `1 + SPEC_STEP` rows
(anchor + SPEC_STEP drafts); the draft attention is non-causal.
- `spark_dsv4_resident_decode_stage_module.c:2393-2485` "SPEC_STEP drafts ... replay
  one batched verify frame"; `:2453` drafts land at `host_tokens[1+row]` (anchor at
  row 0, drafts at rows 1..N); `:2481-2482` `dspark_verify_rows` / `dspark_verify_accept`.
- `:4411` "draft attention: window ring + block kvs, non-causal".

**Verdict:** match — draft is one parallel non-causal block; verify is a single
causal target pass where each verify row's window ends at its own position.

---

## 3. Compressed-KV / compressor state during speculation

**vLLM DFlash** — the draft does *not* re-derive the prefix; it pre-computes a
*compressed* "context KV" from the target's hidden states and stores it in the
draft's separate context slots (eagerly, outside the CUDA graph because the shape
varies per step).
- `dflash/speculator.py:73-78` `context_positions` "Context positions for the K/V
  precompute ... processed by the model's precompute_and_store_context_kv".
- `:421-438` "Pre-insert context K/V into the cache ... Each layer uses the context
  slots of its own kv-cache group."

**SGLang DSpark** — a sliding-window ("SWA") ring is the compressed state; only
*committed* (accepted) tokens get their KV injected, so the draft's speculative
state is rolled back by a commit gate.
- `dspark_kv_inject.py:137-172` `_unified_inject_loc`: "SWA window: only the last
  win tokens per req land in the ring ... commit gate: uncommitted verify tokens
  (col >= commit_len) are dropped."
- `:167-171` `committed = (col < commit_lens)` then `loc = where(committed, loc, -1)`.

**Ours** — the CSA compressor (ratio 4; HCA ratio 128) is the compressed state. It
accumulates a running per-lane state and emits a boundary slot *behind* the raw
token position; the draft itself uses a separate 128-slot ring.
- `spark_dsv4_resident_decode_stage_module.c:2642-2678` compressor: "for boundary rows
  the emitted slot gets norm, rope at the group start position ... and lands at
  position/ratio behind the window"; `SparkDsv4LaunchCompressStep` +
  `SparkDsv4LaunchKvEmission` (`ring_slots=SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS`,
  `rotate`).
- `:2316-2317` / `:2470-2471` `row_emit_positions = position + 1 - ratio` (the
  compressed slot is the group-start position, written when the group's last token
  is processed) — this is our *write direction*.

**Verdict:** both external engines keep the draft's compressed context in a
draft-private buffer and commit it only for accepted rows; ours does the same via
the draft ring + the compressor's boundary-slot emission. The concrete write
direction is the subject of the next section.

---

## 4. Completion / continuation accounting (partial-accept bursts)

**vLLM** — acceptance is a *prefix* stop: match left-to-right, stop at the first
reject, then always append one bonus token; the per-position Gumbel key makes draft
and target noise agree.
- `rejection_sampler_utils.py:541-667`: `accepted_length` accumulates matched drafts;
  `verifying = accepted` (first reject stops); `:667` stores `accepted_length`.
- `:569` `u = tl_rand32(seed, pos)` — acceptance noise is keyed by *position*.
- `:1087-1188` resample the rejected/bonus token; `num_sampled = accepted + 1`.
- `rejection_sampler.py:276-282` `get_num_sampled_and_rejected`.

**SGLang** — `correct_len` (matched drafts) + `bonus`; commit length is the bonus
plus the matched prefix.
- `dspark_verify.py:623-658` `accept_greedy_triton` -> `correct_len, bonus`;
  `finalize_accept_lens_triton` -> `commit_lens`, `new_seq_lens = prefix + commit_lens`;
  `commit_lens = correct_len + 1`.

**Ours** — greedy Leviathan prefix match, emitted burst = 1 (bonus) + accepted, and
the lane advances by that exact amount (never 0).
- `spark_dsv4_resident_decode_stage_module.c:3577-3587` `accepted` = first index where
  `host_output_token_ids[i] != dspark_host_draft_tokens[i]`.
- `:3589-3597` `accepted_token_count = tokens_per_sequence = emitted_token_count =
  1 + accepted`; `lane_next_positions[0] += 1 + accepted`; `cache_lanes[0].
  context_token_count += 1 + accepted`.

**Verdict:** match — both engines and ours emit `1 + accepted` (bonus + matched
prefix), and the continuation advances by exactly that count. This is the
"1+accepted" contract our lease chain must honor (the earlier
CLIENT_LEASE_DISCONNECT bugs came from advancing the lease by a *different* count,
not from this module-side emit).

---

## CSA-write direction — does our fork match?

Both external engines write the speculative compressed state **from the target's
hidden states *down* into a draft-private buffer**, and commit it only for accepted
rows:

- vLLM: target hidden states -> `precompute_and_store_context_kv` -> draft's
  *separate* context slots (`dflash/speculator.py:421-438`).
- SGLang: target hidden states -> `write_target_hidden_kv` -> draft's SWA ring,
  gated by `commit_lens` (`dspark_kv_inject.py:86-94`, `:167-171`).

Ours matches that direction in the sense that matters for correctness:

1. **Speculative KV never lands in the committed cache** — the draft writes only its
   own 128-slot ring (`spark_dsv4_resident_decode_stage_module.c:4474-4507`).
2. **The committed compressed state is the CSA compressor**, which writes a boundary
   slot at the *group-start* position (`position + 1 - ratio`,
   `:2316-2317`/`:2470-2471`), i.e. the compressed slot is written *backward* from
   the triggering token to the group boundary — the same "compressed representation
   lags the raw token stream" shape as vLLM's context-KV precompute and SGLang's
   SWA ring.

The one structural difference to note: vLLM/SGLang derive the draft's context as a
*projection of the target's hidden states* (EAGLE-style), whereas our DSpark draft
is a full replicated transformer that reads the committed block KV + its own ring
(`spark_dsv4_resident_decode_stage_module.c:4411-4422`) and receives the anchor
context via the target-layer taps. The isolation + commit-gate direction is the
same; the *source* of the draft context (hidden-state projection vs block-KV read)
differs by design, not as a bug.

**Bottom line:** our fresh fork's CSA-write direction (speculative/compressed KV into
a draft-private buffer; commit only the accepted prefix) is consistent with both
battle-tested engines. The lease/continuation accounting (`1 + accepted`) also
matches. No direction reversal is needed.
