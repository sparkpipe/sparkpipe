# Rung 3 — DSpark drafter adoption (mined + port plan)

Source: `0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark` (our exact hardware: DGX Spark
GB10 / SM121, 128 GiB unified / 273 GB/s, vLLM 0.27.1-aarch64). It is a RECIPE repo
(vLLM stock + a drafter), not a custom-kernel repo — the 20–58 tok/s comes entirely from
speculation + prefix caching, not a faster GEMM.

## 1. What we can adopt (ranked)

1. **Pull the drafter weights now** — public, ungated, BF16 1.36B, 2.6 GB. Closes the
   draft-weight gate; nothing to wait for.
2. **Block-of-7 emission + probabilistic sampling + Markov/confidence heads** — the
   spec-winning design (3.3× cheaper drafts than MTP).
3. **DFlash forward port + target-hidden taps** — the one real chunk of new module work.
4. **Re-tune k (7 vs 14) on our FP8 + concurrency** — k does NOT transfer across
   quantization (their own data).

## 2. Drafter facts

- Model: `Doopeworld/Qwen3.8-27B-DSpark-vLLM` (public), byte-identical to
  `RadixArk/Qwen3.8-27B-DSpark` (only config `architectures` renamed).
- Production: ported from TorchSpec PR #129, adapted from DeepSeek DeepSpec
  (SpecForge block-diffusion drafter training).
- Geometry (config.json): 5 full-attention layers, hidden 5120, 40 Q / 8 KV heads ×
  head_dim 128, FFN intermediate 10240, vocab 248320, block_size 7,
  target_layer_ids [4,16,28,40,52], mask_token_id 248077, markov_rank 256,
  confidence_head_with_markov true.

### Tensor inventory (62 tensors, all BF16, standard shapes)

- 5 layers × 11: q_proj[5120,5120] k_proj[1024,5120] v_proj[1024,5120] o_proj[5120,5120]
  q_norm[128] k_norm[128] gate/up[10240,5120] down[5120,10240] input/post layernorm[5120]
- fc.weight[5120,25600] — projector: 5 target taps × 5120 → 5120
- markov_w1[248320,256] + markov_w2[248320,256] — low-rank bigram bias
- confidence_head.proj[1,5376] + bias[1] — 5120 hidden + 256 markov → 1
- norm.weight[5120] + hidden_norm.weight[5120] — final norms

KEY: the drafter has NO embed_tokens and NO lm_head — it SHARES the target's token
embedding + lm_head (vLLM's Qwen3DSparkForCausalLM ties them online). Our port reuses
the resident module's existing embedding + head; we only add the 5-layer backbone +
projector + markov + confidence heads. ~1.36B params (5×~220M layer + 131M fc + 127M
markov).

## 3. Why DSpark pays where our MTP D=2 doesn't (the arithmetic)

- Both spec methods verify the same way (target re-runs the proposed block in ONE
  forward; first mismatch discarded). The difference is DRAFT COST, not verification.
- Cost per draft token: MTP 0.153 vs DSpark 0.046 (3.3×). MTP re-runs the in-checkpoint
  head SEQUENTIALLY and pays a full lm_head over ~150K vocab PER draft token (our A4
  finding). DSpark emits the whole 7-token block in ONE 5-layer forward (mask-token
  noise stream + anchor sampling).
- Acceptance (fresh gen): MTP 79.0/50.3/28.9% (k=3/8/15, mean 2.96 tok/pass);
  DSpark k=7 fresh 31.7% (mean 7.91 tok/pass), edit 98.6%. DSpark accepts FEWER tokens
  per pass yet is 46% faster — each pass costs far less.
- Adopt: block_size 7 (one-shot emission, not sequential MTP); probabilistic sampling
  (+23% vs greedy); confidence head (per-position acceptance predictor); the k-vs-
  workload rule (fresh ~30% acceptance ⇒ deep cheap drafts beat shallow expensive ones).

## 4. Kernel side (nothing to adopt)

The 7.88 baseline is vLLM STOCK FP8 decode. Our rung-2 native e4m3x4 decode (8.00)
already beats it. The 20–58 range is speculation + prefix caching, not GEMM. One free
lever: --enable-prefix-caching (14–22× shared-prefix prefill; off-by-default on hybrid).

## 5. Port design + file plan

DSpark = 5-layer full-attention drafter emitting a 7-token block in ONE forward.

1. tools/qwen36_dspark_stagepack.py (new) — pack the 62 tensors → .qwen36sp (reuse
   spark_pack_common + the stagepack wire format; full-attention only).
2. modules/.../source/spark_qwen36_dspark_format.h (new) — drafter pack layout.
3. modules/.../source/spark_qwen36_resident_decode_stage_cuda.cu — DFlash kernels:
   dual-source GQA attention + 5-layer block forward + lm_head + markov/confidence.
4. modules/.../source/spark_qwen36_resident_decode_stage_module.c — tap target hiddens
   @{4,16,28,40,52} + DSPARK_DRAFT_AFTER Execute path.
5. modules/.../source/spark_qwen36_serving_adapter.c — swap MTP phase-one for DSpark
   block draft + probabilistic left-to-right Markov sampling.
6. modules/.../include/sparkpipe/spark_qwen36_resident_decode_stage_firmware.h — DSpark
   frame context (block view + 5 tap buffers).
7. manifest/env — drafter-pack path + spec method switch (mtp|dspark).

Data flow: target decode → capture hiddens @{4,16,28,40,52} → projector fc → drafter
5× (dual-source attn: Q=own, K/V=cat(taps,own), non-causal 7-block) → lm_head 7 logits
→ Markov bias L→R → probabilistic sample 7 → SPECULATIVE_VERIFY 7-token prefill →
accept/replay.

## 6. Reference files

vLLM vllm/model_executor/models/qwen3_dspark.py (Qwen3DSparkForCausalLM, Markov/conf
heads) + qwen3_dflash.py (DFlashQwen3ForCausalLM block-diffusion forward); HF drafter
dspark.py / dflash.py (dual-source attention + anchor sampling). Weights on spark3 at
/home/spark3/extnvme/models/hf/Doopeworld/Qwen3.8-27B-DSpark-vLLM/.
