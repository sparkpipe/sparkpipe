--- lobanov 2026-06-27T17:13:02Z ---
Initial quick benchmarking shows that it might be harder to achieve than it seems on the face of it with a fully-local engine.

Generation baseline on M5 Max on short-ish prompts is 28 ms per token. With current 2-token MTP enabled it's 2ms for prediction and 33ms for fast verification pass. That in practice only gains on average 6-7% because predictions are not always correct. When `--quality` is enabled, a different slower exact verifier kernel is used, which verifies two tokens in 67ms. In practice this gives 28% *reduction* of inference speed over the baseline. Under realistic assumptions of prediction quality and assuming exact verification cost scales linearly with number of tokens there is no gain.

Three avenues to explore:
1) Can exact verifier be made to scale sublinearly with increased number of predicted tokens? Very hard, especially on Metal. It's a research project.
2) Use a different approach for batch verification altogether, e.g. distribution-exact not logit-exact. Research project again, but perhaps easier.
3) Let go of exact verification and use existing batched verifier. It's algorithmically identical, but numerically different, so model precision can suffer. By how much is hard to tell, it would need benchmarking. This is more practical, because we are accepting some quality loss with quantization anyway, and this may allow to trade speed for quality.

--- daaain 2026-06-27T18:31:46Z ---
This sounds incredible! So you already have a working implementation @lobanov?

I've been discussing it with Claude after cloning both repos and sounds like batching could be a way to go (there's a bit at the end in response to your comment):

---

# DSpark on ds4 (DwarfStar): feasibility, tensor map & RAM budget

**Status:** investigation only, no code written yet.
**Target:** DeepSeek-V4-Flash + DSpark draft, Metal backend, Apple-silicon single-user (reference machine: 128 GB M3 Max, asymmetric q2).
**Source artefacts:** DeepSeek released the DSpark checkpoints and the DeepSpec training repo on 27 Jun 2026, MIT-licensed. All claims below are traceable to the files listed under *Evidence* at the bottom.

---

## TL;DR

- **The weights exist and fit the use case.** `deepseek-ai/DeepSeek-V4-Flash-DSpark` is the base checkpoint + a draft module under the `mtp.*` namespace. MIT.
- **It's a bolt-on, not a requant.** The base 43 layers, embedding and LM head are frozen and shared by reference. The existing q2 Flash GGUF is untouched. We only convert the new `mtp.*` tensors.
- **The draft is 3 full MoE layers** (not dense, not tiny) + a vanilla Markov head + a 1-logit confidence head. That's where the RAM cost lives.
- **The draft backbone is the same V4 layer type we already run** (HC + MoE), but with *uncompressed sliding-window attention* (no compressor, no indexer) — which makes the Metal path simpler than feared.
- **RAM is the real constraint, not correctness.** Full draft at q8 ≈ ~20 GB; at q4 ≈ ~11 GB; at iq2 ≈ ~6 GB. We already resident-load MTP-1 (~1 layer), so the *incremental* cost is ~2 layers. Recommendation: ship behind `--draft dspark`, default the draft to q4, reclaim RAM by streaming more **base** experts (never the draft).
- **The 6–7% MTP-1 ceiling is an acceptance artefact, not a verifier wall.** Don't fight the exact verifier (it's slower-than-baseline by construction); use the **lossless batched verifier** + rejection sampling and measure its drift against the q2 noise floor. The economics are bandwidth-bound, so they largely hold on M3 Max too. See *Verifier economics* below.

---

## Resolved questions (all statically verifiable)

| # | Question | Answer | Evidence |
|---|----------|--------|----------|
| 1 | Backbone: plain transformer or our HC layer? | **HC + MoE V4 `Block`** — `DSparkBlock(Block)`. Reuses our layer machinery. | `model.py`: `class DSparkBlock(Block)` |
| 2 | Draft depth | **3 stages** (`n_mtp_layers: 3`) vs base model's **1** (`num_nextn_predict_layers: 1`) and the paper's 5. | `inference/config.json`; base `config.json` |
| 3 | Block size γ | **5** | `dspark_block_size: 5` |
| 4 | Markov head variant | **Vanilla** — `markov_w1` (V×256) + `markov_w2` (256×V) only. No gate, no RNN. | `model.py`: `class DSparkMarkovHead` |
| 5 | Markov rank r | **256** | `dspark_markov_rank: 256` |
| 6 | Confidence head | `Linear(dim + 256 → 1)` + sigmoid; scalar per position. | `model.py`: `class DSparkConfidenceHead` |
| 7 | Draft FFN: MoE or dense? | **Full MoE** — 256 routed experts (fp4) + 1 shared, per stage. | `DSparkBlock` inherits `Block.ffn = MoE(...)` |
| 8 | Draft attention | **Uncompressed sliding-window**, `compress_ratio == 0` asserted. No compressor, no indexer. | `model.py`: `class DSparkAttention` (`assert self.compress_ratio == 0`) |
| 9 | Intra-block attention | **Bidirectional** within the γ block, expressed as an index list (not a triangular mask): every block position sees the window + all block positions. | `get_dspark_topk_idxs` |
| 10 | KV injection source | Mean-pooled hidden states of target layers **[40, 41, 42]** (pooled over the hc copies). | `dspark_target_layer_ids: [40,41,42]`; `model.py`: `if i in self.target_layer_ids: main_hiddens.append(h.mean(dim=2))` |
| 11 | Injection wiring | `main_x = main_norm(main_proj(concat(h40,h41,h42)))`, `main_proj: Linear(12288→4096)`; enters each draft layer as one KV row in its window. | `model.py`: `DSparkBlock.forward_embed` |
| 12 | Block input tokens | `[anchor, noise, noise, noise, noise]`, noise id **128799**. | `dspark_noise_token_id: 128799`; `forward_embed` |
| 13 | Embedding / LM head | **Shared & frozen** from the base model (tied), no separate weights. | `model.py`: `mtp[-1].embed = self.embed; mtp[-1].head = self.head` |
| 14 | Base precision | fp8 e4m3 (ue8m0 scales, 128×128 blocks); experts fp4. | base `config.json` `quantization_config`; `inference/config.json` |

**Correction to an earlier note:** an earlier read suggested the DSpark config altered the base compression schedule for layers 40–42. It does not. The three extra `0` entries in DSpark's `compress_ratios` are the *draft layers themselves* (indices 43/44/45), which run uncompressed. The base 0–42 schedule is unchanged.

---

## Loader spec — `mtp.*` tensor map

Three stages: `mtp.0`, `mtp.1`, `mtp.2`.

**Every stage** (maps onto our existing `ds4_layer_weights`):
`attn.{wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink}`,
`ffn.{gate.weight, gate.bias, experts.{0..255}.{w1,w2,w3}, shared_experts.{w1,w2,w3}}`,
`attn_norm`, `ffn_norm`, `hc_attn_fn/base/scale`, `hc_ffn_fn/base/scale`.

**Stage 0 only:** `mtp.0.main_proj` (`Linear 12288→4096`), `mtp.0.main_norm` (RMSNorm 4096).

**Last stage (`mtp.2`) only:** `norm`, `markov_head.markov_w1` (V×256), `markov_head.markov_w2` (256×V), `confidence_head.proj` (`Linear 4352→1`), `hc_head_fn/base/scale`.

**Not present (shared from base):** token embedding, LM head.

### Forward deltas vs our current MTP-1 path
This is a *new* draft subgraph, not a parameter swap into the existing MTP path:
- **Input projection differs.** MTP-1 uses `e_proj`/`h_proj` (embed + previous hidden). DSpark uses `main_proj`/`main_norm` over the concatenation of three target-layer hiddens, plus noise-token embeddings for the block. → needs a capture hook on layers 40/41/42 during the main forward.
- **Parallel bidirectional block** of γ=5 vs the current recursive single-token chain. Reuse the prefill multi-row path; supply the `get_dspark_topk_idxs`-equivalent visibility (window + all block positions). Since γ=5 ≪ window 128, compute the block densely; never touch the compressed/indexer path.
- **Output head** is new: hc_head collapse → shared LM head → per-position Markov bias loop (r=256) → confidence sigmoid → confidence prefix-cut. The cut generalises our existing `mtp_margin` static threshold.

Verify/rollback reuses what we already have: `spec_frontier_snapshot/restore`, `mtp_n_raw`, `DS4_MTP_KEEP_ACCEPTED` (equivalent to the reference's `past_key_values.crop(start)`).

---

## Quantisation: do we requantise the FFN? No.

- The base model is inert. The existing q2 Flash GGUF stays byte-for-byte. We convert **only** the `mtp.*` tensors.
- In the checkpoint, the draft's experts are **fp4** (same family as base), projections fp8, small heads bf16/fp32. We pick the GGUF quant for the draft independently of the base.
- **Acceptance caveat (affects speed, never correctness):** the draft was trained against the *fp8* target's layer-40/41/42 hiddens. Running the base at q2 perturbs those hiddens → lower accepted length τ. Spec decoding stays lossless regardless. Mitigation that costs nothing: pair DSpark with the existing `q2-q4-imatrix` GGUF (last 6 layers at q4) so the injection sources are higher precision. The shared head is already kept high (`OutQ8`).

---

## RAM budget (the part that matters for a usable laptop)

The draft is three full V4 MoE layers. Per layer ≈ 6.5 B params (256 routed experts dominate: 256 × 3 × 4096 × 2048 ≈ 6.44 B). Three layers ≈ **~19.7 B params**. Stored size by quant:

| Draft quant | Stored size (3 layers) | Notes |
|---|---:|---|
| q8 | **~20 GB** | the "19 GB" figure; highest τ |
| q4 / fp4 | **~11 GB** | recommended default balance |
| iq2 (match base experts) | **~6 GB** | tightest; measure τ loss |

Markov + confidence + norms are negligible (~70 MB total).

**Reframe the increment.** We already resident-load the base model's MTP-1 (1 layer ≈ 6.5 B). The honest *incremental* cost of DSpark over what ds4 loads today is ~2 extra layers:

| Scenario | Draft RAM | Δ over current MTP-1 (≈1 layer) |
|---|---:|---:|
| DSpark @ q8 | ~20 GB | ~ +13 GB |
| DSpark @ q4 | ~11 GB | ~ +5 GB |
| DSpark @ iq2 | ~6 GB | ~ +0 GB (≈ replaces MTP-1 footprint) |

So at iq2 the draft is roughly RAM-neutral versus the MTP-1 we already carry; q4 is a ~5 GB increment; only q8 costs the full headroom.

**MoE means stored ≠ bandwidth.** Only ~6 of 256 experts fire per position, so keeping the draft at higher precision costs *RAM* but only a modest *bandwidth* hit per draft step (the δ in the speedup model). The trade is "spend resident RAM to protect τ," not "spend bandwidth."

**KV cache headroom is mostly safe.** V4's compressed KV is small by design, and the draft's own KV is its 128-row sliding window × 3 layers — negligible. The squeeze is the draft *weights*, not draft KV.

### Levers to stay laptop-usable

1. **Make it opt-in.** Gate behind `--draft dspark`. RAM-constrained or background-only users keep MTP-1 and pay nothing.
2. **Default the draft to q4** (~11 GB, ~+5 GB over MTP-1). Offer iq2 (~RAM-neutral) for tight setups and q8 only when RAM is flush. Treat the draft-quant-vs-τ curve as a measured knob, not a guess.
3. **Reclaim RAM from the base, not the draft.** ds4 already streams base experts (`--ssd-streaming --ssd-streaming-cache-experts NGB`). Stream *more* of the cold **base** experts to free room for an in-RAM draft. The base is amortised over τ accepted tokens, so its streaming penalty is diluted by the speedup. **Do not** SSD-stream the draft — it's on the latency-critical path (the `T_draft` term); paging draft experts per block would erase the win.
4. **Wire a hard memory ceiling.** Keep the draft buffers inside whatever `iogpu`/wired-limit budget we set, so enabling DSpark can't tip the machine into swap and make it unusable.

---

## Verifier economics & the speedup ceiling

This section responds to a maintainer benchmark showing MTP-1 gaining only 6–7% on the fast path and *losing* 28% with `--quality` (exact verifier). The conclusion: the ceiling is acceptance rate, not the verifier — and the verifier choice has a clean, lossless answer.

- **The 6–7% is an acceptance-rate artefact.** The fast path reproduces exactly: 2 ms draft + 33 ms verify(2) = 35 ms ÷ τ≈1.33 ≈ 26.3 ms vs 28 ms baseline ≈ 6.5%. The bottleneck is τ — and τ is precisely what DSpark's semi-autoregressive draft + confidence trim raise. MTP-1 drafts one token of low acceptance; that's the wall, not the kernel.
- **The exact verifier is underwater by construction.** 67 ms / 2 tokens = 33.5 ms/token, already above the 28 ms sequential baseline. Even perfect prediction (τ = γ+1) floors per-token cost above baseline, so *no* acceptance rate rescues it. Avenue 1 (sublinear exact verify on Metal) chases a kernel that loses at τ→∞. Drop it.
- **The fast/batched verifier is bandwidth-bound — which is why speculation can win.** The benchmark shows it: the 2nd verified token adds only ~4 ms to a ~29 ms pass — nearly free, because weights stream once regardless of γ. That sublinearity *is* the speculative thesis. The exact kernel is ~linear (compute-bound) and cannot amortise.
- **"Exact" vs "fast" is reference-drift, not lossless-vs-lossy.** Both are lossless: rejection sampling (Chen et al. 2023 / Leviathan et al. 2023) yields exact samples from *whatever logits the verifier produces*. The batched verifier is distribution-exact w.r.t. its own numerics; it only drifts from the decode kernel ds4 validates against official DeepSeek logits, via fp accumulation order. So avenues 2 and 3 collapse into one practical answer — **batched verifier + rejection sampling** — and the real question is "how far does batched drift from the reference vs how far q2 already drifts." Almost certainly the latter dominates by orders of magnitude (rounding-order vs multi-bit). **Measure it** with `gguf-tools/quality-testing`: score "q2 + batched-verify spec" against official continuations next to "q2 + exact." If batched sits inside the q2 noise floor, ship it.

**Where DSpark clears break-even** (M5 Max, fast path, γ=5; assumes ~6 ms draft, ~48 ms verify(5)):

| τ (accepted) | per-token | vs 28 ms baseline |
|---:|---:|---|
| 2 | ~27 ms | ~break-even |
| 3 | ~18 ms | **~35–40% faster** |
| 4 | ~13.5 ms | **~2×** |

Break-even is τ≈2; real wins need τ≥3. The confidence prefix-cut also bounds wasted verification (the "predictions aren't always correct" loss) by not verifying the low-survival suffix — biggest help on chat.

### M3 Max vs M5 Max: does slower prompt processing change the picture?

**Short answer: predominantly bandwidth-bound, so the speculative *speedup ratio* is largely preserved on M3 Max** — most of the % win survives on a slower absolute baseline. M3's weaker compute bites only where the work is compute-bound, and those are mostly the parts we're abandoning or that DSpark already manages.

From ds4's own bench (q2, short context): M3 Max decode **37.5 ms/tok**, M5 Max **29.2 ms/tok** — a **1.28× gap**, tracking memory bandwidth (~400 GB/s vs higher). Prefill gap is wider: **1.49× short, 1.85× at ~12 k**. M5's compute advantage exceeds its bandwidth advantage, and that asymmetry is the whole story:

- **Per-token economics scale with bandwidth.** Draft and fast-verify both stream weights once, both ~1.28× the baseline. Speedup is a *ratio* (τ ÷ cycle/baseline); numerator and denominator scale together → **% speedup is roughly bandwidth-invariant.** Worked at τ=3: ~1.65× on M5 vs ~1.57× on M3 — close.
- **The marginal verify-token is compute, and M3 has less of it.** The ~4 ms/extra-token on M5 becomes ~6–7 ms on M3 (compute scales 1.5–1.85×). The win erodes *modestly*, and the erosion **grows with γ and context** — larger blocks / long context push the verify pass toward compute-bound sooner on M3. Mitigation: the confidence trim shrinks effective γ exactly when acceptance is low, capping compute-bound verify width. **The trim is more valuable on M3 Max.**
- **The exact verifier is even more hopeless on M3.** Being compute-bound, it scales with M3's worse prefill throughput — the −28% seen on M5 would be a deeper slowdown on M3. Stronger reason to use the batched verifier.
- **Prompt prefill is slower on M3** (1.49–1.85×) but that's a one-time per-request cost, orthogonal to per-token decode.

**Verdict:** RAM-bandwidth-bottlenecked in the decode/fast-verify regime, so DSpark's value proposition survives on M3 Max with a slightly thinner margin. The compounding risk to watch is **q2-lowered τ × M3 compute erosion on chat specifically** — if q2 drops chat τ below ~2.5, M3 could be break-even there while code/math (higher τ) stay clearly positive. One more reason τ-under-q2 is the first number to measure.

---

## Still needs a runtime check (not blocking the loader)

- **τ under q2 base.** The acceptance-rate hit from q2-perturbed injection hiddens is the one number neither the paper (GPU/fp8) nor static analysis gives us. Produced by the step-3 isolated-backbone logit check + a short A/B vs MTP-1.
- **Non-causal block mask inside our window-attention kernel.** Expressible (the kernel already takes explicit `allowed_mask`s for the indexer), but confirm the γ=5 dense block composes cleanly with the raw window.
- **safetensors index** only needed to locate shards / confirm stored dtypes for the converter — no architectural unknowns remain.

---

## Suggested sequencing (each step independently verifiable)

1. Converter: `mtp.*` safetensors → ds4 GGUF (`ds4_dspark_weights` struct). *Verify:* tensor round-trip.
2. Draft backbone in Metal, validated in isolation against the PyTorch reference logits. *Verify:* max-abs logit diff < 1e-2 @ fp16.
3. Markov + confidence heads. *Verify:* identical draft tokens / prefix-cut at temp 0.
4. Wire into the speculative loop behind `--draft dspark`, reusing frontier/rollback. *Verify:* **token-identical** to non-spec greedy decode (spec decoding is lossless — any divergence is a bug; free oracle).
5. Bench with `speed-bench` + `DS4_MTP_TIMING`, A/B vs MTP-1 at short and ~12 k context, code/math vs chat. Sweep draft quant (q8/q4/iq2) for the RAM-vs-τ curve.

---

## Evidence sources

- Draft module reference (authoritative for structure & tensor names):
  `inference/model.py` — classes `DSparkBlock`, `DSparkAttention`, `DSparkMarkovHead`, `DSparkConfidenceHead`, `Transformer.forward_spec`, `get_dspark_topk_idxs`.
  `huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark` → `inference/`
- Hyperparameters: `inference/config.json` (keys: `n_mtp_layers`, `dspark_block_size`, `dspark_markov_rank`, `dspark_target_layer_ids`, `dspark_noise_token_id`, `compress_ratios`, `expert_dtype`, `dtype`).
- Base architecture / quant: `deepseek-ai/DeepSeek-V4-Flash/config.json` (`num_nextn_predict_layers: 1`, `quantization_config`, `compress_ratios`).
- Algorithm reference (training-side, qwen3/gemma4 only — not V4): `github.com/deepseek-ai/DeepSpec`, `deepspec/modeling/dspark/markov_head.py`, `eval/dspark/draft_ops.py`, `eval/dspark/confidence_head.py`. Useful for the loss/calibration story; the V4 module itself is in the HF `inference/` code above.
- Paper: DSpark, arXiv 2606.19348 (semi-autoregressive generation §3.1; confidence-scheduled verification §3.2 — note the hardware-aware scheduler is a multi-request serving optimisation and degenerates to a single-request prefix-cut for our use case).
- ds4 anchors for reuse: `ds4.c` — `metal_graph_eval_mtp_draft_from_hc`, `ds4_session_eval_speculative_argmax`, `ds4_mtp_weights`, `spec_frontier_*`, `DS4_MTP_KEEP_ACCEPTED`, `mtp_margin`. Bench: `speed-bench/`. Quant tooling: `gguf-tools/`.

*All checkpoint files dated 27 Jun 2026, MIT-licensed. Figures (per-layer ≈6.5 B params; q8≈20 GB / q4≈11 GB / iq2≈6 GB) are derived from the config dims and standard bpw; treat as estimates pending the safetensors index and a real conversion.*


--- lobanov 2026-06-27T20:29:25Z ---
Thanks @daaain, Claude is spot on. I don't have an implementation yet, only a benchmarking rig. Rejection sampling with batch verifier is what I also got to after some deliberation. I'll try to get it to work.

I've been pondering building a DFlash+DDTree spec decoding for ds4 for some time, but what stopped me (aside from limited spare time) is lack of credible DFlash drafter for the model. Training one would take weeks on a MacBook, or $100s of cloud compute. This is no longer the case as DSpark model variant ships with one now, so no excuses!

--- rosmur 2026-06-29T00:55:17Z ---
@antirez curious to hear your thoughts here. Feels like this aligns closely to the mission of the project

--- lobanov 2026-06-29T08:57:31Z ---
Unfortunately my first attempt to make DSpark work (in partnership with GLM-5.2) in ds4 don't look encouraging. TLDR is q2-imatrix model and q4 quant of dspark tensors produce low-quality drafts which rarely get accepted. Average acceptance is 1.5-2 tokens only, which causes an average slowdown of 5% of decode at any context length. You can read more about my results [here](https://github.com/lobanov/ds4/blob/dspark/issue468/23_final_verdict_longctx.md).

There are some reusable artifacts available: DSpark drafter GGUF (I can upload to HF ~10.7GB) and Metal DSpark drafter, CPU FP32 oracle for comparing results, all in the same branch if anyone wants to look at it.

Some avenues to explore remain, but it could be that DSpark architecture doesn't lend itself well to be used in a quantized environment.

**Edit**: I forgot to share earlier that simply reusing existing batch verifier for DSpark is a no-go. On a full `ds4-eval` with `--mtp` enabled it loses 7 points compared to default greedy route. This necessitates verifier rework. Most promising approach is to build a rejection sampling-based verifier as in the [original paper on speculative decoding](https://arxiv.org/abs/2211.17192). The caveat is that it's only practical when running decode with temp>0, ideally =1. This will make benchmarking harder, because of sampling variance.

--- daaain 2026-06-29T10:09:55Z ---
Would you be able to try with q8 or is that too big for 128GB? 

--- lobanov 2026-06-29T10:28:11Z ---
The problem with that is storing and loading DSpark tensors in Q8 will require +9GiB of memory, which will leave very little space for any meaningful context. The avenue I will explore next is to dequant stored Q4 drafter experts and route them through F16/F32 path with F32 accumulation to make it numerically similar to how the numpy oracle works. The oracle has shown ~2.8 token prefix acceptance, and if that holds in actual Metal kernel, then we might get a measurable ~20-30% speedup after all.

--- lobanov 2026-06-29T13:22:03Z ---
Interestingly, there are now some independently-created DFlash drafters available on HuggingFace for DeepSeek-V4-Flash, [this one](https://huggingface.co/inference-optimization/dflash-DeepSeek-V4-Flash-all-swa-muon-speculators-50k) in particular looks promising: 8 blocks, 3.6GB at full precision, good acceptances.

For the future, it is worth trying DFlash (+DDTree) as an alternative to DSpark. Even if DSpark proves to be viable in the face of precision loss due to quantization of the model or drafter, it's still a huge bolt-in even in Q4_K eating into the memory budget for a meaningful context.

--- lobanov 2026-06-29T21:22:24Z ---
I found a bug in the implementation that explained dropped acceptance rate. There's still quite a bit of work ahead, but I am cautiously optimistic about 30% gains in decode speeds with Q4_K drafter. Interestingly, the decode speed gains seem to improve as the context gets longer, not the other way around like it is with DFlash...

--- audreyt 2026-06-30T05:23:51Z ---
Opened https://github.com/antirez/ds4/pull/480 with the DSpark/DeepSpec integration work.

Short version: it adds an opt-in Metal DSpark Markov draft path, validates proposed blocks against the target model before commit, refreshes accepted rows from target hidden/raw-KV state, keeps CUDA/ROCm gated off with stubs for now, and includes target-cache/converter test coverage.

Verification in the PR body includes the DSpark runtime tests, converter self-tests, and a representative local M5 Max benchmark. The implementation is documented as correctness-gated rather than a guaranteed speedup because acceptance still depends on the base/draft quantization and prompt.

--- audreyt 2026-06-30T05:33:06Z ---
Update on https://github.com/antirez/ds4/pull/480: I initially marked the PR as draft after re-reading this thread and running extra checks. After a closer check, I marked it as non-draft again.

Caveats still stand:

- @lobanov’s measurements are the best reference point so far: q2-imatrix + q4 DSpark accepted about 1.5-2 tokens and was about 5% slower in that setup, with a later note that a bug was found and Q4_K may still be promising.
- My initial +11% number came from one synthetic numbered-line prompt. It is a useful smoke test, not a representative speed claim.
- The temp=0 code-generation run was not token-identical to plain greedy decode, but that alone is not the right correctness test here.
- The first difference was a near tie (`.` vs `,`, target-logit gap about `0.024`). Replaying the 128 accepted DSpark tokens through the target model stayed inside ds4’s existing speculative threshold (`worst_argmax_gap=0.859 <= 2.0`), and the targeted DSpark test passes with `worst_argmax_gap=0.000`.

So I no longer think that raw text diff is a merge-blocking correctness failure. I marked the PR ready for review again as an opt-in,
correctness-gated experimental DSpark path.

The remaining question is performance: representative acceptance and speed across real prompts and draft quantization choices. I would not treat the current +11% smoke result as a general speedup claim.

--- lobanov 2026-06-30T06:27:16Z ---
Great work @audreyt! Note that with rejection sampling-based spec decode the output will never be identical to temp=0 greeding decode, because it has to run with a non-zero temp. Greedy decode collapses logit distribution which is incompatible to rejection sampling.

--- machiabeli 2026-06-30T07:16:28Z ---
## End-to-end DSpark B2 rejection sampling: findings + bottleneck identification

I wired B2 rejection sampling (Chen/Leviathan 2023) into the DSpark decode loop on top of PR #480's plumbing. Results on M5 Max 128 GB, q2-imatrix base + Q4_K DSpark drafter:

### Correctness: PASS
- temp=0 greedy output is **token-identical** to non-speculative decode
- B2 is distribution-exact by construction (lossless at any temperature)
- Off-by-one in target logits alignment found and fixed (verify row[i] predicts drafts[i+1], not drafts[i])

### Performance: blocked by partial-commit replay

| Configuration | tok/s | vs baseline |
|---|---|---|
| Baseline (no DSpark) | 35.04 | — |
| DSpark full commit only (5/5 accepted) | **~55** | **+57%** |
| DSpark end-to-end (with partial replay) | 23-26 | **-26% to -34%** |

**Why:** The partial-commit path in PR #480 restores the full frontier snapshot then replays accepted tokens one-by-one through the 43-layer target model (~26ms each). This replay cost equals baseline decode cost — so partial commits can never be faster than baseline.

### The gap between component and end-to-end

@lobanov's +30-48% assumed `tok/s = committed / (draft_ms + verify_ms)` = 3.42 / (7 + 75) = ~42 tok/s. This correctly models full commits (where no replay is needed). But with ~80% of cycles being partial commits, the replay dominates:

```
actual_cycle = verify(64ms) + replay(K × 26ms)  where K = partial committed
at K=2: 64 + 52 = 116ms for 2 tokens = 58ms/tok > baseline 28ms/tok
```

### Measured: fast-partial (no replay) achieves 34.63 tok/s

I tested skipping the frontier restore entirely — using the verify pass's KV state directly. This achieved **34.63 tok/s** (~baseline). But the compressed KV has phantom entries from rejected draft tokens that corrupt subsequent attention, causing output divergence ("Args:" instead of "Parameters:").

### The fix: multi-point checkpointing

The replay exists because `spec_frontier_restore` can only restore to position 0 (before any drafts). If it could restore to position K (the partial acceptance point), replay is eliminated.

**Proposed change to `metal_graph_verify_suffix_tops`:** after processing each draft token through each layer, save the compressor frontiers (`layer_n_comp[il]`, `layer_n_index_comp[il]`) + compressed KV state. On partial commit at K, restore from checkpoint K.

This is analogous to the existing `spec_frontier_commit_prefix1` (which handles N=2 MTP-1) but generalized to K=1..5 for DSpark.

### Code available

3 commits on my `work-dspark` branch:
1. B2 rejection sampling (326 lines, log-space stable, env-gated via `DS4_SPEC_TEMP`)
2. Off-by-one fix in target logits alignment + RNG state persistence
3. Skip-logits replay + documentation

Happy to push a PR or contribute to #480 — whichever is preferred.

@lobanov @audreyt — your implementations were the foundation. The converter (audreyt), the root-cause analysis (lobanov), and the numpy oracle (lobanov) were essential.

--- machiabeli 2026-06-30T07:23:40Z ---
**Follow-up: block_size=2 is the optimal configuration for current architecture**

Testing `--mtp-draft 2` (verify 2 tokens instead of 5):

| Config | tok/s (short ctx) | tok/s (long ctx) | Correctness |
|---|---|---|---|
| Baseline | 39.69 | 31.52 | — |
| MTP-1 | 39.46 | — | — |
| DSpark block=5 | 23-26 | 20.42 | token-identical ✅ |
| **DSpark block=2** | **37.27** | **26.52** | **token-identical ✅** |

**Key findings:**
- block=2 achieves **94% of baseline** (vs 66% at block=5) — near break-even
- Acceptance rate at block=2: ~85% full commit (vs ~16% at block=5)
- Verify cost scales: 37ms (block=2) vs 64ms (block=5) — roughly linear
- MTP-1 also shows ~0% speedup on this machine/quant (39.46 vs 39.69)

**Why block=2 is better:** fewer partial commits → less replay overhead. At block=5, 84% of cycles trigger the expensive replay path. At block=2, only 15% do.

The remaining 6% gap at block=2 comes from: draft overhead (~3ms) + snapshot (~1ms) + occasional replay (~26ms on 15% of cycles). Per-token verify checkpointing would eliminate the replay entirely, pushing block=2 above baseline and making block=5 viable (+30-48% as lobanov measured).

`--mtp-draft 2` is the recommended configuration for the current ds4 architecture until per-token checkpointing is implemented.

--- machiabeli 2026-06-30T07:36:38Z ---
**Update: +8.5% speedup achieved on structured output (token-identical, lossless)**

Testing across prompt types reveals DSpark speedup is workload-dependent:

| Prompt | Baseline | DSpark | Speedup | Correctness |
|---|---|---|---|---|
| **JSON structured** | 38.95 | **42.27** | **+8.5%** | token-identical ✅ |
| Code generation | 39.08 | 32.25 | -17.5% | token-identical ✅ |
| Numbered lists | 39.13 | 31.50 | -19.5% | token-identical ✅ |

M5 Max 128GB, q2-imatrix base + Q4_K DSpark drafter, block_size=5, B2 rejection sampling, greedy (temp=0).

**Why structured output is faster:** JSON generation has nearly 100% full 5/5 draft acceptance (the repetitive `{"name": ..., "capital": ..., "population": ...}` pattern is highly predictable). Full commits avoid the partial-commit replay cost entirely → verify(64ms) / 5 tokens = 12.8ms/tok vs baseline 25.7ms/tok.

**Why code/lists are slower:** Creative output has ~2.5 acceptance → frequent partial commits → replay cost (~26ms/tok) eats the verify amortization.

This confirms DSpark is viable on Apple Silicon for structured workloads. The per-token replay architecture is the bottleneck for creative output — the per-token verify checkpointing proposed earlier would unlock the full +30-48% across all workloads.

--- machiabeli 2026-06-30T08:07:01Z ---
**Full workload acceptance matrix — 11 prompts tested (M5 Max 128GB, Q2 base + Q4K DSpark)**

| Prompt Type | Baseline | DSpark | Ratio | Notes |
|---|---|---|---|---|
| **JSON (20 items)** | 38.95 | **42.27** | **1.085** ✅ | Highly repetitive `{"name":..., "capital":..., "population":...}` |
| **Markdown table** | 42.54 | **45.21** | **1.063** ✅ | Repetitive `| Rank | Country | Area |` rows |
| JSON (50 items) | 39.15 | 36.26 | 0.926 | Acceptance decays with length |
| TypeScript types | 46.49 | 41.06 | 0.883 | Structured but diverse field names |
| CSV | 46.06 | 40.86 | 0.887 | Structured data, variable values |
| Code (fibonacci) | 39.08 | 32.25 | 0.825 | Creative logic |
| Code (web scraper) | 39.08 | 32.25 | 0.825 | Creative code |
| Numbered list | 39.13 | 31.50 | 0.805 | Natural language descriptions |
| YAML | 42.84 | 34.15 | 0.797 | K8s config, many unique values |
| Counting (1-100) | 69.73 | 47.32 | 0.679 | Fast baseline, high verify overhead |
| SQL INSERTs | 38.49 | 24.96 | 0.649 | Creative data values |

**Pattern:** Speedup ONLY on short, highly repetitive structured output (JSON, tables) where nearly 100% of draft cycles achieve 5/5 full commit. All other workloads are net-negative due to partial-commit replay cost.

**Next experiment:** Hybrid quant base model (layers 37-42 at Q4K, others IQ2XXS) — hypothesis: higher-quality target features at DSpark capture layers [40,41,42] → higher acceptance → broader speedup. Downloading now (97.6 GB).

--- machiabeli 2026-06-30T08:29:20Z ---
**Peak speedup: +36.4% on repetitive structured output (token-identical, lossless)**

| Prompt | Baseline | DSpark | Speedup |
|---|---|---|---|
| **Repetitive JSON (30x same structure)** | 38.72 | **52.81** | **+36.4%** ✅ |
| JSON (20 diverse items) | 38.85 | 42.29 | +8.9% ✅ |
| Markdown tables | 42.54 | 45.21 | +6.3% ✅ |
| Code generation | 39.37 | 28.73 | -27.0% |

Correctness: token-identical at temp=0 on ALL prompts. M5 Max 128GB, Q2 base + Q4K drafter.

**The speedup scales with draft acceptance rate.** Repetitive output (same JSON structure repeated) achieves near-100% full 5/5 commits → verify(64ms) amortized over 5 tokens = 12.8ms/tok vs baseline 25.8ms/tok. This matches lobanov's theoretical +30-48% component measurements — the end-to-end gap is only the replay overhead on partial commits, which doesn't apply when acceptance is high.

For production use: enable DSpark selectively for structured output (JSON API responses, tool calls, templated content). Use `--mtp dspark.gguf` with the existing B2 rejection sampling code.

--- lobanov 2026-06-30T09:54:25Z ---
Great work @machiabeli! Good luck with multi-point checkpointing.

For anyone who want to experiment, I uploaded Q4_K version of DSpark drafter to a public HF bucket: [ds4-dspark.gguf](https://huggingface.co/buckets/lobanov/ds4/tree/ds4flash-dspark.gguf). That would save you trouble downloading original safetensors and running the quantization yourself.

I'm also in the process of building an imatrix-based Q4 quant to see if it would materially improve acceptance. It'll be a drop-in replacement for the current Q4_K.

Eventually, it'll also be interesting to see if imatrix-based Q2 or Q3 would offer similar acceptance with lower VRAM footprint, but I'm treating it as lower priority work.

--- lobanov 2026-06-30T21:33:02Z ---
Some very good progress in this thread!

I am also trying to make progress on this, but so far all indications that with the acceptance rates we are seeing from the drafter, average generation speed regresses. I commented on your PR @machiabeli, unfortunately I could not reproduce the speedup on my M5 Max.

I am keeping at imatrix-style drafter quantization, but it's not trivial. Naive approach based on drafter tensor activations at various prompt anchors does not (same approach use on IQ2-quant of V4-Flash by @antirez) does not improve the precision of the drafter. I'm trying acceptance-aware imatrix instrumentation and a few other approaches, but so far not much progress.

--- aidiffuser 2026-07-04T07:42:50Z ---
Data point for the "does spec decode help local MoE" question: measured it
on GLM-5.2 (744B MoE, batch-1, 2× M3 Ultra TP) using the model's own MTP
head. Acceptance is fine — α = 0.66–0.82 live. The cost is an itemizable
+36 ms/step toll of which only ~15 ms (MoE expert double-reads at L=2) is
fundamental; the rest is pipeline structure and small-L kernel shape. v1
landed at parity; removing the fixable portion projects 1.15–1.45×, better
at depth. Details in my #458 comment.

--- lobanov 2026-07-05T12:05:39Z ---
Good to see more attempts!

My progress has effectively stalled because the current verifier cost dominates the speculative decode cycle and, despite various optimization strategies, DSpark drafter with the current batch verifier still cannot beat plain decode on M5 Max.

There are three intertwined challenges:
1) Challenge 1: the average accepted prefix length from drafts created by Q4_K-variant of DSpark drafter is ~3.2 in F32 numpy oracle, but drops to around ~2.2 on Metal due to low-precision accumulation (I suspect). It's better on some types of prompts like repetitive JSON or MD tables, but not on general tasks.
2) Challenge 2: increasing precision of the drafter turned out to be difficult and has limited headroom. Even full-precision mixed-tensor DSpark drafter does not perform better in the oracle compared to Q4_K variant.
3) Challenge 3: the draft cycle is verifier-dominated. On M5 Max, verifying 5 tokens in one go using ds4 batch verifier takes ~80ms per cycle, drafting 5 tokens with DSpark takes a further ~10ms, and plain non-speculative decode takes ~26ms/tok. A speculative decode cycle generates one "free" token. This data says that to decode at the speed of plain decode average acceptance needs to be (80+10)/26-1=2.46, and to decode at 20% speed gain (at 21ms/tok) you need to accept 3.29 proposed tokens on average. And, no, despite low average acceptance, there's still enough full-block acceptances, so that when you limit drafter and verifier to L=4, decode speed is lower.

This is all linked in a vicious circle. With verifier-dominated decode cycle, drafter quality needs to improve considerably, but it's capped by the quality of the original drafter model and Metal precision loss. I experimented with various dynamic early-exit policies for drafter and dynamic verifier length, and found that they are prompt-specific, but don't generalize well. I experimented with drafting trees as well, but every additional tree node extends the verifier run, and no precision gains so far outweigh the additional verification cost.

Reducing time it takes for the verifier to run on a draft block is the biggest leaver, but the original batch verifier in ds4 looks optimal enough, and I don't see any obvious headroom. Ultimately, I suspect for such a big model it's bandwidth-bounded and may perform better on M3 Ultra with its 15% faster memory (from specsheet). That said I'm not a Metal expert, and if @aidiffuser can contribute a more efficient kernel for batch verification of small blocks, it'll be great to test.

As an aside, it seems that Q2_K version of the drafter is only 2.5% worse with just ~5GiB footprint, and there's a nice mixed quant Pareto knee in between at around 7.5GiB which only ~0.6% worse than Q4_K, but I didn't pursue this further due to fundamental challenges above.

--- aidiffuser 2026-07-05T14:07:36Z ---
Your Challenge 3 rhymes with what we just measured on a different stack (GLM-5.2 MTP self-speculative decode on MLX/Metal, 2× M3 Ultra tensor-parallel). Two findings that may transfer:

1. **Small-block verify cost is often not fundamental** — it's prefill-shaped kernels running at decode-shaped sizes. Our L=2 verify was 600ms/step until we routed small L away from three prefill-tiled paths (sparse-attention kernel, fused indexer tile, dense KV decompression) → 94ms, no new kernels written. Worth checking whether ds4's batch verifier at L=5 is hitting prefill tilings; per-phase wall-clock barriers (no GPU sync inside the measured window) found ours in one evening.

2. If any of your designs assume draft/verify graph construction overlaps GPU execution: **measure your framework's dispatch admission first.** On MLX, `async_eval` blocks for roughly the full remaining execution of any in-flight large graph — same stream, second stream, even fully independent graphs (378ms block vs 387ms exec; effective queue depth ≈1). Our deferred-accept redesign was byte-correct but could only reclaim the ~7ms pure-Python slice between dispatches; the "graph-build overlap" we'd budgeted for was dispatch admission all along.

Also matching your acceptance observations: ours is strongly content-dependent (0.70–0.79 on technical prose, 0.46–0.48 on synthetic summarize at temp 1.0), and a draft top1–top2 logit-margin gate predicts acceptance well (28% accept below margin 0.5, 96% above 4) — a cheap way to spend verifier runs only on likely-accepts if your drafter exposes logits.

Notes + reproducible microbench: https://gist.github.com/aidiffuser/a3a5d107fbb08eb00d24c0ad0f2e8d62

*Written with Claude (Fable 5)*


--- lobanov 2026-07-05T17:16:52Z ---
Correction on this:

> Challenge 1: the average accepted prefix length from drafts created by Q4_K-variant of DSpark drafter is ~3.2 in F32 numpy oracle, but drops to around ~2.2 on Metal due to low-precision accumulation (I suspect). It's better on some types of prompts like repetitive JSON or MD tables, but not on general tasks.

I rebuilt the numpy oracle using broader set of prompts and anchors and it now converged with observed average acceptances: ~2.3 at temp=0, ~2.1 at temp=1. Full report is [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/summaries/exactness_small_bundles_and_oracle_acceptance.md).

I had hope for about an hour that higher quants of DSpark could improve acceptance, but F16/F32 form of DSpark shows <0.4% difference in acceptance on the oracle. Full report is [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/summaries/dspark_quantization_ceiling.md).

The only two hopes now:
1. A possibility of a faster small-L shaped verifier. I consider it unlikely, because verification pass is memory-bandwidth bound. All these tokens must filter through all 42 layers, which needs all weights to be loaded. GLM-5.2 [concurs with me](https://github.com/lobanov/ds4/blob/dspark-research/issue468/summaries/mtp_verifier_bandwidth_binding.md). Neither of us is expert in Metal kernel optimisations, so please challenge if you have evidence.
2. A possibility that DFlash could work better here, producing on average higher acceptance of generated prefixes. We only need to move it by ~1.1 token accepted on average to get speedup.

It seems that the economics of Apple Silicon just doesn't lend itself well to DSpark. I don't think M3 Ultra can move the needle. RTX Pro 6000 would work very differently here with it's 2TB/s high-bandwidth memory, but I don't have the hardware to try. All one can hope for is 1TB/s+ M5 Ultra 😢 

--- deathcoder 2026-07-06T13:11:38Z ---
im looking into this but im still far way from testing my idea, after watching antirez video describing the possible problem here: we cant obtain fully the gains dspark promises on local hardware because on sparse models there is limited gain in having N tokens to verify if you dont have the experts required for it loaded, and if you have to load them you lose the advantage... i guess this translates into your verifier dominated observation

my idea was first to confirm wether this is actually happening by starting to log the experts needed for the verifier, if this really turns out to be the problem, i was thinking we could expand on dspark idea of increasing acceptance rate of the token by also considering on local hardware if we have the experts needed to verify them efficiently, basically assuming that this check can be done with minimal performance impact, we can stop drafting once we generated a draft token that requires a different set of experts to verify and if this logic performs well, it probably should at most slightly underperform direct decode, but also get the full benefit of draft tokens when they stay on the same expert sets

just wanted to share the idea in case you want to try it... otherwise ill send an update once i get to it and have something to share

hope this makes sense, let me know if it doesnt so i stop wasting time lol

--- lobanov 2026-07-06T13:38:21Z ---
Which video you are referring to @deathcoder? I'd love to see it if it's public!

Regarding your idea, I suspect antirez is referring to streaming weights into GPU cores for per-layer MLP, not into the memory itself, as the model is fully resident anyway unless SSD streaming is enabled, but yes, that's where the memory bandwidth is limiting us. I don't know how much mileage there is in this idea. I empirically tested current batch verifier on M5 Max with different block sizes [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/summaries/mtp_verifier_bench_results.md) and it does show a mild sublinear decline of cost per token verified for L between 2 and 6 tokens. This **could** be attributed to the amortisation of reuse of already loaded expert weights, but to answer definitively it would need further benchmarking. What you are describing is sort of bandwidth-aware scheduler that could abort or truncate the verification run if "experts budget" is exhausted. I don't know anything that rules this out as a plausible hypothesis, but I would try first to understand maximum theoretical gain from such a scheduling policy. For example, you can see how much performance difference does it make if number of experts routed per layer is reduced from standard 6 to, say, 4.

Feel free to build on my research dossier structure if you think it's a good starting point. I found this harness structure efficient for coding agents leading deep-dives into various hypotheses.

--- lobanov 2026-07-06T14:00:25Z ---
I re-read your proposal @deathcoder and there is one thing that I didn't notice in the first read that is worrying. I don't believe there is a reliable way to detect at the drafting stage which per-layer experts will be invoked during the verification. Expert selection happens during the verifier pass at each layer once it computes the masked attention, calculates expert weights and routes to top-6 experts' MLPs at that layer. You can still apply scheduling policy at that point, but full block will have been drafted by the and your leverage is verifier truncation.

--- deathcoder 2026-07-06T14:14:49Z ---
@lobanov https://www.youtube.com/watch?v=7-n0HWtAg2Y in this video he explains the problem of speculative decoding, i think he also spot mentions this in other videos so cant fully remember if i patched together things from other places, but most of the argument comes from that one i would say (its in italian though)

thanks for looking into it btw, i expected there could be issues with the idea, thats why i wanted to start by verifying the hypothesis first, there might be other optimizations we can think of once we know exactly where the bottleneck is... one way of keeping the idea alive might to drop the remaining draft tokens after you know the set of experts i guess... but yeah wanted to iterate on this once im in a better state and i can get the info i need out of the execution pipeline... also im relying mostly on gpt 5.5 for coding and the technical expertise in this domain (using the PRs as reference) but im moving slowly trying to understand what its/im doing, thats why i didnt start from the shared prs, it would be a bit too difficult to understand whats going on for me

--- deathcoder 2026-07-06T17:53:30Z ---
@lobanov i was reading through your shared bench, and i noticed:

> Total verify is NOT flat in K — it grows, and super-linearly at large K... because the union of activated routed experts grows with the suffix (up to 6/token), eroding the weight-amortization that makes batched verify cheap. This is a MoE-specific effect...

maybe i didnt express it correctly in my previous message but i think you reached the same conclusion as antirez, verify becomes the problem if we have to use different sets of experts... even if the model is fully resident you still have this problem

so basically we dont have that much to work with for single user local inference

--- lobanov 2026-07-13T15:37:12Z ---
I kept at it even though sometimes it feels like banging my head against a brick wall 😆 

The issue fundamentally remains as follows. Vendored DSpark tensors quantized to Q4_K yield average acceptance ~2.2 tokens on a broad prompt corpus. Optimized DSpark drafter + shipped batch verifier (BTW it's not greedy-exact, ~0.6% top-1 logits flip) make it approximately equal to baseline decode on M5 Max. Verifier eats most of the performance gains. When the model is served to multiple users, server-side verification batching can amortize it, but this isn't an option with single-player ds4.

Since my last report I tried a couple of things.

Firstly, I had a hypothesis that vendored DSpark tensors are trained on the full-pecision V4-Flash target model and fail to propose long-enough prefixes on our IQ2XXS version. I [tested earlier](https://github.com/lobanov/ds4/blob/dspark-research/issue468/archive/leads/lead_03_acceptance_statistical_power.md) with a PyTorch oracle on a substantial prompt corpus that there is no acceptance gains if the drafter is kept in higher precision, even BF16 proposes blocks with the same quality, but I could not rule out that higher precision *target model* could be more efficient because it'll feed the drafter with different residuals. To test that, I rented 2xH200 from Modal.com and run a patched vLLM to capture residuals and greedy tokens on full-precision V4-Flash with 300 prompts at 128 anchor points (fewer on shorter responses), totalling ~20k anchors. Then I ran the PyTorch oracle to simulate acceptance, and it **did show** 8-10% uplift headroom potential, but it's unclear how much of that is recoverable in IQ2XXS target model version. Cautious estimate is around 5% with custom-trained adaptation, but that alone won't make a big difference. Full report is [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/archive/leads/lead_04_fp_ceiling_capture.md). I didn't follow further leads yet.

Besides, I also captured top-128 logits, so this could be used in future for verifying quantized V4-Flash model output quality. If anyone is interested, I can upload the captures as a HuggingFace dataset.

Second thing I tried is investigating confidence-based scheduling. DSpark has it's own built-in confidence head which sort-of predicts the level of confidence of the drafter. It's useless for stopping drafting earlier, because it needs to be batched for efficiency, but it's useful for scheduling the verifier to look at fewer tokens in a batch. DSpark paper also describes a fitted STS-based scheduler, which I replicated. I've got oracle-based 5% speedup figure, which in itself isn't a decisive win again, but it could stack up nicely. Full report is [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/archive/leads/lead_02_confidence_scheduled_verification.md).

My focus currently is to implement all these advances in the engine, re-measure draft cycle economics, and then re-evaluate the leads.

--- deathcoder 2026-07-13T16:33:27Z ---
hey lebanov thanks for sharing the update im also still looking into this, with oai resets this weekend i made good progress, in my benchmarks tests on short context the acceptance rate is very high almost always 5 tokens, this with a serialized verifier had a negative performance compared to baseline, but i also adapted the fast verifier that i think already exists in ds4 and that was able to achieve a net performance increase from 21 t/s to 29 t/s (my understanding is fast verifier is batching the verification)

i then decided to try my implementation against your longer context benchmarks and i hit a few problems, most important is that fast verifier is not mathematically equivalent to the serialized one so on longer contexts there are drifts, which made me go back to the serialized verifier and try to apply some optimizations there

i can try when i have some time to push my fork so you can have a look, maybe the better acceptance rates can be transferred so we get gains on both attempts

my weekly budget is done for this week so ill probably continue next week unless oai issues another reset

edit: https://github.com/deathcoder/ds4/tree/codex/dspark-observability-0

there is a journal file in there where i had the agent document every step we went through so far

--- lobanov 2026-07-14T09:51:47Z ---
Thanks @deathcoder, let me take a look. What are your draft cycle metrics and what is the prompt corpus you use? I mean, average acceptance per cycle, average committed tokens per cycle, drafting ms, verification ms, and what draft and verification lengths are you using? Also what's your hardware? I tried to fish it out of the journal (nice one!), but couldn't find.

Regarding bit-exactness, it's a separate problem: for non-bit identical logits, temp=0 flips at close top-2 ties frequently enough to diverge materially on long-enough sequences. I'm not focusing on this now. My proxy is distribution-exactness for temp=0.5 and temp=1 (very close), and close functional equivalence at temp=0 using ds4-eval (`--nothink --tokens 2048 --seed 1 --temp 0`). If you make batch verifier more exact, it'll be amazing!

--- deathcoder 2026-07-14T10:56:08Z ---
> If you make batch verifier more exact, it'll be amazing!

i think sol discarded that option and decided to focus on the serial verifier, i have been getting some minor improvements but im still below baseline

this were my latest bench results and sol's analysis:

DSpark Benchmark Summary
Baseline median: 23.54 t/s
Runtime median: 21.60 t/s
Ratio of medians: 0.9176x
Median paired speedup: 0.9172x
Measured pairs: 3
Runtime average accepted depth: 4.923
Runtime target evals avoided: 50.0
Runtime target evals / emitted token: 0.2188
Runtime token positions / target eval: 4.929
Runtime generation sidecar / emitted token: 3.600 ms
Runtime prefill sidecar total: 26.122 ms
Runtime target time / target eval: 194.290 ms
Runtime target time / emitted token: 42.501 ms
Runtime batch outcomes: 13.0 attempts, 12.0 full, 1.0 partial, 0.0 fallbacks
Runtime fast verifier: 0.0 calls, 0.0 failures, 0.0 exact fallbacks
Runtime exact attention-pre outcomes: 602.0/602.0 successful
Generation sidecar breakdown / emitted token: bridge 0.182 ms, stages 0.796/0.777/0.776 ms, head 0.113 ms, chain 0.953 ms

stable result: the promoted exact verifier is correct, but still 8.2% slower than baseline.
The three paired ratios were 0.9172x, 0.9172x, and 0.9196x, with identical output hashes and no thermal warning. Background activity is unlikely to explain a gap this consistent.
The encouraging part is that our optimizations composed almost exactly as predicted:
Earlier exact runtime: 0.8176x
Exact FFN improvement: 1.057x
Attention-pre improvement: 1.061x
Expected combined ratio: about 0.917x
Observed combined ratio: 0.9176x
So nothing was lost during promotion.
The remaining budget is clear:
Baseline: approximately 42.48 ms/token
Runtime: approximately 46.30 ms/token
Exact target verifier: 42.50 ms/emitted token
DSpark sidecar: 3.60 ms/emitted token
Residual overhead: about 0.20 ms/token
Despite evaluating roughly 4.93 positions per target call, the exact verifier still costs nearly one complete baseline token per emitted token. The sidecar then pushes total latency above baseline. We need approximately another 3.8 ms/emitted token, or roughly 9% off target-verifier time, to reach break-even.

now its focusing on trying to improve attention, not sure how far i will get before i finish this oai reset too

for prompts i have everything in speed-bench, basically the strategy so far has been trying to prove a promising improvement on short context than test on code 8k etc that are inside speed-bench/issue468

I have an M1 Ultra 128gb 48c btw

--- lobanov 2026-07-14T11:26:13Z ---
Great, thanks for sharing!

> Runtime average accepted depth: 4.923

This looks suspicious. I read it as your DSpark drafter generates 5 tokens, of which all 5 are accepted 98% of times. This is either incorrect interpretation, or indicates very significant luck on a specific token sequence. My best result is ~3.1 on code sequences and ~2.2 on a broad corpus. I don't even get anywhere near close to that in the oracle using full-precision model residuals. Please could you try to get measurement on a broader corpus (like [this one](https://github.com/lobanov/ds4/tree/dspark-research/issue468/prompts)).

> i think sol discarded that option and decided to focus on the serial verifier

**Edit:** I rewrote the below after reflection and then reverted to the original meaning after a bit more reflection.

This is unlikely the right path in my opinion. Serial verifier feeds each drafted token through all 42 layers and has the same performance bottleneck as plain autoregressive decode, so you are fighting against the memory bandwidth floor. Batched verifier can be sublinear because it can amortize reused expert weights and use GPU matrix-to-matrix operations vs matrix-to-vector. In that sense it's more prefill-like with higher arithmetic intensity (FLOPS/bytes streamed).

--- deathcoder 2026-07-14T12:28:04Z ---
> Please could you try to get measurement on a broader corpus (like [this one](https://github.com/lobanov/ds4/tree/dspark-research/issue468/prompts)).

i didnt check that with the serial verifier but i kind of remember acceptance rate remained high in the long context tests, the problems were with drifting, once i have a promising impl ill get back looking into it since it def plays a role in throughput


> This is unlikely the right path in my opinion. 

i have the same intuition but who am i to question our one and only silicon-based sun, atm im letting it follow this path and if it does produce any good result i will do a final check to make sure those improvements remain real for the dspark path even if we do the same optimizations on the baseline... not getting ahead of myself though, im out of tokens again

btw for the optimization that its worked on so far i think the current approach has been profiling the serial verifier and spotting where batching can be introduced while maintaining math parity, so i think its reworking the serialized into a batched one anyway, it just decided to start from the working one instead of the broken one


--- lobanov 2026-07-15T18:16:56Z ---
Finally, some progress! By stacking Metal-based batch drafter from the #502 (thanks @audreyt, @machiabeli and @stephenlthorn) with the STS-derived dynamic confidence-based verification scheduling, reusing "bonus" token coming of a failed verify (and always requiring to verify one token more than we are confident), I was able to gain ~5% on a broad set of prompts ([this corpus](https://github.com/lobanov/ds4/tree/dspark-research/issue468/prompts/baseline_corpus)) and longer sequences compared to plain baseline, including code and prose. The progress report is [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/summaries/dspark_runtime_milestone_3_progress.md), but code isn't yet a PR-worthy. I just wanted to report a milestone nonetheless.

This is before any meaningful verifier kernel work and attempts to regain the drafting quality with the full-precision residuals, which could well stack up.

**P.S.** current shape of acceptances on the above corpus:

```
 ┌─────────────────────┬───────┬──────────┐
 │ cont drafts         │ P     │ cont × P │
 ├─────────────────────┼───────┼──────────┤
 │ 0 (verified 0 or 1) │ 29.6% │ 0        │
 ├─────────────────────┼───────┼──────────┤
 │ 1 (verified 2)      │ 22.4% │ 0.224    │
 ├─────────────────────┼───────┼──────────┤
 │ 2 (verified 3)      │ 16.9% │ 0.338    │
 ├─────────────────────┼───────┼──────────┤
 │ 3 (verified 4)      │ 9.9%  │ 0.297    │
 ├─────────────────────┼───────┼──────────┤
 │ 4 (verified 5)      │ 21.1% │ 0.844    │
 └─────────────────────┴───────┴──────────┘

 Mean = 0.224 + 0.338 + 0.297 + 0.844 = 1.703

 The distribution is bimodal — it's not a smooth decay, it's two regimes:
 - ~30% of cycles are "reject-early" (verified=1): the first continuation draft is wrong → 0 drafts accepted. These contribute 0 to the mean.
 - ~21% are "accept-all" (verified=5): all 4 drafts right → 4 drafts. These contribute 0.84 (the bulk of the mean).
 - ~49% are "partial" (verified 2–4): contribute ~0.86.
```

**P.P.S.** sadly, it seems that full-precision gain recovery in the drafter with quantised target model isn't a promising lead, disqualifying report is [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/archive/leads/lead_07_upstream_quality_ceiling.md). We may be better off training a drafter on IQ2XXS from scratch, but that's the whole different ball game.

--- lobanov 2026-07-19T13:27:35Z ---
I spent last 72 hours and two weekly limit resets watching gpt-5.6-sol not being able to make any meaningful progress rewriting the batch verifier to make it faster without collapsing acceptance or more exact without making it slower. The worklog is [here](https://github.com/lobanov/ds4/blob/dspark-research-batch-verifier/issue468/pending/lead_08_fused_verify_kernel.md) if anyone is curious enough. This, of course, doesn't mean faster batch verifier is impossible, but this indicates that it's not trivial, so I think I'm going to stop pursuing this lead for now until I get time to look into this properly.

The bit remaining to explore is to see it's possible to fine-tune the drafter so it predicts tokens better on IQ2XXS target. I did a short unsuccessful experiment, as reported above, to see if there's any easy way to recover the precision supposedly lost due to quantized target, but that hasn't ruled out that a more ambitious post-training exercise could make the drafter more accurate.

--- robotnursenyc 2026-07-19T16:21:46Z ---
We implemented the per-token verify checkpointing proposed in this thread and can report measurements. (Disclosure: this work was done with heavy AI assistance — Claude — driving a Mac Studio; all numbers below are from real runs on that hardware, and the branch passes `--server`, `--metal-kernels`, and `--logprob-vectors`.)

**What was built** — the N=2 `prefix1` mechanism generalized to K slots (`DS4_SPEC_CKPT_MAX 8`), ~200 lines in `ds4.c`, no new Metal kernels: during the batched verify encode, the compressor/indexer frontier state (~32 KB/layer) is blitted into slot *t* after each suffix position (same in-stream `ds4_gpu_tensor_copy` pattern as the existing prefix1 captures at the two `t == 0` sites). A partial accept of *k* tokens then commits slot *k−1* and reads its already-computed spec-logits row — no snapshot restore, no second batched pass, no anchor replay. The anchor-only miss unifies with the general partial case (slot 0, row 0).

**Measured on M3 Ultra 256 GB (60-core bin), Flash q4-imatrix, greedy, ctx 32768**, with prompt-lookup (#396) as the drafter so partials occur naturally:

- partial-accept verify cost: **178–186 ms (restore + second pass) → 111–114 ms (single pass)**
- greedy output **byte-identical** across baseline / old partial path / checkpointed path (same SHA-256 for all three)
- extract-style workload throughput 54.6 → 56.6 t/s (CLI); the larger effect is that partial-heavy acceptance patterns stop being punished — exactly the block>2 economics problem quantified upthread

Code: https://github.com/robotnursenyc/ds4/commit/a6252fb79d (branch `dsv4-speculative-stack`, which stacks on #555 + #396). Happy to rebase this into a standalone PR against `main` wired to whichever consumer is preferred — MTP N>2 or the #396/#502 paths — and to drop the `DS4_SPEC_CKPT` diagnostic switch per AGENT.md's no-permanent-flags rule if it lands as default behavior.

--- deathcoder 2026-07-19T17:28:12Z ---
@lobanov yeah im also getting close to throw in the towel, i have ported some of the improvements you shared but overall dspark is still ~12% slower than baseline on my branch... the only silver lining here is that between all optimizations sol has been working on, some of them improved the baseline so even if dspark is not ready yet, my branch currently is ~8% faster than main

my next step is trying to see if some of the wins that open projects got like omlx and mtplx can be ported to ds4 and maybe that will make dspark viable 

--- robotnursenyc 2026-07-19T18:52:36Z ---
its painful this flies on my sparks

--- OPS-NeoRetro 2026-07-20T07:37:10Z ---
Did anyone forget that I actually reviewed the code over a week ago?

--- nhwaani 2026-07-20T11:43:43Z ---
Here are a few data points from our independent research on this issue that may help focus the remaining effort. We did a deep dive on lobanov's research branch (docs 00-24) cross-checked against the actual M5 Max binary and RAM budget, and found several things worth sharing:

## 1. The F32-accum gathered-dense kernel infrastructure already exists

The `kernel_q4_gather_slots6` and `kernel_mul_mv_slots6_q4_K_pair_swiglu_f32` kernels are present in lobanov's metal/moe.metal on the dspark branch. They implement the F32-accumulation path over unchanged Q4_K weights — the fix that recovers the 2.79→1.53 collapse. However, the C-side toggle/env-var plumbing was never completed (no `DS4_METAL_ENABLE_Q4_GATHER_SLOTS` in ds4.c). The infrastructure is there and could be wired with ~50 lines of C — no new Metal kernels needed. This is a concrete path to recovering the ~1.53→~2.2 Metal acceptance.

## 2. Memory budget for higher-precision drafter quants

Verified on the M5 Max 128GB: real headroom is ~29 GiB (the GGUF is reclaimable mmap page cache, not resident). Q8_0 drafter (+9GB) and Q6_K (+3.8GB) both fit. Only F16 (+26GB) breaks the budget. This means Q8_0 is a viable option for anyone wanting to test whether higher drafter precision improves acceptance — it doesn't require a RAM upgrade.

## 3. The +1 token recovery quantification

The guaranteed +1 token/cycle was dropped from the terminal table in doc 23. Restoring it changes the economics materially:
- greedy+1: +9% at 64k
- B2 (A≈1.82)+1: +22% at 64k
- oracle+1: +64% at 64k

This is now partially captured in confidence scheduling (the "bonus token" from failed verifies), but explicitly accounting for the +1 across all cycles may reveal additional headroom.

## 4. Structured lever ranking

From our doc 24 review, ranked by effort/impact:
1. F32 accumulation over Q4_K — memory-neutral, no GGUF rebuild, ~1.53→~2.2 Metal acceptance
2. imatrix at Q4_K — must be collected from drafter's real activations, not target's
3. Q6_K/Q8_0 — only if offline Pareto sweep justifies new routed kernel
4. Target-distillation fine-tuning — highest ceiling, costliest

The key enabler: under F32 accumulation the numpy oracle IS the production model, so the entire quant×imatrix frontier is measurable offline in Python with zero new Metal kernels. This lets you answer "does higher drafter precision help" in ~hours, not weeks.

## 5. Measured cost curves (M5 Max)

For reference, our measurements of the component costs:
- Plain decode: 31.3→35.5 ms/tok (7k→64k, +11% SWA-flat)
- Batch verify (L=5): ~75 ms, context-flat
- Drafter backbone (3 layers, 5 tok): 5.0 ms; total draft ~7.2 ms

These match closely with the numbers others have reported and confirm the verifier-dominated regime.

The acceptance ceiling (~2.2 on broad prompts) and verifier cost (~80ms/cycle) remain the hard constraints. The most promising paths forward given current understanding appear to be: (a) wiring the existing F32-accum kernel to recover Metal acceptance, (b) per-token verify checkpointing (which robotnursenyc implemented) to eliminate the partial-commit replay tax, and (c) exploring whether Q8_0 drafter moves acceptance on the verifier-dominated workloads where it matters most.

--- lobanov 2026-07-20T14:02:54Z ---
Thanks @nhwaani,

That 2.79→1.53 collapse was a drafter-converter bug, which was fixed later, but 2.79 figure is overreported due to prompt selection. Since than moved to [another branch](https://github.com/lobanov/ds4/tree/dspark-research), as that one became very messy, and introduced a broader prompt corpus for more robust benchmarking. Presently, I can see ~5% gains on a broad set of prompts as reported [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/summaries/dspark_runtime_milestone_3_progress.md) and in [this comment](https://github.com/antirez/ds4/issues/468#issuecomment-4983904315) above, and that already includes bonus token harvesting and a Metal drafter reducing 4 tokens drafting to ~6ms. I did not see gains if I used sequential verifier, but only with the batch verifier which scales sublinearly.

Then, there is no apparent way to regain drafter precision by target distillation and further fine-tuning. I'm taking full-precision target + drafter at gold standard (captured via vLLM running on rented 2xH200), and it's about 15-20% higher (in oracle) than with IQ2XXS target+Q4_K drafter, but I could not find a way to practically regain that precision. My latest experiment (last night's) is written up [here](https://github.com/lobanov/ds4/blob/dspark-research/issue468/archive/leads/lead_10_drafter_redistillation.md). Before that I experimented with drafter precisions and seen no acceptance improvement with IQ2XXS target if drafter weights are stored even in F16, so imatrix won't help. If you find a way to do that, it'll help.

Remaining plausible leaver for improving generation speed is to rework the sublinear batch verifier to be faster. There seems to be some headroom purely on memory-bandwidth basis, but so far I was unable to do that.

--- nhwaani 2026-07-20T15:04:33Z ---
Heads up: the Q4_K DSpark GGUF on the HF bucket (`ds4flash-dspark.gguf`, uploaded 2026-06-30) silently disables the drafter on upstream main, so any test using it measures pure overhead, not speculation.

## What's happening

The GGUF's three small-head tensors are stored as bf16:
- `mtp.2.markov_head.markov_w1.weight`
- `mtp.2.markov_head.markov_w2.weight`
- `mtp.2.confidence_head.proj.weight`

This matches the source model — audreyt noted above that the small heads are bf16/fp32 by design. The problem is that upstream main's `dspark_markov_probe_ready()` in ds4.c only accepts F16/F32/Q8_0 for `DS4_DSPARK_LAYOUT_DENSE`:

```c
case DS4_DSPARK_LAYOUT_DENSE:
    return type == DS4_TENSOR_F16 ||
           type == DS4_TENSOR_F32 ||
           type == DS4_TENSOR_Q8_0;
```

bf16 fails the check, so `markov_ready` stays false, `markov_ok` never becomes true, and `dspark_draft_valid` never gets set. The speculation entry gate then hits:

```c
if (!s || !s->dspark_draft_valid || s->dspark_draft_len == 0) {
    s->dspark_stats.no_draft++;
    return n_accept;   // returns 0 accepted
}
```

## Evidence

Running on M5 Max / upstream main `427e281` with `DS4_DSPARK_STATS=1`:

```
DSpark stats cycles=119 first_tokens=119 proposed=0 accepted_draft=0
accept_rate=0.00% avg_accept=0.000 full=0 partial=0 miss_first=0
no_draft=119 no_room=0 invalid=0
verify=0.000 verify_layer=0.000 replay=0.000
```

`proposed=0` across every flag combination I tried (`--dspark`, `--dspark-confidence 0.9`, `--mtp-draft 2/5`, `DS4_DSPARK_SCHEDULER=1`). The verifier never runs because the drafter never produces tokens. The -5% to -15% regressions I measured are the cost of carrying a dormant drafter (target-hidden capture on layers 40-42 + scheduling bookkeeping), not the verifier economics.

Forcing the drafter with `DS4_DSPARK_FAKE_ARGMAX_PROPOSAL=1` confirms speculation can run — `proposed=28, accepted=28, accept_rate=100%` — but that's the target's own argmax, not DSpark.

## Why this likely affects others

- @lobanov's +5% milestone is on the `dspark-research` branch, which has a different DSpark path (no `dspark_markov_probe_ready` gate — I checked, the function doesn't exist there). His results are real but on his branch, not upstream main.
- @machiabeli built their own GGUF ("47/47 byte-exact"), which likely casts bf16→F16 during conversion, so it works on main.
- @deathcoder's "~12% slower than baseline" matches the dormant-drafter overhead pattern exactly — possible they're hitting the same silent disable.

## Two fixes

1. **Re-quantize** the bucket GGUF casting the three small-head tensors bf16→F16 (they're ~70 MB total, negligible).
2. **Or** add bf16 to the accepted `DS4_DSPARK_LAYOUT_DENSE` types in upstream main — the engine already has bf16 support elsewhere (`ds4q_bf16_to_f32` exists in gguf-tools).

Either way, until one lands, anyone testing DSpark on upstream main with the bucket GGUF is measuring overhead, not speculation — which may explain why the negative results across participants are so inconsistent.

Happy to PR the type-check fix if that's the preferred direction.


--- deathcoder 2026-07-26T14:39:11Z ---
hey @nhwaani sol worked through your message but found it doesnt apply to our implementation, thanks for flagging this possibility though

I noticed that dspark support has been merged into main together with other metal improvements, and i evaluated how main currently compares against my branch, what i found:
- my improvements have largely been merged in already, though my branch baseline is still about 2.5% faster than main (down from 8% before the recent updates)
- One more problematic finding is that main dspark seems to diverge from the original model in its generation, ill let sol explain this as im sure he is going to be better than me:

<details>

<summary>I found a reproducible greedy-output drift in the default DSpark runtime on upstream</summary>

I found a reproducible greedy-output drift in the default DSpark runtime on upstream commit `0a7ad776b9068348e6cb09df8cafa9cadd285298`.

This used Metal on an M1 Ultra, the official `q2-imatrix` target and official converted DSpark support GGUF. No optional fast verifier, profiling, statistics, or diagnostic environment variables were enabled. DSpark used its documented default confidence `0.9` and adaptive scheduler.

The first four selected HumanEval prompts matched exactly. On `humaneval_093`, ordinary decoding and DSpark diverged semantically:

- Plain: `swapped = char.swapcase()`
- DSpark: `new_char = char.swapcase()`

Plain output was 542 bytes with SHA-256:

`c5dd0baa4ae31a1019fd8303b865c5ad775d18933c7eac0dc830f1f8951360e1`

DSpark output was 530 bytes with SHA-256:

`cccc2800f96323c55ad1443bab849d4fa97315312fb320ac50b16086891fb68a`

Minimal reproduction:

    git clone https://github.com/antirez/ds4.git
    cd ds4
    git checkout 0a7ad776b9068348e6cb09df8cafa9cadd285298
    make

    ./download_model.sh q2-imatrix
    ./download_model.sh dspark-support

    curl -L \
      https://raw.githubusercontent.com/deathcoder/ds4/521cd51/speed-bench/humaneval-acceptance/samples.jsonl \
      -o /tmp/ds4-humaneval.jsonl

    python3 - <<'PY'
    import json
    for line in open("/tmp/ds4-humaneval.jsonl"):
        row = json.loads(line)
        if row["source_index"] == 93:
            open("/tmp/humaneval_093.txt", "w").write(row["turns"][0])
            break
    PY

    for name in $(env | cut -d= -f1 | grep '^DS4_'); do unset "$name"; done

    ./ds4 --backend metal \
      --model ./ds4flash.gguf \
      --ctx 16384 --nothink --temp 0 --seed 1 -n 128 \
      --prompt-file /tmp/humaneval_093.txt \
      > /tmp/plain.stdout 2> /tmp/plain.stderr

    ./ds4 --backend metal \
      --model ./ds4flash.gguf \
      --ctx 16384 --nothink --temp 0 --seed 1 -n 128 \
      --prompt-file /tmp/humaneval_093.txt \
      --mtp ./gguf/DeepSeek-V4-Flash-DSpark-support.gguf --dspark \
      > /tmp/dspark.stdout 2> /tmp/dspark.stderr

    cmp /tmp/plain.stdout /tmp/dspark.stdout
    diff -u /tmp/plain.stdout /tmp/dspark.stdout
    shasum -a 256 /tmp/plain.stdout /tmp/dspark.stdout

Our audit and guarded comparison runner are on:

https://github.com/deathcoder/ds4/tree/codex/dspark-observability-0

The relevant checkpoint is `521cd51`. The journal records the support-binding validation, activation counters proving that upstream genuinely drafted, and the resulting correctness failure.

</details>

now i dont know how much we should care about this considering that both main and my branch are slower than baseline using dspark on the 128gb mac i tested, so i dont see a reason to use it yet, but having an "error" in the engine is something i still thought its worth trying to raise to @antirez in case he happens to stumble on this thread

--- aidiffuser 2026-08-01T15:06:40Z ---
Following up with three things measured since July — all on DeepSeek-V4-Flash-0731, MLX, 2× M3 Ultra tensor-parallel:

1. **If your "plain decode" baseline runs on mlx-lm's `deepseek_v4` (or derived code), it decays ~4.3 µs/tok per generated token.** The compiled rope helpers take the KV offset as a python int, so every generated token mints a new `mx.compile` cache entry, and lookup is a linear memcmp scan — 12% → 29% of decode-thread wall between ~2k and ~7k generated in our `sample` profiles, plus unbounded host memory (~3 entries/token retained). Fix is passing a 0-d `mx.array` offset (byte-identical outputs): details in the ml-explore/mlx-lm#1189 comment and ml-explore/mlx#3964. Relevant here because a spec-decode A/B against a decaying baseline flatters the drafter more the longer the run.

2. **A stripped-DSpark datapoint for the economics**: we dropped the dspark heads at conversion — the July admission-law finding is unchanged on current MLX (only the ~7 ms python slice overlaps; a drafter must win on serialized arithmetic alone). After the prefill/decode work above, our *non-spec* decode is ~29–31 tok/s and flat through 256k context — that's the bar any DSpark cycle has to beat on this class of hardware. @deathcoder's serialized-verifier direction is exactly what the admission law predicts pays: don't buy overlap, buy cheaper verify.

3. Possibly useful for the batch verifier specifically: we ported the sparse gather-attention kernel so prefill-shaped blocks attend only each query's window + top-k pool rows via per-query index lists (visibility encoded as sentinel-padded sorted indices; attention sinks folded into the online softmax by seeding the running max/denominator). Verify-at-L is a small prefill block, and the "prefill-shaped kernels at decode sizes" tax from the July discussion is precisely what this removes. Kernels + campaign notes: https://gist.github.com/aidiffuser/fef1890680e4eed60d3902511b02a696

*Written with Claude (Fable 5)*
