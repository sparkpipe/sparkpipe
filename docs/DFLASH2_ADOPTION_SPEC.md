# DFlash2 — ADOPTION SPEC for the SparkPipe stack

Mined 2026-08-19. Every claim carries its source URL. Percentages marked *(derived)*
are my arithmetic on numbers published in the cited table, not quoted figures.

**Release state at time of writing:** SGLang support is **merged**
(sgl-project/sglang#35371, merged 2026-08-19T00:07:29Z,
<https://github.com/sgl-project/sglang/pull/35371>). vLLM support is an **open, unmerged
PR** (vllm-project/vllm#52816, opened 2026-08-18, state `open`, 755+/5-, 11 files,
<https://github.com/vllm-project/vllm/pull/52816>). llama.cpp support is an **open,
unmerged PR #27342** — "spec : add DFlash2 support (local convolution + candidate
selector)", `state=open`, `merged=false`, created 2026-08-18
(<https://api.github.com/repos/ggml-org/llama.cpp/pulls/27342>); DFlash **v1** by contrast
merged as #22105 on 2026-06-28. All three engines gate on the checkpoint alone — PR #27342
body: "DFlash2 is enabled when the checkpoint is DFlash2; no need to use extra flag." Blog:
<https://inco.ai/blog/dflash2/>.

> ⚠️ **There is no upstream reference implementation of DFlash2.**
> <https://github.com/z-lab/dflash> is the DFlash **v1** repo: 14 files, top level
> `{.github, .gitignore, LICENSE, README.md, dflash, pyproject.toml}`, and **zero paths
> matching `dflash2`, `conv`, or `select`**; its only release is `v0.1.0` (2026-08-18,
> empty body) (<https://api.github.com/repos/z-lab/dflash/git/trees/HEAD?recursive=1>).
> The blog and both PRs cite it only for *prompt formatting and the benchmark harness*.
> **The three engine PRs plus the checkpoint tensors ARE the specification** — there is no
> training code, no paper, and no canonical forward to diff against. Treat the vLLM PR's
> two unit tests (`test_grouped_conv_matches_reference`,
> `test_selector_edges_match_sequential_reference`) as the authoritative oracle; they are
> the only executable ground truth published.

---

## (a) Architecture delta vs DSpark / DFlash v1

DFlash2 is **DFlash v1 plus two bolt-ons**, carried on a separate architecture string so
existing checkpoints are untouched: "a checkpoint declaring `DFlash2DraftModel` gets
them, and every existing `DFlashDraftModel` checkpoint resolves to the class it resolves
to today, untouched by this PR"
(<https://github.com/vllm-project/vllm/pull/52816>). It does **not** replace attention,
the block structure, or the 5-layer backbone.

### What is NOT changed

| Piece | Status |
|---|---|
| Block-diffusion one-pass block draft | unchanged |
| 5-layer backbone, hidden 5120, GQA attention | unchanged (`DFlash2Qwen3DecoderLayer` *subclasses* `DFlashQwen3DecoderLayer`) |
| Target-hidden taps + `fc` projector | unchanged (`fc.weight [5120, 25600]`, 5 taps × 5120) |
| Shared target embedding + shared target LM head | unchanged (drafter ships no `embed_tokens`/`lm_head`) |
| Verification / rejection sampling | unchanged; "The convolution is block-local and stateless, so it drops into DFlash without changing attention, the LM head, or verification." (<https://inco.ai/blog/dflash2/>) |

### Addition 1 — grouped dynamic depthwise convolution (replaces *within-block attention work*, not attention)

Motivation is explicit: DFlash attention has "two jobs: read the context before the block,
and model the dependencies inside. But it spends less and less on the second: the block's
share of attention falls from **30% in Layer 1 to 8% in Layer 5**… So we split the jobs: a
dedicated module takes the within-block work, and attention keeps reading the context."
(<https://inco.ai/blog/dflash2/>)

Formula (identical in both PR bodies):
`out[i,c] = Σ_t (base[t,c] + δ[i,t,g(c)]) · x[i−t,c]`, "taps zero across the block
boundary. Wraps each sublayer in and out from one projection of its input."
(<https://github.com/vllm-project/vllm/pull/52816>,
<https://github.com/sgl-project/sglang/pull/35371>)

Placement — **four conv applications per layer**, two modules each with a `prepare` (pre)
and `finish` (post) half, from `DFlash2Qwen3DecoderLayer.forward`
(`vllm/model_executor/models/qwen3_dflash2.py`, PR #52816 diff):

```
input_layernorm
attention_conv.prepare(h) -> (h, coefficients)      # conv + emit finish-side deltas
self_attn(...)
attention_conv.finish(h, coefficients)
post_attention_layernorm
mlp_conv.prepare(h) -> (h, coefficients)
mlp(h)
mlp_conv.finish(h, coefficients)
```

Shapes as shipped (Qwen3.8-27B drafter): `conv_kernel_size = 2` (two taps: current + one
back), `conv_group_size = 16` (320 groups over hidden 5120). Per conv module:
`base_kernel [2, 2, 5120]` (dim0 = the two sides, prepare/finish) and
`kernel_projection.weight [1280, 5120]` where `1280 = 2 sides × 2 taps × 320 groups`.
`block_size = 1 + num_speculative_tokens` in vLLM ("the bonus token plus the mask
tokens"). Blog wording: "two-tap dynamic depthwise convolution … Each coefficient
combines a learned base kernel with a small correction computed from the current hidden
state; every 16 channels share one correction. The first position reads the last verified
token's representation" (<https://inco.ai/blog/dflash2/>).

Cost/benefit claimed: "+16.5M added parameters (3%)… The convolutions add **0.7%** to
draft–verify cycle latency; ten more Transformer layers add 15.2%… Average within-block
attention across Layers 4 and 5 also falls from 9.4% to 0.5%"
(<https://inco.ai/blog/dflash2/>).

### Addition 2 — candidate selector (this is what replaces DSpark's Markov head, and replaces per-slot argmax)

The blog names DSpark as the thing being replaced: "Recent methods such as Domino and
[DSpark](https://arxiv.org/abs/2607.05147) buy coherence with **sequential Markov heads
that rewrite each position's full-vocabulary distribution**. But is that costly
autoregressive correction really necessary? No."
(<https://inco.ai/blog/dflash2/>)

Score (blog eq., matching both PR bodies):
`S_t(a,b) = U_t(b) + ⟨ A(a) ⊙ H(h_t), B(b) ⟩` — i.e.
`edge(p→c) = ⟨A[p] ⊙ project(h), B[c]⟩ + unary[c]`. "A and B give each token a compact
256-dimensional embedding, and the two embeddings are matched under a context gate H(h_t)
… In essence, this is a low-rank bilinear attention over adjacent candidates."
(<https://inco.ai/blog/dflash2/>)

So, precisely:

* **Replaced:** "Instead of an independent argmax per slot, keep the target head's top-K
  per slot, score adjacent transitions …, and walk the best path from the verified
  anchor." (<https://github.com/vllm-project/vllm/pull/52816>) → **argmax per slot is
  replaced by a top-K + pairwise-edge + path-walk.**
* **Replaced:** DSpark's full-vocabulary sequential Markov rewrite → a K×K edge lattice
  (K=16, so 256 edges/slot) with **no extra backbone or LM-head pass**
  (<https://inco.ai/blog/dflash2/>).
* **Not replaced:** the attention, the block structure, the projector, the LM head.

Shapes as shipped: `selector_rank = 256`, `selector_top_k = 16`;
`candidate_selector.predecessor_codebook [248320, 256]`,
`candidate_selector.successor_codebook [248320, 256]`,
`candidate_selector.hidden_projection.weight [256, 5120]`. Edge tensor is
`[B, slots, K, K]` (vLLM `_score_edges` einsum `"blpr,blcr->blpc"`; SGLang walk indexes
`scores_ptr + (base + previous) * top_k`).

Losslessness: "At T>0 it walks by inverse CDF and returns q over the K candidates for the
lossless verify" (<https://github.com/sgl-project/sglang/pull/35371>); "greedy walks emit
the point mass at the token they take, and sampled walks return the realized q over the K
candidates, so the verify sees the proposal distribution it actually drew from" (ibid).
vLLM's walk uses Gumbel noise instead of inverse CDF (`tl_rand32`/`tl_rand64` +
`-log(-log u)`) — **the two engines' T>0 walks are not the same kernel**.

### Selector ablation vs the DSpark correction (Qwen3.5-4B, GSM8K, no conv)

| Method | Params | Latency | T=0 | T=1 |
|---|---|---|---|---|
| DFlash | — | — | 4.27 | 3.78 |
| + DSpark correction | +77.8M | +9.6% | 4.49 | 4.08 |
| + path selection (ours) | +2.0M | +0.6% | **4.61** | **4.25** |

Source: <https://inco.ai/blog/dflash2/> Table 2. "It beats the DSpark correction in both
settings with roughly 40× fewer parameters and 16× lower latency overhead."

> ⚠️ **Discrepancy to flag.** That `+2.0M` cannot be the shipped selector. `2×151936×256 =
> 77.8M` reproduces the DSpark row exactly (two `[vocab, 256]` matrices), and
> `3×2560×256 = 1.97M` reproduces the selector row (three `[hidden, 256]` projections) —
> so the ablation's A/B were *projections of the target embedding*. The **released
> checkpoint instead materialises two explicit `[248320, 256]` BF16 codebooks = 127.1M
> params / 254 MB** (verified from the safetensors header of
> `z-lab/Qwen3.8-27B-DFlash2`). Mathematically equivalent, ~64× the storage. Budget the
> shipped number, not the blog's.

### Headroom the design targets

Recall@1 85.4% vs Recall@16 99.5% at position 0; "An oracle that always picks the right
candidate from the top 16 would lift the acceptance length from 4.27 to 6.79. That gap is
pure selection headroom." (<https://inco.ai/blog/dflash2/> Table 1)

---

## (b) Which targets have DFlash2 drafters

**Exactly two targets.** An HF API search for `DFlash2` / `DFlash-2` / `dflash2` returns
**9 repos total**, all created 2026-08-18/19, covering two targets plus mirrors and
repacks (`https://huggingface.co/api/models?search=DFlash2`).

| Repo | Target | Bytes | Notes |
|---|---|---|---|
| `z-lab/Qwen3.8-27B-DFlash2` | `Qwen/Qwen3.8-27B` | **3,848,817,896 B (3.85 GB)** `model.safetensors`, 81 tensors, all BF16 | canonical; mirror of `incoai/…` |
| `incoai/Qwen3.8-27B-DFlash2` | same | same | the repo the launch commands name |
| `z-lab/Muse-Glimmer-30B-DFlash2` | `meta-models/Muse-Glimmer-30B` | **5,544,328,424 B (5.54 GB)** | finetuned *from* Meta's official DFlash drafter `Muse-Glimmer-30B-assistant` |
| `incoai/Muse-Glimmer-30B-DFlash2` | same | same | mirror |
| `z-lab/Qwen3.8-27B-DFlash2-GGUF` / `incoai/…-GGUF` | Qwen3.8-27B | BF16 3,860,293,152 B · Q8_0 2,056,414,752 B · Q4_K_M 1,143,006,752 B | needs llama.cpp **PR #27342** (unmerged); declares `gguf.architecture = "dflash"`, **not** "dflash2"; publishes Q4_K_M-target acceptance (see (c)) |
| `z-lab/Muse-Glimmer-30B-DFlash2-GGUF` / `incoai/…-GGUF` | Muse Glimmer | BF16 5558.1 MB · Q8_0 2959.5 MB · Q4_K_M 1645.7 MB | same |
| `ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4` | Qwen3.8-27B | 1265.6 MB | third-party, proposal-grade; no upstream MLX support claimed |

### Answers to the two targets you asked about

* **Qwen3.8-27B DFlash2 drafter — YES.** `z-lab/Qwen3.8-27B-DFlash2` /
  `incoai/Qwen3.8-27B-DFlash2`. Also pinned in vLLM's test registry as
  `speculative_model="z-lab/Qwen3.8-27B-DFlash2"` against `Qwen/Qwen3.8-27B`
  (`tests/models/registry.py`, PR #52816 diff).
* **DSV4-Flash / DeepSeek-v4 DFlash2 drafter — NO.** HF searches for
  `DSV4-Flash-DFlash2`, `DeepSeek-v4-Flash-DFlash2`, `DSV4 DFlash2`, `deepseek DFlash2`
  return **zero** results. The blog says only "We are releasing **two** DFlash 2 drafters
  today" (<https://inco.ai/blog/dflash2/>). **There is also no Qwen3.6-27B DFlash2** — the
  `lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP` repo you listed is **DFlash v1** on Qwen3.6-27B
  (unsloth `UD-Q4_K_XL` target + DFlash-v1 `Q8_0` draft, llama.cpp PR #22105); it never
  mentions DFlash2 or DSpark.

### config.json — `z-lab/Qwen3.8-27B-DFlash2` (verbatim fields)

```json
{ "architectures": ["DFlash2DraftModel"],
  "dtype": "bfloat16", "is_causal": false,
  "num_hidden_layers": 5, "hidden_size": 5120, "intermediate_size": 17408,
  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
  "layer_types": ["sliding_attention" x5], "sliding_window": 2048,
  "vocab_size": 248320, "num_target_layers": 64, "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
  "dflash_config": { "block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
                     "selector_rank": 256, "selector_top_k": 16,
                     "mask_token_id": 248070,
                     "target_layer_ids": [5, 19, 33, 47, 61] } }
```

`Muse-Glimmer-30B-DFlash2` differs: hidden 6656, intermediate 19968, vocab 202048,
`block_size 16`, `target_layer_ids [1,13,25,37,49]`, `num_target_layers 52`,
`mask_token_id 201818`, plus two fields the Qwen drafter omits —
`output_multiplier: 0.19611613513818404` and `final_logit_softcapping: 20.0`. Same
`conv_kernel_size 2`, `conv_group_size 16`, `selector_rank 256`, `selector_top_k 16`.

### Full tensor delta, DFlash2 (81) vs DFlash v1 (58) — read from safetensors headers

23 new tensors, all BF16, **388.4 MB / 194.19M params (10.1% of the 1.924B drafter)**:

| Tensor | Shape | Count | Params |
|---|---|---|---|
| `layers.{0..4}.attention_conv.base_kernel` | `[2, 2, 5120]` | 5 | 0.10M |
| `layers.{0..4}.attention_conv.kernel_projection.weight` | `[1280, 5120]` | 5 | 32.77M |
| `layers.{0..4}.mlp_conv.base_kernel` | `[2, 2, 5120]` | 5 | 0.10M |
| `layers.{0..4}.mlp_conv.kernel_projection.weight` | `[1280, 5120]` | 5 | 32.77M |
| `candidate_selector.predecessor_codebook` | `[248320, 256]` | 1 | 63.57M |
| `candidate_selector.successor_codebook` | `[248320, 256]` | 1 | 63.57M |
| `candidate_selector.hidden_projection.weight` | `[256, 5120]` | 1 | 1.31M |

The other 58 (`fc.weight [5120,25600]`, `hidden_norm`, `norm`, and the 11 per-layer
attention/MLP/norm tensors × 5) are **name-for-name and shape-for-shape identical to
DFlash v1** (verified against `z-lab/Qwen3.6-27B-DFlash`).

---

## (c) Published numbers

### Target quantization: **BF16 unquantized in every GPU-engine table — but Q4_K_M numbers exist on the llama.cpp path**

`Qwen/Qwen3.8-27B` `text_config.dtype = "bfloat16"`, 55.59 GB, no `quantization_config`
(<https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json>). All vLLM/SGLang launch
commands are bare `vllm serve Qwen/Qwen3.8-27B` / `--model-path Qwen/Qwen3.8-27B` with no
quantization flag, so **every acceptance/throughput table in (c) below is BF16 target +
BF16 draft**. No FP8 or NVFP4 number is published anywhere.

**However, quantized numbers do exist — on the GGUF cards, and they matter for SparkPipe.**
The GGUF READMEs publish acceptance length for a **quantized draft against a Q4_K_M
target** (`llama-server -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M -hfd …-DFlash2-GGUF:<quant>`):

| Draft quant | Qwen3.8-27B (target Q4_K_M) | Muse-Glimmer-30B (target Q4_K_M) |
|---|---:|---:|
| BF16 | 5.28 | 5.45 |
| Q8_0 | 5.13 | 5.58 |
| Q4_K_M | **5.39** | 5.44 |

Sources: <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF>,
<https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2-GGUF>. Two readings, both useful:
**(i)** the drafter is essentially insensitive to its own quantization (5.13–5.39 spans
less than the BF16-vs-Q4 ordering can resolve — Q4_K_M scores *above* BF16, so this is
noise, not a trend); **(ii)** a **Q4_K_M target — including a quantized LM head — reaches
5.28–5.39 acceptance vs 5.46 on the BF16 target** (SGLang card, GSM8K). That is a ~1–3%
give-up, and it directly contradicts reading the unquantized-LM-head rule as an
*algorithmic* requirement. See the reassessed blocker in (e).

### Acceptance length, per-request mean (block 8 = 7 draft tokens), H200, SGLang + FA3

Source: <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2> (= blog Table 4).

> ⚠️ **Baseline-provenance caveat — the DSpark column is an off-label run of that drafter,
> and it matters directly to SparkPipe.** `RadixArk/Qwen3.8-27B-DSpark` states: "A DSpark
> speculator for **Qwen/Qwen3.8-27B-FP8**", "Target model: Qwen/Qwen3.8-27B-FP8",
> "Setting: **FP8 target** and unquantized BF16 draft; DSpark block size 7; sampling
> **temperature 0.6**, top-k 20, top-p 0.95; `max_new_tokens=2048`", and serves with
> `--model-path Qwen/Qwen3.8-27B-FP8 --speculative-draft-model-quantization unquant`
> (<https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>). Its **own** published GSM8K
> acceptance is **4.57**.
>
> The DFlash2 comparison runs that drafter against a **BF16** target at **temperature 1.0**,
> xhigh reasoning, 4096 max new tokens — and reports DSpark GSM8K **4.36** (−4.6% vs the
> drafter's own card). So the baseline is benchmarked outside the configuration it was
> trained and validated for. The gap is small and the harsher sampling plausibly explains
> it, so this is not evidence of a rigged comparison — but it does mean the headline
> **"+32.6% vs DSpark" is measured against a weakened DSpark**. SparkPipe's landed port
> *is* the FP8-target DSpark configuration, so **expect a smaller DFlash2 delta against
> your own baseline than the cards advertise.** Budget on acceptance-length *absolutes*
> (DFlash2 4.80 mean / 5.46 GSM8K) rather than on the DSpark-relative percentage.

| Task | MTP | DSpark | DFlash 2 | vs DSpark *(derived)* |
|---|---:|---:|---:|---:|
| GSM8K | 5.02 | 4.36 | **5.46** | +25.2% |
| MATH-500 | 4.72 | 3.92 | **5.28** | +34.7% |
| HumanEval | 3.91 | 3.30 | **4.39** | +33.0% |
| MBPP | 3.99 | 3.51 | **4.79** | +36.5% |
| MT-Bench | 3.74 | 3.01 | **4.10** | +36.2% |
| **Mean** | 4.28 | 3.62 | **4.80** | **+32.6%** |

### End-to-end throughput, output tok/s (speedup vs autoregressive), Qwen3.8-27B

| conc | Task | AR | MTP | DSpark | DFlash 2 | vs DSpark *(derived)* |
|---|---|---:|---:|---:|---:|---:|
| 1 | GSM8K | 68.9 | 178.5 (2.59×) | 185.3 (2.69×) | **236.1 (3.43×)** | +27.4% |
| 1 | MT-Bench | 68.9 | 134.9 (1.96×) | 137.6 (2.00×) | **184.0 (2.67×)** | +33.7% |
| 8 | GSM8K | 467.2 | 1022.1 (2.19×) | 1040.8 (2.23×) | **1328.7 (2.84×)** | +27.7% |
| 32 | GSM8K | 1329.8 | 1381.1 (1.04×) | 1506.5 (1.13×) | **1922.5 (1.45×)** | +27.6% |
| 32 | MT-Bench | 1507.4 | 1159.7 (0.77×) | 1115.5 (0.74×) | **1525.3 (1.01×)** | +36.7% |

Full 3×5 grid on the card. Over all 15 throughput cells: **mean +34.0% vs DSpark, range
+27.4%…+38.9%** *(derived)*. Speedup vs no-spec: **3.11–3.43× at conc 1, 2.27–2.85× at
conc 8, 1.01–1.45× at conc 32**; blog headline "2.7–3.4× the throughput of autoregressive
decoding … at batch size 1".

### Muse-Glimmer-30B (block 16 = 15 draft tokens)

<https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2>. Acceptance mean: official DFlash
4.44, DSpark 4.48, **DFlash2 5.70** (+27.2% vs DSpark *(derived)*). Throughput conc 1
GSM8K: AR 63.9, DFlash 247.8 (3.88×), DSpark 236.5 (3.70×), **DFlash2 293.7 (4.59×)**.
Speedup vs no-spec 3.08–4.62× (conc 1) down to 1.15–1.68× (conc 32). Over all 15
throughput cells: **mean +29.9% vs DSpark** *(derived)*.

### vLLM's own independent measurement (PR #52816, GSM8K, Qwen3.8-27B, H200)

Acceptance per-request mean 4.27 → **5.34** (+25.2%); pooled ratio 3.64 → **4.65**
(+27.8%). Throughput: conc 1 178.5 → **224.6** (2.79× → 3.51×), conc 8 966.2 → **1211.7**,
conc 32 2220.9 → **2759.4** — **+24.2%…+25.8% vs DSpark** *(derived)*, i.e. the vLLM path
reproduces the direction but lands ~8 points below the SGLang card on throughput.
"Acceptance does not move with concurrency: DFlash 2's pooled ratio reads 4.65, 4.82 and
4.74 at concurrency 1, 8 and 32."

### Measured cost of the two new components (PR #52816, per-component CUDA-graph timing)

At 5 layers / hidden 5120 / vocab 248320 / block 8 / K=16:

| batch | conv | top-k | lattice | walk | selector | conv+selector | serving step | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.113 | 0.041 | 0.014 | 0.006 | 0.061 | **0.174 ms** | 20.70 ms | **0.84%** |
| 8 | 0.121 | 0.084 | 0.016 | 0.006 | 0.106 | **0.227 ms** | 31.80 ms | **0.71%** |
| 32 | 0.173 | 0.173 | 0.018 | 0.006 | 0.197 | **0.370 ms** | 54.93 ms | **0.67%** |

Key facts: "The selector's own math — the lattice and the walk — is 0.020 ms and flat in
batch; the rest of that column is the vocabulary top-k". "The convolution is 20 small
calls over `[8, 5120]` tensors, so it is launch-bound rather than FLOP-bound: 0.477 ms as
eager modules, **0.0096 ms compiled as a single graph** across all 20 calls, and 0.113 ms
with each call compiled on its own… A hand-fused kernel would collect most of the
remaining 12×." SGLang's framing: cost is **1.5–2.7%** of the cycle and falls with batch
(<https://github.com/sgl-project/sglang/pull/35371>).

### ⚠️ The "30% faster than DSpark" claim — NOT VERIFIABLE as a published claim

**No primary source states it.** Checked: the blog, both PR bodies and all comments, all
three model cards, the `z-lab/dflash` repo (README/releases/tags), and the circulating
zhihu post (HTTP 403 direct; retrieved via proxy — it covers **DFlash v1** vs DSpark, not
DFlash2). The blog uses the word "faster" **zero times**, and its only "30%" is
*attention mass* ("30% in Layer 1 to 8% in Layer 5") — nothing to do with speed.

What the primary sources actually claim vs DSpark:
* "on both models, DFlash 2 averages **more than a full token** ahead of DSpark" (blog)
* "**0.48** [tokens] over DSpark" on Qwen3.5-4B (blog Table 3 = +8.7% *(derived)*)
* "beats the DSpark correction … with roughly **40× fewer parameters and 16× lower
  latency overhead**" (blog Table 2)

If "30%" is anyone's paraphrase, it lands on **two metrics at once, both for Qwen3.8-27B
BF16**: acceptance length **+32.6%** and end-to-end output tok/s **mean +34.0%** *(both
derived)*; for Muse-Glimmer-30B the same two are **+27.2%** and **+29.9%**. These nearly
coincide precisely because the drafter's own cost is ~1%.

**It is definitively NOT draft cost.** Draft cost moves the *wrong* way: DFlash2 *adds*
0.7% (conv) + 0.6% (selector) ≈ **1.3% cycle latency** over DFlash v1 (blog). The
DSpark-relative cost claim is a different number entirely (16× lower *overhead* than the
DSpark Markov correction, blog Table 2).

**Treat "30% faster than DSpark" as unattributed.** Quote instead: "+32.6% acceptance
length and +34% end-to-end output tok/s vs `RadixArk/Qwen3.8-27B-DSpark` on BF16
Qwen3.8-27B, H200, SGLang, derived from the published model-card tables."

---

## (d) Launch config

**No new method name, no new server args, no new env vars.** The method stays `dflash`;
DFlash2 is selected purely by the checkpoint's `architectures: ["DFlash2DraftModel"]`.
"No new server args, no new env vars, nothing outside DFlash touched."
(<https://github.com/sgl-project/sglang/pull/35371>)

```bash
# vLLM (PR #52816, unmerged) - num_speculative_tokens = block_size - 1
vllm serve Qwen/Qwen3.8-27B \
  --speculative-config '{"method": "dflash",
                         "model": "incoai/Qwen3.8-27B-DFlash2",
                         "num_speculative_tokens": 7}'

# SGLang (PR #35371, MERGED) - --speculative-num-draft-tokens = block_size
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8

# llama.cpp (PR #27342, unmerged)
llama-server -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M \
  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
  --spec-type draft-dflash --spec-draft-n-max 7
```
Sources: <https://inco.ai/blog/dflash2/>, <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2>.
Muse Glimmer uses `num_speculative_tokens: 15` / `--speculative-num-draft-tokens 16`
(block 16).

| Knob | Value | Note |
|---|---|---|
| method / algorithm | `dflash` / `DFLASH` | **unchanged from DFlash v1** |
| `num_speculative_tokens` (vLLM) | **7** (Qwen3.8-27B), **15** (Muse) | `= block_size − 1` |
| `--speculative-num-draft-tokens` (SGLang) | **8** / **16** | `= block_size` |
| `draft_sample_method` | **not a DFlash2 knob; effectively ignored** | see below |
| Model runner | **V2 forced** | `use_v2_model_runner` returns True for a DFlash2 draft |
| `conv_kernel_size`/`conv_group_size`/`selector_rank`/`selector_top_k` | **from checkpoint only** | each pair must be set together or SGLang raises |

**`draft_sample_method` — the important subtlety.** In vLLM's V2 speculator base,
`self.draft_logits` is allocated **only** when `draft_sample_method == "probabilistic"`.
`DFlash2Speculator.__init__` **unconditionally overwrites it** with a
`[max_num_reqs, num_speculative_steps, vocab_size]` **float32** `-inf` tensor: "The
selector samples a probabilistic path for non-greedy requests, so rejection sampling
always needs the realized proposal distribution." Test
`test_selector_always_keeps_proposal_logits` pins this. **So DFlash2 always pays the
probabilistic-path memory**, whatever `draft_sample_method` says. At
`max_num_reqs=256, steps=7, vocab=248320` that buffer is **1.78 GB fp32** *(derived)* —
material on a 128 GiB unified-memory DGX Spark.

**V1/V2 trap.** "the V1 `DFlashProposer` has no candidate selector, so a DFlash2
checkpoint reaching it would draft as DFlash1 **without raising**". Hence `use_v2_model_runner`
force + `test_dflash2_draft_forces_v2_model_runner`. Any SparkPipe dispatch that mirrors
this must **fail loudly** rather than silently degrade.

**Hard constraint — the target LM head must be unquantized.**
`compute_candidates` raises `"DFlash2 requires an unquantized target LM head for
candidate TopK."` unless `self.lm_head.quant_method` is `UnquantizedEmbeddingMethod`;
SGLang adds `is_dense_head_weight()` allowing only `{float16, bfloat16, float32}` with the
comment "is_floating_point() is True for fp8; list dtypes explicitly." SGLang's test suite
covers "the rejection of a quantized target LM head."

**Known open defect.** vLLM PR #52816 comment from `timothysu` (2026-08-19): with
"Qwen3.8-27B bf16 with the published DFlash2 drafter on sm120", concurrency 4 crashes with
`vectorized_gather_kernel: index out of bounds`, not yet reproducible
(<https://github.com/vllm-project/vllm/pull/52816#issuecomment-5335993561>). **sm120 is
adjacent to SparkPipe's GB10/SM121 target.** Treat concurrency > 1 as unproven.

---

## (e) Adoption path for the landed SparkPipe DSpark port

### What is already landed (inventory)

| Landed piece | Path |
|---|---|
| Packer | `tools/qwen36_dspark_stagepack.py` (294 L) — 17 tensor kinds, 62 tensors, magic `Q6SP`, header `26I2Q`, entry `6I4Q` |
| Pack format / ABI | `modules/qwen36_resident_decode_stage/source/spark_qwen36_dspark_format.h` |
| Neutral drafter constants | `model-families/common/include/sparkpipe/spark_dspark_drafter.h` (ABI v3, GLM52/K3/DSV4-Pro tables) |
| Module forward | `modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_module.c` (2709 L) |
| Draft kernels | `modules/qwen36_resident_decode_stage/source/spark_qwen36_dspark_cuda.cuh` — `SparkQwen36DsparkAttnKernel`, `SparkQwen36DsparkMarkovKernel` |
| Serving adapter | `modules/qwen36_resident_decode_stage/source/spark_qwen36_serving_adapter.c` (1870 L), `SPARK_QWEN36_SERVING_SPEC_METHOD_DSPARK` |
| Parity harness | `tools/qwen36_dspark_reference.py` (236 L) — numpy-only forward oracle |
| Policy | `src/spark_speculation_policy.c` (798 L) |

### Shape reality check — the landed port targets a DIFFERENT drafter geometry

Landed port = `Doopeworld/Qwen3.8-27B-DSpark-vLLM` ≡ `RadixArk/Qwen3.8-27B-DSpark`
(<https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/main/config.json>). Same *target*
model, materially different *drafter*:

| Field | Landed DSpark | DFlash2 | Impact |
|---|---|---|---|
| `layer_types` | `full_attention` ×5 | `sliding_attention` ×5, `sliding_window: 2048` | **attn kernel must gain a window mask** |
| `is_causal` | (absent) | **`false`** | new explicit-causality field, honored ahead of `dflash_config.causal` (see `_dflash_layer_causal` change in PR #52816) |
| `num_attention_heads` | 40 | **32** | `SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS`; `40u` is **hardcoded** in `SparkQwen36DsparkAttnKernel` |
| `intermediate_size` | 10240 | **17408** | FFN shapes |
| `block_size` | 7 | **8** | verify-window / slot arithmetic |
| `target_layer_ids` | `[4,16,28,40,52]` | `[5,19,33,47,61]` | tap plumbing (still 5 taps) |
| `mask_token_id` | 248077 | **248070** | constant |
| `rope` | `yarn`, factor 32 | `default`, theta 1e7 | RoPE table |
| confidence head | present, required | **absent** | policy flags |
| `markov_rank` / `selector_rank` | 256 | 256 | ✅ same |

### Reused UNCHANGED

1. **Pack wire format** — magic `Q6SP`, header `26I2Q`, entry `6I4Q`, the
   global-vs-per-layer entry scheme, SHA256SUMS/PACKAGE_MANIFEST integration. Zero change.
2. **Packer skeleton and 14 of 17 HF-name mappings.** DFlash2 keeps DFlash v1's names, and
   the packer's map already reads them: `KIND_PROJECTOR: "fc.weight"` ✅,
   `KIND_FINAL_NORM: "norm.weight"` ✅, `KIND_HIDDEN_NORM: "hidden_norm.weight"` ✅, and
   all 11 `PER_LAYER_KINDS` (`self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`,
   `input_layernorm`, `post_attention_layernorm`, `mlp.{gate,up,down}_proj`) ✅ —
   **name-for-name identical**. Only shape constants change.
3. **`SPARK_DSPARK_TENSOR_MARKOV_W1/W2` slots at exactly the right shape.** Landed
   `(VOCAB=248320, MARKOV_RANK=256)` ×2 **is bit-identical to DFlash2's
   `predecessor_codebook`/`successor_codebook` `[248320, 256]` ×2 (254 MB)**. The pack
   slot, the `markov_w1_host`/`markov_w2_host` D2H mirror
   (`SparkQwen36ModuleRunMtp…` init path), and the byte accounting all carry over
   verbatim. **This is the single largest reuse win.**
4. **BF16-truncation discipline.** The landed Markov kernel already enforces
   "truncate → BF16 BEFORE the add" so device/host/numpy agree on near-ties. The selector
   needs the same convention; the rule and its test scaffolding exist.
5. **Target-hidden tap capture** (5 taps → `fc`), **shared target embed/lm_head**
   (drafter still ships neither), **mask-token block construction**, **resident
   decode-stage slot machinery**, **`SparkQwen36LaunchHeadScreenedArgmaxScore`**
   (returns value+index → the top-K primitive's ancestor).
6. **Serving adapter frame plumbing** — `SparkQwen36DsparkDraftView`,
   `SPARK_…_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER`, the speculative-frame call path. Only
   the draft-count constant and one method enum move.
7. **Parity-harness architecture** — numpy-only, reads safetensors directly, synthesizes
   taps to test the FORWARD not the target. Structure is exactly right; add layers.

### Requires NEW work

**W1 — Packer: +3 tensor kinds, +2 conv kinds per layer (~1 day).**
17 → 22 kinds. New per-layer: `CONV_ATTN_BASE` `[2,2,5120]`, `CONV_ATTN_PROJ`
`[1280,5120]`, `CONV_MLP_BASE`, `CONV_MLP_PROJ`. New global: `SELECTOR_PRED`
`[248320,256]`, `SELECTOR_SUCC` `[248320,256]`, `SELECTOR_HIDDEN_PROJ` `[256,5120]`.
**Drop** `KIND_CONFIDENCE` (+ its `0xFFFFFFFE` bias rider) — DFlash2 has no confidence
head. Note `base_kernel` is the port's **first rank-3 tensor**; the `6I4Q` entry carries
rows/cols only, so flatten as `[2, 2*5120]` or `[4, 5120]` and pin the choice in the
format header. 81 tensors, 3.85 GB.

**W2 — New kernel: grouped dynamic depthwise conv (~2–3 days). The main kernel work.**
Not a stock separable depthwise conv — the coefficients are **dynamic** (per-token, from
`kernel_projection(x)`) with a learned base and **hard zeroing across the block
boundary** (`position >= tap` where `position = i % block_size`, fast-pathed to
`i & (block_size-1)` for power-of-two blocks). Two taps only, so it is
`out[i] = c0[i]⊙x[i] + c1[i]⊙x[i-1]·(pos≥1)` — a fused elementwise pass, no im2col.
**Fuse aggressively:** upstream measured 0.477 ms eager → **0.0096 ms as one graph** (50×);
they left 12× on the table because PyTorch compiles each of the 20 calls separately.
SparkPipe writes CUDA directly and can take the whole 50×. Reference:
`_grouped_conv` + `test_grouped_conv_matches_reference` in PR #52816 give an exact oracle.

**W3 — Candidate selector: extend the existing Markov kernel (~2 days, NOT from scratch).**
Landed: `bias[v] = Σ_r w1[prev,r]·w2[v,r]` over the **full 248320 vocab**, one predecessor,
then full-vocab argmax, sequential over draft positions.
Needed: `score[p,c] = unary[c] + Σ_r (A[pred_id[p],r]·H[r])·B[cand_id[c],r]` over
**K=16 gathered candidates**, 16 predecessors, with a per-slot context gate.
Three deltas: (i) gather 16 candidate rows instead of striding the vocab — **1552× less
work per slot**; (ii) multiply the predecessor row elementwise by
`H = hidden_projection(h_t)` (`[256,5120]` matvec, new but trivial); (iii) emit a
`[slots,16,16]` lattice instead of a `[slots,vocab]` bias. Oracle:
`_score_edges` + `test_selector_edges_match_sequential_reference`.

**W4 — Top-K over the vocabulary (~1–2 days). Budget this as the selector's real cost.**
Upstream is explicit: "the vocabulary top-k, the selector's largest single cost" — 0.041 ms
(b=1) to 0.173 ms (b=32), i.e. **67–88% of the whole selector column**; the lattice+walk is
0.020 ms and flat. vLLM uses FlashInfer's radix kernel (1.9× `torch.topk` at b=1, 4.5× at
b=32) and falls back otherwise. SparkPipe has **no FlashInfer**, so this is a
first-class own-kernel item: a radix/bitonic top-16 over 248320 BF16 logits per slot.
`SparkQwen36LaunchHeadScreenedArgmaxScore` (fused matvec + value/index reduction over one
row) is the right starting point — generalise its reduction from top-1 to top-16.

**W5 — Path-walk kernel (~1 day). Small, and pick your engine.**
One program per request; K scores in registers; the slot-to-slot dependency is a **loop
inside the program**, not a kernel per slot. **The two upstreams differ at T>0** — decide
deliberately:
* SGLang: softmax over K → inverse CDF from a host-supplied uniform; greedy emits a
  one-hot q. Simpler; folds into the draft CUDA graph.
* vLLM: Gumbel (`tl_rand32`/`tl_rand64`, `-log(-log u)`) seeded per request+position;
  separate `_cache_draft_logits_kernel` scatters K scores into the `[reqs,steps,vocab]`
  fp32 buffer and re-`-inf`s the previous step's slots.

SparkPipe's landed verify loop already threads probabilistic draft probs, so **the
SGLang-style one-hot/realized-q form is the cheaper fit** — it avoids materialising the
1.78 GB fp32 vocab buffer (see (d)).

**W6 — Attention: full → sliding-window 2048, non-causal (~1 day).**
`SparkQwen36DsparkAttnKernel` needs a window bound and must honor `is_causal: false`.
Also unhardcode `40u` (query heads) → 32 and widen FFN 10240 → 17408.

**W7 — Policy and dispatch (~0.5 day).**
Clear `SPARK_DSPARK_POLICY_FLAG_REQUIRE_CONFIDENCE_HEAD` for DFlash2 (no confidence head
→ the confidence-milli floor has no input; either drop the gate or resynthesize a proxy
from the selector's realized q). Keep `REQUIRE_MARKOV_HEAD` **renamed**, not removed —
the tensors are there under a new meaning. Add a
`SPARK_QWEN36_SERVING_SPEC_METHOD_DFLASH2` enum and — mirroring vLLM's V1 trap — make an
unsupported-path dispatch **fail loudly**, never silently draft without the selector.
Add `SPARK_DSPARK_TARGET_QWEN38_DFLASH2` to `spark_dspark_drafter.h` with
`SELECTOR_TOP_K 16`, `CONV_KERNEL_SIZE 2`, `CONV_GROUP_SIZE 16`, and bump
`SPARK_DSPARK_ABI_VERSION` 3 → 4.

**W8 — Parity harness: extend `qwen36_dspark_reference.py` (~1 day).**
Add `_grouped_conv`, the selector lattice, the walk. Constants change:
`N_Q_HEADS 40→32`, `FFN 10240→17408`, `BLOCK 7→8`, `TAP_LAYERS→(5,19,33,47,61)`,
`MASK_TOKEN_ID 248077→248070`, `ROPE` yarn→default, + sliding window 2048. Port both
upstream unit tests as numpy oracles first — they are exact and cheap.

### Quantization constraint for SparkPipe — an ENGINE limit, not an algorithmic one

**vLLM and SGLang both refuse a quantized target LM head**: vLLM raises "DFlash2 requires
an unquantized target LM head for candidate TopK" unless `lm_head.quant_method` is
`UnquantizedEmbeddingMethod`; SGLang's `is_dense_head_weight` admits only
`{float16, bfloat16, float32}` with the comment "is_floating_point() is True for fp8; list
dtypes explicitly." SparkPipe's Qwen3.8-27B work is FP8/NVFP4 oriented
(`qwen38_tp4_build.sh`, `docs/QWEN38-27B_HILLCLIMB.md`), so on a straight port this bites.

**But it is a property of those two implementations, not of DFlash2.** The llama.cpp path
runs DFlash2 against a **Q4_K_M target — quantized LM head included — and reports 5.28–5.39
acceptance** (<https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF>), against 5.46 on the
BF16 target. The selector only needs *top-16 ids + their scores*; nothing requires those
logits to come from a dense BF16 matmul. SparkPipe writes its own head kernel and already
has `SparkQwen36LaunchHeadScreenedArgmaxScore` producing value+index from a **quantized,
shadow-screened** head — i.e. **SparkPipe is better placed here than either upstream.**

Three options, in preference order:
1. **Do the top-K on the quantized head directly** (extend `…ScreenedArgmaxScore` from
   top-1 to top-16, W4). Follows llama.cpp's precedent; costs ~1–3% acceptance per the
   GGUF-vs-BF16 comparison; no extra weight copy.
2. **Keep the LM head in BF16 while the body stays FP8/NVFP4.** Upstream-faithful, exact,
   but costs a 248320×5120 BF16 head resident (**2.54 GB**) on a 128 GiB unified box.
3. Do not adopt on quantized targets. Not warranted given (1).

Either way: **no FP8/NVFP4 acceptance number exists upstream**, so re-tune
`num_speculative_tokens` locally — the landed `docs/RUNG3_DSPARK_ADOPTION.md` already
records that "k does NOT transfer across quantization." Also note the third-party MLX
affine-4 repack **kept the selector codebooks in BF16** while quantizing 47 linear modules
(<https://huggingface.co/ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4>) — a reasonable
default for SparkPipe's own pack: quantize the backbone, leave `[248320,256]×2` alone.

### Verdict

**Adopt — medium effort, ~8–11 engineer-days, low architectural risk.** DFlash2 is
additive to DFlash/DSpark, and SparkPipe's DSpark port already owns the two expensive
structural pieces (the 5-tap projector path and a rank-256 `[vocab,256]×2` bilinear head
at *exactly* DFlash2's selector shapes). The real new work is three kernels — dynamic
depthwise conv (W2), top-16-over-vocab (W4, the sleeper cost), and the path walk (W5) —
plus mechanical reshaping. Nothing requires a new pack format, a new ABI wire layout, a
new serving path, or a second backbone.

**Sequence:** W1 packer → W8 numpy oracle (cheap, exact, de-risks everything) → W6 attn
reshape → W2 conv → W4 top-K **on the quantized head** → W3 selector → W5 walk → W7 policy.
The FP8 LM-head question is no longer a gating blocker (llama.cpp reaches 5.28–5.39
acceptance on a Q4_K_M target); it is a design choice inside W4.

**Four gates before committing serving capacity:**

1. **vLLM #52816 is unmerged** and has an **open sm120 out-of-bounds crash at concurrency
   4** — adjacent to GB10/SM121. Only SGLang is merged.
2. **No DFlash2 drafter exists for DSV4-Flash, DSV4-Pro, K3, GLM5.2, or Qwen3.6-27B**, so
   this is a Qwen3.8-27B-only adoption unless SparkPipe trains its own. The blog invites
   exactly that: "want a drafter for a model you run, including your own fine-tunes, write
   to us" (<https://inco.ai/blog/dflash2/>).
3. **No upstream reference implementation exists** (see the header warning). The three
   engine PRs plus 23 checkpoint tensors are the whole specification; there is no training
   code, no paper, and no canonical forward. **W8 is therefore not optional** — the numpy
   oracle built from the two vLLM unit tests is the only ground truth SparkPipe will have,
   and it must land before W2/W3 rather than after.
4. **Do not budget on the DSpark-relative percentages.** They are measured against an
   FP8-trained DSpark drafter run off-label on a BF16 target (see the caveat in (c)).
   SparkPipe's landed port *is* that FP8 configuration, so the local delta will be smaller
   than +32.6%. Set the acceptance target from DFlash2's **absolutes** — 4.80 mean / 5.46
   GSM8K at block 8 on BF16, or 5.28–5.39 on a Q4_K_M target — measured against SparkPipe's
   own DSpark baseline on identical sampling.

## Perf addendum (independent lane-2 measurement, spark4 GB10, 2026-08-19)

W4 fused cost at production shape (7 rows x 248320 x 5120, K=16): **17.1 ms**,
~149 GB/s effective - the 2.54 GiB BF16 head READ dominates (~54% of GB10
bandwidth). Hard floors for the head read: BF16 ~9.3 ms, FP8 ~4.7 ms,
NVFP4 ~2.3 ms per drafter step. Upstream's 0.041 ms 'top-k' is over an
already-materialised logits row - not comparable to a fused matvec.
Conclusion: option 1 (top-K on a quantized/screened head, feeding the
reduction a candidate list instead of the dense vocab) is the post-adoption
perf lever; the reduction itself is already generic in candidate_count.
Measurement caveat: another lane's process appeared on spark4 mid-run;
correctness unaffected, latencies 17.0-18.2 ms across repeats.

Independent cross-check: lane 2's clean-room implementation (1009 lines,
mutation-tested - 5 of 5 mutations caught, production-shape bit-exact
parity) confirms the landed selector contract; per the DRY law the landed
implementation (cfe1813) remains the single in-tree implementation.
