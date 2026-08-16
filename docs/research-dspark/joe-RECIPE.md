# DeepSeek-V4-Flash-0731 · TP=4 · 4× DGX Spark

**123.13 tok/s** single-stream decode. Here is the config, then every variable I tested to get
there, including the ones that lost. The tuning target throughout is my real agentic workload
(see the README's premise section), not a benchmark, where a knob's answer depends on the
workload, I say so.

```bash
vllm serve /path/to/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731 --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
  --max-model-len 393216 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 8264 \
  --max-cudagraph-capture-size 96 \
  --gpu-memory-utilization 0.85 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --override-generation-config '{"temperature":0.0}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,
      "draft_sample_method":"probabilistic"}' \
  --enable-prefix-caching --async-scheduling --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --moe-backend flashinfer_b12x --enable-flashinfer-autotune \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"thinking":false}' \
  --nnodes 4 --node-rank $RANK --master-addr $MASTER --master-port 25310
```

Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2). 4× GB10, 200G RoCE, `--ulimit
nofile=1048576` (TP=4 opens 4× the NCCL sockets and will fail without it).

Result: KV pool **54.42 GiB / 4,676,943 tokens**, 11.89× concurrency at full context.

---

## Every variable, one at a time

### The batch-aware ladder, I built it, measured it, and removed it

Earlier versions of this recipe used `num_speculative_tokens_per_batch_size:
[[1,4,7],[5,8,5],[9,12,3]]`, deeper drafts at low concurrency, shallower under load. Sounded
smart. Measured (same ~105K prompt, same protocol, one boot apart):

```
                       deep C=1     ms/step     storm C=12 aggregate
k=7 with ladder        64.2         57.7        136.1
k=7 static, no ladder  68.6         54.7        126.6
```

The ladder costs ~7% single-stream for a ~7% aggregate win at C=12, and my real traffic
averages concurrency 1.3–2.4. The mechanism itself carries per-step overhead even when only
the first rung is active. If your workload lives at high sustained concurrency, the ladder may
earn its keep; mine doesn't, so it's gone.

### `num_speculative_tokens`, the full sweep, and why community advice didn't transfer

Everyone told me k=3 (one 4-node cluster runs it) or k=5 (a 2-node recipe corrected to it
it matches the checkpoint's declared `dspark_block_size`). I ran the whole bracket at ~105K
context, one boot each:

```
k=3    40.6 tok/s    69.9 ms/step    accept 61.2%
k=5    49.3          70.3            accept 51.9%
k=7    68.6          54.7            accept 39.4%     <- kept
k=8    55.2          59.7            accept 28.9%
```

Two things worth staring at. First, shallower k has SLOWER steps here, the opposite of the
verify-cost intuition. This image's kernels are built as `dspark7`; its graphs and paths are
specialized for k=7's shapes. Second, acceptance falls as k rises but tokens-per-step more
than compensates until k=8, where acceptance collapses past the drafter's trained reach.
The general lesson: k is a property of your ENGINE BUILD, not of the model. Community numbers
from other engines told me nothing about mine, and measuring took an evening.

### `draft_sample_method: greedy` at temperature 0, the theory that measured zero

At temp 0, vLLM's rejection sampler accepts a draft token only if it equals the target's
argmax. So probabilistic drafting computes a full distribution just to sample from it, greedy
should be strictly better, and I predicted 3–13%. Measured: acceptance +0.2pt, decode within
run noise. The draft distribution at temp 0 is so peaked that sampling lands on the argmax
essentially every time. Probabilistic stays (it's what the non-zero-temperature path needs
anyway). Prediction wrong, question closed, one boot spent.

### `max_num_batched_tokens` 8264 vs 16456, a real trade, measured both ways

Raising the prefill chunk to 16456 cut cold 150K-prompt admission from 83s to 46s TTFT under
concurrent load, and posted the best storm aggregate I measured (140+ tok/s). I ran it in
production for a day and reverted. Why: every prefill chunk stalls ALL concurrent decode
streams for that step, and doubling the chunk doubles the stall. With a warm prefix cache
(96–98% hits here) cold admissions are rare, so the win almost never fires, but the decode
chop under concurrent deep sessions is constant. If your cache runs cold or your traffic is
admission-heavy, take 16456. If your sessions are long-lived and cache-warm, keep 8264.


I changed one thing per boot and measured. Nothing below is inferred from a spec sheet.

### `draft_sample_method`, the single biggest win

| value | acceptance (production shape) | tok/step |
|---|---|---|
| `greedy` | 26.5% | 2.86 |
| **`probabilistic`** | **34.3%** | **3.40** |

**~19% more output speed from one field.** With greedy drafting the draft is a point mass, so
acceptance collapses to `p_target(argmax)`, fine at temperature 0, bad above it. Probabilistic
matches the target's distribution.

I had `probabilistic` in my GLM config for weeks and never carried it across. Both values were
written down; neither note said what the field *did*.

### `num_speculative_tokens` (k), deeper is worse, and not for the obvious reason

| k | capture | acceptance | tok/step | ms/step | **tok/s** (c=1) | tok/s (c=4, per-stream) |
|---|---|---|---|---|---|---|
| **7** | 96 | **72.3%** | **6.059** | **49.53** | **122.27** | **60.64** |
| 8 | 108 | 64.1% | 6.130 | 53.15 | 115.16 | 58.80 |
| 10 | 132 | 46.5% | 5.650 | 55.03 | 102.57 | 53.07 |

At k=7 my position-6 acceptance was still 79.4%, so I expected k=8 and k=10 to pay. They don't.
**Asking the draft head to go deeper degrades acceptance at every position, not just the new
ones**, 72.3% → 64.1% → 46.5%. Step time rises too. Both factors move against you.

I built a model that reproduced the k=7 result to three decimals and predicted 6.745 tok/step at
k=10. Actual: 5.650. **The per-position curve is not invariant to k.** You cannot extrapolate
deeper drafting from a shallower measurement.

### `max_cudagraph_capture_size`, a formula, not a constant

```
capture = max_num_seqs × (k + 1)        # 12 × 8 = 96
```

Three published configs I checked all use **36**. That is correct for *their* shapes, 6 seqs ×
k=5 is exactly 36, and wrong for mine. I copied it, and vLLM logged:

```
WARNING [vllm.py:1804] Truncating max_cudagraph_capture_size to 32
```

32 covers 4 sequences. Above that, decode falls out of CUDA graphs into eager. My p95 concurrency
is 4 and my peak is 14, so roughly half my traffic was running eager, **and my benchmark could
not see it, because I was measuring at concurrency 1**, where the batch is 8 tokens and always fits.

Measured at gmu 0.85: capture 96 → 54.23 GiB, capture 36 → 54.26 GiB. It costs no memory. Size it
correctly and stop thinking about it.

### `gpu_memory_utilization`, pinned between a loss and an OOM

| value | KV pool | concurrency |
|---|---|---|
| 0.80 | 47.34 GiB | 11.03× |
| **0.85** | **54.26 GiB** | **12.62×** |
| 0.90 | will not boot |, |

0.80 costs **6.9 GiB for nothing**. 0.90 doesn't boot: at 0.85 these boxes already sit at 98%
memory with ~2 GiB available, and 0.90 needs ~6 more.

I lowered it to 0.80 because another operator ran 0.80. His topology holds ~77 GB of weights per
box; at TP=4 I hold ~39 GB. **His 0.80 was a fix for his headroom problem, not a better setting.**
Same mistake as capture 36. Twice in one day.

### `max_num_batched_tokens`, 8264, not 8192

With spec decode on, vLLM subtracts draft slots from the prefill budget:

```
max_num_scheduled_tokens = max_num_batched_tokens − (k−1) × max_num_seqs
```

and warns when the result drops below 8192. Every published config I looked at sets `8192` raw
and lands *under* it, at k=6/seqs=16 that's 8112.

```
8264 = 8192 + (7−1) × 12     → effective budget exactly 8192
```

### `max_model_len`, the ceiling buys capacity, not memory

| ctx | KV pool | tokens | bytes/token |
|---|---|---|---|
| 327,680 | 54.23 GiB | 4,138,637 | 14,070 |
| **393,216** | **54.42 GiB** | **4,696,436** | **12,442** |
| 524,288 | 56.24 GiB | 5,630,097 | 10,726 |
| 786,432 | 54.56 GiB | 7,008,697 | 8,357 |
| 1,048,576 | 54.1 GiB | 7,932,105 | 7,324 |

**The pool is flat, 54.1 to 56.2 GiB across a 3.2× range.** Bytes-per-token falls because
sparse attention doesn't retain everything; that's why the model advertises "10% of V3.2's KV at
1M context." Raising the ceiling gets you token capacity, not memory. Memory is `gpu_memory_utilization`.

1M boots fine. I run 393,216 anyway: my largest real prompt is 327.4K, and at a 1M ceiling one
runaway request can occupy ~24% of the pool and preempt others. 393,216 is also the threshold
`reasoning_effort: "max"` requires.

### `cudagraph_mode: FULL_DECODE_ONLY`, free

Same KV, same concurrency, one graph set captured instead of two. No measured gain, no measured
cost. Kept on principle.

### `--override-generation-config '{"temperature":0.0}'`

`--generation-config vllm` means a client that omits `temperature` gets vLLM's default of **1.0**,
and MTP acceptance drops at high temperature. Proof it took effect: before, my omitted-temperature
case tracked temp 1.0 (26.5% vs 22.8%); after, it tracks temp 0 (34.0% vs 32.2%).

Server-side only fixes clients that *omit* the field. One that sends it explicitly still wins.

---

## What I got wrong

The above is what survived. This is what didn't:

- **"The flat acceptance curve is caused by greedy drafting."** Wrong. I switched to probabilistic
  and the curve stayed flat on my production traffic. The mechanism was plausible and the
  prediction failed.
- **"Raising context grew the KV pool 18%."** Wrong, my arithmetic. I derived the smaller pool by
  assuming constant bytes/token, and bytes/token is not constant. The pool never grew.
- **"k=10 will give 6.745 tok/step."** Wrong by 19%. Model fit the data I had and was structurally
  wrong about the direction I hadn't measured.
- **"`reasoning_effort: max` is silently ignored."** Wrong. It needs `max-model-len >= 393216`. I
  measured it on a 262,144 server, below the threshold the feature requires, and reported the
  condition as the conclusion. It works: n=3, `high` mean 3,374 chars vs `max` 4,834, +43%.
- **Two configs copied from other operators** (`gmu 0.80`, `capture 36`) were both wrong for this
  hardware. Both were correct for theirs.

The pattern in every one: I measured under a condition that made the answer impossible, then
reported the condition as the finding.

---

## How this was measured

- **Metric:** `Δ vllm:generation_tokens_total / Δ vllm:request_decode_time_seconds_sum`, server-side,
  decode-only, excludes prefill. Not tokens-over-wall-time; that once made this server look like a
  40 tok/s machine.
- **Warmup:** 2 full-length 2048-token generations, then 6 more discarded. Under-warming read
  **111.86** where properly warmed read **123.13**, an 11 tok/s error, larger than most tuning
  changes on this page.
- **n=10** per point, with sd/min/max reported.
- **Contamination gate:** these are global counters. `accept.py` compares `request_success_total`
  against its own request count and exits 4 rather than report a polluted result. One early sweep
  was 95% someone else's traffic and came back non-monotonic.
- **One variable per boot.** Where I changed three at once, I said so and then isolated them.

## What this does not claim

- Code generation, single stream. My real agentic traffic, ~132K-token prompts, heavy tool use
  measures **~31% acceptance and ~3.2 tok/step** on this same config. Both numbers are true.
  Prompt shape drives acceptance more than any knob here.
- Not a 1M-context throughput claim. Nobody has benchmarked *at* a million-token prompt, me included.
- Single hardware, single model, single vLLM build. The formulas transfer; the values may not.

Reproduce anything here with the scripts in this repo. If your numbers disagree, I want to know.
