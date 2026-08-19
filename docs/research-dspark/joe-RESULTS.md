# DeepSeek-V4-Flash-0731 on 4× DGX Spark, 123.13 tok/s decode, TP=4

**123.13 tok/s** mean single-stream decode, against **104.17** published by techmd on the same
TP=4 topology. Same prompt, same protocol, same metric. Our slowest run beat his fastest.

```
                    ours          techmd TP=4
mean              123.13               104.17
median            123.14               104.31
sd                  1.94                 3.51
min               120.73                98.84
max               127.61               110.23
n                     10                   10
```

Distributions do not overlap: **our min 120.73 > his max 110.23.**

## Where it comes from, both factors, not one

`decode rate = tok/step ÷ step time`, attacked independently:

```
ms/step      48.83   vs   55.46      12% faster steps
tok/step      6.019  vs    5.777     more tokens per step, k=7 vs his k=6
acceptance   71.7%   vs   79.6%      LOWER aggregate, see below
```

The aggregate acceptance number is misleading, and the per-position curve is the real story:

```
          ours    techmd
pos 0    95.0%     96.2%     his draft head is marginally better here
pos 1    92.2%     90.0%
pos 2    91.2%     83.5%
pos 3    91.0%     76.3%
pos 4    90.3%     69.5%
pos 5    86.0%     62.3%
pos 6    79.4%      ,       a 7th position he never attempts
```

His curve collapses with depth; ours holds. By position 5 he is at 62.3% and we are at 86.0%.
His 79.6% aggregate beats our 71.7% only because he averages over 6 positions and we average
over 7, including one that still lands 79.4% of the time. **Deeper drafting is only worth it
if the curve holds, and ours does.**

The single change responsible: **`draft_sample_method: probabilistic`** instead of `greedy`.
With greedy drafting the draft is a point mass, so acceptance collapses to `p_target(argmax)`
and is roughly flat across positions. Probabilistic matches the target's distribution and the
curve decays gracefully, which is what makes k=7 pay.

## The config

```
image        ghcr.io/anemll/dspark-vllm-gx10:0.1.1        (vLLM 0.25.2)
topology     TP=4 across 4x DGX Spark GB10, 200G RoCE
--max-model-len              327680
--max-num-seqs               12
--max-num-batched-tokens     8264          # = 8192 + (k-1)*seqs, see note
--gpu-memory-utilization     0.85
--max-cudagraph-capture-size 36
--kv-cache-dtype             nvfp4_ds_mla
--block-size                 256
--compilation-config         {"cudagraph_mode":"FULL_DECODE_ONLY"}
--override-generation-config {"temperature":0.0}
--speculative-config {"method":"dspark","num_speculative_tokens":7,
                      "draft_sample_method":"probabilistic",
                      "num_speculative_tokens_per_batch_size":[[1,4,7],[5,8,5],[9,12,3]]}
--enable-prefix-caching --async-scheduling --enable-chunked-prefill
--moe-backend flashinfer_b12x --enable-flashinfer-autotune

KV pool      54.26 GiB / 4,135,605 tokens
concurrency  12.62x at full 327,680 context
```

**Three things in there that are not obvious:**

1. **`max-num-batched-tokens 8264`, not 8192.** With spec decode on, vLLM subtracts draft slots
   from the prefill budget: `max_num_scheduled_tokens = max_num_batched_tokens − (k−1) × max_num_seqs`.
   vLLM warns when the result falls below 8192. At 8192 raw with k=7/seqs=12 you land at 8120
   under the threshold. `8264 = 8192 + (7−1)×12` puts the *effective* budget exactly on 8192.
   Every published config I checked sets 8192 raw and silently lands under it.

2. **`draft_sample_method: probabilistic`**, the whole per-position story above.

3. **`FULL_DECODE_ONLY` + capture 36 are free, `gpu_memory_utilization` is not.** Measured:

```
gmu 0.85 · capture 96 · FULL_AND_PIECEWISE   54.23 GiB   12.63x
gmu 0.80 · capture 36 · FULL_DECODE_ONLY     47.34 GiB   11.03x    <- regression
gmu 0.85 · capture 36 · FULL_DECODE_ONLY     54.26 GiB   12.62x    <- kept
```
   Only gmu moves the KV pool. Dropping it to 0.80 cost 6.9 GiB for nothing. At 0.85 these
   boxes sit at 98% memory (~2 GiB available), so 0.90 will not boot.

## The protocol, matched to techmd's exactly

```
prompt       "Write a complete, idiomatic Python implementation of a binary search tree with
              insert, delete, search, traversal, height, docstrings, and tests. Code only."
             (verbatim, changing the prompt changes acceptance and voids comparability)
max_tokens   2048        temperature 0        concurrency 1
warmup       2 full-length 2048-token generations, then 6 further discarded
measured     10 repeats, report mean/median/sd/min/max
metric       delta vllm:generation_tokens_total / delta vllm:request_decode_time_seconds_sum
             server-side, decode-only, EXCLUDES prefill
gate         abort if any foreign request lands in the window (global counters)
```

**Warmup depth is not optional.** techmd measured it swinging results by 2.7 tok/s. Our own
under-warmed run (1 warmup instead of 8) read **111.86** against **123.13** properly warmed
an 11 tok/s error, larger than most tuning changes. If a published number does not state its
warmup, it is not comparable.

Reproduce: `client/bst_parity.py` in the bundle. Raw output, full container argv, and boot-log
fingerprint archived alongside it.

## What this does NOT claim

- **This is a code-generation number.** On our production traffic, 132K-token agentic
  tool-calling, the same server measures ~31% acceptance and ~3.2 tok/step. Prompt shape drives
  draft acceptance more than any config knob. jvr0x put it well: *"The figure is workload, not
  hardware."* His own box collapses from 95.5 to ~47 tok/s on a summarizer.
- **Single stream, concurrency 1.** Aggregate throughput across concurrent sessions is a
  different measurement and this says nothing about it.
- **Not a 1M-context claim.** We run a 327,680 ceiling because our p99 real context is 243.6K
  and our largest observed is 327.4K. At 1M this pool would give 3.35 concurrent requests
  more than the 1.94x others report, but nobody, us included, has benchmarked *at* a
  million-token prompt. Advertised context and measured throughput are different numbers.
```
measured 2026-08-05 · vLLM 0.25.2.dev0+g752a3a504 · 4x DGX Spark GB10 sm_121a · aarch64 CUDA 13
```


---

## Update: the depth sweep, the ladder, and measuring at your real traffic shape

Everything above was measured on a fixed short-prompt protocol, the right ruler for the
head-to-head it was built for, and the wrong ruler for tuning a cluster that serves 150K-token
agentic prompts. I re-ran the whole speculative-decoding question at ~105K context, one
variable per boot, three probes per config (deep single-stream, a C=12 small-request storm,
and a cold deep prefill fired INTO a C=11 storm, the nastiest traffic shape I know how to
generate). A config that errored under any probe was disqualified regardless of speed. None did.

### Depth, closed

```
k     deep tok/s   ms/step   accept    storm agg
3     40.6         69.9      61.2%      91.8
5     49.3         70.3      51.9%     142.0
7     68.6         54.7      39.4%     126.6      <- winner, kept
8     55.2         59.7      28.9%     133.3
```

k=7 wins by 39% over the community-consensus k=5 and 69% over k=3, on THIS image. Shallower
k also steps SLOWER here, which killed my own verify-cost theory: the image's kernels are
dspark7-specialized. Depth is an engine-build property. Measure yours; don't inherit mine.

### The ladder, retired

The batch-aware draft ladder I previously shipped costs ~7% single-stream against static k=7
and only pays at sustained C=12, which my traffic visits rarely. Removed. Details in RECIPE.md.

### Per-position acceptance is the drafter's EKG

`vllm:spec_decode_num_accepted_tokens_per_pos_total` divided position-by-position gives a
conditional acceptance curve. Healthy here: 0.91 / 0.87 / 0.85 / 0.82 / 0.80 / 0.74 / 0.70
smooth decay, with a visible slope break right where the checkpoint declares
`dspark_block_size: 5`. Positions past the declared block still carried 14% of accepted
tokens, so I kept k=7 anyway, but a BROKEN drafter announces itself on this curve as a
front-loaded collapse (0.63 / 0.28 / 0.18 ...) long before you'd diagnose it from tok/s.
Two community deployments lost days to exactly that signature. Watch the curve, not the mean.

### Two numbers, always

Short-protocol numbers sell; production numbers serve. This cluster is 123.13 tok/s on the
former and 57.9 on the latter (84.7 ms/step at ~150K mean prompts, 96–98% prefix-cache hit,
zero preemptions across days of agentic traffic). Publishing one without the other is how
this community keeps confusing itself. The prompt length is part of the number.
