# Embeddings from the engine on one GB10 (2026-11-22)

Qwen3-Embedding-8B served by SparkPipe's own engine: one rank on one GB10,
engine API and rank on the same machine. Nothing here stands for a fleet.

- **Build.** Node files from commit `0f0e678d8` of this change
  (`files.sha256`), CUDA 13.0 for sm_121a. The pack is cut from the
  publisher's checkpoint at revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` by
  `tools/qwen3_embedding_pack.py`.
- **Lane.** 16 sequences, 2,048 rows a wave, 2,048 positions a sequence, a KV
  pool of at most 512 pages (`lane.json`). Up for 21 minutes, 3,551 requests,
  no request failed.
- **Reference.** The publisher's model under torch 2.14.1 and transformers
  5.22.2 (`host.txt`), in two forms: a service that was resident on the same
  GPU the whole time (BF16, fused attention, padded batches filled by token
  count), and the same model loaded a second time by `emb_attrib.py` to run
  one text at a time in four variants.
- **Texts.** 256 texts, 60,954 tokens, 2 to 1,871 tokens a text, median 68
  (`corpus.json`): 64 sentences, 96 paragraphs and 48 runs of paragraphs from
  this repository's documents, 16 other texts (Korean, Chinese, Japanese,
  French, German, Spanish, Russian, Python, C, SQL, single words), and 32
  questions with the model's instruction prefix. The questions are the
  queries and the other 224 texts the passages.

Receipts are in [`qwen3-embedding-8b-one-gb10/`](qwen3-embedding-8b-one-gb10/).

## Agreement with the publisher's code

Our vector against the torch service's vector for the same text
(`agreement.jsonl`, step `agreement`).

| 256 texts | Cosine |
| --- | ---: |
| Lowest | 0.999728 |
| 1st percentile | 0.999789 |
| Median | 0.999887 |
| Highest | 0.999961 |

- The vectors are not bit-equal for any text. The largest difference in one
  of the 4,096 values is 0.0088; the mean is 0.00018.
- The cosine does not depend on the kind of text (mean 0.99986 to 0.99990 in
  every group) and hardly on its length (median distance 1.1e-4 for texts
  under 16 tokens, 1.2e-4 for texts over 1,024).
- Our token counts equal the service's (60,954), and a text sent as text and
  as the publisher tokenizer's ids returns the same bits for all 256 texts.

## Where the difference comes from

The same 256 texts through the publisher's own code in four variants, one
text at a time (`attribution.jsonl`). Distance is 1 minus cosine.

| A | B | Median distance | Largest | Rows bit-equal |
| --- | --- | ---: | ---: | ---: |
| ours | torch service (BF16, padded batches) | 1.13e-4 | 2.72e-4 | 0 |
| ours | torch BF16, one text at a time | 1.22e-4 | 4.30e-4 | 0 |
| ours | torch BF16, eager attention | 2.00e-4 | 5.51e-4 | 0 |
| ours | torch FP32 | 0.98e-4 | 2.48e-4 | 0 |
| torch service | torch BF16, one text at a time | 1.19e-4 | 3.27e-4 | 0 |
| torch service | torch FP32 | 0.97e-4 | 2.30e-4 | 0 |
| torch BF16 | torch FP32 | 1.05e-4 | 2.54e-4 | 0 |
| torch BF16 | torch BF16, eager attention | 1.98e-4 | 4.16e-4 | 0 |
| torch BF16, one text at a time | the same run again | 0 | 0 | 256 |
| torch BF16 | torch BF16 with our rope table | 0 | 0 | 256 |

- **Our distance to the torch service is the distance between two runs of
  the publisher's own code.** The service and the same code run one text at
  a time are 1.19e-4 apart at the median; we are 1.13e-4 from the service.
- **The difference is BF16 rounding.** Every BF16 run sits about 1e-4 from
  the FP32 model and from every other BF16 run. Ours is 0.98e-4 from FP32,
  torch's BF16 run 1.05e-4.
- **What we compute differently.** Attention scores and softmax weights are
  FP32 in our kernels and only the attention output is rounded to BF16;
  projections sum in the engine's own order; the unit-length division is
  FP64 rounded to FP32. The residual sums, norms, rotary terms and the SiLU
  round as the publisher's bfloat16 modules do.
- **The rope table does nothing here.** Our table differs from torch's in
  one of 64 entries by one FP32 step (ours is the correctly rounded value of
  the publisher's formula). torch with our table returns the same bits as
  torch with its own for all 256 texts.

## Ranking

32 queries over 224 passages, scored by dot product
(`agreement.jsonl`, step `ranking`, and `attribution.jsonl`).

| A | B | First passage the same | Top-10 overlap | Adjacent pairs swapped in the top 10 | Largest score difference | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ours | torch service | 32 of 32 | 98.1% | 17 of 288 | 0.0055 | 0.0010 |
| ours | torch FP32 | 32 of 32 | 98.4% | 13 of 288 | 0.0057 | 0.0011 |
| torch service | torch BF16, one text at a time | 32 of 32 | 97.8% | 24 of 288 | 0.0058 | 0.0012 |
| torch service | torch FP32 | 32 of 32 | 98.1% | 17 of 288 | 0.0048 | 0.0010 |
| torch BF16 | torch FP32 | 32 of 32 | 96.9% | 18 of 288 | 0.0064 | 0.0013 |
| torch BF16 | torch BF16, eager attention | 32 of 32 | 97.8% | 25 of 288 | 0.0071 | 0.0014 |

Vectors from both engines in one index, against the service alone:

| Queries | Passages | First passage the same | Top-10 overlap | Largest score difference |
| --- | --- | ---: | ---: | ---: |
| ours | torch service | 32 of 32 | 99.4% | 0.0040 |
| torch service | ours | 32 of 32 | 99.1% | 0.0033 |

**What a ranking needs.** Two passages keep their order when the gap between
their scores is larger than the two score errors together. Between any two of
these runs a score moves by 0.001 on average and by 0.006 at most. In the
service's own ranking the first passage leads the second by 0.032 at the
median (0.0022 at the least), and neighbours inside the top ten are 0.0057
apart at the median. So the first passage is the same in every comparison,
and inside the top ten about one adjacent pair in sixteen changes places:
17 of 288 for ours against the service, 17 to 25 of 288 between two runs of
the publisher's code. A ranking that tolerates the publisher's own BF16 runs
tolerates ours. A decision that hangs on a score difference under 0.006 is
not stable under either.

## The same text, the same bits

Eight texts of 17 to 1,386 tokens (`agreement.jsonl`, step `invariance`).

| | Rows bit-equal |
| --- | ---: |
| ours: alone, and alone again (the second resumes from the cache) | 8 of 8 |
| ours: alone, and in a request of 8 | 8 of 8 |
| ours: alone, and among 32 texts | 8 of 8 |
| ours: a text never seen before, alone and in a request of 8 | 8 of 8 |
| torch service: alone, and among 32 texts | 0 of 8 (cosine 0.999755 at the least) |

All 256 texts sent a second time (as token ids, after the cache held them)
also return the bits of the first time.

## Speed

Medians of three runs, with the range of the three (`agreement.jsonl` step
`speed`, and `speed.jsonl`). Every text carries a prefix that is new in each
run, so nothing resumes from the cache. The torch service was resident on
the same GPU and answered another program's short query every few seconds to
minutes. The mixed texts are the 256 above, 258 tokens a text.

| Request | Ours, texts/s | Ours, tokens/s | torch service, texts/s | torch service, tokens/s |
| --- | ---: | ---: | ---: | ---: |
| one text a request, 48 texts of 37 tokens | 13.1 (13.0-13.1) | 482 | 12.7 (12.5-12.7) | 468 |
| 16 texts a request, 256 mixed texts | 13.2 (12.8-13.5) | 3,390 | 8.1 (8.1-8.3) | 2,095 |
| 64 texts a request, 256 mixed texts | 13.8 (13.7-13.9) | 3,567 | 9.8 (9.7-9.9) | 2,524 |
| 256 texts in one request | 14.1 (13.5-14.3) | 3,648 | 10.9 (10.8-11.1) | 2,823 |
| 64 short texts in one request, 36 tokens each | 96 (74-97) | 3,432 | 61 (50-61) | 2,157 |

One text a request is one after another, so it is also the latency: 76 ms a
text of 37 tokens.

## Memory

Per process from `nvidia-smi` (`gpu.txt`) and the rank's own report
(`engine.log`, `weightd.log`). A GB10's device memory is the machine's
memory.

| | At start | With the KV pool full |
| --- | ---: | ---: |
| Rank daemon: weights, model state, wave buffers | 14,985 MiB | 15,002 MiB |
| weightd: the KV pool | 612 MiB | 4,788 MiB |
| The lane | 15,597 MiB (16.4 GB) | 19,790 MiB (20.8 GB) |

- Weights 15.40 GB, model state 0.39 GB for 16 sequences and 2,048 rows,
  wave buffers 0.02 GB.
- The KV pool starts at 0.46 GB and grows to 4.84 GB: 512 pages of 9.4 MB,
  which is 16 sequences of 2,048 positions at 147 KB a position.
- The machine's available memory fell from 98 GiB to 77 GiB. The torch
  service beside the lane held 14,873 MiB.
- On disk the lane wrote 9.9 GB of KV snapshot files (1,122 pages) for the
  texts that were embedded a second time. They were deleted at the end.

## The chat layer and the memory host

`chat_live.py` runs the chat layer's embeddings call and the memory
connector's host listener from `tools/serving/` against the live engine
(`chat_layer.jsonl`).

- `POST /v1/embeddings` on the chat layer, 256 texts: every row equals the
  engine's own answer bit for bit, and the usage is 60,954 tokens.
- The connector's `hello` lists `embed` among its host methods.
- `embed` with 64 texts answers the model's name, 4,096 dimensions, 1,265
  tokens, and vectors that equal the engine's own bit for bit.
- `embed` with `dimensions` 1024 answers unit-length vectors of 1,024
  values, the first 1,024 of the full vector divided by their own length.
- A text of 3,002 tokens is refused with `context_length_exceeded`.

## Checked on the CPU before the run

- **Pack** (`pack_verify.py`): all 290 tensors of the pack, 15,135,207,424
  bytes, equal the checkpoint's bytes read with safetensors, and every
  tensor of the checkpoint is in the pack.
- **Tokenizer** (`tok_check.py`): the engine's tokenizer returns the
  publisher tokenizer's ids for all 256 texts.
- **Rope table:** 63 of 64 frequencies equal torch's bit for bit. Entry 37
  is `0x1.6453f4p-12` in ours and `0x1.6453f6p-12` in torch's table. The
  power under it is `0x1.6fd745070aa97p+11` in FP64; libm, numpy and torch's
  scalar path round it to `0x1.6fd746p+11`, and torch's vector power on
  aarch64 returns `0x1.6fd744p+11`.

## What this run does not show

- One GB10, one lane geometry, one run of every step but speed.
- No text is longer than 1,871 tokens, and most texts are English prose
  about one system.
- The KV pool filled during the run (512 of 512 pages). `engine.log` shows
  the engine's error-site lines for that: no free page before an eviction
  (8 lines), and a request that asked to resume from a prefix the rank no
  longer held (6 lines; the API counted 5 prefixes computed again). All
  3,551 requests completed.
- `engine.log` and `weightd.log` have the account's home path replaced by
  `$HOME`, and `gpu.txt` is cut to the process list and the memory line.

## Files

| File | What it holds |
| --- | --- |
| `agreement.jsonl` | first light, agreement, ranking, invariance and the first speed run, one line a step |
| `speed.jsonl` | the second and third speed runs |
| `attribution.jsonl` | the four torch variants and every pair of vector sets |
| `chat_layer.jsonl` | the chat layer and memory host check |
| `corpus.json` | the 256 texts and their groups |
| `engine.log`, `weightd.log` | the rank daemon's and weightd's logs for the whole run |
| `gpu.txt`, `host.txt`, `lane.json`, `files.sha256` | GPU memory by process, kernel, driver, compiler and library versions, the lane's geometry, the node files and the pack |
| `emb_check.py`, `emb_attrib.py`, `chat_live.py`, `tok_check.py`, `pack_verify.py` | the scripts that produced the lines above |
