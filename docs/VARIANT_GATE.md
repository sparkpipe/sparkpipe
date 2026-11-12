# Variant accuracy gate

A precision variant (`qwen3.8-27b-fp8`, README "Packs and Models") is offered
only after it passes this gate against the full-precision lane of the same
model, both run from the same firmware build on the same fleet. The gate goes
through the serving API, so it needs no special build and holds for any model
and backend.

## What a variant is

- its own entry in `tools/stage_lane.py` MODELS: the variant's pack, adapter
  library and node target, a KV label of its own, its own API port;
- its own record in `deployment/lanes.json` on the model's mesh lane (the two
  never run at once), and with it a swap entry, a chat-layer entry, LiteLLM
  names and an Open WebUI entry;
- the model's contract, chat template, tokenizer and context window.

The adapter build carries the KV label (`module_tag`), so the swap controller
releases each precision's device KV pool under its own name, and the KV
identity digest already differs through the adapter id and the pack digest: a
variant never opens the other precision's KV pages or snapshots.

## Checks

`tools/ab_api_score.py` scores a frozen token corpus one position at a time:
the document's tokens before the position as `prompt_token_ids`, one greedy
token, the top 20 log-probs, no speculation. Both precisions score the same
positions, in the same order, one request at a time, each on a lane that has
served nothing since it was set up, and `compare` pairs them.

One request at a time is part of the method. A row's log-probs depend
slightly on the rows that share its prefill wave: on the BF16 lane, 96
positions scored eight documents at a time differed from the same positions
scored alone at 2 positions (the best token's log-prob by 0.003, a tail
token's by 0.13; mean KL 2.6e-6), and both of those ran a 17-row prefill
beside other requests' rows. Scored alone, in a fixed order on a fresh lane,
a run repeats bit for bit, which the repeat check holds the variant to.
`--concurrency` is for a quick look, not for a gate run.

| Check | Passes when |
| --- | --- |
| Repeat | the variant, set up afresh and asked its first documents again, returns the same tokens and log-probs at every position |
| Corpus NLL | both one-sided 95% bounds of the relative change in the corpus token's NLL lie inside ±0.5% |
| Corpus next-token accuracy | the variant does not lose significantly: exact McNemar p ≥ 0.05 on the positions only one of the two predicts, or the variant predicts more |
| Depth NLL and accuracy | the same two checks on positions deep in one long document, to the end of the window |
| Suite | on each task suite, run at both precisions with the same settings, the variant does not lose significantly (exact McNemar, α 0.05) |

The bounds are bootstrap bounds: 10,000 replicates, seed 20260929, the more
conservative of the percentile and BCa bound. The unit is the document, or
for the depth comparison the run of positions, since rows of one document are
not independent.

The numbers are not chosen per variant. The ±0.5% NLL margin, α 0.05, the
bootstrap and the 0.001-nat near-tie rule are the ones frozen for the
quantization A/B in `qualification/ab/flash-w1/PLAN.json`.

Top-1 agreement (positions whose two best reference tokens are within 0.001
nats left out) and a lower bound on KL(reference ‖ variant) are reported and
not gated. They say how often the two precisions answer differently, not
whether one is worse: the publisher's FP8 release of GLM 5.3 Flash measured
96.8% agreement and KL 0.0087 with its NLL unchanged
(`qualification/ab/flash-w1/REPORT.md`).

## What the method cannot see

- The API lists 20 tokens a position. NLL is compared where both precisions
  list the corpus token; `compare` reports how many positions that is and how
  many only one side lists.
- KL is taken over the tokens both lists share plus one bucket for the rest,
  which bounds the true KL from below.
- Where a model's next token ends the text, the API returns no token and no
  log-probs. Such a position counts for top-1 agreement (both precisions
  stop there, or they differ) and is left out of KL and NLL; `compare`
  counts them.
- Each request names the last 64-token boundary before its end as a cache
  point, so a document's next position resumes from the cache. Both
  precisions are scored the same way, and a resumed sequence must equal an
  uninterrupted one by the driver contract.

## Running it

The corpus is the frozen CT-short corpus (200 documents: recent and old arXiv
prose and C sources, `qualification/ab/corpora/CT-short.index.json`), put into
the model's tokenizer, and the same documents back to back as one long
document for the depth comparison:

```bash
python3 tools/ab_corpus_retokenize.py --index CT-short.index.json --tokens CT-short.tokens.u32 \
  --source-tokenizer GLM/tokenizer.json --target-tokenizer MODEL/tokenizer.json --target-name MODEL \
  --out-index C.index.json --out-tokens C.tokens.u32
python3 tools/ab_api_score.py join --index C.index.json --tokens C.tokens.u32 --out-index J.index.json --out-tokens J.tokens.u32
```

Then, on each precision's lane in turn; the corpus run and the depth run each
start on a freshly set up lane, and the suite follows the corpus run:

```bash
python3 tools/ab_api_score.py score --endpoint http://127.0.0.1:PORT --index C.index.json --tokens C.tokens.u32 --out ARM.corpus.jsonl
python3 tools/ds4_eval92.py --endpoint http://127.0.0.1:PORT --model MODEL --max-tokens 16000 --temperature 0 \
  --phases batch --concurrency 8 --out ARM.ds4
python3 tools/ab_api_score.py score --endpoint http://127.0.0.1:PORT --index J.index.json --tokens J.tokens.u32 \
  --starts 4096:249856:4096 --count 48 --out ARM.depth.jsonl
```

The suite goes to the engine's chat endpoint, which renders the model's own
template.

The corpus run scores 64 positions a document (256, 272, ... 1264); the depth
run scores 48 positions from every 4,096th token to the end of the window.
The variant's lane is then set up again and scores its first documents a
second time (`--documents 16`, another output file) for the repeat check.
The verdict:

```bash
python3 tools/ab_api_score.py compare BF16.corpus.jsonl VARIANT.corpus.jsonl --out corpus.json
python3 tools/ab_api_score.py compare VARIANT.corpus.jsonl VARIANT.repeat.jsonl --out repeat.json
python3 tools/ab_api_score.py compare BF16.depth.jsonl VARIANT.depth.jsonl --unit run --out depth.json
python3 tools/ab_api_score.py gate --corpus corpus.json --repeat repeat.json --depth depth.json \
  --suite ds4-eval-92=BF16.ds4/batch/summary.json:VARIANT.ds4/batch/summary.json --out verdict.json
```

`gate` exits 0 only when every check passes. A suite run with a failed
request decides nothing and is refused. The receipt of a variant holds the
comparisons, the suite summaries, the verdict and the score files' digests
(`qualification/qwen38_27b/fp8-variant-20261112/`).

## Open: what a variant may lose

The first run, Qwen 3.8 27B FP8, fails the gate as written above: its NLL
and its suite are inside, and its next-token accuracy is 0.18 points lower
on the corpus and 0.72 at depth. The two accuracy checks ask whether a loss
is detectable, and on thousands of positions a small one is; the depth NLL
check asks for both bounds inside ±0.5% and its 61 runs leave the interval
1.5 points wide. The checks were written before the run and are not changed
by it. A quantized variant is expected to lose something; how much it may
lose and still be offered is a product decision, and the checks follow it.
