# Variant measurement

A precision variant (`qwen3.8-27b-fp8`, README "Packs and Models") is offered
with its measured figures against the full-precision lane of the same model,
both run from the same firmware build on the same fleet. Nothing is refused
for losing accuracy: a quantized variant is expected to lose a little, and
the figures are what the user chooses by. One thing is refused, a variant
that does not repeat itself. The measurement goes through the serving API,
so it needs no special build and holds for any model and backend.

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

## What is measured

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
a run repeats bit for bit. `--concurrency` is for a quick look.

| | |
| --- | --- |
| **Refused** | |
| Repeat | the variant, set up afresh and asked its first documents again, does not return the same tokens and log-probs at every position |
| **Published** | |
| NLL | the relative change in the corpus token's NLL, with one-sided 95% bounds |
| Next token | how often each precision's best token is the corpus token, and the positions only one of the two predicts |
| Agreement | how often the two pick the same best token, and a lower bound on KL(reference ‖ variant) |
| Depth | the same three on positions deep in one long document, to the end of the window |
| Suites | each task suite's score at both precisions in several orders of its cases, the change in cases with its bounds, the cases one precision always answers and the other never, and the cases that change with the order; the same for each family of cases the suite names |
| Speed | decode, prefill and memory of both lanes |

The bounds are bootstrap bounds: 10,000 replicates, seed 20260929, the more
conservative of the percentile and BCa bound. The unit is the document, for
the depth comparison the run of positions, since rows of one document are
not independent, and for a suite the case. The 0.001-nat near-tie rule and the bootstrap come from the
quantization A/B plan (`qualification/ab/flash-w1/PLAN.json`). An exact
McNemar p is printed beside every count of positions or cases only one side
gets; it says how surely a difference is there, not whether it matters to a
given user.

## What the method cannot see

- The API lists 20 tokens a position. NLL is compared where both precisions
  list the corpus token; `compare` reports how many positions that is and how
  many only one side lists.
- KL is taken over the tokens both lists share plus one bucket for the rest,
  which bounds the true KL from below.
- Each request names the last 64-token boundary before its end as a cache
  point, so a document's next position resumes from the cache. Both
  precisions are scored the same way, and a resumed sequence must equal an
  uninterrupted one by the driver contract.
- Where a model's next token ends the text, the API returns no token and no
  log-probs. Such a position counts for the best-token figures (both
  precisions stop there, or they differ) and is left out of KL and NLL;
  `compare` counts them.

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
The figures:

```bash
python3 tools/ab_api_score.py compare BF16.corpus.jsonl VARIANT.corpus.jsonl --out corpus.json
python3 tools/ab_api_score.py compare VARIANT.corpus.jsonl VARIANT.repeat.jsonl --out repeat.json
python3 tools/ab_api_score.py compare BF16.depth.jsonl VARIANT.depth.jsonl --unit run --out depth.json
python3 tools/ab_api_score.py gate --corpus corpus.json --repeat repeat.json --depth depth.json \
  --suite ds4-eval-92=BF16.ds4/batch/summary.json,BF16.reversed.ds4/batch/summary.json,BF16.shuffled.ds4/batch/summary.json:VARIANT.ds4/batch/summary.json,VARIANT.reversed.ds4/batch/summary.json,VARIANT.shuffled.ds4/batch/summary.json \
  --out verdict.json
```

`gate` exits 0 when the repeat check holds and prints every figure as
measured. A suite run with a failed request measures nothing and is refused.

A suite's answers are generated several at a time, and a prefill row's
log-probs depend slightly on the rows beside it, so an answer could depend
on which answers generate beside it. Each precision therefore runs the
suite in three orders, each on a lane set up afresh: as listed,
`--order reversed` and `--order shuffled --order-seed 20261112`. `--suite`
takes the runs of a side separated by commas and lists every run's score,
the cases one side always answers and the other never, and the cases that
change between runs of one side. For Qwen 3.8 27B no case changes: both
precisions return the same 92 answers token for token in all three orders
and in a repeat of the first, so there a score belongs to the weights and
not to the run.
The receipt of a variant holds the comparisons, the suite summaries, the
verdict and the score files' digests
(`qualification/qwen38_27b/fp8-variant-20261112/`).

## How this came to be

The first version of this page, written before the first run, also refused a
variant whose NLL bounds left ±0.5% or whose next-token accuracy or suite
score was significantly lower. Qwen 3.8 27B FP8 failed three of those checks:
its next-token accuracy is 0.18 points lower on the corpus and 0.72 at
depth, and the depth NLL interval was wider than the margin. The owner
decided on 2026-11-12 that these are figures to publish, not grounds to
refuse: quantify everything and let the user choose.
