# Qwen 3.8 27B FP8 variant: measured on 2026-11-12

The variant `qwen3.8-27b-fp8` (the publisher's FP8 release,
`Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`) against
the full-precision lane `qwen3.8-27b` (the publisher's BF16 weights,
`Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`), by
`docs/VARIANT_GATE.md`.

The variant repeats itself bit for bit (1,024 of 1,024 positions on a lane
set up afresh), so it is intact. What it gives and what it costs, both lanes
on the same sixteen Sparks from one firmware build:

| | BF16 | FP8 | |
| --- | ---: | ---: | --- |
| Decode, one stream | 48.7 tok/s | 60.1 tok/s | 23% faster |
| Decode, eight streams (the suite run) | 223.8 tok/s | 270.8 tok/s | 21% faster |
| Prefill, 32,768 tokens | 3,026 tok/s | 2,612 tok/s | 14% slower |
| Memory a node at start | 5.8 GiB | 4.3 GiB | 1.5 GiB less |
| Next token right, 12,800 short contexts | 68.562% | 68.383% | 0.18 points lower: 64 positions only BF16 predicts, 41 only FP8 (p 0.031) |
| Next token right, 2,928 positions deep in the window | 61.407% | 60.690% | 0.72 points lower: 26 only BF16, 5 only FP8 (p 0.0002) |
| NLL of the corpus token, short contexts | 0.932 nats | +0.283% [+0.133, +0.449] | |
| NLL of the corpus token, deep in the window | | -0.291% [-1.082, +0.464] | not resolved: the interval spans zero |
| Same best token as BF16 | | 98.12% short, 94.50% deep | |
| ds4-eval 92 | 79 | 74 | 8 cases only BF16 answers, 3 only FP8 (p 0.23) |
| of which GPQA Diamond, 25 cases | 23 | 18 | 5 only BF16, none only FP8 (p 0.06) |
| of which the other three families, 67 cases | 56 | 56 | 3 only BF16, 3 only FP8 |

`verdict.json` is the tool's output. Intervals are one-sided 95% bootstrap
bounds, 10,000 replicates, seed 20260929; p is the exact McNemar test on the
positions or cases only one side gets, and says how surely a difference is
there, not whether it matters to a given user.

## What ran

- Two test lanes on mesh lane 5 of the sixteen Sparks, one at a time, never
  the production roots: `STAGE_MODEL=qwen38_27b` and
  `STAGE_MODEL=qwen38_27b_fp8`, 8 sequences, 1,024 rows, a 262,144-token
  window, 32,768 logical KV pages and at most 8,192 on the device. The
  production lane (Qwen 3.8 Max) was loaded and idle beside them.
- Firmware built on the build host from `9c7e9bd1e`:
  `sparkpipe_model_residentd` `2f1bcec58481…`, the BF16 adapter
  `a7ce4b7dabd8…` and the FP8 adapter `f349c189488b…`. The two adapters
  carry the same GPU code (`.nv_fatbin` `ca3bdaedfd60…`), and the BF16
  adapter's `.text`, `.nv_fatbin` and `.rodata` are those of the build gated
  for the wait prefetch. The hub API is the x86 build of `3a1cff2da`.
- Corpus: CT-short (`qualification/ab/corpora/CT-short.index.json`) in the
  Qwen 3.8 tokenizer, 200 documents of 1,269 to 1,280 tokens
  (`probe.index.json`, tokens SHA-256
  `fb1248163729b3c58fdc192f9c52eefb53d05b24948e9c62313790e0dbc5d825`); the
  depth document is those documents back to back, 255,989 tokens
  (`joined.index.json`).
- Each precision: a fresh lane, the corpus run (64 positions a document,
  12,800 in all), then ds4-eval 92 (8 at a time, 16,000 tokens an answer,
  temperature 0, straight to the engine's chat endpoint), the reference
  prompts and the speed requests; a fresh lane again for the depth run (48
  positions from every 4,096th token to 250,608, 2,928 in all); and for FP8
  a third fresh lane that scored the first 16 documents again.
- Every scoring request is one greedy token with the top 20 log-probs, no
  speculation, one request at a time.

## Corpus, 12,800 positions

| | All | Code | Prose, 2001 | Prose, 2026-07 |
| --- | ---: | ---: | ---: | ---: |
| Documents | 200 | 40 | 40 | 120 |
| Same best token | 98.12% [97.90, 98.32] | 98.98% | 98.24% | 97.79% |
| KL lower bound | 0.0027 [0.0021, 0.0038] | 0.0013 | 0.0018 | 0.0034 |
| NLL, FP8 relative to BF16 | +0.283% [+0.133, +0.449] | +0.330% [-0.051, +0.720] | +0.164% [-0.012, +0.335] | +0.319% [+0.108, +0.553] |
| Next token right, BF16 | 68.562% | 85.391% | 63.477% | 64.648% |
| Next token right, FP8 | 68.383% | 85.352% | 63.516% | 64.349% |
| Only BF16 right / only FP8 right | 64 / 41 | 8 / 7 | 8 / 9 | 48 / 25 |

- The corpus token is among both precisions' 20 listed tokens at 12,006
  positions (93.8%); 21 more list it only at BF16 and 21 only at FP8. NLL is
  over the 12,006. BF16's NLL there is 0.932 nats.
- No position has the same log-probs at both precisions; 7 are near ties.
- With the document as the unit, FP8's next-token accuracy is 0.180 points
  lower [-0.305, -0.055]; 48 documents lose, 29 gain, 123 are unchanged.

## Depth, 2,928 positions to 250,608 tokens

| From token | Positions | Same best token | KL lower bound | NLL, FP8 relative to BF16 | Next token right, BF16 / FP8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| all | 2,928 | 94.50% [93.34, 95.56] | 0.0345 [0.0222, 0.0529] | -0.291% [-1.082, +0.464] | 61.41% / 60.69% |
| 0 | 144 | 95.83% | 0.0618 | +2.282% [-1.194, +3.786] | 51.39% / 51.39% |
| 16,384 | 576 | 93.06% | 0.0357 | -0.153% [-1.879, +1.099] | 53.12% / 52.26% |
| 65,536 | 768 | 91.02% | 0.0571 | -1.355% [-3.556, -0.015] | 49.61% / 48.70% |
| 131,072 | 1,440 | 96.81% | 0.0204 | -0.018% [-1.117, +1.349] | 72.01% / 71.39% |

- The unit is the run of 48 positions, 61 runs.
- At 130 positions one precision or both predict the end of the text, where
  the API lists no log-probs: 86 both, 24 only BF16, 20 only FP8. They count
  for the best-token row and are left out of KL and NLL.
- The depth document is unrelated documents back to back. It tests the
  cache and the recurrent state at depth, not how a model reads one long
  text.

## Suite and reference

ds4-eval 92 in its listed order, 8 at a time, temperature 0, 16,000 tokens
an answer:

| Family | Cases | BF16 | FP8 | Both | Only BF16 | Only FP8 | Neither | Change in cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| All | 92 | 79 | 74 | 71 | 8 | 3 | 10 | -5.0 [-11.0, +0.0], p 0.2266 |
| GPQA Diamond | 25 | 23 | 18 | 18 | 5 | 0 | 2 | -5.0 [-10.0, -2.0], p 0.0625 |
| SuperGPQA | 25 | 22 | 22 | 22 | 0 | 0 | 3 | 0 |
| AIME 2025 | 25 | 18 | 17 | 15 | 3 | 2 | 5 | -1.0 [-5.0, +3.0], p 1 |
| COMPSEC | 17 | 16 | 17 | 16 | 0 | 1 | 0 | +1.0 [+0.0, +4.0], p 1 |

- The whole of the difference is in one family: on GPQA Diamond FP8 answers
  five cases fewer and none more; on the other 67 cases the two score the
  same, three cases each way.
- No request failed; one FP8 answer ran to the token limit.
  `bf16.ds4/summary.json`, `fp8.ds4/summary.json`.
- The score does not move from run to run. Each precision ran the suite a
  second time in the same order, on a lane set up afresh that had served
  nothing else, and returned the same 92 answers token for token: the same
  extracted answer and the same number of generated tokens in every case,
  292,523 tokens at BF16 and 315,157 at FP8 both times
  (`bf16.ds4/summary.2.json`, `fp8.ds4/summary.2.json`). So 79 against 74
  is what these two precisions answer in this order, not a draw.
- What can move a score is which answers generate beside each other: eight
  are generated at a time and a row's log-probs depend slightly on the rows
  beside it (below). This receipt holds the listed order and its repeat;
  the reversed and shuffled orders of `docs/VARIANT_GATE.md` are not in it
  yet. The Fleet List carries an earlier run of the same suite at these two
  precisions, on an earlier build and 16 at a time: BF16 78, FP8 76, 7
  cases only BF16 and 5 only FP8.
- Both lanes return the 71 greedy tokens of the numpy FP8 reference on its
  three chat prompts (`qualification/qwen38_27b/reference/reference3_fp8_numpy.json`).

## Speed, same lanes

| | BF16 | FP8 |
| --- | ---: | ---: |
| Decode, one stream | 20.52 ms a token (48.7 tok/s) | 16.64 ms a token (60.1 tok/s) |
| Prefill, 8,192 tokens | 3,176 tok/s | 2,580 tok/s |
| Prefill, 32,768 tokens | 3,026 tok/s | 2,612 tok/s |
| ds4-eval 92, 8 at a time | 1,307 s, 223.8 tok/s | 1,164 s, 270.8 tok/s |
| Corpus run, 12,800 requests | 1,588 s | 1,483 s |
| Memory a node at start | 5.8 GiB | 4.3 GiB |

Decode is from the API's token timestamps over four 256-token requests after
the corpus and suite runs; prefill is one uncached request with one output
token (BF16 on a fresh lane). FP8 decodes 23% faster and prefills 14 to 19%
slower: prefill is bound by compute and rounds, and the FP8 kernels compute
more per weight.

## Packs

| Rank | Node | BF16 pack (`qwen3.8-27b`) | FP8 pack (`qwen3.8-27b-fp8`) |
| ---: | --- | --- | --- |
| 0 | spark0 | `b8fe6c3f876fbb41dff6501e3301f5b1bcbbb8ad52e979d18757f383615dc329` | `0341a769f92c0a91e5b40c418ab796ef15e1410f373d77ab5c2e62f284cb1578` |
| 1 | spark1 | `7e3f70e9c2063ba4f7ccaf99059519813268ff0ed0029cd994e74f0ed9b7cb16` | `33d5b9613caa605598a017dceae572eef74635b7e9faceb0d4725a9316a63b6e` |
| 2 | spark2 | `9b1f42a105e1a83a02e3264368acd96da0e75529ed0259be8854c148064728bf` | `8aff903bd50b7cfd1305b380fbc162afce22979584cc0cd04d4e34df4fdff3fc` |
| 3 | spark3 | `f179be74f5109520dda5d2ff98ce5354e8dd290df16c2688e8ad42e52ec92a5c` | `ba591702eeb190c88f3c8361078daec7e8fb06e22b38f500aaae073f976236ea` |
| 4 | spark4 | `7dd57e18dc1f5d7211e2f32998cee2bd6090519bdc7a18774d0d65de062ed4ff` | `f62efea881d9ae4ba688f01ed2def5f73063594e0ed9cbba0ca0c96c22d5aadb` |
| 5 | spark5 | `b79e15e4b4ca017d3fab5dfe26f797b6c14f1e8c2750a455f280016415bf349c` | `e97f85c173c8ee288a1cd7f9fbbdd0cc78f1956c7d2d7826a3e624abc16a9bd0` |
| 6 | spark6 | `d1169cf11a978c71a89ffcfcf6a296c10902019192f50b0483b6ff6a79edf6cb` | `62868eb4c4e845d129a8ef56b2b6caae2d4a40dabe26af0d2204aabb4e2a94bc` |
| 7 | spark7 | `864faed920f06a237e0b092f778762c100e8b79b60e341e5d07330ba44dd6b76` | `db8be56135af9d0a4a4a6fe5c5fcfdd07791332ec1e3b7f709ef4bca42a58823` |
| 8 | spark8 | `4c77d2bf8282e8f2e25836bd6fce624ab3c309f2a3be52eb48bd5a8a97fc29fb` | `c083f36e461dd6046010933ddf0ffb48defd98e5e3cb8c75dbf02bcc5202acc6` |
| 9 | spark9 | `3e69f2086952f80a4cf4437cd7f8c286a7ce5148fdf84fb38baa27b6781bf014` | `aabb545779ab9788f9b022b3acf922b3adf6b55329fab12c542447b738b1f011` |
| 10 | sparka | `ffd0354d2f7d07642755d631f614b43983d7171fee328b39d090e906dbb3a6ad` | `8a56b1d336da655d27b7cd9ce7ec31b1b5a881c39b2e8a3702fd6e4b5b757262` |
| 11 | sparkb | `a88a76a24925e5f5f54aec2e7f6950173cafa7a686260eb7b549790e18296d67` | `c5bd2998b2976d6fad811522c650750d2adcec2957a841f15a03390fc44d3c94` |
| 12 | sparkc | `4a1d60bea6d67f3f743b49a0bf46691a6372ea97a8187e8f0e87630bd125e35c` | `f91b8d4b360688cb2383668e98cc873b4337f7b48e1322e9b0717e12c0447498` |
| 13 | sparkd | `6f7cd4d7b5ea518f043f65c7598c9f99447398d0cee1c3e7ab02fd965cb1d601` | `a090ba2061abab068d074fe722e6ab622a0f1e7d1a7f3c680c15ff5b9bb85fcc` |
| 14 | sparke | `dcce3ad7e21c1f0209ab7ed3ad5dfce6d2c6399549dcb2d89f973282018ae464` | `b95ab9b64655b5d943fc654f8e4d9c38f8048a8b998549bbde5c54b09d536618` |
| 15 | sparkf | `7b2736dbfdc145f23f5fc8a379ab3e6a31535a4c86206bc8d234c1c34abc44f9` | `61ca755e957f568cffa616c10fb2a7175689ba0601bdc8243a8068520cb31889` |

## Score files

The score files stay on the hub under
`/srv/workspace/sparkpipe/variant-gate-q27/runs`; the comparisons name them
by SHA-256.

| File | SHA-256 |
| --- | --- |
| `bf16.corpus.jsonl` | `b6f05d35d8facb3e9640ab0f033c22a17927bf5a35297637fe8484cf1098ef4f` |
| `fp8.corpus.jsonl` | `2f25620682121c5b0070e32cd18c5d4f356c761bd28edb7142183780d86a681a` |
| `bf16.depth.jsonl` | `acf632753399e1c5541aad8bb6650ac27a8faa74e268e8e2b33ba582f4ee7f14` |
| `fp8.depth.jsonl` | `7d313847be808c5b3a75ee878a79c74b0a695198d5814ebbd77f7211b57334c5` |
| `fp8.repeat.jsonl` | `34b242f35e6d3028c366fc1237a320f26d5b0c6b35613faa16ea21d22db75a80` |

## What else the run found

- A row's log-probs depend slightly on the rows sharing its prefill wave: 96
  BF16 positions scored eight documents at a time differed from the same
  positions scored alone at 2 (a tail token by 0.13 nats, mean KL 2.6e-6).
  Scored alone on a fresh lane a run repeats bit for bit.
- BF16's first depth attempt stopped at position 8,576, where the model's
  next token ends the text and the API returns no token and no log-probs.
  The scorer now records such a position; BF16's depth run was made again on
  a fresh lane, as FP8's was.
- The run was made against the first version of `docs/VARIANT_GATE.md`,
  which refused a variant whose NLL bounds left ±0.5% or whose next-token
  accuracy or suite score was significantly lower. This variant failed three
  of those checks (next-token accuracy on the corpus and at depth, and the
  width of the depth NLL interval). The owner decided on 2026-11-12 that the
  figures are published for the user to choose by, and that only a variant
  that does not repeat itself is refused; the page and the tool now say so,
  and `verdict.json` is the tool's output on the same score files.
