# ds4-eval retained results

This directory is SparkPipe's canonical record for complete `ds4-eval` runs.
The embedded 92-question set is a capability regression suite, not an official
GPQA, SuperGPQA, AIME, or security leaderboard. The upstream description and
protocol are pinned at
[`antirez/ds4@54b36ed`](https://github.com/antirez/ds4/blob/54b36ed9ba42da31b24f2d1a5feb075c2475dbb1/README.md#capability-evaluation).

## Highest retained result

| Model | Execution | GPQA | SuperGPQA | AIME2025 | COMPSEC | Overall | Exact archive |
|---|---|---:|---:|---:|---:|---:|---|
| Kimi K3 | Kimi coding API, temperature 1.0, 16,000-token limit | 23/25 | 21/25 | 21/25 | 16/17 | **81/92** | [`runs/kimi-k3-api-20260728`](runs/kimi-k3-api-20260728/) |

This is the highest result for which this repository retains and independently
validates the complete run. It is not claimed as an official or directly
comparable all-time record because the recovered project runs below used
different fixture revisions, inference protocols, quantizations, and execution
topologies.

## Locally audited DeepSeek API run

| Model | Execution | GPQA | SuperGPQA | AIME2025 | COMPSEC | Overall | Receipt |
|---|---|---:|---:|---:|---:|---:|---|
| `deepseek-v4-flash` | DeepSeek API, thinking disabled, temperature 0, 4,096-token completion cap, 92-way concurrency | 17/25 | 21/25 | 15/25 | 17/17 | **70/92** | [`REPORT.md`](runs/deepseek-v4-flash-api-20260808/REPORT.md) |

This run completed all 92 requests without API errors and used the exact
prepared `standard4096-suite` prompts. It is a locally audited API comparison,
not the highest retained score; the complete receipt and parameters are in the
[run report](runs/deepseek-v4-flash-api-20260808/REPORT.md).

For context, a separately published DeepSeek V4 Flash 4Expert Q4_K run reports
80/92 (AIME 20/25, GPQA 22/25, SuperGPQA 22/25, COMPSEC 16/17). SparkPipe does
not retain that run's responses, and it is not treated as a locally audited
entry. See the
[pinned model-card report](https://huggingface.co/cloudyu/DeepSeek-V4-Flash-4Expert-GGUF/blob/39a195539a9ad27a28dc70b86f32a1d8651af929/README.md#gguf-evaluation-report--4expert-q4_k-gguf-by-ds4-eval).

## Recovered project results

These records were recovered from fixed commits in
[`experiencenow-ai/ds4_on_spark`](https://github.com/experiencenow-ai/ds4_on_spark).
They preserve project history, but they are not a single apples-to-apples
ranking.

| Model | Execution | GPQA | SuperGPQA | AIME2025 | COMPSEC | Overall | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| `deepseek-v4-flash` served alias | vLLM MXFP4 TP=2, temperature 0, top-p 1, seed 1, concurrency 8, 16,000-token limit | 21/25 | 20/25 | 21/25 | 16/17 | **78/92** | External exact [`JSONL`](https://github.com/experiencenow-ai/ds4_on_spark/blob/0dd28c579ac45ee16b5b3bd24a5f7aebbe87ad38/fixtures/pipeline_quality/vllm-mxfp4-tp2-ds4-eval-20260523T0315Z.jsonl) and [`trace`](https://github.com/experiencenow-ai/ds4_on_spark/blob/0dd28c579ac45ee16b5b3bd24a5f7aebbe87ad38/fixtures/pipeline_quality/vllm-mxfp4-tp2-ds4-eval-20260523T0315Z.trace.txt) |
| `Qwen/Qwen3.6-27B-FP8` | vLLM; run settings not retained | — | — | — | — | **76/92** | [`Profile assertion`](https://github.com/experiencenow-ai/ds4_on_spark/blob/56d95807e16196cf6f385db0b4de6a17bdb4c77d/v2/profiles/models/qwen3_6_27b_fp8_efficient.json) only |
| DeepSeek V4 Flash, checkpoint unbound | antirez IQ2XXS PP=1 CUDA, seed 1, 16,000-token limit | 20/25 | 22/25 | 15/25 | 16/17 | **73/92** | External exact [`JSONL`](https://github.com/experiencenow-ai/ds4_on_spark/blob/0dd28c579ac45ee16b5b3bd24a5f7aebbe87ad38/fixtures/pipeline_quality/lane-d-pp1-redo-20260521T0412Z.jsonl) and [`trace`](https://github.com/experiencenow-ai/ds4_on_spark/blob/0dd28c579ac45ee16b5b3bd24a5f7aebbe87ad38/fixtures/pipeline_quality/lane-d-pp1-redo-20260521T0412Z.trace.txt.gz) |
| `google/gemma-4-31B-it` | DS4 API PP8, temperature 0, thinking off, 4,096-token limit | 18/25 | 20/25 | 18/25 | 16/17 | **72/92** | [`Summary snapshot`](https://github.com/experiencenow-ai/ds4_on_spark/blob/4ee9061bc59f9f913eef5f5e9277df7a330c09fa/v2/profiles/validation/gemma4_ds4_eval_20260606.json) only |
| `google/gemma-4-26B-A4B-it` | DS4 API PP8, temperature 0, thinking off, 4,096-token limit | 17/25 | 21/25 | 16/25 | 15/17 | **69/92** | [`Summary snapshot`](https://github.com/experiencenow-ai/ds4_on_spark/blob/4ee9061bc59f9f913eef5f5e9277df7a330c09fa/v2/profiles/validation/gemma4_ds4_eval_20260606.json) only |
| `google/gemma-4-12B-it` | DS4 API PP8, temperature 0, thinking off, 4,096-token limit | 17/25 | 18/25 | 11/25 | 17/17 | **63/92** | [`Summary snapshot`](https://github.com/experiencenow-ai/ds4_on_spark/blob/4ee9061bc59f9f913eef5f5e9277df7a330c09fa/v2/profiles/validation/gemma4_ds4_eval_20260606.json) only |

The DeepSeek rows retain all 92 generated texts and response traces outside this
repository, but SparkPipe has not imported or independently regraded them.
Their records do not bind the exact model checkpoint revision. The Qwen record
retains no family breakdown, run command, fixture revision, or raw responses.
The Gemma snapshot retains settings and aggregate and family scores, but its raw
`/private/tmp/ds4_bench` response directories were not committed. Summary and
profile assertions cannot be independently regraded.

The current Kimi archive pins `ds4_eval.c` SHA-256
`19545bf6c0a55cb91b7e3120344ec69ad4cfb5c87cf91e82ec4191a590013f23`.
The fixture present at the Gemma snapshot commit hashes to
`138ae8bcb92503a4ff8bd8dd5607fb0b58af118cb34f5ab3e9d89c92e615c22a`,
but all 92 cases match the Kimi fixture on source, ID, domain, title, question,
choices, and answer. The Qwen assertion does not identify its fixture. Its
76/92 score selected the repository's default efficient local mechanical-work
profile, but it is not evidence of an overall prior best: the differently
configured DeepSeek 78/92 run predates and outscores it.

## Kimi K3 archive

[`runs/kimi-k3-api-20260728`](runs/kimi-k3-api-20260728/) contains:

- `REPORT.md`: human-readable result, protocol, and provenance.
- `INTEGRITY.json`: hashes for the report, manifests, and complete response
  checksum stream.
- `summary.json`: all 92 grades and aggregate usage.
- `cases.json`: exact ordered cases, rendered prompts, source commit, and
  `ds4_eval.c` hash.
- `responses/*.json`: all 92 assembled API responses. Each file preserves the
  content, reasoning, response ID/model, finish reason, token usage, exact
  prompts and prompt hash, request settings, grade, source pin, and timings.

The archive preserves exact assembled response text. It does not preserve raw
SSE event bytes, chunk boundaries, or HTTP headers. The response-file checksum
stream, using paths relative to the run directory, is:

```text
f074d5003dcc3b83f1554e28210461b9ea69c4189c3ad6038d3bb0dc2d36c7e1
```

The stream is SHA-256 over 92 filename-sorted lines of:

```text
<file SHA-256><two spaces>responses/<filename><LF>
```

Validate the retained run:

```sh
python3 qualification/ds4_eval/compare_runs.py \
  qualification/ds4_eval/runs/kimi-k3-api-20260728
```

Compare a future normalized run:

```sh
python3 qualification/ds4_eval/compare_runs.py \
  qualification/ds4_eval/runs/kimi-k3-api-20260728 \
  /path/to/candidate-run
```

Validation independently re-extracts and regrades every retained output and
fails closed on checksum drift, missing cases, changed prompts, source drift,
malformed output, or mismatched case identity. Comparison reports score and
family deltas, pass/fail transitions, extracted-answer changes, and exact
content/reasoning match counts with hashes for changed cases.

## Four-bit quality protocol

The Kimi run is a sampled capability reference. It used temperature 1.0 with no
seed, so exact textual equality against it is not a valid quantization gate.

For the locally hosted four-bit model:

1. Run the same pinned 92 cases and archive every exact response.
2. Compare overall and per-family scores against this retained 81/92 reference.
3. Compare local full precision against local four-bit with identical settings,
   `--temp 0 --seed 1`, and native byte-counted `--trace` files. Any source,
   prompt, token-limit, seed, or sampling mismatch is a configuration error.
4. Use upstream's official-continuation target-token NLL test as the direct
   quantization-quality gate. A sampled answer score alone cannot establish
   that four-bit weights preserved the full model distribution.
