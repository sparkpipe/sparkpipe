# DeepSeek V4 Flash API — `ds4_eval` 92-case run

Run completed on 2026-08-08 against all 92 cases in the prepared GA suite.

## Result

| Family | Correct | Total | Accuracy |
|---|---:|---:|---:|
| GPQA Diamond (including modified case) | 17 | 25 | 68.0% |
| SuperGPQA | 21 | 25 | 84.0% |
| AIME2025 | 15 | 25 | 60.0% |
| COMPSEC | 17 | 17 | 100.0% |
| **Overall** | **70** | **92** | **76.1%** |

## API configuration

- Model: `deepseek-v4-flash`
- Thinking: disabled (`{"type":"disabled"}`)
- Maximum completion: `4096` tokens
- Temperature: `0`
- `reasoning_effort`: omitted
- Streaming: disabled
- Requests: 92 submitted concurrently
- Retries: 3 after the initial attempt, exponential 1-second backoff
- Per-request timeout: 600 seconds
- Response model: `deepseek-v4-flash` for all 92 responses
- System fingerprint: `fp_a18b46594c_prod0820_fp8_kvcache_20260402`

The prompts are the exact prepared `standard4096-suite` messages. The source
fixture SHA-256 is
`138ae8bcb92503a4ff8bd8dd5607fb0b58af118cb34f5ab3e9d89c92e615c22a` and the
prepared-suite manifest SHA-256 is
`f4be71b690044f30f78d08fbc13571b41fb241494c9f7e1128e0ea0ea8ec9918`.

All 92 requests completed without transport or API errors. The run used
58,964 completion tokens and 23,885 prompt tokens over 31.128 seconds. The
measured API request rate was 2.956 requests/second; this is an API-wave timing
receipt, not a SparkPipe inference-throughput claim.

## Receipt

The redacted per-case response receipt remains outside the repository at:

`/private/tmp/ds4_api_92_no_thinking_4096_20260808_r2/results.jsonl`

Its SHA-256 is
`51132398130046fc61678c1e15a52f1fd664c62d512dc6a20db4eb94a443ede7`.
The repository records the score and provenance only; the response text is not
committed because it is a large run artifact.

This is a locally audited DeepSeek API result, not the repository's highest
retained score. The current highest retained complete 92-case archive remains
the Kimi K3 result at 81/92.
