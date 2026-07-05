# GLM52 API Stress Test

`tools/sparkpipe_api_stress.py` is the standard dependency-free client for
public API, queue, and batching stress runs. It sends OpenAI-compatible
`/v1/chat/completions` requests, records one JSONL row as each request finishes,
and can sample gateway `/health` in parallel when that endpoint is reachable.

Public website path from the Mac:

```sh
python3 tools/sparkpipe_api_stress.py \
  --url https://sparkpipe.ai/v1/chat/completions \
  --requests 128 \
  --concurrency 32 \
  --max-completion-tokens 256 \
  --stream
```

Run from the website host to also sample the private tunnel health:

```sh
python3 tools/sparkpipe_api_stress.py \
  --url https://sparkpipe.ai/v1/chat/completions \
  --health-url http://127.0.0.1:18080/health \
  --requests 256 \
  --concurrency 64 \
  --max-completion-tokens 256 \
  --stream
```

The progress JSONL defaults to `/private/tmp/sparkpipe_api_stress_*.jsonl`.
Each request result includes HTTP status, elapsed time, first-byte latency,
streaming TTFT, token event count, response bytes, and captured output. The
summary JSON records status counts, request rate, token-event rate, latency
percentiles, and TTFT percentiles.

Suggested first live ladder after `pp13_ready=1`:

```text
requests concurrency max_completion_tokens
32       1           128
64       4           128
128      16          256
256      64          256
512      128         512
```

Stop increasing concurrency once queue depth grows without token-event
throughput improving, TTFT becomes unacceptable, or errors appear.
