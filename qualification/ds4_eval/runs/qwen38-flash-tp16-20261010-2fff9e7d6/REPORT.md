# ds4-eval 92 — qwen3.8-flash

- Endpoint `http://127.0.0.1:8433`, temperature `0.0`, max_tokens `16000`, reasoning_effort `model default`
- Cases `qualification/ds4_eval/runs/kimi-k3-api-20260728/cases.json` sha256 `f0be6aec61f49987ffb19ccbc77f4e24f643b46e7a81652b7c1f63279ac416c9`

| Phase | Concurrency | Passed | Errors | Prompt tok | Cached tok | Completion tok | Wall s | Completion tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | 1 | 0/92 | 0 | 30437 | 0 | 92 | 101.5 | 0.91 |
| batch | 16 | 79/92 | 0 | 30437 | 27328 | 535232 | 2815.5 | 190.1 |

## batch by family

| Family | Passed | Completed |
|---|---:|---:|
| AIME2025 | 22 | 25 |
| COMPSEC | 16 | 17 |
| GPQA Diamond | 21 | 25 |
| SuperGPQA | 20 | 25 |
