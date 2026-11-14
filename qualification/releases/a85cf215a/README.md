# Release a85cf215a

The receipts of release a85cf215a's qualification on 2026-11-04, copied on
2026-11-14 from the hub's `release-a85cf215a/` work directory, where the
kit had left them. The Fleet List's 2026-11-04 speeds (commit ea4fcef9f)
come from this run. The kit's printed output was not kept.

Its one-step tokens differ from the previous release's
(`<model>.old.json`) on one to six of the six prompts of 16 models, and no
model's logprobs are identical: the release changed numerics.

Decode is one request, 256 tokens (`o256`); prefill is a 7K-token prompt
(`ttft6000`).

| Model | UI smoke | COMPSEC-17 | Decode, tok/s | Prefill, tok/s | Note |
| --- | --- | --- | ---: | ---: | --- |
| qwen3.8-max | 5/6 | 5/17 | 11.1 | 40 |  |
| glm-5.3 | 5/6 | – | 21.2 | 1,109 | COMPSEC stopped: HTTP 400 on the effort it chose |
| glm-5.3-flash | 5/6 | – | 47.4 | 1,701 | COMPSEC stopped: HTTP 400 on the effort it chose |
| kimi-k3 | 6/6 | 17/17 | 20.7 | 726 |  |
| deepseek-v4.1-flash | 6/6 | 16/17 | 51.6 | 1,763 |  |
| qwen3.8-27b | 6/6 | 13/17 | 43.8 | 2,224 |  |
| qwen3.8-flash | 6/6 | 15/17 | 45.8 | 1,296 |  |
| gemma-4-31b | 6/6 | – | 45.5 | 871 | COMPSEC not run |
| gemma-4-26b-a4b | 6/6 | 15/17 | 116.5 | 2,320 |  |
| ling-3.0-flash | 6/6 | 14/17 | 82.6 | 2,245 |  |
| ling-3.0-flash-fin | 6/6 | 11/17 | 81.4 | 2,286 |  |
| hy4-preview | 6/6 | 16/17 | 14.6 | 123 |  |
| muse-glimmer-30b | 5/6 | – | 40.4 | 1,376 | COMPSEC not run |
| laguna-s-2.1 | 5/6 | 15/17 | 53.7 | 1,190 |  |
| mimo-v2.6-flash | 6/6 | 17/17 | 50.8 | 198 |  |
| mimo-v2.6-pro | 6/6 | 15/17 | 23.3 | 72 |  |
| k-exaone-2.0 | 5/6 | 11/17 | 26.8 | 759 |  |
| a.x-k2 | 6/6 | 15/17 | 26.7 | 1,394 |  |
| solar-open-2 | 6/6 | – | 59.1 | 2,139 | COMPSEC not run |

UI smoke failures: glm-5.3, glm-5.3-flash, muse-glimmer-30b and
qwen3.8-max refused the effort the smoke asked for with reasoning off
(HTTP 400; the last three pass on a242ef38e, where glm-5.3 waits to be
qualified); k-exaone-2.0 answered 401 to
17 × 23 with reasoning off; laguna-s-2.1 returned no `reasoning_content`
with reasoning on.

## Files

- `<model>.old.json`, `<model>.new.json`: `tools/release_capture.py`
  captures of the previous release and of a85cf215a.
- `<model>.golden-compare-old-new.json`: the logprob golden comparison.
- `<model>.smoke.log`, `<model>.compsec.log`: the UI smoke and COMPSEC-17.
- `<model>.perf.jsonl`: `tools/api_serving_perf.py`.
