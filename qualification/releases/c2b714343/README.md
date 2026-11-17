# Release c2b714343

Release c2b714343 went onto the fleet on 2026-11-20 between 00:42 and 00:45
UTC. It replaced weightd ee01d0d5665e4464 with 351c25b8d10b3326:
- IPC ABI 17, four mesh slots a rank (#1955).
- Page cache ABI 16: prefixes reach disk only after a second read (#1949).

It also carries:
- the capture fixes of #1945 and #1948;
- the release kit of #1937, including glm-5.3's warm path;
- the node baseline of #1957.

Every lane root and hub API directory was installed from main c2b714343.
weightd was ready on all sixteen nodes 12 seconds after it was published.
The kit's `qualify` step ran from 00:45 to 03:03 UTC (`qualify.log`).
`post_qualify.log` holds the checks run after it. `<model>.old.json` is the
same night's capture of release a242ef38e.

## Outputs

On the 20 models with an old capture, every prompt's tokens and top-4
logprobs, one step at a time, equal a242ef38e's (`EXACT 7/7`). The chain and
speculating tokens equal the one-step tokens. The logprob goldens of Kimi K3,
DeepSeek V4.1 Flash and the four Qwen models are identical.

glm-5.3 activates and serves for the first time since release 75a78547b.
Its warm command named a lane root of the old ABI until #1937. It has no old
capture, so `glm-5.3.new.json` is its reference from here on.

Greedy tokens of the prompts sent all at once differ from their one-step
tokens on some prompts, on both releases, and the prompts move from run to
run. qwen3.8-27b-fp8's arith differed from token 10 in the qualify capture,
as on a242ef38e. Three more captures of it after `qualify` were equal on
every path. Seeded sampling inside the batch likewise differs from the same
seed alone at a few prompts of qwen3.8-max, qwen3.8-flash, ling-3.0-flash,
gemma-4-26b-a4b and kimi-k3, in old and new captures alike. A request's
tokens depend on which requests share its batch, which changes with timing.
This is why the API restart check reports a difference for five models
(below), whose restarted APIs give `EXACT 2/2` single requests. This release
did not bring it in.

## Per model

Decode is one request of 256 tokens (`o256`). Prefill is a 7K-token prompt
(`ttft6000`). Release a242ef38e's figures are in brackets. COMPSEC-17 runs
only when a capture comparison fails, which the batch effect above triggers.

| Model | Single requests vs a242ef38e | API restart | UI smoke | COMPSEC-17 | Decode, tok/s | Prefill, tok/s |
| --- | --- | --- | --- | --- | ---: | ---: |
| qwen3.8-max | identical, 7 of 7; goldens identical | sampled batch differs | 6/6 | 10/17 (10/17) | 12.3 (12.2) | 858 (853) |
| glm-5.3 | first capture | equal | 6/6 | 15/17 | 23.0 | 1,260 |
| kimi-k3 | identical, 7 of 7; goldens identical | equal | 6/6 | 17/17 (17/17) | 20.1 (18.8) | 709 (748) |
| kimi-k3.1-pro | identical, 7 of 7 | equal | not listed¹ | not run | 22.5² (21.9) | 490² (510) |
| glm-5.3-flash | identical, 7 of 7 | sampled batch differs | 6/6 | not run | 49.6 (49.8) | 2,112 (2,095) |
| qwen3.8-27b | identical, 7 of 7; goldens identical | equal | 6/6 | not run | 45.5 (44.9) | 3,286 (3,225) |
| qwen3.8-27b-fp8 | identical, 7 of 7; goldens identical | equal | not listed¹ | not run | 63.2² (64.4) | 3,082² (3,120) |
| qwen3.8-flash | identical, 7 of 7; goldens identical | equal | 6/6 | 16/17 (16/17) | 45.0 (45.4) | — |
| deepseek-v4.1-flash | identical, 7 of 7; goldens identical | equal | 6/6 | 16/17 (16/17) | 58.1 (53.7) | — |
| gemma-4-31b | identical, 7 of 7 | equal | 6/6 | not run | 49.4 (48.8) | 1,913 (1,930) |
| gemma-4-26b-a4b | identical, 7 of 7 | sampled batch differs | 6/6 | 16/17 (16/17) | 95.8³ (128.0) | 5,589 (5,893) |
| ling-3.0-flash | identical, 7 of 7 | sampled batch differs | 6/6 | 14/17 (14/17) | 76.9 (77.2) | 5,640 (5,735) |
| ling-3.0-flash-fin | identical, 7 of 7 | equal | 6/6 | not run | 80.2 (79.8) | 5,772 (5,793) |
| hy4-preview | identical, 7 of 7 | equal | 6/6 | not run | 22.3 (21.7) | 371 (372) |
| muse-glimmer-30b | identical, 7 of 7 | equal | 6/6 | not run | 42.0 (42.3) | 3,128 (3,197) |
| laguna-s-2.1 | identical, 7 of 7 | equal | 5/6 (5/6) | not run | 53.1 (52.4) | 2,691 (2,780) |
| mimo-v2.6-flash | identical, 7 of 7 | equal | 6/6 | not run | 56.7 (57.1) | 2,298 (2,314) |
| mimo-v2.6-pro | identical, 7 of 7 | equal | 6/6 | not run | 26.1 (26.0) | 908 (912) |
| k-exaone-2.0 | identical, 7 of 7 | equal | 5/6 (5/6) | not run | 28.5 (28.4) | 930 (945) |
| a.x-k2 | identical, 7 of 7 | equal | 6/6 | not run | 29.0 (28.9) | 1,742 (1,747) |
| solar-open-2 | identical, 7 of 7 | sampled batch differs | 6/6 | not run | 64.2 (63.5) | 2,557 (2,593) |

1. The hub's chat layer predates both models, so `/v1/models` does not list
   them. LiteLLM's catch-all route sent their UI smoke to glm-5.3, which
   swapped the fleet before their perf and long fill ran.
2. Measured after `qualify` with each model activated through the
   controller (`post_qualify.log`).
3. Three repeats after `qualify` decoded 98.2, 115.7 and 123.2 tok/s and
   prefilled 5,681 to 5,783 tok/s (`gemma-4-26b-a4b.perf-repeat.jsonl`).
   This model's single-request decode moves with its speculation from run to
   run. In the API log, its one-row chain steps take a median 11.2 ms a token
   in the old release's capture and 11.1 ms in this release's.

## Long fills

The kit's cold fill sends 33,000 random tokens. It succeeded on 13 models
in `qualify`, and on kimi-k3.1-pro (493 tok/s, 97% of its best) and
qwen3.8-27b-fp8 (3,166) after it:
- kimi-k3 (777 tok/s), k-exaone-2.0 (936), ling-3.0-flash-fin (4,968),
  qwen3.8-27b (3,274) and qwen3.8-flash (2,572) ran at their fastest yet.
- glm-5.3 ran at 1,119 against 1,267, 88% of its best and above the 85%
  regression line.
- Every other model ran within 3% of its best.

qwen3.8-max, gemma-4-31b, hy4-preview, laguna-s-2.1, mimo-v2.6-flash and
mimo-v2.6-pro answer such a prompt with HTTP 400, because it is longer than
their context. The tool fix that sizes the fill to each model's context is
in #1938, which is still open.

The rank-log line count the kit prints (16 on every model) counts the
startup line that contains "default", one a rank.
