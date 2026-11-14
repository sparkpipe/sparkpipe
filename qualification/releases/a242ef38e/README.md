# Release a242ef38e

Release a242ef38e (PRs #1934 and #1936) went onto the fleet on 2026-11-14
between 02:46 and 02:52 UTC. It replaced weightd e457ff8d65ed70b7 with
ee01d0d5665e4464 (IPC ABI 16, four mesh lanes) and moved 17 models to
renumbered lane roots. The kit's `qualify` step then qualified 20 of the 21
served models (`qualify.log`). `<model>.old.json` is the same morning's
capture of release 75a78547b.

glm-5.3 does not activate yet: its warm command in
`deployment/hub/model_swap.json` is written by hand and still names its old
lane root, whose `weightd_warm` speaks ABI 15. PR #1937 moves it; glm-5.3 is
qualified after that PR's `records` step.

## Outputs

On the 19 models with an old capture, every prompt's tokens and top-4
logprobs, one step at a time, equal 75a78547b's (`EXACT 7/7`), and the chain
and speculating tokens equal the one-step tokens. The logprob goldens of
Kimi K3, DeepSeek V4.1 Flash and the four Qwen models are identical.
gemma-4-26b-a4b did not activate on 75a78547b, so its capture here is its
first.

Greedy tokens of the prompts sent all at once differ from their one-step
tokens on some prompts, on both releases, and the prompts move from run to
run:

| Model | Prompt | 75a78547b | a242ef38e |
| --- | --- | --- | --- |
| kimi-k3 | long_2k | from token 25 | from token 25 |
| kimi-k3 | long_8k | equal | from token 32 |
| ling-3.0-flash-fin | story | from token 19 | equal |
| qwen3.8-27b-fp8 | arith | equal | from token 10 |
| qwen3.8-max | story | from token 42 | equal; from token 42 again after the API restart |
| qwen3.8-max | capital | equal | from token 5 after the API restart |

Seeded sampling inside the batch likewise differs from the same seed alone
on kimi-k3, kimi-k3.1-pro, deepseek-v4.1-flash, glm-5.3-flash,
ling-3.0-flash and qwen3.8-flash, on both releases and at different prompts.
A request's tokens depend on which requests share its batch, which changes
with their timing. This release did not bring that in. It is also why the
API restart check fails for deepseek-v4.1-flash and qwen3.8-max: the capture
after the restart drew different batches.

## Per model

Decode is one request, 256 tokens (`o256`); prefill is a 7K-token prompt
(`ttft6000`). Release a85cf215a's figures of 2026-11-04 are in brackets
(`qualification/releases/a85cf215a/`).

| Model | Single requests vs 75a78547b | API restart | UI smoke | COMPSEC-17 | Decode, tok/s | Prefill, tok/s |
| --- | --- | --- | --- | --- | ---: | ---: |
| qwen3.8-max | identical, 7 of 7 prompts; goldens identical | batch differs | 6/6 | 10/17 | 12.2 (11.1) | 853 (40) |
| glm-5.3-flash | identical, 7 of 7 prompts | equal | 6/6 | 13/17 | 49.8 (47.4) | 2,095 (1,701) |
| kimi-k3 | identical, 7 of 7 prompts; goldens identical | equal | 6/6 | 17/17 | 18.8 (20.7) | 748 (726) |
| kimi-k3.1-pro | identical, 7 of 7 prompts | equal | not run¹ | not run¹ | 21.9 | 510 |
| deepseek-v4.1-flash | identical, 7 of 7 prompts; goldens identical | sampled batch differs | 6/6 | 16/17 | 53.7 (51.6) | 2,303 (1,763)⁴ |
| qwen3.8-27b | identical, 7 of 7 prompts; goldens identical | equal | 6/6 | 17/17 | 44.9 (43.8) | 3,225 (2,224) |
| qwen3.8-27b-fp8 | identical, 7 of 7 prompts; goldens identical | equal | not run¹ | not run¹ | 64.4 | 3,120 |
| qwen3.8-flash | identical, 7 of 7 prompts; goldens identical | equal | 6/6 | 16/17 | 45.4 (45.8) | 2,441 (1,296)⁴ |
| gemma-4-31b | identical, 7 of 7 prompts | equal | 6/6 | 15/17 | 48.8 (45.5) | 1,930 (871) |
| gemma-4-26b-a4b | no old capture | equal | 6/6 | 16/17 | 128.0 (116.5) | 5,893 (2,320) |
| ling-3.0-flash | identical, 7 of 7 prompts | equal | 6/6 | 14/17 | 77.2 (82.6) | 5,735 (2,245) |
| ling-3.0-flash-fin | identical, 7 of 7 prompts | equal | 6/6 | 12/17 | 79.8 (81.4) | 5,793 (2,286) |
| hy4-preview | identical, 7 of 7 prompts | equal | 6/6 | 16/17 | 21.7 (14.6) | 372 (123) |
| muse-glimmer-30b | identical, 7 of 7 prompts | equal | 6/6 | 17/17 | 42.3 (40.4) | 3,197 (1,376) |
| laguna-s-2.1 | identical, 7 of 7 prompts | equal | 5/6² | 15/17 | 52.4 (53.7) | 2,780 (1,190) |
| mimo-v2.6-flash | identical, 7 of 7 prompts | equal | 6/6 | 17/17 | 57.1 (50.8) | 2,314 (198) |
| mimo-v2.6-pro | identical, 7 of 7 prompts | equal | 6/6 | 16/17 | 26.0 (23.3) | 912 (72) |
| k-exaone-2.0 | identical, 7 of 7 prompts | equal | 5/6³ | 14/17 | 28.4 (26.8) | 945 (759) |
| a.x-k2 | identical, 7 of 7 prompts | equal | 6/6 | 15/17 | 28.9 (26.7) | 1,747 (1,394) |
| solar-open-2 | identical, 7 of 7 prompts | equal | 6/6 | 12/17 | 63.5 (59.1) | 2,593 (2,139) |

COMPSEC-17 runs at temperature 0, one request at a time, at `minimal`
effort with 2,048 tokens where the model can turn reasoning off and at `low`
with 4,096 where it cannot. Against a85cf215a, which ran it the same way, no
model lost a task: the 14 models scored both times pass 211 tasks against
195. The scores of 9b253f5b4 allowed 512 tokens and do not compare.

1. The hub's chat layer dates from 2026-11-04 and does not list
   kimi-k3.1-pro or qwen3.8-27b-fp8 (main's does). The UI smoke sent its
   first requests before it read the listing, and LiteLLM's `*` route sends
   any name it does not know to glm-5.3: for kimi-k3.1-pro that swapped the
   fleet to glm-5.3, whose activation fails as above, so the perf and long
   fill that followed found no API. Both smokes waited out their timeout.
   COMPSEC reads each model's efforts from the chat layer and was skipped.
   The UI smoke now reads the listing before its first request.
2. laguna-s-2.1 returns no `reasoning_content` with reasoning on, as on
   a85cf215a and 9b253f5b4.
3. k-exaone-2.0 with reasoning off answers 401 to 17 × 23, as on a85cf215a.
4. The first token of the `ttft6000` prompt is a stop for these two models,
   so the perf tool timed no token; the rerun below times the stop.

## Long fill

One cold fill of 33,000 random tokens. Six models serve 32,768 positions
and answered HTTP 400 to 33,000 tokens; the rerun filled them with 32,767.

| Model | Tokens | Seconds | tok/s |
| --- | ---: | ---: | ---: |
| qwen3.8-max | 32,767 | 45.2 | 725 (rerun) |
| glm-5.3-flash | 33,000 | 16.9 | 1,949 |
| kimi-k3 | 33,000 | 44.1 | 748 |
| kimi-k3.1-pro | 33,000 | 65.2 | 506 (rerun) |
| deepseek-v4.1-flash | 33,000 | 12.4 | 2,672 |
| qwen3.8-27b | 33,000 | 10.1 | 3,268 |
| qwen3.8-27b-fp8 | 33,000 | 10.4 | 3,174 |
| qwen3.8-flash | 33,000 | 12.9 | 2,567 |
| gemma-4-31b | 32,767 | 17.9 | 1,832 (rerun) |
| gemma-4-26b-a4b | 33,000 | 6.1 | 5,420 |
| ling-3.0-flash | 33,000 | 6.7 | 4,954 |
| ling-3.0-flash-fin | 33,000 | 6.7 | 4,926 |
| hy4-preview | 32,767 | 89.7 | 365 (rerun) |
| muse-glimmer-30b | 33,000 | 9.7 | 3,418 |
| laguna-s-2.1 | 32,767 | 12.5 | 2,617 (rerun) |
| mimo-v2.6-flash | 32,767 | 14.8 | 2,217 (rerun) |
| mimo-v2.6-pro | 32,767 | 37.1 | 884 (rerun) |
| k-exaone-2.0 | 33,000 | 35.6 | 928 |
| a.x-k2 | 33,000 | 22.8 | 1,446 |
| solar-open-2 | 33,000 | 12.4 | 2,671 |

## Reruns

`rerun.log`: after the qualification, with the long fill and perf tools
fixed the same day, each model activated through the swap controller
again: the six long fills at 32,767 tokens, kimi-k3.1-pro's perf and long
fill, and the `ttft6000` case of deepseek-v4.1-flash and qwen3.8-flash
(`<model>.perf-rerun.jsonl`).

## Rank logs

`rank log error lines` counts lines matching error, fail, fault or abort in
each rank's latest residentd log. Every model's 16 is one line a node that
says "by default". kimi-k3.1-pro's 48 add two shutdown summaries a node,
every failure counter 0, written when the swap to glm-5.3 stopped it.

## Files

- `<model>.old.json`, `<model>.new.json`: `tools/release_capture.py`
  captures of 75a78547b and a242ef38e; `<model>.restart.json` is the capture
  after the API restart.
- `<model>.golden-compare-old-new.json`: the logprob golden comparison.
- `<model>.smoke.log`, `<model>.compsec.log`: the UI smoke and COMPSEC-17.
- `<model>.perf.jsonl`, `<model>.perf-rerun.jsonl`: `tools/api_serving_perf.py`.
- `qualify.log`, `rerun.log`: the kit's output and the reruns.
