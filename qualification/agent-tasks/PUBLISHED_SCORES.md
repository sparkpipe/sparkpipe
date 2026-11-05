# Published agent-task scores, checked 2026-11-05

These are the scores the model providers publish, to sit beside SparkPipe's
own runs. Each one names its page. Each is the provider's own claim with the
provider's own harness, not a SparkPipe measurement. Pages were read on
2026-11-05.

## What the providers publish now

The three open models' providers publish Terminal-Bench 2.1 and DeepSWE
v1.1, and two of them publish Terminal-Bench 3.0 or 4.0. None of them
publishes SWE-bench Verified for these models. Anthropic's newest page
publishes Terminal-Bench 4.0. A SparkPipe SWE-bench Verified score therefore
has no provider figure to compare with for Kimi K3, GLM 5.3 or DeepSeek V4.1
Flash. Terminal-Bench 2.1 does have one: every provider below reports it.

## Terminal-Bench 2.1

| Model | Score | Harness | Source |
| --- | ---: | --- | --- |
| Kimi K3 (max) | 88.3 | Kimi Code | [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3) |
| GLM 5.3 | 88.2 | Claude Code 2.1.207, temperature 1.0, 6 h timeout | [GLM-5.3 model card](https://huggingface.co/zai-org/GLM-5.3) |
| DeepSeek V4.1 Flash (max) | 90.6 | DeepSeek Harness, minimal mode, 1M context, pass@1 over 3 | [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| DeepSeek V4.1 Flash (max) | 88.0 | Claude Code, same settings | same card, scaffold table |
| DeepSeek V4.1 Flash (max) | 84.1 | Codex, same settings | same card, scaffold table |
| DeepSeek V4.1 Flash (max) | 85.0 | OpenCode, same settings | same card, scaffold table |
| Claude Opus 5 | 89.1 | as DeepSeek reports it | DeepSeek-V4.1-Flash model card |
| GPT-5.6 Sol (max) | 88.8 | Codex, as Moonshot reports it | Kimi K3 model card |
| Claude Fable 5 (max, with fallback) | 88.0 | Terminus 2, as Moonshot reports it | Kimi K3 model card |
| Claude Opus 4.8 (max) | 84.6 | Terminus 2, as Moonshot reports it | Kimi K3 model card |

DeepSeek's Terminal-Bench 2.1 runs had no network access, a 1M-token
context, at most 500 steps, temperature 1.0 and top_p 0.95. GLM 5.3's card
gives its Claude Code version. That makes GLM 5.3 with Claude Code (88.2), and
DeepSeek V4.1 Flash with Claude Code (88.0) and with Codex (84.1), the
closest like-for-like checks of SparkPipe serving: same harness, same
benchmark, same model.

OpenAI's own GPT-5.6 page (https://openai.com/index/gpt-5-6/) refused the
fetch (HTTP 403), so the GPT-5.6 Sol figure above is the one Moonshot's card
quotes.

## Terminal-Bench 4.0

| Model | Score | Source |
| --- | ---: | --- |
| Claude Opus 5.5 | 66.4 | [Anthropic, Introducing Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5) (2026-09-22) |
| GPT-6 Astra | 57.9 | same page, "as reported by OpenAI" |
| Claude Fable 5.1 | 55.8 | same page |
| Claude Opus 5 | 52.3 (leaderboard: 51.8, 5 trials, Claude Code) | same page |
| GPT-5.6 Sol | 37.3 | same page, "as reported by OpenAI" |
| GLM 5.3 (max) | 41.9 (mini-swe-agent, pass@1 over 3) | GLM-5.3 model card |
| DeepSeek V4.1 Flash (max) | 31.2 | DeepSeek-V4.1-Flash model card |
| GLM 5.3 | 37.9, as DeepSeek reports it | DeepSeek-V4.1-Flash model card |
| Kimi K3 | 12.6, as DeepSeek reports it | DeepSeek-V4.1-Flash model card |

The Terminal-Bench 4.0 page says the benchmark contains GPU tasks and needs
a GPU-capable sandbox ([run instructions](https://www.tbench.ai/docs/run-terminal-bench-2-1)).

## DeepSWE v1.1

| Model | Score | Harness | Source |
| --- | ---: | --- | --- |
| DeepSeek V4.1 Flash (max) | 74.2 | mini-SWE, N=8 | DeepSeek-V4.1-Flash model card |
| DeepSeek V4.1 Flash (max) | 69.8 / 65.6 | Claude Code / Codex | same card, scaffold table |
| Kimi K3 (max) | 67.5 (67.3 with mini-SWE-agent) | Kimi Code | Kimi K3 model card |
| GLM 5.3 | 66.9 | mini-swe-agent, temperature 0.95, 400K context | GLM-5.3 model card |
| Claude Opus 5 | 74.0, as DeepSeek reports it | | DeepSeek-V4.1-Flash model card |
| GPT-5.6 Sol | 73.0, as DeepSeek and Moonshot report it (72.7 on GLM's card) | | both cards |

## SWE-bench Verified

None of the three open-model cards lists it. For Claude Opus 5, secondary
sources quote 96.0% as a five-trial mean from the
[Opus 5 system card](https://www-cdn.anthropic.com/c5fbac3f0b1280a933ebd26d3cb8bb9f5bdeaf48/Claude%20Opus%205%20System%20Card.pdf).
The card was too large to read with the tool used here, so that number is
unchecked against the primary source.
