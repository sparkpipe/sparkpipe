# Model Support Contract

SparkPipe targets all open-source frontier-level models that can be mapped to a
supported deployment. The runtime is shared; every model has an exact,
checkpoint-derived execution package.

## Product model set

The families that have drivers are listed in the README
[Models](../README.md#models) section, which is authoritative for what the
tree contains. A family is identified by its exact checkpoint contract in
`model_contracts/`:

| Family | Checkpoint (`model_id`) | Contract |
| --- | --- | --- |
| DeepSeek V4.1 Flash | deepseek-ai/DeepSeek-V4.1-Flash | `dsv41_flash_authoritative.json` |
| DeepSeek V4 Flash | deepseek-ai/DeepSeek-V4-Flash-0731 | `dsv4_flash_authoritative.json` |
| DeepSeek V4 Pro | deepseek-ai/DeepSeek-V4-Pro-0813 | `dsv4_pro_authoritative.json` |
| GLM 5.3 Flash | zai-org/GLM-5.3-Flash | `glm53_flash_authoritative.json` |
| GLM 5.3 Full (glm52 module) | RadixArk/GLM-5.3-NVFP4 | `glm53_full_authoritative.json` |
| Kimi K3 | moonshotai/Kimi-K3-MXFP4 | `k3_authoritative.json` |
| Qwen 3.8 Max | Qwen/Qwen3.8-2.4T-A95B | `qwen38_authoritative.json` |
| Qwen 3.8 27B | Qwen/Qwen3.8-27B | `qwen38_27b_authoritative.json` |
| Qwen4 Flash | Qwen/Qwen3.8-Flash-Next | `qwen4_flash_authoritative.json` |
| MiMo 2.6 Flash, Pro | XiaomiMiMo/mimo-v2.6-flash-rl, mimo-v2.6-pro-rl | `mimo26_flash_authoritative.json`, `mimo26_pro_authoritative.json` |
| MiMo 2.5 | XiaomiMiMo/MiMo-V2.5-Base | `mimo25_authoritative.json` |
| Gemma 4 31B, 26B-A4B | google/gemma-4-31B-it, gemma-4-26B-A4B-it | `gemma4_31b_authoritative.json`, `gemma4_26b_a4b_authoritative.json` |
| Ling 3.0 Flash and its finance fine-tune | inclusionAI/Ling-3.0-flash, ling-3.0-flash-fin | `ling_authoritative.json`, `lingfin_authoritative.json` |
| Laguna S 2.1 | poolside/Laguna-S-2.1 | `laguna_authoritative.json` |
| Hunyuan HY4 | AngelSlim/Hy4-preview-GGUF | `hy4_authoritative.json` |
| Muse Glimmer 30B | meta-models/Muse-Glimmer-30B | `muse_glimmer_authoritative.json` |
| MiniMax H3 (text) | no contract; revision `minimax-h3-text-bf16-...` in `deployment/minimax_text_tp4` | none |

`glm52_authoritative.json` (zai-org/GLM-5.2) and `qwen36_authoritative.json`
(Qwen/Qwen3.6-27B) remain in the tree but are not product targets.

Owner direction (2026-09-28):

- GLM 5.2 weights are deprecated; the glm52 module serves GLM 5.3 Full.
- MiMo 2.6 is the MiMo target.
- DeepSeek V4.1 Flash leads the DeepSeek line. V4 Pro 0813 stays, last in
  the driver order, as the base for V4.1 Pro, which will be supported when
  released.
- Kimi K3 moves up into the driver-order slot DSV4 Pro 0813 held
  ([GOALS.md](GOALS.md), model direction).
- Ling 3.0 Flash and its finance fine-tune are the Ling targets; Ling 2.x is
  not.
- Qwen models are for internal use and are not enabled on the external API
  service (license terms).
- MiniMax 2.5 is not a support target.

The catalog is not capped at this table. A new open-source frontier model joins
the product set by satisfying the same package and qualification contract; it
does not require a second serving stack.

## Package boundary

Every supported checkpoint binds:

- upstream model ID, exact revision, and source implementation revision;
- tokenizer and prompt-template identities;
- model geometry, layer kinds, routing, attention, recurrent state, and head;
- weight, activation, accumulation, KV, and scale formats;
- rank-local sharding, PP slices, TP communicators, and storage placement;
- exact native CUDA modules and graph geometry;
- speculation provider and verification contract when enabled; and
- numerical, memory, transport, and performance qualification identities.

Common runtime code never infers one model's constants from another model and
never treats a nearby architecture as compatible because tensor names happen to
match.

## Qualification

A model is ready only when the exact checkpoint and deployment have retained:

1. package and source identity;
2. host contract and bounds tests;
3. exact target CUDA compile and link receipts;
4. real-weight pack validation;
5. numerical comparison with the authoritative implementation;
6. physical route and collective evidence;
7. deterministic prompt, streaming, stop, cancellation, and error behavior;
8. latency and throughput measurements under named request shapes; and
9. clean merged-main release and all-rank ready identity.

A model-family directory, simulator, analytical estimate, successful pack, or
compiled kernel is evidence for its own domain only. None is a production-ready
claim by itself.

## Residency and promotion

The endpoint keeps the complete configured catalog addressable. Models may be
resident, warm, promotable, or unavailable. Promotion from the verified model
store to ready state has a maximum target of one minute and includes shard
placement, binding, prewarm, communicator creation, all-rank agreement, and
atomic publication.

Co-resident models share hardware through priority-aware gang scheduling. A
model switch changes the selected execution plan; it does not change the API or
require applications to reconnect.

Open implementation gaps live in [`../TECHDEBT.md`](../TECHDEBT.md). Measured
model results live in [`../PERFORMANCE_STATUS.md`](../PERFORMANCE_STATUS.md).
