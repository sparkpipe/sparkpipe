# Model Support Contract

SparkPipe targets all open-source frontier-level models that can be mapped to a
supported deployment. The runtime is shared; every model has an exact,
checkpoint-derived execution package.

## Product model set

The families that have drivers are listed in the README
[Models](../README.md#models) section, which is authoritative for what the
tree contains. A family is identified by its exact checkpoint contract in
`model_contracts/`:

| Family | Checkpoint served | Contract |
| --- | --- | --- |
| DeepSeek V4.1 Flash | deepseek-ai/DeepSeek-V4.1-Flash-1017 | `dsv41_flash_authoritative.json` |
| GLM 5.3 Flash | zai-org/GLM-5.3-Flash | `glm53_flash_authoritative.json` |
| GLM 5.3 Full (glm52 family) | zai-org/GLM-5.3 (FP8 release) | `glm53_full_authoritative.json` |
| Kimi K3 | moonshotai/Kimi-K3 | `k3_authoritative.json` |
| Qwen 3.8 Max | a 4-bit expert build of Qwen/Qwen3.8-2.4T-A95B | `qwen38_authoritative.json` |
| Qwen 3.8 27B | Qwen/Qwen3.8-27B-FP8 | `qwen38_27b_authoritative.json` |
| Qwen 3.8 Flash (internal) | a 4-bit expert build of Qwen/Qwen3.8-Flash-Next | `qwen4_flash_authoritative.json` |
| MiMo 2.6 Flash, Pro | XiaomiMiMo/MiMo-V2.6-Flash-RL, MiMo-V2.6-Pro-RL | `mimo26_flash_authoritative.json`, `mimo26_pro_authoritative.json` |
| Gemma 4 31B, 26B-A4B | google/gemma-4-31B-it, gemma-4-26B-A4B-it | `gemma4_31b_authoritative.json`, `gemma4_26b_a4b_authoritative.json` |
| Ling 3.0 Flash and its finance fine-tune | inclusionAI/Ling-3.0-flash, Ling-3.0-flash-Fin | `ling_authoritative.json`, `lingfin_authoritative.json` |
| Laguna S 2.1 | poolside/Laguna-S-2.1 | `laguna_authoritative.json` |
| Hunyuan HY4 preview | tencent/Hy4-preview-FP8 | `hy4_authoritative.json` |
| Muse Glimmer 30B | meta-models/Muse-Glimmer-30B | `muse_glimmer_authoritative.json` |

The Korean models K-EXAONE 2.0 (LG AI Research), A.X K2 (SK Telecom) and
Solar Open 2 (Upstage) are product targets; their contracts and drivers are
open items in [`../TECHDEBT.md`](../TECHDEBT.md#korean-models).

Direction:

- A model at or near the frontier serves at its publisher's full precision
  whenever it fits one fleet. A vetted 4-bit build serves only a model that
  does not fit one fleet at full precision, as Qwen 3.8 Max does; such a model
  moves to full precision across two fleets only once it is the frontier.
- The Korean models come first among new models.
- DeepSeek V4.1 Flash leads the DeepSeek line; DeepSeek V4 Pro is not a
  target. MiMo 2.6 is the MiMo target, Ling 3.0 Flash and its finance
  fine-tune the Ling targets, and GLM 5.3 the GLM target (the glm52 family
  serves GLM 5.3 Full).
- Qwen 3.8 Flash is for internal use only: its license requires a separate
  agreement with Qwen for any model-as-a-service business. Qwen 3.8 27B
  (Apache 2.0) and Qwen 3.8 Max (a separate license only above US$50 million
  of yearly revenue) are offered.

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
