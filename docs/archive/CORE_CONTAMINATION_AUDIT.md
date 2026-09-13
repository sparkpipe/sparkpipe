# Core contamination audit

## Finding

The original build treated a large GLM-5.2 host implementation as part of the common SparkPipe library. That made directory naming suggest a neutral platform while the linked ownership boundary included GLM scheduling, PP13 runtime, stage packs, KV policy, prefix caching, prompt handling, service backends, and HTTP serving behavior.

The problem was not that SparkPipe contains a substantial GLM implementation. GLM-5.2 is a legitimate model-family implementation and proving ground. The problem was that those facilities were reachable through the same compilation and archive boundary as the supposedly model-neutral core.

## Corrected ownership

The proposal makes source ownership explicit through manifests:

| Layer | Manifest | Owned responsibility |
|---|---|---|
| Neutral support | `core/sources.mk` | status, filesystem, JSON, SHA-256 |
| Neutral compiler | `core/sources.mk` | model description, module library, driver compiler |
| Neutral runtime | `core/sources.mk` | driver loader and orchestrator |
| Shared model runtime | `model-families/common/sources.mk` | hidden transport, memory link, KV provider ABI, collectives, tokenizer, stage KV client |
| GLM-5.2 | `model-families/glm52/sources.mk` | all GLM-specific scheduling, KV, PP13, prompt, serving, and topology logic |
| Qwen 3.6 | `model-families/qwen38_27b/sources.mk` | Qwen-specific work control |
| Deployment | `deployment/sources.mk` | release management |

The resulting archives are:

```text
libsparkpipe_core.a
libsparkpipe_compiler.a
libsparkpipe_runtime.a
libsparkpipe_model_common.a
libsparkpipe_deployment.a
libglm52_host.a
libqwen38_27b_host.a
```

Core, compiler, and runtime translation units compile with only:

```text
-Iinclude -Isrc
```

They cannot accidentally include model-family headers because those paths are not available to their compilation commands.

## Automated gate

`tools/audit_core_boundaries.py` and `make architecture_audit` enforce three independent properties:

1. the neutral source manifests contain no model-family implementation;
2. the recursive include closure of neutral sources contains no model-family header;
3. the core/compiler/runtime archives contain no model-family object member or exported model-family symbol.

The current host receipt is:

```text
PASS core-boundary audit: 9 neutral sources, 20 source/header files in include closure, no model-family archive members or exported symbols
```

This gate intentionally checks the built archives as well as paths. Moving files without correcting link ownership would not pass.

## Remaining architectural guardrail

Shared model facilities are not automatically neutral-core facilities. Tokenization, KV provider interfaces, hidden transport, and collectives remain under `model-families/common/`. A facility should move into the neutral ABI only when it is required for routing or lifecycle and can be expressed without model geometry or execution-policy leakage.

The strongest future proof is a second end-to-end production model that requires no changes to the neutral ABI, compiler semantics, loader, or orchestrator.
