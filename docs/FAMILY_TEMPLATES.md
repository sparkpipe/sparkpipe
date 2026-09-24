# Family templates

Most model families were created by copying another family and renaming
it: glm52 → glm5_next → laguna and ling, and qwen38_27b → qwen4_flash. The
qwen38_max, gemma4, muse_glimmer and minimax lineage was built the same way.
Code that is still identical across forks lives once under
`include/sparkpipe/family/`, and each family instantiates it under its own
names.

## The idiom

A translation unit names its family, then includes the shared bodies it uses:

```c
#define SPARK_FAMILY_CAMEL Glm5Next
#define SPARK_FAMILY_UPPER GLM5_NEXT
#define SPARK_FAMILY_LOWER glm5_next
#include "sparkpipe/family/spark_family.h"
#include "sparkpipe/family/serving/spark_serving_quiesce.h"
```

Inside a template:

| Written as | Expands to (glm5_next) |
| --- | --- |
| `SPARK_FAMILY(ServingQuiesce)` | `SparkGlm5NextServingQuiesce` |
| `SPARK_FAMILY_CONST(MODULE_TAG)` | `SPARK_GLM5_NEXT_MODULE_TAG` |
| `SPARK_FAMILY_BARE(LayerBuffers)` | `Glm5NextLayerBuffers` |
| `SPARK_FAMILY_BARE_CONST(TOP_K)` | `GLM5_NEXT_TOP_K` |
| `SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER)` | `"glm5_next"` |

Symbols keep their family names, so call sites, exports and logs do not
change. To find the definition of `SparkGlm5NextServingQuiesce`, drop the
family prefix and search for `SPARK_FAMILY(ServingQuiesce)`.
`tests/family_source.py` expands a source file's template includes for tests
that grep source text.

## Layout

| Directory | Contents |
| --- | --- |
| `serving/` | serving adapter functions |
| `module/` | resident decode stage module functions |
| `stagepack/` | stage-pack format helpers |
| `cuda/` | CUDA host glue shared across lineages |
| `glm/` | GLM-lineage CUDA host glue, layer and unity code, api.h |
| `synth/` | pack synthesizer helpers |
| `validation/` | GPU validator reference math |

A header holds only functions that every family including it shares. When
two groups of families carry different versions of a function (for example
the serial and parallel wave metadata kernels), each version has its own
header, named so that the difference is visible.

## Rule for moving code here

A function moves only if every family that includes the shared body compiles
it to the same object code it produced before: identical host disassembly and
SASS, both at `-O3` and with inlining and IPA disabled. A copy that compiles
differently is drift, not duplication. It stays in its family until someone
decides which behaviour is right.
