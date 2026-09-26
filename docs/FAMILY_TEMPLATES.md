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
| `abi/` | firmware ABI structs shared by the family firmware headers |

A header holds only functions that every family including it shares. When
two groups of families carry different versions of a function (for example
the serial and parallel wave metadata kernels), each version has its own
header, named so that the difference is visible.

## Templates included by headers

A header cannot use `SPARK_FAMILY_CAMEL`: the translation unit that includes
it may define that name itself. Header templates therefore take their own
one-argument macro, which the including header defines just before the first
template and undefines after the last:

```c
#define SPARK_ABI_TYPE(name) SparkQwen38Max##name
#include "sparkpipe/family/abi/spark_abi_linear_view.h"
...
#undef SPARK_ABI_TYPE
```

| Template | Macros | Included by |
| --- | --- | --- |
| `family/abi/spark_abi_*.h` | `SPARK_ABI_TYPE` | `modules/<family>_resident_decode_stage/include/sparkpipe/spark_<family>_resident_decode_stage_firmware.h` |
| `model-families/common/include/sparkpipe/spark_work_control_api.h` | `SPARK_WORK_CONTROL_FN`, `SPARK_WORK_CONTROL_TYPE` | `spark_<family>_work_control.h`; the matching bodies are in `spark_work_control_common.h` |

To find `SparkQwen38MaxLinearView`, search for `SPARK_ABI_TYPE(LinearView)`.
Header templates have no include guard, because each family instantiates
them once. Constants stay in the family header: a macro cannot define a
macro name.

## Rule for moving code here

A function moves only if every family that includes the shared body compiles
it to the same object code it produced before: identical host disassembly and
SASS, both at `-O3` and with inlining and IPA disabled. A copy that compiles
differently is drift, not duplication. It stays in its family until someone
decides which behaviour is right.

A declaration (a struct, a typedef or a prototype) moves only if every
translation unit that includes it preprocesses to the same tokens before
and after the move, line markers aside. That covers the host `.c` files, the
`.cu` files through `nvcc -E`, and the tests and tools that include the
header.

## Adoption gate

`tests/test_template_adoption.py` fails when shared code exists but a family
does not use it:

1. Serving adapters must call the adapter template and must not parse the
   `tp_collective` configuration themselves.
2. Pack synthesizers under `modules/*/tools/` must include
   `spark_pack_synthesize_common.h`.
3. Work-control headers must include `spark_work_control_api.h`, and
   batch-tuning headers must include the shared bucket ladder.
4. No firmware header may define a struct whose body, after the family name
   is removed, equals another firmware header's struct or an `abi/` template.

Families that have not adopted a pattern yet are listed in the test with a
reason. A listed family that adopts fails the gate until it is removed from
the list, so the lists only shrink.
