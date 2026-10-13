# Family templates

Model families were once created by copying another family and renaming
it, and the code that stayed identical across forks lived once under
`include/sparkpipe/family/`, instantiated by each family under its own
names. The drivers now run on the common stage runner
(`inference/runner/`) and the shared kernels (`inference/kernels/`), which
take parameters instead of names. What remains of the template layer is
listed below; new shared code goes to the runner or the kernels with
generic names (`tests/test_dry_law.py`), not here.

## The idiom

A translation unit names its family, then includes the shared bodies it uses:

```c
#define SPARK_FAMILY_CAMEL Glm52
#define SPARK_FAMILY_UPPER GLM52
#define SPARK_FAMILY_LOWER glm52
#include "sparkpipe/family/spark_family.h"
#include "sparkpipe/family/stagepack/spark_stagepack_glm.h"
```

Inside a template:

| Written as | Expands to (glm52) |
| --- | --- |
| `SPARK_FAMILY(StagePackShapeBf16)` | `SparkGlm52StagePackShapeBf16` |
| `SPARK_FAMILY_CONST(MODULE_TAG)` | `SPARK_GLM52_MODULE_TAG` |
| `SPARK_FAMILY_BARE(LayerBuffers)` | `Glm52LayerBuffers` |
| `SPARK_FAMILY_BARE_CONST(TOP_K)` | `GLM52_TOP_K` |
| `SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER)` | `"glm52"` |

Symbols keep their family names, so call sites, exports and logs do not
change. To find a definition, drop the family prefix and search for
`SPARK_FAMILY(<name>)`. `tests/family_source.py` expands a source file's
template includes for tests that grep source text.

## What remains

| Template | Included by |
| --- | --- |
| `stagepack/spark_stagepack_glm.h`, `spark_stagepack_shape_bf16.h`, `spark_stagepack_tp_shards_cols.h` | `model-families/glm52/include/sparkpipe/spark_glm52_stagepack.h` |
| `stagepack/spark_stagepack_kind_is_hc.h`, `spark_stagepack_shape_f32.h` | `modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stagepack_format.h` |
| `glm/spark_glm_head_maxloc.cuh` | `tests/host_cuda/head_sampling_host.cu`, `tests/host_cuda/glm5_next_expert_cover_host.cu`, `tests/test_working_set_rollback.cu` |
| `module/spark_module_combine.h` | `tools/mesh_round_bench.cu` |

The serving, module, ABI, CUDA glue, synthesizer and validation templates
had no remaining users once the drivers moved to the common runner and were
removed on 2026-10-13.

## Rule for moving code between a template and its families

A function moves only if every family that includes the shared body compiles
it to the same object code it produced before: identical host disassembly and
SASS, both at `-O3` and with inlining and IPA disabled. A copy that compiles
differently is drift, not duplication. Compare host objects with
`tools/host_codegen_diff.py` and cubins with `tools/kernel_codegen_diff.py`.
