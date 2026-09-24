import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GemmaWorkspace(unittest.TestCase):
    def test_tp_shards_keep_full_width_residual_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'test.c'
            source.write_text(r'''
#include <assert.h>
#define SparkStageModuleDeviceAllocate TestAllocate
#include "modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_module.c"
#undef SparkStageModuleDeviceAllocate
static void *allocations[64];
static uint64_t sizes[64];
static unsigned count;
SparkStatus TestAllocate(SparkStageModuleLedger *ledger, uint64_t bytes, void **pointer)
{
    (void)ledger;
    assert(count < 64 && bytes > 0);
    *pointer = malloc(bytes);
    assert(*pointer);
    allocations[count] = *pointer;
    sizes[count++] = bytes;
    return SPARK_STATUS_OK;
}
static uint64_t capacity(void *pointer)
{
    for (unsigned i = 0; i < count; ++i) if (allocations[i] == pointer) return sizes[i];
    abort();
}
uint32_t SparkGemma4HeadDirectArgmaxScratchElements(uint32_t rows) { return rows; }
int main(void)
{
    for (unsigned degree = 1; degree <= 16; degree *= 2)
    for (unsigned rows = 1; rows <= 7; rows += 3)
    {
        SparkGemma4ModuleState state = {0};
        SparkGemma4ModuleSlot slot = {0};
        state.max_active_sequence_count = rows;
        state.tp_degree = degree;
        state.sliding_kv_heads_per_rank = 1;
        state.full_kv_heads_per_rank = 1;
        assert(SparkGemma4ModuleAllocateSlot(&state, &slot) == SPARK_STATUS_OK);
        uint64_t full = rows * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
        assert(capacity(slot.attn_output_bf16) >= full);
        assert(capacity(slot.delta_bf16) >= full);
        assert(capacity(slot.mlp_down_bf16) >= full);
        assert(capacity(slot.branch_bf16) >= full);
        assert(capacity(slot.attn_head_output_bf16) >= rows * SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT / degree * SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES);
        assert(capacity(slot.attn_head_output_bf16) >= rows * SPARK_GEMMA4_MODEL_FULL_QUERY_HEAD_COUNT / degree * SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES);
        for (unsigned i = 0; i < count; ++i) free(allocations[i]);
        count = 0;
        cudaStreamDestroy(slot.cuda_stream);
    }
    return 0;
}
''')
            paths = ['.', 'include', 'src', 'runtime', 'tests/cuda_stub', 'model-families/common/include', 'model-families/gemma4/include', 'model-families/gemma4/include/sparkpipe', 'modules/gemma4_resident_decode_stage/include', 'modules/gemma4_resident_decode_stage/source']
            command = ['cc', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-D_DARWIN_C_SOURCE', '-ffunction-sections', '-fdata-sections', *['-I'+str(ROOT/p) for p in paths], str(source), 'runtime/stage_module_common.c', 'tests/cuda_stub/cuda_runtime_stub.c', 'build/libsparkpipe_runtime.a', 'build/libsparkpipe_core.a', '-pthread', '-lm', '-Wl,-dead_strip' if os.uname().sysname == 'Darwin' else '-Wl,--gc-sections', '-o', str(Path(temp)/'test')]
            for flavor, flags in [('dense', []), ('moe', ['-DSPARK_GEMMA4_MOE_BUILD=1', '-DSPARK_GEMMA4_MODEL_MOE_BLOCK=1'])]:
                with self.subTest(flavor=flavor):
                    built = subprocess.run(command + flags, cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(built.returncode, 0, built.stderr)
                    tested = subprocess.run([str(Path(temp)/'test')], text=True, capture_output=True)
                    self.assertEqual(tested.returncode, 0, tested.stdout+tested.stderr)



if __name__ == '__main__':
    unittest.main()
