import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MtpWorkspace(unittest.TestCase):
    def test_tp_shards_keep_full_width_mtp_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'test.c'
            source.write_text(r'''
#include <assert.h>
#define SparkStageModuleDeviceAllocate TestAllocate
#include "modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c"
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
int main(void)
{
    for (unsigned degree = 1; degree <= 16; degree *= 2)
    for (unsigned rows = 1; rows <= 7; rows += 3)
    for (unsigned mtp = 0; mtp <= 1; ++mtp)
    {
        SparkQwen38_27bModuleState state = {0};
        SparkQwen38_27bModuleSlot slot = {0};
        state.max_active_sequence_count = rows;
        state.max_input_row_count = rows + 2;
        state.mtp_armed = mtp;
        state.tp.gdn_conv_channels = SPARK_QWEN38_27B_MODEL_GDN_CONV_CHANNELS / degree;
        state.tp.gdn_value_channels = SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION / degree;
        state.tp.gdn_value_heads = SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT / degree;
        state.tp.attn_query_heads = SPARK_QWEN38_27B_MODEL_ATTN_QUERY_HEAD_COUNT / degree;
        state.tp.attn_kv_heads = 1;
        state.tp.ffn_intermediate = 128;
        assert(SparkQwen38_27bModuleAllocateSlot(&state, &slot) == SPARK_STATUS_OK);
        uint64_t n = state.max_input_row_count;
        assert(capacity(slot.qkv_bf16) >= n * state.tp.gdn_conv_channels * 2);
        assert(capacity(slot.gated_bf16) >= n * state.tp.gdn_value_channels * 2);
        if (mtp)
        {
            assert(capacity(slot.qkv_bf16) >= n * 2 * SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES);
            assert(capacity(slot.gated_bf16) >= n * SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES);
        }
        for (unsigned i = 0; i < count; ++i) free(allocations[i]);
        count = 0;
        cudaStreamDestroy(slot.cuda_stream);
        free(slot.dspark_logits_host);
        free(slot.dspark_hidden_host);
    }
    return 0;
}
''')
            paths = ['.', 'include', 'src', 'runtime', 'tests/cuda_stub', 'model-families/common/include', 'model-families/qwen38_27b/include', 'model-families/qwen38_27b/include/sparkpipe', 'modules/qwen38_27b_resident_decode_stage/include', 'modules/qwen38_27b_resident_decode_stage/source']
            command = ['cc', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-D_DARWIN_C_SOURCE', '-ffunction-sections', '-fdata-sections', *['-I'+str(ROOT/p) for p in paths], str(source), 'runtime/stage_module_common.c', 'tests/cuda_stub/cuda_runtime_stub.c', 'build/libsparkpipe_runtime.a', 'build/libsparkpipe_core.a', '-pthread', '-lm', '-Wl,-dead_strip' if os.uname().sysname == 'Darwin' else '-Wl,--gc-sections', '-o', str(Path(temp)/'test')]
            built = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            tested = subprocess.run([str(Path(temp)/'test')], text=True, capture_output=True)
            self.assertEqual(tested.returncode, 0, tested.stdout+tested.stderr)


if __name__ == '__main__':
    unittest.main()
