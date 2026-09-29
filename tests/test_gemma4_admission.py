import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

SOURCE = r'''
#include <assert.h>
#include "modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_module.c"
static uint32_t admit_lanes(SparkGemma4ModuleState *state, uint32_t rows, uint32_t prefill, uint32_t lanes, uint32_t *reason);
static uint32_t admit(SparkGemma4ModuleState *state, uint32_t rows, uint32_t prefill, uint32_t *reason)
{
    return admit_lanes(state, rows, prefill, prefill ? 1u : rows, reason);
}
static uint32_t admit_lanes(SparkGemma4ModuleState *state, uint32_t rows, uint32_t prefill, uint32_t lanes, uint32_t *reason)
{
    SparkModelDriverAdmissionRequest request;
    SparkModelDriverAdmissionDecision decision;
    memset(&request, 0, sizeof(request));
    request.descriptor_bytes = sizeof(request);
    request.program_id = 1u;
    request.request_id = 1u;
    request.sequence_id = 1u;
    request.active_slot_count = lanes;
    request.new_token_count = rows;
    request.frame_flags = prefill ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
    assert(SparkGemma4ModuleAdmit(state, &request, &decision) == SPARK_STATUS_OK);
    *reason = decision.rejection_reason;
    return decision.accepted;
}
int main(void)
{
    SparkGemma4ModuleState state;
    SparkModelDriverAdmissionRequest request;
    SparkModelDriverAdmissionDecision decision;
    uint32_t reason;
    memset(&state, 0, sizeof(state));
    state.max_active_sequence_count = 8u;
    state.max_input_row_count = 32u;
    state.pipeline_slot_count = 1u;
    state.kv_block_count = 512u;
    SparkStageModuleAtomicStateArrayInitialize(state.slot_states, state.pipeline_slot_count);
    admit(&state, 25u, 1u, &reason);
    assert(reason != SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
    admit(&state, 32u, 1u, &reason);
    assert(reason != SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
    assert(admit(&state, 33u, 1u, &reason) == 0u);
    assert(reason == SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
    admit_lanes(&state, 16u, 1u, 4u, &reason);
    assert(reason != SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
    assert(admit_lanes(&state, 16u, 1u, 9u, &reason) == 0u);
    assert(reason == SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
    assert(admit(&state, 9u, 0u, &reason) == 0u);
    assert(reason == SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
    memset(&request, 0, sizeof(request));
    request.descriptor_bytes = sizeof(request);
    request.program_id = 1u;
    request.control_generation = 2u;
    request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
    assert(SparkGemma4ModuleAdmit(&state, &request, &decision) == SPARK_STATUS_OK);
    assert(decision.accepted == 1u);
    return 0;
}
'''


class GemmaAdmission(unittest.TestCase):
    def test_prefill_rows_and_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'test.c'
            source.write_text(SOURCE)
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
