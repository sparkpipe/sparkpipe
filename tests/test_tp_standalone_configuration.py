import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StandaloneConfiguration(unittest.TestCase):
    def test_model_configuration_parses_boolean_values(self):
        for family, tag, degree in (("gemma4", "Gemma4", 16), ("minimax", "Minimax", 4), ("qwen4_flash", "Qwen4Flash", 4), ("qwen38_max", "Qwen38Max", 16)):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                macro = family.upper()
                collective_check = ""
                collective_stub = ""
                if family == "qwen38_max":
                    collective_stub = """
SparkStatus SparkTpDeviceCollectiveSubmitBf16(SparkTpDeviceCollective *collective, const SparkTpDeviceCollectiveSubmission *submission)
{
    (void)collective;
    (void)submission;
    abort();
}
"""
                    collective_check = """
    state.allow_unqualified_execution = 1u;
    state.tp_degree = 16u;
    state.tp_standalone = 0u;
    assert(SparkQwen38MaxModuleTpAllReduceHidden(&state, 0, 0, 1u) == SPARK_STATUS_INTERNAL_ERROR);
    state.tp_standalone = 1u;
    assert(SparkQwen38MaxModuleTpAllReduceHidden(&state, 0, 0, 1u) == SPARK_STATUS_OK);
"""
                source = Path(temp) / "test.c"
                source.write_text(f'''
#include <assert.h>
#include "modules/{family}_resident_decode_stage/source/spark_{family}_resident_decode_stage_module.c"
{collective_stub}
int main(void)
{{
    Spark{tag}ModuleState state = {{0}};
    setenv("SPARK_{macro}_TP_DEGREE","{degree}",1);
    setenv("SPARK_{macro}_TP_RANK","0",1);
    setenv("SPARK_{macro}_TP_STANDALONE","0",1);
    (void)Spark{tag}ModuleConfigure(&state);
    assert(state.tp_standalone == 0u);
    setenv("SPARK_{macro}_TP_STANDALONE","1",1);
    (void)Spark{tag}ModuleConfigure(&state);
    assert(state.tp_standalone == 1u);
    unsetenv("SPARK_{macro}_TP_STANDALONE");
    (void)Spark{tag}ModuleConfigure(&state);
    assert(state.tp_standalone == 0u);
    setenv("SPARK_{macro}_TP_STANDALONE","2",1);
    assert(Spark{tag}ModuleConfigure(&state) == SPARK_STATUS_INVALID_ARGUMENT);
    setenv("SPARK_{macro}_TP_STANDALONE","garbage",1);
    assert(Spark{tag}ModuleConfigure(&state) == SPARK_STATUS_INVALID_ARGUMENT);
{collective_check}
    return 0;
}}
''')
                paths = [".", "include", "src", "runtime", "tests/cuda_stub", "model-families/common/include", f"model-families/{family}/include", f"model-families/{family}/include/sparkpipe", f"modules/{family}_resident_decode_stage/include", f"modules/{family}_resident_decode_stage/source"]
                command = ["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-D_DARWIN_C_SOURCE", '-DQWEN4_FLASH_MODEL_REVISION="test"', '-DQWEN38_MODEL_REVISION="test"', "-ffunction-sections", "-fdata-sections", *["-I" + str(ROOT / p) for p in paths], str(source), "runtime/stage_module_common.c", "tests/cuda_stub/cuda_runtime_stub.c", "build/libsparkpipe_runtime.a", "build/libsparkpipe_core.a", "-pthread", "-lm", "-Wl,-dead_strip" if os.uname().sysname == "Darwin" else "-Wl,--gc-sections", "-o", str(Path(temp) / "test")]
                built = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                tested = subprocess.run([str(Path(temp) / "test")], text=True, capture_output=True)
                self.assertEqual(tested.returncode, 0, tested.stdout + tested.stderr)


if __name__ == "__main__":
    unittest.main()
