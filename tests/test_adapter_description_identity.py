import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = ROOT / "examples/model_descriptions"
FIELDS = (
    ("model_id", "SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_DRIVER_MODEL_ID)"),
    ("model_revision", "SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION"),
    ("stage_name", "SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_STAGE_NAME)"),
    ("target", "SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_TARGET)"),
    ("model_description_sha256", "SPARK_QWEN38_SERVING_ADAPTER_DRIVER_DESCRIPTION_SHA256"),
)
LIBRARY = "dylib" if os.uname().sysname == "Darwin" else "so"
QWEN38_MAX_BUILD = ["EXPERT_CODEC=fp8", "MODEL_REVISION=d2dc35658bcf77e66643428cb52e774cc3b5bd29", "CONTRACT_SHA256=" + "0" * 64]
CASES = (
    ("gemma4", ".", [f"build/libgemma4_serving_adapter.{LIBRARY}"], "gemma4", "gemma4_resident_decode_stage_bf16_firmware.json", ["-C", "modules/gemma4_resident_decode_stage"]),
    ("gemma4_moe", ".", [f"build/libgemma4_moe_serving_adapter.{LIBRARY}"], "gemma4", "gemma4_26b_resident_decode_stage_firmware.json", ["-C", "modules/gemma4_resident_decode_stage", "-f", "Makefile.moe"]),
    ("muse_glimmer", ".", [f"build/libmuse_glimmer_serving_adapter.{LIBRARY}"], "muse_glimmer", "muse_glimmer_resident_decode_stage_firmware.json", ["-C", "modules/muse_glimmer_resident_decode_stage"]),
    ("minimax", "modules/minimax_resident_decode_stage", ["adapter"], "minimax", "minimax_resident_decode_stage_bf16_firmware.json", ["-C", "modules/minimax_resident_decode_stage"]),
    ("qwen38_max", "modules/qwen38_max_resident_decode_stage", ["adapter", *QWEN38_MAX_BUILD], "qwen38_max", "qwen38_max_resident_decode_stage_firmware.json", ["-C", "modules/qwen38_max_resident_decode_stage", *QWEN38_MAX_BUILD]),
    ("qwen4_flash", "modules/qwen4_flash_resident_decode_stage", ["adapter", "EXPERT_CODEC=fp8"], "qwen4_flash", "qwen4_flash_resident_decode_stage_firmware.json", ["-C", "modules/qwen4_flash_resident_decode_stage"]),
)


def adapter_source(family):
    return ROOT / f"modules/{family}_resident_decode_stage/source/spark_{family}_serving_adapter.c"


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(" ".join(command) + "\n" + result.stdout + result.stderr)
    return result.stdout


def compile_flags(directory, goals, source):
    lines = run(["make", "-n", "-W", os.path.relpath(source, ROOT / directory), *goals], ROOT / directory).replace("\\\n", " ").splitlines()
    commands = [tokens for tokens in map(shlex.split, lines) if "-o" in tokens and any(token.endswith(source.name) for token in tokens)]
    if len(commands) != 1:
        raise AssertionError(f"expected one compile of {source.name}, found {len(commands)}")
    flags, index, tokens = [], 0, commands[0]
    while index < len(tokens):
        if tokens[index] in ("-I", "-D", "-U"):
            flags += tokens[index:index + 2]
            index += 1
        elif tokens[index].startswith(("-I", "-D", "-U", "-std=")):
            flags.append(tokens[index])
        index += 1
    return flags


def adapter_contract(directory, goals, family):
    source = adapter_source(family)
    with tempfile.TemporaryDirectory() as temp:
        probe = Path(temp) / "probe.c"
        probe.write_text(f'#include "{source}"\n' + "".join(f"spark_probe_{name} {expression}\n" for name, expression in FIELDS))
        output = run(["cc", "-E", "-P", *compile_flags(directory, goals, source), "-I" + str(ROOT / "tests/cuda_stub"), str(probe)], ROOT / directory)
    values = {}
    for line in output.splitlines():
        if line.startswith("spark_probe_"):
            name, expansion = line[len("spark_probe_"):].split(" ", 1)
            values[name] = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', expansion))
    return values


def module_identity(arguments):
    output = run(["make", "-s", "--no-print-directory", *arguments, "--eval", "spark-print-%: ; @echo $($*)", "spark-print-MODULE_IDENTIFIER", "spark-print-MODULE_TARGET"], ROOT)
    return output.split()


class AdapterDescriptionIdentity(unittest.TestCase):
    def test_adapters_send_what_their_description_compiles_to(self):
        for name, directory, goals, family, description, module_arguments in CASES:
            with self.subTest(adapter=name):
                path = DESCRIPTIONS / description
                document = json.loads(path.read_text())
                (stage,) = document["stages"]
                modules = {operation["module"] for program in stage["programs"] for operation in program["operations"]}
                identifier, target = module_identity(module_arguments)
                contract = adapter_contract(directory, goals, family)
                self.assertEqual(contract, {"model_id": document["model"]["id"], "model_revision": document["model"]["revision"], "stage_name": stage["name"], "target": stage["target"], "model_description_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                self.assertEqual((stage["target"], modules), (target, {identifier}))


if __name__ == "__main__":
    unittest.main()
