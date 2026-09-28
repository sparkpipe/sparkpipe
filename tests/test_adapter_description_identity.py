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
NAMES = ("model_id", "model_revision", "stage_name", "target", "model_description_sha256")
TEMPLATE = ("SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_DRIVER_MODEL_ID)", "SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION", "SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_STAGE_NAME)", "SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_TARGET)", "SPARK_QWEN38_SERVING_ADAPTER_DRIVER_DESCRIPTION_SHA256")
LIBRARY = "dylib" if os.uname().sysname == "Darwin" else "so"
INVOKED = ["MODEL_REVISION=build-supplied", "CONTRACT_SHA256=" + "0" * 64]


def own(prefix, revision, description_hash):
    return (f"SPARK_{prefix}_SERVING_DRIVER_MODEL_ID", revision, f"SPARK_{prefix}_SERVING_STAGE_NAME", f"SPARK_{prefix}_SERVING_TARGET", description_hash)


def root_case(name, family, fields, description, module_arguments):
    return (name, ".", [f"build/lib{name}_serving_adapter.{LIBRARY}"], family, fields, description, ["-C", f"modules/{family}_resident_decode_stage", *module_arguments])


def module_case(family, fields, description, arguments):
    return (family, f"modules/{family}_resident_decode_stage", ["adapter", *arguments], family, fields, description, ["-C", f"modules/{family}_resident_decode_stage", *arguments])


def glm52_case(codec):
    fields = ("SPARK_GLM52_SERVING_DRIVER_MODEL_ID", None, "SPARK_GLM52_SERVING_STAGE_NAME", "SPARK_GLM52_SERVING_TARGET", "GLM_MODEL_DESCRIPTION_SHA256")
    return module_case("glm52", fields, f"glm52_resident_decode_stage_{codec}_firmware.json", [f"EXPERT_CODEC={codec}", *INVOKED])


CASES = (
    root_case("gemma4", "gemma4", TEMPLATE, "gemma4_resident_decode_stage_bf16_firmware.json", []),
    root_case("gemma4_moe", "gemma4", TEMPLATE, "gemma4_26b_resident_decode_stage_firmware.json", ["-f", "Makefile.moe"]),
    root_case("muse_glimmer", "muse_glimmer", TEMPLATE, "muse_glimmer_resident_decode_stage_firmware.json", []),
    root_case("laguna", "laguna", own("LAGUNA", "LAGUNA_MODEL_REVISION", None), "laguna_resident_decode_stage_firmware.json", ["EXPERT_CODEC=bf16", *INVOKED]),
    root_case("ling", "ling", own("LING", "LING_MODEL_REVISION", "LING_MODEL_DESCRIPTION_SHA256"), "ling_resident_decode_stage_firmware.json", ["EXPERT_CODEC=bf16", *INVOKED]),
    root_case("qwen38_27b", "qwen38_27b", own("QWEN38_27B", "QWEN38_27B_MODEL_REVISION", "QWEN38_27B_CONTRACT_SHA256"), "qwen38_27b_resident_decode_stage_firmware.json", []),
    root_case("dsv4", "dsv4", ("SPARK_DSV4_SERVING_DRIVER_MODEL_ID", "SPARK_DSV4_SERVING_DRIVER_MODEL_REVISION", "SPARK_DSV4_SERVING_DRIVER_STAGE_NAME", None, "SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256"), "dsv4_resident_decode_stage_firmware.json", ["EXPERT_CODEC=fp8", *INVOKED]),
    module_case("minimax", TEMPLATE, "minimax_resident_decode_stage_bf16_firmware.json", []),
    module_case("qwen4_flash", TEMPLATE, "qwen4_flash_resident_decode_stage_firmware.json", ["EXPERT_CODEC=fp8"]),
    module_case("qwen38_max", TEMPLATE[:1] + (None,) + TEMPLATE[2:], "qwen38_max_resident_decode_stage_firmware.json", ["EXPERT_CODEC=fp8", *INVOKED]),
    module_case("glm5_next", own("GLM5_NEXT", None, None), "glm5_next_resident_decode_stage_fp8_firmware.json", ["EXPERT_CODEC=fp8", *INVOKED]),
    *(glm52_case(codec) for codec in ("bf16", "int6", "int7", "int8", "fp8", "nvfp4", "mxfp4")),
)


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


def adapter_contract(directory, goals, family, fields):
    source = ROOT / f"modules/{family}_resident_decode_stage/source/spark_{family}_serving_adapter.c"
    with tempfile.TemporaryDirectory() as temp:
        probe = Path(temp) / "probe.c"
        probe.write_text(f'#include "{source}"\n' + "".join(f"spark_probe_{name} {expression}\n" for name, expression in zip(NAMES, fields) if expression))
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
        for name, directory, goals, family, fields, description, module_arguments in CASES:
            with self.subTest(adapter=name, description=description):
                path = DESCRIPTIONS / description
                document = json.loads(path.read_text())
                (stage,) = document["stages"]
                modules = {operation["module"] for program in stage["programs"] for operation in program["operations"]}
                described = dict(zip(NAMES, (document["model"]["id"], document["model"]["revision"], stage["name"], stage["target"], hashlib.sha256(path.read_bytes()).hexdigest())))
                identifier, target = module_identity(module_arguments)
                self.assertEqual(adapter_contract(directory, goals, family, fields), {name: described[name] for name, expression in zip(NAMES, fields) if expression})
                self.assertEqual((stage["target"], modules), (target, {identifier}))


if __name__ == "__main__":
    unittest.main()
