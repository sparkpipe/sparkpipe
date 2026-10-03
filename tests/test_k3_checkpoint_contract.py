#!/usr/bin/env python3
import copy
import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "model_contracts" / "k3_authoritative.json"
GENERATOR = ROOT / "tools" / "generate_k3_contract.py"
MODEL_HEADER = ROOT / "model-families" / "k3" / "include" / "sparkpipe" / "spark_k3_model.h"
GENERATED_HEADER = ROOT / "inference" / "llms" / "kimi_k3" / "generated_config.h"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_k3_contract", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def header_values(path):
    values = {}
    for name, value in re.findall(r"^#define (\w+) (\"[^\"]*\"|\d+)u?$", path.read_text(), re.M):
        values[name] = value.strip('"') if value.startswith('"') else int(value)
    return values


class K3CheckpointContract(unittest.TestCase):
    def setUp(self):
        self.generator = load_generator()
        self.source = json.loads(CONTRACT.read_text())

    def test_generated_outputs_are_current(self):
        result = subprocess.run([sys.executable, str(GENERATOR), "--check"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_source_names_the_exact_published_checkpoint(self):
        self.generator.validate_source(self.source)
        sources = self.source["sources"]
        self.assertEqual(self.source["model_id"], "moonshotai/Kimi-K3")
        self.assertRegex(self.source["source_revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(sources["hf_revision"], self.source["source_revision"])
        for name, record in sources["pinned_files"].items():
            with self.subTest(file=name):
                self.assertGreater(record["bytes"], 0)
                self.assertTrue(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", record.get("sha256", record.get("git_blob_sha1", ""))))
        self.assertEqual(sources["shard_count"], 96)

    def test_both_chat_terminators_stop_generation(self):
        eos = self.source["eos_token_ids"]
        self.assertEqual(set(eos.values()), {163585, 163586})
        generated = json.loads((ROOT / "model_contracts" / "k3.json").read_text())
        self.assertEqual(generated["eos_token_ids"], eos)

    def test_validation_rejects_a_wrong_checkpoint(self):
        mutations = (
            lambda s: s.__setitem__("model_id", "moonshotai/Kimi-K3-MXFP4"),
            lambda s: s.__setitem__("source_revision", "main"),
            lambda s: s["eos_token_ids"].__setitem__("end_of_message", 163585),
            lambda s: s["eos_token_ids"].pop("end_of_message"),
            lambda s: s["speculation"]["drafter"].__setitem__("aux_hidden_state_layer_ids", [24, 48, 72, 88, 93]),
            lambda s: s["speculation"]["drafter"].__setitem__("vocabulary_size", 248320),
            lambda s: s["speculation"]["drafter"].__setitem__("verifier", "moonshotai/Kimi-K3-MXFP4"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                source = copy.deepcopy(self.source)
                mutate(source)
                with self.assertRaises((ValueError, KeyError)):
                    self.generator.validate_source(source)

    def test_family_and_kernel_headers_carry_the_contract(self):
        model = header_values(MODEL_HEADER)
        tokens = self.source["tokens"]
        self.assertEqual(model["SPARK_K3_MODEL_SOURCE_ID"], self.source["model_id"])
        self.assertEqual(model["SPARK_K3_MODEL_SOURCE_REVISION"], self.source["source_revision"])
        self.assertEqual(model["SPARK_K3_MODEL_END_OF_TEXT_TOKEN_ID"], tokens["end_of_text"])
        self.assertEqual(model["SPARK_K3_MODEL_END_OF_MESSAGE_TOKEN_ID"], tokens["end_of_message"])
        self.assertEqual(model["SPARK_K3_MODEL_BEGIN_OF_TEXT_TOKEN_ID"], tokens["begin_of_text"])
        kernel = header_values(GENERATED_HEADER)
        drafter = self.source["speculation"]["drafter"]
        self.assertEqual([kernel[f"K3_DRAFT_TAP_LAYER_{i}"] for i in range(kernel["K3_DRAFT_TAP_COUNT"])], drafter["aux_hidden_state_layer_ids"])
        self.assertEqual((kernel["K3_DRAFT_LAYERS"], kernel["K3_DRAFT_KV_HEADS"], kernel["K3_DRAFT_HEAD_DIM"], kernel["K3_DRAFT_BLOCK_SIZE"]), (drafter["layer_count"], drafter["kv_head_count"], drafter["head_dimension"], drafter["block_size"]))


if __name__ == "__main__":
    unittest.main()
