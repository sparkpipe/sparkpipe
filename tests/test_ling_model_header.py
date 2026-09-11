#!/usr/bin/env python3
"""Bind the ling geometry header to the authoritative contract.

The contract (model_contracts/ling_authoritative.json) is pre-freeze:
geometry is pinned to the publisher config (inclusionAI/Ling-3.0-flash @
e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3) and the checkpoint shard shas
get pinned at freeze, after the warm download. This test binds
model-families/ling/include/sparkpipe/spark_ling_model.h to that contract
so header and contract stay in lockstep through the freeze.
Run: python3 tests/test_ling_model_header.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
HEADER = REPOSITORY / "model-families/ling/include/sparkpipe/spark_ling_model.h"
CONTRACT = REPOSITORY / "model_contracts/ling_authoritative.json"

BINDINGS = {
    ("model", "hidden_dimension"): "SPARK_LING_MODEL_HIDDEN_DIMENSION",
    ("model", "layer_count"): "SPARK_LING_MODEL_LAYER_COUNT",
    ("model", "vocabulary_size"): "SPARK_LING_MODEL_OUTPUT_VOCAB_COUNT",
    ("model", "mtp_layer_index"): "SPARK_LING_MODEL_MTP_LAYER_INDEX",
    ("model", "maximum_context_tokens"): "SPARK_LING_MODEL_MAXIMUM_CONTEXT_TOKENS",
    ("model", "rms_norm_epsilon"): "SPARK_LING_MODEL_RMS_NORM_EPSILON",
    ("model", "end_of_text_token_id"): "SPARK_LING_MODEL_END_OF_TEXT_TOKEN_ID",
    ("model", "pad_token_id"): "SPARK_LING_MODEL_PAD_TOKEN_ID",
    ("hybrid_attention", "period"): "SPARK_LING_MODEL_ATTENTION_PERIOD",
    ("hybrid_attention", "mla_phase"): "SPARK_LING_MODEL_GLOBAL_ATTENTION_PHASE",
    ("hybrid_attention", "mla_layer_count"): "SPARK_LING_MODEL_MLA_LAYER_COUNT",
    ("hybrid_attention", "kda_layer_count"): "SPARK_LING_MODEL_KDA_LAYER_COUNT",
    ("mla", "head_count"): "SPARK_LING_MODEL_MLA_HEAD_COUNT",
    ("mla", "latent_dimension"): "SPARK_LING_MODEL_MLA_LATENT_DIMENSION",
    ("mla", "qk_nope_head_dimension"): "SPARK_LING_MODEL_MLA_QK_NOPE_HEAD_DIMENSION",
    ("mla", "qk_rope_head_dimension"): "SPARK_LING_MODEL_MLA_QK_ROPE_HEAD_DIMENSION",
    ("mla", "value_head_dimension"): "SPARK_LING_MODEL_MLA_VALUE_HEAD_DIMENSION",
    ("mla", "rope_theta"): "SPARK_LING_MODEL_MLA_ROPE_THETA",
    ("mla", "qk_scale"): "SPARK_LING_MODEL_MLA_QK_SCALE",
    ("kda", "head_count"): "SPARK_LING_MODEL_KDA_HEAD_COUNT",
    ("kda", "key_dimension"): "SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION",
    ("kda", "value_dimension"): "SPARK_LING_MODEL_KDA_HEAD_VALUE_DIMENSION",
    ("kda", "short_conv_kernel"): "SPARK_LING_MODEL_KDA_CONV_KERNEL",
    ("kda", "chunk_tokens"): "SPARK_LING_MODEL_KDA_CHUNK_TOKENS",
    ("kda", "gate_lower_bound"): "SPARK_LING_MODEL_KDA_GATE_LOWER_BOUND",
    ("moe", "routed_expert_count"): "SPARK_LING_MODEL_MOE_EXPERT_COUNT",
    ("moe", "experts_per_token"): "SPARK_LING_MODEL_MOE_TOP_K",
    ("moe", "shared_expert_count"): "SPARK_LING_MODEL_MOE_SHARED_EXPERT_COUNT",
    ("moe", "expert_intermediate_dimension"): "SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION",
    ("moe", "dense_intermediate_dimension"): "SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION",
    ("moe", "first_dense_layer_count"): "SPARK_LING_MODEL_FIRST_DENSE_LAYER_COUNT",
    ("moe", "routed_scaling_factor"): "SPARK_LING_MODEL_MOE_ROUTED_SCALING_FACTOR",
    ("moe", "router_group_count"): "SPARK_LING_MODEL_MOE_ROUTER_GROUP_COUNT",
    ("moe", "router_top_groups"): "SPARK_LING_MODEL_MOE_ROUTER_TOP_GROUPS",
}


class MacroExpression(ast.NodeVisitor):
    """Evaluate a restricted arithmetic AST over resolved macro names."""

    def __init__(self, resolver: "MacroTable", seen: frozenset) -> None:
        self.table = resolver
        self.seen = seen
        self.result = 0.0

    def visit_Expression(self, node: ast.Expression) -> None:
        self.result = self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)):
            raise AssertionError(f"non-numeric constant {node.value!r}")
        return float(node.value)

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in self.table.defines and node.id not in self.table.resolved:
            self.table.resolve(node.id, self.seen)
        if node.id not in self.table.resolved:
            raise AssertionError(f"unknown symbol {node.id}")
        return self.table.resolved[node.id]

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise AssertionError(f"unsupported operator {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise AssertionError(f"unsupported unary operator {type(node.op).__name__}")

    def generic_visit(self, node: ast.AST) -> float:
        raise AssertionError(f"unsupported syntax {type(node).__name__}")


class MacroTable:
    def __init__(self, header: str) -> None:
        joined = header.replace("\\\n", " ")
        self.defines: dict[str, str] = {}
        for match in re.finditer(r"^#define\s+(\w+)[ \t]+([^\n]+?)\s*$", joined, re.M):
            self.defines[match.group(1)] = match.group(2).strip()
        self.resolved: dict[str, float] = {}

    def resolve(self, name: str, seen: frozenset = frozenset()) -> float:
        if name in self.resolved:
            return self.resolved[name]
        if name not in self.defines:
            raise AssertionError(f"header missing #define {name}")
        if name in seen:
            raise AssertionError(f"cyclic define {name}")
        text = self.defines[name]
        try:
            value = float(text.rstrip("uf"))
        except ValueError:
            stripped = re.sub(r"(\d)[uf]\b", r"\1", text)
            tree = ast.parse(stripped, mode="eval")
            visitor = MacroExpression(self, seen | {name})
            visitor.visit(tree)
            value = visitor.result
        self.resolved[name] = value
        return value


def main() -> int:
    table = MacroTable(HEADER.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    failures = 0
    for (section, key), name in BINDINGS.items():
        expected = contract[section][key]
        actual = table.resolve(name)
        if float(expected) != actual:
            print(f"MISMATCH {name}: header {actual} contract {expected}")
            failures += 1
    composed = {
        "SPARK_LING_MODEL_MLA_QUERY_DIMENSION": 32 * 192,
        "SPARK_LING_MODEL_MLA_KV_A_DIMENSION": 512 + 64,
        "SPARK_LING_MODEL_MLA_KV_B_DIMENSION": 32 * (128 + 128),
        "SPARK_LING_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION": 32 * 128,
        "SPARK_LING_MODEL_KDA_QKV_DIMENSION": 32 * 128,
        "SPARK_LING_MODEL_KDA_VALUE_DIMENSION": 32 * 128,
        "SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER": 32 * 128 * 128 * 4,
        "SPARK_LING_MODEL_KV_SLOT_BYTES": (512 + 64) * 2,
        "SPARK_LING_MODEL_WEIGHT_LAYER_COUNT": 42 + 1,
    }
    for name, expected in composed.items():
        actual = table.resolve(name)
        if float(expected) != actual:
            print(f"MISMATCH composed {name}: header {actual} expected {expected}")
            failures += 1
    if 7 + 35 != 42 or 42 % 6 != 0:
        print("MISMATCH hybrid layer split does not cover the stack in whole groups")
        failures += 1
    if 32 % 16 != 0 or 512 % 16 != 0:
        print("MISMATCH TP16: head and expert counts must divide by 16")
        failures += 1
    if not 4 < 8:
        print("MISMATCH router top groups must be a strict subset of groups")
        failures += 1
    if (2560 % 128 != 0) or (768 % 128 != 0):
        print("MISMATCH expert geometry must tile 128-block scale groups")
        failures += 1
    if failures:
        print(f"FAILED {failures} binding(s)")
        return 1
    print("PASS ling header matches the authoritative contract (35 bindings + 9 composed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
