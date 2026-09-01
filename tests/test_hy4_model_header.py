#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(
        ["python3", str(ROOT / "tools" / "generate_hy4_contracts.py"), "--check"],
        check=True,
        cwd=ROOT,
    )
    contract = json.loads(
        (ROOT / "model_contracts" / "hy4.json").read_text(encoding="utf-8"))
    assert contract["model"]["layer_count"] == 78
    assert contract["model"]["mtp_layer_count"] == 0
    assert contract["model"]["vocabulary_size"] == 120832
    assert contract["moe"]["routed_expert_count"] == 256
    assert contract["moe"]["experts_per_token"] == 8
    assert contract["moe"]["score_function"] == "elementwise"
    assert contract["hyper_connections"]["stream_count"] == 4
    assert contract["attention"]["index_top_k"] == 2048
    assert contract["source_index_sha256"] == (
        "12d325844103bac75bd286d14e0e45f87e35e8e60401877282a30b6f26ba6ac6")
    assert contract["runtime"]["speculative_decoding"] == "deferred"

    header = (ROOT / "model-families" / "hy4" / "include" / "sparkpipe" /
              "spark_hy4_model.h").read_text(encoding="utf-8")
    expected_defines = {
        "SPARK_HY4_MODEL_HIDDEN_DIMENSION": "6144u",
        "SPARK_HY4_MODEL_LAYER_COUNT": "78u",
        "SPARK_HY4_MODEL_VOCAB_PER_RANK": "7552u",
        "SPARK_HY4_MODEL_EXPERTS_PER_RANK": "16u",
        "SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK": "4u",
        "SPARK_HY4_MODEL_INDEX_HEADS_PER_RANK": "2u",
        "SPARK_HY4_MODEL_INDEX_TOP_K": "2048u",
        "SPARK_HY4_MODEL_HC_STREAM_COUNT": "4u",
    }
    for name, value in expected_defines.items():
        pattern = re.compile(rf"^#define {re.escape(name)} {re.escape(value)}$",
                             re.M)
        assert pattern.search(header), f"missing define {name} = {value}"
    assert "#define SPARK_HY4_MODEL_IS_INDEXER_FULL_LAYER(layer)" in header
    assert "12d325844103bac75bd286d14e0e45f87e35e8e60401877282a30b6f26ba6ac6" in header

    authoritative = json.loads(
        (ROOT / "model_contracts" / "hy4_authoritative.json").read_text(
            encoding="utf-8"))
    ranks = authoritative["shards"]["ranks"]
    assert len(ranks) == 16
    assert len(set(ranks.values())) == 16
    print("test_hy4_model_header: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
