#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import hy4_fp8_scale_contract as oracle  # noqa: E402

MODULE = ROOT / "modules" / "hy4_resident_decode_stage"
KIND_CODE = {
    "self_attn.q_a_proj.weight": 5,
    "self_attn.q_b_proj.weight": 7,
    "self_attn.kv_a_proj_with_mqa.weight": 8,
    "self_attn.kv_b_proj.weight": 10,
    "self_attn.o_proj.weight": 13,
    "self_attn.indexer.wq_b.weight": 18,
    "self_attn.indexer.wk.weight": 19,
    "mlp.experts.gate_up_proj": 31,
    "mlp.experts.down_proj": 33,
    "mlp.shared_experts.gate_proj.weight": 34,
    "mlp.shared_experts.up_proj.weight": 35,
    "mlp.shared_experts.down_proj.weight": 36,
}
LOCAL_SHAPES = {
    "self_attn.q_a_proj.weight": (2048, 6144),
    "self_attn.q_b_proj.weight": (1024, 2048),
    "self_attn.kv_a_proj_with_mqa.weight": (576, 6144),
    "self_attn.kv_b_proj.weight": (1792, 512),
    "self_attn.o_proj.weight": (6144, 1024),
    "self_attn.indexer.wq_b.weight": (256, 2048),
    "self_attn.indexer.wk.weight": (128, 6144),
    "mlp.experts.gate_up_proj": (16 * 4096, 6144),
    "mlp.experts.down_proj": (16 * 6144, 2048),
    "mlp.shared_experts.gate_proj.weight": (2048, 6144),
    "mlp.shared_experts.up_proj.weight": (2048, 6144),
    "mlp.shared_experts.down_proj.weight": (6144, 2048),
}
RULE_CODE = {oracle.ALIGNED: 0, oracle.REPL_ROWS: 1, oracle.REPL_GROUPS: 2}

PROBE = r"""
#include <stdio.h>
#include <stdlib.h>
#include "spark_hy4_fp8_scale_contract.h"
int main(int argc, char **argv)
{
	SparkHy4Fp8ScaleContract c = {0};
	uint32_t kind = (uint32_t)atoi(argv[1]), rank = (uint32_t)atoi(argv[2]);
	uint32_t ranks = (uint32_t)atoi(argv[3]), rows = (uint32_t)atoi(argv[4]);
	uint32_t cols = (uint32_t)atoi(argv[5]);
	uint32_t srows = (uint32_t)atoi(argv[6]), sgroups = (uint32_t)atoi(argv[7]);
	int32_t status = SparkHy4Fp8ScaleContractValidate(kind,rank,ranks,rows,
	    cols,srows,sgroups,&c);
	if ( argc != 8 )
		return 2;
	printf("%d %u %u %u %u %u %llu %u\n",status,c.rule,c.scale_rows,
	    c.scale_groups,c.rank_row_offset,c.rank_group_offset,
	    (unsigned long long)SparkHy4Fp8ScaleByteIndex(&c,3u,5u),
	    SparkHy4Fp8ScaleZeroCopy(&c));
	return 0;
}
"""


def build_probe(workdir: Path) -> Path:
    source = workdir / "probe.c"
    source.write_text(PROBE)
    binary = workdir / "probe"
    includes = [ROOT / "include", ROOT / "src",
                ROOT / "model-families" / "common" / "include",
                ROOT / "model-families" / "hy4" / "include",
                MODULE / "include", MODULE / "source", ROOT]
    cmd = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"]
    for inc in includes:
        cmd += ["-I", str(inc)]
    cmd += [str(source), "-o", str(binary)]
    subprocess.run(cmd, check=True)
    return binary


def probe(binary, kind, rank, ranks, rows, cols, srows, sgroups):
    out = subprocess.run([str(binary), str(kind), str(rank), str(ranks),
                          str(rows), str(cols), str(srows), str(sgroups)],
                         check=True, capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def check_agreement(binary) -> int:
    checked = 0
    for ranks in (16, 8):
        for rank in (0, 2, ranks - 1):
            for kind, code in KIND_CODE.items():
                rows, cols = LOCAL_SHAPES[kind]
                if ranks == 8 and kind in (
                        "self_attn.q_b_proj.weight",
                        "self_attn.kv_b_proj.weight",
                        "self_attn.indexer.wq_b.weight"):
                    rows *= 2
                if ranks == 8 and kind == "self_attn.o_proj.weight":
                    cols *= 2
                if ranks == 8 and kind.startswith("mlp.experts."):
                    rows *= 2
                got = oracle.contract_of(kind, rank, ranks, rows, cols)
                assert not isinstance(got, str), (kind, ranks, got)
                rule, srows, sgroups, roff, goff = got
                c = probe(binary, code, rank, ranks, rows, cols, srows,
                          sgroups)
                want = [0, RULE_CODE[rule], srows, sgroups, roff, goff,
                        (roff + 3) * sgroups + goff + 5,
                        1 if sgroups == cols // 32 else 0]
                assert c == want, (kind, rank, ranks, c, want)
                checked += 1
    return checked


def check_fail_closed(binary) -> None:
    assert probe(binary, 12, 0, 16, 1024, 6144, 1024, 192)[0] == -1
    assert probe(binary, 7, 16, 16, 1024, 2048, 16384, 64)[0] == -2
    assert probe(binary, 5, 0, 16, 2048, 6150, 2048, 192)[0] == -3
    assert probe(binary, 7, 0, 16, 2048, 2048, 16384, 64)[0] == -6
    assert probe(binary, 13, 0, 16, 6144, 2048, 6144, 512)[0] == -7
    assert probe(binary, 13, 0, 16, 6144, 1024, 6144, 32)[0] == -4
    assert probe(binary, 7, 0, 16, 1024, 2048, 1024, 64)[0] == -4
    assert oracle.contract_of("self_attn.linear_gate.weight", 0, 16,
                              1024, 6144) == "no-rule"
    assert oracle.contract_of("self_attn.o_proj.weight", 0, 16,
                              6144, 2048) == "columns-tile"


def synthetic_header(bad: bool) -> dict:
    header = {"__metadata__": {"tp_degree": "16", "tp_rank": "2"}}
    for prefix in ("model.layers.4.", "model.mtp_layers.0."):
        for kind, (rows, cols) in LOCAL_SHAPES.items():
            got = oracle.contract_of(kind, 2, 16, rows, cols)
            _, srows, sgroups, _, _ = got
            shape = [rows, cols]
            sshape = [srows, sgroups]
            if kind.startswith("mlp.experts."):
                shape = [16, rows // 16, cols]
                sshape = [16, srows // 16, sgroups]
            header[prefix + kind] = {"dtype": "F8_E4M3", "shape": shape,
                                     "data_offsets": [0, 0]}
            header[prefix + kind + "_scale"] = {
                "dtype": "U8", "shape": sshape, "data_offsets": [0, 0]}
    if bad:
        header["model.layers.9.self_attn.o_proj.weight"] = {
            "dtype": "F8_E4M3", "shape": [6144, 1024], "data_offsets": [0, 0]}
        header["model.layers.9.self_attn.o_proj.weight_scale"] = {
            "dtype": "U8", "shape": [6144, 32], "data_offsets": [0, 0]}
    return header


def check_oracle_cli(workdir: Path) -> None:
    good = workdir / "good.json"
    good.write_text(json.dumps(synthetic_header(False)))
    run = subprocess.run([sys.executable,
                          str(ROOT / "tools" / "hy4_fp8_scale_contract.py"),
                          "--header", str(good)], capture_output=True,
                         text=True)
    assert run.returncode == 0, run.stderr
    assert "24 FP8 planes, 0 contract failures" in run.stdout, run.stdout
    bad = workdir / "bad.json"
    bad.write_text(json.dumps(synthetic_header(True)))
    run = subprocess.run([sys.executable,
                          str(ROOT / "tools" / "hy4_fp8_scale_contract.py"),
                          "--header", str(bad)], capture_output=True,
                         text=True)
    assert run.returncode == 1
    assert "model.layers.9.self_attn.o_proj.weight" in run.stderr, run.stderr
    assert "want 6144x512" in run.stderr, run.stderr


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        binary = build_probe(workdir)
        checked = check_agreement(binary)
        check_fail_closed(binary)
        check_oracle_cli(workdir)
    print("test_hy4_fp8_scale_contract: OK (%d C/python agreement cases)"
          % checked)
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
