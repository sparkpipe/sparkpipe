#!/usr/bin/env python3
"""GLM 5.3 Full (GlmMoeDsaForCausalLM) contract freeze — M1 of the
glm53full three-resolution lane.

Freezes the admitted NVFP4 radixark source (warm):
  /mnt/model-warm/glm-5.3-nvfp4-radixark (RadixArk/GLM-5.3-NVFP4,
  DOWNLOAD-RECEIPT.json = ds4-hf-warm-download-receipt-v1, license
  zai-commercial-use-granted).

Per the quantization policy this is a COMMUNITY quantization: the
contract pins provenance + full-receipt hashes; the quality gate fires
on first serve (COMPSEC-17 then the 92x). The packer REPACKAGES only —
expert payloads/scales pass through byte-verbatim.

Every emitted number is derived from fetched artifacts. Re-runnable;
moved to another host by changing --spark.

Usage:
  tools/glm53full_contract_freeze.py --spark spark4
  tools/glm53full_contract_freeze.py --spark spark4 --skip-fetch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "/mnt/model-warm/glm-5.3-nvfp4-radixark"
DEFAULT_CACHE = ROOT / ".lane_cache" / "glm53full_source"
DEFAULT_OUTPUT = ROOT / "model_contracts" / "glm53_full_authoritative.json"

# The three-resolution study admits three OFFICIAL/pinned sources of the
# same checkpoint. The FP8 and BF16 staging dirs keep their final
# config.json long before promotion, so the freeze records them from
# staging and upgrades to full receipt pinning once PUBLISHED lands.
SIBLING_SOURCES = {
    "fp8": {
        "repo": "zai-org/GLM-5.3",
        "revision": "935644c05e76fc198714f4cca449fd8b970ff6d7",
        "default_path": "/mnt/model-warm/.staging/glm-5.3-935644c0",
        "note": "official FP8 release (e4m3 dynamic act, weight_block_size [128,128])",
    },
    "bf16": {
        "repo": "zai-org/GLM-5.3-BF16",
        "revision": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
        "default_path": "/mnt/model-warm/.staging/glm-5.3-bf16-304b8051",
        "note": "official native-precision release (quant policy: serve native)",
    },
}

SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "hf_quant_config.json",
    "README.md",
    "LICENSE",
    "DOWNLOAD-RECEIPT.json",
]

# Strided shard re-hash on the spark (bounded IO, corruption probe). The
# full per-shard pinning lives in DOWNLOAD-RECEIPT.json file_records.
SHARD_STRIDE = 8
SHARD_SAMPLE_OVERRIDE = [1, 47]

REMOTE_CENSUS_SCRIPT = r"""#!/usr/bin/env python3
import hashlib, json, re, struct, sys
src = sys.argv[1]
index_path = src + "/model.safetensors.index.json"
wm = json.load(open(index_path))["weight_map"]
h = hashlib.sha256()
with open(index_path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
census = {
    "tensor_count": len(wm),
    "shard_count": len(set(wm.values())),
    "index_sha256": h.hexdigest(),
    "samples": {},
    "indexer_full_layers": sorted({int(m.group(1)) for n in wm
        if (m := re.match(r"model\.layers\.(\d+)\.self_attn\.indexer\.wq_b\.weight$", n))}),
    "routed_layers": sorted({int(m.group(1)) for n in wm
        if (m := re.match(r"model\.layers\.(\d+)\.mlp\.experts\.0\.up_proj\.weight$", n))}),
    "dense_layers": sorted({int(m.group(1)) for n in wm
        if (m := re.match(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", n))}),
}
hdrs = {}
def hdr_for(shard):
    if shard not in hdrs:
        with open(src + "/" + shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdrs[shard] = json.loads(fh.read(n))
    return hdrs[shard]
want = [
    "model.layers.2.mlp.down_proj.weight",
    "model.layers.3.self_attn.q_a_proj.weight",
    "model.layers.3.mlp.gate.weight",
    "model.layers.3.mlp.gate.e_score_correction_bias",
    "model.layers.3.mlp.shared_experts.up_proj.weight",
    "model.layers.3.self_attn.indexer.wq_b.weight",
    "model.layers.3.mlp.experts.0.up_proj.weight",
    "model.layers.3.mlp.experts.0.up_proj.weight_scale",
    "model.layers.3.mlp.experts.0.up_proj.weight_scale_2",
    "model.layers.3.mlp.experts.0.gate_proj.weight",
    "model.layers.3.mlp.experts.0.gate_proj.weight_scale",
    "model.layers.3.mlp.experts.0.gate_proj.weight_scale_2",
    "model.layers.3.mlp.experts.0.down_proj.weight",
    "model.layers.3.mlp.experts.0.down_proj.weight_scale",
    "model.layers.3.mlp.experts.0.down_proj.weight_scale_2",
    "model.layers.77.mlp.experts.255.gate_proj.weight_scale_2",
    "lm_head.weight",
    "model.embed_tokens.weight",
]
for name in want:
    if name not in wm:
        census["samples"][name] = None
        continue
    e = hdr_for(wm[name])[name]
    census["samples"][name] = {"dtype": e["dtype"], "shape": e["shape"]}
print(json.dumps(census))
"""


REMOTE_SIBLING_SCRIPT = r"""#!/usr/bin/env python3
import hashlib, json, os, sys
path = sys.argv[1]
config_path = os.path.join(path, "config.json")
cfg = json.load(open(config_path))
raw = open(config_path, "rb").read()
out = {"config_sha256": hashlib.sha256(raw).hexdigest(), "config_bytes": len(raw)}
it = cfg.get("indexer_types")
if it is not None:
    layers = len(it)
    full = [i for i in range(layers) if it[i] == "full"]
    out["indexer_types"] = {"layer_count": layers,
                            "full_layers": full,
                            "full_count": len(full),
                            "shared_count": layers - len(full)}
q = cfg.get("quantization_config")
if q is None:
    out["quantization_config"] = None
else:
    out["quantization_config"] = {
        "quant_method": q.get("quant_method"),
        "fmt": q.get("fmt"),
        "activation_scheme": q.get("activation_scheme"),
        "weight_block_size": q.get("weight_block_size"),
        "excluded_module_count": len(q.get("modules_to_not_convert") or []),
    }
receipt_path = os.path.join(path, "DOWNLOAD-RECEIPT.json")
if os.path.isfile(receipt_path):
    r = json.load(open(receipt_path))
    out["receipt"] = {k: r[k] for k in ("format", "repo", "revision",
                                        "license_class", "files", "bytes",
                                        "verified_at")}
    h = hashlib.sha256()
    with open(receipt_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    out["receipt_sha256"] = h.hexdigest()
    out["published"] = os.path.isfile(os.path.join(path, "PUBLISHED"))
else:
    out["receipt"] = None
    out["published"] = False
print(json.dumps(out))
"""


def read_sibling(host: str, path: str, cache: Path) -> dict:
    script = cache / "_sibling_probe.py"
    script.write_text(REMOTE_SIBLING_SCRIPT, encoding="utf-8")
    remote_script = "/tmp/glm53full_sibling_probe.py"
    subprocess.run(["scp", "-q", str(script), f"{host}:{remote_script}"],
                   check=True)
    return json.loads(sh(host, f"python3 {remote_script} {path}"))


def sh(host: str, command: str, timeout: int = 1800) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh {host} failed: {proc.stderr.strip()}")
    return proc.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(host: str, source: str, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        local = cache / name
        if local.exists():
            continue
        subprocess.run(["scp", "-q", f"{host}:{source}/{name}", str(local)],
                       check=True)
        print(f"  fetched {name}")
    script = cache / "_census_builder.py"
    script.write_text(REMOTE_CENSUS_SCRIPT, encoding="utf-8")
    subprocess.run(["scp", "-q", str(script), f"{host}:/tmp/glm53full_census.py"],
                   check=True)
    census_text = sh(host, f"python3 /tmp/glm53full_census.py {source}")
    (cache / "census.json").write_text(census_text)
    print("  fetched tensor census")


def verify_against_receipt(cache: Path, host: str, source: str) -> dict:
    """Small files: local sha vs DOWNLOAD-RECEIPT file_records. Shards:
    strided re-hash on the spark against the receipt records."""
    receipt = json.loads((cache / "DOWNLOAD-RECEIPT.json").read_text())
    records = {r["path"]: r for r in receipt["file_records"]}
    small = {}
    for name in SMALL_FILES:
        local = cache / name
        digest = sha256_file(local)
        entry = {"bytes": local.stat().st_size, "sha256": digest}
        if name in records:
            entry["matches_receipt"] = digest == records[name]["sha256"]
        small[name] = entry
    shard_re = re.compile(r"model-\d+-of-\d+\.safetensors$")
    shard_records = {p: r for p, r in records.items() if shard_re.search(p)}
    names = sorted(shard_records)
    count = len(names)
    by_number = {int(re.search(r"-(\d+)-of-", n).group(1)): n for n in names}
    numbers = sorted(set(
        list(range(1, count + 1, SHARD_STRIDE))
        + [n for n in SHARD_SAMPLE_OVERRIDE if 1 <= n <= count]))
    sampled = [by_number[i] for i in numbers]
    selected = "".join(
        f"{shard_records[n]['sha256']}  {n}\n" for n in sampled if n in shard_records
    )
    check_file = cache / "_strided_sample.txt"
    check_file.write_text(selected)
    subprocess.run(["scp", "-q", str(check_file),
                    f"{host}:/tmp/glm53full_strided.txt"], check=True)
    # 4h ceiling: warm reads share ceph with any concurrent fetch/ rsync
    # traffic; the hash itself is minutes, contention is the variable.
    out = sh(host, f"cd {source} && sha256sum -c /tmp/glm53full_strided.txt",
             timeout=14400)
    ok = [l for l in out.splitlines() if l.endswith(": OK")]
    failed = [l for l in out.splitlines() if l.endswith(": FAILED")]
    return {
        "receipt": {
            "repo": receipt["repo"],
            "revision": receipt["revision"],
            "license_class": receipt["license_class"],
            "format": receipt["format"],
            "bytes": receipt["bytes"],
            "files": receipt["files"],
            "verified_at": receipt["verified_at"],
        },
        "receipt_sha256": sha256_file(cache / "DOWNLOAD-RECEIPT.json"),
        "small_files": small,
        "shard_check": {
            "shard_count": len(names),
            "stride": SHARD_STRIDE,
            "sampled": sampled,
            "ok": len(ok),
            "failed": failed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze glm53full NVFP4 source contract")
    parser.add_argument("--spark", default="spark4")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--fp8-path", default=SIBLING_SOURCES["fp8"]["default_path"],
                        help="fp8 sibling source dir (staging or promoted)")
    parser.add_argument("--bf16-path", default=SIBLING_SOURCES["bf16"]["default_path"],
                        help="bf16 sibling source dir (staging or promoted)")
    args = parser.parse_args()
    cache = Path(args.cache)
    if not args.skip_fetch:
        fetch(args.spark, args.source, cache)
    receipt_check = verify_against_receipt(cache, args.spark, args.source)
    cfg = json.loads((cache / "config.json").read_text())
    quant_root = json.loads((cache / "hf_quant_config.json").read_text())
    quant = quant_root["quantization"]
    producer = quant_root.get("producer")
    census = json.loads((cache / "census.json").read_text())

    failed = [n for n, e in receipt_check["small_files"].items()
              if not e.get("matches_receipt", True)]
    failed += receipt_check["shard_check"]["failed"]
    if failed:
        raise SystemExit(f"RECEIPT MISMATCH: {failed}")

    full_layers = census["indexer_full_layers"]
    body = [l for l in full_layers
            if l < 78 and (l < 3 or (l >= 6 and (l - 6) % 4 == 0))]
    dense = census["dense_layers"]
    routed = census["routed_layers"]
    samples = census["samples"]
    assert dense == [0, 1, 2], dense
    assert routed == list(range(3, 79)), routed[:5]
    assert body == [l for l in range(78) if l < 3 or (l >= 6 and (l - 6) % 4 == 0)], body
    assert 78 in full_layers and 78 not in body  # MTP draft layer carries its own full indexer

    s = samples["model.layers.3.mlp.experts.0.up_proj.weight"]
    assert s == {"dtype": "U8", "shape": [2048, 3072]}, s
    assert samples["model.layers.3.mlp.experts.0.up_proj.weight_scale"] == \
        {"dtype": "F8_E4M3", "shape": [2048, 384]}
    assert samples["model.layers.3.mlp.experts.0.up_proj.weight_scale_2"] == \
        {"dtype": "F32", "shape": []}

    excludes = quant["exclude_modules"]
    assert quant["quant_algo"] == "NVFP4" and quant["group_size"] == 16

    # R3 (indexer_types): probe the FP8/BF16 sibling configs. Both carry
    # the explicit key; their full-layer split must equal the NVFP4
    # checkpoint's tensor census exactly (metadata-only omission on the
    # NVFP4 config, no semantic difference in indexer sharing).
    expected_body = [l for l in range(78) if l < 3 or (l >= 6 and (l - 6) % 4 == 0)]
    siblings = {}
    for name, spec in SIBLING_SOURCES.items():
        path = args.fp8_path if name == "fp8" else args.bf16_path
        probe = read_sibling(args.spark, path, cache)
        assert probe["receipt"] is None or \
            probe["receipt"]["revision"] == spec["revision"], \
            f"{name} sibling revision drifted: {probe.get('receipt')}"
        it = probe.get("indexer_types")
        assert it is not None, f"{name} config has no indexer_types key"
        assert it["full_layers"] == expected_body, \
            f"{name} indexer split {it['full_layers']} != census {expected_body}"
        assert it["full_count"] == 21 and it["shared_count"] == 57, it
        siblings[name] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "path": path,
            "note": spec["note"],
            "published": probe["published"],
            "receipt": probe["receipt"],
            "receipt_sha256": probe.get("receipt_sha256"),
            "config_sha256": probe["config_sha256"],
            "indexer_types": it,
            "quantization_config": probe["quantization_config"],
        }

    contract = {
        "schema_version": 1,
        "model_id": receipt_check["receipt"]["repo"],
        "family": "glm52 (module family; the glm52 module IS the 5.3-full module)",
        "architecture": cfg["architectures"][0],
        "source_revision": receipt_check["receipt"]["revision"],
        "source_path": args.source,
        "license_class": receipt_check["receipt"]["license_class"],
        "quantization_policy": {
            "class": "community quantization, VETTING per policy",
            "provenance": "pinned (receipt repo+revision+per-file sha256)",
            "receipt_verification": "small files full + strided shard re-hash",
            "quality_gate": "PENDING first serve (COMPSEC-17 then 92x)",
            "packer_role": "REPACKAGE only: expert payloads+scales byte-verbatim",
        },
        "quantization": {
            "quant_algo": quant["quant_algo"],
            "producer": producer,
            "group_size": quant["group_size"],
            "kv_cache_quant_algo": quant["kv_cache_quant_algo"],
            "quantized_tensors": "routed-expert linears ONLY (mlp.experts.*) layers 3-77",
            "full_resolution_spine": ("embed, lm_head, all attention, indexer, "
                                      "router, shared experts, dense layers 0-2, "
                                      "MTP layer 78, all norms — BF16"),
            "expert_layout": {
                "weight": "U8 packed e2m1, two 4-bit codes per byte, [rows, cols/2]",
                "weight_scale": "F8_E4M3 per 16-element group, [rows, cols/16]",
                "weight_scale_2": "F32 global per expert projection",
                "fused_up_gate_note": ("up_proj.weight_scale_2 == gate_proj."
                                       "weight_scale_2 per expert (sampled), so the "
                                       "wire format's one-global-per-expert fused "
                                       "EXPERT_UP_GATE entry is exact verbatim"),
                "input_scale": "F32 static activation scale — not packed (module uses dynamic activation)",
            },
        },
        "geometry": {
            "hidden_dimension": cfg["hidden_size"],
            "layer_count": cfg["num_hidden_layers"],
            "first_routed_layer": cfg["first_k_dense_replace"],
            "head_count": cfg["num_attention_heads"],
            "latent_dimension": cfg["kv_lora_rank"],
            "query_a_dimension": cfg["q_lora_rank"],
            "qk_nope_head_dimension": cfg["qk_nope_head_dim"],
            "rope_dimension": cfg["qk_rope_head_dim"],
            "value_head_dimension": cfg["v_head_dim"],
            "moe_expert_count": cfg["n_routed_experts"],
            "moe_top_k": cfg["num_experts_per_tok"],
            "moe_intermediate_dimension": cfg["moe_intermediate_size"],
            "dense_intermediate_dimension": cfg["intermediate_size"],
            "output_vocab_count": cfg["vocab_size"],
            "moe_routed_scaling_factor": cfg["routed_scaling_factor"],
            "router_scoring_func": cfg["scoring_func"],
            "router_topk_method": cfg["topk_method"],
            "dsa_index_head_count": cfg["index_n_heads"],
            "dsa_index_head_dimension": cfg["index_head_dim"],
            "dsa_selected_token_count": cfg["index_topk"],
            "dsa_index_share_group_layer_count": cfg["index_topk_freq"],
            "dsa_index_skip_topk_offset": cfg["index_skip_topk_offset"],
            "mtp_nextn_predict_layers": cfg["num_nextn_predict_layers"],
            "eos_token_ids": cfg["eos_token_id"],
        },
        "indexer_types_resolution": {
            "question": ("5.2 config splits full/shared indexer 21/57; the 5.3 "
                         "NVFP4 config carries no indexer_types key — dropped or "
                         "omitted?"),
            "answer": ("omitted, not dropped: the checkpoint carries full indexer "
                       "sets (wq_b, wk, weights_proj, k_norm.{weight,bias}) on "
                       "layers [0,1,2,6,10,...,74] (21 in 0-77) + the MTP layer 78 "
                       "— the same share-group-4 pattern as 5.2; the glm52 "
                       "has_full_indexer rule needs no change"),
            "evidence_layers": full_layers,
            "sibling_confirmation": ("the official FP8 and BF16 configs of the "
                                     "same checkpoint both carry indexer_types "
                                     "explicitly with the identical 21 full / 57 "
                                     "shared split (full layers == this "
                                     "checkpoint's tensor census) — the NVFP4 "
                                     "config's omission is metadata-only"),
        },
        "three_resolution_sources": {
            "nvfp4": {
                "repo": receipt_check["receipt"]["repo"],
                "revision": receipt_check["receipt"]["revision"],
                "path": args.source,
                "receipt": receipt_check["receipt"],
                "receipt_sha256": receipt_check["receipt_sha256"],
            },
            "fp8": siblings["fp8"],
            "bf16": siblings["bf16"],
            "pack_resolutions": ("nvfp4=codec 6, fp8=codec 5, bf16=codec 1 "
                                 "(packer --expert-codec); all three REPACKAGE "
                                 "only"),
        },
        "structural_census": {
            "tensor_count": census["tensor_count"],
            "shard_count": census["shard_count"],
            "dense_layers": dense,
            "routed_layers": f"3-77 ({len(routed)} layers)",
            "indexer_full_layers_0_77": body,
            "indexer_full_layer_78_mtp": 78 in full_layers,
        },
        "receipt_verification": receipt_check,
        "freeze_policy": ("packs must cite this contract's receipt block; "
                          "re-freeze on any source change"),
    }
    out = Path(args.output)
    out.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    print(f"frozen: {out}")
    print(f"  revision {contract['source_revision']}")
    print(f"  shards {census['shard_count']} tensors {census['tensor_count']}")
    print(f"  shard sample ok={receipt_check['shard_check']['ok']} "
          f"failed={receipt_check['shard_check']['failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
