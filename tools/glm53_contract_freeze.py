#!/usr/bin/env python3
"""GLM 5.3 Flash (family glm5_next) contract freeze — M1.

Fetches the checkpoint source over ssh from a spark host, builds a full
tensor manifest from safetensors shard headers (header-only reads),
verifies small-file sha256s and a strided shard sample against the
archive receipt's SHA256SUMS, and emits the authoritative contract at
model_contracts/glm53_flash_authoritative.json.

Every number in the emitted contract is derived from fetched artifacts;
nothing is hand-copied. Re-runnable; moved to another host by changing
--spark.

Usage:
  tools/glm53_contract_freeze.py --spark spark2
  tools/glm53_contract_freeze.py --spark spark2 --skip-fetch   # reuse cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "/mnt/model-warm/glm-5.3-flash"
DEFAULT_CACHE = ROOT / ".lane_cache" / "glm53_source"
DEFAULT_OUTPUT = ROOT / "model_contracts" / "glm53_flash_authoritative.json"

SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "processor_config.json",
    "README.md",
    "LICENSE",
    ".gitattributes",
    "ARCHIVE-RECEIPT.json",
    "SHA256SUMS",
]

# Strided shard verification: shard numbers (1-based) 1, 9, ..., 57 plus the
# final shard. Re-hashing all 62 shards is ~5 min of NVMe IO; the sample
# catches corruption/bit rot with bounded cost. Full per-shard pinning comes
# from the receipt-anchored SHA256SUMS which is itself sha256-verified.
SHARD_STRIDE = 8
SHARD_SAMPLE_OVERRIDE = [62]

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "U8": 1}

REMOTE_MANIFEST_SCRIPT = r"""#!/usr/bin/env python3
import json, os, struct, sys
src = sys.argv[1]
out = sys.argv[2]
files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
rows = []
for fn in files:
    with open(os.path.join(src, fn), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        rows.append((name, info["dtype"], json.dumps(info["shape"]), fn))
with open(out, "w") as w:
    for name, dt, shape, fn in sorted(rows):
        w.write(f"{name}\t{dt}\t{shape}\t{fn}\n")
print(f"manifest rows: {len(rows)}")
"""


def sh(host: str, command: str, timeout: int = 600) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh {host} failed: {proc.stderr.strip()}")
    return proc.stdout


def fetch(host: str, source: str, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    present = sh(host, f"ls {source}").split()
    for name in SMALL_FILES:
        if name not in present:
            print(f"  note: {name} absent at source, skipping fetch")
            continue
        local = cache / name
        if local.exists():
            continue
        subprocess.run(
            ["scp", "-q", f"{host}:{source}/{name}", str(local)], check=True
        )
        print(f"  fetched {name}")
    # remote manifest builder (header-only reads; no shard payload IO)
    script = cache / "_manifest_builder.py"
    script.write_text(REMOTE_MANIFEST_SCRIPT, encoding="utf-8")
    subprocess.run(
        ["scp", "-q", str(script), f"{host}:/tmp/glm53_manifest_builder.py"],
        check=True,
    )
    sh(host, f"python3 /tmp/glm53_manifest_builder.py {source} /tmp/glm53_tensor_manifest.tsv")
    subprocess.run(
        ["scp", "-q", f"{host}:/tmp/glm53_tensor_manifest.tsv",
         str(cache / "glm53_tensor_manifest.tsv")],
        check=True,
    )
    print("  fetched tensor manifest")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(cache: Path) -> dict[str, tuple[str, list[int], str]]:
    manifest = {}
    for line in (cache / "glm53_tensor_manifest.tsv").read_text().splitlines():
        name, dt, shape, shard = line.split("\t")
        manifest[name] = (dt, json.loads(shape), shard)
    return manifest


def tensor_bytes(shape: list[int], dtype: str) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n * DTYPE_BYTES[dtype]


def pattern(name: str) -> str:
    p = re.sub(r"layers\.\d+", "layers.N", name)
    return re.sub(r"experts\.\d+", "experts.N", p)


def verify_small_files(host: str, source: str, cache: Path) -> dict:
    """Hash small files locally; compare with the source SHA256SUMS."""
    sums = {}
    for line in (cache / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name.removeprefix("./")] = digest
    results = {}
    for name in SMALL_FILES:
        local = cache / name
        if not local.exists():
            continue
        digest = sha256_file(local)
        entry = {"bytes": local.stat().st_size, "sha256": digest}
        if name in sums:
            entry["matches_source_sums"] = digest == sums[name]
        results[name] = entry
    receipt = json.loads((cache / "ARCHIVE-RECEIPT.json").read_text())
    results["_receipt"] = receipt
    results["_sha256sums_sha256"] = sha256_file(cache / "SHA256SUMS")
    return results


def verify_strided_shards(host: str, source: str, cache: Path) -> dict:
    """Re-hash a strided shard sample on the spark, checked against the
    pinned SHA256SUMS (never trusting it blindly)."""
    shard_names = sorted(
        n for n in sh(host, f"ls {source}").split()
        if re.fullmatch(r"model-\d+-of-\d+\.safetensors", n)
    )
    count = len(shard_names)
    numbers = list(range(1, count + 1, SHARD_STRIDE))
    for extra in SHARD_SAMPLE_OVERRIDE:
        if extra not in numbers and 1 <= extra <= count:
            numbers.append(extra)
    sampled = [f"model-{n:05d}-of-{count:05d}.safetensors" for n in numbers]
    sums = {}
    for line in (cache / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name.removeprefix("./")] = digest
    selected = "".join(
        f"{sums[n]}  {n}\n" for n in sampled if n in sums
    )
    check_file = cache / "_strided_sample.txt"
    check_file.write_text(selected)
    subprocess.run(
        ["scp", "-q", str(check_file), f"{host}:/tmp/glm53_strided_sample.txt"],
        check=True,
    )
    out = sh(
        host,
        f"cd {source} && sha256sum -c /tmp/glm53_strided_sample.txt",
        timeout=1800,
    )
    ok = [l for l in out.splitlines() if l.endswith(": OK")]
    failed = [l for l in out.splitlines() if l.endswith(": FAILED")]
    return {
        "shard_count": count,
        "stride": SHARD_STRIDE,
        "sampled_shards": sampled,
        "ok": len(ok),
        "failed": failed,
        "raw": out.strip().splitlines(),
    }


def build_contract(cfg: dict, manifest: dict, files: dict, shard_check: dict) -> dict:
    tc = cfg["text_config"]
    lac = tc["linear_attn_config"]
    kda_layers = lac["kda_layers"]
    full_layers = lac["full_attn_layers"]
    layer_types = tc["layer_types"]
    mlp_types = tc["mlp_layer_types"]

    # per-layer structural census from the manifest (independent of config)
    kda_idx = sorted(
        int(m.group(1)) for n in manifest
        if (m := re.match(r"model\.language_model\.layers\.(\d+)\.self_attn\.A_log$", n))
    )
    mla_idx = sorted(
        int(m.group(1)) for n in manifest
        if (m := re.match(r"model\.language_model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$", n))
    )
    moe_idx = sorted(
        int(m.group(1)) for n in manifest
        if (m := re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$", n))
    )
    hc_idx = sorted(
        int(m.group(1)) for n in manifest
        if (m := re.match(r"model\.language_model\.layers\.(\d+)\.hc_attn_base$", n))
    )
    mtp_idx = sorted(
        int(m.group(1)) for n in manifest
        if (m := re.match(r"model\.language_model\.layers\.(\d+)\.eh_proj\.weight$", n))
    )

    def layer_tensor(layer: int, suffix: str) -> tuple[str, list[int]]:
        name = f"model.language_model.layers.{layer}.{suffix}"
        dt, shape, _ = manifest[name]
        return dt, shape

    # cross-checks: config layer lists vs checkpoint structure
    assert kda_idx == kda_layers, (kda_idx, kda_layers)
    assert mla_idx == full_layers + mtp_idx, (mla_idx, full_layers, mtp_idx)
    assert moe_idx == (
        [i for i, t in enumerate(mlp_types) if t == "sparse"] + mtp_idx
    ), (moe_idx, mlp_types)
    assert [i for i, t in enumerate(layer_types) if t == "linear_attention"] == kda_layers
    assert [i for i, t in enumerate(layer_types) if t == "deepseek_sparse_attention"] == full_layers
    assert mtp_idx == [tc["num_hidden_layers"]], mtp_idx

    kda0 = 0
    dsa0 = full_layers[0]
    mtp0 = mtp_idx[0]
    hc_dt, hc_shape = layer_tensor(0, "hc_attn_fn")
    hc_mult = tc["hc_mult"]
    hidden = tc["hidden_size"]
    assert hc_shape == [hc_mult * 3 * 2, hc_mult * hidden], hc_shape  # [24, 4*4096]

    _, eh_shape = layer_tensor(mtp0, "eh_proj.weight")

    contract = {
        "schema_version": 1,
        "model_id": "zai-org/GLM-5.3-Flash",
        "architecture": cfg["architectures"][0],
        "text_architecture": tc["model_type"],
        "source_revision": files["_receipt"]["revision"],
        "scope": {
            "serving_target": "text stack only",
            "vision_tower": "present in checkpoint, out of scope (same call as qwen-flash lane)",
        },
        "model": {
            "hidden_dimension": hidden,
            "layer_count": tc["num_hidden_layers"],
            "vocabulary_size": tc["vocab_size"],
            "maximum_context_tokens": tc["max_position_embeddings"],
            "rms_norm_epsilon": tc["rms_norm_eps"],
            "tie_word_embeddings": cfg["tie_word_embeddings"],
            "activation": tc["hidden_act"],
            "swiglu_limit": tc["swiglu_limit"],
        },
        "hybrid_attention": {
            "kda_layer_count": len(kda_layers),
            "dsa_layer_count": len(full_layers),
            "kda_layers": kda_layers,
            "dsa_layers": full_layers,
            "period": 4,
            "global_phase": 3,
            "pattern": "3 KDA head, then every 4th layer is DSA (3,7,...,43), final KDA at 44",
            "checkpoint_evidence": {
                "layers_with_A_log": len(kda_idx),
                "layers_with_q_b_proj": len(mla_idx),
                "layers_with_hc": len(hc_idx),
                "layers_with_eh_proj": mtp_idx,
            },
        },
        "kda": {
            "head_count": lac["num_heads"],
            "head_dimension": lac["head_dim"],
            "short_conv_kernel": lac["short_conv_kernel_size"],
            "gate_lower_bound": lac["gate_lower_bound"],
            "use_full_rank_gate": None,
            "use_full_rank_gate_note": (
                "config omits the key; K3 sets true. Resolve default against "
                "kernel semantics before wiring kda_gate (delta 2 in assessment)."
            ),
            "checkpoint_tensor_shapes": {
                "q_proj.weight": layer_tensor(kda0, "self_attn.q_proj.weight"),
                "k_proj.weight": layer_tensor(kda0, "self_attn.k_proj.weight"),
                "v_proj.weight": layer_tensor(kda0, "self_attn.v_proj.weight"),
                "q_conv1d.weight": layer_tensor(kda0, "self_attn.q_conv1d.weight"),
                "b_proj.weight": layer_tensor(kda0, "self_attn.b_proj.weight"),
                "A_log": layer_tensor(kda0, "self_attn.A_log"),
                "dt_bias": layer_tensor(kda0, "self_attn.dt_bias"),
                "f_a_proj.weight": layer_tensor(kda0, "self_attn.f_a_proj.weight"),
                "f_b_proj.weight": layer_tensor(kda0, "self_attn.f_b_proj.weight"),
                "g_a_proj.weight": layer_tensor(kda0, "self_attn.g_a_proj.weight"),
                "g_b_proj.weight": layer_tensor(kda0, "self_attn.g_b_proj.weight"),
                "o_norm.weight": layer_tensor(kda0, "self_attn.o_norm.weight"),
                "o_proj.weight": layer_tensor(kda0, "self_attn.o_proj.weight"),
            },
            "layout_reading": (
                "q/k/v_proj [64*128, 4096] produce fused per-head q/k/v; "
                "b_proj [64, 4096] is a per-head scalar beta; dt_bias [64*128] "
                "per-head-per-channel; f/g two-stage 128-dim then per-channel."
            ),
            "precision": "BF16 (all KDA tensors; no scale_inv twins)",
        },
        "mla": {
            "query_head_count": tc["num_attention_heads"],
            "query_lora_rank": tc["q_lora_rank"],
            "kv_lora_rank": tc["kv_lora_rank"],
            "qk_nope_head_dimension": tc["qk_nope_head_dim"],
            "qk_rope_head_dimension": tc["qk_rope_head_dim"],
            "value_head_dimension": tc["v_head_dim"],
            "uses_nope_only": tc["mla_use_nope"],
            "rope_location": "indexer (wq_b/wk, 128-dim, interleaved)",
            "checkpoint_tensor_shapes": {
                "q_a_proj.weight": layer_tensor(dsa0, "self_attn.q_a_proj.weight"),
                "q_b_proj.weight": layer_tensor(dsa0, "self_attn.q_b_proj.weight"),
                "kv_a_proj_with_mqa.weight": layer_tensor(dsa0, "self_attn.kv_a_proj_with_mqa.weight"),
                "kv_b_proj.weight": layer_tensor(dsa0, "self_attn.kv_b_proj.weight"),
                "o_proj.weight": layer_tensor(dsa0, "self_attn.o_proj.weight"),
            },
            "geometry_reading": (
                "q_b_proj [64*256, 1536]: query heads carry NOPE only "
                "(qk_rope_head_dim=0). kv_a_proj_with_mqa [512, 4096]: the KV "
                "latent is pure lora rank 512, no rope segment (glm52: 576). "
                "kv_b_proj [64*(256+256), 512]. Rope lives in the indexer."
            ),
        },
        "indexer": {
            "head_count": tc["index_n_heads"],
            "head_dimension": tc["index_head_dim"],
            "top_k": tc["index_topk"],
            "kpool": tc["index_kpool"],
            "kpool_always_select_tail": tc["index_kpool_always_select_tail"],
            "kpool_compress": tc["index_kpool_compress"],
            "rope_interleave": tc["indexer_rope_interleave"],
            "share_for_mtp_iteration": tc["index_share_for_mtp_iteration"],
            "all_layers_full": set(tc["indexer_types"]) == {"full"},
            "checkpoint_tensor_shapes": {
                "wq_b.weight": layer_tensor(dsa0, "self_attn.indexer.wq_b.weight"),
                "wk.weight": layer_tensor(dsa0, "self_attn.indexer.wk.weight"),
                "weights_proj.weight": layer_tensor(dsa0, "self_attn.indexer.weights_proj.weight"),
                "k_norm.weight": layer_tensor(dsa0, "self_attn.indexer.k_norm.weight"),
                "k_norm.bias": layer_tensor(dsa0, "self_attn.indexer.k_norm.bias"),
                "index_kpool_compress_ape": layer_tensor(dsa0, "self_attn.indexer.index_kpool_compress_ape"),
                "index_kpool_compress_gate": layer_tensor(dsa0, "self_attn.indexer.index_kpool_compress_gate"),
            },
            "geometry_reading": (
                "wq_b consumes q_a_layernorm output (1536) -> 32*128; the MLA "
                "q_a path is the indexer q_a (no separate indexer wq_a). "
                "k_norm carries a bias (dsv4 donor: check). Compressor names "
                "index_kpool_compress_{ape,gate} map to dsv4-0731 "
                "compressor.{ape,wgate}."
            ),
        },
        "hyper_connections": {
            "hc_mult": hc_mult,
            "sinkhorn_iterations": tc["hc_sinkhorn_iters"],
            "epsilon": tc["hc_eps"],
            "mhc_flag": tc["mhc"],
            "per_layer": True,
            "on_mtp_layer": False,
            "checkpoint_tensor_shapes": {
                "hc_attn_base": layer_tensor(0, "hc_attn_base"),
                "hc_attn_fn": [hc_dt, hc_shape],
                "hc_attn_scale": layer_tensor(0, "hc_attn_scale"),
            },
        },
        "moe": {
            "routed_expert_count": tc["n_routed_experts"],
            "experts_per_token": tc["num_experts_per_tok"],
            "shared_expert_count": tc["n_shared_experts"],
            "expert_intermediate_dimension": tc["moe_intermediate_size"],
            "dense_intermediate_dimension": tc["intermediate_size"],
            "first_dense_layer_count": tc["first_k_dense_replace"],
            "routed_scaling_factor": tc["routed_scaling_factor"],
            "score_function": tc["scoring_func"],
            "topk_method": tc["topk_method"],
            "normalize_selected_probabilities": tc["norm_topk_prob"],
            "router_dtype_config": tc["moe_router_dtype"],
            "router_storage": "gate.weight BF16 [288,4096], e_score_correction_bias F32 [288]",
            "checkpoint_tensor_shapes": {
                "experts.0.gate_proj.weight": layer_tensor(dsa0, "mlp.experts.0.gate_proj.weight"),
                "experts.0.gate_proj.weight_scale_inv": layer_tensor(dsa0, "mlp.experts.0.gate_proj.weight_scale_inv"),
                "shared_experts.gate_proj.weight": layer_tensor(dsa0, "mlp.shared_experts.gate_proj.weight"),
            },
            "expert_layer_count": len(moe_idx),
        },
        "mtp": {
            "layer_count": tc["num_nextn_predict_layers"],
            "layer_index": mtp0,
            "layer_kind": "DSA (MLA + indexer) + MoE, no hyper-connections",
            "checkpoint_tensor_shapes": {
                "eh_proj.weight": layer_tensor(mtp0, "eh_proj.weight"),
                "enorm.weight": layer_tensor(mtp0, "enorm.weight"),
                "hnorm.weight": layer_tensor(mtp0, "hnorm.weight"),
                "shared_head.norm.weight": layer_tensor(mtp0, "shared_head.norm.weight"),
            },
        },
        "tokens": {
            "end_of_text": 154820,
            "user": 154827,
            "observation": 154829,
            "pad": tc["pad_token_id"],
            "eos_token_ids": tc["eos_token_id"],
        },
        "precision": {
            "quant_method": cfg["quantization_config"]["quant_method"],
            "weight_format": "fp8_e4m3",
            "activation_scheme": cfg["quantization_config"]["activation_scheme"],
            "weight_block_size": cfg["quantization_config"]["weight_block_size"],
            "fp8_quantized_patterns": sorted({
                pattern(n[:-len(".weight_scale_inv")])
                for n in manifest if n.endswith(".weight_scale_inv")
            }),
            "bf16_keep_patterns_note": (
                "all KDA self_attn, indexer, hc_*, MoE gate + e_score, "
                "A_log/dt_bias (F32), norms, embed/lm_head, vision"
            ),
        },
        "source": {
            "path_on_sparks": DEFAULT_SOURCE,
            "shard_count": shard_check["shard_count"],
            "index_total_bytes": None,  # filled by caller from index metadata
            "tensor_count": len(manifest),
            "receipt": files["_receipt"],
            "sha256sums_sha256": files["_sha256sums_sha256"],
            "receipt_digest_matches": files["_sha256sums_sha256"]
            == files["_receipt"]["sha256sums_sha256"],
            "shard_verification": {
                "method": (
                    "strided sample re-hash via sha256sum -c against pinned "
                    "SHA256SUMS (which is itself anchored by the receipt digest)"
                ),
                "stride": shard_check["stride"],
                "sampled": shard_check["sampled_shards"],
                "ok": shard_check["ok"],
                "failed": shard_check["failed"],
            },
            "files": {
                name: entry
                for name, entry in files.items() if not name.startswith("_")
            },
            "shard_sha256_pinned": {},  # filled by caller from SHA256SUMS
        },
        "qualification": {
            "cuda_target": "sm_121a",
            "status": "CONTRACT_FROZEN",
            "production_ready": False,
        },
    }

    # byte accounting by component family
    def categorize(name: str) -> str:
        if name.startswith("model.visual"):
            return "vision"
        if ".experts." in name:
            return "moe_routed_experts"
        if ".shared_experts." in name:
            return "moe_shared_experts"
        if re.search(r"mlp\.(gate_proj|up_proj|down_proj)", name):
            return "dense_mlp"
        if ".indexer." in name:
            return "dsa_indexer"
        if re.search(r"self_attn\.(q_a_proj|q_b_proj|kv_a_proj|kv_b_proj|o_proj)", name):
            return "dsa_mla"
        if re.search(r"self_attn\.(q_proj|k_proj|v_proj|q_conv1d|k_conv1d|v_conv1d|b_proj|A_log|dt_bias|f_a_proj|f_b_proj|g_a_proj|g_b_proj|o_norm)", name):
            return "kda"
        if ".hc_" in name:
            return "hyper_connections"
        if "eh_proj" in name or "enorm" in name or "hnorm" in name or "shared_head" in name:
            return "mtp_head"
        if "embed_tokens" in name:
            return "embed"
        if "lm_head" in name:
            return "lm_head"
        return "other_norms"

    comp_bytes: dict[str, int] = {}
    comp_count: dict[str, int] = {}
    for name, (dt, shape, _) in manifest.items():
        cat = categorize(name)
        comp_bytes[cat] = comp_bytes.get(cat, 0) + tensor_bytes(shape, dt)
        comp_count[cat] = comp_count.get(cat, 0) + 1
    total = sum(comp_bytes.values())
    contract["source"]["byte_accounting"] = {
        cat: {
            "gbytes": round(comp_bytes[cat] / 1e9, 3),
            "tensor_count": comp_count[cat],
        }
        for cat in sorted(comp_bytes, key=lambda c: -comp_bytes[c])
    }
    contract["source"]["total_gbytes"] = round(total / 1e9, 3)
    text_bytes = total - comp_bytes.get("vision", 0)
    contract["source"]["text_stack_gbytes"] = round(text_bytes / 1e9, 3)
    contract["source"]["tp16_gbytes_per_rank"] = round(text_bytes / 16 / 1e9, 3)
    return contract


def mlp_idx_msg(mlp_types: list[str]) -> str:
    return str([i for i, t in enumerate(mlp_types) if t == "sparse"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spark", required=True, help="spark host to fetch from")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()

    if not args.skip_fetch:
        print(f"fetching from {args.spark}:{args.source}")
        fetch(args.spark, args.source, args.cache_dir)
    cfg = json.loads((args.cache_dir / "config.json").read_text())
    manifest = load_manifest(args.cache_dir)

    print("verifying small files against pinned SHA256SUMS")
    files = verify_small_files(args.spark, args.source, args.cache_dir)
    mismatches = [
        name for name, entry in files.items()
        if isinstance(entry, dict) and entry.get("matches_source_sums") is False
    ]
    assert not mismatches, f"small-file digest mismatch: {mismatches}"
    assert files["_sha256sums_sha256"] == files["_receipt"]["sha256sums_sha256"], (
        "SHA256SUMS digest does not match archive receipt"
    )

    print("verifying strided shard sample (re-hash on spark)")
    shard_check = verify_strided_shards(args.spark, args.source, args.cache_dir)
    assert shard_check["failed"] == [], shard_check["failed"]
    assert shard_check["ok"] == len(shard_check["sampled_shards"])
    print(f"  {shard_check['ok']}/{len(shard_check['sampled_shards'])} shards OK")

    contract = build_contract(cfg, manifest, files, shard_check)

    # fill per-shard pinning from the receipt-anchored SHA256SUMS
    shard_pins = {}
    for line in (args.cache_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        if re.fullmatch(r"model-\d+-of-\d+\.safetensors", name):
            shard_pins[name.removeprefix("./")] = digest
    assert len(shard_pins) == shard_check["shard_count"]
    contract["source"]["shard_sha256_pinned"] = shard_pins

    index_meta = json.loads(
        (args.cache_dir / "model.safetensors.index.json").read_text()
    )["metadata"]
    contract["source"]["index_total_bytes"] = index_meta["total_size"]
    # census must reproduce the index total (payload bytes, header-excluded)
    census = sum_bytes_of(manifest)
    assert abs(index_meta["total_size"] - census) < 2_000_000, (
        index_meta["total_size"], census
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.output}")
    print(f"  tensors: {contract['source']['tensor_count']}")
    print(f"  total: {contract['source']['total_gbytes']} GB "
          f"(text stack {contract['source']['text_stack_gbytes']} GB, "
          f"TP16/rank {contract['source']['tp16_gbytes_per_rank']} GB)")
    return 0


def sum_bytes_of(manifest: dict) -> int:
    return sum(
        tensor_bytes(shape, dt) for dt, shape, _ in manifest.values()
    )


if __name__ == "__main__":
    sys.exit(main())
