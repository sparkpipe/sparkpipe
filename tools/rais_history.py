#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import rais_score as rs

RELEASE_LABELS: dict[str, tuple[str, str]] = {
    "4fc4a10e3": ("Baseline", "Initial 16-Spark single-switch baseline"),
    "07a6e6bcd": ("Wave-1", "First 16-Spark hill-climb pass (E1-W1)"),
    "d99d5cc81": ("Wave-2", "B=8 speculation + PAIR_GATHER + 2-node A2A"),
    "99fc2a2c1": ("Wave-3", "First 19/19 VALID certified fleet release + 11 TREE_VERIFY"),
    "454edc951": ("Wave-4A", "5x GQA/attn/linear-attn shard pass + MoE fusion"),
    "dff7b5a96": ("Wave-4B", "E2 k3.1-pro +4.5, E4 q3.8-flash +9.2, E11 mimo/hy4, q3.8-27b tree fix"),
    "wave6-ededce093": ("Wave-6", "17/17 swept VALID (44.87 RAIS, Pillar 5 systems physics + cold-boot -40.6 s)"),
    "wave7h-b33686ea2": ("Wave-7H", "18/19 VALID (97.0% cov, 50.34 RAIS; +5.47 vs Wave-6; B64 2D 4x4 deadlock fixed, 0 errors across 19/19 models)"),
}

def _round2(val: float | None) -> float | None:
    if val is None:
        return None
    return round(float(val), 2)


def _extract_raw_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    spec = metrics.get("speculation") or {}
    batch = metrics.get("batch") or {}
    b4 = batch.get("4") or batch.get(4) or {}
    b8 = batch.get("8") or batch.get(8) or {}
    b16 = batch.get("16") or batch.get(16) or {}
    b32 = batch.get("32") or batch.get(32) or {}
    b64 = batch.get("64") or batch.get(64) or {}
    return {
        "b1_short_tok_s": _round2(metrics.get("b1_short_tok_s")),
        "b1_long_tok_s": _round2(metrics.get("b1_long_tok_s")),
        "b1_spec_tok_s": _round2(spec.get("b1_spec_tok_s")),
        "alpha": _round2(spec.get("alpha")),
        "verify_v_b1": _round2(spec.get("verify_v_b1")),
        "verify_v_b8": _round2(spec.get("verify_v_b8")),
        "b4_tok_s": _round2(b4.get("aggregate_tok_s")),
        "b8_tok_s": _round2(b8.get("aggregate_tok_s")),
        "b16_tok_s": _round2(b16.get("aggregate_tok_s")),
        "b32_tok_s": _round2(b32.get("aggregate_tok_s")),
        "b64_tok_s": _round2(b64.get("aggregate_tok_s")),
        "b8_long_tok_s": _round2(metrics.get("b8_long_tok_s")),
        "prefill_8k_tok_s": _round2(metrics.get("prefill_8k_tok_s")),
        "prefill_32k_tok_s": _round2(metrics.get("prefill_32k_tok_s")),
        "warm_8k_ttft_ms": _round2(metrics.get("warm_8k_ttft_ms")),
        "cold_boot_s": _round2(metrics.get("cold_boot_s", metrics.get("swap_s"))),
        "usable_kv_gib_per_rank": _round2(metrics.get("usable_kv_gib_per_rank", metrics.get("usable_kv_gib"))),
        "batch_invariant_exact": metrics.get("batch_invariant_exact"),
    }


def build_snapshot(
    jsonl_path: Path,
    hardware: str = "16x-gb10-1x100g",
    release_sha: str | None = None,
    label: str | None = None,
    note: str | None = None,
    weights_path: Path | None = None,
    divisor: float | None = None,
    basket_id: str | None = None,
) -> dict[str, Any]:
    recs = rs.load_jsonl(jsonl_path)
    if not recs:
        raise ValueError(f"Empty RAIS JSONL: {jsonl_path}")

    sha = release_sha or jsonl_path.parent.name
    default_label, default_note = RELEASE_LABELS.get(sha, (sha, ""))
    eff_label = label or default_label
    eff_note = note if note is not None else default_note

    weights = None
    eff_basket = basket_id
    eff_divisor = divisor
    if weights_path is not None:
        weights, file_basket, file_div = rs.load_weights_file(weights_path)
        eff_basket = eff_basket or file_basket
        if eff_divisor is None and file_div is not None:
            eff_divisor = file_div

    report = rs.score_fleet(
        recs,
        hardware,
        weights_override=weights,
        divisor=eff_divisor if eff_divisor is not None else rs.DEFAULT_DIVISOR,
        basket_id=eff_basket or rs.DEFAULT_BASKET_ID,
    )
    eff_weights = weights if weights is not None else rs.default_fleet_weights()
    by_model = {m["model"]: m for m in report["models"]}
    raw_by_model = {rs.resolve_model_name(r["model"]): r for r in recs}

    utcs = sorted(str(r.get("utc", "")) for r in recs if r.get("utc"))
    utc_end = utcs[-1] if utcs else ""
    utc_start = utcs[0] if utcs else ""

    canonical_models = [k for k in eff_weights if k in by_model]
    w_sum = sum(eff_weights[k] for k in canonical_models) or 1.0

    unpenalized_rais = sum(eff_weights[k] * by_model[k]["raw_rais"] for k in canonical_models) / w_sum
    base_r = sum(eff_weights[k] * by_model[k]["base_roofline_r"] for k in canonical_models) / w_sum
    pillars = {
        p: round(sum(eff_weights[k] * by_model[k]["pillars"][p] for k in canonical_models) / w_sum, 2)
        for p in rs.PILLAR_ORDER
    }
    submetrics = {
        sk: round(
            sum(eff_weights[k] * by_model[k]["submetrics"][sk]["score"] for k in canonical_models) / w_sum,
            2,
        )
        for sk in rs.SUBMETRIC_WEIGHTS
    }

    valid_models = [k for k in canonical_models if by_model[k]["status"] == "VALID"]
    failed_models = {
        k: by_model[k]["gates"]["failed_gates"]
        for k in canonical_models
        if by_model[k]["status"] != "VALID"
    }

    models_payload: dict[str, Any] = {}
    for m in report["models"]:
        name = m["model"]
        raw_metrics = raw_by_model.get(name, {}).get("metrics") or {}
        models_payload[name] = {
            "status": m["status"],
            "rais": _round2(m["certified_rais"]),
            "raw_rais": _round2(m["raw_rais"]),
            "base_r": _round2(m["base_roofline_r"]),
            "pillars": {p: _round2(m["pillars"][p]) for p in rs.PILLAR_ORDER},
            "submetrics": {sk: _round2(m["submetrics"][sk]["score"]) for sk in rs.SUBMETRIC_WEIGHTS},
            "failed_gates": list(m["gates"]["failed_gates"]),
            "raw": _extract_raw_summary(raw_metrics),
        }

    rel_receipt = str(jsonl_path)
    try:
        rel_receipt = str(jsonl_path.resolve().relative_to(TOOLS_DIR.parent.resolve()))
    except ValueError:
        pass

    return {
        "rais_version": report["rais_version"],
        "utc": utc_end,
        "utc_start": utc_start,
        "release": sha,
        "label": eff_label,
        "note": eff_note,
        "hardware": hardware,
        "basket_id": report["basket_id"],
        "index_divisor": report["index_divisor"],
        "receipt": rel_receipt,
        "fleet": {
            "status": report["fleet_status"],
            "valid_models": len(valid_models),
            "total_models": len(canonical_models),
            "coverage_pct": _round2(report["fleet_coverage_pct"]),
            "rais_certified": _round2(report["fleet_rais_covered"]),
            "rais_index": _round2(report["fleet_rais_index"]),
            "rais_valid_weighted": _round2(report["fleet_rais_raw_covered"]),
            "rais_unpenalized": round(unpenalized_rais, 2),
            "base_r": round(base_r, 2),
            "pillars": pillars,
            "submetrics": submetrics,
            "failed_models": failed_models,
        },
        "models": models_payload,
    }


def rebuild_history(
    releases_dir: Path,
    hardware: str = "16x-gb10-1x100g",
) -> list[dict[str, Any]]:
    pattern = f"*/rais-fleet-{hardware}.jsonl"
    files = sorted(releases_dir.glob(pattern))
    snapshots = [build_snapshot(p, hardware=hardware) for p in files]
    snapshots.sort(key=lambda s: (s.get("utc") or "", s.get("release") or ""))
    return snapshots


def write_history_jsonl(snapshots: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(s, separators=(",", ":")) for s in snapshots]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def sanitize_site_snapshots(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for snap in snapshots:
        item = dict(snap)
        item["receipt"] = rs.sanitize_receipt_label(
            snap.get("receipt"),
            hardware=snap.get("hardware"),
            release=snap.get("release"),
        )
        cleaned.append(item)
    return cleaned


def update_site_data(site_data_path: Path, snapshots: list[dict[str, Any]], hardware: str) -> None:
    data = json.loads(site_data_path.read_text(encoding="utf-8"))
    site_snaps = sanitize_site_snapshots(snapshots)
    if data.get("hardware", {}).get("name") == hardware:
        data["history"] = site_snaps
    elif hardware in (data.get("profiles") or {}):
        data["profiles"][hardware]["history"] = site_snaps
    else:
        data["history"] = site_snaps
    site_data_path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rais_history", description="Build and track chronological RAIS v1.1 history with full pillar, sub-metric, and per-model breakdowns.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_reb = sub.add_parser("rebuild", help="Scan qualification/releases/* and rebuild qualification/rais_history.jsonl.")
    p_reb.add_argument("--releases-dir", default="qualification/releases")
    p_reb.add_argument("--hardware", default="16x-gb10-1x100g", choices=sorted(rs.HARDWARE_PROFILES))
    p_reb.add_argument("--output", default="qualification/rais_history.jsonl")
    p_reb.add_argument("--update-site", default=None, help="Optional site/rais-data.json path to embed history into.")

    p_app = sub.add_parser("append", help="Append or replace one release snapshot in qualification/rais_history.jsonl.")
    p_app.add_argument("results", help="Path to release RAIS JSONL file.")
    p_app.add_argument("--hardware", default="16x-gb10-1x100g", choices=sorted(rs.HARDWARE_PROFILES))
    p_app.add_argument("--release", default=None)
    p_app.add_argument("--label", default=None)
    p_app.add_argument("--note", default=None)
    p_app.add_argument("--history", default="qualification/rais_history.jsonl")
    p_app.add_argument("--update-site", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rebuild":
        snaps = rebuild_history(Path(args.releases_dir), hardware=args.hardware)
        write_history_jsonl(snaps, Path(args.output))
        if args.update_site:
            update_site_data(Path(args.update_site), snaps, args.hardware)
        print(f"Wrote {len(snaps)} snapshots to {args.output}")
        for s in snaps:
            fl = s["fleet"]
            p = fl["pillars"]
            print(
                f"  {s['utc']} | {s['label']:8s} ({s['release']}) | valid={fl['valid_models']:2d}/{fl['total_models']} | "
                f"unpen={fl['rais_unpenalized']:5.2f} | valid_wt={fl['rais_valid_weighted']:5.2f} | cert={str(fl['rais_certified']):>5s} | "
                f"R={fl['base_r']:5.2f} | P1={p['P1']:5.2f} P2={p['P2']:5.2f} P3={p['P3']:5.2f} P4={p['P4']:5.2f} P5={p['P5']:5.2f}"
            )
        return 0

    if args.command == "append":
        hist_path = Path(args.history)
        existing = []
        if hist_path.exists():
            existing = [json.loads(l) for l in hist_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        snap = build_snapshot(
            Path(args.results),
            hardware=args.hardware,
            release_sha=args.release,
            label=args.label,
            note=args.note,
        )
        filtered = [x for x in existing if not (x.get("release") == snap["release"] and x.get("hardware") == snap["hardware"])]
        filtered.append(snap)
        filtered.sort(key=lambda s: (s.get("utc") or "", s.get("release") or ""))
        write_history_jsonl(filtered, hist_path)
        if args.update_site:
            update_site_data(Path(args.update_site), filtered, args.hardware)
        print(f"Updated {hist_path} ({len(filtered)} snapshots)")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
