from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path

CHAIN_LINE = re.compile(r"^CHAIN slot=(\d+) stage=\d+ layer=\d+ rows=(\d+)")
REPLAY_LINE = re.compile(r"^GRAPH-REPLAY-TIME slot=(\d+) wall_ns=(\d+) stream_status=(\d+)")
RANK_FILE = re.compile(r"residentd\.rank(\d+)\.log$")


def parse_rank_log(lines) -> list[tuple[int, float]]:
    rows_by_slot: dict[int, int] = {}
    samples: list[tuple[int, float]] = []
    for line in lines:
        match = CHAIN_LINE.match(line)
        if match:
            rows_by_slot[int(match.group(1))] = int(match.group(2))
            continue
        match = REPLAY_LINE.match(line)
        if match and int(match.group(3)) == 0:
            rows = rows_by_slot.get(int(match.group(1)))
            if rows is not None:
                samples.append((rows, int(match.group(2)) / 1e6))
    return samples


def collect_arm(directory: Path) -> dict[int, dict[int, list[float]]]:
    per_rows: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(directory.glob("residentd.rank*.log")):
        rank = int(RANK_FILE.search(path.name).group(1))
        with path.open(errors="replace") as handle:
            for rows, ms in parse_rank_log(handle):
                per_rows[rows][rank].append(ms)
    return per_rows


def summarize_rows(per_rank: dict[int, list[float]]) -> dict:
    pooled = sorted(ms for values in per_rank.values() for ms in values)
    rank_medians = {rank: statistics.median(values) for rank, values in per_rank.items() if values}
    count = len(pooled)
    return {"n": count, "ranks": len(rank_medians),
            "engine_median_ms": round(statistics.median(rank_medians.values()), 3),
            "pooled_median_ms": round(statistics.median(pooled), 3),
            "p10_ms": round(pooled[count // 10], 3), "p90_ms": round(pooled[(9 * count) // 10], 3),
            "min_ms": round(pooled[0], 3), "max_ms": round(pooled[-1], 3)}


def fit_linear(points: list[tuple[int, float, int]]) -> dict:
    weight_sum = sum(weight for _, _, weight in points)
    mean_x = sum(rows * weight for rows, _, weight in points) / weight_sum
    mean_y = sum(ms * weight for _, ms, weight in points) / weight_sum
    variance = sum(weight * (rows - mean_x) ** 2 for rows, _, weight in points)
    if variance == 0.0:
        return {"intercept_ms": mean_y, "per_row_ms": 0.0}
    slope = sum(weight * (rows - mean_x) * (ms - mean_y) for rows, ms, weight in points) / variance
    return {"intercept_ms": mean_y - slope * (mean_x - 1), "per_row_ms": slope}


def fit_verify_cost(arms: dict[str, Path], min_samples: int = 16, extra_points: list[dict] | None = None) -> dict:
    samples: dict[int, dict] = {}
    sources: dict[str, dict] = {}
    merged: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for arm, directory in arms.items():
        per_rows = collect_arm(directory)
        sources[arm] = {str(rows): summarize_rows(per_rank) for rows, per_rank in sorted(per_rows.items())}
        for rows, per_rank in per_rows.items():
            for rank, values in per_rank.items():
                merged[rows][rank].extend(values)
    for rows, per_rank in sorted(merged.items()):
        summary = summarize_rows(per_rank)
        summary["measured"] = summary["n"] >= min_samples
        samples[rows] = summary
    points = [(rows, summary["engine_median_ms"], summary["n"]) for rows, summary in samples.items() if summary["measured"]]
    if len(points) < 2:
        raise ValueError("the verify-cost fit needs measured replay times at two or more row counts")
    fit = fit_linear(points)
    residuals = {str(rows): round(summary["engine_median_ms"] - predict_rows(fit, rows), 3) for rows, summary in samples.items()}
    return {"unit": "ms per verify wave, GRAPH-REPLAY-TIME wall of one rank, engine median over ranks",
            "samples": {str(rows): summary for rows, summary in samples.items()}, "sources": sources,
            "fit": {"form": "T(rows) = intercept + per_row * (rows - 1)", **{key: round(value, 4) for key, value in fit.items()}},
            "residual_ms": residuals, "extra_points": extra_points or []}


def predict_rows(fit: dict, rows: int) -> float:
    return fit["intercept_ms"] + fit["per_row_ms"] * (rows - 1)


def verify_ms(model: dict, k: int, batch: int = 1) -> tuple[float, str]:
    rows = (k + 1) * batch
    sample = model["samples"].get(str(rows))
    if sample is not None and sample.get("measured"):
        return sample["engine_median_ms"], "measured"
    for point in model.get("extra_points", []):
        if point.get("rows") == rows:
            return float(point["ms"]), point.get("label", "extra")
    return predict_rows(model["fit"], rows), "fitted" if rows <= max(int(r) for r in model["samples"]) else "extrapolated"


def host_overhead_ms(k: int, base_ms: float = 2.0, per_row_ms: float = 0.575) -> float:
    return base_ms if k == 0 else base_ms + per_row_ms * (k + 1)
