#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
COSTS = ROOT / "deployment" / "fill_costs.json"
STEP_TOKENS = 1024
MAX_TOKENS = 1 << 21


def seconds_at(points, tokens):
    points = sorted(points)
    if len(points) == 1:
        return points[0][1] * tokens / points[0][0]
    for (low, low_seconds), (high, high_seconds) in zip(points, points[1:]):
        if tokens <= high:
            break
    return low_seconds + (high_seconds - low_seconds) * (tokens - low) / (high - low)


def cutoff(model, required_gain):
    first = min(point[0] for point in model["streamed_seconds"])
    for step in range(1, MAX_TOKENS // STEP_TOKENS + 1):
        tokens = step * STEP_TOKENS
        if tokens < first:
            continue
        streamed = seconds_at(model["streamed_seconds"], tokens) + model["reload_seconds"]
        if streamed <= (1.0 - required_gain) * seconds_at(model["wave_seconds"], tokens):
            return tokens
    return None


def main():
    parser = argparse.ArgumentParser(description="New-token count from which streaming plus its reload beats waves by the required gain")
    parser.add_argument("--costs", default=str(COSTS))
    args = parser.parse_args()
    costs = json.loads(pathlib.Path(args.costs).read_text())
    for name, model in sorted(costs["models"].items()):
        tokens = cutoff(model, costs["required_gain"])
        print(f"{name}: {'never' if tokens is None else tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
