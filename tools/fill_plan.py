#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fill_cutoff

SHARDS = 16


def pass_seconds(plan, tokens, first):
    compute = plan["seconds_per_token"] * tokens + plan["attention_seconds_per_token_pair"] * tokens * (first + tokens / 2.0)
    return max(plan["pass_floor_seconds"], compute)


def choice(plan, tokens, passes):
    rows = math.ceil(tokens / passes / SHARDS / plan["row_align"]) * plan["row_align"]
    pass_tokens = rows * SHARDS
    fill = sum(pass_seconds(plan, min(pass_tokens, tokens - index * pass_tokens), index * pass_tokens) for index in range(passes))
    memory = rows * plan["row_bytes"] + plan["ring_bytes"] + plan["dense_bytes"]
    evicted = max(0.0, memory - plan["free_bytes"])
    if evicted > plan["expert_bytes"]:
        return None
    reload = plan["restart_seconds"] + evicted / plan["reload_bytes_per_second"] if evicted > 0.0 else 0.0
    return {"passes": passes, "node_rows": rows, "memory_gb": round(memory / 1e9, 1), "evicted_gb": round(evicted / 1e9, 1),
            "fill_s": round(fill, 1), "reload_s": round(reload, 1), "total_s": round(fill + reload, 1)}


def best(plan, tokens):
    options = [option for option in (choice(plan, tokens, passes) for passes in range(1, plan["max_passes"] + 1)) if option is not None]
    return min(options, key=lambda option: option["total_s"]) if options else None


def main():
    parser = argparse.ArgumentParser(description="For each prompt size, the RAM that minimizes fill plus expert reload")
    parser.add_argument("--costs", default=str(fill_cutoff.COSTS))
    parser.add_argument("--sizes", default="33000,64000,100000,131072,200000,258319,524288,737280,990881")
    args = parser.parse_args()
    costs = json.loads(pathlib.Path(args.costs).read_text())
    for name, model in sorted(costs["models"].items()):
        if "plan" not in model:
            continue
        for tokens in (int(size) for size in args.sizes.split(",")):
            option = best(model["plan"], tokens)
            waves = fill_cutoff.seconds_at(model["wave_seconds"], tokens)
            stream = option is not None and option["total_s"] <= (1.0 - costs["required_gain"]) * waves
            print(name, tokens, "stream" if stream else "waves", f"waves {waves:.0f} s", option)
    return 0


if __name__ == "__main__":
    sys.exit(main())
