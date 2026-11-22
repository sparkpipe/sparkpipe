#!/usr/bin/env python3
"""Speculation soak: drive one model's API with a fixed edit workload, then compare two drafter arms' counters.

    spec_soak.py run --engine URL --chat-dir DIR --model ID --log API_LOG --minutes N --out DIR [--concurrency C] [--unit UNIT ...]
    spec_soak.py compare FIRST_ARM_DIR SECOND_ARM_DIR [--tolerance 0.02]

`run` builds the workload from this repository's files: rename-a-function edits as tools/spec_edit_corpus.py builds
them, in three context tiers (the file alone; after about 40K tokens of other files; after about 100K). Even tasks
decode greedily, odd tasks sample at temperature 0.7 and top_p 0.95 with a seed fixed per task, so both arms see the
same requests. It sends them round robin, C at a time, until N minutes pass, and writes arm.json: each request's sizes
and times, the API log's byte offsets at the start and the end, and every named unit's NRestarts before and after.

`compare` reads each arm's slice of the API log and sums the model drafter's counters over its `batch_speculation`
lines; a counter that drops marks an API restart and starts a new segment. It prints accepted drafts per round, the
UNAVAILABLE rate and the mean wait for both arms, and passes when the second arm's accepted drafts per round are within
the tolerance of the first's, its UNAVAILABLE rate is no higher, and no named unit restarted during it.
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_edit_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COUNTERS = ("rounds", "frames", "drafted", "kept_drafts", "declined", "unavailable", "backoffs", "failed", "abandoned")
MODEL = re.compile(r" model ((?:\w+=[\d.]+ ?)+)")
TIERS = ((0, 48), (160_000, 16), (400_000, 8))
CHARS = (4_000, 20_000)


def workload():
    sources = sorted(path for path in (ROOT / "tools").glob("*.py") if CHARS[0] <= path.stat().st_size <= CHARS[1])
    backgrounds = sorted(path for path in (ROOT / "runtime").glob("*.c") if path.stat().st_size >= CHARS[1])
    tasks = []
    for context_chars, count in TIERS:
        context, used = [], 0
        for path in backgrounds:
            if used >= context_chars:
                break
            context.append(path.relative_to(ROOT).as_posix())
            used += path.stat().st_size
        prefix = spec_edit_corpus.background(ROOT, context)
        for path in sources:
            if sum(1 for task in tasks if task["context_chars"] == len(prefix)) >= count:
                break
            text = path.read_text()
            name = spec_edit_corpus.first_definition(text)
            if name is None:
                continue
            relative = path.relative_to(ROOT).as_posix()
            for style, rule in spec_edit_corpus.STYLES.items():
                prompt = (f"{prefix}Here is the file {relative}:\n\n{text}\n\nRename the function {name} to {name}_v2 at its definition and at "
                          f"every use in this file, and change nothing else. {rule}")
                tasks.append({"name": f"{style}_{len(tasks):03d}", "messages": [{"role": "user", "content": prompt}], "file_chars": len(text),
                              "context_chars": len(prefix)})
    for index, task in enumerate(tasks):
        task["sampling"] = {"temperature": 0} if index % 2 == 0 else {"temperature": 0.7, "top_p": 0.95, "seed": 1000 + index}
    return tasks


def restarts(units):
    found = {}
    for unit in units:
        shown = subprocess.run(["systemctl", "--user", "show", "-p", "NRestarts", "--value", unit], capture_output=True, text=True)
        found[unit] = int(shown.stdout.strip() or 0) if shown.returncode == 0 and shown.stdout.strip().isdigit() else None
    return found


def run(args):
    frontend = spec_edit_corpus.load_frontend(args.chat_dir)
    config = json.loads((Path(args.chat_dir) / "config.json").read_text())
    model = frontend.Model(next(entry for entry in config["models"] if entry["id"] == args.model), Path(args.chat_dir))
    tasks = workload()
    effort = dict(model.reasoning_effort.get(args.reasoning_effort, {})) if args.reasoning_effort else {}
    rendered = [model.render(frontend.normalize_messages(task["messages"]), None, effort)[1] for task in tasks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = {"model": args.model, "engine": args.engine, "log": args.log, "tasks": len(tasks), "start_offset": Path(args.log).stat().st_size,
           "units_before": restarts(args.unit), "started": time.time(), "requests": []}
    deadline, cursor, lock = time.monotonic() + args.minutes * 60.0, [0], threading.Lock()

    def worker():
        while time.monotonic() < deadline:
            with lock:
                index = cursor[0] % len(tasks)
                cursor[0] += 1
            task = tasks[index]
            budget = min(args.max_tokens, int(task["file_chars"] / 2.5) + 512)
            body = dict(task["sampling"], prompt_token_ids=rendered[index], max_tokens=budget, stop_token_ids=model.stop_token_ids)
            started = time.monotonic()
            try:
                output, finish, first = spec_edit_corpus.stream(args.engine, body)
                row = {"task": task["name"], "prompt_tokens": len(rendered[index]), "output_tokens": len(output), "finish": finish,
                       "seconds": round(time.monotonic() - started, 3), "ttft": round(first or 0.0, 3)}
            except OSError as error:
                row = {"task": task["name"], "error": str(error), "seconds": round(time.monotonic() - started, 3)}
            with lock:
                arm["requests"].append(row)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    arm.update(ended=time.time(), end_offset=Path(args.log).stat().st_size, units_after=restarts(args.unit))
    (out / "arm.json").write_text(json.dumps(arm, indent=1))
    failed = sum(1 for row in arm["requests"] if "error" in row)
    print(f"spec_soak: {len(arm['requests'])} requests ({failed} failed) of {len(tasks)} tasks in {args.minutes} minutes; wrote {out / 'arm.json'}")
    return 0 if failed == 0 else 1


def counters(line):
    match = MODEL.search(line)
    if match is None:
        return None
    values = dict(item.split("=") for item in match.group(1).split())
    found = {key: int(values[key]) for key in COUNTERS if key in values}
    found["wait_total_ms"] = float(values.get("wait_ms", 0.0)) * found.get("rounds", 0)
    return found if len(found) == len(COUNTERS) + 1 else None


def summed(log, start, end):
    data = Path(log).read_bytes()
    before = [counters(line) for line in data[:start].decode(errors="replace").splitlines() if "batch_speculation" in line]
    lines = [counters(line) for line in data[start:end].decode(errors="replace").splitlines() if "batch_speculation" in line]
    previous = next((item for item in reversed(before) if item is not None), None)
    total = dict.fromkeys(list(COUNTERS) + ["wait_total_ms"], 0.0)
    segments = 1
    for current in (item for item in lines if item is not None):
        restarted = previous is None or any(current[key] < previous[key] for key in COUNTERS)
        segments += 1 if restarted and previous is not None else 0
        for key in total:
            total[key] += current[key] - (0.0 if restarted else previous[key])
        previous = current
    return total, segments


def compare(args):
    arms = []
    for directory in args.arms:
        arm = json.loads((Path(directory) / "arm.json").read_text())
        total, segments = summed(arm["log"], arm["start_offset"], arm["end_offset"])
        rounds = max(total["rounds"], 1.0)
        restarted = {unit: (arm["units_before"].get(unit), arm["units_after"].get(unit)) for unit in arm["units_after"]
                     if arm["units_before"].get(unit) != arm["units_after"].get(unit)}
        arms.append({"name": Path(directory).name, "rounds": int(total["rounds"]), "kept_per_round": total["kept_drafts"] / rounds,
                     "drafted_per_round": total["drafted"] / rounds, "unavailable_rate": total["unavailable"] / rounds,
                     "wait_ms": total["wait_total_ms"] / rounds, "failed": int(total["failed"]), "segments": segments, "restarted": restarted,
                     "requests": len(arm["requests"])})
    for arm in arms:
        print(f"{arm['name']}: {arm['requests']} requests, {arm['rounds']} drafter rounds in {arm['segments']} API segment(s); accepted drafts per round "
              f"{arm['kept_per_round']:.4f}, drafted per round {arm['drafted_per_round']:.3f}, UNAVAILABLE rate {arm['unavailable_rate']:.4f}, "
              f"wait {arm['wait_ms']:.2f} ms, failed asks {arm['failed']}, unit restarts {arm['restarted'] or 'none'}")
    first, second = arms
    ratio = second["kept_per_round"] / first["kept_per_round"] if first["kept_per_round"] else float("inf")
    passed = abs(ratio - 1.0) <= args.tolerance and second["unavailable_rate"] <= first["unavailable_rate"] and not second["restarted"]
    print(("PASS" if passed else "FAIL") + f" {second['name']} against {first['name']}: accepted drafts per round x{ratio:.4f} (tolerance "
          f"{args.tolerance}), UNAVAILABLE rate {second['unavailable_rate']:.4f} against {first['unavailable_rate']:.4f}, restarts "
          f"{second['restarted'] or 'none'}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    drive = commands.add_parser("run")
    drive.add_argument("--engine", required=True)
    drive.add_argument("--chat-dir", required=True)
    drive.add_argument("--model", required=True)
    drive.add_argument("--log", required=True)
    drive.add_argument("--minutes", type=float, required=True)
    drive.add_argument("--out", required=True)
    drive.add_argument("--concurrency", type=int, default=8)
    drive.add_argument("--max-tokens", type=int, default=2048)
    drive.add_argument("--reasoning-effort", default="minimal")
    drive.add_argument("--unit", action="append", default=[])
    check = commands.add_parser("compare")
    check.add_argument("arms", nargs=2)
    check.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
