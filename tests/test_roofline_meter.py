#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/roofline_meter.py"
failures = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def step(kind, rows, steps, kept, service, wall, gap, context=100, draft=0.0):
    return (f"batch_step kind={kind} request=100001 rows={rows} steps={steps} drafts=0 trunk=0 tree=0 kept={kept} match=0 class=0 expect=0.00 "
            f"predict_ms=0.00 service_ms={service:.2f} wall_ms={wall:.2f} context={context} gap_ms={gap:.2f} draft_wait_ms={draft:.2f}\n")


def run(*arguments):
    return subprocess.run([sys.executable, str(TOOL), *arguments], capture_output=True, text=True)


def close(value, expected):
    return abs(value - expected) < 1e-3


with tempfile.TemporaryDirectory() as directory:
    api = Path(directory) / "api.log"
    rank = Path(directory) / "rank.log"
    empty = Path(directory) / "empty.log"
    api.write_text("model_api ready\n" + step("single", 1, 1, 1, 25.0, 26.0, 0.1) * 4 + step("chain", 1, 8, 8, 160.0, 162.0, 0.2) * 2
                   + step("chain", 1, 8, 8, 160.0, 162.0, 200.0) + step("verify", 17, 17, 16, 50.0, 51.0, 0.1) + step("prefill", 1000, 1, 1000, 700.0, 705.0, 0.0)
                   + "batch_step kind=single request=7 rows=1\n")
    window = ("STAGE-STEP-TIMING rank=9 steps=64 graph_steps=62 pair_allreduce=0 allreduce=516 submit_us=24000 graph_us=11 source_wait_us=0 "
              "peer_wait_us=8000 copy_us=400 combine_us=400\n")
    rank.write_text("noise\n" + window * 2
                    + "STAGE-CHAIN-HOST tag=m rank=9 chains=64 mean_steps=2.3 admit_ms=0.05 submit_ms=53.00 capture_ms=0.01 sync_ms=0.01 finish_ms=0.29\n"
                    + "STAGE-WAVE-HOST tag=m rank=9 prefill_waves=16 admit_ms=0.1 submit_ms=51.1 capture_ms=0.0 sync_ms=0.0 finish_ms=0.8\n"
                    + "STAGE-STEP-TIMING rank=9 steps=64 graph_steps=62\n")
    empty.write_text("nothing here\n")

    result = run("--api-log", str(api), "--rank-log", str(rank), "--node-gb-per-token", "3.4", "--read-gb-s", "243", "--rounds-per-step", "128", "--json")
    check(result.returncode == 0, "the meter reads an API log and a rank log")
    summary = json.loads(result.stdout)
    floor = 1000.0 * 3.4 / 243.0
    kinds = summary["kinds"]
    check(sorted(kinds) == ["chain", "prefill", "single", "verify"] and kinds["single"]["frames"] == 4 and kinds["chain"]["frames"] == 3,
          "frames are grouped by kind and a batch_step line without its fields is ignored")
    check(close(summary["floor_ms"], floor) and close(summary["floor_tokens_per_second"], 243.0 / 3.4), "the memory floor is the node's bytes over its read rate")
    rank_step = summary["rank_step"]
    check(rank_step["steps"] == 128 and close(rank_step["step_ms"], 24.0) and close(rank_step["collective_ms"], 8.8) and close(summary["collective_share"], 8.8 / 24.0),
          "the rank step and its share in collectives come from the timing windows; a window without its fields is ignored")
    check(close(rank_step["rounds_per_step"], 128.0) and close(rank_step["peer_wait_us_per_round"], 8000.0 / 128.0),
          "the peer wait a round is the step's peer wait over the rounds the caller names")
    single = kinds["single"]["decomposition"]
    check(close(single["total_ms"], 26.1) and close(single["floor_ms"], floor) and close(single["collective_ms"], 25.0 * 8.8 / 24.0)
          and close(single["rank_other_ms"], 25.0 - floor - 25.0 * 8.8 / 24.0) and close(single["host_ms"], 1.0) and close(single["gap_ms"], 0.1),
          "a single step splits into the floor, collectives, the rank's remainder, host and gap, and they add up to its time")
    check(close(sum(value for name, value in single.items() if name != "total_ms"), single["total_ms"]), "the parts of a pass sum to the pass")
    check(close(kinds["single"]["of_roofline"], floor / 26.1), "a single step's share of the roofline is the floor over its time")
    chain = kinds["chain"]
    check(chain["contended_frames"] == 1 and close(chain["per_pass"]["passes"], 16.0) and close(chain["decomposition"]["total_ms"], 20.275)
          and close(chain["of_roofline"], floor / 20.275), "a chain counts each step as a pass and leaves out a frame that waited behind other work")
    verify = kinds["verify"]
    check(close(verify["per_pass"]["passes"], 1.0) and close(verify["per_pass"]["tokens"], 16.0) and close(verify["ms_per_token"], 51.1 / 16.0)
          and close(verify["of_roofline"], floor * 16.0 / 51.1), "a verify frame is one pass that yields its kept tokens")
    check("decomposition" not in kinds["prefill"] and close(kinds["prefill"]["tokens_per_second"], 1000.0 * 1000.0 / 705.0), "prefill reports its rate and no decode split")
    check(close(summary["rank_chain_host"]["outside_submit_ms"], 0.36) and summary["rank_wave_host"]["frames"] == 16, "the rank's host time around a chain and a prefill wave is reported")

    text = run("--api-log", str(api), "--rank-log", str(rank), "--node-gb-per-token", "3.4", "--read-gb-s", "243", "--rounds-per-step", "128", "--label", "test lane")
    check(text.returncode == 0 and text.stdout.startswith("roofline meter: test lane\n") and "over 128 rounds, 62.5 us a round" in text.stdout
          and "chain: 20.27 ms a pass for 1.00 tokens" in text.stdout and "69% of the one-token roofline" in text.stdout
          and "1 frames that waited behind other work left out" in text.stdout, "the text report names the rounds, the chain's pass and its share of the roofline")

    ahead = run("--api-log", str(api), "--rank-log", str(rank), "--node-gb-per-token", "5.0", "--read-gb-s", "243")
    check(ahead.returncode == 0 and "memory floor 20.58 (79%), collectives 9.17 (35%), less 4.74 (18%) of the floor read during the waits, host and wire 1.00 (4%)" in ahead.stdout,
          "a rank whose own work is under its floor read part of the floor during its waits, and the report says how much")

    heavy = Path(directory) / "heavy.log"
    light = Path(directory) / "light.log"
    heavy.write_text("WD-SPINE-PRIVATE spans=1 bytes=3500000000\n"
                     "STAGE-STEP-ARRIVAL rank=0 steps=64 first_us=1280 peer_us=0,2560,3840\n"
                     "STAGE-STEP-TIMING rank=0 steps=64 graph_steps=62 pair_allreduce=0 allreduce=516 submit_us=24000 graph_us=11 source_wait_us=0 "
                     "peer_wait_us=6000 copy_us=500 combine_us=500\n")
    light.write_text("WD-SPINE-PRIVATE spans=1 bytes=3250000000\n"
                     "STAGE-STEP-ARRIVAL rank=2 steps=64 first_us=2560 peer_us=5120,3840,0\n"
                     "STAGE-STEP-TIMING rank=2 steps=64 graph_steps=62 pair_allreduce=0 allreduce=516 submit_us=24000 graph_us=11 source_wait_us=0 "
                     "peer_wait_us=8000 copy_us=500 combine_us=500\n")
    uneven = run("--api-log", str(api), "--rank-log", str(light), "--rank-log", str(heavy), "--node-gb-per-token", "3.5", "--read-gb-s", "243", "--rounds-per-step", "128")
    skew = json.loads(run("--api-log", str(api), "--rank-log", str(light), "--rank-log", str(heavy), "--node-gb-per-token", "3.5", "--read-gb-s", "243", "--json").stdout)
    table = skew["rank_table"]
    check([row["rank"] for row in table] == [0, 2] and close(table[0]["gb"], 3.5) and close(table[0]["own_ms"], 17.0) and close(table[1]["own_ms"], 15.0)
          and close(table[1]["peer_wait_ms"], 8.0), "each rank log gives its rank, its weight bytes and its own work a step, listed by rank")
    check(skew["slowest_rank"] == 0 and close(skew["collective_share"], 7.0 / 24.0)
          and close(skew["kinds"]["single"]["decomposition"]["collective_ms"], 25.0 * 7.0 / 24.0),
          "the split uses the collective share of the rank with the most own work, the one the others wait for")
    check(close(table[0]["first_arrival_us"], 1280.0) and close(table[0]["median_arrival_us"], 3840.0) and close(table[0]["last_arrival_us"], 3840.0)
          and close(table[1]["median_arrival_us"], 5120.0), "a rank's arrival line gives when its first, median and slowest peer land, leaving itself out")
    check(uneven.returncode == 0 and "ranks: own work 15.00 ms a step on rank 2 to 17.00 on rank 0" in uneven.stdout and "weights 3.250 to 3.500 GB a rank" in uneven.stdout
          and "peers land after a round's wait starts, in us, fastest to slowest rank log: first 10.0 to 20.0, median 30.0 to 40.0, last on average 30.0 to 40.0, all 46.9 to 62.5" in uneven.stdout,
          "the text report names the slowest rank, the weight range and the arrivals a round")
    plain = json.loads(run("--api-log", str(api), "--json").stdout)
    unrounded = json.loads(run("--api-log", str(api), "--rank-log", str(rank), "--json").stdout)
    check(unrounded["rank_step"]["rounds_per_step"] is None and unrounded["rank_step"]["peer_wait_us_per_round"] is None, "without the rounds no wait a round is stated")
    check(plain["floor_ms"] is None and plain["rank_step"] is None and "of_roofline" not in plain["kinds"]["single"]
          and close(plain["kinds"]["single"]["decomposition"]["service_ms"], 25.0), "without a rank log and the node's bytes the split is service, host and gap")
    missing = run("--api-log", str(empty))
    check(missing.returncode == 1 and "no batch_step line" in missing.stderr, "an API log with no batch_step line is an error")
    check(run("--api-log", str(api), "--node-gb-per-token", "0").returncode == 2, "a node byte count that is not positive is refused")

print(("FAIL " if failures else "PASS ") + f"roofline meter: {len(failures)} failures")
sys.exit(1 if failures else 0)
