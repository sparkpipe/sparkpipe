#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "glm53full_scorecard"))
import roofline
import runner


def close(a, b, tol):
    return abs(a - b) <= tol


def check_model():
    failures = []
    active = roofline.active_params()
    if not close(active / 1e9, 40.3, 0.3):
        failures.append(f"active params {active / 1e9:.2f} B, expected ~40.3 B")
    m = roofline.step_model(1, 300)
    if m["expert_keys"] != 600:
        failures.append(f"B1 expert keys {m['expert_keys']}, expected 75 x 8 = 600")
    if not close(m["bytes_per_rank"] / 1e9, 4.77, 0.01):
        failures.append(f"B1 bytes {m['bytes_per_rank'] / 1e9:.3f} GB, expected 4.77")
    if m["rounds"] != 236 or m["phases_per_op"] != 1:
        failures.append(f"B1 rounds {m['rounds']} phases {m['phases_per_op']}, expected 236 x 1")
    if not close(m["egress_bytes_per_rank"], 236 * 15 * 6144 * 2, 1):
        failures.append("B1 egress must be 236 one-shot rounds of 15 x 12,288 B")
    m8 = roofline.step_model(8, 400)
    if m8["rounds"] != 472 or m8["phases_per_op"] != 2:
        failures.append(f"B8 crosses the RS+AG threshold: rounds {m8['rounds']}, expected 472")
    if not close(m8["expert_keys"], 75 * 256 * (1 - (1 - 8 / 256) ** 8), 1e-6):
        failures.append("B8 expert keys must follow the uniform-routing expectation")
    if roofline.step_model(1, 4096)["bytes_per_rank"] != roofline.step_model(1, 2048)["bytes_per_rank"] + 2048 * roofline.DEFAULTS["kv_index_bytes_per_token"]:
        failures.append("latent KV reads must cap at the DSA selection while index keys grow with context")
    text, ev = roofline.decode_line(1, 45.0, 300)
    for part in ("memory", "compute", "transport bandwidth", "transport latency", "binding: memory", "236 rounds x 40 us"):
        if part not in text:
            failures.append(f"decode line misses '{part}': {text}")
    if not close(ev["percent"]["memory"], 100 * 4.767e9 / 243e9 / 0.045, 0.5):
        failures.append(f"B1 memory percent {ev['percent']['memory']:.1f}")
    if not close(ev["percent"]["transport latency"], 100 * 236 * 0.04 / 45.0, 0.01):
        failures.append("latency percent must be rounds x floor over the measured step")
    pf = roofline.prefill_model(600)
    if pf["chunks"] != 3 or pf["rows"] != 600:
        failures.append(f"prefill of 600 tokens must run 3 dispatches, got {pf['chunks']}")
    ideal = roofline.prefill_ideal(16384)
    if ideal["single_dispatch_binding"] != "transport bandwidth":
        failures.append(f"a 16K one-dispatch prefill at TP16 is bound by bf16 reduce bytes, got {ideal['single_dispatch_binding']}")
    return failures


def fake_results(path):
    b1 = {"label": "C-b1", "session": "b1", "kind": "b1", "tag": "C", "rc": 0, "sequential": True, "failures": [],
          "per_request": [{"label": f"b1-o128-r{i}", "class": None, "prompt_len": 28, "budget": 128, "tokens": 128,
                           "token_ids": [1] * 128, "inter_token_median_ms": ms, "ttft_s": 0.3, "check": "EXACT"}
                          for i, ms in enumerate((44.0, 45.0, 46.0))]}
    b8 = {"label": "C-b8", "session": "b8", "kind": "decode", "tag": "C", "rc": 0, "sequential": False, "failures": [],
          "tok_s_after_last_first_token": 72.0, "aggregate_tok_s": 60.0,
          "per_request": [{"label": f"c{i}", "prompt_len": 300, "budget": 128, "tokens": 128, "token_ids": [2] * 128,
                           "inter_token_median_ms": 110.0, "ttft_s": 1.0, "first_s": 1.0 + i / 10} for i in range(8)]}
    ttft = {"label": "C-ttft1k", "session": "ttft1k", "kind": "ttft", "tag": "C", "rc": 0, "sequential": True, "failures": [],
            "per_request": [{"label": f"t{i}", "prompt_len": 994, "budget": 4, "tokens": 4, "token_ids": [3] * 4,
                             "inter_token_median_ms": 45.0, "ttft_s": 2.0} for i in range(3)]}
    pl = {"label": "C-pl8", "session": "pl8", "kind": "prefill", "tag": "C", "rc": 0, "sequential": False, "failures": [],
          "per_request": [{"label": f"p{i}", "prompt_len": 1524, "budget": 8, "tokens": 8, "token_ids": [4] * 8,
                           "inter_token_median_ms": 100.0, "ttft_s": 5.0 + i, "first_s": 5.0 + i} for i in range(8)]}
    skipped = {"label": "C-b32", "session": "b32", "tag": "C", "rc": None, "skipped": "needs 32 concurrent sequences, arm has 16"}
    r2 = json.loads(json.dumps(b1))
    r2["tag"] = "R"
    r2["per_request"][1]["token_ids"] = [1] * 60 + [9] + [1] * 67
    path.write_text("\n".join(json.dumps(r) for r in (b1, b8, ttft, pl, skipped, r2)) + "\n")


def check_runner():
    failures = []
    capacity = {"sequences": 16, "rows": 16, "positions": 2048}
    if runner.fits({"kind": "decode", "cases": [{"prompt": [0] * 10, "budget": 8}] * 32}, capacity) is None:
        failures.append("a 32-row decode session must be skipped on a 16-sequence arm")
    if runner.fits({"kind": "compsec17", "cases": [{"prompt": [0] * 10, "budget": 8}] * 17}, capacity) is not None:
        failures.append("compsec17 queues 17 requests on 16 sequences and must run")
    if runner.fits({"kind": "ttft", "sequential": True, "cases": [{"prompt": [0] * 4090, "budget": 4}]}, capacity) is None:
        failures.append("a 4K prompt must be skipped on a 2048-position arm")
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results.jsonl"
        fake_results(results)
        if runner.summary(str(results), tmp, "test") != 0:
            failures.append("summary of passing records must exit 0")
        data = json.loads((Path(tmp) / "summary.json").read_text())
        metrics = {(m["tag"], m["metric"]): m for m in data["metrics"]}
        b1 = metrics.get(("C", "B1 decode o128"))
        if not b1 or not close(b1["step_ms"], 45.0, 1e-9) or "binding: memory" not in b1["roofline"]:
            failures.append(f"B1 metric wrong: {b1}")
        b8 = metrics.get(("C", "B8 decode"))
        if not b8 or not close(b8["tok_s"], 8000 / 110.0, 1e-6) or b8["steady_tok_s_after_last_first_token"] != 72.0:
            failures.append(f"B8 metric wrong: {b8}")
        t = metrics.get(("C", "TTFT 994 prompt tokens"))
        if not t or not close(t["prefill_tok_s"], 497.0, 1e-6):
            failures.append(f"TTFT metric wrong: {t}")
        pl = [m for m in data["metrics"] if m["metric"].startswith("prefill under load")]
        if not pl or not close(pl[0]["prefill_tok_s"], 8 * 1524 / 12.0, 1e-6):
            failures.append(f"prefill under load must divide all prompt tokens by the last first token: {pl}")
        if data["skipped"] != ["C b32: needs 32 concurrent sequences, arm has 16"]:
            failures.append(f"skipped list wrong: {data['skipped']}")
        text = (Path(tmp) / "SUMMARY.md").read_text()
        for part in ("## Roofline lines", "transport latency", "## Gates", "C b1: 3 exact"):
            if part not in text:
                failures.append(f"SUMMARY.md misses '{part}'")
        if runner.compare(str(results), "C", "R", ["b1"]) != 1:
            failures.append("compare must report the token difference in R")
    return failures


def check_window_script():
    failures = []
    script = ROOT / "tools" / "glm53full_scorecard" / "window.sh"
    if subprocess.run(["bash", "-n", str(script)]).returncode != 0:
        failures.append("window.sh does not parse")
    body = script.read_text()
    for forbidden in ("--reclaim ", "weightd_warm --reclaim\n", "declare -A"):
        if forbidden in body:
            failures.append(f"window.sh must not contain '{forbidden.strip()}'")
    if "converge glmfull" not in body:
        failures.append("restore must use the rotation tool's converge glmfull")
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "arms"
        bad.write_text("X|/home/x/root|||\n")
        r = subprocess.run(["bash", str(script), "passive", str(bad)], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "SCORECARD_KIT": tmp, "WINDOW_SSH": "false"})
        if r.returncode != 2 or "root must be" not in r.stderr:
            failures.append(f"an arm outside ~/glmfull-lane6/arm.* must be refused: {r.returncode} {r.stderr}")
        bad.write_text("X|arm.x|LD_PRELOAD=/tmp/x.so||\n")
        r = subprocess.run(["bash", str(script), "passive", str(bad)], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "SCORECARD_KIT": tmp, "WINDOW_SSH": "false"})
        if r.returncode != 2 or "bad env" not in r.stderr:
            failures.append(f"a non-SPARK env must be refused: {r.returncode} {r.stderr}")
    return failures


def main():
    failures = check_model() + check_runner() + check_window_script()
    for f in failures:
        print(f"FAIL {f}")
    print("glm53full scorecard:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
