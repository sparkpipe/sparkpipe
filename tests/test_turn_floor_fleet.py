#!/usr/bin/env python3
"""scripts/turn_floor_fleet.sh walks served models through tools/turn_floor.py on the hub. With TF_DRY=1 it reaches no host and prints what it would do, which is what this test reads: each model's lane record, its steps in order, and what the driver refuses."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "turn_floor_fleet.sh"
failures = []


def check(condition, what):
    if not condition:
        failures.append(what)


def plan(*models, **environment):
    values = dict(os.environ, TF_DRY="1", TF_RUN="tf-test", TF_OUT="/nowhere", TF_TOKENS="/hub/tokens", HOME="/nonexistent-home")
    values.update(environment)
    for name in [name for name, value in values.items() if value is None]:
        del values[name]
    done = subprocess.run(["bash", str(SCRIPT), *models], capture_output=True, text=True, timeout=120, env=values, cwd=str(ROOT))
    return done.returncode, done.stdout, done.stderr


code, out, _ = plan("kimi-k3=k3=8000,100000", "deepseek-v4.1-flash=dsv41=8000,100000,300000=turns,wave,openings,restart")
lines = out.splitlines()
steps = [line.split(" tf ", 1)[1] for line in lines if line.startswith("== ") and " tf " in line]
check(code == 0 and steps[0].startswith("kimi-k3 (lane ") and "port 8459" in steps[0] and "levels 8000,100000, arms turns,wave,restart" in steps[0],
      f"a model's line names its lane record, its levels and the default arms: {steps[:1]}")
check(steps[1:6] == ["kimi-k3 turns", "kimi-k3 wave", "kimi-k3 restart: stop and start", "kimi-k3 restart", "kimi-k3 parts"],
      f"a model runs its turns, the wave wait, a stop and start with one turn after it, then the split: {steps[1:6]}")
check([step for step in steps if step.startswith("deepseek-v4.1-flash ")][1:] ==
      ["deepseek-v4.1-flash turns", "deepseek-v4.1-flash wave", "deepseek-v4.1-flash openings", "deepseek-v4.1-flash restart: stop and start",
       "deepseek-v4.1-flash restart", "deepseek-v4.1-flash parts"], "the openings run only for a model whose arms name them")
hub = [line.strip() for line in lines if line.strip().startswith("hub: ")]
turns = next(line for line in hub if "--arms turns" in line and "kimi-k3" in line)
check("cd /srv/workspace/sparkpipe/turn-floor/tf-test/kimi-k3 && python3 ../turn_floor.py run --endpoint http://127.0.0.1:8459 " in turns and
      "--tokens /hub/tokens/tokens-k3.txt --api-log /srv/workspace/sparkpipe/logs/k3-kv-api.log --label kimi-k3 --output . --contexts 8000,100000 " in turns and
      "--tails 1,40,500,5000 --repeats 3 --spec-repeats 0 --once-from 2000 --decode-tokens 96 --served-tokens 128 --wave-tokens 20000 --arms turns" in turns,
      f"the turns step sends the engine's own port, log and token file with the plan's tails, the long tail once and both decode samples: {turns}")
check(any("--arms resume --event restart" in line for line in hub) and any(line.endswith("turn_floor.py parts .") for line in hub) and
      any(line.endswith("turn_floor.py excerpt . --output engine.log") for line in hub), "the stop and start is followed by the resume arm, and a model ends with its split and its log excerpt")
controller = [line.strip() for line in lines if line.strip().startswith("controller: ")]
check(controller[:3] == ["controller: activate kimi-k3 unless it is active", "controller: deactivate, then activate kimi-k3", "controller: activate kimi-k3 unless it is active"],
      f"the model is changed only through the swap controller: {controller[:3]}")
check(any(line.strip().startswith("copy: hub:/srv/workspace/sparkpipe/turn-floor/tf-test/kimi-k3/") and line.strip().endswith(" to /nowhere/kimi-k3/") for line in lines),
      "the receipts are copied to the folder the caller named")

code, out, _ = plan("kimi-k3=k3=8000", TF_TAILS="50,300", TF_REPEATS="2", TF_ONCE_FROM="0", TF_SERVED="0", TF_WAVE="6000")
check(code == 0 and "--tails 50,300 --repeats 2 --spec-repeats 0 --once-from 0 --decode-tokens 96 --served-tokens 0 --wave-tokens 6000" in out, "the plan's sizes can be set by the caller")
code, out, _ = plan("kimi-k3=k3=8000=turns,bogus")
check(code == 1 and "TF-FAILED kimi-k3: unknown arm 'bogus'" in out and "hub:" not in out, f"an unknown arm stops the model before anything runs: {out[-200:]}")
code, out, _ = plan("no-such-model=k3=8000", "kimi-k3=k3=8000")
check(code == 1 and "TF-FAILED no-such-model: no lane record" in out and "kimi-k3" not in out, "a model without a lane record stops the run, and no later model starts")
code, out, _ = plan("kimi-k3=k3")
check(code == 1 and "a model is MODEL=FAMILY=LEVELS[=ARMS]" in out, "a model needs its token family and its levels")
code, _, error = plan("kimi-k3=k3=8000", TF_RUN=None)
check(code != 0 and "TF_RUN names this run" in error, "the run must be named")
code, _, error = plan()
check(code == 2 and "usage:" in error, "no model prints the usage")

for failure in failures:
    print("FAIL test_turn_floor_fleet: " + failure)
if failures:
    sys.exit(1)
print("ok test_turn_floor_fleet")
