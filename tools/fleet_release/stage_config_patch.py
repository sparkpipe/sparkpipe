#!/usr/bin/env python3
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

STAGE = "config/stage_{rank:02d}.json"


class ConfigError(Exception):
    pass


def render(obj):
    return json.dumps(obj, indent=1)


def flat(obj, prefix=""):
    out = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = json.dumps(value)
    return out


def manifest(path):
    rows = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(None, 1)
        rows[name.strip()] = digest
    return rows


def parse_add(items):
    added = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"--add {item}: want KEY=JSON")
        try:
            added.append((key, json.loads(value)))
        except ValueError:
            raise ConfigError(f"--add {item}: value is not JSON")
    if len({k for k, _ in added}) != len(added):
        raise ConfigError("--add names a member twice")
    return added


def appended_exactly(current, new, added):
    keys = [k for k, _ in added]
    return (list(new)[:len(current)] == list(current)
            and list(new)[len(current):] == keys
            and {k: v for k, v in new.items() if k in current} == current
            and all(new[k] == v for k, v in added))


def patch(production, output, world, added, generated=None, committed=None, drift=None, out=print):
    rows = manifest(production / "MANIFEST")
    if output.exists():
        raise ConfigError(f"{output} exists")
    (output / "config").mkdir(parents=True)
    fail = 0
    drift = drift or {}
    drift_seen = {}
    for rank in range(world):
        name = STAGE.format(rank=rank)
        data = (production / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != rows.get(name):
            out(f"FAIL {name}: production copy does not match the production MANIFEST")
            fail += 1
            continue
        current = json.loads(data)
        if render(current).encode() != data:
            out(f"FAIL {name}: production file is not json.dumps(indent=1) without a trailing newline; refusing to re-render")
            fail += 1
            continue
        if current.get("tp_rank") != rank or current.get("tp_degree") != world:
            out(f"FAIL {name}: tp_rank/tp_degree {current.get('tp_rank')}/{current.get('tp_degree')} want {rank}/{world}")
            fail += 1
            continue
        present = [k for k, _ in added if k in current]
        if present:
            out(f"FAIL {name}: production already carries {', '.join(present)}")
            fail += 1
            continue
        new = dict(current)
        for key, value in added:
            new[key] = value
        text = render(new)
        (output / name).write_text(text)
        if not appended_exactly(current, json.loads(text), added):
            out(f"FAIL {name}: change vs production is not exactly the appended members")
            fail += 1
        note = ""
        if generated is not None:
            if committed is not None and committed[rank] != generated[rank]:
                out(f"FAIL {name}: committed deployment differs from the generator output")
                fail += 1
            fb, fg = flat(new), flat(generated[rank])
            differ = [k for k in sorted(set(fb) | set(fg)) if fb.get(k, "absent") != fg.get(k, "absent")]
            for k in differ:
                drift_seen.setdefault(k, set()).add((fb.get(k, "absent"), fg.get(k, "absent")))
            unexplained = [k for k in differ if k not in drift]
            if unexplained:
                out(f"FAIL {name}: differs from the generator in unexplained members {unexplained}")
                fail += 1
            note = f"  vs generator: {len(differ)} members differ ({', '.join(differ)})"
        out(f"OK   {name} prod {hashlib.sha256(data).hexdigest()[:16]} -> new {hashlib.sha256(text.encode()).hexdigest()[:16]}  vs production: " + " ".join(f"+{k}={json.dumps(v)}" for k, v in added) + note)
    if generated is not None:
        out("== members where the new config differs from the generator (value new | generator), all ranks")
        for k, pairs in sorted(drift_seen.items()):
            vals = "; ".join(f"{a} | {b}" for a, b in sorted(pairs))
            out(f"  {k}: {vals}  -- {drift.get(k, 'UNEXPLAINED')}")
    out("STAGE-CONFIGS " + ("PASS" if fail == 0 else f"FAIL {fail}") + f" output {output}")
    return fail


def check(current_dir, new_dir, world, added, out=print):
    fail = 0
    for rank in range(world):
        name = STAGE.format(rank=rank)
        a = json.loads((current_dir / name).read_text())
        text = (new_dir / name).read_text()
        b = json.loads(text)
        ok = appended_exactly(a, b, added) and render(b) == text
        if not ok:
            out(f"FAIL {name}: not the served config plus exactly " + " ".join(k for k, _ in added))
            fail += 1
    out("STAGE-CONFIG-CHECK " + ("PASS" if fail == 0 else f"FAIL {fail}") + f" ({world} ranks, +" + " +".join(f"{k}={json.dumps(v)}" for k, v in added) + ")")
    return fail


def run_generator(command, world):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(shlex.split(command) + ["--output", tmp], check=True, capture_output=True)
        return {r: json.loads((Path(tmp) / STAGE.format(rank=r)).read_text()) for r in range(world)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Append members to every served stage config (byte format kept) and explain every difference from the generator.")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("patch")
    p.add_argument("--production", required=True, help="copy of the served root with its MANIFEST")
    p.add_argument("--output", required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--add", action="append", required=True, help="KEY=JSON, appended in this order")
    p.add_argument("--generator", help="command that writes config/stage_NN.json under --output DIR")
    p.add_argument("--committed", help="committed deployment directory the generator must reproduce")
    p.add_argument("--drift", help="JSON object: member -> why production differs from the generator")
    c = sub.add_parser("check")
    c.add_argument("--current", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--world", type=int, required=True)
    c.add_argument("--add", action="append", required=True)
    args = ap.parse_args(argv)
    try:
        added = parse_add(args.add)
        if args.command == "check":
            return 1 if check(Path(args.current), Path(args.new), args.world, added) else 0
        if args.committed and not args.generator:
            raise ConfigError("--committed needs --generator")
        generated = run_generator(args.generator, args.world) if args.generator else None
        committed = None
        if args.committed:
            committed = {r: json.loads((Path(args.committed) / STAGE.format(rank=r)).read_text()) for r in range(args.world)}
        drift = None
        if args.drift:
            with open(args.drift) as f:
                drift = json.load(f)
        return 1 if patch(Path(args.production), Path(args.output), args.world, added, generated, committed, drift) else 0
    except (OSError, ValueError, ConfigError, subprocess.CalledProcessError) as e:
        print(f"stage_config_patch.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
