#!/usr/bin/env python3
"""
cross_model_audit.py — flag knobs where OUR OWN models disagree with each other,
or with a vendored reference recipe, for no recorded reason.

★★ WHY THIS EXISTS (2026-08-04)
   Our GLM recipe has used `draft_sample_method: probabilistic` for weeks. Our DeepSeek
   launcher used `greedy`. That single divergence is the leading explanation for DeepSeek
   running at 27-48% spec-decode acceptance instead of ~80%, and it sat there unnoticed
   while a full day was spent tuning TP size, vLLM versions and KV caps.

   It was not that the fact was unknown. It was recorded TWICE — as CONFIGURATION:
     memory/dynamic-speculative-decoding.md:61  ... "draft_sample_method":"greedy" ...
     memory/glm52-0xdfi-1m-comparison.md:22     ... probabilistic draft ...
   A config snapshot has no cross-model reach. Nothing linked the parameter to what it
   DOES, so nothing fired when the other model's launcher was written.

   Prose cannot fix that; a doc is only read when you already suspect something. This
   is a CHECK. It reads every recipe/launcher we own plus the vendored references, and
   prints the knobs where we disagree with ourselves. Run it before any launch change.

USAGE
    python3 cross_model_audit.py            # table + divergences
    python3 cross_model_audit.py --strict   # exit 1 if any UNEXPLAINED divergence
"""
import argparse, json, os, re, sys
from pathlib import Path

HOME = Path.home()
DS = HOME / "Desktop/DEEPSEEK-V4-RESTORE-BUNDLE"
GLM = HOME / "Desktop/GLM52-RESTORE-BUNDLE"

# Knobs that mean the SAME THING across models, with what they actually do. A knob only
# belongs here if a wrong value is silently costly — that is what makes divergence a bug
# rather than a legitimate difference.
KNOBS = {
    "draft_sample_method": (
        "How the DRAFT model samples. `greedy` = point mass, so spec-decode acceptance "
        "collapses to p_target(argmax) — flat across draft positions, and BAD whenever "
        "the target samples at temperature > 0. `probabilistic` matches the target's "
        "distribution. THE 2026-08-04 BUG."),
    "num_speculative_tokens": ("draft depth k. tok/step ceiling is k+1."),
    "kv_cache_dtype": ("KV bytes/token. nvfp4_ds_mla << fp8_ds_mla."),
    "generation_config": ("`vllm` IGNORES the checkpoint's generation_config.json, so a "
                          "client omitting temperature gets vLLM's default 1.0."),
    "gpu_memory_utilization": ("fraction of unified memory vLLM may claim."),
    "max_cudagraph_capture_size": ("graph memory; competes directly with the KV pool."),
    "moe_backend": ("MoE kernel. flashinfer_b12x is BANNED on sm_121 for GLM."),
}

# Divergences that ARE deliberate, with the reason. Anything NOT here is unexplained —
# which is the exact category the greedy-vs-probabilistic miss lived in for weeks.
DELIBERATE = {
    "moe_backend": "GLM must use flashinfer_cutlass — flashinfer_b12x is BANNED on sm_121 "
                   "for GLM (standing rule e). DeepSeek correctly uses flashinfer_b12x.",
    "num_speculative_tokens": "GLM k=2 measured better than k=4/5 on that model; DeepSeek "
                              "runs a batch-aware ladder 7/5/3. Model-specific, measured.",
}

# ★ EDIT THIS FOR YOUR OWN FLEET. These paths are the author's; nothing else in this file
# is site-specific. A "source" is any file that declares serve flags — a launcher script, a
# recipe YAML, a vendored reference config. The audit reads each, extracts the KNOBS below,
# and flags where they DISAGREE with no recorded reason. It needs at least two sources to be
# useful, and it is most useful when they are DIFFERENT MODELS on the same hardware — that is
# the case where a knob silently diverges and nobody notices.
# The live-container row needs no editing: it comes from DSFV4_SSH_HOST / DSFV4_CONTAINER.
SOURCES = [
    ("deepseek/live-tp4", DS / "scripts/launch_rank_tp4.sh"),
    ("deepseek/port-026", DS / "scripts/launch_tp4_port026.sh"),
    ("glm/longctx-v2", GLM / "recipe/glm-dcp2-v2-longctx.yaml"),
    ("glm/speed128k-v2", GLM / "recipe/glm-dcp2-v2-speed128k.yaml"),
    ("ref:jvr0x/ds-dual-tp2", DS / "recipe/jvr0x/deepseek-v4-flash-0731-dual.yaml"),
]


def extract(path: Path) -> dict:
    if not path.exists():
        return {}
    t = path.read_text(errors="replace")
    out = {}
    for k in KNOBS:
        # JSON form inside --speculative-config, then CLI flag form, then YAML key form
        m = (re.search(rf'"{k}"\s*:\s*"([^"]+)"', t)
             or re.search(rf'"{k}"\s*:\s*([0-9.]+)', t)
             or re.search(rf'--{k.replace("_", "-")}[= ]+[\'"]?([^\s\'"\\]+)', t)
             or re.search(rf'^\s*{k}:\s*[\'"]?([^\'"\n]+)', t, re.M))
        if m:
            out[k] = m.group(1).strip()
    return out


def live_container(host=None, name=None):
    """Reads the RUNNING config over ssh. Host/container from env so this file carries
    no site-specific names:  export DSFV4_SSH_HOST=<host>  DSFV4_CONTAINER=<name>"""
    host = host or os.environ.get("DSFV4_SSH_HOST", "localhost")
    name = name or os.environ.get("DSFV4_CONTAINER", "vllm-head")
    """Read the RUNNING config. Launchers hold $VARS; only the live container has values."""
    import subprocess
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                            f'docker inspect {name} --format "{{{{json .Args}}}}"'],
                           capture_output=True, text=True, timeout=30)
        args = json.loads(r.stdout.strip())
    except Exception:
        return {}
    blob = " ".join(args)
    out = {}
    for k in KNOBS:
        m = (re.search(rf'"{k}"\s*:\s*"([^"]+)"', blob)
             or re.search(rf'--{k.replace("_", "-")}\s+([^\s]+)', blob))
        if m:
            out[k] = m.group(1).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    data = {name: extract(p) for name, p in SOURCES}
    live = live_container()
    if live:
        data["LIVE(running)"] = live
    missing = [n for n, p in SOURCES if not p.exists()]
    names = [n for n, _ in SOURCES if n not in missing]
    if live: names.insert(0, "LIVE(running)")

    w = max(len(k) for k in KNOBS) + 2
    print(f"\n  {'knob':<{w}}" + "".join(f"{n[:21]:<23}" for n in names))
    print("  " + "-" * (w + 23 * len(names)))
    divergences = []
    for k in KNOBS:
        vals = {n: data[n].get(k, "—") for n in names}
        real = {v for v in vals.values() if v != "—"}
        flag = "  " if len(real) <= 1 else "! "
        if len(real) > 1:
            divergences.append((k, vals))
        print(f"  {flag}{k:<{w-2}}" + "".join(f"{vals[n][:21]:<23}" for n in names))

    if missing:
        print(f"\n  (not found, skipped: {', '.join(missing)})")

    if not divergences:
        print("\n  ✅ no cross-model divergence on the audited knobs.\n")
        return 0

    unexplained = [(k, v) for k, v in divergences if k not in DELIBERATE]
    known = [(k, v) for k, v in divergences if k in DELIBERATE]
    if known:
        print(f"\n  {len(known)} DELIBERATE divergence(s), reason on file:")
        for k, _ in known:
            print(f"    {k}: {DELIBERATE[k]}")
    if not unexplained:
        print("\n  ✅ no UNEXPLAINED divergence.\n")
        return 0
    print(f"\n  ⚠ {len(unexplained)} UNEXPLAINED knob(s) where our own configs disagree:\n")
    for k, vals in unexplained:
        print(f"  ── {k}")
        print(f"     {KNOBS[k]}")
        for n, v in vals.items():
            if v != "—":
                print(f"       {n:<24} {v}")
        print()
    print("  Each divergence is either a DELIBERATE, model-specific choice — in which case")
    print("  record WHY next to it — or it is a bug that has been sitting there unread.")
    print("  A value that differs across our own models with no stated reason is the exact")
    print("  shape of the greedy-vs-probabilistic miss.\n")
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
