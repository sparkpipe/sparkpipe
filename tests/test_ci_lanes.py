#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ci_lanes

LANE_TOKENS = re.compile(r"""(?<![\w.-])site(?:/|["']\s*[,)/])|sparkpipe_pay|sparkpipe_cli|site_terms|site_kr_networks|brand_logo|hyor-io\.conf|hyor\.io|sparkpipe-ai(?:-ssl|\.conf)|sparkpipe-chat-local\.conf|sparkpipe-lang\.conf|sparkpipe-pay\.|rspamd-(?:dkim_signing|actions|redis)|postfix-ma(?:in|ster)\.cf|dovecot-(?:10-auth|99-sparkpipe)|certbot-reload-mail|PAYMENTS\.md|LITELLM_FRONTEND|docs/brand""")
WHOLE_TREE = re.compile(r"""\bROOT\s*\.rglob\(|os\.walk\(\s*ROOT\b|["']ls-files["']|\(\s*["']tests["']\s*,\s*["']tools["']""")


def check_classification(failures):
    cases = [
        (["site/buy.html", "site/terms/agreement-en.md", "tools/sparkpipe_pay.py", "tests/test_pay.py", "docs/PAYMENTS.md"], False),
        (["deployment/vps/hyor.io/index.html", "deployment/vps/sparkpipe-pay.service", "docs/brand/hyor-ai/README.md"], False),
        (["deployment/vps/postfix-main.cf", "deployment/vps/dovecot-99-sparkpipe.conf", "deployment/vps/certbot-reload-mail-services.sh", "docs/VPS_MAIL.md"], False),
        (["deployment/vps/sparkpipe-ai.conf", "deployment/vps/sparkpipe-chat-local.conf"], False),
        ([], False),
        (["site/buy.html", "inference/kernels/x.cu"], True),
        (["README.md"], False),
        (["TECHDEBT.md", "docs/archive/K3_PACK_FORMAT_V2.md", "site/buy.html"], False),
        (["README.md", "inference/kernels/x.cu"], True),
        (["Makefile"], True),
        (["deployment/vps/chat-local/index.html"], True),
        (["deployment/vps/sparkpipe-public-api.conf"], True),
        (["tools/sparkpipe_cli.py", "tools/sparkpipe_cli_build.py", "tests/test_sparkpipe_cli.py"], False),
        (["tools/client_release_check.py", "tests/test_client_release_check.py"], False),
        (["tools/sparkpipe_cli.py", "tools/serving/chat_frontend.py"], True),
        (["tools/ci_lanes.py"], False),
        (["tests/test_ci_lanes.py", "tools/ci_lanes.py"], False),
        (["LICENSE", "LICENSE-APACHE-2.0", "LICENSE-PERMISSION.md", "NOTICE", "CLA.md", "CONTRIBUTING.md"], False),
        ([".gitignore", ".gitattributes", ".updaterepo-policy", ".github/pull_request_template.md", ".github/ISSUE_TEMPLATE/bug.md", ".github/CODEOWNERS"], False),
        (["docs/images/fleet.svg", "docs/WORKING_STATE.md", "SPEC.md"], False),
        (["LICENSE", "site/index.html", "README.md"], False),
        (["LICENSE", "Makefile"], True),
        (["tools/ci_lanes.py", "inference/kernels/x.cu"], True),
        (["tools/serving/chat_frontend.py"], True),
        (["model-families/k3/README.md"], False),
        (["newdir/x.c"], True),
        ([".github/workflows/cuda13-sm121a-compile.yml"], True),
        (["website/site/x.html"], True),
        (None, True),
    ]
    site = ci_lanes.light_tests(["site/buy.html"])
    if "tests/test_site.py" not in site or "tests/test_ci_lanes.py" not in site or "tests/test_status_truth.py" in site:
        failures.append(f"a site change runs the wrong light tests: {site}")
    repo = ci_lanes.light_tests(["LICENSE", "NOTICE", "tools/ci_lanes.py"])
    for name in ("tests/test_ci_lanes.py", "tests/test_third_party_notices.py", "tests/test_package_manifest.py"):
        if name not in repo:
            failures.append(f"a repository-file change does not run {name}")
    if ci_lanes.light_tests(["docs/images/fleet.svg"]) != ci_lanes.light_tests(["docs/WORKING_STATE.md"]):
        failures.append("a non-Markdown file under docs/ does not run the Markdown readers")
    readme = ci_lanes.light_tests(["README.md"])
    for name in ("tests/test_status_truth.py", "tests/test_lane_records.py", "tests/test_open_webui_github_tool.py", "tests/test_site.py"):
        if name not in readme:
            failures.append(f"a README change does not run {name}")
    for paths, engine in cases:
        if ci_lanes.engine_needed(paths) != engine:
            failures.append(f"{paths}: engine lane {'skipped' if engine else 'run'}")
    if ci_lanes.changed("0" * 40, "HEAD") is not None or ci_lanes.changed("", "HEAD") is not None:
        failures.append("a push without a previous commit does not fall back to the engine lane")


def check_every_file_has_a_lane(failures):
    run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True)
    if run.returncode != 0:
        failures.append(f"git ls-files exited {run.returncode}")
        return 0
    files = [entry for entry in run.stdout.decode("utf-8", "surrogateescape").split("\0") if entry]
    loose = sorted({path.split("/")[0] for path in files if ci_lanes.lane(path) == "unclassified"})
    for top in loose:
        failures.append(f"{top}: no lane; add it to ENGINE_PATHS if the engine build or an engine test reads it, otherwise to LIGHT_PATHS")
    return len(files)


def check_fast_tests_exist(failures):
    for name in ci_lanes.FAST_TESTS:
        if not (ROOT / name).is_file():
            failures.append(f"{name}: in the fast lane but missing")


def check_engine_tests_ignore_site(failures):
    fast = set(ci_lanes.light_tests([]))
    for path in sorted(list((ROOT / "tests").glob("*.py")) + list((ROOT / "tests").glob("*.sh"))):
        name = path.relative_to(ROOT).as_posix()
        if name in fast:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        found = LANE_TOKENS.search(source)
        if found:
            failures.append(f"{name} reads site-lane file {found.group(0)!r}: add it to FAST_TESTS or take the file out of SITE_PATHS")
        found = WHOLE_TREE.search(source)
        if found:
            failures.append(f"{name} scans the whole tree ({found.group(0)!r}): add it to FAST_TESTS")


def check_build_ignores_site(failures):
    build = [ROOT / "Makefile"] + sorted(ROOT.glob("*.mk")) + sorted((ROOT / "tools").glob("*.sh")) + sorted((ROOT / "tools" / "ci").glob("*"))
    for path in build:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found = LANE_TOKENS.search(text)
        if found:
            failures.append(f"{path.relative_to(ROOT)} builds from site-lane file {found.group(0)!r}")
        if path.name == "Makefile" or path.suffix == ".mk":
            found = re.search(r"[A-Za-z0-9_./-]+\.md\b", text)
            if found:
                failures.append(f"{path.relative_to(ROOT)} builds from Markdown {found.group(0)!r}: Markdown changes skip the engine lane")


NODE_MEMORY_ENVELOPE_BYTES = 128 * (1024 ** 3)
SIZE_SUFFIX = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
PROVENANCE_REQUIRED_FIELDS = (
    "lane",
    "drafter_id",
    "upstream_repo",
    "revision",
    "license",
    "verified_pack_format",
)


def parse_memory_bytes(value):
    if isinstance(value, int) and value > 0:
        return value
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT])?", value.strip())
    if match is None:
        return None
    return int(match.group(1)) * SIZE_SUFFIX.get(match.group(2) or "", 1)


def check_drafter_provenance(failures):
    lanes_doc = json.loads((ROOT / "deployment" / "lanes.json").read_text(encoding="utf-8"))
    engine_doc = json.loads((ROOT / "deployment" / "hub" / "draftd_engine.json").read_text(encoding="utf-8"))
    provenance_path = ROOT / "deployment" / "drafter_provenance.json"
    if not provenance_path.is_file():
        failures.append("deployment/drafter_provenance.json is missing")
        return 0
    provenance_doc = json.loads(provenance_path.read_text(encoding="utf-8"))
    entries = provenance_doc.get("drafters")
    if not isinstance(entries, list) or not entries:
        failures.append("deployment/drafter_provenance.json must define a non-empty 'drafters' list")
        return 0
    by_lane = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"deployment/drafter_provenance.json entry #{idx} is not an object")
            continue
        lane_name = entry.get("lane")
        for field in PROVENANCE_REQUIRED_FIELDS:
            val = entry.get(field)
            if not isinstance(val, str) or not val.strip():
                failures.append(f"deployment/drafter_provenance.json lane {lane_name!r}: missing or empty {field!r}")
        if isinstance(lane_name, str) and lane_name.strip():
            if lane_name in by_lane:
                failures.append(f"deployment/drafter_provenance.json duplicate lane {lane_name!r}")
            by_lane[lane_name] = entry
    for lane_name, lane_cfg in sorted(lanes_doc.get("lanes", {}).items()):
        if "draftd" in lane_cfg or "draft_server" in lane_cfg:
            if lane_name not in by_lane:
                failures.append(
                    f"lane {lane_name!r} configures draftd/draft_server in deployment/lanes.json "
                    f"but has no entry in deployment/drafter_provenance.json"
                )
    for engine_lane in engine_doc.get("lanes", []):
        lane_name = engine_lane.get("name")
        drafter = engine_lane.get("drafter", {})
        if lane_name not in by_lane:
            failures.append(
                f"draftd_engine.json lane {lane_name!r} has no entry in deployment/drafter_provenance.json"
            )
            continue
        rec = by_lane[lane_name]
        if drafter.get("model") and rec.get("drafter_id") != drafter.get("model"):
            failures.append(
                f"drafter_provenance.json lane {lane_name!r}: drafter_id={rec.get('drafter_id')!r} "
                f"!= draftd_engine.json model={drafter.get('model')!r}"
            )
        if drafter.get("revision") and rec.get("revision") != drafter.get("revision"):
            failures.append(
                f"drafter_provenance.json lane {lane_name!r}: revision={rec.get('revision')!r} "
                f"!= draftd_engine.json revision={drafter.get('revision')!r}"
            )
    return len(by_lane)


def check_lane_memory_plan(failures):
    import stage_lane

    lanes_doc = json.loads((ROOT / "deployment" / "lanes.json").read_text(encoding="utf-8"))
    runner_src = (ROOT / "inference" / "runner" / "stage_runner.cu").read_text(encoding="utf-8")
    m_max = re.search(r"#define\s+STAGE_RUNNER_GRAPH_MAX_ACTIVE_DEFAULT\s+(\d+)u", runner_src)
    m_bytes = re.search(r"#define\s+STAGE_RUNNER_GRAPH_MAX_BYTES_DEFAULT\s+\((\d+)ull\s*<<\s*(\d+)\)", runner_src)
    if not m_max or int(m_max.group(1)) > 16:
        failures.append(f"stage_runner.cu STAGE_RUNNER_GRAPH_MAX_ACTIVE_DEFAULT must be <= 16 (got {m_max.group(1) if m_max else None})")
    graph_budget = (int(m_bytes.group(1)) << int(m_bytes.group(2))) if m_bytes else 0
    if graph_budget <= 0 or graph_budget > 2 * (1024 ** 3):
        failures.append(f"stage_runner.cu STAGE_RUNNER_GRAPH_MAX_BYTES_DEFAULT must be in (0, 2 GiB] (got {graph_budget})")
    weightd = lanes_doc.get("weightd", {})
    device_max = weightd.get("device_bytes_max", 0)
    kv_reserve = weightd.get("kv_reserve_bytes", 0)
    if device_max <= 0 or kv_reserve <= 0:
        failures.append(f"deployment/lanes.json weightd must set positive device_bytes_max and kv_reserve_bytes (got {weightd})")
        return 0
    if device_max + kv_reserve > NODE_MEMORY_ENVELOPE_BYTES:
        failures.append(
            f"deployment/lanes.json weightd device_bytes_max ({device_max / 1024**3:.1f} GiB) + "
            f"kv_reserve_bytes ({kv_reserve / 1024**3:.1f} GiB) exceeds 128 GiB node envelope"
        )
    lanes = lanes_doc.get("lanes", {})
    for lane_name, lane_cfg in sorted(lanes.items()):
        state_budget = lane_cfg.get("state_budget_bytes")
        state_memory = lane_cfg.get("state_memory_bytes", 0)
        mem_max = parse_memory_bytes(lane_cfg.get("memory_max"))
        model = stage_lane.MODELS.get(lane_cfg.get("model"))
        if not isinstance(state_budget, int) or state_budget <= 0:
            failures.append(f"lane {lane_name!r}: invalid state_budget_bytes={state_budget!r}")
            continue
        if mem_max is None:
            failures.append(f"lane {lane_name!r}: invalid memory_max={lane_cfg.get('memory_max')!r}")
            continue
        if model is None or model.get("node_memory_bytes", 0) <= 0:
            failures.append(f"lane {lane_name!r}: unknown model {lane_cfg.get('model')!r} or missing node_memory_bytes")
            continue
        node_mem = model["node_memory_bytes"]
        full_total = node_mem + kv_reserve + state_budget + state_memory + graph_budget
        if full_total > NODE_MEMORY_ENVELOPE_BYTES:
            failures.append(
                f"lane {lane_name!r} full footprint exceeds 128 GiB node envelope: "
                f"weights={node_mem / 1024**3:.2f} GiB + kv_reserve={kv_reserve / 1024**3:.1f} GiB + "
                f"state_budget={state_budget / 1024**3:.1f} GiB + state_memory={state_memory / 1024**3:.2f} GiB + "
                f"graph_budget={graph_budget / 1024**3:.2f} GiB = {full_total / 1024**3:.2f} GiB > 128.0 GiB"
            )
        total = state_budget + mem_max + kv_reserve
        if total > NODE_MEMORY_ENVELOPE_BYTES:
            failures.append(
                f"lane {lane_name!r} exceeds 128 GiB node envelope: "
                f"state_budget={state_budget / 1024**3:.1f} GiB + "
                f"memory_max={mem_max / 1024**3:.1f} GiB + "
                f"kv_reserve={kv_reserve / 1024**3:.1f} GiB = "
                f"{total / 1024**3:.1f} GiB > {NODE_MEMORY_ENVELOPE_BYTES / 1024**3:.1f} GiB"
            )
        if state_budget + kv_reserve > device_max:
            failures.append(
                f"lane {lane_name!r} state_budget ({state_budget / 1024**3:.1f} GiB) + "
                f"kv_reserve ({kv_reserve / 1024**3:.1f} GiB) exceeds "
                f"weightd.device_bytes_max ({device_max / 1024**3:.1f} GiB)"
            )
    return len(lanes)


def main():
    failures = []
    check_classification(failures)
    tracked = check_every_file_has_a_lane(failures)
    check_fast_tests_exist(failures)
    check_engine_tests_ignore_site(failures)
    check_build_ignores_site(failures)
    provenance_count = check_drafter_provenance(failures)
    memory_plan_lanes = check_lane_memory_plan(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print(f"ci lanes: only the {len(ci_lanes.ENGINE_PATHS)} engine paths run the engine lane, and each of the {tracked} tracked files has a lane; {len(ci_lanes.SITE_PATHS)} site paths, {len(ci_lanes.LIGHT_PATHS)} repository paths and Markdown skip it; no engine test or build rule reads the site paths; {len(ci_lanes.light_tests([]))} light tests cover the site and every whole-tree scan, and Markdown changes add the {len(ci_lanes.markdown_tests())} tests that can read Markdown; drafter provenance covers {provenance_count} drafters and memory plan fits {memory_plan_lanes} lanes inside 128 GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
