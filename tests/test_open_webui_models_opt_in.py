#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "deployment" / "hub" / "open_webui_models.json"
LOCAL_FEATURES = {"code_interpreter"}
LOCAL_TOOLS = {"sparkpipe_chat_timer_tool"}


def check(spec):
    failures = []
    metas = [("defaults", spec["defaults"])] + [(entry["id"], entry) for entry in spec["models"]]
    for name, meta in metas:
        for feature in sorted(set(meta.get("defaultFeatureIds") or []) - LOCAL_FEATURES):
            failures.append(f"{name}: defaultFeatureIds turns on {feature} in every new chat; the privacy policy runs web search and tools only when the user turns them on")
        for tool in sorted(set(meta.get("toolIds") or []) - LOCAL_TOOLS):
            failures.append(f"{name}: toolIds attaches {tool} to every new chat; the privacy policy runs tools only when the user turns them on")
    if (spec["defaults"].get("capabilities") or {}).get("web_search") is not True:
        failures.append("defaults: the web_search capability must stay on so a user can turn web search on in a chat")
    return failures


def main():
    spec = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else SPEC).read_text())
    failures = check(spec)
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print(f"PASS test_open_webui_models_opt_in: {len(spec['models'])} SparkPipe Chat models turn on no web search or outbound tool by default; a user turns them on per chat")


if __name__ == "__main__":
    main()
