#!/usr/bin/env python3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["config/litellm-config.yaml", "deployment/hub/litellm-chatui.yaml"]
WEEK = 7 * 24 * 3600


def main():
    failures = []
    for name in CONFIGS:
        config = yaml.safe_load((ROOT / name).read_text())
        settings = config.get("litellm_settings") or {}
        general = config.get("general_settings") or {}
        if general.get("cancel_on_disconnect") is not True:
            failures.append(f"{name}: a client that disconnects must cancel the upstream call (general_settings.cancel_on_disconnect)")
        if settings.get("num_retries") != 0:
            failures.append(f"{name}: a cancelled or failed request must not be resent (num_retries 0)")
        if not isinstance(settings.get("request_timeout"), int) or settings["request_timeout"] < WEEK:
            failures.append(f"{name}: request_timeout {settings.get('request_timeout')} would drop a request still waiting in the queue; it must be at least {WEEK} s")
        routes = {entry["model_name"]: entry["litellm_params"] for entry in config["model_list"]}
        for model, params in routes.items():
            if model == "*" or model.endswith("-bulk"):
                continue
            twin = routes.get(model + "-bulk")
            if twin is None or twin.get("model") != params.get("model") + "-bulk" or twin.get("api_base") != params.get("api_base"):
                failures.append(f"{name}: {model} needs a {model}-bulk route to the same chat layer, or bulk requests fall through to the '*' route")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS {len(CONFIGS)} LiteLLM configs cancel upstream on disconnect, never retry, let a request wait in the queue for a week, and route every model's -bulk name to its chat layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
