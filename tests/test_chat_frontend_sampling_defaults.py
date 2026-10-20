#!/usr/bin/env python3
"""A chat request that names no sampling fields gets the model's published defaults instead of greedy decoding, explicit fields win one by one, temperature 0 stays greedy, and every deployed model names its defaults and where they come from."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
RequestError = chat_frontend.RequestError

DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def main():
    failures = []
    fields = chat_frontend.sampling_fields
    cases = [
        ("no sampling fields", {}, {"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
        ("an explicit greedy request", {"temperature": 0}, {}),
        ("an explicit temperature keeps the default truncation", {"temperature": 0.5}, {"temperature": 0.5, "top_p": 0.95, "top_k": 20}),
        ("an explicit top_p", {"top_p": 0.8}, {"temperature": 1.0, "top_p": 0.8, "top_k": 20}),
        ("an explicit top_k of -1 turns top-k off", {"top_k": -1}, {"temperature": 1.0, "top_p": 0.95}),
        ("a null temperature means the default", {"temperature": None}, {"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
    ]
    for label, body, expected in cases:
        got = fields(body, DEFAULTS)
        if got != expected:
            failures.append(f"{label}: got {got!r}, expected {expected!r}")
    if fields({}, {}) != {}:
        failures.append("a model without defaults no longer decodes greedily by default")
    base = {"id": "fake", "model_dir": ".", "context_tokens": 1024, "max_output_tokens": 1024, "engine": "http://127.0.0.1:1"}
    for label, extra in (("an unknown key", {"sampling_defaults": {"min_p": 0.1}, "sampling_source": "card"}),
                         ("no source", {"sampling_defaults": {"temperature": 1.0}}),
                         ("an out-of-range temperature", {"sampling_defaults": {"temperature": 3}, "sampling_source": "card"})):
        spec = dict(base, **extra)
        try:
            chat_frontend.Model(spec, ROOT)
            failures.append(f"a config with {label} loaded")
        except SystemExit:
            pass
        except Exception as error:
            failures.append(f"a config with {label} got past the sampling check ({type(error).__name__})")
    for model in json.loads((ROOT / "deployment" / "hub" / "chat_frontend.json").read_text())["models"]:
        if not model.get("sampling_defaults") or not str(model.get("sampling_source", "")).strip():
            failures.append(f"deployment/hub/chat_frontend.json {model['id']} has no sampling_defaults or sampling_source")
    for line in failures:
        print("FAIL " + line)
    if failures:
        return 1
    print("PASS test_chat_frontend_sampling_defaults: unset sampling fields take the model's published defaults, explicit fields win, temperature 0 stays greedy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
