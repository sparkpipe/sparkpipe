#!/usr/bin/env python3
import asyncio
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
FIELDS = importlib.util.spec_from_file_location("request_fields", ROOT / "tests" / "test_chat_frontend_request_fields.py")
request_fields = importlib.util.module_from_spec(FIELDS)
FIELDS.loader.exec_module(request_fields)


def main():
    failures = []
    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    frontend.memory = None
    model = request_fields.FakeModel()
    frontend.models = {model.id: model}
    if frontend.model_for({"model": "fake"}) is not model or frontend.model_for({"model": "fake-bulk"}) is not model:
        failures.append("a model and its -bulk name route to the same engine")
    try:
        frontend.model_for({"model": "other-bulk"})
        failures.append("a -bulk name of a model that is not served is refused")
    except chat_frontend.RequestError as error:
        if error.status != 404:
            failures.append(f"a -bulk name of an unserved model answers 404, not {error.status}")
    messages = [{"role": "user", "content": "hi"}]
    interactive = frontend.prepare(model, {"model": "fake", "messages": messages})[0]["priority"]
    bulk = frontend.prepare(model, {"model": "fake-bulk", "messages": messages})[0]["priority"]
    unnamed = frontend.prepare(model, {"messages": messages})[0]["priority"]
    if not (interactive == unnamed == chat_frontend.INTERACTIVE_PRIORITY and bulk == chat_frontend.BULK_PRIORITY):
        failures.append(f"engine priorities: interactive {interactive}, unnamed {unnamed}, bulk {bulk}")
    if not chat_frontend.INTERACTIVE_PRIORITY > chat_frontend.BULK_PRIORITY > chat_frontend.BACKGROUND_PRIORITY:
        failures.append("interactive work outranks bulk work, which outranks the background memory curator")
    listed = json.loads(asyncio.run(frontend.list_models(None)).body)
    if sorted(entry["id"] for entry in listed["data"]) != ["fake", "fake-bulk"]:
        failures.append(f"/v1/models lists each model and its -bulk name: {listed}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS a request for <model>-bulk runs on that model's engine at bulk priority, below interactive requests and above the background curator, and /v1/models lists both names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
