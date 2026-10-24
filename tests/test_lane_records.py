#!/usr/bin/env python3
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402


def main():
    document = stage_lane.lane_records()
    lanes = document["lanes"]
    reserve = document["weightd"]["kv_reserve_bytes"]
    swap = {entry["id"]: entry for entry in json.loads((ROOT / "deployment/hub/model_swap.json").read_text())["models"]}
    chat = {entry["id"]: entry for entry in json.loads((ROOT / "deployment/hub/chat_frontend.json").read_text())["models"]}
    failures = []
    for missing in sorted(set(swap) - set(lanes)):
        failures.append(f"{missing}: the swap controller serves it but deployment/lanes.json has no record")
    for extra in sorted(set(lanes) - set(swap)):
        failures.append(f"{extra}: a lane record the swap controller does not serve")
    for record_id, record in sorted(lanes.items()):
        model = stage_lane.MODELS.get(record.get("model"))
        if model is None or model["swap_id"] != record_id:
            failures.append(f"{record_id}: model {record.get('model')!r} is no tools/stage_lane.py MODELS entry with this swap id")
            continue
        if record["launcher"] not in ("stage_lane", "k3_lane"):
            failures.append(f"{record_id}: launcher {record['launcher']!r} is neither stage_lane nor k3_lane")
        if record["launcher"] == "stage_lane" and record["runtime_root"] != f"/home/{{host}}/stage-lanes/{record['model']}/lane{record['lane']}/root":
            failures.append(f"{record_id}: a stage lane lives under /home/{{host}}/stage-lanes/{record['model']}/lane{record['lane']}/root, not {record['runtime_root']}")
        native = stage_lane.native_context_tokens(model)
        context = record["context_tokens"]
        reason = record.get("context_reason", "")
        if context > native:
            failures.append(f"{record_id}: context_tokens {context} exceeds the contract maximum {native}")
        elif context < native and not reason.strip():
            failures.append(f"{record_id}: serves {context} of the contract's {native} positions without a context_reason")
        elif context == native and reason:
            failures.append(f"{record_id}: serves the contract maximum, so context_reason must go")
        block = model.get("kv_block_tokens", stage_lane.KV_BLOCK_TOKENS)
        sequence_bytes = context // block * model["kv_page_bytes"]
        physical_bytes = record["kv_physical_pages"] * model["kv_page_bytes"]
        if sequence_bytes > reserve:
            failures.append(f"{record_id}: one {context}-position sequence needs {sequence_bytes} KV bytes per rank, over the weightd KV reserve {reserve}")
        if physical_bytes > reserve:
            failures.append(f"{record_id}: {record['kv_physical_pages']} physical pages promise the scheduler {physical_bytes} bytes per rank, over the weightd KV reserve {reserve}")
        with tempfile.TemporaryDirectory() as directory:
            args = stage_lane.record_arguments(record_id)
            try:
                stage_lane.write_lane(args, pathlib.Path(directory))
            except SystemExit as error:
                failures.append(f"{record_id}: does not render: {error}")
                continue
            hub = json.loads((pathlib.Path(directory) / "model_resident.json").read_text())
            node = json.loads((pathlib.Path(directory) / "deployment.json").read_text())
            if "tokenizer" in node or "draft_server" in node or hub["tokenizer"]["sha256"] != record["tokenizer_sha256"]:
                failures.append(f"{record_id}: the tokenizer and draft server belong to the hub model_resident.json only")
            prefix = record.get("text_prompt_prefix_token_ids")
            if not isinstance(prefix, list) or not all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < model["tokenizer_vocabulary"] for token in prefix):
                failures.append(f"{record_id}: text_prompt_prefix_token_ids must list the ids the publisher tokenizer puts before a raw text prompt (tools/tokenizer_text_prefix.py), [] for none")
            elif hub["tokenizer"].get("text_prompt_prefix_token_ids", []) != prefix:
                failures.append(f"{record_id}: the hub model_resident.json carries text prompt prefix {hub['tokenizer'].get('text_prompt_prefix_token_ids')}, the record {prefix}")
        if record["launcher"] == "stage_lane":
            generated = stage_lane.swap_entry(record["model"], model, record["lane"])
            entry = swap.get(record_id, {})
            for key, value in generated.items():
                if entry.get(key) != value:
                    failures.append(f"{record_id}: deployment/hub/model_swap.json {key} differs from the lane record's swap entry")
        elif swap.get(record_id, {}).get("node_memory_bytes") != model["node_memory_bytes"]:
            failures.append(f"{record_id}: deployment/hub/model_swap.json node_memory_bytes differs from tools/stage_lane.py")
        if record_id in chat and chat[record_id].get("context_tokens") != context:
            failures.append(f"{record_id}: deployment/hub/chat_frontend.json context_tokens {chat[record_id].get('context_tokens')} differs from the engine's {context}")
    for line in failures:
        print("FAILED: " + line)
    if failures:
        return 1
    print(f"PASS {len(lanes)} lane records render, match the swap controller and the chat layer, give the hub API the raw text prompt prefix the record names, and serve the contract maximum or say why not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
