"""The deployed chat layer turns memory on for every model through one shared set of budgets: no model carries its own memory block beyond budget overrides, every model resolves a profile, and the curator suffix each model gets is the reasoning close its format declares, else its reasoning end marker."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
cm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cm)
CONFIG = json.loads((ROOT / "deployment" / "hub" / "chat_frontend.json").read_text())


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_chat_memory_deployment: {message}")


budgets = CONFIG.get("chat_memory", {}).get("budgets")
check(isinstance(budgets, dict) and set(budgets) == set(cm.BUDGET_KEYS), f"chat_memory.budgets must set exactly {cm.BUDGET_KEYS}")
suffixes = {}
for spec in CONFIG["models"]:
    overrides = spec.get("chat_memory", {})
    check(isinstance(overrides, dict) and set(overrides) <= set(cm.OVERRIDE_KEYS), f"{spec['id']} chat_memory may only override {cm.OVERRIDE_KEYS}")
    model = type("Model", (), {"id": spec["id"], "memory": overrides, "reasoning": spec.get("reasoning"), "context_tokens": spec["context_tokens"]})()
    profile = cm.MemoryProfile(budgets, model)
    check(profile.working_tokens + profile.ledger_tokens + 8192 <= profile.prompt_tokens, f"{spec['id']} fits its working set and ledger with 8K to spare: {profile.working_tokens} + {profile.ledger_tokens} of {profile.prompt_tokens}")
    check((profile.curator_suffix is None) == (spec.get("reasoning") is None), f"{spec['id']} gets a curator suffix exactly when it declares reasoning markers")
    suffixes[spec["id"]] = profile.curator_suffix
check(suffixes["kimi-k3"] == "<|close|>think<|sep|><|open|>response<|sep|>", f"K3's derived suffix is the one its memory qualification used: {suffixes['kimi-k3']!r}")
check(suffixes["deepseek-v4.1-flash"] == "</think>", f"DSV4.1's derived suffix closes its reasoning: {suffixes['deepseek-v4.1-flash']!r}")
check(all(suffixes[name] == "\n</think>\n\n" for name in ("qwen3.8-27b", "qwen3.8-max", "qwen3.8-flash")), "the Qwen templates close reasoning with their own newlines")
print(f"PASS test_chat_memory_deployment: {len(suffixes)} models resolve memory profiles from one shared budget set; curator suffixes come from each model's reasoning markers")
