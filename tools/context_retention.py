import importlib.util
from pathlib import Path

LOADER = importlib.util.spec_from_file_location("sparkpipe_working_state", Path(__file__).resolve().parent / "serving" / "working_state.py")
working_state = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(working_state)
RETENTION_SECONDS = working_state.RETENTION_SECONDS
EXPIRY_SWEEP_SECONDS = 1


def expire_contexts(contexts, now, directive):
    wiped, held_until = working_state.read_directive(directive)
    stale = [key for key, context in contexts.items() if working_state.expired(context.used, wiped, held_until, now)]
    for key in stale:
        del contexts[key]
        print(f"CONTEXT-EXPIRED sequence {key} under the working-state directive (wiped {wiped}, held_until {held_until})", flush=True)
    return len(stale)


def evict_contexts(contexts, budget):
    evicted = 0
    while len(contexts) > 1 and sum(context.bytes() for context in contexts.values()) > budget:
        key, _ = contexts.popitem(last=False)
        print(f"CONTEXT-EVICTED sequence {key}", flush=True)
        evicted += 1
    return evicted
