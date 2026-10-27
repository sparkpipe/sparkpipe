RETENTION_SECONDS = 86400
EXPIRY_SWEEP_SECONDS = 60


def expire_contexts(contexts, now):
    stale = [key for key, context in contexts.items() if now - context.used >= RETENTION_SECONDS]
    for key in stale:
        del contexts[key]
        print(f"CONTEXT-EXPIRED sequence {key} unused for {RETENTION_SECONDS} s", flush=True)
    return len(stale)


def evict_contexts(contexts, budget):
    evicted = 0
    while len(contexts) > 1 and sum(context.bytes() for context in contexts.values()) > budget:
        key, _ = contexts.popitem(last=False)
        print(f"CONTEXT-EVICTED sequence {key}", flush=True)
        evicted += 1
    return evicted
