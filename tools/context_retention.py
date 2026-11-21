def evict_contexts(contexts, budget):
    evicted = 0
    while len(contexts) > 1 and sum(context.bytes() for context in contexts.values()) > budget:
        key, _ = contexts.popitem(last=False)
        print(f"CONTEXT-EVICTED sequence {key}", flush=True)
        evicted += 1
    return evicted
