import bisect
import hashlib
import math
import re
from array import array
from collections import OrderedDict

BLOCK_TOKENS = 64
LEDGER_TOKENS = 1 << 23
REQUEST_REFERENCES = 1 << 16
CHAIN_PREFIX = "c_"
PIN_PREFIX = "p_"
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def block_digests(token_ids, block_tokens=BLOCK_TOKENS):
    data = array("I", token_ids).tobytes()
    width = 4 * block_tokens
    digest, digests = b"", []
    for start in range(0, len(data) - width + 1, width):
        digest = hashlib.blake2b(digest + data[start:start + width], digest_size=16).digest()
        digests.append(digest)
    return digests


def short_id(prefix, scope, model_id, digest):
    return prefix + hashlib.blake2b(f"{scope}\0{model_id}\0".encode() + digest, digest_size=12).hexdigest()


class Chain:
    __slots__ = ("id", "scope", "model", "tokens", "digests", "used", "references")

    def __init__(self, chain_id, scope, model_id, tokens, digests, used):
        self.id, self.scope, self.model, self.tokens, self.digests, self.used = chain_id, scope, model_id, tokens, digests, used
        self.references = set()


class Pin:
    __slots__ = ("id", "scope", "model", "tokens", "digests", "source", "created", "used")

    def __init__(self, pin_id, scope, model_id, tokens, digests, source, now):
        self.id, self.scope, self.model, self.tokens, self.digests, self.source = pin_id, scope, model_id, tokens, digests, source
        self.created = self.used = now


class CacheState:
    def __init__(self, block_tokens=BLOCK_TOKENS, ledger_tokens=LEDGER_TOKENS, request_references=REQUEST_REFERENCES):
        self.block_tokens = block_tokens
        self.ledger_tokens = ledger_tokens
        self.request_references = request_references
        self.chains = OrderedDict()
        self.blocks = {}
        self.requests = OrderedDict()
        self.pins = {}
        self.pin_terminals = {}
        self.held_tokens = 0

    def aligned(self, token_ids):
        return len(token_ids) // self.block_tokens * self.block_tokens

    def drop_chain(self, chain_id):
        chain = self.chains.pop(chain_id, None)
        if chain is None:
            return
        self.held_tokens -= len(chain.tokens)
        for digest in chain.digests:
            if self.blocks.get((chain.scope, chain.model, digest)) == chain_id:
                del self.blocks[(chain.scope, chain.model, digest)]
        for reference in chain.references:
            self.requests.pop(reference, None)

    def touch_pins(self, scope, model_id, digests, now):
        for digest in digests:
            pin_id = self.pin_terminals.get((scope, model_id, digest))
            if pin_id is not None:
                self.pins[pin_id].used = now

    def record(self, scope, model_id, request_id, token_ids, now):
        count = self.aligned(token_ids)
        if count == 0 or not isinstance(request_id, str):
            return None
        tokens = array("I", token_ids[:count])
        digests = block_digests(tokens, self.block_tokens)
        holder = self.chains.get(self.blocks.get((scope, model_id, digests[-1])))
        if holder is None:
            inside = set(digests)
            absorbed = {self.blocks.get((scope, model_id, digest)) for digest in digests[:-1]} - {None}
            absorbed = [self.chains[chain_id] for chain_id in absorbed if chain_id in self.chains and self.chains[chain_id].digests[-1] in inside and
                        (scope, model_id, self.chains[chain_id].digests[-1]) not in self.pin_terminals]
            holder = Chain(short_id(CHAIN_PREFIX, scope, model_id, digests[-1]), scope, model_id, tokens, digests, now)
            self.chains[holder.id] = holder
            self.held_tokens += count
            for chain in absorbed:
                moved = [(reference, self.requests[reference][1]) for reference in chain.references if reference in self.requests]
                chain.references = set()
                self.drop_chain(chain.id)
                for reference, length in moved:
                    self.requests[reference] = (holder.id, length)
                    holder.references.add(reference)
            for digest in digests:
                self.blocks[(scope, model_id, digest)] = holder.id
        holder.used = now
        self.chains.move_to_end(holder.id)
        reference = (scope, request_id)
        previous = self.requests.pop(reference, None)
        if previous is not None and previous[0] in self.chains:
            self.chains[previous[0]].references.discard(reference)
        self.requests[reference] = (holder.id, count)
        holder.references.add(reference)
        while len(self.requests) > self.request_references:
            old, (chain_id, _) = self.requests.popitem(last=False)
            if chain_id in self.chains:
                self.chains[chain_id].references.discard(old)
        while self.held_tokens > self.ledger_tokens and len(self.chains) > 1:
            self.drop_chain(next(iter(self.chains)))
        self.touch_pins(scope, model_id, digests, now)
        return holder.id

    def resolve(self, scope, request=None, prefix=None):
        if request is not None:
            found = self.requests.get((scope, request))
            if found is None or found[0] not in self.chains:
                return None
            chain = self.chains[found[0]]
            return chain.model, chain.tokens[:found[1]]
        chain = self.chains.get(prefix)
        if chain is not None and chain.scope == scope:
            return chain.model, chain.tokens
        pin = next((pin for pin in self.pins.values() if pin.scope == scope and short_id(CHAIN_PREFIX, scope, pin.model, pin.digests[-1]) == prefix), None)
        return (pin.model, pin.tokens) if pin is not None else None

    def make_pin(self, scope, model_id, tokens, source, now):
        digests = block_digests(tokens, self.block_tokens)
        return Pin(short_id(PIN_PREFIX, scope, model_id, digests[-1]), scope, model_id, array("I", tokens), digests, source, now)

    def pinned_tokens(self, model_id, extra=None):
        held = {digest for pin in self.pins.values() if pin.model == model_id for digest in pin.digests}
        if extra is not None:
            held.update(extra.digests)
        return len(held) * self.block_tokens

    def add_pin(self, pin):
        self.pins[pin.id] = pin
        self.pin_terminals[(pin.scope, pin.model, pin.digests[-1])] = pin.id

    def remove_pin(self, scope, pin_id):
        pin = self.pins.get(pin_id)
        if pin is None or pin.scope != scope:
            return None
        del self.pins[pin_id]
        self.pin_terminals.pop((pin.scope, pin.model, pin.digests[-1]), None)
        return pin

    def pin_of(self, scope, pin_id):
        pin = self.pins.get(pin_id)
        return pin if pin is not None and pin.scope == scope else None

    def engine_set(self, model_id):
        return [pin for pin in sorted(self.pins.values(), key=lambda pin: (pin.created, pin.id)) if pin.model == model_id]

    def expire(self, rule):
        dropped = [chain_id for chain_id, chain in self.chains.items() if rule(chain.used)]
        for chain_id in dropped:
            self.drop_chain(chain_id)
        models = set()
        for pin in [pin for pin in self.pins.values() if rule(pin.used)]:
            self.remove_pin(pin.scope, pin.id)
            models.add(pin.model)
        return len(dropped), models

    def forget(self, scope):
        for chain_id in [chain_id for chain_id, chain in self.chains.items() if chain.scope == scope]:
            self.drop_chain(chain_id)
        for reference in [reference for reference in self.requests if reference[0] == scope]:
            del self.requests[reference]
        models = set()
        for pin in [pin for pin in self.pins.values() if pin.scope == scope]:
            self.remove_pin(scope, pin.id)
            models.add(pin.model)
        return models

    def pin_view(self, pin):
        chain = short_id(CHAIN_PREFIX, pin.scope, pin.model, pin.digests[-1])
        return {"id": pin.id, "object": "cache.pin", "model": pin.model, "tokens": len(pin.tokens), "chain": chain, "source": dict(pin.source),
                "created": round(pin.created, 3), "last_used": round(pin.used, 3)}

    def chain_view(self, chain):
        pinned = 0
        for index, digest in enumerate(chain.digests):
            if (chain.scope, chain.model, digest) in self.pin_terminals:
                pinned = (index + 1) * self.block_tokens
        return {"id": chain.id, "model": chain.model, "tokens": len(chain.tokens), "pinned": pinned == len(chain.tokens), "pinned_tokens": pinned,
                "last_used": round(chain.used, 3), "requests": len(chain.references)}

    def view(self, scope):
        chains = [self.chain_view(chain) for chain in reversed(self.chains.values()) if chain.scope == scope]
        listed = {pin.id for pin in self.pins.values() if pin.scope == scope}
        known = {chain["id"] for chain in chains}
        for pin in sorted((self.pins[pin_id] for pin_id in listed), key=lambda pin: -pin.used):
            chain = short_id(CHAIN_PREFIX, pin.scope, pin.model, pin.digests[-1])
            if chain not in known:
                chains.append({"id": chain, "model": pin.model, "tokens": len(pin.tokens), "pinned": True, "pinned_tokens": len(pin.tokens),
                               "last_used": round(pin.used, 3), "requests": 0})
        pins = [self.pin_view(pin) for pin in sorted((self.pins[pin_id] for pin_id in listed), key=lambda pin: (pin.created, pin.id))]
        return chains, pins

    def scope_pinned_tokens(self, scope, model_id):
        return len({digest for pin in self.pins.values() if pin.scope == scope and pin.model == model_id for digest in pin.digests}) * self.block_tokens

    def scope_tokens(self, scope, model_id):
        held = {digest for chain in self.chains.values() if chain.scope == scope and chain.model == model_id for digest in chain.digests}
        held.update(digest for pin in self.pins.values() if pin.scope == scope and pin.model == model_id for digest in pin.digests)
        return len(held) * self.block_tokens


def valid_identifier(value):
    return isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None


class PrefillSeed:
    def __init__(self, model_id, spec):
        if not isinstance(spec, dict) or set(spec) != {"points", "source"} or not isinstance(spec["source"], str) or not spec["source"].strip():
            raise ValueError(f"{model_id} cold_prefill needs exactly points ([[prompt tokens, cold prefill ms], ...]) and a source")
        points = spec["points"]
        if (not isinstance(points, list) or not points or
                not all(isinstance(point, list) and len(point) == 2 and all(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
                                                                            for value in point) for point in points)):
            raise ValueError(f"{model_id} cold_prefill points must be [prompt tokens, cold prefill ms] pairs of positive numbers")
        points = sorted((float(tokens), float(ms)) for tokens, ms in points)
        if len({tokens for tokens, _ in points}) != len(points):
            raise ValueError(f"{model_id} cold_prefill points must name each prompt length once")
        self.lengths = [math.log(tokens) for tokens, _ in points]
        self.rates = [ms / tokens for tokens, ms in points]
        self.source = spec["source"]

    def rate(self, prompt_tokens):
        at = math.log(max(prompt_tokens, 1))
        index = bisect.bisect_left(self.lengths, at)
        if index == 0:
            return self.rates[0]
        if index == len(self.lengths):
            return self.rates[-1]
        low, high = self.lengths[index - 1], self.lengths[index]
        share = (at - low) / (high - low)
        return self.rates[index - 1] + share * (self.rates[index] - self.rates[index - 1])
