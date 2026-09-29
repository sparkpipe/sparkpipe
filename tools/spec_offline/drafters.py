from __future__ import annotations

import hashlib
import struct
from collections import Counter, defaultdict
from typing import Callable, Protocol

import numpy as np

from .chain import LOOKUP_MAX_MATCH, LOOKUP_MIN_MATCH, MASK64, LookupHistory, mix64

TapFunction = Callable[[np.ndarray, int], list[int]]


class Drafter(Protocol):
    name: str

    def reset(self, stream_tokens: tuple[int, ...], prompt: int, taps: np.ndarray | None) -> None: ...

    def observe(self, position: int, token: int) -> None: ...

    def expand(self, anchor: int, prefix: tuple[int, ...], width: int) -> list[int]: ...


def chain_from(drafter: Drafter, anchor: int, depth: int) -> list[int]:
    prefix: tuple[int, ...] = ()
    for _ in range(depth):
        choices = drafter.expand(anchor, prefix, 1)
        if not choices:
            break
        prefix += (choices[0],)
    return list(prefix)


class OracleDrafter:
    name = "oracle"

    def __init__(self, vocab: int):
        self.tokens: tuple[int, ...] = ()
        self.vocab = vocab

    def reset(self, stream_tokens, prompt, taps):
        self.tokens = stream_tokens

    def observe(self, position, token):
        pass

    def expand(self, anchor, prefix, width):
        position = anchor + 1 + len(prefix)
        if position >= len(self.tokens):
            return []
        expected = self.tokens[position]
        return [expected] + [(expected + shift) % self.vocab for shift in range(1, width)]


class AdversaryDrafter(OracleDrafter):
    name = "adversary"

    def expand(self, anchor, prefix, width):
        position = anchor + 1 + len(prefix)
        if position >= len(self.tokens):
            return []
        expected = self.tokens[position]
        return [(expected + shift) % self.vocab for shift in range(1, width + 1)]


class SyntheticDrafter(OracleDrafter):
    def __init__(self, accept_milli: int, vocab: int, seed: int = 1):
        super().__init__(vocab)
        self.accept_milli = accept_milli
        self.seed = seed
        self.name = f"synthetic:{accept_milli}"

    def expand(self, anchor, prefix, width):
        index = len(prefix)
        position = anchor + 1 + index
        if position >= len(self.tokens):
            return []
        expected = self.tokens[position]
        draw = mix64((self.seed ^ ((position << 8) & MASK64) ^ index) & MASK64) % 1000
        first = expected if draw < self.accept_milli else (expected + 1) % self.vocab
        alternatives = [token for token in (expected, (expected + 1) % self.vocab, (expected + 2) % self.vocab) if token != first]
        return ([first] + alternatives)[:width]


class LookupDrafter:
    name = "lookup"

    def __init__(self, min_match: int = LOOKUP_MIN_MATCH, max_match: int = LOOKUP_MAX_MATCH):
        self.history = LookupHistory(min_match, max_match)

    def reset(self, stream_tokens, prompt, taps):
        self.history = LookupHistory(self.history.min_match, self.history.max_match)
        self.history.observe(0, stream_tokens[:prompt + 1])

    def observe(self, position, token):
        self.history.observe(position, [token])

    def expand(self, anchor, prefix, width):
        draft = self.history.draft(anchor, len(prefix) + 1)
        if draft is None or len(draft) <= len(prefix) or tuple(draft[:len(prefix)]) != tuple(prefix):
            return []
        return [draft[len(prefix)]]


class SuffixDrafter:
    name = "suffix"

    def __init__(self, min_match: int = 2, max_match: int = 32):
        self.min_match = min_match
        self.max_match = max_match
        self.history: list[int] = []

    def reset(self, stream_tokens, prompt, taps):
        self.history = list(stream_tokens[:prompt + 1])

    def observe(self, position, token):
        del self.history[position:]
        self.history.append(token)

    def expand(self, anchor, prefix, width):
        context = self.history[:anchor + 1] + list(prefix)
        votes: Counter[int] = Counter()
        best_length = 0
        for length in range(min(self.max_match, len(context) - 1), self.min_match - 1, -1):
            pattern = context[-length:]
            for end in range(length, len(context)):
                if context[end - length:end] == pattern:
                    votes[context[end]] += 1
            if votes:
                best_length = length
                break
        if best_length == 0:
            return []
        return [token for token, _ in votes.most_common(width)]


class NgramDrafter:
    def __init__(self, order: int = 3):
        self.order = order
        self.name = f"ngram{order}"
        self.table: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.history: list[int] = []

    def reset(self, stream_tokens, prompt, taps):
        self.table = defaultdict(Counter)
        self.history = []
        for token in stream_tokens[:prompt + 1]:
            self._push(token)

    def _push(self, token):
        if len(self.history) >= self.order:
            self.table[tuple(self.history[-self.order:])][token] += 1
        self.history.append(token)

    def observe(self, position, token):
        while len(self.history) > position:
            self.history.pop()
        self._push(token)

    def expand(self, anchor, prefix, width):
        context = tuple((self.history[:anchor + 1] + list(prefix))[-self.order:])
        if len(context) < self.order:
            return []
        counter = self.table.get(context)
        if not counter:
            return []
        return [token for token, _ in counter.most_common(width)]


class TapDrafter:
    def __init__(self, name: str, function: TapFunction):
        self.name = name
        self.function = function
        self.taps: np.ndarray | None = None
        self.tokens: tuple[int, ...] = ()

    def reset(self, stream_tokens, prompt, taps):
        if taps is None:
            raise ValueError(f"drafter {self.name} needs taps")
        self.taps = taps
        self.tokens = stream_tokens

    def observe(self, position, token):
        pass

    def expand(self, anchor, prefix, width):
        if prefix:
            return []
        return self.function(self.taps[anchor], width)


def probe_tap_function(vocab: int) -> TapFunction:
    def function(tap: np.ndarray, width: int) -> list[int]:
        digest = hashlib.sha256(np.ascontiguousarray(tap, dtype=np.float32).tobytes()).digest()
        first = struct.unpack("<I", digest[:4])[0] % vocab
        return [first] + [(first + shift) % vocab for shift in range(1, width)]
    return function


def bf16_exact(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + ((bits >> 16) & 1) + 0x7FFF) & 0xFFFF0000).view(np.float32)


def probe_tap_for_token(token: int, vocab: int, salt: int, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(salt)
    probe = probe_tap_function(vocab)
    while True:
        candidate = bf16_exact(rng.standard_normal(dimension).astype(np.float32))
        if probe(candidate, 1)[0] == token:
            return candidate


def make_drafter(spec: str, vocab: int, tap_functions: dict[str, TapFunction] | None = None) -> Drafter:
    if spec == "oracle":
        return OracleDrafter(vocab)
    if spec == "adversary":
        return AdversaryDrafter(vocab)
    if spec == "lookup":
        return LookupDrafter()
    if spec == "suffix":
        return SuffixDrafter()
    if spec.startswith("ngram"):
        return NgramDrafter(int(spec[5:] or 3))
    if spec.startswith("synthetic:"):
        return SyntheticDrafter(int(spec.split(":", 1)[1]), vocab=vocab)
    if spec.startswith("tap:"):
        name = spec.split(":", 1)[1]
        functions = dict(tap_functions or {})
        functions.setdefault("probe", probe_tap_function(vocab))
        if name not in functions:
            raise ValueError(f"unknown tap drafter {name}; known: {sorted(functions)}")
        return TapDrafter(spec, functions[name])
    raise ValueError(f"unknown drafter {spec}")
