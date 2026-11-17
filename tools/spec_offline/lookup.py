from __future__ import annotations

LOOKUP_MIN_MATCH = 3
LOOKUP_MAX_MATCH = 8
MASK64 = (1 << 64) - 1


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


class LookupHistory:
    def __init__(self, min_match: int = LOOKUP_MIN_MATCH, max_match: int = LOOKUP_MAX_MATCH):
        self.history: list[int] = []
        self.min_match = min_match
        self.max_match = max_match

    def observe(self, position: int, tokens) -> None:
        if position > len(self.history):
            raise ValueError("lookup history has a gap")
        del self.history[position:]
        self.history.extend(tokens)

    def find(self, anchor: int) -> tuple[int, int]:
        history = self.history
        if anchor >= len(history):
            return 0, 0
        best_length = best_end = 0
        end = anchor
        while end > 0 and best_length < self.max_match:
            length = 0
            while length < self.max_match and length < end and history[end - 1 - length] == history[anchor - length]:
                length += 1
            if length > best_length:
                best_length, best_end = length, end - 1
            end -= 1
        return (best_length, best_end) if best_length >= self.min_match else (0, 0)

    def draft(self, anchor: int, requested: int) -> list[int] | None:
        length, source_end = self.find(anchor)
        if length == 0:
            return None
        tokens = []
        for count in range(requested):
            source = source_end + 1 + count
            if source > anchor:
                break
            tokens.append(self.history[source])
        return tokens or None
