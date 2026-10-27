#!/usr/bin/env python3
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)

CONTROLS = ["<|open|>", "<|op", "<|close|>", "[EOS]", "<|reserved_token_163840|>", "<|reserved_token_16384|>", ".*+?(|)", "x", "é<|"]
FRAGMENTS = ["<", "|", "open", "op", "close", ">", "[", "EOS", "]", "reserved_token_", "16384", "0", ".*+?", "(|)", "", "x", "é", "code ", "\n", "{}"]


class CountingVocabulary:
    def __init__(self, controls):
        self.controls = controls
        self.calls = 0

    def encode(self, text, special):
        self.calls += 1
        return [ord(char) for char in text] + [1 if special else 0]

    def control_strings(self):
        return list(self.controls)


def main():
    failures = []
    rng = random.Random(7)
    pattern = chat_frontend.literal_pattern(tuple(sorted(CONTROLS)))
    for _ in range(4000):
        text = "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(0, 24)))
        if (pattern.search(text) is not None) != any(control in text for control in CONTROLS):
            failures.append(f"the control pattern disagrees with a substring scan on {text!r}")
            break
    if chat_frontend.literal_pattern(()) is not None:
        failures.append("a vocabulary without control strings should have no control pattern")

    vocabulary = CountingVocabulary(CONTROLS)
    messages = [{"role": "user", "content": "plain"}, {"role": "tool", "content": "uses <|open|> inside"}, {"role": "assistant", "content": "[EOS"}]
    marked, _, carried = chat_frontend.carry_controls(messages, None, vocabulary)
    if carried != ["uses <|open|> inside"] or marked[0]["content"] != "plain" or marked[2]["content"] != "[EOS":
        failures.append(f"carry_controls should carry exactly the strings that contain a control string; carried {carried}")

    cache = chat_frontend.PieceCache(vocabulary, 1 << 20)
    pieces = [("first piece", True), ("second piece", False), ("first piece", False), ("first piece", True)]
    direct = [id for piece, special in pieces for id in CountingVocabulary(CONTROLS).encode(piece, special)]
    vocabulary.calls = 0
    _, cold = chat_frontend.encode_pieces(cache, "text", pieces, 1 << 30)
    cold_calls = vocabulary.calls
    _, warm = chat_frontend.encode_pieces(cache, "text", pieces, 1 << 30)
    if cold != direct or warm != direct:
        failures.append("cached encoding must equal encoding every piece directly, cold and warm")
    if cold_calls != 3 or vocabulary.calls != 3:
        failures.append(f"the cache should encode each distinct (piece, special) once; it encoded {cold_calls} cold and {vocabulary.calls - cold_calls} more warm")

    small = chat_frontend.PieceCache(vocabulary, 4096)
    for index in range(500):
        piece = f"piece {index} " * 4
        if list(small.encode(piece, False, 1 << 30)) != vocabulary.encode(piece, False)[:]:
            failures.append("a cache under eviction pressure must still return exact ids")
            break
        if small.bytes > small.budget:
            failures.append(f"the cache holds {small.bytes} bytes over its {small.budget}-byte budget")
            break
    huge = "x" * 10000
    if list(small.encode(huge, True, 1 << 30)) != vocabulary.encode(huge, True) or (huge, True) in small.entries:
        failures.append("a piece larger than the whole budget should be encoded exactly and not kept")

    bounded = chat_frontend.PieceCache(vocabulary, 1 << 20)
    long_piece = "y" * 300
    if list(bounded.encode(long_piece, False, 100)) != vocabulary.encode(long_piece, False) or (long_piece, False) in bounded.entries:
        failures.append("a piece with more tokens than the context has room for should be encoded exactly and not kept")
    _, ids = chat_frontend.encode_pieces(bounded, "text", [("a" * 60, False), ("b" * 60, False)], 100)
    if len(ids) != 122 or ("a" * 60, False) not in bounded.entries or ("b" * 60, False) in bounded.entries:
        failures.append("encode_pieces should keep the pieces that fit the room and not the one that overflows it")
    if list(bounded.encode("y" * 50, False, 100)) != vocabulary.encode("y" * 50, False) or ("y" * 50, False) not in bounded.entries:
        failures.append("a piece within the room should be kept")

    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS the trie control pattern agrees with a substring scan on 4,000 strings, carry_controls carries exactly the control-bearing strings, and the piece cache returns the direct encoding while encoding each distinct piece once within its byte budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
