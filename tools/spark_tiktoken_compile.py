#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path
import struct


SPLIT_MODES = {"byte-level": 1, "letters-and-marks": 2, "digit-runs": 3, "letters": 4}


def compile_tokenizer(ranks_path, config_path, destination, split_mode):
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split mode must be one of {sorted(SPLIT_MODES)}")
    ranks = {}
    for line in ranks_path.read_bytes().splitlines():
        encoded, rank = line.split()
        piece = base64.b64decode(encoded, validate=True)
        if not piece or piece in ranks:
            raise ValueError("empty or duplicate vocabulary piece")
        ranks[piece] = int(rank)
    if set(ranks.values()) != set(range(len(ranks))) or any(bytes([b]) not in ranks for b in range(256)):
        raise ValueError("ranks must be contiguous and contain all 256 bytes")
    added = json.loads(config_path.read_text())["added_tokens_decoder"]
    specials = [(int(token), value["content"].encode(), int(value["special"])) for token, value in added.items()]
    if any(token < len(ranks) or not text for token, text, _ in specials):
        raise ValueError("added token overlaps base vocabulary or has empty text")
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {b: chr(b) for b in direct}
    alphabet.update({b: chr(256 + i) for i, b in enumerate(b for b in range(256) if b not in alphabet)})
    merges = []
    for piece, rank in ranks.items():
        for split in range(1, len(piece)):
            left, right = ranks.get(piece[:split]), ranks.get(piece[split:])
            if left is not None and right is not None:
                merges.append((left, right, rank, rank))
    maximum = max([len(ranks) - 1] + [token for token, _, _ in specials])
    with destination.open("xb") as output:
        output.write(struct.pack("<Q11I", 0x314b4f54535053, 2, 1, 0, 0, 0, 0, maximum, SPLIT_MODES[split_mode], len(ranks), len(merges), len(specials)))
        for piece, rank in sorted(ranks.items(), key=lambda item: item[1]):
            text = "".join(alphabet[b] for b in piece).encode()
            output.write(struct.pack("<II", rank, len(text)) + text)
        for merge in merges:
            output.write(struct.pack("<4I", *merge))
        for token, text, special in sorted(specials):
            output.write(struct.pack("<3I", token, special, len(text)) + text)
    return dict(vocabulary_size=maximum + 1, merge_pairs=len(merges), added_tokens=len(specials), split_mode=split_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-mode", choices=sorted(SPLIT_MODES), required=True)
    args = parser.parse_args()
    print(json.dumps(compile_tokenizer(args.ranks, args.config, args.output, args.split_mode)))


if __name__ == "__main__":
    main()
