#!/usr/bin/env python3
import argparse
import json
import random
import sys
import unicodedata

from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from tokenizers.pre_tokenizers import ByteLevel

PATTERNS = {
    "letters_and_marks": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    "digit_runs": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    "letters": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    "digit_ideograph_sequence": [
        "\\p{N}{1,3}",
        "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    ],
    "letters_possessive": "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
    "byte_level_regex": {"byte_level": "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"},
}

HANDPICKED = [
    "She called it the ‘best’, and it was.",
    "The dogs’ owners’. Their “plan”, however, failed.",
    "Chapter Ⅳ begins",
    "He said, “Yes.”",
    "Paris — France",
    "a b  c  d",
    "東京　大阪　名古屋",
    "価格は　１００円です。",
    "x́́y ́abc a ́b",
    "12Ⅳ²x ①② \U0001d7d9\U0001d7da",
    "a'ſx a'Kx a'ﬆx a'ẞx It'S we'LL they'Re",
    "a\x1cb a \x85b a᠎ b a​ b a  　b",
    "line one\r\n\r\n  line two three four\n\n\t",
    "नमस्ते สวัสดี مرحبا",
    "café café ǅungla ʰello",
    "\U0001F600‍\U0001F600 ❤️ ok… ¿Qué? § €5",
    "\U00020000\U0001d400x \U0001d7ce1",
    "trailing spaces   ",
    "   leading",
    "a b  c",
    "Count from 1 to 30: 1, 2, 3, 12345, 3.14159, -42, 1e10 and ９９９９",
    "東京タワーは333メートルです。ひらがなカタカナ漢字abc",
    "def f(x):\n    return x**2  # 平方\n\n\tprint(f(12))",
    "<b>bold</b> #tag @user .NET 'quote' \"dq\" ~/path_to/file.py",
    "a   b",
    "I'M you'RE we'll it's",
    "x²³ y¹ z⁴",
    "a\u3000b \u3000 c",
    "!'s ?'t x'll",
]

POOLS = [
    list("abcXYZ019.,!?'\"-_()[]{}#$%&*+/<=>@\\^`|~"),
    list(" \t\n\r\x0b\x0c") + [" ", " ", " ", " ", " ", " ", " ", " ", " ", "　", "\x85", "\x1c", "\x1f", "᠎", "​", "﻿"],
    ["é", "ß", "ж", "你", "好", "ส", "न", "ا", "가", "ǅ", "ʰ", "々", "ᛮ", "\U0001d400", "\U00020000", "ſ", "K"],
    ["́", "ः", "⃝", "ั", "्", "️", "‍", "҈", "\U000e0100"],
    ["１", "٣", "०", "Ⅳ", "①", "²", "½", "\U0001d7d9", "〇", "\U00010107"],
    ["‘", "’", "“", "”", "，", "。", "—", "…", "¿", "§", "€", "✓", "\U0001F600", "❤", "­", "", "\U000f0000"],
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL", "'ſ"],
]


CLASS_REGEXES = (("space", "\\s", 1), ("letter", "\\p{L}", 2), ("number", "\\p{N}", 3), ("mark", "\\p{M}", 5), ("punctuation", "[\\p{P}\\p{S}]", 6))


def oniguruma_class_ranges():
    values = [value for value in range(0x80, 0x110000) if not 0xD800 <= value <= 0xDFFF]
    text = "".join(chr(value) for value in values)
    classes = [4] * len(values)
    for _, regex, klass in CLASS_REGEXES:
        kept = [False] * len(values)
        for _, (start, stop) in pre_tokenizers.Split(Regex(regex), "removed").pre_tokenize_str(text):
            for index in range(start, stop):
                kept[index] = True
        for index, is_kept in enumerate(kept):
            if not is_kept:
                if classes[index] != 4:
                    raise SystemExit(f"U+{values[index]:04X} matches two classes")
                classes[index] = klass
    ranges = []
    for value, klass in zip(values, classes):
        if klass == 4:
            continue
        if ranges and ranges[-1][1] == value - 1 and ranges[-1][2] == klass:
            ranges[-1][1] = value
        else:
            ranges.append([value, value, klass])
    return ranges


def corpus():
    generator = random.Random(20260928)
    texts = list(HANDPICKED)
    for _ in range(500):
        chosen = generator.sample(POOLS, 3)
        texts.append("".join(generator.choice(generator.choice(chosen)) for _ in range(generator.randint(2, 24))))
    return texts


def byte_ends(text, pieces):
    ends = []
    for _, (_, stop) in pieces:
        ends.append(len(text[:stop].encode("utf-8")))
    return ends


def tokenizer_for(pattern):
    vocabulary = {character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocabulary, []))
    if isinstance(pattern, dict):
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        return tokenizer
    patterns = pattern if isinstance(pattern, list) else [pattern]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(item), "isolated") for item in patterns]
        + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--glm-tokenizer", required=True)
    arguments = parser.parse_args()
    import tokenizers
    texts = corpus()
    document = {
        "generator": "tools/gen_tokenizer_split_fixture.py",
        "tokenizers_version": tokenizers.__version__,
        "python_unicode_version": unicodedata.unidata_version,
        "texts": texts,
        "patterns": {},
        "glm_ids": [],
        "oniguruma_class_ranges": oniguruma_class_ranges(),
    }
    for name, pattern in PATTERNS.items():
        tokenizer = tokenizer_for(pattern)
        document["patterns"][name] = {
            "regex": pattern["byte_level"] if isinstance(pattern, dict) else pattern,
            "tokenizer": json.loads(tokenizer.to_str()),
            "piece_ends": [byte_ends(text, tokenizer.pre_tokenizer.pre_tokenize_str(text)) for text in texts],
        }
    glm = Tokenizer.from_file(arguments.glm_tokenizer)
    document["glm_ids"] = [glm.encode(text, add_special_tokens=False).ids for text in texts]
    with open(arguments.output, "w") as handle:
        json.dump(document, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
