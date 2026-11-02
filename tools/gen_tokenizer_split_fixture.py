#!/usr/bin/env python3
import argparse
import json
import random
import sys
import unicodedata

from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers
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
    "cased_letters": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    "ideograph_letters": [
        "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    ],
    "cased_words": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?: [\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+)*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?: [\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*)*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    "phrase_letters": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
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
    "HTTPServer helloWorld XMLHttpRequest iPhone DON'T Don'T don't O'Neil",
    "ÀÉÎõ àéÎ ǅungla ǈx ʰello ᵃbcD µs ΣΑΣ σας Ⅻ ⅻ",
    "\"Quoted\" (Paren) [Bracket] {x}: path/to/file.py\r\n/x a/\n/b //\r\n//",
    "x́Ý ́A ́a 'S 'Ab A'B a'BC",
    "東京Tokyo大阪 한국어Korean Ελληνικά",
    "대한민국의 수도는 어디인가요? 한 단어로 답하세요.",
    "계약서에서 '불가항력' 조항이 무엇인지 두 문장으로 설명하세요.\n\n",
    "the capital of France is Paris.\nNew line  two spaces  here ",
    "a b  c d\te f/g h\r\ni ?\n!\n\n/x ?/ a?/",
    "x́ y ́z café über naïve ́ ́a  ́b",
    "word's it'S I'M 's 'm a'll",
    "서울 Seoul 부산 Busan 12 34abc def5 ghi",
    "東京タワーは333メートル・高さ ひらがなabc漢字123 カタカナ・テスト",
    "for i in range(10):\n    print(i)  \n\t\n",
    "The New York Times's best seller list and the quick brown fox's den",
    "HELLO WORLD Foo bar BAZ Qux NASA Rover ABC def GHi jKL",
    "two  spaces break words but one space joins them ",
    "Hello\u00a0World Hello\tWorld Hello World. Hello, World!",
    "iPhone iPad macOS eBay x́y ́z Ýa ́B",
    "대한민국의 수도는 서울 입니다 한국어 문장 Korean Text",
    "word's word'S Word's WORD'S a b'll C D'RE",
    "Σας Σας ΣΑΣ σας ǅungla ǅUNGLA ʰello ʰELLO",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "x!\x01!y a\x01b !\x01\x01! \x01",
    "a \n b\u3000\n c \u2028\t\td\u200b\n e\ufeff  f\u00a0  g\x85\n<x> <x>\n",
    "end with strip   \n\n",
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


CASE_REGEXES = (("upper", "[\\p{Lu}\\p{Lt}]", 1), ("lower", "\\p{Ll}", 2))


def oniguruma_case_ranges():
    values = [value for value in range(0, 0x110000) if not 0xD800 <= value <= 0xDFFF]
    text = "".join(chr(value) for value in values)
    classes = [0] * len(values)
    for _, regex, klass in CASE_REGEXES:
        kept = [False] * len(values)
        for _, (start, stop) in pre_tokenizers.Split(Regex(regex), "removed").pre_tokenize_str(text):
            for index in range(start, stop):
                kept[index] = True
        for index, is_kept in enumerate(kept):
            if not is_kept:
                if classes[index] != 0:
                    raise SystemExit(f"U+{values[index]:04X} matches two cases")
                classes[index] = klass
    ranges = []
    for value, klass in zip(values, classes):
        if klass == 0:
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


STRIP_DROPPED_BYTES = (0x01, 0x0B, 0x1C, 0x1F, 0xC0, 0xFF)
STRIP_MERGES = (("!", "!"), ("a", "b"), ("Ġ", "a"))
STRIP_ADDED = (("  ", True), ("\n", True), ("\n\n", True), ("\t\t", True), ("<x>", False))


def strip_tokenizer():
    level = ByteLevel.alphabet()
    mapping = {}
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    shifted = 0
    for value in range(256):
        if value in printable:
            mapping[value] = chr(value)
        else:
            mapping[value] = chr(256 + shifted)
            shifted += 1
    dropped = {mapping[value] for value in STRIP_DROPPED_BYTES}
    symbols = [character for character in sorted(level) if character not in dropped]
    symbols += ["".join(pair) for pair in STRIP_MERGES]
    tokenizer = Tokenizer(models.BPE({symbol: index for index, symbol in enumerate(symbols)}, list(STRIP_MERGES)))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(PATTERNS["cased_words"]), "isolated"),
         pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.add_tokens([AddedToken(content, lstrip=strip, normalized=False) for content, strip in STRIP_ADDED])
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
        "oniguruma_case_ranges": oniguruma_case_ranges(),
    }
    for name, pattern in PATTERNS.items():
        tokenizer = tokenizer_for(pattern)
        document["patterns"][name] = {
            "regex": pattern["byte_level"] if isinstance(pattern, dict) else pattern,
            "tokenizer": json.loads(tokenizer.to_str()),
            "piece_ends": [byte_ends(text, tokenizer.pre_tokenizer.pre_tokenize_str(text)) for text in texts],
        }
    strip = strip_tokenizer()
    document["added_strip"] = {
        "tokenizer": json.loads(strip.to_str()),
        "ids": [strip.encode(text, add_special_tokens=False).ids for text in texts],
    }
    glm = Tokenizer.from_file(arguments.glm_tokenizer)
    document["glm_ids"] = [glm.encode(text, add_special_tokens=False).ids for text in texts]
    with open(arguments.output, "w") as handle:
        json.dump(document, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
