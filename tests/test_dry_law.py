"""Enforce model-neutral shared infrastructure.

A model token belongs only in its model family, model module, model tool, or
model test. Shared runtime, transport, serving, cache, kernel and common code
may not name any model or driver, in a path or in its content: a generic
function carries a generic name. There is no debt budget; PENDING lists the
shared files still waiting to move to their family, and it may only shrink.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILY_TEMPLATES = "include/sparkpipe/family"
COMMON = (
    "include/sparkpipe",
    "node",
    "ring",
    "serving",
    "api",
    "cache",
    "scheduler",
    "text",
    "src",
    "runtime",
    "inference/stage",
    "inference/kernels",
    "inference/runner",
    "model-families/common",
    "common",
)
MODEL_TOKEN = re.compile(
    r"(?<![a-z])(?:glm|kimi|qwen|dsv4|deepseek|gemma|laguna|minimax|glimmer|hunyuan|hy4)"
    r"|(?<![a-z])(?:muse|mimo)(?![a-z])"
    r"|(?<![a-z0-9])k3(?![a-z0-9])"
    r"|(?<![a-z])ling(?![a-z])"
    r"|(?<![A-Z])GLM|(?<=[a-z0-9_])(?:Glm|Kimi|Qwen|Dsv4|DeepSeek|Deepseek|Mimo|Gemma|Laguna|MiniMax|Minimax|Muse|Glimmer|Hunyuan|Hy4|K3|Ling)(?=[A-Z0-9_]|$)"
    r"|^(?:Glm|Kimi|Qwen|Dsv4|DeepSeek|Mimo|Gemma|Laguna|MiniMax|Minimax|Muse|Glimmer|Hunyuan|Hy4|Ling)(?=[A-Z0-9_]|$)",
    re.MULTILINE,
)
FAMILY_TEMPLATE_TOKEN = re.compile(
    r"glm(?:5[_-]?2|52)|glm5[_-]?next|g5n|laguna|(?<![a-z])ling(?![a-z])|kimi|(?:^|[^a-z0-9])k3(?:[^a-z0-9]|$)|"
    r"qwen|dsv4|deepseek|mimo25",
    re.IGNORECASE,
)
PUBLISHER_ALIASES = ("exaone", "solar", "ax_k2")
PENDING = ()


def family_tokens():
    names = {path.name for path in (ROOT / "model-families").iterdir() if path.is_dir() and path.name != "common"}
    for path in (ROOT / "modules").iterdir():
        if path.is_dir():
            name = path.name.split("_resident_decode_stage")[0]
            names.add(re.sub(r"_(?:tp16|tp4pp4)$", "", name))
    return sorted(names | set(PUBLISHER_ALIASES))


def camel(token):
    return "".join(part[:1].upper() + part[1:] for part in token.split("_"))


def family_pattern(tokens):
    lower = "|".join(re.escape(token) for token in tokens)
    upper = "|".join(re.escape(token.upper()) for token in tokens)
    camels = "|".join(re.escape(camel(token)) for token in tokens)
    return re.compile(rf"(?<![a-z0-9])(?:{lower})(?![a-z])|(?<![A-Z0-9])(?:{upper})(?![A-Z])|"
                      rf"(?<=[a-z0-9_])(?:{camels})(?=[A-Z0-9_]|$)|^(?:{camels})(?=[A-Z0-9_]|$)", re.MULTILINE)


FAMILY_TOKEN = family_pattern(family_tokens())


def pending_entry(relative):
    for entry in PENDING:
        if relative == entry or relative.startswith(entry + "/"):
            return entry
    return None


def shared_files():
    for root in COMMON:
        for path in sorted((ROOT / root).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            yield path, str(path.relative_to(ROOT))


def main():
    failures = 0
    checked = 0
    pending_hits = set()
    tokens = family_tokens()
    for token in tokens:
        for planted in (f"{token}_helper", f"Spark{camel(token)}Helper", f"SPARK_{token.upper()}_LIMIT"):
            if FAMILY_TOKEN.search(planted) is None:
                print(f"  FAIL the family pattern does not see {planted!r} for {token}")
                failures += 1
    for clean in ("scaling", "SparkLinearHelper", "SPARK_SOLARIZE", "glimpse"):
        if FAMILY_TOKEN.search(clean) is not None:
            print(f"  FAIL the family pattern sees a model in {clean!r}")
            failures += 1
    for path, relative in shared_files():
        checked += 1
        text = path.read_text(errors="surrogateescape")
        if relative.startswith(FAMILY_TEMPLATES + "/"):
            if FAMILY_TEMPLATE_TOKEN.search(relative) or FAMILY_TEMPLATE_TOKEN.search(text):
                print(f"  FAIL {relative}: model token in a family template")
                failures += 1
            continue
        if MODEL_TOKEN.search(relative) or MODEL_TOKEN.search(text) or FAMILY_TOKEN.search(relative) or FAMILY_TOKEN.search(text):
            entry = pending_entry(relative)
            if entry is not None:
                pending_hits.add(entry)
                continue
            print(f"  FAIL {relative}: model token in shared code")
            failures += 1
    for entry in PENDING:
        if entry not in pending_hits:
            print(f"  FAIL PENDING entry {entry} is clean or gone: remove it from PENDING")
            failures += 1
    print(f"common files checked {checked} against {len(tokens)} family tokens, pending moves {len(PENDING)}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nPASS shared infrastructure is model-neutral")
    return 0


if __name__ == "__main__":
    sys.exit(main())
