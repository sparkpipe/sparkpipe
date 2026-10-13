#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402
from tokenizer_text_prefix import TokenizerPrefixError, directory_prefix, text_prompt_prefix_token_ids  # noqa: E402

SPECIALS = {
    "<s>": {"id": "<s>", "ids": [7], "tokens": ["<s>"]},
    "</s>": {"id": "</s>", "ids": [8], "tokens": ["</s>"]},
}
ADDED = [{"id": 2, "content": "<bos>", "special": True}, {"id": 7, "content": "<s>", "special": True}]


def special(name):
    return {"SpecialToken": {"id": name, "type_id": 0}}


SEQUENCE_A = {"Sequence": {"id": "A", "type_id": 0}}
WRAPPED = {"type": "TemplateProcessing", "single": [special("<s>"), SEQUENCE_A, special("</s>")], "special_tokens": SPECIALS}
BARE = {"type": "TemplateProcessing", "single": [SEQUENCE_A], "special_tokens": {}}
BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}


def tokenizer(post_processor):
    return {"added_tokens": ADDED, "model": {"vocab": {"hello": 9}}, "post_processor": post_processor}


def expect(failures, label, actual, wanted):
    if actual != wanted:
        failures.append(f"{label}: derived {actual}, wanted {wanted}")


def refuses(failures, label, call):
    try:
        value = call()
    except TokenizerPrefixError:
        return
    failures.append(f"{label}: derived {value} instead of refusing")


def write_directory(directory, tokenizer_value, config_value):
    directory.mkdir(parents=True, exist_ok=True)
    if tokenizer_value is not None:
        (directory / "tokenizer.json").write_text(json.dumps(tokenizer_value))
    if config_value is not None:
        (directory / "tokenizer_config.json").write_text(json.dumps(config_value))
    return directory


def render(work, tokenizer_dir):
    output = work / "rendered"
    command = [sys.executable, str(ROOT / "tools" / "stage_lane.py"), "--model", "laguna", "--lane", "3",
               "--runtime-root", "/home/{host}/root", "--weightd-socket", "/tmp/w.sock",
               "--output-dir", str(output), "--tokenizer-sha256", "a" * 64]
    if tokenizer_dir is not None:
        command += ["--tokenizer-dir", str(tokenizer_dir)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return json.loads((output / "deployment.json").read_text())["tokenizer"]


def main():
    failures = []
    expect(failures, "template leading special, trailing special ignored", text_prompt_prefix_token_ids(tokenizer(WRAPPED), None), [7])
    expect(failures, "sequence of byte level and template", text_prompt_prefix_token_ids(
        tokenizer({"type": "Sequence", "processors": [BYTE_LEVEL, WRAPPED]}), None), [7])
    expect(failures, "bare template", text_prompt_prefix_token_ids(tokenizer(BARE), None), [])
    expect(failures, "byte level", text_prompt_prefix_token_ids(tokenizer(BYTE_LEVEL), None), [])
    expect(failures, "no post processor", text_prompt_prefix_token_ids(tokenizer(None), None), [])
    expect(failures, "bert cls", text_prompt_prefix_token_ids(
        tokenizer({"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}), None), [101])
    expect(failures, "roberta cls", text_prompt_prefix_token_ids(
        tokenizer({"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0], "trim_offsets": True,
                   "add_prefix_space": False}), None), [0])
    expect(failures, "add_bos_token false drops the template prefix",
           text_prompt_prefix_token_ids(tokenizer(WRAPPED), {"add_bos_token": False, "bos_token": "<s>"}), [])
    expect(failures, "add_bos_token true adds bos over a bare template",
           text_prompt_prefix_token_ids(tokenizer(BARE), {"add_bos_token": True, "bos_token": {"content": "<bos>"}}), [2])
    expect(failures, "class defaulting add_bos_token true adds bos",
           text_prompt_prefix_token_ids(tokenizer(BARE), {"tokenizer_class": "GemmaTokenizer", "bos_token": "<bos>"}), [2])
    expect(failures, "class default yields to explicit add_bos_token false",
           text_prompt_prefix_token_ids(tokenizer(BARE), {"tokenizer_class": "GemmaTokenizer", "add_bos_token": False,
                                                          "bos_token": "<bos>"}), [])
    expect(failures, "other class keeps the post processor",
           text_prompt_prefix_token_ids(tokenizer(BARE), {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>"}), [])
    refuses(failures, "unknown post processor", lambda: text_prompt_prefix_token_ids(tokenizer({"type": "Novel"}), None))
    refuses(failures, "template without sequence", lambda: text_prompt_prefix_token_ids(
        tokenizer({"type": "TemplateProcessing", "single": [special("<s>")], "special_tokens": SPECIALS}), None))
    refuses(failures, "bos not in tokenizer", lambda: text_prompt_prefix_token_ids(
        tokenizer(BARE), {"add_bos_token": True, "bos_token": "<missing>"}))
    refuses(failures, "add_bos_token without bos", lambda: text_prompt_prefix_token_ids(tokenizer(BARE), {"add_bos_token": True}))
    refuses(failures, "add_bos_token not boolean", lambda: text_prompt_prefix_token_ids(tokenizer(BARE), {"add_bos_token": "yes"}))
    with tempfile.TemporaryDirectory() as temporary:
        work = pathlib.Path(temporary)
        tiktoken = write_directory(work / "tiktoken", None, {"tokenizer_class": "TikTokenTokenizer", "bos_token": "[BOS]"})
        (tiktoken / "tiktoken.model").write_text("")
        expect(failures, "tiktoken directory", directory_prefix(tiktoken), [])
        refuses(failures, "directory without a tokenizer", lambda: directory_prefix(write_directory(work / "empty", None, None)))
        wrapped = write_directory(work / "wrapped", {**tokenizer(WRAPPED), "post_processor": {
            "type": "TemplateProcessing", "single": [special("<s>"), SEQUENCE_A], "special_tokens": SPECIALS}}, None)
        rendered = render(work, wrapped)
        expect(failures, "rendered deployment carries the derived prefix",
               None if rendered is None else rendered.get("text_prompt_prefix_token_ids"), [7])
        rendered = render(work, write_directory(work / "plain", tokenizer(BYTE_LEVEL), {"add_bos_token": False}))
        expect(failures, "rendered deployment without a prefix omits the field",
               None if rendered is None else sorted(rendered), ["path", "sha256", "vocabulary_size"])
        expect(failures, "a tokenizer section without --tokenizer-dir is refused", render(work, None), None)
    for name, model in stage_lane.MODELS.items():
        if "text_prompt_prefix_token_ids" in model:
            failures.append(f"{name}: hand-written text_prompt_prefix_token_ids; it must come from the tokenizer")
    for line in failures:
        print("FAILED: " + line)
    if failures:
        return 1
    print("PASS raw text prompt prefixes derive from the publisher tokenizer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
