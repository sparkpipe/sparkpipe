#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
RequestError = chat_frontend.RequestError

HUB = ROOT / "deployment" / "hub" / "chat_frontend.json"
QWEN_TEMPLATE = ROOT / "model_contracts" / "references" / "qwen38_27b_chat_template.jinja"
QWEN_DIGEST = ROOT / "model_contracts" / "qwen4_flash_authoritative.json"
QWEN_TEMPLATE_MODELS = ("qwen3.8-27b", "qwen3.8-27b-fp8", "qwen3.8-flash")
THINKING = ("low", "medium", "high")
QWEN_LEVELS = ("reasoning_effort", ("low", "medium", "xhigh"))
GRADED = {
    "qwen3.8-27b": QWEN_LEVELS,
    "qwen3.8-27b-fp8": QWEN_LEVELS,
    "qwen3.8-flash": QWEN_LEVELS,
    "qwen3.8-max": QWEN_LEVELS,
    "glm-5.3": ("reasoning_effort", ("low", "high", "max")),
    "glm-5.3-fp8": ("reasoning_effort", ("low", "high", "max")),
    "glm-5.3-flash": ("reasoning_effort", ("low", "high", "max")),
    "glm-5.3-flash-fp8": ("reasoning_effort", ("low", "high", "max")),
    "sage-s2.1-flash": ("reasoning_effort", ("low", "high", "max")),
    "kimi-k3": ("thinking_effort", ("low", "high", "max")),
    "kimi-k3.1-pro": ("thinking_effort", ("low", "high", "max")),
    "deepseek-v4.1-flash": ("effort", ("low", "high", "max")),
    "muse-glimmer-30b": ("reasoning_strength", ("low", "medium", "high")),
}
SWITCH_ONLY = dict.fromkeys(("gemma-4-31b", "gemma-4-26b-a4b", "hy4-preview", "hy4-fp8", "ling-3.0-flash", "ling-3.0-flash-fin", "ling-3.1-flash", "ling-3.2-air", "laguna-s-2.1",
                              "mimo-v2.6-flash", "mimo-v2.6-pro", "k-exaone-2.0", "a.x-k2"))
SWITCH_ONLY["solar-open-2"] = ("reasoning_effort", {"medium", "high", "xhigh"})
LEVEL_LINE = "Reasoning effort is set to "
OPEN = "<|im_start|>assistant\n<think>\n"
CLOSED = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
QUESTION = [{"role": "user", "content": "Why is the sky blue?"}]


def mapping_failures(models):
    failures = []
    for model in models:
        name, efforts = model["id"], model.get("reasoning_effort", {})
        if (name in GRADED) == (name in SWITCH_ONLY):
            failures.append(f"{name} must be classified once from its publisher's template: graded thinking levels or a thinking switch only")
            continue
        if name in SWITCH_ONLY and SWITCH_ONLY[name] is None:
            if len({json.dumps(efforts.get(label), sort_keys=True) for label in THINKING}) != 1:
                failures.append(f"{name}'s template only switches thinking on, so low, medium and high must render alike: {efforts}")
            continue
        if name in SWITCH_ONLY:
            variable, opening = SWITCH_ONLY[name]
            for label in THINKING:
                if (efforts.get(label) or {}).get(variable) not in opening:
                    failures.append(f"{name}'s template opens thinking only for {variable} in {sorted(opening)}, so {label} must use one of them: {efforts}")
            if "minimal" in efforts and efforts["minimal"].get(variable) in opening:
                failures.append(f"{name} minimal must close thinking, with a {variable} outside {sorted(opening)}")
            continue
        variable, levels = GRADED[name]
        for label, level in zip(THINKING, levels):
            got = (efforts.get(label) or {}).get(variable)
            if got != level:
                failures.append(f"{name} {label} must set the template's {variable} to {level!r}, not {got!r}")
        if variable not in model.get("template_kwargs", []):
            failures.append(f"{name} must accept its template's {variable} in chat_template_kwargs")
    for name in sorted((set(GRADED) | set(SWITCH_ONLY)) - {model["id"] for model in models}):
        failures.append(f"{name} is classified here but not served")
    return failures


def tokenizer_file(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<|im_start|>", "<|im_end|>", "<|endoftext|>"])
    tokenizer.save(str(path))


def rendered(model, body):
    try:
        return model.render(QUESTION, None, chat_frontend.template_settings(model, body))[0]
    except RequestError as error:
        return f"refused: {error.message}"


def template_failures(models):
    failures = []
    digest = json.loads(QWEN_DIGEST.read_text())["digest_freeze"]["files"]["chat_template.jinja"]["sha256"]
    if hashlib.sha256(QWEN_TEMPLATE.read_bytes()).hexdigest() != digest:
        failures.append(f"{QWEN_TEMPLATE.name} is not the publisher's template that {QWEN_DIGEST.name} freezes")
    by_id = {model["id"]: model for model in models}
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        (directory / "chat_template.jinja").write_bytes(QWEN_TEMPLATE.read_bytes())
        (directory / "tokenizer_config.json").write_text("{}")
        tokenizer_file(directory / "tokenizer.json")
        for name in QWEN_TEMPLATE_MODELS:
            model = chat_frontend.Model(dict(by_id[name], model_dir=str(directory)), ROOT)
            texts = {label: rendered(model, {"reasoning_effort": label}) for label in model.reasoning_effort}
            checks = [
                (sorted(texts) == ["high", "low", "medium", "minimal"], f"offers minimal, low, medium and high, not {sorted(texts)}"),
                (texts.get("minimal", "").endswith(CLOSED) and LEVEL_LINE not in texts.get("minimal", ""), "minimal closes the thinking block"),
                (LEVEL_LINE + "low." in texts.get("low", "") and texts.get("low", "").endswith(OPEN), "low renders the template's low instruction"),
                (LEVEL_LINE not in texts.get("medium", "") and texts.get("medium", "").endswith(OPEN), "medium renders the template's medium, which adds no instruction"),
                (LEVEL_LINE + "xhigh." in texts.get("high", "") and texts.get("high", "").endswith(OPEN), "high renders the template's top level, xhigh"),
                (len(set(texts.values())) == len(texts), "every label renders a different prompt"),
                (LEVEL_LINE + "xhigh." in rendered(model, {}), "a request without reasoning_effort gets the template's own default, xhigh"),
                (rendered(model, {"chat_template_kwargs": {"reasoning_effort": "medium"}}) == texts.get("medium"),
                 "chat_template_kwargs can set the template's level directly"),
            ]
            failures.extend(f"{name}: {message}" for passed, message in checks if not passed)
            try:
                chat_frontend.template_settings(model, {"reasoning_effort": "xhigh"})
                failures.append(f"{name}: an effort the model does not offer was accepted")
            except RequestError as error:
                if error.status != 400:
                    failures.append(f"{name}: an effort the model does not offer answered {error.status}, not 400")
    return failures


def main():
    models = json.loads(HUB.read_text())["models"]
    failures = mapping_failures(models) + template_failures(models)
    for line in failures:
        print("FAIL " + line)
    if failures:
        return 1
    print("PASS test_chat_frontend_effort_levels: every served model's low, medium and high reach the levels its publisher's template defines, "
          "and Qwen 3.8's template renders minimal, low, medium and high as four different prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
