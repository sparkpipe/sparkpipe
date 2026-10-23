#!/usr/bin/env python3
"""The chat layer's memory helpers build prompts that line up with what the engine caches: the curator continuation is the answer's exact prompt and generated tokens (any trailing stop token dropped, template closing tokens the model already wrote not repeated) followed by the template's own tokens for the end of the assistant turn, the instruction as plain text in a user turn and the generation prompt, closing an opened reasoning channel with the model's suffix; and a cut renders the template up to a sentinel exactly as the full render would encode it."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
START, END, THINK = "<|im_start|>", "<|im_end|>", "<think>"
WORDS = ["[UNK]", "user", "assistant", "system", "hi", "there", "done", "summarize", "it", "close", "q", "a", "b"]


def vocabulary():
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(WORDS)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([START, END])
    tokenizer.add_tokens([AddedToken(THINK, special=False)])
    return chat_frontend.TokenizersVocabulary.from_tokenizer(tokenizer)


def model(directory, words, closing, opening):
    template = ("{% for m in messages %}" + START + "{{ m.role }} {{ m.content }}{% if m.role == 'assistant' %}" + closing + "{% endif %}" + END + " {% endfor %}"
                "{% if add_generation_prompt %}" + START + "assistant " + opening + "{% endif %}")
    (directory / "chat_template.jinja").write_text(template)
    (directory / "tokenizer_config.json").write_text(json.dumps({}))
    renderer = chat_frontend.TemplateRenderer(directory, {}, words)
    return SimpleNamespace(renderer=renderer, vocabulary=words, template_defaults={}, reasoning={"start": THINK, "end": "close"},
                           stop_token_ids=[words.token_id(END)], render=renderer.render)


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_chat_memory_frontend: {message}")


def main():
    words = vocabulary()
    frontend = chat_frontend.Frontend.__new__(chat_frontend.Frontend)
    token = words.token_id
    messages = [{"role": "system", "content": "hi"}, {"role": "user", "content": "q"}]
    with tempfile.TemporaryDirectory() as scratch:
        plain = model(Path(scratch), words, "", "")
        prompt = plain.render(messages, None, {})[1]
        generated = [token("a"), token("b"), token(END)]
        ids, reasoning_open = frontend.memory_continuation_ids(plain, messages, None, {}, prompt, generated, "summarize it", "close")
        tail = [token(END), token(START), token("user"), token("summarize"), token("it"), token(END), token(START), token("assistant")]
        check(ids == prompt + [token("a"), token("b")] + tail, f"continuation must be prompt + reply + template tail: {ids}")
        check(not reasoning_open, "no reasoning channel is open when the generation prompt does not open one")
        check(frontend.memory_cut_ids(plain, messages + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "there " + chr(0xe000) + "X"}], None, {}, chr(0xe000) + "X")
              == plain.render(messages + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "there"}], None, {})[1][:-3],
              "a cut encodes the template up to the sentinel exactly as the full render does")
        check(frontend.memory_cut_ids(plain, messages, None, {}, "missing") is None, "a cut without its sentinel yields nothing")
    with tempfile.TemporaryDirectory() as scratch:
        closing = model(Path(scratch), words, " done", THINK)
        prompt = closing.render(messages, None, {})[1]
        check(prompt[-1] == token(THINK), "the generation prompt opens the reasoning channel")
        generated = [token("close"), token("a"), token("done")]
        ids, reasoning_open = frontend.memory_continuation_ids(closing, messages, None, {}, prompt, generated, "summarize it", "close")
        check(ids == prompt + generated + [token(END), token(START), token("user"), token("summarize"), token("it"), token(END), token(START), token("assistant"), token(THINK), token("close")],
              f"closing tokens the model already wrote are not repeated, and the suffix closes the reopened reasoning channel: {ids}")
        check(not reasoning_open, "the suffix leaves the reasoning channel closed")
        ids, reasoning_open = frontend.memory_continuation_ids(closing, messages, None, {}, prompt, generated, "summarize it", None)
        check(reasoning_open and ids[-1] == token(THINK), "without a suffix the curator reply is parsed as reasoning first")
    for body, expected in (({}, None), ({"num_ctx": -1}, None), ({"num_ctx": 0}, None), ({"num_ctx": 65536}, 65536), ({"num_ctx": 65536.0}, 65536), ({"num_ctx": "65536"}, 65536), ({"num_ctx": "-1"}, None)):
        check(chat_frontend.context_target(body) == expected, f"Open WebUI's context length {body} is the memory target {expected}")
    for body in ({"num_ctx": "lots"}, {"num_ctx": True}, {"num_ctx": -5}, {"num_ctx": 1.5}):
        try:
            chat_frontend.context_target(body)
            check(False, f"{body} must be refused")
        except chat_frontend.RequestError as error:
            check(error.status == 400, f"{body} is a client error")
    print("PASS test_chat_memory_frontend: curator continuations extend the answer's cached prompt and reply with the template's own tail; cuts match the full render up to the sentinel; Open WebUI's context length becomes the memory target")


if __name__ == "__main__":
    main()
