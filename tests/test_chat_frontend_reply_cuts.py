#!/usr/bin/env python3
"""A history the chat layer renders from text, with no replay record for its replies, still names a cut one token before each reply began and where it ends, both on a control token, so the next turn resumes one token before the previous prompt's end and computes the reply again. The memory host's render_ids, cut_ids and continuation_ids answer with the cuts and the kept cuts the chat layer names for the same messages; cut_ids also keeps its last cut, so a completion on its ids leaves state where the conversation so far ends and the next turn resumes there."""
import asyncio
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
SYSTEM = "You are a careful assistant who reads files before answering. " * 12
SENTINEL = "NEXT"


def user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def said(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def reply_end(harness, prompt, after):
    return prompt.index(harness.tokenizer.token_to_id("<|im_end|>"), after)


async def run(tmp, check):
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        body = {"model": "alpha", "max_tokens": 16, "system": SYSTEM, "messages": [user("What is in the file?")]}
        engine.reply("Looking.</think>It holds a list.")
        status, _, text = await harness.post("/v1/messages", body)
        one = engine.requests[-1]
        first = len(one["prompt_token_ids"])
        check(status == 200 and len(one.get("cuts") or []) == 1 and one["keep"] == one["cuts"], f"a first turn names where its opening ends: {one.get('cuts')}")

        history = [user("What is in the file?"), said("The file holds three names."), user("Which names?")]
        engine.reply("Names.</think>Ada, Bo and Cy.")
        status, headers, text = await harness.post("/v1/messages", dict(body, messages=history))
        two = engine.requests[-1]
        prompt = two["prompt_token_ids"]
        ended = reply_end(harness, prompt, first)
        check(status == 200 and json.loads(text)["sparkpipe"]["replayed_tokens"] == 0, f"the store has no record of that reply, so nothing is replayed: {status}")
        check(two.get("cuts") == one["cuts"] + [first - 1, ended] and two["keep"] == one["cuts"],
              f"a reply rendered from text gets a cut one token before it began and one where it ends, without state: {two.get('cuts')} against {one['cuts'] + [first - 1, ended]}")
        check(prompt[:first - 1] == one["prompt_token_ids"][:first - 1] and harness.tokenizer.decode(prompt[ended:ended + 1], skip_special_tokens=False) == "<|im_end|>",
              "the first cut is one token before the earlier prompt's end and the second falls on the control token that closes the reply")
        check(engine.resumed(two) == first - 1, f"so the next turn resumes one token before the earlier prompt's end by the engine's rules: {engine.resumed(two)} of {first}")

        longer = history + [said("They are Ada, Bo and Cy."), user("Who is the oldest?")]
        engine.reply("Age.</think>Bo.")
        model = harness.frontend.models["alpha"]
        renderer, vocabulary, walk = model.renderer, model.renderer.cache.vocabulary, chat_frontend.SpanWalk
        rendered, counted, pieces_of, encode, count_of = [], [], renderer.pieces, vocabulary.encode, walk.count
        inside = []

        def counted_pieces(messages, tools, kwargs):
            rendered.append(sum(1 for message in messages if message.get("role") == "assistant"))
            return pieces_of(messages, tools, kwargs)

        def counted_encode(piece, special):
            if inside:
                counted.append(len(piece))
            return encode(piece, special)

        def counted_count(self, position):
            inside.append(position)
            try:
                return count_of(self, position)
            finally:
                inside.pop()

        renderer.pieces, vocabulary.encode, walk.count = counted_pieces, counted_encode, counted_count
        try:
            status, _, text = await harness.post("/v1/messages", dict(body, messages=longer))
        finally:
            renderer.pieces, vocabulary.encode, walk.count = pieces_of, encode, count_of
        check(rendered.count(2) == 1, f"a turn renders its conversation once: the reply cuts read the prompt's own pieces, not a second render: {rendered}")
        latest = engine.requests[-1]["prompt_token_ids"]
        whole, remembered = (len(harness.tokenizer.decode(latest[:size], skip_special_tokens=False)) for size in (len(latest), ended))
        check(0 < sum(counted) <= whole - remembered,
              f"the reply cuts tokenize only the text after the last reply they already counted: {sum(counted)} characters of {whole}, {remembered} of them counted by the turn before")
        three = engine.requests[-1]
        second = len(prompt)
        check(status == 200 and three["cuts"][:3] == two["cuts"] and three["cuts"][3:] == [second - 1, reply_end(harness, three["prompt_token_ids"], second)] and
              engine.resumed(three) == second - 1,
              f"every reply of the history has its two cuts, the earlier ones unchanged, and the turn resumes one token before the last prompt's end: {three.get('cuts')} {engine.resumed(three)}")

        engine.reply("Plain.</think>A list.")
        chat = {"model": "alpha", "max_tokens": 16, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is in the other file?"}]}
        status, _, text = await harness.post("/v1/chat/completions", chat)
        opened = engine.requests[-1]
        engine.reply("Plain.</think>Two.")
        status, _, text = await harness.post("/v1/chat/completions", dict(chat, messages=chat["messages"] + [{"role": "assistant", "content": "It holds another list."},
                                                                                                             {"role": "user", "content": "How many entries?"}]))
        followed = engine.requests[-1]
        start = len(opened["prompt_token_ids"])
        check(status == 200 and followed.get("cuts") == opened["cuts"] + [start - 1, reply_end(harness, followed["prompt_token_ids"], start)] and engine.resumed(followed) == start - 1,
              f"the Chat Completions shape names the same two cuts for a reply it renders from text: {followed.get('cuts')} {engine.resumed(followed)}")

        host = chat_frontend.MemoryHost(harness.frontend)
        plain = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Read the log."}, {"role": "assistant", "content": "The log is empty."}]
        asked = {"model": {"id": "alpha"}, "messages": plain + [{"role": "user", "content": SENTINEL + " and then?"}], "tools": None, "kwargs": {}, "sentinel": SENTINEL}
        cut = host.cut_ids(asked)
        whole = host.render_ids(dict(asked, messages=plain + [{"role": "user", "content": "Is it rotated?"}], suffix=None, start=""))
        check(cut["ids"] == whole["ids"][:len(cut["ids"])] and len(cut["cuts"]) == 3 and cut["cuts"] == whole["cuts"] and whole["keep"] == cut["cuts"][:1],
              f"the host's prompt up to a sentinel and its whole prompt name the same cuts, the opening's end and the reply's two, and the whole prompt keeps the opening's end: {cut['cuts']} {whole['cuts']} {whole['keep']}")
        check(cut["keep"] == [cut["cuts"][0], cut["cuts"][-1]] and harness.tokenizer.decode(cut["ids"][cut["cuts"][-1]:cut["cuts"][-1] + 1], skip_special_tokens=False) == "<|im_end|>",
              f"the prompt up to a sentinel also keeps its last cut, where the conversation so far ends: {cut['keep']}")
        await host.complete({"model": {"id": "alpha"}, "ids": cut["ids"], "max_tokens": 1, "reasoning_open": False, "kind": "prewarm", "trace": None, "cuts": cut["cuts"], "keep": cut["keep"]})
        warmed = engine.requests[-1]
        check(warmed.get("cuts") == cut["cuts"] and warmed.get("keep") == cut["keep"], f"a completion passes the host's cuts and kept cuts to the engine unchanged: {warmed.get('cuts')} {warmed.get('keep')}")
        engine.reply("Next.</think>No.")
        status, _, text = await harness.post("/v1/chat/completions", {"model": "alpha", "max_tokens": 16, "messages": plain + [{"role": "user", "content": "Is it rotated?"}]})
        next_turn = engine.requests[-1]
        check(status == 200 and next_turn["prompt_token_ids"] == whole["ids"] and next_turn["cuts"] == cut["cuts"] and engine.resumed(next_turn) == cut["cuts"][-1],
              f"the next turn on those messages names the same cuts and resumes where the completion kept state, at the end of the last reply: {next_turn.get('cuts')} {engine.resumed(next_turn)}")
        prompt_ids = whole["ids"]
        generated = engine.tokens("No.") + [harness.tokenizer.token_to_id("<|im_end|>")]
        continued = host.continuation_ids(dict(asked, messages=plain + [{"role": "user", "content": "Is it rotated?"}], prompt_ids=prompt_ids, generated_ids=generated,
                                               instruction="Summarize the turn.", suffix=None, start=""))
        check(continued["ids"][:len(prompt_ids)] == prompt_ids and continued["cuts"] == cut["cuts"] + [len(prompt_ids) - 1, len(prompt_ids) + len(generated) - 1] and
              continued["keep"] == cut["cuts"][:1],
              f"a continuation names the prompt's cuts, one token before the reply began and where the reply ends: {continued['cuts']} {continued['keep']}")
        missing = host.cut_ids(dict(asked, sentinel="ABSENT"))
        check(missing == {"ids": None, "cuts": [], "keep": []}, f"a prompt without its sentinel has no ids and no cuts: {missing}")


def main():
    check = fixture.Checks("test_chat_frontend_reply_cuts")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), check))
    return check.finish("a reply rendered from text has a cut before and after it, and the memory host's rendering methods answer with the chat layer's cuts")


if __name__ == "__main__":
    raise SystemExit(main())
