#!/usr/bin/env python3
"""The chat layer names a cut where each block of a prompt's head ends, so prompts that share a head resume there: the tools, each system block, the end of the opening and each content block of the first user message but its last. A block's end is not a token boundary, so its cut is the last cache block at or before the tokens of the prompt's text through the block, tokenized by itself, less a margin; the end of the opening falls on a control token and is exact. State is asked for at the tools, the system blocks and the opening, and in the first user message only at a block that carries cache_control. A second conversation that shares the tools and the first system block resumes at that block's cut, and one that shares the first user message's document resumes at the document's cut, by the engine's rules. The Responses and Chat Completions shapes name the same cuts for the same structure."""
import asyncio
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
chat_frontend = fixture.chat_frontend
prompt_cuts = chat_frontend.prompt_cuts
BLOCK = prompt_cuts.BLOCK_TOKENS
MARGIN = prompt_cuts.MARGIN_TOKENS


def passage(seed, sentences):
    words = ["the answer is", "read the file", "list the directory", "call the tool with path", "let me check", "the weather in Paris", "is sunny today"]
    return " ".join("%s %d %s." % (words[(seed + index) % len(words)], seed * 31 + index, words[(seed + 2 * index) % len(words)]) for index in range(sentences))


TOOLS = [{"name": "read%d" % index, "description": passage(40 + index, 3), "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
         for index in range(4)]
RULES, PROJECT, HABITS = passage(1, 14), passage(2, 14), passage(3, 14)
DOCUMENT, NOTES, QUESTION = passage(4, 18), passage(5, 14), "What does the document say?"


def system(*texts, marked=()):
    return [dict({"type": "text", "text": text}, **({"cache_control": {"type": "ephemeral"}} if index in marked else {})) for index, text in enumerate(texts)]


def asks(*texts, marked=()):
    return [{"role": "user", "content": system(*texts, marked=marked)}]


def expected(harness, prompt, block):
    text = harness.tokenizer.decode(prompt, skip_special_tokens=False)
    end = text.index(block) + len(block)
    alone = len(harness.tokenizer.encode(text[:end], add_special_tokens=False).ids)
    return alone, prompt_cuts.structural_cut(alone)


async def run(tmp, check):
    async with fixture.Harness(tmp) as harness:
        engine = harness.engine
        body = {"model": "alpha", "max_tokens": 16, "tools": TOOLS, "system": system(RULES, PROJECT, HABITS, marked=(1,)),
                "messages": asks(DOCUMENT, NOTES, QUESTION, marked=(0,))}
        engine.reply("Reading.</think>It says many things.")
        status, _, text = await harness.post("/v1/messages", body)
        first = engine.requests[-1]
        prompt, cuts, keep = first["prompt_token_ids"], first.get("cuts") or [], first.get("keep") or []
        rendered = harness.tokenizer.decode(prompt, skip_special_tokens=False)
        tools_alone = len(harness.tokenizer.encode(rendered[:rendered.index(RULES)], add_special_tokens=False).ids)
        tools_cut = prompt_cuts.structural_cut(tools_alone)
        rules, project, document, notes = (expected(harness, prompt, block) for block in (RULES, PROJECT, DOCUMENT, NOTES))
        opening = next((cut for cut in cuts if harness.tokenizer.decode(prompt[cut:], skip_special_tokens=False).startswith("<|im_start|>user\n")), None)
        check(status == 200 and cuts == [tools_cut, rules[1], project[1], opening, document[1], notes[1]],
              f"a prompt names the tools' end, each system block's end, the opening's end and each block of the first user message but its last: {cuts} "
              f"against {[tools_cut, rules[1], project[1], opening, document[1], notes[1]]}")
        check(all(cut % BLOCK == 0 and 0 <= alone - MARGIN - cut < BLOCK for alone, cut in (rules, project, document, notes)) and opening % BLOCK != 0,
              f"a block's cut is the last cache block at or before its own token count less the margin, and the opening's end is exact: {rules} {project} {document} {notes} {opening}")
        check(keep == [tools_cut, rules[1], project[1], opening, document[1]],
              f"state is asked for at the tools, the system blocks and the opening, and in the first user message only at the block that carries cache_control: {keep}")
        check(all(prompt[:cut] == harness.tokenizer.encode(rendered, add_special_tokens=False).ids[:cut] for cut in cuts) and
              harness.tokenizer.encode(rendered[:rendered.index(PROJECT) + len(PROJECT)], add_special_tokens=False).ids[:project[1]] == prompt[:project[1]],
              "the tokens before a block's cut are the tokens of the block's own text, so another prompt with the same block has them too")
        engine.reply("Other.</think>Something else.")
        status, _, text = await harness.post("/v1/messages", dict(body, system=system(RULES, passage(9, 14), HABITS), messages=asks("another question")))
        other = engine.requests[-1]
        check(status == 200 and other["cuts"][:2] == [tools_cut, rules[1]] and engine.resumed(other) == rules[1],
              f"a conversation that shares the tools and the first system block resumes at that block's cut by the engine's rules: {other.get('cuts')} {engine.resumed(other)}")
        engine.reply("Again.</think>A second answer.")
        status, _, text = await harness.post("/v1/messages", dict(body, messages=asks(DOCUMENT, passage(8, 14), "And what else?", marked=(0,))))
        again = engine.requests[-1]
        check(status == 200 and again["cuts"][:5] == cuts[:5] and engine.resumed(again) == document[1],
              f"a conversation that shares the opening and the first user message's document resumes at the document's cut: {again.get('cuts')} {engine.resumed(again)}")
        engine.reply("Short.</think>Fine.")
        status, _, text = await harness.post("/v1/messages", {"model": "alpha", "max_tokens": 16, "system": "Be brief.", "messages": [{"role": "user", "content": "hello world"}]})
        short = engine.requests[-1]
        check(status == 200 and "cuts" not in short and "keep" not in short, f"a prompt whose head is under one cache block names no cut: {short.get('cuts')}")

        engine.reply("Reading.</think>It says many things.")
        tools = [{"type": "function", "name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]} for tool in TOOLS]
        status, _, text = await harness.post("/v1/responses", {"model": "alpha", "instructions": RULES, "tools": tools, "max_output_tokens": 16,
                                                               "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": DOCUMENT},
                                                                                                                      {"type": "input_text", "text": QUESTION}]}]})
        asked = engine.requests[-1]
        prompt = asked["prompt_token_ids"]
        shown = harness.tokenizer.decode(prompt, skip_special_tokens=False)
        part = expected(harness, prompt, DOCUMENT)
        ends = next((cut for cut in asked.get("cuts") or [] if harness.tokenizer.decode(prompt[cut:], skip_special_tokens=False).startswith("<|im_start|>user\n")), None)
        check(status == 200 and ends is not None and asked["cuts"][-1] == part[1] and part[1] not in asked["keep"] and ends in asked["keep"] and shown.index(RULES) < shown.index(DOCUMENT),
              f"the Responses shape names the opening's end and the first input part's end, the part without state: {asked.get('cuts')} {asked.get('keep')} {part} {ends}")

        engine.reply("Reading.</think>It says many things.")
        status, _, text = await harness.post("/v1/chat/completions", {"model": "alpha", "max_tokens": 16, "messages": [
            {"role": "system", "content": RULES}, {"role": "system", "content": PROJECT},
            {"role": "user", "content": [{"type": "text", "text": DOCUMENT}, {"type": "text", "text": QUESTION}]}]})
        plain = engine.requests[-1]
        prompt = plain["prompt_token_ids"]
        check(status == 200 and len(plain.get("cuts") or []) >= 2 and plain["keep"] == [cut for cut in plain["cuts"] if cut <= max(plain["keep"])] and
              expected(harness, prompt, RULES)[1] in plain["cuts"],
              f"the Chat Completions shape names the first system message's end and the opening's end: {plain.get('cuts')} {plain.get('keep')} {expected(harness, prompt, RULES)}")


def main():
    check = fixture.Checks("test_chat_frontend_block_cuts")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), check))
    return check.finish("a prompt names a cut where each block of its head ends, and prompts that share a block resume there")


if __name__ == "__main__":
    raise SystemExit(main())
