#!/usr/bin/env python3
"""Open WebUI rebuilds the newest user message on every tool round once a fetch produced a citation: a fixed guideline head, a context block of source tags (file sources with their text, fetched pages as empty markers) and the user's text. The chat layer recognises that wrapper only when its head hashes to the configured value and it parses back exactly; it then renders the user's text without the markers (keeping the head and the file sources when files are attached), labels each fetched result with its citation number, and so keeps every round an extension of the one before. Anything that does not match is left exactly as sent."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_frontend", ROOT / "tools" / "serving" / "chat_frontend.py")
chat_frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_frontend)
HEAD = "### Instructions\nUse the sources below and cite them as [id].\n\n"
WRAPPER = {"head_sha256": hashlib.sha256(HEAD.encode()).hexdigest(), "open": "<context>\n", "close": "\n</context>\n", "separator": "\n"}
QUESTION = "Compare the two release notes."


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL test_chat_frontend_citations: {message}")


def wrapped(sources, base=QUESTION):
    return HEAD + "<context>\n" + "\n".join(sources) + "\n</context>\n" + "\n" + base


def marker(n, url):
    return f'<source id="{n}" name="{url}" resource-id="{url}"></source>'


def fetch(i, url):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "fetch_url", "arguments": json.dumps({"url": url}, separators=(",", ":"))}}]}


def result(i, text):
    return {"role": "tool", "tool_call_id": f"call_{i}", "content": text}


def main():
    a, b = "https://a.example/notes", "https://b.example/notes"
    history = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    round1 = history + [{"role": "user", "content": QUESTION}]
    round2 = history + [{"role": "user", "content": wrapped([marker(1, a)])}, fetch(1, a), result(1, "page A")]
    round3 = history + [{"role": "user", "content": wrapped([marker(1, a), marker(2, b)])}, fetch(1, a), result(1, "page A"), fetch(2, b), result(2, "page B")]
    first, info1 = chat_frontend.canonical_citations(round1, WRAPPER)
    second, info2 = chat_frontend.canonical_citations(round2, WRAPPER)
    third, info3 = chat_frontend.canonical_citations(round3, WRAPPER)
    check(first == round1 and info1 == {}, "a message without the wrapper is left as sent")
    check(second[3]["content"] == QUESTION and third[3]["content"] == QUESTION, "the user's text is rendered without the citation block")
    check(second[5]["content"] == "[This result is source [1]; cite it inline as [1].]\npage A", second[5])
    check(third[7]["content"].startswith("[This result is source [2]; cite it inline as [2].]\n"), third[7])
    check(second == third[:len(second)] and first[:4] == second[:4], "each round's canonical messages extend the previous round's")
    check(info3 == {"citation_markers": 2, "citation_labelled": 2, "citation_unmatched": 0, "citation_files": 0}, info3)
    check(round3[3]["content"].startswith(HEAD), "the caller's messages are not modified")

    document = '<source id="1" name="spec.pdf" resource-type="file" resource-id="f-1">The spec says 40 ms.\n</source> is not the end</source>'
    files2 = history + [{"role": "user", "content": wrapped([document, marker(2, a)])}, fetch(1, a), result(1, "page A")]
    files3 = history + [{"role": "user", "content": wrapped([document, marker(2, a), marker(3, b)])}, fetch(1, a), result(1, "page A"), fetch(2, b), result(2, "page B")]
    files1 = history + [{"role": "user", "content": wrapped([document])}]
    kept1, _ = chat_frontend.canonical_citations(files1, WRAPPER)
    kept2, info = chat_frontend.canonical_citations(files2, WRAPPER)
    kept3, _ = chat_frontend.canonical_citations(files3, WRAPPER)
    check(kept1[3]["content"] == wrapped([document]) and kept2[3]["content"] == kept1[3]["content"] == kept3[3]["content"],
          "with an attached file the head and the file source stay, and the user message is the same in every round")
    check(kept2[5]["content"].startswith("[This result is source [2];") and info["citation_files"] == 1, "fetch numbers follow the file sources")

    duplicate = history + [{"role": "user", "content": wrapped([marker(1, a)])}, fetch(1, a), result(1, "page A"), fetch(2, a), result(2, "page A again")]
    twice, _ = chat_frontend.canonical_citations(duplicate, WRAPPER)
    check(twice[5]["content"].startswith("[This result is source [1];") and twice[7]["content"].startswith("[This result is source [1];"), "a page fetched twice keeps its one number")
    failed = history + [{"role": "user", "content": wrapped([marker(1, a)])}, fetch(1, a), result(1, "page A"), fetch(2, b), result(2, '{"error": "timeout"}')]
    check(chat_frontend.canonical_citations(failed, WRAPPER)[0][7]["content"] == '{"error": "timeout"}', "a failed fetch without a marker gets no label")
    images = round2 + [{"role": "user", "content": [{"type": "text", "text": "Here are the images"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]
    check(chat_frontend.canonical_citations(images, WRAPPER)[0][3]["content"] == QUESTION, "a trailing tool-image message does not hide the wrapped user message")

    forged = history + [{"role": "user", "content": "### Other head\n" + "<context>\n" + marker(1, a) + "\n</context>\n\n" + QUESTION}, fetch(1, a), result(1, "page A")]
    check(chat_frontend.canonical_citations(forged, WRAPPER)[0] == forged, "a block whose head does not hash to the configured head is left as sent")
    broken = history + [{"role": "user", "content": HEAD + "<context>\n" + marker(1, a) + "\n</context>\n" + QUESTION}]
    check(chat_frontend.canonical_citations(broken, WRAPPER)[0] == broken, "a block that does not parse back exactly is left as sent")
    listed = history + [{"role": "user", "content": [{"type": "text", "text": wrapped([marker(1, a)])}]}, fetch(1, a), result(1, "page A")]
    check(chat_frontend.canonical_citations(listed, WRAPPER)[0][3]["content"] == [{"type": "text", "text": QUESTION}], "list content is handled per text part")
    print("PASS test_chat_frontend_citations: wrapped user messages render as the user's text, fetched results carry their citation numbers, rounds extend each other, files keep their sources, and unmatched blocks are left as sent")


if __name__ == "__main__":
    main()
