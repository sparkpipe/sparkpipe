#!/usr/bin/env python3
"""Chat memory's core keeps a conversation's layout cache-friendly and its records safe: a signed identity is accepted only with a valid HS256 signature and an unexpired claim; a chat splits into finished exchanges and the current one; an exchange's key ignores ids and timestamps but not content; the ledger holds only committed exchanges, in commit order, and every other finished exchange stays open in full after it with its id; a commit only appends to the ledger text; the system note is present from the first turn; open nodes come only from the ledger and show the user's message and the answer first, with tool results only as pointers; expand pages long exchanges; tool results get a header with their id, tool, arguments and size, are split into sections at line breaks when long, are capped to their head and tail with a marker, cannot forge ids, section marks or memory tags, collapse to a stub with the kept sections, and page by section through expand; search_history finds literal text (phrase, all words near each other, or any word) in every earlier exchange including tool results and in ledger and group lines, reporting each match's exchange and expand part and capping its output; curator output is validated line by line with long lines cut and the open set trimmed to its budget; groups replace contiguous runs of ledger lines, nest, apply only where their members are, and are planned oldest first only when the ledger is over its budget, down to about half of it; lines are immutable once stored, survive a restart and a torn last record, stay private on disk, and temporary chats never reach disk."""
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import stat
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
cm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cm)


def b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def jwt(payload, secret, alg="HS256"):
    head = b64(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    body = b64(json.dumps(payload).encode())
    signature = b64(hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{signature}"


def chat(n, follow_up=False):
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(1, n):
        messages += [{"role": "user", "content": f"question {i}", "id": f"u{i}"}, {"role": "assistant", "content": f"answer {i}", "timestamp": i}]
    messages.append({"role": "user", "content": f"question {n}"})
    if follow_up:
        messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": "{\"query\": \"x\"}"}}]},
                     {"role": "tool", "tool_call_id": "c1", "name": "search_web", "content": "result"}]
    return messages


secret = "s" * 40
now = time.time()
assert cm.verify_jwt(jwt({"sub": "user-1", "exp": now + 60}, secret), secret)["sub"] == "user-1"
assert cm.verify_jwt(jwt({"sub": "user-1", "exp": now + 60}, "other" * 8), secret) is None
assert cm.verify_jwt(jwt({"sub": "user-1", "exp": now - 1}, secret), secret) is None
assert cm.verify_jwt(jwt({"sub": "user-1"}, secret), secret) is None
assert cm.verify_jwt(jwt({"sub": "user-1", "exp": now + 60}, secret, alg="none"), secret) is None
assert cm.verify_jwt("not.a.jwt", secret) is None and cm.verify_jwt("garbage", secret) is None

head, exchanges, current = cm.split(chat(5))
assert len(head) == 1 and len(exchanges) == 4 and current[0]["content"] == "question 5", (head, exchanges, current)
head, exchanges, current = cm.split(chat(3, follow_up=True))
assert len(exchanges) == 2 and len(current) == 3 and current[-1]["role"] == "tool"
assert cm.split(chat(3) + [{"role": "assistant", "content": "done"}]) is None
images = {"role": "user", "content": [{"type": "text", "text": "Here are the images from the tool results above. Please analyze them."}, {"type": "image_url", "image_url": {"url": "data:,"}}]}
plotted = chat(2) + [{"role": "assistant", "content": "", "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "execute_code", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "c9", "content": "ok"}, images]
head_i, exchanges_i, current_i = cm.split(plotted)
assert len(exchanges_i) == 1 and current_i[-1] is images and current_i[0]["content"] == "question 2", "images from tool results stay in their exchange"
head_i, exchanges_i, current_i = cm.split(plotted + [{"role": "assistant", "content": "a plot"}, {"role": "user", "content": "next"}])
assert len(exchanges_i) == 2 and exchanges_i[1][-1]["content"] == "a plot" and current_i[0]["content"] == "next"
assert cm.split([{"role": "assistant", "content": "hi"}]) is None

a = [{"role": "user", "content": "q", "id": "1"}, {"role": "assistant", "content": "a", "timestamp": 5}]
b = [{"role": "user", "content": "q", "id": "2"}, {"role": "assistant", "content": "a", "timestamp": 9}]
c = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a!"}]
assert cm.exchange_key(a) == cm.exchange_key(b) != cm.exchange_key(c)

head, exchanges, current = cm.split(chat(6))
keys = [cm.exchange_key(e) for e in exchanges]
none = cm.build_layout(head, exchanges, current, {}, [])
users = [m["content"] for m in none["messages"] if m["role"] == "user"]
assert none["lined"] == 0 and none["working"] == [0, 1, 2, 3, 4] and cm.SYSTEM_NOTE in none["messages"][0]["content"] and not any(cm.LEDGER_OPEN in u for u in users)
assert users[0] == "[e1] question 1" and users[-1] == cm.REFERENCE_OPEN + cm.REFERENCE_CLOSE + "[e6] question 6", users
lines_all = {k: f"line {i + 1}" for i, k in enumerate(keys)}
assert cm.build_layout(head, exchanges, current, lines_all, [])["lined"] == 0, "lines alone never take an exchange out of the working set"
committed = cm.build_layout(head, exchanges, current, lines_all, ["e1", "e4", "e9", "e3"], [keys[0], keys[3], keys[1]])
assert committed["lined"] == 3 and committed["working"] == [2, 4] and committed["keys"] == [keys[0], keys[3], keys[1]]
first_user = [m for m in committed["messages"] if m["role"] == "user"][0]["content"]
assert first_user == cm.LEDGER_OPEN + "[e1] line 1\n[e4] line 4\n[e2] line 2\n" + cm.LEDGER_CLOSE + "[e3] question 3", "the ledger lists commits in commit order before the first open exchange"
last = committed["messages"][-1]["content"]
assert last.startswith(cm.REFERENCE_OPEN) and '<history id="e1">' in last and 'id="e4"' in last and 'id="e3"' not in last and last.endswith("[e6] question 6")
assert committed["open"] == ["e1", "e4"], "only committed exchanges can be opened; open ones are shown in full anyway"
more = cm.build_layout(head, exchanges, current, lines_all, [], [keys[0], keys[3], keys[1], keys[2]])
grown = [m for m in more["messages"] if m["role"] == "user"][0]["content"]
assert grown.startswith(first_user[:first_user.index(cm.LEDGER_CLOSE)]), "a commit only appends to the ledger text"
assert cm.build_layout(head, exchanges, current, {}, [], [keys[0]])["lined"] == 0, "a commit without a line keeps the exchange open"
everything = cm.build_layout(head, exchanges, current, lines_all, [], keys)
assert everything["working"] == [] and everything["messages"][-1]["content"].startswith(cm.LEDGER_OPEN + "[e1] line 1") and everything["messages"][-1]["content"].endswith("[e6] question 6")
follow = cm.split(chat(4, follow_up=True))
laid = cm.build_layout(*follow, {}, [])
assert laid["messages"][-1]["role"] == "tool" and laid["messages"][-3]["content"].endswith("[e4] question 4")

long_exchange = [[{"role": "user", "content": "x" * 25}, {"role": "assistant", "content": "y" * 25}]]
one = cm.expand_text(long_exchange, "e1", 1, 20)
assert "part 1 of" in one and "ask for part 2" in one, one
assert "has" in cm.expand_text(long_exchange, "e1", 9, 20)
assert cm.expand_text(long_exchange, "e7", 1, 20).startswith("No exchange")
assert cm.expand_text(long_exchange, "../e1", 1, 20).startswith("No exchange")

missing = [3, 4]
sizes = {"e1": 400, "e2": 900, "e3": 50}
text = 'noise {"lines": {"e4": "The user asked A; the assistant said B.", "e5": "' + "w " * 400 + '"}, "open": ["e2", "e1", "e3", "e9", "e2"]} trailing'
lines, opened, complete = cm.parse_curator(text, missing, 5, sizes, 1000)
assert complete and lines[3].startswith("The user asked A") and len(lines[4]) <= cm.LINE_LIMIT_CHARS + 4 and lines[4].endswith("...")
assert opened == ["e2"], opened
lines, opened, complete = cm.parse_curator('{"lines": {"e4": "only one"}}', missing, 5, sizes, 1000)
assert not complete and lines == {3: "only one"}
assert cm.parse_curator("no json here", missing, 5, sizes, 1000) == (None, [], False)
assert cm.fallback_line([{"role": "user", "content": "What is 2+2? Please"}, {"role": "assistant", "content": "It is 4. Nothing else."}]) == "(auto) The user said: What is 2+2? Please; the assistant answered: It is 4."

with tempfile.TemporaryDirectory() as tmp:
    store = cm.ChatStore(tmp)
    assert store.append("user-1", "chat-abc", {"type": "line", "key": "k1", "line": "first", "source": "curator"})
    assert not store.append("user-1", "chat-abc", {"type": "line", "key": "k1", "line": "changed", "source": "curator"})
    store.append("user-1", "chat-abc", {"type": "open", "after": "k1", "ids": ["e1"]})
    store.append("user-1", "local:temp", {"type": "line", "key": "k9", "line": "temp", "source": "curator"})
    path = store.path("user-1", "chat-abc")
    with path.open("a") as handle:
        handle.write('{"type": "line", "key": "k2", "li')
    again = cm.ChatStore(tmp).load("user-1", "chat-abc")
    assert again["lines"] == {"k1": "first"} and again["open"] == {"k1": ["e1"]}, again
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert not store.path("user-1", "local:temp").exists() and cm.ChatStore(tmp).load("user-1", "local:temp")["lines"] == {}
    assert "user-1" not in str(path) and "chat-abc" not in str(path)

head, exchanges, current = cm.split(chat(41))
keys = [cm.exchange_key(m) for m in exchanges]
lines = {key: f"line {i + 1}" for i, key in enumerate(keys)}
nodes = cm.ledger_nodes(exchanges, lines, {}, list(range(40)))
groups = {"g1": {"members": ["x:" + k for k in keys[:16]], "line": "the first sixteen"},
          "g2": {"members": ["x:" + k for k in keys[16:32]], "line": "the next sixteen"},
          "g3": {"members": ["g1", "g2"], "line": "thirty-two in all"},
          "g4": {"members": ["x:" + keys[5], "x:" + keys[6]], "line": "inside g1 already"},
          "g5": {"members": ["x:missing", "x:" + keys[33]], "line": "not on this branch"}}
grouped = cm.ledger_nodes(exchanges, lines, groups, list(range(40)))
assert [n["id"] for n in grouped] == ["g3"] + [f"e{i}" for i in range(33, 41)], [n["id"] for n in grouped]
text = cm.ledger_text(grouped)
assert text.startswith("[g3: e1-e32] thirty-two in all\n[e33] line 33\n") and text.endswith("[e40] line 40\n"), text
assert set(cm.group_nodes(grouped)) == {"g1", "g2", "g3"} and cm.ledger_text(cm.group_nodes(grouped)["g3"]["children"]) == "[g1: e1-e16] the first sixteen\n[g2: e17-e32] the next sixteen\n"
assert [n["id"] for n in cm.ledger_nodes(exchanges, lines, groups, list(range(40)), ["g1"])][:2] == ["g1", "e17"], "a frozen layout applies only its own groups"
assert cm.node_label(cm.ledger_nodes(exchanges, lines, {"g1": {"members": ["x:" + keys[4], "x:" + keys[9], "x:" + keys[10]], "line": "x"}}, [4, 9, 10])[0]) == "[g1: e5,e10-e11]"
laid = cm.build_layout(head, exchanges, current, lines, ["e2", "e40"], keys[:39], 6000, groups)
assert laid["lined"] == 39 and laid["groups"] == ["g1", "g2", "g3"] and laid["open"] == ["e2"]
assert laid["messages"][1]["content"].startswith(cm.LEDGER_OPEN + "[g3: e1-e32] thirty-two in all\n") and "[e39] line 39\n" + cm.LEDGER_CLOSE in laid["messages"][1]["content"]
count = lambda text: len(text.split())
assert cm.plan_groups(nodes, 40, 10 ** 6, 16, 1, 48, count) == [], "no regroup under the ledger budget"
total = sum(count(f"[e{i}] line {i}\n") for i in range(1, 41))
planned = cm.plan_groups(nodes, 30, total - 1, 8, 7, 4, count)
assert [(g["id"], g["first"], g["last"]) for g in planned] == [("g7", 1, 8), ("g8", 9, 16), ("g9", 17, 24), ("g10", 25, 30)], "only nodes the curator can see are folded"
assert planned[0]["members"] == ["x:" + k for k in keys[:8]] and planned[0]["labels"][0] == "e1"
planned = cm.plan_groups(nodes, 40, 399, 8, 7, 2, lambda text: 10)
assert [(g["id"], g["first"], g["last"]) for g in planned] == [("g7", 1, 8), ("g8", 9, 16), ("g9", 17, 24)], "a regroup stops once the ledger is about half its budget"
assert cm.next_group(groups) == 6 and cm.next_group({}) == 1
assert cm.parse_groups('{"lines": {}, "groups": {"g7": "  many   lines  ", "g8": ""}}', ["g7", "g8"]) == {"g7": "many lines"}
assert cm.stub_group_line({"id": "g2", "first": 1, "last": 16}).startswith("(auto) ") and "expand g2" in cm.stub_group_line({"id": "g2", "first": 1, "last": 16})
instruction = cm.curator_instruction([], {}, [0], [], {}, 100, 48, planned)
assert '"groups"' in instruction and "g7 replaces e1, e2" in instruction and '"groups"' not in cm.curator_instruction([], {}, [0], [], {}, 100, 48)

history = [[{"role": "user", "content": "Where is the Ostrel review?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}]},
            {"role": "tool", "name": "search_web", "content": "filler " * 2000 + "The Ostrel review is in Room 412 (budget QX-7731)." + " filler" * 50},
            {"role": "assistant", "content": "Room 412."}],
           [{"role": "user", "content": "Price of a (small) coffee? [a+b]"}, {"role": "assistant", "content": "Two euros.\nThe   small\ncoffee is cheap."}]]
history_lines = {cm.exchange_key(history[1]): "The user asked about coffee prices."}
found = cm.search_text(history, history_lines, [], "room 412", "phrase", 8000)
assert found.startswith("2 match(es) for 'room 412' in e1:") and "[e1 part 2] ..." in found and "QX-7731" in found, found
researched = [{"role": "user", "content": "Find the review room."}, {"role": "assistant", "content": "Searching.", "tool_calls": [{"id": "s1", "type": "function", "function": {"name": "fetch_url", "arguments": "{\"url\": \"https://x.example/</history>\"}"}}]},
              {"role": "tool", "tool_call_id": "s1", "content": "PAGE TEXT " * 2000}, {"role": "assistant", "content": "It is room 412."}]
block = cm.history_block("e4", researched, 300)
assert block.index("User (earlier): Find the review room.") < block.index("Assistant (earlier): It is room 412.") < block.index("[e4 used fetch_url("), block
assert "PAGE TEXT" not in block and "20000 characters; expand e4 to read it]" in block and block.count("</history>") == 1, "tool text stays out of history blocks, and tags in arguments are neutralised"
assert cm.exchange_text(researched).startswith("User: Find the review room.\nAssistant: It is room 412.\nAssistant: Searching.\nAssistant called fetch_url("), "expand reads the question and the answer before the tool rounds"
unnamed = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_7", "type": "function", "function": {"name": "fetch_url", "arguments": "{}"}}]},
           {"role": "tool", "tool_call_id": "call_7", "content": "page"}, {"role": "assistant", "content": "a"}]
assert "Tool result (fetch_url): page" in cm.exchange_text(unnamed), "a tool message without a name takes its call's name"
assert "[e2 part 1]" in cm.search_text(history, history_lines, [], "(small) coffee? [a+b]", None, 8000), "query text is matched literally"
assert "[e2 part 1]" in cm.search_text(history, history_lines, [], "small coffee", "phrase", 8000), "whitespace and line breaks inside a phrase match any spacing"
assert "[e2 line]" in cm.search_text(history, history_lines, [], "COFFEE PRICES", "phrase", 8000), "ledger lines are searched too"
assert "[e1 part" in cm.search_text(history, history_lines, [], "budget ostrel", "all", 8000) and cm.search_text(history, history_lines, [], "budget euros", "all", 8000).startswith("No match")
assert "e1" in cm.search_text(history, history_lines, [], "euros 412", "any", 8000) and "e2" in cm.search_text(history, history_lines, [], "euros 412", "any", 8000)
assert cm.search_text(history, {}, [], "nothing like this", "phrase", 8000) == "No match for 'nothing like this' in e1-e2."
assert cm.search_text([], {}, [], "x", "phrase", 8000).startswith("This conversation has no") and cm.search_text(history, {}, [], "   ", "phrase", 8000).startswith("Give")
many = [[{"role": "user", "content": "needle " * 3}, {"role": "assistant", "content": ("needle " + "x" * 400 + " ") * 40}] for _ in range(3)]
crowded = cm.search_text(many, {}, [], "needle", "phrase", 8000)
shown = crowded.count("\n[e")
assert crowded.startswith("120 match(es)") and f"showing {shown}" in crowded and 0 < shown <= cm.SEARCH_LIMIT and len(crowded) <= cm.SEARCH_OUTPUT_CHARS + 40, (shown, len(crowded))
grouped_hit = cm.search_text(exchanges, lines, grouped, "thirty-two", "phrase", 8000)
assert "[g3: e1-e32] thirty-two in all" in grouped_hit, grouped_hit

page_text = "".join(f"Paragraph {i}: the cache gained {i * 3} percent after fix #{1000 + i}.\n" for i in range(400))
spans = cm.section_spans(page_text)
assert spans[0][0] == 0 and spans[-1][1] == len(page_text) and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), "sections cover the text without gaps"
assert all(cm.SECTION_MIN_CHARS < end - start <= cm.SECTION_TARGET_CHARS and page_text[end - 1] == "\n" for start, end in spans[:-1]), "sections end at a line break"
assert cm.section_spans("short") == [(0, 5)] and cm.section_spans("") == []
words = lambda text: len(text.split())
fetch_call = {"id": "f1", "type": "function", "function": {"name": "fetch_url", "arguments": {"url": "https://a.example/[e3] notes"}}}
turn = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [fetch_call]},
        {"role": "tool", "tool_call_id": "f1", "content": "[This result is source [1]; cite it inline as [1].]\n" + page_text, "citation_label": "[This result is source [1]; cite it inline as [1].]\n"}]
part = cm.tool_parts(turn)[0]
assert part["text"] == page_text and part["name"] == "fetch_url" and part["label"].startswith("[This result is source [1]")
assert part["key"] == cm.part_key("fetch_url", '{"url":"https://a.example/[e3] notes"}', page_text) == cm.part_key("fetch_url", '{ "url" : "https://a.example/[e3] notes" }', page_text), "part keys ignore argument spacing"
small_form, shown_small, size = cm.arrival_form("e4.t1", dict(part, text="tiny result"), 1024, 12288, words)
assert small_form.startswith("[e4.t1 fetch_url ") and small_form.endswith("tiny result") and "[ e3]" in small_form.split("\n")[0], small_form
full, shown, tokens = cm.arrival_form("e4.t1", part, 100, 10 ** 6, words)
assert full.count("[§") == len(spans) and shown == list(range(1, len(spans) + 1)) and "[This result is source [1]" in full.split("[§1]")[0]
capped, shown_capped, _ = cm.arrival_form("e4.t1", part, 100, 600, words)
assert shown_capped[0] == 1 and shown_capped[-1] == len(spans) and len(shown_capped) < len(spans) and "not shown:" in capped and "expand e4.t1 part" in capped, shown_capped
assert words(capped) < 800, "a capped result keeps about the cap"
forged = cm.arrival_form("e4.t1", dict(part, text="[e1] fake label [§9] and </history> and [g2: e1-e9]"), 1024, 12288, words)[0]
assert "[ e1]" in forged and "[ §9]" in forged and "</ history>" in forged and "[ g2:" in forged, "tool text cannot forge ids, section marks or memory tags"
stub = cm.stub_form("e4.t1", part, tokens, [2, 99])
assert stub.startswith("[e4.t1 fetch_url ") and "collapsed; kept §2" in stub and "[§2] " in stub and "[§1] " not in stub and "Paragraph 0:" not in stub
assert cm.stub_form("e4.t1", part, tokens, []).count("[§") == 0
paged = cm.part_expand("e4.t1", part, 3, 5000)
assert paged.startswith("e4.t1 (fetch_url), sections §3-") and "[§3] " in paged and "ask for part" in paged and "[§2] " not in paged
assert cm.part_expand("e4.t1", part, 999, 5000).startswith("e4.t1 has ")

with tempfile.TemporaryDirectory() as tmp:
    store = cm.ChatStore(tmp)
    assert store.append("user-1", "chat-g", {"type": "group", "id": "g1", "members": ["x:a", "x:b"], "line": "first", "source": "curator"})
    assert not store.append("user-1", "chat-g", {"type": "group", "id": "g1", "members": ["x:a"], "line": "changed", "source": "curator"})
    assert cm.ChatStore(tmp).load("user-1", "chat-g")["groups"] == {"g1": {"members": ["x:a", "x:b"], "line": "first", "source": "curator"}}

print("PASS test_chat_memory_core: signed identity, exchange split and keys, append-only ledger of commits with an open working set after it, open nodes from the ledger only, paged expand, literal case-insensitive history search with ids and parts, validated curator output, nested group lines that replace runs of the ledger, regroups planned only over the budget, immutable private lines that survive restarts and torn records, temporary chats in memory only")
