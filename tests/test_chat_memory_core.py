#!/usr/bin/env python3
"""Chat memory's core keeps a conversation's layout cache-friendly and its records safe: a signed identity is accepted only with a valid HS256 signature and an unexpired claim; a chat splits into finished exchanges and the current one; an exchange's key ignores ids and timestamps but not content; the ledger covers only a gap-free prefix of finished exchanges with lines and always leaves the newest finished exchange raw; adding a line only appends to the ledger text; the system note is present from the first turn; open nodes come only from the ledger; expand pages long exchanges; curator output is validated line by line with long lines cut and the open set trimmed to its budget; lines are immutable once stored, survive a restart and a torn last record, stay private on disk, and temporary chats never reach disk."""
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
assert cm.split([{"role": "assistant", "content": "hi"}]) is None

a = [{"role": "user", "content": "q", "id": "1"}, {"role": "assistant", "content": "a", "timestamp": 5}]
b = [{"role": "user", "content": "q", "id": "2"}, {"role": "assistant", "content": "a", "timestamp": 9}]
c = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a!"}]
assert cm.exchange_key(a) == cm.exchange_key(b) != cm.exchange_key(c)

head, exchanges, current = cm.split(chat(6))
keys = [cm.exchange_key(e) for e in exchanges]
none = cm.build_layout(head, exchanges, current, {}, [])
assert none["lined"] == 0 and cm.SYSTEM_NOTE in none["messages"][0]["content"] and not any(cm.LEDGER_OPEN in str(m["content"]) for m in none["messages"] if m["role"] == "user")
gap = cm.build_layout(head, exchanges, current, {keys[0]: "l1", keys[2]: "l3"}, [])
assert gap["lined"] == 1, gap["lined"]
full = cm.build_layout(head, exchanges, current, {k: f"line {i + 1}" for i, k in enumerate(keys)}, ["e1", "e4", "e9"])
assert full["lined"] == 4, "the newest finished exchange must stay raw"
first_user = [m for m in full["messages"] if m["role"] == "user"][0]["content"]
assert first_user.startswith(cm.LEDGER_OPEN + "[e1] line 1\n") and "[e4] line 4\n" in first_user and "[e5]" not in first_user and first_user.endswith("question 5")
last = full["messages"][-1]["content"]
assert last.startswith(cm.REFERENCE_OPEN) and '<history id="e1">' in last and 'id="e4"' in last and 'id="e9"' not in last and last.endswith("question 6")
assert full["open"] == ["e1", "e4"]
shorter = cm.build_layout(head, exchanges, current, {k: f"line {i + 1}" for i, k in enumerate(keys[:3])}, [])
older = [m for m in shorter["messages"] if m["role"] == "user"][0]["content"]
assert first_user.startswith(older[:older.index(cm.LEDGER_CLOSE)]), "adding a line must only append to the ledger text"
assert cm.build_layout(head, exchanges, current, {k: "x" for k in keys}, [], raw_min=2)["lined"] == 3
follow = cm.split(chat(4, follow_up=True))
laid = cm.build_layout(*follow, {cm.exchange_key(e): "x" for e in follow[1]}, [])
assert laid["messages"][-1]["role"] == "tool" and laid["messages"][-3]["content"].endswith("question 4")

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

print("PASS test_chat_memory_core: signed identity, exchange split and keys, gap-free append-only ledger with a raw newest exchange, open nodes from the ledger only, paged expand, validated curator output, immutable private lines that survive restarts and torn records, temporary chats in memory only")
