import hashlib
import json
import math
import sqlite3
import time
from array import array
from collections import OrderedDict
from pathlib import Path

CONTEXT_CACHE_BYTES = 256 << 20
DEFAULT_BUDGET_BYTES = 4 << 30
VERSION = "sparkpipe-replay-1"
PREFILL_BUCKETS_PER_OCTAVE = 4
PREFILL_NEAREST_BUCKETS = 2
PREFILL_COLD_SHARE = 0.9
PREFILL_MIN_TOKENS = 512
PREFILL_WEIGHT = 0.25

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS turns (key TEXT PRIMARY KEY, scope TEXT NOT NULL, model TEXT NOT NULL, parent TEXT, keep INTEGER NOT NULL, "
    "tokens BLOB NOT NULL, length INTEGER NOT NULL, used REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS turns_used ON turns (used)",
    "CREATE INDEX IF NOT EXISTS turns_scope ON turns (scope)",
    "CREATE TABLE IF NOT EXISTS responses (scope TEXT NOT NULL, id TEXT NOT NULL, parent TEXT, model TEXT NOT NULL, messages TEXT NOT NULL, "
    "created REAL NOT NULL, PRIMARY KEY (scope, id))",
    "CREATE TABLE IF NOT EXISTS aliases (scope TEXT PRIMARY KEY, aliases TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS prefill (model TEXT NOT NULL, bucket INTEGER NOT NULL, ms_per_token REAL NOT NULL, samples INTEGER NOT NULL, "
    "PRIMARY KEY (model, bucket))",
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def call_view(call):
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return [call.get("id"), function.get("name"), arguments]


def assistant_text(content):
    if content is None:
        return ""
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" for part in content):
        content = "".join(str(part.get("text") or "") for part in content)
    return content.strip() if isinstance(content, str) else canonical(content)


def message_view(message):
    role = message.get("role")
    content = message.get("content")
    if role == "assistant":
        return {"role": role, "content": assistant_text(content), "tool_calls": [call_view(call) for call in message.get("tool_calls") or []]}
    view = {"role": role, "content": content if content is not None else ""}
    for key in ("tool_call_id", "name"):
        if message.get(key) is not None:
            view[key] = message[key]
    return view


class Chain:
    def __init__(self, model_id, tools, kwargs):
        self.base = hashlib.sha256(canonical([VERSION, model_id, tools or [], kwargs or {}]).encode()).digest()

    @staticmethod
    def step(digest, message):
        return hashlib.sha256(digest + hashlib.sha256(canonical(message_view(message)).encode()).digest()).digest()

    def digests(self, messages):
        digest, result = self.base, []
        for message in messages:
            digest = self.step(digest, message)
            result.append(digest)
        return result


def splice(context, after, stops):
    head = list(context)
    while head and head[-1] in stops:
        head.pop()
    overlap = next((size for size in range(min(len(head), len(after)), 0, -1) if head[-size:] == after[:size]), 0)
    return head, list(after[overlap:])


def bucket_of(tokens):
    return int(math.floor(math.log2(max(tokens, 1)) * PREFILL_BUCKETS_PER_OCTAVE))


class ReplayStore:
    def __init__(self, path=None, budget_bytes=DEFAULT_BUDGET_BYTES, context_cache_bytes=CONTEXT_CACHE_BYTES):
        self.path = path
        if path:
            Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path) if path else ":memory:", isolation_level=None, check_same_thread=False)
        if path:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
        for statement in SCHEMA:
            self.db.execute(statement)
        self.budget = int(budget_bytes)
        self.bytes = self.db.execute("SELECT COALESCE(SUM(LENGTH(tokens)), 0) FROM turns").fetchone()[0]
        self.contexts = OrderedDict()
        self.context_bytes = 0
        self.context_budget = int(context_cache_bytes)

    def key(self, scope, digest):
        return hashlib.sha256(scope.encode() + b"\0" + digest).hexdigest()

    def find(self, scope, digest):
        key = self.key(scope, digest)
        if key in self.contexts:
            return key
        row = self.db.execute("SELECT 1 FROM turns WHERE key = ? AND scope = ?", (key, scope)).fetchone()
        return key if row else None

    def remember(self, key, tokens):
        old = self.contexts.pop(key, None)
        if old is not None:
            self.context_bytes -= old.itemsize * len(old)
        size = tokens.itemsize * len(tokens)
        if size > self.context_budget:
            return
        self.contexts[key] = tokens
        self.context_bytes += size
        while self.context_bytes > self.context_budget:
            _, dropped = self.contexts.popitem(last=False)
            self.context_bytes -= dropped.itemsize * len(dropped)

    def context(self, key):
        cached = self.contexts.get(key)
        if cached is not None:
            self.contexts.move_to_end(key)
            return cached
        rows, cursor, base = [], key, None
        while cursor is not None:
            if cursor in self.contexts:
                base = array("I", self.contexts[cursor])
                break
            row = self.db.execute("SELECT parent, keep, tokens FROM turns WHERE key = ?", (cursor,)).fetchone()
            if row is None:
                return None
            rows.append(row)
            cursor = row[0]
        tokens = base if base is not None else array("I")
        for _, keep, blob in reversed(rows):
            if keep > len(tokens):
                return None
            del tokens[keep:]
            piece = array("I")
            piece.frombytes(blob)
            tokens.extend(piece)
        self.remember(key, tokens)
        return tokens

    def put(self, scope, digest, model_id, parent, keep, tokens):
        key = self.key(scope, digest)
        blob = array("I", tokens).tobytes()
        now = time.time()
        old = self.db.execute("SELECT LENGTH(tokens) FROM turns WHERE key = ?", (key,)).fetchone()
        self.db.execute("INSERT OR REPLACE INTO turns (key, scope, model, parent, keep, tokens, length, used) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (key, scope, model_id, parent, int(keep), blob, len(tokens) + int(keep), now))
        self.bytes += len(blob) - (old[0] if old else 0)
        if parent is not None:
            self.db.execute("WITH RECURSIVE chain(k) AS (SELECT ? UNION SELECT turns.parent FROM turns JOIN chain ON turns.key = chain.k "
                            "WHERE turns.parent IS NOT NULL) UPDATE turns SET used = ? WHERE key IN chain", (parent, now))
        self.contexts.pop(key, None)
        self.evict()
        return key

    def evict(self):
        while self.bytes > self.budget:
            rows = self.db.execute("SELECT key, LENGTH(tokens) FROM turns ORDER BY used LIMIT 64").fetchall()
            if not rows:
                self.bytes = 0
                return
            for key, size in rows:
                self.db.execute("DELETE FROM turns WHERE key = ?", (key,))
                dropped = self.contexts.pop(key, None)
                if dropped is not None:
                    self.context_bytes -= dropped.itemsize * len(dropped)
                self.bytes -= size
                if self.bytes <= self.budget:
                    return

    def turns(self, scope=None):
        if scope is None:
            return self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM turns WHERE scope = ?", (scope,)).fetchone()[0]

    def clear(self, scope=None):
        tables = ("turns", "responses", "aliases")
        for table in tables:
            if scope is None:
                self.db.execute(f"DELETE FROM {table}")
            else:
                self.db.execute(f"DELETE FROM {table} WHERE scope = ?", (scope,))
        self.contexts.clear()
        self.context_bytes = 0
        self.bytes = self.db.execute("SELECT COALESCE(SUM(LENGTH(tokens)), 0) FROM turns").fetchone()[0]

    def put_response(self, scope, response_id, parent, model_id, messages):
        self.db.execute("INSERT OR REPLACE INTO responses (scope, id, parent, model, messages, created) VALUES (?, ?, ?, ?, ?, ?)",
                        (scope, response_id, parent, model_id, json.dumps(messages, ensure_ascii=False), time.time()))

    def response_messages(self, scope, response_id):
        chain, cursor = [], response_id
        while cursor is not None:
            row = self.db.execute("SELECT parent, messages FROM responses WHERE scope = ? AND id = ?", (scope, cursor)).fetchone()
            if row is None:
                return None
            chain.append(json.loads(row[1]))
            cursor = row[0]
        return [message for messages in reversed(chain) for message in messages]

    def aliases(self, scope):
        row = self.db.execute("SELECT aliases FROM aliases WHERE scope = ?", (scope,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_aliases(self, scope, table):
        self.db.execute("INSERT OR REPLACE INTO aliases (scope, aliases) VALUES (?, ?)", (scope, json.dumps(table, ensure_ascii=False)))

    def observe_prefill(self, model_id, prompt_tokens, prefill_tokens, prefill_ms):
        if prompt_tokens < PREFILL_MIN_TOKENS or prefill_tokens < PREFILL_COLD_SHARE * prompt_tokens or prefill_ms <= 0:
            return
        bucket = bucket_of(prompt_tokens)
        rate = prefill_ms / prefill_tokens
        row = self.db.execute("SELECT ms_per_token, samples FROM prefill WHERE model = ? AND bucket = ?", (model_id, bucket)).fetchone()
        if row is not None:
            rate = row[0] + PREFILL_WEIGHT * (rate - row[0])
        self.db.execute("INSERT OR REPLACE INTO prefill (model, bucket, ms_per_token, samples) VALUES (?, ?, ?, ?)",
                        (model_id, bucket, rate, (row[1] if row else 0) + 1))

    def prefill_rate(self, model_id, prompt_tokens):
        bucket = bucket_of(prompt_tokens)
        row = self.db.execute("SELECT ms_per_token FROM prefill WHERE model = ? AND bucket BETWEEN ? AND ? ORDER BY ABS(bucket - ?) LIMIT 1",
                              (model_id, bucket - PREFILL_NEAREST_BUCKETS, bucket + PREFILL_NEAREST_BUCKETS, bucket)).fetchone()
        return row[0] if row else None
