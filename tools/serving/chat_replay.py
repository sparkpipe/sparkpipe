import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import time
from array import array
from pathlib import Path

DEFAULT_BUDGET_BYTES = 4 << 30
VERSION = "sparkpipe-replay-3"
SIGNATURE_PREFIX = "sp1."
HANDLE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
SCOPE = re.compile(r"(own|key:[A-Za-z0-9_-]{1,64}|account:[A-Za-z0-9][A-Za-z0-9_-]{0,63})")
SCOPE_FILE = re.compile(r"(own|key-[A-Za-z0-9_-]{1,64}|account-[A-Za-z0-9][A-Za-z0-9_-]{0,63})\.sqlite")
PREFILL_FILE = "prefill.sqlite"
SIDE_FILES = ("-wal", "-shm", "-journal")
EVICT_BATCH = 64
TOUCH_SECONDS = 60.0
PREFILL_BUCKETS_PER_OCTAVE = 4
PREFILL_NEAREST_BUCKETS = 2
PREFILL_COLD_SHARE = 0.9
PREFILL_MIN_TOKENS = 512
PREFILL_WEIGHT = 0.25
STORE_ERRORS = (sqlite3.Error, OSError)

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS turns (id TEXT PRIMARY KEY, model TEXT NOT NULL, member TEXT NOT NULL, context BLOB NOT NULL, view BLOB NOT NULL, "
    "tokens BLOB NOT NULL, bytes INTEGER NOT NULL, used REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS turns_used ON turns (used)",
    "CREATE INDEX IF NOT EXISTS turns_view ON turns (model, member, context, view)",
    "CREATE TABLE IF NOT EXISTS handles (handle TEXT PRIMARY KEY, turn TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS handles_turn ON handles (turn)",
    "CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, parent TEXT, model TEXT NOT NULL, member TEXT NOT NULL, messages TEXT NOT NULL, "
    "bytes INTEGER NOT NULL, used REAL NOT NULL, body TEXT)",
    "CREATE INDEX IF NOT EXISTS responses_used ON responses (used)",
    "CREATE TABLE IF NOT EXISTS aliases (one INTEGER PRIMARY KEY CHECK (one = 1), aliases TEXT NOT NULL, bytes INTEGER NOT NULL, used REAL NOT NULL)",
)
PREFILL_SCHEMA = ("CREATE TABLE IF NOT EXISTS prefill (model TEXT NOT NULL, bucket INTEGER NOT NULL, ms_per_token REAL NOT NULL, "
                  "samples INTEGER NOT NULL, PRIMARY KEY (model, bucket))",)
TABLES = ("turns", "responses", "aliases")
ADDED_COLUMNS = (("responses", "body", "TEXT"),)


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
    if message.get("role") == "assistant":
        return ["assistant", assistant_text(message.get("content")), [call_view(call) for call in message.get("tool_calls") or []]]
    return [message.get("role"), message.get("content") if message.get("content") is not None else "", message.get("tool_call_id"), message.get("name")]


def view_digest(model_id, message):
    view = [VERSION, model_id, assistant_text(message.get("content")), [call_view(call) for call in message.get("tool_calls") or []]]
    return hashlib.sha256(canonical(view).encode()).digest()


def context_digests(model_id, tools, kwargs, messages):
    digest = hashlib.sha256(canonical([VERSION, model_id, tools or [], kwargs or {}]).encode()).digest()
    digests = [digest]
    for message in messages:
        digest = hashlib.sha256(digest + hashlib.sha256(canonical(message_view(message)).encode()).digest()).digest()
        digests.append(digest)
    return digests


def member_key(member):
    return member if isinstance(member, str) else ""


def new_turn():
    return secrets.token_hex(12)


def turn_signature(turn, index):
    return f"{SIGNATURE_PREFIX}{turn}.{index}"


def valid_handle(value):
    return isinstance(value, str) and HANDLE.fullmatch(value) is not None


def splice(context, after, stops):
    head = list(context)
    while head and head[-1] in stops:
        head.pop()
    overlap = next((size for size in range(min(len(head), len(after)), 0, -1) if head[-size:] == after[:size]), 0)
    return head, list(after[overlap:])


def bucket_of(tokens):
    return int(math.floor(math.log2(max(tokens, 1)) * PREFILL_BUCKETS_PER_OCTAVE))


def scope_file(scope):
    if not isinstance(scope, str) or SCOPE.fullmatch(scope) is None:
        raise ValueError(f"{scope!r} is not a replay scope")
    return scope.replace(":", "-", 1) + ".sqlite"


def file_scope(name):
    if SCOPE_FILE.fullmatch(name) is None:
        return None
    stem = name[:-len(".sqlite")]
    return stem if stem == "own" else stem.replace("-", ":", 1)


def private_file(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.close(descriptor)
    os.chmod(path, 0o600)


def private_sides(path):
    for suffix in SIDE_FILES:
        side = Path(str(path) + suffix)
        if side.exists():
            os.chmod(side, 0o600)


def unlink_all(path):
    for target in [Path(path)] + [Path(str(path) + suffix) for suffix in SIDE_FILES]:
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def connect(path, schema):
    if path is not None:
        private_file(path)
    db = sqlite3.connect(str(path) if path is not None else ":memory:", isolation_level=None, check_same_thread=False)
    db.execute("PRAGMA secure_delete=ON")
    if path is not None:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
    for statement in schema:
        db.execute(statement)
    if path is not None:
        private_sides(path)
    return db


def widen(db):
    for table, column, kind in ADDED_COLUMNS:
        if column not in {row[1] for row in db.execute(f"PRAGMA table_info({table})")}:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    return db


class ScopeStore:
    def __init__(self, path):
        self.path = path
        self.db = widen(connect(path, SCHEMA))
        self.bytes = sum(self.db.execute(f"SELECT COALESCE(SUM(bytes), 0) FROM {table}").fetchone()[0] for table in TABLES)

    def oldest(self):
        rows = [self.db.execute(f"SELECT MIN(used) FROM {table}").fetchone()[0] for table in ("turns", "responses")]
        rows = [row for row in rows if row is not None]
        return min(rows) if rows else None

    def empty(self):
        return all(self.db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is None for table in TABLES)

    def touch(self, table, column, value, now):
        try:
            self.db.execute(f"UPDATE {table} SET used = ? WHERE {column} = ? AND used < ?", (now, value, now - TOUCH_SECONDS))
        except sqlite3.Error:
            pass

    def find(self, model_id, member, handles, view, context, now):
        for handle in handles:
            row = self.db.execute("SELECT turns.id, turns.tokens FROM handles JOIN turns ON turns.id = handles.turn "
                                  "WHERE handles.handle = ? AND turns.model = ? AND turns.member = ? AND turns.view = ?",
                                  (handle, model_id, member, view)).fetchone()
            if row is not None:
                self.touch("turns", "id", row[0], now)
                return row[1]
        row = self.db.execute("SELECT id, tokens FROM turns WHERE model = ? AND member = ? AND context = ? AND view = ? ORDER BY used DESC LIMIT 1",
                              (model_id, member, context, view)).fetchone()
        if row is None:
            return None
        self.touch("turns", "id", row[0], now)
        return row[1]

    def put(self, turn, model_id, member, context, view, blob, handles, now):
        size = len(blob) + len(context) + sum(len(handle) for handle in handles)
        old = self.db.execute("SELECT bytes FROM turns WHERE id = ?", (turn,)).fetchone()
        self.db.execute("BEGIN")
        try:
            self.db.execute("INSERT OR REPLACE INTO turns (id, model, member, context, view, tokens, bytes, used) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (turn, model_id, member, context, view, blob, size, now))
            self.db.executemany("INSERT OR REPLACE INTO handles (handle, turn) VALUES (?, ?)", [(handle, turn) for handle in handles])
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.bytes += size - (old[0] if old else 0)

    def drop_turns(self, condition, parameters=()):
        rows = self.db.execute(f"SELECT id, bytes FROM turns WHERE {condition}", parameters).fetchall()
        for turn, size in rows:
            self.db.execute("DELETE FROM handles WHERE turn = ?", (turn,))
            self.db.execute("DELETE FROM turns WHERE id = ?", (turn,))
            self.bytes -= size
        return len(rows)

    def drop_rows(self, table, condition, parameters=()):
        size = self.db.execute(f"SELECT COALESCE(SUM(bytes), 0), COUNT(*) FROM {table} WHERE {condition}", parameters).fetchone()
        self.db.execute(f"DELETE FROM {table} WHERE {condition}", parameters)
        self.bytes -= size[0]
        return size[1]

    def evict_one(self):
        turn = self.db.execute("SELECT id, used FROM turns ORDER BY used LIMIT 1").fetchone()
        response = self.db.execute("SELECT id, used FROM responses ORDER BY used LIMIT 1").fetchone()
        if turn is None and response is None:
            return False
        if response is None or (turn is not None and turn[1] <= response[1]):
            self.drop_turns("id IN (SELECT id FROM turns ORDER BY used LIMIT ?)", (EVICT_BATCH,))
        else:
            self.drop_rows("responses", "id IN (SELECT id FROM responses ORDER BY used LIMIT ?)", (EVICT_BATCH,))
        return True

    def expire(self, rule):
        self.db.create_function("sparkpipe_expired", 1, rule, deterministic=True)
        dropped = self.drop_turns("sparkpipe_expired(used)")
        for table in ("responses", "aliases"):
            dropped += self.drop_rows(table, "sparkpipe_expired(used)")
        return dropped

    def checkpoint(self):
        if self.path is not None:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self):
        self.db.close()


class ReplayStore:
    def __init__(self, directory=None, budget_bytes=DEFAULT_BUDGET_BYTES):
        self.directory = Path(directory) if directory is not None else None
        self.budget = int(budget_bytes)
        self.scopes = {}
        if self.directory is not None:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.directory, 0o700)
            for path in sorted(self.directory.iterdir()):
                scope = file_scope(path.name)
                if scope is not None:
                    self.scopes[scope] = ScopeStore(path)
        self.prefill = connect(self.directory / PREFILL_FILE if self.directory is not None else None, PREFILL_SCHEMA)

    @property
    def bytes(self):
        return sum(store.bytes for store in self.scopes.values())

    def store(self, scope, create=False):
        store = self.scopes.get(scope)
        if store is None and create:
            name = scope_file(scope)
            store = self.scopes[scope] = ScopeStore(self.directory / name if self.directory is not None else None)
        return store

    def find(self, scope, member, model_id, handles, view, context):
        store = self.store(scope)
        if store is None:
            return None
        blob = store.find(model_id, member_key(member), [handle for handle in handles if valid_handle(handle)], view, context, time.time())
        if blob is None:
            return None
        tokens = array("I")
        tokens.frombytes(blob)
        return tokens

    def put(self, scope, member, model_id, view, context, tokens, handles, turn=None):
        turn = turn or new_turn()
        handles = sorted({handle for handle in handles if valid_handle(handle)})
        self.store(scope, True).put(turn, model_id, member_key(member), context, view, array("I", tokens).tobytes(), handles, time.time())
        self.evict()
        return turn

    def evict(self):
        while self.bytes > self.budget:
            candidates = [(oldest, scope) for oldest, scope in ((store.oldest(), scope) for scope, store in self.scopes.items()) if oldest is not None]
            if not candidates:
                return
            self.scopes[min(candidates)[1]].evict_one()

    def turns(self, scope=None):
        stores = self.scopes.values() if scope is None else [self.scopes[scope]] if scope in self.scopes else []
        return sum(store.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] for store in stores)

    def put_response(self, scope, member, response_id, parent, model_id, messages, body=None):
        text = json.dumps(messages, ensure_ascii=False)
        shown = json.dumps(body, ensure_ascii=False) if body is not None else None
        size = len(text.encode()) + len(response_id) + (len(shown.encode()) if shown is not None else 0)
        store = self.store(scope, True)
        old = store.db.execute("SELECT bytes FROM responses WHERE id = ?", (response_id,)).fetchone()
        store.db.execute("INSERT OR REPLACE INTO responses (id, parent, model, member, messages, bytes, used, body) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (response_id, parent, model_id, member_key(member), text, size, time.time(), shown))
        store.bytes += size - (old[0] if old else 0)
        self.evict()

    def response_body(self, scope, response_id):
        store = self.store(scope)
        if store is None:
            return None
        row = store.db.execute("SELECT body FROM responses WHERE id = ? AND body IS NOT NULL", (response_id,)).fetchone()
        if row is None:
            return None
        store.touch("responses", "id", response_id, time.time())
        return json.loads(row[0])

    def delete_response(self, scope, response_id):
        store = self.store(scope)
        return store is not None and store.drop_rows("responses", "id = ?", (response_id,)) > 0

    def response_messages(self, scope, member, response_id):
        store = self.store(scope)
        if store is None:
            return None
        chain, cursor, seen, now = [], response_id, set(), time.time()
        while cursor is not None:
            if cursor in seen:
                return None
            seen.add(cursor)
            row = store.db.execute("SELECT parent, messages FROM responses WHERE id = ? AND member = ?", (cursor, member_key(member))).fetchone()
            if row is None:
                return None
            store.touch("responses", "id", cursor, now)
            chain.append(json.loads(row[1]))
            cursor = row[0]
        return [message for messages in reversed(chain) for message in messages]

    def aliases(self, scope):
        store = self.store(scope)
        row = store.db.execute("SELECT aliases FROM aliases WHERE one = 1").fetchone() if store is not None else None
        if row is None:
            return None
        store.touch("aliases", "one", 1, time.time())
        return json.loads(row[0])

    def set_aliases(self, scope, table):
        text = json.dumps(table, ensure_ascii=False)
        store = self.store(scope, True)
        old = store.db.execute("SELECT bytes FROM aliases WHERE one = 1").fetchone()
        store.db.execute("INSERT OR REPLACE INTO aliases (one, aliases, bytes, used) VALUES (1, ?, ?, ?)", (text, len(text.encode()), time.time()))
        store.bytes += len(text.encode()) - (old[0] if old else 0)

    def held(self, scope):
        store = self.store(scope)
        items, size, oldest, newest = 0, 0, None, None
        for table in TABLES if store is not None else ():
            for used, bytes_ in store.db.execute(f"SELECT used, bytes FROM {table}"):
                items, size = items + 1, size + int(bytes_ or 0)
                oldest = float(used) if oldest is None else min(oldest, float(used))
                newest = float(used) if newest is None else max(newest, float(used))
        return {"items": items, "bytes": size, "oldest_used": oldest, "newest_used": newest}

    def forget(self, scope):
        store = self.scopes.pop(scope, None)
        if store is None:
            return False
        store.close()
        if store.path is not None:
            unlink_all(store.path)
        return True

    def clear(self, scope=None):
        for name in [scope] if scope is not None else list(self.scopes):
            self.forget(name)

    def expire(self, rule):
        total = 0
        for scope, store in list(self.scopes.items()):
            dropped = store.expire(rule)
            total += dropped
            if store.empty():
                self.forget(scope)
            elif dropped:
                store.checkpoint()
        return total

    def observe_prefill(self, model_id, prompt_tokens, prefill_tokens, prefill_ms):
        if prompt_tokens < PREFILL_MIN_TOKENS or prefill_tokens < PREFILL_COLD_SHARE * prompt_tokens or prefill_ms <= 0:
            return
        bucket = bucket_of(prompt_tokens)
        rate = prefill_ms / prefill_tokens
        row = self.prefill.execute("SELECT ms_per_token, samples FROM prefill WHERE model = ? AND bucket = ?", (model_id, bucket)).fetchone()
        if row is not None:
            rate = row[0] + PREFILL_WEIGHT * (rate - row[0])
        self.prefill.execute("INSERT OR REPLACE INTO prefill (model, bucket, ms_per_token, samples) VALUES (?, ?, ?, ?)",
                             (model_id, bucket, rate, (row[1] if row else 0) + 1))

    def prefill_rate(self, model_id, prompt_tokens):
        bucket = bucket_of(prompt_tokens)
        row = self.prefill.execute("SELECT ms_per_token FROM prefill WHERE model = ? AND bucket BETWEEN ? AND ? ORDER BY ABS(bucket - ?) LIMIT 1",
                                   (model_id, bucket - PREFILL_NEAREST_BUCKETS, bucket + PREFILL_NEAREST_BUCKETS, bucket)).fetchone()
        return row[0] if row else None

    def close(self):
        for store in self.scopes.values():
            store.close()
        self.prefill.close()
