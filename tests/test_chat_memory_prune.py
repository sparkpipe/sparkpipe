#!/usr/bin/env python3
"""A chat's memory is deleted with the chat. A root job exports the Open WebUI chat list as hashed (user, chat) names, read from the database opened read-only (also while Open WebUI holds it in WAL mode), and writes it atomically with mode 640. Pruning against that list removes the memory file of every chat no longer listed, including every chat of a deleted user, and drops that chat's state from RAM; it keeps the memory of listed chats, files written within the grace period (a chat whose row is not stored yet), and temporary chats, which live only in RAM; it deletes nothing when the list is missing, unreadable, stale or names no users. The exporter and the store hash names the same way."""
import asyncio
import importlib.util
import json
import os
import sqlite3
import stat
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cm = load("chat_memory", ROOT / "tools" / "serving" / "chat_memory.py")
exporter = load("open_webui_chat_list", ROOT / "tools" / "serving" / "open_webui_chat_list.py")
assert exporter.store_name("user-1") == cm.store_name("user-1") and cm.store_name("x") != cm.store_name("y")


def webui(path, users, chats):
    connection = sqlite3.connect(path)
    connection.execute("pragma journal_mode=wal")
    connection.execute("create table user (id text primary key)")
    connection.execute("create table chat (id text primary key, user_id text)")
    connection.executemany("insert into user values (?)", [(u,) for u in users])
    connection.executemany("insert into chat values (?, ?)", [(c, u) for u, c in chats])
    connection.commit()
    return connection


def memory(listing):
    return types.SimpleNamespace(chat_list=str(listing), store=store, prune_grace_seconds=3600, chat_list_max_age_seconds=1800)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "memory"
    store = cm.ChatStore(root)
    for user, chat in (("u1", "c-kept"), ("u1", "c-deleted"), ("u2", "c-gone-user"), ("u1", "c-new")):
        store.append(user, chat, {"type": "line", "key": "k1", "line": f"{chat} line", "source": "curator"})
    store.append("u1", "local:temp", {"type": "line", "key": "k1", "line": "temporary", "source": "curator"})
    old = time.time() - 7200
    for user, chat in (("u1", "c-kept"), ("u1", "c-deleted"), ("u2", "c-gone-user")):
        os.utime(store.path(user, chat), (old, old))
    assert not store.path("u1", "local:temp").exists(), "temporary chats never reach disk"

    database = Path(tmp) / "webui.db"
    holder = webui(database, ["u1"], [("u1", "c-kept")])
    before = database.stat().st_mtime_ns
    listing = exporter.chat_list(database)
    assert listing["users"] == 1 and listing["chats"] == [cm.store_name("u1") + "/" + cm.store_name("c-kept")], listing
    assert database.stat().st_mtime_ns == before, "the database is only read"
    try:
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True).execute("insert into user values ('x')")
        raise AssertionError("a read-only connection must not write")
    except sqlite3.OperationalError:
        pass
    output = Path(tmp) / "open-webui-chats.json"
    exporter.write(listing, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o640 and json.loads(output.read_text())["chats"] == listing["chats"]
    assert not [p for p in Path(tmp).glob(".open-webui-chats-*")], "no temporary file is left behind"

    assert asyncio.run(cm.Memory.prune_once(memory(Path(tmp) / "missing.json"))) is None, "a missing list deletes nothing"
    empty = Path(tmp) / "empty.json"
    exporter.write(dict(listing, users=0, chats=[]), empty)
    assert asyncio.run(cm.Memory.prune_once(memory(empty))) is None, "a list with no users deletes nothing"
    stale = Path(tmp) / "stale.json"
    exporter.write(dict(listing, at=time.time() - 7200), stale)
    assert asyncio.run(cm.Memory.prune_once(memory(stale))) is None, "a stale list deletes nothing"
    assert all(store.path(u, c).exists() for u, c in (("u1", "c-kept"), ("u1", "c-deleted"), ("u2", "c-gone-user"), ("u1", "c-new")))

    assert asyncio.run(cm.Memory.prune_once(memory(output))) == 2
    assert store.path("u1", "c-kept").exists(), "a listed chat keeps its memory"
    assert not store.path("u1", "c-deleted").exists(), "a deleted chat loses its memory"
    assert not store.path("u2", "c-gone-user").exists() and not store.path("u2", "c-gone-user").parent.exists(), "a deleted user's chats and folder go"
    assert store.path("u1", "c-new").exists(), "a file inside the grace period stays until its chat is listed"
    assert ("u1", "c-deleted") not in store.memory and ("u2", "c-gone-user") not in store.memory, "pruned chats leave RAM"
    assert ("u1", "c-kept") in store.memory and ("u1", "local:temp") in store.memory, "listed and temporary chats stay in RAM"
    assert cm.ChatStore(root).load("u1", "c-kept")["lines"] == {"k1": "c-kept line"}

    later = time.time() + 7200
    kept = cm.read_chat_list(output)[2]
    assert store.prune(kept, 3600, now=later) == 1 and not store.path("u1", "c-new").exists(), "past the grace period an unlisted chat's file goes"
    holder.close()

print("PASS test_chat_memory_prune: the root exporter reads Open WebUI read-only and writes hashed chat names atomically; deleted chats and users lose their memory on disk and in RAM; listed, fresh and temporary chats keep theirs; a missing, empty or stale list deletes nothing")
