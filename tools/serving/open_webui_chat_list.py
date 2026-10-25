import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path


def store_name(text):
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def chat_list(database):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        users = connection.execute("select count(*) from user").fetchone()[0]
        chats = sorted(store_name(user) + "/" + store_name(chat) for user, chat in connection.execute("select user_id, id from chat"))
    finally:
        connection.close()
    return {"at": time.time(), "users": users, "chats": chats}


def write(listing, output):
    output = Path(output)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=".open-webui-chats-")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(listing, handle)
    os.chmod(temporary, 0o640)
    if os.geteuid() == 0:
        os.chown(temporary, 0, output.parent.stat().st_gid)
    os.replace(temporary, output)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: open_webui_chat_list.py WEBUI_DB OUTPUT_JSON")
    write(chat_list(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
