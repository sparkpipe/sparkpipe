#!/usr/bin/env python3
"""SparkPipe public API keys: create, revoke and list the keys the chat layer accepts on the public door.

The keys file holds only each key's SHA-256, an id, its creation and expiry
times and whether it is revoked. A new key is printed once and never stored.
"""
import argparse
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path

PREFIX = "sk-sp-"


def load(path):
    if not path.exists():
        return {"keys": []}
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
        raise SystemExit(f"api_keys: {path} is not a keys file")
    return document


def store(path, document):
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps(document, indent=1) + "\n")
    os.replace(temporary, path)


def create(path, days, note, gateway=False):
    document = load(path)
    key = PREFIX + secrets.token_urlsafe(32)
    identifier = "k_" + secrets.token_hex(4)
    while any(entry["id"] == identifier for entry in document["keys"]):
        identifier = "k_" + secrets.token_hex(4)
    now = int(time.time())
    document["keys"].append({"id": identifier, "sha256": hashlib.sha256(key.encode()).hexdigest(), "created": now,
                             "expires": now + days * 86400 if days else None, "revoked": False, "note": note, "gateway": gateway})
    store(path, document)
    return identifier, key


def revoke(path, identifier):
    document = load(path)
    for entry in document["keys"]:
        if entry["id"] == identifier:
            entry["revoked"] = True
            store(path, document)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("create", help="make a key and print it once")
    made.add_argument("--days", type=int, default=0, help="days until the key expires; 0 never expires")
    made.add_argument("--note", default="")
    made.add_argument("--gateway", action="store_true", help="the market gateway's key: its requests name their account in X-SparkPipe-Account")
    gone = commands.add_parser("revoke", help="refuse a key from now on")
    gone.add_argument("id")
    commands.add_parser("list", help="show every key's id, dates and state, never the key")
    arguments = parser.parse_args()
    if arguments.command == "create":
        identifier, key = create(arguments.file, arguments.days, arguments.note, arguments.gateway)
        print(f"{identifier} {key}")
        print("The key is shown only now; the file keeps its SHA-256.", file=sys.stderr)
    elif arguments.command == "revoke":
        if not revoke(arguments.file, arguments.id):
            raise SystemExit(f"api_keys: no key {arguments.id}")
        print(f"{arguments.id} revoked")
    else:
        for entry in load(arguments.file)["keys"]:
            expires = time.strftime("%Y-%m-%d", time.gmtime(entry["expires"])) if entry.get("expires") else "never"
            print(entry["id"], time.strftime("%Y-%m-%d", time.gmtime(entry["created"])), "expires", expires,
                  "revoked" if entry.get("revoked") else "active", *(["gateway"] if entry.get("gateway") is True else []), entry.get("note", ""))


if __name__ == "__main__":
    main()
