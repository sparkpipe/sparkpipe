#!/usr/bin/env python3
"""List, and on the owner's order delete, the files under each node's sparkdata that nothing uses.

A file under /home/<host>/sparkdata is used when one of these names it:

- a lane record: tools/lane_pack_files.py renders every record in
  deployment/lanes.json and gives, host by host, the rank pack, its
  .experts and .sha256, the model's pack siblings and any other sparkdata
  path a rendered config names (a partner pack, for instance);
- a companion of such a file: <file> plus .experts, .sha256, .verified,
  .receipt.json, .ck128 or .digest;
- a swap controller command in deployment/hub/model_swap.json;
- a live config on the node: deployment.json, config/*.json and lane.sh of
  every lane root, model_resident.json, config/*.json and env files of every
  fleet agent root; a named directory keeps everything under it, and a file
  only a live config names is reported, because its record does not name it;
- the keep list deployment/sparkdata_keep.json: packs staged but not served
  yet and the infrastructure directories, each with a reason and a date;
- a fleet agent root: a directory with a release MANIFEST, or a root named
  by ~/.fleet_agent_roots or the fleet-agent unit, which the release channel
  owns.

Everything else is unused. The node walks its sparkdata without following
a link (a link is listed as one), and reports a directory whose whole tree is
unused as one entry. The default is a dry run that prints every unused entry
and the bytes per host. --delete, which only the owner runs, deletes them on
the node, but refuses an entry a running process holds open (file
descriptors, mappings, working directories and executables in /proc, or
lsof where /proc is missing), an entry with anything modified in the last
hour, and every entry on a host that lacks a file a record names, because
its layout is then not the one the records describe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lane_config
import lane_pack_files
import stage_lane

ROOT = stage_lane.ROOT
KEEP = ROOT / "deployment" / "sparkdata_keep.json"
SWAP = ROOT / "deployment" / "hub" / "model_swap.json"
COMPANIONS = (".experts", ".sha256", ".verified", ".receipt.json", ".ck128", ".digest")
MINIMUM_AGE_SECONDS = 3600
SWAP_NAME = re.compile(r"\$HOME/(sparkdata/[^\s\"';|&)]+)")
NODE = r'''
import glob, json, os, re, shutil, stat, subprocess, sys, time
request = json.load(sys.stdin)
home = os.path.expanduser("~")
top = os.path.join(home, "sparkdata")
host = request["host"]
companions = tuple(request["companions"])
referenced = set(request["referenced"])
keep = {entry["path"]: entry["reason"] for entry in request["keep"]}
named = re.compile(r"(?:/home/" + re.escape(host) + r"/|\$HOME/|~/)(sparkdata/[^\s\"',;|&)]+)")

def read(path):
    try:
        with open(path, errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""

agents = set()
names = os.listdir(top) if os.path.isdir(top) else []
for name in names:
    if os.path.lexists(os.path.join(top, name, "MANIFEST")):
        agents.add("sparkdata/" + name)
for line in read(os.path.join(home, ".fleet_agent_roots")).splitlines():
    if line.strip() and "/" not in line.strip():
        agents.add("sparkdata/" + line.strip())
for line in read(os.path.join(home, ".config/systemd/user/fleet-agent.service")).splitlines():
    if line.startswith("ExecStart=") and len(line.split()) > 1:
        agents.update("sparkdata/" + name for name in line.split()[1].split(",") if name and "/" not in name)
configs = []
for root in glob.glob(home + "/stage-lanes/*/lane*/root") + glob.glob(home + "/k3-lanes/lane*/root"):
    configs += [os.path.join(root, "deployment.json"), os.path.join(root, "lane.sh")] + glob.glob(root + "/config/*.json")
for agent in agents:
    root = os.path.join(home, agent)
    configs += [os.path.join(root, "model_resident.json"), os.path.join(root, "agent.env")] + glob.glob(root + "/config/*.json") + glob.glob(root + "/config/env_*.env")
live = {os.path.normpath(match.group(1)) for path in configs for match in named.finditer(read(path))}

def ancestors(relative):
    parts = relative.split("/")
    return ["/".join(parts[:index]) for index in range(len(parts), 1, -1)]

def companion_of(relative, names):
    return any(relative.endswith(suffix) and relative[:-len(suffix)] in names for suffix in companions)

def role(relative):
    if relative in referenced:
        return "named"
    if companion_of(relative, referenced):
        return "companion"
    if any(path in live for path in ancestors(relative)) or companion_of(relative, live):
        return "live"
    for path in ancestors(relative):
        inner = path[len("sparkdata/"):]
        if inner in keep:
            return "keep:" + inner
        if path in agents:
            return "agent"
    return "unused"

roles, sizes, children, recent, links = {}, {}, {}, {}, set()
now = time.time()
for directory, dirnames, filenames in os.walk(top, topdown=False):
    relative_directory = os.path.relpath(directory, home)
    entries = []
    for name in dirnames + filenames:
        path = os.path.join(directory, name)
        relative = os.path.relpath(path, home)
        try:
            info = os.lstat(path)
        except OSError:
            continue
        if stat.S_ISDIR(info.st_mode):
            entries.append(relative)
            continue
        roles[relative] = role(relative)
        sizes[relative] = info.st_size if stat.S_ISREG(info.st_mode) else 0
        recent[relative] = info.st_mtime > now - request["minimum_age_seconds"]
        if stat.S_ISLNK(info.st_mode):
            links.add(relative)
        entries.append(relative)
    children[relative_directory] = entries
    if relative_directory == "sparkdata":
        continue
    own = role(relative_directory)
    unused = own == "unused" and all(roles.get(entry) == "unused" for entry in entries)
    roles[relative_directory] = "unused" if unused else (own if own != "unused" else "mixed")
    sizes[relative_directory] = sum(sizes.get(entry, 0) for entry in entries)
    recent[relative_directory] = os.lstat(directory).st_mtime > now - request["minimum_age_seconds"] or any(recent.get(entry) for entry in entries)

def parent(relative):
    return os.path.dirname(relative)

found = [relative for relative, value in roles.items() if value == "unused" and roles.get(parent(relative)) != "unused"]
kept = {}
for relative, value in roles.items():
    if relative in children or value in ("unused", "mixed"):
        continue
    kept[value.split(":")[0]] = kept.get(value.split(":")[0], 0) + sizes[relative]
missing = sorted(path for path in request["required"] if not os.path.lexists(os.path.join(home, path)))
linked = sorted(path for path in referenced if os.path.islink(os.path.join(home, path)))
live_only = sorted(path for path in live if path not in referenced and os.path.lexists(os.path.join(home, path)) and not path.startswith(tuple(a + "/" for a in agents)))
result = {"entries": [{"path": relative, "bytes": sizes[relative], "kind": "link" if relative in links else "directory" if relative in children else "file",
                       "recent": recent[relative]} for relative in sorted(found)],
          "kept": kept, "missing": missing, "linked": linked, "live": live_only, "agents": sorted(agents), "deleted": [], "refused": []}

def open_paths():
    paths = set()
    if os.path.isdir("/proc/self/fd"):
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            base = os.path.join("/proc", pid)
            try:
                descriptors = [os.path.join("fd", fd) for fd in os.listdir(base + "/fd")]
            except OSError:
                descriptors = []
            for link in ["cwd", "exe"] + descriptors:
                try:
                    paths.add(os.readlink(os.path.join(base, link)).replace(" (deleted)", ""))
                except OSError:
                    pass
            for line in read(base + "/maps").splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    paths.add(fields[5].replace(" (deleted)", ""))
        return paths
    if shutil.which("lsof"):
        listing = subprocess.run(["lsof", "-n", "-P", "-Fn"], capture_output=True, text=True).stdout
        return {line[1:] for line in listing.splitlines() if line.startswith("n/")}
    return None

def real(relative):
    path = os.path.join(home, relative)
    return os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))

if request["delete"]:
    if missing:
        result["refused"] = [{"path": entry["path"], "reason": "the node lacks %d file(s) a record names, first %s; its layout is not the one the records describe" % (len(missing), missing[0])} for entry in result["entries"]]
    else:
        for entry in result["entries"]:
            target = real(entry["path"])
            holders = open_paths()
            if holders is None:
                result["refused"].append({"path": entry["path"], "reason": "no /proc and no lsof: cannot see which files are open"})
            elif any(path == target or path.startswith(target + "/") for path in holders):
                result["refused"].append({"path": entry["path"], "reason": "a running process holds it open"})
            elif entry["recent"]:
                result["refused"].append({"path": entry["path"], "reason": "modified within the last %d s" % request["minimum_age_seconds"]})
            else:
                path = os.path.join(home, entry["path"])
                if entry["kind"] == "directory":
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
                result["deleted"].append(entry)
print(json.dumps(result))
'''


def keep_list(path: Path = KEEP) -> list[dict]:
    entries = json.loads(path.read_text())["keep"]
    for entry in entries:
        if set(entry) != {"path", "since", "reason"} or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["since"]) or not entry["reason"].strip() \
                or entry["path"].startswith("/") or ".." in entry["path"].split("/"):
            raise SystemExit(f"{path}: every keep entry has a sparkdata-relative path, a since date (YYYY-MM-DD) and a reason: {entry}")
    return entries


def swap_names() -> set[str]:
    names = set()
    for entry in json.loads(SWAP.read_text())["models"]:
        names.update(match.group(1) for value in entry.values() if isinstance(value, str) for match in SWAP_NAME.finditer(value))
    return names


def requests(hosts: list[str], keep: list[dict], delete: bool) -> dict[str, dict]:
    rows = [row for record_id in sorted(stage_lane.lane_records()["lanes"]) for row in lane_pack_files.lane_files(record_id)]
    swap = swap_names()
    result = {}
    for host in hosts:
        prefix = f"/home/{host}/"
        named = {path[len(prefix):] for row_host, _, _, path in rows if row_host == host and path.startswith(prefix)}
        result[host] = {"host": host, "referenced": sorted(named | swap), "required": sorted(named),
                        "companions": COMPANIONS, "keep": keep, "minimum_age_seconds": MINIMUM_AGE_SECONDS, "delete": delete}
    return result


def scan(hosts: list[str], keep: list[dict], delete: bool = False) -> dict[str, dict]:
    work = requests(hosts, keep, delete)

    def one(host: str):
        try:
            return host, lane_config.remote(host, NODE, work[host], timeout=3600 if delete else 600)
        except Exception as error:
            return host, {"error": str(error)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        return dict(pool.map(one, hosts))


def gigabytes(value: int) -> str:
    return f"{value / 1e9:.1f} GB"


def report(results: dict[str, dict], delete: bool) -> int:
    failed = 0
    for host, result in results.items():
        if "error" in result:
            print(f"UNREACHABLE {host} {result['error']}")
            failed = 1
            continue
        home = f"/home/{host}/"
        for path in result["missing"]:
            print(f"MISSING {host} {home}{path} is named by a lane record and absent")
        for path in result["linked"]:
            print(f"LINKED {host} {home}{path} is named by a lane record and is a symlink")
        for path in result["live"]:
            print(f"LIVE {host} {home}{path} is named only by a live config on the node, not by its record")
        for entry in result["entries"]:
            suffix = "/" if entry["kind"] == "directory" else " (link)" if entry["kind"] == "link" else ""
            print(f"UNUSED {host} {entry['bytes']} {home}{entry['path']}{suffix}")
        for entry in result["deleted"]:
            print(f"DELETED {host} {entry['bytes']} {home}{entry['path']}")
        for entry in result["refused"]:
            print(f"REFUSED {host} {home}{entry['path']}: {entry['reason']}")
            failed = 1
        unused = sum(entry["bytes"] for entry in result["entries"])
        kept = ", ".join(f"{name} {value} ({gigabytes(value)})" for name, value in sorted(result["kept"].items()))
        freed = f"; deleted {gigabytes(sum(entry['bytes'] for entry in result['deleted']))}" if delete else ""
        print(f"TOTAL {host} {unused} bytes unused ({gigabytes(unused)}) in {len(result['entries'])} entries{freed}; kept: {kept or 'nothing'}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", action="append", choices=stage_lane.HOSTS, help="scan only this host; repeat for more; all 16 by default")
    parser.add_argument("--keep", type=Path, default=KEEP, help="keep list; the repository's deployment/sparkdata_keep.json by default")
    parser.add_argument("--delete", action="store_true", help="delete the unused entries on the nodes (the owner's command)")
    args = parser.parse_args()
    results = scan(args.host or stage_lane.HOSTS, keep_list(args.keep), args.delete)
    return report(results, args.delete)


if __name__ == "__main__":
    sys.exit(main())
