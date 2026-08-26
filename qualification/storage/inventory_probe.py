#!/usr/bin/env python3
"""Read-only physical storage and active-ownership inventory for Spark nodes."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


MODEL_BUDGET_BYTES = 1_000_000_000_000
KV_RESERVE_BYTES = 2_500_000_000_000
MIN_AVAILABLE_BYTES = 2_000_000_000_000
HOST_RE = re.compile(r"^[a-z][a-z0-9-]*$")

REMOTE_SCRIPT = r'''#!/usr/bin/env bash
set +e
root="$1"
section() { printf '\n@@%s@@\n' "$1"; }
section HOSTNAME
hostname 2>&1
section ROOT
printf 'root=%s\n' "$root"
if [ -d "$root" ]; then printf 'exists=1\n'; else printf 'exists=0\n'; fi
section FINDMNT
findmnt -T "$root" -o TARGET,SOURCE,FSTYPE,OPTIONS 2>&1
section LSBLK
lsblk -J -o NAME,KNAME,PKNAME,PATH,TYPE,MODEL,SERIAL,SIZE,FSTYPE,MOUNTPOINTS,TRAN,VENDOR,REV,WWN,HOTPLUG,RM 2>&1
section DF_BYTES
df -B1 --output=source,fstype,size,used,avail,pcent,target "$root" 2>&1
section DF_INODES
df --output=source,itotal,iused,iavail,ipcent,target "$root" 2>&1
section ROOT_STAT
stat -c 'type=%F uid=%u user=%U gid=%g group=%G mtime=%Y path=%n' "$root" 2>&1
section DU_TOP
if [ -d "$root" ]; then timeout 120s du -x -B1 --max-depth=1 "$root" 2>&1; else printf 'missing-root\n'; fi
section TOP_FIND
if [ -d "$root" ]; then timeout 60s find "$root" -xdev -mindepth 1 -maxdepth 2 -type d -printf '%p\t%u\t%g\t%T@\n' 2>&1 | head -5000; else printf 'missing-root\n'; fi
section PROCESSES
ps -eo pid=,user=,comm=,args= --sort=pid 2>&1 | awk 'BEGIN{IGNORECASE=1} /model|vllm|llama|resident|sparkpipe|inference|serve/ {print}' | head -500 2>&1
section SERVICES
systemctl list-units --type=service --state=running --no-legend 2>&1 | grep -Ei 'model|vllm|llama|resident|spark|inference|serve' | head -200
section PROC_DETAIL
for pid in $(ps -eo pid=,args= 2>/dev/null | awk 'BEGIN{IGNORECASE=1} /model|vllm|llama|resident|sparkpipe|inference|serve/ {print $1}' | head -100); do
    printf 'PID=%s\n' "$pid"
    readlink "/proc/$pid/exe" 2>&1
    readlink "/proc/$pid/cwd" 2>&1
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>&1
    printf '\n'
done
section CONFIG_PATHS
ps -eo args= 2>/dev/null | grep -Eo -- '(^|[[:space:]])(--config|--config-file|--model|--model-path|--checkpoint|--checkpoint-path|--kv-cache|--kv-path)(=|[[:space:]])[^[:space:]]+' | head -200
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_sections(raw: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    chunks: list[str] = []
    for line in raw.splitlines(keepends=True):
        match = re.match(r"^@@([A-Z_]+)@@\s*$", line.rstrip("\n"))
        if match:
            if current is not None:
                sections[current] = "".join(chunks)
            current = match.group(1)
            chunks = []
        elif current is not None:
            chunks.append(line)
    if current is not None:
        sections[current] = "".join(chunks)
    return sections


def clean_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def parse_findmnt(value: str) -> dict[str, str] | None:
    lines = clean_lines(value)
    if len(lines) < 2:
        return None
    header = re.split(r"\s+", lines[0])
    fields = re.split(r"\s+", lines[1], maxsplit=len(header) - 1)
    if len(fields) != len(header):
        return None
    return dict(zip(header, fields))


def parse_df(value: str) -> dict[str, object] | None:
    lines = clean_lines(value)
    if len(lines) < 2:
        return None
    fields = re.split(r"\s+", lines[-1], maxsplit=5)
    if len(fields) != 6:
        return None
    source, fstype, total, used, available, percent_target = fields
    match = re.match(r"^(\d+)%\s+(.+)$", percent_target)
    if match is None:
        return None
    return {
        "source": source,
        "fstype": fstype,
        "total_bytes": int(total),
        "used_bytes": int(used),
        "available_bytes": int(available),
        "available_percent": round(100.0 * int(available) / int(total), 3) if int(total) else 0.0,
        "reported_used_percent": int(match.group(1)),
        "target": match.group(2),
    }


def parse_inode_df(value: str) -> dict[str, object] | None:
    lines = clean_lines(value)
    if len(lines) < 2:
        return None
    fields = re.split(r"\s+", lines[-1], maxsplit=5)
    if len(fields) != 6:
        return None
    source, total, used, available, percent, target = fields
    match = re.match(r"^(\d+)%$", percent)
    if match is None:
        return None
    return {
        "source": source,
        "inode_total": int(total),
        "inode_used": int(used),
        "inode_available": int(available),
        "inode_used_percent": int(match.group(1)),
        "target": target,
    }


def parse_du(value: str) -> list[dict[str, object]]:
    rows = []
    for line in clean_lines(value):
        fields = line.split("\t", 1)
        if len(fields) != 2:
            fields = line.split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        path = fields[1]
        rows.append({"path": path, "bytes": int(fields[0])})
    return rows


def parse_lsblk(value: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("blockdevices"), list):
        return None
    return parsed


def physical_devices(lsblk: dict[str, object] | None, source: str | None) -> list[dict[str, object]]:
    if lsblk is None or source is None:
        return []
    devices = lsblk.get("blockdevices", [])
    if not isinstance(devices, list):
        return []
    wanted = os.path.basename(source)
    result = []

    def identity(node: dict[str, object]) -> dict[str, object]:
        return {key: node.get(key) for key in ("name", "path", "kname", "pkname", "type", "model", "serial", "size", "fstype", "mountpoints", "tran", "vendor", "rev", "wwn", "hotplug", "rm")}

    def walk(node: object, parent_chain: list[dict[str, object]] | None = None) -> None:
        if not isinstance(node, dict):
            return
        if parent_chain is None:
            parent_chain = []
        name = str(node.get("name", ""))
        path = str(node.get("path", ""))
        if name == wanted or path == source:
            parent = next((candidate for candidate in reversed(parent_chain) if candidate.get("type") == "disk"), None)
            if parent is None and parent_chain:
                parent = parent_chain[-1]
            result.append({
                "mounted_device": identity(node),
                "physical_parent": identity(parent) if parent is not None else identity(node),
            })
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                walk(child, [node] + parent_chain)

    for device in devices:
        walk(device)
    return result


def extract_references(sections: dict[str, str]) -> list[dict[str, object]]:
    refs = []
    process_lines = clean_lines(sections.get("PROC_DETAIL", ""))
    pid = None
    current: dict[str, object] | None = None
    for line in process_lines:
        match = re.match(r"PID=(\d+)$", line)
        if match:
            if current is not None:
                refs.append(current)
            pid = int(match.group(1))
            current = {"pid": pid, "evidence": []}
        elif current is not None:
            current["evidence"].append(line)
    if current is not None:
        refs.append(current)
    return refs


def extract_absolute_paths(sections: dict[str, str]) -> list[str]:
    text = "\n".join(sections.get(name, "") for name in ("PROCESSES", "SERVICES", "PROC_DETAIL", "CONFIG_PATHS"))
    return sorted(set(re.findall(r"/(?:home|mnt|opt|tmp|var|etc)/[^\s'\"]+", text)))


def path_identity(path: str, sections: dict[str, str]) -> dict[str, object]:
    lower = path.lower()
    if any(token in lower for token in ("kv", "cache", "paged", "resume", "kvcache")):
        kind = "kv_candidate"
    elif any(token in lower for token in ("model", "checkpoint", "weights", "shard", "pack", "dsv4", "qwen", "glm", "k3", "kimi", "minimax", "h3")):
        kind = "model_candidate"
    else:
        kind = "other_candidate"
    references = sections.get("PROCESSES", "") + sections.get("PROC_DETAIL", "") + sections.get("CONFIG_PATHS", "")
    active = "yes" if path in references else "unknown"
    return {"classification": kind, "active_reference": active}


def write_section_artifacts(host: str, raw_dir: Path, raw: bytes, sections: dict[str, str]) -> list[dict[str, object]]:
    files = [{"kind": "command_log", "path": str(raw_dir / f"{host}.log"), "sha256": sha256_bytes(raw), "bytes": len(raw)}]
    section_names = {
        "FINDMNT": "findmnt.txt",
        "LSBLK": "lsblk.json",
        "DF_BYTES": "df-bytes.txt",
        "DF_INODES": "df-inodes.txt",
        "DU_TOP": "du-top.txt",
        "PROCESSES": "processes.txt",
        "SERVICES": "services.txt",
        "PROC_DETAIL": "proc-detail.txt",
        "CONFIG_PATHS": "config-paths.txt",
    }
    for section, filename in section_names.items():
        data = sections.get(section, "").encode("utf-8", "replace")
        path = raw_dir / f"{host}.{filename}"
        path.write_bytes(data)
        files.append({"kind": filename.rsplit(".", 1)[0], "path": str(path), "sha256": sha256_bytes(data), "bytes": len(data)})
    return files


def inventory_host(host: str, root: str, raw_dir: Path, timeout_seconds: int) -> dict[str, object]:
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2", host, "bash", "-s", "--", root,
    ]
    started = time.time()
    proc = subprocess.run(
        command,
        input=REMOTE_SCRIPT.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    raw = proc.stdout
    log_path = raw_dir / f"{host}.log"
    log_path.write_bytes(raw)
    sections = parse_sections(raw.decode("utf-8", "replace"))
    artifacts = write_section_artifacts(host, raw_dir, raw, sections)
    findmnt = parse_findmnt(sections.get("FINDMNT", ""))
    df = parse_df(sections.get("DF_BYTES", ""))
    inode_df = parse_inode_df(sections.get("DF_INODES", ""))
    lsblk = parse_lsblk(sections.get("LSBLK", ""))
    devices = physical_devices(lsblk, str(df.get("source")) if df else None)
    du_rows = parse_du(sections.get("DU_TOP", ""))
    root_exists = "exists=1" in sections.get("ROOT", "")
    observed_host = clean_lines(sections.get("HOSTNAME", ""))[-1] if clean_lines(sections.get("HOSTNAME", "")) else None
    provenance_ok = bool(
        proc.returncode == 0 and root_exists and observed_host == host and findmnt and df and inode_df and lsblk and len(devices) == 1
    )
    source = findmnt.get("SOURCE") if findmnt else None
    mount_target = findmnt.get("TARGET") if findmnt else None
    local_shared = "unknown"
    if source and not source.startswith(("//", "ceph", "fuse.", "nfs", "cifs")) and findmnt and findmnt.get("FSTYPE") not in {"cifs", "nfs", "nfs4", "ceph", "fuse.ceph"}:
        parent = devices[0].get("physical_parent", {}) if devices else {}
        parent_name = str(parent.get("name", ""))
        transport = str(parent.get("tran", "")).lower()
        if parent_name.startswith("nvme") and transport not in {"usb", "firewire"}:
            local_shared = "internal_nvme_candidate"
        elif parent_name.startswith(("sd", "vd", "xvd")) and transport in {"usb", "sata", "nvme"}:
            local_shared = "external_local_nvme_candidate"
        else:
            local_shared = "local_block_device_unclassified"
    elif source or findmnt:
        local_shared = "shared_candidate"
    model_rows = []
    kv_bytes = 0
    model_bytes = 0
    for row in du_rows:
        path = str(row["path"])
        if path == root:
            continue
        identity = path_identity(path, sections)
        row.update(identity)
        row["file_count"] = None
        row["newest_mtime"] = None
        row["owner_model_identity"] = "unknown"
        row["rank_topology"] = "unknown"
        row["classification_evidence"] = "bounded top-level du plus process/config reference scan"
        model_rows.append(row)
        if identity["classification"] == "model_candidate":
            model_bytes += int(row["bytes"])
        if identity["classification"] == "kv_candidate":
            kv_bytes += int(row["bytes"])
    available = int(df["available_bytes"]) if df else None
    eligible = bool(
        provenance_ok and local_shared in {"internal_nvme_candidate", "external_local_nvme_candidate"} and available is not None
        and available >= MIN_AVAILABLE_BYTES and model_bytes <= MODEL_BUDGET_BYTES
        and available - model_bytes >= KV_RESERVE_BYTES
    )
    reasons = []
    if proc.returncode != 0:
        reasons.append(f"ssh_or_remote_exit_{proc.returncode}")
    if observed_host != host:
        reasons.append("hostname_mismatch")
    if not root_exists:
        reasons.append("missing_root")
    if not findmnt or not df or not inode_df:
        reasons.append("missing_filesystem_evidence")
    if len(devices) != 1:
        reasons.append("ambiguous_physical_device")
    if local_shared not in {"internal_nvme_candidate", "external_local_nvme_candidate"}:
        reasons.append("not_proven_local_filesystem")
    if available is not None and available < MIN_AVAILABLE_BYTES:
        reasons.append("available_below_2TB")
    if model_bytes > MODEL_BUDGET_BYTES:
        reasons.append("model_data_above_1TB")
    if available is not None and available - model_bytes < KV_RESERVE_BYTES:
        reasons.append("insufficient_post_model_kv_reserve")
    referenced_paths = extract_absolute_paths(sections)
    referenced_model_paths = [path for path in referenced_paths if path_identity(path, sections)["classification"] == "model_candidate"]
    referenced_kv_paths = [path for path in referenced_paths if path_identity(path, sections)["classification"] == "kv_candidate"]
    return {
        "hostname": host,
        "observed_hostname": observed_host,
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "data_path": root,
        "command": " ".join(shlex.quote(item) for item in command),
        "exit_code": proc.returncode,
        "raw_artifact": {"path": str(log_path), "sha256": sha256_bytes(raw), "bytes": len(raw), "files": artifacts},
        "mount": {
            "findmnt": findmnt,
            "lsblk": lsblk,
            "physical_devices": devices,
            "local_vs_shared": local_shared,
            "provenance_verified": provenance_ok,
        },
        "pressure": {**(df or {}), **(inode_df or {})},
        "active_ownership": {
            "process_evidence": clean_lines(sections.get("PROCESSES", "")),
            "service_evidence": clean_lines(sections.get("SERVICES", "")),
            "process_details": extract_references(sections),
            "config_evidence": clean_lines(sections.get("CONFIG_PATHS", "")),
            "checkpoint_model_revision": "unknown",
            "build_runtime_root": "unknown",
            "referenced_model_paths": referenced_model_paths,
            "referenced_kv_paths": referenced_kv_paths,
        },
        "consumers": model_rows,
        "reclaim_candidates": [],
        "model_bytes": model_bytes,
        "kv_bytes": kv_bytes,
        "model_budget_bytes": MODEL_BUDGET_BYTES,
        "kv_reserve_bytes": KV_RESERVE_BYTES,
        "eligible": eligible,
        "urgent": bool(reasons),
        "decision_reasons": reasons,
        "raw_sections": {
            "findmnt": clean_lines(sections.get("FINDMNT", "")),
            "lsblk_json": bool(lsblk),
            "df_bytes": clean_lines(sections.get("DF_BYTES", "")),
            "df_inodes": clean_lines(sections.get("DF_INODES", "")),
            "du_top": clean_lines(sections.get("DU_TOP", "")),
            "process_service_config": bool(sections.get("PROCESSES") or sections.get("SERVICES") or sections.get("CONFIG_PATHS")),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--root-template", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-workers", required=True, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hosts = [item.strip() for item in args.hosts.split(",") if item.strip()]
    if not hosts or any(not HOST_RE.fullmatch(host) for host in hosts):
        raise SystemExit("invalid host list")
    if "{host}" not in args.root_template:
        raise SystemExit("root template must contain {host}")
    if args.max_workers < 1 or args.max_workers > len(hosts):
        raise SystemExit("max-workers must be between one and the host count")
    output = args.output.resolve()
    raw_dir = output.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict[str, dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(inventory_host, host, args.root_template.format(host=host), raw_dir, 240): host
            for host in hosts
        }
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                results[host] = future.result()
            except (OSError, subprocess.SubprocessError, concurrent.futures.TimeoutError) as error:
                log_path = raw_dir / f"{host}.log"
                raw = f"local probe exception: {type(error).__name__}: {error}\n".encode()
                log_path.write_bytes(raw)
                results[host] = {
                    "hostname": host,
                    "observed_hostname": None,
                    "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "data_path": args.root_template.format(host=host),
                    "exit_code": None,
        "raw_artifact": {"path": str(log_path), "sha256": sha256_bytes(raw), "bytes": len(raw), "files": [{"kind": "command_log", "path": str(log_path), "sha256": sha256_bytes(raw), "bytes": len(raw)}]},
                    "mount": {"findmnt": None, "lsblk": None, "physical_devices": [], "local_vs_shared": "unknown", "provenance_verified": False},
                    "pressure": {},
                    "active_ownership": {"process_evidence": [], "service_evidence": [], "process_details": [], "config_evidence": [], "checkpoint_model_revision": "unknown", "build_runtime_root": "unknown", "referenced_model_paths": [], "referenced_kv_paths": []},
                    "consumers": [], "reclaim_candidates": [], "model_bytes": None, "kv_bytes": None, "model_budget_bytes": MODEL_BUDGET_BYTES, "kv_reserve_bytes": KV_RESERVE_BYTES,
                    "eligible": False, "urgent": True, "decision_reasons": ["local_probe_exception", "unknown_ownership_quarantine"],
                    "raw_sections": {},
                }
    ordered = [results[host] for host in hosts]
    no_action_manifest = [
        {
            "node": row["hostname"],
            "physical_mount": row["mount"].get("findmnt"),
            "exact_path": None,
            "bytes": 0,
            "reason": "No candidate has active_reference=no; every unknown owner remains quarantined.",
            "active_reference_proof": "No verified inactive-reference proof in bounded process/config evidence.",
            "proposed_action": "NO ACTION; coordinator approval and a fresh ownership proof are required.",
            "risk": "Mutation could affect active model/build/KV data.",
            "verification_needed_before_apply": ["service drain/readiness proof", "exact path ownership", "fresh findmnt and lsblk", "hash or recoverable staging receipt"],
        }
        for row in ordered
    ]
    receipt = {
        "schema_version": 1,
        "probe": "qualification/storage/inventory_probe.py",
        "question": "For spark0 through sparkf, what are the observed physical device and mount for /home/{host}/sparkdata, free bytes and inode pressure, active model and KV references, and bounded consumers, and which nodes are eligible for a rank-local NVMe plan under the 1000000000000-byte model ceiling and 2000000000000..2500000000000-byte KV reserve without mutation?",
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "read_only": True,
        "hosts": ordered,
        "thresholds": {"model_budget_bytes": MODEL_BUDGET_BYTES, "minimum_available_bytes": MIN_AVAILABLE_BYTES, "kv_reserve_bytes": KV_RESERVE_BYTES},
        "summary": {
            "host_count": len(hosts),
            "provenance_verified_count": sum(1 for row in ordered if row["mount"]["provenance_verified"]),
            "eligible_hosts": [row["hostname"] for row in ordered if row["eligible"]],
            "urgent_hosts": [row["hostname"] for row in ordered if row["urgent"]],
            "verified_reclaimable_bytes": {host: 0 for host in hosts},
            "no_action_manifest": no_action_manifest,
            "unknown_ownership_quarantined": [row["hostname"] for row in ordered if not row["mount"]["provenance_verified"] or any(item.get("active_reference") == "unknown" for item in row["consumers"])],
        },
        "decision": {
            "ranked_urgent_nodes": [row["hostname"] for row in sorted(ordered, key=lambda item: (not item["urgent"], item["pressure"].get("available_bytes", 0)))],
            "verified_reclaimable_bytes_per_node": {host: 0 for host in hosts},
            "exact_no_action_manifest": no_action_manifest,
            "policy": "Unknown active ownership is quarantined. No reclaim, move, mount, unmount, copy, or service action is authorized by this receipt.",
        },
        "raw_artifacts": [row["raw_artifact"] for row in ordered],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "host_count": len(hosts), "eligible_hosts": receipt["summary"]["eligible_hosts"], "urgent_hosts": receipt["summary"]["urgent_hosts"]}, sort_keys=True))
    return 0 if all(row["mount"]["provenance_verified"] for row in ordered) else 1


if __name__ == "__main__":
    raise SystemExit(main())
