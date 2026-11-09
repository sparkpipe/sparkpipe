#!/usr/bin/env python3

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Tuple

try:
    from . import spark_telemetry_common as telemetry
except ImportError:
    import spark_telemetry_common as telemetry


DEAD_AFTER_S = 300.0
RETRY_AFTER_S = 60.0
SEND_TIMEOUT_S = 20.0
REASON_CHARS = 160
WORKING = "working"
DEAD = "dead"


def failure_reason(row: Dict[str,object]) -> str:
    error = str(row.get("error","") or row.get("fetch_error",""))
    if error:
        return(error[-REASON_CHARS:])
    if telemetry.fnum(row,"sample_count") <= 0.0:
        return("no samples")
    age_s = telemetry.fnum(row,"last_sample_age_s")
    if age_s > telemetry.SAMPLE_STALE_AFTER_S:
        return("no new sample for %d s" % age_s)
    return("")


def advance(previous: Dict[str,Dict[str,object]], rows: Dict[str,Dict[str,object]], now: float) -> Tuple[Dict[str,Dict[str,object]],List[str]]:
    current: Dict[str,Dict[str,object]] = {}
    dead: List[str] = []
    for node,row in sorted(rows.items()):
        before = previous.get(node)
        if failure_reason(row) == "":
            current[node] = {"state":WORKING,"failing_since":0.0}
            continue
        if before is None:
            current[node] = {"state":DEAD,"failing_since":now}
            continue
        since = float(before["failing_since"]) or now
        current[node] = {"state":before["state"],"failing_since":since}
        if before["state"] == WORKING and now - since >= DEAD_AFTER_S:
            dead.append(node)
    return(current,dead)


def utc_text(unix_ts: float) -> str:
    return(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(unix_ts)))


def dead_message(dead: List[str], nodes: Dict[str,Dict[str,object]], rows: Dict[str,Dict[str,object]]) -> Tuple[str,str]:
    lines = ["Telemetry lost these nodes for %d s or more:" % DEAD_AFTER_S,""]
    for node in dead:
        lines.append("%s, failing since %s" % (node,utc_text(float(nodes[node]["failing_since"]))))
        lines.append("  %s" % failure_reason(rows[node]))
    lines.extend(["","You get no further mail for a node until it has recovered and died again."])
    return("SparkPipe: %s dead" % ", ".join(dead),"\n".join(lines) + "\n")


def send_mail(host: str, script: str, subject: str, body: str, ssh_config: str = "") -> str:
    cmd = [
        "ssh",
        *(["-F",ssh_config] if ssh_config else []),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        host,
        "python3 %s --subject %s" % (shlex.quote(script),shlex.quote(subject)),
    ]
    try:
        done = subprocess.run(cmd,input=body,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=SEND_TIMEOUT_S)
    except (OSError,subprocess.SubprocessError) as exc:
        return(str(exc))
    if done.returncode != 0:
        return(done.stderr.strip() or ("ssh exited %d" % done.returncode))
    return("")


def load_state(path: str) -> Dict[str,object]:
    try:
        with open(path,"r",encoding="utf-8") as fp:
            state = json.load(fp)
        return({"nodes":dict(state["nodes"]),"retry_after_unix":float(state["retry_after_unix"])})
    except (OSError,ValueError,KeyError,TypeError):
        return({"nodes":{},"retry_after_unix":0.0})


def notify_dead_nodes(summary: Dict[str,object], state_path: str, host: str, script: str, ssh_config: str = "") -> List[str]:
    now = time.time()
    rows = summary["nodes"]
    state = load_state(state_path)
    nodes,dead = advance(state["nodes"],rows,now)
    retry_after = state["retry_after_unix"]
    sent: List[str] = []
    if dead and now >= retry_after:
        subject,body = dead_message(dead,nodes,rows)
        error = send_mail(host,script,subject,body,ssh_config)
        if error:
            print("alert mail failed: %s" % error,file=sys.stderr,flush=True)
            retry_after = now + RETRY_AFTER_S
        else:
            sent = dead
            for node in dead:
                nodes[node]["state"] = DEAD
    next_state = {"nodes":nodes,"retry_after_unix":retry_after}
    if next_state != state:
        telemetry.write_json_atomic(state_path,next_state)
    return(sent)
