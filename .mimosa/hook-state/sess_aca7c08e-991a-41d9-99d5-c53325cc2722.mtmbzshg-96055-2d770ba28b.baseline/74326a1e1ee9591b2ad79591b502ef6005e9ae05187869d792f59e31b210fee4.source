#!/usr/bin/env python3
"""glm5_next fixed-pack closeout orchestrator (controller side).

Polls the 15 in-flight rank pack builds (rank r builds on sparke-hex r;
rank 0 is complete and verified). Per rank, per poll:
  - no receipt + no builder process  -> REQUEUE-NEEDED flag (build died)
  - builder in D state               -> D-streak counter; >= limit (~30 min)
                                        flags WEDGED-D for the lane to act on
  - receipt line in build log        -> trigger the pack verify on that node
    (committed lane branch shipped as a git bundle; extracted via
    `git archive` — never a hand-copied tools/ dir), then track the verify
    log to VERIFIED / VERIFY-FAIL.

State machine per rank: building -> done -> verifying -> verified|verify-fail.
One line of state per poll goes to stdout. Exit 0 when all ranks verified.
Never signals, kills, or restarts anything on the nodes.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RANKS = list(range(1, 16))
HOSTS = {r: f"spark{r:x}" for r in RANKS}
DEEP_RANKS = {1, 8, 15}
RECEIPT_GREP = "grep -o 'rank{r}.g5nsp: [0-9]+ tensors, [0-9]+ bytes'"

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


def ssh(host: str, cmd: str, timeout: int = 30) -> str:
    try:
        out = subprocess.run(SSH + [host, cmd], capture_output=True,
                             text=True, timeout=timeout)
        return out.stdout.strip()
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"


def poll_rank(r: int) -> dict:
    h = HOSTS[r]
    cmd = f"""
p=$HOME/glm53_packs_fixed/glm5_next_stage.tp16.rank{r}.g5nsp
sz=$(stat -c%s "$p" 2>/dev/null || echo 0)
pid=$(pgrep -f '^python3 .*resident_stagepack.py .*--tp-rank {r} ' | head -1)
if [ -n "$pid" ]; then st=$(ps -o stat= -p $pid | cut -c1-1); et=$(ps -o etimes= -p $pid); else st=X; et=-; fi
rec=$(grep -oE 'rank{r}\\.g5nsp: [0-9]+ tensors, [0-9]+ bytes' /tmp/packbuild-r{r}.log 2>/dev/null | head -1)
vst=none
[ -f /tmp/g5verify-r{r}.log ] && vst=$(tail -c 400 /tmp/g5verify-r{r}.log | grep -qE 'VERIFY-PASS|FAIL' && echo finished || echo running)
echo "$sz|$st|$et|$rec|$vst"
"""
    out = ssh(h, cmd)
    if out in ("__TIMEOUT__", ""):
        return {"rank": r, "host": h, "state": "unreachable"}
    sz, st, et, rec, vst = (out.split("|") + [""] * 5)[:5]
    return {"rank": r, "host": h, "size": int(sz or 0), "proc_state": st,
            "etimes": et, "receipt": rec, "verify_state": vst}


def trigger_verify(r: int, bundle_local: str, deep: bool) -> str:
    h = HOSTS[r]
    flag = "--deep" if deep else ""
    setup = f"""
if [ ! -f /tmp/g5close.bundle ]; then exit 33; fi
git -C ~/g5rt2-src fetch -q /tmp/g5close.bundle 'refs/heads/lane/glm5-closeout:refs/remotes/g5close/branch' || exit 34
rm -rf /tmp/g5close-verify && mkdir -p /tmp/g5close-verify
git -C ~/g5rt2-src archive refs/remotes/g5close/branch tools/ | tar -x -C /tmp/g5close-verify || exit 35
"""
    run = (f"cd /tmp/g5close-verify && nohup python3 tools/glm5_next_pack_verify.py "
           f"--pack $HOME/glm53_packs_fixed/glm5_next_stage.tp16.rank{r}.g5nsp "
           f"--source /mnt/model-warm/glm-5.3-flash --tp-rank {r} {flag} "
           f"> /tmp/g5verify-r{r}.log 2>&1 < /dev/null & echo LAUNCHED")
    out = ssh(h, setup + run, timeout=240)
    if out.endswith("LAUNCHED"):
        return "launched"
    if out.endswith("33"):
        return f"no-bundle-on-{h}"
    if out.endswith("34"):
        return f"fetch-failed-on-{h}"
    if out.endswith("35"):
        return f"archive-failed-on-{h}"
    return f"setup-unclear: {out[-80:]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="/tmp/g5close.bundle")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--d-limit", type=int, default=6,
                    help="consecutive D polls before WEDGED-D (x interval)")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    state = {r: "building" for r in RANKS}
    d_streak = {r: 0 for r in RANKS}

    while True:
        lines = []
        remaining = []
        for r in RANKS:
            if state[r] in ("verified", "verify-fail", "requeue-needed"):
                continue
            info = poll_rank(r)
            st = info.get("state", "ok")
            if st == "unreachable":
                lines.append(f"r{r}({info['host']}): UNREACHABLE")
                remaining.append(r)
                continue
            if info["receipt"]:
                if state[r] == "building" or state[r].startswith("verify-trigger"):
                    vres = "skipped(no-bundle)"
                    if bundle.exists():
                        subprocess.run(["scp", "-q", str(bundle),
                                        f"{info['host']}:/tmp/"], timeout=180)
                        vres = trigger_verify(r, str(bundle), r in DEEP_RANKS)
                    state[r] = "verifying" if vres == "launched" else f"verify-trigger-{vres}"
                    time.sleep(20)  # stagger the verify herd off ceph
                elif state[r] == "verifying":
                    pass
                lines.append(f"r{r}({info['host']}): DONE {info['size']} B, "
                             f"verify={state[r]}")
                if state[r] == "verifying" and info["verify_state"] == "finished":
                    vlog = ssh(info["host"],
                               "tail -2 /tmp/g5verify-r%d.log" % r, timeout=20)
                    if "VERIFY-PASS" in vlog:
                        state[r] = "verified"
                        lines.append(f"r{r}: VERIFIED | {vlog.splitlines()[-1]}")
                    else:
                        state[r] = "verify-fail"
                        lines.append(f"r{r}: VERIFY-FAIL | {vlog}")
                elif state[r] == "verifying":
                    remaining.append(r)
                elif state[r] != "verified":
                    remaining.append(r)
                continue
            # still building (or dead)
            if info["proc_state"] == "X":
                state[r] = "requeue-needed"
                lines.append(f"r{r}({info['host']}): NO PROCESS, NO RECEIPT, "
                             f"size={info['size']} -> REQUEUE-NEEDED")
                continue
            if info["proc_state"] == "D":
                d_streak[r] += 1
            else:
                d_streak[r] = 0
            note = (f" !WEDGED-D({d_streak[r]})"
                    if d_streak[r] >= args.d_limit else "")
            lines.append(f"r{r}({info['host']}): {info['size']} B "
                         f"st={info['proc_state']} et={info['etimes']}s{note}")
            remaining.append(r)
        stamp = time.strftime("%FT%TZ", time.gmtime())
        print(f"{stamp} " + " | ".join(lines), flush=True)
        done = [r for r in RANKS if state[r] in ("verified",)]
        failed = [r for r in RANKS if state[r] in ("verify-fail", "requeue-needed")]
        print(f"{stamp} summary: verified={len(done)}/15 failed={failed} "
              f"pending={len(remaining)}", flush=True)
        if not remaining:
            print(f"{stamp} ORCHESTRATION-COMPLETE verified={sorted(done)} "
                  f"failed={sorted(failed)}", flush=True)
            return 0 if not failed else 1
        if args.once:
            print(f"{stamp} --once: state={json.dumps(state)}", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
