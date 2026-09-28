#!/usr/bin/env python3
import argparse
import pathlib
import shlex
import subprocess
import sys
import threading
import time


def launch(host, rank, args):
    root = args.root
    remote = (
        f"cd {root} && sha=$(cut -d' ' -f1 packs/rank{rank}.sp.sha256) && "
        f"exec systemd-run --user --unit=sp-mimo-tp-r{rank} --collect --wait --pipe --quiet "
        f"-E SPARK_WEIGHTD_SOCKET={args.socket} -E SPARK_WEIGHTD_LANE={args.lane} "
        f"-E SPARK_TP_MESH_RANKS={args.mesh} -E SPARK_TP_WAIT_MODE=hardware "
        f"{root}/bin/mimo26_tp_decode {rank} {root}/packs/rank{rank}.sp $sha "
        f"{root}/fixtures/{args.prompt}.prompt.i32 {root}/fixtures/{args.prompt}.expected.i32 "
        f"{args.expert_pool_bytes} {args.spine_budget_bytes}"
        + (f" {args.dump}" if args.dump else ""))
    return subprocess.Popen(['ssh', host, remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)


def pump(process, log, ready, rank):
    with open(log, 'w') as out:
        for line in process.stdout:
            out.write(line)
            out.flush()
            if line.startswith('M26TP-READY'):
                ready[rank] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosts', required=True)
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--lane', type=int, required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--socket', default='/tmp/spark_weightd.sock')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--expert-pool-bytes', type=int, default=42000000000)
    parser.add_argument('--spine-budget-bytes', type=int, default=2600000000)
    parser.add_argument('--ready-timeout', type=int, default=900)
    parser.add_argument('--dump', default='')
    args = parser.parse_args()
    hosts = args.hosts.split(',')
    if len(hosts) != 4 or len(args.mesh.split(',')) != 4:
        sys.exit('four hosts and four mesh ranks are required')
    logs = pathlib.Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    ready = [False] * 4
    processes, pumps = [], []
    for rank, host in enumerate(hosts):
        process = launch(host, rank, args)
        thread = threading.Thread(target=pump, args=(process, logs / f'{args.prompt}.rank{rank}.log', ready, rank), daemon=True)
        thread.start()
        processes.append(process)
        pumps.append(thread)
    deadline = time.time() + args.ready_timeout
    while not all(ready):
        if time.time() > deadline or any(p.poll() is not None for p in processes):
            for process in processes:
                process.terminate()
            sys.exit(f'M26TP-RUN-FAIL ranks ready={ready} before the release')
        time.sleep(0.2)
    print(f'M26TP-RELEASE all four ranks ready at {time.strftime("%H:%M:%S")}', flush=True)
    for process in processes:
        process.stdin.write('G')
        process.stdin.flush()
    codes = [process.wait() for process in processes]
    for thread in pumps:
        thread.join()
    for rank in range(4):
        for line in (logs / f'{args.prompt}.rank{rank}.log').read_text().splitlines():
            if line.startswith(('M26TP-PASS', 'M26TP-FAIL', 'M26TP-SUMMARY')):
                print(line)
    sys.exit(0 if codes == [0, 0, 0, 0] else 1)


if __name__ == '__main__':
    main()
