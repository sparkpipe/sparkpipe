#!/usr/bin/env python3
import http.client
import json
import random
import re
import subprocess
import sys

API_HOST = "rtx5090"
API_PORT = 8433
ROOT_NAME = "glm53flash.fp8.tp16"
RESIDENTD_LOG = f"~/sparkdata/{ROOT_NAME}/residentd.log"
LOAD_DRIVER = re.compile(r"LoadDriver rc=(-?\d+) .*kv_pages=(\d+)/(\d+)")
CHAIN = re.compile(r"CHAIN-TIME slot=\d+ path=(\w+) steps=(\d+) status=(-?\d+) total_ms=([\d.]+)")
CHAIN_SAMPLE = 200
MAX_STEP_MS = 1000.0
FAILURES = []


def ssh(node, command, timeout=30):
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, command],
            capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as error:
        return 1, str(error)


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS: {name} {detail}".rstrip())
    else:
        print(f"  FAIL: {name} {detail}".rstrip())
        FAILURES.append(f"{name}: {detail}")


def kv_capacity(node):
    rc, out = ssh(node, f"grep 'LoadDriver rc=' {RESIDENTD_LOG} | tail -1")
    match = LOAD_DRIVER.search(out) if rc == 0 else None
    check("driver load reported", match is not None, f"rc={rc} line={out!r}")
    if match is None:
        return
    status, logical, physical = (int(value) for value in match.groups())
    check("driver loaded", status == 0, f"rc={status}")
    check("JIT KV pages configured", logical > 0 and physical > 0, f"kv_pages={logical}/{physical}")


def chain_stalls(node):
    rc, out = ssh(node, f"grep CHAIN-TIME {RESIDENTD_LOG} | tail -{CHAIN_SAMPLE}")
    chains = [(path, int(steps), int(status), float(total)) for path, steps, status, total in CHAIN.findall(out)] if rc == 0 else []
    check("decode chains ran", len(chains) > 0, f"rc={rc} chains={len(chains)}")
    if not chains:
        return
    failed = [chain for chain in chains if chain[2] != 0 or chain[1] == 0]
    check("every chain completed", not failed, f"failed={failed[:3]}")
    worst = max(chain[3] / max(chain[1], 1) for chain in chains)
    check(f"no chain step over {MAX_STEP_MS:.0f} ms", worst <= MAX_STEP_MS, f"worst_step_ms={worst:.1f} chains={len(chains)}")


def complete(prompt, max_tokens):
    connection = http.client.HTTPConnection(API_HOST, API_PORT, timeout=120)
    body = json.dumps({"model": "sparkpipe-model", "prompt_token_ids": prompt, "max_tokens": max_tokens, "temperature": 0})
    try:
        connection.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read().decode()
        return response.status, json.loads(payload) if response.status == 200 else {"error": payload[:200]}
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
        return 0, {"error": str(error)}
    finally:
        connection.close()


def served(name, status, data, max_tokens):
    tokens = data.get("tokens", [])
    finish = data.get("finish_reason")
    complete_tokens = 0 < len(tokens) <= max_tokens and (len(tokens) == max_tokens) == (finish == "length")
    check(name, status == 200 and data.get("status") == 0 and complete_tokens,
          f"status={status} ntok={len(tokens)} finish={finish} error={data.get('error')}")
    return tokens


def long_context():
    status, data = complete(list(range(1, 513)), 4)
    served("512-token context served", status, data, 4)


def repeated_prompt():
    prompt = [random.randrange(1000, 100000) for _ in range(64)]
    status, first = complete(prompt, 8)
    first_tokens = served("fresh 64-token prompt served", status, first, 8)
    status, second = complete(prompt, 8)
    second_tokens = served("repeated 64-token prompt served", status, second, 8)
    check("repeated prompt (prefix-cache hit) decodes the same tokens as the miss", first_tokens == second_tokens,
          f"first={first_tokens} second={second_tokens}")


def main():
    if len(sys.argv) != 2:
        print("usage: test_jit_kv_page_fault.py <node>", file=sys.stderr)
        return 2
    node = sys.argv[1]
    print(f"=== JIT KV cache on {node} ===")
    kv_capacity(node)
    chain_stalls(node)
    long_context()
    repeated_prompt()
    print(f"=== summary: {len(FAILURES)} failures ===")
    for failure in FAILURES:
        print(f"  {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
