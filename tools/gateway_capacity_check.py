#!/usr/bin/env python3
import argparse
import http.client
import ipaddress
import json
import math
import sys
import threading
import time
import urllib.parse

WINDOW_SECONDS = 600
TEST_HOSTS = frozenset({"localhost"})
TEST_LABELS = frozenset({"sandbox", "staging"})
SANDBOX_PREFIXES = ("/pay/sandbox/api/",)


class Refused(Exception):
    pass


def test_host(hostname):
    if hostname in TEST_HOSTS:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return any(label in TEST_LABELS for label in hostname.lower().split("."))


def sandbox_path(path):
    segments = urllib.parse.unquote(path).split("/")
    return path.startswith(SANDBOX_PREFIXES) and "." not in segments and ".." not in segments and "%" not in path


def target(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise Refused(f"{url} is not an http or https URL")
    if not (test_host(parts.hostname) or sandbox_path(parts.path)):
        raise Refused(f"{url} is neither a loopback host, a host with a sandbox or staging label, nor a path under {SANDBOX_PREFIXES[0]}; the capacity check never runs against production")
    return parts


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def worker(parts, method, body, headers, deadline, timeout, results, lock):
    connection = None
    path = (parts.path or "/") + ("?" + parts.query if parts.query else "")
    kind = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    while time.monotonic() < deadline:
        if connection is None:
            connection = kind(parts.hostname, parts.port, timeout=timeout)
        started = time.monotonic()
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            response.read()
            status = response.status
            if response.getheader("Connection", "").lower() == "close":
                connection.close()
                connection = None
        except (OSError, http.client.HTTPException) as error:
            status = type(error).__name__
            connection.close()
            connection = None
        with lock:
            results.append((status, time.monotonic() - started))
    if connection is not None:
        connection.close()


def measure(url, method, body, headers, concurrency, seconds, timeout):
    parts = target(url)
    results, lock = [], threading.Lock()
    started = time.monotonic()
    deadline = started + seconds
    threads = [threading.Thread(target=worker, args=(parts, method, body, headers, deadline, timeout, results, lock), daemon=True) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started
    ok = [latency for status, latency in results if isinstance(status, int) and 200 <= status < 300]
    statuses = {}
    for status, _ in results:
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    rate = len(ok) / elapsed if elapsed > 0 else 0.0
    return {"url": url, "method": method, "concurrency": concurrency, "seconds": round(elapsed, 3), "requests": len(results), "ok": len(ok),
            "statuses": statuses, "throttled": statuses.get("429", 0), "requests_per_second": round(rate, 3),
            "p50_ms": round(percentile(ok, 0.50) * 1000, 3) if ok else None, "p99_ms": round(percentile(ok, 0.99) * 1000, 3) if ok else None,
            "window_seconds": WINDOW_SECONDS, "capacity_per_window": math.floor(rate * WINDOW_SECONDS)}


def parse(argv):
    parser = argparse.ArgumentParser(description="Measure the requests per second a sandbox or staging SparkPipe gateway sustains, for SPARKPIPE_GATEWAY_CAPACITY")
    parser.add_argument("url", help="a gateway route on a sandbox, staging or loopback gateway, such as http://127.0.0.1:18091/v1/chat/completions")
    parser.add_argument("--method", default="POST")
    parser.add_argument("--body-file", help="the request body to send each time")
    parser.add_argument("--key-file", help="a file holding the API key, sent as Authorization: Bearer")
    parser.add_argument("--header", action="append", default=[], help="an extra NAME: VALUE header that is not a secret")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", required=True, help="where to write the JSON result")
    arguments = parser.parse_args(argv)
    if arguments.concurrency < 1 or arguments.seconds <= 0 or arguments.timeout <= 0:
        parser.error("--concurrency, --seconds and --timeout must be positive")
    return arguments


def main(argv):
    arguments = parse(argv)
    headers = {"Content-Type": "application/json"}
    for item in arguments.header:
        name, _, value = item.partition(":")
        if not name.strip() or not value.strip():
            print(f"gateway_capacity_check: --header takes NAME: VALUE, not {item!r}", file=sys.stderr)
            return 2
        headers[name.strip()] = value.strip()
    if arguments.key_file:
        with open(arguments.key_file, encoding="utf-8") as handle:
            headers["Authorization"] = "Bearer " + handle.read().strip()
    body = None
    if arguments.body_file:
        with open(arguments.body_file, "rb") as handle:
            body = handle.read()
    try:
        result = measure(arguments.url, arguments.method.upper(), body, headers, arguments.concurrency, arguments.seconds, arguments.timeout)
    except Refused as error:
        print(f"gateway_capacity_check: {error}", file=sys.stderr)
        return 2
    with open(arguments.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"gateway_capacity_check: {result['ok']} of {result['requests']} requests answered 2xx in {result['seconds']} s at concurrency {result['concurrency']}: "
          f"{result['requests_per_second']} per second, p50 {result['p50_ms']} ms, p99 {result['p99_ms']} ms, capacity_per_window {result['capacity_per_window']}")
    if result["throttled"] or result["ok"] != result["requests"] or not result["ok"]:
        print("gateway_capacity_check: some requests were refused or failed, so this is not the gateway's capacity; raise the sandbox's SPARKPIPE_GATEWAY_CAPACITY, "
              "use a key whose account holds a Fleet, and run again", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
