#!/usr/bin/env python3
import json
import sys
import urllib.request


def arguments(models_endpoint: str, model: str) -> list[str]:
    with urllib.request.urlopen(models_endpoint.rstrip("/") + "/v1/models", timeout=30) as resp:
        listing = json.loads(resp.read())
    for entry in listing.get("data", []):
        if entry.get("id") == model:
            if "reasoning_efforts" not in entry:
                raise SystemExit(f"{models_endpoint}/v1/models lists {model} without reasoning_efforts; point it at the chat layer")
            efforts = entry["reasoning_efforts"]
            if "minimal" in efforts:
                return ["--reasoning-effort", "minimal", "--max-tokens", "2048"]
            if "low" in efforts:
                return ["--reasoning-effort", "low", "--max-tokens", "4096"]
            raise SystemExit(f"{model} lists neither minimal nor low reasoning: {efforts}")
    raise SystemExit(f"{model} is not in {models_endpoint}/v1/models")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compsec_effort.py MODELS_ENDPOINT MODEL", file=sys.stderr)
        return 2
    print(" ".join(arguments(sys.argv[1], sys.argv[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
