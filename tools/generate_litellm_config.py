#!/usr/bin/env python3
"""generate_litellm_config - emit a LiteLLM proxy config that routes
model names to the fleet's per-model sparkpipe APIs (OpenAI-compatible:
/v1/models, /v1/completions, /v1/chat/completions).

  generate_litellm_config.py --registry models.json --out litellm_config.yaml
  generate_litellm_config.py --pair glm53flash.bf16.tp16=http://spark0:8433 ...

registry JSON: [{"name": "<arm>", "base_url": "http://host:port"}, ...]
The proxy serves the admin page at /ui and OpenAI traffic at /v1; the
backend API key (if the fleet requires one) comes from the environment
variable SPARK_API_KEY.
"""
import argparse
import json
import pathlib
import sys

try:
    import yaml
except ImportError:
    yaml = None


def load_registry(path):
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise SystemExit("registry must be a JSON array")
    models = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit("registry entries must be objects")
        name = entry.get("name")
        base = entry.get("base_url")
        if not name or not base:
            raise SystemExit("registry entries need name and base_url")
        models.append((name, base))
    return models


def build_config(models, database_url=None, master_key=False):
    model_list = []
    for name, base in models:
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        model_list.append({
            "model_name": name,
            "litellm_params": {
                "model": "openai/" + name,
                "api_base": base,
                "api_key": "os.environ/SPARK_API_KEY",
            },
        })
    general = {}
    if database_url:
        general["database_url"] = database_url
    if master_key:
        general["master_key"] = "os.environ/SPARK_LITELLM_MASTER_KEY"
    config = {
        "model_list": model_list,
        "litellm_settings": {
            "drop_params": True,
            "num_retries": 1,
            "request_timeout": 600,
        },
    }
    if general:
        config["general_settings"] = general
    return config


def emit(config, out_text):
    if yaml is None:
        raise SystemExit("pyyaml required: pip install pyyaml")
    print(f"litellm config: {out_text} models={len(config['model_list'])}")
    destination = pathlib.Path(out_text)
    if ".." in destination.parts:
        raise SystemExit("REFUSED: path contains '..'")
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry")
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--database-url",
                        help="postgresql:// connection string for LiteLLM's "
                             "virtual keys / spend tracking (optional)")
    parser.add_argument("--master-key", action="store_true",
                        help="require SPARK_LITELLM_MASTER_KEY bearer auth "
                             "(default: no passwords, per operator ruling)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    models = []
    if args.registry:
        models.extend(load_registry(args.registry))
    for pair in args.pair:
        if "=" not in pair:
            raise SystemExit(f"pair must be name=url: {pair}")
        name, base = pair.split("=", 1)
        models.append((name, base))
    if not models:
        raise SystemExit("no models: pass --registry and/or --pair")
    emit(build_config(models, args.database_url, args.master_key), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
