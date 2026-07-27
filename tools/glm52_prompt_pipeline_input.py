#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODEL_DIR = "/home/spark2/models/hf/nvidia/GLM-5.2-NVFP4"
SCHEMA = "sparkpipe.glm52.prompt_pipeline_input.v1"
PREFILL_PLAN_SCHEMA = "sparkpipe.glm52.prompt_prefill_plan.v1"
PREFILL_BLOCK_TOKENS = 16
DEFAULT_PREFILL_CHUNK_TOKENS = 256
TAIL_WINDOW_TOKEN_COUNT = 4


class PromptInputFailure(Exception):
    pass


def parse_token_ids(text: str) -> List[int]:
    result: List[int] = []
    for raw_item in text.replace("\n", ",").replace(" ", ",").split(","):
        item = raw_item.strip()
        if item == "":
            continue
        try:
            token_id = int(item, 10)
        except ValueError as exc:
            raise PromptInputFailure(f"invalid token id: {item}") from exc
        if token_id < 0:
            raise PromptInputFailure(f"negative token id: {token_id}")
        result.append(token_id)
    if not result:
        raise PromptInputFailure("token id list is empty")
    return result


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise PromptInputFailure("set --prompt, --prompt-file, or --token-ids")



def resolve_c_tokenizer_json(args: argparse.Namespace, model_dir: Path) -> Optional[Path]:
    if args.tokenizer_json is not None:
        return Path(args.tokenizer_json)
    env_path = os.environ.get("GLM52_TOKENIZER_JSON")
    if env_path:
        return Path(env_path)
    default_path = model_dir / "tokenizer.json"
    if default_path.is_file():
        return default_path
    return None


def resolve_tool_path(root: Path, tool_path: str) -> Path:
    resolved_tool_path = Path(tool_path)
    if not resolved_tool_path.is_absolute():
        resolved_tool_path = root / resolved_tool_path
    if not resolved_tool_path.is_file():
        raise PromptInputFailure(f"missing C tokenizer binary: {resolved_tool_path}")
    return resolved_tool_path


def tokenize_with_spark_c(args: argparse.Namespace, model_dir: Path, prompt: str) -> Tuple[List[int], str]:
    if args.c_tokenizer_add_bos or args.c_tokenizer_add_eos:
        raise PromptInputFailure("C tokenizer BOS/EOS insertion is not implemented yet")
    tokenizer_json_path = resolve_c_tokenizer_json(args, model_dir)
    if tokenizer_json_path is None:
        raise PromptInputFailure("missing tokenizer.json; set --tokenizer-json or GLM52_TOKENIZER_JSON")
    if not tokenizer_json_path.is_file():
        raise PromptInputFailure(f"missing tokenizer.json: {tokenizer_json_path}")

    root = Path(__file__).resolve().parents[1]
    if args.chat:
        tokenizer_binary = resolve_tool_path(root, args.glm52_tokenizer_binary)
        command = [
            str(tokenizer_binary),
            "--tokenizer-json", str(tokenizer_json_path),
            "--prompt", prompt,
            "--chat",
        ]
        if args.system_prompt is not None:
            command.extend(["--system-prompt", args.system_prompt])
        if args.no_add_generation_prompt:
            command.append("--no-add-generation-prompt")
        if args.c_tokenizer_disable_thinking:
            command.append("--disable-thinking")
        if args.reasoning_effort is not None:
            command.extend(["--reasoning-effort", args.reasoning_effort])
        tokenizer_kind = "spark_c_glm52_chat_byte_bpe"
    else:
        tokenizer_binary = resolve_tool_path(root, args.tokenizer_binary)
        command = [
            str(tokenizer_binary),
            "--tokenizer-json", str(tokenizer_json_path),
            "--prompt", prompt,
        ]
        if args.disable_special_tokens or not args.allow_special_tokens:
            command.append("--disable-special")
        if args.c_tokenizer_add_prefix_space:
            command.append("--add-prefix-space")
        tokenizer_kind = "spark_c_hf_json_byte_bpe"

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip()
        raise PromptInputFailure(f"C tokenizer failed: {error_text}")
    return parse_token_ids(completed.stdout), tokenizer_kind

def tokenize_with_transformers(model_dir: Path, prompt: str, chat: bool, system_prompt: Optional[str], add_generation_prompt: bool) -> Tuple[List[int], str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise PromptInputFailure("transformers is not installed") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    if chat:
        messages: List[Dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
        return [int(token_id) for token_id in token_ids], "transformers_chat_template"
    encoded = tokenizer(prompt, add_special_tokens=True)
    return [int(token_id) for token_id in encoded["input_ids"]], "transformers_encode"


def tokenize_with_tokenizers(model_dir: Path, prompt: str, chat: bool) -> Tuple[List[int], str]:
    if chat:
        raise PromptInputFailure("chat template mode needs transformers for now")
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise PromptInputFailure(f"missing tokenizer.json: {tokenizer_path}")
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:
        raise PromptInputFailure("tokenizers is not installed") from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded = tokenizer.encode(prompt)
    return [int(token_id) for token_id in encoded.ids], "tokenizers_json"


def tokenize_prompt(args: argparse.Namespace, model_dir: Path) -> Tuple[List[int], str]:
    if args.token_ids is not None:
        return parse_token_ids(args.token_ids), "explicit_token_ids"
    prompt = load_prompt(args)
    errors: List[str] = []
    try:
        return tokenize_with_spark_c(args, model_dir, prompt)
    except PromptInputFailure as exc:
        errors.append(str(exc))
    for tokenizer_fn in (tokenize_with_transformers,):
        try:
            return tokenizer_fn(model_dir, prompt, args.chat, args.system_prompt, not args.no_add_generation_prompt)
        except PromptInputFailure as exc:
            errors.append(str(exc))
    try:
        return tokenize_with_tokenizers(model_dir, prompt, args.chat)
    except PromptInputFailure as exc:
        errors.append(str(exc))
    raise PromptInputFailure("unable to tokenize prompt locally: " + "; ".join(errors))


def choose_bootstrap_token(token_ids: List[int], policy: str) -> int:
    if policy == "last":
        return token_ids[-1]
    if policy == "first":
        return token_ids[0]
    raise PromptInputFailure(f"unsupported bootstrap policy: {policy}")


def validate_prefill_chunk_tokens(chunk_tokens: int) -> int:
    if chunk_tokens <= 0:
        raise PromptInputFailure("prefill chunk token count must be positive")
    if chunk_tokens % PREFILL_BLOCK_TOKENS != 0:
        raise PromptInputFailure(f"prefill chunk token count must be a multiple of {PREFILL_BLOCK_TOKENS}")
    return chunk_tokens


def ceil_divide(value: int, divisor: int) -> int:
    if value == 0:
        return 0
    return (value + divisor - 1) // divisor


def build_prefill_chunks(token_ids: List[int], chunk_tokens: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    prefill_token_count = len(token_ids)
    offset = 0
    chunk_index = 0
    while offset < prefill_token_count:
        token_count = min(chunk_tokens, prefill_token_count - offset)
        next_offset = offset + token_count
        chunks.append({
            "chunk_index": chunk_index,
            "token_offset": offset,
            "token_count": token_count,
            "token_ids": token_ids[offset:next_offset],
            "final_prefill_chunk": next_offset == prefill_token_count,
            "scheduled_block_count": ceil_divide(token_count, PREFILL_BLOCK_TOKENS),
            "cache_commit_token_count_after_step": next_offset,
        })
        offset = next_offset
        chunk_index += 1
    return chunks


def write_env_file(path: Path, values: Dict[str, str]) -> None:
    lines = ["# generated by tools/glm52_prompt_pipeline_input.py"]
    for key in sorted(values):
        lines.append(f"export {key}={shlex.quote(values[key])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(args: argparse.Namespace, token_ids: List[int], tokenizer_kind: str, bootstrap_token: int, model_dir: Path) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = "" if args.token_ids is not None else load_prompt(args)
    token_json = output_dir / "prompt_tokens.json"
    token_txt = output_dir / "prompt_tokens.txt"
    prefill_plan_json = output_dir / "prefill_plan.json"
    prefill_chunks_jsonl = output_dir / "prefill_chunks.jsonl"
    prompt_txt = output_dir / "prompt.txt"
    env_path = output_dir / "pipeline_env.sh"
    pipeline_output_dir = Path(args.pipeline_output_dir) if args.pipeline_output_dir is not None else output_dir / "pipeline_run"
    prefill_chunk_tokens = validate_prefill_chunk_tokens(args.prefill_chunk_tokens)
    prefill_chunks = build_prefill_chunks(token_ids, prefill_chunk_tokens)
    prefill_token_count = len(token_ids)
    pipeline_semantics = (
        "exact RING prompt sequence preflight through every prompt token"
        if args.run_full_prompt_sequence else
        "tail-window prompt prefill plus current-token decode for the local validation pipeline")
    prefill_context = (
        "every prompt token feeds the exact RING stage sequence preflight"
        if args.run_full_prompt_sequence else
        "last four prompt tokens feed the current validation context")
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "model_dir": str(model_dir),
        "tokenizer": tokenizer_kind,
        "prompt": prompt_text,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "bootstrap_token_id": bootstrap_token,
        "bootstrap_policy": args.bootstrap_token,
        "prefill_token_ids_file": str(token_txt),
        "prefill_plan_file": str(prefill_plan_json),
        "prefill_chunks_file": str(prefill_chunks_jsonl),
        "prefill_token_count": prefill_token_count,
        "prefill_chunk_count": len(prefill_chunks),
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "prefill_block_tokens": PREFILL_BLOCK_TOKENS,
        "tail_window_decode_input_token_id": bootstrap_token,
        "prefill_context": prefill_context,
        "pipeline_semantics": pipeline_semantics,
    }
    prefill_plan: Dict[str, Any] = {
        "schema": PREFILL_PLAN_SCHEMA,
        "model_dir": str(model_dir),
        "tokenizer": tokenizer_kind,
        "prompt_token_count": len(token_ids),
        "prefill_token_count": prefill_token_count,
        "first_decode_step": "after full prompt prefill",
        "tail_window_decode_input_token_id": bootstrap_token,
        "tail_window_decode_input_token_offset": len(token_ids) - 1,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "prefill_block_tokens": PREFILL_BLOCK_TOKENS,
        "prefill_chunk_count": len(prefill_chunks),
        "chunks": prefill_chunks,
        "production_target": "prompt tokens -> chunked prefill frames -> KV block table -> decode continuation",
    }
    token_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    token_txt.write_text("\n".join(str(token_id) for token_id in token_ids) + "\n", encoding="utf-8")
    prefill_plan_json.write_text(json.dumps(prefill_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prefill_chunks_jsonl.write_text(
        "".join(json.dumps(chunk, sort_keys=True) + "\n" for chunk in prefill_chunks),
        encoding="utf-8")
    prompt_txt.write_text(prompt_text, encoding="utf-8")
    write_env_file(env_path, {
        "GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID": str(bootstrap_token),
        "GLM52_LOCAL_PIPELINE_MAX_PREFILL_TOKENS": str(prefill_chunk_tokens),
        "GLM52_LOCAL_PIPELINE_OUTPUT_DIR": str(pipeline_output_dir),
        "GLM52_PREFILL_CHUNKS_FILE": str(prefill_chunks_jsonl),
        "GLM52_PREFILL_PLAN_FILE": str(prefill_plan_json),
        "GLM52_PREFILL_TOKEN_IDS_FILE": str(token_txt),
        "GLM52_PROMPT_BOOTSTRAP_MODE": "tail_window_prompt_prefill_validation_context",
        "GLM52_PROMPT_TOKEN_ARTIFACT": str(token_json),
        "GLM52_PROMPT_TOKEN_COUNT": str(len(token_ids)),
        "GLM52_PROMPT_PREFILL_CHUNK_COUNT": str(len(prefill_chunks)),
        "GLM52_PROMPT_PREFILL_TOKEN_COUNT": str(prefill_token_count),
    })
    return {
        "token_json": token_json,
        "token_txt": token_txt,
        "prefill_plan_json": prefill_plan_json,
        "prefill_chunks_jsonl": prefill_chunks_jsonl,
        "prompt_txt": prompt_txt,
        "env": env_path,
        "pipeline_output_dir": pipeline_output_dir,
    }


def run_pipeline(args: argparse.Namespace, artifacts: Dict[str, Path], bootstrap_token: int) -> int:
    root = Path(__file__).resolve().parents[1]
    script = Path(args.pipeline_script)
    blocked_env = (
        "GLM52_PREFILL_KV_FROM_EMBEDDINGS",
        "GLM52_WRITE_INPUT_EMBEDDING_HIDDEN_BF16",
        "GLM52_PIPELINE_INPUT_HIDDEN_BF16",
        "GLM52_PIPELINE_OUTPUT_HIDDEN_BF16",
        "GLM52_EXACT_RING_STAGE_SLICE",
        "GLM52_EXACT_RING_STAGE_SLICE_FINAL_TOKEN",
        "GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16",
        "GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN",
        "GLM52_CHAIN_DENSE_LAYERS",
        "GLM52_CHAIN_DENSE_TO_LAYER3_ROUTED_EXPERT_NVFP4_TOPK",
        "GLM52_LOAD_LAYER0_DENSE_BF16",
        "GLM52_LOAD_LAYER0_ATTENTION_BF16",
        "GLM52_LOAD_LAYER3_ROUTER_BF16",
        "GLM52_LOAD_LAYER3_SHARED_EXPERT_BF16",
        "GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4",
        "GLM52_LOAD_LAYER3_ROUTED_EXPERT_NVFP4_TOPK",
    )
    if not script.is_absolute():
        script = root / script
    if not script.is_file():
        raise PromptInputFailure(f"missing pipeline script: {script}")
    env = os.environ.copy()
    for name in blocked_env:
        env.pop(name, None)
    env["GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID"] = str(bootstrap_token)
    env["GLM52_LOCAL_PIPELINE_MAX_PREFILL_TOKENS"] = str(
        json.loads(artifacts["token_json"].read_text(encoding="utf-8"))["prefill_chunk_tokens"])
    env["GLM52_LOCAL_PIPELINE_OUTPUT_DIR"] = str(artifacts["pipeline_output_dir"])
    env["GLM52_PREFILL_CHUNKS_FILE"] = str(artifacts["prefill_chunks_jsonl"])
    env["GLM52_PREFILL_PLAN_FILE"] = str(artifacts["prefill_plan_json"])
    env["GLM52_PREFILL_TOKEN_IDS_FILE"] = str(artifacts["token_txt"])
    env["GLM52_PROMPT_TOKEN_ARTIFACT"] = str(artifacts["token_json"])
    env["GLM52_PROMPT_TOKEN_COUNT"] = str(len(json.loads(artifacts["token_json"].read_text(encoding="utf-8"))["token_ids"]))
    if args.run_full_prompt_sequence:
        env["GLM52_LOCAL_PIPELINE_PROMPT_SEQUENCE"] = "1"
    return subprocess.run([str(script)], cwd=str(root), env=env, check=False).returncode


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a GLM-5.2 prompt and emit SparkPipe prefill/decode input artifacts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file")
    source.add_argument("--token-ids", help="Comma or whitespace separated token ids; skips tokenizer dependency.")
    parser.add_argument("--model-dir", default=os.environ.get("GLM52_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default="build/glm52_prompt_pipeline_input")
    parser.add_argument("--pipeline-output-dir")
    parser.add_argument("--pipeline-script", default="tools/glm52_spark2_local_pipeline_gate.sh")
    parser.add_argument("--bootstrap-token", choices=("last", "first"), default="last")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=DEFAULT_PREFILL_CHUNK_TOKENS)
    parser.add_argument("--allow-tail-window-run", action="store_true", help="Allow --run-pipeline to collapse prompts longer than four tokens to the validation tail window.")
    parser.add_argument("--run-full-prompt-sequence", action="store_true", help="Run the exact RING prompt-sequence preflight instead of the four-token validation tail window.")
    parser.add_argument("--chat", action="store_true", help="Use the HF chat template when transformers is available.")
    parser.add_argument("--system-prompt")
    parser.add_argument("--no-add-generation-prompt", action="store_true")
    parser.add_argument("--run-pipeline", action="store_true")
    parser.set_defaults(tokenizer_json=os.environ.get("GLM52_TOKENIZER_JSON"))
    parser.add_argument("--tokenizer-json", dest="tokenizer_json", help="Path to Hugging Face tokenizer.json for the native C tokenizer.")
    parser.add_argument("--tokenizer-artifact", dest="tokenizer_json", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer-binary", default=os.environ.get("SPARKPIPE_TOKENIZER_BINARY", "build/sparkpipe_tokenize_prompt"))
    parser.add_argument("--allow-special-tokens", action="store_true", help="Let the C tokenizer recognize literal special/control tokens in prompt text.")
    parser.add_argument("--disable-special-tokens", action="store_true", help="Force literal special/control token strings to be encoded as normal text.")
    parser.add_argument("--c-tokenizer-add-prefix-space", action="store_true")
    parser.add_argument("--c-tokenizer-add-bos", action="store_true")
    parser.add_argument("--c-tokenizer-add-eos", action="store_true")
    parser.add_argument("--glm52-tokenizer-binary", default=os.environ.get("SPARKPIPE_GLM52_TOKENIZER_BINARY", "build/sparkpipe_glm52_tokenize"))
    parser.add_argument("--c-tokenizer-disable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("Max", "High"))
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    try:
        args = parse_args(argv)
        model_dir = Path(args.model_dir)
        token_ids, tokenizer_kind = tokenize_prompt(args, model_dir)
        run_requested = args.run_pipeline or args.run_full_prompt_sequence
        if args.run_pipeline and not args.run_full_prompt_sequence and len(token_ids) < TAIL_WINDOW_TOKEN_COUNT:
            raise PromptInputFailure("pipeline prefill mode needs at least four prompt tokens")
        if args.run_pipeline and not args.run_full_prompt_sequence and len(token_ids) > TAIL_WINDOW_TOKEN_COUNT and not args.allow_tail_window_run:
            raise PromptInputFailure("refusing to run long prompt through four-token tail-window pipeline; inspect prefill_plan.json or pass --allow-tail-window-run")
        if args.run_full_prompt_sequence and len(token_ids) > 128:
            raise PromptInputFailure("exact RING prompt-sequence preflight currently supports at most 128 tokens")
        bootstrap_token = choose_bootstrap_token(token_ids, args.bootstrap_token)
        artifacts = write_artifacts(args, token_ids, tokenizer_kind, bootstrap_token, model_dir)
        print(f"glm52_prompt_tokenizer={tokenizer_kind}")
        print(f"glm52_prompt_token_count={len(token_ids)}")
        print(f"glm52_prompt_bootstrap_token={bootstrap_token}")
        print(f"glm52_prompt_tokens_json={artifacts['token_json']}")
        print(f"glm52_prompt_pipeline_env={artifacts['env']}")
        print(f"glm52_prefill_token_ids_file={artifacts['token_txt']}")
        print(f"glm52_prefill_plan_file={artifacts['prefill_plan_json']}")
        print(f"glm52_prefill_chunks_file={artifacts['prefill_chunks_jsonl']}")
        print(f"glm52_prefill_token_count={len(token_ids)}")
        print(f"glm52_prefill_chunk_count={len(build_prefill_chunks(token_ids, validate_prefill_chunk_tokens(args.prefill_chunk_tokens)))}")
        if args.run_full_prompt_sequence:
            print("glm52_prompt_pipeline_semantics=exact_ring_prompt_sequence_preflight")
        else:
            print("glm52_prompt_pipeline_semantics=tail_window_prompt_prefill_validation_context")
        if run_requested:
            return run_pipeline(args, artifacts, bootstrap_token)
        return 0
    except PromptInputFailure as exc:
        print(f"glm52_prompt_pipeline_input_error={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
