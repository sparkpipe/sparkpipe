#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()


def run_prompt_tool(root: Path, output_dir: Path, token_ids: str, extra_args=None, check: bool = True, env=None) -> subprocess.CompletedProcess:
    command = [
        "python3",
        str(root / "tools" / "glm52_prompt_pipeline_input.py"),
        "--token-ids",
        token_ids,
        "--output-dir",
        str(output_dir),
        "--pipeline-output-dir",
        str(output_dir / "pipeline"),
    ]
    if extra_args is not None:
        command.extend(extra_args)
    return subprocess.run(command, cwd=str(root), text=True, capture_output=True, check=check, env=env)


def test_tail_window_artifacts(root: Path) -> None:
    output_dir = root / "build" / "test_glm52_prompt_pipeline_input"
    clean_output_dir(output_dir)
    completed = run_prompt_tool(root, output_dir, "101,202,303,404")
    assert "glm52_prompt_bootstrap_token=404" in completed.stdout
    assert "glm52_prefill_plan_file=" in completed.stdout
    assert "glm52_prefill_chunks_file=" in completed.stdout
    assert "glm52_prompt_pipeline_semantics=tail_window_prompt_prefill_validation_context" in completed.stdout
    token_json = output_dir / "prompt_tokens.json"
    token_txt = output_dir / "prompt_tokens.txt"
    prefill_plan_json = output_dir / "prefill_plan.json"
    prefill_chunks_jsonl = output_dir / "prefill_chunks.jsonl"
    env_file = output_dir / "pipeline_env.sh"
    payload = json.loads(token_json.read_text(encoding="utf-8"))
    prefill_plan = json.loads(prefill_plan_json.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in prefill_chunks_jsonl.read_text(encoding="utf-8").splitlines()]
    assert payload["schema"] == "sparkpipe.glm52.prompt_pipeline_input.v1"
    assert payload["token_ids"] == [101, 202, 303, 404]
    assert payload["bootstrap_token_id"] == 404
    assert payload["prefill_token_ids_file"] == str(token_txt)
    assert payload["prefill_plan_file"] == str(prefill_plan_json)
    assert payload["prefill_chunks_file"] == str(prefill_chunks_jsonl)
    assert payload["prefill_token_count"] == 4
    assert payload["prefill_chunk_count"] == 1
    assert payload["tail_window_decode_input_token_id"] == 404
    assert payload["pipeline_semantics"] == "tail-window prompt prefill plus current-token decode for the local validation pipeline"
    assert prefill_plan["schema"] == "sparkpipe.glm52.prompt_prefill_plan.v1"
    assert prefill_plan["first_decode_step"] == "after full prompt prefill"
    assert prefill_plan["tail_window_decode_input_token_id"] == 404
    assert prefill_plan["tail_window_decode_input_token_offset"] == 3
    assert prefill_plan["prefill_token_count"] == 4
    assert prefill_plan["prefill_chunk_count"] == 1
    assert chunks[0]["token_offset"] == 0
    assert chunks[0]["token_count"] == 4
    assert chunks[0]["token_ids"] == [101, 202, 303, 404]
    assert chunks[0]["final_prefill_chunk"] is True
    env_text = env_file.read_text(encoding="utf-8")
    assert "GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID=404" in env_text
    assert "GLM52_LOCAL_PIPELINE_MAX_PREFILL_TOKENS=256" in env_text
    assert "GLM52_PREFILL_TOKEN_IDS_FILE=" in env_text
    assert "GLM52_PREFILL_PLAN_FILE=" in env_text
    assert "GLM52_PREFILL_CHUNKS_FILE=" in env_text
    assert "GLM52_PROMPT_PREFILL_TOKEN_COUNT=4" in env_text
    assert "GLM52_PROMPT_PREFILL_CHUNK_COUNT=1" in env_text


def test_long_prompt_prefill_chunks(root: Path) -> None:
    output_dir = root / "build" / "test_glm52_prompt_pipeline_input_long"
    token_ids = ",".join(str(1000 + index) for index in range(41))
    clean_output_dir(output_dir)
    completed = run_prompt_tool(
        root,
        output_dir,
        token_ids,
        ["--prefill-chunk-tokens", "16"])
    assert "glm52_prompt_token_count=41" in completed.stdout
    prefill_plan_json = output_dir / "prefill_plan.json"
    prefill_chunks_jsonl = output_dir / "prefill_chunks.jsonl"
    prefill_plan = json.loads(prefill_plan_json.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in prefill_chunks_jsonl.read_text(encoding="utf-8").splitlines()]
    assert prefill_plan["prefill_token_count"] == 41
    assert prefill_plan["prefill_chunk_tokens"] == 16
    assert prefill_plan["prefill_chunk_count"] == 3
    assert prefill_plan["tail_window_decode_input_token_id"] == 1040
    assert [chunk["token_count"] for chunk in chunks] == [16, 16, 9]
    assert [chunk["token_offset"] for chunk in chunks] == [0, 16, 32]
    assert chunks[0]["token_ids"][0] == 1000
    assert chunks[1]["token_ids"][0] == 1016
    assert chunks[2]["token_ids"] == list(range(1032, 1041))
    assert chunks[0]["final_prefill_chunk"] is False
    assert chunks[2]["final_prefill_chunk"] is True


def test_c_prefill_dryrun_when_built(root: Path) -> None:
    output_dir = root / "build" / "test_glm52_prompt_pipeline_input_dryrun"
    dryrun_binary = root / "build" / "sparkpipe_glm52_prefill_dryrun"
    token_ids = ",".join(str(2000 + index) for index in range(21))
    clean_output_dir(output_dir)
    run_prompt_tool(
        root,
        output_dir,
        token_ids,
        ["--prefill-chunk-tokens", "16"])
    if not dryrun_binary.exists():
        return
    completed = subprocess.run(
        [
            str(dryrun_binary),
            "--tokens",
            str(output_dir / "prompt_tokens.txt"),
            "--max-prefill-tokens",
            "16",
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=True)
    lines = completed.stdout.splitlines()
    assert lines[0] == "step\tkind\ttoken_offset\ttoken_count\tremaining\tcommit_after\tprefill_blocks\tkv_blocks\tflags"
    assert lines[1].startswith("0\tprefill\t0\t16\t5\t16\t1\t1\t")
    assert lines[2].startswith("1\tprefill\t16\t5\t0\t21\t1\t2\t")
    assert lines[3].startswith("2\tdecode_ready\t0\t0\t0\t0\t0\t0\t")



def write_tiny_byte_bpe_tokenizer(path: Path) -> None:
    path.write_text(json.dumps({
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "byte_fallback": False,
            "vocab": {
                "a": 1, "b": 2, "c": 3, "ab": 4, "abc": 5,
                "<unk>": 6, "x": 9, "y": 10, "z": 11,
                "xy": 12, "xyz": 13,
            },
            "merges": ["a b", "ab c", "x y", "xy z"],
        },
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "added_tokens": [],
    }), encoding="utf-8")


def test_c_prefill_dryrun_tokenizes_prompt_text_when_built(root: Path) -> None:
    output_dir = root / "build" / "test_glm52_prompt_pipeline_input_c_tokenizer"
    dryrun_binary = root / "build" / "sparkpipe_glm52_prefill_dryrun"
    clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dryrun_binary.exists():
        return
    tokenizer_json = output_dir / "tokenizer.json"
    written_tokens = output_dir / "prompt_tokens.txt"
    write_tiny_byte_bpe_tokenizer(tokenizer_json)
    completed = subprocess.run(
        [
            str(dryrun_binary),
            "--tokenizer-json",
            str(tokenizer_json),
            "--prompt",
            "abcxyz",
            "--max-prefill-tokens",
            "16",
            "--write-tokens",
            str(written_tokens),
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=True)
    lines = completed.stdout.splitlines()
    assert lines[0] == "step\tkind\ttoken_offset\ttoken_count\tremaining\tcommit_after\tprefill_blocks\tkv_blocks\tflags"
    assert lines[1].startswith("0\tprefill\t0\t2\t0\t2\t1\t1\t")
    assert lines[2].startswith("1\tdecode_ready\t0\t0\t0\t0\t0\t0\t")
    assert written_tokens.read_text(encoding="utf-8").splitlines() == ["5", "13"]

def test_local_pipeline_gate_prefill_only(root: Path) -> None:
    output_dir = root / "build" / "test_glm52_prompt_pipeline_gate_prefill_only"
    prompt_dir = output_dir / "prompt"
    pipeline_dir = output_dir / "pipeline"
    fake_model_dir = output_dir / "model"
    token_ids = ",".join(str(3000 + index) for index in range(21))
    clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_model_dir.mkdir(parents=True, exist_ok=True)
    run_prompt_tool(
        root,
        prompt_dir,
        token_ids,
        ["--prefill-chunk-tokens", "16"])
    env = os.environ.copy()
    env.update({
        "NVCC": "true",
        "GLM52_MODEL_DIR": str(fake_model_dir),
        "GLM52_PREFILL_TOKEN_IDS_FILE": str(prompt_dir / "prompt_tokens.txt"),
        "GLM52_LOCAL_PIPELINE_OUTPUT_DIR": str(pipeline_dir),
        "GLM52_LOCAL_PIPELINE_PREFILL_ONLY": "1",
        "GLM52_LOCAL_PIPELINE_MAX_PREFILL_TOKENS": "16",
    })
    completed = subprocess.run(
        ["bash", str(root / "tools" / "glm52_spark2_local_pipeline_gate.sh")],
        cwd=str(root),
        text=True,
        capture_output=True,
        env=env,
        check=True)
    schedule_path = pipeline_dir / "prefill_schedule.tsv"
    assert "glm52_local_pipeline_prefill_only=1" in completed.stdout
    assert "glm52_local_pipeline_prefill_steps=2" in completed.stdout
    assert "glm52_local_pipeline_prefill_tokens=21" in completed.stdout
    schedule_lines = schedule_path.read_text(encoding="utf-8").splitlines()
    assert schedule_lines[1].startswith("0\tprefill\t0\t16\t5\t16\t1\t1\t")
    assert schedule_lines[2].startswith("1\tprefill\t16\t5\t0\t21\t1\t2\t")
    assert schedule_lines[3].startswith("2\tdecode_ready\t0\t0\t0\t0\t0\t0\t")


def test_long_prompt_pipeline_refuses_tail_collapse(root: Path) -> None:
    output_dir = root / "build" / "test_glm52_prompt_pipeline_input_refuse"
    clean_output_dir(output_dir)
    completed = run_prompt_tool(
        root,
        output_dir,
        "101,202,303,404,505",
        ["--run-pipeline"],
        check=False)
    assert completed.returncode == 2
    assert "refusing to run long prompt through four-token tail-window pipeline" in completed.stderr


def test_pipeline_run_scrubs_validation_env(root: Path) -> None:
    output_dir = root / "build" / "test_glm52_prompt_pipeline_input_env_scrub"
    fake_pipeline = output_dir / "capture_pipeline.py"
    capture_path = output_dir / "captured_env.txt"
    inherited_env = os.environ.copy()
    clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_pipeline.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "names = [\n"
        "    'GLM52_PREFILL_KV_FROM_EMBEDDINGS',\n"
        "    'GLM52_WRITE_INPUT_EMBEDDING_HIDDEN_BF16',\n"
        "    'GLM52_EXACT_RING_STAGE_SLICE',\n"
        "    'GLM52_PIPELINE_OUTPUT_HIDDEN_BF16',\n"
        "    'GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID',\n"
        "    'GLM52_LOCAL_PIPELINE_PROMPT_SEQUENCE',\n"
        "    'GLM52_PREFILL_TOKEN_IDS_FILE',\n"
        "]\n"
        "Path(os.environ['SPARK_TEST_ENV_CAPTURE']).write_text('\\n'.join(f'{name}={os.environ.get(name, \"\")}' for name in names) + '\\n', encoding='utf-8')\n",
        encoding="utf-8")
    fake_pipeline.chmod(0o755)
    inherited_env.update({
        "SPARK_TEST_ENV_CAPTURE": str(capture_path),
        "GLM52_PREFILL_KV_FROM_EMBEDDINGS": "1",
        "GLM52_WRITE_INPUT_EMBEDDING_HIDDEN_BF16": "1",
        "GLM52_EXACT_RING_STAGE_SLICE": "1",
        "GLM52_PIPELINE_OUTPUT_HIDDEN_BF16": "/tmp/poison.bf16",
    })
    completed = run_prompt_tool(
        root,
        output_dir,
        "101,202,303,404,505",
        ["--run-full-prompt-sequence", "--pipeline-script", str(fake_pipeline)],
        env=inherited_env)
    assert completed.returncode == 0
    assert "glm52_prompt_pipeline_semantics=exact_ring_prompt_sequence_preflight" in completed.stdout
    captured_env = capture_path.read_text(encoding="utf-8")
    assert "GLM52_PREFILL_KV_FROM_EMBEDDINGS=\n" in captured_env
    assert "GLM52_WRITE_INPUT_EMBEDDING_HIDDEN_BF16=\n" in captured_env
    assert "GLM52_EXACT_RING_STAGE_SLICE=\n" in captured_env
    assert "GLM52_PIPELINE_OUTPUT_HIDDEN_BF16=\n" in captured_env
    assert "GLM52_LOCAL_PIPELINE_INPUT_TOKEN_ID=505\n" in captured_env
    assert "GLM52_LOCAL_PIPELINE_PROMPT_SEQUENCE=1\n" in captured_env
    assert "GLM52_PREFILL_TOKEN_IDS_FILE=" in captured_env


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    test_tail_window_artifacts(root)
    test_long_prompt_prefill_chunks(root)
    test_c_prefill_dryrun_when_built(root)
    test_c_prefill_dryrun_tokenizes_prompt_text_when_built(root)
    test_local_pipeline_gate_prefill_only(root)
    test_long_prompt_pipeline_refuses_tail_collapse(root)
    test_pipeline_run_scrubs_validation_env(root)


if __name__ == "__main__":
    main()
