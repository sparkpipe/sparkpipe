#!/usr/bin/env python3

from pathlib import Path
import subprocess

CODECS = ("int6", "int7", "int8", "fp8", "nvfp4", "mxfp4")
REVISION = "test-model-revision"
CONTRACT_SHA256 = "0" * 64


def run_make(root: Path, codec: str | None) -> subprocess.CompletedProcess[str]:
    command = [
        "make",
        "-C",
        str(root / "modules/glm52_resident_decode_stage"),
        "contract",
    ]
    if codec is not None:
        command.extend(
            (
                f"EXPERT_CODEC={codec}",
                f"MODEL_REVISION={REVISION}",
                f"CONTRACT_SHA256={CONTRACT_SHA256}",
            )
        )
    return subprocess.run(command, capture_output=True, text=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = run_make(root, None)
    assert missing.returncode != 0
    assert "EXPERT_CODEC is required" in missing.stderr

    unsupported = run_make(root, "bf16")
    assert unsupported.returncode != 0
    assert "unsupported EXPERT_CODEC 'bf16'" in unsupported.stderr

    for codec in CODECS:
        result = run_make(root, codec)
        assert result.returncode == 0, result.stderr
    print(f"PASS GLM module host contract for {len(CODECS)} exact codecs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
