#!/usr/bin/env python3
"""The release builder fails closed on adapter undefined symbols.

The firmware adapter .so must dlopen in any host process (the API loads it
for the descriptor), so module_build_release.sh runs the same ldd -r
undefined-symbol gate over lib/model_serving_adapter.so as over the driver
(the minimax lane-10 connect failure: cudaFree left undefined because the
adapter linked runtime statics without -lcudart).
"""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "module_build_release.sh"

FAKE_GIT = '#!/bin/sh\ncase "$1" in rev-parse) printf "a%.0s" $(seq 1 40); printf "\\n";; esac\nexit 0\n'
FAKE_FLOCK = "#!/bin/sh\nexit 0\n"
FAKE_NVCC = "#!/bin/sh\nexit 0\n"
FAKE_MAKE = """#!/bin/sh
case " $* " in
  *" adapter "*)
    mkdir -p build/modules/minimax_resident_decode_stage/bf16
    : > build/modules/minimax_resident_decode_stage/bf16/libminimax_serving_adapter_bf16.so
    ;;
  *" publish "*) mkdir -p build/module_library/active ;;
  *)
    mkdir -p build
    for tool in weightd_warm weightd_receipt sparkpipe_model_residentd sparkpipe_weightd sparkpipe_model_api sparkpipe_model_batch sparkpipe_module_publish; do
      : > "build/$tool"
      chmod +x "build/$tool"
    done
    : > build/libhidden_transport_spark_host_rdma_verbs.so
    cp "$(dirname "$0")/sparkpipe_model_compile" "$(dirname "$0")/sparkpipe_driver_inspect" build/
    ;;
esac
exit 0
"""
FAKE_COMPILE = """#!/bin/sh
output=""
previous=""
for argument in "$@"; do
  if [ "$previous" = "--output" ]; then output="$argument"; fi
  previous="$argument"
done
mkdir -p "$output/stages/stage_000" "$output"
: > "$output/stages/stage_000/model_driver.so"
: > "$output/model_package.json"
exit 0
"""
FAKE_INSPECT = "#!/bin/sh\nexit 0\n"
FAKE_SHA256SUM = '#!/bin/sh\nfor argument in "$@"; do printf "%s  %s\\n" "$(printf "b%.0s" $(seq 1 64))" "$argument"; done\n'


def fake_ldd(body):
    return '#!/bin/sh\ncase "$2" in *model_serving_adapter.so) printf "%s\\n" ' \
        '"' + body + '";; esac\nexit 0\n'


def build_fixture(directory, ldd_body):
    bin_directory = directory / "bin"
    bin_directory.mkdir()
    (bin_directory / "git").write_text(FAKE_GIT)
    (bin_directory / "flock").write_text(FAKE_FLOCK)
    (bin_directory / "nvcc").write_text(FAKE_NVCC)
    (bin_directory / "cc").write_text(FAKE_NVCC)
    (bin_directory / "make").write_text(FAKE_MAKE)
    (bin_directory / "sha256sum").write_text(FAKE_SHA256SUM)
    (bin_directory / "ldd").write_text(fake_ldd(ldd_body))
    for name, body in (("sparkpipe_model_compile", FAKE_COMPILE),
                       ("sparkpipe_driver_inspect", FAKE_INSPECT)):
        path = bin_directory / name
        path.write_text(body)
    for path in bin_directory.iterdir():
        path.chmod(0o755)
    (directory / "tools").mkdir()
    (directory / "tools" / "module_build_release.sh").write_bytes(SCRIPT.read_bytes())
    (directory / "model_contracts").mkdir()
    contract = directory / "model_contracts" / "minimax_bf16.json"
    contract.write_text("{}\n")
    firmware = directory / "firmware.json"
    firmware.write_text('{"stages": [{"target": "cuda.sm121.minimax.resident_decode_stage.bf16"}]}\n')
    return contract, firmware


def run(directory, contract, firmware):
    env = dict(os.environ)
    env.update(
        PATH=str(directory / "bin") + ":" + env["PATH"],
        SPARK_QUEUE_ID="test-adapter-gate",
        FIRMWARE_JSON=str(firmware),
    )
    return subprocess.run(
        ["bash", str(directory / "tools" / "module_build_release.sh"),
         "minimax_resident_decode_stage", "bf16", "fixture-fw",
         "fixture-revision", str(contract)],
        env=env, capture_output=True)


def main():
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        contract, firmware = build_fixture(
            directory, "undefined symbol: cudaFree\\t(fixture adapter)")
        result = run(directory, contract, firmware)
        assert result.returncode == 1, result
        assert b"undefined symbol: cudaFree" in result.stdout, result
        assert b"BUILD-PASS" not in result.stdout, result
        assert not (directory / "build" / "fixture-fw.tar.gz").exists()
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        contract, firmware = build_fixture(directory, "")
        result = run(directory, contract, firmware)
        assert result.returncode == 0, result
        assert b"BUILD-PASS" in result.stdout, result
        assert (directory / "build" / "fixture-fw.tar.gz").exists()
    print("PASS module build release: adapter undefined-symbol gate")


if __name__ == "__main__":
    main()
