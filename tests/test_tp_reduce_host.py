from pathlib import Path
import subprocess
import tempfile

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory)/'tp_reduce'
        subprocess.run([host_cuda_cxx(), '-std=c++17', '-O2', '-I'+str(ROOT), '-I'+str(ROOT/'tests/host_cuda'), '-x', 'c++', str(ROOT/'tests/host_cuda/tp_reduce_host.cu'), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == '__main__':
    main()
