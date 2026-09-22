import argparse
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Compile or explicitly run the isolated CUDA stream-memory-wait qualification probe. No production support is implied.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--compile", action="store_true", help="Compile only; creates no CUDA context")
    action.add_argument("--run", action="store_true", help="Create a CUDA context and measure graph parameter update cost")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gpu-waits", action="store_true", help="With --run, launch three local readiness/cancel/recovery checks")
    mode.add_argument("--rdma-receive", action="store_true", help="With --run, receive NIC-written payloads and readiness on a CUDA-mapped host MR")
    mode.add_argument("--rdma-send", action="store_true", help="With --run, send payloads and readiness over an isolated RC QP; no CUDA context")
    parser.add_argument("--memfd", action="store_true", help="With --run, use explicit MAP_SHARED memfd + portable/mapped CUDA registration; unavailable to sender")
    parser.add_argument("--ib-device", help="Exact RDMA device, required for RDMA modes")
    parser.add_argument("--ib-port", type=int, help="Exact RDMA device port, required for RDMA modes")
    parser.add_argument("--gid-index", type=int, help="Exact GID index, required for RDMA modes")
    parser.add_argument("--address", help="Receiver bind IPv4 or sender peer IPv4, required for RDMA modes")
    parser.add_argument("--tcp-port", type=int, help="Unique handshake TCP port, required for RDMA modes")
    parser.add_argument("--iterations", type=int, help="RDMA repetitions of all three cases, 1..128, default 8")
    parser.add_argument("--cuda-root", type=pathlib.Path, help="CUDA toolkit root, required for --compile")
    parser.add_argument("--arch", help="Explicit GPU compile target, for example sm_121a")
    root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=pathlib.Path, default=root / "build/qualification/tp_stream_memop_probe")
    args = parser.parse_args()
    if args.memfd and (not args.run or args.rdma_send):
        parser.error("--memfd requires a CUDA --run mode and cannot be used by the RDMA sender")
    rdma = args.rdma_receive or args.rdma_send
    rdma_fields = (args.ib_device, args.ib_port, args.gid_index, args.address, args.tcp_port)
    if (args.gpu_waits or rdma) and not args.run:
        parser.error("GPU and RDMA modes require --run")
    if rdma and (any(value is None for value in rdma_fields) or
                 not 1 <= args.ib_port <= 255 or not 0 <= args.gid_index <= 255 or
                 not 1 <= args.tcp_port <= 65535 or
                 (args.iterations is not None and not 1 <= args.iterations <= 128)):
        parser.error("RDMA mode requires explicit device/port/GID/address/TCP port and 1..128 iterations")
    if not rdma and (any(value is not None for value in rdma_fields) or args.iterations is not None):
        parser.error("RDMA parameters require --rdma-receive or --rdma-send")
    if not args.compile and not args.run:
        parser.print_help()
        return 0
    output = args.output.resolve()
    if args.compile:
        if args.cuda_root is None or not args.arch:
            parser.error("--compile requires --cuda-root and --arch")
        compiler = args.cuda_root.resolve() / "bin/nvcc"
        if not compiler.is_file():
            parser.error(f"CUDA compiler does not exist: {compiler}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(compiler), "-std=c++17", f"-arch={args.arch}", "-Xcompiler", "-pthread",
                   str(root / "tools/tp_stream_memop_probe.cu"), "-lcuda", "-libverbs", "-o", str(output)]
    else:
        if args.cuda_root is not None or args.arch:
            parser.error("--cuda-root and --arch apply only to --compile")
        if not output.is_file():
            parser.error(f"compile the probe first; missing binary: {output}")
        command = [str(output), "--run"]
        if args.memfd:
            command.append("--memfd")
        if args.gpu_waits:
            command.append("--gpu-waits")
        if rdma:
            command += ["--rdma-receive" if args.rdma_receive else "--rdma-send",
                        "--ib-device", args.ib_device, "--ib-port", str(args.ib_port),
                        "--gid-index", str(args.gid_index), "--address", args.address,
                        "--tcp-port", str(args.tcp_port), "--iterations", str(args.iterations or 8)]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
