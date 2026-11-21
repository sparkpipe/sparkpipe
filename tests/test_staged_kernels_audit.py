#!/usr/bin/env python3
"""tools/staged_kernels_audit.py requires every staged module that embeds device code to carry its kernels beside it under the
name it is loaded by: a root where each such module has complete cubin and manifest pairs passes, and a missing directory, kernels
left under the module's build name, an unpaired file, an empty directory or a stray file each fail and name the module, and so does
a host library such as the hidden transport once it embeds device code. ELF files without device code, files that are not ELF, and
the cubins inside a kernels directory are not modules."""
import contextlib
import importlib.util
import io
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("staged_kernels_audit", ROOT / "tools" / "staged_kernels_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def elf(path, sections):
    names = b"\0" + b"".join(name.encode() + b"\0" for name in sections + [".shstrtab"])
    offsets, cursor = [], 1
    for name in sections + [".shstrtab"]:
        offsets.append(cursor)
        cursor += len(name) + 1
    header_size, entry_size = 64, 64
    names_offset = header_size
    table_offset = (names_offset + len(names) + 7) // 8 * 8
    count = len(sections) + 2
    header = bytearray(header_size)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", header, 16, 3, 62, 1)
    struct.pack_into("<Q", header, 0x28, table_offset)
    struct.pack_into("<HHHHHH", header, 0x34, header_size, 0, 0, entry_size, count, count - 1)
    table = bytearray(entry_size * count)
    for index, offset in enumerate(offsets, start=1):
        struct.pack_into("<IIQQQQIIQQ", table, index * entry_size, offset, 1, 0, 0, names_offset if index == count - 1 else 0,
                         len(names) if index == count - 1 else 0, 0, 0, 1, 0)
    struct.pack_into("<IIQQQQIIQQ", table, (count - 1) * entry_size, offsets[-1], 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
    body = bytes(header) + names + b"\0" * (table_offset - names_offset - len(names)) + bytes(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def kernels(path, stems, extra=()):
    path.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        elf(path / (stem + ".cubin"), [".text", ".nv.info"])
        (path / (stem + ".manifest")).write_text("nvcc 13.0\n")
    for name in extra:
        (path / name).write_text("x")


def staged(root):
    elf(root / "bin" / "sparkpipe_model_residentd", [".text", ".data"])
    elf(root / "bin" / "weightd_warm", [".text"])
    elf(root / "lib" / "libmodel_serving_adapter.so", [".text", ".nv_fatbin", ".data"])
    kernels(root / "lib" / "libmodel_serving_adapter.so.kernels", ["stage_runner", "model_stage"])
    elf(root / "lib" / "runtime_libs" / "libsparkpipe_device_cuda.so", [".text"])
    elf(root / "lib" / "hidden_transport.so", [".text", ".data"])
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "adapter.json").write_text("{}")


def main():
    failures = []
    with tempfile.TemporaryDirectory(prefix="staged-kernels-") as scratch:
        base = Path(scratch)

        def case(name, change, wanted):
            root = base / name
            staged(root)
            change(root)
            modules, findings = audit.audit(root)
            text = "\n".join(findings)
            if wanted is None:
                if findings or [str(m) for m in modules] != ["lib/libmodel_serving_adapter.so"]:
                    failures.append("%s: expected a pass over the one module with device code, got %s %s" % (name, modules, findings))
            elif not all(fragment in text for fragment in wanted):
                failures.append("%s: expected findings naming %s, got %s" % (name, wanted, findings))

        case("complete", lambda root: None, None)

        adapter_kernels = Path("lib") / "libmodel_serving_adapter.so.kernels"

        def missing(root):
            for child in (root / adapter_kernels).iterdir():
                child.unlink()
            (root / adapter_kernels).rmdir()
        case("missing", missing, ["lib/libmodel_serving_adapter.so embeds device code but has no libmodel_serving_adapter.so.kernels/"])

        def build_name(root):
            missing(root)
            kernels(root / "lib" / "libmodel_tp16_serving_adapter.so.kernels", ["stage_runner", "model_stage"])
        case("build name", build_name, ["lib/libmodel_serving_adapter.so embeds device code but has no libmodel_serving_adapter.so.kernels/",
                                        "lib/libmodel_tp16_serving_adapter.so.kernels/ has no module libmodel_tp16_serving_adapter.so"])
        case("unpaired cubin", lambda root: (root / "lib" / "libmodel_serving_adapter.so.kernels" / "model_stage.manifest").unlink(),
             ["lib/libmodel_serving_adapter.so.kernels/model_stage.cubin has no manifest"])
        case("unpaired manifest", lambda root: (root / "lib" / "libmodel_serving_adapter.so.kernels" / "stage_runner.cubin").unlink(),
             ["lib/libmodel_serving_adapter.so.kernels/stage_runner.manifest has no cubin"])

        def empty(root):
            for child in (root / adapter_kernels).iterdir():
                child.unlink()
        case("empty", empty, ["libmodel_serving_adapter.so.kernels/ beside lib/libmodel_serving_adapter.so holds no cubin"])
        case("stray", lambda root: (root / adapter_kernels / "model_stage.ptx").write_text("x"),
             ["lib/libmodel_serving_adapter.so.kernels/model_stage.ptx is neither a cubin nor a manifest"])
        case("device transport", lambda root: elf(root / "lib" / "hidden_transport.so", [".text", ".nv_fatbin"]),
             ["lib/hidden_transport.so embeds device code but has no hidden_transport.so.kernels/"])

        def relocatable(root):
            elf(root / "lib" / "libextra.so", [".text", "__nv_relfatbin"])
        case("relocatable fat binary", relocatable, ["lib/libextra.so embeds device code but has no libextra.so.kernels/"])

        root = base / "command line"
        staged(root)
        if audit.main([str(root), "--quiet"]) != 0:
            failures.append("the command line fails a complete root")
        missing(root)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = audit.main([str(root), "--quiet"])
        if code != 1 or "FAIL lib/libmodel_serving_adapter.so embeds device code" not in printed.getvalue():
            failures.append("the command line passes a root whose adapter has no kernels, or does not name it: %s" % printed.getvalue())
    for failure in failures:
        print("FAIL %s" % failure)
    if failures:
        return 1
    print("PASS staged kernels audit: every staged module with device code needs complete kernel pairs beside it under its staged name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
