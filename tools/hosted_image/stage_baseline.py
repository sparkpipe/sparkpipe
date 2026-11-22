#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tools" / "devcycle" / "ds4_spark_brickproof.py"
GUARD = ROOT / "tools" / "devcycle" / "spark_dma_guard.py"
IMAGE_NODE_ID = "sparkff"
OFFLINE_VERBS = ("enable", "disable", "mask", "unmask", "set-default")
RUNTIME_VERBS = ("daemon-reload", "restart", "start", "stop", "try-reload-or-restart", "reset-failed", "show",
                 "list-units", "set-property", "is-active", "is-enabled")
LIVE_COMMANDS = ("sysctl", "udevadm", "modprobe", "loginctl", "hostnamectl", "update-grub")
NODE_STEPS = ("install_packages", "install_node_identity", "install_node_layout", "install_sshd_hardening",
              "install_efi_boot_order", "rebuild_initramfs", "install_ceph_policy", "heal_time_sync")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_baseline():
    return load("ds4_spark_brickproof", BASELINE)


def load_guard():
    return load("spark_dma_guard", GUARD)


def classify(module, argv):
    argv = [str(item) for item in argv]
    if argv[0] == "systemctl" and len(argv) > 1:
        if argv[1] in OFFLINE_VERBS:
            return "image", ["systemctl", "--root=/", argv[1], *[item for item in argv[2:] if item != "--now"]]
        if argv[1] in RUNTIME_VERBS:
            return "live", argv
    if argv[0] == "/usr/sbin/sshd" and "-t" in argv:
        return "image", argv
    if argv[0] == module.DMA_GUARD_RUNTIME and argv[1:3] == ["install", "--notify"] and len(argv) == 4:
        return "guard", argv
    if argv[0] in LIVE_COMMANDS or argv[0] == module.DMA_GUARD_RUNTIME:
        return "live", argv
    raise module.BrickproofError(f"the image stager cannot place this baseline command: {argv}")


class Recorder:
    def __init__(self, module, guard):
        self.module = module
        self.guard = guard
        self.applied = []
        self.live = []
        self.node = []

    def run(self, argv, input_bytes=None, timeout=120, check=True, env=None):
        where, command = classify(self.module, argv)
        if where == "live":
            self.live.append(command)
            return subprocess.CompletedProcess(command, 0, b"", b"")
        if where == "guard":
            return self.install_guard(command[3], timeout)
        result = subprocess.run(command, input=input_bytes, capture_output=True, timeout=timeout, env=env,
                                stdin=None if input_bytes is not None else subprocess.DEVNULL)
        self.applied.append({"command": command, "returncode": result.returncode})
        if check and result.returncode != 0:
            raise self.module.BrickproofError(
                f"{' '.join(command)} failed: {(result.stderr or result.stdout).decode('utf-8', 'replace').strip()}")
        return result

    def install_guard(self, target, timeout):
        host, port = self.guard.parse_target(target)
        service, timer = self.guard.unit_text(f"{host}:{port}")
        self.module.atomic_write(self.guard.SERVICE_PATH, service, 0o644)
        self.module.atomic_write(self.guard.TIMER_PATH, timer, 0o644)
        return self.module.run(["systemctl", "enable", self.guard.TIMER_PATH.name])

    def deferred(self, name):
        def record(*arguments, **keywords):
            self.node.append(name)
            return [] if name == "install_ceph_policy" else False
        return record

    def install(self):
        self.module.run = self.run
        self.module.has_configured_remote_storage = lambda: []
        for name in NODE_STEPS:
            if not callable(getattr(self.module, name, None)):
                raise self.module.BrickproofError(f"the baseline has no step {name}")
            setattr(self.module, name, self.deferred(name))

    def report(self, source_commit):
        return {"source_commit": source_commit, "node_id": IMAGE_NODE_ID, "applied": self.applied,
                "deferred_to_node": sorted(set(self.node)), "deferred_live": self.live}


def stage(module, payload, rootfs):
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read)
        status = 0
        try:
            recorder = Recorder(module, load_guard())
            recorder.install()
            os.chroot(rootfs)
            os.chdir("/")
            result = module.remote_apply(payload)
            if result.get("profile") != "staged":
                raise module.BrickproofError(f"baseline applied profile {result.get('profile')}, expected staged")
            document = recorder.report(payload["source_commit"])
        except Exception as error:
            document = {"error": f"{type(error).__name__}: {error}"}
            status = 1
        with os.fdopen(write, "w", encoding="utf-8") as output:
            json.dump(document, output, sort_keys=True)
        os._exit(status)
    os.close(write)
    with os.fdopen(read, "r", encoding="utf-8") as source:
        text = source.read()
    _, status = os.waitpid(child, 0)
    document = json.loads(text) if text else {"error": "the staging process wrote no report"}
    if os.waitstatus_to_exitcode(status) != 0 or "error" in document:
        raise SystemExit(f"stage_baseline: {document.get('error', 'staging failed')}")
    return document


def main(argv):
    parser = argparse.ArgumentParser(description="Write the node-independent baseline into an image root.")
    parser.add_argument("rootfs")
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("stage_baseline: run as root; the baseline is applied inside a chroot of the image")
    rootfs = os.path.realpath(arguments.rootfs)
    if not os.path.isfile(os.path.join(rootfs, "etc/spark-firewall.nft")):
        raise SystemExit("stage_baseline: place the host firewall ruleset in the image before staging the baseline")
    module = load_baseline()
    payload = module.build_payload(ROOT, arguments.source_ref)
    payload["recovery_network"] = module.recovery_network_for_node(IMAGE_NODE_ID)
    document = stage(module, payload, rootfs)
    with open(arguments.report, "w", encoding="utf-8") as output:
        json.dump(document, output, indent=2, sort_keys=True)
    print(f"stage_baseline: {len(document['applied'])} unit and sshd commands applied in the image, "
          f"{len(document['deferred_live'])} live commands and {len(document['deferred_to_node'])} node steps left to node bring-up")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
