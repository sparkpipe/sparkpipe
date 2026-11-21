import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "tools" / "hosted_image"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load("hosted_dependency_audit_for_image", ROOT / "tools" / "hosted_dependency_audit.py")
STAGER = load("hosted_image_stage_baseline", IMAGE / "stage_baseline.py")
BASELINE = STAGER.load_baseline()
PINS = {key: quoted or plain for key, quoted, plain in re.findall(r'^([A-Z0-9_]+)=(?:"([^"]*)"|(\S*))$', (IMAGE / "pins.conf").read_text(), re.M)}


def packages(name):
    lines = (IMAGE / name).read_text().splitlines()
    return [tuple(line.split("=", 1)) for line in lines]


class PinsTest(unittest.TestCase):
    def test_every_package_is_pinned_once_and_none_is_rejected_by_the_image_audit(self):
        entries = packages("image-packages.txt")
        self.assertTrue(all(len(entry) == 2 and entry[0] and re.fullmatch(r"[0-9][A-Za-z0-9.+~:-]*", entry[1]) for entry in entries))
        self.assertEqual(len({entry[0] for entry in entries}), len(entries))
        self.assertEqual([entry for entry in packages("image-packages.txt") if AUDIT.package_reason(*entry)], [])

    def test_kernel_module_firmware_and_tool_pins_agree(self):
        image = dict(packages("image-packages.txt"))
        kernel = PINS["KERNEL_RELEASE"]
        branch = PINS["MODULE_VERSION"].split(".")[0]
        self.assertIn(f"linux-image-{kernel}", image)
        self.assertIn(f"linux-modules-{kernel}", image)
        self.assertEqual(PINS["MODULE_PACKAGE"], f"linux-modules-nvidia-{branch}-open-{kernel}")
        self.assertIn(f"/{PINS['MODULE_PACKAGE']}_{image[f'linux-image-{kernel}']}_{PINS['ARCHITECTURE']}.deb", PINS["MODULE_DEB"])
        self.assertNotIn(PINS["MODULE_PACKAGE"], image)
        self.assertEqual(AUDIT.package_reason(PINS["MODULE_PACKAGE"], image[f"linux-image-{kernel}"]), "")
        self.assertEqual(PINS["MODULE_FILES"].split(), ["nvidia", "nvidia-uvm"])
        self.assertEqual(PINS["FIRMWARE_PACKAGE"], f"nvidia-firmware-{branch}-{PINS['MODULE_VERSION']}")
        self.assertTrue(image[PINS["FIRMWARE_PACKAGE"]].startswith(PINS["MODULE_VERSION"] + "-"))
        for key in ("MMDEBSTRAP_SHA256", "MODULE_SHA256"):
            self.assertRegex(PINS[key], r"^[0-9a-f]{64}$")
        self.assertIn(f"_{PINS['MMDEBSTRAP_VERSION']}-", PINS["MMDEBSTRAP_DEB"])
        self.assertRegex(PINS["SNAPSHOT_URL"], r"^https://snapshot\.ubuntu\.com/ubuntu/[0-9]{8}T[0-9]{6}Z$")

    def test_the_image_carries_every_package_the_baseline_installs(self):
        image = {name for name, version in packages("image-packages.txt")}
        self.assertEqual(sorted(set(BASELINE.required_packages(PINS["KERNEL_RELEASE"])) - image), [])


class StagerTest(unittest.TestCase):
    def staged_apply(self):
        recorder = STAGER.Recorder(BASELINE)
        commands = []
        writes = {}

        def run(argv, input_bytes=None, timeout=120, check=True, env=None):
            where, command = STAGER.classify(BASELINE, argv)
            commands.append((where, tuple(command)))
            return subprocess.CompletedProcess(command, 0, b"", b"")

        payload = {"assets": {path: {"text": "asset\n", "mode": 0o644} for path in BASELINE.STAGED_ASSET_DESTINATIONS},
                   "linked_rank_assets": {}, "fleet_public_key": "ssh-ed25519 AAAA test", "node_layout_roots": ["extnvme"],
                   "recovery_network": BASELINE.recovery_network_for_node(STAGER.IMAGE_NODE_ID), "source_commit": "abc"}
        with tempfile.TemporaryDirectory() as directory:
            ruleset = Path(directory) / "spark-firewall.nft"
            ruleset.write_text("table inet spark_guard {}\n")
            saved = {name: getattr(BASELINE, name) for name in (*STAGER.NODE_STEPS, "run", "has_configured_remote_storage")}
            try:
                recorder.install()
                with mock.patch.object(BASELINE, "run", side_effect=run), \
                        mock.patch.object(BASELINE, "atomic_write", side_effect=lambda path, text, mode: writes.__setitem__(str(path), (text, mode)) or True), \
                        mock.patch.object(BASELINE, "install_sshd_config"), mock.patch.object(BASELINE, "install_recovery_keys"), \
                        mock.patch.object(BASELINE, "persistent_unit_links", return_value=[]), \
                        mock.patch.object(BASELINE, "retire_stale_snap_units", return_value=[]), \
                        mock.patch.object(BASELINE, "ceph_fsids", return_value=[]), \
                        mock.patch.object(BASELINE, "FIREWALL_RULESET", ruleset), mock.patch.object(BASELINE.os, "geteuid", return_value=0):
                    result = BASELINE.remote_apply(payload)
            finally:
                for name, value in saved.items():
                    setattr(BASELINE, name, value)
        return result, recorder, commands, writes

    def test_every_command_of_the_staged_baseline_is_placed_in_the_image_or_left_to_the_live_node(self):
        result, recorder, commands, writes = self.staged_apply()
        self.assertEqual(result["profile"], "staged")
        image = [command for where, command in commands if where == "image"]
        live = [command for where, command in commands if where == "live"]
        for unit in BASELINE.OPTIONAL_BOOT_UNITS:
            self.assertIn(("systemctl", "--root=/", "mask", unit), image)
        for unit in ("earlyoom.service", "ssh.service", "ssh-emergency.service", "sparkpipe-fsck-health.timer",
                     "serial-getty@ttyS0.service", "spark-firewall.service", "fstrim.timer"):
            self.assertIn(("systemctl", "--root=/", "enable", unit), image)
        self.assertIn(("systemctl", "--root=/", "set-default", "multi-user.target"), image)
        self.assertTrue(all("--now" not in command for command in image))
        self.assertIn(("sysctl", "--system"), live)
        self.assertIn(("modprobe", "sch_fq_codel"), live)
        self.assertIn((BASELINE.DMA_GUARD_RUNTIME, "install", "--notify", BASELINE.HUB_NOTIFY), live)
        self.assertIn(("loginctl", "enable-linger", STAGER.IMAGE_NODE_ID), live)
        self.assertEqual(sorted(set(recorder.node)), sorted({"install_packages", "install_node_identity", "install_node_layout",
                                                             "install_sshd_hardening", "install_grub_policy", "rebuild_initramfs"}))
        for path in (*BASELINE.STAGED_ASSET_DESTINATIONS, "/etc/default/earlyoom", "/etc/sysctl.d/99-dirty-cap.conf",
                     "/etc/systemd/system.conf.d/90-ds4-recovery.conf", "/etc/systemd/system/ssh-emergency.service"):
            self.assertIn(path, writes)
        self.assertEqual(writes["/etc/default/earlyoom"], (BASELINE.EARLYOOM_DEFAULT, 0o644))

    def test_a_command_the_stager_cannot_place_fails(self):
        for argv in (["apt-get", "install", "-y", "x"], ["systemctl", "isolate", "rescue.target"], ["/usr/local/sbin/new-tool"]):
            with self.subTest(argv=argv), self.assertRaises(BASELINE.BrickproofError):
                STAGER.classify(BASELINE, argv)

    def test_every_deferred_node_step_exists_in_the_baseline(self):
        for name in STAGER.NODE_STEPS:
            self.assertTrue(callable(getattr(BASELINE, name, None)), name)

    def test_staging_refuses_to_run_without_root_or_a_firewall_ruleset(self):
        completed = subprocess.run([sys.executable, str(IMAGE / "stage_baseline.py"), "/nonexistent", "--source-ref", "HEAD",
                                    "--report", "/dev/null"], capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertRegex(completed.stderr, "run as root|firewall ruleset")


@unittest.skipUnless(sys.platform.startswith("linux"), "the device-node helper uses GNU stat")
class DeviceNodeTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.proc = self.directory / "proc"
        self.dev = self.directory / "dev"
        self.dev.mkdir()
        stubs = self.directory / "bin"
        stubs.mkdir()
        mknod = stubs / "mknod"
        mknod.write_text('#!/bin/sh\n[ "$1" = -m ] && shift 2\nprintf "%s %s %s\\n" "$2" "$3" "$4" > "$1"\n')
        mknod.chmod(0o755)
        self.environment = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}"}

    def run_helper(self, devices, gpus):
        (self.proc / "driver" / "nvidia" / "gpus").mkdir(parents=True, exist_ok=True)
        (self.proc / "devices").write_text("Character devices:\n" + devices)
        for bus, text in gpus.items():
            (self.proc / "driver" / "nvidia" / "gpus" / bus).mkdir()
            (self.proc / "driver" / "nvidia" / "gpus" / bus / "information").write_text(text)
        return subprocess.run(["sh", str(IMAGE / "files" / "usr" / "lib" / "sparkpipe" / "gpu-device-nodes"), str(self.proc), str(self.dev)],
                              capture_output=True, text=True, env=self.environment)

    def nodes(self):
        return {path.name: path.read_text().strip() for path in self.dev.iterdir()}

    def test_control_gpu_and_uvm_nodes_follow_the_registered_devices(self):
        completed = self.run_helper("195 nvidia\n195 nvidiactl\n509 nvidia-uvm\n",
                                    {"000f:01:00.0": "Model: \t\t NVIDIA GB10\nDevice Minor: \t 0\n"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.nodes(), {"nvidiactl": "c 195 255", "nvidia0": "c 195 0", "nvidia-uvm": "c 509 0"})

    def test_nothing_registered_creates_nothing_and_a_partial_module_fails(self):
        self.assertEqual(self.run_helper("1 mem\n", {}).returncode, 0)
        self.assertEqual(self.nodes(), {})
        self.assertEqual(self.run_helper("195 nvidiactl\n", {}).returncode, 1)
        shutil.rmtree(self.proc)
        self.assertEqual(self.run_helper("195 nvidiactl\n", {"000f:01:00.0": "Model: \t\t NVIDIA GB10\n"}).returncode, 1)


class StaticFilesTest(unittest.TestCase):
    def test_image_files_pass_the_image_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            rootfs = Path(directory) / "rootfs"
            shutil.copytree(IMAGE / "files", rootfs)
            status = rootfs / "var" / "lib" / "dpkg" / "status"
            status.parent.mkdir(parents=True)
            status.write_text("\n".join(f"Package: {name}\nStatus: install ok installed\nVersion: {version}\n"
                                        for name, version in packages("image-packages.txt")))
            report = AUDIT.audit_image(str(rootfs), None, [])
        self.assertEqual(report.findings(), [])

    def test_gpu_modules_load_and_their_device_nodes_are_made_by_the_shipped_helper(self):
        files = IMAGE / "files"
        self.assertEqual((files / "etc" / "modules-load.d" / "sparkpipe-gpu.conf").read_text().split(), ["nvidia", "nvidia_uvm"])
        rules = (files / "etc" / "udev" / "rules.d" / "71-sparkpipe-gpu.rules").read_text()
        helpers = set(re.findall(r'RUN\+="([^"]+)"', rules))
        self.assertEqual(helpers, {"/usr/lib/sparkpipe/gpu-device-nodes"})
        self.assertTrue(os.access(files / "usr" / "lib" / "sparkpipe" / "gpu-device-nodes", os.X_OK))
        self.assertEqual({module for module in re.findall(r'KERNEL=="([^"]+)"', rules)}, {"nvidia", "nvidia_uvm"})

    def test_the_kernel_console_is_the_serial_line_the_baseline_keeps_a_getty_on(self):
        grub = (IMAGE / "files" / "etc" / "default" / "grub.d" / "90-sparkpipe-gb10.cfg").read_text()
        command_line = re.search(r'^GRUB_CMDLINE_LINUX="([^"]*)"$', grub, re.M).group(1).split()
        getty = re.search(r"serial-getty@(\w+)\.service", (ROOT / "tools" / "devcycle" / "ds4_spark_brickproof.py").read_text()).group(1)
        self.assertIn(getty, [argument.split("=", 1)[1].split(",")[0] for argument in command_line if argument.startswith("console=")])


class BuildDriverTest(unittest.TestCase):
    def test_missing_inputs_are_refused_before_anything_is_built(self):
        script = str(IMAGE / "build_image.sh")
        self.assertEqual(subprocess.run(["bash", script], capture_output=True).returncode, 2)
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(["bash", script, "--out", f"{directory}/out", "--source-ref", "HEAD", "--release-root", f"{directory}/absent",
                                        "--firewall-ruleset", f"{directory}/absent.nft"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertFalse(Path(directory, "out").exists())


if __name__ == "__main__":
    unittest.main()
