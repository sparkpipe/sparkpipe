import importlib.util
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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


UBUNTU_DEFAULT_GRUB = 'GRUB_DEFAULT=0\nGRUB_TIMEOUT_STYLE=hidden\nGRUB_TIMEOUT=0\nGRUB_CMDLINE_LINUX_DEFAULT="quiet splash"\nGRUB_CMDLINE_LINUX=""\n'
READ_TEXT = Path.read_text
AUDIT = load("hosted_dependency_audit_for_image", ROOT / "tools" / "hosted_dependency_audit.py")
STAGER = load("hosted_image_stage_baseline", IMAGE / "stage_baseline.py")
BASELINE = STAGER.load_baseline()
GUARD = STAGER.load_guard()
PINS = {key: quoted or plain for key, quoted, plain in re.findall(r'^([A-Z0-9_]+)=(?:"([^"]*)"|(\S*))$', (IMAGE / "pins.conf").read_text(), re.M)}


def compaction_helper(base):
    text = (IMAGE / "files" / "usr" / "local" / "sbin" / "sparkpipe-compact").read_text()
    for real, fake in (("/run/sparkpipe-compact", base / "state"), ("/proc/sys/vm/compaction_proactiveness", base / "proactiveness"),
                       ("/proc/sys/vm/min_free_kbytes", base / "reserve"), ("/proc/sys/vm/compact_memory", base / "compact"), ("/proc/buddyinfo", base / "buddyinfo")):
        assert text.count(real) == 1, real
        text = text.replace(real, str(fake))
    helper = base / "sparkpipe-compact"
    helper.write_text(text)
    helper.chmod(0o755)
    return helper


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
        self.assertEqual(PINS["TAILSCALE_DEB"], f"pool/tailscale_{PINS['TAILSCALE_VERSION']}_{PINS['ARCHITECTURE']}.deb")
        self.assertIn("iptables", image)
        self.assertEqual(AUDIT.package_reason("tailscale", PINS["TAILSCALE_VERSION"]), "")
        for key in ("MMDEBSTRAP_SHA256", "MODULE_SHA256", "TAILSCALE_SHA256"):
            self.assertRegex(PINS[key], r"^[0-9a-f]{64}$")
        self.assertIn(f"_{PINS['MMDEBSTRAP_VERSION']}-", PINS["MMDEBSTRAP_DEB"])
        self.assertRegex(PINS["SNAPSHOT_URL"], r"^https://snapshot\.ubuntu\.com/ubuntu/[0-9]{8}T[0-9]{6}Z$")

    def test_the_image_carries_every_package_the_baseline_installs(self):
        image = {name for name, version in packages("image-packages.txt")}
        self.assertEqual(sorted(set(BASELINE.required_packages(PINS["KERNEL_RELEASE"])) - image), [])


class StagerTest(unittest.TestCase):
    def staged_apply(self):
        recorder = STAGER.Recorder(BASELINE, GUARD)
        commands = []
        writes = {}

        def run(argv, input_bytes=None, timeout=120, check=True, env=None):
            where, command = STAGER.classify(BASELINE, argv)
            commands.append((where, tuple(command)))
            if where == "guard":
                return recorder.install_guard(command[3], timeout)
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
                        mock.patch.object(BASELINE, "FIREWALL_RULESET", ruleset), mock.patch.object(BASELINE.os, "geteuid", return_value=0), \
                        mock.patch.object(BASELINE.Path, "read_text", lambda path, *args, **keywords: UBUNTU_DEFAULT_GRUB if str(path) == "/etc/default/grub" else READ_TEXT(path, *args, **keywords)):
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
        self.assertIn(("guard", (BASELINE.DMA_GUARD_RUNTIME, "install", "--notify", BASELINE.HUB_NOTIFY)), commands)
        self.assertIn(("systemctl", "--root=/", "enable", "spark-dma-guard.timer"), image)
        service, timer = GUARD.unit_text(BASELINE.HUB_NOTIFY)
        self.assertEqual(writes[str(GUARD.SERVICE_PATH)], (service, 0o644))
        self.assertEqual(writes[str(GUARD.TIMER_PATH)], (timer, 0o644))
        for path in ("/etc/sysctl.d/99-spark-recovery.conf", "/etc/systemd/system.conf.d/90-ds4-recovery.conf"):
            self.assertIn(path, writes)
        self.assertIn("kernel.hung_task_panic=1", writes["/etc/sysctl.d/99-spark-recovery.conf"][0].split())
        self.assertIn(("loginctl", "enable-linger", STAGER.IMAGE_NODE_ID), live)
        self.assertEqual(sorted(set(recorder.node)), sorted({"install_packages", "install_node_identity", "install_node_layout",
                                                             "install_sshd_hardening", "rebuild_initramfs"}))
        self.assertIn(("update-grub",), live)
        default = BASELINE.shell_assignment(writes["/etc/default/grub"][0], "GRUB_CMDLINE_LINUX_DEFAULT").split()
        self.assertIn("console=ttyS0,921600", default)
        self.assertFalse([token for token in default if token.startswith("ip=")])
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

    def test_boot_leaves_the_kernel_compaction_defaults(self):
        keys = ("vm.compaction_proactiveness", "vm.min_free_kbytes")
        for path in (IMAGE / "files").rglob("*"):
            if path.is_file() and "sysctl" in str(path.relative_to(IMAGE / "files")):
                for key in keys:
                    with self.subTest(path=path, key=key):
                        self.assertNotIn(key, path.read_text())
        later = "\n".join([BASELINE.NO_THRASH_SYSCTL, BASELINE.RECOVERY_SYSCTL, BASELINE.DIRTY_CAP_SYSCTL])
        for key in keys:
            self.assertNotIn(key, later)

    def test_the_compaction_helper_is_the_only_privileged_path(self):
        files = IMAGE / "files"
        rules = sorted(path.name for path in (files / "etc" / "sudoers.d").iterdir())
        self.assertEqual(rules, ["90-sparkpipe-compact"])
        lines = [line for line in (files / "etc" / "sudoers.d" / "90-sparkpipe-compact").read_text().splitlines() if line.strip()]
        self.assertEqual(lines, ["%sparkpipe-compact ALL=(root) NOPASSWD: /usr/local/sbin/sparkpipe-compact raise, /usr/local/sbin/sparkpipe-compact restore"])
        self.assertTrue(os.access(files / "usr" / "local" / "sbin" / "sparkpipe-compact", os.X_OK))
        customize = (IMAGE / "customize_image.sh").read_text()
        self.assertIn('chmod 0440 "$root/etc/sudoers.d/90-sparkpipe-compact"', customize)
        self.assertIn('chroot "$root" visudo -c -q -f /etc/sudoers.d/90-sparkpipe-compact', customize)
        self.assertIn('chroot "$root" groupadd --system sparkpipe-compact', customize)
        self.assertEqual((files / "usr" / "lib" / "tmpfiles.d" / "sparkpipe-compact.conf").read_text().splitlines(),
                         ["d /run/sparkpipe-compact 0755 root root -", "f /run/sparkpipe-compact/window 0644 root root -"])

    @unittest.skipUnless(shutil.which("flock"), "the compaction helper serializes with flock")
    def test_the_helper_raises_once_and_restores_only_when_no_window_is_open(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            helper = compaction_helper(base)
            (base / "proactiveness").write_text("20\n")
            (base / "reserve").write_text("45056\n")
            (base / "buddyinfo").write_text("Node 0, zone   Normal  10 10 10 10 10 10 10 10 10 100 4\n")
            run = lambda verb: subprocess.run([str(helper), verb], capture_output=True, text=True)
            started = time.monotonic()
            raised = run("raise")
            self.assertEqual(raised.returncode, 0)
            self.assertGreaterEqual(time.monotonic() - started, 2.5)
            self.assertLess(time.monotonic() - started, 25.0)
            self.assertIn("sparkpipe-compact: raised;", raised.stdout)
            self.assertEqual(((base / "proactiveness").read_text().strip(), (base / "reserve").read_text().strip(), (base / "compact").read_text().strip()),
                             ("90", "1048576", "1"))
            self.assertEqual((base / "state" / "boot").read_text(), "20 45056\n")
            self.assertEqual(run("raise").returncode, 0)
            self.assertEqual((base / "state" / "boot").read_text(), "20 45056\n")
            holder = subprocess.Popen(["flock", "-s", str(base / "state" / "window"), "sleep", "30"], start_new_session=True)
            try:
                time.sleep(0.3)
                self.assertEqual(run("restore").returncode, 0)
                self.assertEqual((base / "proactiveness").read_text().strip(), "90")
            finally:
                os.killpg(holder.pid, signal.SIGTERM)
                holder.wait()
            self.assertEqual(run("restore").returncode, 0)
            self.assertEqual(((base / "proactiveness").read_text().strip(), (base / "reserve").read_text().strip()), ("20", "45056"))
            self.assertFalse((base / "state" / "boot").exists())
            self.assertEqual(run("restore").returncode, 0)
            self.assertEqual(run("lower").returncode, 2)

    def test_the_gb10_fragment_only_appends_to_the_command_line_the_baseline_owns(self):
        grub = (IMAGE / "files" / "etc" / "default" / "grub.d" / "90-sparkpipe-gb10.cfg").read_text()
        assignments = dict(re.findall(r'^([A-Z_]+)=(.*)$', grub, re.M))
        self.assertNotIn("GRUB_CMDLINE_LINUX_DEFAULT", assignments)
        self.assertNotIn("GRUB_DEFAULT", assignments)
        self.assertTrue(assignments["GRUB_CMDLINE_LINUX"].startswith('"$GRUB_CMDLINE_LINUX '))
        tokens = assignments["GRUB_CMDLINE_LINUX"].strip('"').split()[1:]
        self.assertFalse([token for token in tokens if token.startswith(("console=", "ip=", "fsck."))])


class SelfUpdateTest(unittest.TestCase):
    UNITS = ("apt-daily.timer", "apt-daily.service", "apt-daily-upgrade.timer", "apt-daily-upgrade.service", "motd-news.timer", "motd-news.service")

    def test_the_image_masks_every_update_and_news_unit_and_carries_no_unattended_upgrades(self):
        self.assertEqual((IMAGE / "masked-units.txt").read_text().split(), list(self.UNITS))
        self.assertNotIn("unattended-upgrades", dict(packages("image-packages.txt")))
        self.assertIn('"$here/mask_units.sh" "$root"', (IMAGE / "customize_image.sh").read_text())

    @unittest.skipUnless(shutil.which("systemctl"), "no systemctl on this host")
    def test_a_staged_root_has_the_update_timers_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units = root / "usr" / "lib" / "systemd" / "system"
            units.mkdir(parents=True)
            (root / "etc" / "systemd" / "system").mkdir(parents=True)
            for unit in self.UNITS:
                kind = "Timer" if unit.endswith(".timer") else "Service"
                (units / unit).write_text(f"[Unit]\nDescription={unit}\n[{kind}]\n" + ("OnCalendar=daily\n[Install]\nWantedBy=timers.target\n" if kind == "Timer" else "ExecStart=/bin/true\n"))
            completed = subprocess.run(["bash", str(IMAGE / "mask_units.sh"), str(root)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for unit in self.UNITS:
                self.assertEqual(os.readlink(root / "etc" / "systemd" / "system" / unit), "/dev/null")


class OwnershipTest(unittest.TestCase):
    HOOK = (IMAGE / "customize_image.sh").read_text()

    def test_the_hook_installs_files_root_owned_and_never_copies_build_ownership(self):
        self.assertNotIn("cp -a", self.HOOK)
        self.assertIn('install -D -o 0 -g 0 -m "$mode"', self.HOOK)
        self.assertIn('install_tree "$here/files" "$root"', self.HOOK)
        self.assertIn('install_tree "$out/inputs/release-root" "$root/$RELEASE_ROOT_PATH"', self.HOOK)

    def test_the_hook_fails_on_any_build_user_ownership_and_on_a_root_that_is_not_root_owned_0755(self):
        self.assertIn("-uid +999 -a ! -uid 65534", self.HOOK)
        self.assertIn("-gid +999 -a ! -gid 65534", self.HOOK)
        self.assertIn('= "0:0:755" ]', self.HOOK)
        self.assertLess(self.HOOK.index("install_tree \"$here/files\""), self.HOOK.index("-uid +999"))
        self.assertGreater(self.HOOK.index("-uid +999"), self.HOOK.index("update-initramfs"))

    @unittest.skipUnless(sys.platform.startswith("linux") and os.geteuid() == 0, "install -o 0 needs root on Linux")
    def test_install_tree_gives_root_ownership_and_plain_modes(self):
        function = self.HOOK[self.HOOK.index("install_tree() {"):self.HOOK.index("\n}\n", self.HOOK.index("install_tree() {")) + 3]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "source")
            (source / "etc" / "x").mkdir(parents=True)
            (source / "etc" / "x" / "conf").write_text("c\n")
            (source / "etc" / "x" / "conf").chmod(0o664)
            (source / "bin").mkdir()
            (source / "bin" / "tool").write_text("#!/bin/sh\n")
            (source / "bin" / "tool").chmod(0o775)
            os.chown(source / "etc" / "x" / "conf", 1000, 1000)
            target = Path(directory, "target")
            target.mkdir()
            subprocess.run(["bash", "-c", function + f'install_tree "{source}" "{target}"'], check=True)
            self.assertEqual((os.stat(target / "etc" / "x" / "conf").st_uid, os.stat(target / "etc" / "x" / "conf").st_mode & 0o777), (0, 0o644))
            self.assertEqual(os.stat(target / "bin" / "tool").st_mode & 0o777, 0o755)
            self.assertEqual(os.stat(target / "etc" / "x").st_mode & 0o777, 0o755)


class BuildDriverTest(unittest.TestCase):
    def test_the_entry_scripts_are_executable(self):
        for path in ("build_image.sh", "customize_image.sh", "mask_units.sh", "files/usr/lib/sparkpipe/gpu-device-nodes", "files/usr/lib/sparkpipe/host-keys"):
            with self.subTest(path=path):
                self.assertTrue(os.access(IMAGE / path, os.X_OK))

    def test_missing_inputs_are_refused_before_anything_is_built(self):
        script = str(IMAGE / "build_image.sh")
        self.assertEqual(subprocess.run(["bash", script], capture_output=True).returncode, 2)
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(["bash", script, "--out", f"{directory}/out", "--source-ref", "HEAD", "--release-root", f"{directory}/absent",
                                        "--firewall-ruleset", f"{directory}/absent.nft"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertFalse(Path(directory, "out").exists())

    def test_a_release_root_that_fails_the_audit_is_refused_before_anything_is_built(self):
        script = str(IMAGE / "build_image.sh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "release")
            (root / "lib").mkdir(parents=True)
            (root / "lib" / "libcuda.so.1").write_bytes(b"\x7fELF")
            ruleset = Path(directory, "spark-firewall.nft")
            ruleset.write_text("table inet spark_guard {}\n")
            completed = subprocess.run(["bash", script, "--out", f"{directory}/out", "--source-ref", "HEAD", "--release-root", str(root),
                                        "--firewall-ruleset", str(ruleset)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertIn("FAIL library-file lib/libcuda.so.1: libcuda.so.1", completed.stderr)
            self.assertIn("build_image: the release root fails the dependency audit", completed.stderr)
            self.assertFalse(Path(directory, "out").exists())


if __name__ == "__main__":
    unittest.main()
