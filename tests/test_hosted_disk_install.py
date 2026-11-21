import hashlib
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "hosted_image" / "install_disk.sh"
SOURCE = SCRIPT.read_text()


class InstallDiskInputsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.image = base / "build"
        self.image.mkdir()
        archive = self.image / "rootfs.tar.zst"
        archive.write_bytes(b"rootfs")
        self.digest = hashlib.sha256(b"rootfs").hexdigest()
        self.manifest(f"source_commit abc\nrootfs_sha256 {self.digest}\naudit PASS\n")
        self.identity = base / "identity"
        (self.identity / "files" / "etc").mkdir(parents=True)
        (self.identity / "files" / "etc" / "hostname").write_text("spark1\n")
        (self.identity / "user.conf").write_text("USER_NAME=spark1\nUSER_UID=1000\nUSER_GID=1000\nUSER_GROUPS=sudo,adm,codexadmin\nUSER_SHELL=/bin/bash\n")
        (self.identity / "groups.conf").write_text("codexadmin:1001\n")
        self.work = base / "work"

    def manifest(self, text):
        (self.image / "MANIFEST").write_text(text)

    def install(self, disk="/nonexistent/disk"):
        return subprocess.run(["bash", str(SCRIPT), "--disk", disk, "--image", str(self.image), "--identity", str(self.identity),
                               "--work", str(self.work)], capture_output=True, text=True)

    def refused(self, message, disk="/nonexistent/disk"):
        result = self.install(disk)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(self.work.exists())

    def test_usage_needs_every_input(self):
        self.assertEqual(subprocess.run(["bash", str(SCRIPT), "--disk", "/dev/x"], capture_output=True).returncode, 2)

    def test_a_valid_build_and_bundle_reach_the_block_device_check(self):
        self.refused("/nonexistent/disk is not a block device")

    def test_an_image_without_a_passing_audit_or_with_another_archive_is_refused(self):
        self.manifest(f"rootfs_sha256 {self.digest}\naudit FAIL\n")
        self.refused("does not record a passing dependency audit")
        self.manifest(f"rootfs_sha256 {'0' * 64}\naudit PASS\n")
        self.refused("does not match its MANIFEST")
        (self.image / "MANIFEST").unlink()
        self.refused("holds no rootfs.tar.zst and MANIFEST")

    def test_an_incomplete_or_unsafe_identity_bundle_is_refused(self):
        (self.identity / "user.conf").write_text("USER_NAME=spark1\nUSER_UID=x\n")
        self.refused("user.conf needs")
        (self.identity / "user.conf").write_text("USER_NAME=spark1\nUSER_UID=1000\nUSER_GID=1000\nUSER_GROUPS=sudo;id\nUSER_SHELL=/bin/bash\n")
        self.refused("USER_GROUPS must be a comma list")
        (self.identity / "user.conf").write_text("USER_NAME=spark1\nUSER_UID=1000\nUSER_GID=1000\nUSER_GROUPS=sudo\nUSER_SHELL=/bin/bash\n")
        (self.identity / "files" / "etc" / "link").symlink_to("hostname")
        self.refused("may hold only files and directories")
        (self.identity / "files" / "etc" / "link").unlink()
        keys = self.identity / "files" / "etc" / "ssh"
        keys.mkdir()
        (keys / "ssh_host_ed25519_key").write_text("key")
        self.refused("carries SSH host keys but no initramfs SSH host keys")
        dropbear = self.identity / "files" / "etc" / "dropbear" / "initramfs"
        dropbear.mkdir(parents=True)
        (dropbear / "dropbear_ed25519_host_key").write_text("key")
        self.refused("is not a block device")
        (self.identity / "groups.conf").write_text("codexadmin:1001\nbad name:x\n")
        self.refused("line 'bad name:x' is not name:gid")
        (self.identity / "groups.conf").unlink()
        self.refused("needs files/, user.conf and groups.conf")

    def test_the_bundle_is_read_as_data_and_never_executed(self):
        marker = Path(self.directory.name) / "executed"
        (self.identity / "user.conf").write_text(f"USER_NAME=spark1\nUSER_UID=1000\nUSER_GID=1000\nUSER_GROUPS=sudo\nUSER_SHELL=/bin/bash\ntouch {marker}\n")
        self.refused("is not a block device")
        self.assertFalse(marker.exists())


class InstallDiskContractTest(unittest.TestCase):
    def test_the_install_keeps_home_and_never_deletes(self):
        self.assertNotIn("rm -", SOURCE)
        self.assertIn("! -name home ! -name lost+found", SOURCE)
        self.assertIn("--exclude=./home", SOURCE)
        self.assertIn('mv "$target/$entry" "$aside/"', SOURCE)

    def test_nvram_is_only_read_and_must_not_change(self):
        self.assertIn("--no-nvram", SOURCE)
        self.assertNotIn("efivars", SOURCE)
        calls = re.findall(r"\$\(efibootmgr([^)]*)\)", SOURCE)
        self.assertEqual(calls, [" -v", " -v"])
        self.assertEqual(SOURCE.count("efibootmgr"), len(calls) + SOURCE.count('"efibootmgr cannot read'))
        self.assertIn('[ "$(efibootmgr -v)" = "$nvram" ] || fail', SOURCE)
        self.assertLess(SOURCE.index('nvram="$(efibootmgr -v)"'), SOURCE.index("e2fsck -fp"))

    def test_the_efi_mount_point_exists_before_the_mount(self):
        self.assertLess(SOURCE.index('install -d -o 0 -g 0 -m 0755 "$target/boot/efi"'), SOURCE.index('mount_at "$esp" "$target/boot/efi"'))

    def test_sudoers_are_checked_before_the_bootloader(self):
        self.assertLess(SOURCE.index('chroot "$target" visudo -cq'), SOURCE.index('chroot "$target" grub-install'))

    def test_the_target_is_checked_before_it_is_changed(self):
        self.assertLess(SOURCE.index('findmnt -rn -S "$partition"'), SOURCE.index('mount_at "$root" "$target"'))
        self.assertLess(SOURCE.index('e2fsck -fp "$root"'), SOURCE.index('mount_at "$root" "$target"'))
        self.assertLess(SOURCE.index("previous aside directory"), SOURCE.index('mkdir "$aside"'))
        self.assertLess(SOURCE.index("tar --zstd -xpf"), SOURCE.index("-uid +999 -a ! -uid 65534"))
        self.assertIn("trap unmount_all EXIT", SOURCE)

    def test_the_labels_match_the_image_fstab(self):
        fstab = (ROOT / "tools" / "hosted_image" / "customize_image.sh").read_text()
        for label in ("SPARKROOT", "SPARKESP"):
            self.assertIn(f"LABEL={label}", fstab)
        self.assertIn('e2label "$root" SPARKROOT', SOURCE)
        self.assertIn('fatlabel "$esp" SPARKESP', SOURCE)

    def test_the_script_is_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))


if __name__ == "__main__":
    unittest.main()
