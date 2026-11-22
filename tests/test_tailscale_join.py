import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "tailscale_join.sh"
KEY = "tskey-auth-kFixture7CNTRL-q8Zr4mWx2Lp9Hs3VtYb6"

FAKE_SSH = """#!/bin/bash
while [ "$1" = -o ]; do shift 2; done
target="$1"
shift
printf '%s %s\\n' "$target" "$*" >> "$FAKE_LOG/ssh.log"
exec /bin/sh -c "$*"
"""

FAKE_SUDO = """#!/bin/bash
[ "$1" = -n ] || { echo "fake sudo: expected -n" >&2; exit 1; }
shift
exec "$@"
"""

FAKE_FINDMNT = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_LOG/findmnt.log"
echo "${FAKE_FSTYPE:-tmpfs}"
"""

FAKE_HOSTNAME = """#!/bin/bash
echo "${FAKE_HOSTNAME:-node1}"
"""

FAKE_TAILSCALE = """#!/usr/bin/env python3
import json, os, stat, sys
log = os.environ["FAKE_LOG"]
status_path = os.path.join(log, "status.json")
arguments = sys.argv[1:]
with open(os.path.join(log, "tailscale.log"), "a") as record:
    record.write(" ".join(arguments) + "\\n")
if arguments == ["status", "--json"]:
    with open(status_path) as status:
        sys.stdout.write(status.read())
    sys.exit(0)
if arguments[:1] != ["up"]:
    sys.exit("fake tailscale: unexpected " + " ".join(arguments))
options = dict(argument[2:].split("=", 1) for argument in arguments[1:])
path = options["auth-key"][len("file:"):]
with open(path) as key, open(os.environ["FAKE_KEY_FILE"]) as expected:
    same = key.read() == expected.read()
modes = "%o %o" % (stat.S_IMODE(os.stat(path).st_mode), stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode))
with open(os.path.join(log, "key-seen.log"), "a") as record:
    record.write("%s %s\\n" % ("same" if same else "different", modes))
if os.environ.get("FAKE_UP_FAILS"):
    sys.exit("fake tailscale: login refused")
state = os.environ.get("FAKE_AFTER_STATE", "Running")
node = {"HostName": options["hostname"], "Online": state == "Running", "Tags": options.get("advertise-tags", "tag:from-key").split(","),
        "TailscaleIPs": ["100.64.0.7", "fd7a:115c:a1e0::7"]}
with open(status_path, "w") as status:
    json.dump({"BackendState": state, "Self": node}, status)
"""


def status(state, hostname="", tags=None, online=False):
    return {"BackendState": state, "Self": {"HostName": hostname, "Online": online, "Tags": tags or []}}


class TailscaleJoinTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        for name, text in (("ssh", FAKE_SSH), ("sudo", FAKE_SUDO), ("findmnt", FAKE_FINDMNT), ("hostname", FAKE_HOSTNAME), ("tailscale", FAKE_TAILSCALE)):
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        self.log = self.base / "log"
        self.log.mkdir()
        (self.base / "run").mkdir()
        self.key_dir = self.base / "run" / "join"
        self.key = self.base / ".config" / "sparkpipe" / "tailscale-authkey"
        self.key.parent.mkdir(parents=True)
        self.write_key(KEY + "\n")
        self.set_status(status("NeedsLogin"))

    def write_key(self, text, mode=0o600):
        self.key.write_text(text)
        self.key.chmod(mode)

    def set_status(self, document):
        (self.log / "status.json").write_text(json.dumps(document))

    def join(self, *extra, environment=None, tags=("--tags", "tag:fleet,tag:hosted")):
        env = {"PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}", "FAKE_LOG": str(self.log), "FAKE_KEY_FILE": str(self.key),
               "HOME": str(self.base)}
        env.update(environment or {})
        command = ["bash", str(TOOL), "--ssh", "node1", *tags, "--key-dir", str(self.key_dir), *extra]
        return subprocess.run(command, capture_output=True, text=True, env=env)

    def recorded(self, name):
        path = self.log / name
        return path.read_text() if path.exists() else ""

    def assert_key_never_shown(self, result):
        for text in (result.stdout, result.stderr, self.recorded("ssh.log"), self.recorded("tailscale.log"), self.recorded("findmnt.log"),
                     self.recorded("status.json")):
            self.assertNotIn(KEY, text)
            self.assertNotIn("q8Zr4mWx2Lp9Hs3VtYb6", text)
        self.assertFalse(self.key_dir.exists())

    def assert_refused_before_ssh(self, result, message):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertEqual(self.recorded("ssh.log"), "")
        self.assert_key_never_shown(result)

    def test_a_logged_out_node_joins_from_the_key_file_and_the_key_is_removed(self):
        result = self.join("--operator", "node")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("node1 is Running and online as node1 with tag:fleet,tag:hosted at 100.64.0.7", result.stdout)
        up = [line for line in self.recorded("tailscale.log").splitlines() if line.startswith("up ")]
        self.assertEqual(up, [f"up --auth-key=file:{self.key_dir}/key --hostname=node1 --advertise-tags=tag:fleet,tag:hosted "
                              "--operator=node --timeout=120s"])
        self.assertEqual(self.recorded("key-seen.log"), "same 600 700\n")
        self.assertIn(f"--target {self.key_dir.parent}", self.recorded("findmnt.log"))
        self.assert_key_never_shown(result)

    def test_without_an_operator_or_tags_those_flags_are_left_out_and_the_key_decides_the_tags(self):
        result = self.join(tags=())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        up = [line for line in self.recorded("tailscale.log").splitlines() if line.startswith("up ")]
        self.assertEqual(up, [f"up --auth-key=file:{self.key_dir}/key --hostname=node1 --timeout=120s"])
        self.assertIn("node1 is Running and online as node1 with its key's tags", result.stdout)
        self.assert_key_never_shown(result)

    def test_an_explicit_hostname_and_key_file_override_the_defaults(self):
        other = self.base / "other-key"
        other.write_text(KEY + "\n")
        other.chmod(0o600)
        self.key.unlink()
        result = self.join("--hostname", "renamed", environment={"TAILSCALE_AUTH_KEY_FILE": str(other), "FAKE_KEY_FILE": str(other)})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--hostname=renamed ", self.recorded("tailscale.log"))
        self.assertEqual(self.recorded("key-seen.log"), "same 600 700\n")
        self.assertNotIn("hostname", self.recorded("ssh.log").split("\n")[0].split(" ", 1)[1].split())

    def test_a_node_hostname_that_is_not_a_dns_label_needs_an_explicit_name(self):
        result = self.join(environment={"FAKE_HOSTNAME": "Node_1"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("hostname Node_1 is not a lower-case DNS label; give --hostname", result.stderr)
        self.assertNotIn("up ", self.recorded("tailscale.log"))
        self.assert_key_never_shown(result)

    def test_a_node_already_running_as_itself_gets_no_key(self):
        self.set_status(status("Running", "node1", ["tag:hosted", "tag:fleet"], True))
        for tags in (("--tags", "tag:fleet,tag:hosted"), ()):
            with self.subTest(tags=tags):
                result = self.join(tags=tags)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("already Running as node1", result.stdout)
                self.assertIn("nothing to do, no key was sent", result.stdout)
                self.assertNotIn("up ", self.recorded("tailscale.log"))
                self.assertEqual(self.recorded("key-seen.log"), "")
                self.assert_key_never_shown(result)

    def test_a_node_running_as_another_name_or_waiting_for_approval_is_refused_without_the_key(self):
        for document in (status("Running", "other", ["tag:fleet", "tag:hosted"], True), status("NeedsMachineAuth", "node1")):
            with self.subTest(document["BackendState"]):
                self.set_status(document)
                result = self.join()
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("not logged out", result.stderr)
                self.assertNotIn("up ", self.recorded("tailscale.log"))
                self.assert_key_never_shown(result)

    def test_the_key_file_must_be_private_single_and_a_tailscale_key(self):
        cases = (
            (KEY + "\n", 0o644, "readable or writable by others"),
            (KEY + "\n", 0o620, "readable or writable by others"),
            ("not-a-key\n", 0o600, "holds neither a Tailscale auth key nor an OAuth client secret"),
            (KEY + "\n" + KEY + "\n", 0o600, "must hold one line"),
            ("", 0o600, "must hold one line"),
            ("tskey-client-kFixture7CNTRL-q8Zr4mWx2Lp9Hs3VtYb6\n", 0o600, "without ?ephemeral=false&preauthorized=true"),
            ("tskey-client-kFixture7CNTRL-q8Zr4mWx2Lp9Hs3VtYb6?ephemeral=false\n", 0o600, "without ?ephemeral=false&preauthorized=true"),
        )
        for text, mode, message in cases:
            with self.subTest(message=message, mode=oct(mode)):
                self.write_key(text, mode)
                self.assert_refused_before_ssh(self.join(), message)
        self.write_key("tskey-client-kFixture7CNTRL-q8Zr4mWx2Lp9Hs3VtYb6?ephemeral=false&preauthorized=true\n")
        self.assert_refused_before_ssh(self.join(tags=()), "holds an OAuth client secret, which needs --tags")

    def test_an_oauth_client_secret_with_its_attributes_is_accepted(self):
        self.write_key("tskey-client-kFixture7CNTRL-q8Zr4mWx2Lp9Hs3VtYb6?ephemeral=false&preauthorized=true\n")
        result = self.join()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.recorded("key-seen.log"), "same 600 700\n")
        self.assert_key_never_shown(result)

    def test_a_missing_or_linked_key_file_is_refused(self):
        self.key.unlink()
        self.assert_refused_before_ssh(self.join(), f"{self.key} does not exist; save the auth key there")
        self.write_key(KEY + "\n")
        link = self.base / "link"
        link.symlink_to(self.key)
        self.assert_refused_before_ssh(self.join(environment={"TAILSCALE_AUTH_KEY_FILE": str(link)}), "is not a regular file")

    def test_bad_names_and_tags_are_refused(self):
        for extra, message in ((("--hostname", "Node_1"), "is not a lower-case DNS label"), (("--tags", "fleet"), "is not a comma list of tag:name"),
                               (("--operator", "a b"), "is not a user name"), (("--key-dir", "relative"), "is not an absolute path")):
            with self.subTest(message):
                self.assert_refused_before_ssh(self.join(*extra), message)
        for arguments in ((), ("--hostname", "node1"), ("--ssh",)):
            with self.subTest(arguments):
                result = subprocess.run(["bash", str(TOOL), *arguments], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)

    def test_a_key_directory_off_tmpfs_is_refused_and_nothing_is_written(self):
        result = self.join(environment={"FAKE_FSTYPE": "ext4"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("is not on tmpfs", result.stderr)
        self.assertNotIn("up ", self.recorded("tailscale.log"))
        self.assertEqual(list((self.base / "run").iterdir()), [])
        self.assert_key_never_shown(result)

    def test_a_failed_login_still_removes_the_key(self):
        result = self.join(environment={"FAKE_UP_FAILS": "1"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("tailscale up failed on node1", result.stderr)
        self.assertEqual(self.recorded("key-seen.log"), "same 600 700\n")
        self.assert_key_never_shown(result)

    def test_a_join_that_does_not_reach_running_fails_naming_the_state(self):
        result = self.join(environment={"FAKE_AFTER_STATE": "NeedsMachineAuth"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("the node is NeedsMachineAuth", result.stderr)
        self.assert_key_never_shown(result)

    def test_a_leftover_key_directory_is_refused(self):
        self.key_dir.mkdir()
        result = self.join()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("exists; remove what a previous join left there", result.stderr)
        self.assertEqual(list(self.key_dir.iterdir()), [])

    def test_the_tool_installs_nothing_that_runs_at_boot(self):
        source = TOOL.read_text()
        for word in ("systemctl", "enable", "crontab", "rc.local", "/etc/systemd"):
            self.assertNotIn(word, source)
        self.assertTrue(os.access(TOOL, os.X_OK))
        self.assertTrue(stat.S_IMODE(TOOL.stat().st_mode) & stat.S_IXUSR)


if __name__ == "__main__":
    unittest.main()
