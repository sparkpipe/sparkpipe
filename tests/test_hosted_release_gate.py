import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tools" / "hosted_release_gate.py"
SPEC = importlib.util.spec_from_file_location("hosted_release_gate", GATE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
AUDIT = MODULE.load_audit()
RELEASE_BUILD = ROOT / "scripts" / "release_build.sh"
CUDA_STUB = ROOT / "tests" / "cuda_stub"
HOST_ONLY_PROGRAMS = ("build/sparkpipe_model_api", "build/sparkpipe_model_batch", "build/sparkpipe_tokenize_prompt", "build/weightd_receipt")
BUILD_INPUT_FILES = ("Makefile", "sources.mk")
BUILD_INPUT_DIRECTORIES = ("cache", "config", "deployment", "include", "inference", "model-families", "model_contracts",
                           "modules", "node", "ring", "runtime", "schema", "src", "tests", "text", "tools")
GIT_IDENTITY = {"GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}


def compiler_links_elf():
    compiler = shutil.which("cc")
    if compiler is None:
        return False
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory, "probe.c")
        source.write_text("int spark_probe(void) { return 0; }\n")
        output = Path(directory, "libprobe.so")
        built = subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(output)], capture_output=True)
        return built.returncode == 0 and output.read_bytes()[:4] == b"\x7fELF"


ELF_HOST = compiler_links_elf()
needs_elf = unittest.skipUnless(ELF_HOST, "the host cc does not link ELF objects")


def make_words(variable, *assignments, directory=ROOT):
    with tempfile.NamedTemporaryFile("w", suffix=".mk", delete=False) as printer:
        printer.write("print-%:\n\t@echo $($*)\n")
    try:
        completed = subprocess.run(
            ["make", "-s", "--no-print-directory", "-C", str(directory), "-f", "Makefile", "-f", printer.name,
             f"print-{variable}", *assignments], capture_output=True, text=True, check=True)
    finally:
        os.unlink(printer.name)
    return completed.stdout.split()


def fake_toolkit(directory, libraries=False):
    include = Path(directory, "include")
    include.mkdir(parents=True)
    for header in ("cuda.h", "cuda_runtime.h", "cuda_runtime_api.h"):
        shutil.copy2(CUDA_STUB / header, include / header)
    if libraries:
        lib64 = Path(directory, "lib64")
        lib64.mkdir()
        for name, soname in (("libcuda.so", "libcuda.so.1"), ("libcudart.so", "libcudart.so.13")):
            subprocess.run(["cc", "-shared", "-fPIC", f"-Wl,-soname,{soname}", f"-I{ROOT}", f"-I{ROOT / 'include'}", f"-I{CUDA_STUB}",
                            f"-I{ROOT / 'model-families' / 'common' / 'include'}", str(CUDA_STUB / "cuda_runtime_stub.c"),
                            "-o", str(lib64 / name)], check=True)
    return Path(directory)


def write(path, text, mode=0o755):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


def debt(files, reason="Fixture reason that names why the file needs the library."):
    return {"schema_version": 1, "debt": [{"reason": reason, "files": files}]}


def run_check(root, document, complete=False, deploy=False):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        failures = MODULE.check(str(root), document, complete, deploy)
    return failures, output.getvalue()


class DebtListTest(unittest.TestCase):
    def setUp(self):
        with open(MODULE.DEBT_PATH, "r", encoding="utf-8") as source:
            self.document = json.load(source)

    def test_committed_debt_list_is_valid_and_names_only_shipped_files(self):
        self.assertEqual(MODULE.debt_errors(self.document), [])
        shipped = set(make_words("HOSTED_RELEASE_FILES", "SHARED_LIBRARY_EXT=so"))
        listed = {path for group in self.document["debt"] for path in group["files"]}
        self.assertTrue(listed)
        self.assertEqual(listed - shipped, set())

    def test_the_gate_stages_each_shipped_adapter_with_its_kernels_directory(self):
        adapters = make_words("HOSTED_RELEASE_ADAPTERS", "SHARED_LIBRARY_EXT=so")
        self.assertTrue(adapters)
        self.assertEqual(set(adapters) - set(make_words("HOSTED_RELEASE_FILES", "SHARED_LIBRARY_EXT=so")), set())
        self.assertEqual(make_words("HOSTED_RELEASE_KERNELS", "SHARED_LIBRARY_EXT=so"), [adapter + ".kernels" for adapter in adapters])
        chosen = ["build/sparkpipe_model_api", adapters[0]]
        self.assertEqual(make_words("HOSTED_RELEASE_KERNELS", "SHARED_LIBRARY_EXT=so", "HOSTED_RELEASE_FILES=" + " ".join(chosen)),
                         [adapters[0] + ".kernels"])
        recipe = subprocess.run(["make", "-s", "-n", "-C", str(ROOT), "hosted-release-gate", "HOSTED_RELEASE_FILES=tools/fleet_node_agent.sh",
                                 "HOSTED_RELEASE_ADAPTERS=tools/fleet_node_agent.sh"], capture_output=True, text=True, check=True).stdout
        self.assertIn("--stage tools/fleet_node_agent.sh tools/fleet_node_agent.sh.kernels\n", recipe)

    def test_nvrm_builds_stage_under_the_paths_the_release_ships(self):
        shipped = set(make_words("HOSTED_RELEASE_FILES", "SHARED_LIBRARY_EXT=so"))
        nvrm = make_words("HOSTED_RELEASE_NVRM_FILES", "SHARED_LIBRARY_EXT=so")
        listed = {path for group in self.document["debt"] for path in group["files"]}
        self.assertEqual({"build/sparkpipe_weightd", "build/weightd_warm", "build/weightdctl"} - set(nvrm), set())
        self.assertEqual(set(nvrm) - shipped, set())
        self.assertEqual(set(nvrm) & listed, set())
        chosen = nvrm + ["tools/fleet_node_agent.sh"]
        self.assertEqual(make_words("HOSTED_RELEASE_SOURCES", "SHARED_LIBRARY_EXT=so", "HOSTED_RELEASE_FILES=" + " ".join(chosen)),
                         [path.replace("build/", "build/nvrm/", 1) for path in nvrm] + ["tools/fleet_node_agent.sh"])
        recipe = subprocess.run(["make", "-s", "-n", "-C", str(ROOT), "hosted-release-gate", "HOSTED_RELEASE_FILES=" + " ".join(chosen),
                                 "HOSTED_RELEASE_ADAPTERS=build/weightdctl"], capture_output=True, text=True, check=True).stdout
        staged = " ".join([f"{path.replace('build/', 'build/nvrm/', 1)}={path}" for path in nvrm] + ["tools/fleet_node_agent.sh",
                          "build/nvrm/weightdctl.kernels=build/weightdctl.kernels"])
        self.assertIn(f"--stage {staged}\n", recipe)

    def test_host_only_programs_are_shipped_and_never_debt(self):
        shipped = set(make_words("HOSTED_RELEASE_FILES", "SHARED_LIBRARY_EXT=so"))
        host_only = set(make_words("HOSTED_RELEASE_HOST_ONLY_FILES", "SHARED_LIBRARY_EXT=so"))
        listed = {path for group in self.document["debt"] for path in group["files"]}
        self.assertEqual(set(HOST_ONLY_PROGRAMS) | {"tools/fleet_node_agent.sh"}, host_only)
        self.assertEqual(host_only - shipped, set())
        self.assertEqual(host_only & listed, set())

    def test_validation_refuses_patterns_loose_paths_and_unexplained_entries(self):
        good = debt({"build/a.so": ["needed libcuda.so.1"]})
        self.assertEqual(MODULE.debt_errors(good), [])
        cases = {
            "pattern": debt({"build/*.so": ["needed libcuda.so.1"]}),
            "class": debt({"build/lib[ab].so": ["needed libcuda.so.1"]}),
            "absolute": debt({"/build/a.so": ["needed libcuda.so.1"]}),
            "parent": debt({"../a.so": ["needed libcuda.so.1"]}),
            "unnormalized": debt({"build//a.so": ["needed libcuda.so.1"]}),
            "directory": debt({"build/": ["needed libcuda.so.1"]}),
            "short reason": debt({"build/a.so": ["needed libcuda.so.1"]}, reason="needs it"),
            "no findings": debt({"build/a.so": []}),
            "unsorted findings": debt({"build/a.so": ["needed libcudart.so.13", "needed libcuda.so.1"]}),
            "repeated finding": debt({"build/a.so": ["needed libcuda.so.1", "needed libcuda.so.1"]}),
            "unparsed kind": debt({"build/a.so": ["unparsed bad header"]}),
            "empty kind": debt({"build/a.so": ["empty nothing to audit"]}),
            "no detail": debt({"build/a.so": ["needed"]}),
            "unsorted files": debt({"build/b.so": ["needed libcuda.so.1"], "build/a.so": ["needed libcuda.so.1"]}),
        }
        for name, document in cases.items():
            with self.subTest(name):
                self.assertTrue(MODULE.debt_errors(document))
        twice = debt({"build/a.so": ["needed libcuda.so.1"]})
        twice["debt"].append({"reason": "Another fixture reason that is long enough.", "files": {"build/a.so": ["needed libcudart.so.13"]}})
        self.assertTrue(any("already listed" in error for error in MODULE.debt_errors(twice)))
        repeated = copy.deepcopy(good)
        repeated["debt"].append(copy.deepcopy(good["debt"][0]))
        self.assertTrue(any("repeats the reason" in error for error in MODULE.debt_errors(repeated)))
        self.assertTrue(MODULE.debt_errors({"schema_version": 2, "debt": []}))
        self.assertTrue(MODULE.debt_errors({"schema_version": 1, "debt": [], "patterns": ["*"]}))


class LinkFlagsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.toolkit = fake_toolkit(Path(self.directory.name, "cuda"))

    def test_global_link_flags_name_no_device_library_on_a_cuda_host(self):
        home = f"CUDA_HOME={self.toolkit}"
        self.assertIn("-lcuda", make_words("SPARKPIPE_CUDA_DRIVER_LINK", home))
        self.assertIn("-lcudart", make_words("SPARKPIPE_CUDA_RUNTIME_LINK", home))
        flags = make_words("LDFLAGS", home) + make_words("LDLIBS", home)
        self.assertEqual([flag for flag in flags if "cuda" in flag], [])

    def test_host_only_link_lines_name_no_device_library_and_weightd_tools_only_the_runtime(self):
        host_only = list(HOST_ONLY_PROGRAMS)
        weightd = ["build/weightd_warm", "build/weightdctl"]
        completed = subprocess.run(["make", "-n", "-B", "-C", str(ROOT), f"CUDA_HOME={self.toolkit}", *host_only, *weightd],
                                   capture_output=True, text=True, check=True)
        links = {}
        for line in completed.stdout.splitlines():
            match = re.search(r"\s-o\s+(\S+)\s*$", line)
            if match and " -c " not in f" {line} ":
                links[match.group(1)] = line
        for target in host_only:
            with self.subTest(target):
                self.assertIn(target, links)
                self.assertNotRegex(links[target], r"-lcuda\b|-lcudart\b|lib64")
        for target in weightd:
            with self.subTest(target):
                self.assertRegex(links[target], r"-lcudart\b")
                self.assertNotRegex(links[target], r"-lcuda\b")


class GateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.root = self.base / "root"
        write(self.root / "bin" / "probe.sh", "#!/bin/sh\nnvidia-smi -q\n")
        write(self.root / "bin" / "clean.sh", "#!/bin/sh\necho ready\n")

    def test_listed_finding_passes_and_an_unlisted_one_fails(self):
        failures, output = run_check(self.root, debt({"bin/probe.sh": ["tool-string nvidia-smi"]}))
        self.assertEqual(failures, [])
        self.assertIn(": PASS (2 scanned, 1 findings on the debt list in 1 files, 0 failures)", output)
        failures, output = run_check(self.root, debt({"bin/other.sh": ["tool-string nvidia-smi"]}))
        self.assertEqual(failures, ["FAIL new tool-string bin/probe.sh: nvidia-smi"])
        self.assertIn(": FAIL", output)

    def test_a_listed_finding_that_is_gone_is_stale_and_must_be_removed(self):
        document = debt({"bin/clean.sh": ["tool-string nvidia-smi"], "bin/probe.sh": ["tool-string nvidia-smi"]})
        failures, _ = run_check(self.root, document)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("FAIL stale bin/clean.sh: tool-string nvidia-smi is gone"))

    def test_a_listed_file_the_root_lacks_fails_only_a_complete_root(self):
        document = debt({"bin/absent.so": ["needed libcuda.so.1"], "bin/probe.sh": ["tool-string nvidia-smi"]})
        self.assertEqual(run_check(self.root, document)[0], [])
        failures, _ = run_check(self.root, document, complete=True)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("FAIL missing bin/absent.so"))

    def test_an_empty_root_fails_whatever_the_debt_list_says(self):
        empty = self.base / "empty"
        empty.mkdir()
        failures, _ = run_check(empty, debt({"bin/probe.sh": ["tool-string nvidia-smi"]}))
        self.assertEqual(failures, ["FAIL new empty .: nothing to audit"])

    @needs_elf
    def test_needed_entries_must_match_the_list_exactly(self):
        libraries = self.base / "libraries"
        libraries.mkdir()
        source = write(libraries / "stub.c", "int spark_fixture_symbol(void) { return 1; }\n", 0o644)
        subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-soname,libcuda.so.1", str(source), "-o", str(libraries / "libcuda.so")], check=True)
        program = write(self.base / "program.c", "int main(void) { return 0; }\n", 0o644)
        (self.root / "lib").mkdir()
        subprocess.run(["cc", str(program), "-o", str(self.root / "lib" / "engine"), "-Wl,--no-as-needed",
                        f"-L{libraries}", "-lcuda"], check=True)
        listed = {"bin/probe.sh": ["tool-string nvidia-smi"], "lib/engine": ["needed libcuda.so.1"]}
        self.assertEqual(run_check(self.root, debt(listed))[0], [])
        listed["lib/engine"] = ["needed libcudart.so.13"]
        failures, _ = run_check(self.root, debt(listed))
        self.assertEqual(failures[0], "FAIL new needed lib/engine: libcuda.so.1")
        self.assertTrue(failures[1].startswith("FAIL stale lib/engine: needed libcudart.so.13 is gone"))

    def test_stage_copies_exact_paths_and_refuses_anything_else(self):
        project = self.base / "project"
        write(project / "build" / "tool", "#!/bin/sh\necho ready\n")
        write(project / "tools" / "agent.sh", "#!/bin/sh\nnvidia-smi\n")
        write(project / "build" / "libmodule.so.kernels" / "unit.cubin", "image\n", 0o644)
        write(project / "build" / "libmodule.so.kernels" / "unit.manifest", "kernel unit\n", 0o644)
        write(project / "build" / "linked.kernels" / "unit.cubin", "image\n", 0o644)
        os.symlink("unit.cubin", project / "build" / "linked.kernels" / "alias.cubin")
        (project / "build" / "empty.kernels" / "nested").mkdir(parents=True)
        os.symlink("tool", project / "build" / "alias")
        staged = project / "staged"
        write(staged / "leftover", "old\n")
        shipped = {"build/tool", "tools/agent.sh", "build/libmodule.so.kernels/unit.cubin", "build/libmodule.so.kernels/unit.manifest"}
        cwd = os.getcwd()
        os.chdir(project)
        try:
            MODULE.stage("staged", ["build/tool", "tools/agent.sh", "build/libmodule.so.kernels"])
            self.assertEqual(MODULE.root_files("staged"), shipped)
            self.assertEqual(os.stat("staged/build/tool").st_mode & 0o777, 0o755)
            refused = (["build/*"], ["/etc/hostname"], ["../project/build/tool"], ["build/absent"], [],
                       ["build/alias"], ["build/libmodule.so.kernels/"], ["build/linked.kernels"], ["build/empty.kernels"])
            for sources in refused:
                with self.subTest(sources):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        MODULE.stage("staged", ["build/tool", *sources] if sources else sources)
                    self.assertEqual(raised.exception.code, 2)
            self.assertEqual(MODULE.root_files("staged"), shipped)
        finally:
            os.chdir(cwd)

    def test_stage_puts_a_source_under_another_exact_path(self):
        project = self.base / "project"
        write(project / "build" / "nvrm" / "tool", "#!/bin/sh\necho nvrm\n")
        write(project / "build" / "tool", "#!/bin/sh\necho cuda\n")
        write(project / "build" / "nvrm" / "libmodule.so.kernels" / "unit.cubin", "image\n", 0o644)
        cwd = os.getcwd()
        os.chdir(project)
        try:
            MODULE.stage("staged", ["build/nvrm/tool=build/tool", "build/nvrm/libmodule.so.kernels=build/libmodule.so.kernels"])
            self.assertEqual(MODULE.root_files("staged"), {"build/tool", "build/libmodule.so.kernels/unit.cubin"})
            self.assertEqual(Path("staged/build/tool").read_text(), "#!/bin/sh\necho nvrm\n")
            refused = (["build/nvrm/tool=build/nvrm/tool"], ["build/nvrm/tool=/usr/bin/tool"], ["build/nvrm/tool=build/*"],
                       ["build/nvrm/tool=build/tool=x"], ["build/nvrm/tool=build/tool", "build/tool"],
                       ["build/nvrm/tool=build/tool", "build/nvrm/tool=build/tool"], ["build/absent=build/tool"])
            for sources in refused:
                with self.subTest(sources):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        MODULE.stage("staged", sources)
                    self.assertEqual(raised.exception.code, 2)
            self.assertEqual(MODULE.root_files("staged"), {"build/tool", "build/libmodule.so.kernels/unit.cubin"})
        finally:
            os.chdir(cwd)

    def test_deploy_refuses_a_root_while_the_debt_list_names_any_file(self):
        listed = debt({"bin/probe.sh": ["tool-string nvidia-smi"]})
        self.assertEqual(run_check(self.root, listed)[0], [])
        failures, output = run_check(self.root, listed, deploy=True)
        self.assertEqual(failures, [f"FAIL deploy bin/probe.sh: {MODULE.DEBT_RELATIVE} still lists it, and a root that ships debt is not deployed"])
        self.assertIn(": FAIL", output)
        clean = self.base / "clean"
        write(clean / "bin" / "clean.sh", "#!/bin/sh\necho ready\n")
        self.assertEqual(run_check(clean, {"schema_version": 1, "debt": []}, complete=True, deploy=True)[0], [])

    @needs_elf
    def test_deploy_refuses_a_root_that_mixes_device_backends(self):
        libraries = self.base / "libraries"
        libraries.mkdir()
        source = write(libraries / "device.c", "int spark_device_fixture(void) { return 1; }\n", 0o644)
        subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-soname,libsparkpipe_device_cuda.so", str(source),
                        "-o", str(libraries / "libsparkpipe_device_cuda.so")], check=True)
        program = write(self.base / "program.c", "int main(void) { return 0; }\n", 0o644)
        control = write(self.base / "control.c", 'const char *volatile spark_control = "/dev/nvidiactl";\n'
                        "int main(void) { return spark_control[0] == 0; }\n", 0o644)
        root = self.base / "mixed"
        (root / "build").mkdir(parents=True)
        subprocess.run(["cc", str(program), "-o", str(root / "build" / "cuda_engine"), "-Wl,--no-as-needed", f"-L{libraries}",
                        "-lsparkpipe_device_cuda"], check=True)
        subprocess.run(["cc", str(control), "-o", str(root / "build" / "nvrm_engine")], check=True)
        write(root / "build" / "agent.sh", "#!/bin/sh\nls /dev/nvidiactl\n")
        empty = {"schema_version": 1, "debt": []}
        self.assertEqual(run_check(root, empty)[0], [])
        failures, _ = run_check(root, empty, deploy=True)
        self.assertEqual(failures, ["FAIL deploy build/cuda_engine: built for the cuda device backend in a root that mixes cuda and nvrm",
                                    "FAIL deploy build/nvrm_engine: built for the nvrm device backend in a root that mixes cuda and nvrm"])
        (root / "build" / "cuda_engine").unlink()
        self.assertEqual(run_check(root, empty, deploy=True)[0], [])

    def test_command_line_stages_checks_and_exits_with_the_verdict(self):
        project = self.base / "project"
        write(project / "build" / "tool", "#!/bin/sh\nnvidia-smi\n")
        listed = project / "debt.json"
        listed.write_text(json.dumps(debt({"build/tool": ["tool-string nvidia-smi"]})))
        command = [sys.executable, str(GATE), "check", "--debt", str(listed), "--complete", "root", "--stage", "build/tool"]
        completed = subprocess.run(command, cwd=project, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        listed.write_text(json.dumps(debt({"build/other": ["tool-string nvidia-smi"]})))
        completed = subprocess.run(command, cwd=project, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("FAIL new tool-string build/tool: nvidia-smi", completed.stdout)
        self.assertIn("FAIL missing build/other", completed.stdout)
        listed.write_text(json.dumps(debt({"build/tool": ["tool-string nvidia-smi"]})))
        completed = subprocess.run(command[:3] + ["--deploy"] + command[3:], cwd=project, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("FAIL deploy build/tool:", completed.stdout)
        listed.write_text(json.dumps(debt({"build/other": ["tool-string nvidia-smi"]})))
        findings = subprocess.run([sys.executable, str(GATE), "findings", "--debt", str(listed), "root"],
                                  cwd=project, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(findings.stdout)["debt"], [{"reason": "", "files": {"build/tool": ["tool-string nvidia-smi"]}}])


class ShrinkTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repository = Path(self.directory.name)
        self.git("init", "-q")
        write(self.repository / "README", "fixture\n", 0o644)
        self.commit("start")
        self.start = self.head()
        self.listed = {"build/a.so": ["needed libcuda.so.1", "needed libcudart.so.13"], "build/b": ["needed libcudart.so.13"]}
        self.write_debt(self.listed)
        self.commit("debt")
        self.base = self.head()

    def git(self, *arguments):
        return subprocess.run(["git", "-c", "commit.gpgsign=false", "-C", str(self.repository), *arguments], capture_output=True, text=True, check=True,
                              env={**os.environ, **GIT_IDENTITY}).stdout.strip()

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def head(self):
        return self.git("rev-parse", "HEAD")

    def write_debt(self, files):
        path = self.repository / MODULE.DEBT_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(debt(files)))

    def shrink(self, base):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = MODULE.shrink(base, str(self.repository))
        return code, output.getvalue()

    def test_removing_entries_passes_and_adding_any_fails(self):
        self.write_debt({"build/a.so": ["needed libcuda.so.1", "needed libcudart.so.13"]})
        self.assertEqual(self.shrink(self.base)[0], 0)
        self.write_debt({"build/a.so": ["needed libcuda.so.1"], "build/b": ["needed libcudart.so.13"]})
        self.assertEqual(self.shrink(self.base)[0], 0)
        self.write_debt({**self.listed, "build/c": ["needed libcuda.so.1"]})
        code, output = self.shrink(self.base)
        self.assertEqual(code, 1)
        self.assertIn("FAIL grew build/c: needed libcuda.so.1", output)
        self.write_debt({**self.listed, "build/b": ["needed libcuda.so.1", "needed libcudart.so.13"]})
        code, output = self.shrink(self.base)
        self.assertEqual(code, 1)
        self.assertIn("FAIL grew build/b: needed libcuda.so.1", output)

    def test_the_change_that_introduces_the_list_passes_and_an_unknown_base_is_refused(self):
        code, output = self.shrink(self.start)
        self.assertEqual(code, 0)
        self.assertIn("introduces it with 3 findings", output)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.shrink("0" * 40)
        self.assertEqual(raised.exception.code, 2)


class ReleaseKitTest(unittest.TestCase):
    def test_release_build_ships_exactly_the_gated_node_files(self):
        text = RELEASE_BUILD.read_text()
        node = re.search(r'^NODE="([^"]+)"$', text, re.M)
        self.assertIsNotNone(node)
        self.assertEqual(node.group(1).split(), make_words("HOSTED_RELEASE_NODE_FILES", "SHARED_LIBRARY_EXT=so"))
        hub = re.search(r'^HUB_FILES="([^"]+)"$', text, re.M)
        self.assertIsNotNone(hub)
        self.assertEqual(set(hub.group(1).split()), set(make_words("HOSTED_RELEASE_HUB_FILES")) | {"build/runtime_libs/libsparkpipe_device_cuda.so"})

    def test_hosted_mode_gates_both_builds_for_deployment_and_ships_the_gated_root(self):
        text = RELEASE_BUILD.read_text()
        block = re.search(r'^FW_DIR=build\nif \[ "\$\{HOSTED:-0\}" = 1 \]; then\n(.*?)^fi$', text, re.M | re.S)
        self.assertIsNotNone(block)
        lines = block.group(1).splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("ssh -o BatchMode=yes $HOST", lines[0])
        self.assertIn("make hosted-release-gate HOSTED_RELEASE_FILES='$NODE $SO' HOSTED_RELEASE_GATE_FLAGS=--deploy", lines[0])
        hub = re.search(r'^ssh -o BatchMode=yes (\S+) "rm -rf \$HUB_WORK/api-build-\$SHA ', text, re.M)
        self.assertIsNotNone(hub)
        self.assertIn(f'ssh -o BatchMode=yes {hub.group(1)} "cd $HUB_WORK/api-build-$SHA ', lines[1])
        self.assertIn("make hosted-release-gate HOSTED_RELEASE_FILES='$HUB_FILES $SO' HOSTED_RELEASE_GATE_FLAGS=--deploy", lines[1])
        for line in lines[:2]:
            self.assertTrue(line.rstrip().endswith("exit 1; }"))
        self.assertEqual(lines[2].strip(), "FW_DIR=" + make_words("HOSTED_RELEASE_ROOT")[0] + "/build")
        self.assertLess(block.start(), text.index("rm -rf $OUT"))
        shipped = [line for line in text.splitlines() if line.startswith(("MISSING=", "rm -rf $OUT"))]
        self.assertEqual(len(shipped), 2)
        for line in shipped:
            self.assertIn('"cd ~/build-rp/fw-$SHA-src/$FW_DIR && ', line)


@unittest.skipUnless(ELF_HOST and sys.platform.startswith("linux") and shutil.which("make"), "builds ELF programs with GNU make on Linux")
class HostOnlyBuildTest(unittest.TestCase):
    def test_host_only_programs_link_no_device_library_on_a_cuda_host(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            for name in BUILD_INPUT_FILES:
                shutil.copy2(ROOT / name, source / name)
            for name in BUILD_INPUT_DIRECTORIES:
                shutil.copytree(ROOT / name, source / name, symlinks=True,
                                ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc"))
            toolkit = fake_toolkit(base / "cuda", libraries=True)
            host_only = make_words("HOSTED_RELEASE_HOST_ONLY_FILES", directory=source)
            built = [path for path in host_only if path.startswith("build/")]
            weightd = ["build/weightd_warm", "build/weightdctl"]
            if Path("/usr/include/infiniband/verbs.h").is_file():
                weightd.append("build/sparkpipe_weightd")
            nvrm_weightd = [path.replace("build/", "build/nvrm/", 1) for path in weightd]
            completed = subprocess.run(["make", "-C", str(source), f"-j{os.cpu_count() or 2}", f"CUDA_HOME={toolkit}",
                                        "CC=cc -Wl,--no-as-needed", *built, *weightd, *nvrm_weightd], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout[-4000:] + completed.stderr[-4000:])
            cwd = os.getcwd()
            os.chdir(source)
            try:
                MODULE.stage(str(base / "host_only"), host_only)
                MODULE.stage(str(base / "weightd"), weightd)
                MODULE.stage(str(base / "nvrm_weightd"), nvrm_weightd)
            finally:
                os.chdir(cwd)
            report = AUDIT.audit_root(str(base / "host_only"))
            self.assertEqual(report.scanned, len(host_only))
            self.assertEqual(report.findings(), [])
            weightd_report = AUDIT.audit_root(str(base / "weightd"))
            for program in weightd:
                found = {(kind, detail) for path, kind, detail in weightd_report.found if path == program}
                self.assertEqual(found, {("needed", "libcudart.so.13")}, program)
            nvrm_report = AUDIT.audit_root(str(base / "nvrm_weightd"))
            self.assertEqual(nvrm_report.scanned, len(nvrm_weightd))
            self.assertEqual(nvrm_report.findings(), [])


if __name__ == "__main__":
    unittest.main()
