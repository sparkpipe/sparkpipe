# Host-test fixture processes

Host tests that start `sparkpipe_model_residentd`, `sparkpipe_model_api`,
`sparkpipe_model_batch`, `sparkpipe_weightd` or weightd clients as child
processes must not leave them running when the test fails, is killed, or its
build directory is deleted. Before this rule, a failed `test_model_api_text`,
`test_steploop_admission` or `test_model_pipeline_client` left its fixture
residentds running on sparkf for hours with a deleted working directory.

## The guard

Every such test forks through `TestChildGuardFork()` from
`tests/fixtures/test_child_guard.c` instead of `fork()`. The guard:

- puts each child in its own process group, so cleanup reaches anything the
  child started;
- sets `PR_SET_PDEATHSIG=SIGKILL` in the child on Linux, so the child dies
  even when the test is killed with `SIGKILL` (for example by `timeout -s
  KILL` or an OOM kill);
- on the first fork, registers an `atexit` hook and handlers for every
  signal whose default action ends the process: `SIGABRT` (a failed
  `assert`), `SIGSEGV`, `SIGBUS`, `SIGILL`, `SIGFPE`, `SIGTRAP` (what
  `__builtin_trap` raises on aarch64), `SIGSYS`, `SIGTERM`, `SIGINT`,
  `SIGHUP`, `SIGQUIT`, `SIGPIPE` (a closed output pipe, for example a dropped
  ssh session), `SIGALRM`, `SIGVTALRM`, `SIGPROF`, `SIGXCPU`, `SIGXFSZ`,
  `SIGUSR1` and `SIGUSR2`. Each hook kills every tracked child group with
  `SIGKILL`, reaps it, and then lets the original signal terminate the test.
  It installs a handler only where the test left the default action;
- on the first fork on Linux, sets `PR_SET_PDEATHSIG=SIGTERM` on the test
  itself, unless something already set a parent-death signal for it. When
  the process that started the test dies, the test gets `SIGTERM` and the
  handler above reaps its children. This matters for `make`: on `SIGTERM`,
  GNU make signals only its recipe shell, and the shell dies without
  signalling the test it is waiting for. Without this rule the test went on
  running as an orphan, and if it was stuck its residentds stayed with it.
  If the parent dies between the guard reading it and setting the signal,
  the test raises `SIGTERM` itself. A test whose parent exited before that
  first fork (for example `nohup ./build/test_x &` from a shell that then
  exits) is already an orphan, so the rule does not apply to it;
- exports `SPARK_TEST_FIXTURE_OWNER=<test pid>.<test start ticks>` into the
  environment its children inherit.

The guard never kills a pid the test has already reaped, so a recycled pid is
never signalled. A child that forks again (a grandchild) is covered by the
group kill, not by `PDEATHSIG`.

`tests/test_child_guard.c` proves this. It runs a victim test that starts
exec'd children and then fails by `assert`, `SIGSEGV`, `SIGTERM`, `SIGINT`,
`SIGHUP`, `SIGKILL` (Linux) and `exit(3)`, plus a variant with a grandchild.
Before each failure it checks that the children are alive, in their own
group, and marked with the victim's pid. Afterwards it requires every child to
be gone within 5 s. A control run with a plain `fork()` requires the children
to survive, which shows that the check can see a leak.

`tests/test_fixture_reaping.py` proves it with the real tests and real
residentds on Linux. It runs `build/test_steploop_admission` and
`build/test_model_api_text`, stops the test with `SIGSTOP` once a
residentd child has exec'd, and checks that each residentd is in its own
group, carries the owner marker and names the test's
`/tmp/sparkpipe-steploop-<pid>.json` or `/tmp/sparkpipe-api-text-<pid>.json`
deployment. Then it ends the test in one of these ways:

- `SIGABRT`, `SIGTERM`, `SIGINT`, `SIGHUP` and `SIGPIPE`. The test must die
  by that signal, and its residentds must already be reaped by the test, not
  left to `PDEATHSIG`;
- `SIGKILL`. Every residentd must be gone within 5 s through `PDEATHSIG`;
- `SIGTERM` to a real `make` whose recipe runs the test the way `make test`
  does (`set -e; for ...; do ./$test; done`). The test must die by `SIGTERM`
  instead of running to completion as an orphan, and its residentds must be
  reaped;
- for steploop only, `SIGKILL` to one residentd. The test then fails an
  `assert`, and the remaining residentds must be reaped. In
  `test_model_api_text` a dead rank is a supported state: model_api retries
  the engine with backoff and the test waits up to 60 s for health, so that
  path is not exercised there;
- a normal pass. The test must exit 0 with no fixture left.

After every case no live process may carry the test's deployment path or its
owner marker. The test makes itself a child subreaper, so orphans reparent
to it and it can read the test's exit status after the runner is gone. It
deletes the `/tmp/sparkpipe-*-<pid>` files of the tests it ran.
Against the guard before the runner and `SIGPIPE` rules, the `make` and
`SIGPIPE` cases fail for both tests: the test exits 0 after its runner died,
and the residentds outlive the test.

## Listing strays

`make stray-fixtures` (or `python3 tools/stray_fixture_processes.py`) reads
`/proc` and lists:

- marked processes whose owning test is gone (pid missing, a zombie, or a
  different start time);
- unmarked serving binaries from older tests: a residentd, API, batch or
  weightd binary whose working directory is deleted or whose arguments name
  a `/tmp/sparkpipe-*` deployment, and whose parent is not a live `test_*`
  binary.

It exits 0 when there are no strays, 1 when strays are listed (with a
`kill -KILL` line for them), and 2 on a host without `/proc`. `--all` also
shows the fixtures of tests that are still running. `STRAY_UNDER=<dir>`
(`--under`) limits the list to processes whose exe or cwd is under a
directory, for example your `~/build-<lane>`. Production and lane engines
started from roots are never listed: they carry no marker, their cwd exists,
and their deployment is not under `/tmp/sparkpipe-`.

The tool only lists processes. Kill a stray only if your lane started it.

## The 2026-10-15 hub orphans

On 2026-10-15 the hub had twelve `sparkpipe_model_residentd` processes with
parent pid 1, about 17 days old, each holding about 500 MiB of GPU memory.
They ran with `/tmp/sparkpipe-steploop-<pid>.json` and
`/tmp/sparkpipe-api-text-<pid>.json` deployments from the build trees
`train1-main-edc5e83`, `train1-47b8b77`, `build-jitkv/fix4` and
`wf-tests-audit-runtime-serving-neg`. `edc5e83` and `47b8b77` (2026-09-28)
predate the guard (2026-09-29), so their tests forked residentds with a plain
`fork()`: no process group, no `PDEATHSIG` and no handlers. The other two
trees could not be dated. A tree built before the guard still leaks this way,
so rebuild old trees before running their host tests, and use
`make stray-fixtures` to find leftovers.
