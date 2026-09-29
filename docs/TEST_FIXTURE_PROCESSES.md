# Host-test fixture processes

Host tests that start `sparkpipe_model_residentd`, `sparkpipe_model_api`,
`sparkpipe_model_batch` or weightd clients as child processes must not leave
them running when the test fails, is killed, or its build directory is
deleted. Before this rule, a failed `test_model_api_text`,
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
- on the first fork, registers an `atexit` hook and handlers for `SIGABRT`
  (a failed `assert`), `SIGSEGV`, `SIGBUS`, `SIGILL`, `SIGFPE`, `SIGTERM`,
  `SIGINT`, `SIGHUP`, `SIGQUIT`, `SIGALRM`, `SIGXCPU`, `SIGUSR1` and
  `SIGUSR2`. Each hook kills every tracked child group with `SIGKILL`, reaps
  it, and then lets the original signal terminate the test. It installs a
  handler only where the test left the default action;
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
