# SparkPipe Agent Instructions

## Firmware contract

- Follow [SparkPipe invariants](sparkpipe_invariants.md). Required behavior
  cannot be waived by driver flags, stubs or compatibility paths.
- No comments in code. Put explanations in documentation and PRs. Tests must
  verify behavior and contracts, not comment wording.
- Shared code is model-neutral: a generic function, type, macro or file
  carries a generic name. Model and driver names belong only under that
  model's `model-families/<model>/`, `modules/<driver>/`, tools and tests.
  `tests/test_dry_law.py` enforces this for the runtime, transport, cache,
  serving, kernel and common paths; its `PENDING` list may only shrink.

## Workflow

- Claude is the lead developer. The user merges pull requests by hand, after
  Claude reports a PR ready. A PR may stack on another open PR; say so in
  its description.
- Fleet host roles (the hub, the build host, the Sparks), releases and
  operations are in [`docs/FLEET_RELEASE_RUNBOOK.md`](docs/FLEET_RELEASE_RUNBOOK.md).
  Read it before touching a fleet host. Deploy only by publishing a release
  root; never kill and reload daemons by hand, and never kill anything on
  the hub.

## Source package manifests

- `PACKAGE_MANIFEST.json` and `SHA256SUMS` at the repository root are
  generated for a source package and never committed, so PRs do not
  conflict on them. `tools/source_package_gate.sh [revision]` archives the
  revision (default `HEAD`), generates both manifests inside the archive and
  verifies them with the forbidden-path policy. The CUDA gate, `tools/gates.sh`
  and `make offline-gates` run it.

## GitHub authentication

- Never open a GitHub login flow, request a connector, call `gh auth login`, or
  rely on the active `gh` account for this repository.
- Run every GitHub-facing `git` or `gh` command through
  `tools/sparkpipe_github_pat.sh`. The wrapper reads `GITHUB_PAT` from
  `$SPARKPIPE_PAT_FILE` or, when that is unset, from the first readable
  `sparkpipe/.env` in its candidate list; on this workstation that is
  `/Users/mac/sparkpipe/.env`.
- The wrapper deliberately clears Git credential helpers. This is mandatory:
  the workstation credential helper may otherwise replace the SparkPipe PAT
  with the cached `experiencenow-ai` credential before askpass runs.
- Use `origin` (`https://github.com/sparkpipe/sparkpipe`) as the official push
  and PR target. Do not silently fall back to an `experiencenow-ai` fork.
- Never print the PAT, place it in a command argument, store it in a Git remote,
  or commit it. If the wrapper cannot authenticate, fail loudly and report the
  command error.

Before the first write operation in a task, verify the effective identity:

```sh
tools/sparkpipe_github_pat.sh gh api user --jq .login
```

The expected result is `sparkpipe`.
