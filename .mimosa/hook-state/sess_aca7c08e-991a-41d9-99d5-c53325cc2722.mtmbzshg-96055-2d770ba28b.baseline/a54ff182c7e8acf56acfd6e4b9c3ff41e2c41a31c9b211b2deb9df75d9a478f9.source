# SparkPipe Agent Instructions

## GitHub authentication

- Never open a GitHub login flow, request a connector, call `gh auth login`, or
  rely on the active `gh` account for this repository.
- Run every GitHub-facing `git` or `gh` command through
  `tools/sparkpipe_github_pat.sh`. The wrapper reads `GITHUB_PAT` directly from
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
