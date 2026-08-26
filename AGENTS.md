# SparkPipe Agent Instructions

The non-negotiable execution rules are in
`orchestration/OPERATING_INVARIANTS.md`. Treat any violation as a scheduler or
harness bug: expose and fix it; never hide it with manual assignments, counters,
or status labels.

Model-driver foremen also follow `docs/MODEL_DRIVER_HILLCLIMB.md`. They may
instantiate its parameterized exploratory experiments from current evidence,
but may not invent production architecture or bypass production audit gates.

## Engineering philosophy

- Less code is better. Prefer the smallest, clearest change that solves the
  measured problem. Do not add frameworks, abstraction layers, configuration,
  or generality for hypothetical future needs.
- Find decisive test results quickly. Start with the narrowest test that can
  answer the current question; avoid broad inventory work when the relevant
  production path and test are already known.
- Every coding task is either `exploratory` or `production`. The task contract
  must say which phase applies.

### Exploratory phase

- The deliverable is an answer to a named unknown, not production architecture.
- Exploratory work does not require an independent agent audit. It must still
  provide the decisive test receipt and truthful measured evidence to its Luna
  foreman or the coordinator.
- Reuse the real production implementation and add only the smallest test hook,
  probe, fixture, or launch wrapper needed to expose the hardware behavior.
- Deploy through the assigned runner, observe what the hardware actually does,
  and retain the raw command, identity, and result. Do not polish a probe into a
  generalized subsystem before the variables are known.
- Exploratory code must be clearly test-only or non-shipping. Never create a
  parallel implementation of production behavior merely to make probing easy.

### Production phase

- Begin production design only after the relevant hardware and behavioral
  variables have measured answers.
- Use the strict project C/CUDA rules, correctness gates, and production
  evidence requirements. Remove or contain exploratory shortcuts.
- Choose the simplest production solution supported by the measurements. Every
  extra production line must earn its place through a verified requirement.

### Efficiency metric

- Maximize `validated Solutions / (net production code size squared)`. Failed
  work scores zero; ten times more production code is one hundred times worse.
- Focused test and probe code that calls the real production path is evidence,
  not an additional production implementation, and is excluded from production
  code size. Reusable abstractions added only for a probe, duplicated production
  algorithms, and shipping probe machinery count as production code.

### Actual-results gate

- Agent activity is useful only when it produces one of four outcomes:
  1. a runnable experiment or the narrow code needed for it;
  2. an executed experiment with raw Spark or test output;
  3. analysis tied to that executed result that chooses the next action; or
  4. tested production code implementing a measured design.
- Plans, inventories, status prose, speculative designs, receipt preparation,
  repeated source reading, and ideas without execution are not progress.
- A negative experiment is progress when its hypothesis had a stated plausible
  path to success and its result eliminates a real branch of the search space.
- Generate a compact durable receipt only after an actual result exists. The raw
  run artifact comes first; receipt work must never delay the next experiment.
- Every OxAlpha microtask must name an `action_kind` from the four outcomes
  above, a concrete expected result, why the attempt has reasonable expected
  value, and the command or artifact that proves completion.
- Attribute `UsefulTokens` only after the promised result exists. Tokens spent
  by failed provider legs, plan-only turns, unexecuted ideas, or abandoned
  inventory remain non-useful. Maximize `UsefulTokens / TotalTokens` without
  gaming it by omitting necessary high-value failed experiments.
- Apply the agent-ablation test continuously: if removing an agent would not
  materially change experiments run, results analyzed, or tested code produced,
  redirect or retire that agent.
- A foreman manages by walking around. On every loop, inspect one live BigGulp
  deeply enough to find its current limiting result, then issue a specific
  correction or next experiment. Do not substitute workflow maintenance for
  movement toward the BigGulp's weekly outcome.

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
