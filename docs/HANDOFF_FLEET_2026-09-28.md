# HANDOFF — fleet update through the release channel: merged, verified, pending

**Date:** 2026-09-28 · **Author:** mgr2 (sparkdev session) · **Scope:** i26/i27/i28
integration, the release-channel runbook, the fleet update attempt, and the
state the next session inherits.

## 1. What is merged (all CI-green, package-verified, on main)

| PR | Content | Merge commit |
| --- | --- | --- |
| #1249 (i26) | Nine drivers' TP opens → two family templates; `tools/host_codegen_diff.py` + test; getentropy fix for `node/model_api.c` (macOS API-home build) | `a790ce9` |
| #1250 (i27) | GPU validators/serving adapters/module functions → family templates (net −1,414 lines); `host_codegen_diff` static-renumbering fix | `05a0b25` |
| #1251 (i28) | One JIT KV tier for qwen38_max/qwen4_flash/muse_glimmer; **qwen4_flash prefill regression fix** (refused every prefill since 09-19); `host_codegen_diff` data-object compare | `53d33dd` |
| #1252 | `docs/FLEET_RELEASE_RUNBOOK.md` (the automated channel runbook) + `fleet_sync.sh` generation fix (unit ran the agent from a path `self_update` can never replace, passed a removed argument, drove the retired UPDATE sentinel) | `fd53005` |

Integration method (worked clean, reuse it): dedicated `git worktree` → merge
`origin/main` → resolve the two manifest conflicts by regenerating **in the
clean worktree** → host gates → sweep test droppings (`api_submission.seq` is
created by `test_system_loopback`) → regenerate manifests + verifier → push →
PR → CI (`compile-sm121a` + `host-tests`) → merge → post-merge verifier on the
exact merge commit.

Verified on the merged heads: 116–117/119 C binaries (the 2–3 skips are the
known host reds: nvcc absent; verbs/affinity Linux-only code in
`node/weightd_mesh.c`), 216/226 Python (the same 10 macOS toolchain failures
every time, each reproduced on pristine `main`), `test_dry_law`,
`test_stage_module_teardown`, `test_llm_module_contract` (incl. the new
tier-off/tier-on probe) pass. Identity proofs for the template moves are the
author's in-branch measurements; main-side of each merge changed no
family/module input, so they carry. They have NOT been re-measured
independently (see pending).

## 2. Fleet state right now (stable)

- **16/16 engines `ready`**, heartbeats uniform on the hub
  (`rtx5090:~/current/spark*.json`): weightd `3da98597a88b60b5` (the
  c0d54638-era proven build), residentd `0afa3721f377e980` (the dd3526b-era
  proven build), agent `a05e207588da2cef` (current main, mesh-rank-mask
  fixed).
- The fleet is channel-managed again: `fleet-agent` active on all 16, agents
  pull from the rtx5090 hub (`:8802`), `G5_API_DISABLED=1` drop-in on spark0
  (the serving api remains `g53-api` on the rtx5090).
- **Rollback applied:** the hub root currently carries the dd3526b-era
  binaries (residentd/transport/adapter/driver), NOT the 53d33dd ones. The
  53d33dd driver/adapter/residentd/api are staged on sparkf
  (`~/g5n-rd-build/build`, `~/sparkdata/out`) and the publish receipts are in
  `~/publish-main-53d33dd.log`. Re-staging the hub root from those five files
  re-attempts the upgrade as a single-variable change.
- **flychess untouched:** it lives on the rtx5090 host (stockfish +
  chess-lab eval workers); fleet operations only ever wrote `~/release` and
  `~/current` files there. No spark process outside the sparkpipe trees was
  signalled; every kill was cwd/exe-scoped TERM.

## 3. The fleet update attempt: what happened (evidence trail)

1. Built main `53d33dd` on sparkf and published through the channel. The
   first build attempts failed three build-provenance traps, each paid for:
   - **sparkf's `~/g5n-rd-build` remote refspec was pinned to a single lane
     branch**, so `git fetch origin main` updated only `FETCH_HEAD` and
     `origin/main` stayed frozen at a Sep-18 sha — `module_build_release.sh`
     reset onto a ten-day-old tree while reporting success. Fixed
     (`remote.origin.fetch` now `+refs/heads/*`); **always pass the explicit
     merged SHA as the branch argument**.
   - **`module_library` link-unit cache**: the publish reused a
     `validation=reused` archive whose provenance was unverifiable. With
     `REPOSITORY_ROOT`/`MODULE_LIBRARY_ROOT` forced on the command line the
     publish ran a fresh GPU validation (`validation=executed`, component
     validator PASS, artifact `109fbf0f…`, distinct from the reused
     `ce5b5c6a…`).
   - **adapter artifact name**: `publish_local.sh` computes
     `lib${FAMILY}_serving_adapter_$CODEC.so` from the directory name, but
     the glm5_next module's Makefile emits `libglm5_next_serving_adapter_fp8.so`
     (product name). The script's install fails for this family; the root was
     assembled by hand following `publish_local`'s own recipe (pending fix).
2. Cutover: all 16 nodes applied the new content uniformly — then every
   engine failed at `adapter_initialize` with `io_error status=4`. ERRSITE
   chain (spark9): `runtime/spark_weightd_lazy_pack.c:174` ←
   `runtime/spark_weightd.c:3559` (recv EOF) ← `:3196` (result read) ←
   adapter/module/residentd wrappers. The weightd daemon logs the attach as
   SERVED (the printf at `spark_weightd.c:1694` is the success path) while
   the engine reads EOF — and weightd's code is byte-identical to the proven
   c0d54638 build (no runtime/ changes in the whole i26-i28 delta).
3. Root causes of the cutover failure, both environmental, both fixed or
   isolated:
   - **The hub core root shipped a stale agent.** The current agent passes
     `--mesh-rank-mask 0xffff` (the newer mesh protocol requires
     "all four or none" mesh identity fields — weightd refuses to form the
     mesh without it); the 09-22-era agent in the hub core root does not, and
     `self_update` dutifully rolled the old agent back out over the fixed one
     fleet-wide. Staging the current agent into the hub `core/bin` +
     MANIFEST regenerated fixed it in minutes, via the channel, on all 16.
   - **Full-tape `.wset` sidecars** (Sep 23 hill-m artifacts, 12,096 keys,
     96,768 bytes each) sat next to the channel-root packs. With the sidecar
     present the attach takes the full-tape batch path; moved aside
     fleet-wide (`.wset.aside-20260928`, reversible). The manual deployment's
     packs had no sidecars and lazy-loaded.
4. **Rollback applied:** hub root re-staged from the per-node manual trees
   (`glm-serving-dd3526b2`, `station-core-c0d54638`), MANIFEST + WEIGHTSD_BIN
   re-announced; agents converged fleet-wide to the proven pairing. This
   proves the channel's rollback path works end to end.
5. One more paid-for lesson now in the runbook: **recycled weightds leave a
   stale `/tmp/spark_weightd.sock` file** (the recycle path `kill -9`s after
   TERM-grace and never unlinks), and `ensure_weightd`'s clearing branch only
   runs while a weightd process exists — so a dead socket file deadlocks the
   weightd start loop (`bind: address in use` forever). Clear it
   conditionally (`[ -z "$(pgrep -f sparkpipe_weightd)" ] && rm -f
   /tmp/spark_weightd.sock`); the agent should learn to do this itself.

## 4. Pending (priority order)

1. **End-to-end serving probe on the restored fleet.** `g53-api` is active;
   the first cold completions returned empty within the probe budget. The
   fleet view is 16/16 ready and the attach path is the proven one — watch
   `g53-api` logs / retry the fixture before assuming regression.
2. **Clean re-upgrade attempt of the root to the 53d33dd content.** With the
   mesh-mask agent fixed, the original attach failure may have been purely
   the confound. Re-stage the five 53d33dd files on the hub (they are on
   sparkf), let the channel converge, watch one rank's `adapter_initialize`.
   If the EOF reproduces: bisect i26/i27/i28 glm5_next module/adapter changes
   offline — the ERRSITE trail above is the starting map.
3. **qwen4_flash hardware validation** (i28's owed follow-up): run a prefill
   and decode on hardware (GPU validator or TP1 serve). Before the fix its
   first prefill frame was guaranteed `invalid_argument`.
4. **muse_glimmer tier-on requalify** (`SPARK_MUSE_GLIMMER_STAGE_KV_STORE`),
   per the TECHDEBT requalify list.
5. **Independent identity re-measurement** of the template moves on a Linux
   host with binutils (`host_codegen_diff.py` before/after builds per
   `docs/FAMILY_TEMPLATES.md`); the in-repo claims are the author's.
6. **`module_build_release.sh` / `publish_local.sh` fixes**: adapter-name
   discovery (glob the actual `.so`), never trust the build tree's
   `origin/main` ref (sparkf refspec trap), fail loudly when the module
   library cache resolves outside the build tree, and clear
   `build/module_library/link_units` per the engagement-handoff workaround.
7. **`fleet_node_agent.sh` hardening**: unlink the weightd socket in the
   recycle path (the §3.5 deadlock); consider `rm -f` of the socket before
   every start attempt.
8. **weightd/verbs Darwin host-gate red** (open, weightd lane):
   `node/weightd_mesh.c` verbs include + affinity calls block
   weightd/weightsd/mesh-mock on macOS; needs the Darwin recipe + loud
   refuse-to-run design.
9. **package_inventory exclusion gap**: `.mimosa/` agent state and
   `tools/devcycle/drivers/` binaries walk into locally regenerated
   manifests (CI's SHA256SUMS check reds; 3,036 unexpected paths on i26).
   Regenerate manifests only in a clean tree until the policy grows
   exclusions.
10. **Workstation toolchain**: this Mac's git (<2.28), GNU make (3.81), and
    Apple objdump leave 10 Python tests failing locally (all green on CI).

## 5. Where things are

- Hub: `rtx5090:~/release/{core,glm53flash.fp8.tp16,…}` (rollback content),
  heartbeats `~/current/spark*.json`; serve = `:8802` (tailscale
  `100.123.97.61`).
- sparkf: `~/g5n-rd-build` at `53d33dd` with the full build + publish
  receipts (`~/publish-main-53d33dd.log`, `~/modpub.log`); `~/release` holds
  the staged 53d33dd core+root.
- Nodes: `~/sparkdata/glm53flash.fp8.tp16` = channel root (rollback
  binaries, packs intact, `.wset` sidecars moved to `.wset.aside-20260928`);
  `~/sparkdata/weightd/` = announced weightd; manual trees
  (`~/sparkpipe/glm-serving-*`, `station-core-*`) are retired and can be
  deleted at leisure.
- Coordination: this file + `docs/FLEET_RELEASE_RUNBOOK.md` on main.
