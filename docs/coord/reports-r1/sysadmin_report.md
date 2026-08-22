# SparkPipe GB10 Fleet Inventory — sysadmin_report

- **Collected:** 2026‑08‑22 ~12:04 UTC (box clocks show ≈18:0x local)
- **Method:** `ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 <box>` per box, read‑only collector piped via `bash -s`, fail‑soft with `|| echo UNREACHABLE`. Failed boxes were retried once across all configured transports (`<box>`, `<box>-10g`, `<box>-wifi`, `<box>-200g`) before being marked UNREACHABLE. **Nothing was cleaned, killed, or modified** — report only.
- **Scope:** spark0–9 + sparka–f (16 GB10 boxes). `/tmp` and `/home` are on the same LV as `/` (`/dev/nvme0n1p2`) on every inventoried box, so their usage equals the root figures below.

## Executive summary

Only **7 of 16 boxes are reachable**: spark0, spark2–spark7. Nine are dark (spark1, spark8, spark9, sparka–f). Of the reachable set:

- **3 are unavailable by policy**: spark3 (live qwen27B DFlash2 vLLM serving — untouched), spark0 (build/reference reservation), spark2 (occupied by the GLM dev — inventory only).
- **4 are genuinely free and near‑pristine**: spark4, spark5, spark6, spark7.

### Recommendation — GLM5.2 PP7 (7 × GB10, pipeline‑parallel)

There are **not 7 certifiable‑clean boxes online today**. Recommended squad, in priority order:

| Slot | Box | Basis |
|---|---|---|
| 1 | **spark5** | Free/idle, /tmp 187 M, root 42 % used (2.1 T free), caches <75 M |
| 2 | **spark6** | Free/idle, /tmp 186 M, root 39 % used (2.1 T free), **pre‑staged `vllm_glm52_cuda130_nvfp4/image.tar` (19 G)** |
| 3 | **spark7** | Free/idle, /tmp 186 M, root 40 % used (2.1 T free), **same GLM5.2 image.tar pre‑staged** |
| 4 | **spark4** | Free/idle (no procs, no logins since Aug 3), /tmp 1.0 G, root 67 % used (1.2 T free); some large staging packs listed below |
| 5–7 | *unfilled* | Must come from the 9 recovered dark boxes — see below |

To fill slots 5–7, in order of preference:

1. **Recover any three of spark1/spark8/spark9/sparka–f** and re‑run this inventory before promoting them. spark1/8/9/a/b/c show *banner‑exchange timeouts* (hosts power up but sshd never answers — likely hung/booting); sparkd/e/f show plain connect timeouts (down or unrouted). The `-200g` route through spark0 also failed for all nine, so this is not a transport problem.
2. If deployment cannot wait, the least‑bad fallback is **negotiating temporary use of spark0** with the build/reference owner (idle since Aug 17, load 0.10, 1.9 T free) — requires human sign‑off, not actioned.
3. Do **not** plan around spark2 (active GLM dev) or spark3 (production serving).

**Homogeneity caveat:** all seven inventoried boxes run identical Ubuntu 24.04.4 LTS, kernel 6.17.0‑1026‑nvidia, 20 CPUs, GB10, ~120 Gi unified RAM + 63 Gi swap. However **spark7 is a ThinkStation PGX chassis while the rest are `aitopatom-*` units**; if PP7 requires strict chassis/NIC homogeneity, treat spark7 as the odd one out and prefer replacing it with a recovered `aitopatom` unit when one comes back.

## Reachability

| Box | Result | Detail |
|---|---|---|
| spark0 | REACHED | aitopatom‑9ab9 |
| spark1 | UNREACHABLE | banner‑exchange timeout on all 4 transports |
| spark2 | REACHED | aitopatom‑931a |
| spark3 | REACHED | aitopatom‑a18f |
| spark4 | REACHED | aitopatom‑c342 |
| spark5 | REACHED | aitopatom‑a36d |
| spark6 | REACHED | aitopatom‑c637 |
| spark7 | REACHED | thinkstation‑pgx |
| spark8 | UNREACHABLE | banner‑exchange timeout on all 4 transports |
| spark9 | UNREACHABLE | banner‑exchange timeout on all 4 transports |
| sparka | UNREACHABLE | banner‑exchange timeout on all 4 transports |
| sparkb | UNREACHABLE | banner‑exchange timeout on all 4 transports |
| sparkc | UNREACHABLE | banner‑exchange timeout on all 4 transports |
| sparkd | UNREACHABLE | connect timeout (also via mgmt/fabric aliases) |
| sparke | UNREACHABLE | connect timeout (also via mgmt/fabric aliases) |
| sparkf | UNREACHABLE | connect timeout (also via mgmt/fabric aliases) |

## Per‑box records (reachable)

### spark0 — RESERVED (build/reference) — do not deploy
- Host aitopatom‑9ab9 · Ubuntu 24.04.4 · 6.17.0‑1026‑nvidia · 20 CPU · up 11d23h
- Disk: `/` 3.7 T, **1.6 T used (46 %)**, 1.9 T free (= /tmp and /home); `/home/spark0/extnvme` (sdh) 3.6 T, 156 G used, 5 %
- RAM 119 Gi total / 13 Gi used / 106 Gi avail; swap 63 Gi (1.9 Gi used)
- Load 0.10 / 0.22 / 0.20 · GPU GB10, 0 % util, no compute apps
- Services: **none** (no residentd/jupyter/notebook/vllm; no tmux/screen); last interactive session ended Aug 17
- Junk: **/tmp 262 G** — `k3_slice_0_4.pack` 52 G, `k3_slice_1_3.pack` 48 G, `k3_stage0_0_24.pack.payload` 22 G, `tp4-rank{2,3}.qwen36sp` 15 G ea., `k3_1_3.rank0{0,1,2}.pack` 12 G ea. (stale >50 G packs ✔); `~/.cache` 24 G (uv 14 G, huggingface 9.5 G); journald 3.5 G; `~/src` 48 G build tree; `~/srcdata` 531 G; `~/sparkdata` 616 G incl. multiple 38–93 G `.spstage`/`.pack`; extnvme `.ceph-floor-14-2/osd.img` 100 G
- Status: **RESERVED — keep `/home/spark0/extnvme/models/hf/`**; candidate junk but zero actions taken

### spark2 — OCCUPIED (GLM dev) — inventory only
- Host aitopatom‑931a · same SW stack · up 5d02h
- Disk: `/` 3.7 T, 1.9 T used (53 %), 1.7 T free; `/home/spark2/extnvme` (sda1) 7.3 T, **93 % full**, 568 G free
- RAM 119 Gi / 8.3 Gi used / 111 Gi avail; load 0.35
- Services: none detectable (no residentd/jupyter/notebook/vllm procs) — occupancy is by **human activity**: `~/sparkdata`, `~/sparkpipe`, `~/.cache` touched within the last 3 days (latest Aug 22)
- Junk: minor — /tmp 2.8 G; duplicated `dsv4_flash_stage.spstage` 38 G ×6 across `srcdata/releases/*` (Aug 13–14); `qwen38.fp8.tp1` pack 51 G; extnvme `models` 3.8 T + `cold` 2.9 T (capacity, not junk)
- Status: **OCCUPIED — hands off**

### spark3 — RESERVED (qwen27B DFlash2 serving) — do not touch daemon/driver/env
- Host aitopatom‑a18f · same SW stack · up 3d16h
- Disk: `/` 3.7 T, **3.3 T used (93 %)**, 268 G free; `/home/spark3/extnvme` (sda1) 7.3 T, **95 % full**, 411 G free
- RAM 121 Gi, **116 Gi used**, 5.6 Gi avail (server resident); load 0.28
- Services: **vLLM serving live** — `vllm serve .../Qwen3.8-27B-local --speculative-config {"method":"dflash",...}` PID 3945537, port 8123, `VLLM::EngineCore` PID 3945704 holding **103 204 MiB GPU**
- Junk (**/tmp junk ok** per reservation — listed, not touched): **/tmp 319 G** = `decode_states_tail` 109 G, `decode_states_pw_nospec` 102 G, `dumps_nospec` 64 G, `dumps_spec` 28 G, `pip-build-env-*` 3 × 4.4 G; `~/pro-repo` 1004 G incl. `dsv4_pro_ga.full.spstage` **832 G** (largest single file in fleet) + 93 G + 80 G stage files; `srcdata/releases` duplicates 38–39 G ×5
- Status: **OCCUPIED/PRODUCTION — excluded from deployment**

### spark4 — FREE
- Host aitopatom‑c342 · same SW stack · up 4d01h
- Disk: `/` 3.6 T, 2.3 T used (67 %), **1.2 T free**; `/home/spark4/extnvme` (sda) 3.6 T, 79 %, 764 G free
- RAM 119 Gi / 8.7 Gi used / 110 Gi avail; load 0.05
- Services: none; no logins since Aug 3; last file activity Aug 19
- Junk: /tmp 1.0 G only; `srcdata/dsv4_flash.fp8.pp13/dsv4_flash_stage_v4.spstage` 156 G + rank copies 48 G ×4; `sparkdata/dsv4-staging/packtool/*.spstage` 146 G ×2 + k7-runtime 48 G; `dsv4_pro_tp4_pp4_stage.spstage` 88 G; `k3.stage1.rank00.pack` 49 G; stray `model-…safetensors.part` 31 G (interrupted download); caches trivial (49 M)
- Status: **FREE — deployment candidate #4**

### spark5 — FREE
- Host aitopatom‑a36d · same SW stack · up 4d02h
- Disk: `/` 3.7 T, 1.5 T used (42 %), **2.1 T free**; `/home/spark5/extnvme` (sda1) 3.7 T, **93 % full**, 292 G free
- RAM 121 Gi / 7.7 Gi used / 113 Gi avail; load 0.42 (idle noise)
- Services: none; no sessions; last activity Aug 18
- Junk: /tmp 187 M; `dsv4_pro_tp4_pp4_stage.spstage` 88 G; `k3.stage1.rank01.pack` 49 G; `dsv4-staging/k7-runtime` 48 G; `model-00079….part` 33 G partial; caches 73 M
- Status: **FREE — deployment candidate #1 (cleanest overall)**

### spark6 — FREE
- Host aitopatom‑c637 · same SW stack · up 4d01h
- Disk: `/` 3.6 T, 1.4 T used (39 %), **2.1 T free**; `/home/spark6/extnvme` (sda1) 7.3 T, 86 %, 1.1 T free
- RAM 121 Gi / 8.5 Gi used / 113 Gi avail; load 0.14
- Services: none; no sessions; last activity Aug 18
- Junk: /tmp 186 M (fleet minimum); `dsv4_pro_tp4_pp4_stage.spstage` 88 G; `k3.stage1.rank02.pack` 49 G; k7-runtime 48 G; `ds4_images/vllm_glm52_cuda130_nvfp4/image.tar` 19 G (**GLM5.2 vLLM image already staged**); caches 34 M
- Status: **FREE — deployment candidate #2**

### spark7 — FREE (chassis outlier)
- Host **thinkstation-pgx** (only non‑aitopatom inventoried) · same SW stack · up 4d01h
- Disk: `/` 3.6 T, 1.4 T used (40 %), **2.1 T free**; `/home/spark7/extnvme` (sda) 3.7 T, **91 % full**, 325 G free
- RAM 119 Gi / 7.7 Gi used / 111 Gi avail; load 0.14
- Services: none; no sessions; last activity Aug 18
- Junk: /tmp 186 M; `dsv4_pro_tp4_pp4_stage.spstage` 88 G; `k3.stage1.rank03.pack` 49 G; k7-runtime 48 G; `vllm_glm52_cuda130_nvfp4/image.tar` 19 G (**GLM5.2 image staged**); `extnvme/spark9_coldsata` 569 G (another box's cold storage parked here); caches 47 M
- Status: **FREE — deployment candidate #3, modulo chassis homogeneity caveat**

## Fleet‑wide junk patterns (for the future cleanup pass — NOT executed)

| Pattern | Where | Size | Note |
|---|---|---|---|
| Stale `/tmp` pack/rank accumulation | spark0 | 262 G | `k3_slice_*`, `k3_stage0_*.payload`, `tp4-rank*.qwen36sp`, `k3_1_3.rank*` |
| Stale `/tmp` dump/state accumulation | spark3 (reserved) | 319 G | `decode_states_*`, `dumps_*`, `pip-build-env-*` — explicitly "junk ok" zone |
| Duplicate `dsv4_pro_tp4_pp4_stage.spstage` | all 7 boxes | 88–93 G each | identical role, Aug 17 timestamps |
| `k3.mxfp4.tp4pp4` rank packs | spark4/5/6/7 (+rank00–03 on 0/2/3) | 49–50 G each | one rank pack per box |
| `dsv4-staging/k7-runtime/packs` | spark4/5/6/7 | 48 G each | Aug 18 |
| Release‑dir spstage duplication | spark2/spark3 `srcdata/releases/*` | 38–39 G ×5–7 | old release trees, Aug 13–14 |
| Interrupted downloads `*.safetensors.part` | spark4 (31 G), spark5 (33 G) | ~64 G | deletable candidates |
| Coredumps | all | **none** | `/var/lib/systemd/coredump` empty, `/var/crash` empty everywhere |
| `.ceph-floor-14-2/osd.img` | every extnvme | 100 G each | hidden floor‑fill artifact — review before any deletion |
| journald archives | all | 2.1–3.5 G | minor |
| apt cache | all | ~203–278 M | negligible |

## Compliance notes

- Reservations honored: spark3 daemon/driver/env untouched (read‑only `ps`/`df`/`du` only; its `/tmp` was **not** cleaned despite standing allowance); spark0 `extnvme/models/hf/` intact; spark2 inventoried only.
- No files were created, deleted, or edited on any remote box; the collector ran strictly read‑only commands.
- Raw per‑box collector output retained locally at `/tmp/sparkinv/inv_<box>.txt` for cross‑checking.
