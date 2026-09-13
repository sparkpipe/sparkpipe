# Vendored HIP headers - provenance

Purpose: host-side HIP API subset used to compile-check this module on
machines without a ROCm install (`tools/compile_check.sh`, default vendor
mode). These are unmodified upstream ROCm public headers.

- Upstream repository: https://github.com/ROCm/rocm-systems
- Tag: `therock-7.14` (HIP_VERSION 7.14.60850)
- Fetched: 2026-08-24
- Source paths:
  - `projects/hip/include/hip/*.h` -> `hip/*.h`
  - `projects/clr/hipamd/include/hip/amd_detail/*` -> `hip/amd_detail/*`
- License: ROCm upstream MIT license; see `LICENSE.md` at the upstream repo
  root of `projects/hip`. This vendored subset is for compile-checking only.

Files:

    hip/hip_runtime_api.h        (from projects/hip)
    hip/hip_common.h             (from projects/hip)
    hip/hip_deprecated.h         (from projects/hip)
    hip/hip_vector_types.h       (from projects/hip)
    hip/driver_types.h           (from projects/hip)
    hip/linker_types.h           (from projects/hip)
    hip/texture_types.h          (from projects/hip)
    hip/channel_descriptor.h     (from projects/hip)
    hip/surface_types.h          (from projects/hip)
    hip/hip_texture_types.h      (from projects/hip)
    hip/hip_version.h            (GENERATED - see below)
    hip/amd_detail/host_defines.h               (from projects/clr)
    hip/amd_detail/amd_channel_descriptor.h     (from projects/clr)
    hip/amd_detail/amd_hip_vector_types.h       (from projects/clr)
    hip/amd_detail/amd_hip_runtime_pt_api.h     (from projects/clr)

`hip/hip_version.h` is build-generated upstream and absent from the source
tree; it was written by hand from the tag's `projects/hip VERSION` file
(7.14.60850). No consumer in the vendored API subset reads its macros - they
appear in doc comments only.

Refresh procedure: bump the tag in this file, re-fetch the listed paths, and
re-run `tools/compile_check.sh` (vendor mode must stay green).