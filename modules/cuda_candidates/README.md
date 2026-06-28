# CUDA candidate source reservoir

This directory preserves the iteration-073 CUDA implementation work outside the active SparkPipe build.

A candidate becomes selectable only after it is reshaped into a model-specific firmware link unit, compiled for an exact target, hardware-validated, and published through `sparkpipe_module_publish`.

For a complete CUDA stage, the preferred publication unit is a normal static archive containing:

```text
firmware ABI entry point
model-specific CUDA kernels
host launch code
private helpers
device-link output when required
```

The archive is content-addressed and validated as one immutable artifact. The generated model driver links that exact archive directly. Thin archives are not accepted.

The preserved generic host wrappers are source material only. They are not production firmware and cannot satisfy a model description without an explicit model-specific module boundary and passing validation record.
