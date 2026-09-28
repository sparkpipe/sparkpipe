# GLM Flash numerical gate audit

2026-09-08. The existing synthetic validator ran successfully on clean main
`55006656a730a8c75a78fbf01c4906ae170fe231`, Spark0, through queue job
`glm-synthetic-oracle-5500665`, attempt `8a86853a13754fd6983359322e3d5fb9`.
All participant cgroups stopped with exit 0. The KDA+dense+mHC check measured
relative L2 0.00468 and cosine 0.9999891 against its host oracle. KDA and DSA
attention repeatability passed. This is a synthetic TP1/B1 component result.

The current source has one validation lane regardless of the environment's
maximum-active-sequences setting. The old September 1 handoff describes other
tiers that are absent from this main checkout. Its transport diagnosis is
therefore historical evidence, not proof of today's distributed failure.

The audit found three numerical report results cast to void. They could print
FAIL without failing the binary. The projection probe could also skip its
comparison when readback failed. These checks now contribute to the exit
status and readback failure fails the probe. Remove incidental vector printing
and use a small projection-check helper.

The former error norm subtracted large squared norms, losing small errors to
cancellation. Nonfinite references could also take zero-norm branches and
appear valid. Common `spark_numerical_metrics.h` now accumulates squared errors
directly, rejects nonfinite inputs, defines zero-vector behavior, and validates
finite acceptance thresholds. GLM uses this shared hardware-independent math.
The host test includes a tiny error beside FLT_MAX, NaN/infinities on either
side, zero vectors and a known 3-4-5 error case.

Required qualification still includes DSA numerical comparison, routed expert
MLP, arbitrary multirow execution, real-checkpoint outputs, distributed TP4 and
TP16, TP4xPP4, and complete cache restoration. A component PASS cannot supply
those missing results. The next hill-climbing rule is to test the acceptance
test: inject bad values and prove that the actual gate rejects them before
trusting a performance optimization's green result.
