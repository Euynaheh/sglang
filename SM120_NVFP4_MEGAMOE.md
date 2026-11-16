# SM120 NVFP4 FlashInfer MegaMoE integration

This branch integrates the FlashInfer SM120 NVFP4 target runner with SGLang's
fused MegaMoE path. The publication uses upstream main
`4bf9ca2a3bf7f8dd2e1c2ea6e63d8d8059b46426` and preserves the previous
`sm120-nvfp4-flashinfer-megamoe` branch in its merge history.

## Dependency and execution contract

Use [Euynaheh/flashinfer](https://github.com/Euynaheh/flashinfer) branch
`sm120-nvfp4-megamoe`, tested integration commit
`e1f22cb624b64d7aac65cb221b360092adcde39a`. The standalone kernel provenance
is `f6f6b5ae0f7d16fc7856d1af1d633d421590f44d`, cache ABI `2814542011115`.
The ordinary FlashInfer release version alone does not identify this backend.

Select `--moe-runner-backend flashinfer_megamoe --moe-a2a-backend megamoe`.
The adapter requires SM120, attention TP1, expert TP1, and evenly partitioned
experts without redundant experts or EPLB. Shared experts execute separately;
the resolver enables `--disable-shared-experts-fusion`. Two-batch overlap is
unsupported. The existing shared/routed overlap remains enabled.

`--flashinfer-megamoe-max-num-tokens` sets workspace capacity per EP rank
(default 8192). It must cover the largest resolved prefill and speculative
verify workload. Capacity is distinct from the actual compile bucket: the
adapter passes the maximum local/global-DP row hint to `stage_inputs`, then
calls `compute_staged`, preserving staging, caller reset, K3 and completion
protection inside the complete backend call. Capacity overflow raises an error;
the integrated runner does not fall back to ordinary FusedMoE after weight
ownership has transferred.

NVFP4 uses per-expert FC1/FC2 scales and the `owner_bf16` numerical contract.
Gate/up FC1 alphas must match. This contract permits owner-partial BF16 rounding
and does not promise byte equality with ordinary top-k accumulation.

DSV uses the checkpoint's original MXFP4 DSpark draft with
`--speculative-moe-runner-backend flashinfer_mxfp4` and
`--speculative-moe-a2a-backend none`. The separate draft backend contexts are
restored on exit. A narrow DEP4/DEP8 DSpark configuration exception permits
this target/draft pairing. Loader compatibility preserves serialized draft
quantization metadata. GLM uses its bundled BF16 draft through Triton; the
experimental precision-audit hook is not included in this publication.

## Included compatibility changes

- MegaMoE arguments, capacity validation, FlashInfer buffer/gate bypass,
  zero-copy packed weights, per-expert scales, and stage/compute custom ops.
- SM120 MXFP4 draft selection and separate DSpark runner/A2A contexts.
- GLM5Next architecture recognition in the SM120 attention bridge; KPool NoPE
  history/tail slot preservation, validity-mask consumption and tile padding.
- Public FlashInfer sparse-MLA capability queries with fail-closed legacy
  fallback, retaining the previously tested November 10 compatibility fix.

No recorder, profiling NVTX hooks, timeline analyzer, benchmark client,
private dependency overlay, generated cache, checkpoint or experimental
shared/routed serialization is included. The source release does not vendor
the complete serving environment: prior measurements used Torch 2.14.1+cu130,
CuTe DSL 4.8.1 and an adapted FlashInfer/dependency environment with NCCL/P2P/GIN.
Installing this branch alone does not recreate those private dependencies.

## Validation and limits

Publication validation reruns the adapter, NVFP4 scale/contract, capacity and
attention capability CPU suites on the final candidate. Tests cover mock API
plumbing and validation; they do not execute the full GPU model.

Historical serving evidence for the frozen source family includes .9 DEP4
(27 valid points) and DEP8 (36 valid points), both DSV4-Flash and GLM5.3-Flash.
DSV used fake-prefill 8K/1K, MXFP4 DSpark draft, and native continuous-refill
metrics including ramp/tail. GLM used real semantic 8K/1K, BF16 NEXTN/EAGLE draft
and qualified full-batch cycle windows. These protocols are not interchangeable.
GLM four-GPU qualification is limited to CC4/8/28. Native/TP baselines also
include loader/draft compatibility changes; they were not pristine upstream.

Performance is configuration dependent: DEP4 GLM initially improved about 2.2%,
while DSV DEP4 CC128 regressed about 1.64%; most DEP8 cases regressed relative
to native DEP8. The later GLM CC8/28 timeline observed complete-MoE envelopes
shorter by 3.19%/6.46%; separate uninstrumented runs improved output throughput
by 3.96%/4.10%, with differing speculative acceptance. Shared/routed
serialization regressed and was rejected. No universal speedup is claimed.

The final publication combines the frozen DSV core and GLM attention fixes,
removes instrumentation and restores the public capability query. CPU tests
are rerun for this tree; the complete combined tree is not newly GPU-benchmarked
as part of publication. Existing replay/finite-output/serving evidence does
not establish a new independent arithmetic-reference, byte-exact transport,
model-quality or cold-start-reliability qualification. Clocks were recorded
but not locked. Historical raw experiment records remain outside this repo.
