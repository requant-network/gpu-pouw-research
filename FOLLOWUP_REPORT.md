# PoUW conformance closure and native certificates

Requant Network, 12 November 2026. This supplement resolves concrete protocol/API defects and certificate feasibility questions from the [historical report](FINAL_REPORT.md). The changes are an isolated experimental patch against Pearl commit `b7160514ed3d862fb48e850ba9a70a545cefcb1e`; the original checkout and native verifier remain unchanged.

## Corrected defects

1. The pinned canonical parameter encoding uses 9 bytes for dense pA and 11 for MoE. The GPU finalizer expected 11/13 bytes and used incompatible final BLAKE3 block lengths. [The patch](patches/conformance.patch) fixes both lengths and the block-length calculation; malformed encodings still reject.
2. Canonical BF16 norms now enter the noising API directly as an optional `(m, 2)` tensor. This avoids squaring a rounded norm into FP32 statistics and converting it back. Shape, dtype, finite/nonnegative values and the canonical L2 grid are checked; compile caching distinguishes this variant. The existing exact GPU norm producer is unchanged, and independent rational arithmetic checks the scale chain.
3. The SM120 lottery folded an interleaved lane layout. The unchanged native verifier expects sixteen contiguous chunks in row-major order. The patch implements that order with lane shuffles. Harder share targets exposed the defect that an easy saturated target could conceal. An independent CPU layout check covers 1,024 fold words.

The six-file historical norm+pA-only patch remains an explicit negative control in [patches/history](patches/history/norms-pa-only.patch). The corrected patch contains seven files. Attribution and the upstream ISC notice are retained; no native acceptance rule is weakened.

## Full-path qualification

Two independent matrix seeds exercise Gaussian inputs at K=1024/2048 and a rounding-boundary input at K=1024, each with raw or canonical norms. Corrected runs use three declared pool-share targets; historical runs use the middle target. Each case retains all 35 attempts, including no-hit attempts, proof rejection and the first public synthetic proof/header. The GPU threshold is `min(2^256-1, compact_target*h*w*k)`; the header stays at the documented easy target and the unchanged verifier checks the declared pool-share override.

Across the 48 cases, **471 corrected canonical candidates were accepted**. Historical runs retained **397 rejected candidates**, including Gaussian/canonical counterexamples that isolate the lottery-order mismatch. Under the corrected lottery, the harder boundary case accepted 1/35 raw candidates versus 35/35 with canonical norms. At the stronger target, the raw boundary variant accepted none in either cohort; canonical variants accepted their 9/9 and 6/6 found candidates. No-hit attempts remain in the denominator. Every returned GPU A key matched the canonical CPU derivation byte for byte; direct-norm scale diagnostics matched the independent oracle in the exercised full pipeline.

[conformance-summary.csv](results/conformance-summary.csv) contains found/accepted/rejected/no-hit counts for every case. The JSON records bind the exact executed sources and public artifacts. These campaigns qualify correctness. Their incidental latencies are not a dedicated fair performance comparison, and threshold acceptance is not an exhaustive theorem of bitwise ticket equality.

## Actual native ZK certificates

The unchanged `Fp8Prover.setup(Device.B200)` / `Fp8Verifier` CPU API produced two complete certificates from accepted synthetic plain proofs. Both verified; altering header, public statement or proof rejected in each run.

| Seed | Prover setup, s | Prove, s | Verifier setup, s | Verify, ms | Proof bytes | Peak process RSS, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261112 | 6.216 | 29.842 | 6.588 | 9.42 | 59,554 | 6.230 |
| 20261212 | 6.147 | 27.968 | 6.190 | 9.32 | 59,554 | 6.192 |

These are two feasibility observations under an explicit 48 GiB / 900 s budget. They establish that the real certificate path fits this server; they are not a thirty-sample certificate-throughput estimate or GPU proving result. `Device.B200` is the native parameter-profile argument, not a claim that this server contains a B200. [Certificate source](zk_certificate_study.py), [first result](results/zk-certificate.json) and [replication](results/zk-certificate-replication.json) retain setup, timing, peak memory, budgets, controls and proof/statement hashes.

## Reproduction and decision

Use [prepare_conformance.py](prepare_conformance.py) to create an isolated checkout, then [conformance_study.py](conformance_study.py) with the targets/seeds retained in each result. Its `--help` documents the output and patch arguments. The [public artifact replayer](verify_conformance_artifacts.py) uses the original native CPU verifier for both accepted/rejected plain artifacts and ZK certificates; `python -m pip install -r requirements-reference.txt` installs the pinned BLAKE3 reference; `python verify_conformance.py` performs CPU-only provenance, artifact-hash and layout checks in CI. The [wide API qualification](conformance_validation.py) separately tests dense/MoE finalizers, malformed arguments and wide-exponent norm scaling. All 256 full 96-byte finalizer outputs matched, all 15,810 norm/scale pairs matched independent rational arithmetic, and all seven malformed pA / six malformed norm inputs rejected. CPU CI independently recomputes all 256 key derivations with pinned BLAKE3 and the complete norm-result digest. A fresh native replay also reproduced the 48 public plain verdicts and accepted both ZK certificates while rejecting altered proofs. The [validation record](results/conformance-validation.json) and [native replay](results/conformance-native-replay.json) retain the outcomes.

The completed decision is to retain canonical norms and the corrected lottery order together: either fix alone is insufficient. The original GPU path should not be described as conformant merely because easy-target proofs passed. The evidence supports an offline experimental repair and real certificate feasibility. It does not establish production readiness, arbitrary-shape coverage, formal security, energy efficiency, heterogeneous multi-GPU scaling, network participation or mining economics.

## Server timing boundary

GPU campaigns run serially on the same RTX 5060 Ti. Clocks, power limits and unrelated server services are unchanged. Measurements describe this server and retained workload; small percentage differences are not treated as universal improvements. No energy measurement or heterogeneous multi-GPU throughput is inferred.
