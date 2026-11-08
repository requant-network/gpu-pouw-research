# GPU FP8 useful-work research

Research by [Requant Network](https://requant.network). Campaign date: 9 November 2026, Asia/Yekaterinburg; raw records use UTC.

This is an executed, reproducible research campaign on a GeForce RTX 5060 Ti 16 GB (36 SMs), paired with an AMD Ryzen 5 5500. The host exposes PCIe Gen 3, with an observed x8 link in the environment snapshot. Do not extrapolate transfer timings to a Gen 5 workstation or a datacenter GPU. The software record reports Python 3.12.3, NumPy 2.2.6, CuPy 13.6.0, NVRTC 12.8, runtime API 12090 and driver 610.43.03. CUDA compilation uses 12.8.61 unless explicitly stated otherwise.

Each variant has five warm-ups and thirty measured repetitions per case. A second campaign changes the input seed from 20261109 to 20261110. Tables show milliseconds. p95 uses the nearest-rank definition. Repetitions reuse each case's inputs; they do not establish population confidence intervals. Default dynamic clocks were retained. The experiments ran sequentially on the released GPU.

[Primary raw results](results/gpu-study.json), [independent-seed replication](results/gpu-replication.json), [complete median/p95 table](results/gpu-summary.csv), [environment](results/gpu-environment.json), [methods](METHODS.md), and [executable study](gpu_study.py) are part of this record. `python summarize_results.py` verifies source hashes and regenerates the CSV. Correctness checks run outside the measured intervals.

## Question and numerical contract

Is the speed of FP8 Tensor Core GEMM retained when input noising, quantization, recovery of a useful product, and transfers are accounted for? How does the recovered product's error vary with noise magnitude?

The study uses square matrices of sizes 256, 512, 1024 and 2048; Gaussian, outlier, all-ones rank-one and zero patterns; and noise scales 0, 1/64, 1/4 and 1. E = sigma*U*V is deliberately simple rank-two noise, with Rademacher factors. The noised operands are Q(A+E) and Q(B-E), with finite E4M3 round-to-nearest-even and saturation. They use unit tensor scales, not protocol-specific row-wise scaling.

Without quantization, the exact identity is AB = (A+E)(B-E) - EB + AE + EE. The GPU recovery computes the correction with thin matrices, using the rank-two factors rather than dense correction GEMMs. Nonlinear quantization prevents this identity from exactly recovering AB; the remaining error must be measured.

This sampler is not the Pearl sampler. In particular, correlated rank-two noise and unit scales provide no hardness or unpredictability claim. There is no ticket hashing, commitment construction, chain input, consensus verifier or valid mining certificate here.

## Correctness and coverage

Actual cuBLASLt matmuls use E4M3, FP16 or FP32 inputs and FP32 accumulation/output, with transpose/normal column-major descriptors. FP8 fast accumulation is not enabled. FP32 and FP16 results are checked against independent CPU float64 products. FP8 products are checked against CPU float64 multiplication of the decoded quantized operands, with relative accumulation error below 5e-5. This numerical check does not establish bitwise agreement with a consensus-specific accumulation schedule.

The two seeds together complete 128 workload/noise configurations. The 508-value CPU encoder fixture covers finite values, signed zeros, midpoint ties and saturation. A separate [encoder validation](validate_encoder.py) checks 262144 additional values exactly against the independent CPU encoder for positive and negative noise, including boundary values. [Raw validation record](results/encoder-validation.json).

An initial implementation added zero noise unconditionally, converting -0 to +0. The exact fixture caught it; the final encoder preserves the input when the scale is zero. Every timed transfer/layout path is also checked for exact agreement with the corresponding FP8 GEMM output before accepting its measurements.

## GPU-resident stages

The following slice uses Gaussian inputs with sigma=1/64. Each row retains the same matrix shape, input distribution and precision configuration. FP16 is included as the relevant Tensor Core comparison rather than relying only on an FP32 denominator.

| N | FP32 GEMM | FP16 GEMM | FP8 GEMM | Quantize + FP8 | Quantize + FP8 + peel | Full resident p95 | Replication resident |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 0.0312 | 0.0223 | 0.0214 | 0.0518 | 0.2798 | 0.3011 | 0.2807 |
| 512 | 0.0551 | 0.0346 | 0.0351 | 0.0674 | 0.3007 | 0.3257 | 0.2999 |
| 1024 | 0.2460 | 0.0800 | 0.0536 | 0.0900 | 0.3731 | 0.4616 | 0.3995 |
| 2048 | 1.2543 | 0.3796 | 0.2125 | 0.4047 | 0.9901 | 1.0019 | 0.9897 |

Noise matrices, factors, layout preparation and cuBLAS heuristic construction are precomputed for these resident timers. Quantization writes only packed FP8 values during timing; decoded FP32 values are produced separately for correctness checks. Peeling includes its GPU thin-matrix products and elementwise corrections. Timings include Python/ctypes dispatch and final synchronization.

FP8 GEMM alone is faster than FP16 at the largest size, but quantization consumes much of that margin. With the unfused rank-two recovery included, this useful-product path is slower than a clean FP16 GEMM. That is a negative result for this implementation, not a disproof of useful-work mining. Real inference integration can reuse resident tensors and fuse preparation/recovery; a protocol may mandate a different computation entirely.

## Host layout and PCIe effects

All three paths below include A/B upload, quantization, FP8 GEMM and output D2H. They do not include peeling. Fresh-layout timing also packs B into the required column-major storage on CPU. Prepacked timing moves that layout work outside the timer. Pinned timing additionally reuses page-locked host input/output buffers; allocations and pinning are outside its timer. No claim is made that pinning is free for a one-shot task.

| N | Fresh layout + transfer | Prepacked + transfer | Pinned/prepacked + transfer | Pinned p95 | Replication pinned |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.3527 | 0.2169 | 0.1657 | 0.1785 | 0.1653 |
| 512 | 1.2398 | 0.6593 | 0.5265 | 0.5318 | 0.5205 |
| 1024 | 7.5682 | 2.3130 | 1.9247 | 1.9299 | 1.9252 |
| 2048 | 33.7593 | 9.0211 | 7.7501 | 7.7602 | 7.7516 |

The initial large transfer-inclusive result was dominated in part by CPU transpose/packing, not just PCIe. Separating the layouts demonstrates that distinction. Even with pinned prepacked buffers, repeated host round trips cost far more than the resident GEMM on this machine. A credible mining/inference pipeline should keep reusable operands, noise and intermediate data on GPU. These transfer observations are specific to this host's link and memory path.

## Numerical quality

At N=2048 with Gaussian input, Frobenius errors relative to the clean float64 product are shown below. The peeled-error columns apply the correction in CPU float64 to the actual GPU FP8 product, isolating quantization residuals. The separate GPU FP32 thin-matrix peeling implementation is checked against that corrected result with rtol=2e-4 and atol=0.02; its own error is not substituted for the float64 columns.

| Noise scale | Noised FP8 error | Peeled error | Replication peeled error | Clean FP16 error |
| --- | --- | --- | --- | --- |
| 0.0 | 0.037430 | 0.037430 | 0.037389 | 0.000293 |
| 0.015625 | 0.057927 | 0.037440 | 0.037429 | 0.000293 |
| 0.25 | 0.786902 | 0.046934 | 0.046868 | 0.000293 |
| 1.0 | 6.181635 | 0.192794 | 0.190193 | 0.000293 |

The correction removes much of the deliberate low-rank perturbation, but it does not remove quantization residuals. With small noise, the residual stays near the clean FP8 error; large noise magnifies it. Outlier, rank-one and zero configurations remain in the raw results; zero-reference products use absolute Frobenius error instead of dividing by zero. These matrix errors are not model-accuracy measurements.

## Relation to the protocol frontier

[PIP-3](https://pearl-research-labs.github.io/pips/pip-0003/) is a draft floating-point certificate specification as checked on 9 November 2026. It specifies a richer sampling/scaling and hardware-dependent verification problem. [Komargodski and Weinstein](https://arxiv.org/abs/2504.09971) analyze matrix-based useful work under explicit hardness assumptions; this experiment does not test those assumptions. A public [norm-boundary consistency report](https://github.com/pearl-research-labs/pearl/issues/424) is a useful upstream test lead, not a GPU-confirmed result of our campaign.

The most relevant next research is **a pinned sampler/verifier conformance harness with adversarial rounding cases**, followed by a fused GPU-resident noising/quantization/recovery pipeline with actual ticket/hash costs. Treat FP4 or alternate scales as separate protocol research: greater throughput alone cannot make an incompatible computation valid.

## Decision

Continue as a numerical and protocol-conformance research direction, while rejecting a profitability conclusion from raw FP8 throughput. Retain the exact encoder tests, the FP16 baseline, rank-aware recovery, and the layout ablations. Before any mining claim, a pinned upstream verifier must accept the full artifact, both numerical quality and bit-exact hardware rules must pass, and the complete hash/proof path must be timed. No funds, network mining, reward forecast, energy measurement or model-quality evaluation was part of this campaign.

To reproduce the extra encoder checks after building the bridge, use the same library environment as `run_gpu.sh`, then run `python validate_encoder.py`.
