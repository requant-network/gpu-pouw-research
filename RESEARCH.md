# Research plan

## Question

How do noise, low-rank structure, and FP8 quantization interact with useful matrix-product accuracy and implementation overhead?

## Hypothesis and boundaries

The first experiment implements finite E4M3FN round-to-nearest-even quantization with saturation, verifies an exact integer low-rank-noise peeling identity, and measures product error for a fixed 32-by-32 workload. The numerical noise is deliberately simple and correlated. It is not the Pearl sampler, jackpot policy, verifier, or a protocol-valid mining implementation. Accuracy measurements do not establish unpredictability, hardness, security, or expected rewards. FP8 products are accumulated in Python floating point, not a device-specific FP32 Tensor Core order.

## Completed first experiment

python run.py performs 511 quantizer checks including representable values, midpoint ties, saturation, and rejection of non-finite inputs. It checks exact integer noise peeling and sweeps five noise scales across integer, Gaussian, outlier, zero, and rank-one inputs.

## Completed GPU campaign

[GPU_REPORT.md](GPU_REPORT.md) records the executed two-seed campaign, mathematical contract, independent checks, raw timings, ablations, negative findings and next decisions. All study sources and records are public. This extends the CPU baseline without claiming a complete cryptographic protocol.

## Next experiments

1. Pin an upstream protocol revision and reproduce its sampler and verifier locally.
2. Measure complete encode, quantize, GEMM, hash, and proof preparation stages.
3. Investigate FP4 encoders separately, including scales, numerical quality, and protocol security assumptions.

## Evidence required

Report a full-operation comparison against a pinned baseline, independent correctness checks, memory requirements, and energy observations. Distinguish algorithmic improvement from changed workloads or weaker validation. An upstream verifier must accept any claimed protocol improvement.

## Sources

- [Matrix-multiplication PoUW paper](https://eprint.iacr.org/2025/685)
- [Pearl FP8 draft specification](https://pearl-research-labs.github.io/pips/pip-0003/)
- [Pearl open research problems](https://pearlpolymath.com/)
- [NVIDIA NVFP4 description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
