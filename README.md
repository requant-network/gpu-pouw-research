# GPU PoUW Research

Numerical and systems research into matrix-based proof of useful work.

A research project by [Requant Network](https://requant.network).

**Stage 1: GPU experiments executed, independently checked, and replicated on RTX 5060 Ti.**

Read the [GPU research report and conclusions](GPU_REPORT.md) for measured findings, negative results and the next research decisions.

## Research question

How do noise, low-rank structure, and FP8 quantization interact with useful matrix-product accuracy and implementation overhead?

## Current experiment scope

The first experiment implements finite E4M3FN round-to-nearest-even quantization with saturation, verifies an exact integer low-rank-noise peeling identity, and measures product error for a fixed 32-by-32 workload. The numerical noise is deliberately simple and correlated. It is not the Pearl sampler, jackpot policy, verifier, or a protocol-valid mining implementation. Accuracy measurements do not establish unpredictability, hardness, security, or expected rewards. FP8 products are accumulated in Python floating point, not a device-specific FP32 Tensor Core order.

## Checks

python run.py performs 511 quantizer checks including representable values, midpoint ties, saturation, and rejection of non-finite inputs. It checks exact integer noise peeling and sweeps five noise scales across integer, Gaussian, outlier, zero, and rank-one inputs.

The CUDA diagnostic compares NVIDIA E4M3FN conversions with the Python reference on 508 finite values, signed zeros, midpoint ties, and saturation inputs. The exact fixture length is in its generated binary header. The extended study measures actual cuBLASLt FP8/FP16/FP32 GEMM, quantization, low-rank peeling, layout preparation and transfers; see [GPU report](GPU_REPORT.md). Protocol acceptance remains outside this experiment.

## Reproduce the CPU experiment

Requires Python 3.12 or newer and its standard library. Run without Python's `-O` option, because validation uses assertions.

```sh
python run.py
```

This writes `results/cpu-baseline.json` and regenerates `results/gpu-fixture.bin`. The JSON includes UTC time, environment information, and SHA-256 hashes of the executed experiment sources. The binary fixture is generated locally and excluded from Git.

## Prepare and run the CUDA diagnostic

CUDA 12.8 or newer is required for the Blackwell target used here. Compilation does not execute a GPU workload.

```sh
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_120 cuda/probe.cu -o build/gpu-check
```

When a GPU is available, regenerate the fixture using `python run.py`, then run:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ./build/gpu-check results/gpu-fixture.bin
```

The diagnostic uploads the fixture, warms up ten launches, times 100 launches using CUDA events, and compares every result with the exact fixture. Its timing covers the diagnostic kernel only. CPU preparation and transfers are outside that timer.

## Reproduce the GPU study

Tested on Linux x86_64, Python 3.12, RTX 5060 Ti 16 GB (SM 120), CUDA compiler 12.8.61, NVRTC 12.8, CUDA runtime API 12.9 (12090) and cuBLAS package 12.8.4.1. Install dependencies in a separate environment. CUDA 12.8 toolkit headers must be available; this study uses pinned CUDA 12.x wheels even when a newer toolkit is installed system-wide.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-gpu.txt
export CUDA_PATH=/path/to/cuda-12.8
export CUDA_VISIBLE_DEVICES=0
bash run_gpu.sh
bash run_gpu.sh --seed 20261110 --output results/gpu-replication.json
```

Run without `-O`: assertions are mandatory correctness gates. Each case uses five warm-ups and thirty measured repetitions per variant, with alternating variant order. Two independent input seeds are retained. JIT compilation and plan construction are outside the timers. Read the report for the exact inclusion and exclusion rules of each timing column.

- [GPU research report and conclusions](GPU_REPORT.md)
- [Primary raw GPU campaign](results/gpu-study.json)
- [Independent-seed replication](results/gpu-replication.json)
- [Hardware and software record](results/gpu-environment.json)
- [Derived latency table, median and p95](results/gpu-summary.csv)

Rebuild the latency table and verify the recorded source hashes with `python summarize_results.py` (standard library only).

## Research record

- [Research questions and next experiments](RESEARCH.md)
- [Measurement rules](METHODS.md)
- [Observed results and limitations](RESULTS.md)
- [Raw CPU baseline](results/cpu-baseline.json)

## License and contributions

Original experiment code is MIT licensed. References retain their own licenses. Contributions should include a reproducible command, source revision, environment, correctness evidence, and a description of measurement scope. Small negative results are welcome.

## Primary references

- [Matrix-multiplication PoUW paper](https://eprint.iacr.org/2025/685)
- [Pearl FP8 draft specification](https://pearl-research-labs.github.io/pips/pip-0003/)
- [Pearl open research problems](https://pearlpolymath.com/)
- [NVIDIA NVFP4 description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
