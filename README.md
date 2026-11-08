# GPU PoUW Research

Numerical and systems research into matrix-based proof of useful work.

**Stage 0 вЂ” CPU arithmetic baseline and CUDA diagnostic preparation.**

## Research question

How do noise, low-rank structure, and FP8 quantization interact with useful matrix-product accuracy and implementation overhead?

## Current experiment scope

The first experiment implements finite E4M3FN round-to-nearest-even quantization with saturation, verifies an exact integer low-rank-noise peeling identity, and measures product error for a fixed 32-by-32 workload. The numerical noise is deliberately simple and correlated. It is not the Pearl sampler, jackpot policy, verifier, or a protocol-valid mining implementation. Accuracy measurements do not establish unpredictability, hardness, security, or expected rewards. FP8 products are accumulated in Python floating point, not a device-specific FP32 Tensor Core order.

## Checks

python run.py performs 511 quantizer checks including representable values, midpoint ties, saturation, and rejection of non-finite inputs. It checks exact integer noise peeling and sweeps five noise scales across integer, Gaussian, outlier, zero, and rank-one inputs.

The CUDA diagnostic compares NVIDIA E4M3FN conversions with the Python reference on 508 finite values, signed zeros, midpoint ties, and saturation inputs. The exact fixture length is in its generated binary header. It does not measure Tensor Core GEMM or protocol acceptance.

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

## Research record

- [Research questions and next experiments](RESEARCH.md)
- [Measurement rules](METHODS.md)
- [Observed results and limitations](RESULTS.md)
- [Raw CPU baseline](results/cpu-baseline.json)

## License and contributions

Original experiment code is MIT licensed. References retain their own licenses. Contributions should include a reproducible command, source revision, environment, correctness evidence, and a description of measurement scope. Small negative results are welcome.

## Primary references

- [Matrix-multiplication PoUW paper](https://eprint.iacr.org/2025/685)
- [Pearl FP8 specification](https://pearlresearch.ai/Pearl_Whitepaper.pdf)
- [Pearl open research problems](https://pearlpolymath.com/)
- [NVIDIA NVFP4 description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
