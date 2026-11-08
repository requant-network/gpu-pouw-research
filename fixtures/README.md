# Numerical boundary fixture

The 3328-byte fixture is numerical test data published by GitHub user **yindavidyang** in [Pearl issue #424](https://github.com/pearl-research-labs/pearl/issues/424), referring to revision `684f7045f3b3b7c5b7598f46c06b7dceb506f74b`. We extract the data only; the issue's Python/CUDA reproduction programs are not vendored.

`norm-boundary.bin` stores 1024 signed int8 codes, 128 little-endian uint16 BF16 scales, and 1024 little-endian uint16 original BF16 values, in that order. The corresponding original input is finite and prequantizes to the stated planes. The raw SHA-256 and origin are in [norm-boundary.json](norm-boundary.json).

The Stage-2 tests run the actual GPU kernels at the newer pinned revision `b7160514ed3d862fb48e850ba9a70a545cefcb1e`. Repeated rows and block permutations are deliberately adversarial fixtures; their failure fraction must not be interpreted as the frequency among production workloads.
