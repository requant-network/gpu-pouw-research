# FP8 useful-work conformance: final research report

A research project by [Requant Network](https://requant.network). Campaign completed on 2026-11-09, Asia/Yekaterinburg; raw timestamps use UTC.

## Findings

The actual Pearl GPU miner kernels reproduce a verifier-defined BF16 norm mismatch on a public boundary fixture. An independent integer GPU norm implementation matches the rational CPU oracle, the unchanged upstream Rust norm functions, and a broad BF16 exponent sweep. In a restricted diagnostic pipeline, supplying canonical norms changes FP8 outputs and adds only a few microseconds to a captured prequantization/commitment/noising path.

The complete offline follow-up uses real keyed commitments, rank-32 protocol noise, FP8 mixed GEMM, peeling, ticket publication, Merkle openings and the full native plain-proof verifier. **280 executions are accepted, and 16 altered-header/Merkle controls are rejected.** That includes the raw boundary inputs under the deliberately easy target. Therefore the study confirms an arithmetic mismatch, not an observed invalid-proof rate, consensus failure or exploit.

A second compatibility finding prevents an unmodified all-GPU pipeline on the selected source pin: Python protocol parameters encode dense pA in 9 bytes, while the GPU finalize API rejects anything except 11 or 13 bytes. The full-cycle harness records this rejection and explicitly bridges key finalization through the canonical host function. Its timing includes that bridge; it must not be presented as unmodified GPU-only mining throughput.

## Research questions and decisions

| Question | Evidence | Decision |
| --- | --- | --- |
| Does a fast FP8 GEMM by itself establish useful-work efficiency? | Stage-1 whole-path preparation/quantization/peeling measurements. | No; isolated GEMM gains do not survive every tested pipeline. |
| Does the public norm counterexample reproduce on real SM120 kernels and the newer pinned source? | GPU alpha/beta compared with rational arithmetic and native Rust exact norms. | Yes; the arithmetic disagreement is reproduced. |
| Can exact integer norms be computed efficiently on GPU? | 161,189 valid constant rows, 2,011 expected overflow/grid rejections, 256 heterogeneous rows and 768 Rust comparisons. | Yes within the explicit supported shape/finite-output contract. |
| Does the canonical-norm diagnostic preserve the selected full proof path? | Full offline proof/verifier runs with actual protocol noise, plus negative controls. | Yes in these fixtures; the easy target does not demonstrate universal ticket agreement. |
| Is this a deployable miner or a production fix? | Host key bridge, restricted stats adapter, no network integration or ZK certificate campaign. | No; publish the conformance evidence and the precise integration limits. |

## Pinned sources and attribution

All upstream kernels and native verifier wheels come from the clean Pearl `fp8-scheme` revision **b7160514ed3d862fb48e850ba9a70a545cefcb1e**. The source and native wheel/Cargo-lock hashes are in [pearl-build.json](results/pearl-build.json). Package versions alone are not sufficient to identify this development branch.

The starting counterexample is attributed to yindavidyang's public [Pearl issue #424](https://github.com/pearl-research-labs/pearl/issues/424), originally reviewed at an older revision. [fixtures/README.md](fixtures/README.md) records the numerical-data provenance and layout. We extracted the disclosed numerical fixture; we did not run the issue's program. This campaign adds actual GPU and pinned Rust execution; it does not claim discovery of the original mismatch. The [PIP-0003 page](https://pearl-research-labs.github.io/pips/pip-0003/) is a draft; the executed native API is certificate/plain-proof **v4**. Draft naming must not be confused with the tested implementation version.

## Mathematical and implementation contract

Prequantization commits int8 codes with one BF16 scale per eight entries. Canonical row RMS uses the verifier's exponent-aligned integer frame and per-block truncation, exact square-root rounding to nearest-even, and the specified coarser BF16 norm grid. This is not simply FP32 `sqrt(mean(x*x))`. The infinity norm and BF16-rounded alpha/beta scale chain are likewise protocol-defined. The independent [Fraction-based oracle](experiments/protocol_norms.py) implements these rules without copying the Rust arithmetic bodies.

The retained fixture has canonical RMS bits **0x3f80**, canonical alpha/beta **0x4291 / 0x3b4d**, but the measured upstream GPU scale chain produces **0x428f / 0x3b50**. GPU behavior is compared with compiled, unchanged upstream norm/dtype/square-root functions extracted by [build_rust_norms.py](build_rust_norms.py), with the upstream ISC license retained in the generated build directory. These isolated functions are distinct from the later full native verifier test.

[cuda/canonical_norms.cu](cuda/canonical_norms.cu) performs two-pass exponent/frame reductions, exact integer accumulation and bracket-refined square-root rounding. It accepts K divisible by 512 over the tested K=1,024–65,536 range and rejects unsupported overflow/grid-infinity outputs. Validation covers every nonnegative finite BF16 scale encoding in constant rows for q in {0,1,3,127,-128}, plus heterogeneous-exponent rows and the retained boundary fixtures. This is broad testing, not a proof over every possible row.

The synthetic stats adapter is deliberately restricted to the finite, well-scaled diagnostic rows. It feeds canonical BF16 norms into the unchanged GPU noising scale chain through FP32 stats. Squaring norms in such an adapter is not safe over the entire BF16 exponent range. A production integration should consume canonical norm codes directly and retain all validity checks. No production patch is included here.

## Experiments and measurements

### Isolated conformance and overhead

`protocol_study.py` runs actual upstream prequantization, hash/stats and noising kernels at 64 rows and K=1,024, 4,096, 16,384 and 65,536. Zero keys and all-one factors intentionally isolate the norm calculation. These are not protocol-valid noise/sampler or lottery tests. Per seed, both 64-row boundary cases disagree; all 256 ordinary random rows agree. Permuting eight-entry blocks and repeating the same boundary row does not estimate failure frequency. The exact norm GPU outputs match the rational oracle in all recorded cases; 384 rows per seed also match unchanged Rust norms.

### Complete local proof path

`nonce_study.py` uses M=N=256, K=1,024/2,048 and R=32. A synthetic parent and a fresh proposed header per iteration derive real commitment keys. GPU roots are checked against CPU/native Merkle trees. B is prepared once per job. Actual keyed noise bases, noising, mixed FP8 GEMM, peeling and winner publication run on GPU. The published committed planes match the retained CPU planes; selected rows/columns are opened into `PlainProofV4` and checked by the native Rust verifier. The fixture target is compact **0x207fffff**; no network classes, RPC, wallet, submissions or reward operations are used.

Each profile has five warm-ups and thirty measured attempts for each of two seeds. All 280 attempts produce a valid hit and pass the native verifier. The same 35 header choices recur across profiles and seeds; these are not 280 independent mining-difficulty trials. For each profile/seed, a changed header and a changed Merkle leaf are rejected. Tests cover Gaussian K=1,024 and 2,048, raw boundary K=1,024, and boundary K=1,024 with the restricted canonical adapter. The B matrices differ between profiles, so their output errors/timings are not a paired causal estimate of a repair's effect.

Latency cells are **median / nearest-rank p95, milliseconds**.

### Norm conformance in the controlled diagnostic

| Seed | Input family | K | Rows with scale mismatch | Changed FP8 codes |
| --- | --- | --- | --- | --- |
| 20261111 | known_boundary | 1024 | 64 / 64 | 10694 |
| 20261111 | boundary_permutations | 1024 | 64 / 64 | 10694 |
| 20261111 | random | 1024 | 0 / 64 | 0 |
| 20261111 | random | 4096 | 0 / 64 | 0 |
| 20261111 | random | 16384 | 0 / 64 | 0 |
| 20261111 | random | 65536 | 0 / 64 | 0 |
| 20261112 | known_boundary | 1024 | 64 / 64 | 10694 |
| 20261112 | boundary_permutations | 1024 | 64 / 64 | 10694 |
| 20261112 | random | 1024 | 0 / 64 | 0 |
| 20261112 | random | 4096 | 0 / 64 | 0 |
| 20261112 | random | 16384 | 0 / 64 | 0 |
| 20261112 | random | 65536 | 0 / 64 | 0 |

### Paired captured prequantization/commitment/noising overhead

| Seed | K | Variant | Graph wall | Device interval |
| --- | --- | --- | --- | --- |
| 20261111 | 1024 | original | 0.0515 / 0.0527 | 0.0292 / 0.0298 |
| 20261111 | 1024 | canonical | 0.0551 / 0.0560 | 0.0325 / 0.0333 |
| 20261111 | 65536 | original | 0.3675 / 0.3683 | 0.3450 / 0.3459 |
| 20261111 | 65536 | canonical | 0.3805 / 0.3820 | 0.3582 / 0.3594 |
| 20261112 | 1024 | original | 0.0514 / 0.0522 | 0.0290 / 0.0296 |
| 20261112 | 1024 | canonical | 0.0552 / 0.0559 | 0.0325 / 0.0335 |
| 20261112 | 65536 | original | 0.3682 / 0.3692 | 0.3451 / 0.3462 |
| 20261112 | 65536 | canonical | 0.3807 / 0.3824 | 0.3581 / 0.3596 |

### Full local nonce through native plain-proof verification

| Seed | Input | K | Canonical adapter | Accepted | GPU path + host key bridge | Native verify | Complete local path | Max relative output error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261111 | gaussian | 1024 | False | 35 / 35 | 1.1540 / 1.4237 | 28.7519 / 35.4430 | 34.2180 / 39.8331 | 4.279% |
| 20261111 | gaussian | 2048 | False | 35 / 35 | 1.3516 / 1.4485 | 55.6515 / 65.3774 | 63.0925 / 77.1817 | 4.295% |
| 20261111 | norm_boundary | 1024 | False | 35 / 35 | 1.2836 / 1.4224 | 28.9778 / 37.7458 | 34.3328 / 42.3713 | 4.631% |
| 20261111 | norm_boundary | 1024 | True | 35 / 35 | 1.2954 / 1.5643 | 28.6125 / 33.7531 | 33.4413 / 39.2122 | 4.176% |
| 20261112 | gaussian | 1024 | False | 35 / 35 | 1.0056 / 1.0261 | 28.1981 / 28.5799 | 32.1902 / 32.5673 | 4.299% |
| 20261112 | gaussian | 2048 | False | 35 / 35 | 1.2298 / 1.3456 | 55.1347 / 57.7727 | 61.9022 / 64.5635 | 4.303% |
| 20261112 | norm_boundary | 1024 | False | 35 / 35 | 1.0121 / 1.0411 | 28.1471 / 28.5055 | 32.1748 / 32.5672 | 3.984% |
| 20261112 | norm_boundary | 1024 | True | 35 / 35 | 0.9206 / 0.9804 | 27.9455 / 28.1250 | 31.6115 / 31.8409 | 3.711% |

Raw conformance records: [primary](results/protocol-study.json), [replication](results/protocol-replication.json), [full-exponent GPU validation](results/canonical-norm-validation.json), [Rust primary](results/rust-norm-confirmation.json), [Rust replication](results/rust-norm-replication.json). Full-cycle verdicts are in [nonce-study.json](results/nonce-study.json) and [nonce-replication.json](results/nonce-replication.json).

Full median/p95 records are in [stage2-summary.csv](results/stage2-summary.csv); raw [nonce-study.json](results/nonce-study.json) and [replication](results/nonce-replication.json) include verdicts, opened indices, proof lengths and header hashes. The earlier numerical/cublasLt campaign remains in [GPU_REPORT.md](GPU_REPORT.md).

## Interpretation and limits

The controlled graph comparison measures original versus canonical-norm prequantization/commitment/noising with the same inputs, buffers and factors. It excludes plan/capture/JIT and setup. Device intervals and graph wall time are separately retained. Its small overhead supports further integration work, not a claim that a production fix is already complete.

The full-cycle `gpu_nonce_wall_ms` includes Python launches, synchronization and the required CPU key-finalization bridge. CPU Merkle proof construction and native plain verification are separately timed. `complete_local_nonce_ms` includes header/key setup, GPU path, hit readout, proof construction and verification; it excludes job-constant B preprocessing, compilation, correctness diagnostics and ZK-certificate generation. It verifies every hit because this is a conformance experiment. A production miner would amortize rare-hit proof work differently, so these numbers are not network throughput or profitability predictions.

Useful matrix error is relative Frobenius error against a float64 product of the original BF16 inputs. It is approximately 3.7–4.6% across these profiles. This is a numerical observation, not model-level quality or a protocol security guarantee. Raw boundary proofs accepting at an easy target do not erase the norm mismatch, and do not establish ticket equivalence at realistic difficulty. Hard-target differential tests, exact extracted-ticket comparison and certificate proving remain open.

No nominal-TDP energy estimate, economic reward estimate, hardness proof or consensus vulnerability claim is made. No upstream library sources were patched, and no issue/comment was sent upstream as part of this research.

## Reproduction

Use a separate Linux/Python 3.12 environment with CUDA 13.x and SM120. The measured stack is Torch 2.13.0+cu130, CUTLASS DSL 4.6.2, quack-kernels 0.6.4 and CUDA compiler 13.3.73. [requirements-pearl.txt](requirements-pearl.txt) pins direct dependencies; [pearl-environment.txt](results/pearl-environment.txt) records the complete measured environment. Native packages must be built from the source pin, not substituted with same-version PyPI wheels.

```sh
python3 -m venv .pearl-venv
. .pearl-venv/bin/activate
python -m pip install -r requirements-pearl.txt
python -m pip install maturin==1.16.0
git clone https://github.com/pearl-research-labs/pearl.git upstream-pearl
git -C upstream-pearl checkout b7160514ed3d862fb48e850ba9a70a545cefcb1e
mkdir -p build/wheels results
export RUSTC_BOOTSTRAP=1
export CARGO_TARGET_DIR="$PWD/build/pearl-target"
python -m maturin build --manifest-path upstream-pearl/py-pearl-mining/Cargo.toml --release --locked --jobs 3 --out build/wheels
python -m maturin build --manifest-path upstream-pearl/pearl-blake3/Cargo.toml --release --locked --jobs 3 --out build/wheels
python -m pip install build/wheels/*.whl
nvcc -O3 -std=c++17 -arch=sm_120 -shared -Xcompiler -fPIC cuda/canonical_norms.cu -o build/libcanonical_norms.so
CUDA_VISIBLE_DEVICES=0 python validate_canonical_norms.py
CUDA_VISIBLE_DEVICES=0 python protocol_study.py upstream-pearl
unset CARGO_TARGET_DIR
python build_rust_norms.py upstream-pearl
python confirm_rust_norms.py
CUDA_VISIBLE_DEVICES=0 python nonce_study.py upstream-pearl
CUDA_VISIBLE_DEVICES=0 python nonce_study.py upstream-pearl --seed 20261112 --output results/nonce-replication.json
python collect_pearl_build.py upstream-pearl build/wheels
python verify_stage2.py
```

Rust 1.98.1 with `RUSTC_BOOTSTRAP=1` was used for the pinned development source. Unset `CARGO_TARGET_DIR` before building/running the isolated norm executable, because `confirm_rust_norms.py` expects its default local Cargo target path. For the second isolated-norm seed, run `protocol_study.py --seed 20261112 --output results/protocol-replication.json` and then `confirm_rust_norms.py --study results/protocol-replication.json --output results/rust-norm-replication.json` before regenerating those inputs again. Generated input binaries are ignored; their hashes are recorded. To retain both input sets, move them into separate local directories and pass `--inputs-dir` accordingly.

Run without Python `-O`. Set `CUDA_PATH`, `LD_LIBRARY_PATH`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` to your local toolkit/device as needed. Set `OPENBLAS_NUM_THREADS=4` to match the CPU reference environment. CI runs rational/CPU references and source/result verification; it does not execute these GPU workloads or rebuild the full verifier.

## Conclusion and remaining research

This completed study advances the initial GEMM experiment to arithmetic conformance and full local plain-proof acceptance. The useful next work is a direct canonical-norm interface, resolution of the pinned pA layout mismatch, and differential ticket tests at harder targets, followed by ZK-certificate measurements. Those require new integration studies. The present deliverable is reproducible research evidence, with both positive results and observed limitations retained.

## Follow-up closure, 12 November 2026

The [follow-up report](FOLLOWUP_REPORT.md) resolves the concrete next steps with new evidence and updated conclusions. Its findings supersede the open-work wording in this historical campaign.
