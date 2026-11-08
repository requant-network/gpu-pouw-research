# Research program and decisions

A research project by [Requant Network](https://requant.network).

## Completed campaign

How do real FP8 useful-work kernels agree with canonical verifier arithmetic, and what is the cost of producing locally verifiable candidates?

The [final report](FINAL_REPORT.md) documents the hypotheses, mathematical contract, two-seed measurements, independent checks, negative outcomes and conclusions. Completion refers to this bounded experimental campaign; it does not mean the field has no open questions.

| Completed experiment | Executable source | Primary evidence |
| --- | --- | --- |
| Actual GPU norm mismatch and overhead | [protocol_study.py](protocol_study.py) | [results/protocol-study.json](results/protocol-study.json) |
| Independent full-exponent norm validation | [validate_canonical_norms.py](validate_canonical_norms.py) | [results/canonical-norm-validation.json](results/canonical-norm-validation.json) |
| Unchanged upstream Rust norm comparisons | [confirm_rust_norms.py](confirm_rust_norms.py) | [results/rust-norm-confirmation.json](results/rust-norm-confirmation.json) |
| Complete offline candidate/plain-proof path | [nonce_study.py](nonce_study.py) | [results/nonce-study.json](results/nonce-study.json) |
| Pinned native verifier build | [collect_pearl_build.py](collect_pearl_build.py) | [results/pearl-build.json](results/pearl-build.json) |

The matching replication records and complete median/p95 data are indexed in [stage2-summary.csv](results/stage2-summary.csv). [GPU_REPORT.md](GPU_REPORT.md) preserves the preliminary Stage-1 experiments.

## Work beyond this campaign

This completed study advances the initial GEMM experiment to arithmetic conformance and full local plain-proof acceptance. The useful next work is a direct canonical-norm interface, resolution of the pinned pA layout mismatch, and differential ticket tests at harder targets, followed by ZK-certificate measurements. Those require new integration studies. The present deliverable is reproducible research evidence, with both positive results and observed limitations retained.
