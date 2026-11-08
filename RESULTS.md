# Research results

Research date: 2026-11-09, Asia/Yekaterinburg. Machine-readable records use UTC.

## CPU correctness

511 quantizer checks passed and the integer peeling identity matched exactly. Twenty-five numerical observations follow. The zero-input cases report absolute Frobenius error because their reference product has zero norm.

| Input | Noise scale | Error metric | Observed error |
| --- | ---: | --- | ---: |
| integer | 0.0 | relative_frobenius | 0 |
| integer | 0.015625 | relative_frobenius | 0.0014388962 |
| integer | 0.0625 | relative_frobenius | 0.011557983 |
| integer | 0.25 | relative_frobenius | 0.084065336 |
| integer | 1.0 | relative_frobenius | 0.35189246 |
| gaussian | 0.0 | relative_frobenius | 0.037333936 |
| gaussian | 0.015625 | relative_frobenius | 0.05702989 |
| gaussian | 0.0625 | relative_frobenius | 0.18583584 |
| gaussian | 0.25 | relative_frobenius | 0.84116079 |
| gaussian | 1.0 | relative_frobenius | 6.6549039 |
| outlier | 0.0 | relative_frobenius | 0.042707846 |
| outlier | 0.015625 | relative_frobenius | 0.061096717 |
| outlier | 0.0625 | relative_frobenius | 0.10375364 |
| outlier | 0.25 | relative_frobenius | 0.43411492 |
| outlier | 1.0 | relative_frobenius | 2.1684261 |
| zero | 0.0 | absolute_frobenius | 0 |
| zero | 0.015625 | absolute_frobenius | 0.13049448 |
| zero | 0.0625 | absolute_frobenius | 2.0879116 |
| zero | 0.25 | absolute_frobenius | 33.406586 |
| zero | 1.0 | absolute_frobenius | 534.50538 |
| rank_one | 0.0 | relative_frobenius | 0 |
| rank_one | 0.015625 | relative_frobenius | 0 |
| rank_one | 0.0625 | relative_frobenius | 0.018010646 |
| rank_one | 0.25 | relative_frobenius | 0.073948418 |
| rank_one | 1.0 | relative_frobenius | 0.54486237 |

Noise scales and rank-two correlation are experiment choices. The observed errors do not establish a secure sampling policy or model-level quality.

Both Windows/Python 3.12.10 and Linux x86_64/Python 3.12.3 runs passed. Raw records include source hashes and timing scope:

- [Windows baseline](results/cpu-baseline.json)
- [Linux baseline](results/cpu-linux.json)

## GPU campaign

The RTX 5060 Ti campaign has now executed successfully. The initial `cuda/probe.cu` diagnostic passed against its exact CPU fixture. The extended study and independent-seed replication are documented in [GPU_REPORT.md](GPU_REPORT.md), with full source hashes, timings, correctness checks, negative results, and next research decisions.

- [Primary GPU observations](results/gpu-study.json)
- [Independent input-seed replication](results/gpu-replication.json)
- [Environment](results/gpu-environment.json)
- [Reproduction and measurement rules](METHODS.md)

These experiments measure arithmetic and systems behavior. They do not establish full-protocol security, upstream verifier acceptance, or mining revenue.
