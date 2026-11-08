# Measurement methods

Research date: 2026-11-09, Asia/Yekaterinburg. Raw timestamps use UTC.

## First-stage CPU evidence

The first run establishes arithmetic correctness and creates GPU reference fixtures. Inputs use explicit deterministic seeds in source. Python arbitrary-precision integers provide exact field and ring reference arithmetic. The FP8 experiment states separately where floating-point accumulation is used. Single-run CPU timings are diagnostic observations and do not establish an optimization speedup.

## GPU evidence gates

1. Preserve and record the upstream protocol version, verifier, and security parameters when a real protocol is integrated.
2. Match every diagnostic output against its generated reference fixture before reporting throughput.
3. Separate kernel event time, host/device transfer time, and full operation latency.
4. For performance comparisons, use paired changed-input runs, warm-up policy, at least 30 repetitions, median and p95, device model, toolkit, clock policy, memory use, and competing-load observations.
5. Measure energy with sampled power or an external meter. Do not infer joules per operation from the nominal TDP.
6. Keep failed and negative experiments in the results record.

The initial CUDA diagnostic is a correctness and compilation starting point. Its 100-launch mean is not a complete performance comparison. No GPU workload is executed while the owner's card is occupied.

## Decision gate

A provisional research target is at least 20% improvement in full-operation latency or measured energy while preserving correctness and the selected protocol's parameters. This threshold is a project choice, not an observed result. Isolated kernel gains require an end-to-end follow-up. No profitability or security conclusion follows from these first-stage baselines.

## Executed Stage-1 protocol

`gpu_study.py` executes five warm-up passes and thirty measured passes for every variant and case. Variant order reverses on alternating passes. The second campaign changes the input seed from 20261109 to 20261110. Repetitions reuse a case's arrays: these are latency observations, not thirty independent workload samples. No CUDA clock, voltage, persistence, or power-limit changes were made. The owner released the card before the campaign; no workload was intentionally run concurrently on that card. Background CPU/OS activity and dynamic clocks remain possible confounders.

Report medians and nearest-rank p95 for each seed separately; do not pool independent runs to hide drift. CUDA events cover the stream interval, which can contain gaps while the host enqueues kernels. Host timings cover Python, enqueue and the explicit final synchronization. Captured CUDA Graph comparisons reduce enqueue overhead. Timed allocations normally use CuPy's warmed memory pool. Input generation, independent CPU oracles, JIT compilation, plan/heuristic selection, and correctness comparisons are excluded. Each report lists any additional exclusions. Device-resident and transfer-inclusive paths are separate experiments.

The raw records include exact script/reference source hashes. The environment record deliberately excludes usernames, hostnames, serial numbers, access credentials, and device UUIDs. Consumer-card results from one host do not establish performance on B200/H100, other GPUs, other PCIe links, or an upstream optimized implementation. No measured energy or profitability claim is made.
