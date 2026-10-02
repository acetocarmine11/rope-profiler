# Compatibility and evidence

Snapshot: initial release 0.1.0, 2026-10-02. A dependency range describes permitted
installation; the table describes checks actually executed. The GitHub Actions
workflow defines the CPU compatibility matrix; its run history records online
results for each source revision.

| Component | Installation range | Executed evidence |
| --- | --- | --- |
| Python | >=3.10,<3.13 | Python 3.10.20 and 3.12.13 CPU; 3.11.9 CUDA |
| torch | >=2.4,<3 | 2.4.1 and 2.8.0 CPU; 2.6.0+cu124 CUDA |
| Transformers extra | >=4.51,<4.57 | 4.51.3 and 4.56.2 CPU suites; 4.56.2 CUDA |
| accelerate extra | >=1.3,<2 | 1.10.1 with local Trainer; 1.3.0 in the CUDA environment |
| W&B extra | >=0.19,<1 | 0.30.0 offline tests and local Trainer example |
| Research plotting extra | >=3.8,<4 matplotlib | 3.11.2 in the main CPU environment; synthetic and frozen-paper heatmaps |

The 0.1.0a1 baseline passed 140 tests in each of three local macOS ARM64
combinations (then six installed-file integrity cases were added and passed):

| Python | torch | Transformers | NumPy | Result |
| --- | --- | --- | --- | --- |
| 3.10.20 | 2.4.1 | 4.51.3 | 1.26.4 | 140 passed, 21.15s |
| 3.12.13 | 2.8.0 | 4.51.3 | 2.5.3 | 140 passed, 19.11s |
| 3.12.13 | 2.8.0 | 4.56.2 | 2.5.3 | 140 passed, 5.84s |

The legacy torch 2.4.1 Trainer environment explicitly used `numpy<2` for
NumPy interoperability. Reproduce that combination with this constraint;
unconstrained NumPy 2 on old torch was not validated. The core score wheel
also passed in a fresh environment without NumPy, Transformers or W&B.
All three suite environments used accelerate 1.10.1 and W&B 0.30.0.
The configured Linux CI uses these dependency versions but awaits its own run.

Core installation imports torch and this package. Transformers, W&B and plotting are not
imported by the core entry point, and no run/account/model is initialized.

The 0.1.0 core wheel bundles the two fixed paper top-5% head lists and their
provenance. Their variance ranking was independently recomputed from 49 frozen
settings, then reproduced by the new repository selection tool. Research
selection/scaling utilities ship in the source archive. The ordinary wheel
does not import those utilities. Mean/variance tables and plots use setting-level
Common-pair r_H norm shares, preserving the paper's equal-setting weights.
The [head selection guide](HEAD_SELECTION.md) describes preset resolution and
the source-only research API.

| Execution path | Status |
| --- | --- |
| Native Llama and Qwen3, separate Q/K, CPU eager/SDPA | Checked with real tiny randomly initialized Transformers models |
| Qwen3 post-Q/K normalization, pre-RoPE capture | Checked against the actual module output |
| Selected, reordered query heads with GQA | Checked against native KV references; only required KV heads retained |
| Left/right padding, fixed sampling, dynamic frequencies | Checked |
| Same-weight eager/SDPA numerical parity | Checked on native tiny Llama/Qwen3, including actual pre-RoPE Q/K |
| Qwen3 sliding-window attention | Rejected; requires window-aware token eligibility |
| Trainer `model_init` replacement and report history | Checked using a real Trainer; existing forwards and callbacks preserved |
| Prompt prefill in generate | Checked; decode is excluded |
| CPU two-process statistic reduction | Checked using Gloo; per-example weighting |
| Native paper 8B checkpoints, BF16 SDPA on RTX A6000 | Checked: output preservation, sparse Q/K, scores, intervention, decode, restoration, paired latency and memory accounting |
| NCCL/DDP Trainer, tensor parallel, FSDP | Unverified; no global budget or duplicate-row correction |
| Fused QKV/RoPE, FA2, torch.compile, CUDA graphs, vLLM | Separate adapters and checks needed |
| MPS float64 diagnostics | Unsupported by the device; no MPS claim |
| Cached-prefix continuations, beam-expanded streams | Rejected by the initial adapter |
| Packed/reset/fractional positions, internal attention-mask gaps | Rejected |

The above capture restrictions describe the monitor. The repository band-scaling
helper separately supports native Llama/Qwen3 eager/SDPA inference, prefill and
single-token cache decode. Tests compare its local per-band coefficients with
native Q/K and verify shared-KV head isolation, identity restoration and dynamic
frequency timing. Its mask stays fixed to the context's first native frequency
grid. The helper uses Q-only scaling. Native SDPA CUDA inference was checked on
the two paper 8B checkpoints; fused kernels, vLLM, compilation and parallel model
execution require additional validation. CPU products are checked;
optional accelerator product checks synchronize the device.

In 0.1.0a2 the 14 attention-backend cases, 20 report/lifecycle cases and final
29 release-bundle checks passed in all three environments. The main environment
passed its 203-case combined suite followed by the final release checks (six
new cases). Both 4.51.3 environments then passed the final complete 209-case
suite. Executed runs are recorded in the [validation report](VALIDATION.md).

In 0.1.0a3 the final main suite passed 327 tests. Each 4.51.3 environment
passed 326 tests, with one plotting-file check skipped because matplotlib was
not installed there. The main plotting check and four frozen-paper heatmaps
passed with matplotlib 3.11.2. These are CPU validation results.

The model handles its original full input. Token/head sampling limits diagnostic
storage and calculations. Reports distinguish retained Q/K bytes, cached tensor
storage and score-operation stream intervals; these exclude many temporary allocations and
total forward/report/logging cost. Report frequencies are copied to CPU at the
end of eval; input sampling metadata is read on CPU before capture.

CUDA validation used the frozen 0.1.0a3 wheel on Qwen3-8B and
Llama-3.1-8B-Instruct with batch size 1, native BF16 SDPA, and 4Q/16K/256-distance
diagnostics. Five paired repetitions at each of 512, 2048, and 8192 tokens
measured total monitored wall time including reports and synchronization.
FP64 median extra time was about 40–52 ms per forward: roughly 48–55% at 512
tokens, 12–13% at 2048, and 3.4% at 8192. This is evidence for the tested
configuration and device. See [VALIDATION.md](VALIDATION.md) for the per-model
results and distinctions between sampled Q/K memory, cache, and process peak
memory. CUDA stream intervals can include host launch gaps and are not a sum
of pure kernel times. Distance sampling error and token/head selection error
must be measured separately.

See [CI matrix](../.github/workflows/ci.yml), [sampling](REGIONS.md), and
[release procedure](RELEASING.md).
