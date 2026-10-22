# FLUX.2 Klein LoRA-GA

Optional LoRA-GA initialization is available for **Klein Base 4B**, with the existing
`networks.lora_flux_2` Linear adapters. Without `--loraga`, initialization, training,
and saving retain their existing behavior. PEFT is not a runtime dependency.

Source ownership:

| File | Responsibility |
| --- | --- |
| `networks/loraga.py` | Reference factor construction and exact conversion |
| `training/loraga_streaming.py` | Initialization-only autograd, bounded RAM/CUDA gradient storage, weight cache and disk spill |
| `training/flux2_loraga.py` | CLI validation, estimation, probes, compensation, native FP8, cache/resume metadata |
| `flux_2_train_network.py` | FLUX.2 lifecycle integration and TensorBoard metrics |
| `training/trainer_base.py` | Optional initialization-only exit and resume cursor/loader handling |
| `networks/lora.py` | Optional automatic inference export when initial factors are attached |
| `loraga_export.py` | Manual conversion of raw training-state adapters |
| `tests/test_loraga.py` | Focused CPU/CUDA and full-loop synthetic tests |
| `tests/test_loraga_performance.py` | Storage/caching parity, BF16 tensor-core gradients and CUDA SVD subspaces |

Python paths above are under `src/musubi_tuner`, except tests. This document and
the link in `docs/flux_2.md` cover usage and limitations.

## Algorithm and execution

The factor construction follows the [official implementation](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py#L159),
[paper](https://arxiv.org/html/2407.05000v2), and the adapter-difference conversion
used by [PEFT](https://github.com/huggingface/peft). The upstream implementation
was checked directly for all nine supported direction/scale combinations and
three-batch gradient accumulation. No GaLore or reduced-rank approximation is used.

1. Load the original checkpoint in BF16 on CPU. Reject prequantized FP8 checkpoints.
2. Use the existing cached training batches, paired control latents, noise/timestep
   sampler, and weighted flow-matching loss. No VAE or text encoder is needed for
   gradient estimation. Estimation batches do not advance optimizer steps or epochs.
3. Temporarily mark target weights as requiring gradients. Initialization uses its
   own autograd Linear: CPU BF16 weights are copied for each forward/backward
   operation unless retained in the initialization weight cache. Saved activations
   go to pinned CPU memory during CUDA estimation, allowing asynchronous retrieval
   for backward. FP32 weight gradients accumulate in bounded RAM; excess gradients
   spill to disk. After the first batch measures actual working VRAM, disk-spilled
   gradients are promoted to spare CUDA memory first, and remaining space caches
   immutable BF16 weights across forward, recomputation, and backward. No backbone
   optimizer exists. All blocks use gradient checkpointing during this phase.
4. Process each average gradient separately with FP32 `torch.svd_lowrank`,
   `q=min(4*r,min(shape))`, four power iterations. Default `ArB2r` uses
   `B=U[:,r:2r]`, `A=V[:,:r].T`. `stable` multiplies both factors by
   `out_features**0.25 / sqrt(gamma)`, with upstream gamma default 16.
   SVD runs on CUDA by default when its estimated workspace fits, otherwise CPU.
   Factors return to CPU after each SVD; full backbone gradients never reside in VRAM.
5. Save FP32 initial factors and compatibility metadata. Compensate each CPU
   backbone matrix: `Wc = BF16(FP32(W) - (alpha/r)*(B0@A0))`.
6. Measure deterministic cached-batch output and weight errors. Then use Musubi's
   existing FP8 optimizer and Linear patch on the compensated weights. Freeze the
   backbone, attach initialized FP32 trainable LoRA factors, and enter ordinary
   training. Initialization streaming is removed before block swapping or compilation.

`A` is `lora_down.weight` with shape `[rank,in]`; `B` is `lora_up.weight`
with shape `[out,rank]`. BF16 compensation/forward rounding prevents exact
floating-point function equality; the underlying FP32 algebra is preserved.

The initialization streamer is deliberately separate from Musubi's frozen-weight
H2D-only ring buffers. It transfers weights during backward and supports actual
backbone-weight gradients. Bounded RAM/CUDA accumulation avoids repeated disk I/O
where memory permits, and retained weights reduce H2D traffic. Ordinary training retains its
existing kernels, optimizer, checkpointing, and swap machinery.

For BF16 operands only, the initialization weight-gradient GEMM may use TF32 tensor
cores with FP32 accumulation: TF32 represents every BF16 input value without further
operand truncation. The global matmul flag is restored immediately afterwards;
CPU/FP32-input gradient calculations and SVD precision are unchanged. See the
[PyTorch precision documentation](https://github.com/pytorch/pytorch/blob/v2.6.0/docs/source/notes/cuda.rst#tensorfloat-32-tf32-on-ampere-and-later-devices).

## Arguments

| Argument | Default / meaning |
| --- | --- |
| `--loraga` | Enable two-stage initialization |
| `--loraga_estimation_batches` | `64`; average this many native batches; cycle the loader if necessary |
| `--loraga_seed` | `42`; private estimation/SVD RNG seed |
| `--loraga_direction` | `ArB2r`; also `A2rBr`, `ArBr` |
| `--loraga_scale` | `stable`; also `unit`, `gd` |
| `--loraga_stable_gamma` | `16`; positive finite value |
| `--loraga_init_cache` | Safetensors filename; default `<output_dir>/loraga_init.safetensors` |
| `--loraga_init_only` | Create/reuse and validate initialization, then exit before optimizer setup |
| `--loraga_work_dir` | Disk gradient spill workspace; default system temporary directory |
| `--loraga_gradient_ram_gb` | `-1`: automatic RAM budget leaving 3 GiB free; `0`: no RAM accumulation; positive values set GiB |
| `--loraga_gpu_cache_gb` | `-1`: spare VRAM measured after first backward; `0`: disable CUDA gradient/weight residency; positive values cap GiB |
| `--loraga_gpu_reserve_gb` | `1`: VRAM margin, in addition to measured transient working memory plus 25% growth room |
| `--loraga_svd_device` | `auto`: CUDA when estimated workspace fits; also explicit `cpu` or `cuda` |
| `--loraga_max_quantization_error_ratio` | `2.0`; fail if aggregate weight or probe-output RMSE exceeds this multiple of ordinary FP8 RMSE |

Other upstream direction/scale choices (`random`, `weightS`) and norm clipping are
not exposed. Network kwargs, split adapters, dropout, other network types,
`--network_weights`, `--dim_from_weights`, and merged `--base_weights` are rejected.
Alpha 0/None follows Musubi's existing convention: alpha equals rank.

To reproduce the old storage/device strategy, use `--loraga_gradient_ram_gb 0
--loraga_gpu_cache_gb 0 --loraga_svd_device cpu`. The execution options above do not
invalidate a completed initialization cache. Fresh CPU and CUDA randomized SVDs
can choose different signs/random bases and differ in floating-point rounding;
they use the same reference algorithm and are not promised bit-for-bit identical.

## Windows PowerShell

Run from the repository root using the existing environment. Replace model and
dataset paths. The dataset TOML must already point to cached target/control latents
and cached text conditioning. Keep the full-precision original checkpoint.

```powershell
$env:PYTHONUTF8 = '1'
$python = '.\.venv\Scripts\python.exe'
$trainArgs = @(
    '--model_version', 'klein-base-4b',
    '--dit', 'D:\models\flux-2-klein-base-4b.safetensors',
    '--dataset_config', 'D:\data\editing\train.toml',
    '--network_module', 'networks.lora_flux_2',
    '--network_dim', '64', '--network_alpha', '64',
    '--mixed_precision', 'bf16', '--fp8_base', '--fp8_scaled',
    '--sdpa', '--gradient_checkpointing',
    '--blocks_to_swap', '17', '--block_swap_h2d_only',
    '--block_swap_ring_size', '2',
    '--optimizer_type', 'adamw8bit', '--learning_rate', '1e-4',
    '--lr_scheduler', 'constant', '--max_train_steps', '1000',
    '--max_data_loader_n_workers', '0', '--seed', '42',
    '--output_dir', 'D:\runs\klein-ga', '--output_name', 'editing',
    '--logging_dir', 'D:\runs\klein-ga\logs', '--log_with', 'tensorboard',
    '--save_every_n_steps', '100', '--save_state'
)
$gaArgs = @(
    '--loraga', '--loraga_estimation_batches', '64',
    '--loraga_seed', '42', '--loraga_direction', 'ArB2r',
    '--loraga_scale', 'stable', '--loraga_stable_gamma', '16',
    '--loraga_init_cache', 'D:\runs\klein-ga\init.safetensors',
    '--loraga_work_dir', 'D:\loraga-work',
    '--loraga_gradient_ram_gb', '-1', '--loraga_gpu_cache_gb', '-1',
    '--loraga_gpu_reserve_gb', '1', '--loraga_svd_device', 'auto'
)

# Standalone initialization; no optimizer or training steps.
& $python -m accelerate.commands.launch --num_processes 1 flux_2_train_network.py @trainArgs @gaArgs --loraga_init_only
if ($LASTEXITCODE -ne 0) { throw 'LoRA-GA initialization failed' }

# Training reuses that validated cache.
& $python -m accelerate.commands.launch --num_processes 1 flux_2_train_network.py @trainArgs @gaArgs
if ($LASTEXITCODE -ne 0) { throw 'LoRA-GA training failed' }

# Resume: same arguments and original planned training budget.
& $python -m accelerate.commands.launch --num_processes 1 flux_2_train_network.py @trainArgs @gaArgs `
    --resume 'D:\runs\klein-ga\editing-step00000100-state'
```

Adjust resolution/batch size in the cache configuration and block count to available
memory. These are example commands, not a measured 768px/8GB capacity guarantee.
`--compile` uses the existing compiler path after initialization; Windows still
needs a working backend/Triton installation. `--compile_backend eager` can exercise
the integration without claiming Inductor performance.

For BF16 backbone training, omit both FP8 switches. Existing attention modes remain
subject to their ordinary trainer restrictions; SageAttention training is already
rejected by Musubi. Existing periodic validation and image sampling are unchanged.

## Cache identity and resume

Cache reuse validates SHA256 of every original checkpoint shard, dataset TOML,
latent/control cache, and conditioning cache; absolute cache/checkpoint paths;
model, rank/alpha, target names/shapes, initialization seed/options, objective and
timestep configuration, attention implementation, Torch version, and FP8 settings. Relocating those files invalidates the
cache. This strict check reads the files on reuse and may take time. Corrupt/missing
factors fail validation. Writes publish atomically after successful estimation.

An Accelerate state directory contains raw rank-r adapter weights, optimizer,
scheduler, and native RNG state, plus:

- `loraga_init.safetensors`: initial factors, identity, and initialization metrics.
- `loraga_progress.json`: global optimizer step, epoch, next batch, shuffled dataset
  order, dataset timestep pools, trainer timestep pool, and optimizer/scheduler signature.

Resume reconstructs and requantizes the same compensated original backbone before
loading raw trainable factors and optimizer state. It never re-estimates gradients.
Changing optimizer/scheduler/batch settings, checkpointing/compilation/swapping
configuration, or the planned step budget is rejected.
Missing metadata and resuming without `--loraga` fail explicitly.

Current resume support is **one process, zero DataLoader workers**, with no
persistent workers and a local state directory. Native single-worker cached batch
ordering and private loader RNG are restored. Mid-epoch optimizer-boundary resume
with gradient accumulation and AdamW8bit was tested bit-for-bit against an
uninterrupted synthetic FLUX.2 run. No distributed or cross-version determinism
claim is made.

## Standard inference export

Every ordinary adapter save automatically exports a standard adapter containing
the exact factor difference:

```text
A_export = concat(A_final, A_initial, axis=0)
B_export = concat(B_final, -B_initial, axis=1)
rank_export = 2 * rank
alpha_export = 2 * alpha
```

Thus `(alpha_export/rank_export) * B_export@A_export` equals
`(alpha/rank) * (B_final@A_final - B_initial@A_initial)` in real arithmetic.
Rank 64 / alpha 64 exports rank 128 / alpha 128. There is no silent compression.
Musubi tensor names and ordinary adapter-strength semantics are preserved.

Use these exported files with the **original** backbone through normal inference.
No compensated base checkpoint or initialization sidecar is needed for inference.
Loading the adapter as a runtime branch retains small updates better than merging
them into BF16/FP8 storage. Follow the existing generator's adapter loading options.

For a raw Accelerate adapter, manual conversion is available:

```powershell
& $python -m musubi_tuner.loraga_export `
    --weights 'D:\runs\klein-ga\editing-step00000100-state\model.safetensors' `
    --init 'D:\runs\klein-ga\editing-step00000100-state\loraga_init.safetensors' `
    --output 'D:\runs\klein-ga\editing-export.safetensors'
```

Do not convert an already exported adapter. Resume from the state directory, never
the rank-128 inference adapter. Conversion verifies module shapes and alpha; use
the matching initialization file, since raw adapter tensors alone cannot identify
which run produced them.

FP8 training uses quantized **compensated** weights; inference uses quantized
**original** weights. Their quantization errors differ. Conversion is algebraically
exact for the low-rank difference, not an exact reconstruction of either FP8
rounding residual or the dense BF16 compensation-rounding residual.

## Numerical validation and logging

Before training, one fixed cached batch (including control conditioning) is probed
under original BF16, ordinary FP8, compensated BF16 + initial adapter, and
compensated FP8 + initial adapter representations. Native block/channel FP8
quantization and BF16 dequantization are reused for probes; full-model FP32 copies
are not created. Weight RMSE is aggregated across targeted matrices.

The default ratio limit of 2 is an explicit engineering gate, **not a paper result
or a calibrated Klein quality threshold**. Nonfinite outputs/compensation fail.
Inspect weight/output errors and held-out generations before accepting another
limit. A single probe batch does not establish dataset-wide quality equivalence.
Pre-quantization output error is reported separately because BF16 cancellation
rounding remains even when the factor identity is correct.

Initialization metrics are logged separately to TensorBoard on the first ordinary
training batch. Initialization-only runs print metrics to the console. Metrics
include initialization and estimation/SVD time, estimated batches, cache reuse,
peak estimation VRAM, BF16 output RMSE/relative RMSE, ordinary/compensated FP8
output RMSE, weight RMSE, and both FP8 error ratios. Training loss tags are preserved.
Additional performance metrics separate gradient collection and SVD time, report
RAM/CUDA cache budgets and weight-cache hits, and count gradient payload bytes
read/written to disk. Each gradient batch and each matrix SVD logs its duration.

Historical synthetic fixture measurements before RAM/CUDA caching, on Windows,
RTX 3070 Ti, Torch 2.6.0+cu124, BF16,
two estimated batches, rank/alpha 2/2, hidden size 32, two double/two single blocks:

| Measurement | Observed |
| --- | --- |
| Estimation + SVD | 1.735 s |
| Peak allocated estimation VRAM | 17,113,088 bytes (16.32 MiB) |
| BF16 compensation weight RMSE | 0.0001633 |
| Ordinary FP8 weight RMSE | 0.0023417 |
| Compensated FP8 + initial factors weight RMSE | 0.0024078 |
| Weight error ratio | 1.0282 |
| BF16 restored output RMSE | 0.0005611 |
| BF16 restored output relative RMSE | 0.0010963 |
| Ordinary FP8 output RMSE | 0.0021162 |
| Compensated FP8 + initial adapter output RMSE | 0.0018679 |
| Output error ratio | 0.8827 |

These are fixture results, not evidence of LoRA-GA improving Klein training.
Four-step synthetic swap tests had constant allocated VRAM after warmup:
17,414,656 bytes with ordinary swapping and 17,445,376 bytes with H2D-only swapping.
CUDA context/driver/reserved memory is additional. The full-size pretrained Base 4B
checkpoint and native paired caches subsequently became available: real-model
gradient collection and all 80 SVDs were tested with three batches, rank/alpha
64/64, the existing 768px dataset configuration, and the actual flow-matching loss.
The optimized phase completed in 48.35 seconds (40.24 seconds collecting gradients,
7.88 seconds for SVD). Batch times were 23.39, 6.85, and 6.27 seconds. It used
12.21 GiB of CPU gradient capacity, 1.51 GiB of CUDA gradients, 2.74 GiB of immutable
CUDA weights, and peaked at 5.14 GiB of PyTorch-allocated VRAM. Driver, display,
and allocator-reserved memory are additional. Only 1.51 GiB was written and read
once for initial spill promotion; repeated accumulation used RAM/CUDA.

With pinned activation retrieval enabled, another three-batch run completed in
34.00 seconds (25.85 seconds collecting gradients, 7.95 seconds for SVD), with
batch times 12.18, 6.52, and 6.23 seconds and 4.99 GiB peak allocated VRAM. That run
had more free RAM: its auto plan used 13.24 GiB CPU gradients, 0.53 GiB CUDA
gradients and 3.73 GiB CUDA weights. These auto-budget measurements are not a
controlled attribution of the difference to pinned transfers alone.

The same checkpoint/data/settings with disk accumulation and CPU SVD took 128.17
seconds for its first batch alone. These are initialization performance results,
not convergence or visual-quality comparisons. The disk-only benchmark was stopped
after that measured first batch to avoid further expensive I/O; no full baseline
three-batch speed ratio is claimed. Full 64-batch initialization,
full-size compensated FP8 optimization/export, VAE decoding, and generated-image
consistency remain unmeasured. Synthetic tests still cover those lifecycle
integrations at reduced geometry.

## Memory and storage

Model-shape accounting for Klein Base 4B at rank 64 / alpha 64:

| State | Size |
| --- | --- |
| Backbone | 3,875,544,576 parameters; approximately 7.22 GiB in CPU BF16 |
| Target weights | 3,680,501,760 parameters in 80 modules |
| Gradient accumulators | Approximately 13.71 GiB FP32, distributed over bounded RAM/CUDA and disk spill |
| Initial adapter factors | 92,405,760 FP32 parameters; approximately 352.5 MiB |
| Largest target gradient | 84,934,656 FP32 values; approximately 324 MiB |

CPU RAM additionally holds saved activations, a current accumulator, SVD workspace,
and temporary compensation/quantization matrices. Native CPU quantization can
temporarily retain BF16 and FP8 model storage together. Disk free space is checked
before estimation for the amount exceeding the RAM budget. With RAM accumulation
disabled, reserve at least 16 GiB for gradients, plus cache/state/output files.
CUDA spill promotion can eliminate repeated disk reads/writes after the first
batch, but the initial spill still needs space. A fast SSD is recommended.
No universal system-RAM or 8GB-VRAM fit is promised:
resolution, batch size, attention implementation, other applications, and paging
all matter. Initialization reports resource exhaustion rather than changing the
mathematical procedure. The measured three-batch full-size VRAM result is reported
above; it does not prove the maximum of every bucket or other running applications.
If another bucket exceeds the measured headroom, lower `--loraga_gpu_cache_gb` or
increase `--loraga_gpu_reserve_gb`. For host-memory pressure, lower
`--loraga_gradient_ram_gb`. A running Python process keeps the old implementation
until restarted; changing source files does not patch an active initialization.

## Tests and comparison protocol

```powershell
& $python -m pytest tests/test_loraga.py tests/test_loraga_performance.py `
    tests/test_flux2_validation.py tests/test_lora_dtype_bridging.py `
    tests/test_save_precision.py tests/test_top_level_entrypoints.py -q
```

Coverage includes upstream factor formulas and gradient averaging, CPU/CUDA
gradients with checkpoint recomputation, unchanged estimation weights, BF16
compensation, native FP8 errors, exact rank-difference conversion, native inference
loading onto BF16 and ordinary FP8, conventional zero-up initialization, atomic
cache validation/reuse, raw Accelerate state restoration, real TensorBoard training,
AdamW8bit accumulated-gradient resume, gradient checkpointing, ordinary and
H2D-only block swapping, stable allocations after warmup, and compilation with the
`eager` backend. Inductor and full-sized image-quality tests remain unmeasured.

Recorded results: **81 tests passed**: 30 LoRA-GA lifecycle tests, 9 performance/
storage correctness tests, and 42 focused existing regression tests (validation,
dtype bridging, save precision, and top-level entry points).
The full-loop resume test also covers epoch-boundary and already-completed states.
Ruff and `git diff --check` passed. Existing block-swapping pin-memory deprecation
warnings were observed; no new test failures remain.

The supplied `D:\repos\train.ps1` now calls the training entry point once: it
initializes when no compatible cache exists and then trains in the same process.
An additional initialization-only invocation is unnecessary for that workflow;
the standalone initialization command above remains available when explicitly needed.

For a controlled comparison, run the PowerShell `$trainArgs` command twice with
separate output/log directories: first without `$gaArgs`, then with it. Pin the
checkpoint and cache hashes, seed, rank/alpha 64/64, optimizer, LR/scheduler,
effective batch size, attention, precision, swapping, and total optimization steps.
Use the same independent cached validation dataset with:

```powershell
$validationArgs = @(
    '--validation_dataset_config', 'D:\data\editing\validation.toml',
    '--validation_every_n_steps', '50', '--validation_seed', '1234'
)

# Conventional initialization, identical training and validation settings.
& $python -m accelerate.commands.launch --num_processes 1 flux_2_train_network.py @trainArgs @validationArgs `
    --output_dir 'D:\runs\klein-baseline' --logging_dir 'D:\runs\klein-baseline\logs'
if ($LASTEXITCODE -ne 0) { throw 'Baseline comparison failed' }

# LoRA-GA, same optimization budget; initialization time is additional.
& $python -m accelerate.commands.launch --num_processes 1 flux_2_train_network.py @trainArgs @gaArgs @validationArgs `
    --output_dir 'D:\runs\klein-ga-comparison' --logging_dir 'D:\runs\klein-ga-comparison\logs'
if ($LASTEXITCODE -ne 0) { throw 'LoRA-GA comparison failed' }
```

Keep the LoRA-GA cache outside run directories. Repeat paired seeds (for example
42, 43, 44), preserving all logs and environment versions. Record `loss/current`,
`loss/average`, `val/loss`, optimizer steps to a prespecified validation threshold,
steady-state steps/second after warmup, initialization time separately and total
wall time including initialization. Capture GPU allocated/reserved peaks in the
training process or sample `nvidia-smi` consistently for both runs; state explicitly
which measurement was used. Sampling/validation time should be accounted for
identically in throughput comparisons.

At fixed checkpoints, generate held-out edits using identical references,
prompts, initial noise, sampler, steps, and guidance. Compare training-representation
samples against rank-128 exports on original BF16 and ordinary FP8 backbones;
record output-latent RMSE and decoded-image differences, and assess held-out visual
quality blind to initialization method. Select acceptable full-model tolerances
before judging quality. BF16 fixture output tests use RMSE <0.02; ordinary-FP8
export fixture tests use RMSE <0.06. These limits are not pretrained-Klein image
tolerances. No training-quality advantage is assumed before this comparison.
