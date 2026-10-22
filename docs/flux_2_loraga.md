# FLUX.2 Klein LoRA-GA

Optional LoRA-GA initialization is available for **Klein Base 4B**, with the existing
`networks.lora_flux_2` Linear adapters. Without `--loraga`, initialization, training,
and saving retain their existing behavior. PEFT is not a runtime dependency.

Source ownership:

| File | Responsibility |
| --- | --- |
| `networks/loraga.py` | Reference factor construction and exact conversion |
| `training/loraga_streaming.py` | Initialization-only autograd, CPU offload, disk gradients |
| `training/flux2_loraga.py` | CLI validation, estimation, probes, compensation, native FP8, cache/resume metadata |
| `flux_2_train_network.py` | FLUX.2 lifecycle integration and TensorBoard metrics |
| `training/trainer_base.py` | Optional initialization-only exit and resume cursor/loader handling |
| `networks/lora.py` | Optional automatic inference export when initial factors are attached |
| `loraga_export.py` | Manual conversion of raw training-state adapters |
| `tests/test_loraga.py` | Focused CPU/CUDA and full-loop synthetic tests |

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
   operation. Saved activations go to CPU. FP32 weight gradients are immediately
   accumulated to one disk file per module; no backbone optimizer exists.
   All blocks use gradient checkpointing during this phase.
4. Process each average gradient separately with CPU FP32 `torch.svd_lowrank`,
   `q=min(4*r,min(shape))`, four power iterations. Default `ArB2r` uses
   `B=U[:,r:2r]`, `A=V[:,:r].T`. `stable` multiplies both factors by
   `out_features**0.25 / sqrt(gamma)`, with upstream gamma default 16.
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
backbone-weight gradients. Disk I/O and repeated H2D transfers make initialization
substantially slower than ordinary LoRA steps. Ordinary training retains its
existing kernels, optimizer, checkpointing, and swap machinery.

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
| `--loraga_work_dir` | Disk gradient workspace; default system temporary directory |
| `--loraga_max_quantization_error_ratio` | `2.0`; fail if aggregate weight or probe-output RMSE exceeds this multiple of ordinary FP8 RMSE |

Other upstream direction/scale choices (`random`, `weightS`) and norm clipping are
not exposed. Network kwargs, split adapters, dropout, other network types,
`--network_weights`, `--dim_from_weights`, and merged `--base_weights` are rejected.
Alpha 0/None follows Musubi's existing convention: alpha equals rank.

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
    '--loraga_work_dir', 'D:\loraga-work'
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

Synthetic fixture measurements on Windows, RTX 3070 Ti, Torch 2.6.0+cu124, BF16,
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
CUDA context/driver/reserved memory is additional. Full-sized Klein checkpoint,
VAE decoding, and generated-image consistency were not tested: the required local
Base 4B checkpoint and usable cached dataset were not supplied. GPU tests exercise
real FLUX.2 modules at reduced geometry, not a pretrained 4B model.

## Memory and storage

Model-shape accounting for Klein Base 4B at rank 64 / alpha 64:

| State | Size |
| --- | --- |
| Backbone | 3,875,544,576 parameters; approximately 7.22 GiB in CPU BF16 |
| Target weights | 3,680,501,760 parameters in 80 modules |
| Disk gradient accumulators | Approximately 13.71 GiB FP32, plus temporary file headroom |
| Initial adapter factors | 92,405,760 FP32 parameters; approximately 352.5 MiB |
| Largest target gradient | 84,934,656 FP32 values; approximately 324 MiB |

CPU RAM additionally holds saved activations, a current accumulator, SVD workspace,
and temporary compensation/quantization matrices. Native CPU quantization can
temporarily retain BF16 and FP8 model storage together. Disk free space is checked
before estimation; reserve at least 16 GiB for gradients, plus cache/state/output
files. A fast SSD is recommended. No fixed system-RAM or 8GB-VRAM fit is promised:
resolution, batch size, attention implementation, other applications, and paging
all matter. Initialization reports resource exhaustion rather than changing the
mathematical procedure. Full-sized peak VRAM remains unmeasured.

## Tests and comparison protocol

```powershell
& $python -m pytest tests/test_loraga.py tests/test_flux2_validation.py `
    tests/test_lora_dtype_bridging.py tests/test_save_precision.py -q
```

Coverage includes upstream factor formulas and gradient averaging, CPU/CUDA
gradients with checkpoint recomputation, unchanged estimation weights, BF16
compensation, native FP8 errors, exact rank-difference conversion, native inference
loading onto BF16 and ordinary FP8, conventional zero-up initialization, atomic
cache validation/reuse, raw Accelerate state restoration, real TensorBoard training,
AdamW8bit accumulated-gradient resume, gradient checkpointing, ordinary and
H2D-only block swapping, stable allocations after warmup, and compilation with the
`eager` backend. Inductor and full-sized image-quality tests remain unmeasured.

Recorded results: **30 LoRA-GA tests and 42 focused existing regression tests
passed** (validation, dtype bridging, save precision, and top-level entry points).
The full-loop resume test also covers epoch-boundary and already-completed states.
Ruff and `git diff --check` passed. Existing block-swapping pin-memory deprecation
warnings were observed; no new test failures remain.

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
