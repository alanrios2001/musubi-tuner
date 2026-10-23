# FLUX.2 Klein validation loss on Windows

Validation is optional. Existing commands work without validation flags. It uses the current
model and LoRA, existing latent/text cache readers, and the trainer's `process_batch` and
`compute_loss` objective. No second model, VAE, or text encoder is loaded for validation.

## Split before caching

`split_paired_dataset.py` copies a flat, single-control paired image dataset. Original and
target images must have matching unique stems; extensions may differ. Each target must have
a caption. Inputs remain untouched; the output directory must not already exist.

Create `config/source_groups.json` mapping **every target filename** to its original source ID:

```json
{
  "scene_001_edit_a.png": "scene_001",
  "scene_001_edit_b.png": "scene_001",
  "scene_002.png": "scene_002"
}
```

All crops, resized versions, re-encoded images, and edits derived from one original must have
the same source ID. Obtain these IDs from dataset provenance and audit them before splitting.
The script requires this manifest because arbitrary derivatives cannot be reliably identified
from filenames or pixels alone. It additionally groups identical decoded pixels automatically,
across both original and target images, including transitive connections between source groups.
This catches lossless re-encoding and metadata changes, but cannot infer every lossy/cropped derivative.

Run from the repository root, adjusting the two input paths:

```powershell
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
& $Python .\split_paired_dataset.py `
    --original_directory 'D:\datasets\paired_input\original' `
    --target_directory 'D:\datasets\paired_input\target' `
    --groups_json .\config\source_groups.json `
    --output_directory .\datasets\remove_clothes `
    --validation_fraction 0.1 --seed 42
```

Output:

```text
datasets/remove_clothes/
  train/original/        train/target/       cache_train/
  val/original/          val/target/         cache_val/
  split_manifest.json
```

Captions accompany targets. The script reports train/validation pair counts and writes a
manifest with assignments, source IDs, and decoded pixel hashes. Independent groups are
shuffled with a local seeded RNG, then a prefix closest to the requested validation count
is selected without separating any group. The ratio is approximate (especially for large
groups); 3,573 independent pairs produce approximately 3,216 train / 357 validation pairs.
Multi-control/JSONL datasets can use the trainer's existing cache format, but this standalone
splitter intentionally supports one original per target. Training never invokes the splitter.

## Separate dataset configurations

The checked-in examples are `config/dataset_train.toml` and `config/dataset_val.toml`.
All paths are relative to the working directory, so run commands from the repository root.

`config/dataset_train.toml`:

```toml
[general]
resolution = [768, 768]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "datasets/remove_clothes/train/target"
control_directory = "datasets/remove_clothes/train/original"
cache_directory = "datasets/remove_clothes/cache_train"
num_repeats = 1
```

`config/dataset_val.toml`:

```toml
[general]
resolution = [768, 768]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "datasets/remove_clothes/val/target"
control_directory = "datasets/remove_clothes/val/original"
cache_directory = "datasets/remove_clothes/cache_val"
num_repeats = 1
```

Do not add validation as another entry in the training TOML: every training entry is trained.
Do not mix stale caches into either cache directory. The native loader enumerates cache files,
so it cannot determine whether a cache corresponds to an older split or caption. Rebuild caches
in fresh directories after changing splits, captions, or preprocessing.

## Cache both splits once

Use the same model version and preprocessing settings for both. These commands use this
repository's top-level entry points and CLI. Replace model paths with your actual files;
`$TextEncoder` is the first Qwen3 4B safetensors shard, not its directory.

```powershell
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
$DiT = 'D:\models\flux2-klein-base-4b.safetensors'
$Vae = "D:\repos\models\ae.safetensors"
$TextEncoder = "D:\repos\models\text_encoder\model-00001-of-00002.safetensors"

& $Python .\flux_2_cache_latents.py `
    --dataset_config "D:\repos\config\dataset_train.toml" `
    --vae $Vae `
    --model_version klein-base-4b `
    --vae_dtype bfloat16
if ($LASTEXITCODE -ne 0) { throw 'Training latent cache failed' }

& $Python .\flux_2_cache_text_encoder_outputs.py `
    --dataset_config "D:\repos\config\dataset_train.toml" `
    --text_encoder $TextEncoder `
    --model_version klein-base-4b `
    --fp8_text_encoder `
    --batch_size 1
if ($LASTEXITCODE -ne 0) { throw 'Training text cache failed' }

& $Python .\flux_2_cache_latents.py `
    --dataset_config "D:\repos\config\dataset_val.toml" `
    --vae $Vae `
    --model_version klein-base-4b `
    --vae_dtype bfloat16
if ($LASTEXITCODE -ne 0) { throw 'Validation latent cache failed' }

& $Python .\flux_2_cache_text_encoder_outputs.py `
    --dataset_config "D:\repos\config\dataset_val.toml" `
    --model_version klein-base-4b `
    --text_encoder $TextEncoder `
    --fp8_text_encoder `
    --batch_size 1
if ($LASTEXITCODE -ne 0) { throw 'Validation text cache failed' }
```

Latent caching includes targets and matched control images. Text caching includes caption
conditioning. Training and validation use the same native safetensors formats, aspect-ratio
buckets, resolution calculation, and `bucket_no_upscale` behavior. BF16 VAE caching is an
explicit example choice; keep your existing VAE dtype if different, consistently for both splits.

## Train

Four optional arguments are added to the FLUX.2 parser:

| Argument | Behavior |
| --- | --- |
| `--validation_dataset_config PATH` | Enables a separate held-out cached image dataset. |
| `--validation_every_n_steps N` | Positive interval, required with validation; counts completed optimizer steps. |
| `--validation_max_batches N` | Positive optional limit; omitted means the full held-out dataset. |
| `--validation_seed N` | Identity-based noise/timestep seed; default `42`. |

Validation deliberately evaluates one example per batch, regardless of the TOML batch size,
and ignores training repeats. `validation_max_batches` therefore also limits example count.
With a limit, a fixed filename-sorted prefix is evaluated at every checkpoint. Keep the same
seed, subset, files, sampling settings, cache contents, and precision when comparing runs.

Full example, using variables above and the existing Windows Inductor installation:

```powershell
& $Python -m accelerate.commands.launch `
    --num_processes 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 `
    .\flux_2_train_network.py `
    --model_version klein-base-4b `
    --dit $DiT --vae $Vae --text_encoder $TextEncoder `
    --dataset_config .\config\dataset_train.toml `
    --validation_dataset_config .\config\dataset_val.toml `
    --validation_every_n_steps 100 --validation_seed 42 `
    --sdpa --mixed_precision bf16 `
    --fp8_base --fp8_scaled --fp8_text_encoder `
    --optimizer_type adamw8bit --learning_rate 1e-4 `
    --gradient_accumulation_steps 4 --gradient_checkpointing `
    --blocks_to_swap 13 --block_swap_h2d_only --block_swap_ring_size 2 `
    --compile --compile_backend inductor --compile_mode default `
    --max_data_loader_n_workers 2 --persistent_data_loader_workers `
    --network_module networks.lora_flux_2 --network_dim 64 --network_alpha 64 `
    --timestep_sampling flux2_shift --weighting_scheme none `
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 `
    --log_with tensorboard --logging_dir .\logs `
    --output_dir .\output --output_name klein_edit
```

Use your existing safe swap count if different. Full validation runs every 100 optimizer steps
(400 microsteps with accumulation 4). To cap cost, add `--validation_max_batches 64`. The
example evaluates all held-out pairs. Remove all validation arguments to retain the original
training path without constructing a validation loader.

Existing image sampling remains available in the same run, for example by appending
`--sample_prompts .\config\sample_prompts.txt --sample_every_n_steps 100` with an existing
valid prompt file. Step-triggered sampling and checkpoint saving finish before validation.
Existing epoch-end sampling/saving still runs afterward. Validation itself does not generate
images or encode inputs.

## TensorBoard

```powershell
& $Python -m tensorboard.main --logdir .\logs --port 6006
```

Open `http://localhost:6006`. In Scalars select `train/loss` and `val/loss` from the same run.
The former aliases existing `loss/current`; existing `loss/current`, `loss/average`, and epoch
metrics remain available. `val/num_samples` and `val/evaluation_seconds` are also logged.
All use the existing Accelerator trackers and the global optimizer-step axis.

With validation and a tracker enabled, memory diagnostics are logged automatically:

| TensorBoard tag/prefix | Meaning |
| --- | --- |
| `train/optimizer_step_seconds` | Time for the accumulated training step, including backward and optimizer work, before periodic saving and validation. |
| `memory/train/*` | Memory at every completed optimizer step, before periodic validation. |
| `memory/before_validation/*` | Memory immediately before entering validation. |
| `memory/validation_sampled_peak/*` | Maximum of the measurements taken after each validation batch. |
| `memory/validation_peak/*` | PyTorch allocator high-water marks across validation, including temporary allocations between samples. |
| `memory/validation_end/*` | Memory after releasing validation batch tensors, before restoring training execution. |
| `memory/after_validation/*` | Memory after restoring training modes and preparing block placement. |

Memory values use MiB. `allocated_mib` measures live PyTorch CUDA allocations;
`reserved_mib` includes unused memory held by its caching allocator. On Windows,
`process_dedicated_mib` and `process_shared_mib` also read native GPU Process Memory
performance counters for the training PID, summed across its adapters. These process
counters include memory outside the PyTorch allocator. Shared memory can include pinned
transfer buffers as well as driver fallback; a nonzero value alone does not prove fallback.
Unavailable Windows counters are omitted. No extra dependency or subprocess is required.

Compare `memory/train/process_shared_mib` with `train/optimizer_step_seconds`, and
compare `memory/before_validation/*` with `memory/after_validation/*` at the same step.
The Windows peak is sampled once per batch and can miss shorter spikes; the PyTorch
peak uses allocator statistics reset just before each validation. These readings do not
add explicit CUDA synchronization, clear caches, or change model placement. Diagnostics
start on the next training launch; an already running Python process will not load them.

### Windows CUDA cache budget

FLUX.2 training with block swap on Windows now caps PyTorch's CUDA allocator at
85% of physical VRAM by default, with or without validation or logging. On an
8 GiB GPU this is approximately 6963 MiB. This leaves room for the CUDA context,
desktop, and allocations outside PyTorch. `--cuda_memory_fraction 0.80` changes
the budget; `--cuda_memory_fraction 1` opts out of this headroom. A stricter
allocator limit already set by a caller is preserved. Other platforms and
training without block swap keep their existing default unless the flag is set.

Variable training bucket sizes can grow the native allocator's cached segments
well beyond the live tensors. Windows can satisfy new CUDA allocations using
shared RAM instead of returning an allocation failure; in that case the
allocator does not reach its usual cache-release-and-retry path. Setting the
budget makes the allocator reclaim unused cached segments before expanding
beyond that budget. No per-step `empty_cache()` or model/optimizer changes are
needed. The new budget is installed before model loading.

This bounds PyTorch's allocator, not total GPU usage. If live tensors themselves
cannot fit within the budget, training reports an out-of-memory error rather
than relying on shared RAM; increase block swap or lower resolution in that case.
Confirm the change with `memory/train/reserved_mib` and
`memory/train/process_shared_mib`, alongside `train/optimizer_step_seconds`.

## Execution and reproducibility

- A small shared lifecycle hook runs outside accumulation, after step increment, weight-norm
  regularization, and periodic sampling/saving. Only FLUX.2 implements validation in that hook.
- Validation calls the same `process_batch`, FLUX.2 `call_dit`, noise/timestep sampler, and
  weighted MSE reduction as training. The flow target remains `noise - latents`; controls and
  text use the same packing and dtype paths. Means are weighted by evaluated example count.
- Intentional execution differences: `eval()` disables LoRA dropout and bypasses checkpoint
  recomputation; `torch.no_grad()` avoids graphs and new gradients. Checkpointing and CPU
  activation-offload flags remain unchanged. Validation does not access the optimizer or scheduler.
- SHA-256 of validation seed plus cached item identity (filename stem) seeds each example.
  Singleton evaluation keeps assignments independent of order and configured batch size. No
  timestep pool is borrowed from training; validation does not enforce epoch-wide stratification
  from `num_timestep_buckets`, but uses the same sampling distribution and restrictions.
- Python, NumPy, PyTorch CPU, and initialized CUDA device RNG states are restored, including
  failure paths. Each module's individual mode and the trainer timestep pool are restored.
  The training loader, epoch, sampler, and accumulation state are never advanced by validation.
- The existing swap transition selects forward-only offloading, then restores each offloader's
  mode and prepares training block placement again. H2D CPU masters and ring buffers are reused;
  the complete model is never moved to GPU. Cached tensors are loaded only for the current batch.
- Compiled blocks stay compiled. Eval/no-grad can create a separate graph specialization per
  bucket on first validation; initial evaluation may be slower. No compiler or memory optimization
  is silently disabled. CUDA graph modes and attention kernels retain their existing constraints.
- Stable draws do not force globally deterministic CUDA kernels. Repeated results should match
  within floating-point tolerance on unchanged hardware/software. Cross-device bitwise equality
  is not promised. Distributed ranks currently evaluate the same complete subset independently,
  avoiding padded-example duplication; this is correct but does not parallelize validation.

Validation catches shared resolved training/validation cache or target directories, missing text
caches for discovered latents, empty caches, and missing required tensors when batches load.
It does not re-hash source images during training. Use the audited split manifest and fresh caches
to prevent content leakage. Validation requires paired control caches; ordinary text-to-image
FLUX.2 training remains unchanged when validation is disabled.

## Files and verification

- `src/musubi_tuner/training/flux2_validation.py`: cached loader, deterministic evaluator, state guards.
- `src/musubi_tuner/flux_2_train_network.py`: optional arguments, dataset setup, periodic logging.
- `src/musubi_tuner/training/trainer_base.py`: minimal completed-step lifecycle hook.
- `split_paired_dataset.py`, `config/dataset_{train,val}.toml`: offline split utility and examples.
- `tests/test_flux2_validation.py`, `tests/test_split_paired_dataset.py`: focused regressions.

Run focused tests in the existing environment (install `pytest` if absent):

```powershell
& $Python -m pytest .\tests\test_flux2_validation.py .\tests\test_split_paired_dataset.py -q
```

Verified in the local Windows `.venv` on an RTX 3070 Ti with PyTorch 2.6.0+cu124:

- Focused tests plus neighboring timestep, gradient metric, datasource, and entry-point regressions:
  44 passed. Ruff checks and `git diff --check` passed.
- CUDA smoke tests instantiate a **reduced synthetic Klein architecture**, use real FLUX.2
  conditioning, BF16, SDPA, LoRA, gradient checkpointing, standard/H2D block swapping, repeated
  validation, and then resume backward/optimizer updates. One variant combines scaled FP8 base
  weights, H2D streaming, and compiled blocks. Repeated validation checks live CUDA allocation
  for growth. Compile tests use Dynamo's `eager` backend; Triton is absent locally.
- An actual Accelerator training-loop test verifies accumulation boundaries, three validation
  intervals, resumed training, native sampling scheduling (image generation mocked), and real
  TensorBoard event files. Matched enabled/disabled runs produce identical final LoRA tensors.

These tests do not establish full pretrained 4B, FP8 text encoding, AdamW8bit, Inductor, or
production-resolution VRAM compatibility. Those combinations require an actual run with your
weights and working Windows compiler stack. No optional best-checkpoint saving was added;
ordinary saving behavior remains unchanged.
