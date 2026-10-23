# PoLoRA for FLUX.2 LoRA

PoLoRA is an optional optimizer for Musubi's existing Linear LoRA modules. It
does not replace the adapter, frozen backbone, FP8 implementation, training
objective, sampling, or checkpoint format. Conventional optimizers remain the
default. No PEFT or external PoLoRA installation is required.

## Implementation

The Apache-2.0 reference is vendored unchanged from
[nikhilgsh/polora](https://github.com/nikhilgsh/polora/tree/c9c6e2b7d6977213b152b80243189f9515080b48).
The Musubi wrapper maps `lora_down.weight` to A and `lora_up.weight` to B.
It uses the reference curvature, momentum, polar iteration and spectral
normalization. The reference groups equal-shaped pairs; the wrapper limits
each group to `pair_batch_size` pairs, default 4, to bound temporary memory.
Pair updates are independent. Different batch sizes can produce floating point
rounding differences; they are not approximations of the algorithm.

Unlike the original optimizer, the wrapper serializes every FP32 moment,
diagonal preconditioner and warm-start spectral vector. It migrates these
states when Musubi moves the adapter to CUDA. Numerical optimizer operations
run outside autocast. A training sidecar also restores the step, epoch, batch
cursor and shuffled dataset descriptors. Accelerate restores the adapter,
optimizer, scheduler and RNG states. LoRA-GA uses its existing compensation and
resume sidecars when combined with PoLoRA.

## Arguments

Replace AdamW8bit in the normal FLUX.2 training command with:

```powershell
--optimizer_type polora `
--optimizer_args "pair_batch_size=4" `
--learning_rate 2e-4 `
--network_module networks.lora_flux_2 `
--network_dim 64 --network_alpha 64 `
--max_data_loader_n_workers 0 `
--save_state
```

Remove `--network_args loraplus_lr_ratio=...`. PoLoRA uses a single shared
learning rate; LoRA+ groups are unsupported and rejected. The initial
integration requires unsplit Linear factors, alpha equal to rank, no dropout,
no additional trainable parameters, one training process, and zero dataloader
workers. Other `network_args` are also rejected rather than assumed compatible.

Reference optimizer defaults are `beta1=0.9`, `epsilon=1e-12`, `delta=1e-4`,
`curvature_beta=0.99`, `ns_steps=8`, `higham_iters=8`, `compile=False`.
These can be set through `--optimizer_args key=value`. `compile=True` compiles
the reference spectral kernels and requires a working PyTorch compiler. The
model's ordinary `--compile` option remains separate.

The reference learning rate is 2e-4. It controls spectral update magnitude and
is not interchangeable with an AdamW learning rate. It is a starting point for
an experiment, not a validated optimum for Klein. Compare multiple learning
rates with the same data, seed, effective batch size, training budget and
held-out validation. Compare wall-clock time as well as steps and quality;
there is no established Klein speed or quality improvement.

## Local PowerShell wrapper

The local `D:\repos\train.ps1` accepts:

```powershell
# Explicitly selects PoLoRA, disables the manually configured LoRA+ option,
# enables resumable state and appends _polora to the output name.
& D:\repos\train.ps1 -PoLoRA

# Limit optimizer workspace further if dedicated VRAM is tight.
& D:\repos\train.ps1 -PoLoRA -PoLoRAPairBatchSize 1 -PoLoRALearningRate 2e-4

# Resume an existing PoLoRA state with the same configuration and training budget.
& D:\repos\train.ps1 -PoLoRA -ResumeState 'D:\repos\musubi-tuner\output_768\change_clothes_768_polora-step00000100-state'
```

The wrapper lives outside this repository. Its normal invocation without
`-PoLoRA` keeps the user's existing optimizer selection. It currently leaves
LoRA-GA disabled, as configured before this integration. To combine PoLoRA
and GA in a direct training command, add the existing `--loraga` and cache
arguments documented in [flux_2_loraga.md](flux_2_loraga.md).

At startup, verify the `PoLoRA active: ... Linear pairs` log. A standard
`.safetensors` adapter is sufficient for inference; use the `-state` directory
for training resumption. An AdamW/LoRA+ optimizer state cannot be resumed as
PoLoRA. Start a new experiment or explicitly load compatible adapter weights
as a warm start, without claiming optimizer continuity.

## Memory and validation

BF16 autocast, native FP8-scaled frozen weights, gradient checkpointing,
gradient accumulation and forward/backward or H2D-only block swapping use the
existing Musubi paths. PoLoRA stores FP32 optimizer states: its state may be
smaller than full FP32 Adam, but can be larger than AdamW8bit. Pair batching
limits temporary matrices, not persistent state. Reduce the pair batch size
if necessary; do not assume PoLoRA saves VRAM compared with AdamW8bit.

For the local Klein rank-64 cache (80 target modules), the factor shapes contain
92,405,760 parameters. The persistent FP32 PoLoRA moments alone require
352.5 MiB, plus roughly 5.6 MiB for diagonals and spectral vectors. This is a
shape-based state estimate, not measured full-training peak VRAM. It excludes
adapter weights, gradients, temporary optimizer matrices, backbone and
activations. A full 4B PoLoRA run has not been executed during this integration.

Focused tests cover reference updates from both ordinary zero-B and nonzero-B
initializations, full optimizer state, scheduler LR restoration, invalid
resumes, native pairing, gradient accumulation, CPU-to-CUDA state migration,
FP8/checkpointing/block swapping, steady-state CUDA allocations, and complete
training/resume with and without LoRA-GA. The GPU integration tests use the
real FLUX.2 pipeline with reduced geometry, not the full 4B model. They do not
establish full-model VRAM requirements, convergence or image quality.

Validation on Windows / PyTorch 2.6.0+cu124 / RTX 3070 Ti: 102 tests passed in
the combined PoLoRA, LoRA-GA and validation suite. After adding stricter checks
for missing spectral state and direct TensorBoard assertions, all 18 PoLoRA
tests passed again. Ruff and PowerShell argument-construction checks passed.
The suite reported existing PyTorch deprecation warnings for the `device`
argument in pinned-memory operations. No full-size 4B training or quality
comparison was run.

```powershell
$env:USE_TF = '0'
.\.venv\Scripts\python.exe -m pytest tests/test_polora.py -q
```

Compare exported adapters using the existing inference pipeline, and evaluate
held-out images and loss before selecting PoLoRA over the conventional setup.
