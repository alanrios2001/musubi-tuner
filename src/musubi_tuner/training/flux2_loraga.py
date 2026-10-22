"""Two-stage LoRA-GA initialization for FLUX.2 Klein, using native training loss."""

import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import tempfile
import time

import torch
from accelerate.utils import send_to_device
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.networks import lora_flux_2
from musubi_tuner.networks.loraga import initialize_factors
from musubi_tuner.training.flux2_validation import isolated_rng, seed_rng
from musubi_tuner.training.loraga_streaming import DiskGradients, streamed_linears

logger = logging.getLogger(__name__)
CACHE_NAME = "loraga_init.safetensors"


def add_arguments(parser):
    parser.add_argument("--loraga", action="store_true", help="BF16 gradient initialization before native LoRA training")
    parser.add_argument("--loraga_estimation_batches", type=int, default=64)
    parser.add_argument("--loraga_seed", type=int, default=42)
    parser.add_argument("--loraga_direction", choices=["ArB2r", "A2rBr", "ArBr"], default="ArB2r")
    parser.add_argument("--loraga_scale", choices=["stable", "unit", "gd"], default="stable")
    parser.add_argument("--loraga_stable_gamma", type=float, default=16.0)
    parser.add_argument("--loraga_init_cache", type=str, help="Initialization safetensors path; validate and reuse if present")
    parser.add_argument("--loraga_work_dir", type=str, help="Temporary disk gradient storage (default: system temp)")
    parser.add_argument("--loraga_max_quantization_error_ratio", type=float, default=2.0,
                        help="Fail if aggregate compensated FP8 weight RMSE exceeds this multiple of ordinary FP8 RMSE")


def validate_args(args):
    resume = Path(args.resume) if args.resume else None
    if not args.loraga:
        if resume and (resume / CACHE_NAME).exists():
            raise ValueError("This state requires --loraga to reconstruct the compensated backbone")
        return
    if args.model_version != "klein-base-4b" or args.mixed_precision != "bf16":
        raise ValueError("LoRA-GA currently requires --model_version klein-base-4b --mixed_precision bf16")
    if args.network_module not in ("networks.lora_flux_2", "musubi_tuner.networks.lora_flux_2"):
        raise ValueError("LoRA-GA supports only networks.lora_flux_2 Linear adapters")
    if args.network_args or args.network_dropout or args.base_weights or args.network_weights or args.dim_from_weights:
        raise ValueError("LoRA-GA does not support network_args/dropout/base_weights/network_weights/dim_from_weights; resume with --resume")
    if args.fp8_base != args.fp8_scaled:
        raise ValueError("LoRA-GA FP8 requires both --fp8_base and --fp8_scaled (or neither for BF16)")
    if args.loraga_estimation_batches <= 0 or args.network_dim is None or args.network_dim <= 0:
        raise ValueError("LoRA-GA estimation_batches and network_dim must be positive")
    if not math.isfinite(args.loraga_max_quantization_error_ratio) or args.loraga_max_quantization_error_ratio <= 0:
        raise ValueError("LoRA-GA quantization error ratio must be finite and positive")
    if not math.isfinite(args.loraga_stable_gamma) or args.loraga_stable_gamma <= 0:
        raise ValueError("LoRA-GA stable_gamma must be finite and positive")
    if args.resume_from_huggingface:
        raise ValueError("LoRA-GA resume requires a local Accelerate state directory")
    if args.max_data_loader_n_workers != 0 or args.persistent_data_loader_workers:
        raise ValueError("LoRA-GA deterministic resume currently requires --max_data_loader_n_workers 0 without persistent workers")
    if resume and not (resume / CACHE_NAME).is_file():
        raise ValueError(f"Missing LoRA-GA initialization state: {resume / CACHE_NAME}")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(args, group, targets):
    from musubi_tuner.utils.safetensors_utils import get_split_weight_filenames

    caches = set()
    for dataset in group.datasets:
        for items in dataset.batch_manager.buckets.values():
            for item in items:
                for path in (item.latent_cache_path, item.text_encoder_output_cache_path):
                    if path is None:
                        raise ValueError("LoRA-GA requires cached latent and text conditioning files")
                    caches.add(str(Path(path).resolve()))
    objective_keys = [key for key in vars(args) if any(
        token in key for token in ("timestep", "weighting", "flow_shift", "logit", "mode_scale", "sigmoid", "logsnr")
    )]
    return {
        "version": 1,
        "checkpoint": [[str(Path(p).resolve()), file_hash(p)] for p in (get_split_weight_filenames(args.dit) or [args.dit])],
        "dataset_config": file_hash(args.dataset_config),
        "caches": [[p, file_hash(p)] for p in sorted(caches)],
        "model": args.model_version,
        "rank": args.network_dim,
        "alpha": float(args.network_alpha or args.network_dim),
        "direction": args.loraga_direction, "scale": args.loraga_scale, "gamma": args.loraga_stable_gamma,
        "seed": args.loraga_seed, "batches": args.loraga_estimation_batches,
        "objective": {key: getattr(args, key) for key in objective_keys},
        "fp8_base": args.fp8_base, "fp8_scaled": args.fp8_scaled,
        "targets": {name: list(module.weight.shape) for name, module in targets.items()},
    }


def save_cache(path, initial, metadata, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, (a, b) in initial.items():
        tensors[f"{name}.a"] = a.cpu().contiguous()
        tensors[f"{name}.b"] = b.cpu().contiguous()
    # A completed file is published atomically; failed estimation never leaves a valid cache.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        save_file(tensors, str(temporary), metadata={"identity": json.dumps(metadata, sort_keys=True), "metrics": json.dumps(metrics)})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cache(path, expected):
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
    if json.loads(metadata.get("identity", "null")) != expected:
        raise ValueError(f"Incompatible LoRA-GA initialization cache: {path}")
    state = load_file(str(path))
    initial = {}
    for name, (out_dim, in_dim) in expected["targets"].items():
        a, b = state.pop(f"{name}.a"), state.pop(f"{name}.b")
        if a.shape != (expected["rank"], in_dim) or b.shape != (out_dim, expected["rank"]):
            raise ValueError(f"Invalid LoRA-GA factor shape: {name}")
        if a.dtype != torch.float32 or b.dtype != torch.float32 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Invalid LoRA-GA factor values: {name}")
        initial[name] = a, b
    if state:
        raise ValueError("Unexpected tensors in LoRA-GA initialization cache")
    return initial, json.loads(metadata["metrics"])


def estimate(trainer, args, accelerator, model, network, targets, group, collator):
    from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

    # Match the training loader's native batches, including repeats and paired controls.
    loader = torch.utils.data.DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0,
                                        generator=torch.Generator().manual_seed(args.loraga_seed))
    scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
    started = time.perf_counter()
    pool = trainer.timestep_range_pool
    initial = {}
    disk_bytes = sum(module.weight.numel() * 4 for module in targets.values())
    work_dir = args.loraga_work_dir
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_dir or tempfile.gettempdir()).free < disk_bytes * 1.1:
        raise ValueError(f"LoRA-GA needs at least {disk_bytes} bytes of free gradient storage plus temporary matrix space")
    logger.info("LoRA-GA: %s target gradient bytes on disk; CPU BF16 backbone, FP32 per-matrix SVD", disk_bytes)
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    try:
        with isolated_rng(), tempfile.TemporaryDirectory(dir=work_dir, prefix="musubi-loraga-") as directory:
            seed_rng(args.loraga_seed)
            trainer.timestep_range_pool = []
            sink = DiskGradients(directory)
            model.train()
            # Always checkpoint Phase A, independently of Phase B's setting.
            model.enable_gradient_checkpointing(False)
            with streamed_linears(model, targets, sink, accelerator.device):
                iterator = iter(loader)
                for index in range(args.loraga_estimation_batches):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)
                    batch = send_to_device(batch, accelerator.device)
                    latents = trainer.scale_shift_latents(batch["latents"])
                    noise = torch.randn_like(latents)
                    with accelerator.autocast():
                        loss, _ = trainer.process_batch(args, accelerator, model, network, batch, latents, noise,
                                                       scheduler, torch.bfloat16, torch.float32, None, 0)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite LoRA-GA estimation loss")
                    loss.backward()
                    del loss, batch, latents, noise
                    logger.info("LoRA-GA gradient batch %s/%s", index + 1, args.loraga_estimation_batches)
            for lora in network.unet_loras:
                gradient = sink.take(lora.lora_name, args.loraga_estimation_batches)
                initial[lora.lora_name] = initialize_factors(gradient, lora.lora_dim, lora.scale,
                                                           args.loraga_direction, args.loraga_scale, args.loraga_stable_gamma)
                del gradient
    except (torch.OutOfMemoryError, MemoryError) as error:
        raise RuntimeError("LoRA-GA BF16 initialization exhausted memory. Requires CPU backbone + offloaded activations, "
                           "one FP32 gradient/SVD workspace in RAM and one Linear/attention operation in VRAM. "
                           "Reduce cached resolution/batch size; no alternate initialization was used.") from error
    finally:
        trainer.timestep_range_pool = pool
        model.disable_gradient_checkpointing()
        model.eval()
        gc.collect()
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = {"loraga/initialization_seconds": time.perf_counter() - started,
               "loraga/estimated_batches": args.loraga_estimation_batches}
    if accelerator.device.type == "cuda":
        metrics["loraga/initialization_peak_vram_bytes"] = torch.cuda.max_memory_allocated(accelerator.device)
    return initial, metrics


def compensate(model, initial, targets, scaling, fp8, max_ratio):
    """Compensate one CPU matrix at a time; compare native FP8 rounding errors."""
    sums = [0.0, 0.0, 0.0]
    count = 0

    for name, module in targets.items():
        a, b = initial[name]
        original = module.weight.detach().float()
        offset = (b @ a) * scaling
        residual = (original - offset).to(torch.bfloat16)
        sums[0] += (residual.float() + offset - original).square().sum().item()
        if fp8:
            sums[1] += (quantized(module.weight.detach()).float() - original).square().sum().item()
            sums[2] += (quantized(residual).float() + offset - original).square().sum().item()
        count += original.numel()
        module.weight.data = residual
    metrics = {"loraga/compensation_weight_rmse": math.sqrt(sums[0] / count)}
    if fp8:
        ratio = math.sqrt(sums[2] / max(sums[1], 1e-30))
        metrics.update({"loraga/ordinary_fp8_weight_rmse": math.sqrt(sums[1] / count),
                        "loraga/compensated_fp8_weight_rmse": math.sqrt(sums[2] / count),
                        "loraga/quantization_error_ratio": ratio})
        if ratio > max_ratio:
            raise ValueError(f"LoRA-GA FP8 weight RMSE ratio {ratio:.6g} exceeds configured limit {max_ratio}; "
                             "initialization cache remains reusable. Inspect scaling before explicitly changing the limit.")
    logger.info("LoRA-GA initialization metrics: %s", metrics)
    return metrics


def quantize_model(model):
    from musubi_tuner.modules.fp8_optimization_utils import optimize_state_dict_with_fp8, apply_fp8_monkey_patch
    from musubi_tuner.flux_2.flux2_models import FP8_OPTIMIZATION_TARGET_KEYS, FP8_OPTIMIZATION_EXCLUDE_KEYS

    state = model.state_dict()
    optimize_state_dict_with_fp8(state, "cpu", FP8_OPTIMIZATION_TARGET_KEYS, FP8_OPTIMIZATION_EXCLUDE_KEYS)
    apply_fp8_monkey_patch(model, state, use_scaled_mm=False)
    model.load_state_dict(state, strict=True, assign=True)


def quantized(weight):
    """Native quantizer and BF16 dequantization, as in fp8_linear_forward_patch."""
    from musubi_tuner.modules.fp8_optimization_utils import quantize_weight

    q, scale = quantize_weight("loraga.weight", weight.detach(), torch.float8_e4m3fn, 448.0, -448.0)
    scale = scale.to(weight.dtype)
    if scale.ndim == 3:
        return (q.to(weight.dtype).reshape(scale.shape[0], scale.shape[1], -1) * scale).reshape(weight.shape)
    return q.to(weight.dtype) * scale


def output_probe(trainer, args, accelerator, model, inputs, targets, initial=None, fp8=False):
    """One deterministic cached training batch, using actual target/control conditioning."""
    from musubi_tuner.flux_2.flux2_models import FP8_OPTIMIZATION_TARGET_KEYS, FP8_OPTIMIZATION_EXCLUDE_KEYS

    def transform(path, weight):
        if any(key in path for key in FP8_OPTIMIZATION_TARGET_KEYS) and not any(
            key in path for key in FP8_OPTIMIZATION_EXCLUDE_KEYS
        ):
            return quantized(weight)
        return weight

    additions = None
    if initial:
        scaling = float(args.network_alpha or args.network_dim) / args.network_dim
        additions = {id(targets[name]): (a, b, scaling) for name, (a, b) in initial.items()}
    batch, latents, noise, noisy, timesteps = send_to_device(inputs, accelerator.device)
    model.eval()
    with torch.no_grad(), streamed_linears(model, {}, None, accelerator.device,
                                           weight_transform=transform if fp8 else None, additions=additions), accelerator.autocast():
        output = trainer.call_dit(args, accelerator, model, latents, batch, noise, noisy, timesteps, torch.float32).pred
    return output.detach().float().cpu()


def initialize(trainer, args, accelerator, model):
    if accelerator.num_processes != 1:
        raise ValueError("LoRA-GA initialization currently supports one process only")
    network = lora_flux_2.create_arch_network(1.0, args.network_dim, args.network_alpha, None, None, model)
    targets = {lora.lora_name: lora.org_module for lora in network.unet_loras}
    if not targets:
        raise ValueError("No LoRA-GA targets found")
    for lora in network.unet_loras:
        if type(lora.org_module) is not torch.nn.Linear or lora.split_dims is not None:
            raise ValueError(f"LoRA-GA only supports unsplit Linear adapters: {lora.lora_name}")
        if 2 * lora.lora_dim > min(lora.org_module.weight.shape):
            raise ValueError(f"LoRA-GA rank too large for {lora.lora_name}")
    group, collator = trainer._loraga_dataset
    metadata = identity(args, group, targets)
    # Same noise/timestep/input for all four representations; separate from estimation RNG.
    from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

    with isolated_rng():
        seed_rng(args.loraga_seed)
        pool = trainer.timestep_range_pool
        trainer.timestep_range_pool = []
        try:
            batch = group[0]
            latents = trainer.scale_shift_latents(batch["latents"])
            noise = torch.randn_like(latents)
            scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
            noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
                args, noise, latents, batch["timesteps"], scheduler, torch.device("cpu"), torch.bfloat16)
            inputs = (batch, latents, noise, noisy, timesteps)
        finally:
            trainer.timestep_range_pool = pool
    baseline = output_probe(trainer, args, accelerator, model, inputs, targets)
    ordinary = output_probe(trainer, args, accelerator, model, inputs, targets, fp8=True) if args.fp8_scaled else None
    source = Path(args.resume) / CACHE_NAME if args.resume else Path(
        args.loraga_init_cache or str(Path(args.output_dir) / CACHE_NAME))
    if source.exists():
        initial, metrics = load_cache(source, metadata)
        logger.info("Reusing LoRA-GA initialization: %s", source)
    else:
        initial, metrics = estimate(trainer, args, accelerator, model, network, targets, group, collator)
        save_cache(source, initial, metadata, metrics)
    scaling = float(args.network_alpha or args.network_dim) / args.network_dim
    # Evaluate compensated BF16 before applying persistent native FP8 quantization.
    metrics.update(compensate(model, initial, targets, scaling, args.fp8_scaled, args.loraga_max_quantization_error_ratio))
    restored = output_probe(trainer, args, accelerator, model, inputs, targets, initial)
    metrics["loraga/initial_output_error"] = (restored - baseline).square().mean().sqrt().item()
    metrics["loraga/initial_output_relative_error"] = metrics["loraga/initial_output_error"] / max(
        baseline.square().mean().sqrt().item(), 1e-12)
    if args.fp8_scaled:
        fp8_output = output_probe(trainer, args, accelerator, model, inputs, targets, initial, fp8=True)
        error = (fp8_output - baseline).square().mean().sqrt().item()
        ordinary_error = (ordinary - baseline).square().mean().sqrt().item()
        metrics.update({"loraga/quantization_error": error, "loraga/ordinary_fp8_output_error": ordinary_error,
                        "loraga/output_error_ratio": error / max(ordinary_error, 1e-12)})
        if metrics["loraga/output_error_ratio"] > args.loraga_max_quantization_error_ratio:
            raise ValueError(f"LoRA-GA output error ratio {metrics['loraga/output_error_ratio']:.6g} exceeds configured limit "
                             f"{args.loraga_max_quantization_error_ratio}; inspect initialization metrics before changing the limit")
        quantize_model(model)
    trainer._loraga_initial, trainer._loraga_metadata, trainer._loraga_metrics = initial, metadata, metrics
    del trainer._loraga_dataset
    model.requires_grad_(False)
    return model


def dataset_state(group):
    """Persist shuffled descriptors, not cached tensors or dataset Python objects."""
    return [{"epoch": dataset.current_epoch,
             "buckets": [[list(key), [item.item_key for item in items]]
                         for key, items in dataset.batch_manager.buckets.items()],
             "indices": dataset.batch_manager.bucket_batch_indices,
             "timesteps": dataset.batch_manager.timestep_pool}
            for dataset in group.datasets]


def restore_dataset(group, states):
    if len(group.datasets) != len(states):
        raise ValueError("LoRA-GA resume dataset count mismatch")
    for dataset, state in zip(group.datasets, states):
        manager = dataset.batch_manager
        for key, names in state["buckets"]:
            key = tuple(key)
            items = {item.item_key: item for item in manager.buckets[key]}
            if set(items) != set(names) or len(manager.buckets[key]) != len(names):
                raise ValueError("LoRA-GA resume dataset items mismatch")
            manager.buckets[key] = [items[name] for name in names]
        manager.bucket_batch_indices = [(tuple(key), index) for key, index in state["indices"]]
        manager.timestep_pool = state["timesteps"]
        dataset.current_epoch = state["epoch"]


def register_resume(trainer, args, accelerator, network):
    """Accelerate keeps raw rank-r factors/optimizer/RNG; sidecar reconstructs base."""
    signature_keys = ("seed", "gradient_accumulation_steps", "optimizer_type", "optimizer_args", "learning_rate",
                      "lr_scheduler", "lr_scheduler_args", "lr_warmup_steps", "max_train_steps")
    signature = {key: getattr(args, key, None) for key in signature_keys}

    def save(models, weights, directory):
        save_cache(Path(directory) / CACHE_NAME, trainer._loraga_initial, trainer._loraga_metadata,
                   trainer._loraga_saved_metrics)
        state = dict(trainer._loraga_progress)
        state.update({"signature": signature, "timestep_pool": trainer.timestep_range_pool,
                      "datasets": dataset_state(trainer._loraga_training_group)})
        (Path(directory) / "loraga_progress.json").write_text(json.dumps(state), encoding="utf-8")

    def load(models, directory):
        # Validate before Accelerate mutates the raw network or optimizer.
        initial, _ = load_cache(Path(directory) / CACHE_NAME, trainer._loraga_metadata)
        if any(not torch.equal(tensor, other) for name, pair in initial.items()
               for tensor, other in zip(pair, trainer._loraga_initial[name])):
            raise ValueError("LoRA-GA resume initial factors differ from reconstructed backbone")
        state = json.loads((Path(directory) / "loraga_progress.json").read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("LoRA-GA resume optimizer/scheduler/batch configuration mismatch")
        restore_dataset(trainer._loraga_training_group, state["datasets"])
        trainer.timestep_range_pool = [tuple(value) for value in state["timestep_pool"]]
        trainer._loraga_progress = {key: state[key] for key in ("global_step", "epoch", "next_batch")}

    accelerator.register_save_state_pre_hook(save)
    accelerator.register_load_state_pre_hook(load)
