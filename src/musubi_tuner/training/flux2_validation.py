"""Disk-cached FLUX.2 validation using the trainer's unchanged flow-matching objective."""

from contextlib import contextmanager
import hashlib
from itertools import islice
from multiprocessing import Value
from pathlib import Path
import random
import time

import numpy as np
import torch
from accelerate.utils import send_to_device


def validate_validation_args(args):
    enabled = getattr(args, "validation_dataset_config", None)
    interval = getattr(args, "validation_every_n_steps", None)
    limit = getattr(args, "validation_max_batches", None)
    if enabled and (interval is None or interval <= 0):
        raise ValueError("--validation_dataset_config requires positive --validation_every_n_steps")
    if limit is not None and limit <= 0:
        raise ValueError("--validation_max_batches must be positive")
    if not enabled and (interval is not None or limit is not None):
        raise ValueError("Validation interval/limit requires --validation_dataset_config")


@contextmanager
def isolated_rng():
    """Restore Python, NumPy, CPU and all initialized CUDA RNG states, also on failure."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def example_seed(seed, identity):
    return int.from_bytes(hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).digest()[:8], "little") % (2**63)


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


class CachedValidationDataset(torch.utils.data.Dataset):
    """Reuse BucketBatchManager's native safetensors loader without epoch shuffling.

    Evaluation deliberately uses singleton batches, independent of the configured training
    batch size. This bounds VRAM and gives each identity the same random draw in any order.
    Only descriptors are retained; tensors are loaded from disk on demand.
    """

    def __init__(self, group):
        from musubi_tuner.dataset.bucket import BucketBatchManager

        self.batches = []
        identities = set()
        for dataset in group.datasets:
            buckets = {}
            for resolution, items in dataset.batch_manager.buckets.items():
                # Training repeats have no meaning for held-out metrics.
                unique = {item.item_key: item for item in items}
                buckets[resolution] = sorted(unique.values(), key=lambda item: item.item_key)
            manager = BucketBatchManager(buckets, batch_size=1)
            for index, (resolution, item_index) in enumerate(manager.bucket_batch_indices):
                item = buckets[resolution][item_index]
                if item.item_key in identities:
                    raise ValueError(f"Duplicate validation identity: {item.item_key}")
                identities.add(item.item_key)
                self.batches.append((item.item_key, manager, index))
        self.batches.sort(key=lambda entry: entry[0])
        if not self.batches:
            raise ValueError("No validation cache items found; generate latent and text encoder caches first")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        identity, manager, batch_index = self.batches[index]
        batch = manager[batch_index]
        if "latents" not in batch or "ctx_vec" not in batch or "latents_control_0" not in batch:
            raise ValueError(f"Validation cache for {identity} requires target, control latents and ctx_vec; re-cache it")
        batch["validation_identity"] = identity
        return batch


def build_validation_loader(args, architecture, training_group):
    from musubi_tuner.dataset import config_utils

    with isolated_rng():
        # Blueprint construction only consumes Python RNG; do not queue CUDA seeds before initialization.
        random.seed(args.validation_seed)
        blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            config_utils.load_user_config(args.validation_dataset_config), args, architecture=architecture
        )
        training_caches = {Path(dataset.cache_directory).resolve() for dataset in training_group.datasets}
        training_images = {
            Path(dataset.image_directory).resolve()
            for dataset in training_group.datasets
            if getattr(dataset, "image_directory", None)
        }
        for dataset in blueprint.dataset_group.datasets:
            if not dataset.is_image_dataset:
                raise ValueError("FLUX.2 validation requires an image dataset")
            params = dataset.params
            cache = Path(params.cache_directory or params.image_directory).resolve()
            if cache in training_caches:
                raise ValueError("Training and validation cache directories must be independent")
            if params.image_directory and Path(params.image_directory).resolve() in training_images:
                raise ValueError("Training and validation image directories must be independent")
            params.num_repeats = 1
            params.batch_size = 1
        group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True, shared_epoch=Value("i", 0))
        for dataset in group.datasets:
            cache_files = list(Path(dataset.cache_directory).glob(f"*_{architecture}.safetensors"))
            if len(cache_files) != dataset.num_train_items:
                raise ValueError("Incomplete validation text encoder caches; re-cache before training")
        dataset = CachedValidationDataset(group)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.validation_seed),
        )


@contextmanager
def validation_execution(trainer, transformer, network):
    """Use existing offloader transitions; restore heterogeneous modes and timestep pool."""
    modules = {module: module.training for root in (transformer, network) for module in root.modules()}
    pool = trainer.timestep_range_pool
    offloaders = [
        getattr(transformer, name)
        for name in ("offloader_double", "offloader_single")
        if getattr(transformer, name, None) is not None
    ]
    swap_modes = [(offloader, offloader.forward_only) for offloader in offloaders]
    try:
        transformer.eval()
        network.eval()
        # eval bypasses checkpointing in FLUX.2 blocks; leave checkpoint/offload flags intact.
        if getattr(transformer, "blocks_to_swap", 0):
            transformer.switch_block_swap_for_inference()
        with torch.no_grad():
            yield
    finally:
        trainer.timestep_range_pool = pool
        for module, mode in modules.items():
            module.training = mode
        for offloader, forward_only in swap_modes:
            offloader.set_forward_only(forward_only)
        if getattr(transformer, "blocks_to_swap", 0):
            transformer.prepare_block_swap_before_forward()


def evaluate_validation(
    trainer, args, accelerator, network, transformer, loader, noise_scheduler, dit_dtype, network_dtype, global_step
):
    """No optimizer, scheduler, accumulation or training-loader access is permitted here."""
    started = time.perf_counter()
    model = accelerator.unwrap_model(transformer)
    unwrapped_network = accelerator.unwrap_model(network)
    total_loss = 0.0
    num_samples = 0
    diagnostics = getattr(trainer, "memory_diagnostics", None)
    memory_metrics = {}
    if diagnostics is not None:
        memory_metrics.update(diagnostics.snapshot("memory/before_validation"))
        diagnostics.begin_validation()
    with isolated_rng(), validation_execution(trainer, model, unwrapped_network):
        batches = loader if args.validation_max_batches is None else islice(loader, args.validation_max_batches)
        for batch in batches:
            seed_rng(example_seed(args.validation_seed, batch.pop("validation_identity")))
            # A private timestep pool prevents consumption of training's stratified sampler state.
            trainer.timestep_range_pool = []
            batch = send_to_device(batch, accelerator.device)
            latents = trainer.scale_shift_latents(batch["latents"])
            noise = torch.randn_like(latents)
            with accelerator.autocast():
                loss, _ = trainer.process_batch(
                    args,
                    accelerator,
                    transformer,
                    network,
                    batch,
                    latents,
                    noise,
                    noise_scheduler,
                    dit_dtype,
                    network_dtype,
                    None,
                    global_step,
                )
            count = latents.shape[0]
            total_loss += loss.detach().float().item() * count
            num_samples += count
            if diagnostics is not None:
                for key, value in diagnostics.snapshot("memory/validation_sampled_peak").items():
                    memory_metrics[key] = max(memory_metrics.get(key, value), value)
            del loss, noise, latents, batch
        if diagnostics is not None:
            memory_metrics.update(diagnostics.validation_peaks())
            memory_metrics.update(diagnostics.snapshot("memory/validation_end"))
    if diagnostics is not None:
        memory_metrics.update(diagnostics.snapshot("memory/after_validation"))
    if not num_samples:
        raise ValueError("Validation evaluated no samples")
    return {
        "val/loss": total_loss / num_samples,
        "val/num_samples": num_samples,
        "val/evaluation_seconds": time.perf_counter() - started,
        **memory_metrics,
    }
