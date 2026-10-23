"""Training cursor persistence for ordinary PoLoRA (LoRA-GA owns its cursor)."""

import hashlib
import json
from pathlib import Path


def register_resume(trainer, args, accelerator):
    if accelerator.num_processes != 1:
        raise ValueError("PoLoRA currently supports single-process training only")
    if getattr(args, "loraga", False):
        return
    from musubi_tuner.training.flux2_loraga import dataset_state, restore_dataset

    keys = ("seed", "gradient_accumulation_steps", "optimizer_type", "optimizer_args", "learning_rate",
            "lr_scheduler", "lr_scheduler_args", "lr_warmup_steps", "max_train_steps", "max_grad_norm",
            "network_dim", "network_alpha", "mixed_precision", "fp8_base", "fp8_scaled",
            "gradient_checkpointing", "blocks_to_swap", "block_swap_h2d_only", "block_swap_ring_size",
            "timestep_sampling", "weighting_scheme", "discrete_flow_shift")
    signature = {key: getattr(args, key, None) for key in keys}
    checkpoint = Path(args.dit).resolve()
    signature["checkpoint"] = [str(checkpoint), checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns]
    signature["dataset_config"] = hashlib.sha256(Path(args.dataset_config).read_bytes()).hexdigest()

    def save(models, weights, directory):
        state = dict(trainer._loraga_progress, signature=signature,
                     timestep_pool=trainer.timestep_range_pool,
                     datasets=dataset_state(trainer._loraga_training_group))
        (Path(directory) / "polora_progress.json").write_text(json.dumps(state), encoding="utf-8")

    def load(models, directory):
        path = Path(directory) / "polora_progress.json"
        if not path.is_file():
            raise ValueError("PoLoRA resume requires polora_progress.json from --save_state")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("PoLoRA resume model/dataset/training configuration mismatch")
        restore_dataset(trainer._loraga_training_group, state["datasets"])
        trainer.timestep_range_pool = [tuple(value) for value in state["timestep_pool"]]
        trainer._loraga_progress = {key: state[key] for key in ("global_step", "epoch", "next_batch")}

    accelerator.register_save_state_pre_hook(save)
    accelerator.register_load_state_pre_hook(load)
