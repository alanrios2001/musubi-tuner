"""Native cache, objective, execution-state, and periodic logging regression tests."""

import copy
from contextlib import nullcontext
import random
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import save_file
import torch
from torch.utils._triton import has_triton

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.training.trainer_base import DiTOutput
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.flux2_validation import (
    build_validation_loader,
    evaluate_validation,
    example_seed,
    isolated_rng,
    validate_validation_args,
)


class AcceleratorStub:
    device = torch.device("cpu")
    trackers = [object()]

    def __init__(self):
        self.logs = []

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def log(self, metrics, step):
        self.logs.append((metrics, step))


def args_for_validation(**kwargs):
    args = flux2_setup_parser(setup_parser_common()).parse_args([])
    args.validation_dataset_config = "val.toml"
    args.validation_every_n_steps = 2
    args.timestep_sampling = "flux2_shift"
    args.weighting_scheme = "none"
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def cache_pair(directory, identity, value=1.0, control=True, text=True):
    directory.mkdir(exist_ok=True)
    data = {"latents_2x2_float32": torch.full((4, 2, 2), value)}
    if control:
        data["latents_control_0_2x2_float32"] = torch.full((4, 2, 2), value + 10)
    save_file(data, str(directory / f"{identity}_0032x0032_f2k4b.safetensors"))
    if text:
        save_file({"ctx_vec_float32": torch.full((3, 8), value + 20)}, str(directory / f"{identity}_f2k4b_te.safetensors"))


def cache_config(tmp_path, count=3):
    cache = tmp_path / "cache_val"
    for index in reversed(range(count)):
        cache_pair(cache, f"pair_{index}", float(index))
    image_dir = tmp_path / "target"
    image_dir.mkdir(exist_ok=True)
    config = tmp_path / "val.toml"
    config.write_text(
        "[general]\nresolution = [32, 32]\nenable_bucket = true\nbucket_no_upscale = true\n"
        "batch_size = 2\n[[datasets]]\n"
        f'image_directory = "{image_dir.as_posix()}"\ncache_directory = "{cache.as_posix()}"\nnum_repeats = 2\n',
        encoding="utf-8",
    )
    return config, cache


def training_group(tmp_path):
    return SimpleNamespace(datasets=[SimpleNamespace(cache_directory=tmp_path / "cache_train", image_directory=tmp_path / "train")])


def test_native_cache_loader_keeps_control_target_text_and_unique_examples(tmp_path):
    config, _ = cache_config(tmp_path)
    args = args_for_validation(validation_dataset_config=str(config))
    loader = build_validation_loader(args, "f2k4b", training_group(tmp_path))
    assert len(loader) == 3  # no training repeats; singleton evaluation
    batches = list(loader)
    assert [batch["validation_identity"] for batch in batches] == [f"pair_{index}" for index in range(3)]
    for index, batch in enumerate(batches):
        assert batch["latents"].shape == (1, 4, 2, 2)
        assert torch.all(batch["latents"] == index)
        assert torch.all(batch["latents_control_0"] == index + 10)
        assert torch.all(batch["ctx_vec"] == index + 20)
        assert batch["timesteps"] is None


def test_cache_build_preserves_rng_and_rejects_shared_cache(tmp_path):
    config, cache = cache_config(tmp_path)
    args = args_for_validation(validation_dataset_config=str(config))
    before = rng_snapshot()
    build_validation_loader(args, "f2k4b", training_group(tmp_path))
    assert_rng_equal(before, rng_snapshot())
    group = training_group(tmp_path)
    group.datasets[0].cache_directory = cache
    with pytest.raises(ValueError, match="cache directories"):
        build_validation_loader(args, "f2k4b", group)


def test_missing_text_or_control_cache_fails(tmp_path):
    config, cache = cache_config(tmp_path)
    args = args_for_validation(validation_dataset_config=str(config))
    (cache / "pair_0_f2k4b_te.safetensors").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        build_validation_loader(args, "f2k4b", training_group(tmp_path))
    cache_pair(cache, "pair_0", control=False)
    loader = build_validation_loader(args, "f2k4b", training_group(tmp_path))
    with pytest.raises(ValueError, match="requires target, control"):
        next(iter(loader))


class ObjectiveTrainer(Flux2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.num_timestep_buckets = 4
        self.draws = {}

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype):
        identity = batch["test_identity"]
        self.draws[identity] = (noise.clone(), timesteps.clone())
        assert not torch.is_grad_enabled()
        assert not transformer.training
        return DiTOutput(pred=transformer(noisy_model_input), target=noise - latents)


def toy_batches():
    return [
        dict(
            latents=torch.full((1, 1, 2, 2), float(index)),
            timesteps=None,
            validation_identity=f"pair_{index}",
            test_identity=f"pair_{index}",
        )
        for index in range(3)
    ]


def rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
    )


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])
    assert len(a[3]) == len(b[3])
    assert all(torch.equal(x, y) for x, y in zip(a[3], b[3]))


def test_deterministic_objective_order_rng_parameters_optimizer_scheduler_and_modes():
    trainer = ObjectiveTrainer()
    args = args_for_validation()
    model = torch.nn.Linear(2, 2)
    network = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(2, 2))
    network[0].eval()  # preserve heterogeneous modes, not just root.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    weights = [parameter.clone() for parameter in model.parameters()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    modes = [module.training for root in (model, network) for module in root.modules()]
    pool = trainer.timestep_range_pool = [(0.2, 0.4)]
    before = rng_snapshot()
    accelerator = AcceleratorStub()
    first = evaluate_validation(trainer, args, accelerator, network, model, toy_batches(), None, torch.float32, torch.float32, 2)
    draws = trainer.draws.copy()
    second = evaluate_validation(
        trainer, args, accelerator, network, model, list(reversed(toy_batches())), None, torch.float32, torch.float32, 2
    )
    assert first["val/loss"] == pytest.approx(second["val/loss"], rel=1e-7)
    assert first["val/num_samples"] == 3
    for identity in draws:
        assert all(torch.equal(a, b) for a, b in zip(draws[identity], trainer.draws[identity]))
    assert_rng_equal(before, rng_snapshot())
    assert trainer.timestep_range_pool is pool and pool == [(0.2, 0.4)]
    assert modes == [module.training for root in (model, network) for module in root.modules()]
    assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters()))
    assert all(torch.equal(a, b.grad) for a, b in zip(gradients, model.parameters()))
    assert scheduler.state_dict() == scheduler_state
    for index, state in optimizer.state_dict()["state"].items():
        assert all(torch.equal(value, optimizer_state["state"][index][key]) for key, value in state.items())
    # Training can backpropagate again after validation.
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()


def test_weighted_aggregation_and_max_batches():
    trainer = ObjectiveTrainer()
    trainer.process_batch = lambda *a: (a[4]["loss"], {})
    batches = [
        {"validation_identity": str(i), "latents": torch.zeros(count, 1, 2, 2), "loss": torch.tensor(loss)}
        for i, (count, loss) in enumerate([(2, 1.0), (1, 4.0)])
    ]
    args = args_for_validation()
    model = torch.nn.Linear(2, 2)
    metrics = evaluate_validation(
        trainer, args, AcceleratorStub(), model, model, copy.deepcopy(batches), None, torch.float32, torch.float32, 2
    )
    assert metrics["val/loss"] == 2.0 and metrics["val/num_samples"] == 3
    args.validation_max_batches = 1
    metrics = evaluate_validation(
        trainer, args, AcceleratorStub(), model, model, copy.deepcopy(batches), None, torch.float32, torch.float32, 2
    )
    assert metrics["val/loss"] == 1.0 and metrics["val/num_samples"] == 2


def test_failure_restores_offloader_rng_modes_and_checkpoint_flags():
    trainer = ObjectiveTrainer()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.gradient_checkpointing = True
    model.activation_cpu_offloading = True
    model.blocks_to_swap = 2
    events = []

    class Offloader:
        forward_only = False

        def set_forward_only(self, value):
            self.forward_only = value
            events.append(value)

    model.offloader_double = Offloader()
    model.offloader_single = Offloader()
    model.prepare_block_swap_before_forward = lambda: events.append("prepare")

    def inference():
        model.offloader_double.set_forward_only(True)
        model.offloader_single.set_forward_only(True)
        model.prepare_block_swap_before_forward()

    model.switch_block_swap_for_inference = inference

    def fail(*args):
        random.random()
        np.random.rand()
        torch.rand(1)
        assert model.offloader_double.forward_only and not model.training
        raise RuntimeError("evaluation failure")

    trainer.process_batch = fail
    before = rng_snapshot()
    with pytest.raises(RuntimeError, match="evaluation failure"):
        evaluate_validation(
            trainer, args_for_validation(), AcceleratorStub(), model, model, toy_batches(), None, torch.float32, torch.float32, 2
        )
    assert model.training and model.gradient_checkpointing and model.activation_cpu_offloading
    assert not model.offloader_double.forward_only and not model.offloader_single.forward_only
    assert events == [True, True, "prepare", False, False, "prepare"]
    assert_rng_equal(before, rng_snapshot())


def test_periodic_hook_logs_existing_tracker_axis_and_disabled_is_noop(monkeypatch):
    trainer = Flux2NetworkTrainer()
    accelerator = AcceleratorStub()
    args = args_for_validation()
    trainer.validation_loader = object()
    calls = []

    def evaluate(*values):
        calls.append(values[-1])
        return {"val/loss": 0.25, "val/num_samples": 3, "val/evaluation_seconds": 0.1}

    monkeypatch.setattr("musubi_tuner.training.flux2_validation.evaluate_validation", evaluate)
    for step in (1, 2, 3, 4):
        trainer.on_completed_optimizer_step(args, accelerator, None, None, None, None, None, step)
    assert calls == [2, 4]
    assert [step for _, step in accelerator.logs] == [2, 4]
    assert all(metrics["val/loss"] == 0.25 for metrics, _ in accelerator.logs)
    assert trainer.extra_step_logs(args, {"loss/current": 0.5}) == {"train/loss": 0.5}
    args.validation_dataset_config = None
    del trainer.validation_loader
    trainer.on_completed_optimizer_step(args, accelerator, None, None, None, None, None, 6)
    assert calls == [2, 4]


def test_memory_logged_each_optimizer_step_and_validation_snapshots_follow_modes(monkeypatch):
    trainer = ObjectiveTrainer()
    accelerator = AcceleratorStub()
    args = args_for_validation()
    model = torch.nn.Linear(2, 2)
    trainer.validation_loader = toy_batches()
    snapshots = []

    class Diagnostics:
        samples = iter([10, 30, 20])

        def snapshot(self, prefix):
            snapshots.append((prefix, model.training))
            value = next(self.samples) if prefix.endswith("sampled_peak") else 5
            return {f"{prefix}/allocated_mib": value}

        def begin_validation(self):
            snapshots.append(("reset_peak", model.training))

        def validation_peaks(self):
            return {"memory/validation_peak/allocated_mib": 40}

    trainer.memory_diagnostics = Diagnostics()
    for step in (1, 2, 3):
        trainer.on_completed_optimizer_step(args, accelerator, model, model, None, torch.float32, torch.float32, step)
    train_logs = [(metrics, step) for metrics, step in accelerator.logs if "memory/train/allocated_mib" in metrics]
    assert [step for _, step in train_logs] == [1, 2, 3]
    validation = [metrics for metrics, _ in accelerator.logs if "val/loss" in metrics]
    assert len(validation) == 1
    assert validation[0]["memory/validation_sampled_peak/allocated_mib"] == 30
    assert validation[0]["memory/validation_peak/allocated_mib"] == 40
    assert ("memory/before_validation", True) in snapshots
    assert ("memory/validation_end", False) in snapshots
    assert ("memory/after_validation", True) in snapshots
    args.validation_dataset_config = None
    before = len(snapshots)
    trainer.on_completed_optimizer_step(args, accelerator, model, model, None, None, None, 4)
    assert len(snapshots) == before


def test_disabled_dataset_build_has_no_validation_components(monkeypatch):
    from musubi_tuner.training.trainer_base import NetworkTrainer

    monkeypatch.setattr(NetworkTrainer, "_build_dataset", lambda *args: ("train", "collator", "epoch"))
    trainer = Flux2NetworkTrainer()
    args = args_for_validation(validation_dataset_config=None)
    assert trainer._build_dataset(args) == ("train", "collator", "epoch")
    assert not hasattr(trainer, "validation_loader")


@pytest.mark.parametrize(
    "options",
    [
        {"validation_every_n_steps": None},
        {"validation_every_n_steps": 0},
        {"validation_max_batches": 0},
        {"validation_dataset_config": None},
    ],
)
def test_invalid_validation_options(options):
    with pytest.raises(ValueError):
        validate_validation_args(args_for_validation(**options))


def test_disabled_defaults_and_identity_seed():
    args = flux2_setup_parser(setup_parser_common()).parse_args([])
    validate_validation_args(args)
    assert args.validation_dataset_config is None
    assert example_seed(42, "a") == example_seed(42, "a")
    assert example_seed(42, "a") != example_seed(43, "a") != example_seed(43, "b")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_is_restored():
    torch.zeros(1, device="cuda")  # initialize CUDA before capturing RNG
    before = rng_snapshot()
    with isolated_rng():
        torch.manual_seed(31)
        torch.randn(8, device="cuda")
    assert_rng_equal(before, rng_snapshot())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "swap_mode,compiled,fp8",
    [
        (None, False, False),
        ("standard", False, False),
        ("h2d", False, False),
        ("h2d", True, False),
        ("h2d", True, True),
        pytest.param(
            "h2d", "inductor", True,
            marks=pytest.mark.skipif(
                not has_triton(), reason="Working CUDA Triton unavailable"
            ),
        ),
    ],
)
def test_cuda_flux2_lora_training_resumes_after_repeated_validation(tmp_path, swap_mode, compiled, fp8):
    """Reduced Klein architecture, real BF16/LoRA/checkpointing/SDPA/offload execution.

    This is a synthetic architecture smoke test, not a pretrained 4B quality/VRAM test.
    The Inductor case also exercises CUDA compilation across train/eval transitions.
    Other compiled cases use Dynamo's eager backend without requiring Triton.
    """
    from accelerate.utils import send_to_device
    from musubi_tuner.flux_2.flux2_models import Flux2, Klein4BParams
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
    from musubi_tuner.networks.lora_flux_2 import create_arch_network

    device = torch.device("cuda")
    backend = compiled if isinstance(compiled, str) else "eager"
    args = args_for_validation(gradient_checkpointing=True, compile_backend=backend)
    trainer = Flux2NetworkTrainer()
    trainer.num_timestep_buckets = None
    model = Flux2(
        Klein4BParams(
            in_channels=4,
            context_in_dim=8,
            hidden_size=32,
            num_heads=2,
            axes_dim=[4, 4, 4, 4],
            depth=5,
            depth_single_blocks=8,
            mlp_ratio=2,
        ),
        attn_mode="torch",
    )
    model.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    if fp8:
        from musubi_tuner.flux_2.flux2_models import FP8_OPTIMIZATION_TARGET_KEYS, FP8_OPTIMIZATION_EXCLUDE_KEYS
        from musubi_tuner.modules.fp8_optimization_utils import optimize_state_dict_with_fp8, apply_fp8_monkey_patch

        state = optimize_state_dict_with_fp8(
            model.state_dict(),
            "cpu",
            FP8_OPTIMIZATION_TARGET_KEYS,
            FP8_OPTIMIZATION_EXCLUDE_KEYS,
        )
        apply_fp8_monkey_patch(model, state, use_scaled_mm=False)
        model.load_state_dict(state, assign=True)
        assert any(parameter.dtype == torch.float8_e4m3fn for parameter in model.parameters())
    network = create_arch_network(1.0, 4, 4, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False)
    model.enable_gradient_checkpointing()
    network.to(device=device, dtype=torch.float32)
    if swap_mode:
        trainer.blocks_to_swap = 6
        model.enable_block_swap(6, BlockSwapConfig(device=device, supports_backward=True, h2d_only=swap_mode == "h2d"))
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        trainer.blocks_to_swap = 0
        model.to(device)
    if compiled:
        model = trainer.compile_transformer(args, model)
    model.train()
    network.train()
    config, _ = cache_config(tmp_path)
    args.validation_dataset_config = str(config)
    loader = build_validation_loader(args, "f2k4b", training_group(tmp_path))
    accelerator = AcceleratorStub()
    accelerator.device = device
    accelerator.autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)

    def train_step(step):
        batch = send_to_device(next(iter(loader)), device)
        latents = batch["latents"]
        with accelerator.autocast():
            loss, _ = trainer.process_batch(
                args,
                accelerator,
                model,
                network,
                batch,
                latents,
                torch.randn_like(latents),
                None,
                torch.bfloat16,
                torch.float32,
                None,
                step,
            )
        loss.backward()
        assert any(parameter.grad is not None for parameter in network.parameters())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)

    train_step(0)
    before = [parameter.clone() for parameter in network.parameters()]
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    rng = rng_snapshot()
    first = evaluate_validation(trainer, args, accelerator, network, model, loader, None, torch.bfloat16, torch.float32, 1)
    allocated_after_first = torch.cuda.memory_allocated()
    second = evaluate_validation(trainer, args, accelerator, network, model, loader, None, torch.bfloat16, torch.float32, 1)
    assert torch.cuda.memory_allocated() <= allocated_after_first + 1024 * 1024
    assert first["val/loss"] == pytest.approx(second["val/loss"], rel=1e-6)
    assert all(torch.equal(a, b) for a, b in zip(before, network.parameters()))
    assert scheduler.state_dict() == scheduler_state
    assert_rng_equal(rng, rng_snapshot())
    assert model.training and network.training and model.gradient_checkpointing
    if swap_mode:
        assert not model.offloader_double.forward_only and not model.offloader_single.forward_only
    train_step(1)
    assert any(not torch.equal(a, b) for a, b in zip(before, network.parameters()))
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_training_loop_accumulation_tensorboard_and_disabled_equivalence(tmp_path):
    """Exercise the actual Accelerator training loop and read its TensorBoard events."""
    from safetensors.torch import load_file
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from musubi_tuner.flux_2.flux2_models import Flux2, Klein4BParams

    class SyntheticTrainer(Flux2NetworkTrainer):
        def __init__(self):
            super().__init__()
            self.training_batches = 0
            self.validation_steps = []
            self.sample_steps = []

        def prepare_sampling(self, args, accelerator, vae_dtype):
            return [{}], None

        def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
            # Keep the real periodic sampling/RNG/offloader lifecycle; mock only image generation.
            self.sample_steps.append(steps)
            torch.randn(4)
            torch.randn(4, device=accelerator.device)

        def load_transformer(self, accelerator, args, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype):
            return Flux2(
                Klein4BParams(
                    in_channels=4,
                    context_in_dim=8,
                    hidden_size=32,
                    num_heads=2,
                    axes_dim=[4, 4, 4, 4],
                    depth=1,
                    depth_single_blocks=1,
                    mlp_ratio=2,
                ),
                attn_mode="torch",
            ).to(device=loading_device, dtype=torch.bfloat16)

        def process_batch(self, *args):
            if torch.is_grad_enabled():
                self.training_batches += 1
            return super().process_batch(*args)

        def on_completed_optimizer_step(self, *args):
            if args[0].validation_dataset_config:
                self.validation_steps.append((args[-1], self.training_batches))
            return super().on_completed_optimizer_step(*args)

    train_root, val_root = tmp_path / "train_data", tmp_path / "val_data"
    train_root.mkdir()
    val_root.mkdir()
    train_config, _ = cache_config(train_root, count=4)
    train_config.write_text(
        train_config.read_text().replace("batch_size = 2", "batch_size = 1").replace("num_repeats = 2", "num_repeats = 1")
    )
    val_config, _ = cache_config(val_root, count=2)
    results = []
    for enabled in (True, False):
        output = tmp_path / ("enabled" if enabled else "disabled")
        args = args_for_validation(
            model_version="klein-base-4b",
            dataset_config=str(train_config),
            dit="synthetic",
            output_dir=str(output),
            output_name="smoke",
            network_module="musubi_tuner.networks.lora_flux_2",
            network_dim=4,
            network_alpha=4,
            mixed_precision="bf16",
            dit_dtype=None,
            vae_dtype="float32",
            gradient_checkpointing=True,
            gradient_accumulation_steps=2,
            max_train_steps=3,
            max_train_epochs=None,
            max_data_loader_n_workers=0,
            optimizer_type="AdamW",
            learning_rate=0.001,
            seed=77,
            sdpa=True,
            logging_dir=str(output / "logs"),
            log_with="tensorboard",
            validation_dataset_config=str(val_config) if enabled else None,
            validation_every_n_steps=1 if enabled else None,
            sample_prompts="synthetic",
            sample_every_n_steps=1,
        )
        trainer = SyntheticTrainer()
        trainer.train(args)
        assert trainer.training_batches == 6
        assert trainer.validation_steps == ([(1, 2), (2, 4), (3, 6)] if enabled else [])
        # Native sampling also runs at epoch ends when the step interval matches.
        assert trainer.sample_steps == [1, 2, 2, 3, 3]
        events = list((output / "logs").rglob("events.out.tfevents.*"))
        assert events
        accumulator = EventAccumulator(str(events[0])).Reload()
        assert "train/loss" in accumulator.Tags()["scalars"]
        if enabled:
            assert [event.step for event in accumulator.Scalars("val/loss")] == [1, 2, 3]
            assert all(event.value == 2 for event in accumulator.Scalars("val/num_samples"))
        else:
            assert "val/loss" not in accumulator.Tags()["scalars"]
        results.append(load_file(str(output / "smoke.safetensors")))
    # Validation leaves all future training draws/batch progression and optimizer updates intact.
    assert results[0].keys() == results[1].keys()
    assert all(torch.equal(results[0][key], results[1][key]) for key in results[0])
