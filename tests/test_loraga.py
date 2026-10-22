"""Reference math, bounded gradients, native export/cache/resume and CUDA execution."""

import copy
import gc
import json

from accelerate import Accelerator
import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.networks.loraga import initialize_factors, converted_state_dict
from musubi_tuner.networks import lora_flux_2
from musubi_tuner.training.flux2_loraga import (
    compensate,
    quantize_model,
    save_cache,
    load_cache,
    initialize,
    register_resume,
)
from musubi_tuner.training.loraga_streaming import DiskGradients, streamed_linears
from musubi_tuner.flux_2.flux2_models import Flux2, Flux2Params
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.training.parser_common import setup_parser_common


@pytest.mark.parametrize("direction", ["ArB2r", "A2rBr", "ArBr"])
@pytest.mark.parametrize("scale", ["stable", "unit", "gd"])
def test_reference_factor_construction(direction, scale):
    # Independent transcription of the official lora_ga_init factor formulas.
    gradient = torch.randn(16, 12, generator=torch.Generator().manual_seed(7))
    torch.manual_seed(42)
    u, _, v = torch.svd_lowrank(gradient.float(), q=8, niter=4)
    v = v.T
    if direction == "ArB2r":
        b, a = u[:, 2:4], v[:2]
    elif direction == "A2rBr":
        b, a = u[:, :2], v[2:4]
    else:
        b, a = u[:, 0:4:2], v[1:4:2]
    if scale == "stable":
        a, b = a * 16**0.25 / 16**0.5, b * 16**0.25 / 16**0.5
    elif scale == "gd":
        a, b = a / 0.5, b / 0.5
    torch.manual_seed(42)
    actual_a, actual_b = initialize_factors(gradient, 2, 0.5, direction, scale)
    torch.testing.assert_close(actual_a, a, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_b, b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("gradient,rank,scaling", [(torch.zeros(8, 8), 2, 1), (torch.ones(8, 8), 5, 1), (torch.ones(8, 8), 2, 0)])
def test_invalid_initialization(gradient, rank, scaling):
    with pytest.raises(ValueError):
        initialize_factors(gradient, rank, scaling)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_gradient_estimator_matches_autograd_and_leaves_weights_unchanged(tmp_path, device, checkpointed):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 8))
    original = copy.deepcopy(model.state_dict())
    reference = copy.deepcopy(model).to(device)
    sink = DiskGradients(tmp_path)
    targets = {"first": model[0], "last": model[2]}
    xs = [torch.randn(2, 3, 8, device=device) for _ in range(3)]
    expected = {key: torch.zeros_like(module.weight) for key, module in targets.items()}
    with streamed_linears(model, targets, sink, torch.device(device)):
        for x in xs:
            reference.zero_grad(set_to_none=True)
            reference(x).square().mean().backward()
            expected["first"] += reference[0].weight.grad.cpu() / 3
            expected["last"] += reference[2].weight.grad.cpu() / 3
            output = checkpoint(model, x, use_reentrant=False) if checkpointed else model(x)
            output.square().mean().backward()
            assert all(parameter.grad is None for parameter in model.parameters())
    for name, gradient in expected.items():
        torch.testing.assert_close(sink.take(name, 3), gradient, rtol=2e-5, atol=1e-7)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, original[name], rtol=0, atol=0)
    assert not list(tmp_path.glob("*.pt"))


def test_compensation_fp8_error_and_exact_conversion(tmp_path):
    torch.manual_seed(123)
    original = nn.Linear(64, 64, bias=False).bfloat16()
    model = copy.deepcopy(original)
    a0, b0 = initialize_factors(torch.randn(64, 64), 4, 0.5)
    initial = {"linear": (a0, b0)}
    metrics = compensate(model, initial, {"linear": model}, 0.5, True, 2.0)
    assert metrics["loraga/compensation_weight_rmse"] < 0.001
    assert metrics["loraga/quantization_error_ratio"] < 2.0
    a, b = a0 + torch.randn_like(a0) * 0.01, b0 + torch.randn_like(b0) * 0.01
    state = {"linear.lora_down.weight": a, "linear.lora_up.weight": b, "linear.alpha": torch.tensor(2.0)}
    converted = converted_state_dict(state, initial)
    delta = (b @ a - b0 @ a0) * 0.5
    actual = converted["linear.lora_up.weight"] @ converted["linear.lora_down.weight"] * (converted["linear.alpha"] / 8)
    torch.testing.assert_close(delta, actual, atol=2e-8, rtol=2e-5)
    x = torch.randn(3, 64)
    torch.testing.assert_close(
        nn.functional.linear(x, original.weight.float() + delta),
        nn.functional.linear(x, original.weight.float() + actual),
        atol=2e-7,
        rtol=2e-5,
    )
    print("quantization metrics", metrics)


def test_cache_identity_and_shape_checks(tmp_path):
    path = tmp_path / "init.safetensors"
    initial = {"linear": (torch.randn(2, 8), torch.randn(8, 2))}
    metadata = {"rank": 2, "targets": {"linear": [8, 8]}, "checkpoint": "abc", "dataset": "xyz"}
    save_cache(path, initial, metadata, {"loraga/estimated_batches": 2})
    restored, metrics = load_cache(path, metadata)
    assert metrics["loraga/estimated_batches"] == 2
    torch.testing.assert_close(restored["linear"][0], initial["linear"][0])
    for key in ("checkpoint", "dataset", "rank"):
        with pytest.raises(ValueError, match="Incompatible"):
            load_cache(path, {**metadata, key: "different"})


def tiny_model(depth=2):
    return Flux2(
        Flux2Params(
            in_channels=8,
            context_in_dim=16,
            hidden_size=32,
            num_heads=2,
            depth=depth,
            depth_single_blocks=depth,
            axes_dim=[4, 4, 4, 4],
            mlp_ratio=2,
            use_guidance_embed=False,
        ),
        attn_mode="torch",
    ).bfloat16()


def model_inputs(device):
    return dict(
        x=torch.randn(1, 4, 8, device=device, dtype=torch.bfloat16),
        x_ids=torch.zeros(1, 4, 4, device=device),
        timesteps=torch.tensor([0.5], device=device),
        ctx=torch.randn(1, 3, 16, device=device, dtype=torch.bfloat16),
        ctx_ids=torch.zeros(1, 3, 4, device=device),
        guidance=None,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_export_load_and_frozen_training(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(8)
    original = tiny_model()
    model = copy.deepcopy(original)
    network = lora_flux_2.create_arch_network(1, 2, 1, None, None, model)
    initial = {
        module.lora_name: initialize_factors(torch.randn_like(module.org_module.weight.float()).cpu(), 2, 0.5)
        for module in network.unet_loras
    }
    targets = {module.lora_name: module.org_module for module in network.unet_loras}
    compensate(model, initial, targets, 0.5, False, 2)
    model.requires_grad_(False)
    network.apply_to(None, model, apply_text_encoder=False)
    network.loraga_initial = initial
    with torch.no_grad():
        for module in network.unet_loras:
            a, b = initial[module.lora_name]
            module.lora_down.weight.copy_(a)
            module.lora_up.weight.copy_(b)
    model.to(device)
    original.to(device)
    original_fp8 = copy.deepcopy(original).cpu()
    quantize_model(original_fp8)
    original_fp8.to(device)
    network.to(device)
    inputs = model_inputs(device)
    with torch.autocast(device, dtype=torch.bfloat16):
        expected_initial = original(**inputs).float()
        actual_initial = model(**inputs).float()
    assert (actual_initial - expected_initial).square().mean().sqrt() < 0.02
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    with torch.autocast(device, dtype=torch.bfloat16):
        loss = model(**inputs).float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    file = tmp_path / "export.safetensors"
    network.save_weights(str(file), torch.float32, {})
    state = load_file(str(file))
    assert all(state[f"{module.lora_name}.lora_down.weight"].shape[0] == 4 for module in network.unet_loras)
    inferred = lora_flux_2.create_arch_network_from_weights(1, state, unet=original, for_inference=True)
    inferred.apply_to(None, original, apply_text_encoder=False)
    inferred.load_state_dict(state, strict=True)
    inferred.to(device)
    inferred_fp8 = lora_flux_2.create_arch_network_from_weights(1, state, unet=original_fp8, for_inference=True)
    inferred_fp8.apply_to(None, original_fp8, apply_text_encoder=False)
    inferred_fp8.load_state_dict(state, strict=True)
    inferred_fp8.to(device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        exported_output, trained_output = original(**inputs).float(), model(**inputs).float()
        exported_fp8_output = original_fp8(**inputs).float()
    assert (exported_output - trained_output).square().mean().sqrt() < 0.02
    assert (exported_fp8_output - trained_output).square().mean().sqrt() < 0.06
    with safe_open(str(file), framework="pt") as saved:
        assert saved.metadata()["ss_network_dim"] == "4"
        assert saved.metadata()["ss_network_alpha"] == "2.0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("h2d_only,compiled", [(False, False), (True, False), (True, True)])
def test_phase_b_fp8_checkpointing_and_block_swap(tmp_path, h2d_only, compiled):
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    torch.manual_seed(55)
    model = tiny_model(depth=4)
    network = lora_flux_2.create_arch_network(1, 2, 2, None, None, model)
    targets = {module.lora_name: module.org_module for module in network.unet_loras}
    sink = DiskGradients(tmp_path)
    inputs = model_inputs("cuda")
    model.enable_gradient_checkpointing(False)
    model.train()
    torch.cuda.synchronize()
    with streamed_linears(model, targets, sink, torch.device("cuda")), torch.autocast("cuda", dtype=torch.bfloat16):
        model(**inputs).float().square().mean().backward()
    initial = {name: initialize_factors(sink.take(name, 1), 2, 1) for name in targets}
    compensate(model, initial, targets, 1, True, 3)
    model.disable_gradient_checkpointing()
    quantize_model(model)
    model.requires_grad_(False)
    config = BlockSwapConfig(device=torch.device("cuda"), supports_backward=True, h2d_only=h2d_only, ring_size=2)
    model.enable_block_swap(2, config)
    model.move_to_device_except_swap_blocks(torch.device("cuda"))
    network.apply_to(None, model, apply_text_encoder=False)
    network.to("cuda")
    with torch.no_grad():
        for module in network.unet_loras:
            a, b = initial[module.lora_name]
            module.lora_down.weight.copy_(a)
            module.lora_up.weight.copy_(b)
    model.enable_gradient_checkpointing(False)
    model.train()
    model.prepare_block_swap_before_forward()
    if compiled:
        from types import SimpleNamespace
        from musubi_tuner.utils.model_utils import compile_transformer

        compile_transformer(
            SimpleNamespace(
                compile_backend="eager",
                compile_mode="default",
                compile_dynamic=None,
                compile_fullgraph=False,
                compile_cache_size_limit=None,
            ),
            model,
            [model.double_blocks, model.single_blocks],
            disable_linear=True,
        )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    allocations = []
    for _ in range(4):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**inputs).float().square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del loss
        gc.collect()
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())
    assert max(allocations[1:]) - min(allocations[1:]) < 1024 * 1024
    assert all(parameter.grad is None for parameter in model.parameters())
    print("CUDA phase B allocations", allocations)


def test_conventional_path_has_no_conversion(tmp_path):
    model = tiny_model()
    network = lora_flux_2.create_arch_network(1, 2, 1, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False)
    assert all(torch.count_nonzero(module.lora_up.weight) == 0 for module in network.unet_loras)
    file = tmp_path / "standard.safetensors"
    network.save_weights(str(file), torch.float32, {})
    assert all(load_file(str(file))[f"{module.lora_name}.lora_down.weight"].shape[0] == 2 for module in network.unet_loras)


def cached_training_fixture(tmp_path, fp8=True, count=2):
    cache = tmp_path / "cache"
    cache.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(count):
        save_file(
            {"latents_2x2_float32": torch.randn(8, 2, 2), "latents_control_0_2x2_float32": torch.randn(8, 2, 2)},
            str(cache / f"pair{index}_0032x0032_f2k4b.safetensors"),
        )
        save_file({"ctx_vec_float32": torch.randn(3, 16)}, str(cache / f"pair{index}_f2k4b_te.safetensors"))
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[general]\nresolution = [32, 32]\nbatch_size = 1\n[[datasets]]\n"
        f'image_directory = "{image_dir.as_posix()}"\ncache_directory = "{cache.as_posix()}"\n',
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "backbone.safetensors"
    save_file(tiny_model().state_dict(), str(checkpoint_path))
    args = flux2_setup_parser(setup_parser_common()).parse_args([])
    for key, value in dict(
        loraga=True,
        model_version="klein-base-4b",
        mixed_precision="bf16",
        network_dim=2,
        network_alpha=2,
        network_module="networks.lora_flux_2",
        max_data_loader_n_workers=0,
        fp8_base=fp8,
        fp8_scaled=fp8,
        loraga_estimation_batches=2,
        seed=42,
        timestep_sampling="uniform",
        weighting_scheme="none",
        dataset_config=str(config),
        dit=str(checkpoint_path),
        output_dir=str(tmp_path),
        output_name="test",
        max_train_steps=4,
        sdpa=True,
        loraga_max_quantization_error_ratio=3,
    ).items():
        setattr(args, key, value)
    trainer = Flux2NetworkTrainer()
    trainer.handle_model_specific_args(args)
    trainer._build_dataset(args)
    return trainer, args


class ProbeAccelerator:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_processes = 1

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)


def test_two_stage_native_cached_objective_and_reuse(tmp_path, monkeypatch):
    torch.manual_seed(333)
    trainer, args = cached_training_fixture(tmp_path)
    model = tiny_model()
    model.load_state_dict(load_file(args.dit), strict=True)
    weights = copy.deepcopy(model.state_dict())
    accelerator = ProbeAccelerator()
    initialized = initialize(trainer, args, accelerator, model)
    assert next(initialized.double_blocks[0].img_attn.qkv.parameters()).dtype == torch.float8_e4m3fn
    assert trainer._loraga_metrics["loraga/estimated_batches"] == 2
    assert trainer._loraga_metrics["loraga/output_error_ratio"] < 3
    assert trainer._loraga_metrics["loraga/initial_output_relative_error"] < 0.05
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in initialized.parameters())
    assert not list(tmp_path.glob("musubi-loraga-*"))
    print("two-stage metrics", json.dumps(trainer._loraga_metrics, sort_keys=True))
    first_initial = trainer._loraga_initial
    second = tiny_model()
    second.load_state_dict(weights)
    trainer._loraga_dataset = (trainer._loraga_training_group, None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cache reuse repeated gradient estimation")

    monkeypatch.setattr("musubi_tuner.training.flux2_loraga.estimate", forbidden)
    # Storage and SVD execution choices must not invalidate completed initialization factors.
    args.loraga_gradient_ram_gb = 0
    args.loraga_gpu_cache_gb = 0
    args.loraga_svd_device = "cpu"
    initialize(trainer, args, accelerator, second)
    for key, value in initialized.state_dict().items():
        other = second.state_dict()[key]
        assert value.dtype == other.dtype
        torch.testing.assert_close(value.float(), other.float(), rtol=0, atol=0)
    for name, pair in first_initial.items():
        for a, b in zip(pair, trainer._loraga_initial[name]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_accelerate_state_restores_raw_factors_optimizer_rng_and_metadata(tmp_path):
    trainer, args = cached_training_fixture(tmp_path, fp8=False)
    model = tiny_model()
    network = lora_flux_2.create_arch_network(1, 2, 2, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False)
    accelerator = Accelerator(mixed_precision="bf16")
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    network, optimizer, scheduler = accelerator.prepare(network, optimizer, scheduler)
    trainer._loraga_initial = {
        module.lora_name: (module.lora_down.weight.detach().cpu().clone(), module.lora_up.weight.detach().cpu().clone())
        for module in accelerator.unwrap_model(network).unet_loras
    }
    trainer._loraga_metadata = {
        "rank": 2,
        "targets": {
            module.lora_name: [module.lora_up.weight.shape[0], module.lora_down.weight.shape[1]]
            for module in accelerator.unwrap_model(network).unet_loras
        },
    }
    trainer._loraga_saved_metrics = {"loraga/estimated_batches": 2}
    trainer._loraga_progress = {"global_step": 3, "epoch": 0, "next_batch": 1}
    trainer.timestep_range_pool = [(0.1, 0.2)]
    register_resume(trainer, args, accelerator, network)
    # Native hook excludes backbone state; rank-r factors remain raw for optimizer resume.
    Flux2NetworkTrainer._register_hooks_and_resume.__globals__["NetworkTrainer"]._register_hooks_and_resume(
        trainer, args, accelerator, network
    )
    loss = sum(parameter.square().sum() for parameter in network.parameters())
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    before = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
    expected_lr = scheduler.get_last_lr()
    directory = tmp_path / "state"
    accelerator.save_state(str(directory))
    expected_random = torch.randn(3)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(1)
    scheduler.step()
    trainer._loraga_progress = {"global_step": 99, "epoch": 1, "next_batch": 0}
    trainer.timestep_range_pool = []
    accelerator.load_state(str(directory))
    assert trainer._loraga_progress == {"global_step": 3, "epoch": 0, "next_batch": 1}
    assert trainer.timestep_range_pool == [(0.1, 0.2)]
    assert scheduler.get_last_lr() == expected_lr
    torch.testing.assert_close(torch.randn(3), expected_random, rtol=0, atol=0)
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value.cpu(), before[key], rtol=0, atol=0)
    raw = load_file(str(directory / "model.safetensors"))
    assert next(value for key, value in raw.items() if key.endswith("lora_down.weight")).shape[0] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_training_loop_tensorboard_and_resume_are_identical(tmp_path, monkeypatch):
    from dataclasses import replace
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from musubi_tuner.flux_2 import flux2_utils

    torch.manual_seed(42)
    _, args = cached_training_fixture(tmp_path, count=6)
    info = flux2_utils.FLUX2_MODEL_INFO["klein-base-4b"]
    params = Flux2Params(
        in_channels=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=2,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[4, 4, 4, 4],
        mlp_ratio=2,
        use_guidance_embed=False,
    )
    # Use the real loading/initialization/FP8/training pipeline, only shrink model geometry.
    monkeypatch.setitem(flux2_utils.FLUX2_MODEL_INFO, "klein-base-4b", replace(info, params=params))
    args.dit_dtype = None
    args.vae_dtype = "float32"
    args.gradient_checkpointing = True
    args.gradient_accumulation_steps = 2
    args.optimizer_type = "AdamW8bit"
    args.learning_rate = 0.001
    args.save_every_n_steps = 1
    args.save_state = True
    args.output_dir = str(tmp_path / "uninterrupted")
    args.logging_dir = str(tmp_path / "logs")
    args.log_with = "tensorboard"
    trainer = Flux2NetworkTrainer()
    trainer.train(args)
    expected = load_file(str(tmp_path / "uninterrupted" / "test.safetensors"))
    state = tmp_path / "uninterrupted" / "test-step00000002-state"
    assert state.is_dir()
    events = list((tmp_path / "logs").rglob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(events[0])).Reload()
    assert "loraga/initial_output_error" in accumulator.Tags()["scalars"]
    assert len(accumulator.Scalars("loss/current")) == 8
    # Mid-epoch, epoch boundary, and already-completed states must all resume safely.
    for checkpoint_step in (2, 3, 4):
        args.resume = str(tmp_path / "uninterrupted" / f"test-step{checkpoint_step:08d}-state")
        output = tmp_path / f"resumed-{checkpoint_step}"
        args.output_dir = str(output)
        resumed = Flux2NetworkTrainer()
        resumed.train(args)
        actual = load_file(str(output / "test.safetensors"))
        assert resumed._loraga_progress["global_step"] == 4
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_manual_export_and_double_conversion_rejection(tmp_path):
    from musubi_tuner.loraga_export import export

    initial = {"linear": (torch.randn(2, 8), torch.randn(8, 2))}
    identity = {"rank": 2, "alpha": 1.0, "targets": {"linear": [8, 8]}}
    cache = tmp_path / "init.safetensors"
    save_cache(cache, initial, identity, {})
    raw = {
        "linear.lora_down.weight": torch.randn(2, 8),
        "linear.lora_up.weight": torch.randn(8, 2),
        "linear.alpha": torch.tensor(1.0),
    }
    source, destination = tmp_path / "raw.safetensors", tmp_path / "converted.safetensors"
    save_file(raw, str(source))
    export(str(source), str(cache), str(destination))
    expected = converted_state_dict(raw, initial)
    for name, value in load_file(str(destination)).items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="already converted"):
        export(str(destination), str(cache), str(tmp_path / "again.safetensors"))


def test_prequantized_checkpoint_rejected(tmp_path):
    from musubi_tuner.training.flux2_loraga import validate_checkpoint

    path = tmp_path / "fp8.safetensors"
    save_file({"weight": torch.zeros(8, 8).to(torch.float8_e4m3fn)}, str(path))
    with pytest.raises(ValueError, match="original unquantized"):
        validate_checkpoint(str(path))


def test_init_only_stops_before_optimizer_or_swap(tmp_path, monkeypatch):
    from dataclasses import replace
    from musubi_tuner.flux_2 import flux2_utils

    trainer, args = cached_training_fixture(tmp_path)
    params = Flux2Params(
        in_channels=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=2,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[4, 4, 4, 4],
        mlp_ratio=2,
        use_guidance_embed=False,
    )
    monkeypatch.setitem(
        flux2_utils.FLUX2_MODEL_INFO, "klein-base-4b", replace(flux2_utils.FLUX2_MODEL_INFO["klein-base-4b"], params=params)
    )
    args.loraga_init_only = True
    args.blocks_to_swap = 2  # This geometry cannot swap two blocks; init-only must bypass swapping.
    args.dit_dtype = None
    args.vae_dtype = "float32"

    def forbidden(*args, **kwargs):
        raise AssertionError("Initialization-only run entered optimizer/network setup")

    monkeypatch.setattr(trainer, "_build_network", forbidden)
    monkeypatch.setattr(trainer, "_build_optimizer_and_dataloader", forbidden)
    trainer.train(args)
    assert (tmp_path / "loraga_init.safetensors").is_file()
    assert not (tmp_path / "test.safetensors").exists()
