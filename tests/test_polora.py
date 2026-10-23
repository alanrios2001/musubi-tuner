"""PoLoRA reference parity, complete optimizer state and native trainer integration."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.optimizers._polora_reference.optim import Polora as ReferencePolora
from musubi_tuner.optimizers.polora import Polora, create_optimizer, validate_args


def pairs(count=5, device="cpu", zero_b=False):
    torch.manual_seed(104)
    return [(torch.nn.Parameter(torch.randn(3, 8, device=device)),
             torch.nn.Parameter(torch.zeros(7, 3, device=device) if zero_b else torch.randn(7, 3, device=device)))
            for _ in range(count)]


def gradients(values, seed):
    generator = torch.Generator(device=values[0][0].device).manual_seed(seed)
    for a, b in values:
        for p in (a, b):
            p.grad = torch.randn(p.shape, generator=generator, device=p.device)


@pytest.mark.parametrize("zero_b", [False, True])
@pytest.mark.parametrize("chunk", [1, 4, 8])
def test_reference_update_and_all_state_match(chunk, zero_b):
    actual, expected = pairs(zero_b=zero_b), pairs(zero_b=zero_b)
    optimizer = Polora(actual, pair_batch_size=chunk)
    reference = ReferencePolora(pairs=expected)
    for step in range(3):
        gradients(actual, step)
        gradients(expected, step)
        optimizer.step()
        reference.step()
        for i, (left, right) in enumerate(zip(actual, expected)):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                assert torch.isfinite(a).all()
            for key, value in optimizer.pair_state[i].items():
                # Batch-size rounding can change the near-zero spectral direction;
                # the actual factor updates and curvature remain much closer.
                rtol, atol = (5e-3, 2e-4) if key.startswith("v_sigma") else (2e-5, 2e-6)
                torch.testing.assert_close(value, reference.pair_state[i][key], rtol=rtol, atol=atol)


def test_state_resume_is_bitwise_identical_and_preserves_scheduler():
    original = pairs()
    optimizer = Polora(original)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    gradients(original, 1)
    optimizer.step()
    scheduler.step()
    saved = copy.deepcopy(optimizer.state_dict())
    restored_pairs = [(torch.nn.Parameter(a.detach().clone()), torch.nn.Parameter(b.detach().clone())) for a, b in original]
    restored = Polora(restored_pairs)
    restored.load_state_dict(saved)
    assert restored.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    gradients(original, 2)
    gradients(restored_pairs, 2)
    optimizer.step()
    restored.step()
    for left, right in zip(original, restored_pairs):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert set(saved["polora"]["pairs"][0]) == {"M_A", "M_B", "P", "Q", "v_sigma_A", "v_sigma_B", "v_sigma_DA", "v_sigma_DB"}


def test_incomplete_or_incompatible_resume_is_rejected():
    optimizer = Polora(pairs())
    with pytest.raises(ValueError, match="Missing"):
        optimizer.load_state_dict({"state": {}, "param_groups": []})
    with pytest.raises(ValueError, match="mismatch"):
        Polora(pairs(), beta1=0.8).load_state_dict(optimizer.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    del state["polora"]["pairs"][0]["Q"]
    with pytest.raises(ValueError, match="keys"):
        optimizer.load_state_dict(state)
    gradients(optimizer.pairs, 1)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    del state["polora"]["pairs"][0]["v_sigma_A"]
    with pytest.raises(ValueError, match="keys"):
        optimizer.load_state_dict(state)


def test_missing_gradients_fail_before_any_update_and_closure_runs_once():
    values = pairs()
    optimizer = Polora(values, pair_batch_size=1)
    gradients(values, 1)
    before = values[0][0].detach().clone()
    values[-1][1].grad = None
    with pytest.raises(ValueError, match="dense gradients"):
        optimizer.step()
    assert torch.equal(before, values[0][0])
    calls = []

    def closure():
        calls.append(1)
        gradients(values, 2)
        return 3.0

    assert optimizer.step(closure) == 3
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("network_args", ["loraplus_lr_ratio=4"]),
                                         ("network_alpha", 16), ("max_data_loader_n_workers", 2),
                                         ("network_module", "networks.lora")])
def test_unsupported_configuration_rejected(field, value):
    args = SimpleNamespace(optimizer_type="polora", network_module="networks.lora_flux_2",
                           network_args=None, network_dim=64, network_alpha=64, max_data_loader_n_workers=0)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_args(args)


def test_factory_pairs_native_linear_factors_and_rejects_extra_parameters():
    from musubi_tuner.networks import lora_flux_2
    from test_loraga import tiny_model
    model = tiny_model()
    network = lora_flux_2.create_arch_network(1.0, 2, 2, None, [], model)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    groups, _ = network.prepare_optimizer_params(unet_lr=0.001)
    optimizer = create_optimizer(network, groups, 0.001)
    assert len(optimizer.pairs) == len(network.unet_loras)
    groups.append({"params": [torch.nn.Parameter(torch.ones(1))]})
    with pytest.raises(ValueError, match="extra trainable"):
        create_optimizer(network, groups, 0.001)


def cuda_room():
    return torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2 * 1024**3


@pytest.mark.skipif(not cuda_room(), reason="CUDA unavailable or busy")
@pytest.mark.parametrize("h2d_only", [False, True])
def test_native_fp8_checkpointing_swap_and_optimizer_device_migration(h2d_only):
    import gc
    from test_loraga import tiny_model, model_inputs
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
    from musubi_tuner.networks import lora_flux_2
    from musubi_tuner.training.flux2_loraga import quantize_model

    model = tiny_model(depth=4)
    network = lora_flux_2.create_arch_network(1, 2, 2, None, None, model)
    quantize_model(model)
    model.requires_grad_(False)
    network.apply_to(None, model, apply_text_encoder=False)
    groups, _ = network.prepare_optimizer_params(unet_lr=0.001)
    optimizer = create_optimizer(network, groups, 0.001, pair_batch_size=2)
    assert optimizer.pair_state[0]["M_A"].device.type == "cpu"
    config = BlockSwapConfig(device=torch.device("cuda"), supports_backward=True, h2d_only=h2d_only, ring_size=2)
    model.enable_block_swap(2, config)
    model.move_to_device_except_swap_blocks(torch.device("cuda"))
    network.to("cuda")
    model.enable_gradient_checkpointing(False)
    model.train()
    model.prepare_block_swap_before_forward()
    inputs = model_inputs("cuda")
    allocations = []
    for _ in range(4):
        for _ in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**inputs).float().square().mean() / 4
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del loss
        gc.collect()
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())
    assert optimizer.pair_state[0]["M_A"].device.type == "cuda"
    assert max(allocations[1:]) - min(allocations[1:]) < 1024 * 1024
    assert all(p.grad is None for p in model.parameters())
    restored = create_optimizer(network, groups, 0.001, pair_batch_size=2)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.pair_state[0]["M_A"].dtype == torch.float32


@pytest.mark.skipif(not cuda_room(), reason="CUDA unavailable or less than 2 GiB free; do not interfere with active training")
@pytest.mark.parametrize("loraga", [False, True])
def test_real_fp8_training_and_bitwise_resume(tmp_path, monkeypatch, loraga):
    from test_loraga import cached_training_fixture, Flux2NetworkTrainer, Flux2Params
    from musubi_tuner.flux_2 import flux2_utils

    _, args = cached_training_fixture(tmp_path, count=6)
    info = flux2_utils.FLUX2_MODEL_INFO["klein-base-4b"]
    params = Flux2Params(in_channels=8, context_in_dim=16, hidden_size=32, num_heads=2,
                        depth=2, depth_single_blocks=2, axes_dim=[4, 4, 4, 4],
                        mlp_ratio=2, use_guidance_embed=False)
    monkeypatch.setitem(flux2_utils.FLUX2_MODEL_INFO, "klein-base-4b", replace(info, params=params))
    args.loraga = loraga
    args.dit_dtype = None
    args.vae_dtype = "float32"
    args.gradient_checkpointing = True
    args.gradient_accumulation_steps = 2
    args.optimizer_type = "polora"
    args.learning_rate = 0.001
    args.save_every_n_steps = 1
    args.save_state = True
    args.output_dir = str(tmp_path / "original")
    args.log_with = "tensorboard"
    args.logging_dir = str(tmp_path / "logs")
    args.log_grad_metrics = True
    trainer = Flux2NetworkTrainer()
    trainer.train(args)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event = next((tmp_path / "logs").rglob("events.out.tfevents.*"))
    scalars = EventAccumulator(str(event)).Reload().Tags()["scalars"]
    assert {"loss/current", "train/loss", "grad/norm"}.issubset(scalars)
    assert any(key.startswith("lr/") for key in scalars)
    expected = load_file(str(tmp_path / "original/test.safetensors"))
    for step in (2, 3, 4):
        args.resume = str(tmp_path / f"original/test-step{step:08d}-state")
        args.output_dir = str(tmp_path / f"resume-{step}")
        resumed = Flux2NetworkTrainer()
        resumed.train(args)
        assert resumed._loraga_progress["global_step"] == 4
        for key, tensor in load_file(str(tmp_path / f"resume-{step}/test.safetensors")).items():
            torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
