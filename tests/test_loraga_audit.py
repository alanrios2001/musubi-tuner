"""Independent numerical invariants for the LoRA-GA implementation review."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from musubi_tuner.flux_2.flux2_models import Flux2, Flux2Params
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.networks import lora_flux_2
from musubi_tuner.networks.loraga import initialize_factors
from musubi_tuner.training.flux2_loraga import compensate, output_probe, quantize_model, reused_estimation_metrics
from musubi_tuner.training.loraga_streaming import DiskGradients, streamed_linears
from musubi_tuner.training.parser_common import setup_parser_common


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_svd_ignores_outer_autocast_and_restores_precision(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    gradient = torch.randn(32, 24, generator=torch.Generator().manual_seed(5))
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(42)
        expected = initialize_factors(gradient, 4, 1, device=device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(42)
        with torch.autocast(device, dtype=torch.bfloat16):
            actual = initialize_factors(gradient, 4, 1, device=device)
            assert torch.is_autocast_enabled(device)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_svd_restores_autocast_on_failure(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def fail(*args, **kwargs):
        assert not torch.is_autocast_enabled(device)
        if device == "cuda":
            assert not torch.backends.cuda.matmul.allow_tf32
        raise RuntimeError("SVD failed")

    monkeypatch.setattr(torch, "svd_lowrank", fail)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.autocast(device, dtype=torch.bfloat16):
            with pytest.raises(RuntimeError, match="SVD failed"):
                initialize_factors(torch.ones(8, 8), 2, 1, device=device)
            assert torch.is_autocast_enabled(device)
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def test_cache_timings_preserve_origin_without_reporting_repeated_estimation():
    cached = {
        "loraga/initialization_seconds": 561.5,
        "loraga/gradient_collection_seconds": 553.5,
        "loraga/svd_seconds": 8.0,
        "loraga/estimated_batches": 64,
        "loraga/initialization_peak_vram_bytes": 6085659648,
    }
    metrics = reused_estimation_metrics(cached)
    assert metrics["loraga/estimated_batches"] == 64
    assert metrics["loraga/estimated_batches_this_run"] == 0
    assert metrics["loraga/estimation_seconds"] == metrics["loraga/gradient_collection_seconds"] == metrics["loraga/svd_seconds"] == 0
    assert metrics["loraga/cache_origin/estimation_seconds"] == 561.5
    assert metrics["loraga/cache_origin/gradient_collection_seconds"] == 553.5
    assert metrics["loraga/cache_origin/initialization_peak_vram_bytes"] == 6085659648
    assert "loraga/initialization_peak_vram_bytes" not in metrics
    assert cached["loraga/gradient_collection_seconds"] == 553.5
    # Accelerate sidecars can themselves originate from a cached run.
    assert reused_estimation_metrics(metrics) == metrics


@pytest.mark.parametrize("checkpointed", [False, True])
def test_bf16_estimated_gradients_use_fp32_accumulation(tmp_path, checkpointed):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(104)
    reference = nn.Linear(64, 64, bias=False).bfloat16()
    model = nn.Linear(64, 64, bias=False).bfloat16()
    model.load_state_dict(reference.state_dict())
    sink = DiskGradients(tmp_path)
    oracle = torch.zeros(64, 64)
    with streamed_linears(model, {"linear": model}, sink, torch.device("cpu")):
        for _ in range(3):
            x = torch.randn(2, 13, 64)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                native_output = reference(x)
                native_output.retain_grad()
                actual = checkpoint(model, x, use_reentrant=False) if checkpointed else model(x)
            torch.testing.assert_close(actual, native_output, rtol=0, atol=0)
            native_output.float().square().mean().backward()
            # Differentiate using the rounded BF16 operands, accumulating their products in FP32.
            oracle += native_output.grad.flatten(0, -2).float().T @ x.bfloat16().flatten(0, -2).float()
            actual.float().square().mean().backward()
            assert model.weight.grad is None
    torch.testing.assert_close(sink.take("linear", 3), oracle / 3, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["ArB2r", "A2rBr", "ArBr"])
def test_first_step_gradient_matches_best_rank_2r_projection(direction):
    # A prescribed spectrum provides a dense-SVD oracle independent of the port's formulas.
    torch.manual_seed(101)
    u = torch.linalg.qr(torch.randn(12, 12)).Q
    v = torch.linalg.qr(torch.randn(8, 8)).Q
    singular = torch.arange(8, 0, -1, dtype=torch.float32)
    gradient = (u[:, :8] * singular) @ v.T
    rank, scaling, gamma = 2, 0.5, 16
    a, b = initialize_factors(gradient, rank, scaling, direction, "stable", gamma)
    a, b = nn.Parameter(a), nn.Parameter(b)
    # Differentiate an actual Linear adapter; the frozen compensation contributes no gradient.
    x = torch.eye(8)
    prediction = nn.functional.linear(nn.functional.linear(x, a), b) * scaling
    (prediction * gradient.T).sum().backward()
    first_order_update = scaling * (b.grad @ a + b @ a.grad)
    optimal_projection = (u[:, : 2 * rank] * singular[: 2 * rank]) @ v[:, : 2 * rank].T
    magnitude = scaling**2 * gradient.shape[0]**0.5 / gamma
    torch.testing.assert_close(first_order_update, magnitude * optimal_projection, atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("fp8", [False, True])
def test_output_probe_matches_native_bf16_lora_and_fp8(fp8):
    torch.manual_seed(102)
    model = Flux2(
        Flux2Params(
            in_channels=8, context_in_dim=16, hidden_size=32, num_heads=2,
            depth=1, depth_single_blocks=1, axes_dim=[4, 4, 4, 4],
            mlp_ratio=2, use_guidance_embed=False,
        ),
        attn_mode="torch",
    ).bfloat16().eval()
    network = lora_flux_2.create_arch_network(1, 2, 1, None, None, model)
    targets = {module.lora_name: module.org_module for module in network.unet_loras}
    initial = {name: initialize_factors(torch.randn_like(module.weight.float()), 2, 0.5) for name, module in targets.items()}
    compensate(model, initial, targets, 0.5, False, 2)
    args = flux2_setup_parser(setup_parser_common()).parse_args([])
    args.network_dim, args.network_alpha = 2, 1
    trainer = Flux2NetworkTrainer()
    device = torch.device("cpu")
    accelerator = SimpleNamespace(device=device, autocast=lambda: torch.autocast("cpu", dtype=torch.bfloat16))
    latents, noise = torch.randn(1, 8, 2, 2), torch.randn(1, 8, 2, 2)
    batch = {"ctx_vec": torch.randn(1, 3, 16), "latents_control_0": torch.randn_like(latents)}
    noisy, timesteps = (latents + noise) / 2, torch.tensor([500.0])
    inputs = batch, latents, noise, noisy, timesteps
    measured = output_probe(trainer, args, accelerator, model, inputs, targets, initial, fp8=fp8)
    if fp8:
        quantize_model(model)
    network.apply_to(None, model, apply_text_encoder=False)
    with torch.no_grad():
        for module in network.unet_loras:
            a, b = initial[module.lora_name]
            module.lora_down.weight.copy_(a)
            module.lora_up.weight.copy_(b)
    with torch.no_grad(), accelerator.autocast():
        native = trainer.call_dit(args, accelerator, model, latents, batch, noise, noisy, timesteps, torch.float32).pred
    torch.testing.assert_close(measured, native.float(), rtol=0, atol=0)
