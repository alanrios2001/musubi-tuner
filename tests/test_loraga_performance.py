"""Exact accumulation, bounded residency, and CPU/CUDA initialization parity."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.networks.loraga import initialize_factors
from musubi_tuner.training.loraga_streaming import DiskGradients, GradientStore, WeightCache, streamed_linears


def test_ram_accumulation_is_owned_bounded_and_avoids_disk(tmp_path):
    gradient = torch.ones(8, 8)
    store = GradientStore(tmp_path, ram_bytes=gradient.numel() * 4)
    store.add("ram", gradient)
    gradient.mul_(2)
    store.add("spill", gradient)
    store.add("ram", gradient)
    store.add("spill", gradient)
    assert store.ram_bytes == 256
    assert store.write_bytes == 512
    assert store.read_bytes == 256
    torch.testing.assert_close(store.take("ram", 2), torch.full((8, 8), 1.5), rtol=0, atol=0)
    torch.testing.assert_close(store.take("spill", 2), gradient, rtol=0, atol=0)
    assert store.ram_bytes == 0
    assert not store.tensors and not store.paths


def test_disk_names_remain_unique_after_consumption(tmp_path):
    store = DiskGradients(tmp_path)
    store.add("first", torch.ones(8, 8))
    store.add("second", torch.full((8, 8), 2.0))
    store.take("first", 1)
    store.add("third", torch.full((8, 8), 3.0))
    torch.testing.assert_close(store.take("second", 1), torch.full((8, 8), 2.0))
    torch.testing.assert_close(store.take("third", 1), torch.full((8, 8), 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_gradient_promotion_removes_repeated_disk_io(tmp_path):
    store = GradientStore(tmp_path, ram_bytes=0)
    gradients = {name: torch.full((8, 8), value, device="cuda") for name, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))}
    for name, gradient in gradients.items():
        store.add(name, gradient)
    assert store.promote_spill_to_cuda(512, torch.device("cuda")) == 512
    assert len(store.paths) == 1
    before = store.write_bytes
    for _ in range(3):
        for name, gradient in gradients.items():
            store.add(name, gradient)
    assert store.write_bytes - before == 3 * 256
    for name, gradient in gradients.items():
        actual = store.take(name, 4)
        torch.testing.assert_close(actual.cpu(), gradient.cpu(), rtol=0, atol=0)
    assert store.cuda_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_weight_cache_preserves_gradients_and_releases_residency(tmp_path, checkpointed, pin_memory):
    torch.manual_seed(314)
    model = nn.Sequential(nn.Linear(8, 12, bias=False), nn.Tanh(), nn.Linear(12, 8, bias=False))
    original = copy.deepcopy(model.state_dict())
    reference = copy.deepcopy(model).cuda()
    targets = {"first": model[0], "last": model[2]}
    store = GradientStore(tmp_path, ram_bytes=4096)
    cache = WeightCache()
    expected = {name: torch.zeros_like(module.weight) for name, module in targets.items()}
    with streamed_linears(model, targets, store, torch.device("cuda"), weight_cache=cache, pin_memory=pin_memory):
        cache.fill(model, targets, 4096, torch.device("cuda"))
        assert 0 < cache.bytes <= 4096
        for _ in range(3):
            x = torch.randn(2, 4, 8, device="cuda")
            reference.zero_grad(set_to_none=True)
            reference(x).square().mean().backward()
            expected["first"] += reference[0].weight.grad.cpu() / 3
            expected["last"] += reference[2].weight.grad.cpu() / 3
            output = checkpoint(model, x, use_reentrant=False) if checkpointed else model(x)
            output.square().mean().backward()
    assert cache.hits > 0 and cache.bytes == 0 and not cache.weights
    assert store.write_bytes == 0
    for name, gradient in expected.items():
        torch.testing.assert_close(store.take(name, 3), gradient, rtol=3e-5, atol=1e-7)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bf16_tensorcore_gradient_matches_fp32_and_restores_backend(tmp_path):
    original_flag = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        model = nn.Linear(64, 128, bias=False).bfloat16()
        x = torch.randn(2, 257, 64, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(2, 257, 128, device="cuda", dtype=torch.bfloat16)
        reference = dy.reshape(-1, 128).float().T @ x.reshape(-1, 64).float()
        store = GradientStore(tmp_path, ram_bytes=128 * 64 * 4)
        with streamed_linears(model, {"linear": model}, store, torch.device("cuda")):
            model(x).backward(dy)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        torch.testing.assert_close(store.take("linear", 1), reference.cpu(), rtol=3e-5, atol=2e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_flag


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_svd_preserves_reference_subspaces_and_fp32_cpu_factors():
    torch.manual_seed(42)
    left, _ = torch.linalg.qr(torch.randn(48, 8))
    right, _ = torch.linalg.qr(torch.randn(40, 8))
    gradient = (left * torch.arange(8, 0, -1)) @ right.T
    torch.manual_seed(7)
    a_cpu, b_cpu = initialize_factors(gradient, 4, 0.5)
    torch.manual_seed(7)
    a_gpu, b_gpu = initialize_factors(gradient, 4, 0.5, device="cuda")
    for tensor in (a_gpu, b_gpu):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
    torch.testing.assert_close(a_gpu.T @ a_gpu, a_cpu.T @ a_cpu, rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(b_gpu @ b_gpu.T, b_cpu @ b_cpu.T, rtol=1e-4, atol=2e-6)
