import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from musubi_tuner.training import cuda_memory

MIB = 1024**2


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "inf"])
def test_invalid_budget(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cuda_memory.memory_fraction(value)


@pytest.mark.parametrize(
    "platform,device,block_swap,requested,current,expected",
    [
        ("win32", "cuda:0", True, None, 1.0, 0.85),
        ("win32", "cuda:0", True, None, 0.6, 0.6),
        ("win32", "cuda:0", False, None, 1.0, None),
        ("linux", "cuda:0", True, None, 1.0, None),
        ("linux", "cuda:0", False, 0.75, 1.0, 0.75),
        ("win32", "cuda:0", True, 1.0, 1.0, 1.0),
        ("win32", "cpu", True, None, 1.0, None),
    ],
)
def test_budget_scope_and_existing_limit(monkeypatch, platform, device, block_swap, requested, current, expected):
    monkeypatch.setattr(cuda_memory.sys, "platform", platform)
    get = Mock(return_value=current)
    set_limit = Mock()
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", get)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", set_limit)
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(return_value=SimpleNamespace(total_memory=8192 * MIB)))
    assert cuda_memory.configure_cuda_memory_budget(device, requested, block_swap=block_swap) == expected
    if expected is None:
        get.assert_not_called()
        set_limit.assert_not_called()
    else:
        set_limit.assert_called_once_with(expected, torch.device(device))


def test_budget_installed_before_model_allocations(monkeypatch):
    from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer

    trainer = Flux2NetworkTrainer()
    trainer.model_version_info = object()
    events = []
    monkeypatch.setattr(cuda_memory, "configure_cuda_memory_budget", lambda *a, **kw: events.append("budget"))
    monkeypatch.setattr(
        "musubi_tuner.flux_2.flux2_utils.load_flow_model",
        lambda *a, **kw: events.append("load") or object(),
    )
    args = SimpleNamespace(blocks_to_swap=13, fp8_scaled=True, disable_numpy_memmap=False)
    trainer.load_transformer(SimpleNamespace(device="cuda:0"), args, "unused", "torch", False, "cpu", None)
    assert events == ["budget", "load"]


@pytest.mark.parametrize("device,index", [("cuda", 2), ("cuda:1", 1)])
def test_budget_resolves_implicit_cuda_device(monkeypatch, device, index):
    current_device = Mock(return_value=2)
    monkeypatch.setattr(torch.cuda, "current_device", current_device)

    def get_limit(resolved):
        # PyTorch 2.6 rejects torch.device("cuda") in this API.
        assert resolved == torch.device("cuda", index)
        return 1.0

    setter = Mock()
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", get_limit)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", setter)
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(return_value=SimpleNamespace(total_memory=8192 * MIB)))
    assert cuda_memory.configure_cuda_memory_budget(device, 0.9125) == 0.9125
    setter.assert_called_once_with(0.9125, torch.device("cuda", index))
    if device == "cuda":
        current_device.assert_called_once_with()
    else:
        current_device.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_allocator_recycles_variable_size_cache_before_budget():
    """Scaled reproduction: changing tensor sizes leave live memory flat but grow cached segments.

    Use at most 132 MiB of tensors, not the full training model. With a 64 MiB
    budget the allocator must reclaim those segments without an explicit flush.
    """
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("Regression concerns the native allocator")
    device = torch.device("cuda:0")
    original = torch.cuda.get_per_process_memory_fraction(device)
    total = torch.cuda.get_device_properties(device).total_memory
    baseline = torch.cuda.memory_reserved(device)
    if baseline:
        pytest.skip("Run this allocator reproduction in a fresh test process")
    try:
        peaks = []
        retries = []
        for budget in (1.0, 64 * MIB / total):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_stats(device)["num_alloc_retries"]
            torch.cuda.set_per_process_memory_fraction(1.0, device)
            cuda_memory.configure_cuda_memory_budget(device, budget)
            for size in range(12, 25, 2):
                tensor = torch.empty(size * MIB, dtype=torch.uint8, device=device)
                tensor.fill_(7)
                assert tensor[0].item() == 7
                del tensor
            peaks.append(torch.cuda.max_memory_reserved(device))
            retries.append(torch.cuda.memory_stats(device)["num_alloc_retries"] - before)
        assert peaks[0] > 64 * MIB
        assert peaks[1] <= 64 * MIB
        assert retries[1] > 0
        print(f"variable-size allocator: uncapped={peaks[0] / MIB:.0f} MiB, capped={peaks[1] / MIB:.0f} MiB")
    finally:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(original, device)
