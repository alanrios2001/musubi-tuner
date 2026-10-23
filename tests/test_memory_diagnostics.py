from unittest.mock import Mock

import pytest
import torch

from musubi_tuner.training.memory_diagnostics import MemoryDiagnostics, MIB


def test_cpu_diagnostics_does_not_touch_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", Mock(side_effect=AssertionError("CUDA accessed")))
    diagnostics = MemoryDiagnostics("cpu")
    assert diagnostics.snapshot("memory/train") == {}
    assert diagnostics.validation_peaks() == {}
    diagnostics.begin_validation()


def test_cuda_diagnostics_units_prefixes_and_peak_boundary(monkeypatch):
    # Exercise CUDA telemetry without creating a CUDA context or competing with a live trainer.
    diagnostics = MemoryDiagnostics("cpu")
    diagnostics.device = torch.device("cuda:0")
    diagnostics.windows = Mock()
    diagnostics.windows.read.return_value = {"process_shared_mib": 123.5}
    reset = Mock()
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    for name, value in (
        ("memory_allocated", 2), ("memory_reserved", 3),
        ("max_memory_allocated", 4), ("max_memory_reserved", 5),
    ):
        monkeypatch.setattr(torch.cuda, name, Mock(return_value=value * MIB))
    assert diagnostics.snapshot("memory/train") == {
        "memory/train/allocated_mib": 2,
        "memory/train/reserved_mib": 3,
        "memory/train/process_shared_mib": 123.5,
    }
    reset.assert_not_called()
    diagnostics.begin_validation()
    reset.assert_called_once_with(torch.device("cuda:0"))
    assert diagnostics.validation_peaks() == {
        "memory/validation_peak/allocated_mib": 4,
        "memory/validation_peak/reserved_mib": 5,
    }


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows PDH API")
def test_windows_counter_reader_closes_query():
    from musubi_tuner.training.memory_diagnostics import WindowsProcessGpuMemory

    try:
        reader = WindowsProcessGpuMemory()
    except OSError:
        pytest.skip("GPU Process Memory counters unavailable")
    try:
        # This test process need not have a GPU allocation; missing data must stay absent.
        result = reader.read()
        assert set(result) <= {"process_dedicated_mib", "process_shared_mib"}
        assert all(value >= 0 for value in result.values())
    finally:
        reader.close()
    assert not reader.query
