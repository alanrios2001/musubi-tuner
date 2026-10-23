"""Read-only CUDA allocator and Windows process GPU memory measurements."""

import ctypes
import logging
import os

import torch

logger = logging.getLogger(__name__)
MIB = 1024**2


class WindowsProcessGpuMemory:
    """PDH counters include driver-managed memory, unlike PyTorch allocator stats.

    Counters are summed across this process's adapters. No subprocess or GPU
    allocation is needed. Missing counters are omitted, never reported as zero.
    """

    def __init__(self):
        from ctypes import wintypes

        class CounterValue(ctypes.Structure):
            _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_double)]

        class CounterItem(ctypes.Structure):
            _fields_ = [("name", wintypes.LPWSTR), ("value", CounterValue)]

        self.item_type = CounterItem
        self.query = wintypes.HANDLE()
        self.pdh = ctypes.WinDLL("pdh")
        signatures = {
            "PdhOpenQueryW": [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)],
            "PdhAddEnglishCounterW": [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)],
            "PdhCollectQueryData": [wintypes.HANDLE],
            "PdhGetFormattedCounterArrayW": [
                wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
            ],
            "PdhCloseQuery": [wintypes.HANDLE],
        }
        for name, argtypes in signatures.items():
            function = getattr(self.pdh, name)
            function.argtypes = argtypes
            function.restype = wintypes.DWORD
        self.counters = {}
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)):
            raise OSError("Cannot open GPU memory performance query")
        for label, counter in (("dedicated", "Dedicated Usage"), ("shared", "Shared Usage")):
            handle = wintypes.HANDLE()
            path = rf"\GPU Process Memory(pid_{os.getpid()}_*)\{counter}"
            if not self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)):
                self.counters[label] = handle
        if not self.counters:
            self.close()
            raise OSError("GPU Process Memory performance counters unavailable")

    def read(self):
        from ctypes import wintypes

        if self.pdh.PdhCollectQueryData(self.query):
            return {}
        result = {}
        for label, handle in self.counters.items():
            size, count = wintypes.DWORD(), wintypes.DWORD()
            # PDH_MORE_DATA: first call obtains the required buffer size.
            status = self.pdh.PdhGetFormattedCounterArrayW(handle, 0x200, ctypes.byref(size), ctypes.byref(count), None)
            if status != 0x800007D2 or not size.value:
                continue
            buffer = ctypes.create_string_buffer(size.value)
            if self.pdh.PdhGetFormattedCounterArrayW(handle, 0x200, ctypes.byref(size), ctypes.byref(count), buffer):
                continue
            items = ctypes.cast(buffer, ctypes.POINTER(self.item_type))
            values = [items[i].value.value for i in range(count.value) if items[i].value.status in (0, 1)]
            if values:
                result[f"process_{label}_mib"] = sum(values) / MIB
        return result

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query.value = None

    def __del__(self):
        if getattr(self, "query", None):
            self.close()


class MemoryDiagnostics:
    def __init__(self, device):
        self.device = torch.device(device)
        self.windows = None
        if self.device.type == "cuda" and os.name == "nt":
            try:
                self.windows = WindowsProcessGpuMemory()
            except OSError as error:
                logger.warning("Windows GPU memory diagnostics unavailable: %s", error)

    def snapshot(self, prefix):
        if self.device.type != "cuda":
            return {}
        metrics = {
            "allocated_mib": torch.cuda.memory_allocated(self.device) / MIB,
            "reserved_mib": torch.cuda.memory_reserved(self.device) / MIB,
        }
        if self.windows is not None:
            metrics.update(self.windows.read())
        return {f"{prefix}/{key}": value for key, value in metrics.items()}

    def begin_validation(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def validation_peaks(self):
        if self.device.type != "cuda":
            return {}
        return {
            "memory/validation_peak/allocated_mib": torch.cuda.max_memory_allocated(self.device) / MIB,
            "memory/validation_peak/reserved_mib": torch.cuda.max_memory_reserved(self.device) / MIB,
        }
