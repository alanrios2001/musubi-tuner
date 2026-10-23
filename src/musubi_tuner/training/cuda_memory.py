"""Bound the CUDA cache before Windows can spill it into system memory."""

import argparse
import logging
import math
import sys

import torch

logger = logging.getLogger(__name__)


def memory_fraction(value):
    """argparse type for a nonzero fraction of physical device memory."""
    fraction = float(value)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError("CUDA memory fraction must be greater than 0 and at most 1")
    return fraction


def configure_cuda_memory_budget(device, fraction=None, *, block_swap=False):
    """Cap the cache through the allocator's native reclaim-and-retry mechanism.

    WDDM may satisfy cudaMalloc from shared RAM instead of returning an OOM.
    Without an allocator budget, unused cached blocks then survive indefinitely.
    A native allocator limit makes it release cached blocks and retry first.
    The default leaves headroom for CUDA context, desktop and other GPU users.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None
    if fraction is None:
        if sys.platform != "win32" or not block_swap:
            return None
        fraction = 0.85
    fraction = memory_fraction(fraction)
    # Accelerate can return index-free "cuda" in single-process training.
    # The allocator fraction APIs require an explicit CUDA device index.
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    # Preserve a stricter budget already installed by the caller.
    fraction = min(fraction, torch.cuda.get_per_process_memory_fraction(device))
    torch.cuda.set_per_process_memory_fraction(float(fraction), device)
    limit_mib = torch.cuda.get_device_properties(device).total_memory * fraction / 1024**2
    logger.info(
        "CUDA allocator budget: %.0f MiB (%.1f%%). Unused cached blocks will be reclaimed before exceeding it.",
        limit_mib, 100 * fraction,
    )
    return fraction
