"""Initialization-only CPU weight/activation offload, independent of block swapping."""

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class DiskGradients:
    """One FP32 accumulator per target on disk; only one matrix is read at a time."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = {}
        self._next_index = 0
        self.read_bytes = 0
        self.write_bytes = 0

    def add(self, name, gradient):
        gradient = gradient.detach().to(device="cpu", dtype=torch.float32, copy=True)
        if name in self.paths:
            gradient.add_(torch.load(self.paths[name], map_location="cpu", weights_only=True))
            self.read_bytes += gradient.numel() * gradient.element_size()
        else:
            self.paths[name] = self.directory / f"{self._next_index:05d}.pt"
            self._next_index += 1
        torch.save(gradient, self.paths[name])
        self.write_bytes += gradient.numel() * gradient.element_size()

    def take(self, name, batches):
        if name not in self.paths:
            raise ValueError(f"No gradient collected for LoRA-GA target {name}")
        path = self.paths.pop(name)
        gradient = torch.load(path, map_location="cpu", weights_only=True).div_(batches)
        self.read_bytes += gradient.numel() * gradient.element_size()
        path.unlink()
        return gradient


def available_host_memory():
    """Available physical RAM, without adding a psutil dependency."""
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available
    else:
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            pass
    return 0


class GradientStore(DiskGradients):
    """Bounded FP32 accumulation in RAM, with disk spill and optional CUDA residency."""

    def __init__(self, directory, ram_bytes=0):
        super().__init__(directory)
        self.ram_limit = max(0, int(ram_bytes))
        self.ram_bytes = 0
        self.cuda_bytes = 0
        self.tensors = {}

    def add(self, name, gradient):
        gradient = gradient.detach().float()
        if name in self.tensors:
            target = self.tensors[name]
            target.add_(gradient.to(target.device))
        elif name not in self.paths and self.ram_bytes + gradient.numel() * 4 <= self.ram_limit:
            self.tensors[name] = gradient.to("cpu", copy=True)
            self.ram_bytes += gradient.numel() * 4
        else:
            super().add(name, gradient)

    def promote_spill_to_cuda(self, budget, device):
        """Remove disk traffic first; leave remaining VRAM for immutable weight caching."""
        for name, path in list(self.paths.items()):
            # Tensor serialization adds a small header; inspect only files which can fit.
            if path.stat().st_size > budget - self.cuda_bytes + 4096:
                continue
            gradient = torch.load(path, map_location="cpu", weights_only=True)
            size = gradient.numel() * 4
            if size <= budget - self.cuda_bytes:
                self.tensors[name] = gradient.to(device)
                self.cuda_bytes += size
                self.read_bytes += size
                path.unlink()
                del self.paths[name]
        return self.cuda_bytes

    def take(self, name, batches):
        if name not in self.tensors:
            return super().take(name, batches)
        gradient = self.tensors.pop(name)
        size = gradient.numel() * 4
        if gradient.device.type == "cuda":
            self.cuda_bytes -= size
        else:
            self.ram_bytes -= size
        return gradient.div_(batches)


class WeightCache:
    """Immutable initialization weights retained in measured spare VRAM, not in an LRU ring."""

    def __init__(self):
        self.weights = {}
        self.bytes = 0
        self.hits = 0

    def fill(self, model, targets, budget, device):
        target_ids = {id(module) for module in targets.values()}
        modules = [module for module in model.modules() if isinstance(module, nn.Linear)]
        modules.sort(key=lambda module: (id(module) in target_ids, module.weight.numel()), reverse=True)
        for module in modules:
            size = module.weight.numel() * module.weight.element_size()
            if size <= budget - self.bytes:
                self.weights[id(module.weight)] = module.weight.detach().to(device)
                self.bytes += size

    def get(self, weight, device):
        cached = self.weights.get(id(weight))
        if cached is not None:
            self.hits += 1
            return cached
        return weight.to(device)

    def clear(self):
        self.weights.clear()
        self.bytes = 0


class _StreamLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, sink, name, cache):
        ctx.weight, ctx.sink, ctx.name = weight, sink, name
        ctx.cache = cache
        ctx.input_dtype = x.dtype
        x = x.to(weight.dtype)
        ctx.save_for_backward(x)
        device_weight = weight.to(x.device) if cache is None else cache.get(weight, x.device)
        return F.linear(x, device_weight, None if bias is None else bias.to(x.device))

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            weight = ctx.weight.to(dy.device) if ctx.cache is None else ctx.cache.get(ctx.weight, dy.device)
            dx = (dy.to(ctx.weight.dtype) @ weight).to(ctx.input_dtype)
        if ctx.sink is not None:
            # FP32 GEMM and CPU accumulation, without attaching a full-sized .grad.
            with torch.autocast(device_type=dy.device.type, enabled=False):
                previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    # TF32 exactly represents BF16 input values; accumulation remains FP32.
                    if dy.device.type == "cuda" and x.dtype == dy.dtype == torch.bfloat16:
                        torch.backends.cuda.matmul.allow_tf32 = True
                    gradient = dy.reshape(-1, dy.shape[-1]).float().T @ x.reshape(-1, x.shape[-1]).float()
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            ctx.sink.add(ctx.name, gradient)
        return dx, None, None, None, None, None


@contextmanager
def streamed_linears(model, targets, sink, device, weight_transform=None, additions=None, weight_cache=None, pin_memory=False):
    """Stream immutable BF16 weights for forward AND backward; offload saved tensors.

    Target weights require grad to create autograd edges, but their gradients go
    directly to the disk accumulator. No optimizer or H2D-only offloader is used.
    """
    model.cpu().requires_grad_(False)
    originals = []
    target_names = {id(module): name for name, module in targets.items()}
    try:
        for path, module in model.named_modules():
            if isinstance(module, nn.Linear):
                originals.append((module, module.forward))
                name = target_names.get(id(module))
                module.weight.requires_grad_(name is not None)

                def forward(x, m=module, n=name, p=path):
                    weight = m.weight if weight_transform is None else weight_transform(p, m.weight)
                    output = _StreamLinear.apply(x, weight, m.bias, sink if n is not None else None, n, weight_cache)
                    if additions and id(m) in additions:
                        a, b, scale = additions[id(m)]
                        output = (
                            output
                            + F.linear(F.linear(x.to(output.dtype), a.to(x.device, output.dtype)), b.to(x.device, output.dtype))
                            * scale
                        )
                    return output

                module.forward = forward
            else:
                # Move only direct non-Linear state; never move a complete block.
                for parameter in module.parameters(recurse=False):
                    parameter.data = parameter.data.to(device)
                for key, value in module.named_buffers(recurse=False):
                    module._buffers[key] = value.to(device)
        with torch.autograd.graph.save_on_cpu(pin_memory=pin_memory):
            yield
    finally:
        if weight_cache is not None:
            weight_cache.clear()
        for module, forward in originals:
            module.forward = forward
        model.requires_grad_(False).cpu()
        model.zero_grad(set_to_none=True)
