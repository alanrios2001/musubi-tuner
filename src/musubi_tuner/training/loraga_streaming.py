"""Initialization-only CPU weight/activation offload, independent of block swapping."""

from contextlib import contextmanager
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

    def add(self, name, gradient):
        gradient = gradient.detach().to(device="cpu", dtype=torch.float32)
        if name in self.paths:
            gradient.add_(torch.load(self.paths[name], map_location="cpu", weights_only=True))
        else:
            self.paths[name] = self.directory / f"{len(self.paths):05d}.pt"
        torch.save(gradient, self.paths[name])

    def take(self, name, batches):
        if name not in self.paths:
            raise ValueError(f"No gradient collected for LoRA-GA target {name}")
        path = self.paths.pop(name)
        gradient = torch.load(path, map_location="cpu", weights_only=True).div_(batches)
        path.unlink()
        return gradient


class _StreamLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, sink, name):
        ctx.weight, ctx.sink, ctx.name = weight, sink, name
        ctx.input_dtype = x.dtype
        x = x.to(weight.dtype)
        ctx.save_for_backward(x)
        return F.linear(x, weight.to(x.device), None if bias is None else bias.to(x.device))

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            dx = (dy.to(ctx.weight.dtype) @ ctx.weight.to(dy.device)).to(ctx.input_dtype)
        if ctx.sink is not None:
            # FP32 GEMM and CPU accumulation, without attaching a full-sized .grad.
            with torch.autocast(device_type=dy.device.type, enabled=False):
                gradient = dy.reshape(-1, dy.shape[-1]).float().T @ x.reshape(-1, x.shape[-1]).float()
            ctx.sink.add(ctx.name, gradient)
        return dx, None, None, None, None


@contextmanager
def streamed_linears(model, targets, sink, device, weight_transform=None, additions=None):
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
                    output = _StreamLinear.apply(x, weight, m.bias, sink if n is not None else None, n)
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
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            yield
    finally:
        for module, forward in originals:
            module.forward = forward
        model.requires_grad_(False).cpu()
        model.zero_grad(set_to_none=True)
