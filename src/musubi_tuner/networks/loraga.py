"""LoRA-GA factor construction and lossless export for Musubi Linear LoRAs.

Algorithm adapted from Outsider565/LoRA-GA (Apache-2.0),
peft/src/peft/tuners/lora/layer.py:lora_ga_init. No PEFT dependency.
"""

import math

import torch


def initialize_factors(gradient, rank, scaling, direction="ArB2r", scale="stable", gamma=16.0, device="cpu"):
    """Return A [rank,in], B [out,rank], using the reference randomized SVD."""
    if gradient.ndim != 2 or rank <= 0 or 2 * rank > min(gradient.shape):
        raise ValueError("LoRA-GA requires a matrix with min(in,out) >= 2 * rank")
    if not math.isfinite(scaling) or scaling <= 0 or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("LoRA-GA scaling and stable_gamma must be finite and positive")
    gradient = gradient.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(gradient).all() or not gradient.count_nonzero():
        raise ValueError("LoRA-GA gradient must be finite and nonzero")
    # A float32 input alone does not guarantee float32 randomized-SVD products:
    # an outer autocast context can downcast them (or make CUDA QR fail).
    # TF32 is useful for BF16 gradient operands, but not for the FP32 SVD basis.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if gradient.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            u, _, v = torch.svd_lowrank(gradient, q=min(4 * rank, min(gradient.shape)), niter=4)
    finally:
        if gradient.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    if direction == "ArB2r":
        b, a = u[:, rank : 2 * rank], v[:, :rank].T
    elif direction == "A2rBr":
        b, a = u[:, :rank], v[:, rank : 2 * rank].T
    elif direction == "ArBr":
        b, a = u[:, : 2 * rank : 2], v[:, 1 : 2 * rank : 2].T
    else:
        raise ValueError(f"Unsupported LoRA-GA direction: {direction}")
    if scale == "stable":
        factor = gradient.shape[0] ** 0.25 / gamma**0.5
    elif scale == "gd":
        factor = 1 / scaling
    elif scale == "unit":
        factor = 1.0
    else:
        raise ValueError(f"Unsupported LoRA-GA scale: {scale}")
    return (a * factor).to("cpu").contiguous(), (b * factor).to("cpu").contiguous()


def converted_state_dict(state, initial):
    """Represent s(B A - B0 A0) exactly at rank 2r, preserving alpha/rank."""
    result = dict(state)
    for name, (a0, b0) in initial.items():
        ak, bk, alpha = f"{name}.lora_down.weight", f"{name}.lora_up.weight", f"{name}.alpha"
        a, b = state[ak], state[bk]
        if a.shape != a0.shape or b.shape != b0.shape:
            raise ValueError(f"LoRA-GA export shape mismatch: {name}")
        result[ak] = torch.cat((a, a0.to(a)), dim=0).contiguous()
        result[bk] = torch.cat((b, -b0.to(b)), dim=1).contiguous()
        result[alpha] = state[alpha] * 2
    return result
