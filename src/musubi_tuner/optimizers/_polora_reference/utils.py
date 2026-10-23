"""Eigh-free matrix kernels for the Polora optimizer.

  - ``power_iter_top`` — top singular value / eigenvalue via guarded power
    iteration.
  - ``polar_express_gram_batched`` — polar factor via the PolarExpress
    iteration (Amsel et al., arXiv:2505.16932) on the ``r x r`` Gram.
  - ``ns_inv_sqrt`` — batched SPD inverse square root using the same
    coefficient schedule.
"""
from __future__ import annotations

import torch

__all__ = ["power_iter_top", "polar_express_gram_batched", "ns_inv_sqrt"]


def _normalize_or_zero(X, dim):
    """Normalizes over ``dim``, with zero-norm inputs mapped to zero."""
    norm = X.norm(dim=dim, keepdim=True)
    is_zero = norm == 0
    safe_norm = torch.where(is_zero, torch.ones_like(norm), norm)
    return torch.where(is_zero, torch.zeros_like(X), X / safe_norm)


def power_iter_top(M, symmetric=False, v_init=None, n_iters=8):
    """Computes the largest singular value (or eigenvalue) of ``M`` by guarded
    power iteration, batched over the leading dims.

    Args:
        M: Matrix batch ``(..., m, n)``.
        symmetric: If True, treats ``M`` as symmetric PSD and returns its top
            eigenvalue (iterates ``M v``; value is the Rayleigh quotient
            ``vᵀ M v``). If False, returns the top singular value ``||M||_2``
            (iterates the smaller-side Gram; value is ``||Mᵀ v||``).
            (Default: False)
        v_init: Warm-start vector ``(..., k)`` from a previous call. If None,
            zero, or nonfinite, starts from ``M @ 1``. (Default: None)
        n_iters: Number of power-iteration steps. (Default: 8)

    Returns:
        ``(value, v)``. ``value`` ``(...)`` is floored at the iterated side's
        largest row/col L2 norm — a lower bound on the top value — so a cold
        start cannot under-estimate it. ``v`` ``(..., k)`` is the converged
        top vector, reusable as ``v_init``.
    """
    if M.numel() == 0:
        return torch.zeros(M.shape[:-2], device=M.device, dtype=torch.float32), None
    Mf = M.float() if M.dtype != torch.float32 else M
    if symmetric:
        Mf = 0.5 * (Mf + Mf.transpose(-2, -1))
    Mt = Mf.transpose(-2, -1)
    *batch, m, n = Mf.shape
    left = symmetric or m <= n                 # iterate in the smaller (m-dim) space
    k = m if left else n

    if left:
        v_ones = (Mf @ torch.ones(*batch, n, 1, device=Mf.device, dtype=torch.float32)).squeeze(-1)
        scores = Mf.square().sum(dim=-1)       # row l2^2
    else:
        v_ones = (Mt @ torch.ones(*batch, m, 1, device=Mf.device, dtype=torch.float32)).squeeze(-1)
        scores = Mf.square().sum(dim=-2)       # col l2^2
    floor = scores.amax(dim=-1).sqrt()     # <= the top value

    v_fallback = v_ones                                   # cold start M @ 1; the floor covers a vanishing start

    # Kept in column form (..., k, 1) to avoid per-iter reshapes.
    if v_init is None:
        v = v_fallback
    else:
        v_norm = v_init.norm(dim=-1, keepdim=True)
        restart = (~torch.isfinite(v_norm)) | (v_norm == 0)
        v = torch.where(restart, v_fallback, v_init)
    v = _normalize_or_zero(v, dim=-1).unsqueeze(-1)

    for _ in range(n_iters):
        if symmetric:
            v = Mf @ v
        elif left:
            v = Mf @ (Mt @ v)
        else:
            v = Mt @ (Mf @ v)
        v = _normalize_or_zero(v, dim=-2)

    if symmetric:
        value = (v * (Mf @ v)).sum(dim=(-2, -1))      # Rayleigh quotient
    elif left:
        value = (Mt @ v).norm(dim=(-2, -1))
    else:
        value = (Mf @ v).norm(dim=(-2, -1))
    return torch.maximum(value, floor), v.squeeze(-1)


# PolarExpress per-iteration quintic fits (a, b, c) for p(x) = a*x + b*x^3 + c*x^5
# (Amsel et al., arXiv:2505.16932), raw (pre-safety), from the reference
# optimal_composition Remez fit at worst-case conditioning l=1e-3, cushion 0.02.
_POLAR_EXPRESS_RAW = (
    (8.31968561540051, -23.85945031896673, 17.53144504181025),
    (4.123266419055485, -2.980709974760902, 0.5520797360741728),
    (3.9656114721772022, -2.941248486368672, 0.5589295013119786),
    (3.3312009004415994, -2.4980080275937966, 0.5111272954169234),
    (2.320007312889811, -1.6862169729967622, 0.42068027340235137),
    (1.8951443404954809, -1.2722050191923813, 0.377227344813122),
    (1.875006772051659, -1.250007524486259, 0.37500075245392533),
    (1.8750008025096776, -1.2500016050034328, 0.375000802493755),
    (1.8749954784656357, -1.249990956953079, 0.374995478487443),
    (1.8749954775662068, -1.2499909551542292, 0.3749954775880226),
)
_SAFETY = 1.01


def _polar_coeffs(nsteps):
    """First ``nsteps`` coefficient triples of the composed PolarExpress
    schedule, matching the reference generator at ``num_iters=nsteps``: the
    safety margin divides every iteration except the last, and iterations
    beyond the table repeat the asymptotic quintic."""
    take = _POLAR_EXPRESS_RAW[:nsteps]
    if len(take) < nsteps:
        take = take + (_POLAR_EXPRESS_RAW[-1],) * (nsteps - len(take))
    s, s3, s5 = _SAFETY, _SAFETY ** 3, _SAFETY ** 5
    return tuple((a, b, c) if i == nsteps - 1 else (a / s, b / s3, c / s5)
                 for i, (a, b, c) in enumerate(take))


def polar_express_gram_batched(X, nsteps=8):
    """Computes the polar factor of ``X`` (the orthogonal part ``U`` of the
    polar decomposition ``X = U H``) batchwise, without an SVD.

    Runs the Gram form of the PolarExpress iteration (Dao 2026,
    tridao.me/blog/2026/gram-newton-schulz) with the quintic Remez
    coefficients of Amsel et al. (arXiv:2505.16932): each step applies the
    degree-5 polynomial ``M_t = a_t I + b_t R + c_t R^2`` to the ``r x r``
    Gram ``R = X Xᵀ`` while accumulating ``Q`` so that ``Q @ X -> polar(X)``,
    keeping every iterate ``r x r`` regardless of the long dimension. ``X`` is
    prescaled by its Frobenius norm (an upper bound on ``sigma_max``) so the
    initial top singular value is at most 1; the polar map is scale-invariant,
    so the prescale does not change the result.

    Args:
        X: Matrix batch ``(..., m, n)``. Tall inputs are transposed internally
            and transposed back on return.
        nsteps: Number of polynomial iterations. (Default: 8)

    Returns:
        The polar factor of ``X`` in float32, same shape as ``X``.
    """
    if X.shape[-2] == 0 or X.shape[-1] == 0:
        return X
    X = X.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-2, -1)
    orig_leading = X.shape[:-2]
    X = X.reshape(-1, X.shape[-2], X.shape[-1])
    r = X.shape[-2]
    X_normed = _normalize_or_zero(X, dim=(-2, -1))

    coeffs = _polar_coeffs(nsteps)

    # Disable TF32 so the fp32 Gram iteration keeps full precision.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if X.is_cuda else None
    if X.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        eye = torch.eye(r, dtype=X_normed.dtype, device=X_normed.device)
        R = X_normed @ X_normed.transpose(-2, -1)
        Q = eye.expand(*X_normed.shape[:-2], r, r).clone()
        for a, b, c in coeffs:
            R2 = R @ R
            M = b * R + c * R2
            M.diagonal(dim1=-2, dim2=-1).add_(a)
            Q = M @ Q
            R = M @ R @ M
    finally:
        if X.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    out = (Q @ X_normed).reshape(*orig_leading, r, X_normed.shape[-1])
    return out.transpose(-2, -1) if tall else out


def ns_inv_sqrt(S, nsteps=8, eps=1e-4, eps_relative=False, floor=1e-12):
    """Computes the inverse square root ``(S + delta_eff*I)^{-1/2}`` of a
    symmetric-PSD matrix batch, without an eigendecomposition.

    Runs the coupled Gram Newton-Schulz iteration (Dao 2026,
    tridao.me/blog/2026/gram-newton-schulz, Algorithm 2 corollary) with the
    PolarExpress coefficient schedule: ``R`` is driven to ``I`` while ``Q``
    accumulates the inverse square root. The damped input is prescaled by its
    trace (an upper bound on ``lambda_max``) so the initial top eigenvalue is
    at most 1.

    Args:
        S: Symmetric PSD matrix batch ``(..., r, r)``; symmetrized internally.
        nsteps: Number of Newton-Schulz iterations. (Default: 8)
        eps: Damping added to the diagonal before inverting. Absolute by
            default; scaled by ``lambda_max(S)`` if ``eps_relative=True``.
            (Default: 1e-4)
        eps_relative: If True, damps by ``eps * lambda_max(S)``, which bounds
            the condition number of the damped matrix by ``1 + 1/eps``.
            (Default: False)
        floor: Absolute floor on the relative damping, keeping the damped
            matrix invertible when ``lambda_max(S) = 0``. (Default: 1e-12)

    Returns:
        ``(S + delta_eff*I)^{-1/2}``, same shape and dtype as ``S``.
    """
    S = 0.5 * (S + S.transpose(-2, -1))
    n = S.shape[-1]
    dev, dt = S.device, S.dtype
    eye = torch.eye(n, dtype=dt, device=dev).expand_as(S)
    batch_shape = (*S.shape[:-2], 1, 1)

    if eps_relative:
        lam_max, _ = power_iter_top(S, symmetric=True, n_iters=8)
        eps_eff = (eps * lam_max.reshape(batch_shape)).clamp_min(floor).to(dt)
    else:
        eps_eff = torch.as_tensor(float(eps), device=dev, dtype=dt)
    S_d = S + eps_eff * eye
    tr = S_d.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(batch_shape)
    scale = tr                                 # S_d = scale*R0  =>  S_d^{-1/2} = Q/sqrt(scale)
    R0 = S_d / scale                           # lambda_max(R0) <= 1 since tr >= lambda_max

    coeffs = _polar_coeffs(nsteps)

    R = R0.clone()
    Q = eye.clone()
    use_tf32_guard = (dt == torch.float32) and S.is_cuda
    prev_tf32 = None
    if use_tf32_guard:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for a, b, c in coeffs:
            R2 = R @ R
            M = (b * R) + (c * R2)
            M.diagonal(dim1=-2, dim2=-1).add_(a)
            Q = M @ Q
            R = M @ R @ M
            R = 0.5 * (R + R.transpose(-2, -1))
    finally:
        if use_tf32_guard:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return Q / scale.sqrt()
