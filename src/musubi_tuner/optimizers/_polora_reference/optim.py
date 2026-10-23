"""PoLoRA — spectral-preconditioned LoRA optimizer.

Polora adapts each LoRA factor pair ``(A, B)`` (PEFT convention: ``A`` is
``(r, d_in)``, ``B`` is ``(d_out, r)``, adapter output ``(alpha/r) * B @ A``)
with a preconditioned polar step in a diagonal-Kronecker curvature metric:

  1. Build the ``r x r`` matrices ``C_A = A Q A^T`` and ``C_B = B^T P B``
     from damped, max-normalized versions of the diagonal preconditioners
     ``Q`` / ``P``.
  2. Whiten the Nesterov-corrected momentum by the inverse square root of
     ``C_B`` (resp. ``C_A``), computed eigh-free by Gram Newton-Schulz, apply
     the matrix sign, then unwhiten.
  3. Rescale each factor update to spectral norm
     ``rho = lr / (sigma_max(A) + sigma_max(B))``.

The preconditioners are coupled across the two sides and kept diagonal.
Stored state per pair is the first moment plus two diagonal vectors (``Q`` of
length ``d_in``, ``P`` of length ``d_out``) — about half of Adam's state; the
``r x r`` matrices ``C_A``/``C_B`` are rebuilt from the weights each step, not stored.

``Polora(model, lr=...)`` constructs directly from a PEFT-wrapped model;
non-PEFT use is supported via ``Polora(pairs=[(A, B), ...], lr=...)``.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.optim import Optimizer

from .utils import polar_express_gram_batched, ns_inv_sqrt, power_iter_top


def collect_lora_pairs(model):
    """Discovers the trainable LoRA ``(A, B)`` weight tensors on a PEFT model.

    Pairs whose factors are frozen (``requires_grad=False``) are skipped, so a
    frozen adapter sharing the model (e.g. a DPO reference adapter) is not
    collected.

    Args:
        model: A PEFT-wrapped model (its modules carry ``lora_A`` / ``lora_B``).

    Returns:
        List of ``(A, B)`` parameter pairs — ``A`` is ``(r, in)``, ``B`` is
        ``(out, r)`` — in module-discovery order.
    """
    pairs = []
    for _, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        lora_A, lora_B = mod.lora_A, mod.lora_B
        if hasattr(lora_A, "keys"):  # ModuleDict of adapters (standard PEFT)
            for k in lora_A:
                if k in lora_B:
                    pairs.append((lora_A[k].weight, lora_B[k].weight))
        elif hasattr(lora_A, "weight") and hasattr(lora_B, "weight"):
            pairs.append((lora_A.weight, lora_B.weight))
    return [(A, B) for A, B in pairs if A.requires_grad and B.requires_grad]


class Polora(Optimizer):
    """Spectral-preconditioned LoRA optimizer.

    Each LoRA pair is whitened in a diagonal-Kronecker curvature metric, passed
    through the matrix sign, unwhitened, and rescaled to spectral norm
    ``rho``. See the module docstring for the full update.

    Construct after the model is on its target device; optimizer state is
    allocated on the params' current device. ``state_dict()`` does not include
    the momentum/preconditioner state, so saving and restoring the optimizer
    across runs is not supported.

    Args:
        model: PEFT-wrapped model; ``(A, B)`` adapter pairs are discovered
            automatically. Provide either ``model`` or ``pairs``.
            (Default: None)
        lr: Learning rate; each factor update is rescaled to spectral norm
            ``rho = lr / (sigma_max(A) + sigma_max(B))``. (Default: 2e-4)
        beta1: Momentum coefficient. (Default: 0.9)
        epsilon: Numerical floor for the preconditioner initialization, the
            damping of the ``C_A``/``C_B`` inverse square roots, and the
            spectral-norm normalizations. (Default: 1e-12)
        delta: Relative damping for the ``C_A``/``C_B`` inverse square roots
            and for the normalized ``Q``/``P`` inverse square roots.
            (Default: 1e-4)
        curvature_beta: EMA coefficient for the diagonal preconditioners
            ``Q``/``P``. (Default: 0.99)
        ns_steps: Iterations of the PolarExpress matrix-sign solver. (Default: 8)
        higham_iters: Newton-Schulz iterations for the ``C_A``/``C_B``
            inverse square roots. (Default: 8)
        compile: If True, wraps the two spectral kernels in ``torch.compile``;
            the update is unchanged. (Default: False)
        pairs: Explicit ``[(A, B), ...]`` parameter list for non-PEFT use.
            (Default: None)
    """

    def __init__(self, model=None, lr=2e-4, beta1=0.9, epsilon=1e-12,
                 delta=1e-4, curvature_beta=0.99, ns_steps=8, higham_iters=8,
                 compile=False, *, pairs=None):
        if pairs is None:
            if model is None:
                raise ValueError("Polora requires either `model` (PEFT) or `pairs=[(A,B),...]`.")
            pairs = collect_lora_pairs(model)
        else:
            pairs = list(pairs)
        if not pairs:
            raise ValueError("No trainable LoRA (A,B) tensors found on model.")
        params = [p for A, B in pairs for p in (A, B)]
        super().__init__([{"params": params, "lr": lr}], {})
        self.pairs = pairs
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.beta1 = beta1
        self.curvature_beta = float(curvature_beta)
        self.ns_steps = int(ns_steps)
        self.higham_iters = int(higham_iters)
        # dynamic=True lets the LoRA shape groups share one compiled kernel.
        if compile:
            self._polar_fn = torch.compile(
                polar_express_gram_batched, dynamic=True, fullgraph=False)
            self._invsqrt_fn = torch.compile(
                ns_inv_sqrt, dynamic=True, fullgraph=False)
        else:
            self._polar_fn = polar_express_gram_batched
            self._invsqrt_fn = ns_inv_sqrt

        self.pair_state = {}
        for i, (A, B) in enumerate(pairs):
            r, d_in = A.shape
            d_out = B.shape[0]
            self.pair_state[i] = {
                'M_A': torch.zeros_like(A, dtype=torch.float32),
                'M_B': torch.zeros_like(B, dtype=torch.float32),
                'Q': torch.full((d_in,), self.epsilon, dtype=torch.float32, device=A.device),
                'P': torch.full((d_out,), self.epsilon, dtype=torch.float32, device=B.device),
            }

    @staticmethod
    def _sym(M):
        return 0.5 * (M + M.transpose(-2, -1))

    def _rdinv(self, x):
        """Relative-damped inverse square root along the last dim:
        ``(x / x_max + delta)^{-1/2}``. ``x_max > 0`` from step 1 because the
        preconditioner EMA is seeded at ``epsilon``."""
        xmax = x.amax(dim=-1, keepdim=True)
        return (x / xmax + self.delta).rsqrt()

    def _smax_warm(self, M, states, key):
        """Batched ``sigma_max`` with per-pair warm-start caching, floored at
        ``max(row L2, col L2)`` — a lower bound on ``sigma_max`` — so a cold
        or degenerate start vector cannot under-estimate the top value."""
        cached = [st.get(key) for st in states]
        vi = torch.stack(cached) if all(c is not None for c in cached) else None
        s, v = power_iter_top(M, v_init=vi, n_iters=8)
        for j, st in enumerate(states):
            st[key] = v[j].detach()
        with torch.no_grad():
            Mf = M.detach().float()
            rn = Mf.pow(2).sum(dim=-1).amax(dim=-1).sqrt()
            cn = Mf.pow(2).sum(dim=-2).amax(dim=-1).sqrt()
            s = torch.maximum(s, torch.maximum(rn, cn).reshape(s.shape).to(s.dtype))
        return s

    def _polar_batched(self, X):
        """Matrix sign via PolarExpress (Frobenius-prescaled internally)."""
        return self._polar_fn(X, nsteps=self.ns_steps)

    @torch.no_grad()
    def step(self, closure=None):
        """Applies one Polora update to every LoRA pair, then zeros their grads.

        Gradients on all ``(A, B)`` factors must already be populated; raises
        ``ValueError`` otherwise. See the module docstring for the update.

        Args:
            closure: Callable that re-evaluates the model and returns the loss
                (standard ``torch.optim`` protocol); called under grad.
                (Default: None)

        Returns:
            The loss returned by ``closure``, or None if no closure is given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lr = self.param_groups[0]["lr"]
        cb = self.curvature_beta
        b1, eps = self.beta1, self.epsilon
        pairs = self.pairs
        S = self.pair_state

        for A, B in pairs:
            if A.grad is None or B.grad is None:
                raise ValueError("Gradients are required for Polora update.")

        # One batched bmm / Newton-Schulz / sigma_max per (r, d_in, d_out, device)
        # shape group; rank and device must match for torch.stack.
        groups = defaultdict(list)
        for i, (A, B) in enumerate(pairs):
            groups[(A.shape[0], A.shape[1], B.shape[0], A.device)].append(i)
        for idxs in groups.values():
            G_A = torch.stack([pairs[i][0].grad.float() for i in idxs])
            G_B = torch.stack([pairs[i][1].grad.float() for i in idxs])
            Aw = torch.stack([pairs[i][0].detach().float() for i in idxs])
            Bw = torch.stack([pairs[i][1].detach().float() for i in idxs])
            M_A = torch.stack([S[i]['M_A'] for i in idxs]).mul_(b1).add_(G_A, alpha=1.0 - b1)
            M_B = torch.stack([S[i]['M_B'] for i in idxs]).mul_(b1).add_(G_B, alpha=1.0 - b1)
            Q = torch.stack([S[i]['Q'] for i in idxs])         # d_in metric diagonal
            P = torch.stack([S[i]['P'] for i in idxs])         # d_out metric diagonal

            Q_isqrt = self._rdinv(Q)
            P_isqrt = self._rdinv(P)
            Q_dmp = (Q_isqrt * Q_isqrt).reciprocal()
            P_dmp = (P_isqrt * P_isqrt).reciprocal()
            # Curvature factors, rebuilt each step (not stored).
            C_B = Bw.transpose(-2, -1) @ (P_dmp.unsqueeze(-1) * Bw)
            C_A = (Aw * Q_dmp.unsqueeze(1)) @ Aw.transpose(-2, -1)
            C_B_isqrt = self._invsqrt_fn(self._sym(C_B), nsteps=self.higham_iters,
                                         eps=self.delta, eps_relative=True,
                                         floor=self.epsilon)
            C_A_isqrt = self._invsqrt_fn(self._sym(C_A), nsteps=self.higham_iters,
                                         eps=self.delta, eps_relative=True,
                                         floor=self.epsilon)
            C_B_inv = C_B_isqrt @ C_B_isqrt
            C_A_inv = C_A_isqrt @ C_A_isqrt

            # Nesterov look-ahead.
            M_hat_A = M_A.mul(b1).add(G_A, alpha=1.0 - b1)
            M_hat_B = M_B.mul(b1).add(G_B, alpha=1.0 - b1)

            grp = [S[i] for i in idxs]
            sigma_A = self._smax_warm(Aw, grp, 'v_sigma_A')
            sigma_B = self._smax_warm(Bw, grp, 'v_sigma_B')
            rho = lr / (sigma_B + sigma_A)

            # Whiten -> polar -> unwhiten.
            zA = (C_B_isqrt @ M_hat_A) * Q_isqrt.unsqueeze(1)
            zB = (M_hat_B @ C_A_isqrt) * P_isqrt.unsqueeze(-1)
            zA = self._polar_batched(zA)
            zB = self._polar_batched(zB)
            D_A = (C_B_isqrt @ zA) * Q_isqrt.unsqueeze(1)
            D_B = (zB @ C_A_isqrt) * P_isqrt.unsqueeze(-1)
            sigma_DA = self._smax_warm(D_A, grp, 'v_sigma_DA')
            sigma_DB = self._smax_warm(D_B, grp, 'v_sigma_DB')
            dA = -(rho / sigma_DA.clamp_min(eps)).view(-1, 1, 1) * D_A
            dB = -(rho / sigma_DB.clamp_min(eps)).view(-1, 1, 1) * D_B

            # Preconditioner update: each diagonal accumulates the gradient
            # whitened by the inverse of the other side's r x r matrix.
            r = G_A.shape[1]
            Q.mul_(cb).add_((G_A * (C_B_inv @ G_A)).sum(dim=1), alpha=(1.0 - cb) / r)
            P.mul_(cb).add_((G_B * (G_B @ C_A_inv)).sum(dim=2), alpha=(1.0 - cb) / r)

            for j, i in enumerate(idxs):
                A_, B_ = pairs[i]
                A_.add_(dA[j].to(dtype=A_.dtype, device=A_.device))
                B_.add_(dB[j].to(dtype=B_.dtype, device=B_.device))
                S[i]['M_A'].copy_(M_A[j])
                S[i]['M_B'].copy_(M_B[j])
                S[i]['Q'].copy_(Q[j])
                S[i]['P'].copy_(P[j])
                A_.grad.zero_()
                B_.grad.zero_()

        return loss
