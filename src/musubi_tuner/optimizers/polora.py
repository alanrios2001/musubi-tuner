"""Musubi adapter for the pinned, unmodified Apache-2.0 PoLoRA algorithm."""

import math
from collections import defaultdict

import torch

from ._polora_reference.optim import Polora as ReferencePolora


def validate_args(args):
    if args.optimizer_type.lower() != "polora":
        return
    if args.network_module not in ("networks.lora_flux_2", "musubi_tuner.networks.lora_flux_2"):
        raise ValueError("PoLoRA currently supports networks.lora_flux_2 only")
    if args.network_args:
        raise ValueError("PoLoRA does not support network_args, including LoRA+; remove --network_args")
    if getattr(args, "network_dropout", None):
        raise ValueError("PoLoRA currently requires LoRA without dropout")
    alpha = getattr(args, "network_alpha", None)
    if alpha not in (None, 0, args.network_dim):
        raise ValueError("PoLoRA currently requires network_alpha == network_dim (effective scale 1)")
    if getattr(args, "max_data_loader_n_workers", 0) != 0:
        raise ValueError("PoLoRA deterministic resume currently requires --max_data_loader_n_workers 0")


def create_optimizer(network, trainable_params, lr, **kwargs):
    if network is None:
        raise ValueError("PoLoRA requires the Flux2 trainer's LoRA network")
    pairs, names = [], []
    for module in network.unet_loras:
        if not isinstance(module.lora_down, torch.nn.Linear) or not isinstance(module.lora_up, torch.nn.Linear):
            raise ValueError("PoLoRA supports unsplit Linear LoRA factors only")
        if module.scale != 1 or any(getattr(module, key, None) for key in ("dropout", "rank_dropout", "module_dropout")):
            raise ValueError("PoLoRA requires scale 1 and no dropout")
        pairs.append((module.lora_down.weight, module.lora_up.weight))
        names.append(module.lora_name)
    expected = [p for group in trainable_params for p in (group["params"] if isinstance(group, dict) else [group])]
    actual = [p for pair in pairs for p in pair]
    if len(expected) != len(actual) or {id(p) for p in expected} != {id(p) for p in actual}:
        raise ValueError("PoLoRA requires exactly the paired LoRA factors, without extra trainable parameters")
    if any(isinstance(group, dict) and group.get("lr", lr) != lr for group in trainable_params):
        raise ValueError("PoLoRA requires one shared learning rate; LoRA+ is unsupported")
    return Polora(pairs, names=names, lr=lr, **kwargs)


class Polora(ReferencePolora):
    """Reference updates with bounded workspace, device migration and full state IO.

    Each pair is independent. Chunking only limits the reference's batched
    temporaries; it does not accumulate or approximate updates across pairs.
    """

    def __init__(self, pairs, *, names=None, pair_batch_size=4, lr=2e-4, beta1=0.9,
                 epsilon=1e-12, delta=1e-4, curvature_beta=0.99, ns_steps=8,
                 higham_iters=8, compile=False):
        if isinstance(pair_batch_size, bool) or not isinstance(pair_batch_size, int) or pair_batch_size < 1:
            raise ValueError("pair_batch_size must be a positive integer")
        for key, value in {"lr": lr, "epsilon": epsilon, "delta": delta}.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"PoLoRA {key} must be finite and positive")
        if not 0 <= beta1 < 1 or not 0 <= curvature_beta < 1:
            raise ValueError("PoLoRA EMA coefficients must be in [0, 1)")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (ns_steps, higham_iters)):
            raise ValueError("PoLoRA solver iteration counts must be positive integers")
        pairs = list(pairs)
        for a, b in pairs:
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.device != b.device:
                raise ValueError("PoLoRA expects A[rank,in], B[out,rank] on the same device")
            if not a.requires_grad or not b.requires_grad:
                raise ValueError("PoLoRA factors must be trainable")
        self.pair_batch_size = pair_batch_size
        self.completed_steps = 0
        self.pair_names = list(names) if names is not None else [str(i) for i in range(len(pairs))]
        if len(self.pair_names) != len(pairs) or len(set(self.pair_names)) != len(pairs):
            raise ValueError("PoLoRA pair names must be unique and match pairs")
        self.config = dict(beta1=beta1, epsilon=epsilon, delta=delta, curvature_beta=curvature_beta,
                           ns_steps=ns_steps, higham_iters=higham_iters, compile=compile,
                           pair_batch_size=pair_batch_size)
        super().__init__(pairs=pairs, lr=lr, beta1=beta1, epsilon=epsilon, delta=delta,
                         curvature_beta=curvature_beta, ns_steps=ns_steps,
                         higham_iters=higham_iters, compile=compile)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if any(p.grad is None or p.grad.is_sparse for pair in self.pairs for p in pair):
            raise ValueError("PoLoRA requires dense gradients for every LoRA factor")
        groups = defaultdict(list)
        pairs, states = self.pairs, self.pair_state
        for i, (a, b) in enumerate(pairs):
            groups[(a.shape, b.shape, a.device)].append(i)
            # Musubi constructs optimizers before moving the adapter to CUDA.
            states[i] = {key: value.to(device=a.device, dtype=torch.float32) for key, value in states[i].items()}
        try:
            for indices in groups.values():
                for start in range(0, len(indices), self.pair_batch_size):
                    chunk = indices[start:start + self.pair_batch_size]
                    self.pairs = [pairs[i] for i in chunk]
                    self.pair_state = {j: states[i] for j, i in enumerate(chunk)}
                    with torch.autocast(device_type=self.pairs[0][0].device.type, enabled=False):
                        super().step()
        finally:
            self.pairs, self.pair_state = pairs, states
        self.completed_steps += 1
        return loss

    def _identity(self):
        return [(name, list(a.shape), list(b.shape)) for name, (a, b) in zip(self.pair_names, self.pairs)]

    def state_dict(self):
        state = super().state_dict()
        state["polora"] = dict(version=1, steps=self.completed_steps, identity=self._identity(), config=self.config,
                               pairs=[self.pair_state[i] for i in range(len(self.pairs))])
        return state

    def load_state_dict(self, state_dict):
        payload = state_dict.get("polora")
        if not payload or payload.get("version") != 1:
            raise ValueError("Missing or unsupported PoLoRA optimizer state; upstream incomplete states cannot resume")
        if payload["identity"] != self._identity() or payload["config"] != self.config:
            raise ValueError("PoLoRA checkpoint pair/configuration mismatch")
        if len(payload["pairs"]) != len(self.pairs):
            raise ValueError("PoLoRA checkpoint state count mismatch")
        steps = payload.get("steps")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("PoLoRA checkpoint has invalid step metadata")
        restored = {}
        for i, ((a, b), saved) in enumerate(zip(self.pairs, payload["pairs"])):
            shapes = {"M_A": a.shape, "M_B": b.shape, "Q": (a.shape[1],), "P": (b.shape[0],)}
            shapes.update({"v_sigma_A": (min(a.shape),), "v_sigma_B": (min(b.shape),),
                           "v_sigma_DA": (min(a.shape),), "v_sigma_DB": (min(b.shape),)})
            required = set(shapes) if steps else {"M_A", "M_B", "Q", "P"}
            if not required.issubset(saved) or set(saved) - shapes.keys():
                raise ValueError("PoLoRA checkpoint has invalid state keys")
            restored[i] = {}
            for key, value in saved.items():
                if tuple(value.shape) != tuple(shapes[key]) or value.dtype != torch.float32:
                    raise ValueError(f"PoLoRA checkpoint invalid {key} shape/dtype")
                restored[i][key] = value.detach().to(device=a.device, dtype=torch.float32).clone()
        super().load_state_dict({key: value for key, value in state_dict.items() if key != "polora"})
        self.pair_state = restored
        self.completed_steps = steps
