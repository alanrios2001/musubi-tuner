"""Convert raw Accelerate LoRA-GA rank-r weights to a standard rank-2r adapter."""

import argparse
import json

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.networks.loraga import converted_state_dict
from musubi_tuner.training.flux2_loraga import load_cache


def export(weights, initialization, output):
    with safe_open(initialization, framework="pt", device="cpu") as file:
        identity = json.loads(file.metadata()["identity"])
    initial, _ = load_cache(initialization, identity)
    with safe_open(weights, framework="pt", device="cpu") as file:
        metadata = dict(file.metadata() or {})
    if "ss_loraga_export" in metadata:
        raise ValueError("This adapter is already converted; use raw Accelerate model.safetensors")
    state = load_file(weights)
    expected = {f"{name}.{suffix}" for name in initial for suffix in ("lora_down.weight", "lora_up.weight", "alpha")}
    if set(state) != expected:
        raise ValueError("Raw adapter modules do not match initialization cache")
    for name in initial:
        if float(state[f"{name}.alpha"]) != identity["alpha"]:
            raise ValueError(f"Raw adapter alpha differs from initialization: {name}")
    converted = converted_state_dict(state, initial)
    metadata.update({"ss_network_module": "networks.lora_flux_2", "ss_network_dim": str(identity["rank"] * 2),
                     "ss_network_alpha": str(identity["alpha"] * 2), "ss_loraga_export": "exact_difference_rank_2r"})
    save_file(converted, output, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Raw rank-r Accelerate model.safetensors")
    parser.add_argument("--init", required=True, help="Matching loraga_init.safetensors")
    parser.add_argument("--output", required=True, help="Standard inference adapter safetensors")
    args = parser.parse_args()
    export(args.weights, args.init, args.output)


if __name__ == "__main__":
    main()
