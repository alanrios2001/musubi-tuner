"""LoRA+ optimizer grouping with reference LoRA-GA initialization."""

import pytest
import torch

from musubi_tuner.flux_2_train_network import flux2_setup_parser
from musubi_tuner.networks import lora_flux_2
from musubi_tuner.networks.loraga import initialize_factors
from musubi_tuner.training.flux2_loraga import loraplus_lr_ratio, validate_args, compensate, quantize_model
from musubi_tuner.training.parser_common import setup_parser_common
from test_loraga import cached_training_fixture, tiny_model, ProbeAccelerator, model_inputs


def plus_args(network_args):
    args = flux2_setup_parser(setup_parser_common()).parse_args([])
    args.loraga = True
    args.model_version = "klein-base-4b"
    args.mixed_precision = "bf16"
    args.network_module = "networks.lora_flux_2"
    args.network_dim = 64
    args.max_data_loader_n_workers = 0
    args.network_args = network_args
    return args


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "1e999", "", "invalid"])
def test_invalid_plus_ratio_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        validate_args(plus_args([f"loraplus_lr_ratio={value}"]))


@pytest.mark.parametrize("network_args", [
    ["rank_dropout=0.1"], ["loraplus_lr_ratio"], ["loraplus_lr_ratio=2", "verbose=True"],
    ["loraplus_lr_ratio=2", "loraplus_lr_ratio=4"],
])
def test_only_one_plus_network_option_is_allowed(network_args):
    with pytest.raises(ValueError, match="only supports|Duplicate"):
        validate_args(plus_args(network_args))


@pytest.mark.parametrize("ratio", [None, 1.0, 2.0, 4.0])
def test_valid_plus_ratio_and_conventional_default(ratio):
    args = plus_args(None if ratio is None else [f"loraplus_lr_ratio={ratio}"])
    validate_args(args)
    assert loraplus_lr_ratio(args) == ratio
    # GA validation must not narrow conventional networks' kwargs.
    args.loraga = False
    args.network_args = ["rank_dropout=0.1"]
    validate_args(args)


def test_native_phase_b_uses_distinct_lrs_and_keeps_compensated_fp8_frozen(tmp_path):
    trainer, args = cached_training_fixture(tmp_path)
    args.network_args = ["loraplus_lr_ratio=4"]
    args.learning_rate = 5e-5
    validate_args(args)
    model = tiny_model()
    prototype = lora_flux_2.create_arch_network(1, 2, 2, None, None, model)
    initial = {
        module.lora_name: initialize_factors(torch.randn_like(module.org_module.weight.float()), 2, 1)
        for module in prototype.unet_loras
    }
    targets = {module.lora_name: module.org_module for module in prototype.unet_loras}
    compensate(model, initial, targets, 1, False, 2)
    quantize_model(model)
    model.requires_grad_(False)
    trainer._loraga_initial = initial
    accelerator = ProbeAccelerator()
    accelerator.print = print
    network = trainer._build_network(args, accelerator, model, None, torch.bfloat16)
    assert network.loraga_initial is initial
    for module in network.unet_loras:
        torch.testing.assert_close(module.lora_down.weight, initial[module.lora_name][0], rtol=0, atol=0)
        torch.testing.assert_close(module.lora_up.weight, initial[module.lora_name][1], rtol=0, atol=0)
    groups, descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
    assert descriptions == ["unet", "unet plus"]
    assert [group["lr"] for group in groups] == [5e-5, 2e-4]
    down_ids = {id(module.lora_down.weight) for module in network.unet_loras}
    up_ids = {id(module.lora_up.weight) for module in network.unet_loras}
    assert {id(parameter) for parameter in groups[0]["params"]} == down_ids
    assert {id(parameter) for parameter in groups[1]["params"]} == up_ids
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(**model_inputs("cpu")).float().square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in network.parameters())
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[1] / scheduler.get_last_lr()[0] == 4
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.float(), frozen[key].float(), rtol=0, atol=0)


def test_unequal_rates_change_ga_first_order_gradient_balance():
    # A concrete counterexample to assuming that LoRA+ always improves GA's alignment.
    # In ArB2r, the right-projection term covers singular direction 1, the left term 2.
    gradient = torch.diag(torch.tensor([4.0, 2.0]))
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0], [1.0]])
    equal_rate_tangent = gradient @ a.T @ a + b @ b.T @ gradient
    plus_tangent = 4 * gradient @ a.T @ a + b @ b.T @ gradient
    torch.testing.assert_close(equal_rate_tangent, gradient, rtol=0, atol=0)
    # Even the best overall rescaling cannot remove the unequal weighting.
    best_scale = (plus_tangent * gradient).sum() / plus_tangent.square().sum()
    assert torch.linalg.vector_norm(best_scale * plus_tangent - gradient) > 0
