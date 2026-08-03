#!/usr/bin/env python3
"""8-rank invariant for global mean losses and autograd-preserving FLOPS gather."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from distributed_utils import (
    all_gather_tensor,
    all_gather_tensor_with_grad,
    barrier,
    destroy_distributed,
    global_normalized_loss,
    initialize_distributed,
    unwrap_model,
    wrap_ddp,
)


def main() -> None:
    context = initialize_distributed()
    device = context.device
    dtype = torch.float32
    core = nn.Linear(3, 2, bias=False, dtype=dtype).to(device)
    with torch.no_grad():
        core.weight.copy_(
            torch.tensor(
                [[0.25, -0.5, 0.75], [1.0, -0.125, 0.375]],
                dtype=dtype,
                device=device,
            )
        )
    model = wrap_ddp(core, context)

    rank_value = float(context.rank + 1)
    local_x = torch.tensor(
        [[rank_value, 0.5, -1.0], [-0.25, rank_value, 0.75]],
        dtype=dtype,
        device=device,
    )
    local_y = torch.tensor(
        [[0.1 * rank_value, -0.2], [0.3, -0.05 * rank_value]],
        dtype=dtype,
        device=device,
    )
    local_mask = torch.zeros((2, 2), dtype=dtype, device=device)
    local_mask.reshape(-1)[: (context.rank % 4) + 1] = 1.0

    prediction = model(local_x)
    local_mse_sum = ((prediction - local_y).square() * local_mask).sum()
    local_count = local_mask.sum()
    global_mean_loss = global_normalized_loss(local_mse_sum, local_count, context)
    global_prediction = all_gather_tensor_with_grad(prediction, context)
    global_flops = torch.mean(torch.abs(global_prediction), dim=0).square().sum()
    loss = global_mean_loss + 0.07 * global_flops
    loss.backward()
    ddp_gradient = unwrap_model(model).weight.grad.detach().clone()

    all_x = all_gather_tensor(local_x, context)
    all_y = all_gather_tensor(local_y, context)
    all_mask = all_gather_tensor(local_mask, context)
    reference = nn.Linear(3, 2, bias=False, dtype=dtype).to(device)
    with torch.no_grad():
        reference.weight.copy_(unwrap_model(model).weight.detach())
    reference_prediction = reference(all_x)
    reference_mean = ((reference_prediction - all_y).square() * all_mask).sum() / all_mask.sum()
    reference_flops = torch.mean(torch.abs(reference_prediction), dim=0).square().sum()
    reference_loss = reference_mean + 0.07 * reference_flops
    reference_loss.backward()
    reference_gradient = reference.weight.grad.detach()

    torch.testing.assert_close(ddp_gradient, reference_gradient, rtol=2e-5, atol=2e-6)
    gathered_gradients = all_gather_tensor(ddp_gradient.unsqueeze(0), context)
    torch.testing.assert_close(
        gathered_gradients,
        ddp_gradient.unsqueeze(0).expand_as(gathered_gradients),
        rtol=2e-5,
        atol=2e-6,
    )
    if context.is_main:
        print(
            json.dumps(
                {
                    "ddp_objective_invariant": "passed",
                    "world_size": context.world_size,
                    "global_count": float(all_mask.sum().cpu()),
                    "max_gradient_error": float((ddp_gradient - reference_gradient).abs().max().cpu()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
