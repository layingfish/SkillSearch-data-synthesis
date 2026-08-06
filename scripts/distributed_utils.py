"""Small, dependency-light helpers for single-node PyTorch distributed jobs.

The helpers deliberately use ``torchrun`` environment variables rather than a
second launcher abstraction.  A command behaves exactly as before when it is
started normally, and becomes distributed only when ``WORLD_SIZE > 1``.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: object

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(*, cpu: bool = False, timeout_minutes: int = 30) -> DistributedContext:
    """Initialize DDP from ``torchrun`` variables and return the local device."""
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if world_size == 1:
        device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
        return DistributedContext(rank=0, local_rank=0, world_size=1, device=device)

    if cpu or not torch.cuda.is_available():
        backend = "gloo"
        device = torch.device("cpu")
    else:
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside visible CUDA devices={torch.cuda.device_count()}"
            )
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def destroy_distributed(context: DistributedContext) -> None:
    if not context.enabled:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if not context.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def unwrap_model(model):
    return getattr(model, "module", model)


def wrap_ddp(model, context: DistributedContext, *, find_unused_parameters: bool = False):
    if not context.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel

    if getattr(context.device, "type", None) == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)


def global_normalized_loss(local_loss_sum, local_count, context: DistributedContext):
    """Return a DDP-correct globally normalized differentiable scalar.

    DDP averages gradients across ranks.  Scaling a rank's local numerator by
    ``world_size / global_count`` therefore yields the gradient of the desired
    global ``sum(loss) / sum(count)`` objective even when sequence lengths vary
    between ranks.
    """
    import torch
    import torch.distributed as dist

    count = local_count.detach().to(device=local_loss_sum.device, dtype=local_loss_sum.dtype).clone()
    if context.enabled:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return local_loss_sum * (context.world_size / count.clamp_min(1.0))
    return local_loss_sum / count.clamp_min(1.0)


def reduce_sum_count(local_sum, local_count, context: DistributedContext) -> tuple[float, int]:
    """Collect a detached scalar sum/count pair without imposing batch padding."""
    import torch
    import torch.distributed as dist

    values = torch.stack(
        [
            local_sum.detach().to(device=context.device, dtype=torch.float64),
            local_count.detach().to(device=context.device, dtype=torch.float64),
        ]
    )
    if context.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].cpu()), int(values[1].cpu())


def all_gather_object(value: T, context: DistributedContext) -> list[T]:
    if not context.enabled:
        return [value]
    import torch.distributed as dist

    gathered: list[T | None] = [None] * context.world_size
    dist.all_gather_object(gathered, value)
    return [item for item in gathered if item is not None]


def distributed_epoch_batches(
    rows: list[T],
    per_device_batch_size: int,
    context: DistributedContext,
    *,
    seed: int,
    epoch: int,
) -> Iterable[list[T]]:
    """Yield equal-count, deterministic local batches for one global epoch.

    The final global batch is padded from the beginning exactly like
    ``DistributedSampler``.  Every rank consequently executes the same number
    of optimizer steps, while the effective batch is
    ``per_device_batch_size * world_size``.
    """
    if per_device_batch_size < 1:
        raise ValueError(f"per_device_batch_size must be positive, got {per_device_batch_size}")
    if not rows:
        raise ValueError("cannot make distributed batches from an empty row set")
    indices = list(range(len(rows)))
    random.Random(int(seed) + int(epoch)).shuffle(indices)
    global_batch_size = per_device_batch_size * context.world_size
    padded_size = int(math.ceil(len(indices) / global_batch_size) * global_batch_size)
    if padded_size > len(indices):
        indices.extend(indices[: padded_size - len(indices)])
    for start in range(0, padded_size, global_batch_size):
        global_indices = indices[start : start + global_batch_size]
        local_start = context.rank * per_device_batch_size
        local_indices = global_indices[local_start : local_start + per_device_batch_size]
        yield [rows[index] for index in local_indices]


def distributed_epoch_batches_no_padding(
    rows: list[T],
    per_device_batch_size: int,
    context: DistributedContext,
    *,
    seed: int,
    epoch: int,
) -> Iterable[list[T]]:
    """Yield deterministic local batches without duplicating tail examples.

    Every rank yields the same number of steps.  A rank whose slice of the
    final global batch is empty yields an empty list; model-specific callers
    must execute a graph-connected zero-loss dummy so DDP collectives remain
    aligned without adding that dummy to the objective.
    """
    if per_device_batch_size < 1:
        raise ValueError(f"per_device_batch_size must be positive, got {per_device_batch_size}")
    if not rows:
        raise ValueError("cannot make distributed batches from an empty row set")
    indices = list(range(len(rows)))
    random.Random(int(seed) + int(epoch)).shuffle(indices)
    global_batch_size = per_device_batch_size * context.world_size
    for start in range(0, len(indices), global_batch_size):
        global_indices = indices[start : start + global_batch_size]
        local_start = context.rank * per_device_batch_size
        local_indices = global_indices[local_start : local_start + per_device_batch_size]
        yield [rows[index] for index in local_indices]


def distributed_document_augmentation_epoch_batches(
    query_rows: list[T],
    catalog_rows: list[T],
    catalog_passes: int,
    per_device_batch_size: int,
    context: DistributedContext,
    *,
    seed: int,
    epoch: int,
    no_padding: bool = True,
) -> Iterable[list[T]]:
    """Mix query and document-indexing views for one logical query epoch.

    Every query row is visited once.  ``catalog_passes`` independently shuffled
    copies of the catalog pool are added to the same stream.  The helper keeps
    only integer references, so long document strings are not duplicated in
    memory.  With ``no_padding=True`` the final global batch is not duplicated;
    callers must handle empty local tail batches with a zero-weight example.
    """
    if per_device_batch_size < 1:
        raise ValueError(f"per_device_batch_size must be positive, got {per_device_batch_size}")
    if not query_rows:
        raise ValueError("document augmentation requires a non-empty query row set")
    if catalog_passes < 0:
        raise ValueError(f"catalog_passes must be non-negative, got {catalog_passes}")
    if catalog_passes > 0 and not catalog_rows:
        raise ValueError("catalog_passes is positive but the catalog row set is empty")

    query_count = len(query_rows)
    references = list(range(query_count))
    query_rng = random.Random(int(seed) + int(epoch))
    query_rng.shuffle(references)
    for pass_index in range(catalog_passes):
        catalog_indices = list(range(len(catalog_rows)))
        # Use a separate deterministic permutation for each catalog pass so
        # repeated exposure does not replay the same row order verbatim.
        pass_rng = random.Random(
            int(seed) + int(epoch) * 1_000_003 + (pass_index + 1) * 9_176
        )
        pass_rng.shuffle(catalog_indices)
        references.extend(query_count + index for index in catalog_indices)
    query_rng.shuffle(references)

    global_batch_size = per_device_batch_size * context.world_size
    if no_padding:
        scheduled_size = len(references)
    else:
        scheduled_size = int(math.ceil(len(references) / global_batch_size) * global_batch_size)
        if scheduled_size > len(references):
            references.extend(references[: scheduled_size - len(references)])

    for start in range(0, scheduled_size, global_batch_size):
        global_references = references[start : start + global_batch_size]
        local_start = context.rank * per_device_batch_size
        local_references = global_references[local_start : local_start + per_device_batch_size]
        batch: list[T] = []
        for reference in local_references:
            if reference < query_count:
                batch.append(query_rows[reference])
            else:
                batch.append(catalog_rows[reference - query_count])
        yield batch


def shard_with_indices(rows: list[T], context: DistributedContext) -> list[tuple[int, T]]:
    """Shard inference/evaluation without padding, duplication, or reordering."""
    return [(index, rows[index]) for index in range(context.rank, len(rows), context.world_size)]


def all_gather_tensor_with_grad(tensor, context: DistributedContext):
    """Concatenate equal-shaped local tensors while preserving autograd edges."""
    if not context.enabled:
        return tensor
    import torch
    import torch.distributed.nn.functional as dist_nn

    return torch.cat(tuple(dist_nn.all_gather(tensor)), dim=0)


def all_gather_tensor(tensor, context: DistributedContext):
    """Concatenate equal-shaped tensors without creating autograd edges."""
    if not context.enabled:
        return tensor
    import torch
    import torch.distributed as dist

    gathered = [torch.empty_like(tensor) for _ in range(context.world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)
