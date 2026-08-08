#!/usr/bin/env python3
"""Run the archived LETTER-base catalog stage for an exact 50-step probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "7b883cb7f56667cdaee8bf0227470a2eba2c6882096dc07a2f8f1c6914396b76"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes_sha256(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(-1).cpu().numpy().tobytes()).hexdigest()


def model_state_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(-1).cpu().numpy().tobytes())
    return digest.hexdigest()


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("skill_gr_pipeline_original_20260714", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--foreach", choices=("auto", "true", "false"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer

    asset_dir = Path(__file__).resolve().parent
    source_path = asset_dir / "skill_gr_pipeline.original_20260714.py"
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA256 mismatch: {source_sha256}")
    pipeline = load_pipeline(source_path)

    base = args.repo_root / "outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713"
    inputs = base / "canonical_alias_core75_learned_d4_inputs"
    train_path = inputs / "rows_000.decoder_rows.base_plus_task5_gt2.jsonl"
    dev_path = inputs / "rows_001.decoder_rows.base.jsonl"
    catalog_path = inputs / "canonical_skills.jsonl"
    semantic_ids_path = base / "letter_d4_lightgcn_d32_official_vecdiv_v1_seed42_core75_gt2/semantic_ids.jsonl"

    semantic_ids, semantic_id_mode = pipeline.load_semantic_ids_with_mode(semantic_ids_path)
    sid_config = pipeline.infer_structured_sid_config(
        semantic_ids,
        codebook_size=None,
        dedup_size=None,
        semantic_id_mode=semantic_id_mode,
    )
    codebook_size = sid_config["codebook_size"]
    depth = sid_config["depth"]
    dedup_size = sid_config["dedup_size"]

    query_rows_source = pipeline.load_jsonl(train_path)
    rng = random.Random(args.seed)
    query_train_rows = pipeline.expand_single_sid_query_rows(query_rows_source, semantic_ids)
    query_dev_rows = pipeline.expand_single_sid_query_rows(pipeline.load_jsonl(dev_path), semantic_ids)
    catalog_rows = pipeline.build_single_sid_catalog_rows(
        pipeline.load_jsonl(catalog_path), semantic_ids, "skill document: "
    )
    rng.shuffle(catalog_rows)

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    pipeline.audit_source_token_lengths(
        "query_train", query_train_rows, tokenizer, args.max_source_length, False
    )
    pipeline.audit_source_token_lengths(
        "query_dev", query_dev_rows, tokenizer, args.max_source_length, False
    )
    pipeline.audit_source_token_lengths(
        "catalog", catalog_rows, tokenizer, args.max_source_length, False
    )

    model = pipeline.make_structured_t5_sid_model(
        args.model,
        codebook_size,
        depth,
        dedup_size,
        semantic_id_mode,
    ).to(device)
    initial_model_fingerprint = model_state_fingerprint(model)
    cpu_rng_sha256 = tensor_bytes_sha256(torch.random.get_rng_state())
    cuda_rng_sha256 = tensor_bytes_sha256(torch.cuda.get_rng_state())
    optimizer_kwargs = {"lr": 3e-4, "weight_decay": 0.0}
    if args.foreach != "auto":
        optimizer_kwargs["foreach"] = args.foreach == "true"
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    try:
        from torch.optim.optimizer import _default_to_fused_or_foreach

        inferred_fused, inferred_foreach = _default_to_fused_or_foreach(
            list(model.parameters()), differentiable=False, use_fused=False
        )
    except (ImportError, TypeError):
        inferred_fused, inferred_foreach = None, None

    print(
        json.dumps(
            {
                "event": "probe_header",
                "source_sha256": source_sha256,
                "python_torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "steps": args.steps,
                "initial_model_fingerprint": initial_model_fingerprint,
                "cpu_rng_state_sha256_after_model_init": cpu_rng_sha256,
                "cuda_rng_state_sha256_before_first_forward": cuda_rng_sha256,
                "optimizer_param_group": {
                    key: optimizer.param_groups[0].get(key)
                    for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "foreach", "fused")
                },
                "inferred_fused": inferred_fused,
                "inferred_foreach": inferred_foreach,
                "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model.train()
    rng.shuffle(catalog_rows)
    first_batch_ids = [str(row["index"]) for row in catalog_rows[: args.batch_size]]
    first_batch_payload = "".join(f"{row_id}\n" for row_id in first_batch_ids).encode()
    print(
        json.dumps(
            {
                "event": "first_batch",
                "row_ids": first_batch_ids,
                "row_ids_sha256": hashlib.sha256(first_batch_payload).hexdigest(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for step, start in enumerate(range(0, len(catalog_rows), args.batch_size), start=1):
        if step > args.steps:
            break
        batch_rows = catalog_rows[start : start + args.batch_size]
        batch = pipeline.structured_sid_batch(
            batch_rows,
            tokenizer,
            semantic_ids,
            codebook_size,
            depth,
            dedup_size,
            args.max_source_length,
            device,
            semantic_id_mode,
        )
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(**batch)
        loss.backward()
        optimizer.step()
        if step in {1, args.steps}:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "stage": "catalog",
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
