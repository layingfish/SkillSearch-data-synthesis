#!/usr/bin/env python3
"""Audit and finalize an exact archived single-SID training probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable


PROBE_CATALOG_STEPS = 950
ARCHIVED_COMPARISON_STEPS = (50, 900, 950)
ARCHIVED_STEP_PATTERN = re.compile(
    r"^catalog_step=([0-9]+) train_loss=([0-9]+(?:\.[0-9]+)?)$"
)
EXPECTED_PROBE_MANIFEST = {
    "batch_size": 16,
    "eval_batch_size": 128,
    "max_source_length": 2048,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "seed": 20260626,
    "catalog_rows": 14551,
    "query_train_rows": 129341,
    "query_dev_rows": 2501,
    "catalog_steps": PROBE_CATALOG_STEPS,
    "query_steps": 0,
    "codebook_size": 256,
    "depth": 4,
    "dedup_size": 0,
    "semantic_id_mode": "learned_codes",
    "semantic_ids_sha256": "da2fff997b1979f7d51996c5ddbe4593a18d23bdfd546c61dd61fb783567349c",
    "source_text_policy": "skillrouter_query_1500_chars_with_instruction_v1",
    "formulation": "catalog indexing followed by query->single SID finetuning",
}
EXPECTED_ARCHIVED_MANIFEST = {
    **EXPECTED_PROBE_MANIFEST,
    "catalog_steps": 2730,
    "query_steps": 40420,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rng_state_sha256(rng: random.Random) -> str:
    return hashlib.sha256(pickle.dumps(rng.getstate(), protocol=5)).hexdigest()


def legacy_shuffle_trace(
    row_ids: Iterable[str],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    recorded_batches: int = 50,
) -> dict[str, Any]:
    """Replay the archived persistent-RNG, in-place shuffle sequence."""
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = list(row_ids)
    if not order:
        raise ValueError("row_ids must not be empty")
    rng = random.Random(seed)
    trace: dict[str, Any] = {
        "seed": seed,
        "rows": len(order),
        "batch_size": batch_size,
        "steps_per_epoch": math.ceil(len(order) / batch_size),
        "tail_batch_size": len(order) % batch_size or batch_size,
        "rng_initial_sha256": rng_state_sha256(rng),
    }
    rng.shuffle(order)
    trace["initial_catalog_shuffle"] = {
        "order_sha256": stable_json_sha256(order),
        "first_batch": order[:batch_size],
        "rng_sha256": rng_state_sha256(rng),
    }
    epoch_traces = []
    for epoch in range(epochs):
        rng.shuffle(order)
        first_rows = order[: min(len(order), recorded_batches * batch_size)]
        epoch_traces.append(
            {
                "epoch": epoch + 1,
                "order_sha256": stable_json_sha256(order),
                "first_batch": order[:batch_size],
                "last_batch": order[(len(order) // batch_size) * batch_size :]
                if len(order) % batch_size
                else order[-batch_size:],
                "tail_batch_size": len(order) % batch_size or batch_size,
                "recorded_batch_count": math.ceil(len(first_rows) / batch_size),
                "recorded_rows_sha256": stable_json_sha256(first_rows),
                "rng_sha256": rng_state_sha256(rng),
            }
        )
    trace["epochs"] = epoch_traces
    trace["final_order"] = order
    trace["rng"] = rng
    return trace


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hash_tensor_state(state_dict: dict[str, Any]) -> dict[str, str]:
    import torch

    groups = {
        "full": hashlib.sha256(),
        "backbone": hashlib.sha256(),
        "sid_embedding": hashlib.sha256(),
        "code_heads": hashlib.sha256(),
        "control_head": hashlib.sha256(),
    }
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.view(torch.uint8).numpy().tobytes()
        selected = [groups["full"]]
        if name.startswith("backbone."):
            selected.append(groups["backbone"])
        elif name.startswith("sid_embedding."):
            selected.append(groups["sid_embedding"])
        elif name.startswith("code_heads."):
            selected.append(groups["code_heads"])
        elif name.startswith("control_head."):
            selected.append(groups["control_head"])
        for digest in selected:
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return {name: digest.hexdigest() for name, digest in groups.items()}


def _torch_rng_sha256(state) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


def _token_batch_sha256(tokenizer, texts: list[str], max_length: int) -> str:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return stable_json_sha256(
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    )


def _load_historical_pipeline(source_root: Path):
    scripts_dir = source_root / "scripts"
    pipeline_path = scripts_dir / "skill_gr_pipeline.py"
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"archived pipeline is missing: {pipeline_path}")
    sys.path.insert(0, str(scripts_dir))
    import skill_gr_pipeline  # type: ignore

    loaded_path = Path(skill_gr_pipeline.__file__).resolve()
    if loaded_path != pipeline_path.resolve():
        raise RuntimeError(f"loaded the wrong historical pipeline: {loaded_path}")
    return skill_gr_pipeline


def command_audit(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    historical = _load_historical_pipeline(args.legacy_source)
    semantic_ids, semantic_id_mode = historical.load_semantic_ids_with_mode(args.semantic_ids)
    sid_config = historical.infer_structured_sid_config(
        semantic_ids,
        semantic_id_mode=semantic_id_mode,
    )
    catalog_rows = historical.build_single_sid_catalog_rows(
        historical.load_jsonl(args.skills_jsonl),
        semantic_ids,
        args.catalog_input_prefix,
    )
    query_rows = historical.expand_single_sid_query_rows(
        historical.load_jsonl(args.rows),
        semantic_ids,
    )
    catalog_ids = [str(row["index"]) for row in catalog_rows]
    query_ids = [
        f"{row.get('index')}::{row['positive_skill_id']}" for row in query_rows
    ]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("expanded query row identifiers are not unique")
    trace = legacy_shuffle_trace(
        catalog_ids,
        seed=args.seed,
        epochs=args.catalog_epochs,
        batch_size=args.batch_size,
    )
    ordered_by_id = {str(row["index"]): row for row in catalog_rows}
    first_epoch_ids = trace["epochs"][0]["first_batch"]
    first_epoch_texts = [
        historical.decoder_source_text(ordered_by_id[row_id]) for row_id in first_epoch_ids
    ]
    query_rng = trace["rng"]
    query_rng.shuffle(query_ids)
    trace["query_epoch_1"] = {
        "rows": len(query_ids),
        "order_sha256": stable_json_sha256(query_ids),
        "first_batch": query_ids[: args.batch_size],
        "rng_sha256": rng_state_sha256(query_rng),
    }

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch_rng_before = _torch_rng_sha256(torch.get_rng_state())
    cuda_rng_before = (
        [_torch_rng_sha256(state) for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_batch_sha256 = _token_batch_sha256(tokenizer, first_epoch_texts, args.max_source_length)
    model = historical.make_structured_t5_sid_model(
        str(args.model),
        int(sid_config["codebook_size"]),
        int(sid_config["depth"]),
        int(sid_config["dedup_size"]),
        semantic_id_mode,
    )
    state_hashes = _hash_tensor_state(model.state_dict())
    torch_rng_after = _torch_rng_sha256(torch.get_rng_state())
    cuda_rng_after = (
        [_torch_rng_sha256(state) for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_class = optimizer.__class__.__name__
    optimizer_defaults = {
        key: value
        for key, value in optimizer.defaults.items()
        if isinstance(value, (bool, float, int, str, type(None), tuple))
    }
    del optimizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    critical_files = {
        "rows": args.rows,
        "dev_rows": args.dev_rows,
        "skills_jsonl": args.skills_jsonl,
        "semantic_ids": args.semantic_ids,
        "model_config": args.model / "config.json",
        "model_safetensors": args.model / "model.safetensors",
        "tokenizer_json": args.model / "tokenizer.json",
        "sentencepiece_model": args.model / "spiece.model",
        "historical_pipeline": args.legacy_source / "scripts" / "skill_gr_pipeline.py",
        "probe_launcher": args.launcher,
    }
    missing = [str(path) for path in critical_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"parity audit inputs are missing: {missing}")
    trace.pop("final_order")
    trace.pop("rng")
    payload = {
        "artifact_format": "legacy_baseline_parity_probe_v1",
        "historical_source_commit": args.legacy_commit,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "numpy": _package_version("numpy"),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "configuration": {
            "seed": args.seed,
            "catalog_epochs": args.catalog_epochs,
            "batch_size": args.batch_size,
            "max_source_length": args.max_source_length,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "bf16": bool(args.bf16),
            "sampler": "archived_single_process_persistent_rng_in_place_shuffle_no_tail_padding",
        },
        "file_sha256": {name: file_sha256(path) for name, path in critical_files.items()},
        "semantic_id_mode": semantic_id_mode,
        "sid_config": sid_config,
        "query_rows": len(query_rows),
        "catalog_rows": len(catalog_rows),
        "shuffle_trace": trace,
        "first_catalog_batch_tokenization_sha256": token_batch_sha256,
        "initialization": {
            "torch_rng_before_model_sha256": torch_rng_before,
            "torch_rng_after_model_sha256": torch_rng_after,
            "cuda_rng_before_model_sha256": cuda_rng_before,
            "cuda_rng_after_model_sha256": cuda_rng_after,
            "state_sha256": state_hashes,
            "trainable_parameters": trainable_parameters,
            "optimizer_class": optimizer_class,
            "optimizer_defaults": optimizer_defaults,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True))


def _archived_step_losses(path: Path) -> dict[int, float]:
    matches: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ARCHIVED_STEP_PATTERN.match(line.strip())
        if match:
            step = int(match.group(1))
            if step in ARCHIVED_COMPARISON_STEPS:
                if step in matches:
                    raise ValueError(f"duplicate archived catalog step-{step} loss")
                matches[step] = float(match.group(2))
    missing = set(ARCHIVED_COMPARISON_STEPS) - set(matches)
    if missing:
        raise ValueError(f"archived train log is missing comparison steps: {sorted(missing)}")
    return matches


def _validate_manifest_values(
    manifest: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} manifest mismatch: {mismatches}")


def command_finalize(args: argparse.Namespace) -> None:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (args.training_output / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog_events = [
        row for row in events if row.get("stage") == "catalog" and row.get("event") == "train"
    ]
    steps = [int(row["step"]) for row in catalog_events]
    if steps != list(range(1, PROBE_CATALOG_STEPS + 1)):
        raise ValueError(f"probe catalog train steps are incomplete or reordered: {steps[:5]}...{steps[-5:]}")
    manifest = json.loads(
        (args.training_output / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8")
    )
    _validate_manifest_values(manifest, EXPECTED_PROBE_MANIFEST, label="probe")
    expected_audit_configuration = {
        "seed": manifest["seed"],
        "catalog_epochs": 3,
        "batch_size": manifest["batch_size"],
        "max_source_length": manifest["max_source_length"],
        "lr": manifest["lr"],
        "weight_decay": manifest["weight_decay"],
        "bf16": True,
        "sampler": "archived_single_process_persistent_rng_in_place_shuffle_no_tail_padding",
    }
    if audit.get("configuration") != expected_audit_configuration:
        raise ValueError(
            "audit/training configuration mismatch: "
            f"expected={expected_audit_configuration}, actual={audit.get('configuration')}"
        )
    archived_manifest = json.loads(args.archived_manifest.read_text(encoding="utf-8"))
    _validate_manifest_values(
        archived_manifest,
        EXPECTED_ARCHIVED_MANIFEST,
        label="archived baseline",
    )
    archived_train_log_sha256 = file_sha256(args.archived_train_log)
    archived_manifest_sha256 = file_sha256(args.archived_manifest)
    if archived_train_log_sha256 != args.archived_train_log_sha256:
        raise ValueError("archived train log SHA-256 mismatch")
    if archived_manifest_sha256 != args.archived_manifest_sha256:
        raise ValueError("archived manifest SHA-256 mismatch")
    archived_losses = _archived_step_losses(args.archived_train_log)
    observed_losses = {
        step: float(catalog_events[step - 1]["train_loss"])
        for step in ARCHIVED_COMPARISON_STEPS
    }
    matches = {
        step: f"{observed_losses[step]:.6f}" == f"{archived_losses[step]:.6f}"
        for step in ARCHIVED_COMPARISON_STEPS
    }
    step_one_loss = float(catalog_events[0]["train_loss"])
    aligned = all(matches.values())
    payload = {
        "artifact_format": "legacy_baseline_parity_result_v1",
        "aligned": aligned,
        "comparison": {
            "condition": {
                "source_commit": audit["historical_source_commit"],
                "seed": manifest["seed"],
                "catalog_steps": manifest["catalog_steps"],
                "batch_size": manifest["batch_size"],
                "bf16": audit["configuration"]["bf16"],
                "lr": manifest["lr"],
                "weight_decay": manifest["weight_decay"],
            },
            "step_1_loss": step_one_loss,
            "boundary_losses": {
                "step_910_tail_batch": float(catalog_events[909]["train_loss"]),
                "step_911_second_epoch_first_batch": float(catalog_events[910]["train_loss"]),
            },
            "steps": {
                str(step): {
                    "observed": observed_losses[step],
                    "archived_rounded_6dp": archived_losses[step],
                    "delta_from_archived_rounded_value": observed_losses[step] - archived_losses[step],
                    "six_decimal_match": matches[step],
                }
                for step in ARCHIVED_COMPARISON_STEPS
            },
        },
        "archived_reference": {
            "train_log": str(args.archived_train_log.resolve()),
            "train_log_sha256": archived_train_log_sha256,
            "manifest": str(args.archived_manifest.resolve()),
            "manifest_sha256": archived_manifest_sha256,
        },
        "audit": audit,
        "training_manifest": manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload["comparison"], sort_keys=True))
    if not aligned:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--legacy-source", type=Path, required=True)
    audit.add_argument("--legacy-commit", required=True)
    audit.add_argument("--rows", type=Path, required=True)
    audit.add_argument("--dev-rows", type=Path, required=True)
    audit.add_argument("--skills-jsonl", type=Path, required=True)
    audit.add_argument("--semantic-ids", type=Path, required=True)
    audit.add_argument("--model", type=Path, required=True)
    audit.add_argument("--launcher", type=Path, required=True)
    audit.add_argument("--catalog-input-prefix", default="skill document: ")
    audit.add_argument("--catalog-epochs", type=int, default=3)
    audit.add_argument("--batch-size", type=int, default=16)
    audit.add_argument("--max-source-length", type=int, default=2048)
    audit.add_argument("--lr", type=float, default=3e-4)
    audit.add_argument("--weight-decay", type=float, default=0.0)
    audit.add_argument("--seed", type=int, default=20260626)
    audit.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(func=command_audit)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--audit", type=Path, required=True)
    finalize.add_argument("--training-output", type=Path, required=True)
    finalize.add_argument("--archived-train-log", type=Path, required=True)
    finalize.add_argument("--archived-train-log-sha256", required=True)
    finalize.add_argument("--archived-manifest", type=Path, required=True)
    finalize.add_argument("--archived-manifest-sha256", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
