#!/usr/bin/env python3
"""Run and validate one archived baseline runtime-matrix arm."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from legacy_baseline_parity_probe import _hash_tensor_state, _torch_rng_sha256
except ModuleNotFoundError:
    from scripts.legacy_baseline_parity_probe import _hash_tensor_state, _torch_rng_sha256


EXPECTED_STEPS = 50


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_historical_pipeline(source_root: Path):
    scripts_dir = source_root / "scripts"
    pipeline_path = scripts_dir / "skill_gr_pipeline.py"
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"archived pipeline is missing: {pipeline_path}")
    sys.path.insert(0, str(scripts_dir))
    import skill_gr_pipeline  # type: ignore

    if Path(skill_gr_pipeline.__file__).resolve() != pipeline_path.resolve():
        raise RuntimeError(f"loaded the wrong archived pipeline: {skill_gr_pipeline.__file__}")
    return skill_gr_pipeline


def selected_weight_file(model_dir: Path) -> Path:
    candidates = [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError(
            f"runtime probe model view must expose exactly one weight file, found: {present}"
        )
    return present[0]


def catalog_loss_summary(loss_history: Path, expected_steps: int = EXPECTED_STEPS) -> dict[str, float]:
    events = [
        json.loads(line)
        for line in loss_history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog = [
        row for row in events if row.get("stage") == "catalog" and row.get("event") == "train"
    ]
    steps = [int(row["step"]) for row in catalog]
    if steps != list(range(1, expected_steps + 1)):
        raise ValueError(f"catalog steps are incomplete or reordered: {steps}")
    return {
        "step_1": float(catalog[0]["train_loss"]),
        f"step_{expected_steps}": float(catalog[-1]["train_loss"]),
    }


def command_run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"each runtime arm requires exactly one visible CUDA GPU, got {torch.cuda.device_count()}"
        )
    historical = load_historical_pipeline(args.legacy_source)
    semantic_ids, semantic_id_mode = historical.load_semantic_ids_with_mode(args.semantic_ids)
    sid_config = historical.infer_structured_sid_config(
        semantic_ids,
        semantic_id_mode=semantic_id_mode,
    )
    weight_file = selected_weight_file(args.model)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch_rng_before = _torch_rng_sha256(torch.get_rng_state())
    cuda_rng_before = _torch_rng_sha256(torch.cuda.get_rng_state())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = historical.make_structured_t5_sid_model(
        str(args.model),
        int(sid_config["codebook_size"]),
        int(sid_config["depth"]),
        int(sid_config["dedup_size"]),
        semantic_id_mode,
    )
    state_hashes = _hash_tensor_state(model.state_dict())
    torch_rng_after = _torch_rng_sha256(torch.get_rng_state())
    cuda_rng_after = _torch_rng_sha256(torch.cuda.get_rng_state())
    del tokenizer, model
    gc.collect()

    with tempfile.TemporaryDirectory(prefix=f"skillgr-runtime-{args.arm}.", dir="/tmp") as directory:
        train_output = Path(directory) / "train"
        command = [
            sys.executable,
            str(args.legacy_source / "scripts" / "skill_gr_pipeline.py"),
            "train-single-sid-decoder",
            "--rows",
            str(args.rows),
            "--dev-rows",
            str(args.dev_rows),
            "--semantic-ids",
            str(args.semantic_ids),
            "--skills-jsonl",
            str(args.skills_jsonl),
            "--output-dir",
            str(train_output),
            "--model",
            str(args.model),
            "--catalog-epochs",
            "1",
            "--catalog-max-steps",
            str(args.steps),
            "--epochs",
            "0",
            "--max-source-length",
            "2048",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "128",
            "--lr",
            "0.0003",
            "--weight-decay",
            "0",
            "--logging-steps",
            "1",
            "--eval-steps",
            "1000",
            "--seed",
            str(args.seed),
            "--bf16",
        ]
        subprocess.run(command, check=True, env=os.environ.copy())
        losses = catalog_loss_summary(train_output / "loss_history.jsonl", args.steps)
        manifest = json.loads(
            (train_output / "single_sid_decoder_train_manifest.json").read_text(encoding="utf-8")
        )
    expected_manifest = {
        "batch_size": 16,
        "catalog_rows": 14551,
        "catalog_steps": args.steps,
        "query_steps": 0,
        "seed": args.seed,
        "lr": 3e-4,
        "weight_decay": 0.0,
        "semantic_id_mode": "learned_codes",
        "semantic_ids_sha256": "da2fff997b1979f7d51996c5ddbe4593a18d23bdfd546c61dd61fb783567349c",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"training manifest mismatch: {mismatches}")

    payload: dict[str, Any] = {
        "artifact_format": "legacy_runtime_matrix_arm_v1",
        "arm": args.arm,
        "historical_source_commit": args.legacy_commit,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "huggingface_hub": package_version("huggingface-hub"),
            "safetensors": package_version("safetensors"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "condition": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": 16,
            "bf16": True,
            "lr": 3e-4,
            "weight_decay": 0.0,
            "weight_format": weight_file.name,
            "weight_sha256": file_sha256(weight_file),
        },
        "initialization": {
            "torch_rng_before_model_sha256": torch_rng_before,
            "torch_rng_after_model_sha256": torch_rng_after,
            "cuda_rng_before_model_sha256": cuda_rng_before,
            "cuda_rng_after_model_sha256": cuda_rng_after,
            "state_sha256": state_hashes,
        },
        "losses": losses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "runtime_probe.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "runtime_probe.json")
    print(json.dumps(payload, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--legacy-commit", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--skills-jsonl", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.set_defaults(func=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1:
        raise ValueError(f"steps must be positive, got {args.steps}")
    args.func(args)


if __name__ == "__main__":
    main()
