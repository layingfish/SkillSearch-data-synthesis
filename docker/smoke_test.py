#!/usr/bin/env python3
"""Fast, offline validation of the SkillSearch development image."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/opt/skillsearch/external")
REFERENCE_ROOT = Path("/opt/skillsearch/reference")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required image asset is missing: {path}")


def require_revision(path: Path, expected: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected:
        raise RuntimeError(f"{path} is not pinned to {expected}; found {head}")


def main() -> None:
    for module_name in (
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "transformers",
        "k_means_constrained",
        "vector_quantize_pytorch",
        "tqdm",
        "wandb",
        "tensorboardX",
    ):
        importlib.import_module(module_name)

    require_file(EXTERNAL_ROOT / "LightGCN-PyTorch" / "code" / "model.py")
    require_file(EXTERNAL_ROOT / "LightGCN-PyTorch" / "code" / "world.py")
    require_file(EXTERNAL_ROOT / "LETTER" / "RQ-VAE" / "models" / "rqvae.py")
    require_file(REFERENCE_ROOT / "RIPOR" / "README.md")
    require_file(REFERENCE_ROOT / "PAG" / "README.md")
    require_revision(EXTERNAL_ROOT / "LightGCN-PyTorch", "947ca2b3b1d2d3545b114145710cb06c4e57b3d2")
    require_revision(EXTERNAL_ROOT / "LETTER", "8d0154e28de37dbb6e24871c508ad8ddb1921cda")
    require_revision(REFERENCE_ROOT / "RIPOR", "03fd0cbfafdeeb9cdd866e0c44727fec39a96a52")
    require_revision(REFERENCE_ROOT / "PAG", "49802454fb157fc659c1c6c42bb2d940a9f6bc1b")

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import skill_gr_pipeline  # noqa: F401
    import train_letter_rqvae_official  # noqa: F401
    import train_lightgcn_cf_embeddings  # noqa: F401

    print("SkillSearch image smoke test passed")


if __name__ == "__main__":
    main()
