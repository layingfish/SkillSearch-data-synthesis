#!/usr/bin/env python3
"""Run the archived catalog probe with instantiated Dropout modules set to zero."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import torch


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_known_args()


def main() -> None:
    args, probe_args = parse_args()
    dropout_modules = []
    original_dropout = torch.nn.Dropout

    class ControlledDropout(original_dropout):
        def __init__(self, p=0.5, inplace=False):
            super().__init__(p=0.0, inplace=inplace)
            dropout_modules.append(self)

    torch.nn.Dropout = ControlledDropout
    try:
        sys.argv = [str(args.probe), *probe_args]
        runpy.run_path(str(args.probe), run_name="__main__")
    finally:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps(
                {
                    "dropout_rate_requested": 0.0,
                    "dropout_module_count": len(dropout_modules),
                    "dropout_module_rates": sorted({float(module.p) for module in dropout_modules}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
