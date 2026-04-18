#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.paths import build_pipeline_paths


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    paths = build_pipeline_paths(root)
    parser = argparse.ArgumentParser(description="STEP4 infrarenal neck measurement scaffold.")
    parser.add_argument("--project-root", default=str(root), help="Project root.")
    parser.add_argument("--step3-dir", default=str(paths.step3_dir), help="STEP3 output directory.")
    parser.add_argument("--output-dir", default=str(paths.step4_dir), help="STEP4 output directory.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    step3_dir = Path(args.step3_dir)
    required = [
        step3_dir / "named_segmentscolored.vtp",
        step3_dir / "named_centerlines.vtp",
        step3_dir / "step3_naming_orientation_contract.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("STEP4 scaffold: waiting on STEP3 core outputs:")
        for path in missing:
            print(f"  - {path}")
        return 2
    print("STEP4 scaffold is ready; measurement implementation is intentionally pending STEP3 verification.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

