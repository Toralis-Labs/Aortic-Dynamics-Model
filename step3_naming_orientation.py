#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.paths import build_pipeline_paths


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    paths = build_pipeline_paths(root)
    parser = argparse.ArgumentParser(description="STEP3 naming/orientation scaffold.")
    parser.add_argument("--project-root", default=str(root), help="Project root.")
    parser.add_argument("--step2-dir", default=str(paths.step2_dir), help="STEP2 output directory.")
    parser.add_argument("--output-dir", default=str(paths.step3_dir), help="STEP3 output directory.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    step2_dir = Path(args.step2_dir)
    required = [
        step2_dir / "segmentscolored.vtp",
        step2_dir / "aorta_centerline.vtp",
        step2_dir / "step2_geometry_contract.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("STEP3 scaffold: waiting on STEP2 core outputs:")
        for path in missing:
            print(f"  - {path}")
        return 2
    print("STEP3 scaffold is ready; naming/landmark implementation is intentionally pending STEP2 verification.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

