from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict

from . import __version__

AAA_DIAMETER_THRESHOLD_MM = 30.0


def _normalize_input_path(raw_path: str) -> str:
    value = (raw_path or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {chr(34), chr(39)}:
        value = value[1:-1].strip()
    return value


def _prompt_for_path() -> str:
    try:
        return _normalize_input_path(input("Enter the path to the 3D model file (.vtp): "))
    except EOFError:
        return ""


def _read_key_value_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _default_output_dir(input_path: Path) -> Path:
    return input_path.resolve().parent / f"{input_path.stem}_toralis_output"


def _run_orientation(input_path: Path, output_dir: Path) -> Path:
    from . import orientation

    surface_with_centerlines = output_dir / "oriented_surface_with_centerlines.vtp"
    centerlines = output_dir / "oriented_labeled_centerlines.vtp"
    metadata = output_dir / "oriented_labeled_centerlines_metadata.json"

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "orientation.py",
            "--input",
            str(input_path),
            "--output_surface_with_centerlines",
            str(surface_with_centerlines),
            "--output_centerlines",
            str(centerlines),
            "--metadata",
            str(metadata),
        ]
        exit_code = orientation.main()
    finally:
        sys.argv = argv_backup

    if exit_code != 0:
        raise RuntimeError("Orientation stage failed.")

    return metadata


def _run_measurement(output_dir: Path) -> Path:
    from . import measure_infrarenal_neck

    centerlines = output_dir / "oriented_labeled_centerlines.vtp"
    surface_with_centerlines = output_dir / "oriented_surface_with_centerlines.vtp"
    metadata = output_dir / "oriented_labeled_centerlines_metadata.json"
    colored = output_dir / "infrarenal_neck_colored.vtp"
    report = output_dir / "infrarenal_neck_report.txt"

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "measure_infrarenal_neck.py",
            "--input_centerlines",
            str(centerlines),
            "--input_surface_with_centerlines",
            str(surface_with_centerlines),
            "--input_metadata",
            str(metadata),
            "--output_colored",
            str(colored),
            "--output_report",
            str(report),
        ]
        exit_code = measure_infrarenal_neck.main()
    finally:
        sys.argv = argv_backup

    if exit_code != 0:
        raise RuntimeError(f"Measurement stage failed. See {report}")

    return report


def _write_not_aneurysm(output_dir: Path, input_path: Path, reason: str, report_data: Dict[str, str]) -> Path:
    out_path = output_dir / "toralis_result.txt"
    lines = [
        "status=not_aortic_aneurysm",
        "message=not aortic aneurysm",
        f"input_model={input_path.resolve()}",
        f"reason={reason}",
    ]
    max_eq = report_data.get("max_aneurysm_diameter_eq_mm")
    if max_eq is not None:
        lines.append(f"max_aneurysm_diameter_eq_mm={max_eq}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _write_success_summary(output_dir: Path, input_path: Path, report_path: Path) -> Path:
    out_path = output_dir / "toralis_result.txt"
    lines = [
        "status=success",
        f"input_model={input_path.resolve()}",
        f"measurement_report={report_path.resolve()}",
        f"colored_output={output_dir / 'infrarenal_neck_colored.vtp'}",
        f"orientation_metadata={output_dir / 'oriented_labeled_centerlines_metadata.json'}",
        f"centerlines_output={output_dir / 'oriented_labeled_centerlines.vtp'}",
        f"surface_with_centerlines_output={output_dir / 'oriented_surface_with_centerlines.vtp'}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_pipeline(input_path: Path, output_dir: Path, aaa_threshold_mm: float = AAA_DIAMETER_THRESHOLD_MM) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_orientation(input_path=input_path, output_dir=output_dir)
    report_path = _run_measurement(output_dir=output_dir)
    report_data = _read_key_value_file(report_path)

    if report_data.get("status", "").lower() == "failed":
        raise RuntimeError(report_data.get("error", f"Pipeline failed. See {report_path}"))

    max_eq_mm = _safe_float(report_data.get("max_aneurysm_diameter_eq_mm", "nan"))
    if not math.isfinite(max_eq_mm) or max_eq_mm < aaa_threshold_mm:
        result_path = _write_not_aneurysm(
            output_dir=output_dir,
            input_path=input_path,
            reason=f"maximum equivalent aneurysm diameter {max_eq_mm:.3f} mm is below the AAA threshold of {aaa_threshold_mm:.1f} mm",
            report_data=report_data,
        )
        print("not aortic aneurysm")
        print(f"Result file: {result_path}")
        return 0

    result_path = _write_success_summary(output_dir=output_dir, input_path=input_path, report_path=report_path)
    print(f"AAA measurements generated successfully in: {output_dir}")
    print(f"Summary file: {result_path}")
    print(f"Measurement report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toralis_evar",
        description="Run the Toralis EVAR pipeline on a VTP 3D model and output AAA measurements.",
    )
    parser.add_argument("input_path", nargs="?", help="Path to the input VTP model. If omitted, the CLI prompts for it.")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Optional output directory. Defaults to <input_stem>_toralis_output next to the input file.",
    )
    parser.add_argument(
        "--aaa-threshold-mm",
        type=float,
        default=AAA_DIAMETER_THRESHOLD_MM,
        help="Equivalent diameter threshold used to classify the model as an aortic aneurysm.",
    )
    parser.add_argument("--version", action="version", version=f"toralis {__version__}")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    raw_input_path = _normalize_input_path(args.input_path or _prompt_for_path())
    if not raw_input_path:
        parser.error("an input path is required")

    input_path = Path(raw_input_path).expanduser()
    if not input_path.exists():
        parser.error(f"input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".vtp":
        parser.error("input model must be a .vtp file")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir(input_path)

    try:
        return run_pipeline(input_path=input_path, output_dir=output_dir, aaa_threshold_mm=args.aaa_threshold_mm)
    except Exception as exc:
        sys.stderr.write(f"toralis_evar failed: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
