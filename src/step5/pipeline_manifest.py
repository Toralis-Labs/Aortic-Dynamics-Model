from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.common.json_io import read_json, write_json
from src.common.paths import build_pipeline_paths


ALLOWED_STATUSES = {"success", "requires_review", "failed"}


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def _contract_status(path: Path, status_keys: Iterable[str]) -> tuple[str, list[str], Dict[str, Any]]:
    if not path.exists():
        return "failed", [f"missing contract: {path}"], {}
    try:
        contract = read_json(path)
    except Exception as exc:
        return "failed", [f"could not read contract {path}: {exc}"], {}
    for key in status_keys:
        value = contract.get(key)
        if value in ALLOWED_STATUSES:
            warnings = [str(v) for v in contract.get("warnings", [])]
            return str(value), warnings, contract
    return "failed", [f"contract {path} has no valid status field"], contract


def _file_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing output: {path}"
    if path.is_file() and path.stat().st_size <= 0:
        return False, f"empty output: {path}"
    return True, ""


def _overall_status(step_rows: list[Dict[str, Any]]) -> str:
    statuses = [str(row["step_status"]) for row in step_rows]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "requires_review" for status in statuses):
        return "requires_review"
    return "success"


def _step_row(
    name: str,
    required_outputs: Dict[str, Path],
    *,
    contract_path: Optional[Path] = None,
    status_keys: Iterable[str] = ("step_status", "status", "final_status"),
) -> Dict[str, Any]:
    missing_or_invalid: list[str] = []
    outputs: Dict[str, str] = {}
    for label, path in required_outputs.items():
        outputs[label] = _abs(path)
        ok, message = _file_status(path)
        if not ok:
            missing_or_invalid.append(message)

    contract_status = "success"
    warnings: list[str] = []
    if contract_path is not None and tuple(status_keys):
        contract_status, warnings, _ = _contract_status(contract_path, status_keys)
    elif contract_path is not None:
        if not contract_path.exists():
            contract_status = "failed"
            warnings = [f"missing contract: {contract_path}"]
        else:
            try:
                contract = read_json(contract_path)
                warnings = [str(v) for v in contract.get("warnings", [])]
            except Exception as exc:
                contract_status = "failed"
                warnings = [f"could not read contract {contract_path}: {exc}"]

    step_status = contract_status
    if missing_or_invalid:
        step_status = "failed"
    elif contract_status not in ALLOWED_STATUSES:
        step_status = "failed"

    return {
        "step_name": name,
        "step_status": step_status,
        "required_outputs": outputs,
        "missing_or_invalid": missing_or_invalid,
        "warnings": warnings,
    }


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    paths = build_pipeline_paths(project_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths.step5_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows = [
        _step_row(
            "step1_centerline_network",
            {
                "surface_cleaned": paths.step1_dir / "surface_cleaned.vtp",
                "centerlines_raw_debug": paths.step1_dir / "centerlines_raw_debug.vtp",
                "centerline_network": paths.step1_dir / "centerline_network.vtp",
                "junction_nodes_debug": paths.step1_dir / "junction_nodes_debug.vtp",
                "metadata_json": paths.step1_dir / "centerline_network_metadata.json",
            },
            contract_path=paths.step1_dir / "centerline_network_metadata.json",
            status_keys=(),
        ),
        _step_row(
            "step2_geometry_contract",
            {
                "segments_vtp": paths.step2_dir / "segmentscolored.vtp",
                "aorta_centerline": paths.step2_dir / "aorta_centerline.vtp",
                "contract_json": paths.step2_dir / "step2_geometry_contract.json",
            },
            contract_path=paths.step2_dir / "step2_geometry_contract.json",
        ),
        _step_row(
            "step3_naming_orientation",
            {
                "named_segments_vtp": paths.step3_dir / "named_segmentscolored.vtp",
                "named_centerlines_vtp": paths.step3_dir / "named_centerlines.vtp",
                "contract_json": paths.step3_dir / "step3_naming_orientation_contract.json",
            },
            contract_path=paths.step3_dir / "step3_naming_orientation_contract.json",
        ),
        _step_row(
            "step4_infrarenal_neck",
            {
                "measurements_json": paths.step4_dir / "step4_measurements.json",
                "colored_vtp": paths.step4_dir / "infrarenal_neck_colored.vtp",
            },
            contract_path=paths.step4_dir / "step4_measurements.json",
        ),
    ]
    pipeline_status = _overall_status(step_rows)
    warnings = [
        warning
        for row in step_rows
        for warning in row.get("warnings", [])
    ]
    manifest_path = output_dir / "pipeline_manifest.json"
    summary_path = output_dir / "pipeline_summary.txt"
    manifest = {
        "schema_version": 1,
        "step_name": "step5_pipeline_manifest",
        "step_status": pipeline_status,
        "input_paths": {
            "project_root": _abs(project_root),
            "output_root": _abs(paths.output_root),
        },
        "output_paths": {
            "pipeline_manifest": _abs(manifest_path),
            "pipeline_summary": _abs(summary_path),
        },
        "steps": step_rows,
        "warnings": sorted(set(warnings)),
    }
    write_json(manifest, manifest_path)
    lines = [
        f"Pipeline status: {pipeline_status}",
        "",
        "Steps:",
    ]
    lines.extend(f"- {row['step_name']}: {row['step_status']}" for row in step_rows)
    failures = [
        item
        for row in step_rows
        for item in row.get("missing_or_invalid", [])
    ]
    if failures:
        lines.extend(["", "Missing or invalid outputs:"])
        lines.extend(f"- {item}" for item in failures)
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {item}" for item in sorted(set(warnings)))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="STEP5 pipeline manifest validation.")
    parser.add_argument("--project-root", default=str(project_root), help="Project root.")
    parser.add_argument("--output-dir", default="", help="STEP5 output directory.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_manifest(args)
    print(f"STEP5 completed: {manifest.get('step_status')}")
    return 1 if manifest.get("step_status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
