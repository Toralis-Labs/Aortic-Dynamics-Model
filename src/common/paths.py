from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    output_root: Path
    step1_dir: Path
    step2_dir: Path
    step3_dir: Path
    step4_dir: Path
    default_face_map: Path


def build_pipeline_paths(project_root: str | Path) -> PipelinePaths:
    root = Path(project_root).resolve()
    output = root / "Output files"
    return PipelinePaths(
        project_root=root,
        output_root=output,
        step1_dir=output / "STEP1",
        step2_dir=output / "STEP2",
        step3_dir=output / "STEP3",
        step4_dir=output / "STEP4",
        default_face_map=root / "0044_H_ABAO_AAA" / "face_id_to_name.json",
    )


def script_project_root(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parent

