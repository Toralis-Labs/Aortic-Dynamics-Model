from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    project_root: Path
    input_dir: Path
    output_dir: Path

    surface_vtp: Path
    centerline_network_vtp: Path
    centerline_metadata_json: Path
    input_roles_json: Path

    segmented_surface_vtp: Path
    boundary_rings_vtp: Path
    segmentation_result_json: Path


def build_workspace_paths(project_root: str | Path) -> WorkspacePaths:
    root = Path(project_root).resolve()
    input_dir = root / "inputs"
    output_dir = root / "outputs"

    return WorkspacePaths(
        project_root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        surface_vtp=input_dir / "surface_cleaned.vtp",
        centerline_network_vtp=input_dir / "centerline_network.vtp",
        centerline_metadata_json=input_dir / "centerline_network_metadata.json",
        input_roles_json=input_dir / "input_roles.json",
        segmented_surface_vtp=output_dir / "segmented_surface.vtp",
        boundary_rings_vtp=output_dir / "boundary_rings.vtp",
        segmentation_result_json=output_dir / "segmentation_result.json",
    )


def script_project_root(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parent


# Temporary compatibility alias.
# Remove this after src/step2/geometry_contract.py is refactored to call
# build_workspace_paths directly.
def build_pipeline_paths(project_root: str | Path) -> WorkspacePaths:
    return build_workspace_paths(project_root)