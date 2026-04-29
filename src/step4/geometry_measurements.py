from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import vtk

from src.common.geometry import (
    cumulative_arclength,
    distance,
    equivalent_diameter_from_area,
    point_at_arclength,
    polygon_area_normal,
    projected_major_minor_diameters,
    tangent_at_arclength,
    unit,
)
from src.common.json_io import json_safe, read_json, write_json
from src.common.paths import build_pipeline_paths
from src.common.vtk_helpers import (
    add_int_cell_array,
    add_uchar3_cell_array,
    cell_centers,
    read_vtp,
    write_vtp,
)


SCHEMA_VERSION = "0.1.0"
STEP_NAME = "STEP4_GEOMETRY_MEASUREMENTS"

STATUS_MEASURED = "measured"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_REQUIRES_REVIEW = "requires_review"
STATUS_FAILED = "failed_to_measure"

NECK_REGION_ID = 1
OTHER_REGION_ID = 0
NECK_COLOR = (230, 83, 48)
OTHER_COLOR = (176, 184, 190)


class Step4Failure(RuntimeError):
    pass


@dataclass
class Projection:
    s: float
    point: np.ndarray
    distance: float
    segment_index: int


@dataclass
class SectionMeasurement:
    s: float
    center: np.ndarray
    tangent: np.ndarray
    contour_center: np.ndarray
    area: float
    perimeter: float
    major_diameter: Optional[float]
    minor_diameter: Optional[float]
    equivalent_diameter: Optional[float]
    contour_point_count: int


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _round_float(value: Any, ndigits: int = 6) -> Optional[float]:
    out = _finite_float(value)
    return None if out is None else round(out, ndigits)


def _round_point(value: Any, ndigits: int = 6) -> Optional[list[float]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
        return None
    return [round(float(v), ndigits) for v in arr.tolist()]


def _valid_point(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(float, copy=True)


def _append_unique(values: list[str], message: str) -> None:
    text = str(message).strip()
    if text and text not in values:
        values.append(text)


def _make_contract(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "step_name": STEP_NAME,
        "schema_version": SCHEMA_VERSION,
        "purpose": "3D geometry measurements from Step 3 named anatomy",
        "geometry_only": True,
        "source_inputs": {
            "named_surface_vtp": _abs(paths["named_surface"]),
            "named_centerlines_vtp": _abs(paths["named_centerlines"]),
            "step3_contract_json": _abs(paths["step3_contract"]),
        },
        "outputs": {
            "geometry_measurements_json": _abs(paths["output_json"]),
            "infrarenal_neck_labeled_vtp": _abs(paths["output_labeled_vtp"]),
        },
        "units": {
            "length": "mm",
            "angle": "degrees",
        },
        "landmarks": {
            "lowest_renal_artery_name": None,
            "lowest_renal_ostium_center_xyz": None,
            "lowest_renal_aortic_centerline_s_mm": None,
            "aortic_bifurcation_center_xyz": None,
            "aortic_bifurcation_s_mm": None,
        },
        "aortic_neck": {
            "neck_diameter_D0_major_mm": None,
            "neck_diameter_D0_minor_mm": None,
            "neck_diameter_D0_equivalent_mm": None,
            "neck_diameter_D10_major_mm": None,
            "neck_diameter_D10_minor_mm": None,
            "neck_diameter_D10_equivalent_mm": None,
            "neck_diameter_D15_major_mm": None,
            "neck_diameter_D15_minor_mm": None,
            "neck_diameter_D15_equivalent_mm": None,
            "neck_reference_diameter_mm": None,
            "neck_reference_diameter_source": None,
            "neck_length_mm": None,
            "neck_end_s_mm": None,
            "neck_end_center_xyz": None,
            "proximal_neck_angulation_deg": None,
            "proximal_neck_angulation_category": None,
        },
        "iliac": {
            "left": {
                "iliac_treatment_diameter_mm": None,
                "iliac_reference_diameter_source": None,
                "distal_iliac_seal_zone_length_mm": None,
                "distal_iliac_seal_zone_start_xyz": None,
                "distal_iliac_seal_zone_end_xyz": None,
                "selected_landing_segment": None,
            },
            "right": {
                "iliac_treatment_diameter_mm": None,
                "iliac_reference_diameter_source": None,
                "distal_iliac_seal_zone_length_mm": None,
                "distal_iliac_seal_zone_start_xyz": None,
                "distal_iliac_seal_zone_end_xyz": None,
                "selected_landing_segment": None,
            },
        },
        "measurement_status": {
            "aortic_neck_diameter_measurement_status": None,
            "aortic_neck_length_measurement_status": None,
            "aortic_neck_angulation_measurement_status": None,
            "left_iliac_diameter_measurement_status": None,
            "right_iliac_diameter_measurement_status": None,
            "left_iliac_seal_zone_measurement_status": None,
            "right_iliac_seal_zone_measurement_status": None,
            "overall_geometry_measurement_status": None,
        },
        "metadata": {
            "measurement_confidence": None,
            "warnings": [],
            "open_questions": [],
            "not_available": [],
            "assumptions": [
                "abdominal_aorta_trunk centerline is treated as proximal/inlet to distal/bifurcation when projecting downstream arclength.",
                "neck_reference_diameter_mm uses the maximum available major diameter among D0, D10, and D15.",
                "iliac seal-zone geometry is measured on the distal-most up to 20 mm of the selected iliac segment.",
            ],
            "discovered_arrays": {
                "named_surface": {},
                "named_centerlines": {},
            },
        },
    }


def _attribute_arrays(attrs: Any) -> list[dict[str, Any]]:
    arrays: list[dict[str, Any]] = []
    for idx in range(attrs.GetNumberOfArrays()):
        arr = attrs.GetAbstractArray(idx)
        if arr is None:
            continue
        arrays.append(
            {
                "name": str(arr.GetName() or ""),
                "class_name": str(arr.GetClassName()),
                "components": int(arr.GetNumberOfComponents()),
                "tuples": int(arr.GetNumberOfTuples()),
            }
        )
    return arrays


def inspect_vtp_arrays(polydata: vtk.vtkPolyData) -> dict[str, Any]:
    return {
        "point_data": _attribute_arrays(polydata.GetPointData()),
        "cell_data": _attribute_arrays(polydata.GetCellData()),
        "field_data": _attribute_arrays(polydata.GetFieldData()),
        "point_count": int(polydata.GetNumberOfPoints()),
        "cell_count": int(polydata.GetNumberOfCells()),
        "line_count": int(polydata.GetNumberOfLines()),
        "polygon_count": int(polydata.GetNumberOfPolys()),
    }


def _array_value(arr: Any, idx: int) -> Any:
    if arr is None:
        return None
    if arr.IsA("vtkStringArray"):
        return arr.GetValue(int(idx))
    if arr.GetNumberOfComponents() == 1:
        value = arr.GetVariantValue(int(idx))
        text = value.ToString()
        number = _finite_float(text)
        return number if number is not None else text
    return tuple(float(v) for v in arr.GetTuple(int(idx)))


def _string_like_arrays(attrs: Any, expected_tuples: int) -> list[Any]:
    preferred: list[Any] = []
    fallback: list[Any] = []
    for idx in range(attrs.GetNumberOfArrays()):
        arr = attrs.GetAbstractArray(idx)
        if arr is None or int(arr.GetNumberOfTuples()) != int(expected_tuples):
            continue
        name = str(arr.GetName() or "")
        lowered = name.lower()
        if arr.GetNumberOfComponents() != 1:
            continue
        if lowered in {"segmentname", "segment_name", "vesselname", "vessel_name", "name", "label"}:
            preferred.append(arr)
        elif "name" in lowered or "label" in lowered:
            fallback.append(arr)
    return preferred + fallback


def _numeric_array(attrs: Any, name: str, expected_tuples: int) -> Optional[Any]:
    arr = attrs.GetArray(name)
    if arr is None or int(arr.GetNumberOfTuples()) != int(expected_tuples):
        return None
    return arr


def _segment_ids_for_name(step3_contract: dict[str, Any], segment_name: str) -> list[int]:
    target = _normalize_name(segment_name)
    ids: set[int] = set()
    for key, value in (step3_contract.get("segment_name_map") or {}).items():
        if _normalize_name(value) == target:
            try:
                ids.add(int(key))
            except Exception:
                pass
    row = (step3_contract.get("vessel_priority_classification") or {}).get(segment_name)
    if isinstance(row, dict) and row.get("segment_id") is not None:
        try:
            ids.add(int(row["segment_id"]))
        except Exception:
            pass
    return sorted(ids)


def _cell_matches_point_name_array(polydata: vtk.vtkPolyData, arr: Any, cell_id: int, target_name: str) -> bool:
    cell = polydata.GetCell(int(cell_id))
    point_ids = cell.GetPointIds()
    matches = 0
    total = point_ids.GetNumberOfIds()
    for idx in range(total):
        if _normalize_name(_array_value(arr, point_ids.GetId(idx))) == _normalize_name(target_name):
            matches += 1
    return total > 0 and matches >= max(1, total // 2)


def surface_cell_ids_by_segment_name(
    surface: vtk.vtkPolyData,
    segment_name: str,
    step3_contract: dict[str, Any],
) -> tuple[list[int], str]:
    n_cells = int(surface.GetNumberOfCells())
    cell_data = surface.GetCellData()
    for arr in _string_like_arrays(cell_data, n_cells):
        matches = [
            int(cell_id)
            for cell_id in range(n_cells)
            if _normalize_name(_array_value(arr, cell_id)) == _normalize_name(segment_name)
        ]
        if matches:
            return matches, f"cell_data.{arr.GetName()}"

    segment_ids = _segment_ids_for_name(step3_contract, segment_name)
    if segment_ids:
        segment_id_arr = _numeric_array(cell_data, "SegmentId", n_cells)
        if segment_id_arr is not None:
            target_ids = set(int(v) for v in segment_ids)
            matches = [
                int(cell_id)
                for cell_id in range(n_cells)
                if int(round(float(_array_value(segment_id_arr, cell_id)))) in target_ids
            ]
            if matches:
                return matches, "cell_data.SegmentId+step3_contract.segment_name_map"

    point_data = surface.GetPointData()
    for arr in _string_like_arrays(point_data, int(surface.GetNumberOfPoints())):
        matches = [
            int(cell_id)
            for cell_id in range(n_cells)
            if _cell_matches_point_name_array(surface, arr, cell_id, segment_name)
        ]
        if matches:
            return matches, f"point_data.{arr.GetName()} majority"

    return [], "not_found"


def extract_named_surface_cells(
    surface: vtk.vtkPolyData,
    segment_name: str,
    step3_contract: dict[str, Any],
) -> tuple[vtk.vtkPolyData, list[int], str]:
    cell_ids, source = surface_cell_ids_by_segment_name(surface, segment_name, step3_contract)
    if not cell_ids:
        empty = vtk.vtkPolyData()
        return empty, [], source

    ids = vtk.vtkIdList()
    for cell_id in cell_ids:
        ids.InsertNextId(int(cell_id))

    extract = vtk.vtkExtractCells()
    extract.SetInputData(surface)
    extract.SetCellList(ids)
    extract.Update()

    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(extract.GetOutputPort())
    geometry.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(geometry.GetOutputPort())
    cleaner.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    return out, cell_ids, source


def _polyline_points_for_cell(polydata: vtk.vtkPolyData, cell_id: int) -> Optional[np.ndarray]:
    cell = polydata.GetCell(int(cell_id))
    if cell is None or cell.GetNumberOfPoints() < 2:
        return None
    ids = cell.GetPointIds()
    pts = np.asarray([polydata.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
    if pts.shape[0] < 2:
        return None
    return pts


def _polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(cumulative_arclength(points)[-1])


def extract_named_centerline(
    centerlines: vtk.vtkPolyData,
    segment_name: str,
    step3_contract: dict[str, Any],
) -> tuple[Optional[np.ndarray], str]:
    n_cells = int(centerlines.GetNumberOfCells())
    target = _normalize_name(segment_name)
    candidates: list[tuple[float, int, np.ndarray, str]] = []

    for arr in _string_like_arrays(centerlines.GetCellData(), n_cells):
        for cell_id in range(n_cells):
            if _normalize_name(_array_value(arr, cell_id)) != target:
                continue
            pts = _polyline_points_for_cell(centerlines, cell_id)
            if pts is not None:
                candidates.append((_polyline_length(pts), int(cell_id), pts, f"cell_data.{arr.GetName()}"))
        if candidates:
            candidates.sort(key=lambda row: row[0], reverse=True)
            return candidates[0][2], candidates[0][3]

    segment_ids = _segment_ids_for_name(step3_contract, segment_name)
    if segment_ids:
        arr = _numeric_array(centerlines.GetCellData(), "SegmentId", n_cells)
        if arr is not None:
            target_ids = set(int(v) for v in segment_ids)
            for cell_id in range(n_cells):
                if int(round(float(_array_value(arr, cell_id)))) not in target_ids:
                    continue
                pts = _polyline_points_for_cell(centerlines, cell_id)
                if pts is not None:
                    candidates.append(
                        (
                            _polyline_length(pts),
                            int(cell_id),
                            pts,
                            "cell_data.SegmentId+step3_contract.segment_name_map",
                        )
                    )
            if candidates:
                candidates.sort(key=lambda row: row[0], reverse=True)
                return candidates[0][2], candidates[0][3]

    for arr in _string_like_arrays(centerlines.GetPointData(), int(centerlines.GetNumberOfPoints())):
        for cell_id in range(n_cells):
            if not _cell_matches_point_name_array(centerlines, arr, cell_id, segment_name):
                continue
            pts = _polyline_points_for_cell(centerlines, cell_id)
            if pts is not None:
                candidates.append((_polyline_length(pts), int(cell_id), pts, f"point_data.{arr.GetName()} majority"))
        if candidates:
            candidates.sort(key=lambda row: row[0], reverse=True)
            return candidates[0][2], candidates[0][3]

    return None, "not_found"


def project_point_to_polyline(point: Iterable[float], polyline: np.ndarray) -> Projection:
    p = np.asarray(point, dtype=float).reshape(3)
    pts = np.asarray(polyline, dtype=float)
    if pts.shape[0] == 0:
        return Projection(0.0, np.zeros(3, dtype=float), float("inf"), -1)
    if pts.shape[0] == 1:
        return Projection(0.0, pts[0].copy(), distance(p, pts[0]), 0)

    cumulative = cumulative_arclength(pts)
    best_s = 0.0
    best_point = pts[0].copy()
    best_dist = float("inf")
    best_segment = 0
    for idx in range(pts.shape[0] - 1):
        a = pts[idx]
        b = pts[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1.0e-12:
            t = 0.0
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        q = a + t * ab
        d = distance(p, q)
        if d < best_dist:
            best_dist = d
            best_point = q
            best_s = float(cumulative[idx] + t * distance(a, b))
            best_segment = int(idx)
    return Projection(best_s, best_point, best_dist, best_segment)


def _nested_point(row: dict[str, Any], paths: list[tuple[str, ...]]) -> Optional[np.ndarray]:
    for path in paths:
        value: Any = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        point = _valid_point(value)
        if point is not None:
            return point
    return None


def _segment_start_point_from_contract(step3_contract: dict[str, Any], segment_name: str) -> Optional[np.ndarray]:
    row = (step3_contract.get("proximal_start_metadata") or {}).get(segment_name)
    if isinstance(row, dict):
        point = _nested_point(
            row,
            [
                ("boundary_profile", "boundary_centroid"),
                ("boundary_profile", "centroid"),
                ("centerline_anchor_point",),
            ],
        )
        if point is not None:
            return point
    landmark_row = ((step3_contract.get("landmark_registry") or {}).get("segment_landmarks") or {}).get(segment_name)
    if isinstance(landmark_row, dict):
        point = _nested_point(landmark_row, [("proximal_point",), ("proximal_boundary", "centroid")])
        if point is not None:
            return point
    return None


def _segment_start_point_from_centerline(
    centerlines: vtk.vtkPolyData,
    segment_name: str,
    step3_contract: dict[str, Any],
) -> Optional[np.ndarray]:
    pts, _ = extract_named_centerline(centerlines, segment_name, step3_contract)
    if pts is not None and pts.shape[0] >= 1:
        return np.asarray(pts[0], dtype=float)
    return None


def _aortic_bifurcation_point_from_contract(step3_contract: dict[str, Any]) -> Optional[np.ndarray]:
    landmark = (step3_contract.get("landmark_registry") or {}).get("aorta_end_pre_bifurcation")
    if isinstance(landmark, dict):
        point = _nested_point(
            landmark,
            [
                ("centerline_landmark", "point"),
                ("bifurcation_point",),
                ("boundary_centroid",),
            ],
        )
        if point is not None:
            return point

    row = (step3_contract.get("distal_end_metadata") or {}).get("abdominal_aorta_trunk")
    if isinstance(row, dict):
        point = _nested_point(row, [("centerline_anchor_point",), ("boundary_profile", "boundary_centroid")])
        if point is not None:
            return point
    return None


def resolve_landmarks(
    centerlines: vtk.vtkPolyData,
    step3_contract: dict[str, Any],
    aorta_centerline: Optional[np.ndarray],
    warnings: list[str],
    not_available: list[str],
) -> dict[str, Any]:
    landmarks = {
        "lowest_renal_artery_name": None,
        "lowest_renal_ostium_center_xyz": None,
        "lowest_renal_aortic_centerline_s_mm": None,
        "aortic_bifurcation_center_xyz": None,
        "aortic_bifurcation_s_mm": None,
    }
    if aorta_centerline is None or aorta_centerline.shape[0] < 2:
        _append_unique(warnings, "W_STEP4_AORTA_CENTERLINE_MISSING: abdominal_aorta_trunk centerline could not be resolved.")
        _append_unique(not_available, "abdominal_aorta_trunk_centerline")
        return landmarks

    renal_rows: list[dict[str, Any]] = []
    for renal_name in ("left_renal_artery", "right_renal_artery"):
        ostium = _segment_start_point_from_contract(step3_contract, renal_name)
        source = "step3_contract.proximal_start_metadata"
        if ostium is None:
            ostium = _segment_start_point_from_centerline(centerlines, renal_name, step3_contract)
            source = "named_centerlines.first_point"
        if ostium is None:
            _append_unique(warnings, f"W_STEP4_RENAL_OSTIUM_MISSING: {renal_name} proximal ostium center could not be resolved.")
            _append_unique(not_available, f"{renal_name}_ostium_center")
            continue
        projection = project_point_to_polyline(ostium, aorta_centerline)
        renal_rows.append(
            {
                "name": renal_name,
                "ostium": ostium,
                "s": float(projection.s),
                "projection_distance": float(projection.distance),
                "source": source,
            }
        )

    if renal_rows:
        lowest = max(renal_rows, key=lambda row: float(row["s"]))
        landmarks["lowest_renal_artery_name"] = str(lowest["name"])
        landmarks["lowest_renal_ostium_center_xyz"] = _round_point(lowest["ostium"])
        landmarks["lowest_renal_aortic_centerline_s_mm"] = _round_float(lowest["s"])
        if float(lowest["projection_distance"]) > 3.0:
            _append_unique(
                warnings,
                "W_STEP4_RENAL_PROJECTION_DISTANCE_REVIEW: renal ostium projection to abdominal_aorta_trunk centerline exceeded 3 mm.",
            )
    else:
        _append_unique(warnings, "W_STEP4_RENAL_LANDMARKS_UNRESOLVED: neither renal artery ostium could be resolved.")

    bif_point = _aortic_bifurcation_point_from_contract(step3_contract)
    if bif_point is None:
        bif_point = np.asarray(aorta_centerline[-1], dtype=float)
    bif_projection = project_point_to_polyline(bif_point, aorta_centerline)
    landmarks["aortic_bifurcation_center_xyz"] = _round_point(bif_projection.point)
    landmarks["aortic_bifurcation_s_mm"] = _round_float(bif_projection.s)

    if renal_rows and bif_projection.s <= max(row["s"] for row in renal_rows):
        _append_unique(
            warnings,
            "W_STEP4_AORTA_ORIENTATION_REVIEW: aortic bifurcation arclength was not downstream of renal ostia.",
        )

    return landmarks


def cut_surface_with_plane(surface: vtk.vtkPolyData, origin: np.ndarray, normal: np.ndarray) -> vtk.vtkPolyData:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)
    cutter.SetValue(0, 0.0)
    cutter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(cutter.GetOutputPort())
    cleaner.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cleaner.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(stripper.GetOutput())
    return out


def extract_contours(cut_polydata: vtk.vtkPolyData) -> list[np.ndarray]:
    contours: list[np.ndarray] = []
    for cell_id in range(cut_polydata.GetNumberOfCells()):
        cell = cut_polydata.GetCell(cell_id)
        if cell is None or cell.GetNumberOfPoints() < 3:
            continue
        ids = cell.GetPointIds()
        pts = np.asarray([cut_polydata.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
        if pts.shape[0] >= 3:
            contours.append(pts)
    return contours


def compute_contour_area(points: np.ndarray) -> float:
    area, _, _ = polygon_area_normal(points)
    return float(area)


def compute_contour_perimeter(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    perimeter = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    perimeter += distance(pts[-1], pts[0])
    return perimeter if math.isfinite(perimeter) else 0.0


def compute_major_diameter(points: np.ndarray, normal: np.ndarray) -> Optional[float]:
    major, minor = projected_major_minor_diameters(points, normal)
    if major is None or minor is None:
        return None
    return float(max(major, minor))


def compute_minor_diameter(points: np.ndarray, normal: np.ndarray) -> Optional[float]:
    major, minor = projected_major_minor_diameters(points, normal)
    if major is None or minor is None:
        return None
    return float(min(major, minor))


def compute_equivalent_diameter(area: float) -> Optional[float]:
    return equivalent_diameter_from_area(area)


def choose_contour_closest_to_centerline_origin(contours: list[np.ndarray], origin: np.ndarray, normal: np.ndarray) -> Optional[dict[str, Any]]:
    ranked: list[tuple[float, float, dict[str, Any]]] = []
    for contour in contours:
        area = compute_contour_area(contour)
        if not math.isfinite(area) or area <= 1.0e-8:
            continue
        center = np.mean(contour, axis=0)
        major = compute_major_diameter(contour, normal)
        minor = compute_minor_diameter(contour, normal)
        equivalent = compute_equivalent_diameter(area)
        if major is None or minor is None or equivalent is None:
            continue
        ranked.append(
            (
                distance(center, origin),
                -area,
                {
                    "points": contour,
                    "center": center,
                    "area": float(area),
                    "perimeter": compute_contour_perimeter(contour),
                    "major": float(major),
                    "minor": float(minor),
                    "equivalent": float(equivalent),
                },
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda row: (row[0], row[1]))
    return ranked[0][2]


def measure_section(
    surface: vtk.vtkPolyData,
    centerline: np.ndarray,
    section_s: float,
    *,
    max_s: Optional[float] = None,
) -> Optional[SectionMeasurement]:
    total = float(cumulative_arclength(centerline)[-1]) if centerline.shape[0] >= 2 else 0.0
    if total <= 0.0:
        return None
    if section_s < 0.0 or section_s > total:
        return None
    if max_s is not None and section_s >= float(max_s):
        return None

    center = point_at_arclength(centerline, float(section_s))
    tangent = tangent_at_arclength(centerline, float(section_s), window=1.0)
    if float(np.linalg.norm(tangent)) <= 1.0e-12:
        return None

    cut = cut_surface_with_plane(surface, center, tangent)
    contour = choose_contour_closest_to_centerline_origin(extract_contours(cut), center, tangent)
    if contour is None:
        return None
    return SectionMeasurement(
        s=float(section_s),
        center=center,
        tangent=tangent,
        contour_center=np.asarray(contour["center"], dtype=float),
        area=float(contour["area"]),
        perimeter=float(contour["perimeter"]),
        major_diameter=float(contour["major"]),
        minor_diameter=float(contour["minor"]),
        equivalent_diameter=float(contour["equivalent"]),
        contour_point_count=int(np.asarray(contour["points"]).shape[0]),
    )


def _section_into_contract(contract: dict[str, Any], label: str, section: Optional[SectionMeasurement]) -> None:
    prefix = f"neck_diameter_{label}"
    if section is None:
        contract["aortic_neck"][f"{prefix}_major_mm"] = None
        contract["aortic_neck"][f"{prefix}_minor_mm"] = None
        contract["aortic_neck"][f"{prefix}_equivalent_mm"] = None
        return
    contract["aortic_neck"][f"{prefix}_major_mm"] = _round_float(section.major_diameter)
    contract["aortic_neck"][f"{prefix}_minor_mm"] = _round_float(section.minor_diameter)
    contract["aortic_neck"][f"{prefix}_equivalent_mm"] = _round_float(section.equivalent_diameter)


def _expansion_trigger(baseline: float, diameter: float, delta_s: float) -> bool:
    increase = float(diameter) - float(baseline)
    return (
        increase > 0.10 * float(baseline)
        or (delta_s <= 10.0 and increase > 2.0)
        or (delta_s <= 15.0 and increase > 3.0)
    )


def _find_neck_end(
    scan_sections: list[SectionMeasurement],
    d0_s: float,
    d0_major: float,
) -> tuple[Optional[SectionMeasurement], bool]:
    for idx, section in enumerate(scan_sections):
        if section.major_diameter is None:
            continue
        if not _expansion_trigger(d0_major, float(section.major_diameter), float(section.s) - d0_s):
            continue
        downstream = scan_sections[idx + 1 : idx + 3]
        if not downstream:
            return section, True
        persistent = True
        for downstream_section in downstream:
            if downstream_section.major_diameter is None:
                persistent = False
                break
            if not _expansion_trigger(
                d0_major,
                float(downstream_section.major_diameter),
                float(downstream_section.s) - d0_s,
            ):
                persistent = False
                break
        if persistent:
            return section, True
    if scan_sections:
        return scan_sections[-1], False
    return None, False


def compute_vector_angle_degrees(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    ua = unit(np.asarray(a, dtype=float))
    ub = unit(np.asarray(b, dtype=float))
    if float(np.linalg.norm(ua)) <= 1.0e-12 or float(np.linalg.norm(ub)) <= 1.0e-12:
        return None
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _angulation_category(angle: Optional[float]) -> str:
    if angle is None:
        return "not_available"
    if angle <= 60.0:
        return "<=60_deg"
    if angle <= 90.0:
        return ">60_to_90_deg"
    return ">90_deg"


def measure_aortic_neck(
    contract: dict[str, Any],
    aorta_surface: vtk.vtkPolyData,
    aorta_centerline: Optional[np.ndarray],
    warnings: list[str],
    not_available: list[str],
) -> None:
    statuses = contract["measurement_status"]
    landmarks = contract["landmarks"]
    renal_s = _finite_float(landmarks.get("lowest_renal_aortic_centerline_s_mm"))
    bif_s = _finite_float(landmarks.get("aortic_bifurcation_s_mm"))
    if aorta_centerline is None or aorta_centerline.shape[0] < 2:
        statuses["aortic_neck_diameter_measurement_status"] = STATUS_NOT_AVAILABLE
        statuses["aortic_neck_length_measurement_status"] = STATUS_NOT_AVAILABLE
        statuses["aortic_neck_angulation_measurement_status"] = STATUS_NOT_AVAILABLE
        _append_unique(not_available, "aortic_neck")
        return
    if renal_s is None or bif_s is None:
        statuses["aortic_neck_diameter_measurement_status"] = STATUS_NOT_AVAILABLE
        statuses["aortic_neck_length_measurement_status"] = STATUS_NOT_AVAILABLE
        statuses["aortic_neck_angulation_measurement_status"] = STATUS_NOT_AVAILABLE
        _append_unique(not_available, "aortic_neck_landmarks")
        return
    if aorta_surface.GetNumberOfCells() <= 0:
        statuses["aortic_neck_diameter_measurement_status"] = STATUS_NOT_AVAILABLE
        statuses["aortic_neck_length_measurement_status"] = STATUS_NOT_AVAILABLE
        statuses["aortic_neck_angulation_measurement_status"] = STATUS_NOT_AVAILABLE
        _append_unique(not_available, "abdominal_aorta_trunk_surface")
        _append_unique(warnings, "W_STEP4_AORTA_SURFACE_MISSING: abdominal_aorta_trunk surface cells could not be resolved.")
        return

    section_targets = {
        "D0": renal_s + 1.0,
        "D10": renal_s + 10.0,
        "D15": renal_s + 15.0,
    }
    sections: dict[str, Optional[SectionMeasurement]] = {}
    for label, section_s in section_targets.items():
        if section_s >= bif_s:
            sections[label] = None
            _append_unique(warnings, f"W_STEP4_{label}_BEYOND_BIFURCATION: aortic neck {label} section is not before bifurcation.")
            continue
        try:
            sections[label] = measure_section(aorta_surface, aorta_centerline, section_s, max_s=bif_s)
        except Exception as exc:
            sections[label] = None
            _append_unique(warnings, f"W_STEP4_{label}_SECTION_FAILED: failed to measure aortic neck {label} section ({exc}).")

    for label in ("D0", "D10", "D15"):
        _section_into_contract(contract, label, sections.get(label))

    available_major = [
        (label, section.major_diameter)
        for label, section in sections.items()
        if section is not None and section.major_diameter is not None
    ]
    if available_major:
        max_value = max(float(value) for _, value in available_major)
        max_labels = [label for label, value in available_major if abs(float(value) - max_value) <= 1.0e-9]
        contract["aortic_neck"]["neck_reference_diameter_mm"] = _round_float(max_value)
        contract["aortic_neck"]["neck_reference_diameter_source"] = (
            f"{max_labels[0]}_major" if len(max_labels) == 1 else "max_major_D0_D10_D15"
        )
        statuses["aortic_neck_diameter_measurement_status"] = (
            STATUS_MEASURED if all(sections.get(label) is not None for label in ("D0", "D10", "D15")) else STATUS_REQUIRES_REVIEW
        )
    else:
        statuses["aortic_neck_diameter_measurement_status"] = STATUS_FAILED
        statuses["aortic_neck_length_measurement_status"] = STATUS_FAILED
        statuses["aortic_neck_angulation_measurement_status"] = STATUS_NOT_AVAILABLE
        _append_unique(warnings, "W_STEP4_AORTIC_NECK_DIAMETERS_FAILED: no aortic neck diameter section could be measured.")
        return

    d0 = sections.get("D0")
    if d0 is None or d0.major_diameter is None:
        statuses["aortic_neck_length_measurement_status"] = STATUS_FAILED
        _append_unique(warnings, "W_STEP4_NECK_END_FAILED: D0 baseline diameter was unavailable.")
        return

    d0_s = renal_s + 1.0
    scan_sections: list[SectionMeasurement] = []
    scan_end = max(d0_s, bif_s - 0.1)
    for section_s in np.arange(d0_s, scan_end + 1.0e-6, 1.0):
        try:
            section = measure_section(aorta_surface, aorta_centerline, float(section_s), max_s=bif_s)
        except Exception:
            section = None
        if section is not None and section.major_diameter is not None:
            scan_sections.append(section)

    neck_end, detected = _find_neck_end(scan_sections, d0_s, float(d0.major_diameter))
    if neck_end is None:
        statuses["aortic_neck_length_measurement_status"] = STATUS_FAILED
        _append_unique(warnings, "W_STEP4_NECK_END_FAILED: no valid downstream aortic neck sections were available.")
        return

    contract["aortic_neck"]["neck_end_s_mm"] = _round_float(neck_end.s)
    contract["aortic_neck"]["neck_end_center_xyz"] = _round_point(neck_end.center)
    contract["aortic_neck"]["neck_length_mm"] = _round_float(float(neck_end.s) - renal_s)
    statuses["aortic_neck_length_measurement_status"] = STATUS_MEASURED if detected else STATUS_REQUIRES_REVIEW
    if not detected:
        _append_unique(
            warnings,
            "W_STEP4_NECK_END_REQUIRES_REVIEW: no clear sustained expansion threshold was detected before bifurcation; last valid scanned section was used.",
        )

    try:
        proximal_target = renal_s + 15.0 if renal_s + 15.0 < bif_s else renal_s + 10.0
        if proximal_target >= bif_s:
            proximal_target = min(bif_s, renal_s + max(1.0, bif_s - renal_s) * 0.5)
        vector_a = point_at_arclength(aorta_centerline, proximal_target) - point_at_arclength(aorta_centerline, renal_s)

        distal_target = min(float(neck_end.s) + 30.0, bif_s)
        if distal_target - float(neck_end.s) < 20.0:
            _append_unique(
                warnings,
                "W_STEP4_ANGULATION_SHORT_DISTAL_VECTOR: less than 20 mm was available distal to neck_end for angulation.",
            )
        if distal_target <= float(neck_end.s):
            vector_b = tangent_at_arclength(aorta_centerline, float(neck_end.s), window=2.0)
        else:
            vector_b = point_at_arclength(aorta_centerline, distal_target) - point_at_arclength(aorta_centerline, float(neck_end.s))
        angle = compute_vector_angle_degrees(vector_a, vector_b)
        contract["aortic_neck"]["proximal_neck_angulation_deg"] = _round_float(angle)
        contract["aortic_neck"]["proximal_neck_angulation_category"] = _angulation_category(angle)
        statuses["aortic_neck_angulation_measurement_status"] = STATUS_MEASURED if angle is not None else STATUS_NOT_AVAILABLE
    except Exception as exc:
        statuses["aortic_neck_angulation_measurement_status"] = STATUS_FAILED
        contract["aortic_neck"]["proximal_neck_angulation_category"] = "not_available"
        _append_unique(warnings, f"W_STEP4_ANGULATION_FAILED: proximal neck angulation measurement failed ({exc}).")


def measure_iliac_side(
    side: str,
    contract: dict[str, Any],
    full_surface: vtk.vtkPolyData,
    centerlines: vtk.vtkPolyData,
    step3_contract: dict[str, Any],
    warnings: list[str],
    not_available: list[str],
) -> None:
    side_key = side.lower()
    assert side_key in {"left", "right"}
    diameter_status_key = f"{side_key}_iliac_diameter_measurement_status"
    seal_status_key = f"{side_key}_iliac_seal_zone_measurement_status"
    row = contract["iliac"][side_key]
    statuses = contract["measurement_status"]

    candidate_segments = [f"{side_key}_common_iliac", f"{side_key}_external_iliac"]
    selected_name: Optional[str] = None
    selected_centerline: Optional[np.ndarray] = None
    selected_surface: Optional[vtk.vtkPolyData] = None

    for segment_name in candidate_segments:
        pts, _ = extract_named_centerline(centerlines, segment_name, step3_contract)
        if pts is None or pts.shape[0] < 2:
            continue
        surface, cell_ids, _ = extract_named_surface_cells(full_surface, segment_name, step3_contract)
        if not cell_ids or surface.GetNumberOfCells() <= 0:
            continue
        selected_name = segment_name
        selected_centerline = pts
        selected_surface = surface
        break

    if selected_name is None or selected_centerline is None or selected_surface is None:
        statuses[diameter_status_key] = STATUS_NOT_AVAILABLE
        statuses[seal_status_key] = STATUS_NOT_AVAILABLE
        _append_unique(not_available, f"{side_key}_iliac_landing_segment")
        _append_unique(warnings, f"W_STEP4_{side_key.upper()}_ILIAC_SEGMENT_MISSING: no measurable common or external iliac segment was found.")
        return

    total_length = float(cumulative_arclength(selected_centerline)[-1])
    if total_length <= 0.0:
        statuses[diameter_status_key] = STATUS_FAILED
        statuses[seal_status_key] = STATUS_FAILED
        _append_unique(warnings, f"W_STEP4_{side_key.upper()}_ILIAC_CENTERLINE_INVALID: selected iliac centerline has non-positive length.")
        return

    seal_length = min(20.0, total_length)
    start_s = max(0.0, total_length - seal_length)
    end_s = total_length
    row["selected_landing_segment"] = selected_name
    row["distal_iliac_seal_zone_length_mm"] = _round_float(seal_length)
    row["distal_iliac_seal_zone_start_xyz"] = _round_point(point_at_arclength(selected_centerline, start_s))
    row["distal_iliac_seal_zone_end_xyz"] = _round_point(point_at_arclength(selected_centerline, end_s))

    sample_count = max(3, int(math.floor(seal_length / 2.0)) + 1)
    sample_positions = np.linspace(start_s, end_s, sample_count)
    diameters: list[tuple[float, float]] = []
    failures = 0
    for section_s in sample_positions:
        try:
            section = measure_section(selected_surface, selected_centerline, float(section_s))
        except Exception:
            section = None
        if section is None or section.major_diameter is None:
            failures += 1
            continue
        diameters.append((float(section_s), float(section.major_diameter)))

    if not diameters:
        statuses[diameter_status_key] = STATUS_FAILED
        statuses[seal_status_key] = STATUS_REQUIRES_REVIEW if seal_length < 10.0 else STATUS_MEASURED
        _append_unique(warnings, f"W_STEP4_{side_key.upper()}_ILIAC_DIAMETER_FAILED: selected segment {selected_name} produced no measurable seal-zone sections.")
        return

    _, max_diameter = max(diameters, key=lambda row_value: row_value[1])
    row["iliac_treatment_diameter_mm"] = _round_float(max_diameter)
    row["iliac_reference_diameter_source"] = f"{selected_name}_distal_seal_zone_max_major"

    status = STATUS_MEASURED
    if seal_length < 10.0:
        status = STATUS_REQUIRES_REVIEW
        _append_unique(
            warnings,
            f"W_STEP4_{side_key.upper()}_ILIAC_SEAL_ZONE_SHORT: selected {selected_name} has less than 10 mm available distal seal zone.",
        )
    elif failures > 0:
        status = STATUS_REQUIRES_REVIEW
        _append_unique(
            warnings,
            f"W_STEP4_{side_key.upper()}_ILIAC_SECTION_REVIEW: {failures} sampled section(s) failed in {selected_name}.",
        )

    statuses[diameter_status_key] = status
    statuses[seal_status_key] = status


def _remove_existing_cell_arrays(polydata: vtk.vtkPolyData, names: Iterable[str]) -> None:
    cell_data = polydata.GetCellData()
    for name in names:
        while cell_data.HasArray(str(name)):
            cell_data.RemoveArray(str(name))


def _add_string_cell_array(polydata: vtk.vtkPolyData, name: str, values: Iterable[str]) -> None:
    vals = [str(value) for value in values]
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    arr.SetNumberOfValues(len(vals))
    for idx, value in enumerate(vals):
        arr.SetValue(idx, value)
    polydata.GetCellData().AddArray(arr)


def copy_full_polydata_and_add_step4_cell_arrays(
    full_surface: vtk.vtkPolyData,
    region_ids: list[int],
    mask_values: list[int],
    colors: list[tuple[int, int, int]],
    region_names: list[str],
) -> vtk.vtkPolyData:
    out = vtk.vtkPolyData()
    out.DeepCopy(full_surface)
    _remove_existing_cell_arrays(out, ["Step4RegionId", "InfrarenalNeckMask", "Step4ColorRGB", "Step4RegionName"])
    add_int_cell_array(out, "Step4RegionId", region_ids)
    add_int_cell_array(out, "InfrarenalNeckMask", mask_values)
    add_uchar3_cell_array(out, "Step4ColorRGB", colors)
    _add_string_cell_array(out, "Step4RegionName", region_names)
    return out


def write_infrarenal_neck_labeled_vtp(
    full_surface: vtk.vtkPolyData,
    output_path: Path,
    aorta_cell_ids: list[int],
    lowest_renal_s: Optional[float],
    neck_end_s: Optional[float],
    aorta_centerline: Optional[np.ndarray],
    warnings: list[str],
) -> None:
    n_cells = int(full_surface.GetNumberOfCells())
    region_ids = [OTHER_REGION_ID] * n_cells
    mask_values = [0] * n_cells
    colors = [OTHER_COLOR] * n_cells
    region_names = ["other_model_surface"] * n_cells

    can_label = (
        aorta_centerline is not None
        and aorta_centerline.shape[0] >= 2
        and aorta_cell_ids
        and lowest_renal_s is not None
        and neck_end_s is not None
        and float(neck_end_s) > float(lowest_renal_s)
    )
    if not can_label:
        _append_unique(
            warnings,
            "W_STEP4_NECK_LABEL_UNAVAILABLE: labeled VTP was written with all InfrarenalNeckMask values set to 0.",
        )
    else:
        centers = cell_centers(full_surface)
        aorta_set = set(int(v) for v in aorta_cell_ids)
        lower = float(lowest_renal_s)
        upper = float(neck_end_s)
        for cell_id in aorta_set:
            if cell_id < 0 or cell_id >= n_cells or cell_id >= centers.shape[0]:
                continue
            projection = project_point_to_polyline(centers[cell_id], aorta_centerline)
            if lower <= float(projection.s) <= upper:
                region_ids[cell_id] = NECK_REGION_ID
                mask_values[cell_id] = 1
                colors[cell_id] = NECK_COLOR
                region_names[cell_id] = "infrarenal_neck"

    labeled = copy_full_polydata_and_add_step4_cell_arrays(full_surface, region_ids, mask_values, colors, region_names)
    write_vtp(labeled, output_path)


def _set_overall_status(contract: dict[str, Any]) -> None:
    statuses = [
        value
        for key, value in contract["measurement_status"].items()
        if key != "overall_geometry_measurement_status" and value is not None
    ]
    if not statuses:
        overall = STATUS_NOT_AVAILABLE
    elif any(value == STATUS_FAILED for value in statuses):
        overall = STATUS_FAILED
    elif any(value == STATUS_NOT_AVAILABLE for value in statuses):
        overall = STATUS_NOT_AVAILABLE
    elif any(value == STATUS_REQUIRES_REVIEW for value in statuses):
        overall = STATUS_REQUIRES_REVIEW
    else:
        overall = STATUS_MEASURED
    contract["measurement_status"]["overall_geometry_measurement_status"] = overall
    contract["metadata"]["measurement_confidence"] = {
        STATUS_MEASURED: "high",
        STATUS_REQUIRES_REVIEW: "moderate",
        STATUS_NOT_AVAILABLE: "low",
        STATUS_FAILED: "low",
    }.get(overall, "low")


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    project_root = Path(args.project_root).resolve()
    paths = build_pipeline_paths(project_root)
    step3_dir = Path(args.step3_dir).resolve() if args.step3_dir else paths.step3_dir
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths.step4_dir
    named_surface = Path(args.named_surface).resolve() if args.named_surface else step3_dir / "named_segmentscolored.vtp"
    named_centerlines = Path(args.named_centerlines).resolve() if args.named_centerlines else step3_dir / "named_centerlines.vtp"
    step3_contract = Path(args.step3_contract).resolve() if args.step3_contract else step3_dir / "step3_naming_orientation_contract.json"
    output_json = Path(args.output_json).resolve() if args.output_json else output_dir / "step4_geometry_measurements.json"
    output_labeled_vtp = (
        Path(args.output_labeled_vtp).resolve()
        if args.output_labeled_vtp
        else output_dir / "step4_infrarenal_neck_labeled.vtp"
    )
    return {
        "project_root": project_root,
        "step3_dir": step3_dir,
        "output_dir": output_dir,
        "named_surface": named_surface,
        "named_centerlines": named_centerlines,
        "step3_contract": step3_contract,
        "output_json": output_json,
        "output_labeled_vtp": output_labeled_vtp,
    }


def run_step4(args: argparse.Namespace) -> dict[str, Any]:
    paths = _resolve_paths(args)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    contract = _make_contract(paths)
    warnings = contract["metadata"]["warnings"]
    not_available = contract["metadata"]["not_available"]

    full_surface = read_vtp(paths["named_surface"])
    centerlines = read_vtp(paths["named_centerlines"])
    step3_contract = read_json(paths["step3_contract"])
    contract["metadata"]["discovered_arrays"]["named_surface"] = inspect_vtp_arrays(full_surface)
    contract["metadata"]["discovered_arrays"]["named_centerlines"] = inspect_vtp_arrays(centerlines)
    contract["metadata"]["coordinate_system"] = (
        (step3_contract.get("step2_references") or {}).get("coordinate_system") or {"name": "source_model_coordinates", "units": "mm"}
    )
    contract["metadata"]["source_inputs"] = dict(contract["source_inputs"])
    contract["metadata"]["units"] = dict(contract["units"])

    aorta_centerline, _ = extract_named_centerline(centerlines, "abdominal_aorta_trunk", step3_contract)
    if aorta_centerline is None:
        _append_unique(warnings, "W_STEP4_AORTA_CENTERLINE_NOT_FOUND: abdominal_aorta_trunk was not found in named_centerlines.vtp.")
        _append_unique(not_available, "abdominal_aorta_trunk_centerline")

    aorta_surface, aorta_cell_ids, _ = extract_named_surface_cells(full_surface, "abdominal_aorta_trunk", step3_contract)
    if not aorta_cell_ids:
        _append_unique(warnings, "W_STEP4_AORTA_SURFACE_NOT_FOUND: abdominal_aorta_trunk was not found in named_segmentscolored.vtp.")
        _append_unique(not_available, "abdominal_aorta_trunk_surface")

    contract["landmarks"] = resolve_landmarks(centerlines, step3_contract, aorta_centerline, warnings, not_available)
    measure_aortic_neck(contract, aorta_surface, aorta_centerline, warnings, not_available)
    measure_iliac_side("left", contract, full_surface, centerlines, step3_contract, warnings, not_available)
    measure_iliac_side("right", contract, full_surface, centerlines, step3_contract, warnings, not_available)

    lowest_renal_s = _finite_float(contract["landmarks"].get("lowest_renal_aortic_centerline_s_mm"))
    neck_end_s = _finite_float(contract["aortic_neck"].get("neck_end_s_mm"))
    write_infrarenal_neck_labeled_vtp(
        full_surface,
        paths["output_labeled_vtp"],
        aorta_cell_ids,
        lowest_renal_s,
        neck_end_s,
        aorta_centerline,
        warnings,
    )

    _set_overall_status(contract)
    write_json(json_safe(contract), paths["output_json"])
    return contract


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    paths = build_pipeline_paths(project_root)
    parser = argparse.ArgumentParser(description="STEP4 geometry measurements and infrarenal neck labeling.")
    parser.add_argument("--project-root", default=str(project_root), help="Project root.")
    parser.add_argument("--step3-dir", default=str(paths.step3_dir), help="STEP3 output directory.")
    parser.add_argument("--output-dir", default=str(paths.step4_dir), help="STEP4 output directory.")
    parser.add_argument("--named-surface", default="", help="STEP3 named surface VTP.")
    parser.add_argument("--named-centerlines", default="", help="STEP3 named centerlines VTP.")
    parser.add_argument("--step3-contract", default="", help="STEP3 naming/orientation contract JSON.")
    parser.add_argument("--output-json", default="", help="STEP4 geometry measurements JSON.")
    parser.add_argument("--output-labeled-vtp", default="", help="STEP4 infrarenal neck labeled VTP.")
    parser.add_argument("--debug", action="store_true", help="Show traceback for unexpected errors.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    paths = _resolve_paths(args)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    required = {
        "named surface": paths["named_surface"],
        "named centerlines": paths["named_centerlines"],
        "STEP3 contract": paths["step3_contract"],
    }
    missing = [(name, path) for name, path in required.items() if not path.exists()]
    if missing:
        print("STEP4 missing required input file(s):")
        for name, path in missing:
            print(f"  - {name}: {path}")
        return 2

    try:
        contract = run_step4(args)
    except Exception as exc:
        if args.debug:
            raise
        failure_contract = _make_contract(paths)
        failure_contract["metadata"]["warnings"].append(f"W_STEP4_FAILED: {exc}")
        for key in failure_contract["measurement_status"]:
            failure_contract["measurement_status"][key] = STATUS_FAILED
        failure_contract["measurement_status"]["overall_geometry_measurement_status"] = STATUS_FAILED
        failure_contract["metadata"]["measurement_confidence"] = "low"
        write_json(failure_contract, paths["output_json"])
        print(f"STEP4 failed: {exc}")
        return 1

    status = contract.get("measurement_status", {}).get("overall_geometry_measurement_status")
    print(
        "STEP4 completed: "
        f"{status} | "
        f"json={paths['output_json']} | "
        f"labeled_vtp={paths['output_labeled_vtp']}"
    )
    return 1 if status == STATUS_FAILED else 0
