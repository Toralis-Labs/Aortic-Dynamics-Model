from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import vtk

from src.common.geometry import (
    EPS,
    cumulative_arclength,
    distance,
    equivalent_diameter_from_area,
    orthonormal_frame,
    point_at_arclength,
    polygon_area_normal,
    polyline_length,
    tangent_at_arclength,
    unit,
)
from src.common.json_io import read_json, write_json
from src.common.paths import build_pipeline_paths
from src.common.vtk_helpers import get_cell_array, points_to_numpy, read_vtp, write_vtp


STEP_NAME = "STEP4_EVAR_GEOMETRY_MEASUREMENTS"
MEASUREMENTS_JSON = "step4_measurements.json"
MEASUREMENT_REGIONS_VTP = "step4_evar_geometry_regions.vtp"

TOP_LEVEL_STATUSES = {"success", "requires_review", "failed"}
MEASUREMENT_STATUSES = {
    "measured",
    "derived_summary",
    "unmeasurable",
    "missing_required_landmark",
    "requires_review",
    "not_applicable",
}

AORTIC_NECK_FIELDS = (
    "proximal_neck_diameter_mm",
    "proximal_neck_major_diameter_mm",
    "proximal_neck_minor_diameter_mm",
    "proximal_neck_equivalent_diameter_mm",
    "infrarenal_aortic_neck_treatment_diameter_mm",
    "aortic_treatment_diameter_mm",
    "proximal_neck_length_mm",
    "proximal_neck_angulation_deg",
)

ILIAC_SUMMARY_FIELDS = (
    "iliac_treatment_diameter_mm",
    "left_iliac_treatment_diameter_mm",
    "right_iliac_treatment_diameter_mm",
    "distal_iliac_seal_zone_length_mm",
    "left_distal_iliac_seal_zone_length_mm",
    "right_distal_iliac_seal_zone_length_mm",
)

COMMON_ILIAC_FIELDS = (
    "common_iliac_diameter_mm",
    "left_common_iliac_diameter_mm",
    "right_common_iliac_diameter_mm",
    "common_iliac_length_mm",
    "left_common_iliac_length_mm",
    "right_common_iliac_length_mm",
)

EXTERNAL_ILIAC_FIELDS = (
    "external_iliac_treatment_diameter_mm",
    "left_external_iliac_treatment_diameter_mm",
    "right_external_iliac_treatment_diameter_mm",
    "external_iliac_seal_zone_length_mm",
    "left_external_iliac_seal_zone_length_mm",
    "right_external_iliac_seal_zone_length_mm",
)

INTERNAL_ILIAC_FIELDS = (
    "internal_iliac_treatment_diameter_mm",
    "left_internal_iliac_treatment_diameter_mm",
    "right_internal_iliac_treatment_diameter_mm",
    "internal_iliac_seal_zone_length_mm",
    "left_internal_iliac_seal_zone_length_mm",
    "right_internal_iliac_seal_zone_length_mm",
)

RENAL_TO_INTERNAL_ILIAC_FIELDS = (
    "renal_to_internal_iliac_length_mm",
    "left_renal_to_internal_iliac_length_mm",
    "right_renal_to_internal_iliac_length_mm",
)

ACCESS_FIELDS = (
    "access_vessel_min_diameter_mm",
    "left_access_vessel_min_diameter_mm",
    "right_access_vessel_min_diameter_mm",
    "access_vessel_tortuosity",
    "left_access_vessel_tortuosity",
    "right_access_vessel_tortuosity",
)

REQUIRED_FIELD_GROUPS = {
    "aortic_neck": AORTIC_NECK_FIELDS,
    "iliac_summary": ILIAC_SUMMARY_FIELDS,
    "common_iliac": COMMON_ILIAC_FIELDS,
    "external_iliac": EXTERNAL_ILIAC_FIELDS,
    "internal_iliac": INTERNAL_ILIAC_FIELDS,
    "renal_to_internal_iliac": RENAL_TO_INTERNAL_ILIAC_FIELDS,
    "access": ACCESS_FIELDS,
}

SUMMARY_FIELDS = {
    "iliac_treatment_diameter_mm": ("left_iliac_treatment_diameter_mm", "right_iliac_treatment_diameter_mm", "mm"),
    "distal_iliac_seal_zone_length_mm": (
        "left_distal_iliac_seal_zone_length_mm",
        "right_distal_iliac_seal_zone_length_mm",
        "mm",
    ),
    "common_iliac_diameter_mm": ("left_common_iliac_diameter_mm", "right_common_iliac_diameter_mm", "mm"),
    "common_iliac_length_mm": ("left_common_iliac_length_mm", "right_common_iliac_length_mm", "mm"),
    "external_iliac_treatment_diameter_mm": (
        "left_external_iliac_treatment_diameter_mm",
        "right_external_iliac_treatment_diameter_mm",
        "mm",
    ),
    "external_iliac_seal_zone_length_mm": (
        "left_external_iliac_seal_zone_length_mm",
        "right_external_iliac_seal_zone_length_mm",
        "mm",
    ),
    "internal_iliac_treatment_diameter_mm": (
        "left_internal_iliac_treatment_diameter_mm",
        "right_internal_iliac_treatment_diameter_mm",
        "mm",
    ),
    "internal_iliac_seal_zone_length_mm": (
        "left_internal_iliac_seal_zone_length_mm",
        "right_internal_iliac_seal_zone_length_mm",
        "mm",
    ),
    "renal_to_internal_iliac_length_mm": (
        "left_renal_to_internal_iliac_length_mm",
        "right_renal_to_internal_iliac_length_mm",
        "mm",
    ),
    "access_vessel_min_diameter_mm": (
        "left_access_vessel_min_diameter_mm",
        "right_access_vessel_min_diameter_mm",
        "mm",
    ),
    "access_vessel_tortuosity": ("left_access_vessel_tortuosity", "right_access_vessel_tortuosity", "ratio"),
}

CANONICAL_ALIASES = {
    "abdominal_aorta": ("abdominal_aorta", "abdominal_aorta_trunk", "aorta", "aorta_trunk"),
    "left_renal_artery": ("left_renal_artery", "left_renal", "left_kidney_artery"),
    "right_renal_artery": ("right_renal_artery", "right_renal", "right_kidney_artery"),
    "left_common_iliac": ("left_common_iliac", "left_common_iliac_artery", "left_cia"),
    "right_common_iliac": ("right_common_iliac", "right_common_iliac_artery", "right_cia"),
    "left_external_iliac": ("left_external_iliac", "left_external_iliac_artery", "left_eia"),
    "right_external_iliac": ("right_external_iliac", "right_external_iliac_artery", "right_eia"),
    "left_internal_iliac": (
        "left_internal_iliac",
        "left_internal_iliac_artery",
        "left_iia",
        "left_hypogastric",
    ),
    "right_internal_iliac": (
        "right_internal_iliac",
        "right_internal_iliac_artery",
        "right_iia",
        "right_hypogastric",
    ),
    "left_femoral": ("left_common_femoral", "left_femoral", "left_femoral_artery"),
    "right_femoral": ("right_common_femoral", "right_femoral", "right_femoral_artery"),
}

PRIORITY_SEGMENTS = (
    "abdominal_aorta",
    "left_renal_artery",
    "right_renal_artery",
    "left_common_iliac",
    "right_common_iliac",
    "left_external_iliac",
    "right_external_iliac",
    "left_internal_iliac",
    "right_internal_iliac",
)


class Step4Failure(RuntimeError):
    pass


@dataclass
class Projection:
    point: np.ndarray
    abscissa: float
    distance: float
    segment_index: int
    fraction: float


@dataclass
class CrossSection:
    status: str
    equivalent_diameter_mm: Optional[float]
    major_diameter_mm: Optional[float]
    minor_diameter_mm: Optional[float]
    area_mm2: Optional[float]
    plane_origin: list[float]
    plane_normal: list[float]
    centerline_abscissa_mm: float
    confidence: float
    method: str
    notes: list[str]


@dataclass
class SegmentGeometry:
    canonical_name: str
    actual_name: str
    segment_id: Optional[int]
    centerline_points: np.ndarray
    surface: Optional[vtk.vtkPolyData]
    confidence: float
    centerline_mode: str

    @property
    def is_usable(self) -> bool:
        return self.centerline_points.shape[0] >= 2

    @property
    def length(self) -> float:
        return polyline_length(self.centerline_points)


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _normalize_name(value: str) -> str:
    chars: list[str] = []
    last_was_sep = False
    for char in str(value).strip().lower():
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    return "".join(chars).strip("_")


def _cell_string_values(polydata: vtk.vtkPolyData, name: str) -> list[Optional[str]]:
    arr = polydata.GetCellData().GetAbstractArray(name)
    count = int(polydata.GetNumberOfCells())
    if arr is None or not hasattr(arr, "GetValue"):
        return [None] * count
    return [str(arr.GetValue(i)) for i in range(min(count, arr.GetNumberOfValues()))]


def _cell_numeric_values(polydata: vtk.vtkPolyData, name: str) -> list[Optional[int]]:
    arr = polydata.GetCellData().GetArray(name)
    count = int(polydata.GetNumberOfCells())
    if arr is None:
        return [None] * count
    return [int(arr.GetTuple1(i)) for i in range(min(count, arr.GetNumberOfTuples()))]


def _polyline_points_for_cell(polydata: vtk.vtkPolyData, cell_id: int) -> np.ndarray:
    cell = polydata.GetCell(int(cell_id))
    if cell is None or cell.GetNumberOfPoints() < 2:
        return np.zeros((0, 3), dtype=float)
    ids = cell.GetPointIds()
    return np.asarray([polydata.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)


def _extract_centerlines(polydata: vtk.vtkPolyData, contract: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    names = _cell_string_values(polydata, "SegmentName")
    segment_ids = _cell_numeric_values(polydata, "SegmentId")
    contract_name_by_id = {
        int(segment_id): str(name)
        for segment_id, name in (contract.get("segment_name_map") or {}).items()
        if str(segment_id).lstrip("-").isdigit()
    }

    centerlines: dict[str, np.ndarray] = {}
    ids_by_name: dict[str, int] = {}
    for cell_id in range(polydata.GetNumberOfCells()):
        segment_id = segment_ids[cell_id] if cell_id < len(segment_ids) else None
        name = names[cell_id] if cell_id < len(names) else None
        if not name and segment_id is not None:
            name = contract_name_by_id.get(int(segment_id))
        if not name:
            continue
        points = _polyline_points_for_cell(polydata, cell_id)
        if points.shape[0] < 2:
            continue
        centerlines[str(name)] = points
        if segment_id is not None:
            ids_by_name[str(name)] = int(segment_id)
    return centerlines, ids_by_name


def _available_segment_names(
    contract: dict[str, Any],
    centerlines: dict[str, np.ndarray],
    surface: vtk.vtkPolyData,
) -> list[str]:
    names: list[str] = []
    names.extend(str(v) for v in (contract.get("segment_name_map") or {}).values())
    names.extend(centerlines.keys())
    names.extend(name for name in _cell_string_values(surface, "SegmentName") if name)
    return sorted(set(names), key=lambda value: _normalize_name(value))


def _resolve_segment_names(available_names: Iterable[str]) -> dict[str, Optional[str]]:
    normalized: dict[str, str] = {}
    for name in available_names:
        normalized.setdefault(_normalize_name(name), str(name))

    resolved: dict[str, Optional[str]] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        found = None
        for alias in aliases:
            found = normalized.get(_normalize_name(alias))
            if found:
                break
        resolved[canonical] = found
    return resolved


def _threshold_surface_by_segment_id(surface: vtk.vtkPolyData, segment_id: Optional[int]) -> Optional[vtk.vtkPolyData]:
    if segment_id is None:
        return None
    if get_cell_array(surface, "SegmentId") is None:
        return None

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(surface)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "SegmentId")
    if hasattr(threshold, "SetLowerThreshold"):
        threshold.SetLowerThreshold(float(segment_id))
        threshold.SetUpperThreshold(float(segment_id))
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    else:  # pragma: no cover - older VTK compatibility.
        threshold.ThresholdBetween(float(segment_id), float(segment_id))
    threshold.Update()

    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(threshold.GetOutputPort())
    geometry.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(geometry.GetOutputPort())
    cleaner.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    if out.GetNumberOfCells() <= 0:
        return None
    return out


def _segment_confidence(contract: dict[str, Any], actual_name: str) -> tuple[float, str]:
    flags = ((contract.get("confidence_flags") or {}).get("per_segment") or {}).get(actual_name) or {}
    proximal = flags.get("proximal_start_confidence")
    naming = flags.get("naming_confidence")
    confidence_values = [float(v) for v in (proximal, naming) if v is not None]
    confidence = min(confidence_values) if confidence_values else 0.65
    return float(max(0.0, min(1.0, confidence))), str(flags.get("centerline_mode") or "")


def _build_segments(
    contract: dict[str, Any],
    named_surface: vtk.vtkPolyData,
    named_centerlines: vtk.vtkPolyData,
    warnings: list[str],
) -> tuple[dict[str, Optional[SegmentGeometry]], dict[str, Optional[str]], list[str]]:
    centerlines, centerline_ids_by_name = _extract_centerlines(named_centerlines, contract)
    available_names = _available_segment_names(contract, centerlines, named_surface)
    resolved_names = _resolve_segment_names(available_names)
    contract_id_by_name = {
        str(name): int(segment_id)
        for segment_id, name in (contract.get("segment_name_map") or {}).items()
        if str(segment_id).lstrip("-").isdigit()
    }

    missing_priority: list[str] = []
    segments: dict[str, Optional[SegmentGeometry]] = {}
    for canonical in PRIORITY_SEGMENTS + ("left_femoral", "right_femoral"):
        actual_name = resolved_names.get(canonical)
        if not actual_name:
            segments[canonical] = None
            if canonical in PRIORITY_SEGMENTS:
                missing_priority.append(canonical)
            continue
        segment_id = contract_id_by_name.get(actual_name, centerline_ids_by_name.get(actual_name))
        points = centerlines.get(actual_name)
        if points is None or points.shape[0] < 2:
            segments[canonical] = None
            if canonical in PRIORITY_SEGMENTS:
                missing_priority.append(canonical)
                warnings.append(f"{actual_name} centerline could not be resolved from named_centerlines.vtp.")
            continue
        surface = _threshold_surface_by_segment_id(named_surface, segment_id)
        confidence, centerline_mode = _segment_confidence(contract, actual_name)
        segments[canonical] = SegmentGeometry(
            canonical_name=canonical,
            actual_name=actual_name,
            segment_id=segment_id,
            centerline_points=np.asarray(points, dtype=float),
            surface=surface,
            confidence=confidence,
            centerline_mode=centerline_mode,
        )
    return segments, resolved_names, sorted(set(missing_priority))


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    return cumulative_arclength(points)


def project_point_to_polyline(point: Iterable[float], polyline: np.ndarray) -> Projection:
    pts = np.asarray(polyline, dtype=float)
    p = np.asarray(point, dtype=float).reshape(3)
    if pts.shape[0] < 2:
        raise ValueError("Cannot project onto a polyline with fewer than 2 points.")

    abscissae = cumulative_arclength(pts)
    best: Optional[Projection] = None
    for idx in range(pts.shape[0] - 1):
        start = pts[idx]
        end = pts[idx + 1]
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= EPS:
            continue
        fraction = float(np.clip(np.dot(p - start, segment) / denom, 0.0, 1.0))
        projected = start + fraction * segment
        seg_len = float(np.linalg.norm(segment))
        candidate = Projection(
            point=projected,
            abscissa=float(abscissae[idx] + fraction * seg_len),
            distance=float(np.linalg.norm(p - projected)),
            segment_index=int(idx),
            fraction=fraction,
        )
        if best is None or candidate.distance < best.distance:
            best = candidate
    if best is None:
        raise ValueError("Cannot project onto a degenerate polyline.")
    return best


def get_abscissa_for_projected_point(point: Iterable[float], polyline: np.ndarray) -> float:
    return project_point_to_polyline(point, polyline).abscissa


def _safe_major_minor(points: np.ndarray, normal_hint: Optional[np.ndarray] = None) -> tuple[Optional[float], Optional[float]]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return None, None
    normal = unit(np.asarray(normal_hint, dtype=float)) if normal_hint is not None else np.zeros(3, dtype=float)
    if float(np.linalg.norm(normal)) <= EPS:
        _, normal, _ = polygon_area_normal(pts)
    if float(np.linalg.norm(normal)) <= EPS:
        return None, None
    u, v = orthonormal_frame(normal)
    center = np.mean(pts, axis=0)
    xy = np.column_stack([(pts - center) @ u, (pts - center) @ v])
    major = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    minor = float(min(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    if not math.isfinite(major) or not math.isfinite(minor):
        return None, None
    return major, minor


def _contour_profiles_from_plane(
    surface: vtk.vtkPolyData,
    origin: np.ndarray,
    normal: np.ndarray,
) -> list[dict[str, Any]]:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    n = unit(normal)
    plane.SetNormal(float(n[0]), float(n[1]), float(n[2]))

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(surface)
    cutter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(cutter.GetOutput())
    cleaner.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputData(cleaner.GetOutput())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    contours = stripper.GetOutput()

    profiles: list[dict[str, Any]] = []
    for cell_id in range(contours.GetNumberOfCells()):
        cell = contours.GetCell(cell_id)
        if cell is None:
            continue
        ids = cell.GetPointIds()
        if ids.GetNumberOfIds() < 3:
            continue
        points = np.asarray([contours.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
        closed_gap = distance(points[0], points[-1])
        if closed_gap <= 1.0e-5:
            points = points[:-1]
        if points.shape[0] < 3:
            continue
        area, profile_normal, rms = polygon_area_normal(points)
        if area <= 1.0e-8:
            continue
        centroid = np.mean(points, axis=0)
        major, minor = _safe_major_minor(points, normal_hint=n)
        profiles.append(
            {
                "cell_id": int(cell_id),
                "area_mm2": float(area),
                "equivalent_diameter_mm": equivalent_diameter_from_area(area),
                "major_diameter_mm": major,
                "minor_diameter_mm": minor,
                "centroid": centroid,
                "normal": unit(profile_normal if np.linalg.norm(profile_normal) > EPS else n),
                "rms_planarity": float(rms),
                "closed_gap": float(closed_gap),
                "distance_to_origin": float(distance(centroid, origin)),
                "point_count": int(points.shape[0]),
                "source": "vtk_plane_surface_intersection",
            }
        )
    return profiles


def _slab_projection_profile(
    surface: vtk.vtkPolyData,
    origin: np.ndarray,
    normal: np.ndarray,
) -> Optional[dict[str, Any]]:
    points = points_to_numpy(surface)
    if points.shape[0] < 6:
        return None
    n = unit(normal)
    u, v = orthonormal_frame(n)
    best_points: Optional[np.ndarray] = None
    best_thickness = None
    for thickness in (0.35, 0.5, 0.75, 1.0, 1.5):
        axial = np.abs((points - origin) @ n)
        selected = points[axial <= thickness]
        if selected.shape[0] >= 8:
            best_points = selected
            best_thickness = thickness
            break
    if best_points is None:
        return None
    center = np.mean(best_points, axis=0)
    xy = np.column_stack([(best_points - center) @ u, (best_points - center) @ v])
    major = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    minor = float(min(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    if major <= EPS or minor <= EPS:
        return None
    area = float(math.pi * major * minor / 4.0)
    return {
        "area_mm2": area,
        "equivalent_diameter_mm": equivalent_diameter_from_area(area),
        "major_diameter_mm": major,
        "minor_diameter_mm": minor,
        "centroid": center,
        "normal": n,
        "rms_planarity": None,
        "closed_gap": None,
        "distance_to_origin": float(distance(center, origin)),
        "point_count": int(best_points.shape[0]),
        "source": "nearby_surface_slab_projection",
        "slab_half_thickness_mm": best_thickness,
    }


def _profile_confidence(profile: dict[str, Any]) -> float:
    source = str(profile.get("source"))
    if source == "nearby_surface_slab_projection":
        confidence = 0.46
    else:
        confidence = 0.88
        eq = float(profile.get("equivalent_diameter_mm") or 0.0)
        gap = profile.get("closed_gap")
        if gap is not None:
            gap_f = float(gap)
            if gap_f > max(0.25, 0.20 * max(eq, 1.0)):
                confidence -= 0.28
            elif gap_f > max(0.10, 0.08 * max(eq, 1.0)):
                confidence -= 0.10
        dist = float(profile.get("distance_to_origin") or 0.0)
        if dist > max(1.5, 0.75 * max(eq, 1.0)):
            confidence -= 0.20
        elif dist > max(0.75, 0.35 * max(eq, 1.0)):
            confidence -= 0.08
        rms = profile.get("rms_planarity")
        if rms is not None and float(rms) > 0.25:
            confidence -= 0.08
    return float(max(0.25, min(0.95, confidence)))


def _pick_profile(profiles: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    valid = [p for p in profiles if p.get("equivalent_diameter_mm") is not None]
    if not valid:
        return None
    max_area = max(float(p.get("area_mm2") or 0.0) for p in valid)
    if max_area > 0.0:
        dominant = [p for p in valid if float(p.get("area_mm2") or 0.0) >= 0.25 * max_area]
        if dominant:
            valid = dominant

    def score(profile: dict[str, Any]) -> tuple[int, float, float]:
        eq = float(profile.get("equivalent_diameter_mm") or 0.0)
        gap = profile.get("closed_gap")
        closed_penalty = 0
        if gap is not None and float(gap) > max(0.25, 0.20 * max(eq, 1.0)):
            closed_penalty = 1
        if profile.get("source") == "nearby_surface_slab_projection":
            closed_penalty = 2
        return (closed_penalty, -float(profile.get("area_mm2") or 0.0), float(profile.get("distance_to_origin") or 0.0))

    return sorted(valid, key=score)[0]


def _cross_section_at_abscissa(
    segment: SegmentGeometry,
    abscissa: float,
    method: str,
) -> CrossSection:
    points = segment.centerline_points
    length = segment.length
    s = float(np.clip(abscissa, 0.0, max(length, 0.0)))
    origin = point_at_arclength(points, s)
    normal = tangent_at_arclength(points, s, window=1.0)
    notes: list[str] = []
    if segment.surface is None or segment.surface.GetNumberOfCells() <= 0:
        return CrossSection(
            status="unmeasurable",
            equivalent_diameter_mm=None,
            major_diameter_mm=None,
            minor_diameter_mm=None,
            area_mm2=None,
            plane_origin=origin.tolist(),
            plane_normal=unit(normal).tolist(),
            centerline_abscissa_mm=s,
            confidence=0.0,
            method=method,
            notes=["segment surface was unavailable for orthogonal cross-section measurement"],
        )

    profiles = _contour_profiles_from_plane(segment.surface, origin, normal)
    profile = _pick_profile(profiles)
    if profile is None:
        profile = _slab_projection_profile(segment.surface, origin, normal)
        if profile is not None:
            notes.append("used nearby surface slab projection because plane intersection did not produce a usable contour")
    if profile is None:
        return CrossSection(
            status="unmeasurable",
            equivalent_diameter_mm=None,
            major_diameter_mm=None,
            minor_diameter_mm=None,
            area_mm2=None,
            plane_origin=origin.tolist(),
            plane_normal=unit(normal).tolist(),
            centerline_abscissa_mm=s,
            confidence=0.0,
            method=method,
            notes=["orthogonal cross-section could not be computed at this centerline abscissa"],
        )

    confidence = _profile_confidence(profile)
    if profile.get("closed_gap") is not None and float(profile["closed_gap"]) > 0.25:
        notes.append("plane contour was not fully closed; measurement confidence reduced")
    if profile.get("source") == "nearby_surface_slab_projection":
        notes.append("diameter is approximate from local surface-point projection")
    status = "measured" if confidence >= 0.55 else "requires_review"
    return CrossSection(
        status=status,
        equivalent_diameter_mm=_finite_or_none(profile.get("equivalent_diameter_mm")),
        major_diameter_mm=_finite_or_none(profile.get("major_diameter_mm")),
        minor_diameter_mm=_finite_or_none(profile.get("minor_diameter_mm")),
        area_mm2=_finite_or_none(profile.get("area_mm2")),
        plane_origin=origin.tolist(),
        plane_normal=unit(normal).tolist(),
        centerline_abscissa_mm=s,
        confidence=confidence,
        method=f"{method}; {profile.get('source')}",
        notes=notes,
    )


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _measurement(
    *,
    value: Any = None,
    unit: str = "mm",
    status: str = "unmeasurable",
    method: str = "",
    landmarks_used: Optional[list[str]] = None,
    source_segment_names: Optional[list[str]] = None,
    side: str = "unknown",
    confidence: Optional[float] = None,
    notes: Optional[list[str]] = None,
    **extra: Any,
) -> dict[str, Any]:
    if status not in MEASUREMENT_STATUSES:
        raise ValueError(f"Invalid measurement status: {status}")
    row = {
        "value": value,
        "unit": unit,
        "status": status,
        "method": method,
        "landmarks_used": list(landmarks_used or []),
        "source_segment_names": list(source_segment_names or []),
        "side": side,
        "confidence": confidence,
        "notes": list(notes or []),
    }
    row.update(extra)
    return row


def _missing_measurement(
    *,
    status: str = "missing_required_landmark",
    unit: str = "mm",
    method: str = "",
    side: str = "unknown",
    landmarks_used: Optional[list[str]] = None,
    source_segment_names: Optional[list[str]] = None,
    notes: Optional[list[str]] = None,
    **extra: Any,
) -> dict[str, Any]:
    return _measurement(
        value=None,
        unit=unit,
        status=status,
        method=method,
        side=side,
        confidence=None,
        landmarks_used=landmarks_used or [],
        source_segment_names=source_segment_names or [],
        notes=notes or [],
        **extra,
    )


def _diameter_measurement_from_section(
    section: CrossSection,
    *,
    value_key: str,
    side: str,
    source_segment_names: list[str],
    landmarks_used: list[str],
    method: str,
    notes: Optional[list[str]] = None,
    status_override: Optional[str] = None,
) -> dict[str, Any]:
    value = getattr(section, value_key)
    status = status_override or section.status
    return _measurement(
        value=value,
        unit="mm",
        status=status,
        method=method,
        landmarks_used=landmarks_used,
        source_segment_names=source_segment_names,
        side=side,
        confidence=section.confidence,
        notes=(notes or []) + list(section.notes),
        major_diameter_mm=section.major_diameter_mm,
        minor_diameter_mm=section.minor_diameter_mm,
        equivalent_diameter_mm=section.equivalent_diameter_mm,
        area_mm2=section.area_mm2,
        plane_origin=section.plane_origin,
        plane_normal=section.plane_normal,
        centerline_abscissa_mm=section.centerline_abscissa_mm,
    )


def _summary_measurement(
    left_ref: str,
    right_ref: str,
    left: Optional[dict[str, Any]],
    right: Optional[dict[str, Any]],
    unit: str,
    notes: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "status": "derived_summary",
        "summary_rule": "side_specific_values_preferred",
        "left_ref": left_ref,
        "right_ref": right_ref,
        "left_value": left.get("value") if isinstance(left, dict) else None,
        "right_value": right.get("value") if isinstance(right, dict) else None,
        "unit": unit,
        "notes": list(notes or []),
    }


def _empty_measurement_groups(default_status: str, note: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for group_name, fields in REQUIRED_FIELD_GROUPS.items():
        groups[group_name] = {}
        for field in fields:
            if field in SUMMARY_FIELDS:
                left_ref, right_ref, unit = SUMMARY_FIELDS[field]
                groups[group_name][field] = _summary_measurement(left_ref, right_ref, None, None, unit, notes=[note])
            else:
                unit = "ratio" if "tortuosity" in field else "degrees" if field.endswith("_deg") else "mm"
                groups[group_name][field] = _missing_measurement(status=default_status, unit=unit, notes=[note])
    return groups


def _base_contract(
    *,
    step_status: str,
    input_paths: dict[str, str],
    output_paths: dict[str, str],
    measurement_groups: dict[str, Any],
    warnings: list[str],
    qa: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "step_name": STEP_NAME,
        "step_status": step_status,
        "input_paths": input_paths,
        "output_paths": output_paths,
        "upstream_references": {
            "step3_contract": "step3_naming_orientation_contract.json",
        },
        "units": {
            "length": "mm",
            "angle": "degrees",
        },
        "measurement_groups": measurement_groups,
        "unmeasurable_values": [],
        "warnings": _dedupe(warnings),
        "qa": qa,
    }


def _renal_landmark_point(
    contract: dict[str, Any],
    actual_name: str,
    segment: Optional[SegmentGeometry],
) -> tuple[Optional[np.ndarray], str, float]:
    proximal = (contract.get("proximal_start_metadata") or {}).get(actual_name) or {}
    for key in ("centerline_anchor_point",):
        if proximal.get(key) is not None:
            return np.asarray(proximal[key], dtype=float), f"proximal_start_metadata.{actual_name}.{key}", float(
                proximal.get("confidence", 0.8)
            )
    profile = proximal.get("boundary_profile") or {}
    for key in ("boundary_centroid", "centroid"):
        if profile.get(key) is not None:
            return np.asarray(profile[key], dtype=float), f"proximal_start_metadata.{actual_name}.boundary_profile.{key}", float(
                proximal.get("confidence", 0.7)
            )

    landmarks = ((contract.get("landmark_registry") or {}).get("segment_landmarks") or {}).get(actual_name) or {}
    if landmarks.get("proximal_point") is not None:
        return np.asarray(landmarks["proximal_point"], dtype=float), f"landmark_registry.segment_landmarks.{actual_name}.proximal_point", float(
            landmarks.get("proximal_boundary_confidence") or 0.65
        )

    if segment is not None and segment.centerline_points.shape[0] >= 2:
        return np.asarray(segment.centerline_points[0], dtype=float), f"named_centerlines.{actual_name}.first_point", 0.55
    return None, "", 0.0


def _select_lowest_renal(
    contract: dict[str, Any],
    segments: dict[str, Optional[SegmentGeometry]],
    warnings: list[str],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    aorta = segments.get("abdominal_aorta")
    if aorta is None or not aorta.is_usable:
        return None, {
            "status": "failed",
            "reason": "abdominal aorta centerline unavailable",
            "renal_abscissae_mm": {},
        }

    renal_rows: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        canonical = f"{side}_renal_artery"
        segment = segments.get(canonical)
        if segment is None:
            warnings.append(f"{canonical} segment not resolved; lowest renal selection is incomplete.")
            continue
        point, source, landmark_confidence = _renal_landmark_point(contract, segment.actual_name, segment)
        if point is None:
            warnings.append(f"{segment.actual_name} proximal renal landmark could not be resolved.")
            continue
        projection = project_point_to_polyline(point, aorta.centerline_points)
        confidence = min(segment.confidence, landmark_confidence)
        if projection.distance > 3.0:
            confidence -= 0.25
            warnings.append(
                f"{segment.actual_name} renal landmark projects {projection.distance:.2f} mm from the aorta centerline; review lowest renal selection."
            )
        elif projection.distance > 1.0:
            confidence -= 0.10
        renal_rows[side] = {
            "side": side,
            "segment_name": segment.actual_name,
            "landmark_source": source,
            "landmark_point": point.tolist(),
            "projected_point": projection.point.tolist(),
            "aortic_abscissa_mm": float(projection.abscissa),
            "projection_distance_mm": float(projection.distance),
            "confidence": float(max(0.25, min(0.98, confidence))),
        }

    if not renal_rows:
        warnings.append("No renal artery landmarks could be projected onto the abdominal aorta centerline.")
        return None, {
            "status": "missing_required_landmark",
            "reason": "no renal landmarks resolved",
            "renal_abscissae_mm": {},
        }

    selected = max(renal_rows.values(), key=lambda row: float(row["aortic_abscissa_mm"]))
    if len(renal_rows) == 1:
        selected["confidence"] = float(min(float(selected["confidence"]), 0.65))
        warnings.append("Only one renal artery landmark was available; lowest renal reference requires review.")
    elif abs(float(renal_rows["left"]["aortic_abscissa_mm"]) - float(renal_rows["right"]["aortic_abscissa_mm"])) <= 0.25:
        selected["confidence"] = float(min(float(selected["confidence"]), 0.75))
        selected["tie_with_contralateral_renal"] = True
        warnings.append("Left and right renal landmarks project to nearly identical aortic abscissae; lowest renal side is provisional.")

    qa = {
        "status": "measured" if float(selected["confidence"]) >= 0.55 else "requires_review",
        "renal_abscissae_mm": {
            side: float(row["aortic_abscissa_mm"])
            for side, row in sorted(renal_rows.items())
        },
        "selected_lower_renal_side": selected["side"],
        "selected_lower_renal_segment": selected["segment_name"],
        "selected_lower_renal_abscissa_mm": float(selected["aortic_abscissa_mm"]),
        "selection_method": "max_downstream_abscissa_from_aortic_inlet",
        "confidence": float(selected["confidence"]),
        "details": renal_rows,
    }
    return selected, qa


def _profile_sample_to_json(
    sample: dict[str, Any],
    lowest_renal_abscissa: float,
) -> dict[str, Any]:
    section: CrossSection = sample["section"]
    return {
        "offset_from_lowest_renal_mm": float(sample["offset"]),
        "centerline_abscissa_mm": float(section.centerline_abscissa_mm),
        "equivalent_diameter_mm": section.equivalent_diameter_mm,
        "major_diameter_mm": section.major_diameter_mm,
        "minor_diameter_mm": section.minor_diameter_mm,
        "plane_origin": section.plane_origin,
        "plane_normal": section.plane_normal,
        "confidence": section.confidence,
        "status": section.status,
        "method": section.method,
        "notes": list(section.notes),
    }


def _measure_aortic_profile(
    aorta: SegmentGeometry,
    lowest_renal: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], Optional[dict[str, Any]], dict[str, Any]]:
    lower_s = float(lowest_renal["aortic_abscissa_mm"])
    total = aorta.length
    downstream = max(0.0, total - lower_s)
    if downstream <= 0.5:
        warnings.append("Aortic centerline has insufficient downstream length after the lowest renal reference.")
        return (
            {
                "status": "requires_review",
                "unit": "mm",
                "sample_spacing_mm": 1.0,
                "samples": [],
                "named_offsets": {"D0": None, "D5": None, "D10": None, "D15": None},
                "notes": ["insufficient downstream aortic centerline length"],
            },
            [],
            None,
            {
                "sample_count": 0,
                "measured_sample_count": 0,
                "available_downstream_length_mm": downstream,
                "status": "requires_review",
            },
        )

    sample_spacing = 1.0 if downstream <= 80.0 else 2.0
    offsets = [float(v) for v in np.arange(0.0, downstream + 0.001, sample_spacing)]
    if offsets[-1] < downstream:
        offsets.append(float(downstream))

    samples: list[dict[str, Any]] = []
    for offset in offsets:
        section = _cross_section_at_abscissa(
            aorta,
            lower_s + offset,
            "centerline_orthogonal_aortic_diameter_profile",
        )
        samples.append({"offset": float(offset), "section": section})

    usable = [
        sample
        for sample in samples
        if sample["section"].equivalent_diameter_mm is not None and sample["section"].confidence >= 0.55
    ]
    baseline = None
    proximal_usable = [sample for sample in usable if float(sample["offset"]) <= 3.0]
    if proximal_usable:
        baseline = sorted(
            proximal_usable,
            key=lambda sample: (
                -float(sample["section"].confidence),
                float(sample["offset"]),
                -float(sample["section"].equivalent_diameter_mm or 0.0),
            ),
        )[0]
    if baseline is None and usable:
        baseline = usable[0]

    named_offsets: dict[str, Optional[dict[str, Any]]] = {}
    for name, target in (("D0", 0.0), ("D5", 5.0), ("D10", 10.0), ("D15", 15.0)):
        if target > downstream + 0.25:
            named_offsets[name] = None
            warnings.append(f"{name} profile sample unavailable due to insufficient downstream aortic centerline length.")
            continue
        nearest = min(samples, key=lambda row: abs(float(row["offset"]) - target))
        named_offsets[name] = _profile_sample_to_json(nearest, lower_s)
        if nearest["section"].status != "measured":
            warnings.append(f"{name} aortic profile sample requires review due to low-confidence cross-section geometry.")

    profile_status = "measured" if baseline is not None and all(
        sample["section"].status == "measured"
        for sample in samples
        if float(sample["offset"]) in {0.0, 5.0, 10.0, 15.0} and float(sample["offset"]) <= downstream
    ) else "requires_review"
    if baseline is None:
        warnings.append("Aortic diameter profile did not yield a usable proximal baseline sample.")
    elif float(baseline["offset"]) > 0.5:
        warnings.append(
            f"Proximal aortic baseline used sample {float(baseline['offset']):.1f} mm distal to the lowest renal reference to avoid branch-contour interference."
        )

    profile = {
        "status": profile_status,
        "unit": "mm",
        "sample_spacing_mm": sample_spacing,
        "samples": [_profile_sample_to_json(sample, lower_s) for sample in samples],
        "named_offsets": named_offsets,
        "notes": [
            "profile samples are centerline-orthogonal lumen sections from the lowest renal reference",
        ],
    }
    qa = {
        "sample_count": int(len(samples)),
        "measured_sample_count": int(sum(1 for sample in samples if sample["section"].status == "measured")),
        "requires_review_sample_count": int(sum(1 for sample in samples if sample["section"].status == "requires_review")),
        "available_downstream_length_mm": float(downstream),
        "sample_spacing_mm": sample_spacing,
        "baseline_offset_from_lowest_renal_mm": float(baseline["offset"]) if baseline is not None else None,
        "baseline_diameter_mm": baseline["section"].equivalent_diameter_mm if baseline is not None else None,
        "status": profile_status,
    }
    return profile, samples, baseline, qa


def _detect_proximal_neck_length(
    samples: list[dict[str, Any]],
    baseline_sample: Optional[dict[str, Any]],
    lowest_renal: dict[str, Any],
    warnings: list[str],
    aorta_segment_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold_percent = 10.0
    consecutive_required = 3
    if baseline_sample is None or baseline_sample["section"].equivalent_diameter_mm is None:
        measurement = _missing_measurement(
            status="requires_review",
            unit="mm",
            method="provisional_diameter_expansion_rule",
            side="midline",
            landmarks_used=["lowest_renal_reference"],
            notes=["a usable proximal baseline diameter was not available"],
        )
        measurement.update(
            {
                "expansion_threshold_percent": threshold_percent,
                "consecutive_samples_required": consecutive_required,
                "baseline_diameter_mm": None,
                "candidate_neck_end_abscissa_mm": None,
            }
        )
        return measurement, {
            "status": "requires_review",
            "reason": "missing proximal baseline",
            "expansion_threshold_percent": threshold_percent,
            "consecutive_samples_required": consecutive_required,
        }

    baseline = float(baseline_sample["section"].equivalent_diameter_mm)
    threshold = baseline * (1.0 + threshold_percent / 100.0)
    baseline_offset = float(baseline_sample["offset"])
    run: list[dict[str, Any]] = []
    confirmed_start: Optional[dict[str, Any]] = None
    for sample in samples:
        section = sample["section"]
        offset = float(sample["offset"])
        if offset <= baseline_offset:
            continue
        eq = section.equivalent_diameter_mm
        if eq is not None and section.confidence >= 0.55 and float(eq) >= threshold:
            run.append(sample)
            if len(run) >= consecutive_required:
                confirmed_start = run[0]
                break
        else:
            run = []

    base_extra = {
        "expansion_threshold_percent": threshold_percent,
        "consecutive_samples_required": consecutive_required,
        "baseline_diameter_mm": baseline,
        "baseline_offset_from_lowest_renal_mm": baseline_offset,
        "candidate_neck_end_abscissa_mm": None,
    }
    if confirmed_start is None:
        warnings.append("Provisional proximal neck end was not detected by the diameter-expansion rule; review recommended.")
        measurement = _measurement(
            value=None,
            unit="mm",
            status="requires_review",
            method="provisional_diameter_expansion_rule",
            landmarks_used=["lowest_renal_reference"],
            source_segment_names=[aorta_segment_name, str(lowest_renal["segment_name"])],
            side="midline",
            confidence=None,
            notes=[
                "no sustained downstream diameter expansion was found",
                "profile-derived rule is provisional and not definitive clinical neck-end detection",
            ],
            **base_extra,
        )
        return measurement, {
            "status": "requires_review",
            "reason": "no sustained expansion detected",
            **base_extra,
        }

    candidate_section = confirmed_start["section"]
    candidate_abscissa = float(candidate_section.centerline_abscissa_mm)
    value = max(0.0, candidate_abscissa - float(lowest_renal["aortic_abscissa_mm"]))
    measurement = _measurement(
        value=value,
        unit="mm",
        status="measured",
        method="provisional_diameter_expansion_rule",
        landmarks_used=["lowest_renal_reference", "candidate_neck_end_profile_sample"],
        source_segment_names=[aorta_segment_name, str(lowest_renal["segment_name"])],
        side="midline",
        confidence=min(float(candidate_section.confidence), float(baseline_sample["section"].confidence), 0.82),
        notes=[
            "profile-derived and provisional; not definitive clinical aneurysm-neck-end detection",
        ],
        **{
            **base_extra,
            "candidate_neck_end_abscissa_mm": candidate_abscissa,
            "candidate_neck_end_offset_from_lowest_renal_mm": float(confirmed_start["offset"]),
            "candidate_neck_end_equivalent_diameter_mm": candidate_section.equivalent_diameter_mm,
        },
    )
    warnings.append("Proximal neck length detected using provisional diameter-expansion rule; review recommended.")
    return measurement, {
        "status": "measured",
        "detected": True,
        **measurement,
    }


def _measure_neck_angulation(
    aorta: SegmentGeometry,
    lowest_renal: dict[str, Any],
    neck_length: dict[str, Any],
) -> dict[str, Any]:
    lower_s = float(lowest_renal["aortic_abscissa_mm"])
    total = aorta.length
    downstream = total - lower_s
    upstream = lower_s
    if downstream < 3.0:
        return _missing_measurement(
            status="requires_review",
            unit="degrees",
            method="centerline_axis_angle",
            side="midline",
            notes=["insufficient downstream aortic centerline length for angulation"],
        )

    neck_axis_len = min(15.0, downstream)
    if neck_length.get("value") is not None:
        neck_axis_len = max(3.0, min(neck_axis_len, float(neck_length["value"])))
    neck_start = point_at_arclength(aorta.centerline_points, lower_s)
    neck_end = point_at_arclength(aorta.centerline_points, min(total, lower_s + neck_axis_len))
    neck_axis = unit(neck_end - neck_start)

    if upstream >= 3.0:
        ref_start = point_at_arclength(aorta.centerline_points, max(0.0, lower_s - min(10.0, upstream)))
        ref_end = neck_start
        reference_axis = unit(ref_end - ref_start)
        reference_method = "upstream_aortic_axis"
    else:
        ref_start = neck_start
        ref_end = point_at_arclength(aorta.centerline_points, min(total, lower_s + min(20.0, downstream)))
        reference_axis = unit(ref_end - ref_start)
        reference_method = "downstream_aortic_axis"

    if float(np.linalg.norm(neck_axis)) <= EPS or float(np.linalg.norm(reference_axis)) <= EPS:
        return _missing_measurement(
            status="requires_review",
            unit="degrees",
            method="centerline_axis_angle",
            side="midline",
            notes=["centerline axis vector could not be computed robustly"],
        )

    dot = float(np.clip(np.dot(neck_axis, reference_axis), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    if angle > 90.0:
        angle = 180.0 - angle
    confidence = min(0.82, float(lowest_renal.get("confidence", 0.7)), aorta.confidence)
    return _measurement(
        value=float(angle),
        unit="degrees",
        status="measured" if confidence >= 0.55 else "requires_review",
        method=f"centerline_axis_angle; neck_axis_vs_{reference_method}",
        landmarks_used=["lowest_renal_reference"],
        source_segment_names=[aorta.actual_name],
        side="midline",
        confidence=confidence,
        notes=[],
        neck_axis_vector=neck_axis.tolist(),
        reference_axis_vector=reference_axis.tolist(),
        neck_axis_length_mm=float(neck_axis_len),
    )


def _polyline_subsection(points: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros((0, 3), dtype=float)
    total = polyline_length(pts)
    start = float(np.clip(min(start_s, end_s), 0.0, total))
    end = float(np.clip(max(start_s, end_s), 0.0, total))
    if end - start <= EPS:
        p = point_at_arclength(pts, start)
        return np.vstack([p, p])
    arclengths = cumulative_arclength(pts)
    out: list[np.ndarray] = [point_at_arclength(pts, start)]
    for idx in range(1, pts.shape[0] - 1):
        if start < float(arclengths[idx]) < end:
            out.append(pts[idx].copy())
    out.append(point_at_arclength(pts, end))
    return np.asarray(out, dtype=float)


def _concatenate_paths(parts: list[np.ndarray]) -> np.ndarray:
    merged: list[np.ndarray] = []
    for part in parts:
        pts = np.asarray(part, dtype=float)
        if pts.shape[0] == 0:
            continue
        if not merged:
            merged.extend([p.copy() for p in pts])
            continue
        if distance(merged[-1], pts[0]) <= distance(merged[-1], pts[-1]):
            ordered = pts
        else:
            ordered = pts[::-1]
        if distance(merged[-1], ordered[0]) <= 1.0e-8:
            ordered = ordered[1:]
        merged.extend([p.copy() for p in ordered])
    if not merged:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(merged, dtype=float)


def _measure_renal_to_internal_iliac(
    *,
    side: str,
    aorta: Optional[SegmentGeometry],
    common_iliac: Optional[SegmentGeometry],
    internal_iliac: Optional[SegmentGeometry],
    lowest_renal: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    if lowest_renal is None or aorta is None or not aorta.is_usable:
        return (
            _missing_measurement(
                status="missing_required_landmark",
                unit="mm",
                method="lowest_renal_to_internal_iliac_composite_centerline_length",
                side=side,
                notes=["lowest renal reference or abdominal aorta centerline was unavailable"],
            ),
            np.zeros((0, 3), dtype=float),
        )
    if common_iliac is None or not common_iliac.is_usable:
        return (
            _missing_measurement(
                status="missing_required_landmark",
                unit="mm",
                method="lowest_renal_to_internal_iliac_composite_centerline_length",
                side=side,
                notes=[f"{side} common iliac path was unavailable"],
            ),
            np.zeros((0, 3), dtype=float),
        )
    if internal_iliac is None or not internal_iliac.is_usable:
        return (
            _missing_measurement(
                status="unmeasurable",
                unit="mm",
                method="lowest_renal_to_internal_iliac_composite_centerline_length",
                side=side,
                source_segment_names=[common_iliac.actual_name],
                notes=[f"{side} internal iliac segment not resolved; IBE-specific path length is unmeasurable"],
            ),
            np.zeros((0, 3), dtype=float),
        )

    lower_s = float(lowest_renal["aortic_abscissa_mm"])
    aortic_path = _polyline_subsection(aorta.centerline_points, lower_s, aorta.length)
    path = _concatenate_paths([aortic_path, common_iliac.centerline_points, internal_iliac.centerline_points])
    component_lengths = {
        "aorta_lowest_renal_to_bifurcation_mm": max(0.0, aorta.length - lower_s),
        f"{side}_common_iliac_mm": common_iliac.length,
        f"{side}_internal_iliac_mm": internal_iliac.length,
    }
    confidence = min(aorta.confidence, common_iliac.confidence, internal_iliac.confidence, float(lowest_renal.get("confidence", 0.7)), 0.86)
    return (
        _measurement(
            value=float(sum(component_lengths.values())),
            unit="mm",
            status="measured" if confidence >= 0.55 else "requires_review",
            method="lowest_renal_to_internal_iliac_composite_centerline_length",
            landmarks_used=["lowest_renal_reference", "aortic_bifurcation", f"{side}_internal_iliac_path"],
            source_segment_names=[aorta.actual_name, common_iliac.actual_name, internal_iliac.actual_name],
            side=side,
            confidence=confidence,
            notes=["geometry/path length only; no device adequacy or overlap decision is made"],
            path_component_lengths_mm=component_lengths,
        ),
        path,
    )


def _access_path(
    external_iliac: Optional[SegmentGeometry],
    femoral: Optional[SegmentGeometry],
) -> tuple[np.ndarray, str, list[str], float]:
    if external_iliac is None or not external_iliac.is_usable:
        return np.zeros((0, 3), dtype=float), "unknown_or_incomplete", [], 0.0
    if femoral is not None and femoral.is_usable:
        points = _concatenate_paths([external_iliac.centerline_points, femoral.centerline_points])
        return points, "iliofemoral", [external_iliac.actual_name, femoral.actual_name], min(external_iliac.confidence, femoral.confidence)
    return external_iliac.centerline_points, "iliac_only", [external_iliac.actual_name], external_iliac.confidence


def _measure_access_min_diameter(
    *,
    side: str,
    external_diameter: dict[str, Any],
    femoral: Optional[SegmentGeometry],
    access_extent: str,
) -> dict[str, Any]:
    candidates = [external_diameter]
    femoral_measurement = None
    if femoral is not None and femoral.is_usable:
        femoral_measurement, _ = _measure_segment_diameter(
            femoral,
            side=side,
            method_role="access_vessel_femoral_minimum_diameter",
            missing_status="requires_review",
        )
        candidates.append(femoral_measurement)
    if external_diameter.get("value") is None and femoral_measurement is None:
        return _missing_measurement(
            status="missing_required_landmark",
            unit="mm",
            method="access_path_minimum_lumen_diameter",
            side=side,
            notes=["external iliac access path could not be resolved"],
            access_extent=access_extent,
        )
    measurement = _derived_min_measurement(
        candidates,
        side=side,
        method="access_path_minimum_lumen_diameter",
        notes=["minimum lumen diameter along available access path; no tissue burden assessment"],
    )
    measurement["access_extent"] = access_extent
    return measurement


def _measure_access_tortuosity(
    *,
    side: str,
    path_points: np.ndarray,
    source_segment_names: list[str],
    access_extent: str,
    confidence: float,
) -> dict[str, Any]:
    if path_points.shape[0] < 2:
        return _missing_measurement(
            status="missing_required_landmark",
            unit="ratio",
            method="access_path_tortuosity_path_length_over_straight_line",
            side=side,
            source_segment_names=source_segment_names,
            notes=["access path could not be resolved"],
            access_extent=access_extent,
        )
    path_length = polyline_length(path_points)
    straight = distance(path_points[0], path_points[-1])
    if straight <= EPS:
        return _missing_measurement(
            status="requires_review",
            unit="ratio",
            method="access_path_tortuosity_path_length_over_straight_line",
            side=side,
            source_segment_names=source_segment_names,
            notes=["access path endpoints are coincident or too close for tortuosity"],
            access_extent=access_extent,
        )
    return _measurement(
        value=float(path_length / straight),
        unit="ratio",
        status="measured" if confidence >= 0.55 else "requires_review",
        method="access_path_tortuosity_path_length_over_straight_line",
        landmarks_used=["access_path_start", "access_path_end"],
        source_segment_names=source_segment_names,
        side=side,
        confidence=min(confidence, 0.86),
        notes=["tortuosity is geometric path length divided by straight-line endpoint distance"],
        path_length_mm=float(path_length),
        straight_line_length_mm=float(straight),
        access_extent=access_extent,
    )


class RegionBuilder:
    def __init__(self) -> None:
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.measurement_group: list[str] = []
        self.measurement_name: list[str] = []
        self.side: list[str] = []
        self.status: list[str] = []
        self.confidence: list[float] = []

    def add_polyline(
        self,
        points: np.ndarray,
        *,
        group: str,
        name: str,
        side: str,
        status: str,
        confidence: Optional[float],
    ) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 2:
            return
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(int(pts.shape[0]))
        for idx, point in enumerate(pts):
            pid = self.points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            polyline.GetPointIds().SetId(idx, int(pid))
        self.lines.InsertNextCell(polyline)
        self._add_meta(group, name, side, status, confidence)

    def add_plane(
        self,
        *,
        origin: Iterable[float],
        normal: Iterable[float],
        radius: float,
        group: str,
        name: str,
        side: str,
        status: str,
        confidence: Optional[float],
    ) -> None:
        o = np.asarray(origin, dtype=float)
        n = unit(np.asarray(normal, dtype=float))
        if float(np.linalg.norm(n)) <= EPS:
            n = np.asarray([0.0, 0.0, 1.0], dtype=float)
        u, v = orthonormal_frame(n)
        r = max(0.25, float(radius))
        pts = []
        for idx in range(25):
            angle = 2.0 * math.pi * idx / 24.0
            pts.append(o + r * math.cos(angle) * u + r * math.sin(angle) * v)
        self.add_polyline(np.asarray(pts, dtype=float), group=group, name=name, side=side, status=status, confidence=confidence)

    def _add_meta(
        self,
        group: str,
        name: str,
        side: str,
        status: str,
        confidence: Optional[float],
    ) -> None:
        self.measurement_group.append(str(group))
        self.measurement_name.append(str(name))
        self.side.append(str(side))
        self.status.append(str(status))
        self.confidence.append(float(confidence) if confidence is not None else -1.0)

    def build(self) -> vtk.vtkPolyData:
        out = vtk.vtkPolyData()
        out.SetPoints(self.points)
        out.SetLines(self.lines)
        for name, values in (
            ("measurement_group", self.measurement_group),
            ("measurement_name", self.measurement_name),
            ("side", self.side),
            ("status", self.status),
        ):
            arr = vtk.vtkStringArray()
            arr.SetName(name)
            arr.SetNumberOfValues(len(values))
            for idx, value in enumerate(values):
                arr.SetValue(idx, str(value))
            out.GetCellData().AddArray(arr)
        conf = vtk.vtkDoubleArray()
        conf.SetName("confidence")
        conf.SetNumberOfValues(len(self.confidence))
        for idx, value in enumerate(self.confidence):
            conf.SetValue(idx, float(value))
        out.GetCellData().AddArray(conf)
        return out


def _add_measurement_plane_to_regions(
    builder: RegionBuilder,
    measurement: dict[str, Any],
    *,
    group: str,
    name: str,
    side: str,
) -> None:
    origin = measurement.get("plane_origin")
    normal = measurement.get("plane_normal")
    if origin is None or normal is None:
        return
    radius = 0.5 * float(measurement.get("equivalent_diameter_mm") or measurement.get("value") or 1.0)
    builder.add_plane(
        origin=origin,
        normal=normal,
        radius=radius,
        group=group,
        name=name,
        side=side,
        status=str(measurement.get("status", "unknown")),
        confidence=measurement.get("confidence"),
    )


def _measurement_counts(measurement_groups: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in MEASUREMENT_STATUSES}
    total = 0
    for group in measurement_groups.values():
        for value in group.values():
            if not isinstance(value, dict):
                continue
            status = value.get("status")
            if status in counts:
                counts[str(status)] += 1
                total += 1
    counts["total"] = total
    return counts


def _collect_unmeasurable(measurement_groups: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_name, group in measurement_groups.items():
        for field_name, value in group.items():
            if not isinstance(value, dict):
                continue
            status = value.get("status")
            if status in {"unmeasurable", "missing_required_landmark", "requires_review"}:
                out.append(
                    {
                        "group": group_name,
                        "field": field_name,
                        "status": status,
                        "notes": list(value.get("notes", [])),
                    }
                )
    return out


def _overall_status(
    *,
    step3_status: str,
    measurement_groups: dict[str, Any],
    failed: bool = False,
) -> str:
    if failed:
        return "failed"
    if step3_status == "failed":
        return "failed"
    counts = _measurement_counts(measurement_groups)
    if counts.get("requires_review", 0) or counts.get("missing_required_landmark", 0):
        return "requires_review"
    if step3_status == "requires_review":
        return "requires_review"
    return "success"


def _build_aortic_neck_group(
    *,
    aorta: Optional[SegmentGeometry],
    lowest_renal: Optional[dict[str, Any]],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    group: dict[str, Any] = {}
    regions: list[dict[str, Any]] = []
    if aorta is None or not aorta.is_usable:
        note = "abdominal aorta centerline was unavailable"
        for field in AORTIC_NECK_FIELDS:
            unit = "degrees" if field.endswith("_deg") else "mm"
            group[field] = _missing_measurement(status="missing_required_landmark", unit=unit, side="midline", notes=[note])
        group["aortic_diameter_profile"] = {
            "status": "requires_review",
            "unit": "mm",
            "sample_spacing_mm": None,
            "samples": [],
            "named_offsets": {"D0": None, "D5": None, "D10": None, "D15": None},
            "notes": [note],
        }
        return group, {"status": "missing_required_landmark", "reason": note}, regions

    if lowest_renal is None:
        note = "lowest renal artery reference could not be selected"
        for field in AORTIC_NECK_FIELDS:
            unit = "degrees" if field.endswith("_deg") else "mm"
            group[field] = _missing_measurement(status="missing_required_landmark", unit=unit, side="midline", notes=[note])
        group["aortic_diameter_profile"] = {
            "status": "requires_review",
            "unit": "mm",
            "sample_spacing_mm": None,
            "samples": [],
            "named_offsets": {"D0": None, "D5": None, "D10": None, "D15": None},
            "notes": [note],
        }
        return group, {"status": "missing_required_landmark", "reason": note}, regions

    profile, samples, baseline_sample, profile_qa = _measure_aortic_profile(aorta, lowest_renal, warnings)
    group["aortic_diameter_profile"] = profile
    source_names = [aorta.actual_name]
    landmarks = ["lowest_renal_reference"]
    if baseline_sample is None:
        missing = _missing_measurement(
            status="requires_review",
            unit="mm",
            method="centerline_orthogonal_cross_section_immediately_distal_to_lowest_renal",
            side="midline",
            source_segment_names=source_names,
            landmarks_used=landmarks,
            notes=["proximal neck baseline diameter was not measurable"],
        )
        for field in (
            "proximal_neck_diameter_mm",
            "proximal_neck_major_diameter_mm",
            "proximal_neck_minor_diameter_mm",
            "proximal_neck_equivalent_diameter_mm",
            "infrarenal_aortic_neck_treatment_diameter_mm",
            "aortic_treatment_diameter_mm",
        ):
            group[field] = dict(missing)
    else:
        section = baseline_sample["section"]
        baseline_offset = float(baseline_sample["offset"])
        base_notes = [
            "proximal neck diameter is lumen equivalent diameter immediately distal to lowest renal reference",
        ]
        if baseline_offset > 0.5:
            base_notes.append(
                f"baseline section selected {baseline_offset:.1f} mm distal to renal reference due to local branch-contour quality"
            )
        method = "centerline_orthogonal_cross_section_immediately_distal_to_lowest_renal"
        group["proximal_neck_diameter_mm"] = _diameter_measurement_from_section(
            section,
            value_key="equivalent_diameter_mm",
            side="midline",
            source_segment_names=source_names,
            landmarks_used=landmarks,
            method=method,
            notes=base_notes + ["proximal_neck_diameter_mm aliases proximal_neck_equivalent_diameter_mm"],
        )
        group["proximal_neck_equivalent_diameter_mm"] = _diameter_measurement_from_section(
            section,
            value_key="equivalent_diameter_mm",
            side="midline",
            source_segment_names=source_names,
            landmarks_used=landmarks,
            method=method,
            notes=base_notes,
        )
        group["proximal_neck_major_diameter_mm"] = _diameter_measurement_from_section(
            section,
            value_key="major_diameter_mm",
            side="midline",
            source_segment_names=source_names,
            landmarks_used=landmarks,
            method=method,
            notes=base_notes,
        )
        group["proximal_neck_minor_diameter_mm"] = _diameter_measurement_from_section(
            section,
            value_key="minor_diameter_mm",
            side="midline",
            source_segment_names=source_names,
            landmarks_used=landmarks,
            method=method,
            notes=base_notes,
        )
        for field, source_field in (
            ("infrarenal_aortic_neck_treatment_diameter_mm", "proximal_neck_equivalent_diameter_mm"),
            ("aortic_treatment_diameter_mm", "infrarenal_aortic_neck_treatment_diameter_mm"),
        ):
            group[field] = _measurement(
                value=section.equivalent_diameter_mm,
                unit="mm",
                status="derived_summary",
                method=f"geometry_alias_from_{source_field}",
                landmarks_used=landmarks,
                source_segment_names=source_names,
                side="midline",
                confidence=section.confidence,
                notes=["geometry alias for later matching; no clinical recommendation or device sizing is made"],
                derived_from=source_field,
                major_diameter_mm=section.major_diameter_mm,
                minor_diameter_mm=section.minor_diameter_mm,
                equivalent_diameter_mm=section.equivalent_diameter_mm,
                area_mm2=section.area_mm2,
                plane_origin=section.plane_origin,
                plane_normal=section.plane_normal,
                centerline_abscissa_mm=section.centerline_abscissa_mm,
            )

    neck_length, neck_length_qa = _detect_proximal_neck_length(samples, baseline_sample, lowest_renal, warnings, aorta.actual_name)
    group["proximal_neck_length_mm"] = neck_length
    group["proximal_neck_angulation_deg"] = _measure_neck_angulation(aorta, lowest_renal, neck_length)

    for name in ("D0", "D5", "D10", "D15"):
        sample = profile.get("named_offsets", {}).get(name)
        if isinstance(sample, dict) and sample.get("plane_origin") is not None:
            regions.append({"kind": "plane", "group": "aortic_neck", "name": name, "side": "midline", "measurement": sample})
    if baseline_sample is not None:
        regions.append(
            {
                "kind": "plane",
                "group": "aortic_neck",
                "name": "proximal_neck_measurement_plane",
                "side": "midline",
                "measurement": group["proximal_neck_diameter_mm"],
            }
        )
    if neck_length.get("candidate_neck_end_abscissa_mm") is not None:
        s = float(neck_length["candidate_neck_end_abscissa_mm"])
        origin = point_at_arclength(aorta.centerline_points, s)
        normal = tangent_at_arclength(aorta.centerline_points, s, window=1.0)
        regions.append(
            {
                "kind": "plane",
                "group": "aortic_neck",
                "name": "candidate_neck_end_marker",
                "side": "midline",
                "measurement": {
                    "plane_origin": origin.tolist(),
                    "plane_normal": normal.tolist(),
                    "equivalent_diameter_mm": group["proximal_neck_diameter_mm"].get("equivalent_diameter_mm"),
                    "status": neck_length.get("status"),
                    "confidence": neck_length.get("confidence"),
                },
            }
        )

    qa = {
        "diameter_profile_sampling_summary": profile_qa,
        "neck_length_detection_summary": neck_length_qa,
    }
    return group, qa, regions


def _build_iliac_groups(
    segments: dict[str, Optional[SegmentGeometry]],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    iliac_summary: dict[str, Any] = {}
    common_iliac: dict[str, Any] = {}
    external_iliac: dict[str, Any] = {}
    internal_iliac: dict[str, Any] = {}
    access: dict[str, Any] = {}
    regions: list[dict[str, Any]] = []

    side_data: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        cia = segments.get(f"{side}_common_iliac")
        eia = segments.get(f"{side}_external_iliac")
        iia = segments.get(f"{side}_internal_iliac")
        femoral = segments.get(f"{side}_femoral")

        cia_diameter, _ = _measure_segment_diameter(cia, side=side, method_role="common_iliac_diameter")
        cia_length = _measure_segment_length(cia, side=side, method_role="common_iliac_length")
        eia_diameter, _ = _measure_segment_diameter(eia, side=side, method_role="external_iliac_treatment_diameter")
        eia_length = _measure_segment_length(eia, side=side, method_role="external_iliac_seal_zone_length")
        if iia is None:
            warnings.append(f"{side}_internal_iliac segment not resolved; {side} IBE-specific measurements marked unmeasurable.")
        iia_diameter, _ = _measure_segment_diameter(
            iia,
            side=side,
            method_role="internal_iliac_treatment_diameter",
            missing_status="unmeasurable",
        )
        iia_length = _measure_segment_length(
            iia,
            side=side,
            method_role="internal_iliac_seal_zone_length",
            missing_status="unmeasurable",
        )

        iliac_treatment = _derived_min_measurement(
            [cia_diameter, eia_diameter],
            side=side,
            method="provisional_minimum_available_cia_eia_lumen_diameter",
            notes=["landing-zone selection is provisional and device-independent"],
        )
        distal_seal = _derived_length_from_preferred(
            eia_length,
            cia_length,
            side=side,
            method="provisional_distal_iliac_seal_zone_centerline_length",
        )

        access_points, access_extent, access_sources, access_conf = _access_path(eia, femoral)
        if access_extent == "iliac_only":
            warnings.append("femoral segments not present; access path measured to most distal external iliac endpoint.")
        access_min = _measure_access_min_diameter(
            side=side,
            external_diameter=eia_diameter,
            femoral=femoral,
            access_extent=access_extent,
        )
        access_tortuosity = _measure_access_tortuosity(
            side=side,
            path_points=access_points,
            source_segment_names=access_sources,
            access_extent=access_extent,
            confidence=access_conf,
        )

        side_data[side] = {
            "cia": cia,
            "eia": eia,
            "iia": iia,
            "cia_diameter": cia_diameter,
            "cia_length": cia_length,
            "eia_diameter": eia_diameter,
            "eia_length": eia_length,
            "iia_diameter": iia_diameter,
            "iia_length": iia_length,
            "iliac_treatment": iliac_treatment,
            "distal_seal": distal_seal,
            "access_points": access_points,
            "access_extent": access_extent,
            "access_min": access_min,
            "access_tortuosity": access_tortuosity,
        }

        for segment, group_name, path_name, length_measurement in (
            (cia, "common_iliac", f"{side}_common_iliac_path", cia_length),
            (eia, "external_iliac", f"{side}_external_iliac_path", eia_length),
            (iia, "internal_iliac", f"{side}_internal_iliac_path", iia_length),
        ):
            if segment is not None and segment.is_usable:
                regions.append(
                    {
                        "kind": "path",
                        "group": group_name,
                        "name": path_name,
                        "side": side,
                        "points": segment.centerline_points,
                        "status": length_measurement.get("status"),
                        "confidence": length_measurement.get("confidence"),
                    }
                )
        if access_points.shape[0] >= 2:
            regions.append(
                {
                    "kind": "path",
                    "group": "access",
                    "name": f"{side}_access_path",
                    "side": side,
                    "points": access_points,
                    "status": access_tortuosity.get("status"),
                    "confidence": access_tortuosity.get("confidence"),
                }
            )

    iliac_summary["left_iliac_treatment_diameter_mm"] = side_data["left"]["iliac_treatment"]
    iliac_summary["right_iliac_treatment_diameter_mm"] = side_data["right"]["iliac_treatment"]
    iliac_summary["iliac_treatment_diameter_mm"] = _summary_measurement(
        "left_iliac_treatment_diameter_mm",
        "right_iliac_treatment_diameter_mm",
        iliac_summary["left_iliac_treatment_diameter_mm"],
        iliac_summary["right_iliac_treatment_diameter_mm"],
        "mm",
    )
    iliac_summary["left_distal_iliac_seal_zone_length_mm"] = side_data["left"]["distal_seal"]
    iliac_summary["right_distal_iliac_seal_zone_length_mm"] = side_data["right"]["distal_seal"]
    iliac_summary["distal_iliac_seal_zone_length_mm"] = _summary_measurement(
        "left_distal_iliac_seal_zone_length_mm",
        "right_distal_iliac_seal_zone_length_mm",
        iliac_summary["left_distal_iliac_seal_zone_length_mm"],
        iliac_summary["right_distal_iliac_seal_zone_length_mm"],
        "mm",
    )

    common_iliac["left_common_iliac_diameter_mm"] = side_data["left"]["cia_diameter"]
    common_iliac["right_common_iliac_diameter_mm"] = side_data["right"]["cia_diameter"]
    common_iliac["common_iliac_diameter_mm"] = _summary_measurement(
        "left_common_iliac_diameter_mm",
        "right_common_iliac_diameter_mm",
        common_iliac["left_common_iliac_diameter_mm"],
        common_iliac["right_common_iliac_diameter_mm"],
        "mm",
    )
    common_iliac["left_common_iliac_length_mm"] = side_data["left"]["cia_length"]
    common_iliac["right_common_iliac_length_mm"] = side_data["right"]["cia_length"]
    common_iliac["common_iliac_length_mm"] = _summary_measurement(
        "left_common_iliac_length_mm",
        "right_common_iliac_length_mm",
        common_iliac["left_common_iliac_length_mm"],
        common_iliac["right_common_iliac_length_mm"],
        "mm",
    )

    external_iliac["left_external_iliac_treatment_diameter_mm"] = side_data["left"]["eia_diameter"]
    external_iliac["right_external_iliac_treatment_diameter_mm"] = side_data["right"]["eia_diameter"]
    external_iliac["external_iliac_treatment_diameter_mm"] = _summary_measurement(
        "left_external_iliac_treatment_diameter_mm",
        "right_external_iliac_treatment_diameter_mm",
        external_iliac["left_external_iliac_treatment_diameter_mm"],
        external_iliac["right_external_iliac_treatment_diameter_mm"],
        "mm",
    )
    external_iliac["left_external_iliac_seal_zone_length_mm"] = side_data["left"]["eia_length"]
    external_iliac["right_external_iliac_seal_zone_length_mm"] = side_data["right"]["eia_length"]
    external_iliac["external_iliac_seal_zone_length_mm"] = _summary_measurement(
        "left_external_iliac_seal_zone_length_mm",
        "right_external_iliac_seal_zone_length_mm",
        external_iliac["left_external_iliac_seal_zone_length_mm"],
        external_iliac["right_external_iliac_seal_zone_length_mm"],
        "mm",
    )

    internal_iliac["left_internal_iliac_treatment_diameter_mm"] = side_data["left"]["iia_diameter"]
    internal_iliac["right_internal_iliac_treatment_diameter_mm"] = side_data["right"]["iia_diameter"]
    internal_iliac["internal_iliac_treatment_diameter_mm"] = _summary_measurement(
        "left_internal_iliac_treatment_diameter_mm",
        "right_internal_iliac_treatment_diameter_mm",
        internal_iliac["left_internal_iliac_treatment_diameter_mm"],
        internal_iliac["right_internal_iliac_treatment_diameter_mm"],
        "mm",
    )
    internal_iliac["left_internal_iliac_seal_zone_length_mm"] = side_data["left"]["iia_length"]
    internal_iliac["right_internal_iliac_seal_zone_length_mm"] = side_data["right"]["iia_length"]
    internal_iliac["internal_iliac_seal_zone_length_mm"] = _summary_measurement(
        "left_internal_iliac_seal_zone_length_mm",
        "right_internal_iliac_seal_zone_length_mm",
        internal_iliac["left_internal_iliac_seal_zone_length_mm"],
        internal_iliac["right_internal_iliac_seal_zone_length_mm"],
        "mm",
    )

    access["left_access_vessel_min_diameter_mm"] = side_data["left"]["access_min"]
    access["right_access_vessel_min_diameter_mm"] = side_data["right"]["access_min"]
    access["access_vessel_min_diameter_mm"] = _summary_measurement(
        "left_access_vessel_min_diameter_mm",
        "right_access_vessel_min_diameter_mm",
        access["left_access_vessel_min_diameter_mm"],
        access["right_access_vessel_min_diameter_mm"],
        "mm",
    )
    access["left_access_vessel_tortuosity"] = side_data["left"]["access_tortuosity"]
    access["right_access_vessel_tortuosity"] = side_data["right"]["access_tortuosity"]
    access["access_vessel_tortuosity"] = _summary_measurement(
        "left_access_vessel_tortuosity",
        "right_access_vessel_tortuosity",
        access["left_access_vessel_tortuosity"],
        access["right_access_vessel_tortuosity"],
        "ratio",
    )

    for group_name, group in (
        ("common_iliac", common_iliac),
        ("external_iliac", external_iliac),
        ("internal_iliac", internal_iliac),
        ("access", access),
    ):
        for field, measurement in group.items():
            if isinstance(measurement, dict) and "plane_origin" in measurement:
                regions.append({"kind": "plane", "group": group_name, "name": field, "side": measurement.get("side", "unknown"), "measurement": measurement})

    qa = {
        "access_extent_by_side": {
            "left": side_data["left"]["access_extent"],
            "right": side_data["right"]["access_extent"],
        },
        "side_data": {
            side: {
                "common_iliac_resolved": side_data[side]["cia"] is not None,
                "external_iliac_resolved": side_data[side]["eia"] is not None,
                "internal_iliac_resolved": side_data[side]["iia"] is not None,
            }
            for side in ("left", "right")
        },
    }
    return iliac_summary, common_iliac, external_iliac, internal_iliac, access, regions, qa


def _build_renal_to_internal_iliac_group(
    *,
    segments: dict[str, Optional[SegmentGeometry]],
    lowest_renal: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group: dict[str, Any] = {}
    regions: list[dict[str, Any]] = []
    side_measurements: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        measurement, path = _measure_renal_to_internal_iliac(
            side=side,
            aorta=segments.get("abdominal_aorta"),
            common_iliac=segments.get(f"{side}_common_iliac"),
            internal_iliac=segments.get(f"{side}_internal_iliac"),
            lowest_renal=lowest_renal,
        )
        side_measurements[side] = measurement
        group[f"{side}_renal_to_internal_iliac_length_mm"] = measurement
        if path.shape[0] >= 2:
            regions.append(
                {
                    "kind": "path",
                    "group": "renal_to_internal_iliac",
                    "name": f"{side}_renal_to_internal_iliac_path",
                    "side": side,
                    "points": path,
                    "status": measurement.get("status"),
                    "confidence": measurement.get("confidence"),
                }
            )

    group["renal_to_internal_iliac_length_mm"] = _summary_measurement(
        "left_renal_to_internal_iliac_length_mm",
        "right_renal_to_internal_iliac_length_mm",
        side_measurements["left"],
        side_measurements["right"],
        "mm",
    )
    return group, regions


def _write_regions_vtp(regions: list[dict[str, Any]], path: Path) -> None:
    builder = RegionBuilder()
    for region in regions:
        if region.get("kind") == "path":
            builder.add_polyline(
                np.asarray(region.get("points"), dtype=float),
                group=str(region.get("group", "unknown")),
                name=str(region.get("name", "unknown")),
                side=str(region.get("side", "unknown")),
                status=str(region.get("status", "unknown")),
                confidence=region.get("confidence"),
            )
        elif region.get("kind") == "plane":
            measurement = region.get("measurement") or {}
            origin = measurement.get("plane_origin")
            normal = measurement.get("plane_normal")
            if origin is None or normal is None:
                continue
            radius = 0.5 * float(measurement.get("equivalent_diameter_mm") or measurement.get("value") or 1.0)
            builder.add_plane(
                origin=origin,
                normal=normal,
                radius=radius,
                group=str(region.get("group", "unknown")),
                name=str(region.get("name", "unknown")),
                side=str(region.get("side", "unknown")),
                status=str(measurement.get("status", region.get("status", "unknown"))),
                confidence=measurement.get("confidence", region.get("confidence")),
            )
    write_vtp(builder.build(), path)


def _failure_contract(
    *,
    input_paths: dict[str, str],
    output_paths: dict[str, str],
    warnings: list[str],
    qa: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    note = warnings[0] if warnings else "STEP4 failed before measurement."
    groups = _empty_measurement_groups("unmeasurable", note)
    contract = _base_contract(
        step_status="failed",
        input_paths=input_paths,
        output_paths=output_paths,
        measurement_groups=groups,
        warnings=warnings,
        qa=qa or {},
    )
    contract["unmeasurable_values"] = _collect_unmeasurable(groups)
    return contract


def _validate_required_fields(measurement_groups: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for group_name, fields in REQUIRED_FIELD_GROUPS.items():
        group = measurement_groups.get(group_name)
        if not isinstance(group, dict):
            missing.append(group_name)
            continue
        for field in fields:
            if field not in group:
                missing.append(f"{group_name}.{field}")
    return missing


def run_step4(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    paths = build_pipeline_paths(project_root)
    step3_dir = Path(args.step3_dir).resolve() if args.step3_dir else paths.step3_dir
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths.step4_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    named_segments_path = step3_dir / "named_segmentscolored.vtp"
    named_centerlines_path = step3_dir / "named_centerlines.vtp"
    step3_contract_path = step3_dir / "step3_naming_orientation_contract.json"
    measurements_path = output_dir / MEASUREMENTS_JSON
    regions_path = output_dir / MEASUREMENT_REGIONS_VTP

    input_paths = {
        "named_segments_vtp": _abs(named_segments_path),
        "named_centerlines_vtp": _abs(named_centerlines_path),
        "step3_contract_json": _abs(step3_contract_path),
    }
    output_paths = {
        "measurements_json": MEASUREMENTS_JSON,
        "measurement_regions_vtp": MEASUREMENT_REGIONS_VTP,
    }

    missing_inputs = [
        f"{label}: {path}"
        for label, path in (
            ("named_segmentscolored.vtp", named_segments_path),
            ("named_centerlines.vtp", named_centerlines_path),
            ("step3_naming_orientation_contract.json", step3_contract_path),
        )
        if not path.exists()
    ]
    if missing_inputs:
        warnings = ["Missing required STEP4 input(s): " + "; ".join(missing_inputs)]
        contract = _failure_contract(
            input_paths=input_paths,
            output_paths=output_paths,
            warnings=warnings,
            qa={"missing_inputs": missing_inputs},
        )
        write_json(contract, measurements_path)
        _write_regions_vtp([], regions_path)
        return contract

    try:
        step3_contract = read_json(step3_contract_path)
    except Exception as exc:
        warnings = [f"Could not read STEP3 contract JSON: {step3_contract_path} ({exc})"]
        contract = _failure_contract(input_paths=input_paths, output_paths=output_paths, warnings=warnings)
        write_json(contract, measurements_path)
        _write_regions_vtp([], regions_path)
        return contract

    step3_status = str(step3_contract.get("step_status", "failed"))
    if step3_contract.get("step_name") != "step3_naming_orientation" or step3_status not in TOP_LEVEL_STATUSES:
        warnings = [f"STEP3 contract is unusable or has unexpected schema/status: {step3_contract_path}"]
        contract = _failure_contract(
            input_paths=input_paths,
            output_paths=output_paths,
            warnings=warnings,
            qa={"step3_contract_status": step3_status, "step3_step_name": step3_contract.get("step_name")},
        )
        write_json(contract, measurements_path)
        _write_regions_vtp([], regions_path)
        return contract
    if step3_status == "failed":
        warnings = ["STEP3 contract reports failed status; STEP4 cannot produce trustworthy measurements."]
        contract = _failure_contract(
            input_paths=input_paths,
            output_paths=output_paths,
            warnings=warnings,
            qa={"step3_contract_status": step3_status},
        )
        write_json(contract, measurements_path)
        _write_regions_vtp([], regions_path)
        return contract

    warnings = [str(value) for value in step3_contract.get("warnings", [])]
    if step3_status == "requires_review":
        warnings.append("STEP3 contract status is requires_review; STEP4 measurements inherit review context.")

    try:
        named_surface = read_vtp(named_segments_path)
        named_centerlines = read_vtp(named_centerlines_path)
    except Exception as exc:
        failure_warnings = warnings + [f"Could not read STEP3 VTP input(s): {exc}"]
        contract = _failure_contract(input_paths=input_paths, output_paths=output_paths, warnings=failure_warnings)
        write_json(contract, measurements_path)
        _write_regions_vtp([], regions_path)
        return contract

    segments, resolved_names, missing_priority = _build_segments(step3_contract, named_surface, named_centerlines, warnings)
    aorta = segments.get("abdominal_aorta")
    if aorta is None or not aorta.is_usable:
        failure_warnings = warnings + ["Abdominal aorta segment could not be resolved; STEP4 cannot produce a trustworthy EVAR geometry contract."]
        contract = _failure_contract(
            input_paths=input_paths,
            output_paths=output_paths,
            warnings=failure_warnings,
            qa={
                "step3_contract_status": step3_status,
                "resolved_segment_names": resolved_names,
                "missing_priority_segments": missing_priority,
            },
        )
        write_json(contract, measurements_path)
        _write_regions_vtp([], regions_path)
        return contract

    lowest_renal, lowest_renal_qa = _select_lowest_renal(step3_contract, segments, warnings)
    aortic_neck, aortic_qa, aortic_regions = _build_aortic_neck_group(
        aorta=aorta,
        lowest_renal=lowest_renal,
        warnings=warnings,
    )
    (
        iliac_summary,
        common_iliac,
        external_iliac,
        internal_iliac,
        access,
        iliac_regions,
        iliac_qa,
    ) = _build_iliac_groups(segments, warnings)
    renal_to_internal_iliac, renal_regions = _build_renal_to_internal_iliac_group(
        segments=segments,
        lowest_renal=lowest_renal,
    )

    measurement_groups = {
        "aortic_neck": aortic_neck,
        "iliac_summary": iliac_summary,
        "common_iliac": common_iliac,
        "external_iliac": external_iliac,
        "internal_iliac": internal_iliac,
        "renal_to_internal_iliac": renal_to_internal_iliac,
        "access": access,
    }
    missing_fields = _validate_required_fields(measurement_groups)
    if missing_fields:
        raise Step4Failure("Internal STEP4 implementation did not emit required fields: " + ", ".join(missing_fields))

    unmeasurable_values = _collect_unmeasurable(measurement_groups)
    counts = _measurement_counts(measurement_groups)
    qa = {
        "step3_contract_status": step3_status,
        "resolved_segment_names": {
            canonical: actual for canonical, actual in resolved_names.items() if canonical in PRIORITY_SEGMENTS and actual
        },
        "missing_priority_segments": missing_priority,
        "lowest_renal_selection": lowest_renal_qa,
        "diameter_profile_sampling_summary": aortic_qa.get("diameter_profile_sampling_summary", {}),
        "neck_length_detection_summary": aortic_qa.get("neck_length_detection_summary", {}),
        "access_extent_by_side": iliac_qa.get("access_extent_by_side", {}),
        "measurement_counts": counts,
        "unmeasurable_count": int(counts.get("unmeasurable", 0) + counts.get("missing_required_landmark", 0)),
        "requires_review_count": int(counts.get("requires_review", 0)),
        "field_validation_missing": missing_fields,
    }
    step_status = _overall_status(step3_status=step3_status, measurement_groups=measurement_groups)
    if missing_priority and all(name in missing_priority for name in ("left_common_iliac", "right_common_iliac", "left_external_iliac", "right_external_iliac")):
        step_status = "failed"
        warnings.append("Required iliac anatomy is too incomplete for a trustworthy STEP4 contract.")

    contract = _base_contract(
        step_status=step_status,
        input_paths=input_paths,
        output_paths=output_paths,
        measurement_groups=measurement_groups,
        warnings=warnings,
        qa=qa,
    )
    contract["unmeasurable_values"] = unmeasurable_values

    regions = aortic_regions + iliac_regions + renal_regions
    _write_regions_vtp(regions, regions_path)
    write_json(contract, measurements_path)
    return contract


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    paths = build_pipeline_paths(project_root)
    parser = argparse.ArgumentParser(description="STEP4 EVAR geometry measurement compatibility wrapper.")
    parser.add_argument("--project-root", default=str(project_root), help="Project root.")
    parser.add_argument("--step3-dir", default=str(paths.step3_dir), help="STEP3 output directory.")
    parser.add_argument("--output-dir", default=str(paths.step4_dir), help="STEP4 output directory.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        contract = run_step4(args)
    except Step4Failure as exc:
        print(f"STEP4 failed: {exc}")
        return 1
    status = contract.get("step_status", "failed")
    print(
        "STEP4 completed: "
        f"{status} | "
        f"measurements={contract.get('qa', {}).get('measurement_counts', {}).get('total', 0)} | "
        f"requires_review={contract.get('qa', {}).get('requires_review_count', 0)} | "
        f"unmeasurable={contract.get('qa', {}).get('unmeasurable_count', 0)}"
    )
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())


def _sample_abscissae(points: np.ndarray, max_samples: int = 7) -> list[float]:
    length = polyline_length(points)
    if length <= EPS:
        return []
    if length <= 3.0:
        return [0.5 * length]
    margin = min(2.0, 0.15 * length)
    count = max(1, min(max_samples, int(math.floor((length - 2.0 * margin) / 2.0)) + 1))
    if count <= 1:
        return [0.5 * length]
    return [float(v) for v in np.linspace(margin, length - margin, count)]


def _measure_segment_diameter(
    segment: Optional[SegmentGeometry],
    *,
    side: str,
    method_role: str,
    missing_status: str = "missing_required_landmark",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if segment is None or not segment.is_usable:
        return (
            _missing_measurement(
                status=missing_status,
                unit="mm",
                method=f"{method_role}; centerline_orthogonal_minimum_profile",
                side=side,
                notes=["required segment centerline was not resolved"],
            ),
            [],
        )

    samples: list[dict[str, Any]] = []
    for s in _sample_abscissae(segment.centerline_points):
        section = _cross_section_at_abscissa(
            segment,
            s,
            f"{method_role}; centerline_orthogonal_minimum_profile",
        )
        samples.append({"abscissa": float(s), "section": section})

    usable = [
        sample
        for sample in samples
        if sample["section"].equivalent_diameter_mm is not None
    ]
    if not usable:
        return (
            _missing_measurement(
                status="unmeasurable",
                unit="mm",
                method=f"{method_role}; centerline_orthogonal_minimum_profile",
                side=side,
                source_segment_names=[segment.actual_name],
                notes=["no usable orthogonal cross-section was found along the segment"],
            ),
            samples,
        )

    selected = min(usable, key=lambda row: float(row["section"].equivalent_diameter_mm))
    section = selected["section"]
    status = "measured" if section.confidence >= 0.55 else "requires_review"
    return (
        _diameter_measurement_from_section(
            section,
            value_key="equivalent_diameter_mm",
            side=side,
            source_segment_names=[segment.actual_name],
            landmarks_used=[f"{segment.canonical_name}_centerline"],
            method=f"{method_role}; centerline_orthogonal_minimum_profile",
            status_override=status,
            notes=["minimum equivalent lumen diameter across sampled segment profile"],
        )
        | {
            "sample_count": int(len(samples)),
            "usable_sample_count": int(len(usable)),
            "selected_sample_abscissa_mm": float(selected["abscissa"]),
        },
        samples,
    )


def _measure_segment_length(
    segment: Optional[SegmentGeometry],
    *,
    side: str,
    method_role: str,
    missing_status: str = "missing_required_landmark",
) -> dict[str, Any]:
    if segment is None or not segment.is_usable:
        return _missing_measurement(
            status=missing_status,
            unit="mm",
            method=f"{method_role}; centerline_arc_length",
            side=side,
            notes=["required segment centerline was not resolved"],
        )
    confidence = min(segment.confidence, 0.9)
    return _measurement(
        value=float(segment.length),
        unit="mm",
        status="measured" if confidence >= 0.55 else "requires_review",
        method=f"{method_role}; centerline_arc_length",
        landmarks_used=[f"{segment.canonical_name}_proximal_endpoint", f"{segment.canonical_name}_distal_endpoint"],
        source_segment_names=[segment.actual_name],
        side=side,
        confidence=confidence,
        notes=[],
    )


def _derived_min_measurement(
    candidates: list[dict[str, Any]],
    *,
    side: str,
    method: str,
    notes: Optional[list[str]] = None,
) -> dict[str, Any]:
    usable = [row for row in candidates if row.get("value") is not None]
    source_names = sorted({name for row in usable for name in row.get("source_segment_names", [])})
    if not usable:
        return _missing_measurement(
            status="missing_required_landmark",
            unit="mm",
            method=method,
            side=side,
            notes=(notes or []) + ["no source diameter measurements were available"],
        )
    selected = min(usable, key=lambda row: float(row["value"]))
    review = any(row.get("status") in {"requires_review", "unmeasurable", "missing_required_landmark"} for row in candidates)
    return _measurement(
        value=selected.get("value"),
        unit="mm",
        status="requires_review" if review or selected.get("status") == "requires_review" else "measured",
        method=method,
        landmarks_used=["candidate_iliac_treatment_region"],
        source_segment_names=source_names,
        side=side,
        confidence=selected.get("confidence"),
        notes=(notes or []) + ["side-specific value is the minimum of available candidate iliac lumen diameters"],
        selected_source_measurement=selected.get("method"),
    )


def _derived_length_from_preferred(
    preferred: dict[str, Any],
    fallback: Optional[dict[str, Any]],
    *,
    side: str,
    method: str,
) -> dict[str, Any]:
    source = preferred if preferred.get("value") is not None else fallback
    if source is None or source.get("value") is None:
        return _missing_measurement(
            status="missing_required_landmark",
            unit="mm",
            method=method,
            side=side,
            notes=["no distal iliac seal-zone path could be resolved"],
        )
    status = source.get("status", "measured")
    if source is fallback:
        status = "requires_review"
    return _measurement(
        value=source.get("value"),
        unit="mm",
        status=status,
        method=method,
        landmarks_used=source.get("landmarks_used", []),
        source_segment_names=source.get("source_segment_names", []),
        side=side,
        confidence=source.get("confidence"),
        notes=[
            "provisional distal iliac seal-zone length uses external iliac region when available",
        ]
        + (["fallback used common iliac length because external iliac was unavailable"] if source is fallback else []),
    )
