#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import vtk  # type: ignore


INPUT_CENTERLINES_VTP_PATH = r"C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines.vtp"
INPUT_SURFACE_WITH_CENTERLINES_VTP_PATH = r"C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_surface_with_centerlines.vtp"
INPUT_METADATA_JSON_PATH = r"C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines_metadata.json"
OUTPUT_COLORED_VTP_PATH = r"C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\infrarenal_neck_colored.vtp"
OUTPUT_REPORT_TXT_PATH = r"C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\infrarenal_neck_report.txt"


EPS = 1.0e-12
LABEL_AORTA_TRUNK = 1
LABEL_RIGHT_RENAL = 5
LABEL_LEFT_RENAL = 6
GEOMETRY_TYPE_SURFACE = 1
GEOMETRY_TYPE_CENTERLINE = 2
TRUNK_DIRECTION_REVERSED_WARNING = "W_TRUNK_DIRECTION_REVERSED_FOR_INFRarenal_ANALYSIS"

TRUNK_PRIMARY_NAME = "abdominal_aorta_trunk"
RIGHT_RENAL_PRIMARY_NAME = "right_renal_artery"
LEFT_RENAL_PRIMARY_NAME = "left_renal_artery"
TRUNK_NAME_ALIASES = {
    TRUNK_PRIMARY_NAME,
    "abdominalaortatrunk",
    "aortatrunk",
    "trunk",
}
RIGHT_RENAL_ALIASES = {RIGHT_RENAL_PRIMARY_NAME, "right_renal", "rightrenal", "renalright"}
LEFT_RENAL_ALIASES = {LEFT_RENAL_PRIMARY_NAME, "left_renal", "leftrenal", "renalleft"}


@dataclass
class PolylinePath:
    points: np.ndarray
    point_ids: List[int]
    cumulative_s: np.ndarray
    cell_ids: List[int]


@dataclass
class ResolvedAnchor:
    side: str
    source: str
    trunk_s_mm: float
    point_xyz: np.ndarray
    scaffold_point_id: Optional[int]
    nearest_trunk_point_id: Optional[int]


@dataclass
class SliceMeasurement:
    sample_s_abs_mm: float
    origin_xyz: np.ndarray
    tangent_xyz: np.ndarray
    basis_u_xyz: np.ndarray
    basis_v_xyz: np.ndarray
    contour_2d: np.ndarray
    contour_3d: np.ndarray
    area_mm2: float
    major_mm: float
    minor_mm: float
    eq_mm: float
    contains_origin: bool


def unit(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n < EPS:
        return np.zeros_like(vv, dtype=float)
    return (vv / n).astype(float)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum() or ch == "_")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Metadata JSON must decode to an object: {path}")
    return data


def load_vtp(path: str) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    out = reader.GetOutput()
    if out is None or out.GetNumberOfPoints() < 1:
        raise RuntimeError(f"Failed to read VTP or VTP is empty: {path}")
    pd = vtk.vtkPolyData()
    pd.DeepCopy(out)
    return pd


def write_vtp(pd: vtk.vtkPolyData, path: str, binary: bool = True) -> None:
    ensure_parent_dir(path)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(pd)
    if binary:
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def clone_polydata(pd: vtk.vtkPolyData) -> vtk.vtkPolyData:
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    return out


def iter_array_names(attrs: Any) -> List[str]:
    names: List[str] = []
    if attrs is None:
        return names
    for i in range(attrs.GetNumberOfArrays()):
        arr = attrs.GetAbstractArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if name:
            names.append(str(name))
    return names


def find_array_name(
    attrs: Any,
    preferred_names: Sequence[str],
    contains_tokens: Optional[Sequence[str]] = None,
) -> Optional[str]:
    name_map = {normalize_name(name): name for name in iter_array_names(attrs)}
    for preferred in preferred_names:
        key = normalize_name(preferred)
        if key in name_map:
            return name_map[key]
    if contains_tokens:
        norm_tokens = [normalize_name(tok) for tok in contains_tokens]
        for norm_name, original in name_map.items():
            if all(tok in norm_name for tok in norm_tokens):
                return original
    return None


def get_cell_string_values(pd: vtk.vtkPolyData, array_name: str) -> List[str]:
    arr = pd.GetCellData().GetAbstractArray(array_name)
    if arr is None:
        return []
    values: List[str] = []
    for i in range(pd.GetNumberOfCells()):
        if isinstance(arr, vtk.vtkStringArray):
            values.append(str(arr.GetValue(i)))
        else:
            values.append(str(arr.GetVariantValue(i).ToString()))
    return values


def get_cell_numeric_values(pd: vtk.vtkPolyData, array_name: str, default: float = 0.0) -> List[float]:
    arr = pd.GetCellData().GetArray(array_name)
    if arr is None:
        return [default] * pd.GetNumberOfCells()
    values: List[float] = []
    for i in range(pd.GetNumberOfCells()):
        values.append(float(arr.GetTuple1(i)))
    return values


def get_field_point_id(fd: vtk.vtkFieldData, landmark_key: str) -> Optional[int]:
    names = [f"Landmark_{landmark_key}_PointId", f"Landmark{landmark_key}PointId"]
    for name in names:
        arr = fd.GetArray(name)
        if arr is not None and arr.GetNumberOfTuples() >= 1:
            return int(round(float(arr.GetTuple1(0))))
    return None


def get_metadata_landmark_xyz(meta: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    landmarks = meta.get("landmarks_xyz_canonical", {})
    if not isinstance(landmarks, dict):
        return None
    raw = landmarks.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=float)
    except Exception:
        return None


def get_metadata_canonical_frame_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
    summary = meta.get("canonical_frame_summary", {})
    return summary if isinstance(summary, dict) else {}


def get_metadata_horizontal_frame_source(meta: Dict[str, Any]) -> str:
    direct = meta.get("horizontal_frame_source")
    if direct not in (None, ""):
        return str(direct)
    summary = get_metadata_canonical_frame_summary(meta)
    source = summary.get("source", "unknown")
    return str(source) if source not in (None, "") else "unknown"


def get_metadata_horizontal_frame_confidence(meta: Dict[str, Any]) -> float:
    direct = safe_float(meta.get("horizontal_frame_confidence"), float("nan"))
    if math.isfinite(direct):
        return direct
    summary = get_metadata_canonical_frame_summary(meta)
    return safe_float(summary.get("confidence"), float("nan"))


def get_metadata_ap_orientation_certain(meta: Dict[str, Any]) -> bool:
    if "ap_orientation_certain" in meta:
        return bool(meta.get("ap_orientation_certain"))
    summary = get_metadata_canonical_frame_summary(meta)
    if "ap_sign_warn" in summary:
        return not bool(summary.get("ap_sign_warn"))
    return True


def get_metadata_branch_rows(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows_raw = meta.get("centerline_branch_summaries")
    if not isinstance(rows_raw, list):
        branch_summary = meta.get("branch_summary", {})
        if isinstance(branch_summary, dict):
            rows_raw = branch_summary.get("branches")
    if not isinstance(rows_raw, list):
        return []

    rows: List[Dict[str, Any]] = []
    for fallback_index, raw in enumerate(rows_raw):
        if not isinstance(raw, dict):
            continue
        index_value = raw.get("index", raw.get("polyline_index", fallback_index))
        try:
            index = int(index_value)
        except Exception:
            index = int(fallback_index)
        branch_name = raw.get("branch_name", raw.get("name", ""))
        branch_id = raw.get("branch_id", raw.get("label_id"))
        topology_role = raw.get("topology_role", "")
        rows.append(
            {
                "index": int(index),
                "branch_name": str(branch_name) if branch_name is not None else "",
                "branch_id": (None if branch_id is None else int(branch_id)),
                "topology_role": str(topology_role) if topology_role is not None else "",
            }
        )
    rows.sort(key=lambda row: int(row["index"]))
    return rows


def get_metadata_branch_rows_by_index(meta: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(row["index"]): row for row in get_metadata_branch_rows(meta)}


def get_metadata_branch_cell_order(
    meta: Dict[str, Any],
    branch_aliases: Sequence[str],
    numeric_label_id: Optional[int] = None,
    topology_roles: Optional[Sequence[str]] = None,
) -> List[int]:
    alias_set = {normalize_name(name) for name in branch_aliases}
    role_set = {normalize_name(role) for role in (topology_roles or [])}
    ordered_ids: List[int] = []
    for row in get_metadata_branch_rows(meta):
        row_name = normalize_name(row.get("branch_name"))
        row_role = normalize_name(row.get("topology_role"))
        row_id = row.get("branch_id")
        if row_name in alias_set or (role_set and row_role in role_set) or (numeric_label_id is not None and row_id == int(numeric_label_id)):
            ordered_ids.append(int(row["index"]))
    return ordered_ids


def get_metadata_branch_anchor_entries(meta: Dict[str, Any], branch_aliases: Sequence[str]) -> List[Dict[str, Any]]:
    anchors = meta.get("branch_proximal_anchors", {})
    if not isinstance(anchors, dict):
        return []

    normalized_aliases = {normalize_name(name) for name in branch_aliases}
    matched_entries: List[Dict[str, Any]] = []
    for key, value in anchors.items():
        if normalize_name(key) not in normalized_aliases:
            continue
        if isinstance(value, dict):
            matched_entries.append(dict(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    matched_entries.append(dict(item))
    return matched_entries


def get_xyz_triplet(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
    except Exception:
        return None


def format_mm(x: float) -> str:
    return "nan" if not math.isfinite(x) else f"{x:.3f}"


def join_warnings(warnings: Sequence[str]) -> str:
    clean = [str(w).replace("\n", " ").strip() for w in warnings if str(w).strip()]
    return " | ".join(clean)


def build_polyline_path(points: np.ndarray, point_ids: List[int], cell_ids: List[int]) -> PolylinePath:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise RuntimeError("Polyline path needs at least 2 points.")
    s = np.zeros(pts.shape[0], dtype=float)
    s[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return PolylinePath(points=pts, point_ids=list(point_ids), cumulative_s=s, cell_ids=list(cell_ids))


def reverse_path(path: PolylinePath) -> PolylinePath:
    pts = path.points[::-1].copy()
    point_ids = list(reversed(path.point_ids))
    s = np.zeros(len(pts), dtype=float)
    if len(pts) > 1:
        s[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return PolylinePath(points=pts, point_ids=point_ids, cumulative_s=s, cell_ids=list(reversed(path.cell_ids)))


def polyline_length(path: PolylinePath) -> float:
    return float(path.cumulative_s[-1]) if path.cumulative_s.size else 0.0


def find_stitched_index_for_point_id(path: PolylinePath, point_id: int) -> Optional[int]:
    try:
        return path.point_ids.index(int(point_id))
    except ValueError:
        return None


def project_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < EPS:
        return a.copy(), 0.0, float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = clamp(t, 0.0, 1.0)
    q = a + t * ab
    d = float(np.linalg.norm(p - q))
    return q, t, d


def project_point_to_polyline(path: PolylinePath, p: np.ndarray) -> Dict[str, Any]:
    point = np.asarray(p, dtype=float).reshape(3)
    pts = path.points
    best: Dict[str, Any] = {
        "distance_mm": float("inf"),
        "s_abs_mm": 0.0,
        "segment_index": 0,
        "segment_t": 0.0,
        "projected_xyz": pts[0].copy(),
    }
    for i in range(len(pts) - 1):
        q, t, d = project_point_to_segment(point, pts[i], pts[i + 1])
        if d < best["distance_mm"]:
            seg_len = float(np.linalg.norm(pts[i + 1] - pts[i]))
            best = {
                "distance_mm": d,
                "s_abs_mm": float(path.cumulative_s[i] + t * seg_len),
                "segment_index": i,
                "segment_t": t,
                "projected_xyz": q,
            }
    return best


def interpolate_point_on_path(path: PolylinePath, s_abs_mm: float) -> np.ndarray:
    s = float(clamp(s_abs_mm, 0.0, polyline_length(path)))
    pts = path.points
    if s <= 0.0:
        return pts[0].copy()
    if s >= polyline_length(path):
        return pts[-1].copy()
    idx = int(np.searchsorted(path.cumulative_s, s, side="right") - 1)
    idx = max(0, min(idx, len(pts) - 2))
    s0 = float(path.cumulative_s[idx])
    s1 = float(path.cumulative_s[idx + 1])
    if s1 - s0 < EPS:
        return pts[idx].copy()
    t = (s - s0) / (s1 - s0)
    return (1.0 - t) * pts[idx] + t * pts[idx + 1]


def evaluate_path_point_and_tangent(path: PolylinePath, s_abs_mm: float, ds_mm: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    s = float(clamp(s_abs_mm, 0.0, polyline_length(path)))
    p = interpolate_point_on_path(path, s)
    s_lo = max(0.0, s - ds_mm)
    s_hi = min(polyline_length(path), s + ds_mm)
    if s_hi - s_lo < 1.0e-6:
        if s < polyline_length(path):
            p2 = interpolate_point_on_path(path, min(polyline_length(path), s + max(ds_mm, 1.0)))
            t = unit(p2 - p)
        else:
            p1 = interpolate_point_on_path(path, max(0.0, s - max(ds_mm, 1.0)))
            t = unit(p - p1)
    else:
        p_lo = interpolate_point_on_path(path, s_lo)
        p_hi = interpolate_point_on_path(path, s_hi)
        t = unit(p_hi - p_lo)
    if np.linalg.norm(t) < EPS:
        idx = int(np.searchsorted(path.cumulative_s, s, side="right") - 1)
        idx = max(0, min(idx, len(path.points) - 2))
        t = unit(path.points[idx + 1] - path.points[idx])
    if np.linalg.norm(t) < EPS:
        raise RuntimeError("Failed to evaluate a valid trunk tangent.")
    return p, t


def choose_cell_label_array(pd: vtk.vtkPolyData) -> Tuple[Optional[str], Optional[str]]:
    cd = pd.GetCellData()
    string_name = find_array_name(cd, ["BranchName", "BranchLabelName", "AnatomicalName"])
    numeric_name = find_array_name(cd, ["BranchId", "BranchLabelId", "AnatomicalLabelId"])
    if string_name is None:
        string_name = find_array_name(cd, [], contains_tokens=["branch", "name"])
    if numeric_name is None:
        numeric_name = find_array_name(cd, [], contains_tokens=["branch", "id"])
    return string_name, numeric_name


def choose_geometry_type_arrays(pd: vtk.vtkPolyData) -> Tuple[Optional[str], Optional[str]]:
    cd = pd.GetCellData()
    string_name = find_array_name(cd, ["GeometryType", "GeometryTypeName"])
    numeric_name = find_array_name(cd, ["GeometryTypeId"])
    if string_name is None:
        string_name = find_array_name(cd, [], contains_tokens=["geometry", "type"])
    return string_name, numeric_name


def extract_centerline_polylines(pd: vtk.vtkPolyData, meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    string_label_name, numeric_label_name = choose_cell_label_array(pd)
    string_values = get_cell_string_values(pd, string_label_name) if string_label_name else []
    numeric_values = get_cell_numeric_values(pd, numeric_label_name) if numeric_label_name else [0.0] * pd.GetNumberOfCells()
    metadata_rows_by_index = get_metadata_branch_rows_by_index(meta or {})

    polylines: List[Dict[str, Any]] = []
    for ci in range(pd.GetNumberOfCells()):
        cell = pd.GetCell(ci)
        if cell is None:
            continue
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
            continue
        ids = [int(cell.GetPointId(k)) for k in range(cell.GetNumberOfPoints())]
        if len(ids) < 2:
            continue
        pts = np.array([pd.GetPoint(pid) for pid in ids], dtype=float)
        label_name = string_values[ci] if ci < len(string_values) else ""
        label_id = int(round(numeric_values[ci])) if ci < len(numeric_values) else 0
        meta_row = metadata_rows_by_index.get(int(ci), {})
        if not normalize_name(label_name):
            meta_name = meta_row.get("branch_name")
            if meta_name is not None:
                label_name = str(meta_name)
        if int(label_id) == 0:
            meta_label_id = meta_row.get("branch_id")
            if meta_label_id is not None:
                label_id = int(meta_label_id)
        polylines.append(
            {
                "cell_id": int(ci),
                "point_ids": ids,
                "points": pts,
                "label_name": str(label_name),
                "label_id": int(label_id),
                "topology_role": str(meta_row.get("topology_role", "")),
            }
        )
    if not polylines:
        raise RuntimeError("No centerline polyline cells found in oriented scaffold.")
    return polylines


def select_polylines_by_label(
    polylines: Sequence[Dict[str, Any]],
    aliases: Sequence[str],
    numeric_label_id: Optional[int],
    topology_roles: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    alias_set = {normalize_name(a) for a in aliases}
    role_set = {normalize_name(role) for role in (topology_roles or [])}
    for item in polylines:
        label_name = normalize_name(item.get("label_name"))
        label_id = int(item.get("label_id", -999))
        topology_role = normalize_name(item.get("topology_role"))
        if (
            label_name in alias_set
            or (numeric_label_id is not None and label_id == int(numeric_label_id))
            or (role_set and topology_role in role_set)
        ):
            out.append(item)
    return out


def point_or_coord_from_landmark(
    pd: vtk.vtkPolyData,
    path: PolylinePath,
    field_pid: Optional[int],
    meta_xyz: Optional[np.ndarray],
    label: str,
    warnings: List[str],
    prefer_metadata: bool = True,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source": "unresolved",
        "s_abs_mm": float("nan"),
        "xyz": None,
        "scaffold_point_id": None,
        "nearest_trunk_point_id": None,
    }
    field_xyz = None
    field_result: Optional[Dict[str, Any]] = None
    if field_pid is not None and 0 <= int(field_pid) < pd.GetNumberOfPoints():
        field_xyz = np.array(pd.GetPoint(int(field_pid)), dtype=float)
        idx = find_stitched_index_for_point_id(path, int(field_pid))
        if idx is not None:
            field_result = {
                "source": "centerline_field_point_id",
                "s_abs_mm": float(path.cumulative_s[idx]),
                "xyz": field_xyz,
                "scaffold_point_id": int(field_pid),
                "nearest_trunk_point_id": int(path.point_ids[idx]),
            }
        else:
            proj = project_point_to_polyline(path, field_xyz)
            nearest_idx = int(np.argmin(np.linalg.norm(path.points - field_xyz.reshape(1, 3), axis=1)))
            field_result = {
                "source": "centerline_field_point_id_projected",
                "s_abs_mm": float(proj["s_abs_mm"]),
                "xyz": field_xyz,
                "scaffold_point_id": int(field_pid),
                "nearest_trunk_point_id": int(path.point_ids[nearest_idx]),
            }

    meta_result: Optional[Dict[str, Any]] = None
    if meta_xyz is not None:
        proj_meta = project_point_to_polyline(path, meta_xyz)
        nearest_idx = int(np.argmin(np.linalg.norm(path.points - meta_xyz.reshape(1, 3), axis=1)))
        meta_result = {
            "source": "metadata_landmark_coord",
            "s_abs_mm": float(proj_meta["s_abs_mm"]),
            "xyz": np.asarray(meta_xyz, dtype=float),
            "scaffold_point_id": None,
            "nearest_trunk_point_id": int(path.point_ids[nearest_idx]),
        }

    if meta_result is not None and field_xyz is not None:
        mismatch = float(np.linalg.norm(np.asarray(meta_xyz, dtype=float) - field_xyz))
        if mismatch > 1.0:
            warnings.append(
                f"W_{label.upper()}_FIELD_METADATA_MISMATCH: metadata landmark differs from scaffold point by {mismatch:.3f} mm; metadata was preferred."
                if prefer_metadata
                else f"W_{label.upper()}_FIELD_METADATA_MISMATCH: metadata landmark differs from scaffold point by {mismatch:.3f} mm; scaffold point ID was preferred."
            )

    preferred = [meta_result, field_result] if prefer_metadata else [field_result, meta_result]
    for item in preferred:
        if item is not None:
            result.update(item)
            break
    return result


def resolve_named_landmark_on_trunk(
    center_pd: vtk.vtkPolyData,
    meta: Dict[str, Any],
    trunk_path: PolylinePath,
    landmark_key: str,
    warnings: List[str],
) -> Dict[str, Any]:
    fd = center_pd.GetFieldData()
    field_pid = get_field_point_id(fd, landmark_key)
    meta_xyz = get_metadata_landmark_xyz(meta, landmark_key)
    return point_or_coord_from_landmark(center_pd, trunk_path, field_pid, meta_xyz, landmark_key, warnings, prefer_metadata=True)


def get_anchor_trunk_s_mm(anchor: Any) -> float:
    if anchor is None:
        return float("nan")
    if isinstance(anchor, ResolvedAnchor):
        return safe_float(anchor.trunk_s_mm, float("nan"))
    if isinstance(anchor, dict):
        return safe_float(anchor.get("s_abs_mm"), float("nan"))
    return safe_float(getattr(anchor, "trunk_s_mm", float("nan")), float("nan"))


def validate_trunk_landmark_order(
    trunk_path: PolylinePath,
    inlet: Dict[str, Any],
    bif: Dict[str, Any],
    renals: Dict[str, Any],
    warnings: List[str],
    emit_warning: bool = True,
) -> Dict[str, Any]:
    trunk_len = polyline_length(trunk_path)
    tol = max(1.0e-6, 1.0e-4 * max(trunk_len, 1.0))

    s_inlet = get_anchor_trunk_s_mm(inlet)
    s_bif = get_anchor_trunk_s_mm(bif)
    s_right = get_anchor_trunk_s_mm(renals.get("right"))
    s_left = get_anchor_trunk_s_mm(renals.get("left"))
    resolved_renals = [s for s in (s_right, s_left) if math.isfinite(s)]
    s_lowest_renal = max(resolved_renals) if resolved_renals else float("nan")

    failures: List[str] = []

    if not math.isfinite(s_inlet):
        failures.append("Inlet landmark was unresolved on the trunk path.")
    if not math.isfinite(s_bif):
        failures.append("Bifurcation landmark was unresolved on the trunk path.")

    for label, value in (
        ("s_inlet", s_inlet),
        ("s_bif", s_bif),
        ("s_right_renal", s_right),
        ("s_left_renal", s_left),
    ):
        if math.isfinite(value) and not (-tol <= value <= trunk_len + tol):
            failures.append(f"{label}={value:.6f} falls outside the stitched trunk arclength range [0, {trunk_len:.6f}].")

    if math.isfinite(s_inlet) and math.isfinite(s_bif) and not (s_inlet + tol < s_bif):
        failures.append(f"s_inlet={s_inlet:.6f} must be < s_bif={s_bif:.6f}.")
    if math.isfinite(s_right) and math.isfinite(s_inlet) and not (s_inlet + tol < s_right):
        failures.append(f"s_inlet={s_inlet:.6f} must be < s_right_renal={s_right:.6f}.")
    if math.isfinite(s_right) and math.isfinite(s_bif) and not (s_right + tol < s_bif):
        failures.append(f"s_right_renal={s_right:.6f} must be < s_bif={s_bif:.6f}.")
    if math.isfinite(s_left) and math.isfinite(s_inlet) and not (s_inlet + tol < s_left):
        failures.append(f"s_inlet={s_inlet:.6f} must be < s_left_renal={s_left:.6f}.")
    if math.isfinite(s_left) and math.isfinite(s_bif) and not (s_left + tol < s_bif):
        failures.append(f"s_left_renal={s_left:.6f} must be < s_bif={s_bif:.6f}.")
    if math.isfinite(s_lowest_renal) and math.isfinite(s_bif) and not (s_lowest_renal + tol < s_bif):
        failures.append(f"s_lowest_renal={s_lowest_renal:.6f} must be < s_bif={s_bif:.6f}.")

    if emit_warning and failures:
        warnings.append("W_TRUNK_LANDMARK_ORDER_INVALID: " + " ".join(failures))

    return {
        "valid": not failures,
        "s_inlet": s_inlet,
        "s_bif": s_bif,
        "s_right_renal": s_right,
        "s_left_renal": s_left,
        "s_lowest_renal": s_lowest_renal,
        "failures": failures,
    }


def normalize_trunk_direction_for_infrarenal_analysis(
    center_pd: vtk.vtkPolyData,
    meta: Dict[str, Any],
    trunk_path: PolylinePath,
    warnings: List[str],
) -> PolylinePath:
    trial_warnings: List[str] = []
    inlet = resolve_named_landmark_on_trunk(center_pd, meta, trunk_path, "Inlet", trial_warnings)
    bif = resolve_named_landmark_on_trunk(center_pd, meta, trunk_path, "Bifurcation", trial_warnings)
    renals = {
        "right": resolve_named_landmark_on_trunk(center_pd, meta, trunk_path, "RightRenalOrigin", trial_warnings),
        "left": resolve_named_landmark_on_trunk(center_pd, meta, trunk_path, "LeftRenalOrigin", trial_warnings),
    }
    current_order = validate_trunk_landmark_order(trunk_path, inlet, bif, renals, [], emit_warning=False)
    if current_order["valid"]:
        return trunk_path

    reversed_path = reverse_path(trunk_path)
    reversed_trial_warnings: List[str] = []
    inlet_reversed = resolve_named_landmark_on_trunk(center_pd, meta, reversed_path, "Inlet", reversed_trial_warnings)
    bif_reversed = resolve_named_landmark_on_trunk(center_pd, meta, reversed_path, "Bifurcation", reversed_trial_warnings)
    renals_reversed = {
        "right": resolve_named_landmark_on_trunk(center_pd, meta, reversed_path, "RightRenalOrigin", reversed_trial_warnings),
        "left": resolve_named_landmark_on_trunk(center_pd, meta, reversed_path, "LeftRenalOrigin", reversed_trial_warnings),
    }
    reversed_order = validate_trunk_landmark_order(reversed_path, inlet_reversed, bif_reversed, renals_reversed, [], emit_warning=False)
    if reversed_order["valid"]:
        warnings.append(f"{TRUNK_DIRECTION_REVERSED_WARNING}: trunk arclength was reversed so s increases from inlet toward bifurcation.")
        return reversed_path

    current_msg = " ".join(current_order["failures"]) if current_order["failures"] else "order was unvalidated."
    reversed_msg = " ".join(reversed_order["failures"]) if reversed_order["failures"] else "order was unvalidated."
    raise RuntimeError(
        "Unable to normalize trunk direction for infrarenal analysis. "
        f"Current order failed: {current_msg} Reversed order failed: {reversed_msg}"
    )


def concatenate_ordered_polylines(
    polylines: Sequence[Dict[str, Any]],
    start_xyz: Optional[np.ndarray],
    end_xyz: Optional[np.ndarray],
    warnings: List[str],
    endpoint_tol_mm: float = 1.0e-3,
) -> PolylinePath:
    if not polylines:
        raise RuntimeError("No polylines were supplied for ordered concatenation.")

    ordered_items: List[Dict[str, Any]] = []
    for item in polylines:
        ordered_items.append(
            {
                "cell_id": int(item["cell_id"]),
                "point_ids": list(item["point_ids"]),
                "points": np.array(item["points"], dtype=float),
            }
        )

    first = ordered_items[0]
    if start_xyz is not None:
        d0 = float(np.linalg.norm(first["points"][0] - start_xyz.reshape(3)))
        d1 = float(np.linalg.norm(first["points"][-1] - start_xyz.reshape(3)))
        if d1 + 1.0e-9 < d0:
            first["points"] = first["points"][::-1].copy()
            first["point_ids"] = list(reversed(first["point_ids"]))

    stitched_points = [np.array(p, dtype=float) for p in first["points"]]
    stitched_ids = list(first["point_ids"])
    stitched_cell_ids = [int(first["cell_id"])]

    for item in ordered_items[1:]:
        tail = stitched_points[-1]
        next_points = np.array(item["points"], dtype=float)
        next_ids = list(item["point_ids"])
        d_start = float(np.linalg.norm(next_points[0] - tail))
        d_end = float(np.linalg.norm(next_points[-1] - tail))
        if d_end + 1.0e-9 < d_start:
            next_points = next_points[::-1].copy()
            next_ids = list(reversed(next_ids))
            d_start = float(np.linalg.norm(next_points[0] - tail))
        if d_start > endpoint_tol_mm:
            warnings.append(f"W_TRUNK_STITCH_GAP: trunk sub-paths were stitched across a {d_start:.6f} mm endpoint gap.")
        start_j = 1 if d_start <= max(endpoint_tol_mm, 1.0e-9) else 0
        for p, pid in zip(next_points[start_j:], next_ids[start_j:]):
            stitched_points.append(np.array(p, dtype=float))
            stitched_ids.append(int(pid))
        stitched_cell_ids.append(int(item["cell_id"]))

    path = build_polyline_path(np.vstack(stitched_points), stitched_ids, stitched_cell_ids)
    if start_xyz is not None and end_xyz is not None:
        d_forward = float(np.linalg.norm(path.points[0] - start_xyz) + np.linalg.norm(path.points[-1] - end_xyz))
        d_reverse = float(np.linalg.norm(path.points[-1] - start_xyz) + np.linalg.norm(path.points[0] - end_xyz))
        if d_reverse + 1.0e-9 < d_forward:
            path = reverse_path(path)
    elif start_xyz is not None:
        if float(np.linalg.norm(path.points[-1] - start_xyz)) < float(np.linalg.norm(path.points[0] - start_xyz)):
            path = reverse_path(path)
    return path


def cluster_polyline_endpoint_nodes(
    polylines: Sequence[Dict[str, Any]],
    endpoint_tol_mm: float = 1.0e-3,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    node_xyz: List[np.ndarray] = []
    endpoint_to_node: Dict[Tuple[int, int], int] = {}

    for item in polylines:
        pts = np.asarray(item["points"], dtype=float)
        for endpoint_index, xyz in ((0, pts[0]), (1, pts[-1])):
            assigned = None
            for node_index, node_point in enumerate(node_xyz):
                if float(np.linalg.norm(xyz - node_point)) <= endpoint_tol_mm:
                    assigned = int(node_index)
                    break
            if assigned is None:
                assigned = len(node_xyz)
                node_xyz.append(np.asarray(xyz, dtype=float).reshape(3))
            endpoint_to_node[(int(item["cell_id"]), int(endpoint_index))] = int(assigned)
    return np.vstack(node_xyz) if node_xyz else np.zeros((0, 3), dtype=float), endpoint_to_node


def choose_primary_trunk_polyline_sequence(
    trunk_polylines: Sequence[Dict[str, Any]],
    inlet_xyz: Optional[np.ndarray],
    bif_xyz: Optional[np.ndarray],
) -> Optional[List[Dict[str, Any]]]:
    if len(trunk_polylines) <= 1 or inlet_xyz is None or bif_xyz is None:
        return list(trunk_polylines) if trunk_polylines else None

    node_xyz, endpoint_to_node = cluster_polyline_endpoint_nodes(trunk_polylines)
    if node_xyz.shape[0] == 0:
        return None

    start_node = int(np.argmin(np.linalg.norm(node_xyz - np.asarray(inlet_xyz, dtype=float).reshape(1, 3), axis=1)))
    end_node = int(np.argmin(np.linalg.norm(node_xyz - np.asarray(bif_xyz, dtype=float).reshape(1, 3), axis=1)))

    edges: List[Dict[str, Any]] = []
    adjacency: Dict[int, List[Tuple[int, int]]] = {}
    for edge_index, item in enumerate(trunk_polylines):
        node_a = endpoint_to_node[(int(item["cell_id"]), 0)]
        node_b = endpoint_to_node[(int(item["cell_id"]), 1)]
        length = float(np.sum(np.linalg.norm(np.diff(np.asarray(item["points"], dtype=float), axis=0), axis=1)))
        edges.append(
            {
                "edge_index": int(edge_index),
                "item": item,
                "node_a": int(node_a),
                "node_b": int(node_b),
                "length": float(length),
            }
        )
        adjacency.setdefault(int(node_a), []).append((int(node_b), int(edge_index)))
        adjacency.setdefault(int(node_b), []).append((int(node_a), int(edge_index)))

    best_path: Optional[Tuple[float, List[int]]] = None

    def dfs(node: int, used_edges: set[int], used_nodes: set[int], path_edges: List[int], total_length: float) -> None:
        nonlocal best_path
        if node == end_node:
            candidate = (float(total_length), list(path_edges))
            if best_path is None or candidate[0] > best_path[0] + 1.0e-9:
                best_path = candidate
            elif best_path is not None and abs(candidate[0] - best_path[0]) <= 1.0e-9 and len(candidate[1]) < len(best_path[1]):
                best_path = candidate
            return

        for neighbor, edge_index in adjacency.get(int(node), []):
            if edge_index in used_edges or neighbor in used_nodes:
                continue
            edge = edges[int(edge_index)]
            used_edges.add(int(edge_index))
            used_nodes.add(int(neighbor))
            path_edges.append(int(edge_index))
            dfs(int(neighbor), used_edges, used_nodes, path_edges, float(total_length + edge["length"]))
            path_edges.pop()
            used_nodes.remove(int(neighbor))
            used_edges.remove(int(edge_index))

    dfs(start_node, set(), {start_node}, [], 0.0)
    if best_path is None:
        return None

    ordered_items: List[Dict[str, Any]] = []
    current_node = int(start_node)
    for edge_index in best_path[1]:
        edge = edges[int(edge_index)]
        node_a = int(edge["node_a"])
        node_b = int(edge["node_b"])
        item = edge["item"]
        oriented = {
            "cell_id": int(item["cell_id"]),
            "point_ids": list(item["point_ids"]),
            "points": np.array(item["points"], dtype=float),
        }
        if node_b == current_node and node_a != current_node:
            oriented["point_ids"] = list(reversed(oriented["point_ids"]))
            oriented["points"] = oriented["points"][::-1].copy()
            current_node = int(node_a)
        else:
            current_node = int(node_b)
        ordered_items.append(oriented)
    return ordered_items


def stitch_polylines(
    polylines: Sequence[Dict[str, Any]],
    start_xyz: Optional[np.ndarray],
    end_xyz: Optional[np.ndarray],
    warnings: List[str],
    endpoint_tol_mm: float = 1.0e-3,
) -> PolylinePath:
    if not polylines:
        raise RuntimeError("No polylines were supplied for stitching.")
    if len(polylines) == 1:
        item = polylines[0]
        path = build_polyline_path(item["points"], item["point_ids"], [int(item["cell_id"])])
        if start_xyz is not None and np.linalg.norm(path.points[-1] - start_xyz.reshape(3)) < np.linalg.norm(path.points[0] - start_xyz.reshape(3)):
            path = reverse_path(path)
        return path

    remaining: List[Dict[str, Any]] = []
    for item in polylines:
        remaining.append(
            {
                "cell_id": int(item["cell_id"]),
                "point_ids": list(item["point_ids"]),
                "points": np.array(item["points"], dtype=float),
            }
        )

    if start_xyz is None:
        start_index = 0
        start_reverse = False
    else:
        best: Optional[Tuple[float, int, bool]] = None
        for i, item in enumerate(remaining):
            p0 = item["points"][0]
            p1 = item["points"][-1]
            d0 = float(np.linalg.norm(p0 - start_xyz))
            d1 = float(np.linalg.norm(p1 - start_xyz))
            candidate = (min(d0, d1), i, d1 < d0)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        start_index = best[1]
        start_reverse = best[2]

    item0 = remaining.pop(start_index)
    if start_reverse:
        item0["points"] = item0["points"][::-1].copy()
        item0["point_ids"] = list(reversed(item0["point_ids"]))

    stitched_points = [np.array(p, dtype=float) for p in item0["points"]]
    stitched_ids = list(item0["point_ids"])
    stitched_cell_ids = [int(item0["cell_id"])]

    while remaining:
        tail = stitched_points[-1]
        best_next: Optional[Tuple[float, int, bool]] = None
        for i, item in enumerate(remaining):
            d_start = float(np.linalg.norm(item["points"][0] - tail))
            d_end = float(np.linalg.norm(item["points"][-1] - tail))
            use_reverse = d_end < d_start
            d_best = min(d_start, d_end)
            candidate = (d_best, i, use_reverse)
            if best_next is None or d_best < best_next[0]:
                best_next = candidate
        assert best_next is not None
        gap_mm, next_index, use_reverse = best_next
        next_item = remaining.pop(next_index)
        next_points = np.array(next_item["points"], dtype=float)
        next_ids = list(next_item["point_ids"])
        if use_reverse:
            next_points = next_points[::-1].copy()
            next_ids = list(reversed(next_ids))
        if gap_mm > endpoint_tol_mm:
            warnings.append(
                f"W_TRUNK_STITCH_GAP: trunk sub-paths were stitched across a {gap_mm:.6f} mm endpoint gap."
            )
        start_j = 1 if float(np.linalg.norm(next_points[0] - tail)) <= max(endpoint_tol_mm, 1.0e-9) else 0
        for p, pid in zip(next_points[start_j:], next_ids[start_j:]):
            stitched_points.append(np.array(p, dtype=float))
            stitched_ids.append(int(pid))
        stitched_cell_ids.append(int(next_item["cell_id"]))

    path = build_polyline_path(np.vstack(stitched_points), stitched_ids, stitched_cell_ids)

    if start_xyz is not None and end_xyz is not None:
        d_forward = float(np.linalg.norm(path.points[0] - start_xyz) + np.linalg.norm(path.points[-1] - end_xyz))
        d_reverse = float(np.linalg.norm(path.points[-1] - start_xyz) + np.linalg.norm(path.points[0] - end_xyz))
        if d_reverse + 1.0e-9 < d_forward:
            path = reverse_path(path)
    elif start_xyz is not None:
        if float(np.linalg.norm(path.points[-1] - start_xyz)) < float(np.linalg.norm(path.points[0] - start_xyz)):
            path = reverse_path(path)
    return path


def resolve_trunk_path(center_pd: vtk.vtkPolyData, meta: Dict[str, Any], warnings: List[str]) -> PolylinePath:
    polylines = extract_centerline_polylines(center_pd, meta)
    poly_by_cell_id = {int(item["cell_id"]): item for item in polylines}
    trunk_polylines = select_polylines_by_label(
        polylines,
        sorted(TRUNK_NAME_ALIASES),
        LABEL_AORTA_TRUNK,
        topology_roles=["trunk_path"],
    )
    if not trunk_polylines:
        raise RuntimeError("Could not isolate abdominal aorta trunk from oriented centerline scaffold.")

    fd = center_pd.GetFieldData()
    inlet_field_pid = get_field_point_id(fd, "Inlet")
    bif_field_pid = get_field_point_id(fd, "Bifurcation")
    inlet_meta_xyz = get_metadata_landmark_xyz(meta, "Inlet")
    bif_meta_xyz = get_metadata_landmark_xyz(meta, "Bifurcation")

    inlet_xyz = None
    bif_xyz = None
    if inlet_field_pid is not None and 0 <= inlet_field_pid < center_pd.GetNumberOfPoints():
        inlet_xyz = np.array(center_pd.GetPoint(inlet_field_pid), dtype=float)
    elif inlet_meta_xyz is not None:
        inlet_xyz = inlet_meta_xyz
    if bif_field_pid is not None and 0 <= bif_field_pid < center_pd.GetNumberOfPoints():
        bif_xyz = np.array(center_pd.GetPoint(bif_field_pid), dtype=float)
    elif bif_meta_xyz is not None:
        bif_xyz = bif_meta_xyz

    trunk_cell_order = get_metadata_branch_cell_order(
        meta,
        sorted(TRUNK_NAME_ALIASES),
        LABEL_AORTA_TRUNK,
        topology_roles=["trunk_path"],
    )
    ordered_trunk_polylines = [poly_by_cell_id[cell_id] for cell_id in trunk_cell_order if cell_id in poly_by_cell_id]
    if trunk_cell_order:
        ordered_cell_ids_present = {int(item["cell_id"]) for item in ordered_trunk_polylines}
        for item in sorted(trunk_polylines, key=lambda row: int(row["cell_id"])):
            if int(item["cell_id"]) not in ordered_cell_ids_present:
                ordered_trunk_polylines.append(item)

    primary_trunk_sequence = choose_primary_trunk_polyline_sequence(
        ordered_trunk_polylines if ordered_trunk_polylines else trunk_polylines,
        inlet_xyz,
        bif_xyz,
    )

    if primary_trunk_sequence:
        if trunk_cell_order and len(primary_trunk_sequence) != len(trunk_cell_order):
            warnings.append("W_TRUNK_OVERLAP_PRUNED: overlapping trunk scaffold intervals were reduced to the inlet-to-bifurcation path for downstream measurement.")
        path = concatenate_ordered_polylines(primary_trunk_sequence, inlet_xyz, bif_xyz, warnings)
    elif trunk_cell_order and ordered_trunk_polylines:
        if len(ordered_trunk_polylines) != len(trunk_cell_order):
            warnings.append("W_TRUNK_METADATA_ORDER_PARTIAL: some metadata-ordered trunk segments were missing from the scaffold; available ordered segments were used.")
        path = concatenate_ordered_polylines(ordered_trunk_polylines, inlet_xyz, bif_xyz, warnings)
    else:
        path = stitch_polylines(trunk_polylines, inlet_xyz, bif_xyz, warnings)

    if polyline_length(path) <= 0.0:
        raise RuntimeError("Resolved trunk polyline has zero length.")
    return path


def resolve_renal_anchor_from_branch(
    side: str,
    branch_items: Sequence[Dict[str, Any]],
    trunk_path: PolylinePath,
) -> Optional[ResolvedAnchor]:
    if not branch_items:
        return None
    best: Optional[ResolvedAnchor] = None
    for item in branch_items:
        pts = np.asarray(item["points"], dtype=float)
        if pts.shape[0] < 2:
            continue
        end0_proj = project_point_to_polyline(trunk_path, pts[0])
        end1_proj = project_point_to_polyline(trunk_path, pts[-1])
        use_index = 0 if float(end0_proj["distance_mm"]) <= float(end1_proj["distance_mm"]) else -1
        prox_point = pts[use_index]
        proj = end0_proj if use_index == 0 else end1_proj
        nearest_idx = int(np.argmin(np.linalg.norm(trunk_path.points - prox_point.reshape(1, 3), axis=1)))
        candidate = ResolvedAnchor(
            side=side,
            source=f"{side}_renal_branch_projection",
            trunk_s_mm=float(proj["s_abs_mm"]),
            point_xyz=np.asarray(prox_point, dtype=float),
            scaffold_point_id=int(item["point_ids"][use_index]),
            nearest_trunk_point_id=int(trunk_path.point_ids[nearest_idx]),
        )
        if best is None or candidate.trunk_s_mm < best.trunk_s_mm:
            best = candidate
    return best


def resolve_renal_anchor_from_metadata_branch(
    side: str,
    meta: Dict[str, Any],
    branch_aliases: Sequence[str],
    trunk_path: PolylinePath,
    warnings: List[str],
    landmark_key: Optional[str] = None,
) -> Optional[ResolvedAnchor]:
    best: Optional[Tuple[Tuple[float, float, float], ResolvedAnchor]] = None
    entries = get_metadata_branch_anchor_entries(meta, branch_aliases)
    if not entries:
        return None

    landmark_xyz = get_metadata_landmark_xyz(meta, landmark_key) if landmark_key else None
    for entry in entries:
        point_candidates: List[Tuple[str, np.ndarray]] = []
        geometric_point = get_xyz_triplet(entry.get("geometric_origin_point"))
        topological_point = get_xyz_triplet(entry.get("topological_start_point"))
        if geometric_point is not None:
            point_candidates.append(("metadata_branch_geometric_origin", geometric_point))
        if topological_point is not None:
            point_candidates.append(("metadata_branch_topological_start", topological_point))
        for source_name, xyz in point_candidates:
            proj = project_point_to_polyline(trunk_path, xyz)
            nearest_idx = int(np.argmin(np.linalg.norm(trunk_path.points - xyz.reshape(1, 3), axis=1)))
            parent_projection_distance = safe_float(entry.get("parent_projection_distance"), float("inf"))
            candidate = ResolvedAnchor(
                side=side,
                source=source_name,
                trunk_s_mm=float(proj["s_abs_mm"]),
                point_xyz=np.asarray(xyz, dtype=float),
                scaffold_point_id=None,
                nearest_trunk_point_id=int(trunk_path.point_ids[nearest_idx]),
            )
            discrepancy = float(np.linalg.norm(xyz - landmark_xyz)) if landmark_xyz is not None else 0.0
            if discrepancy > 1.0 and geometric_point is not None and landmark_key:
                warnings.append(
                    f"W_{landmark_key.upper()}_ANCHOR_MISMATCH: branch proximal anchor differs from metadata landmark by {discrepancy:.3f} mm; branch proximal anchor was preferred."
                )
                landmark_xyz = None
            rank = (
                0.0 if source_name == "metadata_branch_geometric_origin" else 1.0,
                parent_projection_distance if math.isfinite(parent_projection_distance) else 1.0e9,
                discrepancy,
            )
            if best is None or rank < best[0]:
                best = (rank, candidate)
    return None if best is None else best[1]


def resolve_renal_anchors(
    center_pd: vtk.vtkPolyData,
    meta: Dict[str, Any],
    trunk_path: PolylinePath,
    warnings: List[str],
) -> Dict[str, Optional[ResolvedAnchor]]:
    fd = center_pd.GetFieldData()
    polylines = extract_centerline_polylines(center_pd, meta)
    right_branches = select_polylines_by_label(polylines, sorted(RIGHT_RENAL_ALIASES), LABEL_RIGHT_RENAL)
    left_branches = select_polylines_by_label(polylines, sorted(LEFT_RENAL_ALIASES), LABEL_LEFT_RENAL)

    resolved: Dict[str, Optional[ResolvedAnchor]] = {"right": None, "left": None}
    for side, branch_items, branch_aliases in (
        ("right", right_branches, sorted(RIGHT_RENAL_ALIASES)),
        ("left", left_branches, sorted(LEFT_RENAL_ALIASES)),
    ):
        key = "RightRenalOrigin" if side == "right" else "LeftRenalOrigin"
        branch_anchor = resolve_renal_anchor_from_metadata_branch(side, meta, branch_aliases, trunk_path, warnings, landmark_key=key)
        if branch_anchor is not None:
            resolved[side] = branch_anchor
            continue

        field_pid = get_field_point_id(fd, key)
        meta_xyz = get_metadata_landmark_xyz(meta, key)

        landmark_info = point_or_coord_from_landmark(
            center_pd,
            trunk_path,
            field_pid,
            meta_xyz,
            key,
            warnings,
            prefer_metadata=True,
        )
        if landmark_info["source"] != "unresolved":
            resolved[side] = ResolvedAnchor(
                side=side,
                source=str(landmark_info["source"]),
                trunk_s_mm=float(landmark_info["s_abs_mm"]),
                point_xyz=np.asarray(landmark_info["xyz"], dtype=float),
                scaffold_point_id=landmark_info["scaffold_point_id"],
                nearest_trunk_point_id=landmark_info["nearest_trunk_point_id"],
            )
            continue

        field_only_info = point_or_coord_from_landmark(
            center_pd,
            trunk_path,
            field_pid,
            None,
            key,
            warnings,
            prefer_metadata=False,
        )
        if field_only_info["source"] != "unresolved":
            resolved[side] = ResolvedAnchor(
                side=side,
                source=str(field_only_info["source"]),
                trunk_s_mm=float(field_only_info["s_abs_mm"]),
                point_xyz=np.asarray(field_only_info["xyz"], dtype=float),
                scaffold_point_id=field_only_info["scaffold_point_id"],
                nearest_trunk_point_id=field_only_info["nearest_trunk_point_id"],
            )
            continue

        branch_projection_anchor = resolve_renal_anchor_from_branch(side, branch_items, trunk_path)
        if branch_projection_anchor is not None:
            resolved[side] = branch_projection_anchor
            warnings.append(
                f"W_{side.upper()}_RENAL_FROM_BRANCH: explicit renal origin landmark was unavailable; used projected renal branch root."
            )

    return resolved


def build_plane_basis(normal_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = unit(normal_xyz)
    if np.linalg.norm(n) < EPS:
        raise RuntimeError("Plane normal is degenerate.")
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(helper, n))) > 0.90:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = unit(np.cross(n, helper))
    if np.linalg.norm(u) < EPS:
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
        u = unit(np.cross(n, helper))
    v = unit(np.cross(n, u))
    if np.linalg.norm(u) < EPS or np.linalg.norm(v) < EPS:
        raise RuntimeError("Failed to build orthogonal in-plane basis.")
    return u, v


def remove_consecutive_duplicate_points(points: np.ndarray, tol_mm: float = 1.0e-6) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] <= 1:
        return pts.copy()
    kept = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - kept[-1])) > tol_mm:
            kept.append(p)
    out = np.vstack(kept)
    if out.shape[0] > 2 and float(np.linalg.norm(out[0] - out[-1])) <= tol_mm:
        out = out[:-1]
    return out


def polygon_area_2d(points_2d: np.ndarray) -> float:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon_2d(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    poly = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
    inside = False
    n = poly.shape[0]
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        intersects = ((yi > y) != (yj > y)) and (x < ((xj - xi) * (y - yi) / (yj - yi + EPS) + xi))
        if intersects:
            inside = not inside
        j = i
    return inside


def point_to_segment_distance_2d(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    p = np.asarray(point_xy, dtype=float).reshape(2)
    a = np.asarray(a_xy, dtype=float).reshape(2)
    b = np.asarray(b_xy, dtype=float).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < EPS:
        return float(np.linalg.norm(p - a))
    t = clamp(float(np.dot(p - a, ab) / denom), 0.0, 1.0)
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def point_near_polygon_boundary_2d(point_xy: np.ndarray, polygon_xy: np.ndarray, tol_mm: float) -> bool:
    poly = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
    if poly.shape[0] < 2:
        return False
    dmin = float("inf")
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        dmin = min(dmin, point_to_segment_distance_2d(point_xy, a, b))
    return dmin <= tol_mm


def points_in_polygon_2d_batch(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if poly.shape[0] < 3:
        return np.zeros(pts.shape[0], dtype=bool)
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    xi = poly[:, 0].reshape(1, -1)
    yi = poly[:, 1].reshape(1, -1)
    xj = np.roll(poly[:, 0], 1).reshape(1, -1)
    yj = np.roll(poly[:, 1], 1).reshape(1, -1)
    crosses = ((yi > y) != (yj > y)) & (x < ((xj - xi) * (y - yi) / (yj - yi + EPS) + xi))
    return (np.count_nonzero(crosses, axis=1) % 2 == 1)


def points_near_polygon_boundary_batch(points_xy: np.ndarray, polygon_xy: np.ndarray, tol_mm: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    poly = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if poly.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype=bool)
    dmin = np.full(pts.shape[0], float("inf"), dtype=float)
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < EPS:
            d = np.linalg.norm(pts - a.reshape(1, 2), axis=1)
        else:
            t = np.clip(((pts - a.reshape(1, 2)) @ ab) / denom, 0.0, 1.0)
            q = a.reshape(1, 2) + t.reshape(-1, 1) * ab.reshape(1, 2)
            d = np.linalg.norm(pts - q, axis=1)
        dmin = np.minimum(dmin, d)
    return dmin <= float(tol_mm)


def detect_unit_scale_mm_per_unit(
    trunk_length_units: float,
    trial_measurement: Optional[SliceMeasurement],
    warnings: List[str],
) -> float:
    if trial_measurement is None:
        return 1.0
    major_units = float(trial_measurement.major_mm)
    eq_units = float(trial_measurement.eq_mm)
    if 10.0 <= float(trunk_length_units) <= 35.0 and 1.0 <= major_units <= 6.0 and 1.0 <= eq_units <= 6.0:
        warnings.append(
            "W_UNIT_SCALE_CM_ASSUMED: geometry magnitudes strongly suggest centimeter coordinates; measurements and centerline distances were converted to millimeters using a x10 scale factor."
        )
        return 10.0
    return 1.0


def convex_hull_2d(points_2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    if pts.shape[0] <= 1:
        return pts.copy()
    pts_unique = np.unique(np.round(pts, decimals=10), axis=0)
    if pts_unique.shape[0] <= 2:
        return pts_unique
    pts_list = sorted([tuple(p) for p in pts_unique.tolist()])

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts_list:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def major_minor_diameters_from_contour(points_2d: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    hull = convex_hull_2d(pts)
    if hull.shape[0] < 2:
        return float("nan"), float("nan")
    best_d2 = -1.0
    best_vec = np.array([1.0, 0.0], dtype=float)
    for i in range(hull.shape[0]):
        for j in range(i + 1, hull.shape[0]):
            dv = hull[j] - hull[i]
            d2 = float(np.dot(dv, dv))
            if d2 > best_d2:
                best_d2 = d2
                best_vec = dv
    major = math.sqrt(max(best_d2, 0.0))
    axis = unit(best_vec)
    if np.linalg.norm(axis) < EPS:
        return float("nan"), float("nan")
    perp = np.array([-axis[1], axis[0]], dtype=float)
    proj = hull @ perp
    minor = float(np.max(proj) - np.min(proj))
    return float(major), float(minor)


def triangulate_polydata(pd: vtk.vtkPolyData) -> vtk.vtkPolyData:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(tri.GetOutput())
    return out


def extract_surface_only_polydata(combined_pd: vtk.vtkPolyData) -> Tuple[vtk.vtkPolyData, List[int]]:
    n_cells = combined_pd.GetNumberOfCells()
    cd = combined_pd.GetCellData()
    is_surface_name = find_array_name(cd, ["IsSurface"])
    geometry_type_name, geometry_type_id = choose_geometry_type_arrays(combined_pd)

    is_surface_values = get_cell_numeric_values(combined_pd, is_surface_name, default=0.0) if is_surface_name else [0.0] * n_cells
    geom_name_values = get_cell_string_values(combined_pd, geometry_type_name) if geometry_type_name else [""] * n_cells
    geom_id_values = get_cell_numeric_values(combined_pd, geometry_type_id, default=0.0) if geometry_type_id else [0.0] * n_cells

    keep_cell_ids: List[int] = []
    polys = vtk.vtkCellArray()
    strips = vtk.vtkCellArray()
    verts = vtk.vtkCellArray()

    for ci in range(n_cells):
        keep = False
        if ci < len(is_surface_values) and float(is_surface_values[ci]) > 0.5:
            keep = True
        if not keep and ci < len(geom_name_values):
            keep = normalize_name(geom_name_values[ci]) == "surface"
        if not keep and ci < len(geom_id_values):
            keep = int(round(float(geom_id_values[ci]))) == GEOMETRY_TYPE_SURFACE
        if not keep:
            cell = combined_pd.GetCell(ci)
            if cell is not None and cell.GetCellType() in (
                vtk.VTK_TRIANGLE,
                vtk.VTK_QUAD,
                vtk.VTK_POLYGON,
                vtk.VTK_TRIANGLE_STRIP,
            ):
                keep = True
        if not keep:
            continue

        cell = combined_pd.GetCell(ci)
        if cell is None:
            continue
        ids = vtk.vtkIdList()
        for k in range(cell.GetNumberOfPoints()):
            ids.InsertNextId(int(cell.GetPointId(k)))
        cell_type = int(cell.GetCellType())
        if cell_type in (vtk.VTK_TRIANGLE, vtk.VTK_QUAD, vtk.VTK_POLYGON):
            polys.InsertNextCell(ids)
        elif cell_type == vtk.VTK_TRIANGLE_STRIP:
            strips.InsertNextCell(ids)
        elif cell_type == vtk.VTK_VERTEX:
            verts.InsertNextCell(ids)
        else:
            continue
        keep_cell_ids.append(int(ci))

    if not keep_cell_ids:
        raise RuntimeError("Could not extract any surface cells from oriented combined VTP.")

    out = vtk.vtkPolyData()
    out.SetPoints(combined_pd.GetPoints())
    out.SetPolys(polys)
    out.SetStrips(strips)
    out.SetVerts(verts)
    out.BuildCells()
    out.BuildLinks()
    return triangulate_polydata(out), keep_cell_ids


def compute_cell_centers(pd: vtk.vtkPolyData) -> np.ndarray:
    cc = vtk.vtkCellCenters()
    cc.SetInputData(pd)
    cc.VertexCellsOff()
    cc.Update()
    out = cc.GetOutput()
    if out is None or out.GetNumberOfPoints() != pd.GetNumberOfCells():
        raise RuntimeError("Failed to compute cell centers.")
    return np.array([out.GetPoint(i) for i in range(out.GetNumberOfPoints())], dtype=float)


def resolve_trunk_surface_cell_ids(
    combined_pd: vtk.vtkPolyData,
    surface_cell_ids: Sequence[int],
    warnings: Optional[List[str]] = None,
) -> np.ndarray:
    surface_ids_arr = np.asarray(surface_cell_ids, dtype=int)
    if surface_ids_arr.size == 0:
        if warnings is not None:
            warnings.append("W_ZONE_SURFACE_EMPTY: no surface cells were available for neck-zone labeling.")
        return np.zeros((0,), dtype=int)

    label_string_name, label_numeric_name = choose_cell_label_array(combined_pd)
    string_values = get_cell_string_values(combined_pd, label_string_name) if label_string_name else []
    numeric_values = get_cell_numeric_values(combined_pd, label_numeric_name, default=float("nan")) if label_numeric_name else []

    if string_values:
        trunk_mask = np.array(
            [
                normalize_name(string_values[int(cell_id)]) in TRUNK_NAME_ALIASES
                if 0 <= int(cell_id) < len(string_values)
                else False
                for cell_id in surface_ids_arr
            ],
            dtype=bool,
        )
        if np.any(trunk_mask):
            return surface_ids_arr[trunk_mask]

    if numeric_values:
        trunk_mask = np.array(
            [
                int(round(float(numeric_values[int(cell_id)]))) == LABEL_AORTA_TRUNK
                if 0 <= int(cell_id) < len(numeric_values) and math.isfinite(float(numeric_values[int(cell_id)]))
                else False
                for cell_id in surface_ids_arr
            ],
            dtype=bool,
        )
        if np.any(trunk_mask):
            if warnings is not None:
                warnings.append(
                    f"W_ZONE_TRUNK_NUMERIC_FALLBACK: no trunk surface cells matched the preferred branch-name labels; used {label_numeric_name or 'numeric label'} == {LABEL_AORTA_TRUNK}."
                )
            return surface_ids_arr[trunk_mask]

    if warnings is not None:
        warnings.append(
            "W_ZONE_TRUNK_SURFACE_EMPTY: no trunk-labeled surface cells were found in the combined VTP; NeckZoneLabel will remain empty."
        )
    return np.zeros((0,), dtype=int)


def project_points_to_polyline_batch(
    path: PolylinePath,
    points_xyz: np.ndarray,
    chunk_size: int = 2048,
) -> Dict[str, np.ndarray]:
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    n_points = int(pts.shape[0])
    if n_points == 0:
        return {
            "s_abs_mm": np.zeros((0,), dtype=float),
            "distance_mm": np.zeros((0,), dtype=float),
            "segment_index": np.zeros((0,), dtype=int),
            "projected_xyz": np.zeros((0, 3), dtype=float),
        }

    seg_a = np.asarray(path.points[:-1], dtype=float)
    seg_b = np.asarray(path.points[1:], dtype=float)
    seg_vec = seg_b - seg_a
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len2 = np.maximum(seg_len * seg_len, EPS)

    s_abs_mm = np.zeros((n_points,), dtype=float)
    distance_mm = np.zeros((n_points,), dtype=float)
    segment_index = np.zeros((n_points,), dtype=int)
    projected_xyz = np.zeros((n_points, 3), dtype=float)

    for start in range(0, n_points, chunk_size):
        stop = min(n_points, start + chunk_size)
        chunk = pts[start:stop]
        rel = chunk[:, None, :] - seg_a[None, :, :]
        t = np.einsum("ijk,jk->ij", rel, seg_vec) / seg_len2.reshape(1, -1)
        t = np.clip(t, 0.0, 1.0)
        proj = seg_a[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
        diff = chunk[:, None, :] - proj
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        best_idx = np.argmin(d2, axis=1)
        row_idx = np.arange(best_idx.shape[0])
        best_t = t[row_idx, best_idx]
        best_proj = proj[row_idx, best_idx]
        best_d2 = d2[row_idx, best_idx]

        s_abs_mm[start:stop] = path.cumulative_s[best_idx] + best_t * seg_len[best_idx]
        distance_mm[start:stop] = np.sqrt(np.maximum(best_d2, 0.0))
        segment_index[start:stop] = best_idx.astype(int)
        projected_xyz[start:stop] = best_proj.astype(float)

    return {
        "s_abs_mm": s_abs_mm,
        "distance_mm": distance_mm,
        "segment_index": segment_index,
        "projected_xyz": projected_xyz,
    }


def nearest_sample_indices(sample_s_abs_mm: np.ndarray, query_s_abs_mm: np.ndarray) -> np.ndarray:
    sample_s = np.asarray(sample_s_abs_mm, dtype=float).reshape(-1)
    query_s = np.asarray(query_s_abs_mm, dtype=float).reshape(-1)
    if sample_s.size == 0 or query_s.size == 0:
        return np.zeros((query_s.size,), dtype=int)

    hi = np.searchsorted(sample_s, query_s, side="left")
    hi = np.clip(hi, 0, sample_s.size - 1)
    lo = np.clip(hi - 1, 0, sample_s.size - 1)
    choose_hi = np.abs(sample_s[hi] - query_s) < np.abs(sample_s[lo] - query_s)
    return np.where(choose_hi, hi, lo).astype(int)


def cut_surface_with_plane(surface_pd: vtk.vtkPolyData, origin_xyz: np.ndarray, normal_xyz: np.ndarray) -> vtk.vtkPolyData:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2]))
    plane.SetNormal(float(normal_xyz[0]), float(normal_xyz[1]), float(normal_xyz[2]))

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface_pd)
    cutter.SetCutFunction(plane)
    cutter.GenerateTrianglesOff()
    cutter.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(cutter.GetOutput())
    clean.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputData(clean.GetOutput())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(stripper.GetOutput())
    return out


def extract_contour_candidates(
    cut_pd: vtk.vtkPolyData,
    origin_xyz: np.ndarray,
    normal_xyz: np.ndarray,
) -> List[Dict[str, Any]]:
    if cut_pd is None or cut_pd.GetNumberOfCells() < 1 or cut_pd.GetNumberOfPoints() < 3:
        return []
    u, v = build_plane_basis(normal_xyz)
    origin = np.asarray(origin_xyz, dtype=float).reshape(3)
    candidates: List[Dict[str, Any]] = []

    for ci in range(cut_pd.GetNumberOfCells()):
        cell = cut_pd.GetCell(ci)
        if cell is None:
            continue
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
            continue
        ids = [int(cell.GetPointId(k)) for k in range(cell.GetNumberOfPoints())]
        if len(ids) < 3:
            continue
        pts3d = np.array([cut_pd.GetPoint(pid) for pid in ids], dtype=float)
        pts2d = np.column_stack(
            [
                np.dot(pts3d - origin.reshape(1, 3), u.reshape(3, 1)).reshape(-1),
                np.dot(pts3d - origin.reshape(1, 3), v.reshape(3, 1)).reshape(-1),
            ]
        )
        pts2d = remove_consecutive_duplicate_points(pts2d)
        if pts2d.shape[0] < 3:
            continue
        close_gap = float(np.linalg.norm(pts2d[0] - pts2d[-1]))
        char_len = float(np.max(np.linalg.norm(pts2d - np.mean(pts2d, axis=0).reshape(1, 2), axis=1))) if pts2d.shape[0] else 0.0
        closed = close_gap <= max(0.75, 0.02 * max(char_len, 1.0))
        if closed and close_gap > 1.0e-6:
            pts2d = np.vstack([pts2d, pts2d[0]])
        pts2d_for_area = pts2d[:-1] if closed and np.linalg.norm(pts2d[0] - pts2d[-1]) < 1.0e-9 else pts2d
        if pts2d_for_area.shape[0] < 3:
            continue
        area = abs(polygon_area_2d(pts2d_for_area))
        if area <= 1.0e-6:
            continue
        contains_origin = point_in_polygon_2d(np.array([0.0, 0.0], dtype=float), pts2d_for_area)
        centroid_2d = np.mean(pts2d_for_area, axis=0)
        centroid_dist = float(np.linalg.norm(centroid_2d))
        major, minor = major_minor_diameters_from_contour(pts2d_for_area)
        candidates.append(
            {
                "cell_id": int(ci),
                "points_3d": pts3d,
                "points_2d": pts2d_for_area,
                "area_mm2": float(area),
                "contains_origin": bool(contains_origin),
                "centroid_dist_mm": float(centroid_dist),
                "major_mm": float(major),
                "minor_mm": float(minor),
                "basis_u_xyz": u.copy(),
                "basis_v_xyz": v.copy(),
                "closed": bool(closed),
            }
        )
    return candidates


def choose_best_contour(candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    containing = [c for c in candidates if c["contains_origin"]]
    if containing:
        containing.sort(key=lambda c: (0 if c["closed"] else 1, -float(c["area_mm2"]), float(c["centroid_dist_mm"])))
        return containing[0]
    ordered = sorted(candidates, key=lambda c: (0 if c["closed"] else 1, float(c["centroid_dist_mm"]), -float(c["area_mm2"])))
    return ordered[0]


def measure_orthogonal_slice(surface_pd: vtk.vtkPolyData, trunk_path: PolylinePath, s_abs_mm: float) -> Optional[SliceMeasurement]:
    origin_xyz, tangent_xyz = evaluate_path_point_and_tangent(trunk_path, s_abs_mm, ds_mm=0.5)
    cut_pd = cut_surface_with_plane(surface_pd, origin_xyz, tangent_xyz)
    candidates = extract_contour_candidates(cut_pd, origin_xyz, tangent_xyz)
    contour = choose_best_contour(candidates)
    if contour is None:
        return None
    area = float(contour["area_mm2"])
    major = float(contour["major_mm"])
    minor = float(contour["minor_mm"])
    eq = math.sqrt(max(0.0, 4.0 * area / math.pi))
    return SliceMeasurement(
        sample_s_abs_mm=float(s_abs_mm),
        origin_xyz=np.asarray(origin_xyz, dtype=float),
        tangent_xyz=np.asarray(tangent_xyz, dtype=float),
        basis_u_xyz=np.asarray(contour["basis_u_xyz"], dtype=float),
        basis_v_xyz=np.asarray(contour["basis_v_xyz"], dtype=float),
        contour_2d=np.asarray(contour["points_2d"], dtype=float),
        contour_3d=np.asarray(contour["points_3d"], dtype=float),
        area_mm2=area,
        major_mm=major,
        minor_mm=minor,
        eq_mm=float(eq),
        contains_origin=bool(contour["contains_origin"]),
    )


def find_valid_slice_near_target(
    surface_pd: vtk.vtkPolyData,
    trunk_path: PolylinePath,
    target_s_abs_mm: float,
    min_s_abs_mm: float,
    max_s_abs_mm: float,
    offsets_mm: Sequence[float],
) -> Optional[SliceMeasurement]:
    tested: set[float] = set()
    for offset in offsets_mm:
        s = float(clamp(target_s_abs_mm + float(offset), min_s_abs_mm, max_s_abs_mm))
        s_key = round(s, 6)
        if s_key in tested:
            continue
        tested.add(s_key)
        result = measure_orthogonal_slice(surface_pd, trunk_path, s)
        if result is not None:
            return result
    return None


def moving_median(values: np.ndarray, window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or window_size <= 1:
        return arr.copy()
    radius = window_size // 2
    out = np.zeros_like(arr, dtype=float)
    for i in range(arr.size):
        lo = max(0, i - radius)
        hi = min(arr.size, i + radius + 1)
        segment = arr[lo:hi]
        segment = segment[np.isfinite(segment)]
        out[i] = float(np.median(segment)) if segment.size else float("nan")
    return out


def moving_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or window_size <= 1:
        return arr.copy()
    radius = window_size // 2
    out = np.zeros_like(arr, dtype=float)
    for i in range(arr.size):
        lo = max(0, i - radius)
        hi = min(arr.size, i + radius + 1)
        segment = arr[lo:hi]
        segment = segment[np.isfinite(segment)]
        out[i] = float(np.mean(segment)) if segment.size else float("nan")
    return out


def sample_measurements_along_path(
    surface_pd: vtk.vtkPolyData,
    trunk_path: PolylinePath,
    s_start_abs_mm: float,
    s_end_abs_mm: float,
    step_mm: float,
) -> List[SliceMeasurement]:
    if s_end_abs_mm < s_start_abs_mm:
        return []
    positions: List[float] = []
    s = float(s_start_abs_mm)
    while s <= s_end_abs_mm + 1.0e-6:
        positions.append(round(float(s), 6))
        s += float(step_mm)
    out: List[SliceMeasurement] = []
    for s_abs in positions:
        m = measure_orthogonal_slice(surface_pd, trunk_path, float(s_abs))
        if m is not None:
            out.append(m)
    return out


def detect_neck_end(
    coarse_measurements: Sequence[SliceMeasurement],
    d0_measurement: SliceMeasurement,
    d0_abs_mm: float,
    unit_scale_mm: float,
    warnings: List[str],
) -> float:
    if len(coarse_measurements) < 3:
        warnings.append("W_NECK_END_FEW_SAMPLES: too few valid trunk samples for neck-end detection.")
        return float("nan")

    positions = np.array([(m.sample_s_abs_mm - d0_abs_mm) * unit_scale_mm for m in coarse_measurements], dtype=float)
    major = np.array([m.major_mm * unit_scale_mm for m in coarse_measurements], dtype=float)
    smooth = moving_mean(moving_median(major, 3), 3)
    baseline = float(d0_measurement.major_mm * unit_scale_mm)

    flags = np.zeros(len(coarse_measurements), dtype=bool)
    for i in range(len(coarse_measurements)):
        if not math.isfinite(float(smooth[i])):
            continue
        pos = float(positions[i])
        hist10 = smooth[(positions >= max(0.0, pos - 10.0)) & (positions <= pos + 1.0e-9)]
        hist15 = smooth[(positions >= max(0.0, pos - 15.0)) & (positions <= pos + 1.0e-9)]
        hist10 = hist10[np.isfinite(hist10)]
        hist15 = hist15[np.isfinite(hist15)]
        delta10 = float(smooth[i] - np.min(hist10)) if hist10.size else 0.0
        delta15 = float(smooth[i] - np.min(hist15)) if hist15.size else 0.0
        cond1 = bool(smooth[i] > 1.10 * baseline)
        cond2 = bool(delta10 >= 2.0 and pos >= 2.0)
        cond3 = bool(delta15 > 3.0 and pos >= 3.0)
        flags[i] = (int(cond1) + int(cond2) + int(cond3)) >= 2

    for i in range(len(flags) - 1):
        if not flags[i]:
            continue
        if flags[i + 1] and abs(float(positions[i + 1] - positions[i])) <= 1.10:
            return float(positions[i])

    warnings.append("W_NECK_END_WEAK: sustained expansion rule did not trigger on coarse 1.0 mm samples.")
    return float("nan")


def refine_neck_end(
    surface_pd: vtk.vtkPolyData,
    trunk_path: PolylinePath,
    candidate_s_from_d0_mm: float,
    d0_measurement: SliceMeasurement,
    d0_abs_mm: float,
    scan_end_abs_mm: float,
    unit_scale_mm: float,
    warnings: List[str],
) -> float:
    if not math.isfinite(candidate_s_from_d0_mm):
        return float("nan")
    refine_start_abs = max(d0_abs_mm, d0_abs_mm + (candidate_s_from_d0_mm - 2.0) / unit_scale_mm)
    refine_end_abs = min(scan_end_abs_mm, d0_abs_mm + (candidate_s_from_d0_mm + 2.0) / unit_scale_mm)
    refined = sample_measurements_along_path(surface_pd, trunk_path, refine_start_abs, refine_end_abs, 0.5 / unit_scale_mm)
    if len(refined) < 4:
        return float(candidate_s_from_d0_mm)

    positions = np.array([(m.sample_s_abs_mm - d0_abs_mm) * unit_scale_mm for m in refined], dtype=float)
    major = np.array([m.major_mm * unit_scale_mm for m in refined], dtype=float)
    smooth = moving_mean(moving_median(major, 5), 3)
    baseline = float(d0_measurement.major_mm * unit_scale_mm)

    flags = np.zeros(len(refined), dtype=bool)
    for i in range(len(refined)):
        if not math.isfinite(float(smooth[i])):
            continue
        pos = float(positions[i])
        hist10 = smooth[(positions >= max(0.0, pos - 10.0)) & (positions <= pos + 1.0e-9)]
        hist15 = smooth[(positions >= max(0.0, pos - 15.0)) & (positions <= pos + 1.0e-9)]
        hist10 = hist10[np.isfinite(hist10)]
        hist15 = hist15[np.isfinite(hist15)]
        delta10 = float(smooth[i] - np.min(hist10)) if hist10.size else 0.0
        delta15 = float(smooth[i] - np.min(hist15)) if hist15.size else 0.0
        cond1 = bool(smooth[i] > 1.10 * baseline)
        cond2 = bool(delta10 >= 2.0 and pos >= 2.0)
        cond3 = bool(delta15 > 3.0 and pos >= 3.0)
        flags[i] = (int(cond1) + int(cond2) + int(cond3)) >= 2

    for i in range(len(flags) - 2):
        if flags[i] and flags[i + 1] and flags[i + 2]:
            if abs(float(positions[i + 1] - positions[i])) <= 0.60 and abs(float(positions[i + 2] - positions[i + 1])) <= 0.60:
                return float(positions[i])

    warnings.append("W_NECK_END_REFINE_UNCHANGED: refined 0.5 mm sampling did not strengthen neck-end confidence.")
    return float(candidate_s_from_d0_mm)


def choose_lowest_renal_anchor(
    resolved: Dict[str, Optional[ResolvedAnchor]],
    warnings: List[str],
) -> Optional[ResolvedAnchor]:
    valid = [anchor for anchor in resolved.values() if anchor is not None and math.isfinite(anchor.trunk_s_mm)]
    if not valid:
        return None
    valid.sort(key=lambda a: float(a.trunk_s_mm), reverse=True)
    if len(valid) == 1:
        warnings.append("W_SINGLE_RENAL_ANCHOR: only one renal origin anchor was resolved; D0 is based on a unilateral renal reference.")
    return valid[0]


def find_stable_segment_surrogate(
    surface_pd: vtk.vtkPolyData,
    trunk_path: PolylinePath,
    unit_scale_mm: float,
    warnings: List[str],
) -> Optional[float]:
    trunk_len = polyline_length(trunk_path)
    if trunk_len * unit_scale_mm < 8.0:
        return None
    s_start = max(2.0 / unit_scale_mm, 0.15 * trunk_len)
    s_end = max(s_start + 2.0 / unit_scale_mm, min(0.55 * trunk_len, trunk_len - 4.0 / unit_scale_mm))
    candidates = sample_measurements_along_path(surface_pd, trunk_path, s_start, s_end, 1.0 / unit_scale_mm)
    if len(candidates) < 6:
        return None
    eq = np.array([m.eq_mm * unit_scale_mm for m in candidates], dtype=float)
    pos = np.array([m.sample_s_abs_mm * unit_scale_mm for m in candidates], dtype=float)
    best_idx = None
    best_score = float("inf")
    for i in range(0, len(candidates) - 4):
        window = eq[i : i + 5]
        if not np.all(np.isfinite(window)):
            continue
        cv = float(np.std(window) / (np.mean(window) + EPS))
        slope = float(abs(window[-1] - window[0]) / max(pos[i + 4] - pos[i], 1.0))
        score = cv + 0.35 * slope
        if score < best_score:
            best_score = score
            best_idx = i
    if best_idx is None:
        return None
    warnings.append("W_RENAL_SURROGATE: explicit renal anchors were unavailable; used a proximal stable-segment surrogate for D0.")
    return float(pos[best_idx])


def build_zone_label_arrays(
    combined_pd: vtk.vtkPolyData,
    trunk_surface_cell_ids: Sequence[int],
    cell_centers_xyz: np.ndarray,
    zone_measurements: Sequence[SliceMeasurement],
    trunk_path: PolylinePath,
    d0_abs_mm: float,
    unit_scale_mm: float,
    warnings: Optional[List[str]] = None,
) -> Tuple[List[int], List[str], List[float]]:
    n_cells = combined_pd.GetNumberOfCells()
    label = [0] * n_cells
    name = ["outside"] * n_cells
    s_from_d0 = [float("nan")] * n_cells

    if not zone_measurements:
        return label, name, s_from_d0

    trunk_surface_ids_arr = np.asarray(trunk_surface_cell_ids, dtype=int)
    if trunk_surface_ids_arr.size == 0:
        return label, name, s_from_d0

    sample_origins = np.array([m.origin_xyz for m in zone_measurements], dtype=float)
    sample_s_abs_mm = np.array([m.sample_s_abs_mm for m in zone_measurements], dtype=float)
    sample_tangents = np.array([m.tangent_xyz for m in zone_measurements], dtype=float)
    sample_basis_u = np.array([m.basis_u_xyz for m in zone_measurements], dtype=float)
    sample_basis_v = np.array([m.basis_v_xyz for m in zone_measurements], dtype=float)
    axial_tol_units = 0.80 / unit_scale_mm
    boundary_tol_units = 1.00 / unit_scale_mm
    plane_tol_units = max(axial_tol_units, 0.75 / unit_scale_mm)
    sample_radius = np.array([float(np.max(np.linalg.norm(m.contour_2d, axis=1))) if m.contour_2d.size else 0.0 for m in zone_measurements], dtype=float)
    sample_bbox = [
        (
            float(np.min(m.contour_2d[:, 0])) - boundary_tol_units,
            float(np.max(m.contour_2d[:, 0])) + boundary_tol_units,
            float(np.min(m.contour_2d[:, 1])) - boundary_tol_units,
            float(np.max(m.contour_2d[:, 1])) + boundary_tol_units,
        )
        for m in zone_measurements
    ]

    trunk_centers = np.asarray(cell_centers_xyz[trunk_surface_ids_arr], dtype=float)
    projections = project_points_to_polyline_batch(trunk_path, trunk_centers)
    projected_s_abs_mm = projections["s_abs_mm"]
    zone_end_rel_mm = max(0.0, float((sample_s_abs_mm[-1] - d0_abs_mm) * unit_scale_mm))

    # Gate the label to the trunk centerline interval and to the two end-cap planes so
    # suprarenal and distal spillover cannot occur even if nearby surface cells fall
    # inside a local slice contour.
    s_interval_mask = (projected_s_abs_mm >= sample_s_abs_mm[0] - 1.0e-6) & (projected_s_abs_mm <= sample_s_abs_mm[-1] + 1.0e-6)
    proximal_plane = np.einsum("ij,j->i", trunk_centers - sample_origins[0].reshape(1, 3), sample_tangents[0])
    distal_plane = np.einsum("ij,j->i", trunk_centers - sample_origins[-1].reshape(1, 3), sample_tangents[-1])
    plane_mask = (proximal_plane >= -plane_tol_units) & (distal_plane <= plane_tol_units)
    interval_mask = s_interval_mask & plane_mask
    if not np.any(interval_mask):
        if warnings is not None:
            warnings.append(
                "W_ZONE_TRUNK_INTERVAL_EMPTY: no trunk-owned surface cells survived the D0-to-D15 interval gating; NeckZoneLabel will remain empty."
            )
        return label, name, s_from_d0

    gated_cell_ids = trunk_surface_ids_arr[interval_mask]
    gated_centers = trunk_centers[interval_mask]
    gated_s_abs_mm = projected_s_abs_mm[interval_mask]
    best_idx = nearest_sample_indices(sample_s_abs_mm, gated_s_abs_mm)
    vec = gated_centers - sample_origins[best_idx]
    axial = np.abs(np.einsum("ij,ij->i", vec, sample_tangents[best_idx]))
    approx_r = sample_radius[best_idx] + boundary_tol_units + axial_tol_units
    radial_d2 = np.einsum("ij,ij->i", vec, vec) - axial * axial
    radial_d2 = np.maximum(radial_d2, 0.0)
    candidate_mask = (axial <= axial_tol_units) & (radial_d2 <= approx_r * approx_r)
    if not np.any(candidate_mask):
        if warnings is not None:
            warnings.append(
                "W_ZONE_TRUNK_CANDIDATES_EMPTY: trunk interval gating left no candidates close enough to the D0-to-D15 slice envelopes; NeckZoneLabel will remain empty."
            )
        return label, name, s_from_d0

    candidate_cell_ids = gated_cell_ids[candidate_mask]
    candidate_centers = gated_centers[candidate_mask]
    candidate_s_abs_mm = gated_s_abs_mm[candidate_mask]
    candidate_idx = best_idx[candidate_mask]
    candidate_vec = vec[candidate_mask]

    labeled_any = False
    for sample_idx in np.unique(candidate_idx):
        local_rows = np.where(candidate_idx == sample_idx)[0]
        local_vec = candidate_vec[local_rows]
        p2 = np.column_stack(
            [
                np.einsum("ij,j->i", local_vec, sample_basis_u[int(sample_idx)]),
                np.einsum("ij,j->i", local_vec, sample_basis_v[int(sample_idx)]),
            ]
        )
        xmin, xmax, ymin, ymax = sample_bbox[int(sample_idx)]
        bbox_mask = (p2[:, 0] >= xmin) & (p2[:, 0] <= xmax) & (p2[:, 1] >= ymin) & (p2[:, 1] <= ymax)
        if not np.any(bbox_mask):
            continue
        p2_bbox = p2[bbox_mask]
        local_rows_bbox = local_rows[bbox_mask]
        contour = zone_measurements[int(sample_idx)].contour_2d
        inside = points_in_polygon_2d_batch(p2_bbox, contour)
        final_mask = inside.copy()
        if np.any(~inside):
            near = points_near_polygon_boundary_batch(p2_bbox[~inside], contour, boundary_tol_units)
            final_mask[~inside] = near
        if not np.any(final_mask):
            continue
        labeled_any = True
        final_rows = local_rows_bbox[final_mask]
        final_cell_ids = candidate_cell_ids[final_rows]
        final_s_rel_mm = np.clip((candidate_s_abs_mm[final_rows] - d0_abs_mm) * unit_scale_mm, 0.0, zone_end_rel_mm)
        for cell_id, s_rel in zip(final_cell_ids.tolist(), final_s_rel_mm.tolist()):
            label[int(cell_id)] = 1
            name[int(cell_id)] = "D0_to_D15_neck_zone"
            s_from_d0[int(cell_id)] = float(s_rel)

    if not labeled_any and warnings is not None:
        warnings.append(
            "W_ZONE_TRUNK_CONTOUR_EMPTY: trunk-only candidates survived interval gating, but none passed the local D0-to-D15 contour inclusion test."
        )
    return label, name, s_from_d0


def add_int_cell_array(cd: vtk.vtkCellData, name: str, values: Sequence[int]) -> None:
    if cd.HasArray(name):
        cd.RemoveArray(name)
    arr = vtk.vtkIntArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(len(values))
    for i, v in enumerate(values):
        arr.SetTuple1(i, int(v))
    cd.AddArray(arr)


def add_double_cell_array(cd: vtk.vtkCellData, name: str, values: Sequence[float]) -> None:
    if cd.HasArray(name):
        cd.RemoveArray(name)
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(len(values))
    for i, v in enumerate(values):
        arr.SetTuple1(i, float(v))
    cd.AddArray(arr)


def add_string_cell_array(cd: vtk.vtkCellData, name: str, values: Sequence[str]) -> None:
    if cd.HasArray(name):
        cd.RemoveArray(name)
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    arr.SetNumberOfValues(len(values))
    for i, v in enumerate(values):
        arr.SetValue(i, str(v))
    cd.AddArray(arr)


def make_report_lines(
    measurement_by_name: Dict[str, Optional[SliceMeasurement]],
    d0_abs_mm: float,
    unit_scale_mm: float,
    lowest_anchor: Optional[ResolvedAnchor],
    renal_reference_mode: str,
    neck_end_s_from_d0_mm: float,
    max_aneurysm_measurement: Optional[SliceMeasurement],
    meta: Dict[str, Any],
    warnings: Sequence[str],
    trunk_direction_reversed: bool,
) -> List[str]:
    lines: List[str] = []
    frame_source = get_metadata_horizontal_frame_source(meta)
    frame_confidence = get_metadata_horizontal_frame_confidence(meta)
    ap_orientation_certain = get_metadata_ap_orientation_certain(meta)
    for key in ("D0", "D5", "D10", "D15"):
        m = measurement_by_name.get(key)
        if m is None:
            lines.append(f"{key}_main_mm=nan")
            lines.append(f"{key}_minor_mm=nan")
            lines.append(f"{key}_eq_mm=nan")
            lines.append(f"{key}_sample_s_mm=nan")
            lines.append(f"{key}_sample_s_on_trunk_mm=nan")
        else:
            lines.append(f"{key}_main_mm={format_mm(m.major_mm * unit_scale_mm)}")
            lines.append(f"{key}_minor_mm={format_mm(m.minor_mm * unit_scale_mm)}")
            lines.append(f"{key}_eq_mm={format_mm(m.eq_mm * unit_scale_mm)}")
            lines.append(f"{key}_sample_s_mm={format_mm((m.sample_s_abs_mm - d0_abs_mm) * unit_scale_mm)}")
            lines.append(f"{key}_sample_s_on_trunk_mm={format_mm(m.sample_s_abs_mm * unit_scale_mm)}")

    lines.append(f"lowest_renal_source={(lowest_anchor.source if lowest_anchor is not None else 'unresolved')}")
    lines.append(
        f"lowest_renal_point_id={(str(lowest_anchor.scaffold_point_id) if lowest_anchor is not None and lowest_anchor.scaffold_point_id is not None else 'nan')}"
    )
    lines.append(
        f"lowest_renal_s_on_trunk_mm={(format_mm(lowest_anchor.trunk_s_mm * unit_scale_mm) if lowest_anchor is not None else 'nan')}"
    )
    lines.append(f"neck_end_s_from_D0_mm={format_mm(neck_end_s_from_d0_mm)}")

    if max_aneurysm_measurement is None:
        lines.append("max_aneurysm_diameter_main_mm=nan")
        lines.append("max_aneurysm_diameter_minor_mm=nan")
        lines.append("max_aneurysm_diameter_eq_mm=nan")
        lines.append("max_aneurysm_location_s_from_D0_mm=nan")
        lines.append("max_aneurysm_location_s_on_trunk_mm=nan")
    else:
        lines.append(f"max_aneurysm_diameter_main_mm={format_mm(max_aneurysm_measurement.major_mm * unit_scale_mm)}")
        lines.append(f"max_aneurysm_diameter_minor_mm={format_mm(max_aneurysm_measurement.minor_mm * unit_scale_mm)}")
        lines.append(f"max_aneurysm_diameter_eq_mm={format_mm(max_aneurysm_measurement.eq_mm * unit_scale_mm)}")
        lines.append(f"max_aneurysm_location_s_from_D0_mm={format_mm((max_aneurysm_measurement.sample_s_abs_mm - d0_abs_mm) * unit_scale_mm)}")
        lines.append(f"max_aneurysm_location_s_on_trunk_mm={format_mm(max_aneurysm_measurement.sample_s_abs_mm * unit_scale_mm)}")

    lines.append(f"renal_reference_mode={renal_reference_mode}")
    lines.append(f"trunk_direction_reversed_for_infrarenal_analysis={str(bool(trunk_direction_reversed)).lower()}")
    lines.append(f"metadata_horizontal_frame_source={frame_source}")
    lines.append(f"metadata_horizontal_frame_confidence={format_mm(frame_confidence)}")
    lines.append(f"metadata_ap_orientation_certain={str(bool(ap_orientation_certain)).lower()}")
    lines.append(f"warnings={join_warnings(warnings)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure infrarenal neck diameters and label the D0-to-D15 neck zone from previously oriented vascular outputs."
    )
    parser.add_argument("--input_centerlines", type=str, default=INPUT_CENTERLINES_VTP_PATH)
    parser.add_argument("--input_surface_with_centerlines", type=str, default=INPUT_SURFACE_WITH_CENTERLINES_VTP_PATH)
    parser.add_argument("--input_metadata", type=str, default=INPUT_METADATA_JSON_PATH)
    parser.add_argument("--output_colored", type=str, default=OUTPUT_COLORED_VTP_PATH)
    parser.add_argument("--output_report", type=str, default=OUTPUT_REPORT_TXT_PATH)
    args = parser.parse_args()

    warnings: List[str] = []

    try:
        meta = load_json(args.input_metadata)
        if str(meta.get("status", "")).lower() == "failed":
            raise RuntimeError("Upstream orientation metadata indicates failure; downstream sizing cannot proceed.")

        center_pd = load_vtp(args.input_centerlines)
        combined_pd = load_vtp(args.input_surface_with_centerlines)
        surface_pd, surface_cell_ids = extract_surface_only_polydata(combined_pd)

        trunk_path = resolve_trunk_path(center_pd, meta, warnings)
        trunk_path = normalize_trunk_direction_for_infrarenal_analysis(center_pd, meta, trunk_path, warnings)
        trunk_len_mm = polyline_length(trunk_path)
        if trunk_len_mm < 5.0:
            raise RuntimeError(f"Resolved trunk length is implausibly short ({trunk_len_mm:.3f} mm).")

        inlet_landmark = resolve_named_landmark_on_trunk(center_pd, meta, trunk_path, "Inlet", warnings)
        bif_landmark = resolve_named_landmark_on_trunk(center_pd, meta, trunk_path, "Bifurcation", warnings)
        renal_anchors = resolve_renal_anchors(center_pd, meta, trunk_path, warnings)
        landmark_order = validate_trunk_landmark_order(trunk_path, inlet_landmark, bif_landmark, renal_anchors, warnings)
        if not landmark_order["valid"]:
            failure_msg = " ".join(landmark_order["failures"]) if landmark_order["failures"] else "unknown landmark-order inconsistency."
            raise RuntimeError(f"Normalized trunk landmark order is inconsistent for infrarenal analysis. {failure_msg}")

        bif_s_abs_mm = float(landmark_order["s_bif"])
        lowest_anchor = choose_lowest_renal_anchor(renal_anchors, warnings)
        renal_reference_mode = "explicit_renal_origin"

        if lowest_anchor is None:
            surrogate_s = find_stable_segment_surrogate(surface_pd, trunk_path, 1.0, warnings)
            if surrogate_s is None:
                raise RuntimeError("Failed to resolve renal origins and failed to find a stable-segment surrogate for D0.")
            lowest_anchor = ResolvedAnchor(
                side="surrogate",
                source="stable_segment_surrogate",
                trunk_s_mm=float(surrogate_s),
                point_xyz=interpolate_point_on_path(trunk_path, surrogate_s),
                scaffold_point_id=None,
                nearest_trunk_point_id=None,
            )
            renal_reference_mode = "morphology_surrogate"
        elif normalize_name(lowest_anchor.source).startswith("metadata_branch"):
            renal_reference_mode = "metadata_branch_origin"
        elif normalize_name(lowest_anchor.source).startswith("metadata"):
            renal_reference_mode = "metadata_landmark"
        elif "field" in normalize_name(lowest_anchor.source):
            renal_reference_mode = "centerline_field_landmark"
        elif "branch" in normalize_name(lowest_anchor.source):
            renal_reference_mode = "renal_branch_projection"

        if not math.isfinite(bif_s_abs_mm):
            raise RuntimeError("Bifurcation landmark could not be resolved on the normalized trunk path.")
        if float(lowest_anchor.trunk_s_mm) >= bif_s_abs_mm:
            raise RuntimeError(
                f"The D0 reference anchor is not proximal to the bifurcation after trunk normalization (s_anchor={lowest_anchor.trunk_s_mm:.6f}, s_bif={bif_s_abs_mm:.6f})."
            )

        if not get_metadata_ap_orientation_certain(meta):
            warnings.append("W_UPSTREAM_AP_UNCERTAIN: upstream metadata reports AP orientation uncertainty, but the oriented input was still used as the trusted canonical reference.")

        trial_max_abs_mm = max(float(lowest_anchor.trunk_s_mm), bif_s_abs_mm - 0.25)
        trial_measurement = find_valid_slice_near_target(
            surface_pd=surface_pd,
            trunk_path=trunk_path,
            target_s_abs_mm=float(lowest_anchor.trunk_s_mm + 0.5),
            min_s_abs_mm=max(0.0, float(lowest_anchor.trunk_s_mm)),
            max_s_abs_mm=max(0.0, trial_max_abs_mm),
            offsets_mm=[0.0, 0.5, -0.25],
        )
        unit_scale_mm = detect_unit_scale_mm_per_unit(trunk_len_mm, trial_measurement, warnings)
        if renal_reference_mode == "morphology_surrogate":
            surrogate_s = find_stable_segment_surrogate(surface_pd, trunk_path, unit_scale_mm, warnings)
            if surrogate_s is not None:
                lowest_anchor.trunk_s_mm = float(surrogate_s)
                lowest_anchor.point_xyz = interpolate_point_on_path(trunk_path, surrogate_s)
        if float(lowest_anchor.trunk_s_mm) >= bif_s_abs_mm:
            raise RuntimeError(
                f"The D0 reference anchor is not proximal to the bifurcation after unit-scale refinement (s_anchor={lowest_anchor.trunk_s_mm:.6f}, s_bif={bif_s_abs_mm:.6f})."
            )

        d0_nominal_abs_mm = float(lowest_anchor.trunk_s_mm + 0.5 / unit_scale_mm)
        d0_max_abs_mm = max(float(lowest_anchor.trunk_s_mm), bif_s_abs_mm - 0.25 / unit_scale_mm)
        d0_measurement = find_valid_slice_near_target(
            surface_pd=surface_pd,
            trunk_path=trunk_path,
            target_s_abs_mm=d0_nominal_abs_mm,
            min_s_abs_mm=max(0.0, float(lowest_anchor.trunk_s_mm)),
            max_s_abs_mm=max(0.0, d0_max_abs_mm),
            offsets_mm=[0.0, 0.5 / unit_scale_mm, -0.25 / unit_scale_mm, 1.0 / unit_scale_mm, -0.5 / unit_scale_mm],
        )
        if d0_measurement is None:
            raise RuntimeError("Failed to obtain a valid orthogonal slice for D0 just distal to the lowest renal anchor.")
        if abs(float(d0_measurement.sample_s_abs_mm - d0_nominal_abs_mm)) > 0.01:
            warnings.append(
                f"W_D0_FALLBACK: D0 required a nearby fallback slice; actual D0 location is {(d0_measurement.sample_s_abs_mm - lowest_anchor.trunk_s_mm) * unit_scale_mm:.3f} mm distal to the renal anchor."
            )

        d0_abs_mm = float(d0_measurement.sample_s_abs_mm)
        scan_guard_mm = max(1.0 / unit_scale_mm, 0.05 * max(0.0, bif_s_abs_mm - d0_abs_mm))
        scan_end_abs_mm = max(d0_abs_mm, bif_s_abs_mm - scan_guard_mm)
        measurement_max_abs_mm = max(d0_abs_mm, bif_s_abs_mm - 0.25 / unit_scale_mm)
        if scan_end_abs_mm < d0_abs_mm + 1.0 / unit_scale_mm:
            warnings.append("W_SCAN_INTERVAL_SHORT: infrarenal trunk interval is very short; distal scan metrics may be truncated.")

        requested_offsets = {"D0": 0.0, "D5": 5.0 / unit_scale_mm, "D10": 10.0 / unit_scale_mm, "D15": 15.0 / unit_scale_mm}
        measurement_by_name: Dict[str, Optional[SliceMeasurement]] = {"D0": d0_measurement}
        for key, rel_mm in requested_offsets.items():
            if key == "D0":
                continue
            target_abs = d0_abs_mm + rel_mm
            m = find_valid_slice_near_target(
                surface_pd=surface_pd,
                trunk_path=trunk_path,
                target_s_abs_mm=target_abs,
                min_s_abs_mm=d0_abs_mm,
                max_s_abs_mm=measurement_max_abs_mm,
                offsets_mm=[0.0, -0.5 / unit_scale_mm, 0.5 / unit_scale_mm, -1.0 / unit_scale_mm, 1.0 / unit_scale_mm],
            )
            if m is None:
                warnings.append(f"W_{key}_MISSING: could not obtain a valid orthogonal slice within the fallback tolerance window.")
            elif abs(float(m.sample_s_abs_mm - target_abs)) > 0.01:
                warnings.append(
                    f"W_{key}_FALLBACK: nominal {key} slice was unavailable; nearest valid slice at offset {(m.sample_s_abs_mm - d0_abs_mm) * unit_scale_mm:.3f} mm from D0 was used."
                )
            measurement_by_name[key] = m

        coarse_samples = sample_measurements_along_path(surface_pd, trunk_path, d0_abs_mm, scan_end_abs_mm, 1.0 / unit_scale_mm)
        coarse_samples.sort(key=lambda m: m.sample_s_abs_mm)
        neck_end_candidate = detect_neck_end(coarse_samples, d0_measurement, d0_abs_mm, unit_scale_mm, warnings)
        neck_end_s_from_d0_mm = refine_neck_end(
            surface_pd=surface_pd,
            trunk_path=trunk_path,
            candidate_s_from_d0_mm=neck_end_candidate,
            d0_measurement=d0_measurement,
            d0_abs_mm=d0_abs_mm,
            scan_end_abs_mm=scan_end_abs_mm,
            unit_scale_mm=unit_scale_mm,
            warnings=warnings,
        )

        max_aneurysm_measurement: Optional[SliceMeasurement] = None
        if coarse_samples:
            coarse_best = max(coarse_samples, key=lambda m: float(m.major_mm))
            refine_start = max(d0_abs_mm, coarse_best.sample_s_abs_mm - 2.0)
            refine_end = min(scan_end_abs_mm, coarse_best.sample_s_abs_mm + 2.0)
            refined_samples = sample_measurements_along_path(surface_pd, trunk_path, refine_start, refine_end, 0.5 / unit_scale_mm)
            refined_samples.sort(key=lambda m: m.sample_s_abs_mm)
            max_aneurysm_measurement = max(refined_samples, key=lambda m: float(m.major_mm)) if refined_samples else coarse_best
        else:
            warnings.append("W_MAX_DIAMETER_UNAVAILABLE: no valid coarse infrarenal scan slices were available for aneurysm-diameter search.")

        zone_end_abs_mm = min(d0_abs_mm + 15.0 / unit_scale_mm, measurement_max_abs_mm)
        zone_measurements = sample_measurements_along_path(surface_pd, trunk_path, d0_abs_mm, zone_end_abs_mm, 0.5 / unit_scale_mm)
        zone_measurements.sort(key=lambda m: m.sample_s_abs_mm)
        if not zone_measurements:
            warnings.append("W_ZONE_LABEL_EMPTY: D0-to-D15 zone slices could not be generated; the colored VTP will carry an empty neck-zone label.")

        cell_centers_xyz = compute_cell_centers(combined_pd)
        trunk_surface_cell_ids = resolve_trunk_surface_cell_ids(combined_pd, surface_cell_ids, warnings=warnings)
        neck_label, neck_name, s_from_d0 = build_zone_label_arrays(
            combined_pd=combined_pd,
            trunk_surface_cell_ids=trunk_surface_cell_ids,
            cell_centers_xyz=cell_centers_xyz,
            zone_measurements=zone_measurements,
            trunk_path=trunk_path,
            d0_abs_mm=d0_abs_mm,
            unit_scale_mm=unit_scale_mm,
            warnings=warnings,
        )

        output_pd = clone_polydata(combined_pd)
        cd = output_pd.GetCellData()
        add_int_cell_array(cd, "NeckZoneLabel", neck_label)
        add_string_cell_array(cd, "NeckZoneName", neck_name)
        add_double_cell_array(cd, "CenterlineDistanceFromD0_mm", s_from_d0)
        add_double_cell_array(cd, "CenterlineDistanceFromD0Clamped_mm", [max(0.0, v) if math.isfinite(v) else float("nan") for v in s_from_d0])

        write_vtp(output_pd, args.output_colored, binary=True)

        report_lines = make_report_lines(
            measurement_by_name=measurement_by_name,
            d0_abs_mm=d0_abs_mm,
            unit_scale_mm=unit_scale_mm,
            lowest_anchor=lowest_anchor,
            renal_reference_mode=renal_reference_mode,
            neck_end_s_from_d0_mm=neck_end_s_from_d0_mm,
            max_aneurysm_measurement=max_aneurysm_measurement,
            meta=meta,
            warnings=warnings,
            trunk_direction_reversed=any(str(w).startswith(TRUNK_DIRECTION_REVERSED_WARNING) for w in warnings),
        )
        report_lines.append(f"trunk_length_mm={format_mm(trunk_len_mm * unit_scale_mm)}")
        report_lines.append(f"scan_end_s_on_trunk_mm={format_mm(scan_end_abs_mm * unit_scale_mm)}")
        report_lines.append(f"scan_end_s_from_D0_mm={format_mm((scan_end_abs_mm - d0_abs_mm) * unit_scale_mm)}")
        report_lines.append(f"geometry_unit_scale_mm_per_unit={format_mm(unit_scale_mm)}")
        report_lines.append(f"colored_vtp_path={os.path.abspath(args.output_colored)}")
        report_lines.append(f"input_centerlines_vtp={os.path.abspath(args.input_centerlines)}")
        report_lines.append(f"input_surface_with_centerlines_vtp={os.path.abspath(args.input_surface_with_centerlines)}")
        report_lines.append(f"input_metadata_json={os.path.abspath(args.input_metadata)}")

        ensure_parent_dir(args.output_report)
        with open(args.output_report, "w", encoding="utf-8") as f:
            for line in report_lines:
                f.write(str(line).strip() + "\n")

        return 0

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        sys.stderr.write(error_text + "\n")
        sys.stderr.write(traceback.format_exc() + "\n")
        try:
            ensure_parent_dir(args.output_report)
            with open(args.output_report, "w", encoding="utf-8") as f:
                f.write("status=failed\n")
                f.write(f"error={error_text}\n")
                f.write(f"warnings={join_warnings(warnings)}\n")
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
