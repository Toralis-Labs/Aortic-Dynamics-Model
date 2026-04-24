#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
First-stage anatomical preprocessing for an unlabeled open-termini abdominal arterial lumen surface (.vtp).

Primary goal:
- Input: unlabeled, open-termini lumen surface VTP representing abdominal arterial tree with branches.
- Output: an oriented, anatomy-aware surface+centerline VTP in a shared canonical frame,
  plus a secondary oriented labeled centerline scaffold VTP and optional metadata JSON
  that later measurement code can consume directly without re-solving anatomy.

Required labeled targets:
- abdominal aorta trunk
- right main iliac
- left main iliac
- right renal
- left renal

Required landmarks:
- proximal aortic inlet tip
- distal aortic bifurcation
- right renal origin on trunk
- left renal origin on trunk

Dependencies:
- vtk
- numpy
- VMTK python bindings (vtkvmtk) REQUIRED

No manual interaction. Deterministic best-effort logic with warnings/confidence outputs.
"""

from __future__ import annotations

# -----------------------------
# User-editable paths (required)
# -----------------------------
INPUT_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\0044_H_ABAO_AAA\\0044_H_ABAO_AAA\\Models\\0156_0001.vtp"
OUTPUT_SURFACE_WITH_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_surface_with_centerlines.vtp"
OUTPUT_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines.vtp"
OUTPUT_METADATA_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines_metadata.json"
OUTPUT_DEBUG_CENTERLINES_RAW_PATH = ""  # optional: e.g. "debug_centerlines_raw.vtp" (empty disables)

import os
import sys
import json
import math
import argparse
import importlib
import platform
import re
import traceback
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

SCRIPT_PATH = os.path.abspath(__file__) if "__file__" in globals() else os.path.abspath(sys.argv[0])
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
_VTK_IMPORT_ERROR = ""
_WINDOWS_DLL_DIR_HANDLES: List[Any] = []
_WINDOWS_DLL_DIRECTORIES_ADDED: List[str] = []
_LAST_WINDOWS_DLL_ATTEMPTS: List[str] = []
_LAST_VMTK_IMPORT_DIAGNOSTICS: Dict[str, Any] = {}
_VMTK_EXTENSION_PROBES = [
    "vmtk.vtkvmtkSegmentationPython",
    "vmtk.vtkvmtkComputationalGeometryPython",
    "vmtk.vtkvmtkDifferentialGeometryPython",
    "vmtk.vtkvmtkMiscPython",
    "vmtk.vtkvmtkRenderingPython",
]
_VMTK_REQUIRED_SYMBOLS = [
    "vtkvmtkCapPolyData",
    "vtkvmtkPolyDataCenterlines",
]


def _normalize_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _resolve_user_path(path: str) -> str:
    path = (path or "").strip()
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))


def _format_exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _prepare_windows_dll_search_paths() -> Dict[str, Any]:
    global _LAST_WINDOWS_DLL_ATTEMPTS

    info: Dict[str, Any] = {
        "platform": os.name,
        "prefixes": [],
        "dll_directories_attempted": [],
        "dll_directories_added": [],
        "dll_add_errors": {},
        "path_prepended": [],
    }
    if os.name != "nt":
        return info

    prefix_candidates: List[str] = []
    prefix_seen = set()
    for raw_prefix in (
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        os.path.dirname(sys.executable),
    ):
        if not raw_prefix:
            continue
        prefix = os.path.abspath(raw_prefix)
        key = _normalize_path_key(prefix)
        if key in prefix_seen or not os.path.isdir(prefix):
            continue
        prefix_seen.add(key)
        prefix_candidates.append(prefix)
    info["prefixes"] = list(prefix_candidates)

    current_path_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    current_path_keys = {_normalize_path_key(p) for p in current_path_entries}
    added_dir_keys = {_normalize_path_key(p) for p in _WINDOWS_DLL_DIRECTORIES_ADDED}
    attempted_keys = set()
    prepend_entries: List[str] = []

    for prefix in prefix_candidates:
        for candidate in (
            prefix,
            os.path.join(prefix, "Library", "bin"),
            os.path.join(prefix, "Scripts"),
            os.path.join(prefix, "bin"),
            os.path.join(prefix, "Lib", "site-packages", "vmtk"),
            os.path.join(prefix, "Lib", "site-packages", "vtkmodules"),
        ):
            if not os.path.isdir(candidate):
                continue
            candidate_abs = os.path.abspath(candidate)
            key = _normalize_path_key(candidate_abs)
            if key in attempted_keys:
                continue
            attempted_keys.add(key)
            info["dll_directories_attempted"].append(candidate_abs)

            if hasattr(os, "add_dll_directory") and key not in added_dir_keys:
                try:
                    _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(candidate_abs))
                    _WINDOWS_DLL_DIRECTORIES_ADDED.append(candidate_abs)
                    added_dir_keys.add(key)
                    info["dll_directories_added"].append(candidate_abs)
                except Exception as exc:
                    info["dll_add_errors"][candidate_abs] = _format_exception_text(exc)

            if key not in current_path_keys:
                prepend_entries.append(candidate_abs)
                current_path_keys.add(key)
                info["path_prepended"].append(candidate_abs)

    if prepend_entries:
        os.environ["PATH"] = os.pathsep.join(prepend_entries + current_path_entries)

    _LAST_WINDOWS_DLL_ATTEMPTS = list(info["dll_directories_attempted"])
    return info


_prepare_windows_dll_search_paths()

import numpy as np

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk  # type: ignore
except Exception as e:  # pragma: no cover
    vtk = None
    vtk_to_numpy = None
    numpy_to_vtk = None
    _VTK_IMPORT_ERROR = str(e)

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import (
        vtkPolyData,
        vtkCellData,
        vtkPointData,
        vtkFieldData,
        vtkStaticPointLocator,
    )


# -----------------------------
# Label schema (stable integers)
# -----------------------------
LABEL_OTHER = 0
LABEL_AORTA_TRUNK = 1
LABEL_RIGHT_ILIAC = 2
LABEL_LEFT_ILIAC = 3
LABEL_RIGHT_RENAL = 4
LABEL_LEFT_RENAL = 5

LABEL_ID_TO_NAME = {
    LABEL_OTHER: "other",
    LABEL_AORTA_TRUNK: "abdominal_aorta_trunk",
    LABEL_RIGHT_ILIAC: "right_main_iliac",
    LABEL_LEFT_ILIAC: "left_main_iliac",
    LABEL_RIGHT_RENAL: "right_renal",
    LABEL_LEFT_RENAL: "left_renal",
}

GEOMETRY_TYPE_SURFACE = 1
GEOMETRY_TYPE_CENTERLINE = 2


@dataclass
class TerminationLoop:
    center: np.ndarray
    area: float
    diameter_eq: float
    normal: np.ndarray
    rms_planarity: float
    n_points: int
    source: str


@dataclass
class AnatomyResult:
    inlet_node: int
    inlet_conf: float
    bif_node: int
    bif_conf: float
    iliac_ep_a: int
    iliac_ep_b: int
    trunk_path: List[int]
    iliac_path_a: List[int]  # bif -> distal
    iliac_path_b: List[int]  # bif -> distal
    iliac_main_path_a: List[int]  # bif -> common iliac (or distal if no split)
    iliac_main_path_b: List[int]
    renal_ep_right: Optional[int]
    renal_ep_left: Optional[int]
    renal_takeoff_right: Optional[int]
    renal_takeoff_left: Optional[int]
    renal_conf: float
    ap_conf: float
    ap_warn: bool
    warnings: List[str]
    confidences: Dict[str, float]
    transform_R: np.ndarray
    transform_origin: np.ndarray
    flipped_for_ap: bool


# -----------------------------
# Basic numeric helpers
# -----------------------------
EPS = 1e-12


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros(3, dtype=float)
    return (v / n).astype(float)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns principal axes as columns (e0,e1,e2) sorted by descending eigenvalue.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return np.eye(3), np.ones(3), np.mean(pts, axis=0) if pts.shape[0] else np.zeros(3)
    c = np.mean(pts, axis=0)
    X = pts - c
    cov = (X.T @ X) / max(1, X.shape[0])
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    # Orthonormalize (numerically safe)
    e0 = unit(V[:, 0])
    e1 = unit(V[:, 1] - np.dot(V[:, 1], e0) * e0)
    e2 = unit(np.cross(e0, e1))
    A = np.column_stack([e0, e1, e2])
    return A, w.astype(float), c.astype(float)


def planar_polygon_area_and_normal(points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Estimate area and normal of an approximately planar closed polygon using PCA plane fit.

    Returns (area, normal_unit, rms_plane_distance).
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, np.zeros(3), float("nan")
    # Remove duplicate last point if it closes the loop
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-8 * (np.linalg.norm(pts[0]) + 1.0):
        pts = pts[:-1]
    if pts.shape[0] < 3:
        return 0.0, np.zeros(3), float("nan")

    A, w, c = pca_axes(pts)
    # For planar set, smallest eigenvector is plane normal; since A columns are descending,
    # normal is e2 (third axis).
    n = unit(A[:, 2])
    u = unit(A[:, 0])
    v = unit(np.cross(n, u))

    X = pts - c
    dists = X @ n
    rms = float(np.sqrt(np.mean(dists * dists))) if dists.size else float("nan")

    x2 = X @ u
    y2 = X @ v
    x_next = np.roll(x2, -1)
    y_next = np.roll(y2, -1)
    area = 0.5 * float(abs(np.sum(x2 * y_next - x_next * y2)))
    return area, n.astype(float), rms


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    d = pts[1:] - pts[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


def compute_tangents(points: np.ndarray) -> np.ndarray:
    """
    Central-difference tangents for polyline points.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    T = np.zeros((n, 3), dtype=float)
    if n < 2:
        return T
    for i in range(n):
        if i == 0:
            v = pts[1] - pts[0]
        elif i == n - 1:
            v = pts[-1] - pts[-2]
        else:
            v = pts[i + 1] - pts[i - 1]
        T[i] = unit(v)
    return T


def compute_abscissa(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    s = np.zeros((n,), dtype=float)
    if n < 2:
        return s
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s[1:] = np.cumsum(d)
    return s


# -----------------------------
# VTK safety helpers
# -----------------------------
def require_vtk() -> None:
    if vtk is None:
        raise RuntimeError(f"VTK import failed: {_VTK_IMPORT_ERROR}")


def load_vtp(path: str) -> "vtkPolyData":
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    pd = reader.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read VTP or empty polydata: {path}")
    return pd


def write_vtp(pd: "vtkPolyData", path: str, binary: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(pd)
    if binary:
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def get_points_numpy(pd: "vtkPolyData") -> np.ndarray:
    pts = pd.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    arr = pts.GetData()
    if vtk_to_numpy is None:
        raise RuntimeError("vtk_to_numpy unavailable")
    return vtk_to_numpy(arr).astype(float)


def clean_and_triangulate_surface(pd: "vtkPolyData") -> "vtkPolyData":
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.PointMergingOn()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOn()
    cleaner.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cleaner.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(tri.GetOutput())

    # Ensure consistent normals as a robustness measure for some VMTK builds
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(out)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()

    out2 = vtk.vtkPolyData()
    out2.DeepCopy(normals.GetOutput())
    out2.BuildLinks()
    return out2


def count_boundary_edges(pd: "vtkPolyData") -> int:
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    edges = fe.GetOutput()
    return int(edges.GetNumberOfCells()) if edges is not None else 0


def build_static_locator(pd: "vtkPolyData") -> "vtkStaticPointLocator":
    loc = vtk.vtkStaticPointLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    return loc


def polyline_length_from_ids(pts: np.ndarray, ids: List[int]) -> float:
    if len(ids) < 2:
        return 0.0
    length = 0.0
    prev = int(ids[0])
    for cur in ids[1:]:
        cur_i = int(cur)
        if cur_i == prev:
            continue
        length += float(np.linalg.norm(pts[cur_i] - pts[prev]))
        prev = cur_i
    return float(length)


def extract_polyline_cells(src: "vtkPolyData", cell_ids: List[int]) -> "vtkPolyData":
    pts_src = src.GetPoints()
    if pts_src is None:
        return vtk.vtkPolyData()

    out = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    old_to_new: Dict[int, int] = {}

    for ci in cell_ids:
        cell = src.GetCell(int(ci))
        if cell is None:
            continue
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
            continue

        raw_ids = [int(cell.GetPointId(k)) for k in range(cell.GetNumberOfPoints())]
        ids: List[int] = []
        for pid in raw_ids:
            if not ids or pid != ids[-1]:
                ids.append(int(pid))
        if len(ids) < 2:
            continue

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(ids))
        for local_idx, old_pid in enumerate(ids):
            new_pid = old_to_new.get(old_pid)
            if new_pid is None:
                p = pts_src.GetPoint(old_pid)
                new_pid = int(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
                old_to_new[old_pid] = new_pid
            polyline.GetPointIds().SetId(local_idx, new_pid)
        lines.InsertNextCell(polyline)

    out.SetPoints(points)
    out.SetLines(lines)
    out.BuildLinks()
    return out


def clean_centerlines_preserve_lines(cl: "vtkPolyData") -> "vtkPolyData":
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(cl)
    cleaner.PointMergingOn()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.Update()

    merged = vtk.vtkPolyData()
    merged.DeepCopy(cleaner.GetOutput())

    pts = get_points_numpy(merged)
    if pts.shape[0] == 0 or merged.GetNumberOfCells() == 0:
        return merged

    bbox = merged.GetBounds()
    diag = float(np.linalg.norm(np.array([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float)))
    # Only remove clearly degenerate line artifacts. We intentionally do not keep
    # only the largest component because clinically meaningful side branches can
    # appear in smaller connected regions before graph reconciliation.
    min_cell_length = max(1e-6 * max(diag, 1.0), 1e-6)

    keep_cells: List[int] = []
    for ci in range(merged.GetNumberOfCells()):
        cell = merged.GetCell(ci)
        if cell is None:
            continue
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
            continue
        ids = [int(cell.GetPointId(k)) for k in range(cell.GetNumberOfPoints())]
        if len(ids) < 2:
            continue
        if polyline_length_from_ids(pts, ids) <= min_cell_length:
            continue
        keep_cells.append(int(ci))

    return extract_polyline_cells(merged, keep_cells)


def find_face_partition_array_name(pd: "vtkPolyData") -> Optional[str]:
    cd = pd.GetCellData()
    if cd is None:
        return None
    n = cd.GetNumberOfArrays()
    best = None
    # Strong preference for ModelFaceID
    for i in range(n):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if not name:
            continue
        if name.lower() == "modelfaceid":
            return name
    # Fallback heuristic
    for i in range(n):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if not name:
            continue
        lname = name.lower()
        if ("face" in lname and "id" in lname) or lname.endswith("faceid") or lname.endswith("_face"):
            best = name
            break
    return best


# -----------------------------
# Termination detection
# -----------------------------
def extract_boundary_loops(pd: "vtkPolyData") -> List[TerminationLoop]:
    loops: List[TerminationLoop] = []
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()

    edges = fe.GetOutput()
    if edges is None or edges.GetNumberOfCells() == 0:
        return loops

    stripper = vtk.vtkStripper()
    stripper.SetInputData(edges)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    out = stripper.GetOutput()
    pts = out.GetPoints()
    if pts is None:
        return loops

    for ci in range(out.GetNumberOfCells()):
        cell = out.GetCell(ci)
        if cell is None:
            continue
        nids = cell.GetNumberOfPoints()
        if nids < 3:
            continue
        coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
        center = np.mean(coords, axis=0)
        area, normal, rms = planar_polygon_area_and_normal(coords)
        diameter_eq = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
        loops.append(
            TerminationLoop(
                center=center.astype(float),
                area=float(area),
                diameter_eq=float(diameter_eq),
                normal=normal.astype(float),
                rms_planarity=float(rms),
                n_points=int(nids),
                source="boundary_loop",
            )
        )
    return loops


def termination_candidates_from_face_partitions(pd_tri: "vtkPolyData", face_array: str) -> List[TerminationLoop]:
    """
    Best-effort cap inference for face-partitioned/capped surfaces (SimVascular-like).
    Treat highly planar faces as termination candidates.
    """
    candidates: List[TerminationLoop] = []
    cd = pd_tri.GetCellData()
    if cd is None:
        return candidates
    arr = cd.GetArray(face_array)
    if arr is None or vtk_to_numpy is None:
        return candidates

    # Compute per-cell area and normals
    cell_size = vtk.vtkCellSizeFilter()
    cell_size.SetInputData(pd_tri)
    cell_size.SetComputeArea(True)
    cell_size.SetComputeLength(False)
    cell_size.SetComputeVolume(False)
    cell_size.SetComputeVertexCount(False)
    cell_size.Update()
    pd_area = cell_size.GetOutput()

    normals_f = vtk.vtkPolyDataNormals()
    normals_f.SetInputData(pd_area)
    normals_f.ComputePointNormalsOff()
    normals_f.ComputeCellNormalsOn()
    normals_f.SplittingOff()
    normals_f.ConsistencyOn()
    normals_f.AutoOrientNormalsOff()
    normals_f.Update()
    pd_n = normals_f.GetOutput()

    centers_f = vtk.vtkCellCenters()
    centers_f.SetInputData(pd_n)
    centers_f.VertexCellsOn()
    centers_f.Update()
    centers_pd = centers_f.GetOutput()
    centers_pts = centers_pd.GetPoints()
    if centers_pts is None:
        return candidates

    face_vals = vtk_to_numpy(pd_n.GetCellData().GetArray(face_array)).astype(np.int64)
    area_vals = vtk_to_numpy(pd_n.GetCellData().GetArray("Area")).astype(float)

    cell_normals_vtk = pd_n.GetCellData().GetNormals()
    if cell_normals_vtk is None:
        cell_normals_vtk = pd_n.GetCellData().GetArray("Normals")
    if cell_normals_vtk is None:
        return candidates
    normal_vals = vtk_to_numpy(cell_normals_vtk).astype(float)
    centers_vals = vtk_to_numpy(centers_pts.GetData()).astype(float)

    if centers_vals.shape[0] != face_vals.shape[0]:
        return candidates

    total_area = float(np.sum(area_vals)) if area_vals.size else 0.0
    if total_area <= 0:
        return candidates

    uniq = np.unique(face_vals)
    face_stats: List[Dict[str, Any]] = []
    for fid in uniq:
        mask = (face_vals == fid)
        if not np.any(mask):
            continue
        a = area_vals[mask]
        a_sum = float(np.sum(a))
        if a_sum <= 0:
            continue
        c = np.sum(centers_vals[mask] * a[:, None], axis=0) / a_sum
        n_sum = np.sum(normal_vals[mask] * a[:, None], axis=0)
        planarity = float(np.linalg.norm(n_sum) / (a_sum + EPS))
        diameter_eq = math.sqrt(4.0 * a_sum / math.pi)
        face_stats.append({"fid": int(fid), "area": a_sum, "center": c, "planarity": planarity, "diameter_eq": diameter_eq})

    if not face_stats:
        return candidates

    areas = np.array([fs["area"] for fs in face_stats], dtype=float)
    max_area = float(np.max(areas)) if areas.size else 0.0

    for fs in face_stats:
        if fs["planarity"] < 0.92:
            continue
        if fs["area"] > 0.60 * total_area:
            continue
        if max_area > 0 and fs["area"] > 0.85 * max_area and len(face_stats) > 3:
            continue
        candidates.append(
            TerminationLoop(
                center=np.array(fs["center"], dtype=float),
                area=float(fs["area"]),
                diameter_eq=float(fs["diameter_eq"]),
                normal=np.zeros(3, dtype=float),
                rms_planarity=float("nan"),
                n_points=0,
                source=f"face_partition:{face_array}",
            )
        )
    return candidates


def detect_terminations(pd_tri: "vtkPolyData", warnings: List[str]) -> Tuple[List[TerminationLoop], str]:
    # Primary: open boundary loops
    if count_boundary_edges(pd_tri) > 0:
        loops = extract_boundary_loops(pd_tri)
        if len(loops) >= 2:
            return loops, "open_termini"

    # Secondary: partitioned/capped face ids
    face_array = find_face_partition_array_name(pd_tri)
    if face_array:
        terms = termination_candidates_from_face_partitions(pd_tri, face_array)
        if len(terms) >= 2:
            warnings.append(f"W_TERMINATIONS_FACEPART: boundary loops not found; using planar face partitions via '{face_array}'.")
            return terms, "capped_partitioned"

    # Tertiary: feature edge loops (closed unpartitioned)
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd_tri)
    fe.BoundaryEdgesOff()
    fe.FeatureEdgesOn()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.SetFeatureAngle(60.0)
    fe.Update()
    edges = fe.GetOutput()
    if edges is not None and edges.GetNumberOfCells():
        stripper = vtk.vtkStripper()
        stripper.SetInputData(edges)
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        out = stripper.GetOutput()
        pts = out.GetPoints()
        if pts is not None:
            loops: List[TerminationLoop] = []
            for ci in range(out.GetNumberOfCells()):
                cell = out.GetCell(ci)
                if cell is None:
                    continue
                nids = cell.GetNumberOfPoints()
                if nids < 6:
                    continue
                coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
                center = np.mean(coords, axis=0)
                area, normal, rms = planar_polygon_area_and_normal(coords)
                diameter_eq = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
                loops.append(
                    TerminationLoop(
                        center=center.astype(float),
                        area=float(area),
                        diameter_eq=float(diameter_eq),
                        normal=normal.astype(float),
                        rms_planarity=float(rms),
                        n_points=int(nids),
                        source="feature_edge_loop",
                    )
                )
            if len(loops) >= 2:
                warnings.append("W_TERMINATIONS_FEATUREEDGES: boundary loops not found; using feature-edge loops (less reliable).")
                return loops, "closed_unpartitioned"

    warnings.append("W_TERMINATIONS_NONE: failed to detect terminations robustly.")
    return [], "unsupported"


# -----------------------------
# Inlet selection (not just largest cap)
# -----------------------------
def choose_inlet_termination(
    terms: List[TerminationLoop],
    surface_points: np.ndarray,
    warnings: List[str],
) -> Tuple[Optional[TerminationLoop], float, np.ndarray]:
    """
    Choose the true proximal superior aortic inlet termination.

    Strategy:
    - Estimate global superior-inferior axis via PCA of surface points.
    - Use a diameter-aware "iliac pair" detector: if the two largest terminations are on the same axial end
      and laterally separated, interpret that end as distal (iliac end), so inlet is on the opposite end.
    - Otherwise, choose the largest-diameter termination that is most extreme on the superior end.

    Returns: (inlet_termination_or_None, confidence, axis_si_unit) where axis is oriented so inlet has positive projection.
    """
    if not terms:
        return None, 0.0, np.array([0.0, 0.0, 1.0], dtype=float)

    A, w, c = pca_axes(surface_points if surface_points.shape[0] > 0 else np.array([t.center for t in terms], dtype=float))
    axis = unit(A[:, 0])
    if np.linalg.norm(axis) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    centers = np.array([t.center for t in terms], dtype=float)
    diams = np.array([max(t.diameter_eq, 0.0) for t in terms], dtype=float)
    areas = np.array([max(t.area, 0.0) for t in terms], dtype=float)

    # Projections along axis (relative to mean of term centers for stability)
    cc = np.mean(centers, axis=0)
    proj = (centers - cc) @ axis

    # Sort by diameter desc
    order = np.argsort(diams)[::-1]
    t1 = order[0]
    t2 = order[1] if len(order) > 1 else None

    # Determine if the largest two diameters likely represent paired iliacs on same distal end
    distal_sign: Optional[float] = None
    iliac_pair_conf = 0.0
    if t2 is not None and diams[t1] > 0 and diams[t2] > 0:
        s1 = float(np.sign(proj[t1])) if abs(proj[t1]) > EPS else 0.0
        s2 = float(np.sign(proj[t2])) if abs(proj[t2]) > EPS else 0.0
        if s1 != 0.0 and s1 == s2:
            dvec = centers[t1] - centers[t2]
            lateral_sep = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
            avg_d = 0.5 * (diams[t1] + diams[t2])
            if avg_d > 0:
                sep_ratio = lateral_sep / (avg_d + EPS)
                if sep_ratio > 1.25:  # strong lateral separation relative to vessel caliber
                    distal_sign = s1
                    iliac_pair_conf = float(clamp((sep_ratio - 1.25) / 1.5 + 0.35, 0.0, 1.0))

    # Determine inlet side sign
    if distal_sign is not None and distal_sign != 0.0:
        inlet_sign = -distal_sign
        candidates = [i for i in range(len(terms)) if float(np.sign(proj[i]) if abs(proj[i]) > EPS else 0.0) == inlet_sign]
        if not candidates:
            # Fallback: take most extreme opposite to distal
            inlet_idx = int(np.argmin(proj)) if distal_sign > 0 else int(np.argmax(proj))
            warnings.append("W_INLET_SIDE_EMPTY: distal-end inferred via iliac-pair, but no terminations on opposite sign; using axial extreme.")
        else:
            # Choose among candidates: favor large diameter and axial extremity
            proj_norm = (proj - np.min(proj)) / (np.ptp(proj) + EPS)
            diam_norm = (diams - np.min(diams)) / (np.ptp(diams) + EPS)
            area_norm = (areas - np.min(areas)) / (np.ptp(areas) + EPS)
            scores = []
            for i in candidates:
                # axial extremity: if inlet_sign is positive, use proj_norm; else use 1-proj_norm
                axial_score = float(proj_norm[i] if inlet_sign > 0 else (1.0 - proj_norm[i]))
                score = 0.55 * float(diam_norm[i]) + 0.25 * axial_score + 0.20 * float(area_norm[i])
                scores.append((score, i))
            scores.sort(reverse=True)
            inlet_idx = int(scores[0][1])
        # Orient axis so inlet has positive projection
        if proj[inlet_idx] < 0:
            axis = -axis
            proj = -proj
        # Confidence: combine iliac_pair_conf with inlet dominance
        diam_sorted = np.sort(diams)[::-1]
        diam_ratio = float(diam_sorted[0] / (diam_sorted[1] + EPS)) if len(diam_sorted) > 1 else 2.0
        inlet_extremity = float(abs(proj[inlet_idx]) / (np.max(abs(proj)) + EPS))
        conf = float(clamp(0.40 + 0.35 * iliac_pair_conf + 0.15 * clamp(diam_ratio - 1.0, 0.0, 1.0) + 0.10 * inlet_extremity, 0.0, 1.0))
        return terms[inlet_idx], conf, axis

    # No distal pair detected: default to largest diameter, but still enforce superior extremity
    # Choose among top diameter candidates: largest diameter with maximal proj (superior)
    top_k = min(4, len(terms))
    top = order[:top_k]
    # favor diameter, but also axial extremity: inlet should be at an axial extreme
    scores = []
    for i in top:
        score = float(diams[i]) + 0.15 * float(proj[i])  # proj encourages superior
        scores.append((score, int(i)))
    scores.sort(reverse=True)
    inlet_idx = int(scores[0][1])

    # Orient axis so inlet is positive
    if proj[inlet_idx] < 0:
        axis = -axis
        proj = -proj

    # Confidence using how unique the inlet is in diameter and axial position
    diam_sorted = np.sort(diams)[::-1]
    diam_ratio = float(diam_sorted[0] / (diam_sorted[1] + EPS)) if len(diam_sorted) > 1 else 2.0
    inlet_extremity = float(abs(proj[inlet_idx]) / (np.max(abs(proj)) + EPS))
    conf = float(clamp(0.35 + 0.30 * clamp(diam_ratio - 1.0, 0.0, 1.0) + 0.35 * inlet_extremity, 0.0, 1.0))
    if conf < 0.55:
        warnings.append("W_INLET_LOWCONF: inlet detection may be ambiguous; used diameter+axis-score selection.")
    return terms[inlet_idx], conf, axis


# -----------------------------
# VMTK import and centerlines
# -----------------------------
def _extract_failing_extension_module(diagnostics: Dict[str, Any]) -> Optional[str]:
    attempts = diagnostics.get("import_attempts", [])
    for attempt in attempts:
        name = str(attempt.get("name", ""))
        err_text = str(attempt.get("error", "")).lower()
        if (
            attempt.get("ok") is False
            and name.startswith("probe ")
            and any(token in err_text for token in ("dll load failed", "specified module could not be found", "cannot open shared object file"))
        ):
            return name.replace("probe ", "", 1)

    pattern = re.compile(r"(vtkvmtk[A-Za-z]+Python)")
    for attempt in attempts:
        err_text = str(attempt.get("error", ""))
        match = pattern.search(err_text)
        if match:
            module_name = match.group(1)
            if module_name.startswith("vmtk."):
                return module_name
            return f"vmtk.{module_name}"
    return None


def _format_vmtk_import_failure_details(diagnostics: Dict[str, Any]) -> str:
    attempts = diagnostics.get("import_attempts", [])
    attempted_names = [str(a.get("name", "")) for a in attempts]
    lines = [
        "This usually means a dependent VMTK DLL could not be found even though the package is installed.",
        f"Attempted import paths: {', '.join(attempted_names) if attempted_names else '<none>'}",
        f"VTK import ok: {bool(diagnostics.get('vtk_import_ok'))}",
        f"Python executable: {diagnostics.get('python_executable', sys.executable)}",
        f"sys.prefix: {diagnostics.get('sys_prefix', sys.prefix)}",
        f"CONDA_PREFIX: {diagnostics.get('conda_prefix') or '<unset>'}",
        f"DLL directories added: {', '.join(diagnostics.get('dll_directories_added', [])) or '<none>'}",
    ]
    failing_extension = diagnostics.get("failing_extension_module")
    if failing_extension:
        lines.append(f"Likely failing extension module: {failing_extension}")
    for attempt in attempts:
        if attempt.get("ok"):
            lines.append(f"{attempt.get('name')}: OK")
        else:
            lines.append(f"{attempt.get('name')}: {attempt.get('error')}")
    return "\n".join(lines)


def resolve_vmtk_import() -> Tuple[Optional[Any], Dict[str, Any]]:
    dll_info = _prepare_windows_dll_search_paths()
    diagnostics: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cwd": os.getcwd(),
        "script_path": SCRIPT_PATH,
        "path_head": [p for p in os.environ.get("PATH", "").split(os.pathsep) if p][:8],
        "dll_search_prefixes": dll_info.get("prefixes", []),
        "dll_directories_attempted": dll_info.get("dll_directories_attempted", []) or list(_LAST_WINDOWS_DLL_ATTEMPTS),
        "dll_directories_added": list(_WINDOWS_DLL_DIRECTORIES_ADDED),
        "dll_add_errors": dll_info.get("dll_add_errors", {}),
        "vtk_import_ok": vtk is not None,
        "vtk_import_error": None if vtk is not None else _VTK_IMPORT_ERROR,
        "import_attempts": [],
        "vmtk_import_ok": False,
        "resolved_vmtk_source": None,
        "failing_extension_module": None,
        "loaded_probe_modules": [],
    }

    attempts: List[Dict[str, Any]] = diagnostics["import_attempts"]

    def _attempt(name: str, importer: Any) -> Optional[Any]:
        try:
            module_obj = importer()
            attempts.append(
                {
                    "name": name,
                    "ok": True,
                    "module": getattr(module_obj, "__name__", type(module_obj).__name__),
                }
            )
            return module_obj
        except Exception as exc:
            attempts.append(
                {
                    "name": name,
                    "ok": False,
                    "error": _format_exception_text(exc),
                }
            )
            return None

    direct_module = _attempt("from vmtk import vtkvmtk", lambda: importlib.import_module("vmtk.vtkvmtk"))
    if direct_module is not None:
        diagnostics["vmtk_import_ok"] = True
        diagnostics["resolved_vmtk_source"] = "vmtk.vtkvmtk"
        return direct_module, diagnostics

    top_level_module = _attempt("import vtkvmtk", lambda: importlib.import_module("vtkvmtk"))
    if top_level_module is not None:
        diagnostics["vmtk_import_ok"] = True
        diagnostics["resolved_vmtk_source"] = "vtkvmtk"
        return top_level_module, diagnostics

    loaded_probe_modules: List[Any] = []
    for module_name in _VMTK_EXTENSION_PROBES:
        probed = _attempt(f"probe {module_name}", lambda name=module_name: importlib.import_module(name))
        if probed is not None:
            loaded_probe_modules.append(probed)
    diagnostics["loaded_probe_modules"] = [getattr(m, "__name__", "") for m in loaded_probe_modules]

    if loaded_probe_modules:
        merged_module = types.ModuleType("vtkvmtk_fallback")
        merged_module.__dict__["__source_modules__"] = [getattr(m, "__name__", "") for m in loaded_probe_modules]
        for probe_module in loaded_probe_modules:
            for attr_name in dir(probe_module):
                if attr_name.startswith("_") or attr_name in merged_module.__dict__:
                    continue
                merged_module.__dict__[attr_name] = getattr(probe_module, attr_name)

        missing_symbols = [name for name in _VMTK_REQUIRED_SYMBOLS if not hasattr(merged_module, name)]
        diagnostics["fallback_missing_symbols"] = missing_symbols
        if not missing_symbols:
            diagnostics["vmtk_import_ok"] = True
            diagnostics["resolved_vmtk_source"] = "extension_fallback"
            return merged_module, diagnostics

    diagnostics["failing_extension_module"] = _extract_failing_extension_module(diagnostics)
    return None, diagnostics


def try_import_vmtk() -> Tuple[Optional[Any], Optional[str]]:
    global _LAST_VMTK_IMPORT_DIAGNOSTICS
    vtkvmtk_mod, diagnostics = resolve_vmtk_import()
    _LAST_VMTK_IMPORT_DIAGNOSTICS = diagnostics
    if vtkvmtk_mod is not None:
        return vtkvmtk_mod, None
    return None, _format_vmtk_import_failure_details(diagnostics)


def debug_vmtk_runtime_report(diagnostics: Optional[Dict[str, Any]] = None) -> str:
    diag = diagnostics
    if diag is None:
        if _LAST_VMTK_IMPORT_DIAGNOSTICS:
            diag = dict(_LAST_VMTK_IMPORT_DIAGNOSTICS)
        else:
            _, diag = resolve_vmtk_import()

    attempts = diag.get("import_attempts", [])
    lines = [
        f"cwd: {os.getcwd()}",
        f"script_path: {SCRIPT_PATH}",
        f"python_executable: {diag.get('python_executable', sys.executable)}",
        f"sys.prefix: {diag.get('sys_prefix', sys.prefix)}",
        f"CONDA_PREFIX: {diag.get('conda_prefix') or '<unset>'}",
        f"vtk import: {'ok' if diag.get('vtk_import_ok') else 'failed'}",
        f"VMTK import: {'ok' if diag.get('vmtk_import_ok') else 'failed'}",
    ]
    if diag.get("resolved_vmtk_source"):
        lines.append(f"VMTK source: {diag.get('resolved_vmtk_source')}")
    if diag.get("failing_extension_module"):
        lines.append(f"failing extension module: {diag.get('failing_extension_module')}")
    if diag.get("dll_directories_added"):
        lines.append(f"dll dirs added: {', '.join(diag.get('dll_directories_added', []))}")
    for attempt in attempts:
        if attempt.get("ok"):
            lines.append(f"{attempt.get('name')}: OK")
        else:
            lines.append(f"{attempt.get('name')}: {attempt.get('error')}")
    return "\n".join(lines)


def cap_surface_if_open(pd_tri: "vtkPolyData", vtkvmtk_mod: Any) -> Tuple["vtkPolyData", bool]:
    if count_boundary_edges(pd_tri) <= 0:
        out = vtk.vtkPolyData()
        out.DeepCopy(pd_tri)
        return out, False
    capper = vtkvmtk_mod.vtkvmtkCapPolyData()
    capper.SetInputData(pd_tri)
    capper.SetDisplacement(0.0)
    capper.SetInPlaneDisplacement(0.0)
    capper.Update()
    capped = vtk.vtkPolyData()
    capped.DeepCopy(capper.GetOutput())
    return capped, True


def compute_centerlines_vmtk(
    pd_tri: "vtkPolyData",
    inlet_center: np.ndarray,
    term_centers: List[np.ndarray],
    warnings: List[str],
) -> Tuple["vtkPolyData", Dict[str, Any]]:
    """
    Compute centerlines using VMTK. Required dependency.

    Seeds:
    - source: closest surface point to inlet_center
    - targets: closest surface points to all other termination centers (filtered for tiny terminations)
    """
    vtkvmtk_mod, err = try_import_vmtk()
    diagnostics = dict(_LAST_VMTK_IMPORT_DIAGNOSTICS) if _LAST_VMTK_IMPORT_DIAGNOSTICS else {}
    if vtkvmtk_mod is None:
        raise RuntimeError(
            "VMTK vtkvmtk not available (required).\n"
            f"{err or _format_vmtk_import_failure_details(diagnostics)}"
        )

    capped, did_cap = cap_surface_if_open(pd_tri, vtkvmtk_mod)
    locator = build_static_locator(capped)

    inlet_pid = int(locator.FindClosestPoint(float(inlet_center[0]), float(inlet_center[1]), float(inlet_center[2])))

    # Filter targets: remove those extremely close to inlet and duplicates
    target_pids = []
    seen = set([inlet_pid])
    for c in term_centers:
        pid = int(locator.FindClosestPoint(float(c[0]), float(c[1]), float(c[2])))
        if pid in seen:
            continue
        p = np.array(capped.GetPoint(pid), dtype=float)
        if float(np.linalg.norm(p - inlet_center)) < 1e-6:
            continue
        seen.add(pid)
        target_pids.append(pid)

    if len(target_pids) < 1:
        raise RuntimeError("Insufficient target seeds for centerline extraction (need >=1).")

    bbox = capped.GetBounds()
    diag = float(np.linalg.norm(np.array([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float)))
    step = max(0.005 * diag, 0.5)  # adaptive resampling step in model units

    info: Dict[str, Any] = {
        "did_cap": bool(did_cap),
        "inlet_pid": int(inlet_pid),
        "n_targets": int(len(target_pids)),
        "resampling_step": float(step),
        "flip_normals": None,
        "vmtk_import_source": diagnostics.get("resolved_vmtk_source"),
    }
    if diagnostics.get("resolved_vmtk_source") == "extension_fallback":
        info["vmtk_fallback_modules"] = diagnostics.get("loaded_probe_modules", [])

    source_ids = vtk.vtkIdList()
    source_ids.InsertNextId(inlet_pid)
    target_ids = vtk.vtkIdList()
    for pid in target_pids:
        target_ids.InsertNextId(int(pid))

    last_err = None
    for flip in (0, 1):
        try:
            cl = vtkvmtk_mod.vtkvmtkPolyDataCenterlines()
            cl.SetInputData(capped)
            cl.SetSourceSeedIds(source_ids)
            cl.SetTargetSeedIds(target_ids)
            cl.SetRadiusArrayName("MaximumInscribedSphereRadius")
            cl.SetCostFunction("1/R")
            cl.SetFlipNormals(int(flip))
            cl.SetAppendEndPointsToCenterlines(1)
            cl.SetCenterlineResampling(1)
            cl.SetResamplingStepLength(float(step))
            cl.Update()

            out = cl.GetOutput()
            if out is None or out.GetNumberOfPoints() < 2 or out.GetNumberOfCells() < 1:
                raise RuntimeError("vtkvmtkPolyDataCenterlines returned empty output.")

            out_clean = clean_centerlines_preserve_lines(out)
            if out_clean.GetNumberOfPoints() < 2 or out_clean.GetNumberOfCells() < 1:
                raise RuntimeError("Centerlines became empty after cleaning.")

            info["flip_normals"] = int(flip)
            info["n_points"] = int(out_clean.GetNumberOfPoints())
            info["n_cells"] = int(out_clean.GetNumberOfCells())
            return out_clean, info
        except Exception as e:
            last_err = e
            warnings.append(f"W_VMTK_CENTERLINES_FAIL_FLIP{flip}: {e}")

    raise RuntimeError(f"Centerline extraction failed for all FlipNormals attempts. Last error: {last_err}")


# -----------------------------
# Centerline graph / topology
# -----------------------------
def build_graph_from_polyline_centerlines(cl: "vtkPolyData") -> Tuple[Dict[int, Dict[int, float]], np.ndarray, List[int]]:
    """
    Build weighted undirected adjacency from line cells.

    Returns: (adjacency, points_numpy, line_cell_ids_used)
    """
    pts = get_points_numpy(cl)
    adjacency: Dict[int, Dict[int, float]] = {}
    used_cells: List[int] = []

    for ci in range(cl.GetNumberOfCells()):
        cell = cl.GetCell(ci)
        if cell is None:
            continue
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
            continue
        nids = cell.GetNumberOfPoints()
        if nids < 2:
            continue
        used_cells.append(ci)
        ids = [int(cell.GetPointId(k)) for k in range(nids)]
        for a, b in zip(ids[:-1], ids[1:]):
            if a == b:
                continue
            w = float(np.linalg.norm(pts[a] - pts[b]))
            if w <= 0:
                continue
            adjacency.setdefault(a, {})
            adjacency.setdefault(b, {})
            if (b not in adjacency[a]) or (w < adjacency[a][b]):
                adjacency[a][b] = w
                adjacency[b][a] = w

    return adjacency, pts, used_cells


def connected_component_nodes(adjacency: Dict[int, Dict[int, float]], start: int) -> set[int]:
    if start not in adjacency:
        return set()
    seen: set[int] = set()
    stack = [int(start)]
    while stack:
        node = int(stack.pop())
        if node in seen:
            continue
        seen.add(node)
        for nei in adjacency.get(node, {}):
            if nei not in seen:
                stack.append(int(nei))
    return seen


def induced_subgraph(adjacency: Dict[int, Dict[int, float]], nodes: set[int]) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {}
    for node in nodes:
        nbrs = {int(nei): float(w) for nei, w in adjacency.get(node, {}).items() if int(nei) in nodes}
        if nbrs:
            out[int(node)] = nbrs
    return out


def count_graph_connected_components(adjacency: Dict[int, Dict[int, float]]) -> int:
    remaining = set(int(n) for n in adjacency.keys())
    n_components = 0
    while remaining:
        seed = next(iter(remaining))
        comp = connected_component_nodes(adjacency, seed)
        if not comp:
            remaining.remove(seed)
            continue
        remaining.difference_update(comp)
        n_components += 1
    return int(n_components)


def edge_key(a: int, b: int) -> Tuple[int, int]:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def path_edge_keys(path: List[int]) -> set[Tuple[int, int]]:
    return {edge_key(a, b) for a, b in zip(path[:-1], path[1:])}


def build_branch_chains_from_graph(adjacency: Dict[int, Dict[int, float]]) -> List[List[int]]:
    """
    Decompose the centerline graph into branch-preserving polylines between
    branchpoints/endpoints. This removes duplicated source-target path overlap
    while keeping clinically meaningful side branches present in the scaffold.
    """
    if not adjacency:
        return []

    deg = node_degrees(adjacency)
    key_nodes = {int(n) for n, d in deg.items() if d != 2}
    visited_edges: set[Tuple[int, int]] = set()
    chains: List[List[int]] = []

    def walk(start: int, nxt: int) -> List[int]:
        path = [int(start), int(nxt)]
        visited_edges.add(edge_key(start, nxt))
        prev = int(start)
        cur = int(nxt)
        while deg.get(cur, 0) == 2 and cur not in key_nodes:
            candidates = [int(v) for v in sorted(adjacency.get(cur, {}).keys()) if int(v) != prev]
            if not candidates:
                break
            candidate = int(candidates[0])
            ek = edge_key(cur, candidate)
            if ek in visited_edges:
                break
            visited_edges.add(ek)
            path.append(candidate)
            prev, cur = cur, candidate
        return path

    for start in sorted(key_nodes):
        for nxt in sorted(adjacency.get(start, {}).keys()):
            ek = edge_key(start, int(nxt))
            if ek in visited_edges:
                continue
            chains.append(walk(int(start), int(nxt)))

    for start in sorted(adjacency.keys()):
        for nxt in sorted(adjacency.get(start, {}).keys()):
            ek = edge_key(int(start), int(nxt))
            if ek in visited_edges:
                continue
            path = walk(int(start), int(nxt))
            prev = path[-2]
            cur = path[-1]
            while True:
                candidates = [int(v) for v in sorted(adjacency.get(cur, {}).keys()) if int(v) != prev]
                if not candidates:
                    break
                candidate = int(candidates[0])
                ek2 = edge_key(cur, candidate)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                path.append(candidate)
                prev, cur = cur, candidate
            chains.append(path)

    return [chain for chain in chains if len(chain) >= 2]


def node_degrees(adjacency: Dict[int, Dict[int, float]]) -> Dict[int, int]:
    return {n: len(nei) for n, nei in adjacency.items()}


def dijkstra(adjacency: Dict[int, Dict[int, float]], start: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    import heapq

    dist: Dict[int, float] = {start: 0.0}
    prev: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, start)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adjacency.get(u, {}).items():
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def path_to_root(prev: Dict[int, int], root: int, node: int) -> List[int]:
    if node == root:
        return [root]
    if node not in prev:
        return []
    path = [node]
    cur = node
    seen = set([cur])
    while cur != root:
        if cur not in prev:
            return []
        cur = prev[cur]
        if cur in seen:
            return []
        seen.add(cur)
        path.append(cur)
    path.reverse()
    return path


def deepest_common_node(path_a: List[int], path_b: List[int], dist: Dict[int, float]) -> Optional[int]:
    if not path_a or not path_b:
        return None
    set_a = set(path_a)
    common = [n for n in path_b if n in set_a]
    if not common:
        return None
    common.sort(key=lambda n: dist.get(n, -1.0))
    return common[-1]


def nearest_node_to_point(nodes: List[int], pts: np.ndarray, p: np.ndarray) -> Tuple[int, float]:
    if not nodes:
        return -1, float("inf")
    pp = np.asarray(p, dtype=float).reshape(3)
    sub = pts[nodes] - pp[None, :]
    d2 = np.sum(sub * sub, axis=1)
    k = int(np.argmin(d2))
    return int(nodes[k]), float(math.sqrt(float(d2[k])))


def pick_inlet_node_from_endpoints(
    endpoints: List[int],
    pts: np.ndarray,
    inlet_center: np.ndarray,
    inlet_center_conf: float,
    warnings: List[str],
) -> Tuple[int, float]:
    if not endpoints:
        warnings.append("W_INLET_NO_ENDPOINTS: centerline graph has no endpoints.")
        return -1, 0.0
    inlet_node, dist_to_center = nearest_node_to_point(endpoints, pts, inlet_center)
    # confidence decreases if inlet endpoint is far from inlet center
    scale = max(10.0, 0.75 * float(np.median([np.linalg.norm(pts[e] - pts[inlet_node]) for e in endpoints]) + 1e-6))
    conf = float(clamp(0.35 + 0.65 * inlet_center_conf - 0.25 * (dist_to_center / (scale + EPS)), 0.0, 1.0))
    if dist_to_center > 0.5 * scale:
        warnings.append(f"W_INLET_ENDPOINT_FAR: inlet endpoint is far from inlet termination center (dist={dist_to_center:.3f}).")
    return inlet_node, conf


def choose_centerline_endpoint_for_termination(
    endpoints: List[int],
    pts: np.ndarray,
    termination: Optional[TerminationLoop],
    exclude: Optional[set[int]] = None,
) -> Tuple[Optional[int], float]:
    if termination is None:
        return None, 0.0
    exclude = exclude or set()
    candidates = [int(ep) for ep in endpoints if int(ep) not in exclude]
    if not candidates:
        return None, 0.0

    center = np.asarray(termination.center, dtype=float).reshape(3)
    dists = np.array([np.linalg.norm(pts[ep] - center) for ep in candidates], dtype=float)
    best_idx = int(np.argmin(dists))
    best_ep = int(candidates[best_idx])
    best_dist = float(dists[best_idx])
    diameter_eq = max(float(termination.diameter_eq), 6.0)
    conf = float(clamp(1.0 - best_dist / (1.25 * diameter_eq + EPS), 0.0, 1.0))
    return best_ep, conf


def choose_distal_iliac_termination_pair(
    terms: List[TerminationLoop],
    inlet_term: Optional[TerminationLoop],
    axis_si: np.ndarray,
    warnings: List[str],
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Use distal surface terminations as the primary iliac clue.

    Strategy:
    - exclude the inlet termination,
    - keep a distal-side pool along the superior/inferior axis,
    - within that pool choose the pair with the strongest combination of size,
      lateral separation, and distal position.
    """
    if len(terms) < 3:
        return None, None, 0.0

    centers = np.array([np.asarray(t.center, dtype=float) for t in terms], dtype=float)
    diameters = np.array([max(float(t.diameter_eq), 0.0) for t in terms], dtype=float)
    axis = unit(axis_si)
    if np.linalg.norm(axis) < EPS:
        A, _, _ = pca_axes(centers)
        axis = unit(A[:, 0])
    if np.linalg.norm(axis) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    inlet_idx: Optional[int] = None
    if inlet_term is not None:
        for i, term in enumerate(terms):
            if term is inlet_term:
                inlet_idx = i
                break
        if inlet_idx is None:
            inlet_center = np.asarray(inlet_term.center, dtype=float).reshape(3)
            d2 = np.sum((centers - inlet_center[None, :]) ** 2, axis=1)
            inlet_idx = int(np.argmin(d2))

    other_indices = [i for i in range(len(terms)) if i != inlet_idx]
    if len(other_indices) < 2:
        return None, None, 0.0

    centroid = np.mean(centers, axis=0)
    proj = (centers - centroid[None, :]) @ axis

    other_proj = proj[other_indices]
    proj_min = float(np.min(other_proj))
    proj_max = float(np.max(other_proj))
    proj_span = max(proj_max - proj_min, EPS)

    distal_cutoff = proj_min + 0.45 * proj_span
    distal_pool = [i for i in other_indices if proj[i] <= distal_cutoff]
    if len(distal_pool) < 2:
        warnings.append("W_BIF_TERM_DISTAL_POOL_SMALL: distal-side termination pool was small; using most distal terminations available.")
        distal_pool = sorted(other_indices, key=lambda i: proj[i])[: min(6, len(other_indices))]

    ranked_pool = sorted(
        distal_pool,
        key=lambda i: (diameters[i], -proj[i]),
        reverse=True,
    )[: min(6, len(distal_pool))]
    if len(ranked_pool) < 2:
        return None, None, 0.0

    pool_centers = centers[ranked_pool, :]
    bbox_diag = float(np.linalg.norm(np.max(pool_centers, axis=0) - np.min(pool_centers, axis=0)))
    bbox_diag = max(bbox_diag, EPS)

    pool_diams = diameters[ranked_pool]
    diam_min = float(np.min(pool_diams))
    diam_span = max(float(np.max(pool_diams)) - diam_min, EPS)

    best_pair: Optional[Tuple[int, int]] = None
    best_metrics: Optional[Tuple[float, float, float, float]] = None
    best_score = -1e18

    for i in range(len(ranked_pool)):
        for j in range(i + 1, len(ranked_pool)):
            ia = ranked_pool[i]
            ib = ranked_pool[j]

            diam_a = float(diameters[ia])
            diam_b = float(diameters[ib])
            mean_diam_norm = 0.5 * ((diam_a - diam_min) / diam_span + (diam_b - diam_min) / diam_span)
            diam_balance = 1.0 - abs(diam_a - diam_b) / (diam_a + diam_b + EPS)

            distal_a = (proj_max - float(proj[ia])) / proj_span
            distal_b = (proj_max - float(proj[ib])) / proj_span
            distalness = 0.5 * (distal_a + distal_b)

            dvec = centers[ia] - centers[ib]
            lateral = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
            lateral_norm = lateral / bbox_diag

            score = 1.40 * mean_diam_norm + 2.20 * lateral_norm + 0.85 * distalness + 0.55 * diam_balance
            if score > best_score:
                best_score = score
                best_pair = (ia, ib)
                best_metrics = (mean_diam_norm, diam_balance, distalness, lateral_norm)

    if best_pair is None or best_metrics is None:
        return None, None, 0.0

    mean_diam_norm, diam_balance, distalness, lateral_norm = best_metrics
    conf = float(
        clamp(
            0.20
            + 0.25 * mean_diam_norm
            + 0.20 * diam_balance
            + 0.20 * distalness
            + 0.30 * clamp(1.75 * lateral_norm, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    if conf < 0.55:
        warnings.append(
            "W_BIF_TERM_PAIR_LOWCONF: distal termination pair was weakly separated; proceeding but keeping graph fallback available."
        )
    return int(best_pair[0]), int(best_pair[1]), conf


def infer_bifurcation_from_endpoint_pair(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    inlet_node: int,
    endpoint_a: int,
    endpoint_b: int,
    dist_from_inlet: Dict[int, float],
    prev: Dict[int, int],
) -> Tuple[Optional[int], float]:
    path_a = path_to_root(prev, inlet_node, endpoint_a)
    path_b = path_to_root(prev, inlet_node, endpoint_b)
    if not path_a or not path_b:
        return None, 0.0

    bif_node = deepest_common_node(path_a, path_b, dist_from_inlet)
    if bif_node is None or bif_node == inlet_node:
        return None, 0.0

    depth = float(dist_from_inlet.get(bif_node, 0.0))
    dist_a = float(dist_from_inlet.get(endpoint_a, 0.0))
    dist_b = float(dist_from_inlet.get(endpoint_b, 0.0))
    len_a = dist_a - depth
    len_b = dist_b - depth
    if len_a <= EPS or len_b <= EPS:
        return None, 0.0

    deg = node_degrees(adjacency)
    depth_norm = clamp(depth / (max(min(dist_a, dist_b), EPS)), 0.0, 1.0)
    symmetry = 1.0 - abs(len_a - len_b) / (len_a + len_b + EPS)
    va = unit(pts[endpoint_a] - pts[bif_node])
    vb = unit(pts[endpoint_b] - pts[bif_node])
    divergence = float(clamp((1.0 - float(np.dot(va, vb))) / 2.0, 0.0, 1.0))
    branchlike = float(clamp((deg.get(bif_node, 0) - 2) / 2.0, 0.0, 1.0))

    conf = float(
        clamp(
            0.15
            + 0.25 * depth_norm
            + 0.20 * symmetry
            + 0.20 * divergence
            + 0.20 * branchlike,
            0.0,
            1.0,
        )
    )
    return int(bif_node), conf


def choose_iliac_endpoints_and_bifurcation(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    inlet_node: int,
    axis_si: np.ndarray,
    warnings: List[str],
    terminations: Optional[List[TerminationLoop]] = None,
    inlet_term: Optional[TerminationLoop] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], float, Dict[int, float], Dict[int, int]]:
    """
    Identify the distal aortic bifurcation into the main iliac systems using centerline topology.

    Returns: (bif_node, ep_a, ep_b, confidence, dist_from_inlet, prev)
    """
    dist, prev = dijkstra(adjacency, inlet_node)
    deg = node_degrees(adjacency)
    endpoints = [n for n, d in deg.items() if d == 1 and n != inlet_node and n in dist]
    if len(endpoints) < 2:
        warnings.append("W_BIF_NOT_ENOUGH_ENDPOINTS: need >=2 distal endpoints for iliac pair inference.")
        return None, None, None, 0.0, dist, prev

    if terminations:
        term_a_idx, term_b_idx, term_pair_conf = choose_distal_iliac_termination_pair(
            terminations,
            inlet_term,
            axis_si,
            warnings,
        )
        if term_a_idx is not None and term_b_idx is not None:
            ep_a, ep_a_conf = choose_centerline_endpoint_for_termination(endpoints, pts, terminations[term_a_idx])
            ep_b, ep_b_conf = choose_centerline_endpoint_for_termination(
                endpoints,
                pts,
                terminations[term_b_idx],
                exclude=({int(ep_a)} if ep_a is not None else set()),
            )
            if ep_a is not None and ep_b is not None and ep_a != ep_b:
                bif_term, bif_term_conf = infer_bifurcation_from_endpoint_pair(
                    adjacency,
                    pts,
                    inlet_node,
                    int(ep_a),
                    int(ep_b),
                    dist,
                    prev,
                )
                if bif_term is not None:
                    conf = float(
                        clamp(
                            0.15
                            + 0.25 * term_pair_conf
                            + 0.25 * min(ep_a_conf, ep_b_conf)
                            + 0.35 * bif_term_conf,
                            0.0,
                            1.0,
                        )
                    )
                    if conf < 0.60:
                        warnings.append(
                            f"W_BIF_TERM_ANCHORED_LOWCONF: termination-anchored bifurcation confidence={conf:.3f}."
                        )
                    return int(bif_term), int(ep_a), int(ep_b), conf, dist, prev
                warnings.append(
                    "W_BIF_TERM_SHARED_ANCESTOR_FAILED: distal termination pair mapped to centerline endpoints, but their shared bifurcation was not robust."
                )
            else:
                warnings.append(
                    "W_BIF_TERM_ENDPOINT_MAP_FAILED: distal surface terminations could not be mapped cleanly to distinct centerline endpoints."
                )

    # Candidate endpoints: farthest K by distance from inlet
    endpoints_sorted = sorted(endpoints, key=lambda n: dist.get(n, -1.0), reverse=True)
    K = min(12, len(endpoints_sorted))
    candidates = endpoints_sorted[:K]

    # Scale for lateral separation normalization
    bbox_min = np.min(pts[list(adjacency.keys())], axis=0)
    bbox_max = np.max(pts[list(adjacency.keys())], axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    max_dist = float(max(dist.get(n, 0.0) for n in candidates)) if candidates else 1.0
    axis = unit(axis_si) if np.linalg.norm(axis_si) > EPS else unit(bbox_max - bbox_min)
    if np.linalg.norm(axis) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    best = None
    best_score = -1e18

    # Precompute paths to root for candidates
    paths = {n: path_to_root(prev, inlet_node, n) for n in candidates}

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            pa = paths.get(a, [])
            pb = paths.get(b, [])
            if not pa or not pb:
                continue
            lca = deepest_common_node(pa, pb, dist)
            if lca is None:
                continue
            # Require branching-like node
            if deg.get(lca, 0) < 3:
                continue

            depth = float(dist.get(lca, 0.0))
            da = float(dist.get(a, 0.0))
            db = float(dist.get(b, 0.0))
            len_a = da - depth
            len_b = db - depth
            if len_a <= 0 or len_b <= 0:
                continue

            symmetry = 1.0 - abs(len_a - len_b) / (len_a + len_b + EPS)

            # Lateral separation in plane orthogonal to axis_si
            dvec = pts[a] - pts[b]
            lateral = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
            lateral_norm = lateral / (diag + EPS)

            depth_norm = depth / (max_dist + EPS)
            min_tail_norm = min(len_a, len_b) / (max_dist + EPS)

            # Prefer distal, bilateral splits, but keep this advisory so real trees
            # with extra branches do not get rejected outright.
            proximal_penalty = clamp((0.45 - depth_norm) / 0.45, 0.0, 1.0)

            score = (
                2.0 * depth_norm
                + 1.25 * symmetry
                + 1.75 * lateral_norm
                + 0.5 * min_tail_norm
                - 0.9 * proximal_penalty
            )

            if score > best_score:
                best_score = score
                best = (lca, a, b, depth_norm, symmetry, lateral_norm)

    if best is None:
        warnings.append("W_BIF_PAIR_SEARCH_FAILED: could not robustly identify iliac pair; falling back to two farthest endpoints.")
        # Fallback: use two farthest endpoints
        a = endpoints_sorted[0]
        b = endpoints_sorted[1]
        pa = path_to_root(prev, inlet_node, a)
        pb = path_to_root(prev, inlet_node, b)
        lca = deepest_common_node(pa, pb, dist)
        if lca is None:
            warnings.append("W_BIF_FALLBACK_FAILED: no common ancestor found; bifurcation unresolved.")
            return None, a, b, 0.0, dist, prev
        conf = 0.20
        return int(lca), int(a), int(b), conf, dist, prev

    lca, a, b, depth_norm, symmetry, lateral_norm = best
    # Confidence: combine symmetry, depth, and lateral separation
    conf = float(clamp(0.25 + 0.35 * depth_norm + 0.25 * symmetry + 0.30 * clamp(lateral_norm * 2.0, 0.0, 1.0), 0.0, 1.0))
    if conf < 0.60:
        warnings.append(f"W_BIF_LOWER_CONF: bifurcation confidence={conf:.3f} (sym={symmetry:.3f}, depthN={depth_norm:.3f}, latN={lateral_norm:.3f}).")
    return int(lca), int(a), int(b), conf, dist, prev


def first_branchpoint_on_path(path: List[int], deg: Dict[int, int], start_index: int = 1) -> Optional[int]:
    """
    Return the first node on path (after path[start_index-1]) that is a branchpoint (deg>=3).
    """
    for k in range(start_index, len(path)):
        n = path[k]
        if deg.get(n, 0) >= 3:
            return int(n)
    return None


# -----------------------------
# Orientation frame construction
# -----------------------------
def project_vector_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(3)
    nn = unit(normal)
    if np.linalg.norm(nn) < EPS:
        return vv.astype(float)
    return (vv - np.dot(vv, nn) * nn).astype(float)


def unit_xy(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(3).copy()
    vv[2] = 0.0
    return unit(vv)


def estimate_superior_axis(inlet_pt: np.ndarray, bif_pt: np.ndarray, all_pts: np.ndarray, warnings: List[str]) -> np.ndarray:
    ez = unit(inlet_pt - bif_pt)
    if np.linalg.norm(ez) < EPS:
        warnings.append("W_FRAME_EZ_DEGEN: inlet/bif are coincident; using PCA axis as z.")
        A, _, _ = pca_axes(all_pts)
        ez = unit(A[:, 0])
        if np.linalg.norm(ez) < EPS:
            ez = np.array([0.0, 0.0, 1.0], dtype=float)
    return ez.astype(float)


def estimate_horizontal_axis_from_iliacs(
    iliac_ep_a_pt: np.ndarray,
    iliac_ep_b_pt: np.ndarray,
    ez: np.ndarray,
    all_pts: np.ndarray,
    warnings: List[str],
) -> np.ndarray:
    a = np.asarray(iliac_ep_a_pt, dtype=float).reshape(3)
    b = np.asarray(iliac_ep_b_pt, dtype=float).reshape(3)
    if a[0] < b[0]:
        left = a
        right = b
    else:
        left = b
        right = a

    ex0 = project_vector_to_plane(right - left, ez)
    ex = unit(ex0)
    if np.linalg.norm(ex) < EPS:
        warnings.append("W_FRAME_EX_DEGEN: iliac endpoints nearly colinear with z; using PCA second axis as x.")
        A, _, _ = pca_axes(all_pts)
        ex = unit(project_vector_to_plane(A[:, 1], ez))
        if np.linalg.norm(ex) < EPS:
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(tmp, ez)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            ex = unit(project_vector_to_plane(tmp, ez))
    return ex.astype(float)


def complete_canonical_frame(
    ez: np.ndarray,
    horizontal_hint: np.ndarray,
    all_pts: np.ndarray,
    warnings: List[str],
) -> Tuple[np.ndarray, float]:
    ez_u = unit(ez)
    ex = unit(project_vector_to_plane(horizontal_hint, ez_u))
    if np.linalg.norm(ex) < EPS:
        warnings.append("W_FRAME_EX_HINT_DEGEN: horizontal hint collapsed after projection; using PCA second axis.")
        A, _, _ = pca_axes(all_pts)
        ex = unit(project_vector_to_plane(A[:, 1], ez_u))
        if np.linalg.norm(ex) < EPS:
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(tmp, ez_u)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            ex = unit(project_vector_to_plane(tmp, ez_u))

    ey = unit(np.cross(ez_u, ex))
    if np.linalg.norm(ey) < EPS:
        warnings.append("W_FRAME_EY_DEGEN: cross(z,x) nearly zero; repairing basis.")
        A, _, _ = pca_axes(all_pts)
        ey = unit(project_vector_to_plane(A[:, 2], ez_u))
        if np.linalg.norm(ey) < EPS:
            ey = unit(np.cross(ez_u, np.array([1.0, 0.0, 0.0], dtype=float)))

    ex = unit(np.cross(ey, ez_u))
    ey = unit(np.cross(ez_u, ex))

    R = np.vstack([ex, ey, ez_u]).astype(float)
    ortho_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
    conf = float(clamp(1.0 - ortho_err, 0.0, 1.0))
    if conf < 0.85:
        warnings.append(f"W_FRAME_ORTHONORMALITY: basis orthonormality confidence={conf:.3f} (err={ortho_err:.3e}).")
    return R, conf


def build_canonical_transform(
    inlet_pt: np.ndarray,
    bif_pt: np.ndarray,
    iliac_ep_a_pt: np.ndarray,
    iliac_ep_b_pt: np.ndarray,
    all_pts: np.ndarray,
    warnings: List[str],
    horizontal_hint: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Construct a deterministic canonical frame by explicitly separating:
    - stable superior/inferior axis estimation (z), and
    - in-plane horizontal axis completion (x/y).

    Returns: (R, origin, orthonormality_confidence)
    Where p_canonical = R @ (p - origin)
    """
    origin = np.asarray(bif_pt, dtype=float).reshape(3)
    ez = estimate_superior_axis(inlet_pt, bif_pt, all_pts, warnings)
    ex_hint = (
        estimate_horizontal_axis_from_iliacs(iliac_ep_a_pt, iliac_ep_b_pt, ez, all_pts, warnings)
        if horizontal_hint is None
        else np.asarray(horizontal_hint, dtype=float).reshape(3)
    )
    R, conf = complete_canonical_frame(ez, ex_hint, all_pts, warnings)
    return R, origin, conf


def apply_transform_points(pts: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """
    p' = R @ (p - origin)
    """
    P = np.asarray(pts, dtype=float)
    o = np.asarray(origin, dtype=float).reshape(3)
    return ((R @ (P - o).T).T).astype(float)


def build_transform_matrix4x4(R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    rot = np.asarray(R, dtype=float).reshape(3, 3)
    o = np.asarray(origin, dtype=float).reshape(3)
    t = -rot @ o
    M = np.eye(4, dtype=float)
    M[0:3, 0:3] = rot
    M[0:3, 3] = t
    return M


def apply_transform_to_polydata(pd: "vtkPolyData", R: np.ndarray, origin: np.ndarray) -> "vtkPolyData":
    M = build_transform_matrix4x4(R, origin)
    vtk_mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            vtk_mat.SetElement(r, c, float(M[r, c]))

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_mat)

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetTransform(transform)
    filt.SetInputData(pd)
    filt.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(filt.GetOutput())
    out.BuildLinks()
    return out


# -----------------------------
# Renal identification
# -----------------------------
def _orient_chain_from_scaffold(chain: List[int], scaffold_set: set[int]) -> Optional[List[int]]:
    if not chain or len(chain) < 2:
        return None
    start_in = int(chain[0]) in scaffold_set
    end_in = int(chain[-1]) in scaffold_set
    if start_in == end_in:
        return None
    if start_in:
        return [int(n) for n in chain]
    return [int(n) for n in reversed(chain)]


def _collect_component_nodes_excluding_scaffold(
    adjacency: Dict[int, Dict[int, float]],
    start_node: int,
    scaffold_set: set[int],
) -> set[int]:
    start = int(start_node)
    if start in scaffold_set:
        return set()

    stack = [start]
    component: set[int] = set()
    while stack:
        node = int(stack.pop())
        if node in component or node in scaffold_set:
            continue
        component.add(node)
        for nbr in adjacency.get(node, {}).keys():
            nbr_i = int(nbr)
            if nbr_i not in component and nbr_i not in scaffold_set:
                stack.append(nbr_i)
    return component


def _path_segment_from_takeoff(prev: Dict[int, int], inlet_node: int, takeoff: int, endpoint: int) -> List[int]:
    path = path_to_root(prev, inlet_node, endpoint)
    if not path or takeoff not in path:
        return []
    return [int(n) for n in path[path.index(takeoff):]]


def _choose_representative_endpoint_for_component(
    component_nodes: set[int],
    takeoff: int,
    pts_c: np.ndarray,
    deg: Dict[int, int],
    dist: Dict[int, float],
    prev: Dict[int, int],
    inlet_node: int,
    trunk_len: float,
) -> Tuple[Optional[Dict[str, Any]], List[int]]:
    endpoints = sorted(int(n) for n in component_nodes if deg.get(int(n), 0) == 1)
    takeoff_dist = float(dist.get(takeoff, float("nan")))
    scale_xy = max(1.0, 0.12 * trunk_len)
    reps: List[Dict[str, Any]] = []

    for ep in endpoints:
        seg = _path_segment_from_takeoff(prev, inlet_node, takeoff, ep)
        if len(seg) < 2:
            continue

        rep_vec = pts_c[ep] - pts_c[takeoff]
        rep_vhat = unit(rep_vec)
        rep_hdir = unit_xy(rep_vec)
        if np.linalg.norm(rep_hdir) < EPS:
            continue

        local_idx = min(2, len(seg) - 1)
        local_vec = pts_c[seg[local_idx]] - pts_c[takeoff]
        if np.linalg.norm(local_vec) < EPS:
            local_vec = rep_vec.copy()
        local_vhat = unit(local_vec)
        local_hdir = unit_xy(local_vec)
        if np.linalg.norm(local_hdir) < EPS:
            local_hdir = rep_hdir.copy()

        branch_len = float(dist.get(ep, float("nan")) - takeoff_dist)
        if not math.isfinite(branch_len) or branch_len <= 0.0:
            continue

        rep_horizontal = float(np.linalg.norm(rep_vhat[:2]))
        local_horizontal = float(np.linalg.norm(local_vhat[:2]))
        local_vertical = abs(float(local_vhat[2]))
        reach_xy = float(np.linalg.norm(rep_vec[:2]))
        len_norm = branch_len / (trunk_len + EPS)

        score_local_horizontal = clamp((local_horizontal - 0.20) / 0.70, 0.0, 1.0)
        score_rep_horizontal = clamp((rep_horizontal - 0.35) / 0.60, 0.0, 1.0)
        score_reach = clamp(reach_xy / scale_xy, 0.0, 1.0)
        score_vertical = clamp((0.88 - local_vertical) / 0.60, 0.0, 1.0)
        score_len = clamp((len_norm - 0.04) / 0.18, 0.0, 1.0) * clamp((0.95 - len_norm) / 0.45, 0.0, 1.0)
        rep_score = float(
            clamp(
                0.34 * score_local_horizontal
                + 0.26 * score_rep_horizontal
                + 0.18 * score_reach
                + 0.12 * score_vertical
                + 0.10 * score_len,
                0.0,
                1.0,
            )
        )

        reps.append(
            {
                "ep": int(ep),
                "path": seg,
                "branch_len": float(branch_len),
                "reach_xy": float(reach_xy),
                "local_vec": local_vec.astype(float),
                "local_hdir": local_hdir.astype(float),
                "local_horizontal": float(local_horizontal),
                "local_vertical": float(local_vertical),
                "rep_vec": rep_vec.astype(float),
                "rep_hdir": rep_hdir.astype(float),
                "rep_horizontal": float(rep_horizontal),
                "representative_score": float(rep_score),
            }
        )

    reps.sort(key=lambda item: float(item["representative_score"]), reverse=True)
    return (reps[0] if reps else None), endpoints


def _should_absorb_chain_into_trunk_scaffold(
    chain_nodes: List[int],
    pts_c: np.ndarray,
    takeoff_dist: float,
    trunk_len: float,
) -> bool:
    if len(chain_nodes) < 2 or trunk_len <= EPS or not math.isfinite(takeoff_dist):
        return False
    if takeoff_dist >= 0.82 * trunk_len:
        return False

    chain_pts = pts_c[np.asarray(chain_nodes, dtype=int)]
    disp = chain_pts[-1] - chain_pts[0]
    s = compute_abscissa(chain_pts)
    chain_len = float(s[-1]) if s.size else float(np.linalg.norm(disp))
    if chain_len <= EPS:
        return False

    horiz_disp = float(np.linalg.norm(disp[:2]))
    vertical_disp = abs(float(disp[2]))
    verticality = vertical_disp / (chain_len + EPS)
    radial = np.linalg.norm(chain_pts[:, :2], axis=1) if chain_pts.size else np.zeros(0, dtype=float)
    max_radius = float(np.max(radial)) if radial.size else 0.0
    end_radius = float(np.linalg.norm(chain_pts[-1, :2]))

    horiz_limit = max(1.00, 0.10 * trunk_len)
    radius_limit = max(1.35, 0.14 * trunk_len)
    return bool(
        horiz_disp <= horiz_limit
        and max_radius <= radius_limit
        and end_radius <= max(1.50, 0.16 * trunk_len)
        and verticality >= 0.68
    )


def _expand_trunk_scaffold_for_renal_scan(
    adjacency: Dict[int, Dict[int, float]],
    pts_c: np.ndarray,
    trunk_path: List[int],
    dist: Dict[int, float],
    bif_node: int,
) -> Tuple[set[int], List[int], List[List[int]]]:
    trunk_len = float(dist.get(bif_node, 0.0))
    scaffold_set: set[int] = {int(n) for n in trunk_path}
    chains = build_branch_chains_from_graph(adjacency)
    absorbed_chain_ids: set[int] = set()

    changed = True
    while changed:
        changed = False
        for chain_id, chain in enumerate(chains):
            oriented = _orient_chain_from_scaffold(chain, scaffold_set)
            if oriented is None:
                continue
            takeoff = int(oriented[0])
            takeoff_dist = float(dist.get(takeoff, float("nan")))
            if not _should_absorb_chain_into_trunk_scaffold(oriented, pts_c, takeoff_dist, trunk_len):
                continue
            new_nodes = [int(n) for n in oriented[1:] if int(n) not in scaffold_set]
            if not new_nodes:
                continue
            scaffold_set.update(new_nodes)
            absorbed_chain_ids.add(int(chain_id))
            changed = True

    return scaffold_set, sorted(int(i) for i in absorbed_chain_ids), chains


def _nearest_node_on_path(path_nodes: List[int], pts_c: np.ndarray, query_node: int) -> int:
    if not path_nodes:
        return int(query_node)
    q = pts_c[int(query_node)]
    best = min((int(n) for n in path_nodes), key=lambda n: float(np.linalg.norm(pts_c[int(n)] - q)))
    return int(best)


def discover_renal_branch_candidates(
    adjacency: Dict[int, Dict[int, float]],
    pts_c: np.ndarray,
    inlet_node: int,
    bif_node: int,
    trunk_set: set,
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    warnings: List[str],
) -> Dict[str, Any]:
    """
    Frame-robust renal candidate discovery using side branches around an
    expanded aortic scaffold. Z is assumed approximately correct, but no
    assumption is made about whether the bilateral renal pair already aligns
    with X or Y in the provisional frame.
    """
    deg = node_degrees(adjacency)
    trunk_len = float(dist.get(bif_node, 0.0))
    if trunk_len <= 0:
        warnings.append("W_RENAL_TRUNKLEN: trunk length invalid; renal detection may fail.")
        trunk_len = 1.0

    trunk_path = [int(n) for n in path_to_root(prev, inlet_node, bif_node)]
    scaffold_set, scaffold_chain_ids, chains = _expand_trunk_scaffold_for_renal_scan(adjacency, pts_c, trunk_path, dist, bif_node)

    candidates: List[Dict[str, Any]] = []
    for chain_id, chain in enumerate(chains):
        oriented = _orient_chain_from_scaffold(chain, scaffold_set)
        if oriented is None or len(oriented) < 2:
            continue
        takeoff = int(oriented[0])
        takeoff_dist = float(dist.get(takeoff, float("nan")))
        if not math.isfinite(takeoff_dist):
            continue
        trunk_takeoff = _nearest_node_on_path(trunk_path, pts_c, takeoff)
        trunk_takeoff_dist = float(dist.get(trunk_takeoff, takeoff_dist))
        if trunk_takeoff_dist < 0.08 * trunk_len or trunk_takeoff_dist > 0.82 * trunk_len:
            continue
        if takeoff in (inlet_node, bif_node):
            continue

        component_nodes = _collect_component_nodes_excluding_scaffold(adjacency, oriented[1], scaffold_set)
        if not component_nodes:
            continue
        if iliac_ep_a in component_nodes or iliac_ep_b in component_nodes:
            continue

        rep, component_endpoints = _choose_representative_endpoint_for_component(
            component_nodes,
            takeoff,
            pts_c,
            deg,
            dist,
            prev,
            inlet_node,
            trunk_len,
        )
        if rep is None:
            continue

        branch_len = float(rep["branch_len"])
        if branch_len <= 0.02 * trunk_len or branch_len > 0.95 * trunk_len:
            continue

        s_rel = float(trunk_takeoff_dist / (trunk_len + EPS))
        local_horizontal = float(rep["local_horizontal"])
        rep_horizontal = float(rep["rep_horizontal"])
        horizontality = float(clamp(0.55 * rep_horizontal + 0.45 * local_horizontal, 0.0, 1.0))
        local_vertical = float(rep["local_vertical"])
        reach_xy = float(rep["reach_xy"])
        len_norm = branch_len / (trunk_len + EPS)
        score_pos = clamp(1.0 - abs(s_rel - 0.38) / 0.34, 0.0, 1.0)
        score_horiz = clamp((horizontality - 0.35) / 0.55, 0.0, 1.0)
        score_side = clamp((0.84 - local_vertical) / 0.60, 0.0, 1.0)
        score_reach = clamp(reach_xy / max(1.0, 0.12 * trunk_len), 0.0, 1.0)
        score_len = clamp((len_norm - 0.04) / 0.18, 0.0, 1.0) * clamp((0.90 - len_norm) / 0.45, 0.0, 1.0)
        visceral_penalty = 0.15 * clamp((0.18 - s_rel) / 0.18, 0.0, 1.0) * clamp((local_vertical - 0.45) / 0.35, 0.0, 1.0)
        score = float(
            clamp(
                0.28 * score_horiz
                + 0.22 * score_reach
                + 0.18 * score_side
                + 0.17 * score_pos
                + 0.10 * score_len
                + 0.05 * float(rep["representative_score"])
                - visceral_penalty,
                0.0,
                1.0,
            )
        )
        if score <= 0.0:
            continue

        hdir = np.asarray(rep["rep_hdir"], dtype=float)
        if np.linalg.norm(hdir) < EPS:
            hdir = np.asarray(rep["local_hdir"], dtype=float)
        hdir = unit_xy(hdir)
        if np.linalg.norm(hdir) < EPS:
            continue

        candidates.append(
            {
                "ep": int(rep["ep"]),
                "takeoff": int(takeoff),
                "takeoff_dist": float(trunk_takeoff_dist),
                "trunk_takeoff": int(trunk_takeoff),
                "branch_root_dist": float(takeoff_dist),
                "s_rel": float(s_rel),
                "branch_len": float(branch_len),
                "hdir": hdir.astype(float),
                "local_hdir": np.asarray(rep["local_hdir"], dtype=float),
                "local_horizontal": float(local_horizontal),
                "rep_horizontal": float(rep_horizontal),
                "horizontality": float(horizontality),
                "vertical": float(local_vertical),
                "reach_xy": float(reach_xy),
                "representative_score": float(rep["representative_score"]),
                "component_endpoint_count": int(len(component_endpoints)),
                "component_node_count": int(len(component_nodes)),
                "source_chain_id": int(chain_id),
                "source_chain_absorbed_to_scaffold": bool(chain_id in scaffold_chain_ids),
                "rep_point": pts_c[int(rep["ep"])].astype(float),
                "score": float(score),
            }
        )

    pair_candidates: List[Dict[str, Any]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            dt = abs(a["takeoff_dist"] - b["takeoff_dist"])
            dt_norm = dt / (0.45 * trunk_len + EPS)
            takeoff_sim = clamp(1.0 - dt_norm, 0.0, 1.0)

            dl = abs(a["branch_len"] - b["branch_len"]) / (a["branch_len"] + b["branch_len"] + EPS)
            len_sim = 1.0 - dl
            len_sim = clamp(len_sim, 0.0, 1.0)

            dir_opp = float(clamp((1.0 - float(np.dot(a["hdir"], b["hdir"]))) / 2.0, 0.0, 1.0))
            if dir_opp < 0.38:
                continue

            axis_vec = np.asarray(a["rep_point"], dtype=float) - np.asarray(b["rep_point"], dtype=float)
            axis_vec[2] = 0.0
            if np.linalg.norm(axis_vec) < EPS:
                axis_vec = a["hdir"] - b["hdir"]
                axis_vec[2] = 0.0
            axis = unit_xy(axis_vec)
            if np.linalg.norm(axis) < EPS:
                continue

            horiz_pair = float(clamp(0.5 * (a["horizontality"] + b["horizontality"]), 0.0, 1.0))
            reach_balance = float(clamp(min(a["reach_xy"], b["reach_xy"]) / (max(a["reach_xy"], b["reach_xy"]) + EPS), 0.0, 1.0))
            axis_span = float(clamp(np.linalg.norm(axis_vec[:2]) / (a["reach_xy"] + b["reach_xy"] + EPS), 0.0, 1.0))
            geometry_score = float(clamp(0.55 * dir_opp + 0.25 * reach_balance + 0.20 * axis_span, 0.0, 1.0))
            axis_conf = float(clamp(0.45 * geometry_score + 0.35 * horiz_pair + 0.20 * axis_span, 0.0, 1.0))
            score = float(
                clamp(
                    0.38 * geometry_score
                    + 0.23 * takeoff_sim
                    + 0.19 * horiz_pair
                    + 0.10 * len_sim
                    + 0.10 * (0.5 * (a["score"] + b["score"])),
                    0.0,
                    1.0,
                )
            )
            pair_candidates.append(
                {
                    "a": a,
                    "b": b,
                    "axis": axis.astype(float),
                    "score": float(score),
                    "confidence": float(clamp(0.15 + 0.85 * score, 0.0, 1.0)),
                    "axis_confidence": float(axis_conf),
                    "geometry_score": float(geometry_score),
                    "direction_opposition": float(dir_opp),
                    "takeoff_similarity": float(takeoff_sim),
                    "horizontality_score": float(horiz_pair),
                    "length_similarity": float(len_sim),
                    "axis_span_score": float(axis_span),
                }
            )

    candidates.sort(key=lambda d: d["score"], reverse=True)
    pair_candidates.sort(key=lambda d: d["score"], reverse=True)
    best_pair = pair_candidates[0] if pair_candidates else None
    if not candidates:
        warnings.append("W_RENAL_CAND_DISCOVERY_EMPTY: no plausible side-branch renal candidates found on the abdominal scaffold.")

    return {
        "trunk_len": float(trunk_len),
        "trunk_scaffold_node_count": int(len(scaffold_set)),
        "trunk_scaffold_extra_node_count": int(max(0, len(scaffold_set) - len(trunk_set))),
        "trunk_scaffold_chain_count": int(len(scaffold_chain_ids)),
        "candidates": candidates,
        "pair_candidates": pair_candidates,
        "best_pair": best_pair,
        "best_single": candidates[0] if candidates else None,
    }


def collect_visceral_branch_axis(
    pts_c: np.ndarray,
    adjacency: Dict[int, Dict[int, float]],
    inlet_node: int,
    bif_node: int,
    trunk_set: set,
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    excluded_endpoints: Optional[set[int]] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    deg = node_degrees(adjacency)
    endpoints = [n for n, d in deg.items() if d == 1 and n != inlet_node and n in dist]
    excluded = {iliac_ep_a, iliac_ep_b}
    if excluded_endpoints:
        excluded.update(int(x) for x in excluded_endpoints if x is not None)
    endpoints = [e for e in endpoints if e not in excluded]

    trunk_len = float(dist.get(bif_node, 0.0))
    if trunk_len <= 0.0:
        return None, 0.0, 0

    vecs: List[np.ndarray] = []
    weights: List[float] = []
    for ep in endpoints:
        path = path_to_root(prev, inlet_node, ep)
        if not path:
            continue
        common = [n for n in path if n in trunk_set]
        if not common:
            continue
        takeoff = max(common, key=lambda n: dist.get(n, -1.0))
        takeoff_dist = float(dist.get(takeoff, float("nan")))
        ep_dist = float(dist.get(ep, float("nan")))
        if not math.isfinite(takeoff_dist) or not math.isfinite(ep_dist):
            continue
        s_rel = takeoff_dist / (trunk_len + EPS)
        if s_rel < 0.08 or s_rel > 0.72:
            continue

        branch_len = ep_dist - takeoff_dist
        if branch_len <= 0.02 * trunk_len:
            continue

        v = pts_c[ep] - pts_c[takeoff]
        if np.linalg.norm(v) < EPS:
            continue
        vhat = unit(v)
        hdir = unit_xy(v)
        if np.linalg.norm(hdir) < EPS:
            continue
        vertical = abs(float(vhat[2]))
        if vertical > 0.80:
            continue

        weight = clamp((0.78 - vertical) / 0.60, 0.0, 1.0) * clamp((0.72 - s_rel) / 0.55, 0.0, 1.0)
        if weight <= 0.0:
            continue

        vecs.append(hdir[:2].astype(float))
        weights.append(float(weight))

    if not vecs:
        return None, 0.0, 0

    if len(vecs) == 1:
        axis = np.array([vecs[0][0], vecs[0][1], 0.0], dtype=float)
        conf = float(clamp(0.25 * weights[0], 0.0, 1.0))
        return unit_xy(axis), conf, 1

    scatter = np.zeros((2, 2), dtype=float)
    for vec, weight in zip(vecs, weights):
        vv = np.asarray(vec, dtype=float).reshape(2, 1)
        scatter += float(weight) * (vv @ vv.T)

    vals, vecs_eig = np.linalg.eigh(scatter)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs_eig = vecs_eig[:, order]
    axis2 = unit(np.array([vecs_eig[0, 0], vecs_eig[1, 0], 0.0], dtype=float))
    dominance = float((vals[0] - vals[1]) / (vals[0] + vals[1] + EPS))
    support = float(clamp(sum(weights) / max(2.0, float(len(weights))), 0.0, 1.0))
    conf = float(clamp(0.20 + 0.55 * dominance + 0.25 * support, 0.0, 1.0))
    return axis2.astype(float), conf, len(vecs)


def collect_ap_orientation_cues(
    pts_c: np.ndarray,
    adjacency: Dict[int, Dict[int, float]],
    inlet_node: int,
    bif_node: int,
    trunk_set: set,
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    renal_eps: Tuple[Optional[int], Optional[int]],
) -> List[Dict[str, Any]]:
    deg = node_degrees(adjacency)
    endpoints = [n for n, d in deg.items() if d == 1 and n != inlet_node and n in dist]
    exclude = {iliac_ep_a, iliac_ep_b}
    if renal_eps[0] is not None:
        exclude.add(int(renal_eps[0]))
    if renal_eps[1] is not None:
        exclude.add(int(renal_eps[1]))
    endpoints = [e for e in endpoints if e not in exclude]

    trunk_len = float(dist.get(bif_node, 0.0))
    if trunk_len <= 0:
        return []

    cues: List[Dict[str, Any]] = []
    for ep in endpoints:
        path = path_to_root(prev, inlet_node, ep)
        if not path:
            continue
        common = [n for n in path if n in trunk_set]
        if not common:
            continue
        takeoff = max(common, key=lambda n: dist.get(n, -1.0))
        takeoff_dist = float(dist.get(takeoff, float("nan")))
        if not math.isfinite(takeoff_dist):
            continue
        s_rel = takeoff_dist / (trunk_len + EPS)
        if s_rel < 0.10 or s_rel > 0.75:
            continue

        idx_take = path.index(takeoff)
        if idx_take >= len(path) - 2:
            continue
        next_node = path[idx_take + 1]
        v = pts_c[next_node] - pts_c[takeoff]
        if np.linalg.norm(v) < EPS:
            continue
        vhat = unit(v)

        anterior = abs(float(vhat[1]))
        lateral = abs(float(vhat[0]))
        vertical = abs(float(vhat[2]))

        if anterior < 0.45:
            continue
        if anterior < lateral:
            continue
        if vertical > 0.80:
            continue

        weight = float(clamp((anterior - 0.45) / 0.45, 0.0, 1.0))
        sign = 1.0 if float(vhat[1]) >= 0.0 else -1.0
        cues.append(
            {
                "ep": int(ep),
                "takeoff": int(takeoff),
                "weight": float(weight),
                "sign_y": float(sign),
                "vhat": vhat.astype(float),
            }
        )
    return cues


def refine_horizontal_axes_using_branch_anatomy(
    R_provisional: np.ndarray,
    pts_c_provisional: np.ndarray,
    iliac_main_a: List[int],
    iliac_main_b: List[int],
    renal_scan: Dict[str, Any],
    adjacency: Dict[int, Dict[int, float]],
    inlet_node: int,
    bif_node: int,
    trunk_set: set,
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    warnings: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    def iliac_reference_axis() -> Tuple[np.ndarray, float]:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if iliac_main_a and iliac_main_b:
            iliac_vec = pts_c_provisional[iliac_main_a[-1]] - pts_c_provisional[iliac_main_b[-1]]
            ref = unit_xy(iliac_vec)
        if np.linalg.norm(ref) < EPS:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)

        xy_extent = float(np.ptp(pts_c_provisional[:, 0]) + np.ptp(pts_c_provisional[:, 1])) if pts_c_provisional.size else 0.0
        iliac_sep = 0.0
        if iliac_main_a and iliac_main_b:
            iliac_sep = float(np.linalg.norm((pts_c_provisional[iliac_main_a[-1]] - pts_c_provisional[iliac_main_b[-1]])[:2]))
        conf = float(clamp(iliac_sep / (xy_extent + EPS), 0.0, 1.0))
        return ref, conf

    def refine_horizontal_frame_from_renal_pair(
        R_in: np.ndarray,
        renal_pair: Dict[str, Any],
        iliac_ref_axis: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pair_axis = unit_xy(np.asarray(renal_pair.get("axis", np.zeros(3, dtype=float)), dtype=float))
        if np.linalg.norm(pair_axis) < EPS:
            return np.asarray(R_in, dtype=float), {
                "rotation_degrees_about_z": 0.0,
                "axis_confidence": 0.0,
                "geometry_score": 0.0,
                "takeoff_similarity_score": 0.0,
                "horizontality_score": 0.0,
            }

        if np.linalg.norm(iliac_ref_axis) >= EPS and float(np.dot(pair_axis[:2], iliac_ref_axis[:2])) < 0.0:
            pair_axis *= -1.0

        ey_axis = np.array([-pair_axis[1], pair_axis[0], 0.0], dtype=float)
        Q = np.vstack([pair_axis, ey_axis, np.array([0.0, 0.0, 1.0], dtype=float)]).astype(float)
        R_out = (Q @ np.asarray(R_in, dtype=float)).astype(float)
        rotation_deg = float(math.degrees(math.atan2(float(pair_axis[1]), float(pair_axis[0]))))
        return R_out, {
            "rotation_degrees_about_z": float(rotation_deg),
            "axis_confidence": float(clamp(renal_pair.get("axis_confidence", renal_pair.get("confidence", 0.0)), 0.0, 1.0)),
            "geometry_score": float(clamp(renal_pair.get("geometry_score", 0.0), 0.0, 1.0)),
            "takeoff_similarity_score": float(clamp(renal_pair.get("takeoff_similarity", 0.0), 0.0, 1.0)),
            "horizontality_score": float(clamp(renal_pair.get("horizontality_score", 0.0), 0.0, 1.0)),
        }

    iliac_ref, iliac_conf = iliac_reference_axis()

    renal_pair = renal_scan.get("best_pair")
    renal_pair_conf = float(renal_pair.get("confidence", 0.0)) if renal_pair is not None else 0.0

    excluded_renal_eps: set[int] = set()
    if renal_pair is not None:
        excluded_renal_eps.add(int(renal_pair["a"]["ep"]))
        excluded_renal_eps.add(int(renal_pair["b"]["ep"]))
    visceral_axis, visceral_conf, visceral_count = collect_visceral_branch_axis(
        pts_c_provisional,
        adjacency,
        inlet_node,
        bif_node,
        trunk_set,
        dist,
        prev,
        iliac_ep_a,
        iliac_ep_b,
        excluded_endpoints=excluded_renal_eps,
    )

    if renal_pair is not None and renal_pair_conf >= 0.35:
        R_refined, renal_info = refine_horizontal_frame_from_renal_pair(R_provisional, renal_pair, iliac_ref)
        source = "renal_pair_primary"
        renal_used = True
        refined = bool(abs(float(renal_info["rotation_degrees_about_z"])) > 1.0)
        if renal_pair_conf < 0.60:
            warnings.append(f"W_FRAME_RENAL_PAIR_LOWCONF: renal-pair confidence={renal_pair_conf:.3f}; applying renal-driven rotation with caution.")
        horizontal_conf = float(
            clamp(
                0.55 * renal_pair_conf
                + 0.20 * float(renal_info["axis_confidence"])
                + 0.15 * float(renal_info["geometry_score"])
                + 0.10 * iliac_conf,
                0.0,
                1.0,
            )
        )
        score_components = {
            "renal_pair": float(renal_pair_conf),
            "renal_axis": float(renal_info["axis_confidence"]),
            "renal_geometry": float(renal_info["geometry_score"]),
            "iliac_sign": float(iliac_conf),
        }
    else:
        if renal_pair is None:
            warnings.append("W_FRAME_RENAL_PRIMARY_FAILED: no bilateral renal pair was available; keeping iliac-based provisional horizontal frame.")
        else:
            warnings.append(f"W_FRAME_RENAL_PAIR_LOWCONF: renal-pair confidence={renal_pair_conf:.3f}; falling back to iliac-based provisional horizontal frame.")
            warnings.append("W_FRAME_RENAL_PRIMARY_FAILED: bilateral renal pair confidence was too low for primary horizontal rotation.")
        R_refined = np.asarray(R_provisional, dtype=float)
        renal_info = {
            "rotation_degrees_about_z": 0.0,
            "axis_confidence": 0.0,
            "geometry_score": float(clamp(renal_pair.get("geometry_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
            "takeoff_similarity_score": float(clamp(renal_pair.get("takeoff_similarity", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
            "horizontality_score": float(clamp(renal_pair.get("horizontality_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        }
        source = "iliac_only_fallback"
        renal_used = False
        refined = False
        horizontal_conf = float(clamp(0.35 + 0.45 * iliac_conf + 0.20 * visceral_conf, 0.0, 1.0))
        score_components = {
            "iliac_sign": float(iliac_conf),
            "visceral_ap_support": float(visceral_conf),
        }

    ortho_err = float(np.linalg.norm(R_refined @ R_refined.T - np.eye(3)))
    ortho_conf = float(clamp(1.0 - ortho_err, 0.0, 1.0))
    if horizontal_conf < 0.60:
        warnings.append(f"W_FRAME_HORIZONTAL_WEAK: horizontal frame confidence={horizontal_conf:.3f} (source={source}).")

    info = {
        "source": source,
        "confidence": float(horizontal_conf),
        "ortho_confidence": float(ortho_conf),
        "renal_refinement_used": bool(renal_used),
        "rotation_degrees_about_z": float(renal_info["rotation_degrees_about_z"]),
        "rotation_degrees": float(abs(float(renal_info["rotation_degrees_about_z"]))),
        "refined": bool(refined),
        "renal_pair_confidence": float(renal_pair_conf),
        "renal_pair_axis_confidence": float(renal_info["axis_confidence"]),
        "renal_pair_geometry_score": float(renal_info["geometry_score"]),
        "renal_pair_takeoff_similarity_score": float(renal_info["takeoff_similarity_score"]),
        "renal_pair_horizontality_score": float(renal_info["horizontality_score"]),
        "visceral_axis_confidence": float(visceral_conf),
        "visceral_axis_count": int(visceral_count),
        "iliac_axis_confidence": float(iliac_conf),
        "score_components": {k: float(v) for k, v in score_components.items()},
    }
    return R_refined, info


def identify_renal_branches(
    adjacency: Dict[int, Dict[int, float]],
    pts_c: np.ndarray,
    inlet_node: int,
    bif_node: int,
    trunk_set: set,
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    warnings: List[str],
    renal_scan: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], float, Dict[str, Any]]:
    """
    Final renal pairing and left/right assignment in the final canonical frame.
    Candidate discovery is frame-robust and can be reused across in-plane rotations.
    """
    scan = renal_scan or discover_renal_branch_candidates(
        adjacency,
        pts_c,
        inlet_node,
        bif_node,
        trunk_set,
        dist,
        prev,
        iliac_ep_a,
        iliac_ep_b,
        warnings,
    )

    candidates = list(scan.get("candidates", []))
    pair_candidates = list(scan.get("pair_candidates", []))
    diag: Dict[str, Any] = {
        "candidate_count": int(len(candidates)),
        "pair_candidate_count": int(len(pair_candidates)),
        "best_pair_available": bool(scan.get("best_pair") is not None),
        "trunk_scaffold_node_count": int(scan.get("trunk_scaffold_node_count", 0)),
        "trunk_scaffold_extra_node_count": int(scan.get("trunk_scaffold_extra_node_count", 0)),
        "trunk_scaffold_chain_count": int(scan.get("trunk_scaffold_chain_count", 0)),
    }

    if not candidates:
        warnings.append("W_RENAL_NOT_ENOUGH_CAND: not enough renal candidates detected.")
        return None, None, None, None, 0.0, diag

    chosen_pair = scan.get("best_pair")
    chosen_score = float(chosen_pair.get("score", 0.0)) if chosen_pair is not None else 0.0

    if chosen_pair is None:
        sided_pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, float]]] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                ax = float(pts_c[int(a["ep"])][0])
                bx = float(pts_c[int(b["ep"])][0])
                if (ax >= 0.0) == (bx >= 0.0):
                    continue
                dt = abs(float(a["takeoff_dist"]) - float(b["takeoff_dist"])) / (float(scan.get("trunk_len", 1.0)) + EPS)
                takeoff_sim = clamp(1.0 - dt / 0.30, 0.0, 1.0)
                dl = abs(float(a["branch_len"]) - float(b["branch_len"])) / (float(a["branch_len"]) + float(b["branch_len"]) + EPS)
                len_sim = clamp(1.0 - dl, 0.0, 1.0)
                horiz_pair = float(0.5 * (float(a["horizontality"]) + float(b["horizontality"])))
                geometry = float(0.5 * (float(a["score"]) + float(b["score"])))
                pair_score = 0.45 * geometry + 0.35 * takeoff_sim + 0.20 * len_sim
                sided_pairs.append(
                    (
                        float(pair_score),
                        a,
                        b,
                        {
                            "geometry_score": float(geometry),
                            "takeoff_similarity": float(takeoff_sim),
                            "horizontality_score": float(horiz_pair),
                        },
                    )
                )
        if sided_pairs:
            sided_pairs.sort(key=lambda item: item[0], reverse=True)
            chosen_score, a, b, chosen_metrics = sided_pairs[0]
            chosen_pair = {
                "a": a,
                "b": b,
                "score": float(chosen_score),
                "geometry_score": float(chosen_metrics["geometry_score"]),
                "takeoff_similarity": float(chosen_metrics["takeoff_similarity"]),
                "horizontality_score": float(chosen_metrics["horizontality_score"]),
            }

    if chosen_pair is None:
        best = max(candidates, key=lambda d: float(d["score"]))
        side = "right" if float(pts_c[int(best["ep"])][0]) >= 0.0 else "left"
        warnings.append(f"W_RENAL_SINGLE: only one candidate renal assigned as {side}.")
        diag["selected_pair_score"] = 0.0
        if side == "right":
            return int(best["ep"]), None, int(best["takeoff"]), None, 0.25, diag
        return None, int(best["ep"]), None, int(best["takeoff"]), 0.25, diag

    a = chosen_pair["a"]
    b = chosen_pair["b"]
    ax = float(pts_c[int(a["ep"])][0])
    bx = float(pts_c[int(b["ep"])][0])
    if ax >= bx:
        right = a
        left = b
    else:
        right = b
        left = a

    conf = float(clamp(0.20 + 0.80 * float(chosen_score), 0.0, 1.0))
    if conf < 0.60:
        warnings.append(f"W_RENAL_LOWER_CONF: renal confidence={conf:.3f}.")

    diag["selected_pair_score"] = float(chosen_score)
    diag["selected_pair_confidence"] = float(conf)
    diag["selected_pair_geometry_score"] = float(chosen_pair.get("geometry_score", 0.0))
    diag["selected_pair_takeoff_similarity_score"] = float(chosen_pair.get("takeoff_similarity", 0.0))
    diag["selected_pair_horizontality_score"] = float(chosen_pair.get("horizontality_score", 0.0))
    diag["right_trunk_takeoff"] = int(right.get("trunk_takeoff", right["takeoff"]))
    diag["left_trunk_takeoff"] = int(left.get("trunk_takeoff", left["takeoff"]))
    return int(right["ep"]), int(left["ep"]), int(right["takeoff"]), int(left["takeoff"]), conf, diag


# -----------------------------
# Anterior/posterior resolution
# -----------------------------
def resolve_anterior_posterior_sign(
    pts_c: np.ndarray,
    adjacency: Dict[int, Dict[int, float]],
    inlet_node: int,
    bif_node: int,
    trunk_set: set,
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    renal_eps: Tuple[Optional[int], Optional[int]],
    warnings: List[str],
) -> Tuple[bool, float, bool]:
    """
    Attempt to resolve +Y (anterior) direction using visceral side branches (non-renal, non-iliac)
    off the abdominal aorta trunk.

    Returns: (need_flip_xy, confidence, warn)
    If need_flip_xy is True, then flip both x and y axes (to keep right-handed) while keeping z fixed.
    """
    cues = collect_ap_orientation_cues(
        pts_c,
        adjacency,
        inlet_node,
        bif_node,
        trunk_set,
        dist,
        prev,
        iliac_ep_a,
        iliac_ep_b,
        renal_eps,
    )

    if not cues:
        warnings.append("W_AP_NO_CUES: insufficient anterior/posterior cues; AP sign left as-is.")
        return False, 0.0, True

    wsum = sum(float(c["weight"]) for c in cues) + EPS
    vote = sum(float(c["weight"]) * float(c["sign_y"]) for c in cues) / wsum
    conf = float(clamp(abs(vote), 0.0, 1.0))
    need_flip = bool(vote < 0.0 and conf >= 0.60)
    warn = bool(conf < 0.60)
    if warn:
        warnings.append(f"W_AP_LOWCONF: AP confidence={conf:.3f} from {len(cues)} cues; AP may be mirrored.")
    return need_flip, conf, warn


# -----------------------------
# Output polydata construction
# -----------------------------
def add_string_array_to_cell_data(cd: "vtkCellData", name: str, values: List[str]) -> None:
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    arr.SetNumberOfValues(len(values))
    for i, v in enumerate(values):
        arr.SetValue(i, str(v))
    cd.AddArray(arr)


def add_scalar_array_to_cell_data(cd: "vtkCellData", name: str, values: List[float], vtk_type: int = vtk.VTK_DOUBLE) -> None:
    arr = vtk.vtkDoubleArray() if vtk_type == vtk.VTK_DOUBLE else vtk.vtkIntArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(len(values))
    for i, v in enumerate(values):
        if vtk_type == vtk.VTK_DOUBLE:
            arr.SetTuple1(i, float(v))
        else:
            arr.SetTuple1(i, int(v))
    cd.AddArray(arr)


def add_vector_array_to_point_data(pd: "vtkPointData", name: str, vectors: np.ndarray) -> None:
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(3)
    arr.SetNumberOfTuples(int(vectors.shape[0]))
    for i in range(int(vectors.shape[0])):
        arr.SetTuple3(i, float(vectors[i, 0]), float(vectors[i, 1]), float(vectors[i, 2]))
    pd.AddArray(arr)


def add_scalar_array_to_point_data(pd: "vtkPointData", name: str, values: np.ndarray, as_int: bool = False) -> None:
    if as_int:
        arr = vtk.vtkIntArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfTuples(int(values.shape[0]))
        for i in range(int(values.shape[0])):
            arr.SetTuple1(i, int(values[i]))
        pd.AddArray(arr)
    else:
        arr = vtk.vtkDoubleArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfTuples(int(values.shape[0]))
        for i in range(int(values.shape[0])):
            arr.SetTuple1(i, float(values[i]))
        pd.AddArray(arr)


def add_field_vector(fd: "vtkFieldData", name: str, vec3: np.ndarray) -> None:
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(3)
    arr.SetNumberOfTuples(1)
    v = np.asarray(vec3, dtype=float).reshape(3)
    arr.SetTuple3(0, float(v[0]), float(v[1]), float(v[2]))
    fd.AddArray(arr)


def add_field_matrix4x4(fd: "vtkFieldData", name: str, M: np.ndarray) -> None:
    mat = np.asarray(M, dtype=float).reshape(4, 4)
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(16)
    arr.SetNumberOfTuples(1)
    arr.SetTuple(0, [float(x) for x in mat.flatten(order="C")])
    fd.AddArray(arr)


def add_field_string_list(fd: "vtkFieldData", name: str, values: List[str]) -> None:
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    arr.SetNumberOfValues(len(values))
    for i, v in enumerate(values):
        arr.SetValue(i, str(v))
    fd.AddArray(arr)


def add_field_scalar(fd: "vtkFieldData", name: str, value: float) -> None:
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(1)
    arr.SetTuple1(0, float(value))
    fd.AddArray(arr)


def add_field_int(fd: "vtkFieldData", name: str, value: int) -> None:
    arr = vtk.vtkIntArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(1)
    arr.SetTuple1(0, int(value))
    fd.AddArray(arr)


def get_cell_scalar_array_values(pd: "vtkPolyData", name: str, default_value: float = 0.0) -> List[float]:
    cd = pd.GetCellData()
    arr = cd.GetArray(name) if cd is not None else None
    n_cells = int(pd.GetNumberOfCells())
    if arr is None:
        return [default_value] * n_cells
    values: List[float] = []
    for i in range(n_cells):
        values.append(float(arr.GetTuple1(i)))
    return values


def get_cell_string_array_values(pd: "vtkPolyData", name: str, default_value: str = "") -> List[str]:
    cd = pd.GetCellData()
    arr = cd.GetAbstractArray(name) if cd is not None else None
    n_cells = int(pd.GetNumberOfCells())
    if arr is None:
        return [default_value] * n_cells
    values: List[str] = []
    for i in range(n_cells):
        if isinstance(arr, vtk.vtkStringArray):
            values.append(str(arr.GetValue(i)))
        else:
            values.append(str(arr.GetVariantValue(i).ToString()))
    return values


def clone_polydata(pd: "vtkPolyData") -> "vtkPolyData":
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    return out


def add_common_output_field_data(
    fd: "vtkFieldData",
    landmarks: Dict[str, Any],
    transform_R: np.ndarray,
    transform_origin: np.ndarray,
    warnings: List[str],
    confidences: Dict[str, float],
    extra_counts: Optional[Dict[str, int]] = None,
) -> None:
    label_ids_sorted = sorted(LABEL_ID_TO_NAME.keys())
    label_names_sorted = [LABEL_ID_TO_NAME[k] for k in label_ids_sorted]
    add_field_string_list(fd, "LabelNames", label_names_sorted)
    add_field_scalar(fd, "LabelCount", float(len(label_ids_sorted)))
    add_field_string_list(fd, "GeometryTypeNames", ["surface", "centerline"])
    add_field_scalar(fd, "GeometryTypeCount", 2.0)

    M = build_transform_matrix4x4(transform_R, transform_origin)
    R = np.asarray(transform_R, dtype=float).reshape(3, 3)
    o = np.asarray(transform_origin, dtype=float).reshape(3)
    add_field_matrix4x4(fd, "CanonicalTransformMatrix4x4", M)
    add_field_vector(fd, "CanonicalOrigin", o)
    add_field_vector(fd, "CanonicalAxisX", R[0, :])
    add_field_vector(fd, "CanonicalAxisY", R[1, :])
    add_field_vector(fd, "CanonicalAxisZ", R[2, :])

    for key, val in landmarks.items():
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 3:
            add_field_vector(fd, f"Landmark_{key}_XYZ", np.array(val, dtype=float))

    add_field_string_list(fd, "Warnings", warnings)
    for k, v in confidences.items():
        add_field_scalar(fd, f"Confidence_{k}", float(v))

    for key, value in (extra_counts or {}).items():
        add_field_int(fd, key, int(value))


def annotate_surface_polydata_for_combined_output(surface_pd: "vtkPolyData") -> "vtkPolyData":
    out = clone_polydata(surface_pd)
    n_cells = int(out.GetNumberOfCells())
    cd = out.GetCellData()
    add_scalar_array_to_cell_data(cd, "GeometryTypeId", [GEOMETRY_TYPE_SURFACE] * n_cells, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "GeometryTypeName", ["surface"] * n_cells)
    add_scalar_array_to_cell_data(cd, "BranchLabelId", [0] * n_cells, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchLabelName", ["surface"] * n_cells)
    add_scalar_array_to_cell_data(cd, "AnatomicalLabelId", [0] * n_cells, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "AnatomicalName", ["surface"] * n_cells)
    add_scalar_array_to_cell_data(cd, "BranchLength", [0.0] * n_cells, vtk_type=vtk.VTK_DOUBLE)
    add_scalar_array_to_cell_data(cd, "IsSurface", [1] * n_cells, vtk_type=vtk.VTK_INT)
    add_scalar_array_to_cell_data(cd, "IsCenterline", [0] * n_cells, vtk_type=vtk.VTK_INT)
    return out


def build_centerline_branch_geometries(
    adjacency_full: Dict[int, Dict[int, float]],
    pts_canonical: np.ndarray,
    dist_from_inlet: Dict[int, float],
    trunk_path: List[int],
    right_iliac_nodes: List[int],
    left_iliac_nodes: List[int],
    right_renal_nodes: List[int],
    left_renal_nodes: List[int],
    inlet_node: int,
    bif_node: int,
    right_renal_takeoff: Optional[int],
    left_renal_takeoff: Optional[int],
) -> List[Dict[str, Any]]:
    trunk_edges = path_edge_keys(trunk_path)
    right_iliac_edges = path_edge_keys(right_iliac_nodes)
    left_iliac_edges = path_edge_keys(left_iliac_nodes)
    right_renal_edges = path_edge_keys(right_renal_nodes)
    left_renal_edges = path_edge_keys(left_renal_nodes)

    def classify_chain(nodes: List[int]) -> Tuple[int, str]:
        edges = path_edge_keys(nodes)
        if edges and edges.issubset(trunk_edges):
            return LABEL_AORTA_TRUNK, LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]
        if edges and edges.issubset(right_iliac_edges):
            return LABEL_RIGHT_ILIAC, LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]
        if edges and edges.issubset(left_iliac_edges):
            return LABEL_LEFT_ILIAC, LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]
        if edges and edges.issubset(right_renal_edges):
            return LABEL_RIGHT_RENAL, LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL]
        if edges and edges.issubset(left_renal_edges):
            return LABEL_LEFT_RENAL, LABEL_ID_TO_NAME[LABEL_LEFT_RENAL]
        return LABEL_OTHER, LABEL_ID_TO_NAME[LABEL_OTHER]

    label_priority = {
        LABEL_AORTA_TRUNK: 0,
        LABEL_RIGHT_ILIAC: 1,
        LABEL_LEFT_ILIAC: 2,
        LABEL_RIGHT_RENAL: 3,
        LABEL_LEFT_RENAL: 4,
        LABEL_OTHER: 5,
    }

    branch_geoms: List[Dict[str, Any]] = []
    chains = build_branch_chains_from_graph(adjacency_full)
    classified: List[Tuple[int, List[int], int, str]] = []
    for chain in chains:
        nodes = [int(n) for n in chain]
        if len(nodes) < 2:
            continue
        d0 = dist_from_inlet.get(nodes[0], float("nan"))
        d1 = dist_from_inlet.get(nodes[-1], float("nan"))
        if math.isfinite(d0) and math.isfinite(d1) and d0 > d1:
            nodes = list(reversed(nodes))
        elif not math.isfinite(d0) and not math.isfinite(d1) and nodes[0] > nodes[-1]:
            nodes = list(reversed(nodes))

        label_id, label_name = classify_chain(nodes)
        classified.append((label_priority.get(label_id, 99), nodes, label_id, label_name))

    classified.sort(key=lambda item: (item[0], tuple(item[1])))

    for _, nodes, label_id, label_name in classified:
        node_to_index = {int(n): i for i, n in enumerate(nodes)}
        landmark_point_ids: Dict[str, int] = {}
        if inlet_node in node_to_index:
            landmark_point_ids["Inlet"] = int(node_to_index[inlet_node])
        if bif_node in node_to_index:
            landmark_point_ids["Bifurcation"] = int(node_to_index[bif_node])
        if right_renal_takeoff is not None and right_renal_takeoff in node_to_index:
            landmark_point_ids["RightRenalOrigin"] = int(node_to_index[right_renal_takeoff])
        if left_renal_takeoff is not None and left_renal_takeoff in node_to_index:
            landmark_point_ids["LeftRenalOrigin"] = int(node_to_index[left_renal_takeoff])

        branch_geoms.append(
            dict(
                label_id=int(label_id),
                name=str(label_name),
                points=pts_canonical[np.array(nodes, dtype=int)],
                landmark_point_ids=landmark_point_ids,
                node_ids=list(nodes),
            )
        )

    return branch_geoms


def build_output_centerlines_polydata(
    branch_geoms: List[Dict[str, Any]],
    landmarks: Dict[str, Any],
    transform_R: np.ndarray,
    transform_origin: np.ndarray,
    warnings: List[str],
    confidences: Dict[str, float],
    extra_counts: Optional[Dict[str, int]] = None,
) -> "vtkPolyData":
    """
    branch_geoms: list of dicts:
        {
            "label_id": int,
            "name": str,
            "points": (N,3) np.ndarray in canonical coords, ordered proximal->distal,
            "landmark_point_ids": dict[str,int] (optional): mapping landmark key -> local index in this branch points
        }
    """
    out = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Global point data arrays (constructed after points are known)
    branch_label_per_point: List[int] = []
    branch_point_index: List[int] = []
    abscissa_per_point: List[float] = []
    tangents_per_point: List[List[float]] = []

    cell_label: List[int] = []
    cell_name: List[str] = []
    cell_length: List[float] = []

    landmark_global_point_ids: Dict[str, int] = {}

    global_pid = 0
    for cell_id, br in enumerate(branch_geoms):
        pts = np.asarray(br["points"], dtype=float)
        if pts.shape[0] < 2:
            continue

        # Insert points
        start_pid = global_pid
        for i in range(pts.shape[0]):
            points.InsertNextPoint(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
            branch_label_per_point.append(int(br["label_id"]))
            branch_point_index.append(int(i))
            global_pid += 1

        # Create polyline cell
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(pts.shape[0])
        for i in range(pts.shape[0]):
            polyline.GetPointIds().SetId(i, start_pid + i)
        lines.InsertNextCell(polyline)

        # Per-point metrics (abscissa, tangent) for this branch
        s = compute_abscissa(pts)
        T = compute_tangents(pts)
        for i in range(pts.shape[0]):
            abscissa_per_point.append(float(s[i]))
            tangents_per_point.append([float(T[i, 0]), float(T[i, 1]), float(T[i, 2])])

        cell_label.append(int(br["label_id"]))
        cell_name.append(str(br["name"]))
        cell_length.append(float(s[-1]) if s.size else 0.0)

        # Map landmarks that lie on this branch to output global point ids
        local_lm = br.get("landmark_point_ids", {}) or {}
        for key, local_idx in local_lm.items():
            if key in landmark_global_point_ids:
                continue
            if 0 <= int(local_idx) < pts.shape[0]:
                landmark_global_point_ids[key] = start_pid + int(local_idx)

    out.SetPoints(points)
    out.SetLines(lines)

    # Attach point data arrays
    pd = out.GetPointData()
    add_scalar_array_to_point_data(pd, "AnatomicalLabelId", np.array(branch_label_per_point, dtype=int), as_int=True)
    add_scalar_array_to_point_data(pd, "BranchPointIndex", np.array(branch_point_index, dtype=int), as_int=True)
    add_scalar_array_to_point_data(pd, "Abscissa", np.array(abscissa_per_point, dtype=float), as_int=False)
    add_vector_array_to_point_data(pd, "Tangent", np.array(tangents_per_point, dtype=float))

    # Attach cell data arrays
    cd = out.GetCellData()
    add_scalar_array_to_cell_data(cd, "GeometryTypeId", [GEOMETRY_TYPE_CENTERLINE] * len(cell_label), vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "GeometryTypeName", ["centerline"] * len(cell_label))
    add_scalar_array_to_cell_data(cd, "BranchLabelId", cell_label, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchLabelName", cell_name)
    add_scalar_array_to_cell_data(cd, "AnatomicalLabelId", cell_label, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "AnatomicalName", cell_name)
    add_scalar_array_to_cell_data(cd, "BranchLength", cell_length, vtk_type=vtk.VTK_DOUBLE)
    add_scalar_array_to_cell_data(cd, "IsSurface", [0] * len(cell_label), vtk_type=vtk.VTK_INT)
    add_scalar_array_to_cell_data(cd, "IsCenterline", [1] * len(cell_label), vtk_type=vtk.VTK_INT)

    # Field data: label schema, landmarks, transform, warnings/confidences
    fd = out.GetFieldData()
    add_common_output_field_data(
        fd,
        landmarks=landmarks,
        transform_R=transform_R,
        transform_origin=transform_origin,
        warnings=warnings,
        confidences=confidences,
        extra_counts=extra_counts,
    )

    # Landmark point ids on the centerline scaffold
    for key, pid in landmark_global_point_ids.items():
        add_field_int(fd, f"Landmark_{key}_PointId", int(pid))

    return out


def build_combined_surface_centerlines_polydata(
    surface_pd: "vtkPolyData",
    centerlines_pd: "vtkPolyData",
    landmarks: Dict[str, Any],
    transform_R: np.ndarray,
    transform_origin: np.ndarray,
    warnings: List[str],
    confidences: Dict[str, float],
    extra_counts: Optional[Dict[str, int]] = None,
) -> "vtkPolyData":
    surface_tagged = annotate_surface_polydata_for_combined_output(surface_pd)
    centerline_tagged = clone_polydata(centerlines_pd)

    app = vtk.vtkAppendPolyData()
    app.AddInputData(surface_tagged)
    app.AddInputData(centerline_tagged)
    app.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(app.GetOutput())
    out.BuildLinks()

    n_surface_cells = int(surface_pd.GetNumberOfCells())
    n_centerline_cells = int(centerlines_pd.GetNumberOfCells())
    combined_cd = out.GetCellData()
    combined_cd.Initialize()
    add_scalar_array_to_cell_data(
        combined_cd,
        "GeometryTypeId",
        [GEOMETRY_TYPE_SURFACE] * n_surface_cells + [GEOMETRY_TYPE_CENTERLINE] * n_centerline_cells,
        vtk_type=vtk.VTK_INT,
    )
    add_string_array_to_cell_data(
        combined_cd,
        "GeometryTypeName",
        ["surface"] * n_surface_cells + ["centerline"] * n_centerline_cells,
    )
    add_scalar_array_to_cell_data(
        combined_cd,
        "BranchLabelId",
        [0] * n_surface_cells + [int(v) for v in get_cell_scalar_array_values(centerlines_pd, "BranchLabelId", 0)],
        vtk_type=vtk.VTK_INT,
    )
    add_string_array_to_cell_data(
        combined_cd,
        "BranchLabelName",
        ["surface"] * n_surface_cells + get_cell_string_array_values(centerlines_pd, "BranchLabelName", "other"),
    )
    add_scalar_array_to_cell_data(
        combined_cd,
        "AnatomicalLabelId",
        [0] * n_surface_cells + [int(v) for v in get_cell_scalar_array_values(centerlines_pd, "AnatomicalLabelId", 0)],
        vtk_type=vtk.VTK_INT,
    )
    add_string_array_to_cell_data(
        combined_cd,
        "AnatomicalName",
        ["surface"] * n_surface_cells + get_cell_string_array_values(centerlines_pd, "AnatomicalName", "other"),
    )
    add_scalar_array_to_cell_data(
        combined_cd,
        "BranchLength",
        [0.0] * n_surface_cells + get_cell_scalar_array_values(centerlines_pd, "BranchLength", 0.0),
        vtk_type=vtk.VTK_DOUBLE,
    )
    add_scalar_array_to_cell_data(
        combined_cd,
        "IsSurface",
        [1] * n_surface_cells + [0] * n_centerline_cells,
        vtk_type=vtk.VTK_INT,
    )
    add_scalar_array_to_cell_data(
        combined_cd,
        "IsCenterline",
        [0] * n_surface_cells + [1] * n_centerline_cells,
        vtk_type=vtk.VTK_INT,
    )

    add_common_output_field_data(
        out.GetFieldData(),
        landmarks=landmarks,
        transform_R=transform_R,
        transform_origin=transform_origin,
        warnings=warnings,
        confidences=confidences,
        extra_counts=extra_counts,
    )
    return out


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> int:
    require_vtk()

    parser = argparse.ArgumentParser(description="First-stage anatomy-aware oriented surface + centerline preprocessing for abdominal arterial tree.")
    parser.add_argument("--input", type=str, default=INPUT_VTP_PATH, help="Input lumen surface VTP path")
    parser.add_argument(
        "--output_surface_with_centerlines",
        type=str,
        default=OUTPUT_SURFACE_WITH_CENTERLINES_VTP_PATH,
        help="Main output oriented surface + centerlines VTP path",
    )
    parser.add_argument(
        "--output_centerlines",
        type=str,
        default=OUTPUT_CENTERLINES_VTP_PATH,
        help="Secondary output oriented labeled centerline scaffold VTP path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Deprecated alias for --output_centerlines (kept for compatibility)",
    )
    parser.add_argument("--metadata", type=str, default=OUTPUT_METADATA_PATH, help="Output metadata JSON path (optional)")
    parser.add_argument("--debug_raw_centerlines", type=str, default=OUTPUT_DEBUG_CENTERLINES_RAW_PATH, help="Optional raw centerlines VTP path")
    args = parser.parse_args()

    input_path = _resolve_user_path(args.input)
    surface_with_centerlines_path = _resolve_user_path(args.output_surface_with_centerlines)
    centerlines_output_arg = args.output if args.output is not None else args.output_centerlines
    centerlines_output_path = _resolve_user_path(centerlines_output_arg)
    meta_path = _resolve_user_path(args.metadata) if args.metadata is not None else ""
    debug_raw_path = _resolve_user_path(args.debug_raw_centerlines) if args.debug_raw_centerlines is not None else ""

    warnings: List[str] = []
    confidences: Dict[str, float] = {}

    try:
        surface = load_vtp(input_path)
        surface_tri = clean_and_triangulate_surface(surface)

        surface_pts = get_points_numpy(surface_tri)

        terms, mode = detect_terminations(surface_tri, warnings)
        if len(terms) < 2:
            raise RuntimeError("Failed to detect enough terminations (need >=2) to seed centerlines.")

        inlet_term, inlet_term_conf, axis_si = choose_inlet_termination(terms, surface_pts, warnings)
        if inlet_term is None:
            raise RuntimeError("Inlet termination could not be determined.")
        confidences["inlet_termination"] = float(inlet_term_conf)

        term_centers = [t.center for t in terms if np.linalg.norm(t.center - inlet_term.center) > 1e-8]
        centerlines, cl_info = compute_centerlines_vmtk(surface_tri, inlet_term.center, term_centers, warnings)

        if debug_raw_path:
            try:
                write_vtp(centerlines, debug_raw_path, binary=True)
            except Exception as e:
                warnings.append(f"W_DEBUG_WRITE_RAW_CENTERLINES: {e}")

        # Build graph
        adjacency_full, cl_pts, _ = build_graph_from_polyline_centerlines(centerlines)
        if not adjacency_full:
            raise RuntimeError("Centerline graph adjacency is empty. Check centerline output line structure.")

        graph_nodes = sorted(int(n) for n in adjacency_full.keys())
        analysis_seed_node, _ = nearest_node_to_point(graph_nodes, cl_pts, inlet_term.center)
        if analysis_seed_node < 0:
            raise RuntimeError("Failed to locate inlet-near centerline node for graph component selection.")

        analysis_nodes = connected_component_nodes(adjacency_full, analysis_seed_node)
        if not analysis_nodes:
            raise RuntimeError("Failed to identify inlet-connected centerline component.")

        adjacency = induced_subgraph(adjacency_full, analysis_nodes)
        if not adjacency:
            raise RuntimeError("Inlet-connected centerline component is empty after graph extraction.")

        total_component_count = count_graph_connected_components(adjacency_full)
        if len(analysis_nodes) < len(adjacency_full):
            warnings.append(
                f"W_CENTERLINE_EXTRA_COMPONENTS: preserved {len(adjacency_full) - len(analysis_nodes)} nodes outside the inlet-connected component; anatomy inference uses the inlet-connected scaffold."
            )

        deg = node_degrees(adjacency)
        endpoints_all = [n for n, d in deg.items() if d == 1]
        if len(endpoints_all) < 2:
            raise RuntimeError("Centerline graph has insufficient endpoints.")

        # Inlet node as nearest endpoint to inlet termination center
        inlet_node, inlet_node_conf = pick_inlet_node_from_endpoints(endpoints_all, cl_pts, inlet_term.center, inlet_term_conf, warnings)
        if inlet_node < 0:
            raise RuntimeError("Failed to identify inlet node on centerlines.")
        confidences["inlet_node"] = float(inlet_node_conf)

        # Identify iliac pair and bifurcation
        bif_node, ep_a, ep_b, bif_conf, dist, prev = choose_iliac_endpoints_and_bifurcation(
            adjacency,
            cl_pts,
            inlet_node,
            axis_si,
            warnings,
            terminations=terms,
            inlet_term=inlet_term,
        )
        if bif_node is None or ep_a is None or ep_b is None:
            raise RuntimeError("Failed to identify aortic bifurcation / iliac endpoints.")
        confidences["aortic_bifurcation"] = float(bif_conf)

        # Trunk path (inlet -> bif)
        trunk_path = path_to_root(prev, inlet_node, bif_node)
        if not trunk_path:
            raise RuntimeError("Failed to reconstruct trunk path inlet->bifurcation.")
        trunk_set = set(trunk_path)

        # Paths from inlet to iliac endpoints
        path_a_full = path_to_root(prev, inlet_node, ep_a)
        path_b_full = path_to_root(prev, inlet_node, ep_b)
        if not path_a_full or not path_b_full:
            raise RuntimeError("Failed to reconstruct paths to iliac endpoints.")

        # Iliac subpaths (bif -> endpoint)
        if bif_node not in path_a_full or bif_node not in path_b_full:
            warnings.append("W_BIF_NOT_ON_ILIAC_PATH: bif not found on one of iliac endpoint paths; using common ancestor as bif.")
        idx_a = path_a_full.index(bif_node) if bif_node in path_a_full else max(0, len(path_a_full) - 2)
        idx_b = path_b_full.index(bif_node) if bif_node in path_b_full else max(0, len(path_b_full) - 2)
        iliac_path_a = path_a_full[idx_a:]
        iliac_path_b = path_b_full[idx_b:]

        # Determine main iliac segments (bif -> first downstream branchpoint OR endpoint)
        bp_a = first_branchpoint_on_path(iliac_path_a, deg, start_index=1)
        bp_b = first_branchpoint_on_path(iliac_path_b, deg, start_index=1)
        if bp_a is not None and bp_a in iliac_path_a:
            iliac_main_a = iliac_path_a[: iliac_path_a.index(bp_a) + 1]
        else:
            iliac_main_a = iliac_path_a
        if bp_b is not None and bp_b in iliac_path_b:
            iliac_main_b = iliac_path_b[: iliac_path_b.index(bp_b) + 1]
        else:
            iliac_main_b = iliac_path_b

        # Build preliminary transform (z superior, x between iliac endpoints)
        inlet_pt = cl_pts[inlet_node]
        bif_pt = cl_pts[bif_node]
        ep_a_pt = cl_pts[iliac_main_a[-1]] if iliac_main_a else cl_pts[ep_a]
        ep_b_pt = cl_pts[iliac_main_b[-1]] if iliac_main_b else cl_pts[ep_b]

        R_provisional, origin, frame_conf = build_canonical_transform(inlet_pt, bif_pt, ep_a_pt, ep_b_pt, cl_pts, warnings)
        cl_pts_c_provisional = apply_transform_points(cl_pts, R_provisional, origin)

        renal_scan = discover_renal_branch_candidates(
            adjacency,
            cl_pts_c_provisional,
            inlet_node,
            bif_node,
            trunk_set,
            dist,
            prev,
            ep_a,
            ep_b,
            warnings,
        )

        R, horizontal_frame_info = refine_horizontal_axes_using_branch_anatomy(
            R_provisional,
            cl_pts_c_provisional,
            iliac_main_a,
            iliac_main_b,
            renal_scan,
            adjacency,
            inlet_node,
            bif_node,
            trunk_set,
            dist,
            prev,
            ep_a,
            ep_b,
            warnings,
        )
        confidences["canonical_frame_orthonormality"] = float(horizontal_frame_info.get("ortho_confidence", frame_conf))
        confidences["horizontal_frame"] = float(horizontal_frame_info.get("confidence", 0.0))

        cl_pts_c_pre_ap = apply_transform_points(cl_pts, R, origin)
        best_pair = renal_scan.get("best_pair")
        renal_eps_for_ap = (
            int(best_pair["a"]["ep"]) if best_pair is not None else None,
            int(best_pair["b"]["ep"]) if best_pair is not None else None,
        )
        need_flip_xy, ap_conf, ap_warn = resolve_anterior_posterior_sign(
            cl_pts_c_pre_ap,
            adjacency,
            inlet_node,
            bif_node,
            trunk_set,
            dist,
            prev,
            ep_a,
            ep_b,
            renal_eps_for_ap,
            warnings,
        )
        confidences["ap_orientation"] = float(ap_conf)
        horizontal_frame_info["ap_sign_confidence"] = float(ap_conf)
        horizontal_frame_info["ap_sign_warn"] = bool(ap_warn)

        flipped_for_ap = False
        if need_flip_xy:
            flipped_for_ap = True
            R_flipped = R.copy()
            R_flipped[0, :] *= -1.0
            R_flipped[1, :] *= -1.0
            R = R_flipped
        if bool(horizontal_frame_info.get("renal_refinement_used", False)) and ap_conf >= 0.60:
            horizontal_frame_info["source"] = "renal_pair_plus_visceral_ap"

        cl_pts_c = apply_transform_points(cl_pts, R, origin)
        surface_tri_c = apply_transform_to_polydata(surface_tri, R, origin)

        # Determine which iliac main path is right/left by x of distal point in canonical frame
        a_x = float(cl_pts_c[iliac_main_a[-1]][0]) if iliac_main_a else float(cl_pts_c[ep_a][0])
        b_x = float(cl_pts_c[iliac_main_b[-1]][0]) if iliac_main_b else float(cl_pts_c[ep_b][0])

        if a_x >= b_x:
            # a is more "right"
            right_iliac_main = iliac_main_a
            left_iliac_main = iliac_main_b
            right_iliac_ep = ep_a
            left_iliac_ep = ep_b
        else:
            right_iliac_main = iliac_main_b
            left_iliac_main = iliac_main_a
            right_iliac_ep = ep_b
            left_iliac_ep = ep_a

        # Now identify renals using final canonical coordinates
        rr_ep, lr_ep, rr_take, lr_take, renal_conf, renal_diag = identify_renal_branches(
            adjacency,
            cl_pts_c,
            inlet_node,
            bif_node,
            trunk_set,
            dist,
            prev,
            right_iliac_ep,
            left_iliac_ep,
            warnings,
            renal_scan=renal_scan,
        )
        confidences["renal_pair"] = float(renal_conf)
        rr_origin_node = int(renal_diag["right_trunk_takeoff"]) if renal_diag.get("right_trunk_takeoff") is not None else rr_take
        lr_origin_node = int(renal_diag["left_trunk_takeoff"]) if renal_diag.get("left_trunk_takeoff") is not None else lr_take

        # If AP confidence was low and renals exist, optionally re-run AP resolution excluding renals
        # (kept deterministic: do not flip again; just warn).
        if ap_warn and (rr_ep is not None or lr_ep is not None):
            warnings.append("W_AP_UNCERTAIN_RENALS_PRESENT: AP orientation uncertain but output still generated deterministically.")

        # Build branch geometries (canonical coordinates) as independent polylines in proximal->distal order
        trunk_pts = cl_pts_c[trunk_path]

        # Ensure inlet at top (+Z): with constructed frame, inlet should have higher z than bif
        if float(trunk_pts[0][2]) < float(trunk_pts[-1][2]):
            # trunk currently oriented bif->inlet; reverse to inlet->bif
            trunk_pts = trunk_pts[::-1]
            trunk_path = trunk_path[::-1]

        # Iliac branches: bif->distal (start at bif)
        # Use main iliac segments. Ensure first point is bif; if not, prepend bif.
        def ensure_starts_with_bif(nodes: List[int]) -> List[int]:
            if not nodes:
                return nodes
            if nodes[0] != bif_node:
                # If bif is in nodes, trim before; else prepend
                if bif_node in nodes:
                    return nodes[nodes.index(bif_node):]
                return [bif_node] + nodes
            return nodes

        right_iliac_nodes = ensure_starts_with_bif(right_iliac_main)
        left_iliac_nodes = ensure_starts_with_bif(left_iliac_main)

        # Renal branches: takeoff->endpoint
        right_renal_nodes: List[int] = []
        left_renal_nodes: List[int] = []
        if rr_ep is not None and rr_take is not None:
            path_full = path_to_root(prev, inlet_node, rr_ep)
            if rr_take in path_full:
                seg = path_full[path_full.index(rr_take):]
                right_renal_nodes = seg
        if lr_ep is not None and lr_take is not None:
            path_full = path_to_root(prev, inlet_node, lr_ep)
            if lr_take in path_full:
                seg = path_full[path_full.index(lr_take):]
                left_renal_nodes = seg
        if len(right_renal_nodes) < 2:
            warnings.append("W_RIGHT_RENAL_MISSING: right renal branch not found or too short.")
        if len(left_renal_nodes) < 2:
            warnings.append("W_LEFT_RENAL_MISSING: left renal branch not found or too short.")

        # Landmarks in canonical coordinates
        inlet_xyz = trunk_pts[0].copy()
        bif_xyz = cl_pts_c[bif_node].copy()

        # Renal origins (on trunk): locate takeoff index along trunk nodes
        trunk_node_to_index = {int(n): i for i, n in enumerate(trunk_path)}
        rr_origin_xyz = None
        lr_origin_xyz = None
        rr_origin_index = None
        lr_origin_index = None
        if rr_origin_node is not None and rr_origin_node not in trunk_node_to_index:
            rr_origin_node = _nearest_node_on_path(trunk_path, cl_pts_c, rr_origin_node)
        if lr_origin_node is not None and lr_origin_node not in trunk_node_to_index:
            lr_origin_node = _nearest_node_on_path(trunk_path, cl_pts_c, lr_origin_node)
        if rr_origin_node is not None and rr_origin_node in trunk_node_to_index:
            rr_origin_index = int(trunk_node_to_index[rr_origin_node])
            rr_origin_xyz = trunk_pts[rr_origin_index].copy()
        if lr_origin_node is not None and lr_origin_node in trunk_node_to_index:
            lr_origin_index = int(trunk_node_to_index[lr_origin_node])
            lr_origin_xyz = trunk_pts[lr_origin_index].copy()

        # Landmarks dict for field data
        landmarks: Dict[str, Any] = {
            "Inlet": inlet_xyz,
            "Bifurcation": bif_xyz,
        }
        if rr_origin_xyz is not None:
            landmarks["RightRenalOrigin"] = rr_origin_xyz
        if lr_origin_xyz is not None:
            landmarks["LeftRenalOrigin"] = lr_origin_xyz

        preserved_branch_geoms = build_centerline_branch_geometries(
            adjacency_full=adjacency_full,
            pts_canonical=cl_pts_c,
            dist_from_inlet=dist,
            trunk_path=trunk_path,
            right_iliac_nodes=right_iliac_nodes,
            left_iliac_nodes=left_iliac_nodes,
            right_renal_nodes=right_renal_nodes,
            left_renal_nodes=left_renal_nodes,
            inlet_node=inlet_node,
            bif_node=bif_node,
            right_renal_takeoff=rr_origin_node,
            left_renal_takeoff=lr_origin_node,
        )
        if not preserved_branch_geoms:
            raise RuntimeError("Failed to construct branch-preserving centerline scaffold output.")

        branch_counts: Dict[str, int] = {}
        for br in preserved_branch_geoms:
            branch_counts[br["name"]] = int(branch_counts.get(br["name"], 0) + 1)

        extra_counts = {
            "SurfacePointCount": int(surface_tri_c.GetNumberOfPoints()),
            "SurfaceCellCount": int(surface_tri_c.GetNumberOfCells()),
            "CenterlineConnectedComponentCount": int(total_component_count),
            "CenterlineAnalysisNodeCount": int(len(analysis_nodes)),
            "CenterlineBranchCount": int(len(preserved_branch_geoms)),
        }

        centerlines_out_pd = build_output_centerlines_polydata(
            branch_geoms=preserved_branch_geoms,
            landmarks=landmarks,
            transform_R=R,
            transform_origin=origin,
            warnings=warnings,
            confidences=confidences,
            extra_counts=extra_counts,
        )
        extra_counts["CenterlinePointCount"] = int(centerlines_out_pd.GetNumberOfPoints())
        extra_counts["CenterlineCellCount"] = int(centerlines_out_pd.GetNumberOfCells())

        combined_out_pd = build_combined_surface_centerlines_polydata(
            surface_pd=surface_tri_c,
            centerlines_pd=centerlines_out_pd,
            landmarks=landmarks,
            transform_R=R,
            transform_origin=origin,
            warnings=warnings,
            confidences=confidences,
            extra_counts=extra_counts,
        )
        extra_counts["CombinedPointCount"] = int(combined_out_pd.GetNumberOfPoints())
        extra_counts["CombinedCellCount"] = int(combined_out_pd.GetNumberOfCells())

        centerlines_out_pd = build_output_centerlines_polydata(
            branch_geoms=preserved_branch_geoms,
            landmarks=landmarks,
            transform_R=R,
            transform_origin=origin,
            warnings=warnings,
            confidences=confidences,
            extra_counts=extra_counts,
        )
        combined_out_pd = build_combined_surface_centerlines_polydata(
            surface_pd=surface_tri_c,
            centerlines_pd=centerlines_out_pd,
            landmarks=landmarks,
            transform_R=R,
            transform_origin=origin,
            warnings=warnings,
            confidences=confidences,
            extra_counts=extra_counts,
        )

        write_vtp(combined_out_pd, surface_with_centerlines_path, binary=True)
        write_vtp(centerlines_out_pd, centerlines_output_path, binary=True)

        # Write metadata JSON (optional but recommended)
        if meta_path:
            os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
            meta: Dict[str, Any] = {
                "input_vtp": os.path.abspath(input_path),
                "output_surface_with_centerlines_vtp": os.path.abspath(surface_with_centerlines_path),
                "output_centerlines_vtp": os.path.abspath(centerlines_output_path),
                "mode": mode,
                "vmtk_info": cl_info,
                "labels": {str(k): v for k, v in LABEL_ID_TO_NAME.items()},
                "landmarks_xyz_canonical": {k: [float(x) for x in np.asarray(v, dtype=float).reshape(3)] for k, v in landmarks.items()},
                "inlet_node_centerline_id": int(inlet_node),
                "bif_node_centerline_id": int(bif_node),
                "right_renal_endpoint_id": int(rr_ep) if rr_ep is not None else None,
                "left_renal_endpoint_id": int(lr_ep) if lr_ep is not None else None,
                "right_renal_takeoff_id": int(rr_origin_node) if rr_origin_node is not None else None,
                "left_renal_takeoff_id": int(lr_origin_node) if lr_origin_node is not None else None,
                "right_renal_branch_root_id": int(rr_take) if rr_take is not None else None,
                "left_renal_branch_root_id": int(lr_take) if lr_take is not None else None,
                "transform": {
                    "R_rows": [[float(x) for x in row] for row in np.asarray(R, dtype=float).reshape(3, 3)],
                    "origin": [float(x) for x in np.asarray(origin, dtype=float).reshape(3)],
                    "flipped_for_ap": bool(flipped_for_ap),
                },
                "horizontal_frame_source": str(horizontal_frame_info.get("source", "iliac_only")),
                "horizontal_frame_confidence": float(horizontal_frame_info.get("confidence", 0.0)),
                "renal_frame_refinement_used": bool(horizontal_frame_info.get("renal_refinement_used", False)),
                "renal_candidate_count": int(renal_diag.get("candidate_count", 0)),
                "renal_pair_candidate_count": int(renal_diag.get("pair_candidate_count", 0)),
                "horizontal_axes_refined_after_renal_scan": bool(horizontal_frame_info.get("refined", False)),
                "renal_rotation_degrees_about_z": float(horizontal_frame_info.get("rotation_degrees_about_z", 0.0)),
                "renal_pair_axis_confidence": float(horizontal_frame_info.get("renal_pair_axis_confidence", 0.0)),
                "renal_pair_geometry_score": float(horizontal_frame_info.get("renal_pair_geometry_score", 0.0)),
                "renal_pair_takeoff_similarity_score": float(horizontal_frame_info.get("renal_pair_takeoff_similarity_score", 0.0)),
                "renal_pair_horizontality_score": float(horizontal_frame_info.get("renal_pair_horizontality_score", 0.0)),
                "horizontal_frame_diagnostics": {
                    "rotation_degrees": float(horizontal_frame_info.get("rotation_degrees", 0.0)),
                    "rotation_degrees_about_z": float(horizontal_frame_info.get("rotation_degrees_about_z", 0.0)),
                    "iliac_axis_confidence": float(horizontal_frame_info.get("iliac_axis_confidence", 0.0)),
                    "renal_pair_confidence": float(horizontal_frame_info.get("renal_pair_confidence", 0.0)),
                    "renal_pair_axis_confidence": float(horizontal_frame_info.get("renal_pair_axis_confidence", 0.0)),
                    "renal_pair_geometry_score": float(horizontal_frame_info.get("renal_pair_geometry_score", 0.0)),
                    "renal_pair_takeoff_similarity_score": float(horizontal_frame_info.get("renal_pair_takeoff_similarity_score", 0.0)),
                    "renal_pair_horizontality_score": float(horizontal_frame_info.get("renal_pair_horizontality_score", 0.0)),
                    "visceral_axis_confidence": float(horizontal_frame_info.get("visceral_axis_confidence", 0.0)),
                    "visceral_axis_count": int(horizontal_frame_info.get("visceral_axis_count", 0)),
                    "ap_sign_confidence": float(horizontal_frame_info.get("ap_sign_confidence", 0.0)),
                    "trunk_scaffold_node_count": int(renal_diag.get("trunk_scaffold_node_count", 0)),
                    "trunk_scaffold_extra_node_count": int(renal_diag.get("trunk_scaffold_extra_node_count", 0)),
                    "trunk_scaffold_chain_count": int(renal_diag.get("trunk_scaffold_chain_count", 0)),
                    "score_components": {str(k): float(v) for k, v in dict(horizontal_frame_info.get("score_components", {})).items()},
                },
                "branch_availability": {
                    "abdominal_aorta_trunk": bool(branch_counts.get(LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK], 0)),
                    "right_main_iliac": bool(branch_counts.get(LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC], 0)),
                    "left_main_iliac": bool(branch_counts.get(LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC], 0)),
                    "right_renal": bool(branch_counts.get(LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL], 0)),
                    "left_renal": bool(branch_counts.get(LABEL_ID_TO_NAME[LABEL_LEFT_RENAL], 0)),
                    "other_branch_count": int(branch_counts.get(LABEL_ID_TO_NAME[LABEL_OTHER], 0)),
                },
                "branch_counts": {k: int(v) for k, v in branch_counts.items()},
                "renals_found": {
                    "right": bool(rr_ep is not None and rr_take is not None),
                    "left": bool(lr_ep is not None and lr_take is not None),
                },
                "ap_orientation_certain": bool((not ap_warn) and ap_conf >= 0.60),
                "counts": {k: int(v) for k, v in extra_counts.items()},
                "confidences": {k: float(v) for k, v in confidences.items()},
                "warnings": list(warnings),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return 0

    except Exception as e:
        sys.stderr.write("ERROR: preprocessing failed.\n")
        sys.stderr.write(f"{e}\n")
        vmtk_related = (
            "VMTK" in str(e)
            or "vtkvmtk" in str(e)
            or bool(_LAST_VMTK_IMPORT_DIAGNOSTICS and not _LAST_VMTK_IMPORT_DIAGNOSTICS.get("vmtk_import_ok"))
        )
        runtime_report = ""
        diagnostics_for_report: Optional[Dict[str, Any]] = None
        if vmtk_related:
            if not _LAST_VMTK_IMPORT_DIAGNOSTICS:
                try_import_vmtk()
            diagnostics_for_report = dict(_LAST_VMTK_IMPORT_DIAGNOSTICS) if _LAST_VMTK_IMPORT_DIAGNOSTICS else {}
            runtime_report = debug_vmtk_runtime_report(diagnostics_for_report)
            sys.stderr.write("\nVMTK runtime diagnostics:\n")
            sys.stderr.write(runtime_report + "\n")
        sys.stderr.write(traceback.format_exc())
        if meta_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
                failure_meta: Dict[str, Any] = {
                    "status": "failed",
                    "input_vtp": os.path.abspath(input_path),
                    "output_surface_with_centerlines_vtp": os.path.abspath(surface_with_centerlines_path),
                    "output_centerlines_vtp": os.path.abspath(centerlines_output_path),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                if runtime_report:
                    failure_meta["vmtk_runtime_report"] = runtime_report
                if diagnostics_for_report:
                    failure_meta["vmtk_import_diagnostics"] = diagnostics_for_report
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(failure_meta, f, indent=2)
            except Exception as meta_exc:
                sys.stderr.write(f"\nWARNING: failed to write metadata/debug report: {meta_exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
