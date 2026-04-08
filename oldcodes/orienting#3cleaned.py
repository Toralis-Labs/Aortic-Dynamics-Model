#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anatomy-aware first-stage preprocessing for abdominal arterial lumen surface models (.vtp).

Input:
  - Unlabeled lumen surface VTP representing an abdominal arterial tree (aorta + branches).

Output (workflow-compatible intent):
  - Oriented surface + centerlines VTP (canonical frame) with BranchId / BranchName labels on cells
  - Oriented labeled centerline scaffold VTP (canonical frame)
  - Metadata JSON with landmarks, transform, warnings, and confidence diagnostics

Mandatory named systems (stable label IDs):
  0 = other
  1 = abdominal aorta trunk
  2 = right main iliac
  3 = left main iliac
  4 = right renal
  5 = left renal

Dependencies:
  - vtk
  - numpy
  - vtkvmtk (VMTK python bindings)

No manual interaction. Deterministic best-effort execution with explicit warnings and confidence.
"""

from __future__ import annotations

# -----------------------------
# User-editable paths (defaults)
# -----------------------------
INPUT_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\0044_H_ABAO_AAA\\0044_H_ABAO_AAA\\Models\\0156_0001.vtp"
OUTPUT_SURFACE_WITH_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_surface_with_centerlines.vtp"
OUTPUT_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines.vtp"
OUTPUT_METADATA_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines_metadata.json"
OUTPUT_DEBUG_CENTERLINES_RAW_PATH = ""  # optional

import os
import sys
import json
import math
import argparse
import platform
import traceback
import importlib
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Iterable, Set

import numpy as np

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import (
        vtkPolyData,
        vtkCellData,
        vtkStaticPointLocator,
    )

# -----------------------------
# Stable label schema
# -----------------------------
LABEL_OTHER = 0
LABEL_AORTA_TRUNK = 1
LABEL_RIGHT_ILIAC = 2
LABEL_LEFT_ILIAC = 3
LABEL_RIGHT_RENAL = 4
LABEL_LEFT_RENAL = 5

LABEL_ID_TO_NAME: Dict[int, str] = {
    LABEL_OTHER: "other",
    LABEL_AORTA_TRUNK: "abdominal_aorta_trunk",
    LABEL_RIGHT_ILIAC: "right_main_iliac",
    LABEL_LEFT_ILIAC: "left_main_iliac",
    LABEL_RIGHT_RENAL: "right_renal",
    LABEL_LEFT_RENAL: "left_renal",
}

LABEL_NAME_TO_ID: Dict[str, int] = {v: k for k, v in LABEL_ID_TO_NAME.items()}

EPS = 1e-12

# -----------------------------
# VTK import (required at runtime)
# -----------------------------
_VTK_IMPORT_ERROR = ""
try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk  # type: ignore
except Exception as e:  # pragma: no cover
    vtk = None
    vtk_to_numpy = None
    numpy_to_vtk = None
    _VTK_IMPORT_ERROR = str(e)


def require_vtk() -> None:
    if vtk is None:
        raise RuntimeError(f"VTK import failed: {_VTK_IMPORT_ERROR}")


# -----------------------------
# VMTK import (Windows-friendly)
# -----------------------------
_WINDOWS_DLL_DIR_HANDLES: List[Any] = []
_WINDOWS_DLL_DIRECTORIES_ADDED: List[str] = []
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


def _prepare_windows_dll_search_paths() -> Dict[str, Any]:
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
    for raw_prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix, os.path.dirname(sys.executable)):
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
            info["dll_directories_attempted"].append(candidate_abs)

            if hasattr(os, "add_dll_directory") and key not in added_dir_keys:
                try:
                    _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(candidate_abs))
                    _WINDOWS_DLL_DIRECTORIES_ADDED.append(candidate_abs)
                    added_dir_keys.add(key)
                    info["dll_directories_added"].append(candidate_abs)
                except Exception as exc:
                    info["dll_add_errors"][candidate_abs] = f"{type(exc).__name__}: {exc}"

            if key not in current_path_keys:
                prepend_entries.append(candidate_abs)
                current_path_keys.add(key)
                info["path_prepended"].append(candidate_abs)

    if prepend_entries:
        os.environ["PATH"] = os.pathsep.join(prepend_entries + current_path_entries)

    return info


def _format_vmtk_import_failure_details(diagnostics: Dict[str, Any]) -> str:
    attempts = diagnostics.get("import_attempts", [])
    attempted_names = [str(a.get("name", "")) for a in attempts]
    lines = [
        "VMTK import failed (vtkvmtk required).",
        "This usually means a dependent VMTK DLL/SO could not be found even though the package is installed.",
        f"Attempted import paths: {', '.join(attempted_names) if attempted_names else '<none>'}",
        f"Python executable: {diagnostics.get('python_executable', sys.executable)}",
        f"sys.prefix: {diagnostics.get('sys_prefix', sys.prefix)}",
        f"CONDA_PREFIX: {diagnostics.get('conda_prefix') or '<unset>'}",
        f"VTK import ok: {bool(diagnostics.get('vtk_import_ok'))}",
        f"DLL directories added: {', '.join(diagnostics.get('dll_directories_added', [])) or '<none>'}",
    ]
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
        "dll_directories_added": dll_info.get("dll_directories_added", []),
        "vtk_import_ok": vtk is not None,
        "vtk_import_error": None if vtk is not None else _VTK_IMPORT_ERROR,
        "import_attempts": [],
        "vmtk_import_ok": False,
        "resolved_vmtk_source": None,
    }

    attempts: List[Dict[str, Any]] = diagnostics["import_attempts"]

    def _attempt(name: str, importer: Any) -> Optional[Any]:
        try:
            module_obj = importer()
            attempts.append({"name": name, "ok": True, "module": getattr(module_obj, "__name__", type(module_obj).__name__)})
            return module_obj
        except Exception as exc:
            attempts.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return None

    direct_module = _attempt("import vmtk.vtkvmtk", lambda: importlib.import_module("vmtk.vtkvmtk"))
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

    return None, diagnostics


# -----------------------------
# Numeric helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros((3,), dtype=float)
    return (v / n).astype(float)


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        c = np.mean(pts, axis=0) if pts.shape[0] else np.zeros((3,), dtype=float)
        return np.eye(3, dtype=float), np.ones((3,), dtype=float), c
    c = np.mean(pts, axis=0)
    X = pts - c
    cov = (X.T @ X) / max(1, X.shape[0])
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    e0 = unit(V[:, 0])
    e1 = unit(V[:, 1] - np.dot(V[:, 1], e0) * e0)
    e2 = unit(np.cross(e0, e1))
    A = np.column_stack([e0, e1, e2]).astype(float)
    return A, w.astype(float), c.astype(float)


def project_vector_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = unit(np.asarray(n, dtype=float).reshape(3))
    return (v - np.dot(v, n) * n).astype(float)


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def compute_abscissa(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros((pts.shape[0],), dtype=float)
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.zeros((pts.shape[0],), dtype=float)
    s[1:] = np.cumsum(d)
    return s


def planar_polygon_area_and_normal(points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, np.zeros((3,), dtype=float), float("nan")
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-8 * (np.linalg.norm(pts[0]) + 1.0):
        pts = pts[:-1]
    if pts.shape[0] < 3:
        return 0.0, np.zeros((3,), dtype=float), float("nan")

    A, _, c = pca_axes(pts)
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


# -----------------------------
# VTK IO / utilities
# -----------------------------
def _resolve_user_path(path: str) -> str:
    path = (path or "").strip()
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.abspath(path)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    return os.path.abspath(os.path.join(script_dir, path))


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


def clone_polydata(pd: "vtkPolyData") -> "vtkPolyData":
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    out.BuildLinks()
    return out


def clean_and_triangulate_surface(pd: "vtkPolyData") -> "vtkPolyData":
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.PointMergingOn()
    cleaner.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cleaner.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(tri.GetOutput())
    out.BuildLinks()
    return out


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


def apply_transform_points(points: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    return (pts - origin[None, :]) @ R.T


def apply_transform_to_polydata(pd: "vtkPolyData", R: np.ndarray, origin: np.ndarray) -> "vtkPolyData":
    R = np.asarray(R, dtype=float).reshape(3, 3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    M = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            M.SetElement(i, j, float(R[i, j]))
        M.SetElement(i, 3, float(-np.dot(R[i, :], origin)))
    M.SetElement(3, 0, 0.0)
    M.SetElement(3, 1, 0.0)
    M.SetElement(3, 2, 0.0)
    M.SetElement(3, 3, 1.0)

    tfm = vtk.vtkTransform()
    tfm.SetMatrix(M)

    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(tfm)
    f.SetInputData(pd)
    f.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(f.GetOutput())
    out.BuildLinks()
    return out


def get_cell_centers_numpy(pd: "vtkPolyData") -> np.ndarray:
    centers_f = vtk.vtkCellCenters()
    centers_f.SetInputData(pd)
    centers_f.VertexCellsOn()
    centers_f.Update()
    out = centers_f.GetOutput()
    pts = out.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    return vtk_to_numpy(pts.GetData()).astype(float)


def get_cell_areas_numpy(pd: "vtkPolyData") -> np.ndarray:
    size_f = vtk.vtkCellSizeFilter()
    size_f.SetInputData(pd)
    size_f.SetComputeArea(True)
    size_f.SetComputeLength(False)
    size_f.SetComputeVolume(False)
    size_f.SetComputeVertexCount(False)
    size_f.Update()
    out = size_f.GetOutput()
    arr = out.GetCellData().GetArray("Area")
    if arr is None:
        return np.ones((pd.GetNumberOfCells(),), dtype=float)
    return vtk_to_numpy(arr).astype(float)


def add_scalar_array_to_cell_data(cd: "vtkCellData", name: str, values: Iterable[float], vtk_type: int) -> None:
    arr = vtk.vtkDataArray.CreateDataArray(vtk_type)
    arr.SetName(name)
    vals = list(values)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(len(vals))
    for i, v in enumerate(vals):
        arr.SetTuple1(i, float(v))
    cd.AddArray(arr)


def add_string_array_to_cell_data(cd: "vtkCellData", name: str, values: Iterable[str]) -> None:
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    vals = list(values)
    arr.SetNumberOfValues(len(vals))
    for i, v in enumerate(vals):
        arr.SetValue(i, str(v))
    cd.AddArray(arr)


# -----------------------------
# Terminations (end holes/caps)
# -----------------------------
@dataclass(frozen=True)
class TerminationLoop:
    center: np.ndarray
    area: float
    diameter_eq: float
    normal: np.ndarray
    rms_planarity: float
    n_points: int
    source: str


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


def find_face_partition_array_name(pd: "vtkPolyData") -> Optional[str]:
    cd = pd.GetCellData()
    if cd is None:
        return None
    for i in range(cd.GetNumberOfArrays()):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if name and name.lower() == "modelfaceid":
            return name
    for i in range(cd.GetNumberOfArrays()):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if not name:
            continue
        lname = name.lower()
        if ("face" in lname and "id" in lname) or lname.endswith("faceid"):
            return name
    return None


def termination_candidates_from_face_partitions(pd_tri: "vtkPolyData", face_array: str) -> List[TerminationLoop]:
    candidates: List[TerminationLoop] = []
    cd = pd_tri.GetCellData()
    if cd is None:
        return candidates
    arr = cd.GetArray(face_array)
    if arr is None or vtk_to_numpy is None:
        return candidates

    size_f = vtk.vtkCellSizeFilter()
    size_f.SetInputData(pd_tri)
    size_f.SetComputeArea(True)
    size_f.SetComputeLength(False)
    size_f.SetComputeVolume(False)
    size_f.SetComputeVertexCount(False)
    size_f.Update()
    pd_area = size_f.GetOutput()

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
    normals_arr = pd_n.GetCellData().GetNormals()
    if normals_arr is None:
        normals_arr = pd_n.GetCellData().GetArray("Normals")
    if normals_arr is None:
        return candidates
    normal_vals = vtk_to_numpy(normals_arr).astype(float)
    centers_vals = vtk_to_numpy(centers_pts.GetData()).astype(float)

    if centers_vals.shape[0] != face_vals.shape[0]:
        return candidates

    total_area = float(np.sum(area_vals)) if area_vals.size else 0.0
    if total_area <= 0.0:
        return candidates

    face_stats: List[Dict[str, Any]] = []
    for fid in np.unique(face_vals):
        mask = face_vals == fid
        if not np.any(mask):
            continue
        a = area_vals[mask]
        a_sum = float(np.sum(a))
        if a_sum <= 0.0:
            continue
        c = np.sum(centers_vals[mask] * a[:, None], axis=0) / (a_sum + EPS)
        n_sum = np.sum(normal_vals[mask] * a[:, None], axis=0)
        planarity = float(np.linalg.norm(n_sum) / (a_sum + EPS))
        diameter_eq = math.sqrt(4.0 * a_sum / math.pi)
        face_stats.append({"fid": int(fid), "area": a_sum, "center": c, "planarity": planarity, "diameter_eq": diameter_eq})

    if not face_stats:
        return candidates

    areas = np.array([fs["area"] for fs in face_stats], dtype=float)
    max_area = float(np.max(areas)) if areas.size else 0.0

    for fs in face_stats:
        if float(fs["planarity"]) < 0.92:
            continue
        if float(fs["area"]) > 0.60 * total_area:
            continue
        if max_area > 0 and float(fs["area"]) > 0.88 * max_area and len(face_stats) > 3:
            continue
        candidates.append(
            TerminationLoop(
                center=np.array(fs["center"], dtype=float),
                area=float(fs["area"]),
                diameter_eq=float(fs["diameter_eq"]),
                normal=np.zeros((3,), dtype=float),
                rms_planarity=float("nan"),
                n_points=0,
                source=f"face_partition:{face_array}",
            )
        )
    return candidates


def detect_terminations(pd_tri: "vtkPolyData", warnings: List[str]) -> Tuple[List[TerminationLoop], str]:
    if count_boundary_edges(pd_tri) > 0:
        loops = extract_boundary_loops(pd_tri)
        if len(loops) >= 2:
            return loops, "open_termini"

    face_array = find_face_partition_array_name(pd_tri)
    if face_array:
        terms = termination_candidates_from_face_partitions(pd_tri, face_array)
        if len(terms) >= 2:
            warnings.append(f"W_TERMINATIONS_FACEPART: boundary loops not found; using planar face partitions via '{face_array}'.")
            return terms, "capped_partitioned"

    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd_tri)
    fe.BoundaryEdgesOff()
    fe.FeatureEdgesOn()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.SetFeatureAngle(60.0)
    fe.Update()
    edges = fe.GetOutput()
    if edges is not None and edges.GetNumberOfCells() > 0:
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
                warnings.append("W_TERMINATIONS_FEATUREEDGES: boundary loops not found; using feature-edge loops (low confidence).")
                return loops, "closed_unpartitioned"

    warnings.append("W_TERMINATIONS_NONE: failed to detect terminations robustly.")
    return [], "unsupported"


# -----------------------------
# VMTK centerlines
# -----------------------------
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
    capped.BuildLinks()
    return capped, True


def build_static_locator(pd: "vtkPolyData") -> "vtkStaticPointLocator":
    loc = vtk.vtkStaticPointLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    return loc


def clean_centerlines_preserve_points(cl: "vtkPolyData") -> "vtkPolyData":
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(cl)
    cleaner.PointMergingOn()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    out.BuildLinks()
    return out


def compute_centerlines_vmtk(
    pd_tri: "vtkPolyData",
    terminations: List[TerminationLoop],
    warnings: List[str],
) -> Tuple["vtkPolyData", Dict[str, Any]]:
    vtkvmtk_mod, diag = resolve_vmtk_import()
    if vtkvmtk_mod is None:
        raise RuntimeError(_format_vmtk_import_failure_details(diag))
    capped, did_cap = cap_surface_if_open(pd_tri, vtkvmtk_mod)
    locator = build_static_locator(capped)

    order = sorted(
        range(len(terminations)),
        key=lambda i: (-float(terminations[i].diameter_eq), -float(terminations[i].area), float(terminations[i].center[0]), float(terminations[i].center[1]), float(terminations[i].center[2])),
    )
    src_idx = int(order[0])
    src_center = np.asarray(terminations[src_idx].center, dtype=float).reshape(3)
    src_pid = int(locator.FindClosestPoint(float(src_center[0]), float(src_center[1]), float(src_center[2])))

    target_pids: List[int] = []
    seen = {src_pid}
    for i in order[1:]:
        c = np.asarray(terminations[int(i)].center, dtype=float).reshape(3)
        pid = int(locator.FindClosestPoint(float(c[0]), float(c[1]), float(c[2])))
        if pid in seen:
            continue
        seen.add(pid)
        target_pids.append(pid)

    if len(target_pids) < 1:
        raise RuntimeError("Insufficient target seeds for centerline extraction (need >=1).")

    bbox = capped.GetBounds()
    diag_len = float(np.linalg.norm(np.array([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float)))
    step = max(0.004 * max(diag_len, 1.0), 0.35)

    info: Dict[str, Any] = {
        "did_cap": bool(did_cap),
        "source_termination_index": int(src_idx),
        "source_pid": int(src_pid),
        "n_targets": int(len(target_pids)),
        "resampling_step": float(step),
        "vmtk_import_source": diag.get("resolved_vmtk_source"),
    }

    source_ids = vtk.vtkIdList()
    source_ids.InsertNextId(src_pid)
    target_ids = vtk.vtkIdList()
    for pid in target_pids:
        target_ids.InsertNextId(int(pid))

    last_err: Optional[BaseException] = None
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

            out_clean = clean_centerlines_preserve_points(out)
            if out_clean.GetNumberOfPoints() < 2 or out_clean.GetNumberOfCells() < 1:
                raise RuntimeError("Centerlines became empty after cleaning.")

            info["flip_normals"] = int(flip)
            info["n_points"] = int(out_clean.GetNumberOfPoints())
            info["n_cells"] = int(out_clean.GetNumberOfCells())
            return out_clean, info
        except Exception as exc:
            last_err = exc
            warnings.append(f"W_VMTK_CENTERLINES_FAIL_FLIP{flip}: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Centerline extraction failed for all FlipNormals attempts. Last error: {last_err}")


# -----------------------------
# Centerline graph + tree helpers
# -----------------------------
def node_degrees(adjacency: Dict[int, Dict[int, float]]) -> Dict[int, int]:
    return {int(n): int(len(nei)) for n, nei in adjacency.items()}


def edge_key(a: int, b: int) -> Tuple[int, int]:
    aa, bb = int(a), int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def build_graph_from_polyline_centerlines(cl: "vtkPolyData") -> Tuple[Dict[int, Dict[int, float]], np.ndarray, np.ndarray]:
    pts = get_points_numpy(cl)
    radii = np.zeros((pts.shape[0],), dtype=float)
    rad_arr = cl.GetPointData().GetArray("MaximumInscribedSphereRadius") if cl.GetPointData() is not None else None
    if rad_arr is not None and vtk_to_numpy is not None:
        try:
            radii = vtk_to_numpy(rad_arr).astype(float).reshape(-1)
        except Exception:
            radii = np.zeros((pts.shape[0],), dtype=float)

    adjacency: Dict[int, Dict[int, float]] = {}
    for ci in range(cl.GetNumberOfCells()):
        cell = cl.GetCell(ci)
        if cell is None:
            continue
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
            continue
        nids = cell.GetNumberOfPoints()
        if nids < 2:
            continue
        ids = [int(cell.GetPointId(k)) for k in range(nids)]
        for u, v in zip(ids[:-1], ids[1:]):
            if u == v:
                continue
            w = float(np.linalg.norm(pts[u] - pts[v]))
            if w <= 0.0:
                continue
            adjacency.setdefault(int(u), {})
            adjacency.setdefault(int(v), {})
            if int(v) not in adjacency[int(u)] or w < adjacency[int(u)][int(v)]:
                adjacency[int(u)][int(v)] = w
                adjacency[int(v)][int(u)] = w
    return adjacency, pts, radii


def connected_component_nodes(adjacency: Dict[int, Dict[int, float]], start: int) -> Set[int]:
    start = int(start)
    if start not in adjacency:
        return set()
    seen: Set[int] = set()
    stack = [start]
    while stack:
        u = int(stack.pop())
        if u in seen:
            continue
        seen.add(u)
        for v in adjacency.get(u, {}).keys():
            if int(v) not in seen:
                stack.append(int(v))
    return seen


def induced_subgraph(adjacency: Dict[int, Dict[int, float]], nodes: Set[int]) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {}
    for u in nodes:
        nbrs = {int(v): float(w) for v, w in adjacency.get(int(u), {}).items() if int(v) in nodes}
        out[int(u)] = nbrs
    return out


def dijkstra(adjacency: Dict[int, Dict[int, float]], start: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    import heapq
    start = int(start)
    dist: Dict[int, float] = {start: 0.0}
    prev: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, start)]
    visited: Set[int] = set()
    while heap:
        d, u = heapq.heappop(heap)
        u = int(u)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adjacency.get(u, {}).items():
            v = int(v)
            nd = float(d + float(w))
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def path_to_root(prev: Dict[int, int], root: int, node: int) -> List[int]:
    root = int(root)
    node = int(node)
    if node == root:
        return [root]
    if node not in prev:
        return []
    path = [node]
    cur = node
    seen = {cur}
    while cur != root:
        if cur not in prev:
            return []
        cur = int(prev[cur])
        if cur in seen:
            return []
        seen.add(cur)
        path.append(cur)
    path.reverse()
    return path


def build_rooted_child_map(prev: Dict[int, int]) -> Dict[int, List[int]]:
    child_map: Dict[int, List[int]] = {}
    for node, parent in prev.items():
        child_map.setdefault(int(parent), []).append(int(node))
    for p in list(child_map.keys()):
        child_map[p] = sorted(child_map[p])
    return child_map


def collect_rooted_subtree_nodes(child_map: Dict[int, List[int]], start: int) -> Set[int]:
    start = int(start)
    seen: Set[int] = set()
    stack = [start]
    while stack:
        u = int(stack.pop())
        if u in seen:
            continue
        seen.add(u)
        for v in child_map.get(u, []):
            if int(v) not in seen:
                stack.append(int(v))
    return seen


def deepest_common_node(path_a: List[int], path_b: List[int], dist: Dict[int, float]) -> Optional[int]:
    if not path_a or not path_b:
        return None
    shared = set(int(n) for n in path_a) & set(int(n) for n in path_b)
    if not shared:
        return None
    return int(max(shared, key=lambda n: float(dist.get(int(n), -1.0))))


@dataclass
class SubtreeSummary:
    parent: int
    start: int
    nodes: Set[int]
    endpoints: List[int]
    representative_endpoint: int
    max_length: float
    total_length: float
    max_endpoint_diameter: float
    mean_top2_diameter: float
    center: np.ndarray
    direction: np.ndarray
    lateral_reach: float


def summarize_rooted_subtree(
    child_map: Dict[int, List[int]],
    pts: np.ndarray,
    radius: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
    parent: int,
    start: int,
    endpoints_set: Set[int],
    endpoint_diam: Dict[int, float],
    axis_hint: np.ndarray,
) -> Optional[SubtreeSummary]:
    parent = int(parent)
    start = int(start)
    if start not in dist or parent not in dist:
        return None
    nodes = collect_rooted_subtree_nodes(child_map, start)
    if not nodes:
        return None

    endpoints = sorted([int(n) for n in nodes if int(n) in endpoints_set])
    if not endpoints:
        endpoints = sorted([int(n) for n in nodes if len(child_map.get(int(n), [])) == 0])
    if not endpoints:
        endpoints = [start]

    rep = int(max(endpoints, key=lambda n: float(dist.get(int(n), float("-inf")))))

    parent_dist = float(dist.get(parent, 0.0))
    max_dist = float(max(float(dist.get(n, parent_dist)) for n in endpoints))
    max_length = max(0.0, max_dist - parent_dist)

    total_length = 0.0
    for n in nodes:
        p = prev.get(int(n))
        if p is None:
            continue
        dn = float(dist.get(int(n), float("nan")))
        dp = float(dist.get(int(p), float("nan")))
        if math.isfinite(dn) and math.isfinite(dp) and dn >= dp:
            total_length += dn - dp

    diams = [float(endpoint_diam.get(int(n), 2.0 * float(radius[int(n)]) if radius.size > int(n) else 0.0)) for n in endpoints]
    diams_sorted = sorted(diams, reverse=True)
    max_d = float(diams_sorted[0]) if diams_sorted else 0.0
    mean_top2 = float(np.mean(diams_sorted[:2])) if diams_sorted else 0.0

    w = np.array([max(float(endpoint_diam.get(int(n), 0.0)), 0.5) for n in endpoints], dtype=float)
    if w.sum() <= EPS:
        w = np.ones((len(endpoints),), dtype=float)
    P = pts[np.array(endpoints, dtype=int)]
    center = (P * w[:, None]).sum(axis=0) / (w.sum() + EPS)
    direction = unit(center - pts[parent])

    ax = unit(axis_hint)
    if np.linalg.norm(ax) < EPS:
        ax = np.array([0.0, 0.0, 1.0], dtype=float)
    reach = 0.0
    for n in endpoints:
        dvec = pts[int(n)] - pts[parent]
        lateral = np.linalg.norm(dvec - np.dot(dvec, ax) * ax)
        reach = max(reach, float(lateral))

    return SubtreeSummary(
        parent=parent,
        start=start,
        nodes=set(int(x) for x in nodes),
        endpoints=[int(x) for x in endpoints],
        representative_endpoint=rep,
        max_length=float(max_length),
        total_length=float(total_length),
        max_endpoint_diameter=float(max_d),
        mean_top2_diameter=float(mean_top2),
        center=center.astype(float),
        direction=direction.astype(float),
        lateral_reach=float(reach),
    )


@dataclass
class IliacBifurcationResult:
    bif_node: int
    child_a: int
    child_b: int
    sys_a: SubtreeSummary
    sys_b: SubtreeSummary
    score: float
    depth_norm: float
    symmetry: float
    lateral_norm: float
    diam_norm: float
    third_child_penalty: float


def find_best_bifurcation_and_iliacs(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    radius: np.ndarray,
    root: int,
    dist: Dict[int, float],
    prev: Dict[int, int],
    endpoints_set: Set[int],
    endpoint_diam: Dict[int, float],
    axis_hint: np.ndarray,
) -> Optional[IliacBifurcationResult]:
    child_map = build_rooted_child_map(prev)
    if not endpoints_set:
        return None

    max_dist = float(max(dist.get(e, 0.0) for e in endpoints_set))
    if max_dist <= EPS:
        return None

    bbox_min = np.min(pts[list(adjacency.keys())], axis=0)
    bbox_max = np.max(pts[list(adjacency.keys())], axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min)) + EPS

    best: Optional[IliacBifurcationResult] = None
    best_score = -1e18

    for b in sorted(child_map.keys()):
        b = int(b)
        if b not in dist:
            continue
        children = [int(c) for c in child_map.get(b, []) if int(c) in dist]
        if len(children) < 2:
            continue

        summaries: List[SubtreeSummary] = []
        for c in children:
            s = summarize_rooted_subtree(child_map, pts, radius, dist, prev, b, c, endpoints_set, endpoint_diam, axis_hint)
            if s is not None and s.max_length > 0.0:
                summaries.append(s)
        if len(summaries) < 2:
            continue

        depth_norm = float(dist.get(b, 0.0)) / (max_dist + EPS)
        if depth_norm < 0.25:
            continue

        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                sa = summaries[i]
                sb = summaries[j]

                len_a, len_b = float(sa.max_length), float(sb.max_length)
                if len_a <= 0.02 * max_dist or len_b <= 0.02 * max_dist:
                    continue

                symmetry = 1.0 - abs(len_a - len_b) / (len_a + len_b + EPS)
                diam_a = float(sa.mean_top2_diameter)
                diam_b = float(sb.mean_top2_diameter)
                diam_sum = diam_a + diam_b
                max_d_all = max(float(max(endpoint_diam.values())) if endpoint_diam else 0.0, 0.0)
                diam_norm = (diam_sum / (2.0 * (max_d_all + EPS))) if max_d_all > EPS else 0.0
                diam_sym = 1.0 - abs(diam_a - diam_b) / (diam_sum + EPS)

                dvec = sa.center - sb.center
                ax = unit(axis_hint)
                lateral = float(np.linalg.norm(dvec - np.dot(dvec, ax) * ax))
                lateral_norm = lateral / diag

                divergence = float(clamp((1.0 - float(np.dot(unit(sa.direction), unit(sb.direction)))) / 2.0, 0.0, 1.0))

                sorted_by_mass = sorted(
                    summaries,
                    key=lambda s: (float(s.max_length), float(s.mean_top2_diameter), float(s.total_length)),
                    reverse=True,
                )
                third_pen = 0.0
                if len(sorted_by_mass) >= 3:
                    third = sorted_by_mass[2]
                    third_pen = float(clamp(third.max_length / (min(len_a, len_b) + EPS) - 0.45, 0.0, 1.0))

                proximal_penalty = float(clamp((0.55 - depth_norm) / 0.55, 0.0, 1.0))

                score = (
                    2.25 * depth_norm
                    + 1.55 * symmetry
                    + 1.10 * diam_norm
                    + 0.65 * diam_sym
                    + 1.55 * lateral_norm
                    + 0.45 * divergence
                    - 1.10 * proximal_penalty
                    - 0.85 * third_pen
                )

                if score > best_score:
                    best_score = score
                    best = IliacBifurcationResult(
                        bif_node=b,
                        child_a=int(sa.start),
                        child_b=int(sb.start),
                        sys_a=sa,
                        sys_b=sb,
                        score=float(score),
                        depth_norm=float(depth_norm),
                        symmetry=float(symmetry),
                        lateral_norm=float(lateral_norm),
                        diam_norm=float(diam_norm),
                        third_child_penalty=float(third_pen),
                    )

    return best


# -----------------------------
# Anatomy solver: inlet, bifurcation, iliacs, renals, frame
# -----------------------------
@dataclass
class RenalCandidate:
    takeoff: int
    child: int
    nodes: Set[int]
    rep_endpoint: int
    takeoff_dist: float
    trunk_rel_s: float
    branch_len: float
    direction: np.ndarray
    horiz: float
    lateral_reach: float
    endpoint_diameter: float
    score: float


def trunk_tangent_at_node(trunk_path: List[int], pts: np.ndarray, node: int) -> np.ndarray:
    node = int(node)
    if node not in trunk_path:
        return np.zeros((3,), dtype=float)
    idx = trunk_path.index(node)
    if idx == 0 and len(trunk_path) >= 2:
        return unit(pts[int(trunk_path[1])] - pts[int(trunk_path[0])])
    if idx == len(trunk_path) - 1 and len(trunk_path) >= 2:
        return unit(pts[int(trunk_path[-1])] - pts[int(trunk_path[-2])])
    if 0 < idx < len(trunk_path) - 1:
        return unit(pts[int(trunk_path[idx + 1])] - pts[int(trunk_path[idx - 1])])
    return np.zeros((3,), dtype=float)


def discover_renal_candidates(
    child_map: Dict[int, List[int]],
    pts: np.ndarray,
    radius: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
    trunk_path: List[int],
    trunk_nodes: Set[int],
    exclude_nodes: Set[int],
    endpoints_set: Set[int],
    endpoint_diam: Dict[int, float],
    axis_hint: np.ndarray,
) -> List[RenalCandidate]:
    candidates: List[RenalCandidate] = []
    trunk_len = float(dist.get(int(trunk_path[-1]), 0.0)) if trunk_path else 0.0
    if trunk_len <= EPS:
        return candidates

    ax = unit(axis_hint)
    if np.linalg.norm(ax) < EPS:
        ax = unit(pts[int(trunk_path[0])] - pts[int(trunk_path[-1])]) if trunk_path else np.array([0.0, 0.0, 1.0], dtype=float)

    for t in trunk_path[1:-1]:
        t = int(t)
        children = [int(c) for c in child_map.get(t, [])]
        for c in children:
            if int(c) in trunk_nodes:
                continue
            if int(c) in exclude_nodes:
                continue
            nodes = collect_rooted_subtree_nodes(child_map, c)
            if not nodes:
                continue
            if nodes & exclude_nodes:
                continue

            eps_in_sub = [int(n) for n in nodes if int(n) in endpoints_set]
            if not eps_in_sub:
                eps_in_sub = [int(n) for n in nodes if len(child_map.get(int(n), [])) == 0]
            if not eps_in_sub:
                eps_in_sub = [c]
            rep = int(max(eps_in_sub, key=lambda n: float(dist.get(int(n), float("-inf")))))

            takeoff_dist = float(dist.get(t, 0.0))
            s_rel = float(takeoff_dist / (trunk_len + EPS))
            if s_rel < 0.12 or s_rel > 0.85:
                pos_score = clamp(1.0 - min(abs(s_rel - 0.35) / 0.28, abs(s_rel - 0.55) / 0.33), 0.0, 1.0)
            else:
                pos_score = clamp(1.0 - abs(s_rel - 0.42) / 0.34, 0.0, 1.0)

            branch_maxdist = float(max(float(dist.get(n, takeoff_dist)) for n in eps_in_sub))
            branch_len = max(0.0, branch_maxdist - takeoff_dist)
            if branch_len < 0.03 * trunk_len:
                continue
            if branch_len > 0.85 * trunk_len:
                continue

            dir_vec = unit(pts[int(c)] - pts[t])
            if np.linalg.norm(dir_vec) < EPS:
                dir_vec = unit(pts[int(rep)] - pts[t])

            tt = trunk_tangent_at_node(trunk_path, pts, t)
            if np.linalg.norm(tt) < EPS:
                tt = ax

            horiz = float(clamp(1.0 - abs(float(np.dot(unit(dir_vec), unit(tt)))), 0.0, 1.0))

            reach = 0.0
            for n in eps_in_sub:
                dv = pts[int(n)] - pts[t]
                reach = max(reach, float(np.linalg.norm(dv - np.dot(dv, tt) * tt)))
            lateral_norm = float(clamp(reach / (0.35 * trunk_len + EPS), 0.0, 1.0))

            diam = float(endpoint_diam.get(rep, 2.0 * float(radius[int(rep)]) if radius.size > int(rep) else 0.0))
            diam_norm = float(clamp(diam / (float(max(endpoint_diam.values())) + EPS) if endpoint_diam else diam / (diam + 1.0), 0.0, 1.0))
            size_score = float(clamp(1.0 - abs(diam_norm - 0.35) / 0.45, 0.0, 1.0))

            score = 1.4 * horiz + 1.15 * pos_score + 0.65 * lateral_norm + 0.45 * size_score + 0.25 * clamp(branch_len / (0.25 * trunk_len + EPS), 0.0, 1.0)
            if score < 1.2:
                continue

            candidates.append(
                RenalCandidate(
                    takeoff=t,
                    child=c,
                    nodes=set(int(x) for x in nodes),
                    rep_endpoint=rep,
                    takeoff_dist=takeoff_dist,
                    trunk_rel_s=s_rel,
                    branch_len=branch_len,
                    direction=dir_vec.astype(float),
                    horiz=horiz,
                    lateral_reach=reach,
                    endpoint_diameter=diam,
                    score=float(score),
                )
            )

    candidates.sort(key=lambda c: (float(c.score), float(c.horiz), float(c.branch_len), float(c.endpoint_diameter), -int(c.takeoff)), reverse=True)
    return candidates


def choose_best_renal_assignment(
    candidates: List[RenalCandidate],
    pts_c: np.ndarray,
    trunk_len: float,
) -> Tuple[Optional[RenalCandidate], Optional[RenalCandidate], float, Dict[str, Any]]:
    diag: Dict[str, Any] = {"candidate_count": int(len(candidates))}
    if not candidates:
        return None, None, 0.0, diag

    x_thr = max(0.02 * float(trunk_len), 1.0)
    right_side: List[RenalCandidate] = []
    left_side: List[RenalCandidate] = []
    mid: List[RenalCandidate] = []
    for c in candidates:
        x = float(pts_c[int(c.rep_endpoint)][0])
        if x > x_thr:
            right_side.append(c)
        elif x < -x_thr:
            left_side.append(c)
        else:
            mid.append(c)

    right_best = right_side[0] if right_side else None
    left_best = left_side[0] if left_side else None

    if right_best is None and mid:
        right_best = mid[0]
        diag["right_from_midline"] = True
    if left_best is None and mid:
        for c in mid:
            if right_best is None or int(c.takeoff) != int(right_best.takeoff):
                left_best = c
                diag["left_from_midline"] = True
                break

    conf = 0.0
    if right_best is None and left_best is None:
        return None, None, 0.0, diag

    if right_best is not None and left_best is not None:
        dt = abs(float(right_best.takeoff_dist) - float(left_best.takeoff_dist)) / (trunk_len + EPS)
        takeoff_sim = clamp(1.0 - dt / 0.28, 0.0, 1.0)
        dl = abs(float(right_best.branch_len) - float(left_best.branch_len)) / (float(right_best.branch_len) + float(left_best.branch_len) + EPS)
        len_sim = clamp(1.0 - dl, 0.0, 1.0)
        z_r = float(pts_c[int(right_best.takeoff)][2])
        z_l = float(pts_c[int(left_best.takeoff)][2])
        asym = clamp(0.5 + 0.5 * clamp((z_l - z_r) / (0.07 * trunk_len + EPS), -1.0, 1.0), 0.0, 1.0)
        side_sep = abs(float(pts_c[int(right_best.rep_endpoint)][0]) - float(pts_c[int(left_best.rep_endpoint)][0]))
        sep_score = clamp(side_sep / (0.25 * trunk_len + EPS), 0.0, 1.0)
        base = 0.55 * takeoff_sim + 0.25 * len_sim + 0.20 * sep_score
        conf = float(clamp(0.35 + 0.45 * base + 0.20 * asym, 0.0, 1.0))
        diag.update(
            {
                "pair_takeoff_sim": float(takeoff_sim),
                "pair_len_sim": float(len_sim),
                "pair_sep_score": float(sep_score),
                "pair_left_minus_right_z": float(z_l - z_r),
                "pair_asym_prior": float(asym),
            }
        )
    else:
        single = right_best if right_best is not None else left_best
        conf = float(clamp(0.25 + 0.45 * clamp(float(single.score) / 3.0, 0.0, 1.0), 0.0, 0.8))
        diag["single_side_only"] = True

    return right_best, left_best, conf, diag


@dataclass
class AnatomySolveResult:
    inlet_node: int
    bif_node: int
    iliac_root_right: int
    iliac_root_left: int
    iliac_rep_right: int
    iliac_rep_left: int
    trunk_path: List[int]
    dist: Dict[int, float]
    prev: Dict[int, int]
    child_map: Dict[int, List[int]]
    trunk_nodes: Set[int]
    iliac_nodes_right: Set[int]
    iliac_nodes_left: Set[int]
    renal_right: Optional[RenalCandidate]
    renal_left: Optional[RenalCandidate]
    renal_nodes_right: Set[int]
    renal_nodes_left: Set[int]
    R: np.ndarray
    origin: np.ndarray
    frame_conf: float
    inlet_conf: float
    bif_conf: float
    laterality_conf: float
    renal_conf: float
    ap_conf: float
    AP_direction_hint: Optional[float]
    warnings: List[str]
    debug: Dict[str, Any]


def solve_anatomy(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    radius: np.ndarray,
    terminations: List[TerminationLoop],
    warnings: List[str],
) -> AnatomySolveResult:
    deg = node_degrees(adjacency)
    endpoints = sorted([int(n) for n, d in deg.items() if int(d) == 1])
    if len(endpoints) < 2:
        raise RuntimeError("Centerline graph has insufficient endpoints.")

    A, _, _ = pca_axes(pts[list(adjacency.keys())])
    axis_hint = unit(A[:, 0])
    if np.linalg.norm(axis_hint) < EPS:
        axis_hint = np.array([0.0, 0.0, 1.0], dtype=float)

    term_centers = np.array([t.center for t in terminations], dtype=float) if terminations else np.zeros((0, 3), dtype=float)

    bbox_min = np.min(pts[list(adjacency.keys())], axis=0)
    bbox_max = np.max(pts[list(adjacency.keys())], axis=0)
    diag_len = float(np.linalg.norm(bbox_max - bbox_min)) + EPS
    match_thresh = 0.08 * diag_len

    endpoint_diam: Dict[int, float] = {}
    endpoint_term: Dict[int, int] = {}
    if term_centers.shape[0] >= 1:
        for ep in endpoints:
            p = pts[int(ep)]
            d2 = np.sum((term_centers - p[None, :]) ** 2, axis=1)
            ti = int(np.argmin(d2))
            d = math.sqrt(float(d2[ti]))
            if d <= match_thresh:
                endpoint_term[int(ep)] = int(ti)
                endpoint_diam[int(ep)] = float(terminations[int(ti)].diameter_eq)
            else:
                endpoint_diam[int(ep)] = float(2.0 * radius[int(ep)]) if radius.size > int(ep) else 0.0
    else:
        for ep in endpoints:
            endpoint_diam[int(ep)] = float(2.0 * radius[int(ep)]) if radius.size > int(ep) else 0.0

    endpoints_set = set(endpoints)
    max_diam = max(endpoint_diam.values()) if endpoint_diam else 0.0

    cand_rows: List[Dict[str, Any]] = []
    best_overall_score = -1e18
    best_root: Optional[int] = None
    best_bif: Optional[IliacBifurcationResult] = None
    best_dist: Dict[int, float] = {}
    best_prev: Dict[int, int] = {}

    for root in endpoints:
        dist, prev = dijkstra(adjacency, root)
        reachable_eps = {e for e in endpoints_set if e in dist and e != root}
        if len(reachable_eps) < 2:
            continue
        bif = find_best_bifurcation_and_iliacs(adjacency, pts, radius, root, dist, prev, reachable_eps, endpoint_diam, axis_hint)
        if bif is None:
            score = -1e9
            cand_rows.append({"root": int(root), "score": float(score), "has_bif": False})
            continue
        root_d = float(endpoint_diam.get(int(root), 0.0))
        root_d_norm = root_d / (max_diam + EPS) if max_diam > EPS else 0.0
        score = float(bif.score + 0.35 * root_d_norm)
        cand_rows.append(
            {
                "root": int(root),
                "score": float(score),
                "bif": int(bif.bif_node),
                "bif_score": float(bif.score),
                "depth_norm": float(bif.depth_norm),
                "symmetry": float(bif.symmetry),
                "lateral_norm": float(bif.lateral_norm),
                "diam_norm": float(bif.diam_norm),
            }
        )
        if score > best_overall_score:
            best_overall_score = score
            best_root = int(root)
            best_bif = bif
            best_dist = dist
            best_prev = prev

    if best_root is None or best_bif is None:
        warnings.append("W_SOLVER_ROOT_BIF_FAIL: multi-hypothesis inlet/bif solve failed; using farthest-endpoint fallback.")
        best_root = int(max(endpoints, key=lambda n: float(endpoint_diam.get(int(n), 0.0))))
        best_dist, best_prev = dijkstra(adjacency, best_root)
        child_map = build_rooted_child_map(best_prev)
        far_eps = sorted([e for e in endpoints if e != best_root and e in best_dist], key=lambda n: float(best_dist.get(int(n), -1.0)), reverse=True)[:2]
        if len(far_eps) < 2:
            raise RuntimeError("Fallback failed: insufficient reachable endpoints.")
        pa = path_to_root(best_prev, best_root, far_eps[0])
        pb = path_to_root(best_prev, best_root, far_eps[1])
        lca = deepest_common_node(pa, pb, best_dist)
        if lca is None:
            raise RuntimeError("Fallback failed: could not find bifurcation LCA.")
        child_map = build_rooted_child_map(best_prev)

        def child_containing(endpoint: int) -> Optional[int]:
            for c in child_map.get(int(lca), []):
                nodes = collect_rooted_subtree_nodes(child_map, int(c))
                if int(endpoint) in nodes:
                    return int(c)
            return None

        ca = child_containing(far_eps[0])
        cb = child_containing(far_eps[1])
        if ca is None or cb is None:
            raise RuntimeError("Fallback failed: could not recover iliac child roots.")
        sa = summarize_rooted_subtree(child_map, pts, radius, best_dist, best_prev, int(lca), int(ca), endpoints_set, endpoint_diam, axis_hint)
        sb = summarize_rooted_subtree(child_map, pts, radius, best_dist, best_prev, int(lca), int(cb), endpoints_set, endpoint_diam, axis_hint)
        if sa is None or sb is None:
            raise RuntimeError("Fallback failed: could not summarize iliac subtrees.")
        best_bif = IliacBifurcationResult(
            bif_node=int(lca),
            child_a=int(ca),
            child_b=int(cb),
            sys_a=sa,
            sys_b=sb,
            score=0.0,
            depth_norm=0.0,
            symmetry=0.0,
            lateral_norm=0.0,
            diam_norm=0.0,
            third_child_penalty=0.0,
        )
        best_overall_score = 0.0
        cand_rows.append({"root": int(best_root), "score": float(best_overall_score), "has_bif": True, "fallback": True})

    sorted_candidates = sorted([r for r in cand_rows if "score" in r], key=lambda r: float(r["score"]), reverse=True)
    inlet_conf = 0.0
    if len(sorted_candidates) >= 2:
        margin = float(sorted_candidates[0]["score"]) - float(sorted_candidates[1]["score"])
        inlet_conf = float(clamp(0.50 + 0.40 * math.tanh(margin / 0.75), 0.0, 1.0))
    else:
        inlet_conf = 0.55
    if inlet_conf < 0.60:
        warnings.append(f"W_INLET_LOWCONF: inlet confidence={inlet_conf:.3f} (multi-hypothesis score margin small).")

    inlet_node = int(best_root)
    bif_node = int(best_bif.bif_node)
    dist = best_dist
    prev = best_prev
    child_map = build_rooted_child_map(prev)

    trunk_path = path_to_root(prev, inlet_node, bif_node)
    if not trunk_path:
        warnings.append("W_TRUNK_PATH_EMPTY: could not reconstruct inlet->bif path; using direct shortest path fallback.")
        trunk_path = [inlet_node, bif_node]
    trunk_nodes = set(int(n) for n in trunk_path)

    sysA = best_bif.sys_a
    sysB = best_bif.sys_b
    repA = int(sysA.representative_endpoint)
    repB = int(sysB.representative_endpoint)

    ez = unit(pts[int(inlet_node)] - pts[int(bif_node)])
    if np.linalg.norm(ez) < EPS and len(trunk_path) >= 2:
        ez = unit(pts[int(trunk_path[-2])] - pts[int(bif_node)])
    if np.linalg.norm(ez) < EPS:
        ez = unit(axis_hint)
    if np.linalg.norm(ez) < EPS:
        ez = np.array([0.0, 0.0, 1.0], dtype=float)

    origin = pts[int(bif_node)].astype(float).copy()

    def build_frame(right_ep: int, left_ep: int) -> Tuple[np.ndarray, float]:
        ex_raw = project_vector_to_plane(pts[int(right_ep)] - pts[int(left_ep)], ez)
        ex = unit(ex_raw)
        if np.linalg.norm(ex) < EPS:
            A2, _, _ = pca_axes(pts[list(adjacency.keys())])
            cand = [unit(project_vector_to_plane(A2[:, k], ez)) for k in range(3)]
            cand = [c for c in cand if np.linalg.norm(c) > 0.5]
            ex = cand[0] if cand else unit(project_vector_to_plane(np.array([1.0, 0.0, 0.0], dtype=float), ez))
        ey = unit(np.cross(ez, ex))
        ex = unit(np.cross(ey, ez))
        ey = unit(np.cross(ez, ex))
        R = np.vstack([ex, ey, ez]).astype(float)
        ortho_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
        conf = float(clamp(1.0 - ortho_err, 0.0, 1.0))
        return R, conf

    nodesA = collect_rooted_subtree_nodes(child_map, int(best_bif.child_a))
    nodesB = collect_rooted_subtree_nodes(child_map, int(best_bif.child_b))
    exclude_base = set(int(n) for n in nodesA | nodesB)
    exclude_for_renal = set(exclude_base)
    exclude_for_renal.discard(int(bif_node))

    trunk_len = float(dist.get(int(bif_node), 0.0))
    renal_candidates = discover_renal_candidates(
        child_map=child_map,
        pts=pts,
        radius=radius,
        dist=dist,
        prev=prev,
        trunk_path=trunk_path,
        trunk_nodes=trunk_nodes,
        exclude_nodes=exclude_for_renal,
        endpoints_set=set(endpoints),
        endpoint_diam=endpoint_diam,
        axis_hint=ez,
    )

    frame_hypos: List[Dict[str, Any]] = []
    for hypo_name, right_child, left_child, right_rep, left_rep in (
        ("A_right", best_bif.child_a, best_bif.child_b, repA, repB),
        ("B_right", best_bif.child_b, best_bif.child_a, repB, repA),
    ):
        R, frame_conf = build_frame(right_rep, left_rep)
        pts_c = apply_transform_points(pts, R, origin)
        rr, lr, renal_conf, renal_diag = choose_best_renal_assignment(renal_candidates, pts_c, trunk_len)
        asym_bonus = 0.0
        if rr is not None and lr is not None:
            asym_bonus = float(clamp((float(pts_c[int(lr.takeoff)][2]) - float(pts_c[int(rr.takeoff)][2])) / (0.05 * trunk_len + EPS), -1.0, 1.0))
        ex = R[0, :]
        x_align = float(np.dot(unit(ex), np.array([1.0, 0.0, 0.0], dtype=float)))
        score = float(best_bif.score + 0.50 * renal_conf + 0.18 * asym_bonus + 0.12 * x_align + 0.10 * frame_conf)
        frame_hypos.append(
            {
                "name": hypo_name,
                "right_child": int(right_child),
                "left_child": int(left_child),
                "right_rep": int(right_rep),
                "left_rep": int(left_rep),
                "R": R,
                "frame_conf": float(frame_conf),
                "renal_right": rr,
                "renal_left": lr,
                "renal_conf": float(renal_conf),
                "renal_diag": renal_diag,
                "x_align": float(x_align),
                "asym_bonus": float(asym_bonus),
                "score": float(score),
            }
        )

    frame_hypos.sort(key=lambda h: float(h["score"]), reverse=True)
    chosen = frame_hypos[0]
    alt = frame_hypos[1] if len(frame_hypos) > 1 else None

    laterality_conf = 0.65
    if alt is not None:
        margin = float(chosen["score"]) - float(alt["score"])
        laterality_conf = float(clamp(0.50 + 0.45 * math.tanh(margin / 0.65), 0.0, 1.0))
    if laterality_conf < 0.60:
        warnings.append(f"W_LATERALITY_LOWCONF: laterality confidence={laterality_conf:.3f}; right/left may be ambiguous.")

    R = np.asarray(chosen["R"], dtype=float).reshape(3, 3)
    frame_conf = float(chosen["frame_conf"])
    rr = chosen.get("renal_right")
    lr = chosen.get("renal_left")
    renal_conf = float(chosen.get("renal_conf", 0.0))

    iliac_nodes_right = collect_rooted_subtree_nodes(child_map, int(chosen["right_child"])) | {int(bif_node)}
    iliac_nodes_left = collect_rooted_subtree_nodes(child_map, int(chosen["left_child"])) | {int(bif_node)}

    renal_nodes_right: Set[int] = set()
    renal_nodes_left: Set[int] = set()
    if rr is not None:
        renal_nodes_right = set(int(n) for n in rr.nodes) | {int(rr.takeoff)}
    if lr is not None:
        renal_nodes_left = set(int(n) for n in lr.nodes) | {int(lr.takeoff)}

    pts_c = apply_transform_points(pts, R, origin)
    ap_votes: List[float] = []
    ap_weights: List[float] = []
    for c in renal_candidates[:min(12, len(renal_candidates))]:
        p_end = pts_c[int(c.rep_endpoint)]
        lateral = abs(float(p_end[0]))
        anterior = abs(float(p_end[1]))
        if anterior > 1.25 * lateral:
            ap_votes.append(math.copysign(1.0, float(p_end[1])))
            ap_weights.append(float(c.score))
    ap_conf = 0.0
    ap_hint = None
    if ap_votes:
        vote = float(np.dot(np.array(ap_votes, dtype=float), np.array(ap_weights, dtype=float))) / (float(np.sum(ap_weights)) + EPS)
        ap_conf = float(clamp(abs(vote), 0.0, 1.0))
        ap_hint = float(vote)
        if ap_conf < 0.60:
            warnings.append(f"W_AP_LOWCONF: AP confidence={ap_conf:.3f}; AP may be mirrored.")

    bif_conf = float(clamp(0.35 + 0.55 * float(best_bif.depth_norm) + 0.25 * float(best_bif.symmetry) + 0.20 * float(best_bif.lateral_norm) + 0.15 * float(best_bif.diam_norm) - 0.30 * float(best_bif.third_child_penalty), 0.0, 1.0))
    if bif_conf < 0.60:
        warnings.append(f"W_BIF_LOWCONF: bifurcation confidence={bif_conf:.3f} (depthN={best_bif.depth_norm:.3f}, symmetry={best_bif.symmetry:.3f}, latN={best_bif.lateral_norm:.3f}).")

    debug = {
        "root_candidates": sorted_candidates[:10],
        "chosen_frame": {k: (v if k not in ("R", "renal_right", "renal_left") else None) for k, v in chosen.items()},
        "frame_hypotheses_scores": [{"name": h["name"], "score": h["score"], "x_align": h["x_align"], "asym_bonus": h["asym_bonus"], "renal_conf": h["renal_conf"]} for h in frame_hypos],
        "renal_candidate_top": [{"takeoff": int(c.takeoff), "rep_endpoint": int(c.rep_endpoint), "score": float(c.score), "s_rel": float(c.trunk_rel_s), "horiz": float(c.horiz), "len": float(c.branch_len)} for c in renal_candidates[:12]],
        "termination_endpoint_match_thresh": float(match_thresh),
    }

    return AnatomySolveResult(
        inlet_node=int(inlet_node),
        bif_node=int(bif_node),
        iliac_root_right=int(chosen["right_child"]),
        iliac_root_left=int(chosen["left_child"]),
        iliac_rep_right=int(chosen["right_rep"]),
        iliac_rep_left=int(chosen["left_rep"]),
        trunk_path=[int(n) for n in trunk_path],
        dist=dist,
        prev=prev,
        child_map=child_map,
        trunk_nodes=set(int(n) for n in trunk_nodes),
        iliac_nodes_right=set(int(n) for n in iliac_nodes_right),
        iliac_nodes_left=set(int(n) for n in iliac_nodes_left),
        renal_right=rr,
        renal_left=lr,
        renal_nodes_right=set(int(n) for n in renal_nodes_right),
        renal_nodes_left=set(int(n) for n in renal_nodes_left),
        R=R,
        origin=origin,
        frame_conf=float(frame_conf),
        inlet_conf=float(inlet_conf),
        bif_conf=float(bif_conf),
        laterality_conf=float(laterality_conf),
        renal_conf=float(renal_conf),
        ap_conf=float(ap_conf),
        AP_direction_hint=ap_hint,
        warnings=warnings,
        debug=debug,
    )


# -----------------------------
# Branch chain decomposition + output polydata
# -----------------------------
def build_branch_chains_from_graph(adjacency: Dict[int, Dict[int, float]]) -> List[List[int]]:
    if not adjacency:
        return []
    deg = node_degrees(adjacency)
    key_nodes = {int(n) for n, d in deg.items() if int(d) != 2}
    visited_edges: Set[Tuple[int, int]] = set()
    chains: List[List[int]] = []

    def walk(start: int, nxt: int) -> List[int]:
        path = [int(start), int(nxt)]
        visited_edges.add(edge_key(start, nxt))
        prev = int(start)
        cur = int(nxt)
        while int(deg.get(cur, 0)) == 2 and cur not in key_nodes:
            nbrs = [int(v) for v in adjacency.get(cur, {}).keys() if int(v) != prev]
            if not nbrs:
                break
            candidate = int(min(nbrs))
            ek = edge_key(cur, candidate)
            if ek in visited_edges:
                break
            visited_edges.add(ek)
            path.append(candidate)
            prev, cur = cur, candidate
        return path

    for start in sorted(key_nodes):
        for nxt in sorted(adjacency.get(int(start), {}).keys()):
            ek = edge_key(start, int(nxt))
            if ek in visited_edges:
                continue
            chains.append(walk(int(start), int(nxt)))

    for u in sorted(adjacency.keys()):
        for v in sorted(adjacency.get(int(u), {}).keys()):
            ek = edge_key(int(u), int(v))
            if ek in visited_edges:
                continue
            chains.append(walk(int(u), int(v)))

    return [c for c in chains if len(c) >= 2]


def build_centerline_scaffold_polydata(
    chains: List[List[int]],
    pts_c: np.ndarray,
    radius: np.ndarray,
    node_label_sets: Dict[int, Set[int]],
    warnings: List[str],
) -> Tuple["vtkPolyData", List[Dict[str, Any]]]:
    prio = {LABEL_RIGHT_RENAL: 0, LABEL_LEFT_RENAL: 0, LABEL_RIGHT_ILIAC: 1, LABEL_LEFT_ILIAC: 1, LABEL_AORTA_TRUNK: 2}

    def edge_owner(u: int, v: int) -> int:
        claims: List[int] = []
        for lid, nodes in node_label_sets.items():
            if lid == LABEL_OTHER:
                continue
            if int(u) in nodes and int(v) in nodes:
                claims.append(int(lid))
        if not claims:
            return LABEL_OTHER
        claims.sort(key=lambda lid: prio.get(int(lid), 9))
        return int(claims[0])

    out = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    branch_ids: List[int] = []
    branch_names: List[str] = []
    branch_lengths: List[float] = []
    geometry_types: List[str] = []
    topo_roles: List[str] = []

    old_to_new: Dict[int, int] = {}

    def ensure_point(old_pid: int) -> int:
        if old_pid in old_to_new:
            return int(old_to_new[old_pid])
        p = pts_c[int(old_pid)]
        nid = int(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
        old_to_new[int(old_pid)] = nid
        return nid

    branch_rows: List[Dict[str, Any]] = []

    out_cell_index = 0
    for chain in chains:
        idx = int(out_cell_index)
        nodes = [int(n) for n in chain]
        edge_labels: List[int] = []
        length = 0.0
        for u, v in zip(nodes[:-1], nodes[1:]):
            lid = edge_owner(u, v)
            edge_labels.append(int(lid))
            length += float(np.linalg.norm(pts_c[int(v)] - pts_c[int(u)]))
        if not edge_labels:
            continue
        counts: Dict[int, int] = {}
        for lid in edge_labels:
            counts[int(lid)] = counts.get(int(lid), 0) + 1
        majority = sorted(counts.items(), key=lambda kv: (int(kv[1]), -prio.get(int(kv[0]), 9)), reverse=True)[0][0]
        lid_chain = int(majority)

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(nodes))
        for k, old_pid in enumerate(nodes):
            new_pid = ensure_point(int(old_pid))
            polyline.GetPointIds().SetId(k, int(new_pid))
        lines.InsertNextCell(polyline)

        branch_ids.append(int(lid_chain))
        branch_names.append(LABEL_ID_TO_NAME.get(int(lid_chain), "other"))
        branch_lengths.append(float(length))
        geometry_types.append("centerline")
        if lid_chain == LABEL_AORTA_TRUNK:
            topo_roles.append("trunk_path_chain")
        elif lid_chain in (LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC):
            topo_roles.append("iliac_system_chain")
        elif lid_chain in (LABEL_RIGHT_RENAL, LABEL_LEFT_RENAL):
            topo_roles.append("renal_system_chain")
        else:
            topo_roles.append("other_chain")

        branch_rows.append(
            {
                "index": int(idx),
                "branch_id": int(lid_chain),
                "branch_name": LABEL_ID_TO_NAME.get(int(lid_chain), "other"),
                "topology_role": topo_roles[-1],
                "length": float(length),
                "point_count": int(len(nodes)),
            }
        )
        out_cell_index += 1

    out.SetPoints(points)
    out.SetLines(lines)
    out.BuildLinks()

    cd = out.GetCellData()
    add_scalar_array_to_cell_data(cd, "BranchId", branch_ids, vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchName", branch_names)
    add_scalar_array_to_cell_data(cd, "BranchLength", branch_lengths, vtk.VTK_DOUBLE)
    add_string_array_to_cell_data(cd, "GeometryType", geometry_types)

    try:
        add_string_array_to_cell_data(cd, "TopologyRole", topo_roles)
    except Exception:
        warnings.append("W_CENTERLINES_TOPOLOGYROLE_FAILED: could not write TopologyRole string array.")

    return out, branch_rows


def summarize_branch_lengths(branch_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts: Dict[str, int] = {}
    lengths: Dict[str, float] = {}
    for r in branch_rows:
        name = str(r.get("branch_name", "other"))
        counts[name] = int(counts.get(name, 0) + 1)
        lengths[name] = float(lengths.get(name, 0.0) + float(r.get("length", 0.0)))
    return counts, lengths


# -----------------------------
# Surface label transfer (topology-aware region growing)
# -----------------------------
def build_surface_cell_adjacency(pd: "vtkPolyData") -> List[List[int]]:
    n_cells = int(pd.GetNumberOfCells())
    adjacency: List[Set[int]] = [set() for _ in range(n_cells)]
    edge_to_cells: Dict[Tuple[int, int], List[int]] = {}

    for ci in range(n_cells):
        cell = pd.GetCell(ci)
        if cell is None:
            continue
        npts = cell.GetNumberOfPoints()
        if npts < 2:
            continue
        ids = [int(cell.GetPointId(k)) for k in range(npts)]
        for k in range(npts):
            a = int(ids[k])
            b = int(ids[(k + 1) % npts])
            ek = edge_key(a, b)
            edge_to_cells.setdefault(ek, []).append(int(ci))

    for cells in edge_to_cells.values():
        if len(cells) < 2:
            continue
        for i in range(len(cells)):
            ci = int(cells[i])
            for j in range(i + 1, len(cells)):
                cj = int(cells[j])
                if ci != cj:
                    adjacency[ci].add(cj)
                    adjacency[cj].add(ci)

    return [sorted(nbrs) for nbrs in adjacency]


def connected_component_from_seed(seed: int, mask: np.ndarray, adjacency: List[List[int]]) -> List[int]:
    seed = int(seed)
    if seed < 0 or seed >= int(mask.size) or not bool(mask[int(seed)]):
        return []
    visited = np.zeros((mask.size,), dtype=bool)
    q = [seed]
    visited[seed] = True
    out: List[int] = []
    while q:
        u = int(q.pop())
        out.append(u)
        for v in adjacency[u]:
            v = int(v)
            if not visited[v] and bool(mask[v]):
                visited[v] = True
                q.append(v)
    return out


def min_distance_to_segments(
    points: np.ndarray,
    seg_p0: np.ndarray,
    seg_p1: np.ndarray,
    seg_r: Optional[np.ndarray] = None,
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    P = np.asarray(points, dtype=float)
    A = np.asarray(seg_p0, dtype=float)
    B = np.asarray(seg_p1, dtype=float)
    if A.shape[0] == 0 or B.shape[0] == 0:
        inf = np.full((P.shape[0],), float("inf"), dtype=float)
        return inf, inf

    if seg_r is None:
        seg_r = np.ones((A.shape[0],), dtype=float)
    R = np.asarray(seg_r, dtype=float).reshape(-1)
    R = np.maximum(R, 1e-3)

    min_abs = np.full((P.shape[0],), float("inf"), dtype=float)
    min_norm = np.full((P.shape[0],), float("inf"), dtype=float)

    AB = B - A
    AB2 = np.sum(AB * AB, axis=1) + EPS

    for start in range(0, P.shape[0], int(chunk)):
        end = min(P.shape[0], start + int(chunk))
        X = P[start:end, None, :]
        A2 = A[None, :, :]
        AB2v = AB[None, :, :]
        t = np.sum((X - A2) * AB2v, axis=2) / AB2[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = A2 + t[:, :, None] * AB2v
        d2 = np.sum((X - proj) ** 2, axis=2)
        d = np.sqrt(np.minimum(d2, 1e30))
        idx = np.argmin(d, axis=1)
        dmin = d[np.arange(d.shape[0]), idx]
        rmin = R[idx]
        min_abs[start:end] = dmin
        min_norm[start:end] = dmin / (rmin + EPS)

    return min_abs, min_norm


def build_label_segments_from_system_sets(
    adjacency: Dict[int, Dict[int, float]],
    pts_c: np.ndarray,
    radius: np.ndarray,
    system_sets: Dict[int, Set[int]],
) -> Dict[int, Dict[str, np.ndarray]]:
    prio = {LABEL_RIGHT_RENAL: 0, LABEL_LEFT_RENAL: 0, LABEL_RIGHT_ILIAC: 1, LABEL_LEFT_ILIAC: 1, LABEL_AORTA_TRUNK: 2}
    labels = [LABEL_RIGHT_RENAL, LABEL_LEFT_RENAL, LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC, LABEL_AORTA_TRUNK]

    edges: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for u, nbrs in adjacency.items():
        for v in nbrs.keys():
            ek = edge_key(int(u), int(v))
            if ek in seen:
                continue
            seen.add(ek)
            edges.append((int(ek[0]), int(ek[1])))

    banks: Dict[int, List[Tuple[int, int]]] = {lid: [] for lid in labels}
    for u, v in edges:
        claims: List[int] = []
        for lid in labels:
            s = system_sets.get(int(lid), set())
            if int(u) in s and int(v) in s:
                claims.append(int(lid))
        if not claims:
            continue
        claims.sort(key=lambda lid: prio.get(int(lid), 9))
        owner = int(claims[0])
        banks[owner].append((int(u), int(v)))

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for lid, elist in banks.items():
        if not elist:
            continue
        p0 = pts_c[np.array([u for u, _ in elist], dtype=int)]
        p1 = pts_c[np.array([v for _, v in elist], dtype=int)]
        if radius.size > 0:
            r = 0.5 * (radius[np.array([u for u, _ in elist], dtype=int)] + radius[np.array([v for _, v in elist], dtype=int)])
            r = np.maximum(r, 1e-3)
        else:
            r = np.ones((len(elist),), dtype=float)
        out[int(lid)] = {"p0": p0.astype(float), "p1": p1.astype(float), "r": r.astype(float)}
    return out


def surface_label_transfer(
    surface_pd: "vtkPolyData",
    cell_centers: np.ndarray,
    cell_areas: np.ndarray,
    adjacency_cells: List[List[int]],
    segment_banks: Dict[int, Dict[str, np.ndarray]],
    landmarks: Dict[str, np.ndarray],
    warnings: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deterministic topology-aware label transfer. Produces cell labels (0..5).
    Uses connected region growing inside per-label candidate masks anchored by labeled centerlines.
    """
    n_cells = int(surface_pd.GetNumberOfCells())
    labels = np.full((n_cells,), LABEL_OTHER, dtype=int)
    if n_cells == 0:
        return labels, {"cell_counts": {}}

    dist_abs: Dict[int, np.ndarray] = {}
    dist_norm: Dict[int, np.ndarray] = {}
    for lid in [LABEL_AORTA_TRUNK, LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC, LABEL_RIGHT_RENAL, LABEL_LEFT_RENAL]:
        bank = segment_banks.get(int(lid))
        if not bank:
            continue
        da, dn = min_distance_to_segments(cell_centers, bank["p0"], bank["p1"], bank.get("r"), chunk=4096)
        dist_abs[int(lid)] = da
        dist_norm[int(lid)] = dn

    if LABEL_AORTA_TRUNK not in dist_abs or LABEL_AORTA_TRUNK not in dist_norm:
        warnings.append("W_SURFACE_TRUNK_DISTANCE_MISSING: trunk centerline segments missing; surface labels remain OTHER.")
        return labels, {"cell_counts": {"other": int(n_cells)}}

    trunk_abs = dist_abs[LABEL_AORTA_TRUNK]
    trunk_norm = dist_norm[LABEL_AORTA_TRUNK]

    trunk_r = float(np.median(segment_banks.get(LABEL_AORTA_TRUNK, {}).get("r", np.array([1.0], dtype=float))))
    trunk_r = max(trunk_r, 1.0)

    bif = np.asarray(landmarks.get("Bifurcation", np.zeros((3,), dtype=float)), dtype=float).reshape(3)
    inlet = np.asarray(landmarks.get("Inlet", None), dtype=float).reshape(3) if "Inlet" in landmarks else None
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    if inlet is not None and np.linalg.norm(inlet - bif) > EPS:
        ez = unit(inlet - bif)

    proj_z = (cell_centers - bif[None, :]) @ ez

    def seed_index_from_distance(candidate_mask: np.ndarray, dist_field: np.ndarray, prefer_point: Optional[np.ndarray] = None) -> int:
        idx = np.where(candidate_mask)[0]
        if idx.size == 0:
            return -1
        if prefer_point is not None:
            p = np.asarray(prefer_point, dtype=float).reshape(3)
            d2 = np.sum((cell_centers[idx] - p[None, :]) ** 2, axis=1)
            return int(idx[int(np.argmin(d2))])
        return int(idx[int(np.argmin(dist_field[idx]))])

    def assign_region_from_seed(lid: int, candidate_mask: np.ndarray, seed_idx: int) -> int:
        if seed_idx < 0:
            return 0
        comp = connected_component_from_seed(int(seed_idx), candidate_mask, adjacency_cells)
        if not comp:
            return 0
        labels[np.array(comp, dtype=int)] = int(lid)
        return int(len(comp))

    renal_params = {
        LABEL_RIGHT_RENAL: {"k_norm": 2.15, "guard_abs": 0.82, "guard_norm": 0.85, "name": "right_renal", "origin_key": "RightRenalOrigin"},
        LABEL_LEFT_RENAL: {"k_norm": 2.15, "guard_abs": 0.82, "guard_norm": 0.85, "name": "left_renal", "origin_key": "LeftRenalOrigin"},
    }
    for lid, prm in renal_params.items():
        lid = int(lid)
        if lid not in dist_norm or lid not in dist_abs or prm["origin_key"] not in landmarks:
            continue
        da = dist_abs[lid]
        dn = dist_norm[lid]
        takeoff = np.asarray(landmarks[prm["origin_key"]], dtype=float).reshape(3)

        renal_r = float(np.median(segment_banks.get(lid, {}).get("r", np.array([1.0], dtype=float))))
        renal_r = max(renal_r, 0.6)

        ostium_r = 2.8 * (renal_r + 0.5 * trunk_r)
        around_ostium = np.sum((cell_centers - takeoff[None, :]) ** 2, axis=1) <= (ostium_r * ostium_r)

        close = (dn <= float(prm["k_norm"])) & (labels == LABEL_OTHER)

        guard = (da <= float(prm["guard_abs"]) * trunk_abs) | (dn <= float(prm["guard_norm"]) * trunk_norm) | around_ostium
        candidate = close & guard

        if np.count_nonzero(candidate) < 12:
            continue

        C = cell_centers[candidate]
        dir_vec = unit(np.mean(C, axis=0) - takeoff)
        if np.linalg.norm(dir_vec) > EPS:
            proj = (cell_centers - takeoff[None, :]) @ dir_vec
            candidate &= (proj >= -0.35 * ostium_r)

        seed_idx = seed_index_from_distance(candidate, dn)
        n_assigned = assign_region_from_seed(lid, candidate, seed_idx)
        if n_assigned == 0:
            warnings.append(f"W_SURFACE_{prm['name'].upper()}_NONE: no surface cells assigned for {prm['name']}.")

    iliac_params = {
        LABEL_RIGHT_ILIAC: {"k_norm": 2.55, "guard_abs": 0.90, "guard_norm": 0.92, "name": "right_main_iliac"},
        LABEL_LEFT_ILIAC: {"k_norm": 2.55, "guard_abs": 0.90, "guard_norm": 0.92, "name": "left_main_iliac"},
    }
    for lid, prm in iliac_params.items():
        lid = int(lid)
        if lid not in dist_norm or lid not in dist_abs:
            continue
        da = dist_abs[lid]
        dn = dist_norm[lid]

        iliac_r = float(np.median(segment_banks.get(lid, {}).get("r", np.array([1.0], dtype=float))))
        iliac_r = max(iliac_r, 0.8)

        around_bif = np.sum((cell_centers - bif[None, :]) ** 2, axis=1) <= ((2.8 * (iliac_r + trunk_r)) ** 2)
        distal = proj_z <= (0.35 * trunk_r)

        close = (dn <= float(prm["k_norm"])) & (labels == LABEL_OTHER)
        guard = (da <= float(prm["guard_abs"]) * trunk_abs) | (dn <= float(prm["guard_norm"]) * trunk_norm) | around_bif
        candidate = close & distal & guard

        if np.count_nonzero(candidate) < 20:
            continue

        seed_idx = seed_index_from_distance(candidate, dn)
        n_assigned = assign_region_from_seed(lid, candidate, seed_idx)
        if n_assigned == 0:
            warnings.append(f"W_SURFACE_{prm['name'].upper()}_NONE: no surface cells assigned for {prm['name']}.")

    trunk_close = (trunk_norm <= 3.15) & (labels == LABEL_OTHER)
    proximal = proj_z >= (-0.25 * trunk_r)
    trunk_candidate = trunk_close & proximal
    if np.count_nonzero(trunk_candidate) > 20:
        seed_idx = seed_index_from_distance(trunk_candidate, trunk_norm, prefer_point=inlet if inlet is not None else None)
        trunk_assigned = assign_region_from_seed(LABEL_AORTA_TRUNK, trunk_candidate, seed_idx)
        if trunk_assigned == 0:
            warnings.append("W_SURFACE_TRUNK_NONE: no trunk surface cells assigned from trunk candidate mask.")
    else:
        warnings.append("W_SURFACE_TRUNK_CANDIDATE_EMPTY: trunk candidate mask empty; trunk region may be missing.")

    def cleanup_label(lid: int) -> None:
        lid = int(lid)
        mask = labels == lid
        if not np.any(mask):
            return
        visited = np.zeros((n_cells,), dtype=bool)
        comps: List[List[int]] = []
        for i in np.where(mask)[0]:
            i = int(i)
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp: List[int] = []
            while stack:
                u = int(stack.pop())
                comp.append(u)
                for v in adjacency_cells[u]:
                    v = int(v)
                    if not visited[v] and bool(mask[v]):
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
        if len(comps) <= 1:
            return
        areas = [float(np.sum(cell_areas[np.array(c, dtype=int)])) for c in comps]
        keep = int(np.argmax(areas))
        for j, comp in enumerate(comps):
            if j == keep:
                continue
            labels[np.array(comp, dtype=int)] = LABEL_OTHER

    for lid in [LABEL_RIGHT_RENAL, LABEL_LEFT_RENAL, LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC, LABEL_AORTA_TRUNK]:
        cleanup_label(lid)

    trunk_cells = set(int(i) for i in np.where(labels == LABEL_AORTA_TRUNK)[0])
    if trunk_cells:
        for lid, name in ((LABEL_RIGHT_RENAL, "right_renal"), (LABEL_LEFT_RENAL, "left_renal")):
            renal_cells = np.where(labels == int(lid))[0]
            if renal_cells.size == 0:
                continue
            touches = False
            for ci in renal_cells:
                for nb in adjacency_cells[int(ci)]:
                    if int(nb) in trunk_cells:
                        touches = True
                        break
                if touches:
                    break
            if not touches:
                warnings.append(f"W_SURFACE_{name.upper()}_DETACHED: labeled {name} surface region does not touch trunk.")

    cell_counts: Dict[str, int] = {}
    for lid, name in LABEL_ID_TO_NAME.items():
        cell_counts[name] = int(np.count_nonzero(labels == int(lid)))

    return labels.astype(int), {"cell_counts": cell_counts}


def derive_surface_point_labels(surface_pd: "vtkPolyData", cell_labels: np.ndarray) -> np.ndarray:
    n_points = int(surface_pd.GetNumberOfPoints())
    n_cells = int(surface_pd.GetNumberOfCells())
    if n_points == 0 or n_cells == 0:
        return np.zeros((n_points,), dtype=int)
    point_votes: List[Dict[int, int]] = [dict() for _ in range(n_points)]
    for ci in range(n_cells):
        cell = surface_pd.GetCell(int(ci))
        if cell is None:
            continue
        lid = int(cell_labels[int(ci)])
        for k in range(cell.GetNumberOfPoints()):
            pid = int(cell.GetPointId(k))
            point_votes[pid][lid] = int(point_votes[pid].get(lid, 0) + 1)
    out = np.zeros((n_points,), dtype=int)
    for pid in range(n_points):
        votes = point_votes[pid]
        if not votes:
            continue
        out[pid] = int(max(votes.items(), key=lambda kv: int(kv[1]))[0])
    return out


# -----------------------------
# Combined output polydata
# -----------------------------
def build_combined_surface_centerlines_polydata(surface_pd: "vtkPolyData", centerlines_pd: "vtkPolyData") -> "vtkPolyData":
    app = vtk.vtkAppendPolyData()
    app.AddInputData(surface_pd)
    app.AddInputData(centerlines_pd)
    app.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(app.GetOutput())
    out.BuildLinks()

    n_surface_cells = int(surface_pd.GetNumberOfCells())
    n_cl_cells = int(centerlines_pd.GetNumberOfCells())
    cd = out.GetCellData()

    def get_cell_array_values(pd: "vtkPolyData", name: str, default: Any) -> List[Any]:
        arr = pd.GetCellData().GetArray(name) if pd.GetCellData() is not None else None
        if arr is None:
            if isinstance(default, str):
                return [str(default)] * int(pd.GetNumberOfCells())
            return [default] * int(pd.GetNumberOfCells())
        if isinstance(arr, vtk.vtkStringArray):
            return [arr.GetValue(i) for i in range(int(pd.GetNumberOfCells()))]
        vals = []
        for i in range(int(pd.GetNumberOfCells())):
            vals.append(arr.GetTuple1(i))
        return vals

    branch_id_vals = get_cell_array_values(surface_pd, "BranchId", 0) + get_cell_array_values(centerlines_pd, "BranchId", 0)
    branch_name_vals = get_cell_array_values(surface_pd, "BranchName", "other") + get_cell_array_values(centerlines_pd, "BranchName", "other")
    branch_len_vals = get_cell_array_values(surface_pd, "BranchLength", 0.0) + get_cell_array_values(centerlines_pd, "BranchLength", 0.0)
    geom_vals = ["surface"] * n_surface_cells + ["centerline"] * n_cl_cells

    add_scalar_array_to_cell_data(cd, "BranchId", [int(v) for v in branch_id_vals], vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchName", [str(v) for v in branch_name_vals])
    add_scalar_array_to_cell_data(cd, "BranchLength", [float(v) for v in branch_len_vals], vtk.VTK_DOUBLE)
    add_string_array_to_cell_data(cd, "GeometryType", geom_vals)

    try:
        topo_vals = get_cell_array_values(surface_pd, "TopologyRole", "surface") + get_cell_array_values(centerlines_pd, "TopologyRole", "centerline")
        add_string_array_to_cell_data(cd, "TopologyRole", [str(v) for v in topo_vals])
    except Exception:
        pass

    return out


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    require_vtk()

    parser = argparse.ArgumentParser(description="First-stage anatomy-aware preprocessing for abdominal arterial lumen surface models (.vtp).")
    parser.add_argument("--input", type=str, default=INPUT_VTP_PATH, help="Input lumen surface VTP path")
    parser.add_argument("--output_surface_with_centerlines", type=str, default=OUTPUT_SURFACE_WITH_CENTERLINES_VTP_PATH, help="Output oriented surface + centerlines VTP path")
    parser.add_argument("--output_centerlines", type=str, default=OUTPUT_CENTERLINES_VTP_PATH, help="Output oriented labeled centerlines VTP path")
    parser.add_argument("--output", type=str, default=None, help="Deprecated alias for --output_centerlines (kept for compatibility)")
    parser.add_argument("--metadata", type=str, default=OUTPUT_METADATA_PATH, help="Output metadata JSON path (optional; empty disables)")
    parser.add_argument("--debug_raw_centerlines", type=str, default=OUTPUT_DEBUG_CENTERLINES_RAW_PATH, help="Optional raw centerlines VTP path (empty disables)")
    args = parser.parse_args()

    input_path = _resolve_user_path(args.input)
    surface_with_centerlines_path = _resolve_user_path(args.output_surface_with_centerlines)
    centerlines_output_path = _resolve_user_path(args.output if args.output is not None else args.output_centerlines)
    meta_path = _resolve_user_path(args.metadata) if args.metadata else ""
    debug_raw_path = _resolve_user_path(args.debug_raw_centerlines) if args.debug_raw_centerlines else ""

    warnings: List[str] = []
    debug: Dict[str, Any] = {}

    try:
        surface = load_vtp(input_path)
        surface_tri = clean_and_triangulate_surface(surface)

        terminations, mode = detect_terminations(surface_tri, warnings)
        if len(terminations) < 2:
            raise RuntimeError("Failed to detect enough terminations (need >=2) to seed centerlines.")

        centerlines_raw, cl_info = compute_centerlines_vmtk(surface_tri, terminations, warnings)
        debug["centerlines_info"] = cl_info

        if debug_raw_path:
            try:
                write_vtp(centerlines_raw, debug_raw_path, binary=True)
            except Exception as exc:
                warnings.append(f"W_DEBUG_WRITE_RAW_CENTERLINES: {type(exc).__name__}: {exc}")

        adjacency_full, cl_pts, cl_rad = build_graph_from_polyline_centerlines(centerlines_raw)
        if not adjacency_full:
            raise RuntimeError("Centerline adjacency graph is empty.")

        nodes_all = set(int(n) for n in adjacency_full.keys())
        comps: List[Set[int]] = []
        remaining = set(nodes_all)
        while remaining:
            seed = next(iter(remaining))
            comp = connected_component_nodes(adjacency_full, seed)
            comps.append(comp)
            remaining.difference_update(comp)
        comps.sort(key=lambda s: (len(s), -min(s)), reverse=True)
        main_nodes = comps[0]
        if len(comps) > 1:
            warnings.append(f"W_CENTERLINE_MULTI_COMPONENT: {len(comps)} components; using largest with {len(main_nodes)} nodes.")
        adjacency = induced_subgraph(adjacency_full, main_nodes)

        anatomy = solve_anatomy(adjacency, cl_pts, cl_rad, terminations, warnings)
        debug.update(anatomy.debug)

        R = anatomy.R
        origin = anatomy.origin
        cl_pts_c = apply_transform_points(cl_pts, R, origin)

        landmarks: Dict[str, np.ndarray] = {}
        landmarks["Bifurcation"] = np.array([0.0, 0.0, 0.0], dtype=float)
        landmarks["Inlet"] = cl_pts_c[int(anatomy.inlet_node)].astype(float)
        if anatomy.renal_right is not None:
            landmarks["RightRenalOrigin"] = cl_pts_c[int(anatomy.renal_right.takeoff)].astype(float)
        if anatomy.renal_left is not None:
            landmarks["LeftRenalOrigin"] = cl_pts_c[int(anatomy.renal_left.takeoff)].astype(float)

        system_sets: Dict[int, Set[int]] = {
            LABEL_AORTA_TRUNK: set(int(n) for n in anatomy.trunk_nodes),
            LABEL_RIGHT_ILIAC: set(int(n) for n in anatomy.iliac_nodes_right),
            LABEL_LEFT_ILIAC: set(int(n) for n in anatomy.iliac_nodes_left),
            LABEL_RIGHT_RENAL: set(int(n) for n in anatomy.renal_nodes_right),
            LABEL_LEFT_RENAL: set(int(n) for n in anatomy.renal_nodes_left),
            LABEL_OTHER: set(),
        }

        chains = build_branch_chains_from_graph(adjacency)
        centerlines_scaffold_pd, branch_rows = build_centerline_scaffold_polydata(chains, cl_pts_c, cl_rad, system_sets, warnings)
        branch_counts, length_by_branch = summarize_branch_lengths(branch_rows)

        surface_c = apply_transform_to_polydata(surface_tri, R, origin)
        surface_c.BuildLinks()

        cell_centers = get_cell_centers_numpy(surface_c)
        cell_areas = get_cell_areas_numpy(surface_c)
        adjacency_cells = build_surface_cell_adjacency(surface_c)

        segment_banks = build_label_segments_from_system_sets(adjacency, cl_pts_c, cl_rad, system_sets)
        landmarks_np = {k: np.asarray(v, dtype=float).reshape(3) for k, v in landmarks.items()}
        cell_labels, surface_summary = surface_label_transfer(surface_c, cell_centers, cell_areas, adjacency_cells, segment_banks, landmarks_np, warnings)

        surface_cd = surface_c.GetCellData()
        add_scalar_array_to_cell_data(surface_cd, "BranchId", [int(v) for v in cell_labels.tolist()], vtk.VTK_INT)
        add_string_array_to_cell_data(surface_cd, "BranchName", [LABEL_ID_TO_NAME.get(int(v), "other") for v in cell_labels.tolist()])
        add_scalar_array_to_cell_data(surface_cd, "BranchLength", [0.0] * int(surface_c.GetNumberOfCells()), vtk.VTK_DOUBLE)
        add_string_array_to_cell_data(surface_cd, "GeometryType", ["surface"] * int(surface_c.GetNumberOfCells()))

        combined_pd = build_combined_surface_centerlines_polydata(surface_c, centerlines_scaffold_pd)

        write_vtp(combined_pd, surface_with_centerlines_path, binary=True)
        write_vtp(centerlines_scaffold_pd, centerlines_output_path, binary=True)

        if meta_path:
            os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
            meta: Dict[str, Any] = {
                "input_vtp": os.path.abspath(input_path),
                "output_surface_with_centerlines_vtp": os.path.abspath(surface_with_centerlines_path),
                "output_centerlines_vtp": os.path.abspath(centerlines_output_path),
                "mode": str(mode),
                "branch_names": sorted(branch_counts.keys()),
                "branch_counts": {k: int(v) for k, v in branch_counts.items()},
                "centerline_length_by_branch": {k: float(v) for k, v in length_by_branch.items()},
                "centerline_branch_summaries": branch_rows,
                "surface_cell_counts_by_branch": {k: int(v) for k, v in surface_summary.get("cell_counts", {}).items() if int(v) > 0},
                "landmarks_xyz_canonical": {k: [float(x) for x in np.asarray(v, dtype=float).reshape(3)] for k, v in landmarks.items()},
                "transform": {
                    "R_rows": [[float(x) for x in row] for row in np.asarray(R, dtype=float).reshape(3, 3)],
                    "origin": [float(x) for x in np.asarray(origin, dtype=float).reshape(3)],
                    "flipped_for_ap": False,
                },
                "renals_found": {
                    "right": bool(anatomy.renal_right is not None),
                    "left": bool(anatomy.renal_left is not None),
                },
                "confidence": {
                    "inlet": float(anatomy.inlet_conf),
                    "bifurcation": float(anatomy.bif_conf),
                    "laterality": float(anatomy.laterality_conf),
                    "renal": float(anatomy.renal_conf),
                    "ap": float(anatomy.ap_conf),
                    "frame": float(anatomy.frame_conf),
                },
                "warnings": list(warnings),
                "debug": debug,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return 0

    except Exception as exc:
        sys.stderr.write("ERROR: preprocessing failed.\n")
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.write(traceback.format_exc() + "\n")
        if meta_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
                failure_meta = {
                    "input_vtp": os.path.abspath(input_path),
                    "output_surface_with_centerlines_vtp": os.path.abspath(surface_with_centerlines_path),
                    "output_centerlines_vtp": os.path.abspath(centerlines_output_path),
                    "mode": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "warnings": list(warnings),
                    "traceback": traceback.format_exc(),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(failure_meta, f, indent=2)
            except Exception:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
