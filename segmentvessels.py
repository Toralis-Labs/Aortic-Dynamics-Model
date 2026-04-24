#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import glob
import importlib
import platform
import subprocess
import traceback
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

"""
Geometry-only vascular surface segment decomposition.

Primary deliverable:
- Input: one vascular lumen .vtp file
- Output: one cleaned surface .vtp with cell-wise SegmentId labeling and segment colors,
  plus a metadata JSON describing segment boundaries and centerline paths.

Dependencies:
- vtk
- numpy
- VMTK python bindings (vtkvmtk) REQUIRED

No manual interaction. Deterministic best-effort behavior with warnings and fallbacks.
"""

# -----------------------------
# User-editable paths (defaults)
# -----------------------------
INPUT_VTP_PATH = r"C:\Users\ibrah\OneDrive\Desktop\Fluids Project\Vascular specific\0044_H_ABAO_AAA\0156_0001.vtp"
DEFAULT_OUTPUT_DIR = r"C:\Users\ibrah\OneDrive\Desktop\Fluids Project\Vascular specific\Output files"
OUTPUT_SEGMENTS_VTP_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "segmentscolored.vtp")
OUTPUT_METADATA_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "segments_metadata.json")
OUTPUT_SURFACE_CLEANED_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "surface_cleaned.vtp")
OUTPUT_CENTERLINES_DEBUG_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "centerlines_debug.vtp")
OUTPUT_JUNCTION_PROFILES_DEBUG_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "junction_profiles_debug.vtp")
OUTPUT_SEGMENT_BOUNDARIES_DEBUG_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "segment_boundaries_debug.vtp")
WRITE_DEBUG_OUTPUTS = True

SCRIPT_PATH = os.path.abspath(__file__) if "__file__" in globals() else os.path.abspath(sys.argv[0])
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
_AUTO_VMTK_REEXEC_ENV = "SEGMENT_VMTK_REEXEC_ACTIVE"
_AUTO_VMTK_PYTHON_ENV = "VMTK_PYTHON_EXE"
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


def _iter_vmtk_python_candidates() -> List[str]:
    candidates: List[str] = []
    seen = set()

    def _add(path: str) -> None:
        path = (path or "").strip()
        if not path:
            return
        try:
            candidate = os.path.abspath(path)
        except Exception:
            return
        if not os.path.isfile(candidate):
            return
        key = _normalize_path_key(candidate)
        if key in seen or key == _normalize_path_key(sys.executable):
            return
        seen.add(key)
        candidates.append(candidate)

    _add(os.environ.get(_AUTO_VMTK_PYTHON_ENV, ""))
    _add(os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe"))

    user_home = os.path.expanduser("~")
    common_conda_roots = [
        os.path.join(user_home, "miniconda3"),
        os.path.join(user_home, "anaconda3"),
        os.path.join(user_home, "mambaforge"),
        os.path.join(user_home, "miniforge3"),
    ]
    common_env_names = ("vmtk_env", "vmtk", "simvascular", "sv")

    for root in common_conda_roots:
        for env_name in common_env_names:
            _add(os.path.join(root, "envs", env_name, "python.exe"))
        for pattern in ("*vmtk*", "*vascular*"):
            for match in glob.glob(os.path.join(root, "envs", pattern, "python.exe")):
                _add(match)

    return candidates


def _python_supports_required_vmtk(python_exe: str) -> bool:
    probe = r"""
import importlib
import os
import sys
import types

required = ("vtkvmtkCapPolyData", "vtkvmtkPolyDataCenterlines")
prefix = os.path.abspath(sys.prefix)
dirs = [
    prefix,
    os.path.join(prefix, "Library", "bin"),
    os.path.join(prefix, "Scripts"),
    os.path.join(prefix, "bin"),
    os.path.join(prefix, "Lib", "site-packages", "vmtk"),
    os.path.join(prefix, "Lib", "site-packages", "vtkmodules"),
]
for candidate in dirs:
    if hasattr(os, "add_dll_directory") and os.path.isdir(candidate):
        try:
            os.add_dll_directory(candidate)
        except Exception:
            pass

try:
    mod = importlib.import_module("vmtk.vtkvmtk")
except Exception:
    mod = None

if mod is None:
    merged = types.ModuleType("vtkvmtk_fallback")
    loaded = []
    for name in (
        "vmtk.vtkvmtkSegmentationPython",
        "vmtk.vtkvmtkComputationalGeometryPython",
        "vmtk.vtkvmtkDifferentialGeometryPython",
        "vmtk.vtkvmtkMiscPython",
        "vmtk.vtkvmtkRenderingPython",
    ):
        try:
            loaded.append(importlib.import_module(name))
        except Exception:
            continue
    for probe_module in loaded:
        for attr_name in dir(probe_module):
            if attr_name.startswith("_") or attr_name in merged.__dict__:
                continue
            merged.__dict__[attr_name] = getattr(probe_module, attr_name)
    mod = merged

sys.exit(0 if all(hasattr(mod, name) for name in required) else 1)
""".strip()
    try:
        completed = subprocess.run(
            [python_exe, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0


def _maybe_reexec_with_vmtk_python() -> None:
    if os.name != "nt" or __name__ != "__main__":
        return
    if os.environ.get(_AUTO_VMTK_REEXEC_ENV) == "1":
        return
    if _python_supports_required_vmtk(sys.executable):
        return

    for python_exe in _iter_vmtk_python_candidates():
        if not _python_supports_required_vmtk(python_exe):
            continue
        sys.stderr.write(f"INFO: relaunching with VMTK-capable interpreter: {python_exe}\n")
        env = os.environ.copy()
        env[_AUTO_VMTK_REEXEC_ENV] = "1"
        env[_AUTO_VMTK_PYTHON_ENV] = python_exe
        completed = subprocess.run([python_exe, SCRIPT_PATH, *sys.argv[1:]], env=env, check=False)
        raise SystemExit(int(completed.returncode))


_maybe_reexec_with_vmtk_python()


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

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk  # type: ignore
except Exception as e:  # pragma: no cover
    vtk = None
    vtk_to_numpy = None
    numpy_to_vtk = None
    _VTK_IMPORT_ERROR = str(e)

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStaticPointLocator


@dataclass
class BoundaryProfile:
    center: np.ndarray
    area: float
    diameter_eq: float
    normal: np.ndarray
    rms_planarity: float
    n_points: int
    source: str
    profile_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    boundary_type: str = "terminal_profile"
    face_id: Optional[int] = None
    cap_id: Optional[int] = None
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)


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
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return np.eye(3), np.ones(3), np.mean(pts, axis=0) if pts.shape[0] else np.zeros(3)
    c = np.mean(pts, axis=0)
    x = pts - c
    cov = (x.T @ x) / max(1, x.shape[0])
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    e0 = unit(v[:, 0])
    e1 = unit(v[:, 1] - np.dot(v[:, 1], e0) * e0)
    e2 = unit(np.cross(e0, e1))
    a = np.column_stack([e0, e1, e2])
    return a, w.astype(float), c.astype(float)


def planar_polygon_area_and_normal(points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, np.zeros(3), float("nan")
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-8 * (np.linalg.norm(pts[0]) + 1.0):
        pts = pts[:-1]
    if pts.shape[0] < 3:
        return 0.0, np.zeros(3), float("nan")

    a, _, c = pca_axes(pts)
    n = unit(a[:, 2])
    u = unit(a[:, 0])
    v = unit(np.cross(n, u))
    x = pts - c
    dists = x @ n
    rms = float(np.sqrt(np.mean(dists * dists))) if dists.size else float("nan")
    x2 = x @ u
    y2 = x @ v
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


def compute_abscissa(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    s = np.zeros((n,), dtype=float)
    if n < 2:
        return s
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s[1:] = np.cumsum(d)
    return s


def project_point_to_segment(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, float, float]:
    q = np.asarray(point, dtype=float).reshape(3)
    a = np.asarray(p0, dtype=float).reshape(3)
    b = np.asarray(p1, dtype=float).reshape(3)
    v = b - a
    vv = float(np.dot(v, v))
    if vv <= EPS:
        diff = q - a
        return a.astype(float), 0.0, float(np.dot(diff, diff))
    t = clamp(float(np.dot(q - a, v) / vv), 0.0, 1.0)
    proj = a + t * v
    diff = q - proj
    return proj.astype(float), float(t), float(np.dot(diff, diff))


def project_point_to_polyline(point: np.ndarray, polyline_points: np.ndarray) -> Optional[Dict[str, Any]]:
    pts = np.asarray(polyline_points, dtype=float)
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] == 1:
        diff = np.asarray(point, dtype=float).reshape(3) - pts[0]
        return {
            "point": pts[0].astype(float),
            "segment_index": 0,
            "t": 0.0,
            "distance2": float(np.dot(diff, diff)),
            "abscissa": 0.0,
        }

    s = compute_abscissa(pts)
    best: Optional[Dict[str, Any]] = None
    for idx in range(pts.shape[0] - 1):
        proj, t, d2 = project_point_to_segment(point, pts[idx], pts[idx + 1])
        abscissa = float(s[idx] + t * np.linalg.norm(pts[idx + 1] - pts[idx]))
        if best is None or d2 < float(best["distance2"]):
            best = {
                "point": proj.astype(float),
                "segment_index": int(idx),
                "t": float(t),
                "distance2": float(d2),
                "abscissa": float(abscissa),
            }
    return best


def polyline_point_at_abscissa(points: np.ndarray, target_abscissa: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((3,), dtype=float)
    if pts.shape[0] == 1:
        return pts[0].astype(float)

    s = compute_abscissa(pts)
    total = float(s[-1])
    if total <= EPS:
        return pts[0].astype(float)

    target = clamp(float(target_abscissa), 0.0, total)
    idx = int(np.searchsorted(s, target, side="right") - 1)
    idx = int(max(0, min(idx, pts.shape[0] - 2)))
    ds = float(s[idx + 1] - s[idx])
    if ds <= EPS:
        return pts[idx].astype(float)
    t = clamp((target - float(s[idx])) / ds, 0.0, 1.0)
    return ((1.0 - t) * pts[idx] + t * pts[idx + 1]).astype(float)


def polyline_tangent_at_abscissa(points: np.ndarray, target_abscissa: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros((3,), dtype=float)

    s = compute_abscissa(pts)
    total = float(s[-1]) if s.size else 0.0
    target = clamp(float(target_abscissa), 0.0, total)
    idx = int(np.searchsorted(s, target, side="right") - 1)
    idx = int(max(0, min(idx, pts.shape[0] - 2)))
    return unit(pts[idx + 1] - pts[idx])


def build_orthonormal_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = unit(np.asarray(normal, dtype=float).reshape(3))
    if np.linalg.norm(n) < EPS:
        return np.array([1.0, 0.0, 0.0], dtype=float), np.array([0.0, 1.0, 0.0], dtype=float)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n))) > 0.85:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = unit(np.cross(n, ref))
    v = unit(np.cross(n, u))
    return u.astype(float), v.astype(float)


def make_circle_points(center: np.ndarray, normal: np.ndarray, radius: float, n_points: int = 32) -> np.ndarray:
    u, v = build_orthonormal_frame(normal)
    c = np.asarray(center, dtype=float).reshape(3)
    r = max(float(radius), 1e-3)
    pts = []
    for k in range(max(8, int(n_points))):
        ang = 2.0 * math.pi * float(k) / float(max(8, int(n_points)))
        pts.append(c + r * math.cos(ang) * u + r * math.sin(ang) * v)
    return np.asarray(pts, dtype=float)


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


def write_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_vtp(path: str) -> "vtkPolyData":
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(reader.GetOutput())
    return out


def get_points_numpy(pd: "vtkPolyData") -> np.ndarray:
    pts = pd.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    arr = pts.GetData()
    if vtk_to_numpy is None:
        raise RuntimeError("vtk_to_numpy unavailable")
    return vtk_to_numpy(arr).astype(float)


def get_point_array_numpy(pd: "vtkPolyData", name: str) -> Optional[np.ndarray]:
    arr = pd.GetPointData().GetArray(name)
    if arr is None or vtk_to_numpy is None:
        return None
    return vtk_to_numpy(arr)


def get_cell_array_numpy(pd: "vtkPolyData", name: str) -> Optional[np.ndarray]:
    arr = pd.GetCellData().GetArray(name)
    if arr is None or vtk_to_numpy is None:
        return None
    return vtk_to_numpy(arr)


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


def build_cell_centers(pd: "vtkPolyData") -> np.ndarray:
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(pd)
    cell_centers.VertexCellsOff()
    cell_centers.Update()
    return get_points_numpy(cell_centers.GetOutput())


def prune_polydata_arrays(
    pd: "vtkPolyData",
    keep_point_arrays: Optional[List[str]] = None,
    keep_cell_arrays: Optional[List[str]] = None,
    keep_field_arrays: Optional[List[str]] = None,
) -> "vtkPolyData":
    keep_point = set(str(x) for x in (keep_point_arrays or []))
    keep_cell = set(str(x) for x in (keep_cell_arrays or []))
    keep_field = set(str(x) for x in (keep_field_arrays or []))

    out = vtk.vtkPolyData()
    out.DeepCopy(pd)

    for dataset, keep in (
        (out.GetPointData(), keep_point),
        (out.GetCellData(), keep_cell),
        (out.GetFieldData(), keep_field),
    ):
        if dataset is None:
            continue
        remove_names: List[str] = []
        for i in range(dataset.GetNumberOfArrays()):
            arr = dataset.GetArray(i)
            if arr is None:
                continue
            name = arr.GetName()
            if not name:
                continue
            if name not in keep:
                remove_names.append(name)
        for name in remove_names:
            dataset.RemoveArray(name)
    return out


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


def extract_largest_connected_region(pd: "vtkPolyData") -> "vtkPolyData":
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(conn.GetOutputPort())
    geom.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(geom.GetOutput())
    return out


def find_face_partition_array_name(pd: "vtkPolyData") -> Optional[str]:
    cd = pd.GetCellData()
    if cd is None:
        return None
    n = cd.GetNumberOfArrays()
    best = None
    for i in range(n):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if not name:
            continue
        if name.lower() == "modelfaceid":
            return name
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


def sanitize_surface_for_segmentation(pd_in: "vtkPolyData", warnings: List[str]) -> "vtkPolyData":
    if pd_in is None or pd_in.GetNumberOfPoints() < 3 or pd_in.GetNumberOfCells() < 1:
        raise RuntimeError("Input VTP is empty or invalid.")

    pd = vtk.vtkPolyData()
    pd.DeepCopy(pd_in)

    original_lines = int(pd.GetNumberOfLines())
    original_verts = int(pd.GetNumberOfVerts())
    original_strips = int(pd.GetNumberOfStrips())
    if original_lines > 0 or original_verts > 0:
        warnings.append(
            f"W_INPUT_MIXED_GEOMETRY: input contained non-surface geometry "
            f"(lines={original_lines}, verts={original_verts}, strips={original_strips}); only polygonal surface cells are kept."
        )

    tri = clean_and_triangulate_surface(pd)
    largest = extract_largest_connected_region(tri)
    tri2 = clean_and_triangulate_surface(largest)

    face_array = find_face_partition_array_name(tri2)
    keep_cell_arrays = [x for x in [face_array, "CapID"] if x]
    cleaned = prune_polydata_arrays(tri2, keep_point_arrays=[], keep_cell_arrays=keep_cell_arrays, keep_field_arrays=[])
    cleaned = clean_and_triangulate_surface(cleaned)

    if cleaned.GetNumberOfPoints() < 3 or cleaned.GetNumberOfCells() < 1:
        raise RuntimeError("Cleaned surface became empty.")
    if cleaned.GetNumberOfPolys() < 1:
        raise RuntimeError("Cleaned surface has no polygonal cells.")

    cleaned.BuildLinks()
    return cleaned


def polydata_from_cell_ids(pd: "vtkPolyData", cell_ids: List[int]) -> "vtkPolyData":
    ids = vtk.vtkIdTypeArray()
    for cid in cell_ids:
        ids.InsertNextValue(int(cid))
    node = vtk.vtkSelectionNode()
    node.SetFieldType(vtk.vtkSelectionNode.CELL)
    node.SetContentType(vtk.vtkSelectionNode.INDICES)
    node.SetSelectionList(ids)
    selection = vtk.vtkSelection()
    selection.AddNode(node)
    extract = vtk.vtkExtractSelection()
    extract.SetInputData(0, pd)
    extract.SetInputData(1, selection)
    extract.Update()
    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(extract.GetOutputPort())
    geom.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(geom.GetOutput())
    return out


def extract_boundary_loops(pd: "vtkPolyData", source_name: str = "boundary_loop") -> List[BoundaryProfile]:
    loops: List[BoundaryProfile] = []
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
            BoundaryProfile(
                center=center.astype(float),
                area=float(area),
                diameter_eq=float(diameter_eq),
                normal=normal.astype(float),
                rms_planarity=float(rms),
                n_points=int(nids),
                source=str(source_name),
                profile_points=np.asarray(coords, dtype=float),
                boundary_type="terminal_profile",
            )
        )
    return loops


def extract_face_regions(pd_tri: "vtkPolyData", face_array_name: str, cap_array_name: str = "CapID") -> Dict[int, Dict[str, Any]]:
    cd = pd_tri.GetCellData()
    if cd is None:
        return {}
    face_arr = cd.GetArray(face_array_name)
    if face_arr is None or vtk_to_numpy is None:
        return {}

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
    pd_normals = normals_f.GetOutput()

    centers_f = vtk.vtkCellCenters()
    centers_f.SetInputData(pd_normals)
    centers_f.VertexCellsOff()
    centers_f.Update()
    centers_pd = centers_f.GetOutput()
    centers_pts = centers_pd.GetPoints()
    if centers_pts is None:
        return {}

    face_vals = vtk_to_numpy(pd_normals.GetCellData().GetArray(face_array_name)).astype(np.int64)
    cap_arr = pd_normals.GetCellData().GetArray(cap_array_name)
    cap_vals = vtk_to_numpy(cap_arr).astype(np.int64) if cap_arr is not None else np.zeros_like(face_vals)
    area_vals = vtk_to_numpy(pd_normals.GetCellData().GetArray("Area")).astype(float)
    centers_vals = vtk_to_numpy(centers_pts.GetData()).astype(float)

    cell_normals_vtk = pd_normals.GetCellData().GetNormals()
    if cell_normals_vtk is None:
        cell_normals_vtk = pd_normals.GetCellData().GetArray("Normals")
    normal_vals = vtk_to_numpy(cell_normals_vtk).astype(float) if cell_normals_vtk is not None else np.zeros_like(centers_vals)

    regions: Dict[int, Dict[str, Any]] = {}
    total_surface_area = float(np.sum(area_vals)) if area_vals.size else 0.0

    for face_id in sorted(int(v) for v in np.unique(face_vals).tolist()):
        mask = np.asarray(face_vals == int(face_id), dtype=bool)
        if not np.any(mask):
            continue
        areas = area_vals[mask]
        centers = centers_vals[mask]
        normals = normal_vals[mask] if normal_vals.shape[0] == face_vals.shape[0] else np.zeros((centers.shape[0], 3), dtype=float)
        cap_subset = cap_vals[mask] if cap_vals.shape[0] == face_vals.shape[0] else np.zeros((centers.shape[0],), dtype=np.int64)
        total_area = float(np.sum(areas))
        if total_area <= EPS:
            centroid = np.mean(centers, axis=0) if centers.shape[0] else np.zeros((3,), dtype=float)
            mean_normal = np.zeros((3,), dtype=float)
            planarity_score = 0.0
        else:
            centroid = np.sum(centers * areas[:, None], axis=0) / total_area
            n_sum = np.sum(normals * areas[:, None], axis=0)
            mean_normal = unit(n_sum)
            planarity_score = float(np.linalg.norm(n_sum) / (total_area + EPS))

        cap_id = int(max(set(int(v) for v in cap_subset.tolist()), key=lambda v: int(np.count_nonzero(cap_subset == v)))) if cap_subset.size else 0
        cell_ids = np.flatnonzero(mask).astype(int).tolist()
        region_pd = polydata_from_cell_ids(pd_tri, cell_ids)
        loop_profiles = extract_boundary_loops(region_pd, source_name=f"{face_array_name}_region_boundary")
        loop_profiles = sorted(loop_profiles, key=lambda loop: (-float(loop.area), -int(loop.n_points)))
        boundary_profile = loop_profiles[0] if loop_profiles else None
        boundary_pts = np.asarray(boundary_profile.profile_points, dtype=float) if boundary_profile is not None else np.zeros((0, 3), dtype=float)
        boundary_area, boundary_normal, boundary_rms = planar_polygon_area_and_normal(boundary_pts) if boundary_pts.shape[0] >= 3 else (0.0, np.zeros(3), float("nan"))
        boundary_diameter = math.sqrt(4.0 * boundary_area / math.pi) if boundary_area > 0.0 else 0.0

        regions[int(face_id)] = {
            "face_id": int(face_id),
            "cell_count": int(np.count_nonzero(mask)),
            "total_area": float(total_area),
            "centroid": np.asarray(centroid, dtype=float),
            "mean_normal": np.asarray(mean_normal, dtype=float),
            "diameter_eq": float(math.sqrt(4.0 * total_area / math.pi)) if total_area > 0.0 else 0.0,
            "cap_id": int(cap_id),
            "cell_ids": cell_ids,
            "planarity_score": float(planarity_score),
            "total_surface_area": float(total_surface_area),
            "boundary_loop_count": int(len(loop_profiles)),
            "boundary_profile_points": np.asarray(boundary_pts, dtype=float),
            "boundary_area": float(boundary_area),
            "boundary_normal": np.asarray(boundary_normal, dtype=float),
            "boundary_rms_planarity": float(boundary_rms),
            "boundary_diameter_eq": float(boundary_diameter),
        }
    return regions


def face_region_to_boundary_profile(region: Dict[str, Any], source_prefix: str = "model_face_id") -> BoundaryProfile:
    profile_points = np.asarray(region.get("boundary_profile_points", np.zeros((0, 3), dtype=float)), dtype=float)
    boundary_area = float(region.get("boundary_area", 0.0))
    area = boundary_area if boundary_area > 0.0 else float(region.get("total_area", 0.0))
    diameter = float(region.get("boundary_diameter_eq", 0.0))
    if diameter <= 0.0 and area > 0.0:
        diameter = float(math.sqrt(4.0 * area / math.pi))
    normal = np.asarray(region.get("boundary_normal", region.get("mean_normal", np.zeros((3,), dtype=float))), dtype=float).reshape(3)
    rms_planarity = float(region.get("boundary_rms_planarity", float("nan")))
    if np.linalg.norm(normal) < EPS:
        normal = np.asarray(region.get("mean_normal", np.zeros((3,), dtype=float)), dtype=float).reshape(3)
    return BoundaryProfile(
        center=np.asarray(region.get("centroid", np.zeros((3,), dtype=float)), dtype=float).reshape(3),
        area=float(area),
        diameter_eq=float(diameter),
        normal=unit(normal),
        rms_planarity=float(rms_planarity),
        n_points=int(profile_points.shape[0]),
        source=f"{source_prefix}:{int(region.get('face_id', -1))}",
        profile_points=profile_points,
        boundary_type="terminal_profile",
        face_id=(None if region.get("face_id") is None else int(region.get("face_id"))),
        cap_id=(None if region.get("cap_id") is None else int(region.get("cap_id"))),
    )


def termination_candidates_from_face_partitions(
    pd_tri: "vtkPolyData",
    face_array: str,
    warnings: Optional[List[str]] = None,
) -> Tuple[List[BoundaryProfile], Dict[int, Dict[str, Any]]]:
    work_warnings = warnings if warnings is not None else []
    regions = extract_face_regions(pd_tri, face_array_name=face_array, cap_array_name="CapID")
    if not regions:
        return [], {}

    region_list = list(regions.values())
    region_list.sort(key=lambda region: (-float(region.get("total_area", 0.0)), int(region.get("face_id", -1))))
    total_surface_area = float(region_list[0].get("total_surface_area", 0.0)) if region_list else 0.0
    max_region_area = float(max(float(region.get("total_area", 0.0)) for region in region_list)) if region_list else 0.0
    cap_candidates = [region for region in region_list if int(region.get("cap_id", 0)) > 0 and int(region.get("boundary_loop_count", 0)) >= 1]

    if len(cap_candidates) >= 2:
        terms = [face_region_to_boundary_profile(region, source_prefix=f"face_partition:{face_array}") for region in cap_candidates]
        terms.sort(key=lambda term: (-float(term.area), -float(term.diameter_eq), str(term.source)))
        return terms, regions

    candidates: List[BoundaryProfile] = []
    for region in region_list:
        area = float(region.get("total_area", 0.0))
        planarity = float(region.get("planarity_score", 0.0))
        if area <= 0.0:
            continue
        if int(region.get("boundary_loop_count", 0)) < 1:
            continue
        if planarity < 0.92:
            continue
        if total_surface_area > 0.0 and area > 0.60 * total_surface_area:
            continue
        if max_region_area > 0.0 and area > 0.85 * max_region_area and len(region_list) > 3:
            continue
        candidates.append(face_region_to_boundary_profile(region, source_prefix=f"face_partition:{face_array}"))

    if len(candidates) >= 2:
        work_warnings.append(f"W_TERMINATIONS_FACEPART: boundary loops not found; using planar face partitions via '{face_array}'.")
        candidates.sort(key=lambda term: (-float(term.area), -float(term.diameter_eq), str(term.source)))
        return candidates, regions
    return [], regions


def detect_terminations(
    pd_tri: "vtkPolyData",
    warnings: List[str],
) -> Tuple[List[BoundaryProfile], str, Dict[int, Dict[str, Any]]]:
    if count_boundary_edges(pd_tri) > 0:
        loops = extract_boundary_loops(pd_tri, source_name="boundary_loop")
        if len(loops) >= 2:
            loops.sort(key=lambda loop: (-float(loop.area), -float(loop.diameter_eq), str(loop.source)))
            return loops, "open_termini", {}

    face_array = find_face_partition_array_name(pd_tri)
    if face_array:
        terms, regions = termination_candidates_from_face_partitions(pd_tri, face_array, warnings=warnings)
        if len(terms) >= 2:
            return terms, "capped_partitioned", regions

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
            loops: List[BoundaryProfile] = []
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
                    BoundaryProfile(
                        center=center.astype(float),
                        area=float(area),
                        diameter_eq=float(diameter_eq),
                        normal=normal.astype(float),
                        rms_planarity=float(rms),
                        n_points=int(nids),
                        source="feature_edge_loop",
                        profile_points=np.asarray(coords, dtype=float),
                        boundary_type="terminal_profile",
                    )
                )
            if len(loops) >= 2:
                warnings.append("W_TERMINATIONS_FEATUREEDGES: boundary loops not found; using feature-edge loops (less reliable).")
                loops.sort(key=lambda loop: (-float(loop.area), -float(loop.diameter_eq), str(loop.source)))
                return loops, "closed_unpartitioned", {}

    warnings.append("W_TERMINATIONS_NONE: failed to detect terminations robustly.")
    return [], "unsupported", {}


def choose_root_termination(
    terms: List[BoundaryProfile],
    surface_points: np.ndarray,
    warnings: List[str],
) -> Tuple[Optional[BoundaryProfile], float, np.ndarray]:
    if not terms:
        return None, 0.0, np.array([0.0, 0.0, 1.0], dtype=float)

    A, _, _ = pca_axes(surface_points if surface_points.shape[0] > 0 else np.array([t.center for t in terms], dtype=float))
    axis = unit(A[:, 0])
    if np.linalg.norm(axis) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    centers = np.array([t.center for t in terms], dtype=float)
    diams = np.array([max(t.diameter_eq, 0.0) for t in terms], dtype=float)
    areas = np.array([max(t.area, 0.0) for t in terms], dtype=float)
    cc = np.mean(centers, axis=0)
    proj = (centers - cc) @ axis

    order = np.argsort(diams)[::-1]
    t1 = int(order[0])
    t2 = int(order[1]) if len(order) > 1 else None

    distal_sign: Optional[float] = None
    paired_end_conf = 0.0
    if t2 is not None and diams[t1] > 0 and diams[t2] > 0:
        s1 = float(np.sign(proj[t1])) if abs(proj[t1]) > EPS else 0.0
        s2 = float(np.sign(proj[t2])) if abs(proj[t2]) > EPS else 0.0
        if s1 != 0.0 and s1 == s2:
            dvec = centers[t1] - centers[t2]
            lateral_sep = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
            avg_d = 0.5 * (diams[t1] + diams[t2])
            if avg_d > 0:
                sep_ratio = lateral_sep / (avg_d + EPS)
                if sep_ratio > 1.25:
                    distal_sign = s1
                    paired_end_conf = float(clamp((sep_ratio - 1.25) / 1.5 + 0.35, 0.0, 1.0))

    if distal_sign is not None and distal_sign != 0.0:
        root_sign = -distal_sign
        candidates = [i for i in range(len(terms)) if float(np.sign(proj[i]) if abs(proj[i]) > EPS else 0.0) == root_sign]
        if not candidates:
            root_idx = int(np.argmin(proj)) if distal_sign > 0 else int(np.argmax(proj))
            warnings.append("W_ROOT_SIDE_EMPTY: paired distal terminals inferred, but no terminal on the opposite axial side; using axial extreme.")
        else:
            proj_norm = (proj - np.min(proj)) / (np.ptp(proj) + EPS)
            diam_norm = (diams - np.min(diams)) / (np.ptp(diams) + EPS)
            area_norm = (areas - np.min(areas)) / (np.ptp(areas) + EPS)
            scores = []
            for i in candidates:
                axial_score = float(proj_norm[i] if root_sign > 0 else (1.0 - proj_norm[i]))
                score = 0.55 * float(diam_norm[i]) + 0.25 * axial_score + 0.20 * float(area_norm[i])
                scores.append((score, i))
            scores.sort(reverse=True)
            root_idx = int(scores[0][1])
        if proj[root_idx] < 0:
            axis = -axis
            proj = -proj
        diam_sorted = np.sort(diams)[::-1]
        diam_ratio = float(diam_sorted[0] / (diam_sorted[1] + EPS)) if len(diam_sorted) > 1 else 2.0
        root_extremity = float(abs(proj[root_idx]) / (np.max(abs(proj)) + EPS))
        conf = float(clamp(0.40 + 0.35 * paired_end_conf + 0.15 * clamp(diam_ratio - 1.0, 0.0, 1.0) + 0.10 * root_extremity, 0.0, 1.0))
        return terms[root_idx], conf, axis

    top_k = min(4, len(terms))
    top = order[:top_k]
    scores = []
    for i in top:
        score = float(diams[i]) + 0.15 * float(abs(proj[i]))
        scores.append((score, int(i)))
    scores.sort(reverse=True)
    root_idx = int(scores[0][1])
    if proj[root_idx] < 0:
        axis = -axis
    diam_sorted = np.sort(diams)[::-1]
    diam_ratio = float(diam_sorted[0] / (diam_sorted[1] + EPS)) if len(diam_sorted) > 1 else 2.0
    root_extremity = float(abs(proj[root_idx]) / (np.max(abs(proj)) + EPS))
    conf = float(clamp(0.35 + 0.30 * clamp(diam_ratio - 1.0, 0.0, 1.0) + 0.35 * root_extremity, 0.0, 1.0))
    if conf < 0.55:
        warnings.append("W_ROOT_LOWCONF: root termination may be ambiguous; used diameter+axis score selection.")
    return terms[root_idx], conf, axis


def _extract_failing_extension_module(diagnostics: Dict[str, Any]) -> Optional[str]:
    attempts = diagnostics.get("import_attempts", [])
    for attempt in attempts:
        if attempt.get("ok"):
            continue
        name = str(attempt.get("name", ""))
        if name.startswith("probe "):
            return name.replace("probe ", "", 1).strip() or None
    return None


def _format_vmtk_import_failure_details(diagnostics: Optional[Dict[str, Any]]) -> str:
    if not diagnostics:
        return "No VMTK diagnostics available."
    attempts = diagnostics.get("import_attempts", [])
    lines = [
        f"python_executable: {diagnostics.get('python_executable', sys.executable)}",
        f"sys.prefix: {diagnostics.get('sys_prefix', sys.prefix)}",
        f"CONDA_PREFIX: {diagnostics.get('conda_prefix') or '<unset>'}",
        f"vtk import: {'ok' if diagnostics.get('vtk_import_ok') else 'failed'}",
        f"VMTK import: {'ok' if diagnostics.get('vmtk_import_ok') else 'failed'}",
    ]
    if diagnostics.get("failing_extension_module"):
        lines.append(f"failing extension module: {diagnostics.get('failing_extension_module')}")
    if diagnostics.get("dll_directories_added"):
        lines.append(f"dll dirs added: {', '.join(diagnostics.get('dll_directories_added', []))}")
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


def compute_centerlines_vmtk(
    pd_tri: "vtkPolyData",
    root_center: np.ndarray,
    term_centers: List[np.ndarray],
    warnings: List[str],
) -> Tuple["vtkPolyData", Dict[str, Any]]:
    vtkvmtk_mod, err = try_import_vmtk()
    diagnostics = dict(_LAST_VMTK_IMPORT_DIAGNOSTICS) if _LAST_VMTK_IMPORT_DIAGNOSTICS else {}
    if vtkvmtk_mod is None:
        raise RuntimeError(
            "VMTK vtkvmtk not available (required).\n"
            f"{err or _format_vmtk_import_failure_details(diagnostics)}"
        )

    capped, did_cap = cap_surface_if_open(pd_tri, vtkvmtk_mod)
    locator = build_static_locator(capped)

    root_pid = int(locator.FindClosestPoint(float(root_center[0]), float(root_center[1]), float(root_center[2])))
    target_pids: List[int] = []
    seen = {root_pid}
    for c in term_centers:
        pid = int(locator.FindClosestPoint(float(c[0]), float(c[1]), float(c[2])))
        if pid in seen:
            continue
        p = np.array(capped.GetPoint(pid), dtype=float)
        if float(np.linalg.norm(p - root_center)) < 1e-6:
            continue
        seen.add(pid)
        target_pids.append(int(pid))

    if len(target_pids) < 1:
        raise RuntimeError("Insufficient target seeds for centerline extraction (need >=1).")

    bbox = capped.GetBounds()
    diag = float(np.linalg.norm(np.array([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float)))
    step = max(0.005 * diag, 0.5)

    info: Dict[str, Any] = {
        "did_cap": bool(did_cap),
        "root_pid": int(root_pid),
        "n_targets": int(len(target_pids)),
        "resampling_step": float(step),
        "flip_normals": None,
        "vmtk_import_source": diagnostics.get("resolved_vmtk_source"),
    }
    if diagnostics.get("resolved_vmtk_source") == "extension_fallback":
        info["vmtk_fallback_modules"] = diagnostics.get("loaded_probe_modules", [])

    source_ids = vtk.vtkIdList()
    source_ids.InsertNextId(root_pid)
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


def compute_centerlines_with_root_trials(
    pd_tri: "vtkPolyData",
    terms: List[BoundaryProfile],
    root_term: BoundaryProfile,
    warnings: List[str],
) -> Tuple["vtkPolyData", Dict[str, Any], BoundaryProfile]:
    root_order: List[BoundaryProfile] = [root_term]
    for term in sorted(terms, key=lambda t: (-float(t.diameter_eq), -float(t.area), str(t.source))):
        if term is root_term:
            continue
        root_order.append(term)

    last_err: Optional[BaseException] = None
    for idx, candidate_root in enumerate(root_order[: max(3, min(len(root_order), 5))]):
        try:
            other_centers = [np.asarray(term.center, dtype=float) for term in terms if term is not candidate_root]
            cl, info = compute_centerlines_vmtk(pd_tri, np.asarray(candidate_root.center, dtype=float), other_centers, warnings)
            info["root_trial_index"] = int(idx)
            info["root_source"] = str(candidate_root.source)
            return cl, info, candidate_root
        except Exception as exc:
            last_err = exc
            warnings.append(f"W_CENTERLINES_ROOT_TRIAL_FAIL_{idx}: {exc}")
    raise RuntimeError(f"Centerline extraction failed for all root trials. Last error: {last_err}")


def build_graph_from_polyline_centerlines(cl: "vtkPolyData") -> Tuple[Dict[int, Dict[int, float]], np.ndarray, List[int]]:
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
        dedup_ids: List[int] = []
        for pid in ids:
            if not dedup_ids or pid != dedup_ids[-1]:
                dedup_ids.append(int(pid))
        for a, b in zip(dedup_ids[:-1], dedup_ids[1:]):
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


def node_degrees(adjacency: Dict[int, Dict[int, float]]) -> Dict[int, int]:
    return {int(n): int(len(nei)) for n, nei in adjacency.items()}


def edge_key(a: int, b: int) -> Tuple[int, int]:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def build_branch_chains_from_graph(adjacency: Dict[int, Dict[int, float]]) -> List[List[int]]:
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
                if cur == path[0]:
                    break
            chains.append(path)

    dedup: List[List[int]] = []
    seen = set()
    for path in chains:
        if len(path) < 2:
            continue
        k0 = tuple(path)
        k1 = tuple(reversed(path))
        key = min(k0, k1)
        if key in seen:
            continue
        seen.add(key)
        dedup.append([int(v) for v in path])
    return dedup


def dijkstra_shortest_paths(
    adjacency: Dict[int, Dict[int, float]],
    root: int,
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    import heapq

    dist: Dict[int, float] = {int(root): 0.0}
    parent: Dict[int, Optional[int]] = {int(root): None}
    heap: List[Tuple[float, int]] = [(0.0, int(root))]
    seen: set[int] = set()
    while heap:
        cur_d, node = heapq.heappop(heap)
        node = int(node)
        if node in seen:
            continue
        seen.add(node)
        for nei, weight in adjacency.get(node, {}).items():
            cand = float(cur_d + weight)
            if cand + 1e-12 < dist.get(int(nei), float("inf")):
                dist[int(nei)] = cand
                parent[int(nei)] = int(node)
                heapq.heappush(heap, (cand, int(nei)))
    return dist, parent


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def add(self, x: int) -> None:
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        x = int(x)
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return int(self.parent[x])

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def map_terminations_to_centerline_endpoints(
    endpoints: List[int],
    pts: np.ndarray,
    terms: List[BoundaryProfile],
    root_term: BoundaryProfile,
    warnings: List[str],
) -> Dict[int, Dict[str, Any]]:
    if not endpoints:
        warnings.append("W_ENDPOINTS_NONE: centerline graph has no endpoints.")
        return {}

    mapping: Dict[int, Dict[str, Any]] = {}
    used_endpoints: set[int] = set()
    root_idx = int(terms.index(root_term))
    endpoint_candidates = [int(ep) for ep in endpoints]

    def _assign(term_idx: int) -> None:
        term = terms[term_idx]
        best_ep = None
        best_dist = float("inf")
        for ep in endpoint_candidates:
            if ep in used_endpoints:
                continue
            dist = float(np.linalg.norm(pts[int(ep)] - np.asarray(term.center, dtype=float)))
            if dist < best_dist:
                best_dist = dist
                best_ep = int(ep)
        if best_ep is None:
            return
        used_endpoints.add(int(best_ep))
        conf = float(clamp(1.0 - best_dist / (1.25 * max(float(term.diameter_eq), 1.0) + EPS), 0.0, 1.0))
        mapping[int(term_idx)] = {
            "endpoint_node": int(best_ep),
            "distance": float(best_dist),
            "confidence": float(conf),
        }

    _assign(root_idx)
    pair_scores: List[Tuple[float, float, int, int]] = []
    for term_idx, term in enumerate(terms):
        if term_idx == root_idx:
            continue
        diameter_eq = max(float(term.diameter_eq), 1.0)
        for ep in endpoints:
            if int(ep) in used_endpoints:
                continue
            dist = float(np.linalg.norm(pts[int(ep)] - np.asarray(term.center, dtype=float)))
            score = float(dist / (diameter_eq + EPS))
            pair_scores.append((score, dist, int(term_idx), int(ep)))

    for _, dist, term_idx, ep in sorted(pair_scores, key=lambda item: (item[0], item[1], item[2], item[3])):
        if term_idx in mapping or int(ep) in used_endpoints:
            continue
        term = terms[int(term_idx)]
        conf = float(clamp(1.0 - dist / (1.25 * max(float(term.diameter_eq), 1.0) + EPS), 0.0, 1.0))
        mapping[int(term_idx)] = {
            "endpoint_node": int(ep),
            "distance": float(dist),
            "confidence": float(conf),
        }
        used_endpoints.add(int(ep))

    for term_idx, term in enumerate(terms):
        if term_idx in mapping:
            continue
        warnings.append(f"W_TERMINATION_ENDPOINT_MAP_FAILED: failed to map termination '{term.source}' to a unique centerline endpoint.")
    return mapping


def collapse_junction_clusters(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    resampling_step: float,
) -> Tuple[Dict[int, int], List[List[int]], Dict[int, List[int]], Dict[int, int]]:
    deg = node_degrees(adjacency)
    chains = build_branch_chains_from_graph(adjacency)
    dsu = _DisjointSet()
    for node, degree in deg.items():
        if degree >= 3:
            dsu.add(int(node))

    short_chain_len = max(4.5 * float(resampling_step), 1.5)
    short_chain_nodes = 5

    for chain in chains:
        if len(chain) < 2:
            continue
        a = int(chain[0])
        b = int(chain[-1])
        if deg.get(a, 0) >= 3 and deg.get(b, 0) >= 3:
            length = polyline_length(pts[np.asarray(chain, dtype=int)])
            if len(chain) <= short_chain_nodes and length <= short_chain_len:
                dsu.union(a, b)

    supernode_for_keynode: Dict[int, int] = {}
    cluster_members: Dict[int, List[int]] = {}
    next_supernode = 0
    junction_cluster_to_supernode: Dict[int, int] = {}

    for node, degree in sorted(deg.items()):
        if degree == 1:
            sid = next_supernode
            next_supernode += 1
            supernode_for_keynode[int(node)] = sid
            cluster_members[sid] = [int(node)]
        elif degree >= 3:
            root = dsu.find(int(node))
            sid = junction_cluster_to_supernode.get(root)
            if sid is None:
                sid = next_supernode
                next_supernode += 1
                junction_cluster_to_supernode[root] = sid
                cluster_members[sid] = []
            supernode_for_keynode[int(node)] = sid
            cluster_members[sid].append(int(node))

    return supernode_for_keynode, chains, cluster_members, deg


def build_supernode_graph_and_segments(
    chains: List[List[int]],
    supernode_for_keynode: Dict[int, int],
    pts: np.ndarray,
    radii: Optional[np.ndarray],
) -> Tuple[Dict[int, Dict[int, float]], List[Dict[str, Any]]]:
    super_adj: Dict[int, Dict[int, float]] = {}
    segments: List[Dict[str, Any]] = []

    for chain in chains:
        if len(chain) < 2:
            continue
        a = int(chain[0])
        b = int(chain[-1])
        if a not in supernode_for_keynode or b not in supernode_for_keynode:
            continue
        sa = int(supernode_for_keynode[a])
        sb = int(supernode_for_keynode[b])
        if sa == sb:
            continue
        poly_points = pts[np.asarray(chain, dtype=int)]
        segment_length = polyline_length(poly_points)
        seg_radii = np.asarray(radii[np.asarray(chain, dtype=int)], dtype=float) if radii is not None and radii.shape[0] > max(chain) else None
        segments.append(
            {
                "supernode_a": int(sa),
                "supernode_b": int(sb),
                "node_path": [int(v) for v in chain],
                "path_points": np.asarray(poly_points, dtype=float),
                "path_radii": (None if seg_radii is None else np.asarray(seg_radii, dtype=float)),
                "length": float(segment_length),
            }
        )
        super_adj.setdefault(int(sa), {})[int(sb)] = float(segment_length)
        super_adj.setdefault(int(sb), {})[int(sa)] = float(segment_length)

    return super_adj, segments


def orient_boundary_normal(profile: BoundaryProfile, inward_direction: np.ndarray) -> BoundaryProfile:
    out = BoundaryProfile(
        center=np.asarray(profile.center, dtype=float).reshape(3),
        area=float(profile.area),
        diameter_eq=float(profile.diameter_eq),
        normal=unit(np.asarray(profile.normal, dtype=float).reshape(3)),
        rms_planarity=float(profile.rms_planarity),
        n_points=int(profile.n_points),
        source=str(profile.source),
        profile_points=np.asarray(profile.profile_points, dtype=float),
        boundary_type=str(profile.boundary_type),
        face_id=(None if profile.face_id is None else int(profile.face_id)),
        cap_id=(None if profile.cap_id is None else int(profile.cap_id)),
        fallback_used=bool(profile.fallback_used),
        warnings=list(profile.warnings),
    )
    inward = unit(np.asarray(inward_direction, dtype=float).reshape(3))
    if np.linalg.norm(inward) < EPS:
        return out
    if np.linalg.norm(out.normal) < EPS:
        out.normal = inward.astype(float)
        return out
    if float(np.dot(out.normal, inward)) < 0.0:
        out.normal = (-out.normal).astype(float)
    return out


def extract_surface_profile_from_plane(
    surface: "vtkPolyData",
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    anchor_point: np.ndarray,
    source_name: str,
    boundary_type: str,
    fallback_radius: float,
) -> BoundaryProfile:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(plane_origin[0]), float(plane_origin[1]), float(plane_origin[2]))
    plane.SetNormal(float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]))

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)
    cutter.GenerateTrianglesOff()
    cutter.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    cut = stripper.GetOutput()
    pts = cut.GetPoints()

    candidates: List[Tuple[float, float, int, np.ndarray, float, np.ndarray, float]] = []
    if pts is not None:
        anchor = np.asarray(anchor_point, dtype=float).reshape(3)
        for ci in range(cut.GetNumberOfCells()):
            cell = cut.GetCell(ci)
            if cell is None:
                continue
            nids = cell.GetNumberOfPoints()
            if nids < 6:
                continue
            coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
            centroid = np.mean(coords, axis=0)
            area, normal, rms = planar_polygon_area_and_normal(coords)
            if area <= 0.0:
                continue
            diameter_eq = math.sqrt(4.0 * area / math.pi)
            dist = float(np.linalg.norm(centroid - anchor))
            score = float(dist / (diameter_eq + EPS))
            candidates.append((score, dist, int(nids), coords, float(area), normal, float(rms)))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], -item[2]))
        _, _, npts, coords, area, normal, rms = candidates[0]
        centroid = np.mean(coords, axis=0)
        diameter_eq = math.sqrt(4.0 * area / math.pi)
        return BoundaryProfile(
            center=np.asarray(centroid, dtype=float),
            area=float(area),
            diameter_eq=float(diameter_eq),
            normal=unit(np.asarray(normal, dtype=float)),
            rms_planarity=float(rms),
            n_points=int(npts),
            source=str(source_name),
            profile_points=np.asarray(coords, dtype=float),
            boundary_type=str(boundary_type),
            fallback_used=False,
        )

    circle_pts = make_circle_points(np.asarray(plane_origin, dtype=float), np.asarray(plane_normal, dtype=float), max(float(fallback_radius), 1.0), n_points=32)
    area = math.pi * max(float(fallback_radius), 1.0) ** 2
    return BoundaryProfile(
        center=np.asarray(plane_origin, dtype=float).reshape(3),
        area=float(area),
        diameter_eq=float(2.0 * max(float(fallback_radius), 1.0)),
        normal=unit(np.asarray(plane_normal, dtype=float).reshape(3)),
        rms_planarity=0.0,
        n_points=int(circle_pts.shape[0]),
        source=str(source_name),
        profile_points=np.asarray(circle_pts, dtype=float),
        boundary_type=str(boundary_type),
        fallback_used=True,
        warnings=[f"W_PROFILE_FALLBACK: used synthetic circular profile for '{source_name}'."],
    )


def build_terminal_boundary_profile(
    term: Optional[BoundaryProfile],
    fallback_center: np.ndarray,
    inward_direction: np.ndarray,
    fallback_radius: float,
    label: str,
) -> BoundaryProfile:
    if term is not None:
        return orient_boundary_normal(term, inward_direction)
    circle_pts = make_circle_points(fallback_center, inward_direction, max(float(fallback_radius), 1.0), n_points=32)
    area = math.pi * max(float(fallback_radius), 1.0) ** 2
    return BoundaryProfile(
        center=np.asarray(fallback_center, dtype=float).reshape(3),
        area=float(area),
        diameter_eq=float(2.0 * max(float(fallback_radius), 1.0)),
        normal=unit(np.asarray(inward_direction, dtype=float).reshape(3)),
        rms_planarity=0.0,
        n_points=int(circle_pts.shape[0]),
        source=f"terminal_fallback:{label}",
        profile_points=np.asarray(circle_pts, dtype=float),
        boundary_type="fallback_terminal_profile",
        fallback_used=True,
        warnings=[f"W_TERMINAL_PROFILE_FALLBACK: used synthetic terminal profile for '{label}'."],
    )


def local_radius_from_path(path_radii: Optional[np.ndarray], fallback_radius: float, at_start: bool) -> float:
    if path_radii is not None and path_radii.size > 0:
        if at_start:
            return max(float(np.nanmedian(path_radii[: min(3, path_radii.size)])), 1.0)
        return max(float(np.nanmedian(path_radii[max(0, path_radii.size - 3) :])), 1.0)
    return max(float(fallback_radius), 1.0)


def extract_junction_boundary_profile(
    surface: "vtkPolyData",
    path_points: np.ndarray,
    path_radii: Optional[np.ndarray],
    at_start: bool,
    resampling_step: float,
    segment_id: int,
) -> BoundaryProfile:
    total_length = polyline_length(path_points)
    if total_length <= EPS:
        raise RuntimeError(f"Degenerate centerline path for segment {segment_id}.")

    local_radius = local_radius_from_path(path_radii, fallback_radius=max(total_length * 0.1, 1.0), at_start=at_start)
    offset = max(1.35 * local_radius, 2.0 * float(resampling_step), 0.75)
    offset = min(offset, max(0.35 * total_length, min(total_length * 0.45, offset)))
    if total_length > 2.0 * offset:
        boundary_abscissa = offset if at_start else (total_length - offset)
    else:
        boundary_abscissa = 0.30 * total_length if at_start else 0.70 * total_length

    plane_origin = polyline_point_at_abscissa(path_points, boundary_abscissa)
    tangent = polyline_tangent_at_abscissa(path_points, boundary_abscissa)
    if at_start:
        inward = unit(tangent)
    else:
        inward = unit(-tangent)

    profile = extract_surface_profile_from_plane(
        surface=surface,
        plane_origin=np.asarray(plane_origin, dtype=float),
        plane_normal=np.asarray(inward, dtype=float),
        anchor_point=np.asarray(plane_origin, dtype=float),
        source_name=f"junction_profile:segment_{int(segment_id)}:{'proximal' if at_start else 'distal'}",
        boundary_type="junction_profile",
        fallback_radius=float(local_radius),
    )
    return orient_boundary_normal(profile, inward)


def build_boundary_debug_polydata(records: List[Dict[str, Any]]) -> "vtkPolyData":
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    segment_arr = vtk.vtkIntArray()
    segment_arr.SetName("SegmentId")
    side_arr = vtk.vtkIntArray()
    side_arr.SetName("BoundarySide")
    type_arr = vtk.vtkStringArray()
    type_arr.SetName("BoundaryType")
    fallback_arr = vtk.vtkIntArray()
    fallback_arr.SetName("FallbackUsed")

    for rec in records:
        pts_np = np.asarray(rec["profile_points"], dtype=float)
        if pts_np.shape[0] < 2:
            continue
        poly = vtk.vtkPolyLine()
        poly.GetPointIds().SetNumberOfIds(int(pts_np.shape[0]))
        for idx, p in enumerate(pts_np):
            pid = int(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
            poly.GetPointIds().SetId(idx, pid)
        lines.InsertNextCell(poly)
        segment_arr.InsertNextValue(int(rec["segment_id"]))
        side_arr.InsertNextValue(int(rec["boundary_side"]))
        type_arr.InsertNextValue(str(rec["boundary_type"]))
        fallback_arr.InsertNextValue(int(bool(rec["fallback_used"])))

    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)
    out.GetCellData().AddArray(segment_arr)
    out.GetCellData().AddArray(side_arr)
    out.GetCellData().AddArray(type_arr)
    out.GetCellData().AddArray(fallback_arr)
    return out


def build_segment_centerlines_debug_polydata(segments: List[Dict[str, Any]]) -> "vtkPolyData":
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    seg_arr = vtk.vtkIntArray()
    seg_arr.SetName("SegmentId")
    parent_arr = vtk.vtkIntArray()
    parent_arr.SetName("ParentSegmentId")
    length_arr = vtk.vtkDoubleArray()
    length_arr.SetName("SegmentLength")

    for seg in segments:
        pts_np = np.asarray(seg["path_points_oriented"], dtype=float)
        if pts_np.shape[0] < 2:
            continue
        poly = vtk.vtkPolyLine()
        poly.GetPointIds().SetNumberOfIds(int(pts_np.shape[0]))
        for idx, p in enumerate(pts_np):
            pid = int(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
            poly.GetPointIds().SetId(idx, pid)
        lines.InsertNextCell(poly)
        seg_arr.InsertNextValue(int(seg["segment_id"]))
        parent_arr.InsertNextValue(int(seg["parent_segment_id"]) if seg["parent_segment_id"] is not None else -1)
        length_arr.InsertNextValue(float(seg["length"]))

    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)
    out.GetCellData().AddArray(seg_arr)
    out.GetCellData().AddArray(parent_arr)
    out.GetCellData().AddArray(length_arr)
    return out


def hsv_to_rgb_uint8(h: float, s: float, v: float) -> Tuple[int, int, int]:
    h = float(h % 1.0)
    s = clamp(float(s), 0.0, 1.0)
    v = clamp(float(v), 0.0, 1.0)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(round(255.0 * r)), int(round(255.0 * g)), int(round(255.0 * b))


def build_segment_sample_locator(segments: List[Dict[str, Any]], sample_step: float) -> Tuple["vtkStaticPointLocator", Dict[int, int]]:
    points = vtk.vtkPoints()
    sample_segment_map: Dict[int, int] = {}

    for seg in segments:
        pts_np = np.asarray(seg["path_points_oriented"], dtype=float)
        if pts_np.shape[0] == 0:
            continue
        s = compute_abscissa(pts_np)
        total = float(s[-1]) if s.size else 0.0
        n_samples = max(int(math.ceil(total / max(float(sample_step), 0.5))) + 1, int(pts_np.shape[0]))
        for k in range(n_samples):
            target = total * float(k) / float(max(n_samples - 1, 1))
            p = polyline_point_at_abscissa(pts_np, target)
            pid = int(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
            sample_segment_map[pid] = int(seg["segment_id"])

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(pd)
    locator.BuildLocator()
    return locator, sample_segment_map


def point_between_segment_boundaries(point: np.ndarray, segment: Dict[str, Any]) -> bool:
    p = np.asarray(point, dtype=float).reshape(3)
    prox = segment["proximal_boundary"]
    dist = segment["distal_boundary"]
    prox_n = unit(np.asarray(prox.normal, dtype=float))
    dist_n = unit(np.asarray(dist.normal, dtype=float))
    prox_tol = float(segment.get("proximal_plane_tolerance", 0.0))
    dist_tol = float(segment.get("distal_plane_tolerance", 0.0))
    prox_ok = float(np.dot(p - np.asarray(prox.center, dtype=float), prox_n)) >= -prox_tol
    dist_ok = float(np.dot(p - np.asarray(dist.center, dtype=float), dist_n)) >= -dist_tol
    return bool(prox_ok and dist_ok)


def prepared_projection_data(points: np.ndarray, radii: Optional[np.ndarray]) -> Dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return {
            "points": pts,
            "p0": np.zeros((0, 3), dtype=float),
            "v": np.zeros((0, 3), dtype=float),
            "vv": np.zeros((0,), dtype=float),
            "lengths": np.zeros((0,), dtype=float),
            "abscissa": compute_abscissa(pts),
            "radii": (None if radii is None else np.asarray(radii, dtype=float)),
        }
    p0 = pts[:-1]
    v = pts[1:] - pts[:-1]
    vv = np.sum(v * v, axis=1)
    lengths = np.sqrt(np.maximum(vv, 0.0))
    return {
        "points": pts,
        "p0": p0,
        "v": v,
        "vv": vv,
        "lengths": lengths,
        "abscissa": compute_abscissa(pts),
        "radii": (None if radii is None else np.asarray(radii, dtype=float)),
    }


def distance_to_prepared_polyline(point: np.ndarray, prep: Dict[str, Any]) -> Dict[str, Any]:
    pts = np.asarray(prep["points"], dtype=float)
    if pts.shape[0] == 0:
        return {"distance2": float("inf"), "radius": 1.0, "abscissa": 0.0}
    if pts.shape[0] == 1:
        diff = np.asarray(point, dtype=float).reshape(3) - pts[0]
        radius_arr = prep.get("radii")
        radius = float(radius_arr[0]) if radius_arr is not None and np.asarray(radius_arr).size > 0 else 1.0
        return {"distance2": float(np.dot(diff, diff)), "radius": max(radius, 1.0), "abscissa": 0.0}

    q = np.asarray(point, dtype=float).reshape(1, 3)
    p0 = np.asarray(prep["p0"], dtype=float)
    v = np.asarray(prep["v"], dtype=float)
    vv = np.asarray(prep["vv"], dtype=float)
    lengths = np.asarray(prep["lengths"], dtype=float)
    safe_vv = np.where(vv > EPS, vv, 1.0)
    t = np.sum((q - p0) * v, axis=1) / safe_vv
    t = np.clip(t, 0.0, 1.0)
    proj = p0 + t[:, None] * v
    diff = q - proj
    d2 = np.sum(diff * diff, axis=1)
    best_idx = int(np.argmin(d2))
    abscissa = float(prep["abscissa"][best_idx] + t[best_idx] * lengths[best_idx])

    radius_arr = prep.get("radii")
    if radius_arr is None or np.asarray(radius_arr).size == 0:
        radius = 1.0
    else:
        radii = np.asarray(radius_arr, dtype=float)
        r0 = float(radii[min(best_idx, radii.shape[0] - 1)])
        r1 = float(radii[min(best_idx + 1, radii.shape[0] - 1)])
        radius = max((1.0 - float(t[best_idx])) * r0 + float(t[best_idx]) * r1, 1.0)

    return {"distance2": float(d2[best_idx]), "radius": float(radius), "abscissa": float(abscissa)}


def choose_segment_for_point(
    point: np.ndarray,
    candidate_segment_ids: List[int],
    segment_lookup: Dict[int, Dict[str, Any]],
) -> Tuple[int, bool]:
    p = np.asarray(point, dtype=float).reshape(3)
    gated_scores: List[Tuple[float, float, int]] = []
    fallback_scores: List[Tuple[float, float, int]] = []

    for seg_id in candidate_segment_ids:
        seg = segment_lookup.get(int(seg_id))
        if seg is None:
            continue
        proj = distance_to_prepared_polyline(p, seg["projection"])
        radius = max(float(proj["radius"]), float(seg.get("mean_radius", 1.0)), 1.0)
        score = float(math.sqrt(float(proj["distance2"])) / radius)
        fallback_scores.append((score, float(proj["distance2"]), int(seg_id)))
        if point_between_segment_boundaries(p, seg):
            gated_scores.append((score, float(proj["distance2"]), int(seg_id)))

    if gated_scores:
        gated_scores.sort(key=lambda item: (item[0], item[1], item[2]))
        return int(gated_scores[0][2]), False
    if fallback_scores:
        fallback_scores.sort(key=lambda item: (item[0], item[1], item[2]))
        return int(fallback_scores[0][2]), True
    raise RuntimeError("No candidate segments available for point assignment.")


def assign_surface_cells_to_segments(
    surface: "vtkPolyData",
    segments: List[Dict[str, Any]],
    resampling_step: float,
    warnings: List[str],
) -> Tuple[np.ndarray, Dict[int, int], int]:
    centers = build_cell_centers(surface)
    if centers.shape[0] != surface.GetNumberOfCells():
        raise RuntimeError("Failed to compute surface cell centers.")

    sample_step = max(0.75 * float(resampling_step), 0.5)
    locator, sample_segment_map = build_segment_sample_locator(segments, sample_step=sample_step)
    segment_lookup = {int(seg["segment_id"]): seg for seg in segments}

    labels = np.full((surface.GetNumberOfCells(),), -1, dtype=np.int32)
    fallback_counts: Dict[int, int] = {int(seg["segment_id"]): 0 for seg in segments}
    total_fallback = 0
    id_list = vtk.vtkIdList()
    n_samples = max(1, min(24, len(sample_segment_map)))

    for ci, center in enumerate(centers):
        id_list.Reset()
        locator.FindClosestNPoints(int(n_samples), float(center[0]), float(center[1]), float(center[2]), id_list)
        candidate_segment_ids: List[int] = []
        seen_ids = set()
        for k in range(id_list.GetNumberOfIds()):
            sample_pid = int(id_list.GetId(k))
            seg_id = int(sample_segment_map.get(sample_pid, -1))
            if seg_id < 0 or seg_id in seen_ids:
                continue
            seen_ids.add(seg_id)
            candidate_segment_ids.append(seg_id)
        if not candidate_segment_ids:
            candidate_segment_ids = [int(seg["segment_id"]) for seg in segments]
        chosen_seg, used_fallback = choose_segment_for_point(center, candidate_segment_ids, segment_lookup)
        labels[int(ci)] = int(chosen_seg)
        if used_fallback:
            total_fallback += 1
            fallback_counts[int(chosen_seg)] = fallback_counts.get(int(chosen_seg), 0) + 1

    unassigned = int(np.count_nonzero(labels < 0))
    if unassigned > 0:
        warnings.append(f"W_CELL_ASSIGNMENT_UNASSIGNED: {unassigned} cells remained unassigned after main pass.")
        for idx in np.flatnonzero(labels < 0).tolist():
            labels[int(idx)] = int(segments[0]["segment_id"])

    if total_fallback > 0:
        warnings.append(
            f"W_CELL_ASSIGNMENT_FALLBACK: {total_fallback} / {surface.GetNumberOfCells()} cells required fallback assignment outside strict boundary gates."
        )
    return labels.astype(np.int32), fallback_counts, int(total_fallback)


def add_segment_arrays_to_surface(surface: "vtkPolyData", labels: np.ndarray) -> "vtkPolyData":
    out = vtk.vtkPolyData()
    out.DeepCopy(surface)

    seg_arr = numpy_to_vtk(np.asarray(labels, dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    seg_arr.SetName("SegmentId")
    out.GetCellData().AddArray(seg_arr)
    out.GetCellData().SetScalars(seg_arr)

    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    unique_ids = sorted(int(v) for v in np.unique(labels).tolist())
    color_map: Dict[int, Tuple[int, int, int]] = {}
    for order, seg_id in enumerate(unique_ids):
        h = (0.61803398875 * (float(order) + 1.0)) % 1.0
        color_map[int(seg_id)] = hsv_to_rgb_uint8(h, 0.62, 0.95)
    for idx, seg_id in enumerate(labels.tolist()):
        colors[int(idx)] = np.asarray(color_map[int(seg_id)], dtype=np.uint8)

    color_arr = numpy_to_vtk(colors.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    color_arr.SetName("SegmentColorRGB")
    color_arr.SetNumberOfComponents(3)
    out.GetCellData().AddArray(color_arr)
    return out


def np3_to_list(arr: np.ndarray) -> List[float]:
    a = np.asarray(arr, dtype=float).reshape(3)
    return [float(a[0]), float(a[1]), float(a[2])]


def points_to_lists(points: np.ndarray) -> List[List[float]]:
    pts = np.asarray(points, dtype=float)
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pts]


def segment_confidence(segment: Dict[str, Any], total_cells: int) -> float:
    score = 0.30
    if not bool(segment["proximal_boundary"].fallback_used):
        score += 0.20
    if not bool(segment["distal_boundary"].fallback_used):
        score += 0.20
    if float(segment["length"]) > 2.0:
        score += 0.10
    if float(segment.get("mean_radius", 1.0)) > 1.0:
        score += 0.05
    fallback_ratio = float(segment.get("fallback_cell_count", 0)) / max(int(total_cells), 1)
    score += 0.15 * (1.0 - clamp(fallback_ratio * 8.0, 0.0, 1.0))
    return float(clamp(score, 0.0, 1.0))


def build_metadata(
    input_path: str,
    output_surface_path: str,
    output_metadata_path: str,
    segments: List[Dict[str, Any]],
    warnings: List[str],
    termination_mode: str,
    centerline_info: Dict[str, Any],
    total_cells: int,
) -> Dict[str, Any]:
    segments_json: List[Dict[str, Any]] = []
    for seg in sorted(segments, key=lambda item: int(item["segment_id"])):
        prox = seg["proximal_boundary"]
        dist = seg["distal_boundary"]
        segments_json.append(
            {
                "segment_id": int(seg["segment_id"]),
                "proximal_boundary_type": str(prox.boundary_type),
                "distal_boundary_type": str(dist.boundary_type),
                "proximal_boundary_centroid": np3_to_list(prox.center),
                "distal_boundary_centroid": np3_to_list(dist.center),
                "proximal_boundary_normal": np3_to_list(prox.normal),
                "distal_boundary_normal": np3_to_list(dist.normal),
                "proximal_boundary_profile_points": points_to_lists(prox.profile_points),
                "distal_boundary_profile_points": points_to_lists(dist.profile_points),
                "centerline_path_point_ids": [int(v) for v in seg["node_path_oriented"]],
                "centerline_path_xyz": points_to_lists(seg["path_points_oriented"]),
                "parent_segment_id": (None if seg["parent_segment_id"] is None else int(seg["parent_segment_id"])),
                "child_segment_ids": [int(v) for v in seg["child_segment_ids"]],
                "warnings": list(seg["warnings"]),
                "confidence": float(segment_confidence(seg, total_cells)),
                "fallback_flags": {
                    "proximal_boundary_fallback": bool(prox.fallback_used),
                    "distal_boundary_fallback": bool(dist.fallback_used),
                    "cell_assignment_fallback_count": int(seg.get("fallback_cell_count", 0)),
                },
                "length": float(seg["length"]),
                "mean_radius": float(seg.get("mean_radius", 0.0)),
            }
        )

    return {
        "input_vtp_path": os.path.abspath(input_path),
        "output_segments_vtp_path": os.path.abspath(output_surface_path),
        "output_metadata_json_path": os.path.abspath(output_metadata_path),
        "termination_detection_mode": str(termination_mode),
        "centerline_info": centerline_info,
        "segment_count": int(len(segments)),
        "segments": segments_json,
        "warnings": list(warnings),
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    if vtk is None or vtk_to_numpy is None or numpy_to_vtk is None:
        raise RuntimeError(f"VTK import failed: {_VTK_IMPORT_ERROR}")

    warnings: List[str] = []
    input_path = _resolve_user_path(args.input)
    if not input_path or not os.path.isfile(input_path):
        raise RuntimeError(f"Input VTP not found: {input_path}")

    surface_in = read_vtp(input_path)
    surface_clean = sanitize_surface_for_segmentation(surface_in, warnings)
    if args.debug:
        write_vtp(surface_clean, args.surface_cleaned)

    terms, termination_mode, face_regions = detect_terminations(surface_clean, warnings)
    if len(terms) < 2:
        raise RuntimeError("Failed to detect at least two terminal boundaries. Cannot compute a vascular segment tree.")

    surface_points = get_points_numpy(surface_clean)
    root_term, root_conf, root_axis = choose_root_termination(terms, surface_points, warnings)
    if root_term is None:
        raise RuntimeError("Failed to choose a root termination.")

    centerlines, centerline_info, resolved_root_term = compute_centerlines_with_root_trials(surface_clean, terms, root_term, warnings)
    if resolved_root_term is not root_term:
        root_term = resolved_root_term
        warnings.append(f"W_ROOT_TRIAL_CHANGED: centerline extraction used alternate root termination '{root_term.source}'.")
    centerline_info["root_confidence"] = float(root_conf)
    centerline_info["root_axis"] = np3_to_list(root_axis)
    centerline_info["root_source"] = str(root_term.source)

    adjacency, cl_pts, _ = build_graph_from_polyline_centerlines(centerlines)
    if not adjacency:
        raise RuntimeError("Centerline graph is empty after VMTK extraction.")

    radii_raw = get_point_array_numpy(centerlines, "MaximumInscribedSphereRadius")
    radii = np.asarray(radii_raw, dtype=float).reshape(-1) if radii_raw is not None else None

    deg = node_degrees(adjacency)
    endpoints = [int(n) for n, d in sorted(deg.items()) if int(d) == 1]
    if len(endpoints) < 2:
        raise RuntimeError("Centerline graph does not contain enough endpoints.")

    term_to_endpoint = map_terminations_to_centerline_endpoints(endpoints, cl_pts, terms, root_term, warnings)
    root_term_idx = int(terms.index(root_term))
    root_endpoint = int(term_to_endpoint[root_term_idx]["endpoint_node"]) if root_term_idx in term_to_endpoint else int(endpoints[0])

    supernode_for_keynode, chains, _, _ = collapse_junction_clusters(adjacency, cl_pts, float(centerline_info["resampling_step"]))
    super_adj, segment_records = build_supernode_graph_and_segments(chains, supernode_for_keynode, cl_pts, radii)
    if not segment_records:
        raise RuntimeError("Failed to derive any centerline segments from the centerline graph.")

    endpoint_to_supernode = {int(node): int(supernode_for_keynode[int(node)]) for node in endpoints if int(node) in supernode_for_keynode}
    term_supernode_map: Dict[int, int] = {}
    for term_idx, mapping in term_to_endpoint.items():
        ep = int(mapping["endpoint_node"])
        if ep in endpoint_to_supernode:
            term_supernode_map[int(term_idx)] = int(endpoint_to_supernode[ep])

    root_supernode = int(term_supernode_map.get(root_term_idx, endpoint_to_supernode.get(root_endpoint, -1)))
    if root_supernode < 0:
        raise RuntimeError("Failed to map the chosen root termination to the reduced segment graph.")

    dist_super, _ = dijkstra_shortest_paths(super_adj, root_supernode)
    ordered_segments: List[Dict[str, Any]] = []
    for seg_id, seg in enumerate(
        sorted(
            segment_records,
            key=lambda item: (min(int(item["supernode_a"]), int(item["supernode_b"])), max(int(item["supernode_a"]), int(item["supernode_b"]))),
        ),
        start=1,
    ):
        sa = int(seg["supernode_a"])
        sb = int(seg["supernode_b"])
        da = float(dist_super.get(sa, float("inf")))
        db = float(dist_super.get(sb, float("inf")))
        if da <= db:
            proximal_supernode = sa
            distal_supernode = sb
            node_path_oriented = list(seg["node_path"])
            path_points_oriented = np.asarray(seg["path_points"], dtype=float)
            path_radii_oriented = None if seg["path_radii"] is None else np.asarray(seg["path_radii"], dtype=float)
        else:
            proximal_supernode = sb
            distal_supernode = sa
            node_path_oriented = list(reversed(seg["node_path"]))
            path_points_oriented = np.asarray(seg["path_points"], dtype=float)[::-1]
            path_radii_oriented = None if seg["path_radii"] is None else np.asarray(seg["path_radii"], dtype=float)[::-1]

        ordered_segments.append(
            {
                "segment_id": int(seg_id),
                "supernode_proximal": int(proximal_supernode),
                "supernode_distal": int(distal_supernode),
                "node_path_oriented": [int(v) for v in node_path_oriented],
                "path_points_oriented": np.asarray(path_points_oriented, dtype=float),
                "path_radii_oriented": (None if path_radii_oriented is None else np.asarray(path_radii_oriented, dtype=float)),
                "length": float(seg["length"]),
                "warnings": [],
            }
        )

    segments_by_proximal_supernode: Dict[int, List[int]] = {}
    for seg in ordered_segments:
        segments_by_proximal_supernode.setdefault(int(seg["supernode_proximal"]), []).append(int(seg["segment_id"]))

    supernode_to_term: Dict[int, BoundaryProfile] = {}
    for term_idx, supernode_id in term_supernode_map.items():
        supernode_to_term[int(supernode_id)] = terms[int(term_idx)]

    boundary_debug_records: List[Dict[str, Any]] = []
    for seg in ordered_segments:
        path_points = np.asarray(seg["path_points_oriented"], dtype=float)
        path_radii = seg["path_radii_oriented"]
        mean_radius = max(float(np.nanmedian(path_radii)) if path_radii is not None and np.asarray(path_radii).size > 0 else max(float(seg["length"]) * 0.05, 1.0), 1.0)
        seg["mean_radius"] = float(mean_radius)

        start_inward = unit(path_points[1] - path_points[0]) if path_points.shape[0] >= 2 else np.array([0.0, 0.0, 1.0], dtype=float)
        end_inward = unit(path_points[-2] - path_points[-1]) if path_points.shape[0] >= 2 else -start_inward

        prox_super = int(seg["supernode_proximal"])
        dist_supernode = int(seg["supernode_distal"])

        prox_term = supernode_to_term.get(prox_super)
        if prox_term is not None:
            proximal_boundary = build_terminal_boundary_profile(
                prox_term,
                fallback_center=path_points[0],
                inward_direction=start_inward,
                fallback_radius=mean_radius,
                label=f"segment_{seg['segment_id']}_proximal",
            )
        else:
            proximal_boundary = extract_junction_boundary_profile(
                surface=surface_clean,
                path_points=path_points,
                path_radii=path_radii,
                at_start=True,
                resampling_step=float(centerline_info["resampling_step"]),
                segment_id=int(seg["segment_id"]),
            )

        dist_term = supernode_to_term.get(dist_supernode)
        if dist_term is not None:
            distal_boundary = build_terminal_boundary_profile(
                dist_term,
                fallback_center=path_points[-1],
                inward_direction=end_inward,
                fallback_radius=mean_radius,
                label=f"segment_{seg['segment_id']}_distal",
            )
        else:
            distal_boundary = extract_junction_boundary_profile(
                surface=surface_clean,
                path_points=path_points,
                path_radii=path_radii,
                at_start=False,
                resampling_step=float(centerline_info["resampling_step"]),
                segment_id=int(seg["segment_id"]),
            )

        seg["proximal_boundary"] = proximal_boundary
        seg["distal_boundary"] = distal_boundary
        seg["proximal_plane_tolerance"] = float(max(mean_radius * 0.45, 0.5 * float(centerline_info["resampling_step"]), 0.5))
        seg["distal_plane_tolerance"] = float(max(mean_radius * 0.45, 0.5 * float(centerline_info["resampling_step"]), 0.5))
        seg["projection"] = prepared_projection_data(path_points, path_radii)
        seg["warnings"].extend(list(proximal_boundary.warnings))
        seg["warnings"].extend(list(distal_boundary.warnings))

        boundary_debug_records.append(
            {
                "segment_id": int(seg["segment_id"]),
                "boundary_side": 0,
                "boundary_type": str(proximal_boundary.boundary_type),
                "profile_points": np.asarray(proximal_boundary.profile_points, dtype=float),
                "fallback_used": bool(proximal_boundary.fallback_used),
            }
        )
        boundary_debug_records.append(
            {
                "segment_id": int(seg["segment_id"]),
                "boundary_side": 1,
                "boundary_type": str(distal_boundary.boundary_type),
                "profile_points": np.asarray(distal_boundary.profile_points, dtype=float),
                "fallback_used": bool(distal_boundary.fallback_used),
            }
        )

    parent_segment_for_supernode: Dict[int, Optional[int]] = {int(root_supernode): None}
    for seg in sorted(ordered_segments, key=lambda item: float(dist_super.get(int(item["supernode_proximal"]), float("inf")))):
        prox_super = int(seg["supernode_proximal"])
        dist_supernode = int(seg["supernode_distal"])
        seg["parent_segment_id"] = parent_segment_for_supernode.get(prox_super)
        parent_segment_for_supernode[dist_supernode] = int(seg["segment_id"])

    for seg in ordered_segments:
        child_ids = segments_by_proximal_supernode.get(int(seg["supernode_distal"]), [])
        seg["child_segment_ids"] = sorted(int(v) for v in child_ids if int(v) != int(seg["segment_id"]))

    labels, fallback_counts, total_fallback = assign_surface_cells_to_segments(
        surface=surface_clean,
        segments=ordered_segments,
        resampling_step=float(centerline_info["resampling_step"]),
        warnings=warnings,
    )
    for seg in ordered_segments:
        seg["fallback_cell_count"] = int(fallback_counts.get(int(seg["segment_id"]), 0))

    surface_out = add_segment_arrays_to_surface(surface_clean, labels)
    write_vtp(surface_out, args.output_vtp)

    if args.debug:
        write_vtp(build_segment_centerlines_debug_polydata(ordered_segments), args.centerlines_debug)
        boundary_debug = build_boundary_debug_polydata(boundary_debug_records)
        junction_debug = build_boundary_debug_polydata(
            [rec for rec in boundary_debug_records if str(rec["boundary_type"]).startswith("junction")]
        )
        write_vtp(boundary_debug, args.segment_boundaries_debug)
        write_vtp(junction_debug, args.junction_profiles_debug)

    metadata = build_metadata(
        input_path=input_path,
        output_surface_path=args.output_vtp,
        output_metadata_path=args.metadata_json,
        segments=ordered_segments,
        warnings=warnings,
        termination_mode=termination_mode,
        centerline_info=centerline_info,
        total_cells=int(surface_clean.GetNumberOfCells()),
    )
    metadata["face_partition_region_count"] = int(len(face_regions))
    metadata["cell_assignment_total_fallback"] = int(total_fallback)
    write_json(metadata, args.metadata_json)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose a vascular lumen surface into geometric vessel segments and write SegmentId-labeled surface output."
    )
    parser.add_argument("--input", default=INPUT_VTP_PATH, help="Input vascular lumen .vtp file.")
    parser.add_argument("--output-vtp", default=OUTPUT_SEGMENTS_VTP_PATH, help="Main output surface VTP path.")
    parser.add_argument("--metadata-json", default=OUTPUT_METADATA_PATH, help="Metadata JSON output path.")
    parser.add_argument("--surface-cleaned", default=OUTPUT_SURFACE_CLEANED_PATH, help="Optional cleaned surface debug VTP path.")
    parser.add_argument("--centerlines-debug", default=OUTPUT_CENTERLINES_DEBUG_PATH, help="Optional centerlines debug VTP path.")
    parser.add_argument("--junction-profiles-debug", default=OUTPUT_JUNCTION_PROFILES_DEBUG_PATH, help="Optional junction profile debug VTP path.")
    parser.add_argument("--segment-boundaries-debug", default=OUTPUT_SEGMENT_BOUNDARIES_DEBUG_PATH, help="Optional segment boundary debug VTP path.")
    parser.add_argument("--debug", dest="debug", action="store_true", default=WRITE_DEBUG_OUTPUTS, help="Write debug outputs.")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug outputs.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except Exception:
        sys.stderr.write("ERROR: vascular segment decomposition failed.\n")
        sys.stderr.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
