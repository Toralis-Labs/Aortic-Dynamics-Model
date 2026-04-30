#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import heapq
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

SCRIPT_PATH = os.path.abspath(__file__) if "__file__" in globals() else os.path.abspath(sys.argv[0])
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
_AUTO_VMTK_REEXEC_ENV = "CENTERLINE_NETWORK_VMTK_REEXEC_ACTIVE"
_AUTO_VMTK_PYTHON_ENV = "CENTERLINE_NETWORK_VMTK_PYTHON"
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
    seen: Set[str] = set()

    def _add(candidate: str) -> None:
        candidate = (candidate or "").strip()
        if not candidate:
            return
        try:
            candidate_abs = os.path.abspath(candidate)
        except Exception:
            return
        if not os.path.isfile(candidate_abs):
            return
        key = _normalize_path_key(candidate_abs)
        if key in seen or key == _normalize_path_key(sys.executable):
            return
        seen.add(key)
        candidates.append(candidate_abs)

    _add(os.environ.get(_AUTO_VMTK_PYTHON_ENV, ""))
    _add(os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe"))
    _add(os.path.join(SCRIPT_DIR, ".tools", "m", "envs", "vmtk-step2", "python.exe"))
    _add(os.path.join(SCRIPT_DIR, ".tools", "micromamba_root", "envs", "vmtk-step2", "python.exe"))

    home = os.path.expanduser("~")
    conda_roots = [
        os.path.join(home, "miniconda3"),
        os.path.join(home, "anaconda3"),
        os.path.join(home, "mambaforge"),
        os.path.join(home, "miniforge3"),
    ]
    env_names = ("vmtk-step2", "vmtk_env", "vmtk", "simvascular", "sv")
    for root in conda_roots:
        for env_name in env_names:
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

    for candidate in _iter_vmtk_python_candidates():
        if not _python_supports_required_vmtk(candidate):
            continue
        sys.stderr.write(f"INFO: relaunching with VMTK-capable interpreter: {candidate}\n")
        env = os.environ.copy()
        env[_AUTO_VMTK_REEXEC_ENV] = "1"
        env[_AUTO_VMTK_PYTHON_ENV] = candidate
        completed = subprocess.run([candidate, SCRIPT_PATH, *sys.argv[1:]], env=env, check=False)
        raise SystemExit(int(completed.returncode))


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

    prefixes: List[str] = []
    seen: Set[str] = set()
    for raw_prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix, os.path.dirname(sys.executable)):
        if not raw_prefix:
            continue
        prefix = os.path.abspath(raw_prefix)
        key = _normalize_path_key(prefix)
        if key in seen or not os.path.isdir(prefix):
            continue
        seen.add(key)
        prefixes.append(prefix)
    info["prefixes"] = list(prefixes)

    path_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    path_keys = {_normalize_path_key(p) for p in path_entries}
    attempted: Set[str] = set()
    added_keys = {_normalize_path_key(p) for p in _WINDOWS_DLL_DIRECTORIES_ADDED}
    prepend_entries: List[str] = []

    for prefix in prefixes:
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
            if key in attempted:
                continue
            attempted.add(key)
            info["dll_directories_attempted"].append(candidate_abs)
            if hasattr(os, "add_dll_directory") and key not in added_keys:
                try:
                    _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(candidate_abs))
                    _WINDOWS_DLL_DIRECTORIES_ADDED.append(candidate_abs)
                    added_keys.add(key)
                    info["dll_directories_added"].append(candidate_abs)
                except Exception as exc:
                    info["dll_add_errors"][candidate_abs] = _format_exception_text(exc)
            if key not in path_keys:
                prepend_entries.append(candidate_abs)
                path_keys.add(key)
                info["path_prepended"].append(candidate_abs)

    if prepend_entries:
        os.environ["PATH"] = os.pathsep.join(prepend_entries + path_entries)
    _LAST_WINDOWS_DLL_ATTEMPTS = list(info["dll_directories_attempted"])
    return info


_maybe_reexec_with_vmtk_python()
_prepare_windows_dll_search_paths()

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # type: ignore
except Exception as exc:  # pragma: no cover
    vtk = None
    numpy_to_vtk = None
    vtk_to_numpy = None
    _VTK_IMPORT_ERROR = str(exc)

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStaticPointLocator


EPS = 1e-12
NODE_TYPE_ORDINARY = 0
NODE_TYPE_ROOT = 1
NODE_TYPE_TERMINAL = 2
NODE_TYPE_JUNCTION = 3


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


@dataclass
class RawCenterlinePath:
    path_id: int
    termination_index: int
    termination_source: str
    points: np.ndarray
    radii: np.ndarray
    root_distance: float
    terminal_distance: float
    flip_normals: int
    length: float


@dataclass
class SampledCenterlinePath:
    path_id: int
    termination_index: int
    termination_source: str
    points: np.ndarray
    tangents: np.ndarray
    radii: np.ndarray
    abscissa: np.ndarray
    length: float


@dataclass
class JunctionNode:
    node_id: int
    point: np.ndarray
    degree: int
    node_type: int
    support_count: int


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


def unit(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(vv))
    if n < EPS:
        return np.zeros((3,), dtype=float)
    return (vv / n).astype(float)


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
        center = np.mean(pts, axis=0) if pts.shape[0] else np.zeros((3,), dtype=float)
        return np.eye(3, dtype=float), np.ones((3,), dtype=float), center.astype(float)
    center = np.mean(pts, axis=0)
    x = pts - center
    cov = (x.T @ x) / max(1, x.shape[0])
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    e0 = unit(v[:, 0])
    e1 = unit(v[:, 1] - np.dot(v[:, 1], e0) * e0)
    e2 = unit(np.cross(e0, e1))
    return np.column_stack([e0, e1, e2]).astype(float), w.astype(float), center.astype(float)


def planar_polygon_area_and_normal(points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, np.zeros((3,), dtype=float), float("nan")
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-8 * (np.linalg.norm(pts[0]) + 1.0):
        pts = pts[:-1]
    if pts.shape[0] < 3:
        return 0.0, np.zeros((3,), dtype=float), float("nan")
    axes, _, center = pca_axes(pts)
    n = unit(axes[:, 2])
    u = unit(axes[:, 0])
    v = unit(np.cross(n, u))
    x = pts - center
    d = x @ n
    rms = float(np.sqrt(np.mean(d * d))) if d.size else float("nan")
    x2 = x @ u
    y2 = x @ v
    x_next = np.roll(x2, -1)
    y_next = np.roll(y2, -1)
    area = 0.5 * float(abs(np.sum(x2 * y_next - x_next * y2)))
    return float(area), n.astype(float), float(rms)


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def compute_abscissa(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return np.zeros((pts.shape[0],), dtype=float)
    s = np.zeros((pts.shape[0],), dtype=float)
    s[1:] = np.cumsum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
    return s


def project_point_to_segment(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, float, float]:
    q = np.asarray(point, dtype=float).reshape(3)
    a = np.asarray(p0, dtype=float).reshape(3)
    b = np.asarray(p1, dtype=float).reshape(3)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= EPS:
        diff = q - a
        return a.astype(float), 0.0, float(np.dot(diff, diff))
    t = clamp(float(np.dot(q - a, ab) / denom), 0.0, 1.0)
    proj = a + t * ab
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
    for i in range(pts.shape[0] - 1):
        proj, t, d2 = project_point_to_segment(point, pts[i], pts[i + 1])
        ab = float(s[i] + t * np.linalg.norm(pts[i + 1] - pts[i]))
        if best is None or d2 < float(best["distance2"]):
            best = {
                "point": proj.astype(float),
                "segment_index": int(i),
                "t": float(t),
                "distance2": float(d2),
                "abscissa": float(ab),
            }
    return best


def polyline_point_at_abscissa(points: np.ndarray, target_abscissa: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((3,), dtype=float)
    if pts.shape[0] == 1:
        return pts[0].astype(float)
    s = compute_abscissa(pts)
    total = float(s[-1]) if s.size else 0.0
    if total <= EPS:
        return pts[0].astype(float)
    target = clamp(float(target_abscissa), 0.0, total)
    idx = int(np.searchsorted(s, target, side="right") - 1)
    idx = max(0, min(idx, pts.shape[0] - 2))
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
    idx = max(0, min(idx, pts.shape[0] - 2))
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
    c = np.asarray(center, dtype=float).reshape(3)
    u, v = build_orthonormal_frame(normal)
    r = max(float(radius), 1e-6)
    out: List[np.ndarray] = []
    n = max(8, int(n_points))
    for k in range(n):
        ang = 2.0 * math.pi * float(k) / float(n)
        out.append(c + r * math.cos(ang) * u + r * math.sin(ang) * v)
    return np.asarray(out, dtype=float)


def _require_vtk() -> None:
    if vtk is None or vtk_to_numpy is None or numpy_to_vtk is None:
        raise RuntimeError(f"VTK import failed: {_VTK_IMPORT_ERROR}")


def _new_vtk_polydata() -> "vtkPolyData":
    _require_vtk()
    return getattr(vtk, "vtkPolyData")()


def _new_vtk_static_point_locator() -> "vtkStaticPointLocator":
    _require_vtk()
    return getattr(vtk, "vtkStaticPointLocator")()


def _new_vtk_polydata_normals() -> Any:
    _require_vtk()
    return getattr(vtk, "vtkPolyDataNormals")()


def _new_vtk_polydata_connectivity_filter() -> Any:
    _require_vtk()
    return getattr(vtk, "vtkPolyDataConnectivityFilter")()


def write_vtp(pd: "vtkPolyData", path: str, binary: bool = True) -> None:
    _require_vtk()
    if not path:
        return
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if not math.isfinite(v):
            return None
        return v
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(value.__dict__)
    return value


def write_json(data: Dict[str, Any], path: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2, ensure_ascii=False)


def read_vtp(path: str) -> "vtkPolyData":
    _require_vtk()
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    out = _new_vtk_polydata()
    out.DeepCopy(reader.GetOutput())
    return out


def get_points_numpy(pd: "vtkPolyData") -> np.ndarray:
    _require_vtk()
    pts = pd.GetPoints()
    if pts is None or pts.GetData() is None:
        return np.zeros((0, 3), dtype=float)
    return vtk_to_numpy(pts.GetData()).astype(float)


def get_point_array_numpy(pd: "vtkPolyData", name: str) -> Optional[np.ndarray]:
    _require_vtk()
    arr = pd.GetPointData().GetArray(name)
    if arr is None:
        return None
    return vtk_to_numpy(arr)


def get_cell_array_numpy(pd: "vtkPolyData", name: str) -> Optional[np.ndarray]:
    _require_vtk()
    arr = pd.GetCellData().GetArray(name)
    if arr is None:
        return None
    return vtk_to_numpy(arr)


def count_boundary_edges(pd: "vtkPolyData") -> int:
    _require_vtk()
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    out = fe.GetOutput()
    return int(out.GetNumberOfCells()) if out is not None else 0


def build_static_locator(pd: "vtkPolyData") -> "vtkStaticPointLocator":
    _require_vtk()
    locator = _new_vtk_static_point_locator()
    locator.SetDataSet(pd)
    locator.BuildLocator()
    return locator


def build_cell_centers(pd: "vtkPolyData") -> np.ndarray:
    _require_vtk()
    centers = vtk.vtkCellCenters()
    centers.SetInputData(pd)
    centers.VertexCellsOff()
    centers.Update()
    return get_points_numpy(centers.GetOutput())


def prune_polydata_arrays(
    pd: "vtkPolyData",
    keep_point_arrays: Optional[List[str]] = None,
    keep_cell_arrays: Optional[List[str]] = None,
    keep_field_arrays: Optional[List[str]] = None,
) -> "vtkPolyData":
    _require_vtk()
    keep_point = set(str(x) for x in (keep_point_arrays or []))
    keep_cell = set(str(x) for x in (keep_cell_arrays or []))
    keep_field = set(str(x) for x in (keep_field_arrays or []))
    out = _new_vtk_polydata()
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
    _require_vtk()
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

    normals = _new_vtk_polydata_normals()
    normals.SetInputConnection(tri.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()

    out = _new_vtk_polydata()
    out.DeepCopy(normals.GetOutput())
    out.BuildLinks()
    return out


def extract_largest_connected_region(pd: "vtkPolyData") -> "vtkPolyData":
    _require_vtk()
    conn = _new_vtk_polydata_connectivity_filter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(conn.GetOutputPort())
    geom.Update()
    out = _new_vtk_polydata()
    out.DeepCopy(geom.GetOutput())
    return out


def sanitize_surface_for_segmentation(pd_in: "vtkPolyData", warnings: List[str]) -> "vtkPolyData":
    _require_vtk()
    if pd_in is None or pd_in.GetNumberOfPoints() < 3 or pd_in.GetNumberOfCells() < 1:
        raise RuntimeError("Input VTP is empty or invalid.")

    pd = _new_vtk_polydata()
    pd.DeepCopy(pd_in)
    original_lines = int(pd.GetNumberOfLines())
    original_verts = int(pd.GetNumberOfVerts())
    original_strips = int(pd.GetNumberOfStrips())
    if original_lines > 0 or original_verts > 0 or original_strips > 0:
        warnings.append(
            "W_INPUT_MIXED_GEOMETRY: input contained non-surface geometry "
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


def find_face_partition_array_name(pd: "vtkPolyData") -> Optional[str]:
    _require_vtk()
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
        if ("face" in lname and "id" in lname) or lname.endswith("faceid") or lname.endswith("_face"):
            return name
    return None


def polydata_from_cell_ids(pd: "vtkPolyData", cell_ids: List[int]) -> "vtkPolyData":
    _require_vtk()
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
    out = _new_vtk_polydata()
    out.DeepCopy(geom.GetOutput())
    return out


def extract_boundary_loops(pd: "vtkPolyData", source_name: str = "boundary_loop") -> List[BoundaryProfile]:
    _require_vtk()
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    edges = fe.GetOutput()
    if edges is None or edges.GetNumberOfCells() == 0:
        return []

    stripper = vtk.vtkStripper()
    stripper.SetInputData(edges)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    out = stripper.GetOutput()
    pts = out.GetPoints()
    if pts is None:
        return []

    loops: List[BoundaryProfile] = []
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
        diameter_eq = math.sqrt(4.0 * area / math.pi) if area > 0.0 else 0.0
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
    _require_vtk()
    cd = pd_tri.GetCellData()
    if cd is None:
        return {}
    face_arr = cd.GetArray(face_array_name)
    if face_arr is None:
        return {}

    cell_size = vtk.vtkCellSizeFilter()
    cell_size.SetInputData(pd_tri)
    if hasattr(cell_size, "SetComputeArea"):
        cell_size.SetComputeArea(True)
    elif hasattr(cell_size, "ComputeAreaOn"):
        cell_size.ComputeAreaOn()
    if hasattr(cell_size, "SetComputeLength"):
        cell_size.SetComputeLength(False)
    if hasattr(cell_size, "SetComputeVolume"):
        cell_size.SetComputeVolume(False)
    if hasattr(cell_size, "SetComputeVertexCount"):
        cell_size.SetComputeVertexCount(False)
    cell_size.Update()
    pd_area = cell_size.GetOutput()

    normals = _new_vtk_polydata_normals()
    normals.SetInputData(pd_area)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.Update()
    pd_normals = normals.GetOutput()

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
    total_surface_area = float(np.sum(area_vals)) if area_vals.size else 0.0

    regions: Dict[int, Dict[str, Any]] = {}
    for face_id in sorted(int(v) for v in np.unique(face_vals).tolist()):
        mask = np.asarray(face_vals == int(face_id), dtype=bool)
        if not np.any(mask):
            continue
        areas = area_vals[mask]
        centers = centers_vals[mask]
        normals_subset = normal_vals[mask] if normal_vals.shape[0] == face_vals.shape[0] else np.zeros((centers.shape[0], 3), dtype=float)
        cap_subset = cap_vals[mask] if cap_vals.shape[0] == face_vals.shape[0] else np.zeros((centers.shape[0],), dtype=np.int64)
        total_area = float(np.sum(areas))
        if total_area <= EPS:
            centroid = np.mean(centers, axis=0) if centers.shape[0] else np.zeros((3,), dtype=float)
            mean_normal = np.zeros((3,), dtype=float)
            planarity_score = 0.0
        else:
            centroid = np.sum(centers * areas[:, None], axis=0) / total_area
            nsum = np.sum(normals_subset * areas[:, None], axis=0)
            mean_normal = unit(nsum)
            planarity_score = float(np.linalg.norm(nsum) / (total_area + EPS))
        cap_id = 0
        if cap_subset.size:
            unique_cap = [int(v) for v in np.unique(cap_subset).tolist()]
            if unique_cap:
                unique_cap.sort(key=lambda v: int(np.count_nonzero(cap_subset == v)), reverse=True)
                cap_id = int(unique_cap[0])
        cell_ids = np.flatnonzero(mask).astype(int).tolist()
        region_pd = polydata_from_cell_ids(pd_tri, cell_ids)
        loop_profiles = extract_boundary_loops(region_pd, source_name=f"{face_array_name}_region_boundary")
        loop_profiles.sort(key=lambda prof: (-float(prof.area), -int(prof.n_points)))
        boundary_profile = loop_profiles[0] if loop_profiles else None
        boundary_pts = np.asarray(boundary_profile.profile_points, dtype=float) if boundary_profile is not None else np.zeros((0, 3), dtype=float)
        boundary_area, boundary_normal, boundary_rms = planar_polygon_area_and_normal(boundary_pts) if boundary_pts.shape[0] >= 3 else (0.0, np.zeros((3,), dtype=float), float("nan"))
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
    if np.linalg.norm(normal) < EPS:
        normal = np.asarray(region.get("mean_normal", np.zeros((3,), dtype=float)), dtype=float).reshape(3)
    return BoundaryProfile(
        center=np.asarray(region.get("centroid", np.zeros((3,), dtype=float)), dtype=float).reshape(3),
        area=float(area),
        diameter_eq=float(diameter),
        normal=unit(normal),
        rms_planarity=float(region.get("boundary_rms_planarity", float("nan"))),
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
        work_warnings.append(f"W_TERMINATIONS_FACEPART: using planar face partitions via '{face_array}'.")
        candidates.sort(key=lambda term: (-float(term.area), -float(term.diameter_eq), str(term.source)))
        return candidates, regions
    return [], regions


def detect_terminations(pd_tri: "vtkPolyData", warnings: List[str]) -> Tuple[List[BoundaryProfile], str, Dict[int, Dict[str, Any]]]:
    if count_boundary_edges(pd_tri) > 0:
        loops = extract_boundary_loops(pd_tri, source_name="boundary_loop")
        if len(loops) >= 2:
            loops.sort(key=lambda prof: (-float(prof.area), -float(prof.diameter_eq), str(prof.source)))
            return loops, "open_termini", {}

    face_array = find_face_partition_array_name(pd_tri)
    if face_array:
        terms, regions = termination_candidates_from_face_partitions(pd_tri, face_array, warnings=warnings)
        if len(terms) >= 2:
            return terms, "capped_partitioned", regions

    _require_vtk()
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
                        fallback_used=True,
                    )
                )
            if len(loops) >= 2:
                warnings.append("W_TERMINATIONS_FEATUREEDGES: using feature-edge fallback loops (least reliable).")
                loops.sort(key=lambda prof: (-float(prof.area), -float(prof.diameter_eq), str(prof.source)))
                return loops, "feature_edge_fallback", {}
    warnings.append("W_TERMINATIONS_NONE: failed to detect terminations robustly.")
    return [], "unsupported", {}


def choose_root_termination(
    terms: List[BoundaryProfile],
    surface_points: np.ndarray,
    warnings: List[str],
) -> Tuple[Optional[BoundaryProfile], float, np.ndarray]:
    if not terms:
        return None, 0.0, np.array([0.0, 0.0, 1.0], dtype=float)
    points_for_axis = surface_points if surface_points.shape[0] > 0 else np.array([term.center for term in terms], dtype=float)
    axes, _, _ = pca_axes(points_for_axis)
    axis = unit(axes[:, 0])
    if np.linalg.norm(axis) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    centers = np.array([term.center for term in terms], dtype=float)
    diams = np.array([max(float(term.diameter_eq), 0.0) for term in terms], dtype=float)
    areas = np.array([max(float(term.area), 0.0) for term in terms], dtype=float)
    cc = np.mean(centers, axis=0)
    proj = (centers - cc) @ axis

    order = np.argsort(diams)[::-1]
    root_idx = int(order[0])
    t1 = int(order[0])
    t2 = int(order[1]) if len(order) > 1 else None
    distal_sign: Optional[float] = None
    paired_end_conf = 0.0

    if t2 is not None and diams[t1] > 0.0 and diams[t2] > 0.0:
        s1 = float(np.sign(proj[t1])) if abs(proj[t1]) > EPS else 0.0
        s2 = float(np.sign(proj[t2])) if abs(proj[t2]) > EPS else 0.0
        if s1 != 0.0 and s1 == s2:
            dvec = centers[t1] - centers[t2]
            lateral_sep = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
            avg_d = 0.5 * (diams[t1] + diams[t2])
            if avg_d > 0.0:
                sep_ratio = lateral_sep / (avg_d + EPS)
                if sep_ratio > 1.25:
                    distal_sign = s1
                    paired_end_conf = float(clamp((sep_ratio - 1.25) / 1.5 + 0.35, 0.0, 1.0))

    if distal_sign is not None and distal_sign != 0.0:
        root_sign = -distal_sign
        candidates = [i for i in range(len(terms)) if float(np.sign(proj[i]) if abs(proj[i]) > EPS else 0.0) == root_sign]
        if candidates:
            proj_norm = (proj - np.min(proj)) / (np.ptp(proj) + EPS)
            diam_norm = (diams - np.min(diams)) / (np.ptp(diams) + EPS)
            area_norm = (areas - np.min(areas)) / (np.ptp(areas) + EPS)
            scored: List[Tuple[float, int]] = []
            for i in candidates:
                axial_score = float(proj_norm[i] if root_sign > 0 else (1.0 - proj_norm[i]))
                score = 0.55 * float(diam_norm[i]) + 0.25 * axial_score + 0.20 * float(area_norm[i])
                scored.append((score, int(i)))
            scored.sort(reverse=True)
            root_idx = int(scored[0][1])
        else:
            warnings.append("W_ROOT_SIDE_EMPTY: inferred distal pair, but no root-side candidate remained; using axial extreme.")
            root_idx = int(np.argmin(proj)) if distal_sign > 0 else int(np.argmax(proj))
        if proj[root_idx] < 0.0:
            axis = -axis
            proj = -proj
        diam_sorted = np.sort(diams)[::-1]
        diam_ratio = float(diam_sorted[0] / (diam_sorted[1] + EPS)) if len(diam_sorted) > 1 else 2.0
        root_extremity = float(abs(proj[root_idx]) / (np.max(abs(proj)) + EPS))
        conf = float(clamp(0.40 + 0.35 * paired_end_conf + 0.15 * clamp(diam_ratio - 1.0, 0.0, 1.0) + 0.10 * root_extremity, 0.0, 1.0))
        return terms[root_idx], conf, axis

    top_k = min(4, len(terms))
    scored2: List[Tuple[float, int]] = []
    for i in order[:top_k]:
        scored2.append((float(diams[i]) + 0.15 * float(abs(proj[i])), int(i)))
    scored2.sort(reverse=True)
    root_idx = int(scored2[0][1])
    if proj[root_idx] < 0.0:
        axis = -axis
    diam_sorted = np.sort(diams)[::-1]
    diam_ratio = float(diam_sorted[0] / (diam_sorted[1] + EPS)) if len(diam_sorted) > 1 else 2.0
    root_extremity = float(abs(proj[root_idx]) / (np.max(abs(proj)) + EPS))
    conf = float(clamp(0.35 + 0.30 * clamp(diam_ratio - 1.0, 0.0, 1.0) + 0.35 * root_extremity, 0.0, 1.0))
    if conf < 0.55:
        warnings.append("W_ROOT_LOWCONF: root choice is low-confidence; used diameter+axis heuristic.")
    return terms[root_idx], conf, axis


def _extract_failing_extension_module(diagnostics: Dict[str, Any]) -> Optional[str]:
    for attempt in diagnostics.get("import_attempts", []):
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
            attempts.append({"name": name, "ok": True, "module": getattr(module_obj, "__name__", type(module_obj).__name__)})
            return module_obj
        except Exception as exc:
            attempts.append({"name": name, "ok": False, "error": _format_exception_text(exc)})
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
        mod = _attempt(f"probe {module_name}", lambda name=module_name: importlib.import_module(name))
        if mod is not None:
            loaded_probe_modules.append(mod)
    diagnostics["loaded_probe_modules"] = [getattr(m, "__name__", "") for m in loaded_probe_modules]

    if loaded_probe_modules:
        merged = types.ModuleType("vtkvmtk_fallback")
        merged.__dict__["__source_modules__"] = [getattr(m, "__name__", "") for m in loaded_probe_modules]
        for mod in loaded_probe_modules:
            for attr_name in dir(mod):
                if attr_name.startswith("_") or attr_name in merged.__dict__:
                    continue
                merged.__dict__[attr_name] = getattr(mod, attr_name)
        missing = [name for name in _VMTK_REQUIRED_SYMBOLS if not hasattr(merged, name)]
        diagnostics["fallback_missing_symbols"] = list(missing)
        if not missing:
            diagnostics["vmtk_import_ok"] = True
            diagnostics["resolved_vmtk_source"] = "extension_fallback"
            return merged, diagnostics

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
        out = _new_vtk_polydata()
        out.DeepCopy(pd_tri)
        return out, False
    capper = vtkvmtk_mod.vtkvmtkCapPolyData()
    capper.SetInputData(pd_tri)
    if hasattr(capper, "SetDisplacement"):
        capper.SetDisplacement(0.0)
    if hasattr(capper, "SetInPlaneDisplacement"):
        capper.SetInPlaneDisplacement(0.0)
    capper.Update()
    out = _new_vtk_polydata()
    out.DeepCopy(capper.GetOutput())
    out = clean_and_triangulate_surface(out)
    return out, True


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
    _require_vtk()
    pts_src = src.GetPoints()
    if pts_src is None:
        return _new_vtk_polydata()

    out = _new_vtk_polydata()
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
    _require_vtk()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(cl)
    cleaner.PointMergingOn()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.Update()
    merged = _new_vtk_polydata()
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
        used_cells.append(int(ci))
        ids = [int(cell.GetPointId(k)) for k in range(nids)]
        dedup: List[int] = []
        for pid in ids:
            if not dedup or pid != dedup[-1]:
                dedup.append(int(pid))
        for a, b in zip(dedup[:-1], dedup[1:]):
            if a == b:
                continue
            w = float(np.linalg.norm(pts[a] - pts[b]))
            if w <= EPS:
                continue
            adjacency.setdefault(int(a), {})
            adjacency.setdefault(int(b), {})
            if b not in adjacency[a] or w < adjacency[a][b]:
                adjacency[a][b] = float(w)
                adjacency[b][a] = float(w)
    return adjacency, pts, used_cells


def node_degrees(adjacency: Dict[int, Dict[int, float]]) -> Dict[int, int]:
    return {int(node): int(len(nei)) for node, nei in adjacency.items()}


def dijkstra_shortest_paths(adjacency: Dict[int, Dict[int, float]], root: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    dist: Dict[int, float] = {int(root): 0.0}
    parent: Dict[int, Optional[int]] = {int(root): None}
    heap: List[Tuple[float, int]] = [(0.0, int(root))]
    seen: Set[int] = set()
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


def shortest_path_between_nodes(adjacency: Dict[int, Dict[int, float]], start: int, goal: int) -> List[int]:
    if int(start) == int(goal):
        return [int(start)]
    dist, parent = dijkstra_shortest_paths(adjacency, int(start))
    if int(goal) not in dist:
        return []
    path: List[int] = []
    cur: Optional[int] = int(goal)
    while cur is not None:
        path.append(int(cur))
        cur = parent.get(int(cur))
    path.reverse()
    return path if path and path[0] == int(start) else []


def adjacency_connected_components(adjacency: Dict[int, Dict[int, float]]) -> List[List[int]]:
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for start in sorted(adjacency.keys()):
        if int(start) in seen:
            continue
        stack = [int(start)]
        seen.add(int(start))
        comp: List[int] = []
        while stack:
            node = int(stack.pop())
            comp.append(int(node))
            for nei in adjacency.get(node, {}).keys():
                nei_i = int(nei)
                if nei_i in seen:
                    continue
                seen.add(nei_i)
                stack.append(nei_i)
        comp.sort()
        comps.append(comp)
    return comps


def edge_key(a: int, b: int) -> Tuple[int, int]:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def build_branch_chains_from_graph(adjacency: Dict[int, Dict[int, float]]) -> List[List[int]]:
    if not adjacency:
        return []
    deg = node_degrees(adjacency)
    key_nodes = {int(node) for node, degree in deg.items() if degree != 2}
    visited_edges: Set[Tuple[int, int]] = set()
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
    seen_paths: Set[Tuple[int, ...]] = set()
    for path in chains:
        if len(path) < 2:
            continue
        k0 = tuple(int(v) for v in path)
        k1 = tuple(reversed(k0))
        key = min(k0, k1)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        dedup.append([int(v) for v in path])
    return dedup


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


def _nearest_graph_node(point: np.ndarray, pts: np.ndarray, candidate_nodes: List[int]) -> int:
    if not candidate_nodes:
        candidate_nodes = list(range(pts.shape[0]))
    best = int(candidate_nodes[0])
    best_d = float("inf")
    q = np.asarray(point, dtype=float).reshape(3)
    for node in candidate_nodes:
        d = float(np.linalg.norm(pts[int(node)] - q))
        if d < best_d:
            best_d = d
            best = int(node)
    return int(best)


def _extract_ordered_path_from_centerline_result(
    pd: "vtkPolyData",
    root_center: np.ndarray,
    target_center: np.ndarray,
    warnings: List[str],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    adjacency, pts, _ = build_graph_from_polyline_centerlines(pd)
    if pts.shape[0] < 2 or not adjacency:
        raise RuntimeError("Centerline result is empty after graph conversion.")

    deg = node_degrees(adjacency)
    endpoints = sorted([int(node) for node, degree in deg.items() if degree == 1])
    if len(endpoints) < 2:
        endpoints = list(sorted(adjacency.keys()))
    root_node = _nearest_graph_node(root_center, pts, endpoints)
    remaining = [int(node) for node in endpoints if int(node) != int(root_node)]
    target_node = _nearest_graph_node(target_center, pts, remaining if remaining else endpoints)

    node_path = shortest_path_between_nodes(adjacency, int(root_node), int(target_node))
    if len(node_path) < 2:
        if pd.GetNumberOfCells() == 1:
            cell = pd.GetCell(0)
            ids = [int(cell.GetPointId(k)) for k in range(cell.GetNumberOfPoints())]
            p0 = pts[ids[0]]
            p1 = pts[ids[-1]]
            if float(np.linalg.norm(p0 - root_center)) > float(np.linalg.norm(p1 - root_center)):
                ids = list(reversed(ids))
            node_path = ids
        else:
            raise RuntimeError("Failed to reconstruct a usable root-to-terminal centerline path.")

    out_points = pts[np.asarray(node_path, dtype=int)]
    radii_all = get_point_array_numpy(pd, "MaximumInscribedSphereRadius")
    if radii_all is not None and radii_all.shape[0] > max(node_path):
        out_radii = np.asarray(radii_all[np.asarray(node_path, dtype=int)], dtype=float).reshape(-1)
    else:
        warnings.append("W_CENTERLINE_RADIUS_MISSING: MaximumInscribedSphereRadius missing on centerline output; using geometric fallback.")
        step = np.median(np.linalg.norm(out_points[1:] - out_points[:-1], axis=1)) if out_points.shape[0] > 1 else 1.0
        out_radii = np.full((out_points.shape[0],), max(float(step), 1.0), dtype=float)
    return out_points.astype(float), out_radii.astype(float), float(np.linalg.norm(out_points[0] - root_center)), float(np.linalg.norm(out_points[-1] - target_center))


def _path_to_polydata(path: RawCenterlinePath) -> "vtkPolyData":
    _require_vtk()
    pd = _new_vtk_polydata()
    pts_vtk = vtk.vtkPoints()
    n = int(path.points.shape[0])
    for i in range(n):
        p = path.points[i]
        pts_vtk.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        line.GetPointIds().SetId(i, int(i))
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(line)
    pd.SetPoints(pts_vtk)
    pd.SetLines(lines)

    radius_arr = numpy_to_vtk(np.asarray(path.radii, dtype=float), deep=1)
    radius_arr.SetName("MaximumInscribedSphereRadius")
    pd.GetPointData().AddArray(radius_arr)

    ab_arr = numpy_to_vtk(compute_abscissa(path.points).astype(float), deep=1)
    ab_arr.SetName("Abscissa")
    pd.GetPointData().AddArray(ab_arr)

    point_path = numpy_to_vtk(np.full((n,), int(path.path_id), dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
    point_path.SetName("PathId")
    pd.GetPointData().AddArray(point_path)
    point_term = numpy_to_vtk(np.full((n,), int(path.termination_index), dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
    point_term.SetName("TerminationIndex")
    pd.GetPointData().AddArray(point_term)

    cell_path = numpy_to_vtk(np.array([int(path.path_id)], dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
    cell_path.SetName("PathId")
    pd.GetCellData().AddArray(cell_path)
    cell_term = numpy_to_vtk(np.array([int(path.termination_index)], dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
    cell_term.SetName("TerminationIndex")
    pd.GetCellData().AddArray(cell_term)
    cell_flip = numpy_to_vtk(np.array([int(path.flip_normals)], dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
    cell_flip.SetName("FlipNormals")
    pd.GetCellData().AddArray(cell_flip)
    cell_len = numpy_to_vtk(np.array([float(path.length)], dtype=float), deep=1)
    cell_len.SetName("PathLength")
    pd.GetCellData().AddArray(cell_len)
    return pd


def _append_polydata(parts: List["vtkPolyData"]) -> "vtkPolyData":
    _require_vtk()
    append = vtk.vtkAppendPolyData()
    for part in parts:
        append.AddInputData(part)
    append.Update()
    out = _new_vtk_polydata()
    out.DeepCopy(append.GetOutput())
    return out


def compute_centerlines_vmtk(
    pd_tri: "vtkPolyData",
    root_center: np.ndarray,
    term_centers: List[np.ndarray],
    term_indices: List[int],
    term_sources: List[str],
    warnings: List[str],
) -> Tuple["vtkPolyData", List[RawCenterlinePath], Dict[str, Any]]:
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
    use_term_centers: List[np.ndarray] = []
    use_term_indices: List[int] = []
    use_term_sources: List[str] = []
    seen_target_pids: Set[int] = {int(root_pid)}
    for center, term_idx, term_source in zip(term_centers, term_indices, term_sources):
        pid = int(locator.FindClosestPoint(float(center[0]), float(center[1]), float(center[2])))
        if pid in seen_target_pids:
            warnings.append(f"W_TARGET_DUPLICATE_SEED: skipping duplicated target seed for termination '{term_source}'.")
            continue
        seen_target_pids.add(pid)
        target_pids.append(int(pid))
        use_term_centers.append(np.asarray(center, dtype=float).reshape(3))
        use_term_indices.append(int(term_idx))
        use_term_sources.append(str(term_source))

    if not target_pids:
        raise RuntimeError("No valid target seeds remained after root/target filtering.")

    bbox = capped.GetBounds()
    diag = float(np.linalg.norm(np.array([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float)))
    step = max(0.005 * max(diag, 1.0), 0.5)

    info: Dict[str, Any] = {
        "did_cap": bool(did_cap),
        "root_pid": int(root_pid),
        "n_targets": int(len(target_pids)),
        "resampling_step": float(step),
        "flip_normals": None,
        "vmtk_import_source": diagnostics.get("resolved_vmtk_source"),
        "target_seed_ids": [int(pid) for pid in target_pids],
    }
    if diagnostics.get("resolved_vmtk_source") == "extension_fallback":
        info["vmtk_fallback_modules"] = diagnostics.get("loaded_probe_modules", [])

    source_ids = vtk.vtkIdList()
    source_ids.InsertNextId(int(root_pid))

    raw_paths: List[RawCenterlinePath] = []
    last_err: Optional[BaseException] = None
    for flip in (0, 1):
        raw_paths = []
        try:
            for target_pid, target_center, term_idx, term_source in zip(target_pids, use_term_centers, use_term_indices, use_term_sources):
                target_ids = vtk.vtkIdList()
                target_ids.InsertNextId(int(target_pid))
                cl_filter = vtkvmtk_mod.vtkvmtkPolyDataCenterlines()
                cl_filter.SetInputData(capped)
                cl_filter.SetSourceSeedIds(source_ids)
                cl_filter.SetTargetSeedIds(target_ids)
                cl_filter.SetRadiusArrayName("MaximumInscribedSphereRadius")
                cl_filter.SetCostFunction("1/R")
                cl_filter.SetFlipNormals(int(flip))
                cl_filter.SetAppendEndPointsToCenterlines(1)
                cl_filter.SetCenterlineResampling(1)
                cl_filter.SetResamplingStepLength(float(step))
                cl_filter.Update()

                single_out = cl_filter.GetOutput()
                if single_out is None or single_out.GetNumberOfPoints() < 2 or single_out.GetNumberOfCells() < 1:
                    raise RuntimeError(f"vtkvmtkPolyDataCenterlines returned empty output for target seed {target_pid}.")
                clean_single = clean_centerlines_preserve_lines(single_out)
                if clean_single.GetNumberOfPoints() < 2 or clean_single.GetNumberOfCells() < 1:
                    raise RuntimeError(f"Centerline output became empty after cleaning for target seed {target_pid}.")
                ordered_points, ordered_radii, root_dist, term_dist = _extract_ordered_path_from_centerline_result(
                    clean_single,
                    np.asarray(root_center, dtype=float),
                    np.asarray(target_center, dtype=float),
                    warnings,
                )
                if ordered_points.shape[0] < 2 or polyline_length(ordered_points) <= EPS:
                    raise RuntimeError(f"Extracted path for target seed {target_pid} was degenerate.")
                raw_paths.append(
                    RawCenterlinePath(
                        path_id=int(len(raw_paths)),
                        termination_index=int(term_idx),
                        termination_source=str(term_source),
                        points=np.asarray(ordered_points, dtype=float),
                        radii=np.asarray(ordered_radii, dtype=float),
                        root_distance=float(root_dist),
                        terminal_distance=float(term_dist),
                        flip_normals=int(flip),
                        length=float(polyline_length(ordered_points)),
                    )
                )
            parts = [_path_to_polydata(path) for path in raw_paths]
            raw_debug = _append_polydata(parts)
            info["flip_normals"] = int(flip)
            info["n_points"] = int(raw_debug.GetNumberOfPoints())
            info["n_cells"] = int(raw_debug.GetNumberOfCells())
            return raw_debug, raw_paths, info
        except Exception as exc:
            last_err = exc
            warnings.append(f"W_VMTK_CENTERLINES_FAIL_FLIP{flip}: {exc}")
    raise RuntimeError(f"Centerline extraction failed for all FlipNormals attempts. Last error: {last_err}")


def compute_centerlines_with_root_trials(
    pd_tri: "vtkPolyData",
    terms: List[BoundaryProfile],
    root_term: BoundaryProfile,
    warnings: List[str],
) -> Tuple["vtkPolyData", List[RawCenterlinePath], Dict[str, Any], BoundaryProfile]:
    root_order: List[BoundaryProfile] = [root_term]
    for term in sorted(terms, key=lambda t: (-float(t.diameter_eq), -float(t.area), str(t.source))):
        if term is root_term:
            continue
        root_order.append(term)

    last_err: Optional[BaseException] = None
    max_trials = max(3, min(len(root_order), 5))
    for trial_idx, candidate_root in enumerate(root_order[:max_trials]):
        try:
            target_centers = [np.asarray(term.center, dtype=float) for term in terms if term is not candidate_root]
            target_indices = [int(i) for i, term in enumerate(terms) if term is not candidate_root]
            target_sources = [str(term.source) for term in terms if term is not candidate_root]
            raw_pd, raw_paths, info = compute_centerlines_vmtk(
                pd_tri,
                np.asarray(candidate_root.center, dtype=float),
                target_centers,
                target_indices,
                target_sources,
                warnings,
            )
            info["root_trial_index"] = int(trial_idx)
            info["root_source"] = str(candidate_root.source)
            return raw_pd, raw_paths, info, candidate_root
        except Exception as exc:
            last_err = exc
            warnings.append(f"W_CENTERLINES_ROOT_TRIAL_FAIL_{trial_idx}: {exc}")
    raise RuntimeError(f"Centerline extraction failed for all root trials. Last error: {last_err}")


def _resample_scalar_by_abscissa(source_ab: np.ndarray, source_values: np.ndarray, target_ab: np.ndarray) -> np.ndarray:
    if source_values.shape[0] == 0:
        return np.zeros((target_ab.shape[0],), dtype=float)
    if source_values.shape[0] == 1 or source_ab.shape[0] <= 1:
        return np.full((target_ab.shape[0],), float(source_values[0]), dtype=float)
    return np.interp(target_ab, source_ab, np.asarray(source_values, dtype=float)).astype(float)


def resample_centerline_path(path: RawCenterlinePath, step: float) -> SampledCenterlinePath:
    pts = np.asarray(path.points, dtype=float)
    radii = np.asarray(path.radii, dtype=float).reshape(-1)
    s = compute_abscissa(pts)
    total = float(s[-1]) if s.size else 0.0
    if pts.shape[0] < 2 or total <= EPS:
        return SampledCenterlinePath(
            path_id=int(path.path_id),
            termination_index=int(path.termination_index),
            termination_source=str(path.termination_source),
            points=pts[:1].astype(float),
            tangents=np.zeros((min(1, pts.shape[0]), 3), dtype=float),
            radii=radii[:1].astype(float) if radii.size else np.ones((min(1, pts.shape[0]),), dtype=float),
            abscissa=np.zeros((min(1, pts.shape[0]),), dtype=float),
            length=float(total),
        )
    sample_step = max(float(step), 0.25)
    targets = np.arange(0.0, total, sample_step, dtype=float)
    if targets.size == 0 or abs(float(targets[-1]) - total) > 1e-8:
        targets = np.concatenate([targets, np.array([total], dtype=float)])
    points_new = np.vstack([polyline_point_at_abscissa(pts, float(t)) for t in targets])
    tangents_new = np.vstack([polyline_tangent_at_abscissa(pts, float(t)) for t in targets])
    radii_new = _resample_scalar_by_abscissa(
        s,
        radii if radii.size == pts.shape[0] else np.full((pts.shape[0],), max(sample_step, 1.0), dtype=float),
        targets,
    )
    return SampledCenterlinePath(
        path_id=int(path.path_id),
        termination_index=int(path.termination_index),
        termination_source=str(path.termination_source),
        points=np.asarray(points_new, dtype=float),
        tangents=np.asarray(tangents_new, dtype=float),
        radii=np.asarray(radii_new, dtype=float),
        abscissa=np.asarray(targets, dtype=float),
        length=float(total),
    )


def _sample_match_threshold(radius_a: float, radius_b: float, step: float) -> float:
    return float(max(0.85 * step, min(2.5 * step, 0.20 * (float(radius_a) + float(radius_b)) + 0.25 * step)))


def _pairwise_shared_prefix_matches(
    path_a: SampledCenterlinePath,
    path_b: SampledCenterlinePath,
    step: float,
) -> Dict[str, Any]:
    n = min(path_a.points.shape[0], path_b.points.shape[0])
    if n == 0:
        return {"last_stable_index": -1, "matched_pairs": [], "window": 0, "stable_length_mm": 0.0}
    window = int(max(3, min(6, math.ceil(2.5 / max(float(step), 0.25)))))
    window = min(window, n)
    overlap = np.zeros((n,), dtype=bool)
    match_j = np.full((n,), -1, dtype=np.int32)

    for i in range(n):
        pa = path_a.points[i]
        ta = unit(path_a.tangents[i])
        ra = float(path_a.radii[i]) if path_a.radii.shape[0] > i else float(step)
        best_score = float("inf")
        best_j = -1
        for j in (i - 1, i, i + 1):
            if j < 0 or j >= path_b.points.shape[0]:
                continue
            pb = path_b.points[j]
            tb = unit(path_b.tangents[j])
            rb = float(path_b.radii[j]) if path_b.radii.shape[0] > j else float(step)
            ab_diff = abs(float(path_a.abscissa[i]) - float(path_b.abscissa[j]))
            if ab_diff > 1.5 * float(step):
                continue
            dist = float(np.linalg.norm(pa - pb))
            align = abs(float(np.dot(ta, tb)))
            dthr = _sample_match_threshold(ra, rb, step)
            if dist > dthr or align < 0.90:
                continue
            score = float(dist / (dthr + EPS) + 0.5 * (1.0 - align) + 0.25 * ab_diff / (float(step) + EPS))
            if score < best_score:
                best_score = score
                best_j = int(j)
        if best_j >= 0:
            overlap[i] = True
            match_j[i] = int(best_j)

    last_stable = -1
    stable_started = False
    bad_windows = 0
    for start in range(0, n - window + 1):
        frac = float(np.count_nonzero(overlap[start:start + window])) / float(window)
        if frac >= 0.80:
            stable_started = True
            last_stable = max(last_stable, start + window - 1)
            bad_windows = 0
        elif stable_started:
            bad_windows += 1
            if bad_windows >= 2:
                break
    if last_stable < 0 and overlap[0]:
        last_stable = 0

    matched_pairs: List[Tuple[int, int]] = []
    if last_stable >= 0:
        for i in range(last_stable + 1):
            j = int(match_j[i])
            if j >= 0:
                matched_pairs.append((int(i), int(j)))
    stable_length_mm = float(min(path_a.abscissa[last_stable], path_b.abscissa[matched_pairs[-1][1]]) if matched_pairs and last_stable >= 0 else 0.0)
    return {
        "last_stable_index": int(last_stable),
        "matched_pairs": matched_pairs,
        "window": int(window),
        "stable_length_mm": float(stable_length_mm),
    }


def _simplify_cluster_sequence(seq: List[int]) -> List[int]:
    out: List[int] = []
    for cid in seq:
        cid_i = int(cid)
        if out and cid_i == out[-1]:
            continue
        if len(out) >= 2 and cid_i == out[-2]:
            out.pop()
            continue
        out.append(cid_i)
    return out


def _keep_root_component(
    adjacency: Dict[int, Dict[int, float]],
    root_node: int,
    warnings: List[str],
    component_label: str,
    length_threshold: float,
) -> Tuple[Dict[int, Dict[int, float]], List[Dict[str, Any]]]:
    if not adjacency:
        return {}, []
    comps = adjacency_connected_components(adjacency)
    root_comp_set: Set[int] = set()
    for comp in comps:
        if int(root_node) in comp:
            root_comp_set = {int(node) for node in comp}
            break
    removed: List[Dict[str, Any]] = []
    if not root_comp_set:
        return adjacency, removed
    for comp in comps:
        comp_set = {int(node) for node in comp}
        if comp_set == root_comp_set:
            continue
        comp_edge_len = 0.0
        seen_edges: Set[Tuple[int, int]] = set()
        for node in comp:
            for nei, w in adjacency.get(int(node), {}).items():
                ek = edge_key(int(node), int(nei))
                if ek in seen_edges:
                    continue
                seen_edges.add(ek)
                comp_edge_len += float(w)
        removed.append({"nodes": list(sorted(comp_set)), "length": float(comp_edge_len), "component_label": str(component_label)})
        if comp_edge_len > length_threshold:
            warnings.append(f"W_{component_label}_DISCONNECTED_COMPONENT: removing disconnected component of length {comp_edge_len:.3f}.")
    new_adj: Dict[int, Dict[int, float]] = {}
    for node in sorted(root_comp_set):
        for nei, w in adjacency.get(int(node), {}).items():
            if int(nei) not in root_comp_set:
                continue
            new_adj.setdefault(int(node), {})[int(nei)] = float(w)
    return new_adj, removed


def _make_junction_debug_polydata(nodes: List[JunctionNode]) -> "vtkPolyData":
    _require_vtk()
    pd = _new_vtk_polydata()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    node_ids: List[int] = []
    node_types: List[int] = []
    degrees: List[int] = []
    supports: List[int] = []
    for item in nodes:
        pid = int(points.InsertNextPoint(float(item.point[0]), float(item.point[1]), float(item.point[2])))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
        node_ids.append(int(item.node_id))
        node_types.append(int(item.node_type))
        degrees.append(int(item.degree))
        supports.append(int(item.support_count))
    pd.SetPoints(points)
    pd.SetVerts(verts)
    if node_ids:
        arr_id = numpy_to_vtk(np.asarray(node_ids, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_id.SetName("NodeId")
        pd.GetPointData().AddArray(arr_id)
        arr_type = numpy_to_vtk(np.asarray(node_types, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_type.SetName("NodeType")
        pd.GetPointData().AddArray(arr_type)
        arr_deg = numpy_to_vtk(np.asarray(degrees, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_deg.SetName("Degree")
        pd.GetPointData().AddArray(arr_deg)
        arr_support = numpy_to_vtk(np.asarray(supports, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_support.SetName("SupportCount")
        pd.GetPointData().AddArray(arr_support)
    return pd


def reconstruct_shared_centerline_network(
    raw_paths: List[RawCenterlinePath],
    reconstruction_step_hint: float,
    warnings: List[str],
) -> Tuple["vtkPolyData", "vtkPolyData", Dict[str, Any]]:
    if not raw_paths:
        raise RuntimeError("No raw centerline paths are available for network reconstruction.")

    reconstruction_step = float(max(0.4, min(1.25, 0.5 * max(reconstruction_step_hint, 0.5))))
    sampled_paths: List[SampledCenterlinePath] = []
    for path in raw_paths:
        if path.points.shape[0] < 2 or polyline_length(path.points) <= EPS:
            warnings.append(f"W_RAW_PATH_DEGENERATE: skipping degenerate raw centerline path {path.path_id}.")
            continue
        sampled_paths.append(resample_centerline_path(path, reconstruction_step))
    if not sampled_paths:
        raise RuntimeError("No usable raw centerline paths remained after resampling.")

    offsets: List[int] = []
    total_samples = 0
    dsu = _DisjointSet()
    for path in sampled_paths:
        offsets.append(total_samples)
        for local_idx in range(path.points.shape[0]):
            dsu.add(total_samples + int(local_idx))
        total_samples += int(path.points.shape[0])

    root_gid = offsets[0]
    for offset in offsets[1:]:
        dsu.union(root_gid, int(offset))

    overlap_records: List[Dict[str, Any]] = []
    for i in range(len(sampled_paths)):
        for j in range(i + 1, len(sampled_paths)):
            overlap_info = _pairwise_shared_prefix_matches(sampled_paths[i], sampled_paths[j], reconstruction_step)
            overlap_records.append(
                {
                    "path_a": int(sampled_paths[i].path_id),
                    "path_b": int(sampled_paths[j].path_id),
                    "stable_length_mm": float(overlap_info["stable_length_mm"]),
                    "match_count": int(len(overlap_info["matched_pairs"])),
                    "window": int(overlap_info["window"]),
                }
            )
            if overlap_info["last_stable_index"] < 0 or overlap_info["stable_length_mm"] < 1.5 * reconstruction_step:
                continue
            for ia, jb in overlap_info["matched_pairs"]:
                dsu.union(int(offsets[i] + ia), int(offsets[j] + jb))

    cluster_members_global: Dict[int, List[int]] = {}
    for global_id in range(total_samples):
        root = dsu.find(int(global_id))
        cluster_members_global.setdefault(int(root), []).append(int(global_id))

    cluster_ids_by_root: Dict[int, int] = {}
    root_id = dsu.find(root_gid)
    sorted_roots = [int(root_id)] + [int(r) for r in sorted(cluster_members_global.keys()) if int(r) != int(root_id)]
    for new_id, root in enumerate(sorted_roots):
        cluster_ids_by_root[int(root)] = int(new_id)

    sample_point_records: List[np.ndarray] = []
    sample_radius_records: List[float] = []
    sample_path_records: List[int] = []
    sample_ab_records: List[float] = []
    for path in sampled_paths:
        for local_idx in range(path.points.shape[0]):
            sample_point_records.append(np.asarray(path.points[local_idx], dtype=float))
            sample_radius_records.append(float(path.radii[local_idx]) if path.radii.shape[0] > local_idx else float(reconstruction_step))
            sample_path_records.append(int(path.path_id))
            sample_ab_records.append(float(path.abscissa[local_idx]))

    n_clusters = len(cluster_ids_by_root)
    cluster_points = np.zeros((n_clusters, 3), dtype=float)
    cluster_radii = np.zeros((n_clusters,), dtype=float)
    cluster_support_paths: Dict[int, Set[int]] = {cid: set() for cid in range(n_clusters)}

    for root, members in cluster_members_global.items():
        cid = int(cluster_ids_by_root[int(root)])
        pts = np.asarray([sample_point_records[mid] for mid in members], dtype=float)
        rads = np.asarray([sample_radius_records[mid] for mid in members], dtype=float)
        cluster_points[cid] = np.mean(pts, axis=0)
        cluster_radii[cid] = float(np.mean(rads)) if rads.size else float(reconstruction_step)
        for mid in members:
            cluster_support_paths[cid].add(int(sample_path_records[mid]))

    path_cluster_sequences: Dict[int, List[int]] = {}
    edge_support_paths: Dict[Tuple[int, int], Set[int]] = {}
    edge_length_sum: Dict[Tuple[int, int], float] = {}
    edge_count: Dict[Tuple[int, int], int] = {}
    for path_idx, path in enumerate(sampled_paths):
        seq: List[int] = []
        for local_idx in range(path.points.shape[0]):
            gid = offsets[path_idx] + int(local_idx)
            cid = int(cluster_ids_by_root[dsu.find(int(gid))])
            seq.append(int(cid))
        seq = _simplify_cluster_sequence(seq)
        if len(seq) < 2:
            warnings.append(f"W_CLUSTER_SEQ_DEGENERATE: reconstructed cluster sequence for path {path.path_id} collapsed to <2 nodes.")
            continue
        path_cluster_sequences[int(path.path_id)] = [int(v) for v in seq]
        for a, b in zip(seq[:-1], seq[1:]):
            if int(a) == int(b):
                continue
            ek = edge_key(int(a), int(b))
            pa = cluster_points[int(a)]
            pb = cluster_points[int(b)]
            obs_len = float(np.linalg.norm(pb - pa))
            if obs_len <= EPS:
                obs_len = reconstruction_step
            edge_support_paths.setdefault(ek, set()).add(int(path.path_id))
            edge_length_sum[ek] = float(edge_length_sum.get(ek, 0.0) + obs_len)
            edge_count[ek] = int(edge_count.get(ek, 0) + 1)

    cluster_adjacency: Dict[int, Dict[int, float]] = {}
    for ek, support_paths in edge_support_paths.items():
        a, b = int(ek[0]), int(ek[1])
        mean_len = float(edge_length_sum.get(ek, 0.0) / max(edge_count.get(ek, 1), 1))
        if mean_len <= EPS:
            continue
        cluster_adjacency.setdefault(a, {})[b] = float(mean_len)
        cluster_adjacency.setdefault(b, {})[a] = float(mean_len)

    root_cluster = int(cluster_ids_by_root[dsu.find(root_gid)])
    cluster_adjacency, removed_cluster_components = _keep_root_component(
        cluster_adjacency,
        root_cluster,
        warnings,
        component_label="CLUSTER_GRAPH",
        length_threshold=max(2.0, 4.0 * reconstruction_step),
    )

    terminal_clusters = sorted(
        {
            int(path_cluster_sequences[path.path_id][-1])
            for path in sampled_paths
            if int(path.path_id) in path_cluster_sequences and path_cluster_sequences[int(path.path_id)]
        }
    )

    supernode_for_keynode, chains, cluster_members, _ = collapse_junction_clusters(cluster_adjacency, cluster_points, reconstruction_step)
    if int(root_cluster) not in supernode_for_keynode:
        supernode_for_keynode[int(root_cluster)] = int(max(supernode_for_keynode.values()) + 1 if supernode_for_keynode else 0)
        cluster_members[int(supernode_for_keynode[int(root_cluster)])] = [int(root_cluster)]
    for terminal_cluster in terminal_clusters:
        if terminal_cluster not in supernode_for_keynode:
            supernode_for_keynode[int(terminal_cluster)] = int(max(supernode_for_keynode.values()) + 1 if supernode_for_keynode else 0)
            cluster_members[int(supernode_for_keynode[int(terminal_cluster)])] = [int(terminal_cluster)]

    supernode_points: Dict[int, np.ndarray] = {}
    supernode_support_counts: Dict[int, int] = {}
    for sid, members in cluster_members.items():
        pts = cluster_points[np.asarray(members, dtype=int)]
        supernode_points[int(sid)] = np.mean(pts, axis=0).astype(float)
        support_paths_union: Set[int] = set()
        for member in members:
            support_paths_union |= cluster_support_paths.get(int(member), set())
        supernode_support_counts[int(sid)] = int(len(support_paths_union))

    super_adjacency: Dict[int, Dict[int, float]] = {}
    segments: List[Dict[str, Any]] = []
    seen_segment_pairs: Set[Tuple[int, int, Tuple[int, ...]]] = set()
    for edge_id, chain in enumerate(chains):
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
        path_clusters = [int(v) for v in chain]
        unique_key = (min(sa, sb), max(sa, sb), tuple(path_clusters))
        if unique_key in seen_segment_pairs:
            continue
        seen_segment_pairs.add(unique_key)

        support_counts_chain: List[int] = []
        length_chain = 0.0
        for u, v in zip(path_clusters[:-1], path_clusters[1:]):
            ek = edge_key(int(u), int(v))
            support_counts_chain.append(int(len(edge_support_paths.get(ek, set()))))
            length_chain += float(cluster_adjacency.get(int(u), {}).get(int(v), np.linalg.norm(cluster_points[int(v)] - cluster_points[int(u)])))
        support_count = int(min(support_counts_chain)) if support_counts_chain else 1

        segment_points_list: List[np.ndarray] = [supernode_points[int(sa)]]
        for interior in path_clusters[1:-1]:
            segment_points_list.append(cluster_points[int(interior)])
        segment_points_list.append(supernode_points[int(sb)])
        segment_points = np.asarray(segment_points_list, dtype=float)
        if segment_points.shape[0] < 2 or polyline_length(segment_points) <= EPS:
            continue
        segments.append(
            {
                "edge_id": int(edge_id),
                "supernode_a": int(sa),
                "supernode_b": int(sb),
                "cluster_chain": [int(v) for v in path_clusters],
                "support_count": int(support_count),
                "length": float(length_chain if length_chain > EPS else polyline_length(segment_points)),
                "path_points": segment_points,
                "edge_type": int(1 if support_count >= 2 else 0),
            }
        )
        super_adjacency.setdefault(int(sa), {})[int(sb)] = float(length_chain if length_chain > EPS else polyline_length(segment_points))
        super_adjacency.setdefault(int(sb), {})[int(sa)] = float(length_chain if length_chain > EPS else polyline_length(segment_points))

    root_supernode = int(supernode_for_keynode[int(root_cluster)])
    super_adjacency, removed_super_components = _keep_root_component(
        super_adjacency,
        root_supernode,
        warnings,
        component_label="SUPER_GRAPH",
        length_threshold=max(2.0, 4.0 * reconstruction_step),
    )
    active_supernodes: Set[int] = set(super_adjacency.keys())
    active_supernodes.add(int(root_supernode))
    segments = [seg for seg in segments if int(seg["supernode_a"]) in active_supernodes and int(seg["supernode_b"]) in active_supernodes]

    super_deg = node_degrees(super_adjacency)
    terminal_supernodes = sorted(
        {
            int(supernode_for_keynode[int(tc)])
            for tc in terminal_clusters
            if int(tc) in supernode_for_keynode and int(supernode_for_keynode[int(tc)]) in active_supernodes and int(supernode_for_keynode[int(tc)]) != int(root_supernode)
        }
    )
    junction_supernodes = sorted([int(node) for node, degree in super_deg.items() if degree >= 3])

    network_pd = _new_vtk_polydata()
    network_points = vtk.vtkPoints()
    network_lines = vtk.vtkCellArray()
    point_node_ids: List[int] = []
    point_node_types: List[int] = []
    point_degrees: List[int] = []
    point_supports: List[int] = []

    shared_pid_for_supernode: Dict[int, int] = {}

    def _get_or_add_supernode_pid(sid: int) -> int:
        sid = int(sid)
        if sid in shared_pid_for_supernode:
            return int(shared_pid_for_supernode[sid])
        p = supernode_points[int(sid)]
        pid = int(network_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
        shared_pid_for_supernode[int(sid)] = pid
        point_node_ids.append(int(sid))
        if sid == int(root_supernode):
            node_type = NODE_TYPE_ROOT
        elif sid in terminal_supernodes:
            node_type = NODE_TYPE_TERMINAL
        elif sid in junction_supernodes:
            node_type = NODE_TYPE_JUNCTION
        else:
            node_type = NODE_TYPE_ORDINARY
        point_node_types.append(int(node_type))
        point_degrees.append(int(super_deg.get(int(sid), 0)))
        point_supports.append(int(supernode_support_counts.get(int(sid), 1)))
        return pid

    cell_edge_ids: List[int] = []
    cell_edge_types: List[int] = []
    cell_lengths: List[float] = []
    cell_supports: List[int] = []
    cell_start_ids: List[int] = []
    cell_end_ids: List[int] = []

    for seg in segments:
        points_seg = np.asarray(seg["path_points"], dtype=float)
        if points_seg.shape[0] < 2 or polyline_length(points_seg) <= EPS:
            continue
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(points_seg.shape[0])
        start_pid = _get_or_add_supernode_pid(int(seg["supernode_a"]))
        line.GetPointIds().SetId(0, int(start_pid))
        for i in range(1, points_seg.shape[0] - 1):
            p = points_seg[i]
            pid = int(network_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
            line.GetPointIds().SetId(i, pid)
            point_node_ids.append(-1)
            point_node_types.append(int(NODE_TYPE_ORDINARY))
            point_degrees.append(2)
            point_supports.append(int(seg["support_count"]))
        end_pid = _get_or_add_supernode_pid(int(seg["supernode_b"]))
        line.GetPointIds().SetId(points_seg.shape[0] - 1, int(end_pid))
        network_lines.InsertNextCell(line)
        cell_edge_ids.append(int(seg["edge_id"]))
        cell_edge_types.append(int(seg["edge_type"]))
        cell_lengths.append(float(seg["length"]))
        cell_supports.append(int(seg["support_count"]))
        cell_start_ids.append(int(seg["supernode_a"]))
        cell_end_ids.append(int(seg["supernode_b"]))

    network_pd.SetPoints(network_points)
    network_pd.SetLines(network_lines)
    if point_node_ids:
        arr_node_id = numpy_to_vtk(np.asarray(point_node_ids, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_node_id.SetName("NodeId")
        network_pd.GetPointData().AddArray(arr_node_id)
        arr_node_type = numpy_to_vtk(np.asarray(point_node_types, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_node_type.SetName("NodeType")
        network_pd.GetPointData().AddArray(arr_node_type)
        arr_degree = numpy_to_vtk(np.asarray(point_degrees, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_degree.SetName("NodeDegree")
        network_pd.GetPointData().AddArray(arr_degree)
        arr_support = numpy_to_vtk(np.asarray(point_supports, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_support.SetName("SupportCount")
        network_pd.GetPointData().AddArray(arr_support)
    if cell_edge_ids:
        arr_edge_id = numpy_to_vtk(np.asarray(cell_edge_ids, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_edge_id.SetName("EdgeId")
        network_pd.GetCellData().AddArray(arr_edge_id)
        arr_edge_type = numpy_to_vtk(np.asarray(cell_edge_types, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_edge_type.SetName("EdgeType")
        network_pd.GetCellData().AddArray(arr_edge_type)
        arr_edge_len = numpy_to_vtk(np.asarray(cell_lengths, dtype=float), deep=1)
        arr_edge_len.SetName("EdgeLength")
        network_pd.GetCellData().AddArray(arr_edge_len)
        arr_edge_support = numpy_to_vtk(np.asarray(cell_supports, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_edge_support.SetName("SupportCount")
        network_pd.GetCellData().AddArray(arr_edge_support)
        arr_start = numpy_to_vtk(np.asarray(cell_start_ids, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_start.SetName("StartNodeId")
        network_pd.GetCellData().AddArray(arr_start)
        arr_end = numpy_to_vtk(np.asarray(cell_end_ids, dtype=np.int32), deep=1, array_type=vtk.VTK_INT)
        arr_end.SetName("EndNodeId")
        network_pd.GetCellData().AddArray(arr_end)

    junction_nodes: List[JunctionNode] = []
    for sid in sorted(active_supernodes):
        if sid == int(root_supernode):
            node_type = NODE_TYPE_ROOT
        elif sid in terminal_supernodes:
            node_type = NODE_TYPE_TERMINAL
        elif sid in junction_supernodes:
            node_type = NODE_TYPE_JUNCTION
        else:
            node_type = NODE_TYPE_ORDINARY
        junction_nodes.append(
            JunctionNode(
                node_id=int(sid),
                point=np.asarray(supernode_points[int(sid)], dtype=float),
                degree=int(super_deg.get(int(sid), 0)),
                node_type=int(node_type),
                support_count=int(supernode_support_counts.get(int(sid), 1)),
            )
        )
    junction_debug_pd = _make_junction_debug_polydata(junction_nodes)

    metadata = {
        "reconstruction_step": float(reconstruction_step),
        "raw_path_count": int(len(raw_paths)),
        "sampled_path_count": int(len(sampled_paths)),
        "sample_cluster_count": int(n_clusters),
        "root_cluster_id": int(root_cluster),
        "root_node_id": int(root_supernode),
        "terminal_node_ids": [int(v) for v in terminal_supernodes],
        "junction_node_ids": [int(v) for v in junction_supernodes],
        "cluster_edge_count": int(sum(len(v) for v in cluster_adjacency.values()) // 2),
        "network_edge_count": int(len(segments)),
        "network_node_count": int(len(active_supernodes)),
        "path_cluster_sequences": {str(k): [int(v) for v in vals] for k, vals in path_cluster_sequences.items()},
        "pairwise_overlap_records": overlap_records,
        "removed_cluster_components": removed_cluster_components,
        "removed_super_components": removed_super_components,
    }
    return network_pd, junction_debug_pd, metadata


def default_output_paths(input_path: str) -> Dict[str, str]:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(SCRIPT_DIR, "Output files", "STEP1")
    return {
        "surface_cleaned": os.path.join(out_dir, f"{stem}_surface_cleaned.vtp"),
        "centerlines_raw_debug": os.path.join(out_dir, f"{stem}_centerlines_raw_debug.vtp"),
        "centerline_network_output": os.path.join(out_dir, f"{stem}_centerline_network.vtp"),
        "junction_nodes_debug": os.path.join(out_dir, f"{stem}_junction_nodes_debug.vtp"),
        "metadata_json": os.path.join(out_dir, f"{stem}_centerline_network_metadata.json"),
    }


def _boundary_profile_to_dict(profile: BoundaryProfile) -> Dict[str, Any]:
    return {
        "center": np.asarray(profile.center, dtype=float).tolist(),
        "area": float(profile.area),
        "diameter_eq": float(profile.diameter_eq),
        "normal": np.asarray(profile.normal, dtype=float).tolist(),
        "rms_planarity": float(profile.rms_planarity) if math.isfinite(float(profile.rms_planarity)) else None,
        "n_points": int(profile.n_points),
        "source": str(profile.source),
        "boundary_type": str(profile.boundary_type),
        "face_id": None if profile.face_id is None else int(profile.face_id),
        "cap_id": None if profile.cap_id is None else int(profile.cap_id),
        "fallback_used": bool(profile.fallback_used),
        "warnings": list(profile.warnings),
    }


def _raw_path_to_dict(path: RawCenterlinePath) -> Dict[str, Any]:
    return {
        "path_id": int(path.path_id),
        "termination_index": int(path.termination_index),
        "termination_source": str(path.termination_source),
        "length": float(path.length),
        "n_points": int(path.points.shape[0]),
        "flip_normals": int(path.flip_normals),
        "root_distance": float(path.root_distance),
        "terminal_distance": float(path.terminal_distance),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and clean a vascular centerline network from a lumen VTP surface.")
    parser.add_argument("--input", required=True, help="Input vascular lumen surface (.vtp).")
    parser.add_argument("--surface-cleaned", default="", help="Optional cleaned surface VTP output path.")
    parser.add_argument("--centerlines-raw-debug", default="", help="Optional raw centerlines debug VTP output path.")
    parser.add_argument("--centerline-network-output", default="", help="Cleaned reconstructed centerline network VTP output path.")
    parser.add_argument("--junction-nodes-debug", default="", help="Optional junction node debug VTP output path.")
    parser.add_argument("--metadata-json", default="", help="Metadata JSON output path.")
    parser.add_argument("--debug", action="store_true", help="Enable extra traceback printing on failure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_vtk()

    input_path = _resolve_user_path(args.input)
    if not input_path or not os.path.isfile(input_path):
        raise RuntimeError(f"Input file not found: {args.input}")
    defaults = default_output_paths(input_path)
    output_surface_cleaned = _resolve_user_path(args.surface_cleaned) or defaults["surface_cleaned"]
    output_centerlines_raw = _resolve_user_path(args.centerlines_raw_debug) or defaults["centerlines_raw_debug"]
    output_network = _resolve_user_path(args.centerline_network_output) or defaults["centerline_network_output"]
    output_junctions = _resolve_user_path(args.junction_nodes_debug) or defaults["junction_nodes_debug"]
    output_metadata = _resolve_user_path(args.metadata_json) or defaults["metadata_json"]

    warnings: List[str] = []
    confidence_notes: List[str] = []

    print("[1/8] Reading input", flush=True)
    pd_in = read_vtp(input_path)
    if pd_in is None or pd_in.GetNumberOfPoints() < 3 or pd_in.GetNumberOfCells() < 1:
        raise RuntimeError("Input VTP is empty or invalid.")

    print("[2/8] Sanitizing surface", flush=True)
    surface_clean = sanitize_surface_for_segmentation(pd_in, warnings)

    print("[3/8] Detecting terminations", flush=True)
    terms, termination_mode, face_regions = detect_terminations(surface_clean, warnings)
    if len(terms) < 2:
        raise RuntimeError("Failed to detect at least two terminal regions.")

    print("[4/8] Choosing root", flush=True)
    surface_points = get_points_numpy(surface_clean)
    root_term, root_confidence, root_axis = choose_root_termination(terms, surface_points, warnings)
    if root_term is None:
        raise RuntimeError("Failed to choose a root termination.")
    initial_root_source = str(root_term.source)
    if root_confidence < 0.6:
        confidence_notes.append("Root selection confidence is modest; downstream centerlines may need manual review.")

    print("[5/8] Computing centerlines", flush=True)
    raw_centerlines_pd, raw_paths, centerline_info, final_root_term = compute_centerlines_with_root_trials(surface_clean, terms, root_term, warnings)
    if final_root_term is not root_term:
        warnings.append(f"W_ROOT_CHANGED_BY_TRIALS: root trial fallback replaced '{root_term.source}' with '{final_root_term.source}'.")
        root_term = final_root_term
        confidence_notes.append("Root trial fallback changed the effective inlet seed after extraction failure.")

    print("[6/8] Building graph", flush=True)
    raw_graph, _, raw_cells = build_graph_from_polyline_centerlines(raw_centerlines_pd)
    raw_graph_deg = node_degrees(raw_graph)
    raw_endpoints = [int(node) for node, degree in raw_graph_deg.items() if degree == 1]

    print("[7/8] Reconstructing shared network", flush=True)
    network_pd, junction_debug_pd, network_info = reconstruct_shared_centerline_network(
        raw_paths,
        float(centerline_info.get("resampling_step", 1.0)),
        warnings,
    )
    if network_pd.GetNumberOfPoints() < 2 or network_pd.GetNumberOfCells() < 1:
        raise RuntimeError("Reconstructed network is empty.")
    final_graph, _, final_cells = build_graph_from_polyline_centerlines(network_pd)

    print("[8/8] Writing outputs", flush=True)
    write_vtp(surface_clean, output_surface_cleaned)
    write_vtp(raw_centerlines_pd, output_centerlines_raw)
    write_vtp(network_pd, output_network)
    write_vtp(junction_debug_pd, output_junctions)

    root_node_id = int(network_info.get("root_node_id", -1))
    terminal_node_ids = [int(v) for v in network_info.get("terminal_node_ids", [])]
    junction_node_ids = [int(v) for v in network_info.get("junction_node_ids", [])]

    metadata: Dict[str, Any] = {
        "input_path": input_path,
        "output_paths": {
            "surface_cleaned": output_surface_cleaned,
            "centerlines_raw_debug": output_centerlines_raw,
            "centerline_network_output": output_network,
            "junction_nodes_debug": output_junctions,
            "metadata_json": output_metadata,
        },
        "termination_detection_mode": str(termination_mode),
        "termination_count": int(len(terms)),
        "terminations": [_boundary_profile_to_dict(term) for term in terms],
        "face_partition_region_count": int(len(face_regions)),
        "root_selection": {
            "heuristic_source": initial_root_source,
            "selected_source": str(root_term.source),
            "changed_by_root_trial": bool(initial_root_source != str(root_term.source)),
            "selected_center": np.asarray(root_term.center, dtype=float).tolist(),
            "selected_diameter_eq": float(root_term.diameter_eq),
            "selected_area": float(root_term.area),
            "heuristic_confidence": float(root_confidence),
            "heuristic_root_axis": np.asarray(root_axis, dtype=float).tolist(),
        },
        "centerline_extraction": {
            "info": centerline_info,
            "raw_path_count": int(len(raw_paths)),
            "raw_paths": [_raw_path_to_dict(path) for path in raw_paths],
            "raw_centerline_points": int(raw_centerlines_pd.GetNumberOfPoints()),
            "raw_centerline_cells": int(raw_centerlines_pd.GetNumberOfCells()),
            "raw_graph_node_count": int(len(raw_graph)),
            "raw_graph_edge_count": int(sum(len(v) for v in raw_graph.values()) // 2),
            "raw_graph_endpoint_count": int(len(raw_endpoints)),
            "raw_graph_polyline_cell_count": int(len(raw_cells)),
        },
        "cleaned_network": {
            "node_count": int(network_info.get("network_node_count", 0)),
            "edge_count": int(network_info.get("network_edge_count", 0)),
            "polydata_point_count": int(network_pd.GetNumberOfPoints()),
            "polydata_cell_count": int(network_pd.GetNumberOfCells()),
            "polydata_graph_node_count": int(len(final_graph)),
            "polydata_graph_edge_count": int(sum(len(v) for v in final_graph.values()) // 2),
            "polydata_graph_polyline_cell_count": int(len(final_cells)),
            "root_node_id": int(root_node_id),
            "terminal_node_ids": terminal_node_ids,
            "junction_node_ids": junction_node_ids,
            "network_info": network_info,
        },
        "warnings": warnings,
        "confidence_notes": confidence_notes,
    }
    write_json(metadata, output_metadata)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write(f"ERROR: {_format_exception_text(exc)}\n")
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise SystemExit(1)
