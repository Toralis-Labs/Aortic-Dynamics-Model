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

"""
Geometry-only vascular surface segment decomposition with parent-aware and
daughter-aware proximal junction refinement.

Primary deliverable:
- Input: one vascular lumen .vtp file
- Output: one cleaned surface .vtp with cell-wise SegmentId labeling and segment
  colors, plus metadata JSON describing segment boundaries and centerline paths.

Dependencies:
- vtk
- numpy
- VMTK python bindings (vtkvmtk) REQUIRED

No manual interaction. Deterministic best-effort behavior with warnings and
explicit low-confidence synthetic boundary artifacts when refinement fails.
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
OUTPUT_OSTIUM_CROSSSECTIONS_DEBUG_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "ostium_crosssections_debug.vtp")
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
    boundary_method: str = "terminal"
    face_id: Optional[int] = None
    cap_id: Optional[int] = None
    fallback_used: bool = False
    synthetic: bool = False
    confidence: float = 0.0
    connection_zone_score: float = 0.0
    parent_projection_point: Optional[np.ndarray] = None
    parent_projection_abscissa: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class TransitionInterface:
    interface_id: int
    child_segment_id: int
    parent_segment_id: int
    contour_points: np.ndarray
    contour_centroid: np.ndarray
    contour_normal: np.ndarray
    parent_projection_point: Optional[np.ndarray]
    parent_projection_abscissa: Optional[float]
    partition_normal: np.ndarray
    partition_axis_u: np.ndarray
    partition_axis_v: np.ndarray
    confidence: float
    connection_zone_score: float
    method_tag: str
    representative_child_abscissa: Optional[float] = None
    stable_zone_start_abscissa: Optional[float] = None
    stable_zone_end_abscissa: Optional[float] = None
    stable_zone_start_index: int = -1
    stable_zone_end_index: int = -1
    representative_index: int = -1
    local_spacing: float = 0.0
    child_window: float = 0.0
    parent_window: float = 0.0
    patch_radius: float = 0.0
    child_radius: float = 0.0
    parent_radius: float = 0.0
    contour_quality: float = 0.0
    axis_stability: float = 0.0
    synthetic: bool = False
    low_confidence: bool = False
    local_partition_success: bool = False
    local_partition_mode: str = "uninitialized"
    local_patch_cell_ids: List[int] = field(default_factory=list)
    local_barrier_cell_ids: List[int] = field(default_factory=list)
    local_child_cell_ids: List[int] = field(default_factory=list)
    local_parent_cell_ids: List[int] = field(default_factory=list)
    local_child_seed_cell_ids: List[int] = field(default_factory=list)
    local_parent_seed_cell_ids: List[int] = field(default_factory=list)
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


def polygon_perimeter(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    closed = pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-8 * (np.linalg.norm(pts[0]) + 1.0):
        closed = np.vstack([pts, pts[0]])
    return float(np.sum(np.linalg.norm(closed[1:] - closed[:-1], axis=1)))


def contour_shape_metrics(points: np.ndarray, normal_hint: Optional[np.ndarray] = None) -> Dict[str, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return {
            "area": 0.0,
            "diameter_eq": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "eccentricity": 1.0,
            "axis_ratio": float("inf"),
            "closure_gap": float("inf"),
            "closure_score": 0.0,
            "normal_alignment": 0.0,
        }
    area, normal, _ = planar_polygon_area_and_normal(pts)
    diameter_eq = math.sqrt(4.0 * area / math.pi) if area > 0.0 else 0.0
    perimeter = polygon_perimeter(pts)
    circularity = clamp(4.0 * math.pi * area / (perimeter * perimeter + EPS), 0.0, 1.0) if perimeter > 0.0 else 0.0
    closure_gap = float(np.linalg.norm(pts[0] - pts[-1]))
    scale = max(diameter_eq, 1.0)
    closure_score = 1.0 - clamp(closure_gap / (0.30 * scale + EPS), 0.0, 1.0)
    a, _, c = pca_axes(pts)
    plane_n = unit(normal_hint if normal_hint is not None else normal)
    u = unit(a[:, 0] - np.dot(a[:, 0], plane_n) * plane_n)
    if np.linalg.norm(u) < EPS:
        u, v = build_orthonormal_frame(plane_n)
    else:
        v = unit(np.cross(plane_n, u))
    x = pts - c
    qx = x @ u
    qy = x @ v
    if qx.size > 1:
        cov_xy = np.cov(qx, qy)
        cross = float(cov_xy[0, 1])
    else:
        cross = 0.0
    cov2 = np.array([[float(np.var(qx)), cross], [cross, float(np.var(qy))]], dtype=float)
    try:
        w2, _ = np.linalg.eigh(cov2)
        w2 = np.sort(np.maximum(w2, 0.0))[::-1]
    except Exception:
        w2 = np.array([1.0, 0.0], dtype=float)
    axis_ratio = math.sqrt(float(w2[0] / max(w2[1], EPS))) if w2[0] > 0.0 else float("inf")
    eccentricity = math.sqrt(max(0.0, 1.0 - float(w2[1] / max(w2[0], EPS)))) if w2[0] > 0.0 else 1.0
    alignment = abs(float(np.dot(unit(normal), plane_n)))
    return {
        "area": float(area),
        "diameter_eq": float(diameter_eq),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "eccentricity": float(clamp(eccentricity, 0.0, 1.0)),
        "axis_ratio": float(axis_ratio),
        "closure_gap": float(closure_gap),
        "closure_score": float(clamp(closure_score, 0.0, 1.0)),
        "normal_alignment": float(clamp(alignment, 0.0, 1.0)),
    }


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
    writer.SetFileName(os.path.abspath(path))
    writer.SetInputData(pd)
    if binary:
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def write_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def read_vtp(path: str) -> "vtkPolyData":
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(os.path.abspath(path))
    reader.Update()
    pd = reader.GetOutput()
    if pd is None or pd.GetNumberOfPoints() < 1:
        raise RuntimeError(f"Failed to read VTP or empty VTP: {path}")
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
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
    best = None
    for i in range(cd.GetNumberOfArrays()):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if not name:
            continue
        if name.lower() == "modelfaceid":
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
                boundary_method="terminal_profile",
                confidence=0.95,
                connection_zone_score=0.0,
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
        boundary_method="terminal_profile",
        face_id=(None if region.get("face_id") is None else int(region.get("face_id"))),
        cap_id=(None if region.get("cap_id") is None else int(region.get("cap_id"))),
        confidence=0.90,
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
                        boundary_method="terminal_profile",
                        confidence=0.65,
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
    sign = 1.0
    n = unit(np.asarray(profile.normal, dtype=float))
    inward = unit(np.asarray(inward_direction, dtype=float))
    if np.linalg.norm(n) > EPS and np.linalg.norm(inward) > EPS and float(np.dot(n, inward)) < 0.0:
        sign = -1.0
    out = BoundaryProfile(
        center=np.asarray(profile.center, dtype=float).reshape(3),
        area=float(profile.area),
        diameter_eq=float(profile.diameter_eq),
        normal=(sign * n).astype(float),
        rms_planarity=float(profile.rms_planarity),
        n_points=int(profile.n_points),
        source=str(profile.source),
        profile_points=np.asarray(profile.profile_points, dtype=float),
        boundary_type=str(profile.boundary_type),
        boundary_method=str(profile.boundary_method),
        face_id=profile.face_id,
        cap_id=profile.cap_id,
        fallback_used=bool(profile.fallback_used),
        synthetic=bool(profile.synthetic),
        confidence=float(profile.confidence),
        connection_zone_score=float(profile.connection_zone_score),
        parent_projection_point=(None if profile.parent_projection_point is None else np.asarray(profile.parent_projection_point, dtype=float).reshape(3)),
        parent_projection_abscissa=(None if profile.parent_projection_abscissa is None else float(profile.parent_projection_abscissa)),
        warnings=list(profile.warnings),
    )
    return out


def boundary_profile_from_contour(
    contour_points: np.ndarray,
    normal_hint: np.ndarray,
    source: str,
    boundary_type: str,
    boundary_method: str,
    confidence: float,
    connection_zone_score: float,
    fallback_used: bool = False,
    synthetic: bool = False,
    parent_projection_point: Optional[np.ndarray] = None,
    parent_projection_abscissa: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> BoundaryProfile:
    pts = np.asarray(contour_points, dtype=float)
    area, normal, rms = planar_polygon_area_and_normal(pts)
    diameter_eq = math.sqrt(4.0 * area / math.pi) if area > 0.0 else 0.0
    center = np.mean(pts, axis=0) if pts.shape[0] else np.zeros((3,), dtype=float)
    if np.linalg.norm(normal) < EPS:
        normal = unit(normal_hint)
    return BoundaryProfile(
        center=np.asarray(center, dtype=float).reshape(3),
        area=float(area),
        diameter_eq=float(diameter_eq),
        normal=unit(normal),
        rms_planarity=float(rms),
        n_points=int(pts.shape[0]),
        source=str(source),
        profile_points=np.asarray(pts, dtype=float),
        boundary_type=str(boundary_type),
        boundary_method=str(boundary_method),
        fallback_used=bool(fallback_used),
        synthetic=bool(synthetic),
        confidence=float(clamp(confidence, 0.0, 1.0)),
        connection_zone_score=float(clamp(connection_zone_score, 0.0, 1.0)),
        parent_projection_point=(None if parent_projection_point is None else np.asarray(parent_projection_point, dtype=float).reshape(3)),
        parent_projection_abscissa=(None if parent_projection_abscissa is None else float(parent_projection_abscissa)),
        warnings=list(warnings or []),
    )


def build_terminal_boundary_profile(
    term: Optional[BoundaryProfile],
    fallback_center: np.ndarray,
    inward_direction: np.ndarray,
    fallback_radius: float,
    label: str,
) -> BoundaryProfile:
    if term is not None:
        out = orient_boundary_normal(term, inward_direction)
        out.boundary_method = "terminal_profile"
        out.confidence = max(float(out.confidence), 0.90)
        return out
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
        boundary_method="terminal_synthetic",
        fallback_used=True,
        synthetic=True,
        confidence=0.20,
        warnings=[f"W_TERMINAL_PROFILE_FALLBACK: used synthetic terminal profile for '{label}'."],
    )


def path_radius_at_abscissa(path_points: np.ndarray, path_radii: Optional[np.ndarray], target_abscissa: float, fallback_radius: float) -> float:
    if path_radii is None:
        return max(float(fallback_radius), 1.0)
    radii = np.asarray(path_radii, dtype=float).reshape(-1)
    if radii.size == 0:
        return max(float(fallback_radius), 1.0)
    s = compute_abscissa(path_points)
    if s.size == 0 or float(s[-1]) <= EPS:
        return max(float(np.nanmedian(radii)), float(fallback_radius), 1.0)
    target = clamp(float(target_abscissa), 0.0, float(s[-1]))
    idx = int(np.searchsorted(s, target, side="right") - 1)
    idx = int(max(0, min(idx, len(s) - 2))) if len(s) >= 2 else 0
    if len(s) < 2:
        return max(float(radii[0]), float(fallback_radius), 1.0)
    ds = float(s[idx + 1] - s[idx])
    if ds <= EPS:
        return max(float(radii[min(idx, radii.size - 1)]), float(fallback_radius), 1.0)
    t = clamp((target - float(s[idx])) / ds, 0.0, 1.0)
    r0 = float(radii[min(idx, radii.size - 1)])
    r1 = float(radii[min(idx + 1, radii.size - 1)])
    return max((1.0 - t) * r0 + t * r1, float(fallback_radius), 1.0)


def characteristic_path_radius(path_radii: Optional[np.ndarray], fallback_radius: float = 1.0) -> float:
    if path_radii is None:
        return max(float(fallback_radius), 1.0)
    radii = np.asarray(path_radii, dtype=float).reshape(-1)
    if radii.size == 0:
        return max(float(fallback_radius), 1.0)
    return max(float(np.nanmedian(radii[: min(5, radii.size)])), float(fallback_radius), 1.0)


def section_spacing_from_diameter(diameter: float, resampling_step: float) -> float:
    return float(clamp(max(0.18 * float(diameter), 0.45 * float(resampling_step), 0.20), 0.20, 0.75))


def sample_abscissae(start_s: float, end_s: float, spacing: float) -> List[float]:
    if end_s <= start_s + EPS:
        return [float(start_s)]
    n = max(2, int(math.ceil((float(end_s) - float(start_s)) / max(float(spacing), 1e-3))) + 1)
    return [float(start_s + (float(end_s) - float(start_s)) * k / float(max(n - 1, 1))) for k in range(n)]


def local_interface_spacing(
    local_radius: float,
    resampling_step: float,
    connection_evidence: float = 0.0,
    contour_quality: float = 1.0,
    axis_stability: float = 1.0,
) -> float:
    base = max(0.14, 0.16 * max(2.0 * float(local_radius), 1.0), 0.38 * float(resampling_step))
    density_scale = 1.0
    density_scale -= 0.28 * clamp(float(connection_evidence), 0.0, 1.0)
    density_scale -= 0.14 * clamp(1.0 - float(axis_stability), 0.0, 1.0)
    density_scale -= 0.12 * clamp(1.0 - float(contour_quality), 0.0, 1.0)
    return float(clamp(base * density_scale, 0.12, 0.80))


def local_interface_window(
    local_radius: float,
    total_length: float,
    spacing: float,
    connection_evidence: float = 0.0,
    contour_quality: float = 1.0,
    axis_stability: float = 1.0,
    role: str = "child",
) -> float:
    scale = 2.8 if role == "child" else 3.6
    window = max(scale * max(2.0 * float(local_radius), 1.0), 4.0 * float(spacing))
    window *= 1.0 + 0.18 * clamp(float(connection_evidence), 0.0, 1.0)
    window *= 1.0 + 0.10 * clamp(1.0 - float(axis_stability), 0.0, 1.0)
    window *= 1.0 + 0.08 * clamp(1.0 - float(contour_quality), 0.0, 1.0)
    lo = min(max(3.0 * float(spacing), 0.12 * max(float(total_length), 1.0)), max(float(total_length), float(spacing)))
    hi_factor = 0.40 if role == "child" else 0.55
    hi = max(lo, hi_factor * max(float(total_length), float(spacing)))
    return float(clamp(window, lo, hi))


def escalate_transition_sampling_window(
    center_s: float,
    total_length: float,
    spacing: float,
    window: float,
) -> List[float]:
    half_window = max(0.6 * float(window), 2.0 * float(spacing))
    start_s = clamp(float(center_s) - half_window, 0.0, float(total_length))
    end_s = clamp(float(center_s) + half_window, start_s, float(total_length))
    return sample_abscissae(start_s, end_s, max(0.5 * float(spacing), 0.12))


def point_plane_signed_distance(point: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray) -> float:
    return float(np.dot(np.asarray(point, dtype=float).reshape(3) - np.asarray(plane_origin, dtype=float).reshape(3), unit(np.asarray(plane_normal, dtype=float).reshape(3))))


def point_to_polyline_distance(point: np.ndarray, polyline_points: np.ndarray, closed: bool = True) -> float:
    pts = np.asarray(polyline_points, dtype=float)
    if pts.shape[0] == 0:
        return float("inf")
    if pts.shape[0] == 1:
        return float(np.linalg.norm(np.asarray(point, dtype=float).reshape(3) - pts[0]))
    best = float("inf")
    n_seg = pts.shape[0] if closed else pts.shape[0] - 1
    for idx in range(max(0, n_seg)):
        p0 = pts[idx]
        p1 = pts[(idx + 1) % pts.shape[0]]
        proj, _, d2 = project_point_to_segment(point, p0, p1)
        _ = proj
        best = min(best, math.sqrt(max(float(d2), 0.0)))
    return float(best)


def detect_stable_transition_zone(
    samples: List[Dict[str, Any]],
    scores: np.ndarray,
    local_scale: float,
) -> Dict[str, Any]:
    if not samples or scores.size == 0:
        return {
            "success": False,
            "representative_index": -1,
            "start_index": -1,
            "end_index": -1,
            "zone_score": 0.0,
            "smoothed_scores": np.zeros((0,), dtype=float),
        }

    smoothed = np.asarray(moving_average(scores.tolist(), 1), dtype=float)
    qualities = np.asarray([float(s.get("quality", 0.0)) for s in samples], dtype=float)
    axes = np.asarray([float(s.get("axis_stability", 0.0)) for s in samples], dtype=float)
    peak_idx = int(np.argmax(smoothed))
    peak_score = float(smoothed[peak_idx])
    high = clamp(max(0.36, 0.82 * peak_score), 0.36, 0.92)
    low = clamp(max(0.26, 0.66 * peak_score), 0.24, high)

    start = peak_idx
    end = peak_idx
    misses = 0
    while start > 0:
        candidate = float(smoothed[start - 1])
        if candidate >= low:
            start -= 1
            misses = 0
            continue
        misses += 1
        if misses >= 2:
            break
        start -= 1
    misses = 0
    while end + 1 < len(samples):
        candidate = float(smoothed[end + 1])
        if candidate >= low:
            end += 1
            misses = 0
            continue
        misses += 1
        if misses >= 2:
            break
        end += 1

    zone_indices = list(range(start, end + 1))
    has_high = any(float(smoothed[i]) >= high for i in zone_indices)
    abscissae = [float(samples[i]["abscissa"]) for i in zone_indices]
    span = (max(abscissae) - min(abscissae)) if abscissae else 0.0
    representative_idx = max(
        zone_indices,
        key=lambda i: (
            0.70 * float(smoothed[i])
            + 0.20 * float(qualities[i])
            + 0.10 * float(axes[i]),
            -abs(i - peak_idx),
        ),
    )
    zone_score = float(np.mean([smoothed[i] for i in zone_indices])) if zone_indices else 0.0
    stable = bool(has_high and (len(zone_indices) >= 2 or span >= 0.55 * max(float(local_scale), 0.5)))
    return {
        "success": stable,
        "representative_index": int(representative_idx),
        "start_index": int(start),
        "end_index": int(end),
        "zone_score": float(clamp(zone_score, 0.0, 1.0)),
        "smoothed_scores": smoothed,
    }


def transition_partition_frame(
    contour_centroid: np.ndarray,
    contour_normal: np.ndarray,
    parent_projection_point: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = unit(np.asarray(contour_normal, dtype=float).reshape(3))
    if parent_projection_point is not None:
        axis_u = np.asarray(contour_centroid, dtype=float).reshape(3) - np.asarray(parent_projection_point, dtype=float).reshape(3)
        axis_u = axis_u - np.dot(axis_u, normal) * normal
        axis_u = unit(axis_u)
    else:
        axis_u, _ = build_orthonormal_frame(normal)
    if np.linalg.norm(axis_u) < EPS:
        axis_u, _ = build_orthonormal_frame(normal)
    axis_v = unit(np.cross(normal, axis_u))
    return normal.astype(float), axis_u.astype(float), axis_v.astype(float)


def concatenate_segment_paths(parent_points: np.ndarray, child_points: np.ndarray) -> np.ndarray:
    a = np.asarray(parent_points, dtype=float)
    b = np.asarray(child_points, dtype=float)
    if a.shape[0] == 0:
        return b.astype(float)
    if b.shape[0] == 0:
        return a.astype(float)
    if np.linalg.norm(a[-1] - b[0]) < 1e-6:
        return np.vstack([a, b[1:]]).astype(float)
    return np.vstack([a, b]).astype(float)


def concatenate_segment_radii(parent_radii: Optional[np.ndarray], child_radii: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if parent_radii is None and child_radii is None:
        return None
    if parent_radii is None:
        return np.asarray(child_radii, dtype=float).copy()
    if child_radii is None:
        return np.asarray(parent_radii, dtype=float).copy()
    a = np.asarray(parent_radii, dtype=float).reshape(-1)
    b = np.asarray(child_radii, dtype=float).reshape(-1)
    if a.size == 0:
        return b.astype(float)
    if b.size == 0:
        return a.astype(float)
    return np.concatenate([a, b[1:] if b.size > 1 else b]).astype(float)


def plane_section_candidates(
    surface: "vtkPolyData",
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    anchor_point: np.ndarray,
    source_name: str,
    boundary_type: str,
) -> List[Dict[str, Any]]:
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
    if pts is None:
        return []

    candidates: List[Dict[str, Any]] = []
    anchor = np.asarray(anchor_point, dtype=float).reshape(3)
    plane_n = unit(np.asarray(plane_normal, dtype=float))

    for ci in range(cut.GetNumberOfCells()):
        cell = cut.GetCell(ci)
        if cell is None:
            continue
        nids = cell.GetNumberOfPoints()
        if nids < 6:
            continue
        coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
        if coords.shape[0] >= 2 and np.linalg.norm(coords[0] - coords[-1]) < 1e-6:
            coords = coords[:-1]
        if coords.shape[0] < 6:
            continue
        metrics = contour_shape_metrics(coords, normal_hint=plane_n)
        area = float(metrics["area"])
        if area <= EPS:
            continue
        centroid = np.mean(coords, axis=0)
        dist = float(np.linalg.norm(centroid - anchor))
        diameter_eq = float(metrics["diameter_eq"])
        dist_score = 1.0 - clamp(dist / (1.40 * max(diameter_eq, 1.0) + EPS), 0.0, 1.0)
        density_score = clamp(float(coords.shape[0]) / 32.0, 0.0, 1.0)
        quality = clamp(
            0.30 * float(metrics["circularity"])
            + 0.20 * float(metrics["closure_score"])
            + 0.20 * float(metrics["normal_alignment"])
            + 0.15 * density_score
            + 0.15 * dist_score,
            0.0,
            1.0,
        )
        profile = boundary_profile_from_contour(
            contour_points=coords,
            normal_hint=plane_n,
            source=source_name,
            boundary_type=boundary_type,
            boundary_method="section_candidate",
            confidence=quality,
            connection_zone_score=0.0,
        )
        candidates.append(
            {
                "profile": profile,
                "quality": float(quality),
                "anchor_distance": float(dist),
                "area": float(metrics["area"]),
                "diameter_eq": float(metrics["diameter_eq"]),
                "circularity": float(metrics["circularity"]),
                "eccentricity": float(metrics["eccentricity"]),
                "axis_ratio": float(metrics["axis_ratio"]),
                "closure_score": float(metrics["closure_score"]),
                "normal_alignment": float(metrics["normal_alignment"]),
                "point_count": int(coords.shape[0]),
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["quality"]),
            float(item["anchor_distance"]) / (max(float(item["diameter_eq"]), 1.0) + EPS),
            -float(item["area"]),
        )
    )
    return candidates


def path_local_axis_stability(path_points: np.ndarray, abscissa: float, lookahead: float) -> float:
    total = float(polyline_length(path_points))
    if total <= EPS:
        return 0.0
    p0 = polyline_tangent_at_abscissa(path_points, abscissa)
    p1 = polyline_tangent_at_abscissa(path_points, clamp(abscissa + 0.5 * lookahead, 0.0, total))
    p2 = polyline_tangent_at_abscissa(path_points, clamp(abscissa + lookahead, 0.0, total))
    vals = [abs(float(np.dot(p0, p1))), abs(float(np.dot(p0, p2))), abs(float(np.dot(p1, p2)))]
    return float(clamp(sum(vals) / max(len(vals), 1), 0.0, 1.0))


def build_section_sample(
    surface: "vtkPolyData",
    family: str,
    path_points: np.ndarray,
    path_radii: Optional[np.ndarray],
    abscissa: float,
    anchor_point: np.ndarray,
    source_name: str,
    boundary_type: str,
    fallback_radius: float,
    partner_path: Optional[np.ndarray] = None,
    partner_prep: Optional[Dict[str, Any]] = None,
    partner_label: str = "partner",
    segment_id: int = -1,
) -> Dict[str, Any]:
    total = float(polyline_length(path_points))
    s = clamp(float(abscissa), 0.0, total)
    origin = polyline_point_at_abscissa(path_points, s)
    tangent = polyline_tangent_at_abscissa(path_points, s)
    if np.linalg.norm(tangent) < EPS:
        tangent = unit(anchor_point - origin)
    if np.linalg.norm(tangent) < EPS:
        tangent = np.array([0.0, 0.0, 1.0], dtype=float)
    local_radius = path_radius_at_abscissa(path_points, path_radii, s, fallback_radius=max(float(fallback_radius), 1.0))
    candidates = plane_section_candidates(
        surface=surface,
        plane_origin=origin,
        plane_normal=tangent,
        anchor_point=anchor_point,
        source_name=source_name,
        boundary_type=boundary_type,
    )

    if candidates:
        best = candidates[0]
        profile = orient_boundary_normal(best["profile"], tangent)
        quality = float(best["quality"])
        area = float(best["area"])
        diameter_eq = float(best["diameter_eq"])
        eccentricity = float(best["eccentricity"])
        circularity = float(best["circularity"])
        contour_count = int(len(candidates))
        closure_score = float(best["closure_score"])
    else:
        circle_pts = make_circle_points(origin, tangent, local_radius, n_points=32)
        profile = BoundaryProfile(
            center=np.asarray(origin, dtype=float),
            area=float(math.pi * local_radius * local_radius),
            diameter_eq=float(2.0 * local_radius),
            normal=unit(tangent),
            rms_planarity=0.0,
            n_points=int(circle_pts.shape[0]),
            source=f"{source_name}:synthetic",
            profile_points=np.asarray(circle_pts, dtype=float),
            boundary_type=str(boundary_type),
            boundary_method="synthetic_section",
            fallback_used=True,
            synthetic=True,
            confidence=0.05,
            warnings=[f"W_SYNTHETIC_SECTION: no valid contour for {source_name} at abscissa {s:.3f}."],
        )
        quality = 0.0
        area = float(profile.area)
        diameter_eq = float(profile.diameter_eq)
        eccentricity = 1.0
        circularity = 0.0
        contour_count = 0
        closure_score = 0.0

    partner_projection = None
    partner_distance = float("inf")
    partner_abscissa = float("nan")
    if partner_prep is not None:
        proj = distance_to_prepared_polyline(origin, partner_prep)
        partner_distance = float(math.sqrt(max(float(proj["distance2"]), 0.0)))
        partner_abscissa = float(proj["abscissa"])
        partner_projection = polyline_point_at_abscissa(np.asarray(partner_prep["points"], dtype=float), partner_abscissa)
    elif partner_path is not None:
        proj = project_point_to_polyline(origin, partner_path)
        if proj is not None:
            partner_distance = float(math.sqrt(max(float(proj["distance2"]), 0.0)))
            partner_abscissa = float(proj["abscissa"])
            partner_projection = np.asarray(proj["point"], dtype=float)

    axis_stability = path_local_axis_stability(path_points, s, lookahead=max(diameter_eq, local_radius, 0.5))

    return {
        "segment_id": int(segment_id),
        "family": str(family),
        "abscissa": float(s),
        "origin": np.asarray(origin, dtype=float),
        "tangent": np.asarray(tangent, dtype=float),
        "local_radius": float(local_radius),
        "best_profile": profile,
        "candidate_count": int(contour_count),
        "quality": float(quality),
        "area": float(area),
        "diameter_eq": float(diameter_eq),
        "eccentricity": float(clamp(eccentricity, 0.0, 1.0)),
        "circularity": float(clamp(circularity, 0.0, 1.0)),
        "closure_score": float(clamp(closure_score, 0.0, 1.0)),
        "axis_stability": float(clamp(axis_stability, 0.0, 1.0)),
        "partner_distance": float(partner_distance),
        "partner_abscissa": float(partner_abscissa),
        "partner_projection": (None if partner_projection is None else np.asarray(partner_projection, dtype=float)),
        "source_name": str(source_name),
        "partner_label": str(partner_label),
    }


def sample_section_family(
    surface: "vtkPolyData",
    family: str,
    path_points: np.ndarray,
    path_radii: Optional[np.ndarray],
    sample_positions: List[float],
    anchor_points: Optional[List[np.ndarray]],
    source_prefix: str,
    boundary_type: str,
    fallback_radius: float,
    partner_path: Optional[np.ndarray] = None,
    partner_prep: Optional[Dict[str, Any]] = None,
    partner_label: str = "partner",
    segment_id: int = -1,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx, s in enumerate(sample_positions):
        anchor = np.asarray(anchor_points[idx], dtype=float) if anchor_points is not None and idx < len(anchor_points) else polyline_point_at_abscissa(path_points, s)
        sample = build_section_sample(
            surface=surface,
            family=family,
            path_points=path_points,
            path_radii=path_radii,
            abscissa=float(s),
            anchor_point=np.asarray(anchor, dtype=float),
            source_name=f"{source_prefix}:{family}:{idx}",
            boundary_type=boundary_type,
            fallback_radius=float(fallback_radius),
            partner_path=partner_path,
            partner_prep=partner_prep,
            partner_label=partner_label,
            segment_id=int(segment_id),
        )
        sample["sample_index"] = int(idx)
        samples.append(sample)
    return samples


def moving_average(values: List[float], window: int) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    win = max(1, int(window))
    out: List[float] = []
    for i in range(arr.size):
        lo = max(0, i - win)
        hi = min(arr.size, i + win + 1)
        out.append(float(np.mean(arr[lo:hi])))
    return out


def profile_scores_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    n = len(samples)
    if n == 0:
        return {
            "quality": np.zeros((0,), dtype=float),
            "area": np.zeros((0,), dtype=float),
            "diameter": np.zeros((0,), dtype=float),
            "ecc": np.zeros((0,), dtype=float),
            "circ": np.zeros((0,), dtype=float),
            "partner_dist": np.zeros((0,), dtype=float),
            "axis_stability": np.zeros((0,), dtype=float),
        }
    return {
        "quality": np.asarray([float(s["quality"]) for s in samples], dtype=float),
        "area": np.asarray([float(s["area"]) for s in samples], dtype=float),
        "diameter": np.asarray([float(s["diameter_eq"]) for s in samples], dtype=float),
        "ecc": np.asarray([float(s["eccentricity"]) for s in samples], dtype=float),
        "circ": np.asarray([float(s["circularity"]) for s in samples], dtype=float),
        "partner_dist": np.asarray([float(s["partner_distance"]) for s in samples], dtype=float),
        "axis_stability": np.asarray([float(s["axis_stability"]) for s in samples], dtype=float),
    }


def compute_daughter_transition_scores(
    samples: List[Dict[str, Any]],
    daughter_diameter: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    feats = profile_scores_from_samples(samples)
    n = int(feats["quality"].size)
    if n == 0:
        return np.zeros((0,), dtype=float), {"ref_area": 0.0, "ref_ecc": 1.0, "ref_partner_dist": 0.0}

    tail_start = max(0, n - max(3, n // 3))
    valid_tail = feats["quality"][tail_start:] > 0.05
    if np.any(valid_tail):
        ref_area = float(np.nanmedian(feats["area"][tail_start:][valid_tail]))
        ref_ecc = float(np.nanmedian(feats["ecc"][tail_start:][valid_tail]))
        ref_partner_dist = float(np.nanmedian(feats["partner_dist"][tail_start:][valid_tail]))
    else:
        ref_area = float(np.nanmedian(feats["area"])) if feats["area"].size else 0.0
        ref_ecc = float(np.nanmedian(feats["ecc"])) if feats["ecc"].size else 1.0
        ref_partner_dist = float(np.nanmedian(feats["partner_dist"])) if feats["partner_dist"].size else 0.0
    ref_area = max(ref_area, math.pi * max(0.25 * daughter_diameter, 0.25) ** 2)
    ref_partner_dist = max(ref_partner_dist, 0.50 * max(daughter_diameter, 1.0))

    scores = np.zeros((n,), dtype=float)
    tube_scores = np.zeros((n,), dtype=float)
    transition_scores = np.zeros((n,), dtype=float)
    for i in range(n):
        quality = float(feats["quality"][i])
        area_dev = abs(float(feats["area"][i]) - ref_area) / (ref_area + EPS)
        area_score = 1.0 - clamp(area_dev / 0.60, 0.0, 1.0)
        ecc_norm = clamp((float(feats["ecc"][i]) - ref_ecc) / 0.40, 0.0, 1.0)
        sep_norm = clamp(float(feats["partner_dist"][i]) / (1.25 * max(daughter_diameter, 1.0)), 0.0, 1.0)
        axis = float(feats["axis_stability"][i])
        tube_score = clamp(
            0.32 * quality
            + 0.22 * area_score
            + 0.18 * (1.0 - ecc_norm)
            + 0.15 * sep_norm
            + 0.13 * axis,
            0.0,
            1.0,
        )
        connection = clamp(
            0.32 * (1.0 - sep_norm)
            + 0.28 * ecc_norm
            + 0.18 * (1.0 - area_score)
            + 0.12 * (1.0 - axis)
            + 0.10 * (1.0 - float(feats["circ"][i])),
            0.0,
            1.0,
        )
        tube_scores[i] = tube_score
        transition_scores[i] = connection

    future_tube = np.asarray(
        [float(np.mean(tube_scores[i : min(n, i + 3)])) for i in range(n)],
        dtype=float,
    )
    prev_conn = np.asarray(
        [float(np.mean(transition_scores[max(0, i - 2) : i + 1])) for i in range(n)],
        dtype=float,
    )
    for i in range(n):
        scores[i] = clamp(
            0.28 * float(feats["quality"][i])
            + 0.32 * future_tube[i]
            + 0.24 * prev_conn[i]
            + 0.16 * min(float(transition_scores[i]), 0.90),
            0.0,
            1.0,
        )
    return scores, {
        "ref_area": float(ref_area),
        "ref_ecc": float(ref_ecc),
        "ref_partner_dist": float(ref_partner_dist),
        "tube_scores": tube_scores,
        "transition_scores": transition_scores,
    }


def compute_parent_event_scores(
    samples: List[Dict[str, Any]],
    daughter_diameter: float,
    daughter_path: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    feats = profile_scores_from_samples(samples)
    n = int(feats["quality"].size)
    if n == 0:
        return np.zeros((0,), dtype=float), {"ref_area": 0.0}

    head_stop = max(1, min(n, max(3, n // 3)))
    valid_head = feats["quality"][:head_stop] > 0.05
    if np.any(valid_head):
        ref_area = float(np.nanmedian(feats["area"][:head_stop][valid_head]))
        ref_ecc = float(np.nanmedian(feats["ecc"][:head_stop][valid_head]))
    else:
        ref_area = float(np.nanmedian(feats["area"])) if feats["area"].size else 0.0
        ref_ecc = float(np.nanmedian(feats["ecc"])) if feats["ecc"].size else 0.0
    ref_area = max(ref_area, math.pi * max(0.35 * daughter_diameter, 0.25) ** 2)

    daughter_prep = prepared_projection_data(daughter_path, None)
    raw_scores = np.zeros((n,), dtype=float)
    stable_scores = np.zeros((n,), dtype=float)
    for i, sample in enumerate(samples):
        quality = float(feats["quality"][i])
        area_inflation = clamp((float(feats["area"][i]) - ref_area) / (0.70 * ref_area + EPS), 0.0, 1.0)
        ecc_jump = clamp((float(feats["ecc"][i]) - ref_ecc) / 0.35, 0.0, 1.0)
        multiloop = clamp((int(sample["candidate_count"]) - 1) / 2.0, 0.0, 1.0)
        dist_to_daughter = float(math.sqrt(max(distance_to_prepared_polyline(np.asarray(sample["origin"], dtype=float), daughter_prep)["distance2"], 0.0)))
        daughter_prox = 1.0 - clamp(dist_to_daughter / (1.8 * max(daughter_diameter, 1.0)), 0.0, 1.0)
        raw_scores[i] = clamp(
            0.24 * quality
            + 0.24 * area_inflation
            + 0.22 * ecc_jump
            + 0.16 * multiloop
            + 0.14 * daughter_prox,
            0.0,
            1.0,
        )
        stable_scores[i] = clamp(
            0.35 * quality
            + 0.20 * (1.0 - area_inflation)
            + 0.20 * (1.0 - ecc_jump)
            + 0.15 * (1.0 - multiloop)
            + 0.10 * float(feats["axis_stability"][i]),
            0.0,
            1.0,
        )
    persistence = np.asarray(moving_average(raw_scores.tolist(), 1), dtype=float)
    event_scores = np.clip(0.65 * raw_scores + 0.35 * persistence, 0.0, 1.0)
    return event_scores, {
        "ref_area": float(ref_area),
        "stable_scores": stable_scores,
        "raw_scores": raw_scores,
    }


def nearest_parent_sample_score(parent_samples: List[Dict[str, Any]], parent_scores: np.ndarray, target_s: float) -> Tuple[float, float]:
    if not parent_samples or parent_scores.size == 0:
        return 0.0, float("nan")
    idx = int(np.argmin([abs(float(s["abscissa"]) - float(target_s)) for s in parent_samples]))
    return float(parent_scores[idx]), float(parent_samples[idx]["abscissa"])


def synthetic_transition_boundary(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    label: str,
    boundary_type: str,
    boundary_method: str,
    parent_projection_point: Optional[np.ndarray] = None,
    parent_projection_abscissa: Optional[float] = None,
    confidence: float = 0.10,
    connection_zone_score: float = 0.0,
    extra_warning: str = "",
) -> BoundaryProfile:
    circle_pts = make_circle_points(center, normal, max(float(radius), 1.0), n_points=32)
    warnings = [f"W_SYNTHETIC_BOUNDARY: used low-confidence synthetic contour for '{label}'."]
    if extra_warning:
        warnings.append(str(extra_warning))
    return BoundaryProfile(
        center=np.asarray(center, dtype=float).reshape(3),
        area=float(math.pi * max(float(radius), 1.0) ** 2),
        diameter_eq=float(2.0 * max(float(radius), 1.0)),
        normal=unit(np.asarray(normal, dtype=float).reshape(3)),
        rms_planarity=0.0,
        n_points=int(circle_pts.shape[0]),
        source=f"synthetic:{label}",
        profile_points=np.asarray(circle_pts, dtype=float),
        boundary_type=str(boundary_type),
        boundary_method=str(boundary_method),
        fallback_used=True,
        synthetic=True,
        confidence=float(clamp(confidence, 0.0, 1.0)),
        connection_zone_score=float(clamp(connection_zone_score, 0.0, 1.0)),
        parent_projection_point=(None if parent_projection_point is None else np.asarray(parent_projection_point, dtype=float).reshape(3)),
        parent_projection_abscissa=(None if parent_projection_abscissa is None else float(parent_projection_abscissa)),
        warnings=warnings,
    )


def refine_child_proximal_boundary(
    surface: "vtkPolyData",
    parent_seg: Dict[str, Any],
    child_seg: Dict[str, Any],
    resampling_step: float,
    warnings: List[str],
) -> Tuple[BoundaryProfile, Dict[str, Any], List[Dict[str, Any]], Optional[TransitionInterface]]:
    child_path = np.asarray(child_seg["path_points_oriented"], dtype=float)
    parent_path = np.asarray(parent_seg["path_points_oriented"], dtype=float)
    child_radii = child_seg.get("path_radii_oriented")
    parent_radii = parent_seg.get("path_radii_oriented")
    child_len = float(polyline_length(child_path))
    parent_len = float(polyline_length(parent_path))
    daughter_radius = characteristic_path_radius(child_radii, fallback_radius=max(float(child_seg.get("mean_radius", 1.0)), 1.0))
    daughter_diam = 2.0 * daughter_radius
    parent_anchor_radius = path_radius_at_abscissa(
        parent_path,
        parent_radii,
        max(parent_len - max(0.5 * float(resampling_step), 0.25), 0.0),
        fallback_radius=max(float(parent_seg.get("mean_radius", daughter_radius)), daughter_radius),
    )
    child_axis0 = path_local_axis_stability(child_path, 0.0, lookahead=max(daughter_diam, resampling_step, 0.5))
    parent_axis0 = path_local_axis_stability(parent_path, max(parent_len - daughter_diam, 0.0), lookahead=max(daughter_diam, resampling_step, 0.5))
    child_spacing = local_interface_spacing(daughter_radius, resampling_step, contour_quality=0.75, axis_stability=child_axis0)
    parent_spacing = local_interface_spacing(parent_anchor_radius, resampling_step, contour_quality=0.75, axis_stability=parent_axis0)
    child_window = local_interface_window(daughter_radius, child_len, child_spacing, contour_quality=0.75, axis_stability=child_axis0, role="child")
    parent_window = local_interface_window(parent_anchor_radius, parent_len, parent_spacing, contour_quality=0.75, axis_stability=parent_axis0, role="parent")
    child_start_s = min(0.35 * child_spacing, 0.10 * max(daughter_diam, 1.0))
    child_end_s = clamp(child_window, child_start_s + 0.5 * child_spacing, child_len)
    child_samples_s = sample_abscissae(child_start_s, child_end_s, child_spacing)
    parent_start_s = clamp(parent_len - parent_window, 0.0, parent_len)
    parent_end_s = clamp(parent_len - 0.35 * parent_spacing, parent_start_s, parent_len)
    parent_samples_s = sample_abscissae(parent_start_s, parent_end_s, parent_spacing)

    if child_len <= EPS or parent_len <= EPS:
        label = f"segment_{int(child_seg['segment_id'])}_proximal"
        synthetic = synthetic_transition_boundary(
            center=child_path[0] if child_path.shape[0] else np.zeros((3,), dtype=float),
            normal=polyline_tangent_at_abscissa(child_path, 0.0),
            radius=max(float(child_seg.get("mean_radius", 1.0)), 1.0),
            label=label,
            boundary_type="junction_profile",
            boundary_method="synthetic_low_confidence",
            confidence=0.05,
            extra_warning="Degenerate parent or child path during junction refinement.",
        )
        interface_normal, axis_u, axis_v = transition_partition_frame(synthetic.center, synthetic.normal, None)
        interface = TransitionInterface(
            interface_id=-1,
            child_segment_id=int(child_seg["segment_id"]),
            parent_segment_id=int(parent_seg["segment_id"]),
            contour_points=np.asarray(synthetic.profile_points, dtype=float),
            contour_centroid=np.asarray(synthetic.center, dtype=float),
            contour_normal=np.asarray(synthetic.normal, dtype=float),
            parent_projection_point=None,
            parent_projection_abscissa=None,
            partition_normal=np.asarray(interface_normal, dtype=float),
            partition_axis_u=np.asarray(axis_u, dtype=float),
            partition_axis_v=np.asarray(axis_v, dtype=float),
            confidence=float(synthetic.confidence),
            connection_zone_score=0.0,
            method_tag="synthetic_degenerate_interface",
            representative_child_abscissa=0.0,
            stable_zone_start_abscissa=0.0,
            stable_zone_end_abscissa=0.0,
            stable_zone_start_index=-1,
            stable_zone_end_index=-1,
            representative_index=-1,
            local_spacing=float(min(child_spacing, parent_spacing)),
            child_window=float(child_window),
            parent_window=float(parent_window),
            patch_radius=float(max(1.5 * daughter_diam, 3.0 * max(child_spacing, parent_spacing))),
            child_radius=float(daughter_radius),
            parent_radius=float(parent_anchor_radius),
            synthetic=True,
            low_confidence=True,
            local_partition_mode="synthetic_degenerate",
            warnings=list(synthetic.warnings),
        )
        return synthetic, {"success": False, "parent_event_abscissa": float("nan"), "parent_event_score": 0.0}, [], interface

    parent_prep = prepared_projection_data(parent_path, parent_radii)
    child_prep = prepared_projection_data(child_path, child_radii)
    child_anchor_points = [polyline_point_at_abscissa(child_path, s) for s in child_samples_s]
    parent_anchor_points = [polyline_point_at_abscissa(parent_path, s) for s in parent_samples_s]

    child_samples = sample_section_family(
        surface=surface,
        family="daughter",
        path_points=child_path,
        path_radii=child_radii,
        sample_positions=child_samples_s,
        anchor_points=child_anchor_points,
        source_prefix=f"segment_{int(child_seg['segment_id'])}_proximal_refine",
        boundary_type="junction_profile",
        fallback_radius=float(daughter_radius),
        partner_path=parent_path,
        partner_prep=parent_prep,
        partner_label="parent",
        segment_id=int(child_seg["segment_id"]),
    )
    parent_samples = sample_section_family(
        surface=surface,
        family="parent",
        path_points=parent_path,
        path_radii=parent_radii,
        sample_positions=parent_samples_s,
        anchor_points=parent_anchor_points,
        source_prefix=f"segment_{int(child_seg['segment_id'])}_parent_context",
        boundary_type="junction_context",
        fallback_radius=max(float(parent_seg.get("mean_radius", daughter_radius)), daughter_radius),
        partner_path=child_path,
        partner_prep=child_prep,
        partner_label="daughter",
        segment_id=int(child_seg["segment_id"]),
    )

    daughter_scores, daughter_info = compute_daughter_transition_scores(child_samples, daughter_diam)
    parent_scores, parent_info = compute_parent_event_scores(parent_samples, daughter_diam, daughter_path=child_path)
    parent_event_idx = int(np.argmax(parent_scores)) if parent_scores.size else -1
    parent_event_s = float(parent_samples[parent_event_idx]["abscissa"]) if parent_event_idx >= 0 else float("nan")
    parent_event_score = float(parent_scores[parent_event_idx]) if parent_event_idx >= 0 else 0.0
    parent_event_point = polyline_point_at_abscissa(parent_path, parent_event_s) if parent_event_idx >= 0 else None

    combined_scores = np.zeros((len(child_samples),), dtype=float)
    for i, sample in enumerate(child_samples):
        projected_parent_s = float(sample["partner_abscissa"]) if math.isfinite(float(sample["partner_abscissa"])) else parent_event_s
        near_parent_score, near_parent_s = nearest_parent_sample_score(parent_samples, parent_scores, projected_parent_s)
        parent_consistency = 1.0 if not math.isfinite(parent_event_s) else 1.0 - clamp(abs(projected_parent_s - parent_event_s) / (1.25 * max(daughter_diam, 1.0)), 0.0, 1.0)
        combined_scores[i] = clamp(
            0.55 * float(daughter_scores[i])
            + 0.25 * float(near_parent_score)
            + 0.20 * float(parent_consistency),
            0.0,
            1.0,
        )
        sample["daughter_transition_score"] = float(daughter_scores[i])
        sample["nearest_parent_event_score"] = float(near_parent_score)
        sample["nearest_parent_abscissa"] = float(near_parent_s)
        sample["combined_connection_score"] = float(combined_scores[i])

    if combined_scores.size:
        best_child_idx = int(np.argmax(combined_scores))
        best_score = float(combined_scores[best_child_idx])
    else:
        best_child_idx = -1
        best_score = 0.0

    debug_records: List[Dict[str, Any]] = []
    for i, sample in enumerate(child_samples):
        prof = sample["best_profile"]
        debug_records.append(
            {
                "segment_id": int(child_seg["segment_id"]),
                "parent_segment_id": int(parent_seg["segment_id"]),
                "family": "daughter",
                "sample_index": int(sample["sample_index"]),
                "sample_abscissa": float(sample["abscissa"]),
                "score": float(sample.get("combined_connection_score", 0.0)),
                "quality": float(sample["quality"]),
                "candidate_count": int(sample["candidate_count"]),
                "chosen": int(i == best_child_idx),
                "profile_points": np.asarray(prof.profile_points, dtype=float),
                "center": np.asarray(prof.center, dtype=float),
                "normal": np.asarray(prof.normal, dtype=float),
            }
        )
    for i, sample in enumerate(parent_samples):
        prof = sample["best_profile"]
        debug_records.append(
            {
                "segment_id": int(child_seg["segment_id"]),
                "parent_segment_id": int(parent_seg["segment_id"]),
                "family": "parent",
                "sample_index": int(sample["sample_index"]),
                "sample_abscissa": float(sample["abscissa"]),
                "score": float(parent_scores[i]) if i < parent_scores.size else 0.0,
                "quality": float(sample["quality"]),
                "candidate_count": int(sample["candidate_count"]),
                "chosen": int(i == parent_event_idx),
                "profile_points": np.asarray(prof.profile_points, dtype=float),
                "center": np.asarray(prof.center, dtype=float),
                "normal": np.asarray(prof.normal, dtype=float),
            }
        )

    low_confidence = best_child_idx < 0 or best_score < 0.42
    if best_child_idx < 0:
        label = f"segment_{int(child_seg['segment_id'])}_proximal"
        synthetic = synthetic_transition_boundary(
            center=child_path[0],
            normal=polyline_tangent_at_abscissa(child_path, child_start_s),
            radius=max(daughter_radius, 1.0),
            label=label,
            boundary_type="junction_profile",
            boundary_method="synthetic_low_confidence",
            parent_projection_point=parent_event_point,
            parent_projection_abscissa=(None if not math.isfinite(parent_event_s) else parent_event_s),
            confidence=0.10,
            connection_zone_score=0.0,
            extra_warning="No valid child samples for coordinated parent-daughter refinement.",
        )
        warnings.append(f"W_JUNCTION_REFINEMENT_EMPTY: segment {int(child_seg['segment_id'])} used synthetic proximal boundary because no valid samples were produced.")
        interface_normal, axis_u, axis_v = transition_partition_frame(synthetic.center, synthetic.normal, parent_event_point)
        interface = TransitionInterface(
            interface_id=-1,
            child_segment_id=int(child_seg["segment_id"]),
            parent_segment_id=int(parent_seg["segment_id"]),
            contour_points=np.asarray(synthetic.profile_points, dtype=float),
            contour_centroid=np.asarray(synthetic.center, dtype=float),
            contour_normal=np.asarray(synthetic.normal, dtype=float),
            parent_projection_point=(None if parent_event_point is None else np.asarray(parent_event_point, dtype=float)),
            parent_projection_abscissa=(None if not math.isfinite(parent_event_s) else float(parent_event_s)),
            partition_normal=np.asarray(interface_normal, dtype=float),
            partition_axis_u=np.asarray(axis_u, dtype=float),
            partition_axis_v=np.asarray(axis_v, dtype=float),
            confidence=float(synthetic.confidence),
            connection_zone_score=0.0,
            method_tag="synthetic_low_confidence_interface",
            representative_child_abscissa=float(child_start_s),
            stable_zone_start_abscissa=float(child_start_s),
            stable_zone_end_abscissa=float(child_start_s),
            stable_zone_start_index=-1,
            stable_zone_end_index=-1,
            representative_index=-1,
            local_spacing=float(min(child_spacing, parent_spacing)),
            child_window=float(child_window),
            parent_window=float(parent_window),
            patch_radius=float(max(1.5 * daughter_diam, 3.0 * max(child_spacing, parent_spacing))),
            child_radius=float(daughter_radius),
            parent_radius=float(parent_anchor_radius),
            synthetic=True,
            low_confidence=True,
            local_partition_mode="synthetic_low_confidence",
            warnings=list(synthetic.warnings),
        )
        return synthetic, {
            "success": False,
            "parent_event_abscissa": float(parent_event_s),
            "parent_event_score": float(parent_event_score),
            "parent_projection_point": (None if parent_event_point is None else np.asarray(parent_event_point, dtype=float)),
            "child_best_score": float(best_score),
            "daughter_ref_area": float(daughter_info["ref_area"]),
        }, debug_records, interface

    zone = detect_stable_transition_zone(child_samples, combined_scores, local_scale=max(daughter_diam, child_spacing))
    representative_idx = int(zone["representative_index"]) if int(zone["representative_index"]) >= 0 else int(best_child_idx)
    if representative_idx < 0 and combined_scores.size:
        representative_idx = int(np.argmax(combined_scores))

    representative_sample = child_samples[representative_idx] if representative_idx >= 0 else None
    representative_quality = float(representative_sample["quality"]) if representative_sample is not None else 0.0
    representative_axis = float(representative_sample["axis_stability"]) if representative_sample is not None else 0.0
    representative_radius = (
        path_radius_at_abscissa(child_path, child_radii, float(representative_sample["abscissa"]), daughter_radius)
        if representative_sample is not None
        else daughter_radius
    )
    dense_child_spacing = local_interface_spacing(
        representative_radius,
        resampling_step=resampling_step,
        connection_evidence=float(np.max(combined_scores)) if combined_scores.size else 0.0,
        contour_quality=representative_quality,
        axis_stability=representative_axis,
    )
    dense_parent_spacing = local_interface_spacing(
        parent_anchor_radius,
        resampling_step=resampling_step,
        connection_evidence=float(parent_event_score),
        contour_quality=representative_quality,
        axis_stability=representative_axis,
    )

    if (not zone["success"]) or float(zone["zone_score"]) < 0.50 or dense_child_spacing < 0.90 * child_spacing:
        dense_center = float(representative_sample["abscissa"]) if representative_sample is not None else float(child_samples[min(best_child_idx, len(child_samples) - 1)]["abscissa"])
        child_dense_positions = escalate_transition_sampling_window(dense_center, child_len, dense_child_spacing, child_window)
        parent_dense_center = parent_event_s if math.isfinite(parent_event_s) else max(parent_len - parent_window, 0.0)
        parent_dense_positions = escalate_transition_sampling_window(parent_dense_center, parent_len, dense_parent_spacing, parent_window)

        child_positions = sorted(set(round(v, 6) for v in child_samples_s + child_dense_positions))
        parent_positions = sorted(set(round(v, 6) for v in parent_samples_s + parent_dense_positions))
        child_samples_s = [float(v) for v in child_positions]
        parent_samples_s = [float(v) for v in parent_positions]
        child_anchor_points = [polyline_point_at_abscissa(child_path, s) for s in child_samples_s]
        parent_anchor_points = [polyline_point_at_abscissa(parent_path, s) for s in parent_samples_s]
        child_samples = sample_section_family(
            surface=surface,
            family="daughter",
            path_points=child_path,
            path_radii=child_radii,
            sample_positions=child_samples_s,
            anchor_points=child_anchor_points,
            source_prefix=f"segment_{int(child_seg['segment_id'])}_proximal_refine_dense",
            boundary_type="junction_profile",
            fallback_radius=float(daughter_radius),
            partner_path=parent_path,
            partner_prep=parent_prep,
            partner_label="parent",
            segment_id=int(child_seg["segment_id"]),
        )
        parent_samples = sample_section_family(
            surface=surface,
            family="parent",
            path_points=parent_path,
            path_radii=parent_radii,
            sample_positions=parent_samples_s,
            anchor_points=parent_anchor_points,
            source_prefix=f"segment_{int(child_seg['segment_id'])}_parent_context_dense",
            boundary_type="junction_context",
            fallback_radius=max(float(parent_seg.get("mean_radius", daughter_radius)), daughter_radius),
            partner_path=child_path,
            partner_prep=child_prep,
            partner_label="daughter",
            segment_id=int(child_seg["segment_id"]),
        )
        daughter_scores, daughter_info = compute_daughter_transition_scores(child_samples, daughter_diam)
        parent_scores, parent_info = compute_parent_event_scores(parent_samples, daughter_diam, daughter_path=child_path)
        parent_event_idx = int(np.argmax(parent_scores)) if parent_scores.size else -1
        parent_event_s = float(parent_samples[parent_event_idx]["abscissa"]) if parent_event_idx >= 0 else float("nan")
        parent_event_score = float(parent_scores[parent_event_idx]) if parent_event_idx >= 0 else 0.0
        parent_event_point = polyline_point_at_abscissa(parent_path, parent_event_s) if parent_event_idx >= 0 else None

        combined_scores = np.zeros((len(child_samples),), dtype=float)
        debug_records = []
        for i, sample in enumerate(child_samples):
            projected_parent_s = float(sample["partner_abscissa"]) if math.isfinite(float(sample["partner_abscissa"])) else parent_event_s
            near_parent_score, near_parent_s = nearest_parent_sample_score(parent_samples, parent_scores, projected_parent_s)
            parent_consistency = 1.0 if not math.isfinite(parent_event_s) else 1.0 - clamp(abs(projected_parent_s - parent_event_s) / (1.25 * max(daughter_diam, 1.0)), 0.0, 1.0)
            combined_scores[i] = clamp(
                0.55 * float(daughter_scores[i])
                + 0.25 * float(near_parent_score)
                + 0.20 * float(parent_consistency),
                0.0,
                1.0,
            )
            sample["daughter_transition_score"] = float(daughter_scores[i])
            sample["nearest_parent_event_score"] = float(near_parent_score)
            sample["nearest_parent_abscissa"] = float(near_parent_s)
            sample["combined_connection_score"] = float(combined_scores[i])
        best_child_idx = int(np.argmax(combined_scores)) if combined_scores.size else -1
        best_score = float(combined_scores[best_child_idx]) if best_child_idx >= 0 else 0.0
        zone = detect_stable_transition_zone(child_samples, combined_scores, local_scale=max(daughter_diam, dense_child_spacing))

        for i, sample in enumerate(child_samples):
            prof = sample["best_profile"]
            debug_records.append(
                {
                    "segment_id": int(child_seg["segment_id"]),
                    "parent_segment_id": int(parent_seg["segment_id"]),
                    "family": "daughter",
                    "sample_index": int(sample["sample_index"]),
                    "sample_abscissa": float(sample["abscissa"]),
                    "score": float(sample.get("combined_connection_score", 0.0)),
                    "quality": float(sample["quality"]),
                    "candidate_count": int(sample["candidate_count"]),
                    "chosen": int(i == int(zone["representative_index"]) if int(zone["representative_index"]) >= 0 else i == best_child_idx),
                    "profile_points": np.asarray(prof.profile_points, dtype=float),
                    "center": np.asarray(prof.center, dtype=float),
                    "normal": np.asarray(prof.normal, dtype=float),
                    "stable_zone_member": int(int(zone["start_index"]) <= i <= int(zone["end_index"])),
                }
            )
        for i, sample in enumerate(parent_samples):
            prof = sample["best_profile"]
            debug_records.append(
                {
                    "segment_id": int(child_seg["segment_id"]),
                    "parent_segment_id": int(parent_seg["segment_id"]),
                    "family": "parent",
                    "sample_index": int(sample["sample_index"]),
                    "sample_abscissa": float(sample["abscissa"]),
                    "score": float(parent_scores[i]) if i < parent_scores.size else 0.0,
                    "quality": float(sample["quality"]),
                    "candidate_count": int(sample["candidate_count"]),
                    "chosen": int(i == parent_event_idx),
                    "profile_points": np.asarray(prof.profile_points, dtype=float),
                    "center": np.asarray(prof.center, dtype=float),
                    "normal": np.asarray(prof.normal, dtype=float),
                }
            )
        child_spacing = dense_child_spacing
        parent_spacing = dense_parent_spacing

    representative_idx = int(zone["representative_index"]) if int(zone["representative_index"]) >= 0 else int(best_child_idx)
    if representative_idx < 0:
        representative_idx = int(best_child_idx)
    for rec in debug_records:
        if str(rec.get("family", "")) == "daughter":
            sample_idx = int(rec.get("sample_index", -1))
            rec["stable_zone_member"] = int(int(zone["start_index"]) <= sample_idx <= int(zone["end_index"]))
            rec["chosen"] = int(sample_idx == representative_idx)
    low_confidence = representative_idx < 0 or (not zone["success"]) or float(zone["zone_score"]) < 0.42
    best_score = float(combined_scores[representative_idx]) if representative_idx >= 0 and combined_scores.size else float(best_score)
    best_sample = child_samples[representative_idx]
    best_profile = orient_boundary_normal(best_sample["best_profile"], best_sample["tangent"])
    projected_parent_s = float(best_sample["partner_abscissa"]) if math.isfinite(float(best_sample["partner_abscissa"])) else parent_event_s
    if not math.isfinite(projected_parent_s):
        projected_parent_s = float(parent_len)
    parent_projection_point = polyline_point_at_abscissa(parent_path, projected_parent_s)
    chosen_profile = boundary_profile_from_contour(
        contour_points=np.asarray(best_profile.profile_points, dtype=float),
        normal_hint=np.asarray(best_sample["tangent"], dtype=float),
        source=f"segment_{int(child_seg['segment_id'])}_proximal_refined",
        boundary_type="junction_profile",
        boundary_method="refined_parent_daughter_sections",
        confidence=float(max(best_score, float(zone["zone_score"]))),
        connection_zone_score=float(best_score),
        fallback_used=bool(low_confidence or best_profile.fallback_used),
        synthetic=bool(low_confidence and best_profile.synthetic),
        parent_projection_point=np.asarray(parent_projection_point, dtype=float),
        parent_projection_abscissa=float(projected_parent_s),
        warnings=list(best_profile.warnings),
    )
    chosen_profile = orient_boundary_normal(chosen_profile, best_sample["tangent"])
    chosen_profile.confidence = float(clamp(max(best_score, float(zone["zone_score"])), 0.0, 1.0))
    chosen_profile.connection_zone_score = float(clamp(best_score, 0.0, 1.0))
    if low_confidence:
        chosen_profile.fallback_used = True
        chosen_profile.synthetic = bool(best_profile.synthetic or chosen_profile.synthetic)
        chosen_profile.warnings.append(
            f"W_JUNCTION_REFINEMENT_LOWCONF: segment {int(child_seg['segment_id'])} proximal refinement stayed low-confidence (score={best_score:.3f}, stable_zone={bool(zone['success'])})."
        )

    child_local_radius = path_radius_at_abscissa(child_path, child_radii, float(best_sample["abscissa"]), daughter_radius)
    parent_local_radius = path_radius_at_abscissa(parent_path, parent_radii, float(projected_parent_s), parent_anchor_radius)
    partition_normal, axis_u, axis_v = transition_partition_frame(chosen_profile.center, chosen_profile.normal, parent_projection_point)
    interface = TransitionInterface(
        interface_id=-1,
        child_segment_id=int(child_seg["segment_id"]),
        parent_segment_id=int(parent_seg["segment_id"]),
        contour_points=np.asarray(chosen_profile.profile_points, dtype=float),
        contour_centroid=np.asarray(chosen_profile.center, dtype=float),
        contour_normal=np.asarray(chosen_profile.normal, dtype=float),
        parent_projection_point=np.asarray(parent_projection_point, dtype=float),
        parent_projection_abscissa=float(projected_parent_s),
        partition_normal=np.asarray(partition_normal, dtype=float),
        partition_axis_u=np.asarray(axis_u, dtype=float),
        partition_axis_v=np.asarray(axis_v, dtype=float),
        confidence=float(chosen_profile.confidence),
        connection_zone_score=float(best_score),
        method_tag="refined_parent_daughter_interface",
        representative_child_abscissa=float(best_sample["abscissa"]),
        stable_zone_start_abscissa=(float(child_samples[int(zone["start_index"])]["abscissa"]) if int(zone["start_index"]) >= 0 else float(best_sample["abscissa"])),
        stable_zone_end_abscissa=(float(child_samples[int(zone["end_index"])]["abscissa"]) if int(zone["end_index"]) >= 0 else float(best_sample["abscissa"])),
        stable_zone_start_index=int(zone["start_index"]),
        stable_zone_end_index=int(zone["end_index"]),
        representative_index=int(representative_idx),
        local_spacing=float(min(child_spacing, parent_spacing)),
        child_window=float(child_window),
        parent_window=float(parent_window),
        patch_radius=float(max(1.35 * chosen_profile.diameter_eq, 2.25 * max(child_local_radius, parent_local_radius, 1.0), 3.0 * max(child_spacing, parent_spacing))),
        child_radius=float(child_local_radius),
        parent_radius=float(parent_local_radius),
        contour_quality=float(best_sample["quality"]),
        axis_stability=float(best_sample["axis_stability"]),
        synthetic=bool(chosen_profile.synthetic),
        low_confidence=bool(low_confidence),
        local_partition_mode="pending",
        warnings=list(chosen_profile.warnings),
    )
    return chosen_profile, {
        "success": bool(zone["success"]),
        "parent_event_abscissa": float(parent_event_s),
        "parent_event_score": float(parent_event_score),
        "parent_projection_point": np.asarray(parent_projection_point, dtype=float),
        "child_best_score": float(best_score),
        "child_best_abscissa": float(best_sample["abscissa"]),
        "daughter_ref_area": float(daughter_info["ref_area"]),
        "parent_ref_area": float(parent_info["ref_area"]),
        "child_spacing": float(child_spacing),
        "parent_spacing": float(parent_spacing),
        "stable_zone_score": float(zone["zone_score"]),
        "stable_zone_start_abscissa": interface.stable_zone_start_abscissa,
        "stable_zone_end_abscissa": interface.stable_zone_end_abscissa,
    }, debug_records, interface


def build_parent_distal_boundary_from_children(
    surface: "vtkPolyData",
    parent_seg: Dict[str, Any],
    child_interfaces: List[TransitionInterface],
    resampling_step: float,
    warnings: List[str],
) -> Tuple[BoundaryProfile, List[Dict[str, Any]]]:
    parent_path = np.asarray(parent_seg["path_points_oriented"], dtype=float)
    parent_radii = parent_seg.get("path_radii_oriented")
    parent_len = float(polyline_length(parent_path))
    parent_radius = characteristic_path_radius(parent_radii, fallback_radius=max(float(parent_seg.get("mean_radius", 1.0)), 1.0))
    if parent_len <= EPS:
        synthetic = synthetic_transition_boundary(
            center=parent_path[-1] if parent_path.shape[0] else np.zeros((3,), dtype=float),
            normal=-polyline_tangent_at_abscissa(parent_path, 0.0),
            radius=max(parent_radius, 1.0),
            label=f"segment_{int(parent_seg['segment_id'])}_distal",
            boundary_type="junction_transition_parent",
            boundary_method="synthetic_parent_low_confidence",
            confidence=0.05,
            extra_warning="Degenerate parent path during distal boundary refinement.",
        )
        return synthetic, []

    usable = [
        iface
        for iface in child_interfaces
        if int(iface.parent_segment_id) == int(parent_seg["segment_id"]) and iface.contour_points.shape[0] >= 3
    ]
    if not usable:
        synthetic = synthetic_transition_boundary(
            center=polyline_point_at_abscissa(parent_path, max(parent_len - 0.5 * max(parent_radius, 1.0), 0.0)),
            normal=-polyline_tangent_at_abscissa(parent_path, max(parent_len - 0.5 * max(parent_radius, 1.0), 0.0)),
            radius=max(parent_radius, 1.0),
            label=f"segment_{int(parent_seg['segment_id'])}_distal",
            boundary_type="junction_transition_parent",
            boundary_method="synthetic_parent_low_confidence",
            confidence=0.10,
            extra_warning="No child-coupled transition interfaces were available for parent summary boundary.",
        )
        warnings.append(f"W_PARENT_DISTAL_SYNTHETIC: segment {int(parent_seg['segment_id'])} used synthetic distal boundary because no child-coupled interface summary existed.")
        return synthetic, []

    usable.sort(
        key=lambda iface: (
            float(iface.confidence),
            float(iface.connection_zone_score),
            0.0 if iface.parent_projection_abscissa is None else -float(iface.parent_projection_abscissa),
        ),
        reverse=True,
    )
    chosen_interface = usable[0]
    parent_projection_abscissa = (
        float(chosen_interface.parent_projection_abscissa)
        if chosen_interface.parent_projection_abscissa is not None
        else max(parent_len - max(parent_radius, 1.0), 0.0)
    )
    parent_projection_point = (
        np.asarray(chosen_interface.parent_projection_point, dtype=float)
        if chosen_interface.parent_projection_point is not None
        else polyline_point_at_abscissa(parent_path, parent_projection_abscissa)
    )
    parent_inward = -polyline_tangent_at_abscissa(parent_path, parent_projection_abscissa)
    chosen_profile = boundary_profile_from_contour(
        contour_points=np.asarray(chosen_interface.contour_points, dtype=float),
        normal_hint=np.asarray(parent_inward, dtype=float),
        source=f"segment_{int(parent_seg['segment_id'])}_distal_refined",
        boundary_type="junction_transition_parent",
        boundary_method="child_coupled_parent_transition_summary",
        confidence=float(chosen_interface.confidence),
        connection_zone_score=float(chosen_interface.connection_zone_score),
        fallback_used=bool(chosen_interface.low_confidence or chosen_interface.synthetic),
        synthetic=bool(chosen_interface.synthetic),
        parent_projection_point=np.asarray(parent_projection_point, dtype=float),
        parent_projection_abscissa=float(parent_projection_abscissa),
        warnings=list(chosen_interface.warnings),
    )
    chosen_profile = orient_boundary_normal(chosen_profile, np.asarray(parent_inward, dtype=float))
    if len(usable) > 1:
        chosen_profile.warnings.append(
            f"W_PARENT_DISTAL_SUMMARY: segment {int(parent_seg['segment_id'])} has {len(usable)} child-coupled interfaces; distal boundary is a representative summary only and final partition uses per-child interfaces."
        )
    if float(chosen_interface.confidence) < 0.40:
        chosen_profile.fallback_used = True
        chosen_profile.warnings.append(
            f"W_PARENT_DISTAL_LOWCONF: segment {int(parent_seg['segment_id'])} distal summary stayed low-confidence (score={float(chosen_interface.confidence):.3f})."
        )

    debug_records: List[Dict[str, Any]] = []
    for i, iface in enumerate(usable):
        debug_records.append(
            {
                "segment_id": int(parent_seg["segment_id"]),
                "parent_segment_id": int(parent_seg["segment_id"]),
                "family": "parent_child_coupled",
                "sample_index": int(i),
                "sample_abscissa": float(iface.parent_projection_abscissa) if iface.parent_projection_abscissa is not None else 0.0,
                "score": float(iface.confidence),
                "quality": float(iface.contour_quality),
                "candidate_count": 1,
                "chosen": int(i == 0),
                "profile_points": np.asarray(iface.contour_points, dtype=float),
                "center": np.asarray(iface.contour_centroid, dtype=float),
                "normal": np.asarray(iface.contour_normal, dtype=float),
            }
        )
    return chosen_profile, debug_records


def build_boundary_debug_polydata(records: List[Dict[str, Any]]) -> "vtkPolyData":
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    segment_arr = vtk.vtkIntArray()
    segment_arr.SetName("SegmentId")
    side_arr = vtk.vtkIntArray()
    side_arr.SetName("BoundarySide")
    type_arr = vtk.vtkStringArray()
    type_arr.SetName("BoundaryType")
    method_arr = vtk.vtkStringArray()
    method_arr.SetName("BoundaryMethod")
    fallback_arr = vtk.vtkIntArray()
    fallback_arr.SetName("FallbackUsed")
    conf_arr = vtk.vtkDoubleArray()
    conf_arr.SetName("BoundaryConfidence")

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
        method_arr.InsertNextValue(str(rec.get("boundary_method", "")))
        fallback_arr.InsertNextValue(int(bool(rec.get("fallback_used", False))))
        conf_arr.InsertNextValue(float(rec.get("confidence", 0.0)))

    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)
    out.GetCellData().AddArray(segment_arr)
    out.GetCellData().AddArray(side_arr)
    out.GetCellData().AddArray(type_arr)
    out.GetCellData().AddArray(method_arr)
    out.GetCellData().AddArray(fallback_arr)
    out.GetCellData().AddArray(conf_arr)
    return out


def build_section_debug_polydata(records: List[Dict[str, Any]]) -> "vtkPolyData":
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    seg_arr = vtk.vtkIntArray()
    seg_arr.SetName("SegmentId")
    parent_arr = vtk.vtkIntArray()
    parent_arr.SetName("ParentSegmentId")
    family_arr = vtk.vtkStringArray()
    family_arr.SetName("SectionFamily")
    index_arr = vtk.vtkIntArray()
    index_arr.SetName("SampleIndex")
    abs_arr = vtk.vtkDoubleArray()
    abs_arr.SetName("SampleAbscissa")
    score_arr = vtk.vtkDoubleArray()
    score_arr.SetName("SectionScore")
    qual_arr = vtk.vtkDoubleArray()
    qual_arr.SetName("SectionQuality")
    count_arr = vtk.vtkIntArray()
    count_arr.SetName("CandidateCount")
    chosen_arr = vtk.vtkIntArray()
    chosen_arr.SetName("Chosen")
    stable_arr = vtk.vtkIntArray()
    stable_arr.SetName("StableZoneMember")

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
        seg_arr.InsertNextValue(int(rec["segment_id"]))
        parent_arr.InsertNextValue(int(rec.get("parent_segment_id", -1)))
        family_arr.InsertNextValue(str(rec.get("family", "")))
        index_arr.InsertNextValue(int(rec.get("sample_index", -1)))
        abs_arr.InsertNextValue(float(rec.get("sample_abscissa", 0.0)))
        score_arr.InsertNextValue(float(rec.get("score", 0.0)))
        qual_arr.InsertNextValue(float(rec.get("quality", 0.0)))
        count_arr.InsertNextValue(int(rec.get("candidate_count", 0)))
        chosen_arr.InsertNextValue(int(bool(rec.get("chosen", False))))
        stable_arr.InsertNextValue(int(bool(rec.get("stable_zone_member", False))))

    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)
    out.GetCellData().AddArray(seg_arr)
    out.GetCellData().AddArray(parent_arr)
    out.GetCellData().AddArray(family_arr)
    out.GetCellData().AddArray(index_arr)
    out.GetCellData().AddArray(abs_arr)
    out.GetCellData().AddArray(score_arr)
    out.GetCellData().AddArray(qual_arr)
    out.GetCellData().AddArray(count_arr)
    out.GetCellData().AddArray(chosen_arr)
    out.GetCellData().AddArray(stable_arr)
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


def build_cell_center_locator(centers: np.ndarray) -> "vtkStaticPointLocator":
    points = vtk.vtkPoints()
    for p in np.asarray(centers, dtype=float):
        points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(pd)
    locator.BuildLocator()
    return locator


def build_surface_cell_adjacency(surface: "vtkPolyData") -> List[List[int]]:
    n_cells = int(surface.GetNumberOfCells())
    neighbors: List[Set[int]] = [set() for _ in range(n_cells)]
    edge_to_cells: Dict[Tuple[int, int], List[int]] = {}
    id_list = vtk.vtkIdList()
    for ci in range(n_cells):
        surface.GetCellPoints(int(ci), id_list)
        ids = [int(id_list.GetId(k)) for k in range(id_list.GetNumberOfIds())]
        if len(ids) < 2:
            continue
        for k in range(len(ids)):
            a = int(ids[k])
            b = int(ids[(k + 1) % len(ids)])
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edge_to_cells.setdefault(edge, []).append(int(ci))
    for cell_ids in edge_to_cells.values():
        uniq = sorted(set(int(v) for v in cell_ids))
        if len(uniq) < 2:
            continue
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                neighbors[uniq[i]].add(int(uniq[j]))
                neighbors[uniq[j]].add(int(uniq[i]))
    return [sorted(int(v) for v in row) for row in neighbors]


def flood_reachable_cells(
    seed_ids: Set[int],
    allowed_ids: Set[int],
    adjacency: List[List[int]],
) -> Set[int]:
    if not seed_ids or not allowed_ids:
        return set()
    visited: Set[int] = set(int(v) for v in seed_ids if int(v) in allowed_ids)
    queue: List[int] = list(sorted(visited))
    head = 0
    while head < len(queue):
        cur = int(queue[head])
        head += 1
        for nei in adjacency[cur]:
            nei = int(nei)
            if nei not in allowed_ids or nei in visited:
                continue
            visited.add(nei)
            queue.append(nei)
    return visited


def segment_relative_distance(
    point: np.ndarray,
    segment: Dict[str, Any],
) -> Tuple[float, float, float]:
    proj = distance_to_prepared_polyline(np.asarray(point, dtype=float).reshape(3), segment["projection"])
    radius = max(float(proj["radius"]), float(segment.get("mean_radius", 1.0)), 1.0)
    rel = float(math.sqrt(max(float(proj["distance2"]), 0.0)) / radius)
    return rel, float(proj["abscissa"]), float(radius)


def partition_transition_interface_on_surface(
    centers: np.ndarray,
    adjacency: List[List[int]],
    interface: TransitionInterface,
    segment_lookup: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    child_seg = segment_lookup.get(int(interface.child_segment_id))
    parent_seg = segment_lookup.get(int(interface.parent_segment_id))
    if child_seg is None or parent_seg is None:
        interface.local_partition_success = False
        interface.local_partition_mode = "missing_segments"
        return {"success": False, "blocked_edges": set(), "patch_cells": []}

    child_path = np.asarray(child_seg["path_points_oriented"], dtype=float)
    parent_path = np.asarray(parent_seg["path_points_oriented"], dtype=float)
    child_radii = child_seg.get("path_radii_oriented")
    parent_radii = parent_seg.get("path_radii_oriented")
    child_s0 = 0.0 if interface.representative_child_abscissa is None else float(interface.representative_child_abscissa)
    parent_s0 = float(polyline_length(parent_path)) if interface.parent_projection_abscissa is None else float(interface.parent_projection_abscissa)
    patch_radius = max(float(interface.patch_radius), 2.0 * max(float(interface.child_radius), float(interface.parent_radius), 1.0))
    spacing = max(float(interface.local_spacing), 0.12)
    confidence_scale = 0.75 if interface.low_confidence or interface.synthetic else 1.0
    effective_patch_radius = patch_radius * confidence_scale
    barrier_band = max(0.50 * spacing, 0.22 * effective_patch_radius, 0.24 * max(float(interface.child_radius), float(interface.parent_radius), 1.0))
    patch_cells: List[int] = []
    metrics: Dict[int, Dict[str, float]] = {}

    for ci, center in enumerate(np.asarray(centers, dtype=float)):
        child_proj = distance_to_prepared_polyline(center, child_seg["projection"])
        parent_proj = distance_to_prepared_polyline(center, parent_seg["projection"])
        child_radius = path_radius_at_abscissa(child_path, child_radii, float(child_proj["abscissa"]), float(interface.child_radius or child_seg.get("mean_radius", 1.0)))
        parent_radius = path_radius_at_abscissa(parent_path, parent_radii, float(parent_proj["abscissa"]), float(interface.parent_radius or parent_seg.get("mean_radius", 1.0)))
        child_dist = math.sqrt(max(float(child_proj["distance2"]), 0.0))
        parent_dist = math.sqrt(max(float(parent_proj["distance2"]), 0.0))
        signed = point_plane_signed_distance(center, interface.contour_centroid, interface.partition_normal)
        near_child = (
            float(child_proj["abscissa"]) >= child_s0 - 0.75 * spacing
            and float(child_proj["abscissa"]) <= child_s0 + float(interface.child_window) + spacing
            and child_dist <= max(3.25 * child_radius, effective_patch_radius)
        )
        near_parent = (
            float(parent_proj["abscissa"]) >= parent_s0 - float(interface.parent_window) - spacing
            and float(parent_proj["abscissa"]) <= parent_s0 + 0.75 * spacing
            and parent_dist <= max(3.25 * parent_radius, effective_patch_radius)
        )
        near_contour = float(np.linalg.norm(center - np.asarray(interface.contour_centroid, dtype=float))) <= 1.20 * effective_patch_radius
        if not (near_child or near_parent or near_contour):
            continue
        child_rel = child_dist / max(child_radius, 1.0)
        parent_rel = parent_dist / max(parent_radius, 1.0)
        contour_dist = point_to_polyline_distance(center, interface.contour_points, closed=True)
        patch_cells.append(int(ci))
        metrics[int(ci)] = {
            "signed": float(signed),
            "child_rel": float(child_rel),
            "parent_rel": float(parent_rel),
            "child_abscissa": float(child_proj["abscissa"]),
            "parent_abscissa": float(parent_proj["abscissa"]),
            "contour_dist": float(contour_dist),
            "child_radius": float(child_radius),
            "parent_radius": float(parent_radius),
        }

    patch_set = set(int(v) for v in patch_cells)
    if not patch_set:
        interface.local_partition_success = False
        interface.local_partition_mode = "empty_patch"
        return {"success": False, "blocked_edges": set(), "patch_cells": []}

    barrier_cells: Set[int] = set()
    patch_center_ids = sorted(patch_set)
    patch_centers = np.asarray([centers[idx] for idx in patch_center_ids], dtype=float)
    for contour_point in np.asarray(interface.contour_points, dtype=float):
        if patch_centers.shape[0] == 0:
            break
        d2 = np.sum((patch_centers - contour_point.reshape(1, 3)) ** 2, axis=1)
        nearest_patch_idx = int(np.argmin(d2))
        barrier_cells.add(int(patch_center_ids[nearest_patch_idx]))
    for ci in patch_set:
        m = metrics[int(ci)]
        if float(m["contour_dist"]) <= barrier_band and abs(float(m["signed"])) <= max(1.20 * barrier_band, 0.35 * effective_patch_radius):
            barrier_cells.add(int(ci))

    child_seed_offset = max(0.35 * float(interface.child_radius or child_seg.get("mean_radius", 1.0)), 0.80 * barrier_band)
    parent_seed_offset = max(0.35 * float(interface.parent_radius or parent_seg.get("mean_radius", 1.0)), 0.80 * barrier_band)
    child_seeds: Set[int] = set()
    parent_seeds: Set[int] = set()
    child_allowed: Set[int] = set()
    parent_allowed: Set[int] = set()
    child_costs: Dict[int, float] = {}
    parent_costs: Dict[int, float] = {}

    for ci in patch_set:
        if ci in barrier_cells:
            continue
        m = metrics[int(ci)]
        child_range_pen = abs(float(m["child_abscissa"]) - child_s0) / max(float(interface.child_window) + spacing, spacing)
        parent_range_pen = abs(float(m["parent_abscissa"]) - parent_s0) / max(float(interface.parent_window) + spacing, spacing)
        child_cost = float(m["child_rel"]) + 0.70 * clamp(-float(m["signed"]) / max(float(m["child_radius"]), 1.0), 0.0, 2.0) + 0.25 * child_range_pen
        parent_cost = float(m["parent_rel"]) + 0.70 * clamp(float(m["signed"]) / max(float(m["parent_radius"]), 1.0), 0.0, 2.0) + 0.25 * parent_range_pen
        child_costs[int(ci)] = float(child_cost)
        parent_costs[int(ci)] = float(parent_cost)
        child_allowed_here = (
            float(m["child_abscissa"]) >= child_s0 - 0.75 * spacing
            and float(m["child_abscissa"]) <= child_s0 + float(interface.child_window) + spacing
            and child_cost <= parent_cost + 0.60
            and float(m["signed"]) >= -0.55 * max(float(m["child_radius"]), 1.0)
        )
        parent_allowed_here = (
            float(m["parent_abscissa"]) >= parent_s0 - float(interface.parent_window) - spacing
            and float(m["parent_abscissa"]) <= parent_s0 + 0.75 * spacing
            and parent_cost <= child_cost + 0.60
            and float(m["signed"]) <= 0.55 * max(float(m["parent_radius"]), 1.0)
        )
        if child_allowed_here:
            child_allowed.add(int(ci))
        if parent_allowed_here:
            parent_allowed.add(int(ci))
        if child_allowed_here and float(m["signed"]) >= child_seed_offset and float(m["child_rel"]) <= min(1.85, float(m["parent_rel"]) + 0.20):
            child_seeds.add(int(ci))
        if parent_allowed_here and float(m["signed"]) <= -parent_seed_offset and float(m["parent_rel"]) <= min(1.85, float(m["child_rel"]) + 0.20):
            parent_seeds.add(int(ci))

    if not child_seeds and child_costs:
        candidate = min((ci for ci in child_allowed), key=lambda ci: (child_costs[ci], parent_costs.get(ci, float("inf"))), default=None)
        if candidate is not None:
            child_seeds.add(int(candidate))
    if not parent_seeds and parent_costs:
        candidate = min((ci for ci in parent_allowed), key=lambda ci: (parent_costs[ci], child_costs.get(ci, float("inf"))), default=None)
        if candidate is not None:
            parent_seeds.add(int(candidate))

    if (interface.low_confidence or interface.synthetic) and (len(child_seeds) < 1 or len(parent_seeds) < 1):
        interface.local_partition_success = False
        interface.local_partition_mode = "conservative_skip"
        interface.local_patch_cell_ids = sorted(int(v) for v in patch_set)
        interface.local_barrier_cell_ids = sorted(int(v) for v in barrier_cells)
        return {"success": False, "blocked_edges": set(), "patch_cells": sorted(int(v) for v in patch_set)}

    child_reached = flood_reachable_cells(child_seeds, child_allowed, adjacency)
    parent_reached = flood_reachable_cells(parent_seeds, parent_allowed, adjacency)
    local_labels: Dict[int, int] = {}
    child_cells: Set[int] = set()
    parent_cells: Set[int] = set()

    for ci in sorted(patch_set):
        if ci in child_reached and ci not in parent_reached:
            local_labels[int(ci)] = int(interface.child_segment_id)
            child_cells.add(int(ci))
        elif ci in parent_reached and ci not in child_reached:
            local_labels[int(ci)] = int(interface.parent_segment_id)
            parent_cells.add(int(ci))
        elif ci in child_reached and ci in parent_reached:
            if child_costs.get(int(ci), float("inf")) <= parent_costs.get(int(ci), float("inf")):
                local_labels[int(ci)] = int(interface.child_segment_id)
                child_cells.add(int(ci))
            else:
                local_labels[int(ci)] = int(interface.parent_segment_id)
                parent_cells.add(int(ci))
        elif ci in barrier_cells:
            if float(metrics[int(ci)]["signed"]) >= 0.0:
                local_labels[int(ci)] = int(interface.child_segment_id)
                child_cells.add(int(ci))
            else:
                local_labels[int(ci)] = int(interface.parent_segment_id)
                parent_cells.add(int(ci))
        else:
            child_cost = child_costs.get(int(ci), float("inf"))
            parent_cost = parent_costs.get(int(ci), float("inf"))
            if child_cost <= parent_cost:
                local_labels[int(ci)] = int(interface.child_segment_id)
                child_cells.add(int(ci))
            elif parent_cost < float("inf"):
                local_labels[int(ci)] = int(interface.parent_segment_id)
                parent_cells.add(int(ci))

    blocked_edges: Set[Tuple[int, int]] = set()
    for ci in patch_set:
        for nei in adjacency[int(ci)]:
            nei = int(nei)
            if nei not in patch_set:
                continue
            li = local_labels.get(int(ci), -1)
            lj = local_labels.get(int(nei), -1)
            if li >= 0 and lj >= 0 and li != lj:
                a, b = (int(ci), int(nei)) if int(ci) < int(nei) else (int(nei), int(ci))
                blocked_edges.add((a, b))

    interface.local_partition_success = bool(child_cells and parent_cells)
    interface.local_partition_mode = "contour_barrier_region_grow" if interface.local_partition_success else "weak_partition"
    interface.local_patch_cell_ids = sorted(int(v) for v in patch_set)
    interface.local_barrier_cell_ids = sorted(int(v) for v in barrier_cells)
    interface.local_child_cell_ids = sorted(int(v) for v in child_cells)
    interface.local_parent_cell_ids = sorted(int(v) for v in parent_cells)
    interface.local_child_seed_cell_ids = sorted(int(v) for v in child_seeds)
    interface.local_parent_seed_cell_ids = sorted(int(v) for v in parent_seeds)
    return {
        "success": bool(interface.local_partition_success),
        "local_labels": local_labels,
        "blocked_edges": blocked_edges,
        "patch_cells": sorted(int(v) for v in patch_set),
        "barrier_cells": sorted(int(v) for v in barrier_cells),
    }


def seed_segment_core_cells(
    centers: np.ndarray,
    locator: "vtkStaticPointLocator",
    labels: np.ndarray,
    interface_patch_mask: np.ndarray,
    segments: List[Dict[str, Any]],
) -> List[Tuple[int, int]]:
    id_list = vtk.vtkIdList()
    candidates: Dict[int, Tuple[float, int]] = {}
    for seg in segments:
        pts = np.asarray(seg["path_points_oriented"], dtype=float)
        total = float(seg["length"])
        if pts.shape[0] < 2 or total <= EPS:
            continue
        radius = max(float(seg.get("mean_radius", 1.0)), 1.0)
        start_s = 0.10 * total if total <= 4.0 * radius else 0.18 * total
        end_s = 0.90 * total if total <= 4.0 * radius else 0.82 * total
        sample_count = max(4, min(14, int(math.ceil(total / max(radius, 0.5))) + 1))
        samples_s = [float(start_s + (end_s - start_s) * k / float(max(sample_count - 1, 1))) for k in range(sample_count)]
        for s in samples_s:
            p = polyline_point_at_abscissa(pts, s)
            id_list.Reset()
            locator.FindClosestNPoints(8, float(p[0]), float(p[1]), float(p[2]), id_list)
            for k in range(id_list.GetNumberOfIds()):
                ci = int(id_list.GetId(k))
                if ci < 0 or ci >= labels.shape[0] or labels[ci] >= 0 or bool(interface_patch_mask[ci]):
                    continue
                rel, abscissa, local_radius = segment_relative_distance(centers[ci], seg)
                inside, violation = point_between_segment_boundaries(centers[ci], seg)
                if (not inside) or float(violation) > 0.25 or float(rel) > 1.65:
                    continue
                abscissa_penalty = abs(float(abscissa) - s) / max(0.35 * total, local_radius, 1.0)
                score = float(rel + 0.20 * abscissa_penalty)
                prev = candidates.get(int(ci))
                if prev is None or score < float(prev[0]):
                    candidates[int(ci)] = (float(score), int(seg["segment_id"]))
    selected = [(int(ci), int(seg_id)) for ci, (score, seg_id) in sorted(candidates.items(), key=lambda item: (float(item[1][0]), int(item[1][1]), int(item[0])))]
    return selected


def propagate_segment_labels(
    centers: np.ndarray,
    adjacency: List[List[int]],
    labels: np.ndarray,
    assignment_modes: np.ndarray,
    blocked_edges: Set[Tuple[int, int]],
    segment_lookup: Dict[int, Dict[str, Any]],
) -> np.ndarray:
    n_cells = int(labels.shape[0])
    best_cost = np.full((n_cells,), np.inf, dtype=float)
    best_label = np.asarray(labels, dtype=np.int32).copy()
    frozen = (assignment_modes > 0) & (assignment_modes < 3) & (labels >= 0)
    pq: List[Tuple[float, int, int]] = []
    for ci in range(n_cells):
        seg_id = int(labels[ci])
        if seg_id < 0:
            continue
        best_cost[ci] = 0.0
        heapq.heappush(pq, (0.0, int(ci), int(seg_id)))

    while pq:
        cost, ci, seg_id = heapq.heappop(pq)
        if cost > float(best_cost[ci]) + 1e-12 or int(best_label[ci]) != int(seg_id):
            continue
        seg = segment_lookup.get(int(seg_id))
        if seg is None:
            continue
        for nei in adjacency[int(ci)]:
            nei = int(nei)
            edge = (ci, nei) if ci < nei else (nei, ci)
            if edge in blocked_edges:
                continue
            if frozen[nei] and int(best_label[nei]) != int(seg_id):
                continue
            proj = distance_to_prepared_polyline(centers[nei], seg["projection"])
            radius = max(float(proj["radius"]), float(seg.get("mean_radius", 1.0)), 1.0)
            inside, violation = point_between_segment_boundaries(centers[nei], seg)
            step = float(np.linalg.norm(centers[nei] - centers[ci]) / radius)
            boundary_penalty = 0.0 if inside else 0.40 + 0.95 * float(violation)
            new_cost = float(cost + step + boundary_penalty)
            if new_cost + 1e-9 < float(best_cost[nei]):
                best_cost[nei] = float(new_cost)
                best_label[nei] = int(seg_id)
                heapq.heappush(pq, (float(new_cost), int(nei), int(seg_id)))
                if not frozen[nei]:
                    assignment_modes[nei] = max(int(assignment_modes[nei]), 3)
    return best_label.astype(np.int32)


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


def point_between_segment_boundaries(point: np.ndarray, segment: Dict[str, Any]) -> Tuple[bool, float]:
    p = np.asarray(point, dtype=float).reshape(3)
    prox = segment["proximal_boundary"]
    dist = segment["distal_boundary"]
    prox_n = unit(np.asarray(prox.normal, dtype=float))
    dist_n = unit(np.asarray(dist.normal, dtype=float))
    prox_tol = float(segment.get("proximal_plane_tolerance", 0.0))
    dist_tol = float(segment.get("distal_plane_tolerance", 0.0))
    prox_signed = float(np.dot(p - np.asarray(prox.center, dtype=float), prox_n))
    dist_signed = float(np.dot(p - np.asarray(dist.center, dtype=float), dist_n))
    prox_ok = prox_signed >= -prox_tol
    dist_ok = dist_signed >= -dist_tol
    violation = 0.0
    if not prox_ok:
        violation += (-prox_signed - prox_tol) / max(float(segment.get("mean_radius", 1.0)), 1.0)
    if not dist_ok:
        violation += (-dist_signed - dist_tol) / max(float(segment.get("mean_radius", 1.0)), 1.0)
    return bool(prox_ok and dist_ok), float(max(violation, 0.0))


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
    scored: List[Tuple[float, float, int, bool]] = []
    for seg_id in candidate_segment_ids:
        seg = segment_lookup.get(int(seg_id))
        if seg is None:
            continue
        proj = distance_to_prepared_polyline(p, seg["projection"])
        radius = max(float(proj["radius"]), float(seg.get("mean_radius", 1.0)), 1.0)
        base_score = float(math.sqrt(float(proj["distance2"])) / radius)
        inside, violation = point_between_segment_boundaries(p, seg)
        boundary_penalty = 0.0 if inside else 1.5 + 1.25 * float(violation)
        confidence_penalty = 0.30 * (1.0 - float(seg.get("segment_confidence_cached", 0.5)))
        score = float(base_score + boundary_penalty + confidence_penalty)
        scored.append((score, float(proj["distance2"]), int(seg_id), not inside))

    if not scored:
        raise RuntimeError("No candidate segments available for point assignment.")
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return int(scored[0][2]), bool(scored[0][3])


def assign_surface_cells_to_segments(
    surface: "vtkPolyData",
    segments: List[Dict[str, Any]],
    transition_interfaces: List[TransitionInterface],
    resampling_step: float,
    warnings: List[str],
) -> Tuple[np.ndarray, Dict[int, int], int, Dict[str, np.ndarray]]:
    centers = build_cell_centers(surface)
    if centers.shape[0] != surface.GetNumberOfCells():
        raise RuntimeError("Failed to compute surface cell centers.")

    adjacency = build_surface_cell_adjacency(surface)
    segment_lookup = {int(seg["segment_id"]): seg for seg in segments}
    labels = np.full((surface.GetNumberOfCells(),), -1, dtype=np.int32)
    assignment_modes = np.zeros((surface.GetNumberOfCells(),), dtype=np.int32)
    interface_ids = np.full((surface.GetNumberOfCells(),), -1, dtype=np.int32)
    barrier_mask = np.zeros((surface.GetNumberOfCells(),), dtype=np.int32)
    interface_patch_mask = np.zeros((surface.GetNumberOfCells(),), dtype=np.int32)
    owner_strength = np.zeros((surface.GetNumberOfCells(),), dtype=float)
    blocked_edges: Set[Tuple[int, int]] = set()

    fallback_counts: Dict[int, int] = {int(seg["segment_id"]): 0 for seg in segments}
    total_fallback = 0
    local_interfaces_used = 0

    for order, interface in enumerate(
        sorted(
            transition_interfaces,
            key=lambda item: (
                float(item.confidence),
                float(item.connection_zone_score),
                0 if not item.low_confidence else -1,
            ),
            reverse=True,
        ),
        start=1,
    ):
        interface.interface_id = int(order)
        result = partition_transition_interface_on_surface(centers, adjacency, interface, segment_lookup)
        for ci in result.get("patch_cells", []):
            interface_patch_mask[int(ci)] = 1
        for ci in result.get("barrier_cells", []):
            barrier_mask[int(ci)] = 1
            if interface_ids[int(ci)] < 0:
                interface_ids[int(ci)] = int(interface.interface_id)
        if not bool(result.get("success", False)):
            warnings.extend([w for w in interface.warnings if w not in warnings])
            continue
        local_interfaces_used += 1
        for edge in result.get("blocked_edges", set()):
            blocked_edges.add((int(edge[0]), int(edge[1])))
        for ci, seg_id in result.get("local_labels", {}).items():
            if int(seg_id) < 0:
                continue
            if assignment_modes[int(ci)] == 1 and owner_strength[int(ci)] > float(interface.confidence) + 1e-9:
                continue
            labels[int(ci)] = int(seg_id)
            assignment_modes[int(ci)] = 1
            interface_ids[int(ci)] = int(interface.interface_id)
            owner_strength[int(ci)] = float(interface.confidence)

    local_partition_labels = labels.copy()
    if not transition_interfaces:
        warnings.append("W_INTERFACE_PARTITION_NONE: no parent-child transition interfaces were available; final labeling will rely more heavily on propagation and fallback.")
    elif local_interfaces_used == 0:
        warnings.append("W_INTERFACE_PARTITION_UNUSED: no transition interface produced a trusted local partition; fallback pressure may remain elevated.")

    center_locator = build_cell_center_locator(centers)
    core_seeds = seed_segment_core_cells(centers, center_locator, labels, interface_patch_mask, segments)
    for ci, seg_id in core_seeds:
        if labels[int(ci)] >= 0:
            continue
        labels[int(ci)] = int(seg_id)
        assignment_modes[int(ci)] = 2

    labels = propagate_segment_labels(
        centers=centers,
        adjacency=adjacency,
        labels=labels,
        assignment_modes=assignment_modes,
        blocked_edges=blocked_edges,
        segment_lookup=segment_lookup,
    )

    sample_step = max(0.60 * float(resampling_step), 0.35)
    locator, sample_segment_map = build_segment_sample_locator(segments, sample_step=sample_step)
    id_list = vtk.vtkIdList()
    n_samples = max(8, min(32, len(sample_segment_map)))
    for ci, center in enumerate(centers):
        if labels[int(ci)] >= 0:
            continue
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
        assignment_modes[int(ci)] = 4
        if used_fallback:
            total_fallback += 1
            fallback_counts[int(chosen_seg)] = fallback_counts.get(int(chosen_seg), 0) + 1

    unassigned = int(np.count_nonzero(labels < 0))
    if unassigned > 0:
        warnings.append(f"W_CELL_ASSIGNMENT_UNASSIGNED: {unassigned} cells remained unassigned after main pass.")
        for idx in np.flatnonzero(labels < 0).tolist():
            labels[int(idx)] = int(segments[0]["segment_id"])
            assignment_modes[int(idx)] = 4
            total_fallback += 1
            fallback_counts[int(segments[0]["segment_id"])] = fallback_counts.get(int(segments[0]["segment_id"]), 0) + 1

    if total_fallback > 0:
        warnings.append(
            f"W_CELL_ASSIGNMENT_FALLBACK: {total_fallback} / {surface.GetNumberOfCells()} cells required low-confidence global recovery after local interface partitioning and propagation."
        )
    debug_arrays = {
        "AssignmentMode": assignment_modes.astype(np.int32),
        "TransitionInterfaceId": interface_ids.astype(np.int32),
        "InterfaceBarrier": barrier_mask.astype(np.int32),
        "InterfacePatchMask": interface_patch_mask.astype(np.int32),
        "LocalPartitionSegmentId": local_partition_labels.astype(np.int32),
    }
    return labels.astype(np.int32), fallback_counts, int(total_fallback), debug_arrays


def add_segment_arrays_to_surface(surface: "vtkPolyData", labels: np.ndarray, extra_arrays: Optional[Dict[str, np.ndarray]] = None) -> "vtkPolyData":
    out = vtk.vtkPolyData()
    out.DeepCopy(surface)

    seg_arr = numpy_to_vtk(np.asarray(labels, dtype=np.int32), deep=True, array_type=vtk.VTK_INT)
    seg_arr.SetName("SegmentId")
    out.GetCellData().AddArray(seg_arr)

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
    for name, values in sorted((extra_arrays or {}).items(), key=lambda item: str(item[0])):
        arr_np = np.asarray(values)
        if arr_np.shape[0] != labels.shape[0]:
            continue
        if np.issubdtype(arr_np.dtype, np.integer):
            arr = numpy_to_vtk(arr_np.astype(np.int32), deep=True, array_type=vtk.VTK_INT)
        else:
            arr = numpy_to_vtk(arr_np.astype(float), deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetName(str(name))
        out.GetCellData().AddArray(arr)
    out.GetCellData().SetScalars(color_arr)
    return out


def np3_to_list(arr: Optional[np.ndarray]) -> Optional[List[float]]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).reshape(3)
    return [float(a[0]), float(a[1]), float(a[2])]


def points_to_lists(points: np.ndarray) -> List[List[float]]:
    pts = np.asarray(points, dtype=float)
    return [[float(p[0]), float(p[1]), float(p[2])] for p in pts]


def merge_graph_induced_root_pseudosegments(
    segments: List[Dict[str, Any]],
    transition_interfaces: List[TransitionInterface],
    warnings: List[str],
    resampling_step: float,
) -> Tuple[List[Dict[str, Any]], List[TransitionInterface]]:
    segs = [dict(seg) for seg in segments]
    interfaces = list(transition_interfaces)
    changed = True
    while changed:
        changed = False
        lookup = {int(seg["segment_id"]): seg for seg in segs}
        root_candidates = [seg for seg in segs if seg.get("parent_segment_id") is None and len(seg.get("child_segment_ids", [])) == 1]
        for root_seg in sorted(root_candidates, key=lambda item: int(item["segment_id"])):
            child_id = int(root_seg["child_segment_ids"][0])
            child_seg = lookup.get(int(child_id))
            if child_seg is None:
                continue
            root_len = float(root_seg.get("length", 0.0))
            root_radius = max(float(root_seg.get("mean_radius", 1.0)), 1.0)
            tangent_parent = polyline_tangent_at_abscissa(np.asarray(root_seg["path_points_oriented"], dtype=float), max(root_len - 0.5 * max(root_radius, 1.0), 0.0))
            tangent_child = polyline_tangent_at_abscissa(np.asarray(child_seg["path_points_oriented"], dtype=float), min(0.5 * max(root_radius, 1.0), float(child_seg.get("length", 0.0))))
            tangent_align = abs(float(np.dot(unit(tangent_parent), unit(tangent_child))))
            interface = next((iface for iface in interfaces if int(iface.parent_segment_id) == int(root_seg["segment_id"]) and int(iface.child_segment_id) == int(child_id)), None)
            prox_boundary = child_seg.get("proximal_boundary")
            interface_conf = float(interface.confidence) if interface is not None else float(prox_boundary.confidence if prox_boundary is not None else 0.0)
            interface_supported = bool(interface is not None and (not interface.low_confidence) and (not interface.synthetic) and float(interface.confidence) >= 0.55)
            short_threshold = max(2.4 * root_radius, 4.0 * float(resampling_step))
            if root_len > short_threshold or tangent_align < 0.82 or interface_supported:
                continue
            root_seg["path_points_oriented"] = concatenate_segment_paths(np.asarray(root_seg["path_points_oriented"], dtype=float), np.asarray(child_seg["path_points_oriented"], dtype=float))
            root_seg["path_radii_oriented"] = concatenate_segment_radii(root_seg.get("path_radii_oriented"), child_seg.get("path_radii_oriented"))
            root_seg["node_path_oriented"] = list(root_seg["node_path_oriented"]) + list(child_seg["node_path_oriented"][1:] if len(child_seg["node_path_oriented"]) > 1 else child_seg["node_path_oriented"])
            root_seg["supernode_distal"] = int(child_seg["supernode_distal"])
            root_seg["child_segment_ids"] = [int(v) for v in child_seg.get("child_segment_ids", [])]
            root_seg["length"] = float(polyline_length(np.asarray(root_seg["path_points_oriented"], dtype=float)))
            root_seg["warnings"] = list(root_seg.get("warnings", [])) + list(child_seg.get("warnings", []))
            root_seg.setdefault("merged_segment_ids", []).append(int(child_id))
            if "distal_boundary" in child_seg:
                root_seg["distal_boundary"] = child_seg["distal_boundary"]
            for grandchild_id in child_seg.get("child_segment_ids", []):
                grandchild = lookup.get(int(grandchild_id))
                if grandchild is not None:
                    grandchild["parent_segment_id"] = int(root_seg["segment_id"])
            for iface in interfaces:
                if int(iface.parent_segment_id) == int(child_id):
                    iface.parent_segment_id = int(root_seg["segment_id"])
                if int(iface.child_segment_id) == int(child_id) and int(iface.parent_segment_id) == int(root_seg["segment_id"]):
                    iface.low_confidence = True
            interfaces = [
                iface
                for iface in interfaces
                if not (int(iface.parent_segment_id) == int(root_seg["segment_id"]) and int(iface.child_segment_id) == int(child_id))
            ]
            segs = [seg for seg in segs if int(seg["segment_id"]) != int(child_id)]
            warnings.append(
                f"W_ROOT_PSEUDOSEGMENT_MERGED: merged short proximal segment {int(root_seg['segment_id'])} with child {int(child_id)} because the graph split was not supported by a strong surface-defined transition (confidence={interface_conf:.3f})."
            )
            changed = True
            break
    lookup = {int(seg["segment_id"]): seg for seg in segs}
    for seg in segs:
        seg["child_segment_ids"] = sorted(int(v) for v in seg.get("child_segment_ids", []) if int(v) in lookup)
    return segs, interfaces


def transition_interface_to_json(interface: TransitionInterface) -> Dict[str, Any]:
    return {
        "interface_id": int(interface.interface_id),
        "child_segment_id": int(interface.child_segment_id),
        "parent_segment_id": int(interface.parent_segment_id),
        "contour_points": points_to_lists(interface.contour_points),
        "contour_centroid": np3_to_list(interface.contour_centroid),
        "contour_normal": np3_to_list(interface.contour_normal),
        "parent_projection_point": np3_to_list(interface.parent_projection_point),
        "parent_projection_abscissa": (None if interface.parent_projection_abscissa is None else float(interface.parent_projection_abscissa)),
        "partition_normal": np3_to_list(interface.partition_normal),
        "partition_axis_u": np3_to_list(interface.partition_axis_u),
        "partition_axis_v": np3_to_list(interface.partition_axis_v),
        "confidence": float(interface.confidence),
        "connection_zone_score": float(interface.connection_zone_score),
        "method_tag": str(interface.method_tag),
        "representative_child_abscissa": (None if interface.representative_child_abscissa is None else float(interface.representative_child_abscissa)),
        "stable_zone_start_abscissa": (None if interface.stable_zone_start_abscissa is None else float(interface.stable_zone_start_abscissa)),
        "stable_zone_end_abscissa": (None if interface.stable_zone_end_abscissa is None else float(interface.stable_zone_end_abscissa)),
        "stable_zone_indices": [int(interface.stable_zone_start_index), int(interface.stable_zone_end_index)],
        "representative_index": int(interface.representative_index),
        "local_spacing": float(interface.local_spacing),
        "child_window": float(interface.child_window),
        "parent_window": float(interface.parent_window),
        "patch_radius": float(interface.patch_radius),
        "child_radius": float(interface.child_radius),
        "parent_radius": float(interface.parent_radius),
        "contour_quality": float(interface.contour_quality),
        "axis_stability": float(interface.axis_stability),
        "synthetic": bool(interface.synthetic),
        "low_confidence": bool(interface.low_confidence),
        "local_partition_success": bool(interface.local_partition_success),
        "local_partition_mode": str(interface.local_partition_mode),
        "local_patch_cell_count": int(len(interface.local_patch_cell_ids)),
        "local_barrier_cell_count": int(len(interface.local_barrier_cell_ids)),
        "local_child_cell_count": int(len(interface.local_child_cell_ids)),
        "local_parent_cell_count": int(len(interface.local_parent_cell_ids)),
        "warnings": list(interface.warnings),
    }


def segment_confidence(segment: Dict[str, Any], total_cells: int) -> float:
    score = 0.20
    prox = segment["proximal_boundary"]
    dist = segment["distal_boundary"]
    score += 0.20 * float(clamp(prox.confidence, 0.0, 1.0))
    score += 0.15 * float(clamp(dist.confidence, 0.0, 1.0))
    if not bool(prox.synthetic):
        score += 0.10
    if not bool(dist.synthetic):
        score += 0.08
    if float(segment["length"]) > 2.0:
        score += 0.07
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
    transition_interfaces: List[TransitionInterface],
    warnings: List[str],
    termination_mode: str,
    centerline_info: Dict[str, Any],
    total_cells: int,
    assignment_mode_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    interface_lookup = {int(iface.interface_id): iface for iface in transition_interfaces}
    segments_json: List[Dict[str, Any]] = []
    for seg in sorted(segments, key=lambda item: int(item["segment_id"])):
        prox = seg["proximal_boundary"]
        dist = seg["distal_boundary"]
        prox_interface = interface_lookup.get(int(seg.get("proximal_transition_interface_id", -1)))
        segments_json.append(
            {
                "segment_id": int(seg["segment_id"]),
                "proximal_boundary_type": str(prox.boundary_type),
                "distal_boundary_type": str(dist.boundary_type),
                "proximal_boundary_method": str(prox.boundary_method),
                "distal_boundary_method": str(dist.boundary_method),
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
                "proximal_boundary_confidence": float(prox.confidence),
                "distal_boundary_confidence": float(dist.confidence),
                "connection_zone_score": float(prox.connection_zone_score),
                "parent_projection_point": np3_to_list(prox.parent_projection_point),
                "parent_projection_abscissa": (None if prox.parent_projection_abscissa is None else float(prox.parent_projection_abscissa)),
                "boundary_confidence": {
                    "proximal": float(prox.confidence),
                    "distal": float(dist.confidence),
                },
                "boundary_is_synthetic": {
                    "proximal": bool(prox.synthetic),
                    "distal": bool(dist.synthetic),
                },
                "boundary_confidence_flags": {
                    "proximal_fallback_used": bool(prox.fallback_used),
                    "distal_fallback_used": bool(dist.fallback_used),
                },
                "interface_method_tag": (str(prox_interface.method_tag) if prox_interface is not None else str(prox.boundary_method)),
                "boundary_confidence_detail": {
                    "proximal_boundary_confidence": float(prox.confidence),
                    "distal_boundary_confidence": float(dist.confidence),
                    "connection_zone_score": float(prox.connection_zone_score),
                },
                "local_partition": {
                    "interface_id": (None if prox_interface is None else int(prox_interface.interface_id)),
                    "success": bool(prox_interface.local_partition_success) if prox_interface is not None else False,
                    "mode": (str(prox_interface.local_partition_mode) if prox_interface is not None else "not_applicable"),
                },
                "fallback_flags": {
                    "proximal_boundary_fallback": bool(prox.fallback_used),
                    "distal_boundary_fallback": bool(dist.fallback_used),
                    "cell_assignment_fallback_count": int(seg.get("fallback_cell_count", 0)),
                },
                "length": float(seg["length"]),
                "mean_radius": float(seg.get("mean_radius", 0.0)),
                "child_coupled_parent_interface_ids": [int(v) for v in seg.get("child_transition_interface_ids", [])],
                "merged_segment_ids": [int(v) for v in seg.get("merged_segment_ids", [])],
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
        "transition_interfaces": [transition_interface_to_json(iface) for iface in sorted(transition_interfaces, key=lambda item: int(item.interface_id))],
        "cell_assignment_mode_counts": dict(assignment_mode_counts or {}),
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
    centerline_info["endpoint_count"] = int(len(endpoints))
    centerline_info["junction_keynode_count"] = int(sum(1 for _, d in deg.items() if int(d) >= 3))

    term_to_endpoint = map_terminations_to_centerline_endpoints(endpoints, cl_pts, terms, root_term, warnings)
    root_term_idx = int(terms.index(root_term))
    root_endpoint = int(term_to_endpoint[root_term_idx]["endpoint_node"]) if root_term_idx in term_to_endpoint else int(endpoints[0])

    supernode_for_keynode, chains, _, _ = collapse_junction_clusters(adjacency, cl_pts, float(centerline_info["resampling_step"]))
    super_adj, segment_records = build_supernode_graph_and_segments(chains, supernode_for_keynode, cl_pts, radii)
    if not segment_records:
        raise RuntimeError("Failed to derive any centerline segments from the centerline graph.")
    centerline_info["reduced_supernode_count"] = int(len(super_adj))
    centerline_info["segment_record_count"] = int(len(segment_records))

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

    parent_segment_for_supernode: Dict[int, Optional[int]] = {int(root_supernode): None}
    for seg in sorted(ordered_segments, key=lambda item: float(dist_super.get(int(item["supernode_proximal"]), float("inf")))):
        prox_super = int(seg["supernode_proximal"])
        dist_supernode = int(seg["supernode_distal"])
        seg["parent_segment_id"] = parent_segment_for_supernode.get(prox_super)
        parent_segment_for_supernode[dist_supernode] = int(seg["segment_id"])

    for seg in ordered_segments:
        child_ids = segments_by_proximal_supernode.get(int(seg["supernode_distal"]), [])
        seg["child_segment_ids"] = sorted(int(v) for v in child_ids if int(v) != int(seg["segment_id"]))

    segment_lookup: Dict[int, Dict[str, Any]] = {int(seg["segment_id"]): seg for seg in ordered_segments}
    supernode_to_term: Dict[int, BoundaryProfile] = {}
    for term_idx, supernode_id in term_supernode_map.items():
        supernode_to_term[int(supernode_id)] = terms[int(term_idx)]

    section_debug_records: List[Dict[str, Any]] = []
    transition_interfaces: List[TransitionInterface] = []

    for seg in ordered_segments:
        path_points = np.asarray(seg["path_points_oriented"], dtype=float)
        path_radii = seg["path_radii_oriented"]
        mean_radius = max(float(np.nanmedian(path_radii)) if path_radii is not None and np.asarray(path_radii).size > 0 else max(float(seg["length"]) * 0.05, 1.0), 1.0)
        seg["mean_radius"] = float(mean_radius)
        seg["merged_segment_ids"] = []
        seg["child_transition_interface_ids"] = []
        seg["proximal_transition_interface_id"] = -1

    segment_lookup = {int(seg["segment_id"]): seg for seg in ordered_segments}
    for seg in ordered_segments:
        path_points = np.asarray(seg["path_points_oriented"], dtype=float)
        start_inward = unit(path_points[1] - path_points[0]) if path_points.shape[0] >= 2 else np.array([0.0, 0.0, 1.0], dtype=float)
        end_inward = unit(path_points[-2] - path_points[-1]) if path_points.shape[0] >= 2 else -start_inward

        prox_super = int(seg["supernode_proximal"])
        dist_supernode = int(seg["supernode_distal"])

        prox_term = supernode_to_term.get(prox_super)
        if prox_term is not None or seg["parent_segment_id"] is None:
            proximal_boundary = build_terminal_boundary_profile(
                prox_term,
                fallback_center=path_points[0],
                inward_direction=start_inward,
                fallback_radius=float(seg["mean_radius"]),
                label=f"segment_{seg['segment_id']}_proximal",
            )
        else:
            parent_seg = segment_lookup[int(seg["parent_segment_id"])]
            proximal_boundary, parent_info, child_debug, transition_interface = refine_child_proximal_boundary(
                surface=surface_clean,
                parent_seg=parent_seg,
                child_seg=seg,
                resampling_step=float(centerline_info["resampling_step"]),
                warnings=warnings,
            )
            if transition_interface is not None:
                transition_interfaces.append(transition_interface)
            section_debug_records.extend(child_debug)
            _ = parent_info

        seg["proximal_boundary"] = proximal_boundary
        seg["warnings"].extend(list(proximal_boundary.warnings))
        seg["proximal_plane_tolerance"] = float(max(float(seg["mean_radius"]) * 0.40, 0.45 * float(centerline_info["resampling_step"]), 0.35))

        if dist_supernode in supernode_to_term:
            distal_boundary = build_terminal_boundary_profile(
                supernode_to_term.get(dist_supernode),
                fallback_center=path_points[-1],
                inward_direction=end_inward,
                fallback_radius=float(seg["mean_radius"]),
                label=f"segment_{seg['segment_id']}_distal",
            )
            seg["distal_boundary"] = distal_boundary
            seg["warnings"].extend(list(distal_boundary.warnings))
    ordered_segments, transition_interfaces = merge_graph_induced_root_pseudosegments(
        ordered_segments,
        transition_interfaces,
        warnings=warnings,
        resampling_step=float(centerline_info["resampling_step"]),
    )
    segment_lookup = {int(seg["segment_id"]): seg for seg in ordered_segments}
    for seg in ordered_segments:
        seg["child_segment_ids"] = []
        path_points = np.asarray(seg["path_points_oriented"], dtype=float)
        path_radii = seg.get("path_radii_oriented")
        seg["length"] = float(polyline_length(path_points))
        seg["mean_radius"] = max(float(np.nanmedian(path_radii)) if path_radii is not None and np.asarray(path_radii).size > 0 else max(float(seg["length"]) * 0.05, 1.0), 1.0)
        seg["child_transition_interface_ids"] = []
        seg["proximal_transition_interface_id"] = -1
    for seg in ordered_segments:
        parent_id = seg.get("parent_segment_id")
        if parent_id is not None and int(parent_id) in segment_lookup:
            segment_lookup[int(parent_id)]["child_segment_ids"].append(int(seg["segment_id"]))
    for seg in ordered_segments:
        seg["child_segment_ids"] = sorted(int(v) for v in seg.get("child_segment_ids", []))

    for seg in ordered_segments:
        if "distal_boundary" in seg:
            continue
        child_ifaces = [iface for iface in transition_interfaces if int(iface.parent_segment_id) == int(seg["segment_id"])]
        if child_ifaces:
            distal_boundary, parent_debug = build_parent_distal_boundary_from_children(
                surface=surface_clean,
                parent_seg=seg,
                child_interfaces=child_ifaces,
                resampling_step=float(centerline_info["resampling_step"]),
                warnings=warnings,
            )
            section_debug_records.extend(parent_debug)
        else:
            path_points = np.asarray(seg["path_points_oriented"], dtype=float)
            end_inward = unit(path_points[-2] - path_points[-1]) if path_points.shape[0] >= 2 else np.array([0.0, 0.0, 1.0], dtype=float)
            distal_boundary = synthetic_transition_boundary(
                center=path_points[-1],
                normal=end_inward,
                radius=max(float(seg.get("mean_radius", 1.0)), 1.0),
                label=f"segment_{seg['segment_id']}_distal",
                boundary_type="junction_transition_parent",
                boundary_method="synthetic_parent_low_confidence",
                confidence=0.10,
                extra_warning="No terminal and no child refinement available for distal boundary.",
            )
            warnings.append(f"W_DISTAL_BOUNDARY_SYNTHETIC: segment {int(seg['segment_id'])} used synthetic distal boundary because no refined distal context existed.")
        seg["distal_boundary"] = distal_boundary
        seg["warnings"].extend(list(distal_boundary.warnings))
        seg["distal_plane_tolerance"] = float(max(float(seg.get("mean_radius", 1.0)) * 0.40, 0.45 * float(centerline_info["resampling_step"]), 0.35))

    for seg in ordered_segments:
        seg["projection"] = prepared_projection_data(np.asarray(seg["path_points_oriented"], dtype=float), seg["path_radii_oriented"])

    transition_interfaces = [
        iface
        for iface in transition_interfaces
        if int(iface.child_segment_id) in segment_lookup and int(iface.parent_segment_id) in segment_lookup
    ]

    for seg in ordered_segments:
        seg["segment_confidence_cached"] = float(segment_confidence(seg, max(surface_clean.GetNumberOfCells(), 1)))

    labels, fallback_counts, total_fallback, surface_debug_arrays = assign_surface_cells_to_segments(
        surface=surface_clean,
        segments=ordered_segments,
        transition_interfaces=transition_interfaces,
        resampling_step=float(centerline_info["resampling_step"]),
        warnings=warnings,
    )
    for iface in transition_interfaces:
        segment_lookup[int(iface.child_segment_id)]["proximal_transition_interface_id"] = int(iface.interface_id)
        segment_lookup[int(iface.parent_segment_id)]["child_transition_interface_ids"].append(int(iface.interface_id))
    for seg in ordered_segments:
        seg["child_transition_interface_ids"] = sorted(set(int(v) for v in seg.get("child_transition_interface_ids", [])))
    for seg in ordered_segments:
        seg["fallback_cell_count"] = int(fallback_counts.get(int(seg["segment_id"]), 0))
        seg["segment_confidence_cached"] = float(segment_confidence(seg, int(surface_clean.GetNumberOfCells())))

    boundary_debug_records: List[Dict[str, Any]] = []
    for seg in ordered_segments:
        prox = seg["proximal_boundary"]
        dist = seg["distal_boundary"]
        boundary_debug_records.append(
            {
                "segment_id": int(seg["segment_id"]),
                "boundary_side": 0,
                "boundary_type": str(prox.boundary_type),
                "boundary_method": str(prox.boundary_method),
                "profile_points": np.asarray(prox.profile_points, dtype=float),
                "fallback_used": bool(prox.fallback_used),
                "confidence": float(prox.confidence),
            }
        )
        boundary_debug_records.append(
            {
                "segment_id": int(seg["segment_id"]),
                "boundary_side": 1,
                "boundary_type": str(dist.boundary_type),
                "boundary_method": str(dist.boundary_method),
                "profile_points": np.asarray(dist.profile_points, dtype=float),
                "fallback_used": bool(dist.fallback_used),
                "confidence": float(dist.confidence),
            }
        )
    for iface in transition_interfaces:
        boundary_debug_records.append(
            {
                "segment_id": int(iface.child_segment_id),
                "boundary_side": 2,
                "boundary_type": "transition_interface",
                "boundary_method": str(iface.method_tag),
                "profile_points": np.asarray(iface.contour_points, dtype=float),
                "fallback_used": bool(iface.low_confidence or iface.synthetic),
                "confidence": float(iface.confidence),
            }
        )

    assignment_mode_counts = {
        "local_interface_partition": int(np.count_nonzero(surface_debug_arrays["AssignmentMode"] == 1)),
        "segment_core_seed": int(np.count_nonzero(surface_debug_arrays["AssignmentMode"] == 2)),
        "geodesic_propagation": int(np.count_nonzero(surface_debug_arrays["AssignmentMode"] == 3)),
        "fallback_global_recovery": int(np.count_nonzero(surface_debug_arrays["AssignmentMode"] == 4)),
    }

    surface_out = add_segment_arrays_to_surface(surface_clean, labels, extra_arrays=surface_debug_arrays)
    write_vtp(surface_out, args.output_vtp)

    if args.debug:
        write_vtp(build_segment_centerlines_debug_polydata(ordered_segments), args.centerlines_debug)
        write_vtp(build_section_debug_polydata(section_debug_records), args.ostium_crosssections_debug)
        write_vtp(build_boundary_debug_polydata(boundary_debug_records), args.segment_boundaries_debug)

    metadata = build_metadata(
        input_path=input_path,
        output_surface_path=args.output_vtp,
        output_metadata_path=args.metadata_json,
        segments=ordered_segments,
        transition_interfaces=transition_interfaces,
        warnings=warnings,
        termination_mode=termination_mode,
        centerline_info=centerline_info,
        total_cells=int(surface_clean.GetNumberOfCells()),
        assignment_mode_counts=assignment_mode_counts,
    )
    metadata["face_partition_region_count"] = int(len(face_regions))
    metadata["cell_assignment_total_fallback"] = int(total_fallback)
    metadata["debug_outputs"] = {
        "surface_cleaned_vtp": os.path.abspath(args.surface_cleaned),
        "centerlines_debug_vtp": os.path.abspath(args.centerlines_debug),
        "ostium_crosssections_debug_vtp": os.path.abspath(args.ostium_crosssections_debug),
        "segment_boundaries_debug_vtp": os.path.abspath(args.segment_boundaries_debug),
    }
    write_json(metadata, args.metadata_json)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a vascular lumen surface into geometric vessel segments using "
            "centerline topology plus coordinated parent-daughter cross-section refinement."
        )
    )
    parser.add_argument("--input", default=INPUT_VTP_PATH, help="Input vascular lumen .vtp file.")
    parser.add_argument("--output-vtp", default=OUTPUT_SEGMENTS_VTP_PATH, help="Main output surface VTP path.")
    parser.add_argument("--metadata-json", default=OUTPUT_METADATA_PATH, help="Metadata JSON output path.")
    parser.add_argument("--surface-cleaned", default=OUTPUT_SURFACE_CLEANED_PATH, help="Optional cleaned surface debug VTP path.")
    parser.add_argument("--centerlines-debug", default=OUTPUT_CENTERLINES_DEBUG_PATH, help="Optional centerlines debug VTP path.")
    parser.add_argument("--ostium-crosssections-debug", default=OUTPUT_OSTIUM_CROSSSECTIONS_DEBUG_PATH, help="Coordinated parent-daughter section debug VTP path.")
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
