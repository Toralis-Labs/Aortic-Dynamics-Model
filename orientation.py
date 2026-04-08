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
INPUT_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\0044_H_ABAO_AAA\\0156_0001.vtp"
OUTPUT_SURFACE_WITH_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_surface_with_centerlines.vtp"
OUTPUT_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines.vtp"
OUTPUT_METADATA_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines_metadata.json"
OUTPUT_DEBUG_CENTERLINES_RAW_PATH = ""  # optional: e.g. "debug_centerlines_raw.vtp" (empty disables)

import os
import sys
import json
import math
import argparse
import glob
import importlib
import platform
import re
import subprocess
import traceback
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

SCRIPT_PATH = os.path.abspath(__file__) if "__file__" in globals() else os.path.abspath(sys.argv[0])
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
_AUTO_VMTK_REEXEC_ENV = "ORIENT_VMTK_REEXEC_ACTIVE"
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
LABEL_CELIAC_TRUNK = 2
LABEL_SUPERIOR_MESENTERIC_ARTERY = 3
LABEL_INFERIOR_MESENTERIC_ARTERY = 4
LABEL_RIGHT_RENAL = 5
LABEL_LEFT_RENAL = 6
LABEL_RIGHT_ILIAC = 7
LABEL_LEFT_ILIAC = 8
LABEL_RIGHT_INTERNAL_ILIAC = 9
LABEL_RIGHT_EXTERNAL_ILIAC = 10
LABEL_LEFT_INTERNAL_ILIAC = 11
LABEL_LEFT_EXTERNAL_ILIAC = 12

# Legacy aliases retained so the automatic fallback path can keep using the old helper names.
LABEL_RIGHT_RENAL_ARTERY = LABEL_RIGHT_RENAL
LABEL_LEFT_RENAL_ARTERY = LABEL_LEFT_RENAL
LABEL_RIGHT_COMMON_ILIAC = LABEL_RIGHT_ILIAC
LABEL_LEFT_COMMON_ILIAC = LABEL_LEFT_ILIAC

LABEL_ID_TO_NAME = {
    LABEL_OTHER: "other",
    LABEL_AORTA_TRUNK: "abdominal_aorta_trunk",
    LABEL_CELIAC_TRUNK: "celiac_trunk",
    LABEL_SUPERIOR_MESENTERIC_ARTERY: "superior_mesenteric_artery",
    LABEL_INFERIOR_MESENTERIC_ARTERY: "inferior_mesenteric_artery",
    LABEL_RIGHT_RENAL: "right_renal_artery",
    LABEL_LEFT_RENAL: "left_renal_artery",
    LABEL_RIGHT_ILIAC: "right_common_iliac",
    LABEL_LEFT_ILIAC: "left_common_iliac",
    LABEL_RIGHT_INTERNAL_ILIAC: "right_internal_iliac",
    LABEL_RIGHT_EXTERNAL_ILIAC: "right_external_iliac",
    LABEL_LEFT_INTERNAL_ILIAC: "left_internal_iliac",
    LABEL_LEFT_EXTERNAL_ILIAC: "left_external_iliac",
}

LABEL_NAME_TO_ID = {str(v): int(k) for k, v in LABEL_ID_TO_NAME.items()}
LABEL_PRIORITY_ORDER = {
    LABEL_AORTA_TRUNK: 0,
    LABEL_CELIAC_TRUNK: 1,
    LABEL_SUPERIOR_MESENTERIC_ARTERY: 2,
    LABEL_INFERIOR_MESENTERIC_ARTERY: 3,
    LABEL_RIGHT_RENAL: 4,
    LABEL_LEFT_RENAL: 5,
    LABEL_RIGHT_ILIAC: 6,
    LABEL_LEFT_ILIAC: 7,
    LABEL_RIGHT_INTERNAL_ILIAC: 8,
    LABEL_RIGHT_EXTERNAL_ILIAC: 9,
    LABEL_LEFT_INTERNAL_ILIAC: 10,
    LABEL_LEFT_EXTERNAL_ILIAC: 11,
    LABEL_OTHER: 99,
}

DEFAULT_PRELABELED_TERMINAL_FACE_MAP: Dict[int, Dict[str, Any]] = {
    2: {"name": "abdominal_aorta_inlet", "cap_id": 1, "terminal_type": "inlet"},
    3: {"name": "celiac_branch", "cap_id": 2, "terminal_type": "outlet"},
    4: {"name": "celiac_artery", "cap_id": 2, "terminal_type": "outlet"},
    5: {"name": "left_external_iliac", "cap_id": 2, "terminal_type": "outlet"},
    6: {"name": "right_external_iliac", "cap_id": 2, "terminal_type": "outlet"},
    7: {"name": "inferior_mesenteric_artery", "cap_id": 2, "terminal_type": "outlet"},
    8: {"name": "left_internal_iliac", "cap_id": 2, "terminal_type": "outlet"},
    9: {"name": "right_internal_iliac", "cap_id": 2, "terminal_type": "outlet"},
    10: {"name": "left_renal_artery", "cap_id": 2, "terminal_type": "outlet"},
    11: {"name": "right_renal_artery", "cap_id": 2, "terminal_type": "outlet"},
    12: {"name": "superior_mesenteric_artery", "cap_id": 2, "terminal_type": "outlet"},
}

PRELABELED_CELIAC_OUTLET_NAMES = ("celiac_branch", "celiac_artery")
PRELABELED_RIGHT_ILIAC_OUTLET_NAMES = ("right_internal_iliac", "right_external_iliac")
PRELABELED_LEFT_ILIAC_OUTLET_NAMES = ("left_internal_iliac", "left_external_iliac")
PRELABELED_DIRECT_AORTIC_BRANCH_LABELS = {
    "superior_mesenteric_artery": LABEL_SUPERIOR_MESENTERIC_ARTERY,
    "inferior_mesenteric_artery": LABEL_INFERIOR_MESENTERIC_ARTERY,
    "right_renal_artery": LABEL_RIGHT_RENAL,
    "left_renal_artery": LABEL_LEFT_RENAL,
}

@dataclass
class TerminationLoop:
    center: np.ndarray
    area: float
    diameter_eq: float
    normal: np.ndarray
    rms_planarity: float
    n_points: int
    source: str


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
            "distance_sq": float(np.dot(diff, diff)),
            "segment_index": 0,
            "t": 0.0,
            "abscissa": 0.0,
            "tangent": np.zeros((3,), dtype=float),
        }

    s = compute_abscissa(pts)
    best: Optional[Dict[str, Any]] = None
    for idx in range(pts.shape[0] - 1):
        proj, t, d2 = project_point_to_segment(point, pts[idx], pts[idx + 1])
        tangent = unit(pts[idx + 1] - pts[idx])
        abscissa = float(s[idx] + t * float(np.linalg.norm(pts[idx + 1] - pts[idx])))
        item = {
            "point": proj.astype(float),
            "distance_sq": float(d2),
            "segment_index": int(idx),
            "t": float(t),
            "abscissa": float(abscissa),
            "tangent": tangent.astype(float),
        }
        if best is None or float(item["distance_sq"]) < float(best["distance_sq"]):
            best = item
    return best


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
# Pre-labeled terminal faces
# -----------------------------
def resolve_terminal_face_map_path(input_path: str, explicit_path: Optional[str]) -> str:
    explicit_resolved = _resolve_user_path(explicit_path or "")
    if explicit_resolved:
        return explicit_resolved
    input_abs = _resolve_user_path(input_path)
    base_dir = os.path.dirname(os.path.abspath(input_abs))
    return os.path.join(base_dir, "face_id_to_name.json")


def load_terminal_face_map(path: str, warnings: Optional[List[str]] = None) -> Tuple[Dict[int, Dict[str, Any]], str]:
    work_warnings = warnings if warnings is not None else []
    resolved = _resolve_user_path(path)
    if resolved and os.path.isfile(resolved):
        with open(resolved, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Terminal face map must be a JSON object keyed by face id: {resolved}")
        parsed: Dict[int, Dict[str, Any]] = {}
        for key, value in raw.items():
            try:
                face_id = int(key)
            except Exception as exc:
                raise RuntimeError(f"Terminal face map key is not an integer face id: {key!r}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Terminal face map entry for face id {face_id} must be an object.")
            parsed[int(face_id)] = {
                "name": str(value.get("name", f"face_{face_id}")),
                "cap_id": (None if value.get("cap_id") is None else int(value.get("cap_id"))),
                "terminal_type": str(value.get("terminal_type", "outlet")).strip().lower() or "outlet",
            }
        return parsed, resolved

    work_warnings.append(
        "W_TERMINAL_FACE_MAP_DEFAULT: terminal face map sidecar not found; using built-in default ModelFaceID mapping."
    )
    return {
        int(fid): {
            "name": str(info.get("name", f"face_{fid}")),
            "cap_id": (None if info.get("cap_id") is None else int(info.get("cap_id"))),
            "terminal_type": str(info.get("terminal_type", "outlet")).strip().lower() or "outlet",
        }
        for fid, info in DEFAULT_PRELABELED_TERMINAL_FACE_MAP.items()
    }, "__builtin_default__"


def extract_model_face_regions(
    pd_tri: "vtkPolyData",
    face_array_name: str = "ModelFaceID",
    cap_array_name: str = "CapID",
) -> Dict[int, Dict[str, Any]]:
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
    for face_id in sorted(int(v) for v in np.unique(face_vals).tolist()):
        mask = np.asarray(face_vals == int(face_id), dtype=bool)
        if not np.any(mask):
            continue
        areas = area_vals[mask]
        cells = centers_vals[mask]
        normals = normal_vals[mask] if normal_vals.shape[0] == face_vals.shape[0] else np.zeros((cells.shape[0], 3), dtype=float)
        cap_subset = cap_vals[mask] if cap_vals.shape[0] == face_vals.shape[0] else np.zeros((cells.shape[0],), dtype=np.int64)
        total_area = float(np.sum(areas))
        if total_area <= EPS:
            centroid = np.mean(cells, axis=0) if cells.shape[0] else np.zeros((3,), dtype=float)
            mean_normal = np.zeros((3,), dtype=float)
        else:
            centroid = np.sum(cells * areas[:, None], axis=0) / total_area
            mean_normal = unit(np.sum(normals * areas[:, None], axis=0))
        cap_id = int(max(set(int(v) for v in cap_subset.tolist()), key=lambda v: int(np.count_nonzero(cap_subset == v)))) if cap_subset.size else 0
        regions[int(face_id)] = {
            "face_id": int(face_id),
            "cell_count": int(np.count_nonzero(mask)),
            "total_area": float(total_area),
            "centroid": np.asarray(centroid, dtype=float),
            "mean_normal": np.asarray(mean_normal, dtype=float),
            "diameter_eq": float(math.sqrt(4.0 * total_area / math.pi)) if total_area > 0.0 else 0.0,
            "cap_id": int(cap_id),
            "cell_ids": np.flatnonzero(mask).astype(int).tolist(),
        }
    return regions


def face_region_to_termination_loop(region: Dict[str, Any], source_prefix: str = "model_face_id") -> TerminationLoop:
    return TerminationLoop(
        center=np.asarray(region.get("centroid", np.zeros((3,), dtype=float)), dtype=float).reshape(3),
        area=float(region.get("total_area", 0.0)),
        diameter_eq=float(region.get("diameter_eq", 0.0)),
        normal=unit(np.asarray(region.get("mean_normal", np.zeros((3,), dtype=float)), dtype=float).reshape(3)),
        rms_planarity=float("nan"),
        n_points=int(region.get("cell_count", 0)),
        source=f"{source_prefix}:{int(region.get('face_id', -1))}",
    )


def build_prelabeled_terminal_regions(
    pd_tri: "vtkPolyData",
    terminal_face_map: Dict[int, Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    work_warnings = warnings if warnings is not None else []
    face_regions = extract_model_face_regions(pd_tri, face_array_name="ModelFaceID", cap_array_name="CapID")
    if not face_regions:
        return {}, [], None

    mapped_regions: List[Dict[str, Any]] = []
    inlet_region: Optional[Dict[str, Any]] = None
    for face_id, mapping in sorted(terminal_face_map.items()):
        region = face_regions.get(int(face_id))
        if region is None:
            work_warnings.append(f"W_MODEL_FACE_ID_MISSING: mapped face id {int(face_id)} was not found on the surface.")
            continue
        merged = dict(region)
        merged["name"] = str(mapping.get("name", f"face_{face_id}"))
        merged["terminal_type"] = str(mapping.get("terminal_type", "outlet")).strip().lower() or "outlet"
        merged["mapped_cap_id"] = mapping.get("cap_id")
        mapped_regions.append(merged)
        if merged["terminal_type"] == "inlet":
            inlet_region = merged

    if inlet_region is None:
        work_warnings.append("W_MODEL_FACE_INLET_MISSING: no inlet face was identified from the terminal face map.")
    return face_regions, mapped_regions, inlet_region


def assign_face_regions_to_centerline_endpoints(
    endpoints: List[int],
    pts: np.ndarray,
    mapped_regions: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    work_warnings = warnings if warnings is not None else []
    if not endpoints:
        return {}

    matches: Dict[str, Dict[str, Any]] = {}
    inlet_regions = [region for region in mapped_regions if str(region.get("terminal_type", "")) == "inlet"]
    outlet_regions = [region for region in mapped_regions if str(region.get("terminal_type", "")) != "inlet"]

    used_endpoints: set[int] = set()
    if inlet_regions:
        inlet_region = inlet_regions[0]
        inlet_ep, inlet_conf = choose_centerline_endpoint_for_termination(
            endpoints,
            pts,
            face_region_to_termination_loop(inlet_region),
            exclude=None,
        )
        if inlet_ep is not None:
            used_endpoints.add(int(inlet_ep))
            matches[str(inlet_region["name"])] = {
                "face_id": int(inlet_region["face_id"]),
                "name": str(inlet_region["name"]),
                "terminal_type": str(inlet_region["terminal_type"]),
                "endpoint_node": int(inlet_ep),
                "endpoint_xyz": np.asarray(pts[int(inlet_ep)], dtype=float),
                "centroid": np.asarray(inlet_region["centroid"], dtype=float),
                "cap_id": int(inlet_region.get("cap_id", 0)),
                "distance": float(np.linalg.norm(pts[int(inlet_ep)] - np.asarray(inlet_region["centroid"], dtype=float))),
                "confidence": float(inlet_conf),
            }
        else:
            work_warnings.append("W_MODEL_FACE_INLET_ENDPOINT_MAP_FAILED: failed to map inlet face centroid to a centerline endpoint.")

    candidate_pairs: List[Tuple[float, float, int, str, Dict[str, Any], int]] = []
    outlet_order = sorted(outlet_regions, key=lambda region: (-float(region.get("total_area", 0.0)), int(region.get("face_id", -1))))
    for order_index, region in enumerate(outlet_order):
        term = face_region_to_termination_loop(region)
        diameter_eq = max(float(term.diameter_eq), 1e-3)
        for ep in endpoints:
            ep_i = int(ep)
            if ep_i in used_endpoints:
                continue
            dist_to_center = float(np.linalg.norm(pts[ep_i] - term.center))
            score = float(dist_to_center / diameter_eq)
            candidate_pairs.append((score, dist_to_center, int(order_index), str(region["name"]), region, ep_i))

    assigned_regions: set[str] = set(matches.keys())
    assigned_endpoints: set[int] = set(int(v["endpoint_node"]) for v in matches.values())
    for _, dist_to_center, _, name, region, ep_i in sorted(candidate_pairs, key=lambda item: (item[0], item[1], item[2], item[3], item[5])):
        if name in assigned_regions or ep_i in assigned_endpoints:
            continue
        diameter_eq = max(float(region.get("diameter_eq", 0.0)), 1e-3)
        conf = float(clamp(1.0 - dist_to_center / (1.25 * diameter_eq + EPS), 0.0, 1.0))
        matches[str(name)] = {
            "face_id": int(region["face_id"]),
            "name": str(region["name"]),
            "terminal_type": str(region["terminal_type"]),
            "endpoint_node": int(ep_i),
            "endpoint_xyz": np.asarray(pts[int(ep_i)], dtype=float),
            "centroid": np.asarray(region["centroid"], dtype=float),
            "cap_id": int(region.get("cap_id", 0)),
            "distance": float(dist_to_center),
            "confidence": float(conf),
        }
        assigned_regions.add(str(name))
        assigned_endpoints.add(int(ep_i))

    for region in outlet_order:
        if str(region["name"]) in matches:
            continue
        work_warnings.append(
            f"W_MODEL_FACE_OUTLET_ENDPOINT_MAP_FAILED: failed to map outlet face '{str(region['name'])}' (face id {int(region['face_id'])}) to a unique centerline endpoint."
        )
    return matches


def path_segment_between_nodes(path: List[int], start_node: int, end_node: int) -> List[int]:
    if not path:
        return []
    try:
        i0 = int(path.index(int(start_node)))
        i1 = int(path.index(int(end_node)))
    except ValueError:
        return []
    if i0 > i1:
        i0, i1 = i1, i0
    return [int(n) for n in path[i0 : i1 + 1]]


def deepest_common_node_across_paths(paths: List[List[int]], dist: Dict[int, float]) -> Optional[int]:
    valid_paths = [list(path) for path in paths if path]
    if not valid_paths:
        return None
    common = set(int(n) for n in valid_paths[0])
    for path in valid_paths[1:]:
        common.intersection_update(int(n) for n in path)
    if not common:
        return None
    return max((int(n) for n in common), key=lambda n: float(dist.get(int(n), float("-inf"))))


def compute_rooted_descendant_terminal_sets(
    child_map: Dict[int, List[int]],
    root: int,
    terminal_node_to_names: Dict[int, List[str]],
) -> Dict[int, set[str]]:
    memo: Dict[int, set[str]] = {}

    def visit(node: int) -> set[str]:
        node_i = int(node)
        if node_i in memo:
            return set(memo[node_i])

        names = {str(v) for v in terminal_node_to_names.get(node_i, [])}
        for child in child_map.get(node_i, []):
            names.update(visit(int(child)))
        memo[node_i] = set(names)
        return set(names)

    visit(int(root))
    return {int(k): {str(v) for v in vals} for k, vals in memo.items()}


def trim_prelabeled_leaf_branch_paths(
    branch_paths_raw: Dict[str, List[int]],
    branch_terminals: Dict[str, List[str]],
    branch_parent_names: Dict[str, Optional[str]],
    dist: Dict[int, float],
) -> Tuple[Dict[str, List[int]], Dict[str, Optional[int]], Dict[str, str]]:
    trimmed_paths: Dict[str, List[int]] = {}
    ownership_start_nodes: Dict[str, Optional[int]] = {}
    ownership_modes: Dict[str, str] = {}

    for label_name, raw_nodes in branch_paths_raw.items():
        nodes = [int(n) for n in raw_nodes]
        if len(nodes) < 2:
            trimmed_paths[str(label_name)] = list(nodes)
            ownership_start_nodes[str(label_name)] = (None if not nodes else int(nodes[0]))
            ownership_modes[str(label_name)] = "short_or_empty"
            continue

        parent_name = branch_parent_names.get(str(label_name))
        terminal_names = [str(v) for v in branch_terminals.get(str(label_name), [])]
        trimmed_nodes = list(nodes)
        mode = "explicit_interval"

        if parent_name is not None and len(terminal_names) == 1:
            sibling_paths: List[List[int]] = []
            for other_name, other_parent in branch_parent_names.items():
                if str(other_name) == str(label_name) or str(other_parent) != str(parent_name):
                    continue
                other_nodes = [int(n) for n in branch_paths_raw.get(str(other_name), [])]
                if len(other_nodes) >= 2:
                    sibling_paths.append(other_nodes)

            shared_nodes = [
                int(common)
                for common in (
                    deepest_common_node(nodes, sibling_nodes, dist) for sibling_nodes in sibling_paths
                )
                if common is not None and int(common) in nodes
            ]
            if shared_nodes:
                unique_start = max(shared_nodes, key=lambda n: float(dist.get(int(n), float("-inf"))))
                idx = int(nodes.index(int(unique_start)))
                if idx < len(nodes) - 1:
                    trimmed_nodes = [int(n) for n in nodes[idx:]]
                    mode = "sibling_divergence"

        trimmed_paths[str(label_name)] = [int(n) for n in trimmed_nodes]
        ownership_start_nodes[str(label_name)] = (None if not trimmed_nodes else int(trimmed_nodes[0]))
        ownership_modes[str(label_name)] = str(mode)

    return trimmed_paths, ownership_start_nodes, ownership_modes


def assign_prelabeled_branch_ownership_edges(
    prev: Dict[int, int],
    topology: Dict[str, Any],
) -> Tuple[Dict[str, List[List[int]]], Dict[str, int], Dict[int, List[str]]]:
    inlet_node = int(topology.get("inlet_node", -1))
    branch_paths = {
        str(name): [int(n) for n in nodes]
        for name, nodes in dict(topology.get("branch_paths", {})).items()
        if len(nodes) >= 2
    }
    branch_terminals = {
        str(name): {str(v) for v in vals}
        for name, vals in dict(topology.get("branch_terminals", {})).items()
    }
    aorta_name = LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]
    terminal_matches = dict(topology.get("terminal_matches", {}))

    outlet_terminal_names = sorted(
        str(name)
        for name, match in terminal_matches.items()
        if str(name) != "abdominal_aorta_inlet" and int(match.get("endpoint_node", -1)) >= 0
    )
    if aorta_name not in branch_terminals:
        branch_terminals[aorta_name] = set(outlet_terminal_names)
    else:
        branch_terminals[aorta_name].update(str(v) for v in outlet_terminal_names)

    terminal_node_to_names: Dict[int, List[str]] = {}
    for name, match in terminal_matches.items():
        if str(name) == "abdominal_aorta_inlet":
            continue
        endpoint_node = int(match.get("endpoint_node", -1))
        if endpoint_node < 0:
            continue
        terminal_node_to_names.setdefault(int(endpoint_node), []).append(str(name))

    child_map = build_rooted_child_map(prev)
    descendant_terminals = compute_rooted_descendant_terminal_sets(child_map, inlet_node, terminal_node_to_names)
    branch_path_edges = {str(name): path_edge_keys(nodes) for name, nodes in branch_paths.items()}
    branch_priority = {
        str(name): int(LABEL_PRIORITY_ORDER.get(LABEL_NAME_TO_ID.get(str(name), LABEL_OTHER), 999))
        for name in set(branch_paths.keys()) | set(branch_terminals.keys())
    }

    edge_owner: Dict[Tuple[int, int], str] = {}
    owned_edges_by_name: Dict[str, set[Tuple[int, int]]] = {
        str(name): set() for name in set(branch_paths.keys()) | set(branch_terminals.keys())
    }
    owned_edges_by_name.setdefault(LABEL_ID_TO_NAME[LABEL_OTHER], set())

    for child, parent in prev.items():
        edge = edge_key(int(parent), int(child))
        downstream = {str(v) for v in descendant_terminals.get(int(child), set())}
        if not downstream:
            owner_name = LABEL_ID_TO_NAME[LABEL_OTHER]
        else:
            coverage_candidates = [
                str(name)
                for name, terminals in branch_terminals.items()
                if downstream.issubset(set(str(v) for v in terminals))
            ]
            if coverage_candidates:
                owner_name = min(
                    coverage_candidates,
                    key=lambda name: (
                        len(branch_terminals.get(str(name), set())),
                        0 if edge in branch_path_edges.get(str(name), set()) else 1,
                        branch_priority.get(str(name), 999),
                        str(name),
                    ),
                )
            else:
                fallback_candidates = [
                    str(name)
                    for name, terminals in branch_terminals.items()
                    if set(str(v) for v in terminals).intersection(downstream)
                    and (str(name) == aorta_name or len(terminals) > 1)
                ]
                if fallback_candidates:
                    owner_name = min(
                        fallback_candidates,
                        key=lambda name: (
                            0 if str(name) == aorta_name else 1,
                            -len(branch_terminals.get(str(name), set())),
                            branch_priority.get(str(name), 999),
                            str(name),
                        ),
                    )
                else:
                    owner_name = LABEL_ID_TO_NAME[LABEL_OTHER]

        edge_owner[edge] = str(owner_name)
        owned_edges_by_name.setdefault(str(owner_name), set()).add(edge)

    branch_owned_edges = {
        str(name): [[int(a), int(b)] for a, b in sorted(edges)]
        for name, edges in owned_edges_by_name.items()
        if edges
    }
    branch_owned_edge_counts = {str(name): int(len(edges)) for name, edges in owned_edges_by_name.items() if edges}
    descendant_terminals_json = {
        int(node): sorted(str(v) for v in vals)
        for node, vals in descendant_terminals.items()
        if vals
    }
    return branch_owned_edges, branch_owned_edge_counts, descendant_terminals_json


def infer_prelabeled_branch_topology(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    inlet_node: int,
    terminal_matches: Dict[str, Dict[str, Any]],
    dist: Dict[int, float],
    prev: Dict[int, int],
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    work_warnings = warnings if warnings is not None else []
    terminal_paths: Dict[str, List[int]] = {}
    for name, match in terminal_matches.items():
        ep = int(match.get("endpoint_node", -1))
        if ep < 0:
            continue
        path = path_to_root(prev, inlet_node, ep)
        if not path:
            work_warnings.append(f"W_PRELABELED_PATH_EMPTY: failed to recover inlet-rooted path for terminal '{name}'.")
            continue
        terminal_paths[str(name)] = [int(n) for n in path]

    right_paths = [terminal_paths[name] for name in PRELABELED_RIGHT_ILIAC_OUTLET_NAMES if name in terminal_paths]
    left_paths = [terminal_paths[name] for name in PRELABELED_LEFT_ILIAC_OUTLET_NAMES if name in terminal_paths]
    if not right_paths or not left_paths:
        raise RuntimeError("Pre-labeled workflow requires both right and left iliac outlet paths to recover the aortic bifurcation.")

    bif_candidates: List[int] = []
    for r_name in PRELABELED_RIGHT_ILIAC_OUTLET_NAMES:
        for l_name in PRELABELED_LEFT_ILIAC_OUTLET_NAMES:
            if r_name not in terminal_paths or l_name not in terminal_paths:
                continue
            bif = deepest_common_node(terminal_paths[r_name], terminal_paths[l_name], dist)
            if bif is not None:
                bif_candidates.append(int(bif))
    if not bif_candidates:
        raise RuntimeError("Failed to recover the aortic bifurcation from the pre-labeled iliac outlet topology.")
    bif_node = max((int(n) for n in bif_candidates), key=lambda n: float(dist.get(int(n), float("-inf"))))

    trunk_path = path_to_root(prev, inlet_node, bif_node)
    if not trunk_path:
        raise RuntimeError("Failed to reconstruct the abdominal aorta trunk path from inlet to bifurcation.")
    trunk_set = set(int(n) for n in trunk_path)

    def _side_split(names: Tuple[str, str], fallback_node: int) -> int:
        available = [terminal_paths[name] for name in names if name in terminal_paths]
        if len(available) >= 2:
            split = deepest_common_node_across_paths(available, dist)
            if split is not None:
                return int(split)
        return int(fallback_node)

    right_split_node = _side_split(PRELABELED_RIGHT_ILIAC_OUTLET_NAMES, bif_node)
    left_split_node = _side_split(PRELABELED_LEFT_ILIAC_OUTLET_NAMES, bif_node)

    branch_parent_names: Dict[str, Optional[str]] = {
        LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]: None,
        LABEL_ID_TO_NAME[LABEL_CELIAC_TRUNK]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_SUPERIOR_MESENTERIC_ARTERY]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_INFERIOR_MESENTERIC_ARTERY]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_LEFT_RENAL]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC],
        LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC],
        LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC],
        LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC],
    }

    branch_paths_raw: Dict[str, List[int]] = {LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]: [int(n) for n in trunk_path]}
    branch_takeoffs: Dict[str, Optional[int]] = {LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]: None}
    branch_terminals: Dict[str, List[str]] = {
        LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]: sorted(
            str(name) for name in terminal_paths.keys() if str(name) != "abdominal_aorta_inlet"
        )
    }

    def _add_branch(label_name: str, path_nodes: List[int], terminal_names: List[str], takeoff_node: Optional[int]) -> None:
        if len(path_nodes) < 2:
            work_warnings.append(f"W_PRELABELED_BRANCH_SHORT: branch '{label_name}' was too short to preserve explicitly.")
            return
        branch_paths_raw[str(label_name)] = [int(n) for n in path_nodes]
        branch_takeoffs[str(label_name)] = (None if takeoff_node is None else int(takeoff_node))
        branch_terminals[str(label_name)] = [str(v) for v in terminal_names]

    def _path_for(name: str) -> List[int]:
        return list(terminal_paths.get(str(name), []))

    def _segment_from_trunk(name: str) -> Tuple[List[int], Optional[int]]:
        path = _path_for(name)
        if not path:
            return [], None
        takeoff = deepest_common_node(path, trunk_path, dist)
        if takeoff is None:
            return [], None
        return path_segment_between_nodes(path, int(takeoff), int(path[-1])), int(takeoff)

    right_ref_name = next((name for name in PRELABELED_RIGHT_ILIAC_OUTLET_NAMES if name in terminal_paths), None)
    left_ref_name = next((name for name in PRELABELED_LEFT_ILIAC_OUTLET_NAMES if name in terminal_paths), None)
    if right_ref_name is None or left_ref_name is None:
        raise RuntimeError("Failed to recover representative right/left iliac outlet paths from the pre-labeled map.")

    right_ref_path = _path_for(right_ref_name)
    left_ref_path = _path_for(left_ref_name)
    right_common_path = path_segment_between_nodes(right_ref_path, bif_node, right_split_node)
    left_common_path = path_segment_between_nodes(left_ref_path, bif_node, left_split_node)
    _add_branch(LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC], right_common_path, list(PRELABELED_RIGHT_ILIAC_OUTLET_NAMES), bif_node)
    _add_branch(LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC], left_common_path, list(PRELABELED_LEFT_ILIAC_OUTLET_NAMES), bif_node)

    outlet_label_names = {
        "right_internal_iliac": LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC],
        "right_external_iliac": LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC],
        "left_internal_iliac": LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC],
        "left_external_iliac": LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC],
    }
    outlet_split_map = {
        "right_internal_iliac": right_split_node,
        "right_external_iliac": right_split_node,
        "left_internal_iliac": left_split_node,
        "left_external_iliac": left_split_node,
    }
    for outlet_name, label_name in outlet_label_names.items():
        if outlet_name not in terminal_paths:
            continue
        seg = path_segment_between_nodes(terminal_paths[outlet_name], outlet_split_map[outlet_name], terminal_matches[outlet_name]["endpoint_node"])
        _add_branch(label_name, seg, [outlet_name], outlet_split_map[outlet_name])

    direct_branch_takeoffs: Dict[str, Optional[int]] = {}
    for terminal_name, label_id in PRELABELED_DIRECT_AORTIC_BRANCH_LABELS.items():
        seg, takeoff = _segment_from_trunk(terminal_name)
        direct_branch_takeoffs[terminal_name] = takeoff
        if seg:
            _add_branch(LABEL_ID_TO_NAME[label_id], seg, [terminal_name], takeoff)

    celiac_present = [name for name in PRELABELED_CELIAC_OUTLET_NAMES if name in terminal_paths]
    celiac_takeoff_node: Optional[int] = None
    celiac_split_node: Optional[int] = None
    dropped_terminal_faces: List[str] = []
    if celiac_present:
        rep_name = sorted(celiac_present)[0]
        rep_path = _path_for(rep_name)
        if len(celiac_present) >= 2:
            celiac_split_node = deepest_common_node_across_paths([_path_for(name) for name in celiac_present], dist)
        if celiac_split_node is None:
            celiac_split_node = int(rep_path[-1]) if rep_path else None
        if celiac_split_node is not None:
            celiac_split_path = path_to_root(prev, inlet_node, int(celiac_split_node))
            celiac_takeoff_node = deepest_common_node(celiac_split_path, trunk_path, dist)
            celiac_seg = path_segment_between_nodes(celiac_split_path, int(celiac_takeoff_node), int(celiac_split_node)) if celiac_takeoff_node is not None else []
            _add_branch(LABEL_ID_TO_NAME[LABEL_CELIAC_TRUNK], celiac_seg, list(celiac_present), celiac_takeoff_node)
        if len(celiac_present) > 1:
            dropped_terminal_faces.extend(sorted(str(name) for name in celiac_present))

    right_hint_nodes = [int(terminal_matches[name]["endpoint_node"]) for name in PRELABELED_RIGHT_ILIAC_OUTLET_NAMES if name in terminal_matches]
    left_hint_nodes = [int(terminal_matches[name]["endpoint_node"]) for name in PRELABELED_LEFT_ILIAC_OUTLET_NAMES if name in terminal_matches]
    if not right_hint_nodes or not left_hint_nodes:
        raise RuntimeError("Failed to recover right/left iliac endpoint hints for canonical orientation.")

    branch_paths, branch_ownership_start_nodes, branch_ownership_modes = trim_prelabeled_leaf_branch_paths(
        branch_paths_raw=branch_paths_raw,
        branch_terminals=branch_terminals,
        branch_parent_names=branch_parent_names,
        dist=dist,
    )

    branch_parent_attachment_nodes = {
        str(k): (None if v is None else int(v)) for k, v in branch_takeoffs.items()
    }
    branch_takeoffs = {
        str(k): (
            None
            if branch_ownership_start_nodes.get(str(k)) is None
            else int(branch_ownership_start_nodes.get(str(k)))
        )
        for k in branch_paths.keys()
    }
    direct_branch_parent_attachment_nodes = {
        str(k): (None if v is None else int(v)) for k, v in direct_branch_takeoffs.items()
    }
    direct_branch_takeoffs = {}
    for terminal_name, label_id in PRELABELED_DIRECT_AORTIC_BRANCH_LABELS.items():
        label_name = LABEL_ID_TO_NAME[label_id]
        start_node = branch_ownership_start_nodes.get(str(label_name))
        direct_branch_takeoffs[str(terminal_name)] = (None if start_node is None else int(start_node))

    celiac_parent_attachment_node = celiac_takeoff_node
    celiac_takeoff_node = branch_takeoffs.get(LABEL_ID_TO_NAME[LABEL_CELIAC_TRUNK], celiac_takeoff_node)

    topology_out: Dict[str, Any] = {
        "inlet_node": int(inlet_node),
        "bifurcation_node": int(bif_node),
        "trunk_path": [int(n) for n in trunk_path],
        "trunk_node_set": set(int(n) for n in trunk_path),
        "right_common_iliac_split_node": int(right_split_node),
        "left_common_iliac_split_node": int(left_split_node),
        "direct_branch_takeoffs": {str(k): (None if v is None else int(v)) for k, v in direct_branch_takeoffs.items()},
        "direct_branch_parent_attachment_nodes": {
            str(k): (None if v is None else int(v)) for k, v in direct_branch_parent_attachment_nodes.items()
        },
        "celiac_takeoff_node": (None if celiac_takeoff_node is None else int(celiac_takeoff_node)),
        "celiac_parent_attachment_node": (
            None if celiac_parent_attachment_node is None else int(celiac_parent_attachment_node)
        ),
        "celiac_split_node": (None if celiac_split_node is None else int(celiac_split_node)),
        "branch_paths": {str(k): [int(n) for n in v] for k, v in branch_paths.items()},
        "branch_paths_raw": {str(k): [int(n) for n in v] for k, v in branch_paths_raw.items()},
        "branch_takeoffs": {str(k): (None if v is None else int(v)) for k, v in branch_takeoffs.items()},
        "branch_parent_attachment_nodes": {
            str(k): (None if v is None else int(v)) for k, v in branch_parent_attachment_nodes.items()
        },
        "branch_terminals": {str(k): [str(v) for v in vals] for k, vals in branch_terminals.items()},
        "branch_parent_names": {str(k): (None if v is None else str(v)) for k, v in branch_parent_names.items()},
        "branch_ownership_start_nodes": {
            str(k): (None if v is None else int(v)) for k, v in branch_ownership_start_nodes.items()
        },
        "branch_ownership_modes": {str(k): str(v) for k, v in branch_ownership_modes.items()},
        "terminal_paths": {str(k): [int(n) for n in v] for k, v in terminal_paths.items()},
        "terminal_matches": {
            str(k): {
                **{kk: vv for kk, vv in v.items() if kk not in ("endpoint_xyz", "centroid")},
                "endpoint_xyz": [float(x) for x in np.asarray(v.get("endpoint_xyz", np.zeros((3,), dtype=float)), dtype=float).reshape(3)],
                "centroid": [float(x) for x in np.asarray(v.get("centroid", np.zeros((3,), dtype=float)), dtype=float).reshape(3)],
            }
            for k, v in terminal_matches.items()
        },
        "right_orientation_hint_nodes": [int(v) for v in right_hint_nodes],
        "left_orientation_hint_nodes": [int(v) for v in left_hint_nodes],
        "dropped_terminal_faces": [str(v) for v in dropped_terminal_faces],
    }

    branch_owned_edges, branch_owned_edge_counts, descendant_terminals_json = assign_prelabeled_branch_ownership_edges(
        prev=prev,
        topology=topology_out,
    )
    topology_out["branch_owned_edges"] = {str(k): [[int(a), int(b)] for a, b in v] for k, v in branch_owned_edges.items()}
    topology_out["branch_owned_edge_counts"] = {str(k): int(v) for k, v in branch_owned_edge_counts.items()}
    topology_out["node_descendant_terminals"] = {
        str(k): [str(v) for v in vals] for k, vals in descendant_terminals_json.items()
    }

    return topology_out


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


def build_adjacency_from_edge_keys(
    adjacency_full: Dict[int, Dict[int, float]],
    edges: set[Tuple[int, int]],
) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {}
    for a, b in sorted((edge_key(int(u), int(v)) for u, v in edges)):
        w = adjacency_full.get(int(a), {}).get(int(b))
        if w is None:
            w = adjacency_full.get(int(b), {}).get(int(a))
        if w is None:
            continue
        out.setdefault(int(a), {})[int(b)] = float(w)
        out.setdefault(int(b), {})[int(a)] = float(w)
    return out


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


def build_rooted_child_map(prev: Dict[int, int]) -> Dict[int, List[int]]:
    child_map: Dict[int, List[int]] = {}
    for node, parent in prev.items():
        child_map.setdefault(int(parent), []).append(int(node))
    for parent, children in child_map.items():
        child_map[parent] = sorted(int(child) for child in children)
    return child_map


def collect_rooted_subtree_nodes(child_map: Dict[int, List[int]], start: int) -> set[int]:
    seen: set[int] = set()
    stack = [int(start)]
    while stack:
        node = int(stack.pop())
        if node in seen:
            continue
        seen.add(node)
        for child in reversed(child_map.get(node, [])):
            if int(child) not in seen:
                stack.append(int(child))
    return seen


def summarize_rooted_subtree(
    child_map: Dict[int, List[int]],
    pts: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
    anchor_node: int,
    start: int,
) -> Optional[Dict[str, Any]]:
    anchor_node = int(anchor_node)
    start = int(start)
    anchor_dist = float(dist.get(anchor_node, float("nan")))
    if not math.isfinite(anchor_dist):
        return None

    subtree_nodes = collect_rooted_subtree_nodes(child_map, start)
    if not subtree_nodes:
        return None

    endpoints = sorted(int(n) for n in subtree_nodes if len(child_map.get(int(n), [])) == 0)
    if not endpoints:
        endpoints = [start]

    representative_endpoint = max(
        endpoints,
        key=lambda n: (float(dist.get(int(n), float("-inf"))), int(n)),
    )

    subtree_max_dist = max(float(dist.get(int(n), anchor_dist)) for n in endpoints)
    subtree_max_length = max(0.0, subtree_max_dist - anchor_dist)
    subtree_total_length = 0.0
    for node in subtree_nodes:
        parent = prev.get(int(node))
        if parent is None:
            continue
        parent_dist = float(dist.get(int(parent), float("nan")))
        node_dist = float(dist.get(int(node), float("nan")))
        if math.isfinite(parent_dist) and math.isfinite(node_dist) and node_dist >= parent_dist:
            subtree_total_length += node_dist - parent_dist

    spatial_reach = 0.0
    for node in endpoints:
        spatial_reach = max(spatial_reach, float(np.linalg.norm(pts[int(node)] - pts[anchor_node])))

    return {
        "anchor_node": int(anchor_node),
        "start": int(start),
        "nodes": set(int(n) for n in subtree_nodes),
        "endpoints": [int(n) for n in endpoints],
        "representative_endpoint": int(representative_endpoint),
        "subtree_max_length": float(subtree_max_length),
        "subtree_total_length": float(subtree_total_length),
        "endpoint_count": int(len(endpoints)),
        "node_count": int(len(subtree_nodes)),
        "spatial_reach": float(spatial_reach),
    }


def build_raw_stem_path(
    child_map: Dict[int, List[int]],
    takeoff: int,
    stem_start: int,
) -> Dict[str, Any]:
    takeoff = int(takeoff)
    stem_start = int(stem_start)
    stem_path = [takeoff, stem_start]
    cur = stem_start
    first_split: Optional[int] = None
    while True:
        children = [int(child) for child in child_map.get(cur, [])]
        if len(children) != 1:
            if len(children) >= 2:
                first_split = int(cur)
            break
        nxt = int(children[0])
        stem_path.append(nxt)
        cur = nxt
    return {
        "raw_stem_path": [int(n) for n in stem_path],
        "raw_stem_terminal": int(stem_path[-1]),
        "raw_first_split": (int(first_split) if first_split is not None else None),
    }


def rooted_subtree_rank_key(summary: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
    return (
        float(summary.get("subtree_max_length", 0.0)),
        float(summary.get("subtree_total_length", 0.0)),
        float(summary.get("spatial_reach", 0.0)),
        int(summary.get("endpoint_count", 0)),
        -int(summary.get("start", -1)),
    )


def is_substantial_named_stem_child(
    summary: Dict[str, Any],
    dominant: Dict[str, Any],
    parent_system: Dict[str, Any],
) -> bool:
    dominant_max = max(float(dominant.get("subtree_max_length", 0.0)), EPS)
    dominant_total = max(float(dominant.get("subtree_total_length", 0.0)), EPS)
    dominant_reach = max(float(dominant.get("spatial_reach", 0.0)), EPS)
    system_max = max(float(parent_system.get("subtree_max_length", 0.0)), EPS)

    ratio_votes = 0
    if float(summary.get("subtree_max_length", 0.0)) >= 0.60 * dominant_max:
        ratio_votes += 1
    if float(summary.get("subtree_total_length", 0.0)) >= 0.60 * dominant_total:
        ratio_votes += 1
    if float(summary.get("spatial_reach", 0.0)) >= 0.60 * dominant_reach:
        ratio_votes += 1
    if int(summary.get("endpoint_count", 0)) >= 2:
        ratio_votes += 1

    absolute_ok = (
        float(summary.get("subtree_max_length", 0.0)) >= max(0.75, 0.20 * system_max)
        and float(summary.get("spatial_reach", 0.0)) >= max(0.50, 0.15 * system_max)
    )
    return bool(absolute_ok and ratio_votes >= 2)


def build_named_stem_path(
    child_map: Dict[int, List[int]],
    pts: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
    takeoff: int,
    stem_start: int,
    parent_system: Dict[str, Any],
) -> Dict[str, Any]:
    takeoff = int(takeoff)
    stem_start = int(stem_start)
    named_path = [takeoff, stem_start]
    cur = stem_start
    first_major_split: Optional[int] = None
    split_summaries: List[Dict[str, Any]] = []
    visited = {takeoff, stem_start}

    while True:
        children = [int(child) for child in child_map.get(cur, [])]
        if not children:
            break
        if len(children) == 1:
            nxt = int(children[0])
            if nxt in visited:
                break
            named_path.append(nxt)
            visited.add(nxt)
            cur = nxt
            continue

        child_summaries = [
            summary
            for summary in (
                summarize_rooted_subtree(child_map, pts, dist, prev, cur, child)
                for child in children
            )
            if summary is not None
        ]
        if not child_summaries:
            break
        child_summaries.sort(key=rooted_subtree_rank_key, reverse=True)
        split_summaries = child_summaries

        dominant = child_summaries[0]
        substantial = [
            summary
            for summary in child_summaries
            if is_substantial_named_stem_child(summary, dominant, parent_system)
        ]
        if len(substantial) >= 2:
            first_major_split = int(cur)
            break

        nxt = int(dominant["start"])
        if nxt in visited:
            break
        named_path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return {
        "named_stem_path": [int(n) for n in named_path],
        "named_stem_terminal": int(named_path[-1]),
        "named_first_major_split": (int(first_major_split) if first_major_split is not None else None),
        "named_split_child_summaries": [
            {
                "start": int(summary["start"]),
                "subtree_max_length": float(summary["subtree_max_length"]),
                "subtree_total_length": float(summary["subtree_total_length"]),
                "endpoint_count": int(summary["endpoint_count"]),
                "spatial_reach": float(summary["spatial_reach"]),
            }
            for summary in split_summaries
        ],
    }


def describe_rooted_child_system(
    child_map: Dict[int, List[int]],
    pts: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
    takeoff: int,
    stem_start: int,
) -> Optional[Dict[str, Any]]:
    # Collapse every descendant leaf that shares the same first off-parent stem.
    takeoff = int(takeoff)
    stem_start = int(stem_start)
    takeoff_dist = float(dist.get(takeoff, float("nan")))
    stem_start_dist = float(dist.get(stem_start, float("nan")))
    if not math.isfinite(takeoff_dist) or not math.isfinite(stem_start_dist):
        return None

    subtree_summary = summarize_rooted_subtree(child_map, pts, dist, prev, takeoff, stem_start)
    if subtree_summary is None:
        return None

    subtree_nodes = set(int(n) for n in subtree_summary["nodes"])
    endpoints = [int(n) for n in subtree_summary["endpoints"]]
    representative_endpoint = int(subtree_summary["representative_endpoint"])

    raw_stem_info = build_raw_stem_path(child_map, takeoff, stem_start)
    named_stem_info = build_named_stem_path(
        child_map,
        pts,
        dist,
        prev,
        takeoff,
        stem_start,
        parent_system=subtree_summary,
    )
    stem_path = [int(n) for n in named_stem_info["named_stem_path"]]
    stem_terminal = int(named_stem_info["named_stem_terminal"])

    local_idx = min(2, len(stem_path) - 1)
    local_vec = pts[stem_path[local_idx]] - pts[takeoff]
    stem_vec = pts[stem_terminal] - pts[takeoff]
    rep_vec = pts[representative_endpoint] - pts[takeoff]
    if np.linalg.norm(local_vec) < EPS:
        local_vec = stem_vec.copy()
    if np.linalg.norm(local_vec) < EPS:
        local_vec = rep_vec.copy()
    if np.linalg.norm(stem_vec) < EPS:
        stem_vec = rep_vec.copy()

    center_source = endpoints if endpoints else sorted(subtree_nodes)
    subtree_center = np.mean(pts[np.asarray(center_source, dtype=int)], axis=0)
    direction_vec = subtree_center - pts[takeoff]
    if np.linalg.norm(direction_vec) < EPS:
        direction_vec = rep_vec.copy()

    subtree_max_length = float(subtree_summary["subtree_max_length"])
    subtree_total_length = float(subtree_summary["subtree_total_length"])
    raw_stem_terminal = int(raw_stem_info["raw_stem_terminal"])
    raw_stem_length = float(max(0.0, float(dist.get(raw_stem_terminal, takeoff_dist)) - takeoff_dist))
    named_stem_length = float(max(0.0, float(dist.get(stem_terminal, takeoff_dist)) - takeoff_dist))

    return {
        "takeoff": int(takeoff),
        "takeoff_dist": float(takeoff_dist),
        "stem_start": int(stem_start),
        "raw_stem_path": [int(n) for n in raw_stem_info["raw_stem_path"]],
        "raw_stem_terminal": int(raw_stem_terminal),
        "raw_first_split": raw_stem_info["raw_first_split"],
        "raw_stem_length": float(raw_stem_length),
        "named_stem_path": [int(n) for n in named_stem_info["named_stem_path"]],
        "named_stem_terminal": int(stem_terminal),
        "named_first_major_split": named_stem_info["named_first_major_split"],
        "named_stem_length": float(named_stem_length),
        "named_split_child_summaries": list(named_stem_info["named_split_child_summaries"]),
        "stem_path": [int(n) for n in stem_path],
        "stem_terminal": int(stem_terminal),
        "first_split": raw_stem_info["raw_first_split"],
        "descendant_nodes": set(int(n) for n in subtree_nodes),
        "endpoints": [int(n) for n in endpoints],
        "representative_endpoint": int(representative_endpoint),
        "representative_dist": float(dist.get(int(representative_endpoint), takeoff_dist)),
        "stem_length": float(named_stem_length),
        "subtree_max_length": float(subtree_max_length),
        "subtree_total_length": float(subtree_total_length),
        "endpoint_count": int(len(endpoints)),
        "node_count": int(len(subtree_nodes)),
        "subtree_spatial_reach": float(subtree_summary["spatial_reach"]),
        "subtree_center": np.asarray(subtree_center, dtype=float),
        "local_vector": np.asarray(local_vec, dtype=float),
        "stem_vector": np.asarray(stem_vec, dtype=float),
        "representative_vector": np.asarray(rep_vec, dtype=float),
        "direction_vector": np.asarray(direction_vec, dtype=float),
    }


def build_direct_child_systems_for_parent_path(
    parent_path: List[int],
    pts: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
) -> List[Dict[str, Any]]:
    if not parent_path:
        return []

    child_map = build_rooted_child_map(prev)
    systems: List[Dict[str, Any]] = []
    for idx, takeoff in enumerate(parent_path):
        takeoff = int(takeoff)
        continue_child = int(parent_path[idx + 1]) if idx + 1 < len(parent_path) else None
        for stem_start in child_map.get(takeoff, []):
            stem_start = int(stem_start)
            if continue_child is not None and stem_start == continue_child:
                continue
            system = describe_rooted_child_system(child_map, pts, dist, prev, takeoff, stem_start)
            if system is not None:
                systems.append(system)

    systems.sort(key=lambda item: (float(item["takeoff_dist"]), int(item["takeoff"]), int(item["stem_start"])))
    return systems


def rooted_child_system_key(system: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if system is None:
        return (-1, -1)
    return int(system.get("takeoff", -1)), int(system.get("stem_start", -1))


def rooted_child_system_node_set(system: Optional[Dict[str, Any]], include_takeoff: bool = True) -> set[int]:
    if system is None:
        return set()
    nodes = set(int(n) for n in system.get("descendant_nodes", set()))
    if include_takeoff and system.get("takeoff") is not None:
        nodes.add(int(system["takeoff"]))
    return nodes


def find_rooted_child_system_by_key(
    systems: List[Dict[str, Any]],
    target: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    target_key = rooted_child_system_key(target)
    if target_key == (-1, -1):
        return None
    for system in systems:
        if rooted_child_system_key(system) == target_key:
            return system
    return None


def find_rooted_child_system_for_endpoint(
    child_map: Dict[int, List[int]],
    pts: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
    takeoff: int,
    endpoint: int,
) -> Optional[Dict[str, Any]]:
    takeoff = int(takeoff)
    endpoint = int(endpoint)
    matches: List[Dict[str, Any]] = []
    for stem_start in sorted(int(child) for child in child_map.get(takeoff, [])):
        system = describe_rooted_child_system(child_map, pts, dist, prev, takeoff, stem_start)
        if system is None:
            continue
        if endpoint in rooted_child_system_node_set(system, include_takeoff=False):
            matches.append(system)
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            0 if endpoint in set(int(n) for n in item.get("endpoints", [])) else 1,
            float(item.get("takeoff_dist", float("inf"))),
            int(item.get("stem_start", -1)),
        )
    )
    return matches[0]


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
) -> Tuple[Optional[int], Optional[int], Optional[int], float, Dict[int, float], Dict[int, int], Dict[str, Any]]:
    """
    Identify the distal aortic bifurcation into the main iliac systems using centerline topology.

    Returns: (bif_node, ep_a, ep_b, confidence, dist_from_inlet, prev, selection)
    where selection preserves the paired rooted child systems that were chosen.
    """
    dist, prev = dijkstra(adjacency, inlet_node)
    deg = node_degrees(adjacency)
    endpoints = [n for n, d in deg.items() if d == 1 and n != inlet_node and n in dist]
    if len(endpoints) < 2:
        warnings.append("W_BIF_NOT_ENOUGH_ENDPOINTS: need >=2 distal endpoints for iliac pair inference.")
        return None, None, None, 0.0, dist, prev, {"system_a": None, "system_b": None, "source": "unresolved"}

    endpoints_sorted = sorted(endpoints, key=lambda n: dist.get(n, -1.0), reverse=True)

    bbox_min = np.min(pts[list(adjacency.keys())], axis=0)
    bbox_max = np.max(pts[list(adjacency.keys())], axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    max_dist = float(max(dist.get(n, 0.0) for n in endpoints_sorted)) if endpoints_sorted else 1.0
    axis = unit(axis_si) if np.linalg.norm(axis_si) > EPS else unit(bbox_max - bbox_min)
    if np.linalg.norm(axis) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    child_map = build_rooted_child_map(prev)
    best = None
    best_score = -1e18

    for bif_node in sorted(int(n) for n in child_map.keys() if int(n) in dist):
        # Candidate aortic bifurcations are evaluated as pairs of rooted child systems,
        # not pairs of distal leaves.
        rooted_children = [int(child) for child in child_map.get(int(bif_node), []) if int(child) in dist]
        if len(rooted_children) < 2:
            continue

        systems: List[Dict[str, Any]] = []
        for stem_start in rooted_children:
            system = describe_rooted_child_system(child_map, pts, dist, prev, bif_node, stem_start)
            if system is not None:
                systems.append(system)
        if len(systems) < 2:
            continue

        depth = float(dist.get(bif_node, 0.0))
        depth_norm = depth / (max_dist + EPS)
        proximal_penalty = clamp((0.45 - depth_norm) / 0.45, 0.0, 1.0)

        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                sys_a = systems[i]
                sys_b = systems[j]

                len_a = float(sys_a["subtree_max_length"])
                len_b = float(sys_b["subtree_max_length"])
                if len_a <= 0.0 or len_b <= 0.0:
                    continue

                symmetry = 1.0 - abs(len_a - len_b) / (len_a + len_b + EPS)

                dvec = np.asarray(sys_a["subtree_center"], dtype=float) - np.asarray(sys_b["subtree_center"], dtype=float)
                lateral = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
                lateral_norm = lateral / (diag + EPS)

                va = unit(np.asarray(sys_a["direction_vector"], dtype=float))
                vb = unit(np.asarray(sys_b["direction_vector"], dtype=float))
                divergence = float(clamp((1.0 - float(np.dot(va, vb))) / 2.0, 0.0, 1.0))

                min_tail_norm = min(len_a, len_b) / (max_dist + EPS)
                score = (
                    2.00 * depth_norm
                    + 1.15 * symmetry
                    + 1.60 * lateral_norm
                    + 0.55 * min_tail_norm
                    + 0.35 * divergence
                    - 0.90 * proximal_penalty
                )

                if score > best_score:
                    best_score = score
                    best = (int(bif_node), sys_a, sys_b, depth_norm, symmetry, lateral_norm, divergence)

    if best is None:
        warnings.append(
            "W_BIF_DIRECT_CHILD_SEARCH_FAILED: could not identify paired distal child systems; falling back to two farthest endpoints."
        )
        a = endpoints_sorted[0]
        b = endpoints_sorted[1]
        pa = path_to_root(prev, inlet_node, a)
        pb = path_to_root(prev, inlet_node, b)
        lca = deepest_common_node(pa, pb, dist)
        if lca is None:
            warnings.append("W_BIF_FALLBACK_FAILED: no common ancestor found; bifurcation unresolved.")
            return None, a, b, 0.0, dist, prev, {"system_a": None, "system_b": None, "source": "fallback_failed"}
        fallback_sys_a = find_rooted_child_system_for_endpoint(child_map, pts, dist, prev, int(lca), int(a))
        fallback_sys_b = find_rooted_child_system_for_endpoint(child_map, pts, dist, prev, int(lca), int(b))
        if fallback_sys_a is None or fallback_sys_b is None:
            warnings.append(
                "W_BIF_FALLBACK_SYSTEM_RECOVERY_FAILED: bifurcation fallback found distal endpoints but could not fully recover rooted iliac systems."
            )
        conf = 0.20
        return int(lca), int(a), int(b), conf, dist, prev, {
            "bif_node": int(lca),
            "system_a": fallback_sys_a,
            "system_b": fallback_sys_b,
            "source": "endpoint_pair_fallback",
        }

    bif_node, sys_a, sys_b, depth_norm, symmetry, lateral_norm, divergence = best
    ep_a = int(sys_a["representative_endpoint"])
    ep_b = int(sys_b["representative_endpoint"])

    term_support = 0.0
    if terminations:
        term_a_idx, term_b_idx, term_pair_conf = choose_distal_iliac_termination_pair(
            terminations,
            inlet_term,
            axis_si,
            warnings,
        )
        if term_a_idx is not None and term_b_idx is not None:
            term_ep_a, term_ep_a_conf = choose_centerline_endpoint_for_termination(endpoints, pts, terminations[term_a_idx])
            term_ep_b, term_ep_b_conf = choose_centerline_endpoint_for_termination(
                endpoints,
                pts,
                terminations[term_b_idx],
                exclude=({int(term_ep_a)} if term_ep_a is not None else set()),
            )
            if term_ep_a is not None and term_ep_b is not None and term_ep_a != term_ep_b:
                sys_a_endpoints = set(int(n) for n in sys_a["endpoints"])
                sys_b_endpoints = set(int(n) for n in sys_b["endpoints"])
                matched = (term_ep_a in sys_a_endpoints and term_ep_b in sys_b_endpoints) or (
                    term_ep_a in sys_b_endpoints and term_ep_b in sys_a_endpoints
                )
                partial = (term_ep_a in sys_a_endpoints or term_ep_a in sys_b_endpoints or term_ep_b in sys_a_endpoints or term_ep_b in sys_b_endpoints)
                if matched:
                    term_support = 0.08 * clamp(term_pair_conf, 0.0, 1.0) * min(term_ep_a_conf, term_ep_b_conf)
                elif partial:
                    warnings.append(
                        "W_BIF_TERM_TOPOLOGY_PARTIAL_MATCH: distal surface terminations only partially matched the topology-selected iliac systems."
                    )
                elif term_pair_conf >= 0.55:
                    warnings.append(
                        "W_BIF_TERM_TOPOLOGY_MISMATCH: distal surface terminations disagreed with topology-selected iliac systems; keeping topology result."
                    )
            else:
                warnings.append(
                    "W_BIF_TERM_ENDPOINT_MAP_FAILED: distal surface terminations could not be mapped cleanly to distinct centerline endpoints."
                )

    conf = float(
        clamp(
            0.22
            + 0.32 * depth_norm
            + 0.22 * symmetry
            + 0.22 * clamp(lateral_norm * 2.0, 0.0, 1.0)
            + 0.07 * divergence
            + term_support,
            0.0,
            1.0,
        )
    )
    if conf < 0.60:
        warnings.append(
            f"W_BIF_LOWER_CONF: bifurcation confidence={conf:.3f} (sym={symmetry:.3f}, depthN={depth_norm:.3f}, latN={lateral_norm:.3f})."
        )
    return int(bif_node), int(ep_a), int(ep_b), conf, dist, prev, {
        "bif_node": int(bif_node),
        "system_a": sys_a,
        "system_b": sys_b,
        "source": "paired_direct_child_systems",
    }


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
    excluded_system_nodes: Optional[set[int]] = None,
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
    excluded_nodes = set(int(n) for n in (excluded_system_nodes or set()))

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
        if excluded_nodes and component_nodes.intersection(excluded_nodes):
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
        attachment_gap = abs(float(takeoff_dist) - float(trunk_takeoff_dist))
        attachment_score = clamp(1.0 - attachment_gap / (0.08 * trunk_len + EPS), 0.0, 1.0)
        score_pos = clamp(1.0 - abs(s_rel - 0.34) / 0.24, 0.0, 1.0)
        score_horiz = clamp((horizontality - 0.35) / 0.55, 0.0, 1.0)
        score_side = clamp((0.84 - local_vertical) / 0.60, 0.0, 1.0)
        score_reach = clamp(reach_xy / max(1.0, 0.12 * trunk_len), 0.0, 1.0)
        score_len = clamp((len_norm - 0.04) / 0.18, 0.0, 1.0) * clamp((0.90 - len_norm) / 0.45, 0.0, 1.0)
        cranial_penalty = 0.20 * clamp((0.20 - s_rel) / 0.12, 0.0, 1.0)
        caudal_penalty = 0.10 * clamp((s_rel - 0.62) / 0.18, 0.0, 1.0)
        visceral_penalty = 0.18 * clamp((0.22 - s_rel) / 0.14, 0.0, 1.0) * clamp((local_vertical - 0.25) / 0.40, 0.0, 1.0)
        attachment_penalty = 0.10 * clamp((0.65 - attachment_score) / 0.65, 0.0, 1.0)
        score = float(
            clamp(
                0.26 * score_horiz
                + 0.20 * score_reach
                + 0.16 * score_side
                + 0.16 * score_pos
                + 0.10 * score_len
                + 0.07 * attachment_score
                + 0.05 * float(rep["representative_score"])
                - cranial_penalty
                - caudal_penalty
                - visceral_penalty
                - attachment_penalty,
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
                "attachment_score": float(attachment_score),
                "renal_zone_score": float(score_pos),
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
            mean_s_rel = float(0.5 * (a["s_rel"] + b["s_rel"]))
            takeoff_zone = float(clamp(1.0 - abs(mean_s_rel - 0.34) / 0.20, 0.0, 1.0))
            attachment_score = float(clamp(min(a.get("attachment_score", 0.0), b.get("attachment_score", 0.0)), 0.0, 1.0))
            cranial_pair_penalty = 0.18 * clamp((0.20 - min(a["s_rel"], b["s_rel"])) / 0.12, 0.0, 1.0)
            caudal_pair_penalty = 0.10 * clamp((max(a["s_rel"], b["s_rel"]) - 0.62) / 0.18, 0.0, 1.0)
            score = float(
                clamp(
                    0.30 * geometry_score
                    + 0.20 * takeoff_sim
                    + 0.15 * horiz_pair
                    + 0.10 * len_sim
                    + 0.10 * (0.5 * (a["score"] + b["score"]))
                    + 0.10 * takeoff_zone
                    + 0.05 * attachment_score
                    - cranial_pair_penalty
                    - caudal_pair_penalty,
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
                    "takeoff_zone_score": float(takeoff_zone),
                    "horizontality_score": float(horiz_pair),
                    "length_similarity": float(len_sim),
                    "axis_span_score": float(axis_span),
                    "attachment_score": float(attachment_score),
                    "mean_takeoff_level": float(mean_s_rel),
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
    excluded_system_nodes: Optional[set[int]] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    excluded = {iliac_ep_a, iliac_ep_b}
    if excluded_endpoints:
        excluded.update(int(x) for x in excluded_endpoints if x is not None)
    excluded_nodes = set(int(n) for n in (excluded_system_nodes or set()))

    trunk_len = float(dist.get(bif_node, 0.0))
    if trunk_len <= 0.0:
        return None, 0.0, 0

    trunk_path = path_to_root(prev, inlet_node, bif_node)
    if not trunk_path:
        return None, 0.0, 0

    systems = build_direct_child_systems_for_parent_path(trunk_path, pts_c, dist, prev)
    vecs: List[np.ndarray] = []
    weights: List[float] = []
    for system in systems:
        if excluded_nodes and rooted_child_system_node_set(system, include_takeoff=False).intersection(excluded_nodes):
            continue
        if excluded.intersection(int(n) for n in system["endpoints"]):
            continue
        takeoff_dist = float(system["takeoff_dist"])
        branch_len = float(system["subtree_max_length"])
        if not math.isfinite(takeoff_dist) or branch_len <= 0.0:
            continue
        s_rel = takeoff_dist / (trunk_len + EPS)
        if s_rel < 0.08 or s_rel > 0.72:
            continue

        if branch_len <= 0.02 * trunk_len:
            continue

        v = np.asarray(system["stem_vector"], dtype=float)
        if np.linalg.norm(v) < EPS:
            v = np.asarray(system["representative_vector"], dtype=float)
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
    excluded_system_nodes: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    exclude = {iliac_ep_a, iliac_ep_b}
    if renal_eps[0] is not None:
        exclude.add(int(renal_eps[0]))
    if renal_eps[1] is not None:
        exclude.add(int(renal_eps[1]))
    excluded_nodes = set(int(n) for n in (excluded_system_nodes or set()))

    trunk_len = float(dist.get(bif_node, 0.0))
    if trunk_len <= 0:
        return []

    trunk_path = path_to_root(prev, inlet_node, bif_node)
    if not trunk_path:
        return []

    systems = build_direct_child_systems_for_parent_path(trunk_path, pts_c, dist, prev)
    cues: List[Dict[str, Any]] = []
    for system in systems:
        if excluded_nodes and rooted_child_system_node_set(system, include_takeoff=False).intersection(excluded_nodes):
            continue
        if exclude.intersection(int(n) for n in system["endpoints"]):
            continue
        takeoff = int(system["takeoff"])
        takeoff_dist = float(system["takeoff_dist"])
        if not math.isfinite(takeoff_dist):
            continue
        s_rel = takeoff_dist / (trunk_len + EPS)
        if s_rel < 0.10 or s_rel > 0.75:
            continue

        v = np.asarray(system["local_vector"], dtype=float)
        if np.linalg.norm(v) < EPS:
            v = np.asarray(system["stem_vector"], dtype=float)
        if np.linalg.norm(v) < EPS:
            v = np.asarray(system["representative_vector"], dtype=float)
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
                "ep": int(system["representative_endpoint"]),
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
    iliac_excluded_system_nodes: Optional[set[int]] = None,
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
        visceral_axis_hint: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pair_axis = unit_xy(np.asarray(renal_pair.get("axis", np.zeros(3, dtype=float)), dtype=float))
        if np.linalg.norm(pair_axis) < EPS:
            return np.asarray(R_in, dtype=float), {
                "rotation_degrees_about_z": 0.0,
                "axis_confidence": 0.0,
                "geometry_score": 0.0,
                "takeoff_similarity_score": 0.0,
                "horizontality_score": 0.0,
                "takeoff_zone_score": 0.0,
                "axis_alignment_with_iliacs": 0.0,
                "visceral_orthogonality": 0.0,
            }

        axis_alignment = 0.0
        if np.linalg.norm(iliac_ref_axis) >= EPS and float(np.dot(pair_axis[:2], iliac_ref_axis[:2])) < 0.0:
            pair_axis *= -1.0
        if np.linalg.norm(iliac_ref_axis) >= EPS:
            axis_alignment = float(clamp(abs(float(np.dot(pair_axis[:2], iliac_ref_axis[:2]))), 0.0, 1.0))

        visceral_orthogonality = 0.0
        visceral_axis_xy = (
            unit_xy(np.asarray(visceral_axis_hint, dtype=float).reshape(3))
            if visceral_axis_hint is not None
            else np.zeros((3,), dtype=float)
        )
        if np.linalg.norm(visceral_axis_xy) >= EPS:
            visceral_orthogonality = float(
                clamp(
                    math.sqrt(max(0.0, 1.0 - float(np.dot(pair_axis[:2], visceral_axis_xy[:2])) ** 2)),
                    0.0,
                    1.0,
                )
            )

        ey_axis = np.array([-pair_axis[1], pair_axis[0], 0.0], dtype=float)
        Q = np.vstack([pair_axis, ey_axis, np.array([0.0, 0.0, 1.0], dtype=float)]).astype(float)
        R_out = (Q @ np.asarray(R_in, dtype=float)).astype(float)
        rotation_deg = float(math.degrees(math.atan2(float(pair_axis[1]), float(pair_axis[0]))))
        return R_out, {
            "rotation_degrees_about_z": float(rotation_deg),
            "axis_confidence": float(clamp(renal_pair.get("axis_confidence", renal_pair.get("confidence", 0.0)), 0.0, 1.0)),
            "geometry_score": float(clamp(renal_pair.get("geometry_score", 0.0), 0.0, 1.0)),
            "takeoff_similarity_score": float(clamp(renal_pair.get("takeoff_similarity", 0.0), 0.0, 1.0)),
            "takeoff_zone_score": float(clamp(renal_pair.get("takeoff_zone_score", 0.0), 0.0, 1.0)),
            "horizontality_score": float(clamp(renal_pair.get("horizontality_score", 0.0), 0.0, 1.0)),
            "axis_alignment_with_iliacs": float(axis_alignment),
            "visceral_orthogonality": float(visceral_orthogonality),
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
        excluded_system_nodes=iliac_excluded_system_nodes,
    )

    default_renal_info = {
        "rotation_degrees_about_z": 0.0,
        "axis_confidence": 0.0,
        "geometry_score": float(clamp(renal_pair.get("geometry_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "takeoff_similarity_score": float(clamp(renal_pair.get("takeoff_similarity", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "takeoff_zone_score": float(clamp(renal_pair.get("takeoff_zone_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "horizontality_score": float(clamp(renal_pair.get("horizontality_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "axis_alignment_with_iliacs": 0.0,
        "visceral_orthogonality": 0.0,
    }

    allow_renal_rotation = False
    renal_candidate_info = dict(default_renal_info)
    if renal_pair is not None:
        _, renal_candidate_info = refine_horizontal_frame_from_renal_pair(R_provisional, renal_pair, iliac_ref, visceral_axis)
        rotation_abs = abs(float(renal_candidate_info["rotation_degrees_about_z"]))
        renal_alignment = float(renal_candidate_info["axis_alignment_with_iliacs"])
        visceral_ortho = float(renal_candidate_info["visceral_orthogonality"])
        allow_renal_rotation = bool(
            renal_pair_conf >= 0.45
            and renal_alignment >= 0.72
            and (
                rotation_abs <= 35.0
                or (renal_pair_conf >= 0.78 and visceral_ortho >= 0.65 and rotation_abs <= 55.0)
            )
        )

    if renal_pair is not None and allow_renal_rotation:
        R_refined, renal_info = refine_horizontal_frame_from_renal_pair(R_provisional, renal_pair, iliac_ref, visceral_axis)
        source = "renal_pair_consistent_with_iliacs"
        renal_used = True
        refined = bool(abs(float(renal_info["rotation_degrees_about_z"])) > 1.0)
        if renal_pair_conf < 0.60 or abs(float(renal_info["rotation_degrees_about_z"])) > 25.0:
            warnings.append(
                f"W_FRAME_RENAL_PAIR_CAUTION: renal-pair confidence={renal_pair_conf:.3f}, rotation={abs(float(renal_info['rotation_degrees_about_z'])):.1f} deg."
            )
        horizontal_conf = float(
            clamp(
                0.42 * renal_pair_conf
                + 0.18 * float(renal_info["axis_confidence"])
                + 0.12 * float(renal_info["geometry_score"])
                + 0.12 * float(renal_info["takeoff_zone_score"])
                + 0.08 * float(renal_info["axis_alignment_with_iliacs"])
                + 0.08 * float(renal_info["visceral_orthogonality"])
                + 0.15 * iliac_conf
                + 0.05 * visceral_conf,
                0.0,
                1.0,
            )
        )
        score_components = {
            "renal_pair": float(renal_pair_conf),
            "renal_axis": float(renal_info["axis_confidence"]),
            "renal_geometry": float(renal_info["geometry_score"]),
            "renal_takeoff_zone": float(renal_info["takeoff_zone_score"]),
            "renal_iliac_alignment": float(renal_info["axis_alignment_with_iliacs"]),
            "visceral_orthogonality": float(renal_info["visceral_orthogonality"]),
            "iliac_axis": float(iliac_conf),
            "visceral_axis": float(visceral_conf),
        }
    else:
        if renal_pair is None:
            warnings.append("W_FRAME_RENAL_PRIMARY_FAILED: no bilateral renal pair was available; keeping iliac-based provisional horizontal frame.")
        else:
            warnings.append(
                "W_FRAME_RENAL_PRIMARY_REJECTED: renal-pair axis was not sufficiently consistent with the iliac lateral axis; keeping iliac-based horizontal frame."
            )
        R_refined = np.asarray(R_provisional, dtype=float)
        renal_info = dict(default_renal_info)
        renal_info["axis_alignment_with_iliacs"] = float(renal_candidate_info.get("axis_alignment_with_iliacs", 0.0))
        renal_info["visceral_orthogonality"] = float(renal_candidate_info.get("visceral_orthogonality", 0.0))
        source = "iliac_primary"
        renal_used = False
        refined = False
        horizontal_conf = float(clamp(0.42 + 0.40 * iliac_conf + 0.18 * visceral_conf, 0.0, 1.0))
        score_components = {
            "iliac_axis": float(iliac_conf),
            "visceral_axis": float(visceral_conf),
            "renal_pair": float(renal_pair_conf),
            "renal_iliac_alignment": float(renal_info["axis_alignment_with_iliacs"]),
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
        "renal_pair_takeoff_zone_score": float(renal_info["takeoff_zone_score"]),
        "renal_pair_horizontality_score": float(renal_info["horizontality_score"]),
        "renal_axis_alignment_with_iliacs": float(renal_info["axis_alignment_with_iliacs"]),
        "visceral_axis_orthogonality": float(renal_info["visceral_orthogonality"]),
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
    excluded_system_nodes: Optional[set[int]] = None,
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
        excluded_system_nodes=excluded_system_nodes,
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

    chosen_pair = None
    chosen_score = 0.0
    for pair in [scan.get("best_pair")] + pair_candidates:
        if pair is None:
            continue
        geometry_score = float(pair.get("geometry_score", 0.0))
        direction_opposition = float(pair.get("direction_opposition", 1.0))
        takeoff_zone = float(pair.get("takeoff_zone_score", 0.0))
        attachment_score = float(pair.get("attachment_score", 0.0))
        if geometry_score < 0.40 or direction_opposition < 0.45:
            continue
        if takeoff_zone < 0.20 and float(pair.get("score", 0.0)) < 0.75:
            continue
        if attachment_score < 0.25 and float(pair.get("score", 0.0)) < 0.75:
            continue
        chosen_pair = pair
        chosen_score = float(pair.get("score", 0.0))
        break

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
                takeoff_zone = float(0.5 * (float(a.get("renal_zone_score", 0.0)) + float(b.get("renal_zone_score", 0.0))))
                attachment_score = float(min(float(a.get("attachment_score", 0.0)), float(b.get("attachment_score", 0.0))))
                pair_score = 0.35 * geometry + 0.25 * takeoff_sim + 0.15 * len_sim + 0.15 * takeoff_zone + 0.10 * attachment_score
                sided_pairs.append(
                    (
                        float(pair_score),
                        a,
                        b,
                        {
                            "geometry_score": float(geometry),
                            "takeoff_similarity": float(takeoff_sim),
                            "takeoff_zone_score": float(takeoff_zone),
                            "horizontality_score": float(horiz_pair),
                            "attachment_score": float(attachment_score),
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
                "takeoff_zone_score": float(chosen_metrics["takeoff_zone_score"]),
                "horizontality_score": float(chosen_metrics["horizontality_score"]),
                "attachment_score": float(chosen_metrics["attachment_score"]),
            }

    if chosen_pair is None:
        best = max(candidates, key=lambda d: float(d["score"]))
        if float(best.get("score", 0.0)) < 0.25 or float(best.get("renal_zone_score", 0.0)) < 0.10:
            warnings.append("W_RENAL_SINGLE_REJECTED: best single side-branch candidate was not anatomically convincing enough to label as renal.")
            diag["selected_pair_score"] = 0.0
            return None, None, None, None, 0.0, diag
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
    diag["selected_pair_takeoff_zone_score"] = float(chosen_pair.get("takeoff_zone_score", 0.0))
    diag["selected_pair_horizontality_score"] = float(chosen_pair.get("horizontality_score", 0.0))
    diag["selected_pair_attachment_score"] = float(chosen_pair.get("attachment_score", 0.0))
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
    excluded_system_nodes: Optional[set[int]] = None,
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
        excluded_system_nodes=excluded_system_nodes,
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


def add_string_array_to_field_data(fd: "vtkFieldData", name: str, values: List[str]) -> None:
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    arr.SetNumberOfValues(len(values))
    for i, value in enumerate(values):
        arr.SetValue(i, str(value))
    fd.AddArray(arr)


def add_vector_array_to_field_data(fd: "vtkFieldData", name: str, values: List[np.ndarray]) -> None:
    arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(3)
    arr.SetNumberOfTuples(len(values))
    for i, value in enumerate(values):
        xyz = np.asarray(value, dtype=float).reshape(3)
        arr.SetTuple3(i, float(xyz[0]), float(xyz[1]), float(xyz[2]))
    fd.AddArray(arr)


def attach_landmarks_to_polydata_field_data(pd: "vtkPolyData", landmarks: Dict[str, Any]) -> None:
    if pd is None or not landmarks:
        return
    fd = pd.GetFieldData()
    if fd is None:
        fd = vtk.vtkFieldData()
        pd.SetFieldData(fd)
    names = sorted(str(name) for name in landmarks.keys())
    coords = [np.asarray(landmarks[name], dtype=float).reshape(3) for name in names]
    add_string_array_to_field_data(fd, "LandmarkName", names)
    add_vector_array_to_field_data(fd, "LandmarkXYZ", coords)


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


def compute_polydata_cell_centers_numpy(pd: "vtkPolyData") -> np.ndarray:
    cc = vtk.vtkCellCenters()
    cc.SetInputData(pd)
    cc.VertexCellsOff()
    cc.Update()
    out = cc.GetOutput()
    pts = out.GetPoints()
    if pts is None or pts.GetNumberOfPoints() == 0:
        return np.zeros((0, 3), dtype=float)
    return vtk_to_numpy(pts.GetData()).astype(float)


def compute_polydata_cell_areas(pd: "vtkPolyData") -> np.ndarray:
    cell_size = vtk.vtkCellSizeFilter()
    cell_size.SetInputData(pd)
    cell_size.SetComputeArea(True)
    cell_size.SetComputeLength(False)
    cell_size.SetComputeVolume(False)
    cell_size.SetComputeVertexCount(False)
    cell_size.Update()
    out = cell_size.GetOutput()
    arr = out.GetCellData().GetArray("Area") if out.GetCellData() is not None else None
    if arr is None:
        return np.zeros((int(pd.GetNumberOfCells()),), dtype=float)
    return vtk_to_numpy(arr).astype(float)


def build_surface_cell_adjacency(pd: "vtkPolyData") -> List[List[int]]:
    n_cells = int(pd.GetNumberOfCells())
    adjacency = [set() for _ in range(n_cells)]
    edge_to_cells: Dict[Tuple[int, int], List[int]] = {}

    for ci in range(n_cells):
        cell = pd.GetCell(ci)
        if cell is None:
            continue
        ids = [int(cell.GetPointId(k)) for k in range(cell.GetNumberOfPoints())]
        if len(ids) < 2:
            continue
        for k in range(len(ids)):
            a = int(ids[k])
            b = int(ids[(k + 1) % len(ids)])
            ek = edge_key(a, b)
            edge_to_cells.setdefault(ek, []).append(int(ci))

    for cells in edge_to_cells.values():
        if len(cells) < 2:
            continue
        for i in range(len(cells)):
            ci = int(cells[i])
            for j in range(i + 1, len(cells)):
                cj = int(cells[j])
                if ci == cj:
                    continue
                adjacency[ci].add(cj)
                adjacency[cj].add(ci)

    return [sorted(int(v) for v in nbrs) for nbrs in adjacency]


def build_surface_point_to_cells(pd: "vtkPolyData") -> List[List[int]]:
    n_points = int(pd.GetNumberOfPoints())
    point_to_cells: List[List[int]] = [[] for _ in range(n_points)]
    for ci in range(int(pd.GetNumberOfCells())):
        cell = pd.GetCell(ci)
        if cell is None:
            continue
        for k in range(cell.GetNumberOfPoints()):
            pid = int(cell.GetPointId(k))
            if 0 <= pid < n_points:
                point_to_cells[pid].append(int(ci))
    return point_to_cells


def build_surface_label_segment_bank(
    branch_geoms: List[Dict[str, Any]],
    label_id_key: str = "label_id",
    label_name_key: str = "name",
) -> Dict[int, Dict[str, Any]]:
    label_bank: Dict[int, Dict[str, Any]] = {}
    for br in branch_geoms:
        label_id = int(br.get(label_id_key, br.get("label_id", LABEL_OTHER)))
        label_name = str(br.get(label_name_key, br.get("name", LABEL_ID_TO_NAME.get(label_id, "other"))))
        pts = np.asarray(br.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
        if pts.shape[0] < 2:
            continue

        abscissa = compute_abscissa(pts)
        tangents = compute_tangents(pts)
        landmark_point_ids = {str(k): int(v) for k, v in dict(br.get("landmark_point_ids", {}) or {}).items()}
        item = label_bank.setdefault(
            label_id,
            {
                "name": label_name,
                "segment_p0": [],
                "segment_p1": [],
                "polyline_lengths": [],
                "branches": [],
            },
        )
        item["segment_p0"].append(pts[:-1].astype(float))
        item["segment_p1"].append(pts[1:].astype(float))
        item["polyline_lengths"].append(float(abscissa[-1]))
        item["branches"].append(
            {
                "points": pts.astype(float),
                "abscissa": abscissa.astype(float),
                "tangents": tangents.astype(float),
                "landmark_point_ids": landmark_point_ids,
                "length": float(abscissa[-1]),
                "label_id": int(label_id),
                "label_name": str(label_name),
                "node_ids": [int(v) for v in list(br.get("node_ids", []))],
                "topology_role": str(br.get("topology_role", "unassigned")),
                "topology_parent_takeoff": br.get("topology_parent_takeoff"),
                "topology_takeoff_node": br.get("topology_takeoff_node"),
                "topology_parent_attachment_node": br.get("topology_parent_attachment_node"),
                "topology_ownership_start_node": br.get("topology_ownership_start_node"),
                "topology_parent_label_id": br.get("topology_parent_label_id"),
                "topology_parent_name": br.get("topology_parent_name"),
                "topology_ownership_mode": str(br.get("topology_ownership_mode", "owned_edges")),
                "topology_owned_edge_count": int(br.get("topology_owned_edge_count", 0)),
                "proximal_topological_point": np.asarray(
                    br.get("proximal_topological_point", pts[0] if pts.shape[0] else np.zeros((3,), dtype=float)),
                    dtype=float,
                ).reshape(3),
                "proximal_geometric_origin_point": np.asarray(
                    br.get("proximal_geometric_origin_point", pts[0] if pts.shape[0] else np.zeros((3,), dtype=float)),
                    dtype=float,
                ).reshape(3),
                "proximal_geometric_origin_source": str(br.get("proximal_geometric_origin_source", "topological_start")),
                "proximal_parent_projection_distance": br.get("proximal_parent_projection_distance"),
                "proximal_parent_abscissa": br.get("proximal_parent_abscissa"),
                "proximal_direction": np.asarray(
                    br.get("proximal_direction", tangents[0] if tangents.shape[0] else np.zeros((3,), dtype=float)),
                    dtype=float,
                ).reshape(3),
                "proximal_outward_direction": np.asarray(
                    br.get("proximal_outward_direction", tangents[0] if tangents.shape[0] else np.zeros((3,), dtype=float)),
                    dtype=float,
                ).reshape(3),
                "proximal_parent_tangent": np.asarray(
                    br.get("proximal_parent_tangent", tangents[0] if tangents.shape[0] else np.zeros((3,), dtype=float)),
                    dtype=float,
                ).reshape(3),
                "proximal_seed_point": np.asarray(
                    br.get("proximal_seed_point", pts[0] if pts.shape[0] else np.zeros((3,), dtype=float)),
                    dtype=float,
                ).reshape(3),
                "proximal_seed_offset": float(br.get("proximal_seed_offset", 0.0)),
                "proximal_local_step": float(br.get("proximal_local_step", 0.0)),
            }
        )

    out: Dict[int, Dict[str, Any]] = {}
    for label_id, item in label_bank.items():
        if not item["segment_p0"]:
            continue
        seg_p0 = np.concatenate(item["segment_p0"], axis=0).astype(float)
        seg_p1 = np.concatenate(item["segment_p1"], axis=0).astype(float)
        out[int(label_id)] = {
            "name": str(item["name"]),
            "segment_p0": seg_p0,
            "segment_p1": seg_p1,
            "polyline_count": int(len(item["polyline_lengths"])),
            "segment_count": int(seg_p0.shape[0]),
            "branches": list(item["branches"]),
        }
    return out


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


def transform_single_point(point: Any, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    arr = np.asarray(point, dtype=float).reshape(1, 3)
    return apply_transform_points(arr, R, origin)[0].astype(float)


def polyline_plane_reference(
    points: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    prefer: str = "first",
) -> Dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    origin = np.asarray(plane_origin, dtype=float).reshape(3)
    normal = unit(np.asarray(plane_normal, dtype=float).reshape(3))
    if pts.shape[0] == 0 or np.linalg.norm(normal) < EPS:
        return {
            "point": origin.astype(float),
            "type": "fallback_origin",
            "segment_index": None,
            "t": None,
            "distance_abs": float("inf"),
        }

    signed = np.dot(pts - origin[None, :], normal)
    crossings: List[Tuple[int, float, np.ndarray]] = []
    zero_tol = 1e-8
    for idx in range(max(0, pts.shape[0] - 1)):
        d0 = float(signed[idx])
        d1 = float(signed[idx + 1])
        if abs(d0) <= zero_tol and abs(d1) <= zero_tol:
            crossings.append((int(idx), 0.0, pts[idx].astype(float)))
            continue
        if abs(d0) <= zero_tol:
            crossings.append((int(idx), 0.0, pts[idx].astype(float)))
            continue
        if abs(d1) <= zero_tol:
            crossings.append((int(idx), 1.0, pts[idx + 1].astype(float)))
            continue
        if d0 * d1 < 0.0:
            t = float(abs(d0) / (abs(d0) + abs(d1) + EPS))
            crossings.append((int(idx), t, ((1.0 - t) * pts[idx] + t * pts[idx + 1]).astype(float)))

    if crossings:
        idx, t, point = crossings[0] if str(prefer).lower() != "last" else crossings[-1]
        return {
            "point": np.asarray(point, dtype=float).reshape(3),
            "type": "intersection",
            "segment_index": int(idx),
            "t": float(t),
            "distance_abs": 0.0,
        }

    best_idx = int(np.argmin(np.abs(signed)))
    return {
        "point": pts[best_idx].astype(float),
        "type": "nearest_vertex",
        "segment_index": None,
        "t": None,
        "distance_abs": float(abs(signed[best_idx])),
    }


def extract_local_surface_patch(
    surface_pd: "vtkPolyData",
    center: np.ndarray,
    radius: float,
) -> "vtkPolyData":
    sphere = vtk.vtkSphere()
    sphere.SetCenter(float(center[0]), float(center[1]), float(center[2]))
    sphere.SetRadius(float(max(radius, 1e-3)))

    extract = vtk.vtkExtractGeometry()
    extract.SetInputData(surface_pd)
    extract.SetImplicitFunction(sphere)
    extract.ExtractInsideOn()
    extract.ExtractBoundaryCellsOn()
    extract.Update()

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(extract.GetOutput())
    geom.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(geom.GetOutput())
    clean.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(clean.GetOutput())
    return out


def extract_plane_cut_components(
    surface_patch: "vtkPolyData",
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    neighborhood_center: np.ndarray,
    neighborhood_radius: float,
) -> List[Dict[str, Any]]:
    out_components: List[Dict[str, Any]] = []
    if surface_patch is None or int(surface_patch.GetNumberOfCells()) <= 0:
        return out_components

    plane = vtk.vtkPlane()
    plane.SetOrigin(float(plane_origin[0]), float(plane_origin[1]), float(plane_origin[2]))
    plane.SetNormal(float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]))

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface_patch)
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

    stripped = stripper.GetOutput()
    if stripped is None or int(stripped.GetNumberOfCells()) <= 0:
        stripped = clean.GetOutput()
    if stripped is None:
        return out_components

    center = np.asarray(neighborhood_center, dtype=float).reshape(3)
    radius = float(max(neighborhood_radius, 1e-3))

    for cell_id in range(int(stripped.GetNumberOfCells())):
        cell = stripped.GetCell(cell_id)
        if cell is None:
            continue
        n_pts = int(cell.GetNumberOfPoints())
        if n_pts < 2:
            continue
        pts = np.zeros((n_pts, 3), dtype=float)
        for k in range(n_pts):
            pid = int(cell.GetPointId(k))
            pts[k, :] = np.asarray(stripped.GetPoint(pid), dtype=float)
        min_center_dist = float(np.min(np.linalg.norm(pts - center[None, :], axis=1))) if pts.size else float("inf")
        if min_center_dist > radius:
            continue
        area, _, _ = planar_polygon_area_and_normal(pts) if pts.shape[0] >= 3 else (0.0, np.zeros((3,), dtype=float), float("nan"))
        out_components.append(
            {
                "points": pts.astype(float),
                "centroid": np.mean(pts, axis=0).astype(float),
                "length": float(polyline_length(pts)),
                "area": float(area),
                "min_center_distance": float(min_center_dist),
            }
        )

    out_components.sort(
        key=lambda item: (
            float(item.get("min_center_distance", float("inf"))),
            -float(item.get("length", 0.0)),
        )
    )
    return out_components


def classify_split_cross_section(
    components: List[Dict[str, Any]],
    child_ref_a: np.ndarray,
    child_ref_b: np.ndarray,
    local_step: float,
    child_separation: float,
) -> Dict[str, Any]:
    child_a = np.asarray(child_ref_a, dtype=float).reshape(3)
    child_b = np.asarray(child_ref_b, dtype=float).reshape(3)
    step = float(max(local_step, 1e-3))
    separation = float(max(child_separation, step))
    min_length = float(max(1.25 * step, 0.12 * separation))
    match_threshold = float(max(2.5 * step, 0.45 * separation))

    significant = [comp for comp in components if float(comp.get("length", 0.0)) >= min_length]
    if len(significant) < 2:
        return {
            "is_split": False,
            "component_count": int(len(significant)),
            "matched_children": 0,
            "match_threshold": float(match_threshold),
        }

    dists_a = [float(np.min(np.linalg.norm(np.asarray(comp["points"], dtype=float) - child_a[None, :], axis=1))) for comp in significant]
    dists_b = [float(np.min(np.linalg.norm(np.asarray(comp["points"], dtype=float) - child_b[None, :], axis=1))) for comp in significant]
    idx_a = int(np.argmin(dists_a))
    idx_b = int(np.argmin(dists_b))
    distinct = idx_a != idx_b
    matched_a = float(dists_a[idx_a]) <= match_threshold
    matched_b = float(dists_b[idx_b]) <= match_threshold
    matched_children = int(matched_a) + int(matched_b)

    return {
        "is_split": bool(distinct and matched_a and matched_b),
        "component_count": int(len(significant)),
        "matched_children": int(matched_children if distinct else max(int(matched_a), int(matched_b))),
        "match_threshold": float(match_threshold),
        "child_a_component_index": int(idx_a),
        "child_b_component_index": int(idx_b),
        "child_a_distance": float(dists_a[idx_a]),
        "child_b_distance": float(dists_b[idx_b]),
    }


def refine_bifurcation_by_geometric_separation(
    surface_pd: "vtkPolyData",
    parent_points: np.ndarray,
    child_a_points: np.ndarray,
    child_b_points: np.ndarray,
    topology_candidate_point: np.ndarray,
    topology_candidate_node: Optional[int],
    junction_name: str,
    child_a_name: str,
    child_b_name: str,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    work_warnings = warnings if warnings is not None else []

    parent_pts = np.asarray(parent_points, dtype=float)
    child_a_pts = np.asarray(child_a_points, dtype=float)
    child_b_pts = np.asarray(child_b_points, dtype=float)
    candidate_point = np.asarray(topology_candidate_point, dtype=float).reshape(3)

    result: Dict[str, Any] = {
        "junction_name": str(junction_name),
        "child_a_name": str(child_a_name),
        "child_b_name": str(child_b_name),
        "topology_candidate_node": (None if topology_candidate_node is None else int(topology_candidate_node)),
        "topology_candidate_point_raw": candidate_point.astype(float),
        "status": "topology_fallback",
        "used_geometric_refinement": False,
        "confidence": 0.0,
        "bracketed_transition": False,
        "refined_split_point_raw": candidate_point.astype(float),
        "refined_parent_point_raw": candidate_point.astype(float),
        "refined_child_points_raw": {
            str(child_a_name): child_a_pts[0].astype(float) if child_a_pts.shape[0] else candidate_point.astype(float),
            str(child_b_name): child_b_pts[0].astype(float) if child_b_pts.shape[0] else candidate_point.astype(float),
        },
        "scan_axis_raw": np.zeros((3,), dtype=float),
        "search_window": {},
        "classification": {},
    }

    if parent_pts.shape[0] < 2 or child_a_pts.shape[0] < 2 or child_b_pts.shape[0] < 2:
        work_warnings.append(f"W_GEOM_{str(junction_name).upper()}_SHORT_PATHS: insufficient centerline support for geometric refinement.")
        return result

    parent_len = float(polyline_length(parent_pts))
    child_a_len = float(polyline_length(child_a_pts))
    child_b_len = float(polyline_length(child_b_pts))
    local_step = float(
        max(
            np.median(np.linalg.norm(parent_pts[1:] - parent_pts[:-1], axis=1)) if parent_pts.shape[0] >= 2 else 0.0,
            np.median(np.linalg.norm(child_a_pts[1:] - child_a_pts[:-1], axis=1)) if child_a_pts.shape[0] >= 2 else 0.0,
            np.median(np.linalg.norm(child_b_pts[1:] - child_b_pts[:-1], axis=1)) if child_b_pts.shape[0] >= 2 else 0.0,
            1e-3,
        )
    )

    parent_dir = unit(parent_pts[-1] - parent_pts[-2])
    child_ref_offset = float(
        min(
            max(3.0 * local_step, 0.12 * min(child_a_len, child_b_len)),
            0.40 * min(child_a_len, child_b_len),
        )
    )
    if child_ref_offset <= EPS:
        child_ref_offset = float(max(2.0 * local_step, 1e-3))
    child_a_dir = unit(polyline_point_at_abscissa(child_a_pts, child_ref_offset) - child_a_pts[0])
    child_b_dir = unit(polyline_point_at_abscissa(child_b_pts, child_ref_offset) - child_b_pts[0])
    child_mean_dir = unit(child_a_dir + child_b_dir)
    scan_axis = unit(parent_dir + child_mean_dir)
    if np.linalg.norm(scan_axis) < EPS:
        scan_axis = parent_dir.copy()
    if np.linalg.norm(scan_axis) < EPS:
        scan_axis = unit(child_a_dir + child_b_dir)
    if np.linalg.norm(scan_axis) < EPS:
        scan_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    child_ref_a = polyline_point_at_abscissa(child_a_pts, child_ref_offset)
    child_ref_b = polyline_point_at_abscissa(child_b_pts, child_ref_offset)
    child_separation = float(
        max(
            np.linalg.norm(project_vector_to_plane(child_ref_a - child_ref_b, scan_axis)),
            np.linalg.norm(child_ref_a - child_ref_b),
            2.0 * local_step,
        )
    )

    upstream = float(
        min(
            max(3.0 * local_step, 0.40 * child_separation),
            0.45 * parent_len if parent_len > EPS else max(3.0 * local_step, 0.40 * child_separation),
        )
    )
    downstream = float(
        min(
            max(4.0 * local_step, 0.85 * child_separation),
            0.50 * min(child_a_len, child_b_len) if min(child_a_len, child_b_len) > EPS else max(4.0 * local_step, 0.85 * child_separation),
        )
    )
    patch_radius = float(max(1.50 * child_separation, upstream + downstream + 3.0 * local_step, 6.0 * local_step))
    result["scan_axis_raw"] = scan_axis.astype(float)
    result["search_window"] = {
        "upstream": float(upstream),
        "downstream": float(downstream),
        "patch_radius": float(patch_radius),
        "local_step": float(local_step),
        "child_reference_offset": float(child_ref_offset),
    }

    surface_patch = extract_local_surface_patch(surface_pd, candidate_point, patch_radius)
    if surface_patch is None or int(surface_patch.GetNumberOfCells()) <= 0:
        work_warnings.append(f"W_GEOM_{str(junction_name).upper()}_PATCH_EMPTY: local surface patch extraction failed; using topology candidate.")
        return result

    sample_offsets = np.linspace(-float(upstream), float(downstream), 17)
    sample_rows: List[Dict[str, Any]] = []
    first_split_index: Optional[int] = None
    last_single_index: Optional[int] = None
    split_plane_origin: Optional[np.ndarray] = None

    for idx, offset in enumerate(sample_offsets.tolist()):
        plane_origin = candidate_point + float(offset) * scan_axis
        child_ref_info_a = polyline_plane_reference(child_a_pts, plane_origin, scan_axis, prefer="first")
        child_ref_info_b = polyline_plane_reference(child_b_pts, plane_origin, scan_axis, prefer="first")
        cut_components = extract_plane_cut_components(
            surface_patch=surface_patch,
            plane_origin=plane_origin,
            plane_normal=scan_axis,
            neighborhood_center=plane_origin,
            neighborhood_radius=max(0.80 * patch_radius, 4.0 * local_step),
        )
        classification = classify_split_cross_section(
            components=cut_components,
            child_ref_a=np.asarray(child_ref_info_a["point"], dtype=float),
            child_ref_b=np.asarray(child_ref_info_b["point"], dtype=float),
            local_step=float(local_step),
            child_separation=float(max(child_separation, np.linalg.norm(np.asarray(child_ref_info_a["point"]) - np.asarray(child_ref_info_b["point"])))),
        )
        is_split = bool(classification.get("is_split", False))
        if is_split and first_split_index is None:
            first_split_index = int(idx)
            split_plane_origin = np.asarray(plane_origin, dtype=float).reshape(3)
        if not is_split:
            last_single_index = int(idx)

        sample_rows.append(
            {
                "offset": float(offset),
                "is_split": bool(is_split),
                "component_count": int(classification.get("component_count", 0)),
                "matched_children": int(classification.get("matched_children", 0)),
            }
        )

    result["classification"]["coarse_samples"] = list(sample_rows)

    if first_split_index is None:
        work_warnings.append(f"W_GEOM_{str(junction_name).upper()}_NO_SPLIT: geometric split was not detected; using topology candidate.")
        return result

    low = float(sample_offsets[first_split_index - 1]) if first_split_index > 0 else float(sample_offsets[first_split_index])
    high = float(sample_offsets[first_split_index])
    bracketed = bool(first_split_index > 0 and not bool(sample_rows[first_split_index - 1]["is_split"]))

    if bracketed:
        for _ in range(8):
            mid = 0.5 * (low + high)
            plane_origin = candidate_point + mid * scan_axis
            child_ref_info_a = polyline_plane_reference(child_a_pts, plane_origin, scan_axis, prefer="first")
            child_ref_info_b = polyline_plane_reference(child_b_pts, plane_origin, scan_axis, prefer="first")
            cut_components = extract_plane_cut_components(
                surface_patch=surface_patch,
                plane_origin=plane_origin,
                plane_normal=scan_axis,
                neighborhood_center=plane_origin,
                neighborhood_radius=max(0.80 * patch_radius, 4.0 * local_step),
            )
            classification = classify_split_cross_section(
                components=cut_components,
                child_ref_a=np.asarray(child_ref_info_a["point"], dtype=float),
                child_ref_b=np.asarray(child_ref_info_b["point"], dtype=float),
                local_step=float(local_step),
                child_separation=float(max(child_separation, np.linalg.norm(np.asarray(child_ref_info_a["point"]) - np.asarray(child_ref_info_b["point"])))),
            )
            if bool(classification.get("is_split", False)):
                high = float(mid)
                split_plane_origin = np.asarray(plane_origin, dtype=float).reshape(3)
            else:
                low = float(mid)

    if split_plane_origin is None:
        split_plane_origin = candidate_point + high * scan_axis

    parent_ref_info = polyline_plane_reference(parent_pts, split_plane_origin, scan_axis, prefer="last")
    child_ref_info_a = polyline_plane_reference(child_a_pts, split_plane_origin, scan_axis, prefer="first")
    child_ref_info_b = polyline_plane_reference(child_b_pts, split_plane_origin, scan_axis, prefer="first")
    final_components = extract_plane_cut_components(
        surface_patch=surface_patch,
        plane_origin=split_plane_origin,
        plane_normal=scan_axis,
        neighborhood_center=split_plane_origin,
        neighborhood_radius=max(0.80 * patch_radius, 4.0 * local_step),
    )
    final_classification = classify_split_cross_section(
        components=final_components,
        child_ref_a=np.asarray(child_ref_info_a["point"], dtype=float),
        child_ref_b=np.asarray(child_ref_info_b["point"], dtype=float),
        local_step=float(local_step),
        child_separation=float(max(child_separation, np.linalg.norm(np.asarray(child_ref_info_a["point"]) - np.asarray(child_ref_info_b["point"])))),
    )

    split_midpoint = 0.5 * (
        np.asarray(child_ref_info_a["point"], dtype=float).reshape(3)
        + np.asarray(child_ref_info_b["point"], dtype=float).reshape(3)
    )
    geometric_success = bool(final_classification.get("is_split", False))
    result.update(
        {
            "status": ("geometric_refined" if bracketed and geometric_success else "geometric_unbracketed" if geometric_success else "topology_fallback"),
            "used_geometric_refinement": bool(geometric_success),
            "confidence": float(
                clamp(
                    (0.85 if bracketed else 0.55)
                    * (0.50 + 0.25 * min(1.0, float(final_classification.get("matched_children", 0)) / 2.0) + 0.25 * min(1.0, float(final_classification.get("component_count", 0)) / 2.0)),
                    0.0,
                    1.0,
                )
            ) if geometric_success else 0.0,
            "bracketed_transition": bool(bracketed),
            "refined_parameter": float(high),
            "refined_parent_point_raw": np.asarray(parent_ref_info["point"], dtype=float).reshape(3),
            "refined_split_point_raw": split_midpoint.astype(float),
            "refined_child_points_raw": {
                str(child_a_name): np.asarray(child_ref_info_a["point"], dtype=float).reshape(3),
                str(child_b_name): np.asarray(child_ref_info_b["point"], dtype=float).reshape(3),
            },
            "classification": {
                "component_count": int(final_classification.get("component_count", 0)),
                "matched_children": int(final_classification.get("matched_children", 0)),
                "match_threshold": float(final_classification.get("match_threshold", 0.0)),
                "coarse_samples": list(sample_rows),
                "last_single_index": (None if last_single_index is None else int(last_single_index)),
                "first_split_index": int(first_split_index),
            },
        }
    )

    if not geometric_success:
        work_warnings.append(f"W_GEOM_{str(junction_name).upper()}_FINAL_CLASSIFY_FAILED: split plane refinement did not validate; using topology candidate.")
    elif not bracketed:
        work_warnings.append(
            f"W_GEOM_{str(junction_name).upper()}_UNBRACKETED: geometric split was detected without a proximal single-lumen bracket; using best split plane with reduced confidence."
        )

    return result


def refine_prelabeled_geometric_bifurcations(
    surface_pd: "vtkPolyData",
    pts: np.ndarray,
    topology: Dict[str, Any],
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    work_warnings = warnings if warnings is not None else []
    branch_paths_raw = {
        str(name): [int(n) for n in nodes]
        for name, nodes in dict(topology.get("branch_paths_raw", {})).items()
    }

    def _points_for(name: str) -> np.ndarray:
        nodes = [int(n) for n in branch_paths_raw.get(str(name), [])]
        if not nodes:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(pts[np.asarray(nodes, dtype=int)], dtype=float)

    out: Dict[str, Any] = {}

    aortic_parent = _points_for(LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK])
    right_common = _points_for(LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC])
    left_common = _points_for(LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC])
    bif_node = topology.get("bifurcation_node")
    if aortic_parent.shape[0] >= 2 and right_common.shape[0] >= 2 and left_common.shape[0] >= 2 and bif_node is not None:
        out["aortic_bifurcation"] = refine_bifurcation_by_geometric_separation(
            surface_pd=surface_pd,
            parent_points=aortic_parent,
            child_a_points=right_common,
            child_b_points=left_common,
            topology_candidate_point=np.asarray(pts[int(bif_node)], dtype=float),
            topology_candidate_node=int(bif_node),
            junction_name="aortic_bifurcation",
            child_a_name=LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC],
            child_b_name=LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC],
            warnings=work_warnings,
        )

    right_split_node = topology.get("right_common_iliac_split_node")
    right_internal = _points_for(LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC])
    right_external = _points_for(LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC])
    if right_common.shape[0] >= 2 and right_internal.shape[0] >= 2 and right_external.shape[0] >= 2 and right_split_node is not None:
        out["right_common_iliac_split"] = refine_bifurcation_by_geometric_separation(
            surface_pd=surface_pd,
            parent_points=right_common,
            child_a_points=right_internal,
            child_b_points=right_external,
            topology_candidate_point=np.asarray(pts[int(right_split_node)], dtype=float),
            topology_candidate_node=int(right_split_node),
            junction_name="right_common_iliac_split",
            child_a_name=LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC],
            child_b_name=LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC],
            warnings=work_warnings,
        )

    left_split_node = topology.get("left_common_iliac_split_node")
    left_internal = _points_for(LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC])
    left_external = _points_for(LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC])
    if left_common.shape[0] >= 2 and left_internal.shape[0] >= 2 and left_external.shape[0] >= 2 and left_split_node is not None:
        out["left_common_iliac_split"] = refine_bifurcation_by_geometric_separation(
            surface_pd=surface_pd,
            parent_points=left_common,
            child_a_points=left_internal,
            child_b_points=left_external,
            topology_candidate_point=np.asarray(pts[int(left_split_node)], dtype=float),
            topology_candidate_node=int(left_split_node),
            junction_name="left_common_iliac_split",
            child_a_name=LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC],
            child_b_name=LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC],
            warnings=work_warnings,
        )

    return out


def transform_geometric_junctions(
    junctions_raw: Dict[str, Any],
    R: np.ndarray,
    origin: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, info in dict(junctions_raw).items():
        item = dict(info)
        if info.get("topology_candidate_point_raw") is not None:
            item["topology_candidate_point_canonical"] = transform_single_point(info["topology_candidate_point_raw"], R, origin)
        if info.get("refined_parent_point_raw") is not None:
            item["refined_parent_point_canonical"] = transform_single_point(info["refined_parent_point_raw"], R, origin)
        if info.get("refined_split_point_raw") is not None:
            item["refined_split_point_canonical"] = transform_single_point(info["refined_split_point_raw"], R, origin)
        child_points_raw = dict(info.get("refined_child_points_raw", {}) or {})
        if child_points_raw:
            item["refined_child_points_canonical"] = {
                str(name): transform_single_point(value, R, origin)
                for name, value in child_points_raw.items()
            }
        if info.get("scan_axis_raw") is not None:
            axis = np.asarray(info["scan_axis_raw"], dtype=float).reshape(3)
            item["scan_axis_canonical"] = (np.asarray(R, dtype=float).reshape(3, 3) @ axis).astype(float)
        out[str(key)] = item
    return out


def _select_primary_branch_geometry(
    branch_geoms: List[Dict[str, Any]],
    branch_name: str,
) -> Optional[Dict[str, Any]]:
    candidates = [br for br in branch_geoms if str(br.get("name", "")) == str(branch_name)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda br: float(polyline_length(np.asarray(br.get("points", np.zeros((0, 3), dtype=float)), dtype=float))),
    )


def replace_branch_start_point(branch: Dict[str, Any], point: np.ndarray) -> None:
    pts = np.asarray(branch.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
    if pts.shape[0] == 0:
        return
    pts = pts.copy()
    pts[0] = np.asarray(point, dtype=float).reshape(3)
    branch["points"] = pts.astype(float)


def replace_branch_end_point(branch: Dict[str, Any], point: np.ndarray) -> None:
    pts = np.asarray(branch.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
    if pts.shape[0] == 0:
        return
    pts = pts.copy()
    pts[-1] = np.asarray(point, dtype=float).reshape(3)
    branch["points"] = pts.astype(float)


def apply_geometric_junctions_to_prelabeled_branch_geometries(
    branch_geoms: List[Dict[str, Any]],
    junctions_canonical: Dict[str, Any],
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    work_warnings = warnings if warnings is not None else []
    if not junctions_canonical:
        return branch_geoms

    trunk = _select_primary_branch_geometry(branch_geoms, LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK])
    right_common = _select_primary_branch_geometry(branch_geoms, LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC])
    left_common = _select_primary_branch_geometry(branch_geoms, LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC])
    right_internal = _select_primary_branch_geometry(branch_geoms, LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC])
    right_external = _select_primary_branch_geometry(branch_geoms, LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC])
    left_internal = _select_primary_branch_geometry(branch_geoms, LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC])
    left_external = _select_primary_branch_geometry(branch_geoms, LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC])

    aortic = dict(junctions_canonical.get("aortic_bifurcation", {}) or {})
    if aortic and bool(aortic.get("used_geometric_refinement")):
        split_point = np.asarray(aortic.get("refined_split_point_canonical", aortic.get("refined_split_point_raw")), dtype=float).reshape(3)
        child_points = {
            str(k): np.asarray(v, dtype=float).reshape(3)
            for k, v in dict(aortic.get("refined_child_points_canonical", aortic.get("refined_child_points_raw", {})) or {}).items()
        }
        if trunk is not None:
            replace_branch_end_point(trunk, split_point)
            trunk.setdefault("landmark_point_ids", {})
            trunk["landmark_point_ids"]["Bifurcation"] = int(max(0, np.asarray(trunk["points"]).shape[0] - 1))
            trunk["landmark_point_ids"]["AorticBifurcation"] = int(max(0, np.asarray(trunk["points"]).shape[0] - 1))
        if right_common is not None and LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC] in child_points:
            replace_branch_start_point(right_common, child_points[LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]])
            right_common.setdefault("landmark_point_ids", {})
            right_common["landmark_point_ids"].pop("Bifurcation", None)
            right_common["landmark_point_ids"].pop("AorticBifurcation", None)
            right_common["landmark_point_ids"]["RightCommonIliacStart"] = 0
            right_common["force_proximal_geometric_origin_to_topological"] = True
        if left_common is not None and LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC] in child_points:
            replace_branch_start_point(left_common, child_points[LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]])
            left_common.setdefault("landmark_point_ids", {})
            left_common["landmark_point_ids"].pop("Bifurcation", None)
            left_common["landmark_point_ids"].pop("AorticBifurcation", None)
            left_common["landmark_point_ids"]["LeftCommonIliacStart"] = 0
            left_common["force_proximal_geometric_origin_to_topological"] = True
    elif aortic:
        work_warnings.append("W_GEOM_AORTIC_BIF_NOT_APPLIED: aortic geometric refinement was unavailable; branch ownership remains topology-derived at the bifurcation.")

    right_split = dict(junctions_canonical.get("right_common_iliac_split", {}) or {})
    if right_split and bool(right_split.get("used_geometric_refinement")):
        split_point = np.asarray(right_split.get("refined_split_point_canonical", right_split.get("refined_split_point_raw")), dtype=float).reshape(3)
        child_points = {
            str(k): np.asarray(v, dtype=float).reshape(3)
            for k, v in dict(right_split.get("refined_child_points_canonical", right_split.get("refined_child_points_raw", {})) or {}).items()
        }
        if right_common is not None:
            replace_branch_end_point(right_common, split_point)
            right_common.setdefault("landmark_point_ids", {})
            right_common["landmark_point_ids"]["RightCommonIliacSplit"] = int(max(0, np.asarray(right_common["points"]).shape[0] - 1))
        if right_internal is not None and LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC] in child_points:
            replace_branch_start_point(right_internal, child_points[LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC]])
            right_internal.setdefault("landmark_point_ids", {})
            right_internal["landmark_point_ids"].pop("RightCommonIliacSplit", None)
            right_internal["landmark_point_ids"]["RightInternalIliacStart"] = 0
            right_internal["force_proximal_geometric_origin_to_topological"] = True
        if right_external is not None and LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC] in child_points:
            replace_branch_start_point(right_external, child_points[LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC]])
            right_external.setdefault("landmark_point_ids", {})
            right_external["landmark_point_ids"].pop("RightCommonIliacSplit", None)
            right_external["landmark_point_ids"]["RightExternalIliacStart"] = 0
            right_external["force_proximal_geometric_origin_to_topological"] = True
    elif right_split:
        work_warnings.append("W_GEOM_RIGHT_ILIAC_SPLIT_NOT_APPLIED: right common/internal/external geometric refinement was unavailable; branch ownership remains topology-derived.")

    left_split = dict(junctions_canonical.get("left_common_iliac_split", {}) or {})
    if left_split and bool(left_split.get("used_geometric_refinement")):
        split_point = np.asarray(left_split.get("refined_split_point_canonical", left_split.get("refined_split_point_raw")), dtype=float).reshape(3)
        child_points = {
            str(k): np.asarray(v, dtype=float).reshape(3)
            for k, v in dict(left_split.get("refined_child_points_canonical", left_split.get("refined_child_points_raw", {})) or {}).items()
        }
        if left_common is not None:
            replace_branch_end_point(left_common, split_point)
            left_common.setdefault("landmark_point_ids", {})
            left_common["landmark_point_ids"]["LeftCommonIliacSplit"] = int(max(0, np.asarray(left_common["points"]).shape[0] - 1))
        if left_internal is not None and LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC] in child_points:
            replace_branch_start_point(left_internal, child_points[LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC]])
            left_internal.setdefault("landmark_point_ids", {})
            left_internal["landmark_point_ids"].pop("LeftCommonIliacSplit", None)
            left_internal["landmark_point_ids"]["LeftInternalIliacStart"] = 0
            left_internal["force_proximal_geometric_origin_to_topological"] = True
        if left_external is not None and LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC] in child_points:
            replace_branch_start_point(left_external, child_points[LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC]])
            left_external.setdefault("landmark_point_ids", {})
            left_external["landmark_point_ids"].pop("LeftCommonIliacSplit", None)
            left_external["landmark_point_ids"]["LeftExternalIliacStart"] = 0
            left_external["force_proximal_geometric_origin_to_topological"] = True
    elif left_split:
        work_warnings.append("W_GEOM_LEFT_ILIAC_SPLIT_NOT_APPLIED: left common/internal/external geometric refinement was unavailable; branch ownership remains topology-derived.")

    return branch_geoms


def find_surface_bank_landmark(
    bank_entry: Optional[Dict[str, Any]],
    landmark_key: str,
    tangent_mode: str = "forward",
) -> Optional[Dict[str, Any]]:
    if not bank_entry:
        return None

    candidates: List[Dict[str, Any]] = []
    for branch in list(bank_entry.get("branches", [])):
        landmark_point_ids = dict(branch.get("landmark_point_ids", {}) or {})
        if landmark_key not in landmark_point_ids:
            continue
        pts = np.asarray(branch.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
        if pts.shape[0] == 0:
            continue
        tangents = np.asarray(branch.get("tangents", compute_tangents(pts)), dtype=float)
        abscissa = np.asarray(branch.get("abscissa", compute_abscissa(pts)), dtype=float)
        idx = int(max(0, min(int(landmark_point_ids[landmark_key]), pts.shape[0] - 1)))
        if tangent_mode == "backward":
            if idx > 0:
                tangent = pts[idx] - pts[idx - 1]
            elif pts.shape[0] > 1:
                tangent = pts[1] - pts[0]
            else:
                tangent = tangents[idx]
        elif tangent_mode == "forward":
            if idx < pts.shape[0] - 1:
                tangent = pts[idx + 1] - pts[idx]
            elif idx > 0:
                tangent = pts[idx] - pts[idx - 1]
            else:
                tangent = tangents[idx]
        else:
            tangent = tangents[idx]
        candidates.append(
            {
                "point": pts[idx].astype(float),
                "tangent": unit(tangent),
                "index": int(idx),
                "points": pts.astype(float),
                "abscissa": abscissa.astype(float),
                "length": float(branch.get("length", abscissa[-1] if abscissa.size else 0.0)),
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("length", 0.0)))


def get_longest_surface_bank_branch(bank_entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not bank_entry:
        return None
    branches = list(bank_entry.get("branches", []))
    if not branches:
        return None
    return max(branches, key=lambda item: float(item.get("length", 0.0)))


def nearest_cell_to_point(
    cell_centers: np.ndarray,
    target_point: np.ndarray,
    allowed_mask: Optional[np.ndarray] = None,
) -> int:
    pts = np.asarray(cell_centers, dtype=float)
    if pts.shape[0] == 0:
        return -1
    target = np.asarray(target_point, dtype=float).reshape(3)
    if allowed_mask is None:
        candidate_ids = np.arange(pts.shape[0], dtype=int)
    else:
        candidate_ids = np.flatnonzero(np.asarray(allowed_mask, dtype=bool))
    if candidate_ids.size == 0:
        return -1
    d2 = np.sum((pts[candidate_ids] - target[None, :]) ** 2, axis=1)
    return int(candidate_ids[int(np.argmin(d2))])


def connected_components_from_seeds(
    seed_cells: List[int],
    allowed_mask: np.ndarray,
    adjacency: List[List[int]],
) -> List[int]:
    keep: set[int] = set()
    allowed = np.asarray(allowed_mask, dtype=bool).copy()
    for seed in seed_cells:
        seed_i = int(seed)
        if seed_i < 0 or seed_i >= allowed.shape[0]:
            continue
        allowed[seed_i] = True
        keep.update(int(v) for v in connected_component_from_seed(seed_i, allowed, adjacency))
    return sorted(int(v) for v in keep)


def summarize_surface_label_transfer(
    cell_labels: np.ndarray,
    point_labels: np.ndarray,
    adjacency: List[List[int]],
    cell_areas: np.ndarray,
    seed_cells: Dict[int, int],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    cell_labels_arr = np.asarray(cell_labels, dtype=int)
    point_labels_arr = np.asarray(point_labels, dtype=int)
    cell_areas_arr = np.asarray(cell_areas, dtype=float)
    if cell_areas_arr.size != cell_labels_arr.size:
        cell_areas_arr = np.zeros((cell_labels_arr.size,), dtype=float)

    cell_counts: Dict[str, int] = {}
    point_counts: Dict[str, int] = {}
    component_counts: Dict[str, int] = {}
    area_by_label: Dict[str, float] = {}
    for lid, name in LABEL_ID_TO_NAME.items():
        mask = np.asarray(cell_labels_arr == int(lid), dtype=bool)
        cell_counts[str(name)] = int(np.count_nonzero(mask))
        point_counts[str(name)] = int(np.count_nonzero(point_labels_arr == int(lid)))
        component_counts[str(name)] = int(len(label_connected_components(cell_labels_arr, adjacency, int(lid))))
        area_by_label[str(name)] = float(np.sum(cell_areas_arr[mask])) if cell_areas_arr.size else 0.0

    contact_map = compute_surface_label_contact_map(cell_labels_arr, adjacency)
    present_nonzero = [str(name) for name, count in cell_counts.items() if int(count) > 0]
    target_names = [str(v) for v in list(settings.get("surface_target_labels", []))]
    if not target_names:
        target_names = [str(name) for name in LABEL_ID_TO_NAME.values()]
    missing_target_labels = [
        str(name)
        for name in target_names
        if str(name) != LABEL_ID_TO_NAME[LABEL_OTHER] and int(cell_counts.get(str(name), 0)) <= 0
    ]
    contact_map_named: Dict[str, Dict[str, int]] = {}
    for src_label, dst_map in contact_map.items():
        src_name = LABEL_ID_TO_NAME.get(int(src_label), str(src_label))
        contact_map_named[str(src_name)] = {}
        for dst_label, count in dst_map.items():
            dst_name = LABEL_ID_TO_NAME.get(int(dst_label), str(dst_label))
            contact_map_named[str(src_name)][str(dst_name)] = int(count)

    validation = {
        "present_labels": list(present_nonzero),
        "target_labels": list(target_names),
        "missing_target_labels": list(missing_target_labels),
        "non_target_surface_labels_present": [str(name) for name in present_nonzero if str(name) not in target_names],
        "component_counts": {str(k): int(v) for k, v in component_counts.items()},
    }

    return {
        "active_surface_labels": list(target_names),
        "forbidden_surface_labels": [],
        "cell_counts": cell_counts,
        "point_counts": point_counts,
        "component_counts": component_counts,
        "area_by_label": area_by_label,
        "touches": contact_map_named,
        "seed_cells": {LABEL_ID_TO_NAME[int(k)]: int(v) for k, v in seed_cells.items()},
        "validation": validation,
        "settings": dict(settings),
    }


def min_distance_sq_points_to_segments(points: np.ndarray, seg_p0: np.ndarray, seg_p1: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    p0 = np.asarray(seg_p0, dtype=float)
    p1 = np.asarray(seg_p1, dtype=float)
    out = np.full((pts.shape[0],), float("inf"), dtype=float)
    if pts.shape[0] == 0 or p0.shape[0] == 0 or p1.shape[0] == 0:
        return out

    seg_v = p1 - p0
    seg_vv = np.sum(seg_v * seg_v, axis=1)
    seg_vv = np.maximum(seg_vv, EPS)

    for start in range(0, pts.shape[0], max(1, int(chunk_size))):
        stop = min(pts.shape[0], start + max(1, int(chunk_size)))
        P = pts[start:stop]
        W = P[:, None, :] - p0[None, :, :]
        t = np.sum(W * seg_v[None, :, :], axis=2) / seg_vv[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = p0[None, :, :] + t[:, :, None] * seg_v[None, :, :]
        d2 = np.sum((P[:, None, :] - proj) ** 2, axis=2)
        out[start:stop] = np.min(d2, axis=1)

    return out


def estimate_surface_label_radius(
    distance_sq: np.ndarray,
    segment_count: int,
    seed_cells: Optional[List[int]] = None,
) -> float:
    dist = np.sqrt(np.maximum(np.asarray(distance_sq, dtype=float), 0.0))
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return 1.0

    sample_count = int(max(16, min(int(finite.size), max(16, 6 * max(1, int(segment_count))))))
    sample = finite
    if finite.size > sample_count:
        sample = np.partition(finite, sample_count - 1)[:sample_count]

    radius = float(np.percentile(sample, 70.0)) if sample.size else float(np.median(finite))
    if seed_cells:
        seed_ids = np.asarray(
            [int(v) for v in seed_cells if 0 <= int(v) < dist.shape[0]],
            dtype=int,
        )
        if seed_ids.size:
            seed_vals = dist[seed_ids]
            seed_vals = seed_vals[np.isfinite(seed_vals)]
            if seed_vals.size:
                radius = max(radius, float(np.percentile(seed_vals, 65.0)))

    if not math.isfinite(radius) or radius <= EPS:
        radius = float(np.percentile(finite, 50.0)) if finite.size else 1.0
    return float(max(radius, 0.20))


def build_surface_branch_candidate_mask(
    branch: Dict[str, Any],
    cell_centers: np.ndarray,
    label_radius: float,
    parent_distance_sq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    n_cells = int(cell_centers.shape[0])
    empty_mask = np.zeros((n_cells,), dtype=bool)
    inf_cost = np.full((n_cells,), float("inf"), dtype=float)

    points = np.asarray(branch.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
    if points.shape[0] == 0:
        return empty_mask, inf_cost, [], {
            "mask_cell_count": 0,
            "seed_count": 0,
            "radius_limit": 0.0,
            "branch_radius": 0.0,
        }

    if points.shape[0] >= 2:
        branch_distance_sq = min_distance_sq_points_to_segments(cell_centers, points[:-1], points[1:]).astype(float)
    else:
        diff = cell_centers - points[0][None, :]
        branch_distance_sq = np.sum(diff * diff, axis=1).astype(float)
    branch_distance = np.sqrt(np.maximum(branch_distance_sq, 0.0))

    branch_length = float(branch.get("length", polyline_length(points)))
    if points.shape[0] >= 2:
        fallback_local_step = float(np.linalg.norm(points[1] - points[0]))
    else:
        fallback_local_step = 0.0
    local_step = float(branch.get("proximal_local_step", fallback_local_step))
    branch_radius = estimate_surface_label_radius(branch_distance_sq, max(1, points.shape[0] - 1))
    radius_base = max(float(label_radius), float(branch_radius))

    start_point = np.asarray(
        branch.get("proximal_geometric_origin_point", points[0]),
        dtype=float,
    ).reshape(3)
    topological_point = np.asarray(
        branch.get("proximal_topological_point", points[0]),
        dtype=float,
    ).reshape(3)
    start_axis = unit(
        np.asarray(
            branch.get(
                "proximal_direction",
                points[1] - points[0] if points.shape[0] >= 2 else np.zeros((3,), dtype=float),
            ),
            dtype=float,
        ).reshape(3)
    )
    if np.linalg.norm(start_axis) <= EPS and points.shape[0] >= 2:
        start_axis = unit(points[min(1, points.shape[0] - 1)] - points[0])

    outward_axis = unit(
        np.asarray(
            branch.get("proximal_outward_direction", start_axis),
            dtype=float,
        ).reshape(3)
    )
    parent_axis = unit(
        np.asarray(
            branch.get("proximal_parent_tangent", np.zeros((3,), dtype=float)),
            dtype=float,
        ).reshape(3)
    )
    if points.shape[0] >= 2:
        end_axis = unit(points[-1] - points[-2])
    else:
        end_axis = start_axis.copy()
    end_point = np.asarray(points[-1], dtype=float).reshape(3)

    label_id = int(branch.get("label_id", LABEL_OTHER))
    ownership_mode = str(branch.get("topology_ownership_mode", ""))
    topology_role = str(branch.get("topology_role", ""))
    is_interval = ownership_mode == "explicit_interval" or topology_role in ("named_stem", "trunk_path")
    is_short_interval = is_interval and branch_length <= max(3.0 * radius_base, 3.0 * max(local_step, EPS))

    radius_factor = 2.50
    if label_id == LABEL_AORTA_TRUNK:
        radius_factor = 2.85
    elif label_id in (LABEL_CELIAC_TRUNK, LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC):
        radius_factor = 2.95
    elif label_id in (
        LABEL_SUPERIOR_MESENTERIC_ARTERY,
        LABEL_INFERIOR_MESENTERIC_ARTERY,
        LABEL_RIGHT_RENAL,
        LABEL_LEFT_RENAL,
    ):
        radius_factor = 2.65
    elif label_id in (
        LABEL_RIGHT_INTERNAL_ILIAC,
        LABEL_RIGHT_EXTERNAL_ILIAC,
        LABEL_LEFT_INTERNAL_ILIAC,
        LABEL_LEFT_EXTERNAL_ILIAC,
    ):
        radius_factor = 2.90
    if is_interval:
        radius_factor += 0.25
    if is_short_interval:
        radius_factor += 0.55

    radius_limit = max(radius_factor * max(radius_base, EPS), 1.65 * max(local_step, EPS))
    proximal_margin = max(1.10 * max(radius_base, EPS), 0.75 * max(local_step, EPS))
    distal_margin = max(1.20 * max(radius_base, EPS), 0.90 * max(local_step, EPS))
    if is_short_interval:
        proximal_margin = max(proximal_margin, 1.70 * max(radius_base, EPS))
        distal_margin = max(distal_margin, 1.70 * max(radius_base, EPS))

    mask = np.asarray(branch_distance <= radius_limit, dtype=bool)
    start_proj = np.zeros((n_cells,), dtype=float)
    if np.linalg.norm(start_axis) > EPS:
        start_proj = np.sum((cell_centers - start_point[None, :]) * start_axis[None, :], axis=1)
        mask &= np.asarray(start_proj >= -proximal_margin, dtype=bool)

    end_proj = np.zeros((n_cells,), dtype=float)
    if np.linalg.norm(end_axis) > EPS:
        end_proj = np.sum((cell_centers - end_point[None, :]) * end_axis[None, :], axis=1)
        mask &= np.asarray(end_proj <= distal_margin, dtype=bool)

    takeoff_radius = max(2.50 * max(radius_base, EPS), 1.75 * max(local_step, EPS))
    distal_radius = max(2.35 * max(radius_base, EPS), 1.60 * max(local_step, EPS))
    near_takeoff = np.asarray(
        np.sum((cell_centers - start_point[None, :]) ** 2, axis=1) <= (takeoff_radius * takeoff_radius),
        dtype=bool,
    )
    near_topological = np.asarray(
        np.sum((cell_centers - topological_point[None, :]) ** 2, axis=1) <= (takeoff_radius * takeoff_radius),
        dtype=bool,
    )
    near_distal = np.asarray(
        np.sum((cell_centers - end_point[None, :]) ** 2, axis=1) <= (distal_radius * distal_radius),
        dtype=bool,
    )

    if label_id not in (
        LABEL_AORTA_TRUNK,
        LABEL_RIGHT_INTERNAL_ILIAC,
        LABEL_RIGHT_EXTERNAL_ILIAC,
        LABEL_LEFT_INTERNAL_ILIAC,
        LABEL_LEFT_EXTERNAL_ILIAC,
    ) and np.linalg.norm(outward_axis) > EPS:
        outward_proj = np.sum((cell_centers - start_point[None, :]) * outward_axis[None, :], axis=1)
        outward_margin = max(0.75 * max(radius_base, EPS), 0.50 * max(local_step, EPS))
        mask &= np.asarray((outward_proj >= -outward_margin) | near_takeoff | near_topological, dtype=bool)

    parent_distance = None
    parent_penalty = np.zeros((n_cells,), dtype=float)
    if parent_distance_sq is not None:
        parent_distance = np.sqrt(np.maximum(np.asarray(parent_distance_sq, dtype=float), 0.0))
        parent_slack = max(0.35 * max(radius_base, EPS), 0.30 * max(local_step, EPS))
        raw_penalty = np.maximum(branch_distance - parent_distance - parent_slack, 0.0) / max(radius_base, EPS)
        parent_penalty = 0.40 * raw_penalty
        if np.linalg.norm(parent_axis) > EPS:
            parent_proj = np.abs(np.sum((cell_centers - start_point[None, :]) * parent_axis[None, :], axis=1))
            parent_band = np.asarray(parent_proj <= max(2.50 * max(radius_base, EPS), 1.80 * max(local_step, EPS)), dtype=bool)
            parent_penalty = np.where(parent_band | near_takeoff | near_topological, 0.0, parent_penalty)
        else:
            parent_penalty = np.where(near_takeoff | near_topological, 0.0, parent_penalty)

    local_cost = branch_distance / max(radius_base, EPS)
    if np.linalg.norm(start_axis) > EPS:
        local_cost += 0.10 * np.maximum(-start_proj, 0.0) / max(proximal_margin, EPS)
    if np.linalg.norm(end_axis) > EPS:
        local_cost += 0.12 * np.maximum(end_proj, 0.0) / max(distal_margin, EPS)
    local_cost += parent_penalty
    local_cost = np.where(mask, local_cost, float("inf")).astype(float)

    if not np.any(mask):
        relaxed_limit = max(1.35 * radius_limit, 3.00 * max(radius_base, EPS))
        mask = np.asarray(branch_distance <= relaxed_limit, dtype=bool)
        if np.linalg.norm(start_axis) > EPS:
            mask &= np.asarray(start_proj >= -(proximal_margin + radius_base), dtype=bool)
        if np.linalg.norm(end_axis) > EPS:
            mask &= np.asarray(end_proj <= (distal_margin + radius_base), dtype=bool)
        local_cost = np.where(mask, local_cost, float("inf")).astype(float)

    seed_points: List[np.ndarray] = [
        np.asarray(branch.get("proximal_seed_point", start_point), dtype=float).reshape(3),
    ]
    characteristic_step = max(radius_base, local_step, EPS)
    if branch_length > 2.0 * characteristic_step:
        seed_points.append(polyline_point_at_abscissa(points, 0.50 * branch_length))
    if branch_length > 4.0 * characteristic_step:
        seed_points.append(polyline_point_at_abscissa(points, min(0.80 * branch_length, branch_length - 0.75 * characteristic_step)))

    seed_mask = np.asarray(mask, dtype=bool)
    if np.linalg.norm(start_axis) > EPS:
        seed_mask &= np.asarray(start_proj >= -0.25 * proximal_margin, dtype=bool)

    seed_cells: List[int] = []
    seen_seed_ids: set[int] = set()
    for seed_point in seed_points:
        seed_cell = nearest_cell_to_point(cell_centers, seed_point, allowed_mask=seed_mask)
        if seed_cell < 0:
            seed_cell = nearest_cell_to_point(cell_centers, seed_point, allowed_mask=mask)
        if seed_cell < 0:
            seed_cell = nearest_cell_to_point(cell_centers, seed_point, allowed_mask=None)
        if seed_cell >= 0 and int(seed_cell) not in seen_seed_ids:
            seen_seed_ids.add(int(seed_cell))
            seed_cells.append(int(seed_cell))

    return mask.astype(bool), local_cost.astype(float), seed_cells, {
        "mask_cell_count": int(np.count_nonzero(mask)),
        "seed_count": int(len(seed_cells)),
        "radius_limit": float(radius_limit),
        "branch_radius": float(branch_radius),
        "radius_base": float(radius_base),
        "proximal_margin": float(proximal_margin),
        "distal_margin": float(distal_margin),
        "length": float(branch_length),
        "parent_penalty_mean": float(np.mean(parent_penalty[np.isfinite(local_cost)])) if np.any(np.isfinite(local_cost)) else 0.0,
    }


def run_seeded_surface_label_watershed(
    adjacency: List[List[int]],
    label_eligible_masks: Dict[int, np.ndarray],
    label_local_costs: Dict[int, np.ndarray],
    label_seed_lists: Dict[int, List[int]],
    label_depths: Dict[int, int],
) -> np.ndarray:
    import heapq

    n_cells = int(len(adjacency))
    labels = np.full((n_cells,), LABEL_OTHER, dtype=int)
    finalized = np.zeros((n_cells,), dtype=bool)
    heap: List[Tuple[float, Tuple[int, int, int], int, int]] = []

    for lid, seed_list in label_seed_lists.items():
        mask = np.asarray(label_eligible_masks.get(int(lid), np.zeros((n_cells,), dtype=bool)), dtype=bool)
        local_cost = np.asarray(label_local_costs.get(int(lid), np.full((n_cells,), float("inf"), dtype=float)), dtype=float)
        if not np.any(mask):
            continue
        tie_priority = (-int(label_depths.get(int(lid), 0)), int(LABEL_PRIORITY_ORDER.get(int(lid), 999)), int(lid))
        for seed in seed_list:
            seed_i = int(seed)
            if seed_i < 0 or seed_i >= n_cells or not bool(mask[seed_i]) or not math.isfinite(float(local_cost[seed_i])):
                continue
            heapq.heappush(heap, (0.0, tie_priority, int(lid), seed_i))

    while heap:
        total_cost, tie_priority, lid, cell_id = heapq.heappop(heap)
        ci = int(cell_id)
        if ci < 0 or ci >= n_cells or bool(finalized[ci]):
            continue
        mask = np.asarray(label_eligible_masks.get(int(lid), np.zeros((n_cells,), dtype=bool)), dtype=bool)
        local_cost = np.asarray(label_local_costs.get(int(lid), np.full((n_cells,), float("inf"), dtype=float)), dtype=float)
        if not bool(mask[ci]) or not math.isfinite(float(local_cost[ci])):
            continue

        finalized[ci] = True
        labels[ci] = int(lid)
        current_local = float(local_cost[ci])

        for nbr in adjacency[ci]:
            nbr_i = int(nbr)
            if nbr_i < 0 or nbr_i >= n_cells or bool(finalized[nbr_i]) or not bool(mask[nbr_i]):
                continue
            nbr_local = float(local_cost[nbr_i])
            if not math.isfinite(nbr_local):
                continue
            step_cost = 0.05 + 0.50 * (current_local + nbr_local)
            heapq.heappush(heap, (float(total_cost + step_cost), tie_priority, int(lid), nbr_i))

    return labels.astype(int)


def fill_surface_other_by_majority(
    labels: np.ndarray,
    adjacency: List[List[int]],
    label_eligible_masks: Dict[int, np.ndarray],
    label_local_costs: Dict[int, np.ndarray],
    label_depths: Dict[int, int],
    passes: int = 6,
) -> np.ndarray:
    out = np.asarray(labels, dtype=int).copy()
    n_cells = int(out.shape[0])

    for _ in range(max(0, int(passes))):
        prev = out.copy()
        changed = False
        for ci in range(n_cells):
            if int(prev[ci]) != LABEL_OTHER:
                continue

            counts: Dict[int, int] = {}
            best_costs: Dict[int, float] = {}
            any_candidate_counts: Dict[int, int] = {}
            any_candidate_costs: Dict[int, float] = {}

            for nbr in adjacency[int(ci)]:
                lid = int(prev[int(nbr)])
                if lid == LABEL_OTHER:
                    continue

                any_candidate_counts[lid] = int(any_candidate_counts.get(lid, 0) + 1)
                local_cost = np.asarray(
                    label_local_costs.get(int(lid), np.full((n_cells,), float("inf"), dtype=float)),
                    dtype=float,
                )
                any_candidate_costs[lid] = min(float(any_candidate_costs.get(lid, float("inf"))), float(local_cost[int(ci)]))

                eligible_mask = np.asarray(
                    label_eligible_masks.get(int(lid), np.zeros((n_cells,), dtype=bool)),
                    dtype=bool,
                )
                if bool(eligible_mask[int(ci)]):
                    counts[lid] = int(counts.get(lid, 0) + 1)
                    best_costs[lid] = min(float(best_costs.get(lid, float("inf"))), float(local_cost[int(ci)]))

            candidate_counts = counts if counts else any_candidate_counts
            candidate_costs = best_costs if counts else any_candidate_costs
            if not candidate_counts:
                continue

            best_label = min(
                candidate_counts.keys(),
                key=lambda lid: (
                    -int(candidate_counts[lid]),
                    float(candidate_costs.get(int(lid), float("inf"))),
                    -int(label_depths.get(int(lid), 0)),
                    int(LABEL_PRIORITY_ORDER.get(int(lid), 999)),
                    int(lid),
                ),
            )
            if int(candidate_counts[best_label]) >= 2 or len(adjacency[int(ci)]) <= 2:
                out[int(ci)] = int(best_label)
                changed = True
        if not changed:
            break

    return out.astype(int)


def fill_small_other_surface_components(
    labels: np.ndarray,
    adjacency: List[List[int]],
    cell_areas: np.ndarray,
    label_depths: Dict[int, int],
    max_cells: int = 256,
) -> np.ndarray:
    out = np.asarray(labels, dtype=int).copy()
    components = label_connected_components(out, adjacency, LABEL_OTHER)
    if not components:
        return out

    area_arr = np.asarray(cell_areas, dtype=float)
    positive_areas = area_arr[area_arr > 0.0]
    median_area = float(np.median(positive_areas)) if positive_areas.size else 0.0
    area_limit = float(max(0.0, 40.0 * median_area))

    for comp in components:
        comp_ids = np.asarray([int(v) for v in comp], dtype=int)
        comp_area = float(np.sum(area_arr[comp_ids])) if area_arr.size == out.size else 0.0
        if comp_ids.size > max(1, int(max_cells)) and (area_limit <= 0.0 or comp_area > area_limit):
            continue

        counts: Dict[int, int] = {}
        for ci in comp_ids.tolist():
            for nbr in adjacency[int(ci)]:
                lid = int(out[int(nbr)])
                if lid == LABEL_OTHER:
                    continue
                counts[lid] = int(counts.get(lid, 0) + 1)
        if not counts:
            continue

        best_label = min(
            counts.keys(),
            key=lambda lid: (
                -int(counts[lid]),
                -int(label_depths.get(int(lid), 0)),
                int(LABEL_PRIORITY_ORDER.get(int(lid), 999)),
                int(lid),
            ),
        )
        out[comp_ids] = int(best_label)

    return out.astype(int)


def fill_surface_other_components_from_boundary(
    labels: np.ndarray,
    adjacency: List[List[int]],
    relaxed_label_costs: Dict[int, np.ndarray],
    label_depths: Dict[int, int],
) -> np.ndarray:
    import heapq

    out = np.asarray(labels, dtype=int).copy()
    components = label_connected_components(out, adjacency, LABEL_OTHER)
    if not components:
        return out

    for comp in components:
        comp_set = {int(v) for v in comp}
        heap: List[Tuple[float, Tuple[int, int, int], int, int]] = []
        seen_seeds: set[Tuple[int, int]] = set()

        for ci in comp_set:
            for nbr in adjacency[int(ci)]:
                lid = int(out[int(nbr)])
                if lid == LABEL_OTHER:
                    continue
                local_cost = np.asarray(
                    relaxed_label_costs.get(int(lid), np.full((out.shape[0],), float("inf"), dtype=float)),
                    dtype=float,
                )
                if not math.isfinite(float(local_cost[int(ci)])):
                    continue
                seed_key = (int(lid), int(ci))
                if seed_key in seen_seeds:
                    continue
                seen_seeds.add(seed_key)
                tie_priority = (-int(label_depths.get(int(lid), 0)), int(LABEL_PRIORITY_ORDER.get(int(lid), 999)), int(lid))
                heapq.heappush(heap, (float(local_cost[int(ci)]), tie_priority, int(lid), int(ci)))

        if not heap:
            continue

        claimed: Dict[int, int] = {}
        while heap:
            total_cost, tie_priority, lid, cell_id = heapq.heappop(heap)
            ci = int(cell_id)
            if ci not in comp_set or ci in claimed:
                continue

            local_cost = np.asarray(
                relaxed_label_costs.get(int(lid), np.full((out.shape[0],), float("inf"), dtype=float)),
                dtype=float,
            )
            if not math.isfinite(float(local_cost[ci])):
                continue

            claimed[ci] = int(lid)
            current_local = float(local_cost[ci])
            for nbr in adjacency[ci]:
                nbr_i = int(nbr)
                if nbr_i not in comp_set or nbr_i in claimed:
                    continue
                nbr_local = float(local_cost[nbr_i])
                if not math.isfinite(nbr_local):
                    continue
                step_cost = 0.05 + 0.50 * (current_local + nbr_local)
                heapq.heappush(heap, (float(total_cost + step_cost), tie_priority, int(lid), nbr_i))

        for ci, lid in claimed.items():
            out[int(ci)] = int(lid)

    return out.astype(int)


def recover_missing_surface_labels(
    labels: np.ndarray,
    adjacency: List[List[int]],
    label_eligible_masks: Dict[int, np.ndarray],
    label_local_costs: Dict[int, np.ndarray],
    relaxed_label_costs: Dict[int, np.ndarray],
    label_seed_lists: Dict[int, List[int]],
    label_depths: Dict[int, int],
    recovery_margin: float = 0.20,
) -> np.ndarray:
    out = np.asarray(labels, dtype=int).copy()
    n_cells = int(out.shape[0])
    recovery_order = sorted(
        [int(lid) for lid in label_seed_lists.keys()],
        key=lambda lid: (-int(label_depths.get(int(lid), 0)), int(LABEL_PRIORITY_ORDER.get(int(lid), 999)), int(lid)),
    )

    for lid in recovery_order:
        if np.any(out == int(lid)):
            continue

        eligible_mask = np.asarray(
            label_eligible_masks.get(int(lid), np.zeros((n_cells,), dtype=bool)),
            dtype=bool,
        )
        if not np.any(eligible_mask):
            continue
        seed_list = [int(v) for v in label_seed_lists.get(int(lid), [])]
        if not seed_list:
            continue

        rescue_component = connected_components_from_seeds(seed_list, eligible_mask, adjacency)
        if not rescue_component:
            continue
        comp_ids = np.asarray([int(v) for v in rescue_component], dtype=int)
        lid_cost = np.asarray(
            label_local_costs.get(int(lid), np.full((n_cells,), float("inf"), dtype=float)),
            dtype=float,
        )[comp_ids]
        owner_cost = np.full((comp_ids.size,), float("inf"), dtype=float)
        current_labels = np.asarray(out[comp_ids], dtype=int)

        for owner in np.unique(current_labels):
            owner_i = int(owner)
            owner_mask = np.asarray(current_labels == owner_i, dtype=bool)
            if owner_i == LABEL_OTHER:
                owner_cost[owner_mask] = float("inf")
                continue
            owner_field = np.asarray(
                relaxed_label_costs.get(
                    owner_i,
                    label_local_costs.get(int(owner_i), np.full((n_cells,), float("inf"), dtype=float)),
                ),
                dtype=float,
            )
            owner_cost[owner_mask] = owner_field[comp_ids[owner_mask]]

        claim_local = np.asarray(lid_cost <= (owner_cost + float(recovery_margin)), dtype=bool)
        for seed in seed_list:
            claim_local |= np.asarray(comp_ids == int(seed), dtype=bool)
        if not np.any(claim_local):
            continue

        claim_mask = np.zeros((n_cells,), dtype=bool)
        claim_mask[comp_ids[claim_local]] = True
        reclaimed = connected_components_from_seeds(seed_list, claim_mask, adjacency)
        if not reclaimed:
            continue
        out[np.asarray(reclaimed, dtype=int)] = int(lid)

    return out.astype(int)


def connected_component_from_seed(seed_cell: int, allowed_mask: np.ndarray, adjacency: List[List[int]]) -> List[int]:
    n_cells = int(len(adjacency))
    if n_cells == 0:
        return []
    seed = int(max(0, min(seed_cell, n_cells - 1)))
    if not bool(allowed_mask[seed]):
        return [seed]

    visited = {seed}
    stack = [seed]
    component: List[int] = []
    while stack:
        cur = int(stack.pop())
        component.append(cur)
        for nbr in adjacency[cur]:
            nbr_i = int(nbr)
            if nbr_i in visited or not bool(allowed_mask[nbr_i]):
                continue
            visited.add(nbr_i)
            stack.append(nbr_i)
    return component


def label_connected_components(labels: np.ndarray, adjacency: List[List[int]], label_id: int) -> List[List[int]]:
    cell_ids = np.flatnonzero(np.asarray(labels, dtype=int) == int(label_id))
    if cell_ids.size == 0:
        return []

    remaining = set(int(v) for v in cell_ids.tolist())
    components: List[List[int]] = []
    while remaining:
        seed = int(next(iter(remaining)))
        stack = [seed]
        remaining.remove(seed)
        comp: List[int] = []
        while stack:
            cur = int(stack.pop())
            comp.append(cur)
            for nbr in adjacency[cur]:
                nbr_i = int(nbr)
                if nbr_i in remaining and int(labels[nbr_i]) == int(label_id):
                    remaining.remove(nbr_i)
                    stack.append(nbr_i)
        components.append(comp)
    return components


def find_shortest_cell_path_between_sets(
    adjacency: List[List[int]],
    source_cells: set[int],
    target_cells: set[int],
    allowed_mask: np.ndarray,
) -> List[int]:
    if not source_cells or not target_cells:
        return []

    source = {int(v) for v in source_cells if 0 <= int(v) < len(adjacency)}
    target = {int(v) for v in target_cells if 0 <= int(v) < len(adjacency)}
    if not source or not target:
        return []
    if source & target:
        return [int(next(iter(source & target)))]

    queue = [int(v) for v in sorted(source)]
    parent: Dict[int, int] = {int(v): -1 for v in queue}
    head = 0
    meet: Optional[int] = None

    while head < len(queue):
        cur = int(queue[head])
        head += 1
        if cur in target:
            meet = cur
            break
        for nbr in adjacency[cur]:
            nbr_i = int(nbr)
            if nbr_i in parent:
                continue
            if not bool(allowed_mask[nbr_i]):
                continue
            parent[nbr_i] = cur
            queue.append(nbr_i)

    if meet is None:
        return []

    path: List[int] = []
    cur = int(meet)
    while cur != -1:
        path.append(cur)
        cur = int(parent.get(cur, -1))
    path.reverse()
    return path


def choose_majority_neighbor_label(labels: np.ndarray, adjacency: List[List[int]], cell_id: int) -> int:
    counts: Dict[int, int] = {}
    for nbr in adjacency[int(cell_id)]:
        lid = int(labels[int(nbr)])
        counts[lid] = int(counts.get(lid, 0) + 1)
    if not counts:
        return int(labels[int(cell_id)])

    best_label = min(
        counts.keys(),
        key=lambda lid: (-counts[lid], LABEL_PRIORITY_ORDER.get(int(lid), 999), int(lid)),
    )
    return int(best_label)


def smooth_surface_labels(
    labels: np.ndarray,
    adjacency: List[List[int]],
    immutable_cells: set[int],
    passes: int = 2,
) -> np.ndarray:
    out = np.asarray(labels, dtype=int).copy()
    for _ in range(max(0, int(passes))):
        prev = out.copy()
        changed = False
        for ci in range(prev.shape[0]):
            if int(ci) in immutable_cells:
                continue
            nbrs = adjacency[int(ci)]
            if len(nbrs) < 2:
                continue
            counts: Dict[int, int] = {}
            for nbr in nbrs:
                lid = int(prev[int(nbr)])
                counts[lid] = int(counts.get(lid, 0) + 1)
            best_label = choose_majority_neighbor_label(prev, adjacency, int(ci))
            best_count = int(counts.get(best_label, 0))
            cur_label = int(prev[int(ci)])
            if best_label == cur_label:
                continue
            if best_count >= max(2, int(math.ceil(0.60 * len(nbrs)))):
                out[int(ci)] = int(best_label)
                changed = True
        if not changed:
            break
    return out


def derive_surface_point_labels(pd: "vtkPolyData", cell_labels: np.ndarray) -> np.ndarray:
    point_to_cells = build_surface_point_to_cells(pd)
    point_labels = np.full((int(pd.GetNumberOfPoints()),), LABEL_OTHER, dtype=int)

    for pid, cells in enumerate(point_to_cells):
        if not cells:
            continue
        counts: Dict[int, int] = {}
        for ci in cells:
            lid = int(cell_labels[int(ci)])
            counts[lid] = int(counts.get(lid, 0) + 1)
        best = min(
            counts.keys(),
            key=lambda lid: (-counts[lid], LABEL_PRIORITY_ORDER.get(int(lid), 999), int(lid)),
        )
        point_labels[int(pid)] = int(best)
    return point_labels


def compute_surface_label_contact_map(labels: np.ndarray, adjacency: List[List[int]]) -> Dict[int, Dict[int, int]]:
    contact: Dict[int, Dict[int, int]] = {}
    for ci, nbrs in enumerate(adjacency):
        li = int(labels[int(ci)])
        for nbr in nbrs:
            cj = int(nbr)
            if cj <= ci:
                continue
            lj = int(labels[cj])
            if li == lj:
                continue
            contact.setdefault(li, {})
            contact.setdefault(lj, {})
            contact[li][lj] = int(contact[li].get(lj, 0) + 1)
            contact[lj][li] = int(contact[lj].get(li, 0) + 1)
    return contact


def build_surface_label_transfer(
    surface_pd: "vtkPolyData",
    branch_geoms: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    work_warnings = warnings if warnings is not None else []
    n_cells = int(surface_pd.GetNumberOfCells())
    n_points = int(surface_pd.GetNumberOfPoints())
    if n_cells == 0 or n_points == 0:
        cell_labels = np.zeros((n_cells,), dtype=int)
        point_labels = np.zeros((n_points,), dtype=int)
        summary = summarize_surface_label_transfer(
            cell_labels=cell_labels,
            point_labels=point_labels,
            adjacency=[[] for _ in range(n_cells)],
            cell_areas=np.zeros((n_cells,), dtype=float),
            seed_cells={},
            settings={
                "surface_target_labels": [str(name) for name in LABEL_ID_TO_NAME.values()],
                "surface_excluded_labels": [],
                "surface_cell_count": int(n_cells),
                "surface_point_count": int(n_points),
            },
        )
        return cell_labels, point_labels, summary

    cell_centers = compute_polydata_cell_centers_numpy(surface_pd)
    cell_areas = compute_polydata_cell_areas(surface_pd)
    adjacency = build_surface_cell_adjacency(surface_pd)
    label_bank = build_surface_label_segment_bank(branch_geoms)
    if not label_bank:
        work_warnings.append("W_SURFACE_LABEL_BANK_EMPTY: no centerline branch segments available for surface transfer.")
        cell_labels = np.zeros((n_cells,), dtype=int)
        point_labels = derive_surface_point_labels(surface_pd, cell_labels)
        summary = summarize_surface_label_transfer(
            cell_labels=cell_labels,
            point_labels=point_labels,
            adjacency=adjacency,
            cell_areas=cell_areas,
            seed_cells={},
            settings={
                "surface_target_labels": [str(name) for name in LABEL_ID_TO_NAME.values()],
                "surface_excluded_labels": [],
                "surface_cell_count": int(n_cells),
                "surface_point_count": int(n_points),
            },
        )
        return cell_labels, point_labels, summary

    target_label_ids = sorted(
        int(lid)
        for lid, bank in label_bank.items()
        if int(lid) != LABEL_OTHER and bank is not None and int(bank.get("segment_count", 0)) > 0
    )
    if not target_label_ids:
        work_warnings.append("W_SURFACE_NO_TARGET_SEEDS: no centerline branch segments available for surface transfer.")
        cell_labels = np.zeros((n_cells,), dtype=int)
        point_labels = derive_surface_point_labels(surface_pd, cell_labels)
        summary = summarize_surface_label_transfer(
            cell_labels=cell_labels,
            point_labels=point_labels,
            adjacency=adjacency,
            cell_areas=cell_areas,
            seed_cells={},
            settings={
                "surface_target_labels": [str(name) for name in LABEL_ID_TO_NAME.values()],
                "surface_excluded_labels": [],
                "surface_cell_count": int(n_cells),
                "surface_point_count": int(n_points),
            },
        )
        return cell_labels, point_labels, summary

    distance_sq: Dict[int, np.ndarray] = {}
    for lid in target_label_ids:
        bank = label_bank.get(int(lid))
        if bank is None:
            continue
        distance_sq[int(lid)] = min_distance_sq_points_to_segments(cell_centers, bank["segment_p0"], bank["segment_p1"]).astype(float)

    if not distance_sq:
        work_warnings.append("W_SURFACE_NO_DISTANCE_FIELDS: no branch distance fields could be computed for surface transfer.")
        cell_labels = np.zeros((n_cells,), dtype=int)
        point_labels = derive_surface_point_labels(surface_pd, cell_labels)
        summary = summarize_surface_label_transfer(
            cell_labels=cell_labels,
            point_labels=point_labels,
            adjacency=adjacency,
            cell_areas=cell_areas,
            seed_cells={},
            settings={
                "surface_target_labels": [str(name) for name in LABEL_ID_TO_NAME.values()],
                "surface_excluded_labels": [],
                "surface_cell_count": int(n_cells),
                "surface_point_count": int(n_points),
            },
        )
        return cell_labels, point_labels, summary

    label_order = sorted(distance_sq.keys(), key=lambda lid: (LABEL_PRIORITY_ORDER.get(int(lid), 999), int(lid)))
    label_eligible_masks: Dict[int, np.ndarray] = {}
    label_local_costs: Dict[int, np.ndarray] = {}
    label_seed_lists: Dict[int, List[int]] = {}
    label_mask_debug: Dict[str, Any] = {}
    label_parent_ids: Dict[int, Optional[int]] = {}
    label_radius_estimates: Dict[int, float] = {}
    label_candidate_pruned_counts: Dict[str, int] = {}

    for lid in label_order:
        bank = label_bank.get(int(lid))
        if bank is None:
            continue

        branches = list(bank.get("branches", []))
        if not branches:
            continue

        parent_label_id: Optional[int] = None
        for branch in branches:
            raw_parent = branch.get("topology_parent_label_id")
            if raw_parent is None:
                continue
            parent_label_id = int(raw_parent)
            if parent_label_id == int(lid):
                parent_label_id = None
            if parent_label_id is not None:
                break
        label_parent_ids[int(lid)] = (None if parent_label_id is None else int(parent_label_id))

        label_radius = estimate_surface_label_radius(
            distance_sq[int(lid)],
            int(bank.get("segment_count", 0)),
            seed_cells=None,
        )
        label_radius_estimates[int(lid)] = float(label_radius)

        eligible_mask = np.zeros((n_cells,), dtype=bool)
        local_cost = np.full((n_cells,), float("inf"), dtype=float)
        seed_list: List[int] = []
        branch_debug_items: List[Dict[str, Any]] = []
        parent_distance_sq = None if parent_label_id is None else distance_sq.get(int(parent_label_id))

        for branch in branches:
            branch_mask, branch_cost, branch_seeds, branch_debug = build_surface_branch_candidate_mask(
                branch=branch,
                cell_centers=cell_centers,
                label_radius=float(label_radius),
                parent_distance_sq=parent_distance_sq,
            )
            if np.any(branch_mask):
                eligible_mask |= np.asarray(branch_mask, dtype=bool)
                local_cost = np.minimum(local_cost, np.asarray(branch_cost, dtype=float))
            seed_list.extend(int(v) for v in branch_seeds)
            branch_debug_items.append(
                {
                    "mask_cell_count": int(branch_debug.get("mask_cell_count", 0)),
                    "seed_count": int(branch_debug.get("seed_count", 0)),
                    "radius_limit": float(branch_debug.get("radius_limit", 0.0)),
                    "branch_radius": float(branch_debug.get("branch_radius", 0.0)),
                    "radius_base": float(branch_debug.get("radius_base", 0.0)),
                    "length": float(branch_debug.get("length", 0.0)),
                }
            )

        seed_list = [int(v) for v in sorted(set(seed_list))]
        if not seed_list and np.any(eligible_mask):
            eligible_ids = np.flatnonzero(np.asarray(eligible_mask, dtype=bool))
            if eligible_ids.size:
                seed_list = [int(eligible_ids[int(np.argmin(local_cost[eligible_ids]))])]

        if not seed_list:
            work_warnings.append(
                f"W_SURFACE_LABEL_SEED_FAILED: unable to place a surface seed for {LABEL_ID_TO_NAME.get(int(lid), str(lid))}."
            )
            continue

        keep_component = connected_components_from_seeds(seed_list, eligible_mask, adjacency)
        connected_mask = np.zeros((n_cells,), dtype=bool)
        connected_mask[np.asarray(keep_component, dtype=int)] = True
        pruned_mask = np.asarray(eligible_mask & (~connected_mask), dtype=bool)
        label_candidate_pruned_counts[LABEL_ID_TO_NAME.get(int(lid), str(lid))] = int(np.count_nonzero(pruned_mask))
        eligible_mask = np.asarray(connected_mask, dtype=bool)
        local_cost = np.where(eligible_mask, local_cost, float("inf")).astype(float)

        label_eligible_masks[int(lid)] = eligible_mask.astype(bool)
        label_local_costs[int(lid)] = local_cost.astype(float)
        label_seed_lists[int(lid)] = seed_list
        label_mask_debug[LABEL_ID_TO_NAME.get(int(lid), str(lid))] = {
            "eligible_cell_count": int(np.count_nonzero(eligible_mask)),
            "seed_count": int(len(seed_list)),
            "parent_label": (None if parent_label_id is None else LABEL_ID_TO_NAME.get(int(parent_label_id), str(parent_label_id))),
            "radius_estimate": float(label_radius),
            "branch_masks": list(branch_debug_items),
        }

    def _label_depth(label_id: int, memo: Dict[int, int]) -> int:
        lid_i = int(label_id)
        if lid_i in memo:
            return int(memo[lid_i])
        parent = label_parent_ids.get(lid_i)
        if parent is None or int(parent) == lid_i:
            memo[lid_i] = 0
        else:
            memo[lid_i] = int(1 + _label_depth(int(parent), memo))
        return int(memo[lid_i])

    depth_memo: Dict[int, int] = {}
    label_depths = {
        int(lid): int(_label_depth(int(lid), depth_memo))
        for lid in label_order
        if int(lid) in label_seed_lists
    }

    used_seed_cells: set[int] = set()
    seed_resolution_order = sorted(
        [int(lid) for lid in label_order if int(lid) in label_seed_lists],
        key=lambda lid: (-int(label_depths.get(int(lid), 0)), int(LABEL_PRIORITY_ORDER.get(int(lid), 999)), int(lid)),
    )
    for lid in seed_resolution_order:
        eligible_mask = np.asarray(label_eligible_masks.get(int(lid), np.zeros((n_cells,), dtype=bool)), dtype=bool)
        local_cost = np.asarray(label_local_costs.get(int(lid), np.full((n_cells,), float("inf"), dtype=float)), dtype=float)
        original_seeds = [int(v) for v in label_seed_lists.get(int(lid), [])]
        resolved_seeds: List[int] = []

        for seed in original_seeds:
            seed_i = int(seed)
            if seed_i < 0 or seed_i >= n_cells or seed_i in used_seed_cells or not bool(eligible_mask[seed_i]):
                continue
            if not math.isfinite(float(local_cost[seed_i])):
                continue
            resolved_seeds.append(seed_i)
            used_seed_cells.add(seed_i)

        if not resolved_seeds and np.any(eligible_mask):
            candidate_ids = np.flatnonzero(np.asarray(eligible_mask, dtype=bool))
            if candidate_ids.size:
                ranked_ids = candidate_ids[np.argsort(local_cost[candidate_ids], kind="mergesort")]
                for cand in ranked_ids.tolist():
                    cand_i = int(cand)
                    if cand_i in used_seed_cells or not math.isfinite(float(local_cost[cand_i])):
                        continue
                    resolved_seeds.append(cand_i)
                    used_seed_cells.add(cand_i)
                    break

        if not resolved_seeds:
            resolved_seeds = [int(v) for v in original_seeds[:1]]
        label_seed_lists[int(lid)] = resolved_seeds

    cell_labels = run_seeded_surface_label_watershed(
        adjacency=adjacency,
        label_eligible_masks=label_eligible_masks,
        label_local_costs=label_local_costs,
        label_seed_lists=label_seed_lists,
        label_depths=label_depths,
    )
    cell_labels = fill_surface_other_by_majority(
        labels=cell_labels,
        adjacency=adjacency,
        label_eligible_masks=label_eligible_masks,
        label_local_costs=label_local_costs,
        label_depths=label_depths,
        passes=6,
    )
    relaxed_label_costs = {
        int(lid): (
            np.sqrt(np.maximum(np.asarray(distance_sq[int(lid)], dtype=float), 0.0))
            / max(float(label_radius_estimates.get(int(lid), 1.0)), EPS)
        ).astype(float)
        for lid in label_order
        if int(lid) in distance_sq
    }
    cell_labels = fill_surface_other_components_from_boundary(
        labels=cell_labels,
        adjacency=adjacency,
        relaxed_label_costs=relaxed_label_costs,
        label_depths=label_depths,
    )
    cell_labels = recover_missing_surface_labels(
        labels=cell_labels,
        adjacency=adjacency,
        label_eligible_masks=label_eligible_masks,
        label_local_costs=label_local_costs,
        relaxed_label_costs=relaxed_label_costs,
        label_seed_lists=label_seed_lists,
        label_depths=label_depths,
        recovery_margin=0.20,
    )
    cell_labels = fill_small_other_surface_components(
        labels=cell_labels,
        adjacency=adjacency,
        cell_areas=cell_areas,
        label_depths=label_depths,
        max_cells=256,
    )

    point_labels = derive_surface_point_labels(surface_pd, cell_labels)
    seed_cells = {
        int(lid): int(seed_list[0])
        for lid, seed_list in label_seed_lists.items()
        if seed_list
    }
    summary_settings = {
        "surface_target_labels": [LABEL_ID_TO_NAME.get(int(lid), str(lid)) for lid in label_order],
        "surface_excluded_labels": [],
        "surface_cell_count": int(n_cells),
        "surface_point_count": int(n_points),
        "seed_label_count": int(len(seed_cells)),
        "distance_labels": [LABEL_ID_TO_NAME.get(int(lid), str(lid)) for lid in label_order],
        "transfer_mode": "centerline_scaffold_seeded_graph_growth",
        "label_eligible_cell_counts": {
            str(name): int(info.get("eligible_cell_count", 0)) for name, info in label_mask_debug.items()
        },
        "label_seed_counts": {str(name): int(info.get("seed_count", 0)) for name, info in label_mask_debug.items()},
        "label_parent_map": {str(name): info.get("parent_label") for name, info in label_mask_debug.items()},
        "label_depths": {
            LABEL_ID_TO_NAME.get(int(lid), str(lid)): int(label_depths.get(int(lid), 0)) for lid in label_order if int(lid) in label_depths
        },
        "label_radius_estimates": {
            LABEL_ID_TO_NAME.get(int(lid), str(lid)): float(label_radius_estimates.get(int(lid), 0.0))
            for lid in label_order
            if int(lid) in label_radius_estimates
        },
        "label_candidate_pruned_counts": {
            str(name): int(v) for name, v in label_candidate_pruned_counts.items()
        },
    }
    summary = summarize_surface_label_transfer(
        cell_labels=cell_labels,
        point_labels=point_labels,
        adjacency=adjacency,
        cell_areas=cell_areas,
        seed_cells=seed_cells,
        settings=summary_settings,
    )

    validation = dict(summary.get("validation", {}))
    for missing_name in list(validation.get("missing_target_labels", [])):
        work_warnings.append(f"W_SURFACE_TARGET_EMPTY: no surface cells assigned to {missing_name}.")
    for unexpected_name in list(validation.get("non_target_surface_labels_present", [])):
        work_warnings.append(f"W_SURFACE_UNEXPECTED_LABEL_PRESENT: unexpected surface label present: {unexpected_name}.")

    return cell_labels.astype(int), point_labels.astype(int), summary


def annotate_surface_polydata_for_combined_output(
    surface_pd: "vtkPolyData",
    branch_geoms: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Tuple["vtkPolyData", Dict[str, Any]]:
    out = clone_polydata(surface_pd)
    cell_labels, _, summary = build_surface_label_transfer(out, branch_geoms, warnings=warnings)

    n_cells = int(out.GetNumberOfCells())
    cd = out.GetCellData()
    add_scalar_array_to_cell_data(cd, "BranchId", [int(v) for v in cell_labels.tolist()], vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchName", [LABEL_ID_TO_NAME.get(int(v), "other") for v in cell_labels.tolist()])
    add_scalar_array_to_cell_data(cd, "BranchLength", [0.0] * n_cells, vtk_type=vtk.VTK_DOUBLE)
    add_string_array_to_cell_data(cd, "GeometryType", ["surface"] * n_cells)

    return out, summary


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
    trunk_child_systems: Optional[List[Dict[str, Any]]] = None,
    right_iliac_system: Optional[Dict[str, Any]] = None,
    left_iliac_system: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    trunk_edges = path_edge_keys(trunk_path)
    right_iliac_edges = path_edge_keys(right_iliac_nodes)
    left_iliac_edges = path_edge_keys(left_iliac_nodes)
    right_renal_edges = path_edge_keys(right_renal_nodes)
    left_renal_edges = path_edge_keys(left_renal_nodes)
    named_system_labels = {
        rooted_child_system_key(right_iliac_system): LABEL_RIGHT_ILIAC,
        rooted_child_system_key(left_iliac_system): LABEL_LEFT_ILIAC,
    }
    named_system_labels = {k: v for k, v in named_system_labels.items() if k != (-1, -1)}
    system_entries: List[Dict[str, Any]] = []
    for system in trunk_child_systems or []:
        system_key = rooted_child_system_key(system)
        system_entries.append(
            {
                "key": system_key,
                "system": system,
                "nodes": rooted_child_system_node_set(system, include_takeoff=True),
                "stem_edges": path_edge_keys([int(n) for n in system.get("named_stem_path", system.get("stem_path", []))]),
                "named_label_id": named_system_labels.get(system_key),
            }
        )

    def classify_chain(nodes: List[int]) -> Tuple[int, str, Dict[str, Any]]:
        edges = path_edge_keys(nodes)
        if edges and edges.issubset(trunk_edges):
            return LABEL_AORTA_TRUNK, LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK], {
                "role": "trunk_path",
                "parent_takeoff": None,
                "parent_stem_start": None,
                "parent_label_id": LABEL_AORTA_TRUNK,
                "parent_name": LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
            }
        if edges and edges.issubset(right_iliac_edges):
            return LABEL_RIGHT_ILIAC, LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC], {
                "role": "named_stem",
                "parent_takeoff": int(right_iliac_system["takeoff"]) if right_iliac_system is not None else None,
                "parent_stem_start": int(right_iliac_system["stem_start"]) if right_iliac_system is not None else None,
                "parent_label_id": LABEL_RIGHT_ILIAC,
                "parent_name": LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC],
            }
        if edges and edges.issubset(left_iliac_edges):
            return LABEL_LEFT_ILIAC, LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC], {
                "role": "named_stem",
                "parent_takeoff": int(left_iliac_system["takeoff"]) if left_iliac_system is not None else None,
                "parent_stem_start": int(left_iliac_system["stem_start"]) if left_iliac_system is not None else None,
                "parent_label_id": LABEL_LEFT_ILIAC,
                "parent_name": LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC],
            }
        if edges and edges.issubset(right_renal_edges):
            return LABEL_RIGHT_RENAL, LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL], {
                "role": "named_branch",
                "parent_takeoff": int(right_renal_takeoff) if right_renal_takeoff is not None else None,
                "parent_stem_start": int(right_renal_nodes[1]) if len(right_renal_nodes) >= 2 else None,
                "parent_label_id": LABEL_RIGHT_RENAL,
                "parent_name": LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL],
            }
        if edges and edges.issubset(left_renal_edges):
            return LABEL_LEFT_RENAL, LABEL_ID_TO_NAME[LABEL_LEFT_RENAL], {
                "role": "named_branch",
                "parent_takeoff": int(left_renal_takeoff) if left_renal_takeoff is not None else None,
                "parent_stem_start": int(left_renal_nodes[1]) if len(left_renal_nodes) >= 2 else None,
                "parent_label_id": LABEL_LEFT_RENAL,
                "parent_name": LABEL_ID_TO_NAME[LABEL_LEFT_RENAL],
            }

        node_set = set(int(n) for n in nodes)
        for entry in system_entries:
            if not node_set.issubset(entry["nodes"]):
                continue
            named_label_id = entry["named_label_id"]
            if named_label_id is not None and edges and edges.issubset(entry["stem_edges"]):
                return named_label_id, LABEL_ID_TO_NAME[named_label_id], {
                    "role": "named_stem",
                    "parent_takeoff": int(entry["system"]["takeoff"]),
                    "parent_stem_start": int(entry["system"]["stem_start"]),
                    "parent_label_id": int(named_label_id),
                    "parent_name": LABEL_ID_TO_NAME[named_label_id],
                }
            return LABEL_OTHER, LABEL_ID_TO_NAME[LABEL_OTHER], {
                "role": "direct_child_descendant",
                "parent_takeoff": int(entry["system"]["takeoff"]),
                "parent_stem_start": int(entry["system"]["stem_start"]),
                "parent_label_id": int(named_label_id) if named_label_id is not None else None,
                "parent_name": LABEL_ID_TO_NAME[named_label_id] if named_label_id is not None else "trunk_child_system",
            }
        return LABEL_OTHER, LABEL_ID_TO_NAME[LABEL_OTHER], {
            "role": "unassigned",
            "parent_takeoff": None,
            "parent_stem_start": None,
            "parent_label_id": None,
            "parent_name": None,
        }

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
    classified: List[Tuple[int, List[int], int, str, Dict[str, Any]]] = []
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

        label_id, label_name, topology_info = classify_chain(nodes)
        classified.append((label_priority.get(label_id, 99), nodes, label_id, label_name, topology_info))

    classified.sort(key=lambda item: (item[0], tuple(item[1])))

    for _, nodes, label_id, label_name, topology_info in classified:
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
                topology_role=str(topology_info.get("role", "unassigned")),
                topology_parent_takeoff=topology_info.get("parent_takeoff"),
                topology_takeoff_node=topology_info.get("parent_takeoff"),
                topology_parent_attachment_node=topology_info.get("parent_takeoff"),
                topology_ownership_start_node=(int(nodes[0]) if nodes else None),
                topology_parent_stem_start=topology_info.get("parent_stem_start"),
                topology_parent_label_id=topology_info.get("parent_label_id"),
                topology_parent_name=topology_info.get("parent_name"),
            )
        )

    return branch_geoms


def build_prelabeled_branch_geometries(
    adjacency_full: Dict[int, Dict[int, float]],
    pts_canonical: np.ndarray,
    dist_from_inlet: Dict[int, float],
    topology: Dict[str, Any],
) -> List[Dict[str, Any]]:
    branch_paths = {
        str(name): [int(n) for n in nodes]
        for name, nodes in dict(topology.get("branch_paths", {})).items()
        if len(nodes) >= 2
    }
    branch_takeoffs = {str(name): value for name, value in dict(topology.get("branch_takeoffs", {})).items()}
    branch_parent_names = {
        str(name): (None if value is None else str(value))
        for name, value in dict(topology.get("branch_parent_names", {})).items()
    }
    branch_ownership_modes = {
        str(name): str(value) for name, value in dict(topology.get("branch_ownership_modes", {})).items()
    }
    branch_ownership_start_nodes = {
        str(name): (None if value is None else int(value))
        for name, value in dict(topology.get("branch_ownership_start_nodes", {})).items()
    }
    branch_parent_attachment_nodes = {
        str(name): (None if value is None else int(value))
        for name, value in dict(topology.get("branch_parent_attachment_nodes", {})).items()
    }
    direct_branch_takeoffs = dict(topology.get("direct_branch_takeoffs", {}))
    owned_edges_input = dict(topology.get("branch_owned_edges", {}))

    owner_edge_sets: Dict[str, set[Tuple[int, int]]] = {}
    if owned_edges_input:
        for name, pairs in owned_edges_input.items():
            edge_set: set[Tuple[int, int]] = set()
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                edge_set.add(edge_key(int(pair[0]), int(pair[1])))
            if edge_set:
                owner_edge_sets[str(name)] = edge_set
    else:
        owner_edge_sets = {str(name): path_edge_keys(nodes) for name, nodes in branch_paths.items()}

    parent_name_map = {
        LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]: None,
        LABEL_ID_TO_NAME[LABEL_CELIAC_TRUNK]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_SUPERIOR_MESENTERIC_ARTERY]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_INFERIOR_MESENTERIC_ARTERY]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_LEFT_RENAL]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]: LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK],
        LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC],
        LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC],
        LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC],
        LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC]: LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC],
    }
    parent_name_map.update({str(k): (None if v is None else str(v)) for k, v in branch_parent_names.items()})

    topology_role_map = {
        LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]: "trunk_path",
        LABEL_ID_TO_NAME[LABEL_CELIAC_TRUNK]: "named_stem",
        LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]: "named_stem",
        LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]: "named_stem",
        LABEL_ID_TO_NAME[LABEL_OTHER]: "unassigned",
    }

    landmark_nodes = {
        "Inlet": int(topology.get("inlet_node", -1)) if topology.get("inlet_node") is not None else -1,
        "Bifurcation": int(topology.get("bifurcation_node", -1)) if topology.get("bifurcation_node") is not None else -1,
        "RightRenalOrigin": int(direct_branch_takeoffs.get("right_renal_artery", -1))
        if direct_branch_takeoffs.get("right_renal_artery") is not None
        else -1,
        "LeftRenalOrigin": int(direct_branch_takeoffs.get("left_renal_artery", -1))
        if direct_branch_takeoffs.get("left_renal_artery") is not None
        else -1,
        "CeliacTakeoff": int(topology.get("celiac_takeoff_node", -1)) if topology.get("celiac_takeoff_node") is not None else -1,
        "CeliacSplit": int(topology.get("celiac_split_node", -1)) if topology.get("celiac_split_node") is not None else -1,
        "RightCommonIliacSplit": int(topology.get("right_common_iliac_split_node", -1))
        if topology.get("right_common_iliac_split_node") is not None
        else -1,
        "LeftCommonIliacSplit": int(topology.get("left_common_iliac_split_node", -1))
        if topology.get("left_common_iliac_split_node") is not None
        else -1,
    }

    owner_names_by_priority = sorted(
        owner_edge_sets.keys(),
        key=lambda name: (
            LABEL_PRIORITY_ORDER.get(LABEL_NAME_TO_ID.get(str(name), LABEL_OTHER), 999),
            0 if str(name) != LABEL_ID_TO_NAME[LABEL_OTHER] else 1,
            str(name),
        ),
    )

    branch_geoms: List[Dict[str, Any]] = []
    for owner_name in owner_names_by_priority:
        owner_edges = {edge_key(int(a), int(b)) for a, b in owner_edge_sets.get(str(owner_name), set())}
        if not owner_edges:
            continue
        owner_adj = build_adjacency_from_edge_keys(adjacency_full, owner_edges)
        if not owner_adj:
            continue

        chains = build_branch_chains_from_graph(owner_adj)
        oriented_chains: List[List[int]] = []
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
            oriented_chains.append([int(n) for n in nodes])

        oriented_chains.sort(
            key=lambda nodes: (
                float(dist_from_inlet.get(int(nodes[0]), float("inf"))) if nodes else float("inf"),
                tuple(int(n) for n in nodes),
            )
        )

        label_id = int(LABEL_NAME_TO_ID.get(str(owner_name), LABEL_OTHER))
        parent_name = parent_name_map.get(str(owner_name))
        topology_info = {
            "role": str(topology_role_map.get(str(owner_name), "named_branch" if label_id != LABEL_OTHER else "unassigned")),
            "parent_takeoff": branch_takeoffs.get(str(owner_name)),
            "parent_attachment": branch_parent_attachment_nodes.get(str(owner_name)),
            "ownership_start": branch_ownership_start_nodes.get(str(owner_name)),
            "parent_label_id": (None if parent_name is None else int(LABEL_NAME_TO_ID.get(str(parent_name), LABEL_OTHER))),
            "parent_name": parent_name,
            "ownership_mode": str(branch_ownership_modes.get(str(owner_name), "owned_edges")),
            "owned_edge_count": int(len(owner_edges)),
        }

        for nodes in oriented_chains:
            node_to_index = {int(n): idx for idx, n in enumerate(nodes)}
            landmark_point_ids: Dict[str, int] = {}
            for key, node in landmark_nodes.items():
                if int(node) >= 0 and int(node) in node_to_index:
                    landmark_point_ids[str(key)] = int(node_to_index[int(node)])

            branch_geoms.append(
                {
                    "label_id": int(label_id),
                    "name": str(owner_name),
                    "points": pts_canonical[np.asarray(nodes, dtype=int)],
                    "landmark_point_ids": landmark_point_ids,
                    "node_ids": [int(n) for n in nodes],
                    "topology_role": str(topology_info.get("role", "unassigned")),
                    "topology_parent_takeoff": topology_info.get("parent_takeoff"),
                    "topology_takeoff_node": topology_info.get("parent_takeoff"),
                    "topology_parent_attachment_node": topology_info.get("parent_attachment"),
                    "topology_ownership_start_node": (
                        topology_info.get("ownership_start")
                        if topology_info.get("ownership_start") is not None
                        else (int(nodes[0]) if nodes else None)
                    ),
                    "topology_parent_stem_start": None,
                    "topology_parent_label_id": topology_info.get("parent_label_id"),
                    "topology_parent_name": topology_info.get("parent_name"),
                    "topology_ownership_mode": str(topology_info.get("ownership_mode", "owned_edges")),
                    "topology_owned_edge_count": int(topology_info.get("owned_edge_count", 0)),
                }
            )
    return branch_geoms


def enrich_branch_geometries_with_proximal_anchors(
    branch_geoms: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    work_warnings = warnings if warnings is not None else []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for br in branch_geoms:
        grouped.setdefault(str(br.get("name", LABEL_ID_TO_NAME[LABEL_OTHER])), []).append(br)

    for br in branch_geoms:
        pts = np.asarray(br.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
        if pts.shape[0] == 0:
            continue

        branch_length = float(polyline_length(pts))
        local_step = float(np.linalg.norm(pts[1] - pts[0])) if pts.shape[0] >= 2 else 0.0
        topological_point = pts[0].astype(float)
        geometric_point = topological_point.copy()
        parent_tangent = unit(pts[1] - pts[0]) if pts.shape[0] >= 2 else np.zeros((3,), dtype=float)
        geometric_source = "topological_start"
        parent_projection_distance = float("nan")
        parent_projection_abscissa = float("nan")

        parent_name = br.get("topology_parent_name")
        lock_geometric_origin = bool(br.get("force_proximal_geometric_origin_to_topological", False))
        best_parent_proj: Optional[Dict[str, Any]] = None
        if parent_name is not None:
            for parent_br in grouped.get(str(parent_name), []):
                parent_pts = np.asarray(parent_br.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
                proj = project_point_to_polyline(topological_point, parent_pts)
                if proj is None:
                    continue
                if best_parent_proj is None or float(proj["distance_sq"]) < float(best_parent_proj["distance_sq"]):
                    best_parent_proj = {**proj, "points": parent_pts}
            if best_parent_proj is not None:
                parent_tangent = unit(np.asarray(best_parent_proj["tangent"], dtype=float).reshape(3))
                parent_projection_distance = math.sqrt(max(float(best_parent_proj["distance_sq"]), 0.0))
                parent_projection_abscissa = float(best_parent_proj["abscissa"])
                if lock_geometric_origin:
                    geometric_point = topological_point.copy()
                    geometric_source = "explicit_geometric_separation"
                else:
                    geometric_point = np.asarray(best_parent_proj["point"], dtype=float).reshape(3)
                    geometric_source = "projected_to_parent"
                if (
                    math.isfinite(parent_projection_distance)
                    and branch_length > EPS
                    and parent_projection_distance > max(2.5, 0.90 * branch_length)
                ):
                    work_warnings.append(
                        f"W_BRANCH_GEOMETRIC_ORIGIN_FAR: {str(br.get('name', 'other'))} parent projection distance={parent_projection_distance:.3f}."
                    )
        if lock_geometric_origin and best_parent_proj is None:
            geometric_point = topological_point.copy()
            geometric_source = "explicit_geometric_separation"

        if branch_length > EPS:
            seed_offset = min(max(2.0 * max(local_step, EPS), 0.08 * branch_length), 0.35 * branch_length)
            seed_point = polyline_point_at_abscissa(pts, seed_offset)
        else:
            seed_offset = 0.0
            seed_point = topological_point.copy()

        proximal_direction = unit(seed_point - geometric_point)
        if np.linalg.norm(proximal_direction) < EPS and pts.shape[0] >= 2:
            proximal_direction = unit(pts[1] - pts[0])
        if np.linalg.norm(proximal_direction) < EPS:
            proximal_direction = parent_tangent.copy()

        outward_direction = proximal_direction.copy()
        if np.linalg.norm(parent_tangent) > EPS:
            projected = proximal_direction - float(np.dot(proximal_direction, parent_tangent)) * parent_tangent
            if np.linalg.norm(projected) > EPS:
                outward_direction = unit(projected)

        br["proximal_topological_point"] = topological_point.astype(float)
        br["proximal_geometric_origin_point"] = geometric_point.astype(float)
        br["proximal_geometric_origin_source"] = str(geometric_source)
        br["proximal_parent_projection_distance"] = (
            None if not math.isfinite(parent_projection_distance) else float(parent_projection_distance)
        )
        br["proximal_parent_abscissa"] = (
            None if not math.isfinite(parent_projection_abscissa) else float(parent_projection_abscissa)
        )
        br["proximal_direction"] = proximal_direction.astype(float)
        br["proximal_outward_direction"] = outward_direction.astype(float)
        br["proximal_parent_tangent"] = parent_tangent.astype(float)
        br["proximal_seed_point"] = np.asarray(seed_point, dtype=float).reshape(3)
        br["proximal_seed_offset"] = float(seed_offset)
        br["proximal_local_step"] = float(local_step)
        br.setdefault("topology_takeoff_node", br.get("topology_parent_takeoff"))
        br.setdefault("topology_parent_attachment_node", br.get("topology_parent_takeoff"))
        br.setdefault(
            "topology_ownership_start_node",
            int(br["node_ids"][0]) if list(br.get("node_ids", [])) else None,
        )

    return branch_geoms


def summarize_branch_proximal_anchors(branch_geoms: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for idx, br in enumerate(branch_geoms):
        name = str(br.get("name", LABEL_ID_TO_NAME[LABEL_OTHER]))
        out.setdefault(name, []).append(
            {
                "polyline_index": int(idx),
                "branch_id": int(br.get("label_id", LABEL_OTHER)),
                "parent_name": (None if br.get("topology_parent_name") is None else str(br.get("topology_parent_name"))),
                "topology_takeoff_node": (
                    None if br.get("topology_takeoff_node") is None else int(br.get("topology_takeoff_node"))
                ),
                "topology_parent_attachment_node": (
                    None
                    if br.get("topology_parent_attachment_node") is None
                    else int(br.get("topology_parent_attachment_node"))
                ),
                "topology_ownership_start_node": (
                    None
                    if br.get("topology_ownership_start_node") is None
                    else int(br.get("topology_ownership_start_node"))
                ),
                "topological_start_point": _jsonable_point(br.get("proximal_topological_point", np.zeros((3,), dtype=float))),
                "geometric_origin_point": _jsonable_point(
                    br.get("proximal_geometric_origin_point", br.get("proximal_topological_point", np.zeros((3,), dtype=float)))
                ),
                "geometric_origin_source": str(br.get("proximal_geometric_origin_source", "topological_start")),
                "parent_projection_distance": (
                    None
                    if br.get("proximal_parent_projection_distance") is None
                    else float(br.get("proximal_parent_projection_distance"))
                ),
                "direction": _jsonable_point(br.get("proximal_direction", np.zeros((3,), dtype=float))),
                "parent_tangent": _jsonable_point(br.get("proximal_parent_tangent", np.zeros((3,), dtype=float))),
            }
        )
    return out


def build_output_centerlines_polydata(
    branch_geoms: List[Dict[str, Any]],
) -> "vtkPolyData":
    out = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    cell_label: List[int] = []
    cell_name: List[str] = []
    cell_length: List[float] = []

    global_pid = 0
    for br in branch_geoms:
        pts = np.asarray(br["points"], dtype=float)
        if pts.shape[0] < 2:
            continue

        start_pid = global_pid
        for i in range(pts.shape[0]):
            points.InsertNextPoint(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
            global_pid += 1

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(pts.shape[0])
        for i in range(pts.shape[0]):
            polyline.GetPointIds().SetId(i, start_pid + i)
        lines.InsertNextCell(polyline)

        s = compute_abscissa(pts)
        cell_label.append(int(br["label_id"]))
        cell_name.append(str(br["name"]))
        cell_length.append(float(s[-1]) if s.size else 0.0)

    out.SetPoints(points)
    out.SetLines(lines)

    cd = out.GetCellData()
    add_scalar_array_to_cell_data(cd, "BranchId", cell_label, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchName", cell_name)
    add_scalar_array_to_cell_data(cd, "BranchLength", cell_length, vtk_type=vtk.VTK_DOUBLE)
    add_string_array_to_cell_data(cd, "GeometryType", ["centerline"] * len(cell_label))

    return out


def build_combined_surface_centerlines_polydata(
    surface_pd: "vtkPolyData",
    centerlines_pd: "vtkPolyData",
) -> "vtkPolyData":
    surface_tagged = clone_polydata(surface_pd)

    app = vtk.vtkAppendPolyData()
    app.AddInputData(surface_tagged)
    app.AddInputData(centerlines_pd)
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
        "BranchId",
        [int(v) for v in get_cell_scalar_array_values(surface_tagged, "BranchId", 0)]
        + [int(v) for v in get_cell_scalar_array_values(centerlines_pd, "BranchId", 0)],
        vtk_type=vtk.VTK_INT,
    )
    add_string_array_to_cell_data(
        combined_cd,
        "BranchName",
        get_cell_string_array_values(surface_tagged, "BranchName", "other")
        + get_cell_string_array_values(centerlines_pd, "BranchName", "other"),
    )
    add_scalar_array_to_cell_data(
        combined_cd,
        "BranchLength",
        [0.0] * n_surface_cells + get_cell_scalar_array_values(centerlines_pd, "BranchLength", 0.0),
        vtk_type=vtk.VTK_DOUBLE,
    )
    add_string_array_to_cell_data(
        combined_cd,
        "GeometryType",
        ["surface"] * n_surface_cells + ["centerline"] * n_centerline_cells,
    )
    return out


def summarize_centerline_branch_geometries(branch_geoms: List[Dict[str, Any]]) -> Dict[str, Any]:
    length_by_branch_label: Dict[str, float] = {}
    count_by_branch_label: Dict[str, int] = {}
    branch_rows: List[Dict[str, Any]] = []

    for idx, br in enumerate(branch_geoms):
        length = float(polyline_length(np.asarray(br.get("points", np.zeros((0, 3), dtype=float)), dtype=float)))
        branch_name = str(br.get("name", LABEL_ID_TO_NAME[LABEL_OTHER]))

        length_by_branch_label[branch_name] = float(length_by_branch_label.get(branch_name, 0.0) + length)
        count_by_branch_label[branch_name] = int(count_by_branch_label.get(branch_name, 0) + 1)

        branch_rows.append(
            {
                "index": int(idx),
                "branch_id": int(br.get("label_id", LABEL_OTHER)),
                "branch_name": branch_name,
                "topology_role": str(br.get("topology_role", "unassigned")),
                "topology_parent_name": (None if br.get("topology_parent_name") is None else str(br.get("topology_parent_name"))),
                "topology_ownership_mode": str(br.get("topology_ownership_mode", "owned_edges")),
                "length": float(length),
                "point_count": int(np.asarray(br.get("points", np.zeros((0, 3), dtype=float)), dtype=float).shape[0]),
            }
        )

    return {
        "length_by_branch_label": {k: float(v) for k, v in length_by_branch_label.items()},
        "count_by_branch_label": {k: int(v) for k, v in count_by_branch_label.items()},
        "branches": branch_rows,
    }


def _jsonable_point(value: Any) -> List[float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _jsonable_face_regions(mapped_face_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for region in mapped_face_regions:
        face_id = int(region.get("face_id", -1))
        out[str(face_id)] = {
            "face_id": int(face_id),
            "name": str(region.get("name", f"face_{face_id}")),
            "terminal_type": str(region.get("terminal_type", "outlet")),
            "cap_id": int(region.get("cap_id", 0)),
            "mapped_cap_id": (None if region.get("mapped_cap_id") is None else int(region.get("mapped_cap_id"))),
            "cell_count": int(region.get("cell_count", 0)),
            "total_area": float(region.get("total_area", 0.0)),
            "diameter_eq": float(region.get("diameter_eq", 0.0)),
            "centroid": _jsonable_point(region.get("centroid", np.zeros((3,), dtype=float))),
            "mean_normal": _jsonable_point(region.get("mean_normal", np.zeros((3,), dtype=float))),
        }
    return out


def _jsonable_terminal_matches(terminal_matches: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, match in terminal_matches.items():
        out[str(name)] = {
            "face_id": int(match.get("face_id", -1)),
            "name": str(match.get("name", name)),
            "terminal_type": str(match.get("terminal_type", "outlet")),
            "endpoint_node": int(match.get("endpoint_node", -1)),
            "endpoint_xyz": _jsonable_point(match.get("endpoint_xyz", np.zeros((3,), dtype=float))),
            "centroid": _jsonable_point(match.get("centroid", np.zeros((3,), dtype=float))),
            "cap_id": int(match.get("cap_id", 0)),
            "distance": float(match.get("distance", 0.0)),
            "confidence": float(match.get("confidence", 0.0)),
        }
    return out


def _jsonable_geometric_junctions(junctions: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, info in dict(junctions or {}).items():
        item = dict(info or {})
        out[str(key)] = {
            "junction_name": str(item.get("junction_name", key)),
            "status": str(item.get("status", "unknown")),
            "used_geometric_refinement": bool(item.get("used_geometric_refinement", False)),
            "confidence": float(item.get("confidence", 0.0)),
            "bracketed_transition": bool(item.get("bracketed_transition", False)),
            "topology_candidate_node": (
                None if item.get("topology_candidate_node") is None else int(item.get("topology_candidate_node"))
            ),
            "topology_candidate_point_raw": (
                None if item.get("topology_candidate_point_raw") is None else _jsonable_point(item.get("topology_candidate_point_raw"))
            ),
            "topology_candidate_point_canonical": (
                None
                if item.get("topology_candidate_point_canonical") is None
                else _jsonable_point(item.get("topology_candidate_point_canonical"))
            ),
            "refined_parent_point_raw": (
                None if item.get("refined_parent_point_raw") is None else _jsonable_point(item.get("refined_parent_point_raw"))
            ),
            "refined_parent_point_canonical": (
                None
                if item.get("refined_parent_point_canonical") is None
                else _jsonable_point(item.get("refined_parent_point_canonical"))
            ),
            "refined_split_point_raw": (
                None if item.get("refined_split_point_raw") is None else _jsonable_point(item.get("refined_split_point_raw"))
            ),
            "refined_split_point_canonical": (
                None
                if item.get("refined_split_point_canonical") is None
                else _jsonable_point(item.get("refined_split_point_canonical"))
            ),
            "refined_child_points_raw": {
                str(name): _jsonable_point(value)
                for name, value in dict(item.get("refined_child_points_raw", {}) or {}).items()
            },
            "refined_child_points_canonical": {
                str(name): _jsonable_point(value)
                for name, value in dict(item.get("refined_child_points_canonical", {}) or {}).items()
            },
            "scan_axis_raw": (
                None if item.get("scan_axis_raw") is None else _jsonable_point(item.get("scan_axis_raw"))
            ),
            "scan_axis_canonical": (
                None if item.get("scan_axis_canonical") is None else _jsonable_point(item.get("scan_axis_canonical"))
            ),
            "search_window": {
                str(k): float(v) for k, v in dict(item.get("search_window", {}) or {}).items()
            },
            "classification": {
                "component_count": int(dict(item.get("classification", {}) or {}).get("component_count", 0)),
                "matched_children": int(dict(item.get("classification", {}) or {}).get("matched_children", 0)),
                "match_threshold": float(dict(item.get("classification", {}) or {}).get("match_threshold", 0.0)),
                "last_single_index": (
                    None
                    if dict(item.get("classification", {}) or {}).get("last_single_index") is None
                    else int(dict(item.get("classification", {}) or {}).get("last_single_index"))
                ),
                "first_split_index": (
                    None
                    if dict(item.get("classification", {}) or {}).get("first_split_index") is None
                    else int(dict(item.get("classification", {}) or {}).get("first_split_index"))
                ),
                "coarse_samples": [
                    {
                        "offset": float(sample.get("offset", 0.0)),
                        "is_split": bool(sample.get("is_split", False)),
                        "component_count": int(sample.get("component_count", 0)),
                        "matched_children": int(sample.get("matched_children", 0)),
                    }
                    for sample in list(dict(item.get("classification", {}) or {}).get("coarse_samples", []))
                ],
            },
        }
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
    parser.add_argument(
        "--terminal_face_map",
        type=str,
        default=None,
        help="Optional JSON sidecar mapping ModelFaceID values to named terminals. Defaults to face_id_to_name.json next to the input VTP.",
    )
    args = parser.parse_args()

    input_path = _resolve_user_path(args.input)
    surface_with_centerlines_path = _resolve_user_path(args.output_surface_with_centerlines)
    centerlines_output_arg = args.output if args.output is not None else args.output_centerlines
    centerlines_output_path = _resolve_user_path(centerlines_output_arg)
    meta_path = _resolve_user_path(args.metadata) if args.metadata is not None else ""
    debug_raw_path = _resolve_user_path(args.debug_raw_centerlines) if args.debug_raw_centerlines is not None else ""
    terminal_face_map_path = resolve_terminal_face_map_path(input_path, args.terminal_face_map)

    warnings: List[str] = []
    terminal_face_map_source = ""
    terminal_face_map: Dict[int, Dict[str, Any]] = {}
    face_regions_raw: Dict[int, Dict[str, Any]] = {}
    mapped_face_regions: List[Dict[str, Any]] = []
    terminal_matches: Dict[str, Dict[str, Any]] = {}
    topology: Dict[str, Any] = {}
    horizontal_frame_info: Dict[str, Any] = {}
    centerline_runtime_info: Dict[str, Any] = {}
    terminal_detection_mode = "automatic_fallback"

    try:
        surface = load_vtp(input_path)
        surface_tri = clean_and_triangulate_surface(surface)

        surface_pts = get_points_numpy(surface_tri)

        try:
            terminal_face_map, terminal_face_map_source = load_terminal_face_map(terminal_face_map_path, warnings=warnings)
        except Exception as face_map_exc:
            warnings.append(f"W_TERMINAL_FACE_MAP_LOAD_FAILED: {face_map_exc}")
            terminal_face_map = {}
            terminal_face_map_source = ""

        inlet_term: Optional[TerminationLoop] = None
        inlet_term_conf = 0.0
        axis_si = np.array([0.0, 0.0, 1.0], dtype=float)
        terms: List[TerminationLoop] = []
        mode = "automatic_fallback"
        use_prelabeled = False
        inlet_face_region: Optional[Dict[str, Any]] = None

        if terminal_face_map:
            face_regions_raw, mapped_face_regions, inlet_face_region = build_prelabeled_terminal_regions(
                surface_tri,
                terminal_face_map,
                warnings=warnings,
            )
            outlet_face_regions = [region for region in mapped_face_regions if str(region.get("terminal_type", "")) != "inlet"]
            if inlet_face_region is not None and outlet_face_regions:
                terminal_detection_mode = "prelabeled_model_face_id"
                use_prelabeled = True
                mode = terminal_detection_mode
                inlet_term = face_region_to_termination_loop(inlet_face_region)
                inlet_term_conf = 1.0
                terms = [face_region_to_termination_loop(region) for region in mapped_face_regions]
                term_centers = [np.asarray(region["centroid"], dtype=float) for region in outlet_face_regions]
                centerlines, centerline_runtime_info = compute_centerlines_vmtk(surface_tri, inlet_term.center, term_centers, warnings)
            else:
                warnings.append("W_PRELABELED_INCOMPLETE: pre-labeled ModelFaceID workflow was unavailable or incomplete; falling back to automatic termination inference.")

        if not use_prelabeled:
            terminal_detection_mode = "automatic_fallback"
            terms, mode = detect_terminations(surface_tri, warnings)
            if len(terms) < 2:
                raise RuntimeError("Failed to detect enough terminations (need >=2) to seed centerlines.")

            inlet_term, inlet_term_conf, axis_si = choose_inlet_termination(terms, surface_pts, warnings)
            if inlet_term is None:
                raise RuntimeError("Inlet termination could not be determined.")

            term_centers = [t.center for t in terms if np.linalg.norm(t.center - inlet_term.center) > 1e-8]
            centerlines, centerline_runtime_info = compute_centerlines_vmtk(surface_tri, inlet_term.center, term_centers, warnings)

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

        if len(analysis_nodes) < len(adjacency_full):
            warnings.append(
                f"W_CENTERLINE_EXTRA_COMPONENTS: preserved {len(adjacency_full) - len(analysis_nodes)} nodes outside the inlet-connected component; anatomy inference uses the inlet-connected scaffold."
            )

        deg = node_degrees(adjacency)
        endpoints_all = [n for n, d in deg.items() if d == 1]
        if len(endpoints_all) < 2:
            raise RuntimeError("Centerline graph has insufficient endpoints.")

        if use_prelabeled:
            terminal_matches = assign_face_regions_to_centerline_endpoints(
                endpoints=endpoints_all,
                pts=cl_pts,
                mapped_regions=mapped_face_regions,
                warnings=warnings,
            )
            inlet_match = terminal_matches.get("abdominal_aorta_inlet")
            if inlet_match is None:
                raise RuntimeError("Pre-labeled workflow failed to map the abdominal_aorta_inlet face to a centerline endpoint.")

            inlet_node = int(inlet_match["endpoint_node"])
            dist, prev = dijkstra(adjacency, inlet_node)
            topology = infer_prelabeled_branch_topology(
                adjacency=adjacency,
                pts=cl_pts,
                inlet_node=inlet_node,
                terminal_matches=terminal_matches,
                dist=dist,
                prev=prev,
                warnings=warnings,
            )
            geometric_junctions_raw = refine_prelabeled_geometric_bifurcations(
                surface_pd=surface_tri,
                pts=cl_pts,
                topology=topology,
                warnings=warnings,
            )
            topology["geometric_junctions_raw"] = geometric_junctions_raw

            bif_node = int(topology["bifurcation_node"])
            trunk_path = [int(n) for n in topology.get("trunk_path", [])]
            if len(trunk_path) < 2:
                raise RuntimeError("Pre-labeled workflow failed to reconstruct the abdominal aorta trunk path.")

            right_hint_nodes = [int(n) for n in topology.get("right_orientation_hint_nodes", [])]
            left_hint_nodes = [int(n) for n in topology.get("left_orientation_hint_nodes", [])]
            if not right_hint_nodes or not left_hint_nodes:
                raise RuntimeError("Pre-labeled workflow failed to recover right/left iliac orientation hints.")

            right_hint_centroid = np.mean(cl_pts[np.asarray(right_hint_nodes, dtype=int)], axis=0)
            left_hint_centroid = np.mean(cl_pts[np.asarray(left_hint_nodes, dtype=int)], axis=0)
            horizontal_hint = np.asarray(right_hint_centroid - left_hint_centroid, dtype=float).reshape(3)
            if np.linalg.norm(horizontal_hint) < EPS:
                horizontal_hint = None

            inlet_pt = np.asarray(cl_pts[inlet_node], dtype=float)
            aortic_geom = dict(geometric_junctions_raw.get("aortic_bifurcation", {}) or {})
            bif_pt = np.asarray(
                aortic_geom.get("refined_split_point_raw", cl_pts[bif_node]),
                dtype=float,
            ).reshape(3)
            R, origin, frame_conf = build_canonical_transform(
                inlet_pt=inlet_pt,
                bif_pt=bif_pt,
                iliac_ep_a_pt=np.asarray(right_hint_centroid, dtype=float),
                iliac_ep_b_pt=np.asarray(left_hint_centroid, dtype=float),
                all_pts=cl_pts,
                warnings=warnings,
                horizontal_hint=horizontal_hint,
            )
            cl_pts_c = apply_transform_points(cl_pts, R, origin)

            right_hint_x = float(np.mean(cl_pts_c[np.asarray(right_hint_nodes, dtype=int), 0]))
            left_hint_x = float(np.mean(cl_pts_c[np.asarray(left_hint_nodes, dtype=int), 0]))
            right_left_flip_applied = False
            if right_hint_x < left_hint_x:
                right_left_flip_applied = True
                R_flipped = np.asarray(R, dtype=float).copy()
                R_flipped[0, :] *= -1.0
                R_flipped[1, :] *= -1.0
                R = R_flipped
                cl_pts_c = apply_transform_points(cl_pts, R, origin)
                right_hint_x = float(np.mean(cl_pts_c[np.asarray(right_hint_nodes, dtype=int), 0]))
                left_hint_x = float(np.mean(cl_pts_c[np.asarray(left_hint_nodes, dtype=int), 0]))

            superior_flip_applied = False
            inlet_z = float(cl_pts_c[inlet_node, 2])
            bif_z = float(transform_single_point(bif_pt, R, origin)[2])
            if inlet_z < bif_z:
                superior_flip_applied = True
                R_flipped = np.asarray(R, dtype=float).copy()
                R_flipped[1, :] *= -1.0
                R_flipped[2, :] *= -1.0
                R = R_flipped
                cl_pts_c = apply_transform_points(cl_pts, R, origin)
                inlet_z = float(cl_pts_c[inlet_node, 2])
                bif_z = float(transform_single_point(bif_pt, R, origin)[2])

            surface_tri_c = apply_transform_to_polydata(surface_tri, R, origin)
            geometric_junctions_canonical = transform_geometric_junctions(geometric_junctions_raw, R, origin)
            topology["geometric_junctions_canonical"] = geometric_junctions_canonical
            direct_branch_takeoffs = dict(topology.get("direct_branch_takeoffs", {}))
            rr_origin_node = direct_branch_takeoffs.get("right_renal_artery")
            lr_origin_node = direct_branch_takeoffs.get("left_renal_artery")

            landmarks: Dict[str, Any] = {
                "Inlet": np.asarray(cl_pts_c[inlet_node], dtype=float),
                "Bifurcation": transform_single_point(bif_pt, R, origin),
            }
            if aortic_geom:
                aortic_c = dict(geometric_junctions_canonical.get("aortic_bifurcation", {}) or {})
                split_point = aortic_c.get("refined_split_point_canonical")
                child_points = dict(aortic_c.get("refined_child_points_canonical", {}) or {})
                if split_point is not None:
                    landmarks["AorticBifurcation"] = np.asarray(split_point, dtype=float).reshape(3)
                    landmarks["Bifurcation"] = np.asarray(split_point, dtype=float).reshape(3)
                if LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC] in child_points:
                    landmarks["RightCommonIliacStart"] = np.asarray(child_points[LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]], dtype=float).reshape(3)
                if LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC] in child_points:
                    landmarks["LeftCommonIliacStart"] = np.asarray(child_points[LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]], dtype=float).reshape(3)
            right_split_c = dict(geometric_junctions_canonical.get("right_common_iliac_split", {}) or {})
            if right_split_c.get("refined_split_point_canonical") is not None:
                landmarks["RightCommonIliacSplit"] = np.asarray(right_split_c["refined_split_point_canonical"], dtype=float).reshape(3)
            for label_name, landmark_name in (
                (LABEL_ID_TO_NAME[LABEL_RIGHT_INTERNAL_ILIAC], "RightInternalIliacStart"),
                (LABEL_ID_TO_NAME[LABEL_RIGHT_EXTERNAL_ILIAC], "RightExternalIliacStart"),
            ):
                if label_name in dict(right_split_c.get("refined_child_points_canonical", {}) or {}):
                    landmarks[landmark_name] = np.asarray(right_split_c["refined_child_points_canonical"][label_name], dtype=float).reshape(3)
            left_split_c = dict(geometric_junctions_canonical.get("left_common_iliac_split", {}) or {})
            if left_split_c.get("refined_split_point_canonical") is not None:
                landmarks["LeftCommonIliacSplit"] = np.asarray(left_split_c["refined_split_point_canonical"], dtype=float).reshape(3)
            for label_name, landmark_name in (
                (LABEL_ID_TO_NAME[LABEL_LEFT_INTERNAL_ILIAC], "LeftInternalIliacStart"),
                (LABEL_ID_TO_NAME[LABEL_LEFT_EXTERNAL_ILIAC], "LeftExternalIliacStart"),
            ):
                if label_name in dict(left_split_c.get("refined_child_points_canonical", {}) or {}):
                    landmarks[landmark_name] = np.asarray(left_split_c["refined_child_points_canonical"][label_name], dtype=float).reshape(3)
            if rr_origin_node is not None:
                landmarks["RightRenalOrigin"] = np.asarray(cl_pts_c[int(rr_origin_node)], dtype=float)
            if lr_origin_node is not None:
                landmarks["LeftRenalOrigin"] = np.asarray(cl_pts_c[int(lr_origin_node)], dtype=float)

            horizontal_frame_info = {
                "source": "prelabeled_terminal_faces",
                "confidence": 1.0,
                "frame_orthonormality_confidence": float(frame_conf),
                "laterality_confidence": 1.0,
                "laterality_source": "prelabeled_terminal_faces",
                "right_hint_centroid_raw": _jsonable_point(right_hint_centroid),
                "left_hint_centroid_raw": _jsonable_point(left_hint_centroid),
                "right_hint_mean_x_canonical": float(right_hint_x),
                "left_hint_mean_x_canonical": float(left_hint_x),
                "inlet_z_canonical": float(inlet_z),
                "bifurcation_z_canonical": float(bif_z),
                "right_left_flip_applied": bool(right_left_flip_applied),
                "superior_flip_applied": bool(superior_flip_applied),
            }

            preserved_branch_geoms = build_prelabeled_branch_geometries(
                adjacency_full=adjacency_full,
                pts_canonical=cl_pts_c,
                dist_from_inlet=dist,
                topology=topology,
            )
            if not preserved_branch_geoms:
                raise RuntimeError("Failed to construct branch-preserving centerline scaffold output for the pre-labeled workflow.")
            preserved_branch_geoms = apply_geometric_junctions_to_prelabeled_branch_geometries(
                preserved_branch_geoms,
                geometric_junctions_canonical,
                warnings=warnings,
            )
            preserved_branch_geoms = enrich_branch_geometries_with_proximal_anchors(preserved_branch_geoms, warnings=warnings)

            branch_counts: Dict[str, int] = {}
            for br in preserved_branch_geoms:
                branch_counts[br["name"]] = int(branch_counts.get(br["name"], 0) + 1)
            branch_geometry_summary = summarize_centerline_branch_geometries(preserved_branch_geoms)
            branch_anchor_summary = summarize_branch_proximal_anchors(preserved_branch_geoms)

            surface_tagged_pd, surface_label_summary = annotate_surface_polydata_for_combined_output(
                surface_pd=surface_tri_c,
                branch_geoms=preserved_branch_geoms,
                warnings=warnings,
            )
            centerlines_out_pd = build_output_centerlines_polydata(branch_geoms=preserved_branch_geoms)
            combined_out_pd = build_combined_surface_centerlines_polydata(
                surface_pd=surface_tagged_pd,
                centerlines_pd=centerlines_out_pd,
            )
            attach_landmarks_to_polydata_field_data(centerlines_out_pd, landmarks)
            attach_landmarks_to_polydata_field_data(combined_out_pd, landmarks)

            write_vtp(combined_out_pd, surface_with_centerlines_path, binary=True)
            write_vtp(centerlines_out_pd, centerlines_output_path, binary=True)

            if meta_path:
                os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
                inlet_face_stats = next(
                    (region for region in mapped_face_regions if str(region.get("terminal_type", "")) == "inlet"),
                    None,
                )
                outlet_face_stats = [
                    region for region in mapped_face_regions if str(region.get("terminal_type", "")) != "inlet"
                ]
                meta = {
                    "status": "success",
                    "input_vtp": os.path.abspath(input_path),
                    "output_surface_with_centerlines_vtp": os.path.abspath(surface_with_centerlines_path),
                    "output_centerlines_vtp": os.path.abspath(centerlines_output_path),
                    "mode": str(terminal_detection_mode),
                    "warnings": [str(w) for w in warnings],
                    "terminal_detection_mode": str(terminal_detection_mode),
                    "terminal_face_map_path": os.path.abspath(terminal_face_map_path) if terminal_face_map_path else "",
                    "sidecar_mapping_used": {
                        "source": str(terminal_face_map_source),
                        "mapping": {
                            str(fid): {
                                "name": str(info.get("name", f"face_{fid}")),
                                "cap_id": (None if info.get("cap_id") is None else int(info.get("cap_id"))),
                                "terminal_type": str(info.get("terminal_type", "outlet")),
                            }
                            for fid, info in sorted(terminal_face_map.items())
                        },
                    },
                    "detected_face_stats_per_mapped_face": _jsonable_face_regions(mapped_face_regions),
                    "chosen_inlet_face": (
                        None
                        if inlet_face_stats is None
                        else {
                            "face_id": int(inlet_face_stats.get("face_id", -1)),
                            "name": str(inlet_face_stats.get("name", "abdominal_aorta_inlet")),
                            "centroid": _jsonable_point(inlet_face_stats.get("centroid", np.zeros((3,), dtype=float))),
                        }
                    ),
                    "outlet_faces": [
                        {
                            "face_id": int(region.get("face_id", -1)),
                            "name": str(region.get("name", f"face_{idx}")),
                            "centroid": _jsonable_point(region.get("centroid", np.zeros((3,), dtype=float))),
                        }
                        for idx, region in enumerate(outlet_face_stats)
                    ],
                    "prelabeled_terminal_matches": _jsonable_terminal_matches(terminal_matches),
                    "centerline_runtime": json.loads(json.dumps(centerline_runtime_info, default=str)),
                    "canonical_frame_summary": dict(horizontal_frame_info),
                    "identified_bifurcation_node": int(topology.get("bifurcation_node", -1)),
                    "identified_bifurcation_point_xyz_canonical": _jsonable_point(landmarks.get("Bifurcation", np.zeros((3,), dtype=float))),
                    "identified_common_iliac_split_nodes": {
                        "right": int(topology.get("right_common_iliac_split_node", -1)),
                        "left": int(topology.get("left_common_iliac_split_node", -1)),
                    },
                    "identified_renal_takeoff_nodes": {
                        "right": (None if rr_origin_node is None else int(rr_origin_node)),
                        "left": (None if lr_origin_node is None else int(lr_origin_node)),
                    },
                    "geometric_junction_refinement": _jsonable_geometric_junctions(geometric_junctions_canonical),
                    "branch_names": sorted(str(name) for name in branch_counts.keys()),
                    "branch_counts": {k: int(v) for k, v in branch_counts.items()},
                    "branch_summary": dict(branch_geometry_summary),
                    "branch_proximal_anchors": dict(branch_anchor_summary),
                    "centerline_length_by_branch": {
                        k: float(v) for k, v in dict(branch_geometry_summary.get("length_by_branch_label", {})).items()
                    },
                    "surface_cell_counts_by_branch": {
                        k: int(v)
                        for k, v in dict(surface_label_summary.get("cell_counts", {})).items()
                        if int(v) > 0
                    },
                    "landmarks_xyz_canonical": {k: _jsonable_point(v) for k, v in landmarks.items()},
                    "transform": {
                        "R_rows": [[float(x) for x in row] for row in np.asarray(R, dtype=float).reshape(3, 3)],
                        "origin": _jsonable_point(origin),
                        "flipped_for_ap": False,
                    },
                    "junction_nodes": {
                        "aortic_bifurcation_node": int(topology.get("bifurcation_node", -1)),
                        "right_common_iliac_split_node": int(topology.get("right_common_iliac_split_node", -1)),
                        "left_common_iliac_split_node": int(topology.get("left_common_iliac_split_node", -1)),
                        "aortic_bifurcation_point_xyz_canonical": _jsonable_point(landmarks.get("AorticBifurcation", landmarks.get("Bifurcation", np.zeros((3,), dtype=float)))),
                        "right_common_iliac_start_xyz_canonical": (
                            None if landmarks.get("RightCommonIliacStart") is None else _jsonable_point(landmarks.get("RightCommonIliacStart"))
                        ),
                        "left_common_iliac_start_xyz_canonical": (
                            None if landmarks.get("LeftCommonIliacStart") is None else _jsonable_point(landmarks.get("LeftCommonIliacStart"))
                        ),
                        "right_common_iliac_split_point_xyz_canonical": (
                            None if landmarks.get("RightCommonIliacSplit") is None else _jsonable_point(landmarks.get("RightCommonIliacSplit"))
                        ),
                        "left_common_iliac_split_point_xyz_canonical": (
                            None if landmarks.get("LeftCommonIliacSplit") is None else _jsonable_point(landmarks.get("LeftCommonIliacSplit"))
                        ),
                        "celiac_takeoff_node": (
                            None if topology.get("celiac_takeoff_node") is None else int(topology.get("celiac_takeoff_node"))
                        ),
                        "celiac_parent_attachment_node": (
                            None
                            if topology.get("celiac_parent_attachment_node") is None
                            else int(topology.get("celiac_parent_attachment_node"))
                        ),
                        "celiac_split_node": (
                            None if topology.get("celiac_split_node") is None else int(topology.get("celiac_split_node"))
                        ),
                    },
                    "renals_found": {
                        "right": bool(LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL] in topology.get("branch_paths", {})),
                        "left": bool(LABEL_ID_TO_NAME[LABEL_LEFT_RENAL] in topology.get("branch_paths", {})),
                    },
                    "prelabeled_topology": {
                        "inlet_node": int(topology.get("inlet_node", -1)),
                        "bifurcation_node": int(topology.get("bifurcation_node", -1)),
                        "right_common_iliac_split_node": int(topology.get("right_common_iliac_split_node", -1)),
                        "left_common_iliac_split_node": int(topology.get("left_common_iliac_split_node", -1)),
                        "celiac_takeoff_node": (
                            None if topology.get("celiac_takeoff_node") is None else int(topology.get("celiac_takeoff_node"))
                        ),
                        "celiac_parent_attachment_node": (
                            None
                            if topology.get("celiac_parent_attachment_node") is None
                            else int(topology.get("celiac_parent_attachment_node"))
                        ),
                        "celiac_split_node": (
                            None if topology.get("celiac_split_node") is None else int(topology.get("celiac_split_node"))
                        ),
                        "branch_paths": {
                            str(k): [int(n) for n in v] for k, v in dict(topology.get("branch_paths", {})).items()
                        },
                        "branch_paths_raw": {
                            str(k): [int(n) for n in v] for k, v in dict(topology.get("branch_paths_raw", {})).items()
                        },
                        "branch_takeoffs": {
                            str(k): (None if v is None else int(v)) for k, v in dict(topology.get("branch_takeoffs", {})).items()
                        },
                        "branch_parent_attachment_nodes": {
                            str(k): (None if v is None else int(v))
                            for k, v in dict(topology.get("branch_parent_attachment_nodes", {})).items()
                        },
                        "branch_terminals": {
                            str(k): [str(x) for x in vals] for k, vals in dict(topology.get("branch_terminals", {})).items()
                        },
                        "branch_parent_names": {
                            str(k): (None if v is None else str(v)) for k, v in dict(topology.get("branch_parent_names", {})).items()
                        },
                        "branch_ownership_start_nodes": {
                            str(k): (None if v is None else int(v))
                            for k, v in dict(topology.get("branch_ownership_start_nodes", {})).items()
                        },
                        "branch_ownership_modes": {
                            str(k): str(v) for k, v in dict(topology.get("branch_ownership_modes", {})).items()
                        },
                        "branch_owned_edge_counts": {
                            str(k): int(v) for k, v in dict(topology.get("branch_owned_edge_counts", {})).items()
                        },
                        "geometric_junction_refinement": _jsonable_geometric_junctions(topology.get("geometric_junctions_canonical", {})),
                        "direct_branch_takeoffs": {
                            str(k): (None if v is None else int(v))
                            for k, v in dict(topology.get("direct_branch_takeoffs", {})).items()
                        },
                        "direct_branch_parent_attachment_nodes": {
                            str(k): (None if v is None else int(v))
                            for k, v in dict(topology.get("direct_branch_parent_attachment_nodes", {})).items()
                        },
                    },
                    "dropped_or_merged_branches": [str(v) for v in topology.get("dropped_terminal_faces", [])],
                    "fallback_usage": {
                        "used": False,
                        "reason": "",
                    },
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

            return 0

        # Inlet node as nearest endpoint to inlet termination center
        inlet_node, _ = pick_inlet_node_from_endpoints(endpoints_all, cl_pts, inlet_term.center, inlet_term_conf, warnings)
        if inlet_node < 0:
            raise RuntimeError("Failed to identify inlet node on centerlines.")

        # Identify iliac pair and bifurcation
        bif_node, ep_a, ep_b, _, dist, prev, iliac_selection = choose_iliac_endpoints_and_bifurcation(
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

        # Trunk path (inlet -> bif)
        trunk_path = path_to_root(prev, inlet_node, bif_node)
        if not trunk_path:
            raise RuntimeError("Failed to reconstruct trunk path inlet->bifurcation.")
        trunk_set = set(trunk_path)
        trunk_child_systems = build_direct_child_systems_for_parent_path(trunk_path, cl_pts, dist, prev)
        iliac_system_a = find_rooted_child_system_by_key(trunk_child_systems, iliac_selection.get("system_a")) or iliac_selection.get("system_a")
        iliac_system_b = find_rooted_child_system_by_key(trunk_child_systems, iliac_selection.get("system_b")) or iliac_selection.get("system_b")
        if iliac_system_a is None and bif_node is not None and ep_a is not None:
            child_map = build_rooted_child_map(prev)
            iliac_system_a = find_rooted_child_system_for_endpoint(child_map, cl_pts, dist, prev, bif_node, ep_a)
        if iliac_system_b is None and bif_node is not None and ep_b is not None:
            child_map = build_rooted_child_map(prev)
            iliac_system_b = find_rooted_child_system_for_endpoint(child_map, cl_pts, dist, prev, bif_node, ep_b)
        if iliac_system_a is None or iliac_system_b is None:
            warnings.append(
                "W_ILIAC_SYSTEM_SELECTION_INCOMPLETE: rooted iliac system recovery was incomplete; falling back to representative endpoint paths where required."
            )

        # Define the named common iliac stems from the major-split-aware named stem path,
        # not the raw first-split stem path.
        iliac_main_a = [int(n) for n in iliac_system_a.get("named_stem_path", iliac_system_a.get("stem_path", []))] if iliac_system_a is not None else []
        iliac_main_b = [int(n) for n in iliac_system_b.get("named_stem_path", iliac_system_b.get("stem_path", []))] if iliac_system_b is not None else []
        if not iliac_main_a and ep_a is not None:
            path_a_full = path_to_root(prev, inlet_node, ep_a)
            if path_a_full:
                if bif_node not in path_a_full:
                    warnings.append("W_BIF_NOT_ON_ILIAC_PATH: bif not found on one iliac endpoint path; using representative endpoint fallback.")
                idx_a = path_a_full.index(bif_node) if bif_node in path_a_full else max(0, len(path_a_full) - 2)
                iliac_path_a = path_a_full[idx_a:]
                bp_a = first_branchpoint_on_path(iliac_path_a, deg, start_index=1)
                iliac_main_a = iliac_path_a[: iliac_path_a.index(bp_a) + 1] if bp_a is not None and bp_a in iliac_path_a else iliac_path_a
        if not iliac_main_b and ep_b is not None:
            path_b_full = path_to_root(prev, inlet_node, ep_b)
            if path_b_full:
                if bif_node not in path_b_full:
                    warnings.append("W_BIF_NOT_ON_ILIAC_PATH: bif not found on one iliac endpoint path; using representative endpoint fallback.")
                idx_b = path_b_full.index(bif_node) if bif_node in path_b_full else max(0, len(path_b_full) - 2)
                iliac_path_b = path_b_full[idx_b:]
                bp_b = first_branchpoint_on_path(iliac_path_b, deg, start_index=1)
                iliac_main_b = iliac_path_b[: iliac_path_b.index(bp_b) + 1] if bp_b is not None and bp_b in iliac_path_b else iliac_path_b
        iliac_excluded_nodes = rooted_child_system_node_set(iliac_system_a, include_takeoff=False)
        iliac_excluded_nodes.update(rooted_child_system_node_set(iliac_system_b, include_takeoff=False))

        # Build preliminary transform (z superior, x between iliac endpoints)
        inlet_pt = cl_pts[inlet_node]
        bif_pt = cl_pts[bif_node]
        ep_a_pt = cl_pts[iliac_main_a[-1]] if iliac_main_a else cl_pts[ep_a]
        ep_b_pt = cl_pts[iliac_main_b[-1]] if iliac_main_b else cl_pts[ep_b]

        R_provisional, origin, _ = build_canonical_transform(inlet_pt, bif_pt, ep_a_pt, ep_b_pt, cl_pts, warnings)
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
            excluded_system_nodes=iliac_excluded_nodes,
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
            iliac_excluded_system_nodes=iliac_excluded_nodes,
        )
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
            excluded_system_nodes=iliac_excluded_nodes,
        )
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
        laterality_conf = float(
            clamp(
                0.50 * float(horizontal_frame_info.get("confidence", 0.0))
                + 0.20 * float(horizontal_frame_info.get("iliac_axis_confidence", 0.0))
                + 0.20 * float(horizontal_frame_info.get("renal_axis_alignment_with_iliacs", 0.0))
                + 0.10 * float(ap_conf),
                0.0,
                1.0,
            )
        )
        horizontal_frame_info["laterality_confidence"] = float(laterality_conf)
        horizontal_frame_info["laterality_source"] = "canonical_x"
        if laterality_conf < 0.60:
            warnings.append(f"W_LATERALITY_LOWCONF: laterality confidence={laterality_conf:.3f}; side labels may be ambiguous.")

        # Determine which iliac main path is right/left by x of distal point in canonical frame
        a_x = float(cl_pts_c[iliac_main_a[-1]][0]) if iliac_main_a else float(cl_pts_c[ep_a][0])
        b_x = float(cl_pts_c[iliac_main_b[-1]][0]) if iliac_main_b else float(cl_pts_c[ep_b][0])

        if a_x >= b_x:
            # a is more "right"
            right_iliac_system = iliac_system_a
            left_iliac_system = iliac_system_b
            right_iliac_main = iliac_main_a
            left_iliac_main = iliac_main_b
            right_iliac_ep = ep_a
            left_iliac_ep = ep_b
        else:
            right_iliac_system = iliac_system_b
            left_iliac_system = iliac_system_a
            right_iliac_main = iliac_main_b
            left_iliac_main = iliac_main_a
            right_iliac_ep = ep_b
            left_iliac_ep = ep_a
        if right_iliac_system is not None:
            right_iliac_ep = int(right_iliac_system["representative_endpoint"])
        if left_iliac_system is not None:
            left_iliac_ep = int(left_iliac_system["representative_endpoint"])

        # Now identify renals using final canonical coordinates
        rr_ep, lr_ep, rr_take, lr_take, _, renal_diag = identify_renal_branches(
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
            excluded_system_nodes=iliac_excluded_nodes,
        )
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
            trunk_child_systems=trunk_child_systems,
            right_iliac_system=right_iliac_system,
            left_iliac_system=left_iliac_system,
        )
        if not preserved_branch_geoms:
            raise RuntimeError("Failed to construct branch-preserving centerline scaffold output.")
        preserved_branch_geoms = enrich_branch_geometries_with_proximal_anchors(preserved_branch_geoms, warnings=warnings)

        branch_counts: Dict[str, int] = {}
        for br in preserved_branch_geoms:
            branch_counts[br["name"]] = int(branch_counts.get(br["name"], 0) + 1)
        branch_geometry_summary = summarize_centerline_branch_geometries(preserved_branch_geoms)
        branch_anchor_summary = summarize_branch_proximal_anchors(preserved_branch_geoms)

        surface_tagged_pd, surface_label_summary = annotate_surface_polydata_for_combined_output(
            surface_pd=surface_tri_c,
            branch_geoms=preserved_branch_geoms,
            warnings=warnings,
        )

        centerlines_out_pd = build_output_centerlines_polydata(
            branch_geoms=preserved_branch_geoms,
        )

        combined_out_pd = build_combined_surface_centerlines_polydata(
            surface_pd=surface_tagged_pd,
            centerlines_pd=centerlines_out_pd,
        )
        attach_landmarks_to_polydata_field_data(centerlines_out_pd, landmarks)
        attach_landmarks_to_polydata_field_data(combined_out_pd, landmarks)

        write_vtp(combined_out_pd, surface_with_centerlines_path, binary=True)
        write_vtp(centerlines_out_pd, centerlines_output_path, binary=True)

        # Write metadata JSON (optional but recommended)
        if meta_path:
            os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
            meta: Dict[str, Any] = {
                "status": "success",
                "input_vtp": os.path.abspath(input_path),
                "output_surface_with_centerlines_vtp": os.path.abspath(surface_with_centerlines_path),
                "output_centerlines_vtp": os.path.abspath(centerlines_output_path),
                "mode": mode,
                "warnings": [str(w) for w in warnings],
                "terminal_detection_mode": str(terminal_detection_mode),
                "terminal_face_map_path": os.path.abspath(terminal_face_map_path) if terminal_face_map_path else "",
                "sidecar_mapping_used": {
                    "source": str(terminal_face_map_source),
                    "mapping": {
                        str(fid): {
                            "name": str(info.get("name", f"face_{fid}")),
                            "cap_id": (None if info.get("cap_id") is None else int(info.get("cap_id"))),
                            "terminal_type": str(info.get("terminal_type", "outlet")),
                        }
                        for fid, info in sorted(terminal_face_map.items())
                    },
                },
                "detected_face_stats_per_mapped_face": _jsonable_face_regions(mapped_face_regions),
                "branch_names": sorted(str(name) for name in branch_counts.keys()),
                "branch_counts": {k: int(v) for k, v in branch_counts.items()},
                "centerline_length_by_branch": {
                    k: float(v) for k, v in dict(branch_geometry_summary.get("length_by_branch_label", {})).items()
                },
                "centerline_branch_summaries": list(branch_geometry_summary.get("branches", [])),
                "branch_proximal_anchors": dict(branch_anchor_summary),
                "centerline_runtime": json.loads(json.dumps(centerline_runtime_info, default=str)),
                "surface_cell_counts_by_branch": {
                    k: int(v)
                    for k, v in dict(surface_label_summary.get("cell_counts", {})).items()
                    if int(v) > 0
                },
                "landmarks_xyz_canonical": {k: [float(x) for x in np.asarray(v, dtype=float).reshape(3)] for k, v in landmarks.items()},
                "canonical_frame_summary": dict(horizontal_frame_info),
                "identified_bifurcation_node": int(bif_node),
                "identified_common_iliac_split_nodes": {
                    "right": int(right_iliac_nodes[-1]) if len(right_iliac_nodes) >= 2 else int(bif_node),
                    "left": int(left_iliac_nodes[-1]) if len(left_iliac_nodes) >= 2 else int(bif_node),
                },
                "identified_renal_takeoff_nodes": {
                    "right": (None if rr_origin_node is None else int(rr_origin_node)),
                    "left": (None if lr_origin_node is None else int(lr_origin_node)),
                },
                "junction_nodes": {
                    "aortic_bifurcation_node": int(bif_node),
                    "right_common_iliac_split_node": int(right_iliac_nodes[-1]) if len(right_iliac_nodes) >= 2 else int(bif_node),
                    "left_common_iliac_split_node": int(left_iliac_nodes[-1]) if len(left_iliac_nodes) >= 2 else int(bif_node),
                },
                "transform": {
                    "R_rows": [[float(x) for x in row] for row in np.asarray(R, dtype=float).reshape(3, 3)],
                    "origin": [float(x) for x in np.asarray(origin, dtype=float).reshape(3)],
                    "flipped_for_ap": bool(flipped_for_ap),
                },
                "renals_found": {
                    "right": bool(rr_ep is not None and rr_take is not None),
                    "left": bool(lr_ep is not None and lr_take is not None),
                },
                "dropped_or_merged_branches": [],
                "fallback_usage": {
                    "used": bool(terminal_detection_mode != "prelabeled_model_face_id"),
                    "reason": ("automatic termination inference" if terminal_detection_mode != "prelabeled_model_face_id" else ""),
                },
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
                    "terminal_detection_mode": str(terminal_detection_mode),
                    "terminal_face_map_path": os.path.abspath(terminal_face_map_path) if terminal_face_map_path else "",
                    "sidecar_mapping_used": {
                        "source": str(terminal_face_map_source),
                        "mapping": {
                            str(fid): {
                                "name": str(info.get("name", f"face_{fid}")),
                                "cap_id": (None if info.get("cap_id") is None else int(info.get("cap_id"))),
                                "terminal_type": str(info.get("terminal_type", "outlet")),
                            }
                            for fid, info in sorted(terminal_face_map.items())
                        },
                    },
                    "detected_face_stats_per_mapped_face": _jsonable_face_regions(mapped_face_regions),
                    "prelabeled_terminal_matches": _jsonable_terminal_matches(terminal_matches),
                    "prelabeled_topology": {
                        "bifurcation_node": (
                            None if topology.get("bifurcation_node") is None else int(topology.get("bifurcation_node"))
                        ),
                        "right_common_iliac_split_node": (
                            None
                            if topology.get("right_common_iliac_split_node") is None
                            else int(topology.get("right_common_iliac_split_node"))
                        ),
                        "left_common_iliac_split_node": (
                            None
                            if topology.get("left_common_iliac_split_node") is None
                            else int(topology.get("left_common_iliac_split_node"))
                        ),
                        "direct_branch_takeoffs": {
                            str(k): (None if v is None else int(v))
                            for k, v in dict(topology.get("direct_branch_takeoffs", {})).items()
                        },
                        "geometric_junction_refinement_raw": _jsonable_geometric_junctions(topology.get("geometric_junctions_raw", {})),
                        "geometric_junction_refinement_canonical": _jsonable_geometric_junctions(topology.get("geometric_junctions_canonical", {})),
                    },
                    "canonical_frame_summary": dict(horizontal_frame_info),
                    "centerline_runtime": json.loads(json.dumps(centerline_runtime_info, default=str)),
                    "warnings": [str(w) for w in warnings],
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
