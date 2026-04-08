#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

INPUT_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\0044_H_ABAO_AAA\\0044_H_ABAO_AAA\\Models\\0156_0001.vtp"
OUTPUT_SURFACE_WITH_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_surface_with_centerlines.vtp"
OUTPUT_CENTERLINES_VTP_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines.vtp"
OUTPUT_METADATA_PATH = "C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\Output files\\oriented_labeled_centerlines_metadata.json"
OUTPUT_DEBUG_CENTERLINES_RAW_PATH = ""

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

LABEL_PRIORITY = {
    LABEL_AORTA_TRUNK: 0,
    LABEL_RIGHT_ILIAC: 1,
    LABEL_LEFT_ILIAC: 2,
    LABEL_RIGHT_RENAL: 3,
    LABEL_LEFT_RENAL: 4,
    LABEL_OTHER: 5,
}

EPS = 1.0e-12
SCRIPT_PATH = os.path.abspath(__file__) if "__file__" in globals() else os.path.abspath(sys.argv[0])
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
_WINDOWS_DLL_HANDLES: List[Any] = []
_VTK_IMPORT_ERROR = ""
_LAST_VMTK_DIAGNOSTICS: Dict[str, Any] = {}

try:
    import numpy as np
except Exception as exc:
    raise RuntimeError(f"numpy is required: {exc}")

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk  # type: ignore
except Exception as exc:
    vtk = None
    vtk_to_numpy = None
    numpy_to_vtk = None
    _VTK_IMPORT_ERROR = str(exc)

if TYPE_CHECKING:
    from vtkmodules.vtkCommonDataModel import (
        vtkPolyData,
        vtkCellData,
        vtkStaticPointLocator,
    )


@dataclass
class TerminationLoop:
    center: np.ndarray
    area: float
    diameter_eq: float
    normal: np.ndarray
    rms_planarity: float
    n_points: int
    source: str


def _normalize_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _resolve_user_path(path: str) -> str:
    path = (path or "").strip()
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))


def _prepare_windows_dll_search_paths() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": os.name,
        "dll_directories_added": [],
        "path_prepended": [],
    }
    if os.name != "nt":
        return info
    prefixes = []
    seen = set()
    for prefix in [os.environ.get("CONDA_PREFIX"), sys.prefix, os.path.dirname(sys.executable)]:
        if not prefix:
            continue
        prefix = os.path.abspath(prefix)
        key = _normalize_path_key(prefix)
        if key in seen or not os.path.isdir(prefix):
            continue
        seen.add(key)
        prefixes.append(prefix)

    current_path = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    current_keys = {_normalize_path_key(p) for p in current_path}
    prepend = []

    for prefix in prefixes:
        for candidate in [
            prefix,
            os.path.join(prefix, "Library", "bin"),
            os.path.join(prefix, "Scripts"),
            os.path.join(prefix, "bin"),
            os.path.join(prefix, "Lib", "site-packages", "vmtk"),
            os.path.join(prefix, "Lib", "site-packages", "vtkmodules"),
        ]:
            if not os.path.isdir(candidate):
                continue
            key = _normalize_path_key(candidate)
            if hasattr(os, "add_dll_directory"):
                try:
                    _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(candidate))
                    info["dll_directories_added"].append(candidate)
                except Exception:
                    pass
            if key not in current_keys:
                current_keys.add(key)
                prepend.append(candidate)
                info["path_prepended"].append(candidate)

    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + current_path)
    return info


_prepare_windows_dll_search_paths()


def require_vtk() -> None:
    if vtk is None or vtk_to_numpy is None or numpy_to_vtk is None:
        raise RuntimeError(f"vtk is required: {_VTK_IMPORT_ERROR}")


def unit(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(vv))
    if n < EPS:
        return np.zeros((3,), dtype=float)
    return (vv / n).astype(float)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        center = np.mean(pts, axis=0) if pts.shape[0] else np.zeros((3,), dtype=float)
        return np.eye(3, dtype=float), np.ones((3,), dtype=float), center.astype(float)
    center = np.mean(pts, axis=0)
    x = pts - center[None, :]
    cov = (x.T @ x) / max(1, int(pts.shape[0]))
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return v[:, order].astype(float), w[order].astype(float), center.astype(float)


def planar_polygon_area_and_normal(points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, np.zeros((3,), dtype=float), float("inf")
    center = np.mean(pts, axis=0)
    x = pts - center[None, :]
    cov = (x.T @ x) / max(1, int(pts.shape[0]))
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)
    normal = unit(v[:, order[0]])
    e0 = unit(v[:, order[2]])
    e1 = unit(v[:, order[1]])
    q = np.column_stack([e0, e1])
    uv = x @ q
    area2 = 0.0
    for i in range(uv.shape[0]):
        j = (i + 1) % uv.shape[0]
        area2 += float(uv[i, 0] * uv[j, 1] - uv[j, 0] * uv[i, 1])
    area = 0.5 * abs(area2)
    rms = float(np.sqrt(np.mean(((x @ normal) ** 2))))
    return float(area), normal.astype(float), rms


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


def compute_tangents(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    if n == 1:
        return np.zeros((1, 3), dtype=float)
    tangents = np.zeros((n, 3), dtype=float)
    tangents[0] = unit(pts[1] - pts[0])
    tangents[-1] = unit(pts[-1] - pts[-2])
    for i in range(1, n - 1):
        tangents[i] = unit(pts[i + 1] - pts[i - 1])
    return tangents.astype(float)


def compute_abscissa(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=float)
    if n == 1:
        return np.zeros((1,), dtype=float)
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    out = np.zeros((n,), dtype=float)
    out[1:] = np.cumsum(seg)
    return out.astype(float)


def project_vector_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(3)
    nn = unit(normal)
    if float(np.linalg.norm(nn)) < EPS:
        return vv.astype(float)
    return (vv - np.dot(vv, nn) * nn).astype(float)


def unit_xy(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(3).copy()
    vv[2] = 0.0
    return unit(vv)


def load_vtp(path: str) -> vtkPolyData:
    require_vtk()
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = reader.GetOutput()
    if out is None or out.GetNumberOfPoints() <= 0:
        raise RuntimeError(f"failed to read VTP: {path}")
    pd = vtk.vtkPolyData()
    pd.DeepCopy(out)
    return pd


def write_vtp(pd: vtkPolyData, path: str, binary: bool = True) -> None:
    require_vtk()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(pd)
    if binary:
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    writer.EncodeAppendedDataOff()
    writer.Write()


def clone_polydata(pd: vtkPolyData) -> vtkPolyData:
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    return out


def get_points_numpy(pd: vtkPolyData) -> np.ndarray:
    pts = pd.GetPoints()
    if pts is None or pts.GetData() is None or vtk_to_numpy is None:
        return np.zeros((0, 3), dtype=float)
    arr = vtk_to_numpy(pts.GetData())
    return np.asarray(arr, dtype=float).reshape((-1, 3))


def clean_and_triangulate_surface(pd: vtkPolyData) -> vtkPolyData:
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.PointMergingOn()
    cleaner.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(cleaner.GetOutput())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()

    cleaner2 = vtk.vtkCleanPolyData()
    cleaner2.SetInputData(tri.GetOutput())
    cleaner2.ConvertLinesToPointsOff()
    cleaner2.ConvertPolysToLinesOff()
    cleaner2.ConvertStripsToPolysOff()
    cleaner2.PointMergingOn()
    cleaner2.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner2.GetOutput())
    out.BuildLinks()
    return out


def count_boundary_edges(pd: vtkPolyData) -> int:
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    out = fe.GetOutput()
    return int(out.GetNumberOfCells()) if out is not None else 0


def build_static_locator(pd: vtkPolyData) -> vtkStaticPointLocator:
    loc = vtk.vtkStaticPointLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    return loc


def clean_centerlines_preserve_lines(pd: vtkPolyData) -> vtkPolyData:
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.PointMergingOn()
    cleaner.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    return out


def find_face_partition_array_name(pd: vtkPolyData) -> Optional[str]:
    cd = pd.GetCellData()
    if cd is None or vtk_to_numpy is None:
        return None
    preferred = [
        "ModelFaceID",
        "ModelFaceIds",
        "ModelFaceID_",
        "FaceID",
        "FaceIds",
        "GroupIds",
        "GroupId",
        "CapID",
        "EntityIds",
        "RegionId",
        "ModelFaceEntityIds",
    ]
    candidates = []
    for i in range(cd.GetNumberOfArrays()):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if not name or arr.GetNumberOfComponents() != 1:
            continue
        try:
            vals = vtk_to_numpy(arr)
        except Exception:
            continue
        vals = np.asarray(vals)
        if vals.size == 0:
            continue
        if not np.issubdtype(vals.dtype, np.integer) and not np.issubdtype(vals.dtype, np.floating):
            continue
        uniq = np.unique(vals)
        if uniq.size < 2 or uniq.size > min(512, vals.size):
            continue
        if np.issubdtype(vals.dtype, np.floating):
            if not np.allclose(vals, np.round(vals)):
                continue
        candidates.append((str(name), int(uniq.size)))
    if not candidates:
        return None
    for pref in preferred:
        for name, _ in candidates:
            if name == pref:
                return name
    for name, _ in candidates:
        lname = name.lower()
        if "face" in lname or "group" in lname or "cap" in lname or "region" in lname:
            return name
    return candidates[0][0]


def compute_polydata_cell_centers_numpy(pd: vtkPolyData) -> np.ndarray:
    cc = vtk.vtkCellCenters()
    cc.SetInputData(pd)
    cc.VertexCellsOff()
    cc.Update()
    out = cc.GetOutput()
    pts = out.GetPoints()
    if pts is None or pts.GetData() is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(vtk_to_numpy(pts.GetData()), dtype=float).reshape((-1, 3))


def compute_polydata_cell_areas(pd: vtkPolyData) -> np.ndarray:
    cs = vtk.vtkCellSizeFilter()
    cs.SetInputData(pd)
    cs.SetComputeArea(True)
    cs.SetComputeLength(False)
    cs.SetComputeVolume(False)
    cs.SetComputeVertexCount(False)
    cs.Update()
    arr = cs.GetOutput().GetCellData().GetArray("Area")
    if arr is None or vtk_to_numpy is None:
        return np.zeros((int(pd.GetNumberOfCells()),), dtype=float)
    return np.asarray(vtk_to_numpy(arr), dtype=float).reshape((-1,))


def build_surface_point_to_cells(pd: vtkPolyData) -> List[List[int]]:
    n_points = int(pd.GetNumberOfPoints())
    point_to_cells: List[List[int]] = [[] for _ in range(n_points)]
    for ci in range(int(pd.GetNumberOfCells())):
        cell = pd.GetCell(ci)
        if cell is None:
            continue
        for k in range(int(cell.GetNumberOfPoints())):
            pid = int(cell.GetPointId(k))
            if 0 <= pid < n_points:
                point_to_cells[pid].append(int(ci))
    return point_to_cells


def build_surface_cell_adjacency(pd: vtkPolyData) -> List[List[int]]:
    point_to_cells = build_surface_point_to_cells(pd)
    adjacency = [set() for _ in range(int(pd.GetNumberOfCells()))]
    for cells in point_to_cells:
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


def add_string_array_to_cell_data(cd: vtkCellData, name: str, values: List[str]) -> None:
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    arr.SetNumberOfValues(len(values))
    for i, v in enumerate(values):
        arr.SetValue(i, str(v))
    cd.AddArray(arr)


def add_scalar_array_to_cell_data(cd: vtkCellData, name: str, values: List[float], vtk_type: int = vtk.VTK_DOUBLE) -> None:
    if vtk_type == vtk.VTK_INT:
        arr = vtk.vtkIntArray()
    else:
        arr = vtk.vtkDoubleArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(len(values))
    for i, v in enumerate(values):
        arr.SetTuple1(i, int(v) if vtk_type == vtk.VTK_INT else float(v))
    cd.AddArray(arr)


def get_cell_scalar_array_values(pd: vtkPolyData, name: str, default_value: float = 0.0) -> List[float]:
    cd = pd.GetCellData()
    arr = cd.GetArray(name) if cd is not None else None
    n = int(pd.GetNumberOfCells())
    if arr is None or vtk_to_numpy is None:
        return [default_value] * n
    try:
        vals = vtk_to_numpy(arr)
    except Exception:
        return [default_value] * n
    vals = np.asarray(vals).reshape((-1,))
    if vals.shape[0] != n:
        return [default_value] * n
    return [safe_float(v, default_value) for v in vals.tolist()]


def get_cell_string_array_values(pd: vtkPolyData, name: str, default_value: str = "") -> List[str]:
    cd = pd.GetCellData()
    arr = cd.GetAbstractArray(name) if cd is not None else None
    n = int(pd.GetNumberOfCells())
    if arr is None:
        return [default_value] * n
    out = []
    for i in range(n):
        try:
            out.append(str(arr.GetVariantValue(i).ToString()))
        except Exception:
            try:
                out.append(str(arr.GetValue(i)))
            except Exception:
                out.append(default_value)
    return out


def extract_boundary_loops(pd: vtkPolyData) -> List[TerminationLoop]:
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

    for ci in range(int(out.GetNumberOfCells())):
        cell = out.GetCell(ci)
        if cell is None:
            continue
        nids = int(cell.GetNumberOfPoints())
        if nids < 3:
            continue
        coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
        center = np.mean(coords, axis=0)
        area, normal, rms = planar_polygon_area_and_normal(coords)
        diameter_eq = math.sqrt(max(0.0, 4.0 * area / math.pi)) if area > 0.0 else 0.0
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


def termination_candidates_from_face_partitions(pd_tri: vtkPolyData, face_array: str) -> List[TerminationLoop]:
    candidates: List[TerminationLoop] = []
    cd = pd_tri.GetCellData()
    if cd is None or vtk_to_numpy is None:
        return candidates
    arr = cd.GetArray(face_array)
    if arr is None:
        return candidates

    cs = vtk.vtkCellSizeFilter()
    cs.SetInputData(pd_tri)
    cs.SetComputeArea(True)
    cs.SetComputeLength(False)
    cs.SetComputeVolume(False)
    cs.SetComputeVertexCount(False)
    cs.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(cs.GetOutput())
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.Update()
    pdn = normals.GetOutput()

    centers = vtk.vtkCellCenters()
    centers.SetInputData(pdn)
    centers.VertexCellsOn()
    centers.Update()
    centers_pts = centers.GetOutput().GetPoints()
    if centers_pts is None:
        return candidates

    try:
        face_vals = np.asarray(vtk_to_numpy(pdn.GetCellData().GetArray(face_array)), dtype=np.int64)
        area_vals = np.asarray(vtk_to_numpy(pdn.GetCellData().GetArray("Area")), dtype=float)
        normal_arr = pdn.GetCellData().GetNormals()
        if normal_arr is None:
            normal_arr = pdn.GetCellData().GetArray("Normals")
        if normal_arr is None:
            return candidates
        normal_vals = np.asarray(vtk_to_numpy(normal_arr), dtype=float).reshape((-1, 3))
        center_vals = np.asarray(vtk_to_numpy(centers_pts.GetData()), dtype=float).reshape((-1, 3))
    except Exception:
        return candidates

    if center_vals.shape[0] != face_vals.shape[0]:
        return candidates

    total_area = float(np.sum(area_vals))
    if total_area <= 0.0:
        return candidates

    uniq = np.unique(face_vals)
    face_stats: List[Dict[str, Any]] = []
    for fid in uniq:
        mask = face_vals == int(fid)
        if not np.any(mask):
            continue
        a = area_vals[mask]
        a_sum = float(np.sum(a))
        if a_sum <= 0.0:
            continue
        c = np.sum(center_vals[mask] * a[:, None], axis=0) / a_sum
        n_sum = np.sum(normal_vals[mask] * a[:, None], axis=0)
        planarity = float(np.linalg.norm(n_sum) / (a_sum + EPS))
        diameter_eq = math.sqrt(max(0.0, 4.0 * a_sum / math.pi))
        face_stats.append(
            {
                "fid": int(fid),
                "area": float(a_sum),
                "center": c.astype(float),
                "planarity": float(planarity),
                "diameter_eq": float(diameter_eq),
            }
        )
    if not face_stats:
        return candidates

    max_area = max(float(fs["area"]) for fs in face_stats)
    for fs in face_stats:
        if float(fs["planarity"]) < 0.92:
            continue
        if float(fs["area"]) > 0.60 * total_area:
            continue
        if max_area > 0.0 and float(fs["area"]) > 0.85 * max_area and len(face_stats) > 3:
            continue
        candidates.append(
            TerminationLoop(
                center=np.asarray(fs["center"], dtype=float),
                area=float(fs["area"]),
                diameter_eq=float(fs["diameter_eq"]),
                normal=np.zeros((3,), dtype=float),
                rms_planarity=float("nan"),
                n_points=0,
                source=f"face_partition:{face_array}",
            )
        )
    return candidates


def detect_terminations(pd_tri: vtkPolyData, warnings: List[str]) -> Tuple[List[TerminationLoop], str]:
    if count_boundary_edges(pd_tri) > 0:
        loops = extract_boundary_loops(pd_tri)
        if len(loops) >= 2:
            return loops, "open_termini"

    face_array = find_face_partition_array_name(pd_tri)
    if face_array:
        terms = termination_candidates_from_face_partitions(pd_tri, face_array)
        if len(terms) >= 2:
            warnings.append(f"W_TERMINATIONS_FACEPART: used planar face partitions via '{face_array}'.")
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
            for ci in range(int(out.GetNumberOfCells())):
                cell = out.GetCell(ci)
                if cell is None:
                    continue
                nids = int(cell.GetNumberOfPoints())
                if nids < 6:
                    continue
                coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
                center = np.mean(coords, axis=0)
                area, normal, rms = planar_polygon_area_and_normal(coords)
                diameter_eq = math.sqrt(max(0.0, 4.0 * area / math.pi)) if area > 0.0 else 0.0
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
                warnings.append("W_TERMINATIONS_FEATUREEDGES: used feature-edge loops because boundary loops and face partitions were unavailable.")
                return loops, "closed_unpartitioned"

    warnings.append("W_TERMINATIONS_NONE: failed to detect robust terminations.")
    return [], "unsupported"


def generate_inlet_hypotheses(
    terms: List[TerminationLoop],
    surface_points: np.ndarray,
    warnings: List[str],
    max_hypotheses: int = 3,
) -> List[Dict[str, Any]]:
    if not terms:
        return []
    centers = np.asarray([t.center for t in terms], dtype=float).reshape((-1, 3))
    diam = np.asarray([max(0.0, float(t.diameter_eq)) for t in terms], dtype=float)
    area = np.asarray([max(0.0, float(t.area)) for t in terms], dtype=float)

    if surface_points.shape[0] >= 3:
        axes, _, _ = pca_axes(surface_points)
    else:
        axes, _, _ = pca_axes(centers)
    base_axis = unit(axes[:, 0])
    if float(np.linalg.norm(base_axis)) < EPS:
        base_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    centroid = np.mean(centers, axis=0)
    base_proj = (centers - centroid[None, :]) @ base_axis
    max_abs_proj = max(EPS, float(np.max(np.abs(base_proj))))

    diam_norm = (diam - float(np.min(diam))) / (float(np.ptp(diam)) + EPS)
    area_norm = (area - float(np.min(area))) / (float(np.ptp(area)) + EPS)

    hypotheses: List[Dict[str, Any]] = []
    for i, term in enumerate(terms):
        axis = base_axis.copy()
        proj = base_proj.copy()
        if float(proj[i]) < 0.0:
            axis *= -1.0
            proj *= -1.0

        opposite = [j for j in range(len(terms)) if j != i and float(proj[j]) < 0.0]
        distal_pair_score = 0.0
        distal_pair = None
        if len(opposite) >= 2:
            opp_pool = sorted(opposite, key=lambda j: (diam[j], -proj[j], -j), reverse=True)[: min(6, len(opposite))]
            best_pair_score = -1.0
            best_pair = None
            for a_idx in range(len(opp_pool)):
                for b_idx in range(a_idx + 1, len(opp_pool)):
                    ia = int(opp_pool[a_idx])
                    ib = int(opp_pool[b_idx])
                    dvec = centers[ia] - centers[ib]
                    lateral = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
                    lateral_ratio = lateral / (0.5 * (diam[ia] + diam[ib]) + EPS)
                    depth_score = 0.5 * ((-proj[ia]) / max_abs_proj + (-proj[ib]) / max_abs_proj)
                    pair_size = 0.5 * (diam_norm[ia] + diam_norm[ib])
                    balance = 1.0 - abs(diam[ia] - diam[ib]) / (diam[ia] + diam[ib] + EPS)
                    score = float(
                        clamp(
                            0.35 * pair_size
                            + 0.35 * clamp((lateral_ratio - 0.75) / 1.5, 0.0, 1.0)
                            + 0.20 * clamp(depth_score, 0.0, 1.0)
                            + 0.10 * clamp(balance, 0.0, 1.0),
                            0.0,
                            1.0,
                        )
                    )
                    if score > best_pair_score:
                        best_pair_score = score
                        best_pair = (ia, ib)
            if best_pair is not None:
                distal_pair_score = float(best_pair_score)
                distal_pair = tuple(int(v) for v in best_pair)

        extremity_score = abs(float(proj[i])) / max_abs_proj
        size_score = 0.55 * float(diam_norm[i]) + 0.45 * float(area_norm[i])
        inlet_conf = float(clamp(0.32 * size_score + 0.33 * extremity_score + 0.35 * distal_pair_score, 0.0, 1.0))
        hypotheses.append(
            {
                "inlet_index": int(i),
                "termination": term,
                "axis_si": axis.astype(float),
                "confidence": float(inlet_conf),
                "distal_pair": distal_pair,
                "extremity_score": float(extremity_score),
                "size_score": float(size_score),
                "distal_pair_score": float(distal_pair_score),
            }
        )

    hypotheses.sort(
        key=lambda h: (
            -float(h["confidence"]),
            -float(h["extremity_score"]),
            -float(h["size_score"]),
            int(h["inlet_index"]),
        )
    )
    seen = set()
    out: List[Dict[str, Any]] = []
    for hyp in hypotheses:
        idx = int(hyp["inlet_index"])
        if idx in seen:
            continue
        seen.add(idx)
        out.append(hyp)
        if len(out) >= max(1, int(max_hypotheses)):
            break

    if out and float(out[0]["confidence"]) < 0.55:
        warnings.append(f"W_INLET_LOWCONF: best inlet hypothesis confidence={float(out[0]['confidence']):.3f}.")
    return out


def _resolve_vmtk_import() -> Tuple[Optional[Any], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "vtk_import_ok": bool(vtk is not None),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "import_attempts": [],
    }
    if vtk is None:
        return None, diagnostics

    module_names = [
        "vmtk.vtkvmtkComputationalGeometryPython",
        "vtkvmtkComputationalGeometryPython",
    ]
    for name in module_names:
        try:
            mod = importlib.import_module(name)
            ok = hasattr(mod, "vtkvmtkCapPolyData") and hasattr(mod, "vtkvmtkPolyDataCenterlines")
            diagnostics["import_attempts"].append({"name": name, "ok": ok, "error": ""})
            if ok:
                diagnostics["resolved_vmtk_source"] = name
                diagnostics["vmtk_import_ok"] = True
                return mod, diagnostics
        except Exception as exc:
            diagnostics["import_attempts"].append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    diagnostics["vmtk_import_ok"] = False
    return None, diagnostics


def try_import_vmtk() -> Tuple[Optional[Any], Optional[str]]:
    global _LAST_VMTK_DIAGNOSTICS
    mod, diag = _resolve_vmtk_import()
    _LAST_VMTK_DIAGNOSTICS = dict(diag)
    if mod is not None:
        return mod, None
    attempts = diag.get("import_attempts", [])
    msg = "VMTK vtkvmtk bindings are required but could not be imported."
    if attempts:
        details = "; ".join(f"{a.get('name')}: {a.get('error')}" for a in attempts)
        msg += " " + details
    return None, msg


def cap_surface_if_open(pd_tri: vtkPolyData, vtkvmtk_mod: Any) -> Tuple[vtkPolyData, bool]:
    if count_boundary_edges(pd_tri) <= 0:
        out = vtk.vtkPolyData()
        out.DeepCopy(pd_tri)
        return out, False
    capper = vtkvmtk_mod.vtkvmtkCapPolyData()
    capper.SetInputData(pd_tri)
    capper.SetDisplacement(0.0)
    capper.SetInPlaneDisplacement(0.0)
    capper.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(capper.GetOutput())
    return out, True


def compute_centerlines_vmtk(
    pd_tri: vtkPolyData,
    inlet_center: np.ndarray,
    term_centers: List[np.ndarray],
    warnings: List[str],
) -> Tuple[vtkPolyData, Dict[str, Any]]:
    vtkvmtk_mod, err = try_import_vmtk()
    if vtkvmtk_mod is None:
        raise RuntimeError(err or "VMTK bindings unavailable.")

    capped, did_cap = cap_surface_if_open(pd_tri, vtkvmtk_mod)
    locator = build_static_locator(capped)
    inlet_pid = int(locator.FindClosestPoint(float(inlet_center[0]), float(inlet_center[1]), float(inlet_center[2])))

    target_pids: List[int] = []
    seen = {int(inlet_pid)}
    for c in term_centers:
        pid = int(locator.FindClosestPoint(float(c[0]), float(c[1]), float(c[2])))
        if pid in seen:
            continue
        seen.add(pid)
        p = np.asarray(capped.GetPoint(pid), dtype=float)
        if float(np.linalg.norm(p - inlet_center)) < 1.0e-6:
            continue
        target_pids.append(int(pid))

    if len(target_pids) < 1:
        raise RuntimeError("Insufficient target seeds for VMTK centerline extraction.")

    bbox = capped.GetBounds()
    diag = float(np.linalg.norm(np.asarray([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float)))
    step = max(0.005 * diag, 0.5)

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
                raise RuntimeError("VMTK returned empty centerlines.")
            out_clean = clean_centerlines_preserve_lines(out)
            if out_clean.GetNumberOfPoints() < 2 or out_clean.GetNumberOfCells() < 1:
                raise RuntimeError("Centerlines empty after cleaning.")
            info = {
                "did_cap": bool(did_cap),
                "inlet_pid": int(inlet_pid),
                "n_targets": int(len(target_pids)),
                "flip_normals": int(flip),
                "resampling_step": float(step),
                "n_points": int(out_clean.GetNumberOfPoints()),
                "n_cells": int(out_clean.GetNumberOfCells()),
            }
            return out_clean, info
        except Exception as exc:
            last_err = exc
            warnings.append(f"W_VMTK_CENTERLINES_FAIL_FLIP{flip}: {exc}")
    raise RuntimeError(f"VMTK centerline extraction failed: {last_err}")


def build_graph_from_polyline_centerlines(pd: vtkPolyData) -> Tuple[Dict[int, Dict[int, float]], np.ndarray, List[int]]:
    pts = get_points_numpy(pd)
    adjacency: Dict[int, Dict[int, float]] = {}
    used_cells: List[int] = []
    for ci in range(int(pd.GetNumberOfCells())):
        cell = pd.GetCell(ci)
        if cell is None:
            continue
        if cell.GetCellType() not in (vtk.VTK_POLY_LINE, vtk.VTK_LINE):
            continue
        nids = int(cell.GetNumberOfPoints())
        if nids < 2:
            continue
        used_cells.append(int(ci))
        ids = [int(cell.GetPointId(k)) for k in range(nids)]
        for a, b in zip(ids[:-1], ids[1:]):
            if a == b:
                continue
            w = float(np.linalg.norm(pts[a] - pts[b]))
            if w <= 0.0:
                continue
            adjacency.setdefault(a, {})
            adjacency.setdefault(b, {})
            if b not in adjacency[a] or w < adjacency[a][b]:
                adjacency[a][b] = w
                adjacency[b][a] = w
    return adjacency, pts, used_cells


def connected_component_nodes(adjacency: Dict[int, Dict[int, float]], start: int) -> set[int]:
    if int(start) not in adjacency:
        return set()
    seen: set[int] = set()
    stack = [int(start)]
    while stack:
        node = int(stack.pop())
        if node in seen:
            continue
        seen.add(node)
        for nbr in adjacency.get(node, {}):
            if int(nbr) not in seen:
                stack.append(int(nbr))
    return seen


def induced_subgraph(adjacency: Dict[int, Dict[int, float]], nodes: set[int]) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {}
    for node in nodes:
        nbrs = {int(n): float(w) for n, w in adjacency.get(int(node), {}).items() if int(n) in nodes}
        if nbrs:
            out[int(node)] = nbrs
    return out


def node_degrees(adjacency: Dict[int, Dict[int, float]]) -> Dict[int, int]:
    return {int(n): int(len(v)) for n, v in adjacency.items()}


def dijkstra(adjacency: Dict[int, Dict[int, float]], start: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    import heapq

    start = int(start)
    dist: Dict[int, float] = {start: 0.0}
    prev: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, start)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if int(u) in visited:
            continue
        visited.add(int(u))
        for v, w in adjacency.get(int(u), {}).items():
            nd = float(d) + float(w)
            if int(v) not in dist or nd < dist[int(v)]:
                dist[int(v)] = float(nd)
                prev[int(v)] = int(u)
                heapq.heappush(heap, (float(nd), int(v)))
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
    return [int(v) for v in path]


def edge_key(a: int, b: int) -> Tuple[int, int]:
    aa = int(a)
    bb = int(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def path_edge_keys(path: List[int]) -> set[Tuple[int, int]]:
    return {edge_key(a, b) for a, b in zip(path[:-1], path[1:])}


def graph_edges_for_node_set(adjacency: Dict[int, Dict[int, float]], nodes: set[int]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for a in nodes:
        for b in adjacency.get(int(a), {}):
            if int(b) in nodes:
                out.add(edge_key(int(a), int(b)))
    return out


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
        prev_node = int(start)
        cur = int(nxt)
        while deg.get(int(cur), 0) == 2 and int(cur) not in key_nodes:
            candidates = [int(v) for v in sorted(adjacency.get(int(cur), {}).keys()) if int(v) != prev_node]
            if not candidates:
                break
            cand = int(candidates[0])
            ek = edge_key(cur, cand)
            if ek in visited_edges:
                break
            visited_edges.add(ek)
            path.append(cand)
            prev_node, cur = cur, cand
        return path

    for start in sorted(key_nodes):
        for nxt in sorted(adjacency.get(int(start), {}).keys()):
            ek = edge_key(int(start), int(nxt))
            if ek in visited_edges:
                continue
            chains.append(walk(int(start), int(nxt)))

    for start in sorted(adjacency.keys()):
        for nxt in sorted(adjacency.get(int(start), {}).keys()):
            ek = edge_key(int(start), int(nxt))
            if ek in visited_edges:
                continue
            path = walk(int(start), int(nxt))
            prev_node = path[-2]
            cur = path[-1]
            while True:
                candidates = [int(v) for v in sorted(adjacency.get(int(cur), {}).keys()) if int(v) != prev_node]
                if not candidates:
                    break
                cand = int(candidates[0])
                ek2 = edge_key(cur, cand)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                path.append(cand)
                prev_node, cur = cur, cand
            chains.append(path)

    return [list(map(int, c)) for c in chains if len(c) >= 2]


def build_rooted_child_map(prev: Dict[int, int]) -> Dict[int, List[int]]:
    child_map: Dict[int, List[int]] = {}
    for node, parent in prev.items():
        child_map.setdefault(int(parent), []).append(int(node))
    for parent in list(child_map.keys()):
        child_map[parent] = sorted(int(v) for v in child_map[parent])
    return child_map


def collect_rooted_subtree_nodes(child_map: Dict[int, List[int]], start: int) -> set[int]:
    seen: set[int] = set()
    stack = [int(start)]
    while stack:
        node = int(stack.pop())
        if node in seen:
            continue
        seen.add(node)
        for child in reversed(child_map.get(int(node), [])):
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
    if anchor_node not in dist or start not in dist:
        return None
    anchor_dist = float(dist[anchor_node])
    subtree_nodes = collect_rooted_subtree_nodes(child_map, start)
    if not subtree_nodes:
        return None
    endpoints = sorted(int(n) for n in subtree_nodes if len(child_map.get(int(n), [])) == 0)
    if not endpoints:
        endpoints = [int(start)]
    representative_endpoint = max(endpoints, key=lambda n: (float(dist.get(int(n), float("-inf"))), int(n)))
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
        spatial_reach = max(spatial_reach, float(np.linalg.norm(pts[int(node)] - pts[int(anchor_node)])))

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


def build_raw_stem_path(child_map: Dict[int, List[int]], takeoff: int, stem_start: int) -> Dict[str, Any]:
    takeoff = int(takeoff)
    stem_start = int(stem_start)
    stem_path = [takeoff, stem_start]
    cur = int(stem_start)
    first_split = None
    while True:
        children = [int(v) for v in child_map.get(int(cur), [])]
        if len(children) != 1:
            if len(children) >= 2:
                first_split = int(cur)
            break
        nxt = int(children[0])
        stem_path.append(nxt)
        cur = nxt
    return {
        "raw_stem_path": [int(v) for v in stem_path],
        "raw_stem_terminal": int(stem_path[-1]),
        "raw_first_split": int(first_split) if first_split is not None else None,
    }


def rooted_subtree_rank_key(summary: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
    return (
        float(summary.get("subtree_max_length", 0.0)),
        float(summary.get("subtree_total_length", 0.0)),
        float(summary.get("spatial_reach", 0.0)),
        int(summary.get("endpoint_count", 0)),
        -int(summary.get("start", -1)),
    )


def is_substantial_named_stem_child(summary: Dict[str, Any], dominant: Dict[str, Any], parent_system: Dict[str, Any]) -> bool:
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
    cur = int(stem_start)
    first_major_split = None
    split_summaries: List[Dict[str, Any]] = []
    visited = {takeoff, stem_start}

    while True:
        children = [int(v) for v in child_map.get(int(cur), [])]
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
            summary for summary in child_summaries if is_substantial_named_stem_child(summary, dominant, parent_system)
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
        "named_stem_path": [int(v) for v in named_path],
        "named_stem_terminal": int(named_path[-1]),
        "named_first_major_split": int(first_major_split) if first_major_split is not None else None,
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
    takeoff = int(takeoff)
    stem_start = int(stem_start)
    if takeoff not in dist or stem_start not in dist:
        return None
    subtree_summary = summarize_rooted_subtree(child_map, pts, dist, prev, takeoff, stem_start)
    if subtree_summary is None:
        return None

    raw_stem = build_raw_stem_path(child_map, takeoff, stem_start)
    parent_like = dict(subtree_summary)
    named_stem = build_named_stem_path(child_map, pts, dist, prev, takeoff, stem_start, parent_like)

    rep_endpoint = int(subtree_summary["representative_endpoint"])
    rep_point = np.asarray(pts[rep_endpoint], dtype=float)
    takeoff_point = np.asarray(pts[takeoff], dtype=float)
    stem_terminal = int(named_stem["named_stem_terminal"])
    stem_terminal_point = np.asarray(pts[stem_terminal], dtype=float)
    endpoints = [int(v) for v in subtree_summary["endpoints"]]
    subtree_center = np.mean(np.asarray([pts[v] for v in endpoints], dtype=float), axis=0) if endpoints else rep_point

    system = {
        "takeoff": int(takeoff),
        "stem_start": int(stem_start),
        "takeoff_dist": float(dist.get(takeoff, 0.0)),
        "nodes": set(int(v) for v in subtree_summary["nodes"]),
        "endpoints": [int(v) for v in endpoints],
        "representative_endpoint": int(rep_endpoint),
        "subtree_max_length": float(subtree_summary["subtree_max_length"]),
        "subtree_total_length": float(subtree_summary["subtree_total_length"]),
        "endpoint_count": int(subtree_summary["endpoint_count"]),
        "node_count": int(subtree_summary["node_count"]),
        "spatial_reach": float(subtree_summary["spatial_reach"]),
        "raw_stem_path": list(raw_stem["raw_stem_path"]),
        "raw_stem_terminal": int(raw_stem["raw_stem_terminal"]),
        "raw_first_split": raw_stem["raw_first_split"],
        "named_stem_path": list(named_stem["named_stem_path"]),
        "named_stem_terminal": int(named_stem["named_stem_terminal"]),
        "named_first_major_split": named_stem["named_first_major_split"],
        "named_split_child_summaries": list(named_stem["named_split_child_summaries"]),
        "stem_path": list(named_stem["named_stem_path"]),
        "stem_terminal": int(named_stem["named_stem_terminal"]),
        "takeoff_point": takeoff_point.astype(float),
        "subtree_center": np.asarray(subtree_center, dtype=float),
        "direction_vector": unit(rep_point - takeoff_point),
        "stem_vector": unit(stem_terminal_point - takeoff_point),
        "representative_vector": np.asarray(rep_point - takeoff_point, dtype=float),
        "local_vector": np.asarray(rep_point - np.asarray(pts[int(stem_start)], dtype=float), dtype=float),
    }
    return system


def build_direct_child_systems_for_parent_path(
    parent_path: List[int],
    pts: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
) -> List[Dict[str, Any]]:
    child_map = build_rooted_child_map(prev)
    parent_path = [int(v) for v in parent_path]
    next_on_path: Dict[int, Optional[int]] = {}
    for i, node in enumerate(parent_path):
        next_on_path[int(node)] = int(parent_path[i + 1]) if i + 1 < len(parent_path) else None

    systems: List[Dict[str, Any]] = []
    for node in parent_path:
        trunk_child = next_on_path.get(int(node))
        for child in child_map.get(int(node), []):
            if trunk_child is not None and int(child) == int(trunk_child):
                continue
            sys_info = describe_rooted_child_system(child_map, pts, dist, prev, int(node), int(child))
            if sys_info is not None:
                systems.append(sys_info)
    systems.sort(
        key=lambda s: (
            float(s.get("takeoff_dist", 0.0)),
            -float(s.get("subtree_total_length", 0.0)),
            int(s.get("representative_endpoint", -1)),
        )
    )
    return systems


def rooted_child_system_key(system: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if system is None:
        return (-1, -1)
    return (int(system.get("takeoff", -1)), int(system.get("stem_start", -1)))


def rooted_child_system_node_set(system: Optional[Dict[str, Any]], include_takeoff: bool = True) -> set[int]:
    if system is None:
        return set()
    nodes = set(int(v) for v in system.get("nodes", set()))
    if include_takeoff and system.get("takeoff") is not None:
        nodes.add(int(system["takeoff"]))
    return nodes


def find_rooted_child_system_for_endpoint(
    child_map: Dict[int, List[int]],
    pts: np.ndarray,
    dist: Dict[int, float],
    prev: Dict[int, int],
    takeoff: int,
    endpoint: int,
    inlet_node: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if inlet_node is None:
        inlet_node = int(min(dist.keys(), key=lambda k: dist[k])) if dist else int(takeoff)
    path = path_to_root(prev, int(inlet_node), int(endpoint))
    if not path or int(takeoff) not in path:
        return None
    idx = path.index(int(takeoff))
    if idx >= len(path) - 1:
        return None
    stem_start = int(path[idx + 1])
    return describe_rooted_child_system(child_map, pts, dist, prev, int(takeoff), int(stem_start))


def nearest_node_to_point(nodes: List[int], pts: np.ndarray, p: np.ndarray) -> Tuple[int, float]:
    if not nodes:
        return -1, float("inf")
    pp = np.asarray(p, dtype=float).reshape(3)
    d2 = np.sum((pts[np.asarray(nodes, dtype=int)] - pp[None, :]) ** 2, axis=1)
    best_i = int(np.argmin(d2))
    return int(nodes[best_i]), float(math.sqrt(max(0.0, float(d2[best_i]))))


def pick_inlet_node_from_endpoints(
    endpoints: List[int],
    pts: np.ndarray,
    inlet_center: np.ndarray,
    inlet_conf: float,
    warnings: List[str],
) -> Tuple[int, float]:
    if not endpoints:
        return -1, 0.0
    ep, dist0 = nearest_node_to_point(list(endpoints), pts, inlet_center)
    conf = float(clamp(0.40 + 0.40 * clamp(inlet_conf, 0.0, 1.0) + 0.20 * clamp(1.0 / (1.0 + dist0), 0.0, 1.0), 0.0, 1.0))
    if conf < 0.55:
        warnings.append(f"W_INLET_NODE_LOWCONF: inlet-node confidence={conf:.3f}.")
    return int(ep), float(conf)


def deepest_common_node(path_a: List[int], path_b: List[int], dist: Dict[int, float]) -> Optional[int]:
    if not path_a or not path_b:
        return None
    common = set(int(v) for v in path_a).intersection(int(v) for v in path_b)
    if not common:
        return None
    return max(common, key=lambda n: (float(dist.get(int(n), float("-inf"))), int(n)))


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
    dists = np.asarray([np.linalg.norm(pts[int(ep)] - center) for ep in candidates], dtype=float)
    best_idx = int(np.argmin(dists))
    best_ep = int(candidates[best_idx])
    best_dist = float(dists[best_idx])
    diameter_eq = max(6.0, float(termination.diameter_eq))
    conf = float(clamp(1.0 - best_dist / (1.25 * diameter_eq + EPS), 0.0, 1.0))
    return int(best_ep), float(conf)


def choose_distal_iliac_termination_pair(
    terms: List[TerminationLoop],
    inlet_term: Optional[TerminationLoop],
    axis_si: np.ndarray,
    warnings: List[str],
) -> Tuple[Optional[int], Optional[int], float]:
    if len(terms) < 3:
        return None, None, 0.0
    centers = np.asarray([t.center for t in terms], dtype=float).reshape((-1, 3))
    diam = np.asarray([max(0.0, float(t.diameter_eq)) for t in terms], dtype=float)
    axis = unit(axis_si)
    if float(np.linalg.norm(axis)) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    inlet_idx = None
    if inlet_term is not None:
        inlet_center = np.asarray(inlet_term.center, dtype=float).reshape(3)
        d2 = np.sum((centers - inlet_center[None, :]) ** 2, axis=1)
        inlet_idx = int(np.argmin(d2))

    other_idx = [i for i in range(len(terms)) if i != inlet_idx]
    if len(other_idx) < 2:
        return None, None, 0.0

    centroid = np.mean(centers, axis=0)
    proj = (centers - centroid[None, :]) @ axis
    other_proj = proj[other_idx]
    proj_min = float(np.min(other_proj))
    proj_max = float(np.max(other_proj))
    proj_span = max(EPS, proj_max - proj_min)
    distal_cutoff = proj_min + 0.45 * proj_span
    distal_pool = [i for i in other_idx if float(proj[i]) <= distal_cutoff]
    if len(distal_pool) < 2:
        warnings.append("W_BIF_TERM_DISTAL_POOL_SMALL: distal termination pool was small.")
        distal_pool = sorted(other_idx, key=lambda i: float(proj[i]))[: min(6, len(other_idx))]

    ranked_pool = sorted(distal_pool, key=lambda i: (diam[i], -proj[i], -i), reverse=True)[: min(6, len(distal_pool))]
    if len(ranked_pool) < 2:
        return None, None, 0.0

    bbox_diag = float(np.linalg.norm(np.max(centers[ranked_pool], axis=0) - np.min(centers[ranked_pool], axis=0)))
    bbox_diag = max(bbox_diag, EPS)
    pool_diams = diam[ranked_pool]
    diam_min = float(np.min(pool_diams))
    diam_span = max(float(np.max(pool_diams)) - diam_min, EPS)

    best_pair = None
    best_score = -1.0
    best_metrics = None
    for i in range(len(ranked_pool)):
        for j in range(i + 1, len(ranked_pool)):
            ia = int(ranked_pool[i])
            ib = int(ranked_pool[j])
            diam_a = float(diam[ia])
            diam_b = float(diam[ib])
            mean_diam_norm = 0.5 * ((diam_a - diam_min) / diam_span + (diam_b - diam_min) / diam_span)
            diam_balance = 1.0 - abs(diam_a - diam_b) / (diam_a + diam_b + EPS)
            distal_a = (proj_max - float(proj[ia])) / proj_span
            distal_b = (proj_max - float(proj[ib])) / proj_span
            distalness = 0.5 * (distal_a + distal_b)
            dvec = centers[ia] - centers[ib]
            lateral = float(np.linalg.norm(dvec - np.dot(dvec, axis) * axis))
            lateral_norm = lateral / bbox_diag
            score = 1.45 * mean_diam_norm + 2.00 * lateral_norm + 0.90 * distalness + 0.55 * diam_balance
            if score > best_score:
                best_score = float(score)
                best_pair = (ia, ib)
                best_metrics = (mean_diam_norm, diam_balance, distalness, lateral_norm)
    if best_pair is None or best_metrics is None:
        return None, None, 0.0
    mean_diam_norm, diam_balance, distalness, lateral_norm = best_metrics
    conf = float(
        clamp(
            0.18
            + 0.24 * mean_diam_norm
            + 0.22 * diam_balance
            + 0.18 * distalness
            + 0.30 * clamp(1.8 * lateral_norm, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    if conf < 0.55:
        warnings.append("W_BIF_TERM_PAIR_LOWCONF: distal termination pair support was weak.")
    return int(best_pair[0]), int(best_pair[1]), float(conf)


def generate_bifurcation_hypotheses(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    inlet_node: int,
    axis_si: np.ndarray,
    warnings: List[str],
    terminations: Optional[List[TerminationLoop]] = None,
    inlet_term: Optional[TerminationLoop] = None,
    max_hypotheses: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[int, float], Dict[int, int]]:
    inlet_node = int(inlet_node)
    dist, prev = dijkstra(adjacency, inlet_node)
    deg = node_degrees(adjacency)
    endpoints = [int(n) for n, d in deg.items() if int(d) == 1 and int(n) != inlet_node and int(n) in dist]
    if len(endpoints) < 2:
        warnings.append("W_BIF_NOT_ENOUGH_ENDPOINTS: insufficient distal endpoints for bifurcation inference.")
        return [], dist, prev

    bbox_min = np.min(pts[list(adjacency.keys())], axis=0)
    bbox_max = np.max(pts[list(adjacency.keys())], axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    axis = unit(axis_si)
    if float(np.linalg.norm(axis)) < EPS:
        axis = unit(bbox_max - bbox_min)
    if float(np.linalg.norm(axis)) < EPS:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    child_map = build_rooted_child_map(prev)
    endpoints_sorted = sorted(endpoints, key=lambda n: (float(dist.get(int(n), -1.0)), int(n)), reverse=True)
    max_dist = max(EPS, float(max(dist.get(int(n), 0.0) for n in endpoints_sorted)))
    term_pair_bonus: Optional[Dict[str, Any]] = None
    if terminations:
        ta, tb, tconf = choose_distal_iliac_termination_pair(terminations, inlet_term, axis, warnings)
        if ta is not None and tb is not None:
            ep_a, ep_a_conf = choose_centerline_endpoint_for_termination(endpoints, pts, terminations[ta])
            ep_b, ep_b_conf = choose_centerline_endpoint_for_termination(endpoints, pts, terminations[tb], exclude=({ep_a} if ep_a is not None else set()))
            if ep_a is not None and ep_b is not None and int(ep_a) != int(ep_b):
                term_pair_bonus = {
                    "ep_a": int(ep_a),
                    "ep_b": int(ep_b),
                    "conf": float(tconf * min(ep_a_conf, ep_b_conf)),
                }

    hypotheses: List[Dict[str, Any]] = []
    for bif_node in sorted(int(n) for n in child_map.keys() if int(n) in dist):
        rooted_children = [int(child) for child in child_map.get(int(bif_node), []) if int(child) in dist]
        if len(rooted_children) < 2:
            continue
        systems: List[Dict[str, Any]] = []
        for stem_start in rooted_children:
            sys_info = describe_rooted_child_system(child_map, pts, dist, prev, int(bif_node), int(stem_start))
            if sys_info is not None:
                systems.append(sys_info)
        if len(systems) < 2:
            continue

        depth = float(dist.get(int(bif_node), 0.0))
        depth_norm = clamp(depth / (max_dist + EPS), 0.0, 1.0)
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
                divergence = clamp((1.0 - float(np.dot(va, vb))) / 2.0, 0.0, 1.0)
                min_tail_norm = min(len_a, len_b) / (max_dist + EPS)

                term_support = 0.0
                if term_pair_bonus is not None:
                    eps_a = set(int(v) for v in sys_a["endpoints"])
                    eps_b = set(int(v) for v in sys_b["endpoints"])
                    tea = int(term_pair_bonus["ep_a"])
                    teb = int(term_pair_bonus["ep_b"])
                    matched = (tea in eps_a and teb in eps_b) or (tea in eps_b and teb in eps_a)
                    if matched:
                        term_support = 0.10 * float(term_pair_bonus["conf"])

                raw_score = (
                    2.10 * depth_norm
                    + 1.10 * symmetry
                    + 1.55 * lateral_norm
                    + 0.55 * min_tail_norm
                    + 0.30 * divergence
                    + term_support
                    - 0.95 * proximal_penalty
                )
                conf = float(
                    clamp(
                        0.18
                        + 0.30 * depth_norm
                        + 0.22 * symmetry
                        + 0.20 * clamp(2.0 * lateral_norm, 0.0, 1.0)
                        + 0.06 * divergence
                        + term_support,
                        0.0,
                        1.0,
                    )
                )
                hypotheses.append(
                    {
                        "bif_node": int(bif_node),
                        "system_a": sys_a,
                        "system_b": sys_b,
                        "ep_a": int(sys_a["representative_endpoint"]),
                        "ep_b": int(sys_b["representative_endpoint"]),
                        "confidence": float(conf),
                        "score": float(raw_score),
                        "depth_norm": float(depth_norm),
                        "symmetry": float(symmetry),
                        "lateral_norm": float(lateral_norm),
                        "divergence": float(divergence),
                    }
                )

    hypotheses.sort(
        key=lambda h: (
            -float(h["score"]),
            -float(h["confidence"]),
            -float(h["depth_norm"]),
            -float(h["lateral_norm"]),
            int(h["bif_node"]),
        )
    )

    unique: List[Dict[str, Any]] = []
    seen_keys = set()
    for hyp in hypotheses:
        key = (int(hyp["bif_node"]), rooted_child_system_key(hyp["system_a"]), rooted_child_system_key(hyp["system_b"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(hyp)
        if len(unique) >= max(1, int(max_hypotheses)):
            break

    if not unique and len(endpoints_sorted) >= 2:
        a = int(endpoints_sorted[0])
        b = int(endpoints_sorted[1])
        pa = path_to_root(prev, inlet_node, a)
        pb = path_to_root(prev, inlet_node, b)
        lca = deepest_common_node(pa, pb, dist)
        if lca is not None:
            sys_a = find_rooted_child_system_for_endpoint(child_map, pts, dist, prev, int(lca), a, inlet_node=inlet_node)
            sys_b = find_rooted_child_system_for_endpoint(child_map, pts, dist, prev, int(lca), b, inlet_node=inlet_node)
            if sys_a is not None and sys_b is not None:
                unique.append(
                    {
                        "bif_node": int(lca),
                        "system_a": sys_a,
                        "system_b": sys_b,
                        "ep_a": int(a),
                        "ep_b": int(b),
                        "confidence": 0.22,
                        "score": 0.22,
                        "depth_norm": float(clamp(float(dist.get(int(lca), 0.0)) / (max_dist + EPS), 0.0, 1.0)),
                        "symmetry": 0.0,
                        "lateral_norm": 0.0,
                        "divergence": 0.0,
                    }
                )
                warnings.append("W_BIF_FALLBACK: used farthest-endpoint fallback for bifurcation.")
    if unique and float(unique[0]["confidence"]) < 0.60:
        warnings.append(f"W_BIF_LOWER_CONF: best bifurcation confidence={float(unique[0]['confidence']):.3f}.")
    return unique, dist, prev


def estimate_superior_axis(inlet_pt: np.ndarray, bif_pt: np.ndarray, all_pts: np.ndarray, warnings: List[str]) -> np.ndarray:
    ez = unit(np.asarray(inlet_pt, dtype=float) - np.asarray(bif_pt, dtype=float))
    if float(np.linalg.norm(ez)) < EPS:
        warnings.append("W_FRAME_EZ_DEGEN: inlet and bifurcation were nearly coincident; falling back to PCA.")
        axes, _, _ = pca_axes(all_pts)
        ez = unit(axes[:, 0])
    if float(np.linalg.norm(ez)) < EPS:
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
    ex = unit(project_vector_to_plane(a - b, ez))
    if float(np.linalg.norm(ex)) < EPS:
        warnings.append("W_FRAME_EX_DEGEN: iliac lateral axis was degenerate; falling back to PCA.")
        axes, _, _ = pca_axes(all_pts)
        ex = unit(project_vector_to_plane(axes[:, 1], ez))
    if float(np.linalg.norm(ex)) < EPS:
        ex = unit(project_vector_to_plane(np.array([1.0, 0.0, 0.0], dtype=float), ez))
    return ex.astype(float)


def complete_canonical_frame(ez: np.ndarray, horizontal_hint: np.ndarray, all_pts: np.ndarray, warnings: List[str]) -> Tuple[np.ndarray, float]:
    ez_u = unit(ez)
    ex = unit(project_vector_to_plane(horizontal_hint, ez_u))
    if float(np.linalg.norm(ex)) < EPS:
        warnings.append("W_FRAME_EX_HINT_DEGEN: horizontal hint collapsed after projection; using PCA.")
        axes, _, _ = pca_axes(all_pts)
        ex = unit(project_vector_to_plane(axes[:, 1], ez_u))
        if float(np.linalg.norm(ex)) < EPS:
            ex = unit(project_vector_to_plane(np.array([1.0, 0.0, 0.0], dtype=float), ez_u))
    ey = unit(np.cross(ez_u, ex))
    if float(np.linalg.norm(ey)) < EPS:
        warnings.append("W_FRAME_EY_DEGEN: frame cross-product degenerate; repairing basis.")
        axes, _, _ = pca_axes(all_pts)
        ey = unit(project_vector_to_plane(axes[:, 2], ez_u))
        if float(np.linalg.norm(ey)) < EPS:
            ey = unit(np.cross(ez_u, np.array([1.0, 0.0, 0.0], dtype=float)))
    ex = unit(np.cross(ey, ez_u))
    ey = unit(np.cross(ez_u, ex))
    r = np.vstack([ex, ey, ez_u]).astype(float)
    ortho_err = float(np.linalg.norm(r @ r.T - np.eye(3)))
    conf = float(clamp(1.0 - ortho_err, 0.0, 1.0))
    return r.astype(float), float(conf)


def build_canonical_transform(
    inlet_pt: np.ndarray,
    bif_pt: np.ndarray,
    iliac_ep_a_pt: np.ndarray,
    iliac_ep_b_pt: np.ndarray,
    all_pts: np.ndarray,
    warnings: List[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    inlet_pt = np.asarray(inlet_pt, dtype=float).reshape(3)
    bif_pt = np.asarray(bif_pt, dtype=float).reshape(3)
    ez = estimate_superior_axis(inlet_pt, bif_pt, all_pts, warnings)
    ex_hint = estimate_horizontal_axis_from_iliacs(iliac_ep_a_pt, iliac_ep_b_pt, ez, all_pts, warnings)
    r, conf = complete_canonical_frame(ez, ex_hint, all_pts, warnings)
    origin = bif_pt.astype(float)
    return r.astype(float), origin.astype(float), float(conf)


def apply_transform_points(points: np.ndarray, r: np.ndarray, origin: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    rot = np.asarray(r, dtype=float).reshape((3, 3))
    org = np.asarray(origin, dtype=float).reshape((3,))
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    return ((pts - org[None, :]) @ rot.T).astype(float)


def apply_transform_to_polydata(pd: vtkPolyData, r: np.ndarray, origin: np.ndarray) -> vtkPolyData:
    out = clone_polydata(pd)
    pts = get_points_numpy(out)
    pts_c = apply_transform_points(pts, r, origin)
    new_pts = vtk.vtkPoints()
    new_pts.SetNumberOfPoints(int(pts_c.shape[0]))
    for i in range(int(pts_c.shape[0])):
        new_pts.SetPoint(i, float(pts_c[i, 0]), float(pts_c[i, 1]), float(pts_c[i, 2]))
    out.SetPoints(new_pts)
    out.BuildLinks()
    return out


def _orient_chain_from_scaffold(chain: List[int], scaffold_set: set[int]) -> Optional[List[int]]:
    if len(chain) < 2:
        return None
    head_in = int(chain[0]) in scaffold_set
    tail_in = int(chain[-1]) in scaffold_set
    if head_in and not tail_in:
        return [int(v) for v in chain]
    if tail_in and not head_in:
        return [int(v) for v in reversed(chain)]
    return None


def _collect_component_nodes_excluding_scaffold(
    adjacency: Dict[int, Dict[int, float]],
    start: int,
    scaffold_set: set[int],
) -> set[int]:
    start = int(start)
    if start in scaffold_set:
        return set()
    seen: set[int] = set()
    stack = [start]
    while stack:
        node = int(stack.pop())
        if node in seen or node in scaffold_set:
            continue
        seen.add(node)
        for nbr in adjacency.get(node, {}):
            if int(nbr) not in seen and int(nbr) not in scaffold_set:
                stack.append(int(nbr))
    return seen


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
    component_nodes = set(int(v) for v in component_nodes)
    endpoints = sorted(int(n) for n in component_nodes if deg.get(int(n), 0) == 1)
    if not endpoints:
        endpoints = sorted(component_nodes, key=lambda n: (float(dist.get(int(n), float("-inf"))), int(n)), reverse=True)[:1]
    if not endpoints:
        return None, []
    best = None
    best_score = -1.0
    for ep in endpoints:
        path = path_to_root(prev, int(inlet_node), int(ep))
        if not path or int(takeoff) not in path:
            continue
        idx = path.index(int(takeoff))
        seg_nodes = path[idx:]
        seg_pts = pts_c[np.asarray(seg_nodes, dtype=int)]
        branch_len = polyline_length(seg_pts)
        if branch_len <= EPS:
            continue
        vec = np.asarray(seg_pts[-1] - seg_pts[0], dtype=float)
        total = max(EPS, float(np.linalg.norm(vec)))
        reach_xy = float(np.linalg.norm(vec[:2]))
        vertical = abs(float(vec[2])) / total
        local_horizontal = clamp(reach_xy / total, 0.0, 1.0)
        rep_hdir = unit_xy(vec)
        local_hdir = rep_hdir.copy()
        rep_horizontal = local_horizontal
        representative_score = float(
            clamp(
                0.45 * local_horizontal
                + 0.30 * clamp(reach_xy / max(1.0, 0.12 * trunk_len), 0.0, 1.0)
                + 0.25 * clamp(branch_len / max(1.0, 0.12 * trunk_len), 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        cand = {
            "ep": int(ep),
            "seg_nodes": list(seg_nodes),
            "branch_len": float(branch_len),
            "local_hdir": local_hdir.astype(float),
            "rep_hdir": rep_hdir.astype(float),
            "local_horizontal": float(local_horizontal),
            "rep_horizontal": float(rep_horizontal),
            "local_vertical": float(vertical),
            "reach_xy": float(reach_xy),
            "representative_score": float(representative_score),
            "rep_point": np.asarray(pts_c[int(ep)], dtype=float),
        }
        if representative_score > best_score:
            best_score = float(representative_score)
            best = cand
    return best, endpoints


def _should_absorb_chain_into_trunk_scaffold(oriented: List[int], pts_c: np.ndarray, takeoff_dist: float, trunk_len: float) -> bool:
    if len(oriented) < 2:
        return False
    pts = pts_c[np.asarray(oriented, dtype=int)]
    length = polyline_length(pts)
    if length <= 0.0:
        return False
    vec = np.asarray(pts[-1] - pts[0], dtype=float)
    total = max(EPS, float(np.linalg.norm(vec)))
    verticality = abs(float(vec[2])) / total
    horizontality = float(np.linalg.norm(vec[:2])) / total
    if takeoff_dist < 0.08 * trunk_len or takeoff_dist > 0.85 * trunk_len:
        return False
    return bool(length <= 0.22 * trunk_len and verticality >= 0.70 and horizontality <= 0.42)


def _expand_trunk_scaffold_for_renal_scan(
    adjacency: Dict[int, Dict[int, float]],
    pts_c: np.ndarray,
    trunk_path: List[int],
    dist: Dict[int, float],
    bif_node: int,
) -> Tuple[set[int], List[int], List[List[int]]]:
    trunk_len = float(dist.get(int(bif_node), 0.0))
    scaffold_set: set[int] = {int(v) for v in trunk_path}
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
            takeoff_dist = float(dist.get(int(takeoff), float("nan")))
            if not math.isfinite(takeoff_dist):
                continue
            if not _should_absorb_chain_into_trunk_scaffold(oriented, pts_c, takeoff_dist, trunk_len):
                continue
            new_nodes = [int(v) for v in oriented[1:] if int(v) not in scaffold_set]
            if not new_nodes:
                continue
            scaffold_set.update(new_nodes)
            absorbed_chain_ids.add(int(chain_id))
            changed = True

    return scaffold_set, sorted(int(v) for v in absorbed_chain_ids), chains


def _nearest_node_on_path(path_nodes: List[int], pts_c: np.ndarray, query_node: int) -> int:
    if not path_nodes:
        return int(query_node)
    q = np.asarray(pts_c[int(query_node)], dtype=float)
    return int(min((int(v) for v in path_nodes), key=lambda n: float(np.linalg.norm(pts_c[int(n)] - q))))


def discover_renal_branch_candidates(
    adjacency: Dict[int, Dict[int, float]],
    pts_c: np.ndarray,
    inlet_node: int,
    bif_node: int,
    trunk_set: set[int],
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    warnings: List[str],
    excluded_system_nodes: Optional[set[int]] = None,
) -> Dict[str, Any]:
    deg = node_degrees(adjacency)
    trunk_len = float(dist.get(int(bif_node), 0.0))
    if trunk_len <= 0.0:
        warnings.append("W_RENAL_TRUNKLEN: invalid trunk length for renal scan.")
        trunk_len = 1.0
    excluded_nodes = set(int(v) for v in (excluded_system_nodes or set()))
    trunk_path = [int(v) for v in path_to_root(prev, int(inlet_node), int(bif_node))]
    scaffold_set, scaffold_chain_ids, chains = _expand_trunk_scaffold_for_renal_scan(adjacency, pts_c, trunk_path, dist, bif_node)

    candidates: List[Dict[str, Any]] = []
    for chain_id, chain in enumerate(chains):
        oriented = _orient_chain_from_scaffold(chain, scaffold_set)
        if oriented is None or len(oriented) < 2:
            continue
        takeoff = int(oriented[0])
        takeoff_dist = float(dist.get(int(takeoff), float("nan")))
        if not math.isfinite(takeoff_dist):
            continue
        trunk_takeoff = _nearest_node_on_path(trunk_path, pts_c, takeoff)
        trunk_takeoff_dist = float(dist.get(int(trunk_takeoff), takeoff_dist))
        if trunk_takeoff_dist < 0.08 * trunk_len or trunk_takeoff_dist > 0.82 * trunk_len:
            continue
        if takeoff in (int(inlet_node), int(bif_node)):
            continue

        component_nodes = _collect_component_nodes_excluding_scaffold(adjacency, int(oriented[1]), scaffold_set)
        if not component_nodes:
            continue
        if excluded_nodes and component_nodes.intersection(excluded_nodes):
            continue
        if int(iliac_ep_a) in component_nodes or int(iliac_ep_b) in component_nodes:
            continue

        rep, component_endpoints = _choose_representative_endpoint_for_component(
            component_nodes, takeoff, pts_c, deg, dist, prev, inlet_node, trunk_len
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
        hdir = unit_xy(np.asarray(rep["rep_hdir"], dtype=float))
        if float(np.linalg.norm(hdir)) < EPS:
            hdir = unit_xy(np.asarray(rep["local_hdir"], dtype=float))
        if float(np.linalg.norm(hdir)) < EPS:
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
                "source_chain_absorbed_to_scaffold": bool(int(chain_id) in scaffold_chain_ids),
                "rep_point": np.asarray(rep["rep_point"], dtype=float),
                "score": float(score),
            }
        )

    pair_candidates: List[Dict[str, Any]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            dt = abs(float(a["takeoff_dist"]) - float(b["takeoff_dist"]))
            dt_norm = dt / (0.45 * trunk_len + EPS)
            takeoff_sim = clamp(1.0 - dt_norm, 0.0, 1.0)
            dl = abs(float(a["branch_len"]) - float(b["branch_len"])) / (float(a["branch_len"]) + float(b["branch_len"]) + EPS)
            len_sim = clamp(1.0 - dl, 0.0, 1.0)

            dir_opp = clamp((1.0 - float(np.dot(a["hdir"], b["hdir"]))) / 2.0, 0.0, 1.0)
            if dir_opp < 0.38:
                continue

            axis_vec = np.asarray(a["rep_point"], dtype=float) - np.asarray(b["rep_point"], dtype=float)
            axis_vec[2] = 0.0
            if float(np.linalg.norm(axis_vec)) < EPS:
                axis_vec = np.asarray(a["hdir"], dtype=float) - np.asarray(b["hdir"], dtype=float)
                axis_vec[2] = 0.0
            axis = unit_xy(axis_vec)
            if float(np.linalg.norm(axis)) < EPS:
                continue

            horiz_pair = clamp(0.5 * (float(a["horizontality"]) + float(b["horizontality"])), 0.0, 1.0)
            reach_balance = clamp(min(float(a["reach_xy"]), float(b["reach_xy"])) / (max(float(a["reach_xy"]), float(b["reach_xy"])) + EPS), 0.0, 1.0)
            axis_span = clamp(float(np.linalg.norm(axis_vec[:2])) / (float(a["reach_xy"]) + float(b["reach_xy"]) + EPS), 0.0, 1.0)
            geometry_score = clamp(0.55 * dir_opp + 0.25 * reach_balance + 0.20 * axis_span, 0.0, 1.0)
            axis_conf = clamp(0.45 * geometry_score + 0.35 * horiz_pair + 0.20 * axis_span, 0.0, 1.0)
            mean_s_rel = 0.5 * (float(a["s_rel"]) + float(b["s_rel"]))
            takeoff_zone = clamp(1.0 - abs(mean_s_rel - 0.34) / 0.20, 0.0, 1.0)
            attachment_score = clamp(min(float(a["attachment_score"]), float(b["attachment_score"])), 0.0, 1.0)
            cranial_pair_penalty = 0.18 * clamp((0.20 - min(float(a["s_rel"]), float(b["s_rel"]))) / 0.12, 0.0, 1.0)
            caudal_pair_penalty = 0.10 * clamp((max(float(a["s_rel"]), float(b["s_rel"])) - 0.62) / 0.18, 0.0, 1.0)
            score = float(
                clamp(
                    0.30 * geometry_score
                    + 0.20 * takeoff_sim
                    + 0.15 * horiz_pair
                    + 0.10 * len_sim
                    + 0.10 * (0.5 * (float(a["score"]) + float(b["score"])))
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

    candidates.sort(key=lambda d: (-float(d["score"]), -float(d["representative_score"]), int(d["ep"])))
    pair_candidates.sort(key=lambda d: (-float(d["score"]), -float(d["confidence"])))
    if not candidates:
        warnings.append("W_RENAL_CAND_DISCOVERY_EMPTY: no plausible renal side-branch candidates were found.")
    return {
        "trunk_len": float(trunk_len),
        "trunk_scaffold_node_count": int(len(scaffold_set)),
        "trunk_scaffold_extra_node_count": int(max(0, len(scaffold_set) - len(trunk_set))),
        "trunk_scaffold_chain_count": int(len(scaffold_chain_ids)),
        "candidates": list(candidates),
        "pair_candidates": list(pair_candidates),
        "best_pair": pair_candidates[0] if pair_candidates else None,
        "best_single": candidates[0] if candidates else None,
    }


def collect_visceral_branch_axis(
    pts_c: np.ndarray,
    adjacency: Dict[int, Dict[int, float]],
    inlet_node: int,
    bif_node: int,
    trunk_path: List[int],
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    excluded_endpoints: Optional[set[int]] = None,
    excluded_system_nodes: Optional[set[int]] = None,
) -> Tuple[Optional[np.ndarray], float, int]:
    excluded = {int(iliac_ep_a), int(iliac_ep_b)}
    if excluded_endpoints:
        excluded.update(int(v) for v in excluded_endpoints if v is not None)
    excluded_nodes = set(int(v) for v in (excluded_system_nodes or set()))
    trunk_len = float(dist.get(int(bif_node), 0.0))
    if trunk_len <= 0.0 or not trunk_path:
        return None, 0.0, 0

    systems = build_direct_child_systems_for_parent_path(trunk_path, pts_c, dist, prev)
    vecs: List[np.ndarray] = []
    weights: List[float] = []
    for system in systems:
        if excluded_nodes and rooted_child_system_node_set(system, include_takeoff=False).intersection(excluded_nodes):
            continue
        if excluded.intersection(int(v) for v in system["endpoints"]):
            continue
        takeoff_dist = float(system["takeoff_dist"])
        branch_len = float(system["subtree_max_length"])
        if not math.isfinite(takeoff_dist) or branch_len <= 0.02 * trunk_len:
            continue
        s_rel = takeoff_dist / (trunk_len + EPS)
        if s_rel < 0.08 or s_rel > 0.72:
            continue
        v = np.asarray(system["stem_vector"], dtype=float)
        if float(np.linalg.norm(v)) < EPS:
            v = np.asarray(system["representative_vector"], dtype=float)
        hdir = unit_xy(v)
        if float(np.linalg.norm(hdir)) < EPS:
            continue
        vhat = unit(v)
        vertical = abs(float(vhat[2]))
        if vertical > 0.80:
            continue
        weight = clamp((0.78 - vertical) / 0.60, 0.0, 1.0) * clamp((0.72 - s_rel) / 0.55, 0.0, 1.0)
        if weight <= 0.0:
            continue
        vecs.append(np.asarray(hdir[:2], dtype=float))
        weights.append(float(weight))

    if not vecs:
        return None, 0.0, 0
    if len(vecs) == 1:
        axis = np.array([vecs[0][0], vecs[0][1], 0.0], dtype=float)
        conf = clamp(0.25 * weights[0], 0.0, 1.0)
        return unit_xy(axis), float(conf), 1

    scatter = np.zeros((2, 2), dtype=float)
    for vec, weight in zip(vecs, weights):
        vv = np.asarray(vec, dtype=float).reshape((2, 1))
        scatter += float(weight) * (vv @ vv.T)

    vals, vecs_eig = np.linalg.eigh(scatter)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs_eig = vecs_eig[:, order]
    axis2 = unit(np.array([vecs_eig[0, 0], vecs_eig[1, 0], 0.0], dtype=float))
    dominance = float((vals[0] - vals[1]) / (vals[0] + vals[1] + EPS))
    support = clamp(sum(weights) / max(2.0, float(len(weights))), 0.0, 1.0)
    conf = clamp(0.20 + 0.55 * dominance + 0.25 * support, 0.0, 1.0)
    return axis2.astype(float), float(conf), int(len(vecs))


def collect_ap_orientation_cues(
    pts_c: np.ndarray,
    inlet_node: int,
    bif_node: int,
    trunk_path: List[int],
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    renal_eps: Tuple[Optional[int], Optional[int]],
    excluded_system_nodes: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    exclude = {int(iliac_ep_a), int(iliac_ep_b)}
    if renal_eps[0] is not None:
        exclude.add(int(renal_eps[0]))
    if renal_eps[1] is not None:
        exclude.add(int(renal_eps[1]))
    excluded_nodes = set(int(v) for v in (excluded_system_nodes or set()))
    trunk_len = float(dist.get(int(bif_node), 0.0))
    if trunk_len <= 0.0 or not trunk_path:
        return []

    systems = build_direct_child_systems_for_parent_path(trunk_path, pts_c, dist, prev)
    cues: List[Dict[str, Any]] = []
    for system in systems:
        if excluded_nodes and rooted_child_system_node_set(system, include_takeoff=False).intersection(excluded_nodes):
            continue
        if exclude.intersection(int(v) for v in system["endpoints"]):
            continue
        s_rel = float(system["takeoff_dist"]) / (trunk_len + EPS)
        if s_rel < 0.10 or s_rel > 0.75:
            continue
        v = np.asarray(system["local_vector"], dtype=float)
        if float(np.linalg.norm(v)) < EPS:
            v = np.asarray(system["stem_vector"], dtype=float)
        if float(np.linalg.norm(v)) < EPS:
            v = np.asarray(system["representative_vector"], dtype=float)
        if float(np.linalg.norm(v)) < EPS:
            continue
        vhat = unit(v)
        anterior = abs(float(vhat[1]))
        lateral = abs(float(vhat[0]))
        vertical = abs(float(vhat[2]))
        if anterior < 0.45 or anterior < lateral or vertical > 0.80:
            continue
        weight = clamp((anterior - 0.45) / 0.45, 0.0, 1.0)
        sign = 1.0 if float(vhat[1]) >= 0.0 else -1.0
        cues.append(
            {
                "ep": int(system["representative_endpoint"]),
                "takeoff": int(system["takeoff"]),
                "weight": float(weight),
                "sign_y": float(sign),
                "vhat": vhat.astype(float),
            }
        )
    return cues


def refine_horizontal_axes_using_branch_anatomy(
    r_provisional: np.ndarray,
    pts_c_provisional: np.ndarray,
    iliac_main_a: List[int],
    iliac_main_b: List[int],
    renal_scan: Dict[str, Any],
    inlet_node: int,
    bif_node: int,
    trunk_path: List[int],
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
            ref = unit_xy(pts_c_provisional[int(iliac_main_a[-1])] - pts_c_provisional[int(iliac_main_b[-1])])
        if float(np.linalg.norm(ref)) < EPS:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
        xy_extent = float(np.ptp(pts_c_provisional[:, 0]) + np.ptp(pts_c_provisional[:, 1])) if pts_c_provisional.size else 0.0
        iliac_sep = 0.0
        if iliac_main_a and iliac_main_b:
            iliac_sep = float(np.linalg.norm((pts_c_provisional[int(iliac_main_a[-1])] - pts_c_provisional[int(iliac_main_b[-1])])[:2]))
        conf = clamp(iliac_sep / (xy_extent + EPS), 0.0, 1.0)
        return ref.astype(float), float(conf)

    def refine_from_renal_pair(pair: Dict[str, Any], iliac_ref_axis: np.ndarray, visceral_axis_hint: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        pair_axis = unit_xy(np.asarray(pair.get("axis", np.zeros((3,), dtype=float)), dtype=float))
        if float(np.linalg.norm(pair_axis)) < EPS:
            return r_provisional.copy(), {
                "rotation_degrees_about_z": 0.0,
                "axis_confidence": 0.0,
                "geometry_score": 0.0,
                "takeoff_similarity_score": 0.0,
                "takeoff_zone_score": 0.0,
                "horizontality_score": 0.0,
                "axis_alignment_with_iliacs": 0.0,
                "visceral_orthogonality": 0.0,
            }
        if float(np.linalg.norm(iliac_ref_axis)) >= EPS and float(np.dot(pair_axis[:2], iliac_ref_axis[:2])) < 0.0:
            pair_axis *= -1.0
        axis_alignment = clamp(abs(float(np.dot(pair_axis[:2], iliac_ref_axis[:2]))), 0.0, 1.0) if float(np.linalg.norm(iliac_ref_axis)) >= EPS else 0.0
        visceral_orthogonality = 0.0
        if visceral_axis_hint is not None:
            visceral_axis_xy = unit_xy(np.asarray(visceral_axis_hint, dtype=float))
            if float(np.linalg.norm(visceral_axis_xy)) >= EPS:
                visceral_orthogonality = clamp(
                    math.sqrt(max(0.0, 1.0 - float(np.dot(pair_axis[:2], visceral_axis_xy[:2])) ** 2)),
                    0.0,
                    1.0,
                )
        ey_axis = np.array([-pair_axis[1], pair_axis[0], 0.0], dtype=float)
        q = np.vstack([pair_axis, ey_axis, np.array([0.0, 0.0, 1.0], dtype=float)]).astype(float)
        r_out = (q @ np.asarray(r_provisional, dtype=float)).astype(float)
        rot_deg = float(math.degrees(math.atan2(float(pair_axis[1]), float(pair_axis[0]))))
        return r_out, {
            "rotation_degrees_about_z": float(rot_deg),
            "axis_confidence": float(clamp(pair.get("axis_confidence", pair.get("confidence", 0.0)), 0.0, 1.0)),
            "geometry_score": float(clamp(pair.get("geometry_score", 0.0), 0.0, 1.0)),
            "takeoff_similarity_score": float(clamp(pair.get("takeoff_similarity", 0.0), 0.0, 1.0)),
            "takeoff_zone_score": float(clamp(pair.get("takeoff_zone_score", 0.0), 0.0, 1.0)),
            "horizontality_score": float(clamp(pair.get("horizontality_score", 0.0), 0.0, 1.0)),
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
        {},
        inlet_node,
        bif_node,
        trunk_path,
        dist,
        prev,
        iliac_ep_a,
        iliac_ep_b,
        excluded_endpoints=excluded_renal_eps,
        excluded_system_nodes=iliac_excluded_system_nodes,
    )

    default_info = {
        "rotation_degrees_about_z": 0.0,
        "axis_confidence": 0.0,
        "geometry_score": float(clamp(renal_pair.get("geometry_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "takeoff_similarity_score": float(clamp(renal_pair.get("takeoff_similarity", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "takeoff_zone_score": float(clamp(renal_pair.get("takeoff_zone_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "horizontality_score": float(clamp(renal_pair.get("horizontality_score", 0.0), 0.0, 1.0)) if renal_pair is not None else 0.0,
        "axis_alignment_with_iliacs": 0.0,
        "visceral_orthogonality": 0.0,
    }

    if renal_pair is not None:
        r_candidate, renal_info = refine_from_renal_pair(renal_pair, iliac_ref, visceral_axis)
        rotation_abs = abs(float(renal_info["rotation_degrees_about_z"]))
        renal_alignment = float(renal_info["axis_alignment_with_iliacs"])
        visceral_ortho = float(renal_info["visceral_orthogonality"])
        allow = bool(
            renal_pair_conf >= 0.45
            and renal_alignment >= 0.70
            and (rotation_abs <= 35.0 or (renal_pair_conf >= 0.78 and visceral_ortho >= 0.65 and rotation_abs <= 55.0))
        )
    else:
        r_candidate = r_provisional.copy()
        renal_info = dict(default_info)
        allow = False

    if allow:
        r_refined = r_candidate
        source = "renal_pair_consistent_with_iliacs"
        refined = bool(abs(float(renal_info["rotation_degrees_about_z"])) > 1.0)
        horizontal_conf = clamp(
            0.42 * renal_pair_conf
            + 0.18 * float(renal_info["axis_confidence"])
            + 0.12 * float(renal_info["geometry_score"])
            + 0.10 * float(renal_info["takeoff_zone_score"])
            + 0.08 * float(renal_info["axis_alignment_with_iliacs"])
            + 0.10 * float(renal_info["visceral_orthogonality"])
            + 0.15 * iliac_conf
            + 0.05 * visceral_conf,
            0.0,
            1.0,
        )
        if renal_pair_conf < 0.60 or abs(float(renal_info["rotation_degrees_about_z"])) > 25.0:
            warnings.append(
                f"W_FRAME_RENAL_PAIR_CAUTION: renal-pair confidence={renal_pair_conf:.3f}, rotation={abs(float(renal_info['rotation_degrees_about_z'])):.1f} deg."
            )
    else:
        r_refined = r_provisional.copy()
        renal_info = dict(default_info)
        source = "iliac_primary"
        refined = False
        horizontal_conf = clamp(0.42 + 0.40 * iliac_conf + 0.18 * visceral_conf, 0.0, 1.0)
        if renal_pair is None:
            warnings.append("W_FRAME_RENAL_PRIMARY_FAILED: no bilateral renal pair was available for horizontal-frame refinement.")
        else:
            warnings.append("W_FRAME_RENAL_PRIMARY_REJECTED: renal-pair axis was not sufficiently consistent with the iliac axis.")

    ortho_err = float(np.linalg.norm(r_refined @ r_refined.T - np.eye(3)))
    ortho_conf = float(clamp(1.0 - ortho_err, 0.0, 1.0))
    if horizontal_conf < 0.60:
        warnings.append(f"W_FRAME_HORIZONTAL_WEAK: horizontal-frame confidence={horizontal_conf:.3f} (source={source}).")

    info = {
        "source": str(source),
        "confidence": float(horizontal_conf),
        "ortho_confidence": float(ortho_conf),
        "renal_refinement_used": bool(allow),
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
    }
    return r_refined.astype(float), info


def identify_renal_branches(
    adjacency: Dict[int, Dict[int, float]],
    pts_c: np.ndarray,
    inlet_node: int,
    bif_node: int,
    trunk_set: set[int],
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    warnings: List[str],
    renal_scan: Optional[Dict[str, Any]] = None,
    excluded_system_nodes: Optional[set[int]] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], float, Dict[str, Any]]:
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
                horiz_pair = 0.5 * (float(a["horizontality"]) + float(b["horizontality"]))
                geometry = 0.5 * (float(a["score"]) + float(b["score"]))
                takeoff_zone = 0.5 * (float(a.get("renal_zone_score", 0.0)) + float(b.get("renal_zone_score", 0.0)))
                attachment_score = min(float(a.get("attachment_score", 0.0)), float(b.get("attachment_score", 0.0)))
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
            sided_pairs.sort(key=lambda item: (-float(item[0]), -float(item[3]["geometry_score"])))
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
        best = max(candidates, key=lambda d: (float(d["score"]), float(d["renal_zone_score"]), -int(d["ep"])))
        if float(best.get("score", 0.0)) < 0.25 or float(best.get("renal_zone_score", 0.0)) < 0.10:
            warnings.append("W_RENAL_SINGLE_REJECTED: best single candidate was not anatomically convincing enough.")
            diag["selected_pair_score"] = 0.0
            return None, None, None, None, 0.0, diag
        side = "right" if float(pts_c[int(best["ep"])][0]) >= 0.0 else "left"
        warnings.append(f"W_RENAL_SINGLE: only one renal branch was confidently assigned as {side}.")
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
        warnings.append(f"W_RENAL_LOWER_CONF: renal-assignment confidence={conf:.3f}.")
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


def resolve_anterior_posterior_sign(
    pts_c: np.ndarray,
    inlet_node: int,
    bif_node: int,
    trunk_path: List[int],
    dist: Dict[int, float],
    prev: Dict[int, int],
    iliac_ep_a: int,
    iliac_ep_b: int,
    renal_eps: Tuple[Optional[int], Optional[int]],
    warnings: List[str],
    excluded_system_nodes: Optional[set[int]] = None,
) -> Tuple[bool, float, bool]:
    cues = collect_ap_orientation_cues(
        pts_c,
        inlet_node,
        bif_node,
        trunk_path,
        dist,
        prev,
        iliac_ep_a,
        iliac_ep_b,
        renal_eps,
        excluded_system_nodes=excluded_system_nodes,
    )
    if not cues:
        warnings.append("W_AP_NO_CUES: insufficient anterior-posterior cues; AP sign left as-is.")
        return False, 0.0, True
    wsum = sum(float(c["weight"]) for c in cues) + EPS
    vote = sum(float(c["weight"]) * float(c["sign_y"]) for c in cues) / wsum
    conf = float(clamp(abs(vote), 0.0, 1.0))
    need_flip = bool(vote < 0.0 and conf >= 0.60)
    warn = bool(conf < 0.60)
    if warn:
        warnings.append(f"W_AP_LOWCONF: AP confidence={conf:.3f} from {len(cues)} cues.")
    return need_flip, float(conf), bool(warn)


def build_surface_label_segment_bank(
    branch_geoms: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    bank: Dict[int, Dict[str, Any]] = {}
    for br in branch_geoms:
        display_label = int(br.get("label_id", LABEL_OTHER))
        parent_label = br.get("topology_parent_label_id")
        label_targets: List[int] = []
        if display_label != LABEL_OTHER:
            label_targets.append(int(display_label))
        elif parent_label is not None and int(parent_label) == LABEL_AORTA_TRUNK:
            label_targets.append(LABEL_AORTA_TRUNK)

        pts = np.asarray(br.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
        if pts.shape[0] < 2 or not label_targets:
            continue
        abscissa = compute_abscissa(pts)
        tangents = compute_tangents(pts)
        landmark_point_ids = {str(k): int(v) for k, v in dict(br.get("landmark_point_ids", {}) or {}).items()}

        for label_id in label_targets:
            entry = bank.setdefault(
                int(label_id),
                {
                    "name": LABEL_ID_TO_NAME[int(label_id)],
                    "segment_p0": [],
                    "segment_p1": [],
                    "branches": [],
                },
            )
            entry["segment_p0"].append(pts[:-1].astype(float))
            entry["segment_p1"].append(pts[1:].astype(float))
            entry["branches"].append(
                {
                    "points": pts.astype(float),
                    "abscissa": abscissa.astype(float),
                    "tangents": tangents.astype(float),
                    "landmark_point_ids": dict(landmark_point_ids),
                    "length": float(abscissa[-1] if abscissa.size else 0.0),
                }
            )

    out: Dict[int, Dict[str, Any]] = {}
    for label_id, entry in bank.items():
        if not entry["segment_p0"]:
            continue
        seg_p0 = np.concatenate(entry["segment_p0"], axis=0).astype(float)
        seg_p1 = np.concatenate(entry["segment_p1"], axis=0).astype(float)
        out[int(label_id)] = {
            "name": str(entry["name"]),
            "segment_p0": seg_p0,
            "segment_p1": seg_p1,
            "segment_count": int(seg_p0.shape[0]),
            "branches": list(entry["branches"]),
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
    idx = max(0, min(idx, int(pts.shape[0]) - 2))
    ds = float(s[idx + 1] - s[idx])
    if ds <= EPS:
        return pts[idx].astype(float)
    t = clamp((target - float(s[idx])) / ds, 0.0, 1.0)
    return ((1.0 - t) * pts[idx] + t * pts[idx + 1]).astype(float)


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
        idx = int(max(0, min(int(landmark_point_ids[landmark_key]), int(pts.shape[0]) - 1)))
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


def nearest_cell_to_point(cell_centers: np.ndarray, target_point: np.ndarray, allowed_mask: Optional[np.ndarray] = None) -> int:
    pts = np.asarray(cell_centers, dtype=float)
    if pts.shape[0] == 0:
        return -1
    target = np.asarray(target_point, dtype=float).reshape(3)
    if allowed_mask is None:
        candidate_ids = np.arange(int(pts.shape[0]), dtype=int)
    else:
        candidate_ids = np.flatnonzero(np.asarray(allowed_mask, dtype=bool))
    if candidate_ids.size == 0:
        return -1
    d2 = np.sum((pts[candidate_ids] - target[None, :]) ** 2, axis=1)
    return int(candidate_ids[int(np.argmin(d2))])


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

    for start in range(0, int(pts.shape[0]), max(1, int(chunk_size))):
        stop = min(int(pts.shape[0]), start + max(1, int(chunk_size)))
        p = pts[start:stop]
        w = p[:, None, :] - p0[None, :, :]
        t = np.sum(w * seg_v[None, :, :], axis=2) / seg_vv[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = p0[None, :, :] + t[:, :, None] * seg_v[None, :, :]
        d2 = np.sum((p[:, None, :] - proj) ** 2, axis=2)
        out[start:stop] = np.min(d2, axis=1)
    return out


def connected_component_from_seed(seed_cell: int, allowed_mask: np.ndarray, adjacency: List[List[int]]) -> List[int]:
    n_cells = int(len(adjacency))
    if n_cells == 0:
        return []
    seed = int(max(0, min(int(seed_cell), n_cells - 1)))
    if not bool(allowed_mask[seed]):
        return [seed]
    visited = {seed}
    stack = [seed]
    comp: List[int] = []
    while stack:
        cur = int(stack.pop())
        comp.append(cur)
        for nbr in adjacency[cur]:
            nbr = int(nbr)
            if nbr in visited or not bool(allowed_mask[nbr]):
                continue
            visited.add(nbr)
            stack.append(nbr)
    return comp


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
                nbr = int(nbr)
                if nbr in remaining and int(labels[nbr]) == int(label_id):
                    remaining.remove(nbr)
                    stack.append(nbr)
        components.append(comp)
    return components


def choose_majority_neighbor_label(labels: np.ndarray, adjacency: List[List[int]], cell_id: int) -> int:
    counts: Dict[int, int] = {}
    for nbr in adjacency[int(cell_id)]:
        lid = int(labels[int(nbr)])
        counts[lid] = int(counts.get(lid, 0) + 1)
    if not counts:
        return int(labels[int(cell_id)])
    best = min(counts.keys(), key=lambda lid: (-counts[lid], LABEL_PRIORITY.get(int(lid), 99), int(lid)))
    return int(best)


def smooth_surface_labels(labels: np.ndarray, adjacency: List[List[int]], immutable_cells: set[int], passes: int = 2) -> np.ndarray:
    out = np.asarray(labels, dtype=int).copy()
    for _ in range(max(0, int(passes))):
        prev = out.copy()
        changed = False
        for ci in range(int(prev.shape[0])):
            if int(ci) in immutable_cells:
                continue
            nbrs = adjacency[int(ci)]
            if len(nbrs) < 2:
                continue
            counts: Dict[int, int] = {}
            for nbr in nbrs:
                lid = int(prev[int(nbr)])
                counts[lid] = int(counts.get(lid, 0) + 1)
            best = choose_majority_neighbor_label(prev, adjacency, int(ci))
            best_count = int(counts.get(int(best), 0))
            cur_label = int(prev[int(ci)])
            if int(best) == cur_label:
                continue
            if best_count >= max(2, int(math.ceil(0.60 * len(nbrs)))):
                out[int(ci)] = int(best)
                changed = True
        if not changed:
            break
    return out.astype(int)


def derive_surface_point_labels(pd: vtkPolyData, cell_labels: np.ndarray) -> np.ndarray:
    point_to_cells = build_surface_point_to_cells(pd)
    point_labels = np.full((int(pd.GetNumberOfPoints()),), LABEL_OTHER, dtype=int)
    for pid, cells in enumerate(point_to_cells):
        if not cells:
            continue
        counts: Dict[int, int] = {}
        for ci in cells:
            lid = int(cell_labels[int(ci)])
            counts[lid] = int(counts.get(lid, 0) + 1)
        best = min(counts.keys(), key=lambda lid: (-counts[lid], LABEL_PRIORITY.get(int(lid), 99), int(lid)))
        point_labels[int(pid)] = int(best)
    return point_labels.astype(int)


def compute_surface_label_contact_map(labels: np.ndarray, adjacency: List[List[int]]) -> Dict[int, Dict[int, int]]:
    labels = np.asarray(labels, dtype=int)
    contact: Dict[int, Dict[int, int]] = {}
    for i, nbrs in enumerate(adjacency):
        li = int(labels[int(i)])
        for j in nbrs:
            if int(j) <= int(i):
                continue
            lj = int(labels[int(j)])
            if li == lj:
                continue
            contact.setdefault(li, {})
            contact.setdefault(lj, {})
            contact[li][lj] = int(contact[li].get(lj, 0) + 1)
            contact[lj][li] = int(contact[lj].get(li, 0) + 1)
    return contact


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
    cell_counts = {LABEL_ID_TO_NAME[int(lid)]: int(np.count_nonzero(cell_labels_arr == int(lid))) for lid in LABEL_ID_TO_NAME}
    point_counts = {LABEL_ID_TO_NAME[int(lid)]: int(np.count_nonzero(point_labels_arr == int(lid))) for lid in LABEL_ID_TO_NAME}
    cell_areas_by_label = {
        LABEL_ID_TO_NAME[int(lid)]: float(np.sum(np.asarray(cell_areas, dtype=float)[cell_labels_arr == int(lid)]))
        for lid in LABEL_ID_TO_NAME
    }
    components_by_label = {
        LABEL_ID_TO_NAME[int(lid)]: [list(map(int, comp)) for comp in label_connected_components(cell_labels_arr, adjacency, int(lid))]
        for lid in LABEL_ID_TO_NAME
    }
    component_counts = {name: int(len(comps)) for name, comps in components_by_label.items()}
    contact_map = compute_surface_label_contact_map(cell_labels_arr, adjacency)
    validation = {
        "abdominal_aorta_trunk_exists": bool(cell_counts[LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]] > 0),
        "right_main_iliac_exists": bool(cell_counts[LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]] > 0),
        "left_main_iliac_exists": bool(cell_counts[LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]] > 0),
        "right_renal_exists": bool(cell_counts[LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL]] > 0),
        "left_renal_exists": bool(cell_counts[LABEL_ID_TO_NAME[LABEL_LEFT_RENAL]] > 0),
        "abdominal_aorta_trunk_component_count": int(component_counts[LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]]),
        "right_main_iliac_component_count": int(component_counts[LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]]),
        "left_main_iliac_component_count": int(component_counts[LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]]),
        "right_renal_component_count": int(component_counts[LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL]]),
        "left_renal_component_count": int(component_counts[LABEL_ID_TO_NAME[LABEL_LEFT_RENAL]]),
        "abdominal_aorta_trunk_single_component": bool(component_counts[LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]] <= 1),
        "right_main_iliac_single_component": bool(component_counts[LABEL_ID_TO_NAME[LABEL_RIGHT_ILIAC]] <= 1),
        "left_main_iliac_single_component": bool(component_counts[LABEL_ID_TO_NAME[LABEL_LEFT_ILIAC]] <= 1),
        "right_renal_single_component": bool(component_counts[LABEL_ID_TO_NAME[LABEL_RIGHT_RENAL]] <= 1),
        "left_renal_single_component": bool(component_counts[LABEL_ID_TO_NAME[LABEL_LEFT_RENAL]] <= 1),
        "right_renal_touches_trunk": bool(contact_map.get(LABEL_RIGHT_RENAL, {}).get(LABEL_AORTA_TRUNK, 0) > 0),
        "left_renal_touches_trunk": bool(contact_map.get(LABEL_LEFT_RENAL, {}).get(LABEL_AORTA_TRUNK, 0) > 0),
        "right_main_iliac_touches_trunk": bool(contact_map.get(LABEL_RIGHT_ILIAC, {}).get(LABEL_AORTA_TRUNK, 0) > 0),
        "left_main_iliac_touches_trunk": bool(contact_map.get(LABEL_LEFT_ILIAC, {}).get(LABEL_AORTA_TRUNK, 0) > 0),
    }
    return {
        "cell_counts": dict(cell_counts),
        "point_counts": dict(point_counts),
        "cell_areas": dict(cell_areas_by_label),
        "component_counts": dict(component_counts),
        "contact_map": {
            LABEL_ID_TO_NAME[int(k)]: {LABEL_ID_TO_NAME[int(kk)]: int(vv) for kk, vv in vv_map.items()}
            for k, vv_map in contact_map.items()
        },
        "seed_cells": {LABEL_ID_TO_NAME[int(k)]: int(v) for k, v in seed_cells.items()},
        "validation": dict(validation),
        "settings": dict(settings),
    }


def build_surface_label_transfer(
    surface_pd: vtkPolyData,
    branch_geoms: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    work_warnings = warnings if warnings is not None else []
    n_cells = int(surface_pd.GetNumberOfCells())
    n_points = int(surface_pd.GetNumberOfPoints())
    if n_cells == 0 or n_points == 0:
        cell_labels = np.zeros((n_cells,), dtype=int)
        point_labels = np.zeros((n_points,), dtype=int)
        summary = summarize_surface_label_transfer(cell_labels, point_labels, [[] for _ in range(n_cells)], np.zeros((n_cells,), dtype=float), {}, {})
        return cell_labels, point_labels, summary

    cell_centers = compute_polydata_cell_centers_numpy(surface_pd)
    cell_areas = compute_polydata_cell_areas(surface_pd)
    adjacency = build_surface_cell_adjacency(surface_pd)
    label_bank = build_surface_label_segment_bank(branch_geoms)

    if not label_bank:
        work_warnings.append("W_SURFACE_LABEL_BANK_EMPTY: no centerline branches were available for surface transfer.")
        cell_labels = np.zeros((n_cells,), dtype=int)
        point_labels = derive_surface_point_labels(surface_pd, cell_labels)
        summary = summarize_surface_label_transfer(cell_labels, point_labels, adjacency, cell_areas, {}, {})
        return cell_labels, point_labels, summary

    distance_sq: Dict[int, np.ndarray] = {}
    for lid in [LABEL_AORTA_TRUNK, LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC, LABEL_RIGHT_RENAL, LABEL_LEFT_RENAL]:
        entry = label_bank.get(int(lid))
        if entry is None:
            continue
        distance_sq[int(lid)] = min_distance_sq_points_to_segments(cell_centers, entry["segment_p0"], entry["segment_p1"])

    trunk_bank = label_bank.get(LABEL_AORTA_TRUNK)
    right_iliac_bank = label_bank.get(LABEL_RIGHT_ILIAC)
    left_iliac_bank = label_bank.get(LABEL_LEFT_ILIAC)
    right_renal_bank = label_bank.get(LABEL_RIGHT_RENAL)
    left_renal_bank = label_bank.get(LABEL_LEFT_RENAL)

    trunk_inlet = find_surface_bank_landmark(trunk_bank, "Inlet", tangent_mode="forward")
    trunk_bif = find_surface_bank_landmark(trunk_bank, "Bifurcation", tangent_mode="backward")
    if trunk_bank and trunk_inlet is None:
        longest = get_longest_surface_bank_branch(trunk_bank)
        if longest is not None:
            pts = np.asarray(longest["points"], dtype=float)
            trunk_inlet = {
                "point": pts[0].astype(float),
                "tangent": unit(pts[min(1, int(pts.shape[0]) - 1)] - pts[0]) if pts.shape[0] > 1 else np.zeros((3,), dtype=float),
            }
            work_warnings.append("W_SURFACE_INLET_FALLBACK: used proximal trunk endpoint for inlet seeding.")
    if trunk_bank and trunk_bif is None:
        longest = get_longest_surface_bank_branch(trunk_bank)
        if longest is not None:
            pts = np.asarray(longest["points"], dtype=float)
            trunk_bif = {
                "point": pts[-1].astype(float),
                "tangent": unit(pts[-1] - pts[max(0, int(pts.shape[0]) - 2)]) if pts.shape[0] > 1 else np.zeros((3,), dtype=float),
            }
            work_warnings.append("W_SURFACE_BIF_FALLBACK: used distal trunk endpoint for bifurcation clipping.")

    distal_to_bif_mask = np.zeros((n_cells,), dtype=bool)
    bif_pt = None
    if trunk_bif is not None and float(np.linalg.norm(np.asarray(trunk_bif.get("tangent", np.zeros((3,), dtype=float)), dtype=float))) > EPS:
        bif_pt = np.asarray(trunk_bif["point"], dtype=float).reshape(3)
        bif_tangent = unit(np.asarray(trunk_bif["tangent"], dtype=float).reshape(3))
        distal_to_bif_mask = np.asarray(((cell_centers - bif_pt[None, :]) @ bif_tangent) > 0.0, dtype=bool)
    else:
        work_warnings.append("W_SURFACE_BIF_CLIP_UNAVAILABLE: bifurcation clipping plane could not be constructed.")

    x_extent = float(np.ptp(cell_centers[:, 0])) if cell_centers.shape[0] else 0.0
    side_tol = 0.02 * x_extent
    right_side_mask = np.asarray(cell_centers[:, 0] >= -side_tol, dtype=bool)
    left_side_mask = np.asarray(cell_centers[:, 0] <= side_tol, dtype=bool)

    cell_labels = np.full((n_cells,), LABEL_OTHER, dtype=int)
    seed_cells: Dict[int, int] = {}
    immutable_cells: set[int] = set()

    trunk_dist_sq = distance_sq.get(LABEL_AORTA_TRUNK, np.full((n_cells,), float("inf"), dtype=float))
    right_iliac_dist_sq = distance_sq.get(LABEL_RIGHT_ILIAC, np.full((n_cells,), float("inf"), dtype=float))
    left_iliac_dist_sq = distance_sq.get(LABEL_LEFT_ILIAC, np.full((n_cells,), float("inf"), dtype=float))
    right_renal_dist_sq = distance_sq.get(LABEL_RIGHT_RENAL, np.full((n_cells,), float("inf"), dtype=float))
    left_renal_dist_sq = distance_sq.get(LABEL_LEFT_RENAL, np.full((n_cells,), float("inf"), dtype=float))

    renal_specs = [
        (LABEL_RIGHT_RENAL, right_renal_bank, "RightRenalOrigin", "right_renal"),
        (LABEL_LEFT_RENAL, left_renal_bank, "LeftRenalOrigin", "left_renal"),
    ]
    renal_masks: Dict[int, np.ndarray] = {
        LABEL_RIGHT_RENAL: np.zeros((n_cells,), dtype=bool),
        LABEL_LEFT_RENAL: np.zeros((n_cells,), dtype=bool),
    }
    renal_settings: Dict[str, Any] = {}

    for lid, bank, landmark_key, summary_key in renal_specs:
        if bank is None or lid not in distance_sq:
            renal_settings[summary_key] = {"assigned_cells": 0, "reason": "missing_centerline_branch"}
            continue
        branch = get_longest_surface_bank_branch(bank)
        if branch is None:
            renal_settings[summary_key] = {"assigned_cells": 0, "reason": "missing_bank_branch"}
            continue
        branch_points = np.asarray(branch["points"], dtype=float)
        branch_abscissa = np.asarray(branch.get("abscissa", compute_abscissa(branch_points)), dtype=float)
        branch_len = float(branch.get("length", branch_abscissa[-1] if branch_abscissa.size else 0.0))
        lm = find_surface_bank_landmark(bank, landmark_key, tangent_mode="forward")
        if lm is None:
            lm = find_surface_bank_landmark(trunk_bank, landmark_key, tangent_mode="forward")
        if lm is None:
            lm = {"point": branch_points[0].astype(float), "tangent": unit(branch_points[min(1, branch_points.shape[0] - 1)] - branch_points[0]) if branch_points.shape[0] > 1 else np.zeros((3,), dtype=float)}
            work_warnings.append(f"W_SURFACE_{summary_key.upper()}_LANDMARK_FALLBACK: used proximal renal endpoint as takeoff.")
        takeoff_point = np.asarray(lm["point"], dtype=float).reshape(3)
        takeoff_tangent = unit(np.asarray(lm.get("tangent", np.zeros((3,), dtype=float)), dtype=float))
        if float(np.linalg.norm(takeoff_tangent)) < EPS and branch_points.shape[0] > 1:
            takeoff_tangent = unit(branch_points[1] - branch_points[0])

        local_step = float(np.linalg.norm(branch_points[1] - branch_points[0])) if branch_points.shape[0] > 1 else 0.0
        seed_offset = min(max(1.5 * local_step, 0.08 * branch_len), 0.35 * max(0.0, branch_len)) if branch_len > EPS else 0.0
        renal_seed_point = polyline_point_at_abscissa(branch_points, seed_offset)

        axial_proj = (
            np.sum((cell_centers - takeoff_point[None, :]) * takeoff_tangent[None, :], axis=1)
            if float(np.linalg.norm(takeoff_tangent)) > EPS
            else np.zeros((n_cells,), dtype=float)
        )
        forward_mask = np.asarray(axial_proj >= -0.25 * max(local_step, EPS), dtype=bool) if float(np.linalg.norm(takeoff_tangent)) > EPS else np.ones((n_cells,), dtype=bool)
        if bif_pt is not None:
            forward_mask &= np.asarray(~distal_to_bif_mask, dtype=bool)
        seed_cell = nearest_cell_to_point(cell_centers, renal_seed_point, allowed_mask=forward_mask)
        if seed_cell < 0:
            seed_cell = nearest_cell_to_point(cell_centers, renal_seed_point, allowed_mask=None)
        if seed_cell < 0:
            renal_settings[summary_key] = {"assigned_cells": 0, "reason": "missing_surface_seed"}
            continue

        seed_cells[int(lid)] = int(seed_cell)
        immutable_cells.add(int(seed_cell))

        dist_sq = distance_sq[int(lid)]
        seed_radius = math.sqrt(max(0.0, float(dist_sq[int(seed_cell)])))
        distance_limit = max(2.40 * max(seed_radius, EPS), 1.25 * max(local_step, EPS))
        ostium_radius = max(1.75 * max(seed_radius, EPS), float(np.linalg.norm(renal_seed_point - takeoff_point)) + 0.75 * max(seed_radius, EPS))
        takeoff_back_margin = max(0.35 * max(seed_radius, EPS), 0.25 * max(local_step, EPS))
        branch_facing_mask = np.asarray(axial_proj >= -takeoff_back_margin, dtype=bool) if float(np.linalg.norm(takeoff_tangent)) > EPS else np.ones((n_cells,), dtype=bool)
        ostium_bridge_mask = (
            np.asarray(np.sum((cell_centers - takeoff_point[None, :]) ** 2, axis=1) <= (ostium_radius * ostium_radius), dtype=bool) & branch_facing_mask
        )
        trunk_guard_mask = np.asarray(dist_sq <= (1.10 ** 2) * trunk_dist_sq, dtype=bool)
        candidate_mask = np.asarray(dist_sq <= (distance_limit * distance_limit), dtype=bool)
        candidate_mask &= branch_facing_mask
        candidate_mask &= np.asarray(trunk_guard_mask | ostium_bridge_mask, dtype=bool)
        if bif_pt is not None:
            candidate_mask &= np.asarray(~distal_to_bif_mask, dtype=bool)
        candidate_mask[int(seed_cell)] = True

        comp = connected_component_from_seed(int(seed_cell), candidate_mask, adjacency)
        renal_mask = np.zeros((n_cells,), dtype=bool)
        if comp:
            renal_mask[np.asarray(comp, dtype=int)] = True
        renal_masks[int(lid)] = renal_mask
        renal_settings[summary_key] = {
            "assigned_cells": int(np.count_nonzero(renal_mask)),
            "seed_cell": int(seed_cell),
            "seed_radius": float(seed_radius),
            "distance_limit": float(distance_limit),
        }

    overlap = np.asarray(renal_masks[LABEL_RIGHT_RENAL] & renal_masks[LABEL_LEFT_RENAL], dtype=bool)
    if np.any(overlap):
        overlap_ids = np.flatnonzero(overlap)
        for ci in overlap_ids.tolist():
            if float(right_renal_dist_sq[int(ci)]) <= float(left_renal_dist_sq[int(ci)]):
                renal_masks[LABEL_LEFT_RENAL][int(ci)] = False
            else:
                renal_masks[LABEL_RIGHT_RENAL][int(ci)] = False
        work_warnings.append(f"W_SURFACE_RENAL_OVERLAP_RESOLVED: resolved {int(np.count_nonzero(overlap))} overlapping renal cells.")

    cell_labels[renal_masks[LABEL_RIGHT_RENAL]] = LABEL_RIGHT_RENAL
    cell_labels[renal_masks[LABEL_LEFT_RENAL]] = LABEL_LEFT_RENAL

    iliac_specs = [
        (LABEL_RIGHT_ILIAC, right_iliac_bank, right_side_mask, left_iliac_dist_sq, "right_main_iliac"),
        (LABEL_LEFT_ILIAC, left_iliac_bank, left_side_mask, right_iliac_dist_sq, "left_main_iliac"),
    ]
    iliac_masks: Dict[int, np.ndarray] = {
        LABEL_RIGHT_ILIAC: np.zeros((n_cells,), dtype=bool),
        LABEL_LEFT_ILIAC: np.zeros((n_cells,), dtype=bool),
    }
    iliac_settings: Dict[str, Any] = {}
    for lid, bank, side_mask, other_iliac_dist_sq, summary_key in iliac_specs:
        if bank is None or lid not in distance_sq:
            iliac_settings[summary_key] = {"assigned_cells": 0, "reason": "missing_centerline_branch"}
            continue
        branch = get_longest_surface_bank_branch(bank)
        if branch is None:
            iliac_settings[summary_key] = {"assigned_cells": 0, "reason": "missing_bank_branch"}
            continue
        branch_points = np.asarray(branch["points"], dtype=float)
        if branch_points.shape[0] == 0:
            iliac_settings[summary_key] = {"assigned_cells": 0, "reason": "empty_branch"}
            continue
        distal_point = branch_points[-1].astype(float)
        own_dist_sq = distance_sq[int(lid)]
        seed_mask = np.asarray(distal_to_bif_mask, dtype=bool) & np.asarray(side_mask, dtype=bool)
        seed_cell = nearest_cell_to_point(cell_centers, distal_point, allowed_mask=seed_mask)
        if seed_cell < 0:
            seed_cell = nearest_cell_to_point(cell_centers, distal_point, allowed_mask=None)
        if seed_cell < 0:
            iliac_settings[summary_key] = {"assigned_cells": 0, "reason": "missing_surface_seed"}
            continue
        seed_cells[int(lid)] = int(seed_cell)
        immutable_cells.add(int(seed_cell))

        seed_radius = math.sqrt(max(0.0, float(own_dist_sq[int(seed_cell)])))
        branch_len = float(branch.get("length", polyline_length(branch_points)))
        distance_limit = max(2.50 * max(seed_radius, EPS), 0.10 * max(branch_len, 1.0))
        candidate_mask = np.asarray(distal_to_bif_mask, dtype=bool) & np.asarray(side_mask, dtype=bool)
        candidate_mask &= np.asarray((own_dist_sq <= (distance_limit * distance_limit)) | (own_dist_sq <= 1.20 * other_iliac_dist_sq), dtype=bool)
        candidate_mask[int(seed_cell)] = True

        comp = connected_component_from_seed(int(seed_cell), candidate_mask, adjacency)
        mask = np.zeros((n_cells,), dtype=bool)
        if comp:
            mask[np.asarray(comp, dtype=int)] = True
        iliac_masks[int(lid)] = mask
        iliac_settings[summary_key] = {
            "assigned_cells": int(np.count_nonzero(mask)),
            "seed_cell": int(seed_cell),
            "seed_radius": float(seed_radius),
            "distance_limit": float(distance_limit),
        }

    cell_labels[iii := iliac_masks[LABEL_RIGHT_ILIAC]] = LABEL_RIGHT_ILIAC
    cell_labels[iii2 := iliac_masks[LABEL_LEFT_ILIAC]] = np.where(iii2 & (cell_labels == LABEL_OTHER), LABEL_LEFT_ILIAC, cell_labels[iii2])

    overlap_iliac = np.asarray((cell_labels == LABEL_RIGHT_ILIAC) & iliac_masks[LABEL_LEFT_ILIAC], dtype=bool)
    if np.any(overlap_iliac):
        for ci in np.flatnonzero(overlap_iliac).tolist():
            if float(right_iliac_dist_sq[int(ci)]) <= float(left_iliac_dist_sq[int(ci)]):
                cell_labels[int(ci)] = LABEL_RIGHT_ILIAC
            else:
                cell_labels[int(ci)] = LABEL_LEFT_ILIAC

    remaining_distal = np.asarray((cell_labels == LABEL_OTHER) & distal_to_bif_mask, dtype=bool)
    if np.any(remaining_distal):
        right_score = right_iliac_dist_sq + np.where(right_side_mask, 0.0, np.percentile(right_iliac_dist_sq[np.isfinite(right_iliac_dist_sq)], 75) if np.any(np.isfinite(right_iliac_dist_sq)) else 0.0)
        left_score = left_iliac_dist_sq + np.where(left_side_mask, 0.0, np.percentile(left_iliac_dist_sq[np.isfinite(left_iliac_dist_sq)], 75) if np.any(np.isfinite(left_iliac_dist_sq)) else 0.0)
        choose_right = right_score <= left_score
        cell_labels[np.asarray(remaining_distal & choose_right, dtype=bool)] = LABEL_RIGHT_ILIAC
        cell_labels[np.asarray(remaining_distal & ~choose_right, dtype=bool)] = LABEL_LEFT_ILIAC

    trunk_seed = -1
    if trunk_inlet is not None:
        trunk_seed_mask = np.asarray(~distal_to_bif_mask, dtype=bool) if bif_pt is not None else np.ones((n_cells,), dtype=bool)
        trunk_seed = nearest_cell_to_point(cell_centers, np.asarray(trunk_inlet["point"], dtype=float), allowed_mask=trunk_seed_mask)
        if trunk_seed >= 0:
            seed_cells[LABEL_AORTA_TRUNK] = int(trunk_seed)
            immutable_cells.add(int(trunk_seed))
    if trunk_seed < 0:
        trunk_seed = nearest_cell_to_point(cell_centers, np.mean(cell_centers, axis=0), allowed_mask=np.asarray(~distal_to_bif_mask, dtype=bool) if bif_pt is not None else None)
        if trunk_seed >= 0:
            seed_cells[LABEL_AORTA_TRUNK] = int(trunk_seed)
            immutable_cells.add(int(trunk_seed))
            work_warnings.append("W_SURFACE_TRUNK_SEED_FALLBACK: used generic proximal surface seed for trunk.")

    remaining_proximal = np.asarray((cell_labels == LABEL_OTHER) & (~distal_to_bif_mask if bif_pt is not None else np.ones((n_cells,), dtype=bool)), dtype=bool)
    if trunk_seed >= 0:
        trunk_candidate = np.asarray(remaining_proximal, dtype=bool)
        trunk_candidate |= np.asarray(trunk_dist_sq <= np.minimum.reduce([
            right_renal_dist_sq,
            left_renal_dist_sq,
            right_iliac_dist_sq,
            left_iliac_dist_sq,
            trunk_dist_sq * 1.0,
        ]), dtype=bool)
        if bif_pt is not None:
            trunk_candidate &= np.asarray(~distal_to_bif_mask, dtype=bool)
        trunk_candidate[int(trunk_seed)] = True
        trunk_comp = connected_component_from_seed(int(trunk_seed), trunk_candidate, adjacency)
        if trunk_comp:
            trunk_mask = np.zeros((n_cells,), dtype=bool)
            trunk_mask[np.asarray(trunk_comp, dtype=int)] = True
            trunk_mask &= np.asarray(cell_labels == LABEL_OTHER, dtype=bool)
            cell_labels[trunk_mask] = LABEL_AORTA_TRUNK
    else:
        work_warnings.append("W_SURFACE_TRUNK_SEED_FAILED: unable to place trunk surface seed.")

    cell_labels[np.asarray((cell_labels == LABEL_OTHER) & (~distal_to_bif_mask if bif_pt is not None else np.ones((n_cells,), dtype=bool)), dtype=bool)] = LABEL_AORTA_TRUNK

    for lid in [LABEL_AORTA_TRUNK, LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC, LABEL_RIGHT_RENAL, LABEL_LEFT_RENAL]:
        comps = label_connected_components(cell_labels, adjacency, int(lid))
        if len(comps) <= 1:
            continue
        keep_comp = None
        if int(lid) in seed_cells:
            seed = int(seed_cells[int(lid)])
            for comp in comps:
                if seed in comp:
                    keep_comp = set(int(v) for v in comp)
                    break
        if keep_comp is None:
            keep_comp = set(max(comps, key=lambda c: len(c)))
        for comp in comps:
            comp_set = set(int(v) for v in comp)
            if comp_set == keep_comp:
                continue
            for ci in comp:
                cell_labels[int(ci)] = choose_majority_neighbor_label(cell_labels, adjacency, int(ci))

    cell_labels = smooth_surface_labels(cell_labels, adjacency, immutable_cells, passes=2)
    point_labels = derive_surface_point_labels(surface_pd, cell_labels)

    settings = {
        "surface_cell_count": int(n_cells),
        "surface_point_count": int(n_points),
        "distal_to_bifurcation_cell_count": int(np.count_nonzero(distal_to_bif_mask)),
        "renal_regions": dict(renal_settings),
        "iliac_regions": dict(iliac_settings),
    }
    summary = summarize_surface_label_transfer(cell_labels, point_labels, adjacency, cell_areas, seed_cells, settings)

    validation = dict(summary.get("validation", {}))
    if not bool(validation.get("abdominal_aorta_trunk_exists", False)):
        work_warnings.append("W_SURFACE_ABDOMINAL_AORTA_TRUNK_EMPTY: no surface cells assigned to trunk.")
    if not bool(validation.get("right_main_iliac_exists", False)):
        work_warnings.append("W_SURFACE_RIGHT_MAIN_ILIAC_EMPTY: no surface cells assigned to right iliac.")
    if not bool(validation.get("left_main_iliac_exists", False)):
        work_warnings.append("W_SURFACE_LEFT_MAIN_ILIAC_EMPTY: no surface cells assigned to left iliac.")
    if not bool(validation.get("right_renal_exists", False)):
        work_warnings.append("W_SURFACE_RIGHT_RENAL_EMPTY: no surface cells assigned to right renal.")
    if not bool(validation.get("left_renal_exists", False)):
        work_warnings.append("W_SURFACE_LEFT_RENAL_EMPTY: no surface cells assigned to left renal.")
    if not bool(validation.get("right_renal_touches_trunk", False)):
        work_warnings.append("W_SURFACE_RIGHT_RENAL_DETACHED: right renal surface does not touch trunk.")
    if not bool(validation.get("left_renal_touches_trunk", False)):
        work_warnings.append("W_SURFACE_LEFT_RENAL_DETACHED: left renal surface does not touch trunk.")
    if not bool(validation.get("right_main_iliac_touches_trunk", False)):
        work_warnings.append("W_SURFACE_RIGHT_MAIN_ILIAC_DETACHED: right iliac surface does not touch trunk.")
    if not bool(validation.get("left_main_iliac_touches_trunk", False)):
        work_warnings.append("W_SURFACE_LEFT_MAIN_ILIAC_DETACHED: left iliac surface does not touch trunk.")

    return cell_labels.astype(int), point_labels.astype(int), summary


def annotate_surface_polydata_for_combined_output(
    surface_pd: vtkPolyData,
    branch_geoms: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> Tuple[vtkPolyData, Dict[str, Any]]:
    out = clone_polydata(surface_pd)
    cell_labels, _, summary = build_surface_label_transfer(out, branch_geoms, warnings=warnings)
    n_cells = int(out.GetNumberOfCells())
    cd = out.GetCellData()
    add_scalar_array_to_cell_data(cd, "BranchId", [int(v) for v in cell_labels.tolist()], vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchName", [LABEL_ID_TO_NAME.get(int(v), "other") for v in cell_labels.tolist()])
    add_scalar_array_to_cell_data(cd, "BranchLength", [0.0] * n_cells, vtk_type=vtk.VTK_DOUBLE)
    add_string_array_to_cell_data(cd, "GeometryType", ["surface"] * n_cells)
    add_scalar_array_to_cell_data(cd, "ParentBranchId", [int(v) for v in cell_labels.tolist()], vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "ParentBranchName", [LABEL_ID_TO_NAME.get(int(v), "other") for v in cell_labels.tolist()])
    add_string_array_to_cell_data(cd, "TopologyRole", ["surface"] * n_cells)
    return out, summary


def build_centerline_branch_geometries(
    adjacency_full: Dict[int, Dict[int, float]],
    pts_canonical: np.ndarray,
    dist_from_inlet: Dict[int, float],
    trunk_path: List[int],
    right_iliac_system: Optional[Dict[str, Any]],
    left_iliac_system: Optional[Dict[str, Any]],
    right_renal_system: Optional[Dict[str, Any]],
    left_renal_system: Optional[Dict[str, Any]],
    inlet_node: int,
    bif_node: int,
    right_renal_origin: Optional[int],
    left_renal_origin: Optional[int],
    trunk_child_systems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    trunk_path = [int(v) for v in trunk_path]
    trunk_edges = path_edge_keys(trunk_path)
    trunk_set = set(int(v) for v in trunk_path)

    named_systems = {
        LABEL_RIGHT_ILIAC: right_iliac_system,
        LABEL_LEFT_ILIAC: left_iliac_system,
        LABEL_RIGHT_RENAL: right_renal_system,
        LABEL_LEFT_RENAL: left_renal_system,
    }
    system_node_sets = {int(lid): rooted_child_system_node_set(sys, include_takeoff=True) for lid, sys in named_systems.items()}
    system_edge_sets = {int(lid): graph_edges_for_node_set(adjacency_full, nodes) for lid, nodes in system_node_sets.items()}
    system_stem_edges = {
        int(lid): path_edge_keys([int(v) for v in (sys.get("named_stem_path", sys.get("stem_path", [])) if sys is not None else [])])
        for lid, sys in named_systems.items()
    }

    named_system_keys = {rooted_child_system_key(sys) for sys in named_systems.values() if sys is not None}
    trunk_descendant_entries = [
        sys for sys in trunk_child_systems if rooted_child_system_key(sys) not in named_system_keys
    ]

    branch_geoms: List[Dict[str, Any]] = []
    chains = build_branch_chains_from_graph(adjacency_full)
    classified: List[Tuple[int, float, List[int], Dict[str, Any]]] = []

    for chain in chains:
        nodes = [int(v) for v in chain]
        if len(nodes) < 2:
            continue
        d0 = dist_from_inlet.get(int(nodes[0]), float("nan"))
        d1 = dist_from_inlet.get(int(nodes[-1]), float("nan"))
        if math.isfinite(d0) and math.isfinite(d1):
            if d0 > d1:
                nodes = list(reversed(nodes))
        elif not math.isfinite(d0) and not math.isfinite(d1) and nodes[0] > nodes[-1]:
            nodes = list(reversed(nodes))

        chain_edges = path_edge_keys(nodes)
        node_set = set(int(v) for v in nodes)

        label_id = LABEL_OTHER
        parent_label_id = None
        role = "unassigned"
        parent_name = None

        if chain_edges and chain_edges.issubset(trunk_edges):
            label_id = LABEL_AORTA_TRUNK
            parent_label_id = LABEL_AORTA_TRUNK
            role = "trunk_path"
            parent_name = LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]
        else:
            matched_named = False
            for lid in [LABEL_RIGHT_ILIAC, LABEL_LEFT_ILIAC, LABEL_RIGHT_RENAL, LABEL_LEFT_RENAL]:
                nodes_l = system_node_sets.get(int(lid), set())
                edges_l = system_edge_sets.get(int(lid), set())
                stem_edges_l = system_stem_edges.get(int(lid), set())
                if chain_edges and chain_edges.issubset(edges_l) and node_set.issubset(nodes_l):
                    label_id = int(lid)
                    parent_label_id = int(lid)
                    role = "named_stem" if stem_edges_l and chain_edges.issubset(stem_edges_l) else "named_system_descendant"
                    parent_name = LABEL_ID_TO_NAME[int(lid)]
                    matched_named = True
                    break
            if not matched_named:
                attached = False
                for sys in trunk_descendant_entries:
                    sys_nodes = rooted_child_system_node_set(sys, include_takeoff=True)
                    if node_set.issubset(sys_nodes):
                        label_id = LABEL_OTHER
                        parent_label_id = LABEL_AORTA_TRUNK
                        role = "trunk_descendant"
                        parent_name = LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]
                        attached = True
                        break
                if not attached and node_set.intersection(trunk_set):
                    label_id = LABEL_OTHER
                    parent_label_id = LABEL_AORTA_TRUNK
                    role = "trunk_adjacent"
                    parent_name = LABEL_ID_TO_NAME[LABEL_AORTA_TRUNK]

        pts = pts_canonical[np.asarray(nodes, dtype=int)]
        landmark_point_ids: Dict[str, int] = {}
        node_to_index = {int(n): i for i, n in enumerate(nodes)}
        if int(inlet_node) in node_to_index:
            landmark_point_ids["Inlet"] = int(node_to_index[int(inlet_node)])
        if int(bif_node) in node_to_index:
            landmark_point_ids["Bifurcation"] = int(node_to_index[int(bif_node)])
        if right_renal_origin is not None and int(right_renal_origin) in node_to_index:
            landmark_point_ids["RightRenalOrigin"] = int(node_to_index[int(right_renal_origin)])
        if left_renal_origin is not None and int(left_renal_origin) in node_to_index:
            landmark_point_ids["LeftRenalOrigin"] = int(node_to_index[int(left_renal_origin)])

        branch = {
            "label_id": int(label_id),
            "name": LABEL_ID_TO_NAME[int(label_id)],
            "points": pts.astype(float),
            "node_ids": list(nodes),
            "landmark_point_ids": dict(landmark_point_ids),
            "topology_role": str(role),
            "topology_parent_label_id": int(parent_label_id) if parent_label_id is not None else None,
            "topology_parent_name": str(parent_name) if parent_name is not None else None,
        }
        min_dist = min(float(dist_from_inlet.get(int(n), float("inf"))) for n in nodes)
        classified.append((LABEL_PRIORITY.get(int(label_id), 99), float(min_dist), list(nodes), branch))

    classified.sort(key=lambda item: (int(item[0]), float(item[1]), tuple(int(v) for v in item[2])))
    for _, _, _, branch in classified:
        branch_geoms.append(branch)
    return branch_geoms


def build_output_centerlines_polydata(branch_geoms: List[Dict[str, Any]]) -> vtkPolyData:
    out = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    cell_label: List[int] = []
    cell_name: List[str] = []
    cell_length: List[float] = []
    parent_label: List[int] = []
    parent_name: List[str] = []
    topology_role: List[str] = []

    global_pid = 0
    for br in branch_geoms:
        pts = np.asarray(br["points"], dtype=float)
        if pts.shape[0] < 2:
            continue
        start_pid = global_pid
        for i in range(int(pts.shape[0])):
            points.InsertNextPoint(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
            global_pid += 1

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(int(pts.shape[0]))
        for i in range(int(pts.shape[0])):
            polyline.GetPointIds().SetId(i, int(start_pid + i))
        lines.InsertNextCell(polyline)

        s = compute_abscissa(pts)
        cell_label.append(int(br["label_id"]))
        cell_name.append(str(br["name"]))
        cell_length.append(float(s[-1]) if s.size else 0.0)
        parent_lid = br.get("topology_parent_label_id", None)
        parent_label.append(int(parent_lid) if parent_lid is not None else int(br["label_id"]))
        parent_name.append(str(br.get("topology_parent_name") or LABEL_ID_TO_NAME.get(int(parent_label[-1]), "other")))
        topology_role.append(str(br.get("topology_role", "unassigned")))

    out.SetPoints(points)
    out.SetLines(lines)
    cd = out.GetCellData()
    add_scalar_array_to_cell_data(cd, "BranchId", cell_label, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "BranchName", cell_name)
    add_scalar_array_to_cell_data(cd, "BranchLength", cell_length, vtk_type=vtk.VTK_DOUBLE)
    add_string_array_to_cell_data(cd, "GeometryType", ["centerline"] * len(cell_label))
    add_scalar_array_to_cell_data(cd, "ParentBranchId", parent_label, vtk_type=vtk.VTK_INT)
    add_string_array_to_cell_data(cd, "ParentBranchName", parent_name)
    add_string_array_to_cell_data(cd, "TopologyRole", topology_role)
    return out


def build_combined_surface_centerlines_polydata(surface_pd: vtkPolyData, centerlines_pd: vtkPolyData) -> vtkPolyData:
    app = vtk.vtkAppendPolyData()
    app.AddInputData(surface_pd)
    app.AddInputData(centerlines_pd)
    app.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(app.GetOutput())
    out.BuildLinks()

    n_surface = int(surface_pd.GetNumberOfCells())
    n_center = int(centerlines_pd.GetNumberOfCells())

    cd = out.GetCellData()
    cd.Initialize()
    add_scalar_array_to_cell_data(
        cd,
        "BranchId",
        [int(v) for v in get_cell_scalar_array_values(surface_pd, "BranchId", LABEL_OTHER)]
        + [int(v) for v in get_cell_scalar_array_values(centerlines_pd, "BranchId", LABEL_OTHER)],
        vtk_type=vtk.VTK_INT,
    )
    add_string_array_to_cell_data(
        cd,
        "BranchName",
        get_cell_string_array_values(surface_pd, "BranchName", "other")
        + get_cell_string_array_values(centerlines_pd, "BranchName", "other"),
    )
    add_scalar_array_to_cell_data(
        cd,
        "BranchLength",
        [0.0] * n_surface + get_cell_scalar_array_values(centerlines_pd, "BranchLength", 0.0),
        vtk_type=vtk.VTK_DOUBLE,
    )
    add_string_array_to_cell_data(
        cd,
        "GeometryType",
        ["surface"] * n_surface + ["centerline"] * n_center,
    )
    add_scalar_array_to_cell_data(
        cd,
        "ParentBranchId",
        [int(v) for v in get_cell_scalar_array_values(surface_pd, "ParentBranchId", LABEL_OTHER)]
        + [int(v) for v in get_cell_scalar_array_values(centerlines_pd, "ParentBranchId", LABEL_OTHER)],
        vtk_type=vtk.VTK_INT,
    )
    add_string_array_to_cell_data(
        cd,
        "ParentBranchName",
        get_cell_string_array_values(surface_pd, "ParentBranchName", "other")
        + get_cell_string_array_values(centerlines_pd, "ParentBranchName", "other"),
    )
    add_string_array_to_cell_data(
        cd,
        "TopologyRole",
        get_cell_string_array_values(surface_pd, "TopologyRole", "surface")
        + get_cell_string_array_values(centerlines_pd, "TopologyRole", "unassigned"),
    )
    return out


def summarize_centerline_branch_geometries(branch_geoms: List[Dict[str, Any]]) -> Dict[str, Any]:
    length_by_branch: Dict[str, float] = {}
    count_by_branch: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for idx, br in enumerate(branch_geoms):
        pts = np.asarray(br.get("points", np.zeros((0, 3), dtype=float)), dtype=float)
        length = float(polyline_length(pts))
        branch_name = str(br.get("name", LABEL_ID_TO_NAME[LABEL_OTHER]))
        length_by_branch[branch_name] = float(length_by_branch.get(branch_name, 0.0) + length)
        count_by_branch[branch_name] = int(count_by_branch.get(branch_name, 0) + 1)
        rows.append(
            {
                "index": int(idx),
                "branch_id": int(br.get("label_id", LABEL_OTHER)),
                "branch_name": branch_name,
                "topology_role": str(br.get("topology_role", "unassigned")),
                "topology_parent_label_id": br.get("topology_parent_label_id"),
                "topology_parent_name": br.get("topology_parent_name"),
                "length": float(length),
                "point_count": int(pts.shape[0]),
            }
        )
    return {
        "length_by_branch_label": {k: float(v) for k, v in length_by_branch.items()},
        "count_by_branch_label": {k: int(v) for k, v in count_by_branch.items()},
        "branches": list(rows),
    }


def evaluate_interpretation(
    cl_pts: np.ndarray,
    adjacency: Dict[int, Dict[int, float]],
    inlet_node: int,
    inlet_hypothesis: Dict[str, Any],
    bif_hypothesis: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    inlet_node = int(inlet_node)
    bif_node = int(bif_hypothesis["bif_node"])
    dist = bif_hypothesis["dist"]
    prev = bif_hypothesis["prev"]
    child_map = build_rooted_child_map(prev)
    trunk_path = path_to_root(prev, inlet_node, bif_node)
    if not trunk_path:
        raise RuntimeError("Failed to reconstruct trunk path.")
    trunk_set = set(int(v) for v in trunk_path)

    iliac_system_a = bif_hypothesis["system_a"]
    iliac_system_b = bif_hypothesis["system_b"]
    iliac_main_a = [int(v) for v in iliac_system_a.get("named_stem_path", iliac_system_a.get("stem_path", []))]
    iliac_main_b = [int(v) for v in iliac_system_b.get("named_stem_path", iliac_system_b.get("stem_path", []))]
    if not iliac_main_a:
        iliac_main_a = [int(v) for v in path_to_root(prev, inlet_node, int(iliac_system_a["representative_endpoint"])) if int(v) == bif_node or int(v) in rooted_child_system_node_set(iliac_system_a)]
    if not iliac_main_b:
        iliac_main_b = [int(v) for v in path_to_root(prev, inlet_node, int(iliac_system_b["representative_endpoint"])) if int(v) == bif_node or int(v) in rooted_child_system_node_set(iliac_system_b)]

    iliac_excluded_nodes = rooted_child_system_node_set(iliac_system_a, include_takeoff=False)
    iliac_excluded_nodes.update(rooted_child_system_node_set(iliac_system_b, include_takeoff=False))

    inlet_pt = np.asarray(cl_pts[int(inlet_node)], dtype=float)
    bif_pt = np.asarray(cl_pts[int(bif_node)], dtype=float)
    iliac_pt_a = np.asarray(cl_pts[int(iliac_main_a[-1] if iliac_main_a else iliac_system_a["representative_endpoint"])], dtype=float)
    iliac_pt_b = np.asarray(cl_pts[int(iliac_main_b[-1] if iliac_main_b else iliac_system_b["representative_endpoint"])], dtype=float)

    r_provisional, origin, frame_conf = build_canonical_transform(inlet_pt, bif_pt, iliac_pt_a, iliac_pt_b, cl_pts, warnings)
    cl_pts_c_provisional = apply_transform_points(cl_pts, r_provisional, origin)

    renal_scan = discover_renal_branch_candidates(
        adjacency,
        cl_pts_c_provisional,
        inlet_node,
        bif_node,
        trunk_set,
        dist,
        prev,
        int(iliac_system_a["representative_endpoint"]),
        int(iliac_system_b["representative_endpoint"]),
        warnings,
        excluded_system_nodes=iliac_excluded_nodes,
    )

    r_refined, frame_info = refine_horizontal_axes_using_branch_anatomy(
        r_provisional,
        cl_pts_c_provisional,
        iliac_main_a,
        iliac_main_b,
        renal_scan,
        inlet_node,
        bif_node,
        trunk_path,
        dist,
        prev,
        int(iliac_system_a["representative_endpoint"]),
        int(iliac_system_b["representative_endpoint"]),
        warnings,
        iliac_excluded_system_nodes=iliac_excluded_nodes,
    )

    cl_pts_c_pre_ap = apply_transform_points(cl_pts, r_refined, origin)
    best_pair = renal_scan.get("best_pair")
    renal_eps_for_ap = (
        int(best_pair["a"]["ep"]) if best_pair is not None else None,
        int(best_pair["b"]["ep"]) if best_pair is not None else None,
    )
    need_flip_xy, ap_conf, ap_warn = resolve_anterior_posterior_sign(
        cl_pts_c_pre_ap,
        inlet_node,
        bif_node,
        trunk_path,
        dist,
        prev,
        int(iliac_system_a["representative_endpoint"]),
        int(iliac_system_b["representative_endpoint"]),
        renal_eps_for_ap,
        warnings,
        excluded_system_nodes=iliac_excluded_nodes,
    )

    flipped_for_ap = False
    r = np.asarray(r_refined, dtype=float).copy()
    if need_flip_xy:
        flipped_for_ap = True
        r[0, :] *= -1.0
        r[1, :] *= -1.0
    cl_pts_c = apply_transform_points(cl_pts, r, origin)

    x_a = float(cl_pts_c[int(iliac_system_a["representative_endpoint"])][0])
    x_b = float(cl_pts_c[int(iliac_system_b["representative_endpoint"])][0])
    if x_a >= x_b:
        right_iliac_system = iliac_system_a
        left_iliac_system = iliac_system_b
        right_iliac_main = iliac_main_a
        left_iliac_main = iliac_main_b
    else:
        right_iliac_system = iliac_system_b
        left_iliac_system = iliac_system_a
        right_iliac_main = iliac_main_b
        left_iliac_main = iliac_main_a

    rr_ep, lr_ep, rr_take, lr_take, renal_conf, renal_diag = identify_renal_branches(
        adjacency,
        cl_pts_c,
        inlet_node,
        bif_node,
        trunk_set,
        dist,
        prev,
        int(right_iliac_system["representative_endpoint"]),
        int(left_iliac_system["representative_endpoint"]),
        warnings,
        renal_scan=renal_scan,
        excluded_system_nodes=iliac_excluded_nodes,
    )

    rr_origin_node = int(renal_diag["right_trunk_takeoff"]) if renal_diag.get("right_trunk_takeoff") is not None else rr_take
    lr_origin_node = int(renal_diag["left_trunk_takeoff"]) if renal_diag.get("left_trunk_takeoff") is not None else lr_take

    right_renal_system = None
    left_renal_system = None
    if rr_ep is not None and rr_origin_node is not None:
        right_renal_system = find_rooted_child_system_for_endpoint(child_map, cl_pts_c, dist, prev, int(rr_origin_node), int(rr_ep), inlet_node=inlet_node)
    if lr_ep is not None and lr_origin_node is not None:
        left_renal_system = find_rooted_child_system_for_endpoint(child_map, cl_pts_c, dist, prev, int(lr_origin_node), int(lr_ep), inlet_node=inlet_node)

    trunk_child_systems = build_direct_child_systems_for_parent_path(trunk_path, cl_pts_c, dist, prev)

    max_graph_depth = max(EPS, max(float(v) for v in dist.values()))
    trunk_depth_norm = clamp(float(dist.get(int(bif_node), 0.0)) / max_graph_depth, 0.0, 1.0)
    laterality_conf = clamp(
        0.35 * float(frame_info.get("iliac_axis_confidence", 0.0))
        + 0.25 * float(frame_info.get("renal_axis_alignment_with_iliacs", 0.0))
        + 0.20 * float(ap_conf)
        + 0.20 * float(renal_conf),
        0.0,
        1.0,
    )
    score = float(
        clamp(
            0.23 * float(inlet_hypothesis["confidence"])
            + 0.27 * float(bif_hypothesis["confidence"])
            + 0.14 * float(frame_info.get("confidence", 0.0))
            + 0.14 * float(laterality_conf)
            + 0.12 * float(renal_conf)
            + 0.10 * float(trunk_depth_norm),
            0.0,
            1.0,
        )
    )

    return {
        "score": float(score),
        "inlet_confidence": float(inlet_hypothesis["confidence"]),
        "bifurcation_confidence": float(bif_hypothesis["confidence"]),
        "renal_confidence": float(renal_conf),
        "laterality_confidence": float(laterality_conf),
        "ap_confidence": float(ap_conf),
        "frame_confidence": float(frame_info.get("confidence", 0.0)),
        "frame_info": dict(frame_info),
        "ap_warn": bool(ap_warn),
        "flipped_for_ap": bool(flipped_for_ap),
        "r": np.asarray(r, dtype=float),
        "origin": np.asarray(origin, dtype=float),
        "cl_pts_c": np.asarray(cl_pts_c, dtype=float),
        "dist": dict(dist),
        "prev": dict(prev),
        "child_map": dict(child_map),
        "trunk_path": list(trunk_path),
        "trunk_child_systems": list(trunk_child_systems),
        "right_iliac_system": right_iliac_system,
        "left_iliac_system": left_iliac_system,
        "right_iliac_main": list(right_iliac_main),
        "left_iliac_main": list(left_iliac_main),
        "right_renal_ep": rr_ep,
        "left_renal_ep": lr_ep,
        "right_renal_takeoff": rr_take,
        "left_renal_takeoff": lr_take,
        "right_renal_origin": rr_origin_node,
        "left_renal_origin": lr_origin_node,
        "right_renal_system": right_renal_system,
        "left_renal_system": left_renal_system,
        "renal_scan": dict(renal_scan),
        "renal_diag": dict(renal_diag),
        "inlet_node": int(inlet_node),
        "bif_node": int(bif_node),
        "inlet_term": inlet_hypothesis["termination"],
        "axis_si": np.asarray(inlet_hypothesis["axis_si"], dtype=float),
    }


def solve_best_interpretation(
    surface_tri: vtkPolyData,
    surface_pts: np.ndarray,
    terms: List[TerminationLoop],
    warnings: List[str],
    debug_raw_path: str = "",
) -> Dict[str, Any]:
    inlet_hypotheses = generate_inlet_hypotheses(terms, surface_pts, warnings, max_hypotheses=3)
    if not inlet_hypotheses:
        raise RuntimeError("No inlet hypotheses were generated.")

    all_interpretations: List[Dict[str, Any]] = []
    first_successful_raw_centerlines = None

    for inlet_rank, inlet_hyp in enumerate(inlet_hypotheses):
        local_warnings: List[str] = []
        try:
            inlet_term = inlet_hyp["termination"]
            term_centers = [np.asarray(t.center, dtype=float) for t in terms if t is not inlet_term]
            centerlines, cl_info = compute_centerlines_vmtk(surface_tri, np.asarray(inlet_term.center, dtype=float), term_centers, local_warnings)
            if first_successful_raw_centerlines is None:
                first_successful_raw_centerlines = centerlines

            adjacency_full, cl_pts, _ = build_graph_from_polyline_centerlines(centerlines)
            if not adjacency_full:
                raise RuntimeError("Centerline graph adjacency was empty.")
            analysis_seed_node, _ = nearest_node_to_point(sorted(int(v) for v in adjacency_full.keys()), cl_pts, np.asarray(inlet_term.center, dtype=float))
            if analysis_seed_node < 0:
                raise RuntimeError("Failed to locate inlet-near centerline node.")
            analysis_nodes = connected_component_nodes(adjacency_full, analysis_seed_node)
            adjacency = induced_subgraph(adjacency_full, analysis_nodes)
            if not adjacency:
                raise RuntimeError("Inlet-connected centerline component was empty.")

            deg = node_degrees(adjacency)
            endpoints_all = [int(n) for n, d in deg.items() if int(d) == 1]
            inlet_node, inlet_node_conf = pick_inlet_node_from_endpoints(endpoints_all, cl_pts, np.asarray(inlet_term.center, dtype=float), float(inlet_hyp["confidence"]), local_warnings)
            if inlet_node < 0:
                raise RuntimeError("Failed to identify inlet node on centerlines.")

            bif_hypotheses, dist, prev = generate_bifurcation_hypotheses(
                adjacency,
                cl_pts,
                inlet_node,
                np.asarray(inlet_hyp["axis_si"], dtype=float),
                local_warnings,
                terminations=terms,
                inlet_term=inlet_term,
                max_hypotheses=3,
            )
            if not bif_hypotheses:
                raise RuntimeError("No bifurcation hypotheses were generated.")

            for bif_rank, bif_hyp in enumerate(bif_hypotheses):
                eval_warnings = list(local_warnings)
                try:
                    bif_hyp = dict(bif_hyp)
                    bif_hyp["dist"] = dict(dist)
                    bif_hyp["prev"] = dict(prev)
                    interp = evaluate_interpretation(cl_pts, adjacency, inlet_node, inlet_hyp, bif_hyp, eval_warnings)
                    interp["warnings"] = list(eval_warnings)
                    interp["centerlines_raw"] = centerlines
                    interp["centerline_info"] = dict(cl_info)
                    interp["adjacency_full"] = dict(adjacency_full)
                    interp["adjacency_analysis"] = dict(adjacency)
                    interp["cl_pts"] = np.asarray(cl_pts, dtype=float)
                    interp["termination_mode_rank"] = int(inlet_rank)
                    interp["bifurcation_rank"] = int(bif_rank)
                    interp["inlet_node_confidence"] = float(inlet_node_conf)
                    all_interpretations.append(interp)
                except Exception as exc:
                    warnings.append(f"W_INTERPRETATION_EVAL_FAIL_{inlet_rank}_{bif_rank}: {exc}")
        except Exception as exc:
            warnings.append(f"W_INLET_HYPOTHESIS_FAIL_{inlet_rank}: {exc}")

    if not all_interpretations:
        raise RuntimeError("No valid anatomical interpretations were found.")
    all_interpretations.sort(
        key=lambda d: (
            -float(d["score"]),
            -float(d["bifurcation_confidence"]),
            -float(d["inlet_confidence"]),
            -float(d["renal_confidence"]),
            -float(d["laterality_confidence"]),
        )
    )
    best = all_interpretations[0]
    if debug_raw_path and first_successful_raw_centerlines is not None:
        try:
            write_vtp(first_successful_raw_centerlines, debug_raw_path, binary=True)
        except Exception as exc:
            warnings.append(f"W_DEBUG_WRITE_RAW_CENTERLINES: {exc}")
    return best


def main() -> int:
    require_vtk()

    parser = argparse.ArgumentParser(description="Anatomy-aware first-stage preprocessing for abdominal arterial lumen surfaces.")
    parser.add_argument("--input", type=str, default=INPUT_VTP_PATH, help="Input lumen surface VTP path")
    parser.add_argument("--output_surface_with_centerlines", type=str, default=OUTPUT_SURFACE_WITH_CENTERLINES_VTP_PATH, help="Output oriented surface+centerlines VTP path")
    parser.add_argument("--output_centerlines", type=str, default=OUTPUT_CENTERLINES_VTP_PATH, help="Output oriented labeled centerlines VTP path")
    parser.add_argument("--output", type=str, default=None, help="Deprecated alias for --output_centerlines")
    parser.add_argument("--metadata", type=str, default=OUTPUT_METADATA_PATH, help="Output metadata JSON path")
    parser.add_argument("--debug_raw_centerlines", type=str, default=OUTPUT_DEBUG_CENTERLINES_RAW_PATH, help="Optional raw centerlines VTP path")
    args = parser.parse_args()

    input_path = _resolve_user_path(args.input)
    surface_output_path = _resolve_user_path(args.output_surface_with_centerlines)
    centerline_output_arg = args.output if args.output is not None else args.output_centerlines
    centerline_output_path = _resolve_user_path(centerline_output_arg)
    metadata_path = _resolve_user_path(args.metadata) if args.metadata is not None else ""
    debug_raw_path = _resolve_user_path(args.debug_raw_centerlines) if args.debug_raw_centerlines is not None else ""

    warnings: List[str] = []

    try:
        surface = load_vtp(input_path)
        surface_tri = clean_and_triangulate_surface(surface)
        surface_pts = get_points_numpy(surface_tri)

        terms, mode = detect_terminations(surface_tri, warnings)
        if len(terms) < 2:
            raise RuntimeError("Failed to detect enough terminations for centerline seeding.")

        best = solve_best_interpretation(surface_tri, surface_pts, terms, warnings, debug_raw_path=debug_raw_path)
        warnings.extend(best.get("warnings", []))

        surface_tri_c = apply_transform_to_polydata(surface_tri, best["r"], best["origin"])

        inlet_node = int(best["inlet_node"])
        bif_node = int(best["bif_node"])
        trunk_path = [int(v) for v in best["trunk_path"]]
        cl_pts_c = np.asarray(best["cl_pts_c"], dtype=float)
        dist = dict(best["dist"])
        adjacency_full = dict(best["adjacency_full"])
        right_iliac_system = best["right_iliac_system"]
        left_iliac_system = best["left_iliac_system"]
        right_renal_system = best["right_renal_system"]
        left_renal_system = best["left_renal_system"]

        trunk_pts = cl_pts_c[np.asarray(trunk_path, dtype=int)]
        if trunk_pts.shape[0] >= 2 and float(trunk_pts[0, 2]) < float(trunk_pts[-1, 2]):
            trunk_path = list(reversed(trunk_path))
            trunk_pts = trunk_pts[::-1]

        trunk_node_to_index = {int(n): i for i, n in enumerate(trunk_path)}
        rr_origin_node = best.get("right_renal_origin")
        lr_origin_node = best.get("left_renal_origin")
        if rr_origin_node is not None and int(rr_origin_node) not in trunk_node_to_index:
            rr_origin_node = _nearest_node_on_path(trunk_path, cl_pts_c, int(rr_origin_node))
        if lr_origin_node is not None and int(lr_origin_node) not in trunk_node_to_index:
            lr_origin_node = _nearest_node_on_path(trunk_path, cl_pts_c, int(lr_origin_node))

        landmarks: Dict[str, np.ndarray] = {
            "Inlet": np.asarray(trunk_pts[0], dtype=float),
            "Bifurcation": np.asarray(cl_pts_c[int(bif_node)], dtype=float),
        }
        if rr_origin_node is not None and int(rr_origin_node) in trunk_node_to_index:
            landmarks["RightRenalOrigin"] = np.asarray(cl_pts_c[int(rr_origin_node)], dtype=float)
        if lr_origin_node is not None and int(lr_origin_node) in trunk_node_to_index:
            landmarks["LeftRenalOrigin"] = np.asarray(cl_pts_c[int(lr_origin_node)], dtype=float)

        branch_geoms = build_centerline_branch_geometries(
            adjacency_full=adjacency_full,
            pts_canonical=cl_pts_c,
            dist_from_inlet=dist,
            trunk_path=trunk_path,
            right_iliac_system=right_iliac_system,
            left_iliac_system=left_iliac_system,
            right_renal_system=right_renal_system,
            left_renal_system=left_renal_system,
            inlet_node=inlet_node,
            bif_node=bif_node,
            right_renal_origin=int(rr_origin_node) if rr_origin_node is not None else None,
            left_renal_origin=int(lr_origin_node) if lr_origin_node is not None else None,
            trunk_child_systems=list(best["trunk_child_systems"]),
        )
        if not branch_geoms:
            raise RuntimeError("Failed to construct branch-preserving centerline scaffold.")

        branch_geom_summary = summarize_centerline_branch_geometries(branch_geoms)
        branch_counts: Dict[str, int] = {}
        for br in branch_geoms:
            branch_counts[str(br["name"])] = int(branch_counts.get(str(br["name"]), 0) + 1)

        surface_tagged_pd, surface_label_summary = annotate_surface_polydata_for_combined_output(surface_tri_c, branch_geoms, warnings=warnings)
        centerlines_out_pd = build_output_centerlines_polydata(branch_geoms)
        combined_out_pd = build_combined_surface_centerlines_polydata(surface_tagged_pd, centerlines_out_pd)

        write_vtp(combined_out_pd, surface_output_path, binary=True)
        write_vtp(centerlines_out_pd, centerline_output_path, binary=True)

        if metadata_path:
            os.makedirs(os.path.dirname(os.path.abspath(metadata_path)) or ".", exist_ok=True)
            meta = {
                "status": "ok",
                "input_vtp": os.path.abspath(input_path),
                "output_surface_with_centerlines_vtp": os.path.abspath(surface_output_path),
                "output_centerlines_vtp": os.path.abspath(centerline_output_path),
                "mode": str(mode),
                "branch_names": sorted(str(name) for name in branch_counts.keys()),
                "branch_counts": {k: int(v) for k, v in branch_counts.items()},
                "centerline_length_by_branch": {
                    k: float(v) for k, v in dict(branch_geom_summary.get("length_by_branch_label", {})).items()
                },
                "centerline_branch_summaries": list(branch_geom_summary.get("branches", [])),
                "surface_cell_counts_by_branch": {
                    k: int(v) for k, v in dict(surface_label_summary.get("cell_counts", {})).items() if int(v) > 0
                },
                "landmarks_xyz_canonical": {
                    k: [float(x) for x in np.asarray(v, dtype=float).reshape(3)] for k, v in landmarks.items()
                },
                "transform": {
                    "R_rows": [[float(x) for x in row] for row in np.asarray(best["r"], dtype=float).reshape((3, 3))],
                    "origin": [float(x) for x in np.asarray(best["origin"], dtype=float).reshape((3,))],
                    "flipped_for_ap": bool(best["flipped_for_ap"]),
                },
                "renals_found": {
                    "right": bool(best.get("right_renal_ep") is not None and best.get("right_renal_origin") is not None),
                    "left": bool(best.get("left_renal_ep") is not None and best.get("left_renal_origin") is not None),
                },
                "confidences": {
                    "inlet": float(best["inlet_confidence"]),
                    "bifurcation": float(best["bifurcation_confidence"]),
                    "renal_assignment": float(best["renal_confidence"]),
                    "laterality": float(best["laterality_confidence"]),
                    "ap": float(best["ap_confidence"]),
                    "frame": float(best["frame_confidence"]),
                },
                "warnings": [str(w) for w in warnings],
                "diagnostics": {
                    "termination_count": int(len(terms)),
                    "termination_sources": [str(t.source) for t in terms],
                    "inlet_node": int(best["inlet_node"]),
                    "bif_node": int(best["bif_node"]),
                    "inlet_node_confidence": float(best.get("inlet_node_confidence", 0.0)),
                    "frame_info": dict(best.get("frame_info", {})),
                    "renal_diag": dict(best.get("renal_diag", {})),
                    "renal_scan": {
                        "candidate_count": int(best["renal_scan"].get("candidates", []) and len(best["renal_scan"]["candidates"]) or 0),
                        "pair_candidate_count": int(best["renal_scan"].get("pair_candidates", []) and len(best["renal_scan"]["pair_candidates"]) or 0),
                        "best_pair_confidence": float(best["renal_scan"]["best_pair"]["confidence"]) if best["renal_scan"].get("best_pair") is not None else 0.0,
                    },
                    "surface_transfer": {
                        "validation": dict(surface_label_summary.get("validation", {})),
                        "contact_map": dict(surface_label_summary.get("contact_map", {})),
                    },
                    "vmtk_import_diagnostics": dict(_LAST_VMTK_DIAGNOSTICS),
                },
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        return 0

    except Exception as exc:
        sys.stderr.write("ERROR: preprocessing failed.\n")
        sys.stderr.write(f"{exc}\n")
        sys.stderr.write(traceback.format_exc())
        if metadata_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(metadata_path)) or ".", exist_ok=True)
                meta = {
                    "status": "failed",
                    "input_vtp": os.path.abspath(input_path),
                    "output_surface_with_centerlines_vtp": os.path.abspath(surface_output_path),
                    "output_centerlines_vtp": os.path.abspath(centerline_output_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "warnings": [str(w) for w in warnings],
                    "vmtk_import_diagnostics": dict(_LAST_VMTK_DIAGNOSTICS),
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as meta_exc:
                sys.stderr.write(f"\nWARNING: failed to write metadata: {meta_exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
