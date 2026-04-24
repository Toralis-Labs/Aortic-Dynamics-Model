#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EVAR measurements from unlabeled abdominal aorta surface VTP.

Design goals:
- One-script, no manual interaction.
- Robust to open-termini lumen surfaces, wall-only surfaces, and capped/face-partitioned SimVascular-style surfaces
  where arrays such as ModelFaceID may be present.
- Best-effort anatomy inference from geometry/topology only.
- Output is a plain .txt file with all requested metrics (mm) and explicit warnings/confidence flags.

Notes:
- Centerline extraction: uses VMTK (vtkvmtk) if available. If unavailable/fails, script still writes output file with NaNs
  and attempts a limited fallback for maximum diameter estimation via slicing.
- Diameter definition is equivalent diameter from cross-sectional area A in an orthogonal plane to the local centerline:
    D_eq = sqrt(4*A/pi)
"""

from __future__ import annotations

INPUT_VTP_PATH = r"C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\0044_H_ABAO_AAA\\0044_H_ABAO_AAA\\Models\\0156_0001.vtp"
OUTPUT_TXT_PATH = r"C:\\Users\\ibrah\\OneDrive\\Desktop\\Fluids Project\\Vascular specific\\evar_measurements.txt"

import os
import sys
import math
import traceback
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any
import numpy as np

if TYPE_CHECKING:
    from vtkmodules.vtkCommonCore import vtkIdList
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStaticPointLocator

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
except Exception as _vtk_e:  # pragma: no cover
    vtk = None
    vtk_to_numpy = None
    _VTK_IMPORT_ERROR = str(_vtk_e)


# ----------------------------
# Constants and output schema
# ----------------------------

NAN = float("nan")

PRIMARY_MEASUREMENTS = [
    "Proximal_neck_D0",
    "Proximal_neck_D5",
    "Proximal_neck_D10",
    "Proximal_neck_D15",
    "Proximal_neck_length",
    "Right_common_iliac_D0",
    "Right_common_iliac_D10",
    "Right_common_iliac_D15",
    "Right_common_iliac_D20",
    "Right_common_iliac_length",
    "Left_common_iliac_D0",
    "Left_common_iliac_D10",
    "Left_common_iliac_D15",
    "Left_common_iliac_D20",
    "Left_common_iliac_length",
    "Length_lowest_renal_aortic_bifurcation",
    "Length_lowest_renal_iliac_bifurcation_right",
    "Length_lowest_renal_iliac_bifurcation_left",
    "Right_external_iliac_diameter",
    "Left_external_iliac_diameter",
    "Maximum_aneurysm_diameter",
]

EXTRA_OUTPUTS = [
    "Right_external_iliac_min_diameter",
    "Left_external_iliac_min_diameter",
    "Right_external_iliac_distal20mm_avg_diameter",
    "Left_external_iliac_distal20mm_avg_diameter",
    "Right_external_iliac_tortuosity",
    "Left_external_iliac_tortuosity",
    "Right_external_iliac_max_angulation_deg",
    "Left_external_iliac_max_angulation_deg",
    "D0_within_sac",
    "D5_within_sac",
    "D10_within_sac",
    "D15_within_sac",
]

META_OUTPUTS = [
    "Input_mode",
    "VMTK_available",
    "Scale_factor_to_mm",
    "Scale_confidence",
    "Canonical_frame_confidence",
    "warn_scale_inference",
    "warn_centerlines",
    "warn_ap_orientation",
    "warn_iliac_lr_orientation",
    "warn_inlet_identification",
    "warn_aortic_bifurcation",
    "warn_lowest_renal",
    "warn_right_common_iliac_bifurcation",
    "warn_left_common_iliac_bifurcation",
    "warn_right_external_iliac",
    "warn_left_external_iliac",
    "warn_left_right_mirror_ambiguity",
    "conf_ap_orientation",
    "conf_iliac_lr_orientation",
    "conf_inlet_identification",
    "conf_aortic_bifurcation",
    "conf_lowest_renal",
    "conf_right_external_iliac",
    "conf_left_external_iliac",
]

ALL_OUTPUT_KEYS = PRIMARY_MEASUREMENTS + EXTRA_OUTPUTS + META_OUTPUTS

SEMANTIC_TERMINATION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "inflow": ("inflow", "aorta_inflow"),
    "ext_iliac_right": ("ext_iliac_right", "external_iliac_right", "right_external_iliac"),
    "ext_iliac_left": ("ext_iliac_left", "external_iliac_left", "left_external_iliac"),
    "int_iliac_right": ("int_iliac_right", "internal_iliac_right", "right_internal_iliac"),
    "int_iliac_left": ("int_iliac_left", "internal_iliac_left", "left_internal_iliac"),
    "renal_right": ("renal_right", "right_renal"),
    "renal_left": ("renal_left", "left_renal"),
    "sma": ("sma",),
    "ima": ("ima",),
    "celiac_hepatic": ("celiac_hepatic", "celiac"),
    "celiac_splenic": ("celiac_splenic", "celiac"),
}

VENTRAL_SEMANTIC_KEYS = ("ima", "sma", "celiac_hepatic", "celiac_splenic")


# ----------------------------
# Basic helpers
# ----------------------------

def add_warning(warnings: List[str], code: str, msg: str) -> None:
    warnings.append(f"{code}: {msg}")


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return NAN
        if isinstance(x, (bool, np.bool_)):
            return float(int(x))
        return float(x)
    except Exception:
        return NAN


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_semantic_label(label: Any) -> str:
    text = str(label).strip().lower()
    return text.replace("-", "_").replace(" ", "_")


def canonical_semantic_label(label: Any) -> str:
    norm = normalize_semantic_label(label)
    for canonical, aliases in SEMANTIC_TERMINATION_ALIASES.items():
        if norm == canonical:
            return canonical
        if norm in aliases:
            return canonical
    return norm


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=float)
    return (v / n).astype(float)


def polyline_length(points: np.ndarray) -> float:
    if points is None or len(points) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def make_results_template() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for k in ALL_OUTPUT_KEYS:
        if k in ["Input_mode", "VMTK_available"]:
            results[k] = ""
        elif k.startswith("warn_"):
            results[k] = 0
        elif k.startswith("conf_") or k.endswith("_confidence") or k.endswith("_conf"):
            results[k] = NAN
        elif k.endswith("_within_sac"):
            results[k] = "NaN"
        else:
            results[k] = NAN
    results["Input_mode"] = "unknown"
    results["VMTK_available"] = "false"
    results["Scale_factor_to_mm"] = NAN
    results["Scale_confidence"] = NAN
    results["Canonical_frame_confidence"] = NAN
    return results


def write_output_txt(path: str, results: Dict[str, Any], warnings: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: List[str] = []
    for k in ALL_OUTPUT_KEYS:
        v = results.get(k, NAN)
        if isinstance(v, str):
            lines.append(f"{k}={v}")
        elif isinstance(v, (bool, np.bool_)):
            lines.append(f"{k}={'true' if bool(v) else 'false'}")
        else:
            if v is None:
                lines.append(f"{k}=NaN")
            else:
                try:
                    fv = float(v)
                    if math.isnan(fv):
                        lines.append(f"{k}=NaN")
                    elif math.isinf(fv):
                        lines.append(f"{k}=Inf" if fv > 0 else f"{k}=-Inf")
                    else:
                        lines.append(f"{k}={fv:.6f}")
                except Exception:
                    lines.append(f"{k}={v}")
    # Append warnings section
    lines.append("")
    lines.append("Warnings_count=%d" % len(warnings))
    for i, w in enumerate(warnings, start=1):
        lines.append(f"WARNING_{i:03d}={w}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# VTK helpers (safe wrappers)
# ----------------------------

def vtk_available() -> bool:
    return vtk is not None


def load_vtp(path: str) -> vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    out = reader.GetOutput()
    if out is None:
        raise RuntimeError("vtkXMLPolyDataReader produced no output.")
    return out


def deep_copy_polydata(pd: vtkPolyData) -> vtkPolyData:
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    return out


def clean_and_triangulate(pd: vtkPolyData) -> vtkPolyData:
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

    out = tri.GetOutput()
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
    edges = fe.GetOutput()
    return int(edges.GetNumberOfCells()) if edges is not None else 0


def compute_polydata_surface_area(pd: vtkPolyData) -> float:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.Update()

    mass = vtk.vtkMassProperties()
    mass.SetInputConnection(tri.GetOutputPort())
    mass.Update()
    return float(mass.GetSurfaceArea())


def get_points_numpy(pd: vtkPolyData) -> np.ndarray:
    pts = pd.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    arr = pts.GetData()
    if vtk_to_numpy is None:
        raise RuntimeError("vtk_to_numpy not available")
    return vtk_to_numpy(arr).astype(float)


def apply_linear_transform_to_polydata(pd: vtkPolyData, R: np.ndarray, t: np.ndarray) -> vtkPolyData:
    """
    Apply x' = R x + t to all points in a vtkPolyData.
    """
    tf = vtk.vtkTransform()
    m = vtk.vtkMatrix4x4() 
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, float(R[i, j]))
    m.SetElement(0, 3, float(t[0]))
    m.SetElement(1, 3, float(t[1]))
    m.SetElement(2, 3, float(t[2]))
    m.SetElement(3, 3, 1.0)
    tf.SetMatrix(m)

    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(tf)
    f.SetInputData(pd)
    f.Update()
    return f.GetOutput()


def transform_terminations(
    terminations: List[Dict[str, Any]],
    R: np.ndarray,
    t: np.ndarray,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for term in terminations:
        tt = dict(term)
        c = np.array(term.get("center", np.zeros(3)), dtype=float)
        tt["center"] = (R @ c.reshape(3, 1)).reshape(3) + t
        if "normal" in term and term["normal"] is not None:
            n = np.array(term.get("normal", np.zeros(3)), dtype=float)
            tt["normal"] = (R @ n.reshape(3, 1)).reshape(3)
        out.append(tt)
    return out


def extract_largest_connected_region(pd: vtkPolyData) -> vtkPolyData:
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(conn.GetOutput())
    return out


def extract_largest_connected_region_lines(pd: vtkPolyData) -> vtkPolyData:
    """
    Preserve polyline cells while cleaning merged-point connectivity on centerlines.
    """
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(conn.GetOutput())
    cleaner.PointMergingOn()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    return out


def polydata_line_stats(pd: vtkPolyData) -> Tuple[int, int, int]:
    lines_obj = pd.GetLines() if pd is not None else None
    n_line_cells = int(lines_obj.GetNumberOfCells()) if lines_obj is not None else 0
    n_points = int(pd.GetNumberOfPoints()) if pd is not None else 0
    n_cells = int(pd.GetNumberOfCells()) if pd is not None else 0
    return n_points, n_cells, n_line_cells


def build_static_point_locator(pd: vtkPolyData) -> vtkStaticPointLocator:
    loc = vtk.vtkStaticPointLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    return loc


# ----------------------------
# Termination detection
# ----------------------------

def planar_polygon_area_and_normal(points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Estimate area of (approximately planar) closed polygon via PCA plane fit and 2D shoelace.
    Returns: (area, normal_unit, rms_plane_distance)
    """
    if len(points) < 3:
        return 0.0, np.zeros(3), NAN
    pts = np.asarray(points, dtype=float)
    # Remove duplicate last point if present
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-8 * (np.linalg.norm(pts[0]) + 1.0):
        pts = pts[:-1]
    if len(pts) < 3:
        return 0.0, np.zeros(3), NAN
    c = np.mean(pts, axis=0)
    X = pts - c
    # PCA
    cov = (X.T @ X) / max(len(X), 1)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)
    # normal is smallest eigenvector
    n = unit(V[:, idx[0]])
    u = unit(V[:, idx[2]])  # largest
    v = unit(np.cross(n, u))
    # Plane distances
    dists = X @ n
    rms = float(np.sqrt(np.mean(dists * dists)))
    # Project
    x2 = X @ u
    y2 = X @ v
    # Shoelace
    x_next = np.roll(x2, -1)
    y_next = np.roll(y2, -1)
    area = 0.5 * float(abs(np.sum(x2 * y_next - x_next * y2)))
    return area, n, rms


def extract_boundary_loops(pd: vtkPolyData) -> List[Dict[str, Any]]:
    """
    Extract boundary edge loops (open-termini mode).
    Returns list of dicts: {center, area, normal, rms_planarity, n_points, source}
    """
    loops: List[Dict[str, Any]] = []
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
        loops.append(
            dict(
                center=center,
                area=area,
                diameter_eq=math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0,
                normal=normal,
                rms_planarity=rms,
                n_points=int(nids),
                source="boundary_loop",
            )
        )
    return loops


def extract_feature_edge_loops(pd: vtkPolyData, feature_angle_deg: float = 60.0) -> List[Dict[str, Any]]:
    """
    Fallback termination inference for closed, unpartitioned models:
    extract feature edges (sharp seams), then try to form closed loops.
    """
    loops: List[Dict[str, Any]] = []
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOff()
    fe.FeatureEdgesOn()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.SetFeatureAngle(feature_angle_deg)
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
        if nids < 6:
            continue
        coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
        center = np.mean(coords, axis=0)
        area, normal, rms = planar_polygon_area_and_normal(coords)
        loops.append(
            dict(
                center=center,
                area=area,
                diameter_eq=math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0,
                normal=normal,
                rms_planarity=rms,
                n_points=int(nids),
                source="feature_edge_loop",
            )
        )
    return loops


def find_face_partition_array_name(pd: vtkPolyData) -> Optional[str]:
    """
    Attempt to find a SimVascular-style face partition array in cell data.
    Prefer 'ModelFaceID', otherwise look for arrays containing 'face' and 'id'.
    """
    cd = pd.GetCellData()
    if cd is None:
        return None
    n_arrays = cd.GetNumberOfArrays()
    candidates: List[Tuple[int, str]] = []
    for i in range(n_arrays):
        arr = cd.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if not name:
            continue
        lname = name.lower()
        if "modelfaceid" == lname:
            return name
        if ("face" in lname and "id" in lname) or (lname.endswith("faceid")):
            candidates.append((i, name))
    # Pick the best candidate with reasonable number of unique ids
    best_name = None
    best_score = -1.0
    for _, name in candidates:
        arr = cd.GetArray(name)
        if arr is None:
            continue
        try:
            vals = vtk_to_numpy(arr)
        except Exception:
            continue
        if vals.size == 0:
            continue
        # prefer int-like arrays
        uniq = np.unique(vals.astype(np.int64))
        if len(uniq) <= 1:
            continue
        if len(uniq) > 5000:
            continue
        score = 0.0
        if "face" in name.lower():
            score += 1.0
        if "id" in name.lower():
            score += 1.0
        score += 1.0 / (1.0 + float(len(uniq)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def termination_candidates_from_face_partitions(pd_tri: vtkPolyData, face_array: str) -> List[Dict[str, Any]]:
    """
    Compute per-face-id planarity and area-weighted centroids to identify cap-like patches.
    Returns termination candidates only (cap-likely faces).
    """
    cd = pd_tri.GetCellData()
    face_arr = cd.GetArray(face_array) if cd is not None else None
    if face_arr is None or vtk_to_numpy is None:
        return []

    # Compute cell area in C++ (fast)
    cell_size = vtk.vtkCellSizeFilter()
    cell_size.SetInputData(pd_tri)
    cell_size.SetComputeArea(True)
    cell_size.SetComputeLength(False)
    cell_size.SetComputeVolume(False)
    cell_size.SetComputeVertexCount(False)
    cell_size.Update()
    pd_area = cell_size.GetOutput()

    # Compute cell normals
    normals_f = vtk.vtkPolyDataNormals()
    normals_f.SetInputData(pd_area)
    normals_f.ComputePointNormalsOff()
    normals_f.ComputeCellNormalsOn()
    normals_f.SplittingOff()
    normals_f.ConsistencyOn()
    normals_f.AutoOrientNormalsOff()
    normals_f.Update()
    pd_n = normals_f.GetOutput()

    # Cell centers
    centers_f = vtk.vtkCellCenters()
    centers_f.SetInputData(pd_n)
    centers_f.VertexCellsOn()
    centers_f.Update()
    centers_pd = centers_f.GetOutput()
    centers_pts = centers_pd.GetPoints()

    face_vals = vtk_to_numpy(pd_n.GetCellData().GetArray(face_array)).astype(np.int64)
    area_vals = vtk_to_numpy(pd_n.GetCellData().GetArray("Area")).astype(float)

    cell_normals_vtk = pd_n.GetCellData().GetNormals()
    if cell_normals_vtk is None:
        # try by name
        cell_normals_vtk = pd_n.GetCellData().GetArray("Normals")
    if cell_normals_vtk is None:
        return []
    normal_vals = vtk_to_numpy(cell_normals_vtk).astype(float)

    if centers_pts is None:
        return []

    centers_vals = vtk_to_numpy(centers_pts.GetData()).astype(float)
    if centers_vals.shape[0] != face_vals.shape[0]:
        # Fallback: do not trust centers
        return []

    total_area = float(np.sum(area_vals)) if area_vals.size > 0 else 0.0
    if total_area <= 0:
        return []

    terminations: List[Dict[str, Any]] = []
    uniq_ids = np.unique(face_vals)

    # Compute stats per face id
    face_stats: List[Dict[str, Any]] = []
    for fid in uniq_ids:
        mask = (face_vals == fid)
        if not np.any(mask):
            continue
        a = area_vals[mask]
        a_sum = float(np.sum(a))
        if a_sum <= 0:
            continue
        c = np.sum(centers_vals[mask] * a[:, None], axis=0) / a_sum
        n_sum = np.sum(normal_vals[mask] * a[:, None], axis=0)
        planarity = float(np.linalg.norm(n_sum) / (a_sum + 1e-12))
        diameter_eq = math.sqrt(4.0 * a_sum / math.pi)
        face_stats.append(
            dict(
                face_id=int(fid),
                area=a_sum,
                center=c,
                planarity=planarity,
                diameter_eq=diameter_eq,
            )
        )

    if not face_stats:
        return terminations

    # Heuristics:
    # - caps are often very planar (planarity ~ 1)
    # - caps are small compared to the wall area
    # - there should typically be multiple caps
    # Use relative thresholds (unitless), safe pre-scale.
    areas = np.array([fs["area"] for fs in face_stats], dtype=float)
    max_area = float(np.max(areas)) if areas.size > 0 else 0.0
    # Consider cap-candidates among planar faces excluding the largest face (likely wall)
    for fs in face_stats:
        if fs["planarity"] < 0.92:
            continue
        # exclude extremely large planar face: likely wall or large planar artifact
        if fs["area"] > 0.60 * total_area:
            continue
        # exclude faces that are too large relative to top face area
        if max_area > 0 and fs["area"] > 0.80 * max_area and len(face_stats) > 3:
            continue
        terminations.append(
            dict(
                center=np.array(fs["center"], dtype=float),
                area=float(fs["area"]),
                diameter_eq=float(fs["diameter_eq"]),
                normal=np.zeros(3),
                rms_planarity=NAN,
                n_points=int(0),
                source=f"face_partition:{face_array}",
                face_id=int(fs["face_id"]),
                planarity=float(fs["planarity"]),
            )
        )

    # If none selected, return empty to allow other modes
    return terminations


def detect_terminations_and_mode(pd_tri: vtkPolyData, warnings: List[str]) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """
    Determine input mode and collect termination candidates.

    Returns: (terminations, mode, face_array_name)
    mode in {'open_termini', 'capped_partitioned', 'closed_unpartitioned', 'unsupported'}
    """
    # Mode 1: open boundaries
    n_boundary_edges = count_boundary_edges(pd_tri)
    if n_boundary_edges > 0:
        loops = extract_boundary_loops(pd_tri)
        if loops:
            return loops, "open_termini", None

    # Mode 2: face-partitioned / capped (e.g., ModelFaceID exists)
    face_array = find_face_partition_array_name(pd_tri)
    if face_array:
        terms = termination_candidates_from_face_partitions(pd_tri, face_array)
        # Need at least 2 terminations (inlet + outlet) for centerlines
        if len(terms) >= 2:
            return terms, "capped_partitioned", face_array
        else:
            add_warning(warnings, "W_FACEPART_001", f"Found cell-data face array '{face_array}', but could not robustly identify cap faces.")
    # Mode 3: closed but not partitioned - try feature edges seams
    loops2 = extract_feature_edge_loops(pd_tri, feature_angle_deg=60.0)
    if len(loops2) >= 2:
        return loops2, "closed_unpartitioned", None

    return [], "unsupported", face_array


def candidate_mesh_surfaces_dirs(input_vtp_path: str) -> List[str]:
    abs_path = os.path.abspath(input_vtp_path)
    stem = os.path.splitext(os.path.basename(abs_path))[0]
    parent_dir = os.path.dirname(abs_path)
    case_root = os.path.dirname(parent_dir)
    candidates: List[str] = []

    preferred = os.path.join(case_root, "Simulations", stem, "mesh-complete", "mesh-surfaces")
    candidates.append(preferred)

    sims_root = os.path.join(case_root, "Simulations")
    if os.path.isdir(sims_root):
        for sim_name in sorted(os.listdir(sims_root)):
            candidates.append(os.path.join(sims_root, sim_name, "mesh-complete", "mesh-surfaces"))

    out: List[str] = []
    seen = set()
    for cand in candidates:
        norm = os.path.normpath(cand)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def load_semantic_termination_priors(input_vtp_path: str) -> List[Dict[str, Any]]:
    if vtk is None:
        return []

    priors: List[Dict[str, Any]] = []
    mesh_surfaces_dir = None
    for cand in candidate_mesh_surfaces_dirs(input_vtp_path):
        inflow_file = os.path.join(cand, "inflow.vtp")
        if os.path.isdir(cand) and os.path.exists(inflow_file):
            mesh_surfaces_dir = cand
            break
    if mesh_surfaces_dir is None:
        return priors

    for name in sorted(os.listdir(mesh_surfaces_dir)):
        if not name.lower().endswith(".vtp"):
            continue
        path = os.path.join(mesh_surfaces_dir, name)
        try:
            pd = clean_and_triangulate(load_vtp(path))
            pts = get_points_numpy(pd)
            if pts.shape[0] < 3:
                continue
            area = compute_polydata_surface_area(pd)
            if not math.isfinite(area) or area <= 0:
                continue
            center = np.mean(pts, axis=0)
            priors.append(
                dict(
                    semantic_label=canonical_semantic_label(os.path.splitext(name)[0]),
                    center=np.array(center, dtype=float),
                    area=float(area),
                    diameter_eq=float(math.sqrt(4.0 * area / math.pi)),
                    source_path=path,
                )
            )
        except Exception:
            continue
    return priors


def annotate_terminations_with_semantic_labels(
    terminations: List[Dict[str, Any]],
    input_vtp_path: str,
) -> int:
    priors = load_semantic_termination_priors(input_vtp_path)
    if not priors or not terminations:
        return 0

    scored_pairs: List[Tuple[float, float, int, int]] = []
    for ti, term in enumerate(terminations):
        tc = np.array(term.get("center", np.zeros(3)), dtype=float)
        ta = max(float(term.get("area", 0.0)), 1e-12)
        td = max(float(term.get("diameter_eq", 0.0)), 1e-6)
        for pi, prior in enumerate(priors):
            pc = np.array(prior["center"], dtype=float)
            pa = max(float(prior["area"]), 1e-12)
            pd = max(float(prior["diameter_eq"]), 1e-6)
            dist = float(np.linalg.norm(tc - pc))
            score = dist / max(td, pd, 1e-6) + 0.35 * abs(math.log(ta / pa))
            scored_pairs.append((score, dist, ti, pi))

    assigned_terms = set()
    assigned_priors = set()
    matched = 0
    for _, dist, ti, pi in sorted(scored_pairs, key=lambda item: (item[0], item[1])):
        if ti in assigned_terms or pi in assigned_priors:
            continue
        term = terminations[ti]
        prior = priors[pi]
        match_scale = max(
            float(term.get("diameter_eq", 0.0)),
            float(prior.get("diameter_eq", 0.0)),
            1e-6,
        )
        if dist > max(0.5, 0.75 * match_scale):
            continue
        term["semantic_label"] = str(prior["semantic_label"])
        term["semantic_confidence"] = float(clamp(1.0 - dist / (1.5 * match_scale + 1e-6), 0.0, 1.0))
        term["semantic_match_distance"] = float(dist)
        term["semantic_source"] = str(prior["source_path"])
        assigned_terms.add(ti)
        assigned_priors.add(pi)
        matched += 1
    return matched


def find_termination_by_semantic_label(
    terminations: List[Dict[str, Any]],
    *labels: str,
) -> Optional[Dict[str, Any]]:
    wanted = {canonical_semantic_label(label) for label in labels}
    candidates = [
        term
        for term in terminations
        if canonical_semantic_label(term.get("semantic_label", "")) in wanted
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda term: float(term.get("semantic_confidence", 0.0)))


def choose_centerline_endpoint_for_termination(
    endpoints: List[int],
    pts: np.ndarray,
    termination: Optional[Dict[str, Any]],
    exclude: Optional[set[int]] = None,
) -> Tuple[Optional[int], float]:
    if termination is None:
        return None, 0.0
    exclude = exclude or set()
    candidates = [int(ep) for ep in endpoints if int(ep) not in exclude]
    if not candidates:
        return None, 0.0

    center = np.array(termination.get("center", np.zeros(3)), dtype=float)
    dists = np.array([np.linalg.norm(pts[ep] - center) for ep in candidates], dtype=float)
    idx = int(np.argmin(dists))
    best_ep = int(candidates[idx])
    best_dist = float(dists[idx])
    diameter_eq = max(float(termination.get("diameter_eq", 0.0)), 6.0)
    conf = float(clamp(1.0 - best_dist / (1.25 * diameter_eq + 1e-6), 0.0, 1.0))
    return best_ep, conf


def map_semantic_terminations_to_centerline_endpoints(
    endpoints: List[int],
    pts: np.ndarray,
    terminations: List[Dict[str, Any]],
    exclude: Optional[List[int]] = None,
) -> Dict[str, Any]:
    used = {int(x) for x in (exclude or [])}
    mapping: Dict[str, Any] = {}
    for canonical, aliases in SEMANTIC_TERMINATION_ALIASES.items():
        term = find_termination_by_semantic_label(terminations, canonical, *aliases)
        if term is None:
            continue
        ep, conf = choose_centerline_endpoint_for_termination(
            endpoints,
            pts,
            term,
            exclude=(used if canonical != "inflow" else set()),
        )
        if ep is None:
            continue
        mapping[canonical] = int(ep)
        mapping[f"{canonical}_conf"] = float(conf)
        if canonical != "inflow":
            used.add(int(ep))
    return mapping


def resolve_frame_sign_from_semantics(
    pts_canonical: np.ndarray,
    terminations_canonical: List[Dict[str, Any]],
    semantic_endpoints: Dict[str, Any],
) -> Tuple[int, float, float, int, int]:
    vote = 0.0
    lr_conf = 0.0
    ap_conf = 0.0

    right_ext = semantic_endpoints.get("ext_iliac_right")
    left_ext = semantic_endpoints.get("ext_iliac_left")
    if right_ext is not None and left_ext is not None:
        x_sep = float(pts_canonical[int(right_ext)][0] - pts_canonical[int(left_ext)][0])
        x_mag = abs(x_sep)
        endpoint_conf = min(
            float(semantic_endpoints.get("ext_iliac_right_conf", 0.0)),
            float(semantic_endpoints.get("ext_iliac_left_conf", 0.0)),
        )
        lr_conf = float(
            clamp(
                0.45 + 0.30 * endpoint_conf + 0.25 * clamp(x_mag / 30.0, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        vote += lr_conf if x_sep >= 0.0 else -lr_conf

    ventral_terms = [
        find_termination_by_semantic_label(terminations_canonical, key)
        for key in VENTRAL_SEMANTIC_KEYS
    ]
    ventral_terms = [term for term in ventral_terms if term is not None]
    if ventral_terms:
        y_vals = np.array([float(term["center"][1]) for term in ventral_terms], dtype=float)
        weights = np.array(
            [max(0.25, min(1.0, float(term.get("diameter_eq", 0.0)) / 6.0)) for term in ventral_terms],
            dtype=float,
        )
        mean_y = float(np.average(y_vals, weights=weights)) if np.any(weights > 0) else float(np.mean(y_vals))
        ap_conf = float(
            clamp(
                0.35
                + 0.25 * clamp(abs(mean_y) / 10.0, 0.0, 1.0)
                + 0.40 * clamp(len(ventral_terms) / 3.0, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        vote += ap_conf if mean_y >= 0.0 else -ap_conf

    need_flip = int(vote < 0.0)
    lr_warn = int(right_ext is None or left_ext is None or lr_conf < 0.55)
    ap_warn = int((not ventral_terms) or ap_conf < 0.55)
    return need_flip, lr_conf, ap_conf, lr_warn, ap_warn


def find_branch_takeoff_on_trunk(
    adjacency: Dict[int, Dict[int, float]],
    inlet_node: int,
    endpoint_node: Optional[int],
    trunk_nodes: List[int],
    dist_from_inlet: Dict[int, float],
) -> Tuple[Optional[int], List[int], float]:
    if endpoint_node is None:
        return None, [], NAN
    path_ie, _ = shortest_path(adjacency, inlet_node, int(endpoint_node))
    if not path_ie:
        return None, [], NAN
    trunk_set = set(int(n) for n in trunk_nodes)
    common = [int(n) for n in path_ie if int(n) in trunk_set]
    if not common:
        return None, path_ie, NAN
    takeoff = max(common, key=lambda n: dist_from_inlet.get(n, -1.0))
    return int(takeoff), path_ie, float(dist_from_inlet.get(takeoff, NAN))


def find_shared_branch_bifurcation(
    adjacency: Dict[int, Dict[int, float]],
    root_node: int,
    endpoint_a: Optional[int],
    endpoint_b: Optional[int],
    dist_from_root: Dict[int, float],
) -> Tuple[Optional[int], float]:
    if root_node < 0 or endpoint_a is None or endpoint_b is None:
        return None, 0.0

    path_a, _ = shortest_path(adjacency, root_node, int(endpoint_a))
    path_b, _ = shortest_path(adjacency, root_node, int(endpoint_b))
    if len(path_a) < 2 or len(path_b) < 2:
        return None, 0.0

    bif_node = path_common_node_with_max_distance(path_a, path_b, dist_from_root)
    if bif_node is None:
        return None, 0.0

    idx_a = path_a.index(bif_node)
    idx_b = path_b.index(bif_node)
    rem_a = len(path_a) - idx_a - 1
    rem_b = len(path_b) - idx_b - 1
    if rem_a < 1 or rem_b < 1 or bif_node in {int(endpoint_a), int(endpoint_b)}:
        return None, 0.0

    min_ep_dist = max(
        min(float(dist_from_root.get(int(endpoint_a), 0.0)), float(dist_from_root.get(int(endpoint_b), 0.0))),
        1e-6,
    )
    conf = float(
        clamp(
            0.35
            + 0.25 * clamp(min(rem_a, rem_b) / 3.0, 0.0, 1.0)
            + 0.40 * clamp(float(dist_from_root.get(int(bif_node), 0.0)) / min_ep_dist, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    return int(bif_node), conf


def build_semantic_renal_candidates(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    inlet_node: int,
    bif_node: int,
    trunk_nodes: List[int],
    dist_from_inlet: Dict[int, float],
    semantic_endpoints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    bif_dist = float(dist_from_inlet.get(bif_node, 0.0))
    out: List[Dict[str, Any]] = []
    for key in ("renal_left", "renal_right"):
        ep = semantic_endpoints.get(key)
        if ep is None:
            continue
        takeoff, _, takeoff_dist = find_branch_takeoff_on_trunk(
            adjacency,
            inlet_node,
            int(ep),
            trunk_nodes,
            dist_from_inlet,
        )
        if takeoff is None or not math.isfinite(takeoff_dist):
            continue
        if takeoff_dist > bif_dist - 5.0:
            continue
        vec = np.array(pts[int(ep)] - pts[int(takeoff)], dtype=float)
        vec_perp = vec.copy()
        vec_perp[2] = 0.0
        lateral_ratio = float(np.linalg.norm(vec_perp) / (np.linalg.norm(vec) + 1e-12))
        out.append(
            dict(
                endpoint=int(ep),
                takeoff=int(takeoff),
                takeoff_dist=float(takeoff_dist),
                x=float(pts[int(ep)][0]),
                z=float(pts[int(ep)][2]),
                r=lateral_ratio,
                semantic_label=key,
            )
        )
    return out


# ----------------------------
# Scale inference
# ----------------------------

def principal_axis_length(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    c = np.mean(points, axis=0)
    X = points - c
    try:
        cov = (X.T @ X) / max(X.shape[0], 1)
        w, V = np.linalg.eigh(cov)
        axis = unit(V[:, np.argmax(w)])
        proj = X @ axis
        return float(np.max(proj) - np.min(proj))
    except Exception:
        return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))


def infer_scale_to_mm(pd_tri: vtkPolyData, terminations: List[Dict[str, Any]], warnings: List[str]) -> Tuple[float, float, int]:
    """
    Infer scale factor to convert current units to mm.

    Returns (scale_factor, confidence in [0,1], warn_flag)
    """
    pts = get_points_numpy(pd_tri)
    L_raw = principal_axis_length(pts)
    if L_raw <= 0:
        add_warning(warnings, "W_SCALE_001", "Could not compute a stable length scale from geometry; defaulting to mm scale_factor=1.")
        return 1.0, 0.0, 1

    diameters = np.array([t.get("diameter_eq", 0.0) for t in terminations], dtype=float)
    diameters = diameters[np.isfinite(diameters)]
    D_med_raw = float(np.median(np.sort(diameters)[-min(5, len(diameters)):])) if len(diameters) > 0 else NAN

    # Candidate scale factors: try mm, cm->mm, m->mm, dm->mm, inch->mm
    candidates = [0.1, 1.0, 10.0, 25.4, 100.0, 1000.0]
    scores: List[float] = []

    def penalty_range(x: float, lo: float, hi: float) -> float:
        if not math.isfinite(x) or x <= 0:
            return 10.0
        if x < lo:
            return (lo / x) ** 2
        if x > hi:
            return (x / hi) ** 2
        return 0.0

    for f in candidates:
        L = L_raw * f
        score = 0.0
        # plausible abdominal aorta+iliac extent in mm (broad)
        score += penalty_range(L, 80.0, 800.0)
        score += abs(math.log(max(L, 1e-6) / 300.0))
        if math.isfinite(D_med_raw) and D_med_raw > 0:
            Dm = D_med_raw * f
            # plausible median termination diameter in mm
            score += 0.75 * penalty_range(Dm, 4.0, 60.0)
            score += 0.35 * abs(math.log(max(Dm, 1e-6) / 20.0))
        scores.append(score)

    best_idx = int(np.argmin(scores))
    best_f = float(candidates[best_idx])
    sorted_scores = np.sort(np.array(scores))
    if len(sorted_scores) >= 2:
        gap = float(sorted_scores[1] - sorted_scores[0])
        conf = float(1.0 / (1.0 + math.exp(-gap)))  # logistic
    else:
        conf = 0.0

    warn = 0
    if conf < 0.65:
        warn = 1
        add_warning(
            warnings,
            "W_SCALE_002",
            f"Scale inference ambiguous (confidence={conf:.3f}). Using scale_factor_to_mm={best_f}. Raw length={L_raw:.3f}.",
        )
    return best_f, conf, warn


# ----------------------------
# VMTK centerlines
# ----------------------------

def try_import_vmtk() -> Tuple[Optional[Any], Optional[str]]:
    """
    Try importing vtkvmtk module (VMTK python bindings).
    Returns (vtkvmtk_module, error_message_if_any)
    """
    try:
        from vmtk import vtkvmtk  # type: ignore
        return vtkvmtk, None
    except Exception as e1:
        try:
            import vtkvmtk  # type: ignore
            return vtkvmtk, None
        except Exception as e2:
            return None, f"{e1} | {e2}"


def cap_surface_if_open(pd_tri: vtkPolyData, vtkvmtk_mod: Any) -> Tuple[vtkPolyData, Optional[vtkIdList]]:
    """
    If surface has boundaries, cap it using vtkvmtkCapPolyData and return (capped_surface, capCenterIds).
    If no boundaries, returns (pd_tri, None).
    """
    if count_boundary_edges(pd_tri) <= 0:
        return pd_tri, None
    capper = vtkvmtk_mod.vtkvmtkCapPolyData()
    capper.SetInputData(pd_tri)
    capper.SetDisplacement(0.0)
    capper.SetInPlaneDisplacement(0.0)
    capper.Update()
    capped = capper.GetOutput()
    cap_ids = capper.GetCapCenterIds()
    return capped, cap_ids


def compute_centerlines_vmtk(
    pd_tri: vtkPolyData,
    terminations: List[Dict[str, Any]],
    mode: str,
    warnings: List[str],
) -> Tuple[Optional[vtkPolyData], Dict[str, Any]]:
    """
    Compute centerlines using VMTK.

    Returns (centerlines_polydata_or_None, info_dict).
    """
    info: Dict[str, Any] = dict(
        used_flip_normals=0,
        used_cap_center_ids=False,
        inlet_seed=None,
        n_targets=0,
    )

    vtkvmtk_mod, err = try_import_vmtk()
    if vtkvmtk_mod is None:
        add_warning(warnings, "W_VMTK_001", f"VMTK (vtkvmtk) not available: {err}")
        return None, info

    # Determine inlet/outlet termination candidates based on area (largest is inlet)
    if not terminations or len(terminations) < 2:
        add_warning(warnings, "W_VMTK_002", "Not enough termination candidates to seed centerline extraction (need >=2).")
        return None, info

    # Filter out extremely tiny termination candidates relative to max area (unitless / scale-invariant)
    areas = np.array([max(0.0, float(t.get("area", 0.0))) for t in terminations], dtype=float)
    max_area = float(np.max(areas))
    keep = []
    for t in terminations:
        a = max(0.0, float(t.get("area", 0.0)))
        if max_area <= 0:
            keep.append(t)
        else:
            if a >= 1e-4 * max_area:
                keep.append(t)
    if len(keep) >= 2:
        terminations_use = keep
    else:
        terminations_use = terminations

    # Pick inlet termination: largest area
    inlet_term = max(terminations_use, key=lambda d: float(d.get("area", 0.0)))
    inlet_center = np.array(inlet_term["center"], dtype=float)

    # Prepare seeds
    capped_surface, cap_center_ids = cap_surface_if_open(pd_tri, vtkvmtk_mod)
    locator = build_static_point_locator(capped_surface)

    source_ids = vtk.vtkIdList()
    target_ids = vtk.vtkIdList()

    if cap_center_ids is not None and cap_center_ids.GetNumberOfIds() > 1:
        # When CapCenterIds is set, SourceSeedIds and TargetSeedIds are indices into the cap list.
        info["used_cap_center_ids"] = True
        info["n_caps"] = int(cap_center_ids.GetNumberOfIds())
        cap_centers = np.array([capped_surface.GetPoint(cap_center_ids.GetId(i)) for i in range(cap_center_ids.GetNumberOfIds())], dtype=float)

        # Map each cap index to nearest termination candidate to estimate area and choose inlet reliably
        term_centers = np.array([np.array(t["center"], dtype=float) for t in terminations_use], dtype=float)
        term_areas = np.array([float(t.get("area", 0.0)) for t in terminations_use], dtype=float)
        cap_area_est = np.zeros((cap_centers.shape[0],), dtype=float)
        for i in range(cap_centers.shape[0]):
            d2 = np.sum((term_centers - cap_centers[i][None, :]) ** 2, axis=1)
            j = int(np.argmin(d2))
            cap_area_est[i] = term_areas[j]

        inlet_cap_idx = int(np.argmax(cap_area_est))
        source_ids.InsertNextId(inlet_cap_idx)

        # Targets: all other caps above tiny relative area
        max_cap_area = float(np.max(cap_area_est)) if cap_area_est.size > 0 else 0.0
        for i in range(cap_centers.shape[0]):
            if i == inlet_cap_idx:
                continue
            if max_cap_area > 0 and cap_area_est[i] < 1e-4 * max_cap_area:
                continue
            target_ids.InsertNextId(i)

        if target_ids.GetNumberOfIds() == 0:
            # fall back to all other caps
            for i in range(cap_centers.shape[0]):
                if i != inlet_cap_idx:
                    target_ids.InsertNextId(i)

        info["inlet_seed"] = inlet_cap_idx
        info["n_targets"] = int(target_ids.GetNumberOfIds())

    else:
        # Closed surface or cap ids unavailable: use closest surface point ids to termination centers.
        info["used_cap_center_ids"] = False

        # Inlet point id
        inlet_pid = int(locator.FindClosestPoint(inlet_center))
        source_ids.InsertNextId(inlet_pid)
        info["inlet_seed"] = inlet_pid

        for t in terminations_use:
            c = np.array(t["center"], dtype=float)
            pid = int(locator.FindClosestPoint(c))
            if pid == inlet_pid:
                continue
            target_ids.InsertNextId(pid)

        # Ensure at least one target. If all collapse, take farthest point in set
        if target_ids.GetNumberOfIds() == 0:
            # choose a far point id: max distance from inlet among termination centers
            far_t = max(terminations_use, key=lambda d: float(np.linalg.norm(np.array(d["center"], float) - inlet_center)))
            far_pid = int(locator.FindClosestPoint(np.array(far_t["center"], float)))
            if far_pid != inlet_pid:
                target_ids.InsertNextId(far_pid)

        info["n_targets"] = int(target_ids.GetNumberOfIds())

    if source_ids.GetNumberOfIds() == 0 or target_ids.GetNumberOfIds() == 0:
        add_warning(warnings, "W_VMTK_003", "Failed to construct VMTK seed id lists (source/target).")
        return None, info

    # Try centerline extraction with two normal-orientation options
    for flip in [0, 1]:
        try:
            cl_filter = vtkvmtk_mod.vtkvmtkPolyDataCenterlines()
            cl_filter.SetInputData(capped_surface)
            if info["used_cap_center_ids"] and cap_center_ids is not None:
                cl_filter.SetCapCenterIds(cap_center_ids)
            cl_filter.SetSourceSeedIds(source_ids)
            cl_filter.SetTargetSeedIds(target_ids)
            cl_filter.SetRadiusArrayName("MaximumInscribedSphereRadius")
            cl_filter.SetCostFunction("1/R")
            cl_filter.SetFlipNormals(int(flip))
            cl_filter.SetAppendEndPointsToCenterlines(1)
            cl_filter.SetCenterlineResampling(1)
            cl_filter.SetResamplingStepLength(1.0)
            cl_filter.Update()
            out = cl_filter.GetOutput()
            if out is None or out.GetNumberOfPoints() < 2 or out.GetNumberOfCells() < 1:
                raise RuntimeError("vtkvmtkPolyDataCenterlines returned empty output.")
            out = extract_largest_connected_region_lines(out)
            n_pts, n_cells, n_lines = polydata_line_stats(out)
            if n_pts < 2 or n_cells < 1 or n_lines < 1:
                raise RuntimeError(
                    "Centerlines lost line structure after post-processing: "
                    f"points={n_pts}, cells={n_cells}, line_cells={n_lines}"
                )
            info["used_flip_normals"] = int(flip)
            info["centerline_points"] = n_pts
            info["centerline_cells"] = n_cells
            info["centerline_line_cells"] = n_lines
            return out, info
        except Exception as e:
            add_warning(warnings, "W_VMTK_004", f"Centerline extraction attempt flip_normals={flip} failed: {e}")

    return None, info


# ----------------------------
# Graph construction on centerlines
# ----------------------------

def build_graph_from_centerlines(cl: vtkPolyData) -> Tuple[Dict[int, Dict[int, float]], np.ndarray]:
    """
    Build an undirected weighted graph from polyline cells.
    Returns adjacency dict: {i: {j: w}}, and points array (N,3).
    """
    n_pts, n_cells, n_line_cells = polydata_line_stats(cl)
    pts = get_points_numpy(cl)
    adjacency: Dict[int, Dict[int, float]] = {i: {} for i in range(pts.shape[0])}

    for ci in range(cl.GetNumberOfCells()):
        cell = cl.GetCell(ci)
        if cell is None:
            continue
        nids = cell.GetNumberOfPoints()
        if nids < 2:
            continue
        ids = [cell.GetPointId(k) for k in range(nids)]
        for a, b in zip(ids[:-1], ids[1:]):
            if a == b:
                continue
            w = float(np.linalg.norm(pts[a] - pts[b]))
            if w <= 0:
                continue
            # keep minimum weight if parallel edges appear
            if b not in adjacency[a] or w < adjacency[a][b]:
                adjacency[a][b] = w
                adjacency[b][a] = w
    # Remove isolated nodes
    adjacency = {k: v for k, v in adjacency.items() if len(v) > 0}
    if n_line_cells == 0 or not adjacency:
        print(
            "DEBUG centerlines graph input: "
            f"points={n_pts} cells={n_cells} line_cells={n_line_cells} graph_nodes={len(adjacency)}",
            file=sys.stderr,
        )
    return adjacency, pts


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


def shortest_path(adjacency: Dict[int, Dict[int, float]], start: int, goal: int) -> Tuple[List[int], float]:
    import heapq
    heap: List[Tuple[float, int]] = [(0.0, start)]
    dist: Dict[int, float] = {start: 0.0}
    prev: Dict[int, int] = {}
    visited = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            break
        for v, w in adjacency.get(u, {}).items():
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if goal not in dist:
        return [], float("inf")
    # reconstruct
    path = [goal]
    cur = goal
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path, dist[goal]


def path_common_node_with_max_distance(path1: List[int], path2: List[int], dist_from_root: Dict[int, float]) -> Optional[int]:
    s1 = set(path1)
    common = [n for n in path2 if n in s1]
    if not common:
        return None
    common_sorted = sorted(common, key=lambda n: dist_from_root.get(n, -1.0))
    return common_sorted[-1]


def interpolate_along_polyline(points: np.ndarray, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given ordered polyline points, interpolate point and tangent at arc-length s from start.
    Returns (p, tangent_unit).
    """
    if points.shape[0] < 2:
        return (points[0] if points.shape[0] == 1 else np.zeros(3)), np.array([0.0, 0.0, 1.0])
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_clamped = float(clamp(s, 0.0, float(cum[-1])))
    idx = int(np.searchsorted(cum, s_clamped, side="right") - 1)
    idx = max(0, min(idx, len(seg_len) - 1))
    ds = s_clamped - cum[idx]
    if seg_len[idx] < 1e-12:
        p = points[idx].copy()
        t = unit(seg[idx])
        return p, t
    alpha = ds / seg_len[idx]
    p = points[idx] * (1.0 - alpha) + points[idx + 1] * alpha
    # tangent from local segment or neighbor segments if available
    if 1 <= idx < len(seg) - 1:
        t = unit(points[idx + 1] - points[idx - 1])
    else:
        t = unit(points[idx + 1] - points[idx])
    return p.astype(float), t.astype(float)


# ----------------------------
# Cross-section diameter computation (area-equivalent)
# ----------------------------

def cross_section_area_vtk(surface: vtkPolyData, origin: np.ndarray, normal: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Compute cross-section area by cutting the input surface with a plane (origin, normal),
    triangulating the resulting contours, and extracting the connected region closest to origin.
    Returns (area, info).
    """
    info: Dict[str, Any] = dict(n_contour_cells=0, used_fallback=False)
    n = unit(np.array(normal, dtype=float))
    if np.linalg.norm(n) < 1e-12:
        return NAN, info

    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(n[0]), float(n[1]), float(n[2]))

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(surface)
    cutter.GenerateCutScalarsOff()
    cutter.SetNumberOfContours(1)
    cutter.SetValue(0, 0.0)
    cutter.Update()
    cut = cutter.GetOutput()
    if cut is None or cut.GetNumberOfCells() == 0:
        return NAN, info
    info["n_contour_cells"] = int(cut.GetNumberOfCells())

    # Clean to merge nearly-identical points along the contour
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(cut)
    clean.PointMergingOn()
    clean.Update()

    # Triangulate contours
    triang = vtk.vtkContourTriangulator()
    triang.SetInputConnection(clean.GetOutputPort())
    triang.Update()
    tri_pd = triang.GetOutput()
    if tri_pd is None or tri_pd.GetNumberOfCells() == 0:
        # Fallback: attempt to build loops with stripper and compute planar area
        info["used_fallback"] = True
        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(clean.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        out = stripper.GetOutput()
        pts = out.GetPoints()
        if pts is None or out.GetNumberOfCells() == 0:
            return NAN, info
        areas = []
        for ci in range(out.GetNumberOfCells()):
            cell = out.GetCell(ci)
            if cell is None:
                continue
            nids = cell.GetNumberOfPoints()
            if nids < 3:
                continue
            coords = np.array([pts.GetPoint(cell.GetPointId(i)) for i in range(nids)], dtype=float)
            area, _, _ = planar_polygon_area_and_normal(coords)
            areas.append(area)
        if not areas:
            return NAN, info
        return float(max(areas)), info

    # Extract the region closest to origin
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(tri_pd)
    conn.SetExtractionModeToClosestPointRegion()
    conn.SetClosestPoint(float(origin[0]), float(origin[1]), float(origin[2]))
    conn.Update()
    region = conn.GetOutput()
    if region is None or region.GetNumberOfCells() == 0:
        return NAN, info

    # Compute area by summing triangle areas (ensure triangulated)
    tri2 = vtk.vtkTriangleFilter()
    tri2.SetInputData(region)
    tri2.Update()
    region_tri = tri2.GetOutput()
    pts = region_tri.GetPoints()
    if pts is None:
        return NAN, info

    area_sum = 0.0
    for ci in range(region_tri.GetNumberOfCells()):
        cell = region_tri.GetCell(ci)
        if cell is None:
            continue
        if cell.GetNumberOfPoints() != 3:
            continue
        p0 = np.array(pts.GetPoint(cell.GetPointId(0)), dtype=float)
        p1 = np.array(pts.GetPoint(cell.GetPointId(1)), dtype=float)
        p2 = np.array(pts.GetPoint(cell.GetPointId(2)), dtype=float)
        area_sum += 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
    return float(area_sum), info


def cross_section_diameter_eq(surface: vtkPolyData, origin: np.ndarray, tangent: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns (D_eq_mm, area_mm2, info)
    """
    area, info = cross_section_area_vtk(surface, origin, tangent)
    if not math.isfinite(area) or area <= 0:
        return NAN, NAN, info
    d = math.sqrt(4.0 * area / math.pi)
    return float(d), float(area), info


# ----------------------------
# Profile smoothing and aneurysm start detection
# ----------------------------

def smooth_1d(y: np.ndarray, window_mm: float, step_mm: float) -> np.ndarray:
    """
    Simple moving-average smoothing without SciPy, using window length in mm.
    """
    if y.size == 0:
        return y
    if not math.isfinite(window_mm) or window_mm <= 0 or not math.isfinite(step_mm) or step_mm <= 0:
        return y.copy()
    w = int(max(3, round(window_mm / step_mm)))
    if w % 2 == 0:
        w += 1
    if w >= y.size:
        return y.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    # pad by reflection
    pad = w // 2
    ypad = np.pad(y, pad_width=pad, mode="reflect")
    ys = np.convolve(ypad, kernel, mode="valid")
    return ys.astype(float)


def detect_aneurysm_start_distance(s_rel: np.ndarray, D: np.ndarray, step_mm: float) -> Tuple[float, float]:
    """
    Detect aneurysm start using derivative/persistence logic on smoothed diameter profile.

    Returns (aneurysm_start_mm, confidence).
    If not detected, returns (NaN, 0).
    """
    if s_rel.size < 10 or D.size != s_rel.size:
        return NAN, 0.0
    # Remove NaNs
    mask = np.isfinite(D) & np.isfinite(s_rel)
    if np.count_nonzero(mask) < 10:
        return NAN, 0.0
    s = s_rel[mask]
    d = D[mask]
    # Sort by s to be safe
    order = np.argsort(s)
    s = s[order]
    d = d[order]

    # Smooth (5 mm window)
    ds = float(step_mm) if step_mm > 0 else float(np.median(np.diff(s)))
    d_s = smooth_1d(d, window_mm=5.0, step_mm=ds)

    # Derivative
    dd = np.gradient(d_s, s)

    # Baseline noise from first 15 mm (or first quarter)
    baseline_end = min(float(15.0), float(0.25 * (s[-1] - s[0])))
    base_mask = s <= (s[0] + baseline_end)
    if np.count_nonzero(base_mask) < 5:
        base_mask = np.arange(len(s)) < max(5, len(s) // 5)
    base_dd = dd[base_mask]
    med = float(np.median(base_dd))
    mad = float(np.median(np.abs(base_dd - med))) + 1e-8
    # Threshold: require slope greater than baseline + 3*MAD and also positive
    thr = max(0.02, med + 3.0 * mad)

    # Persistence criteria
    persistence_mm = 5.0
    persistence_n = int(max(3, round(persistence_mm / ds)))
    increase_mm = 2.0  # secondary check
    increase_frac = 0.08

    n = len(s)
    for i in range(0, n - persistence_n - 1):
        if dd[i] <= thr:
            continue
        j = i + persistence_n
        dd_seg = dd[i:j]
        if np.mean(dd_seg) <= max(thr * 0.6, 0.01):
            continue
        delta = float(d_s[j] - d_s[i])
        if delta < increase_mm and delta < increase_frac * float(max(d_s[i], 1e-6)):
            continue
        # sustained widening: require most of dd_seg positive
        if np.count_nonzero(dd_seg > 0) < int(0.7 * len(dd_seg)):
            continue
        # found
        conf = float(clamp((np.mean(dd_seg) - thr) / (abs(thr) + 0.05) + 0.5, 0.0, 1.0))
        return float(s[i]), conf

    return NAN, 0.0


# ----------------------------
# Anatomy inference and measurements
# ----------------------------

def pick_inlet_node(
    endpoints: List[int],
    pts: np.ndarray,
    terminations: List[Dict[str, Any]],
    warnings: List[str],
) -> Tuple[int, float, int]:
    """
    Choose inlet centerline endpoint node using termination candidate with maximum area.
    Returns (inlet_node, confidence, warn_flag)
    """
    if not endpoints:
        add_warning(warnings, "W_INLET_001", "No centerline endpoints detected.")
        return -1, 0.0, 1

    if terminations:
        inlet_term = max(terminations, key=lambda d: float(d.get("area", 0.0)))
        c = np.array(inlet_term["center"], dtype=float)
        dists = np.array([np.linalg.norm(pts[e] - c) for e in endpoints], dtype=float)
        idx = int(np.argmin(dists))
        inlet = int(endpoints[idx])
        # confidence: distance relative to median termination diameter
        deqs = np.array([float(t.get("diameter_eq", 0.0)) for t in terminations], dtype=float)
        med_d = float(np.median(deqs[deqs > 0])) if np.any(deqs > 0) else 20.0
        conf = float(clamp(1.0 - (dists[idx] / (med_d + 1e-6)), 0.0, 1.0))
        warn = 0
        if dists[idx] > 0.5 * med_d:
            warn = 1
            add_warning(warnings, "W_INLET_002", f"Inlet endpoint is far from largest termination center (dist={dists[idx]:.2f}mm).")
        return inlet, conf, warn

    # Fallback: choose endpoint with maximum Z coordinate (assume superior)
    z = pts[endpoints, 2]
    inlet = int(endpoints[int(np.argmax(z))])
    add_warning(warnings, "W_INLET_003", "No termination candidates; inlet chosen as endpoint with maximum Z (orientation may be wrong).")
    return inlet, 0.3, 1


def infer_provisional_aortic_bifurcation_and_iliac_subtrees(
    adjacency: Dict[int, Dict[int, float]],
    endpoints: List[int],
    inlet_node: int,
    pts: np.ndarray,
    dist_from_inlet: Dict[int, float],
) -> Dict[str, Any]:
    """
    Infer a provisional aortic bifurcation and the two downstream iliac subtrees
    using graph topology rather than only farthest-endpoint geometry.
    """
    info: Dict[str, Any] = dict(
        bif_node=None,
        groups=[],
        confidence=0.0,
        warn=1,
        pair_endpoints=(),
    )

    distal_endpoints = [e for e in endpoints if e != inlet_node and e in dist_from_inlet]
    if len(distal_endpoints) < 2:
        return info

    candidate_endpoints = sorted(
        distal_endpoints,
        key=lambda n: dist_from_inlet.get(n, -1.0),
        reverse=True,
    )[: min(10, len(distal_endpoints))]

    inlet_paths: Dict[int, List[int]] = {}
    for ep in candidate_endpoints:
        path_ie, _ = shortest_path(adjacency, inlet_node, ep)
        if path_ie:
            inlet_paths[ep] = path_ie
    if len(inlet_paths) < 2:
        return info

    max_root_dist = max(float(dist_from_inlet.get(ep, 0.0)) for ep in inlet_paths)
    best_pair: Optional[Tuple[int, int, int, int, int]] = None
    best_metrics: Dict[str, float] = {}
    best_score = -1e18

    for i, a in enumerate(candidate_endpoints):
        if a not in inlet_paths:
            continue
        for b in candidate_endpoints[i + 1 :]:
            if b not in inlet_paths:
                continue
            bif_node = path_common_node_with_max_distance(inlet_paths[a], inlet_paths[b], dist_from_inlet)
            if bif_node is None or bif_node == inlet_node:
                continue

            path_bif_a, len_a = shortest_path(adjacency, bif_node, a)
            path_bif_b, len_b = shortest_path(adjacency, bif_node, b)
            if len(path_bif_a) < 2 or len(path_bif_b) < 2:
                continue
            child_a = int(path_bif_a[1])
            child_b = int(path_bif_b[1])
            if child_a == child_b:
                continue

            va = unit(pts[a] - pts[bif_node])
            vb = unit(pts[b] - pts[bif_node])
            divergence = float(clamp((1.0 - float(np.dot(va, vb))) / 2.0, 0.0, 1.0))
            balance = float(clamp(1.0 - abs(float(len_a) - float(len_b)) / (float(len_a) + float(len_b) + 1e-6), 0.0, 1.0))
            common_frac = float(clamp(float(dist_from_inlet.get(bif_node, 0.0)) / (max_root_dist + 1e-6), 0.0, 1.0))

            score = 25.0 * divergence + 15.0 * balance + 20.0 * common_frac + 0.30 * (float(len_a) + float(len_b))
            if score > best_score:
                best_score = score
                best_pair = (int(bif_node), child_a, child_b, int(a), int(b))
                best_metrics = dict(
                    divergence=divergence,
                    balance=balance,
                    common_frac=common_frac,
                )

    if best_pair is None:
        return info

    bif_node, child_a, child_b, seed_a, seed_b = best_pair
    child_entries: Dict[int, List[Tuple[int, float, List[int]]]] = {child_a: [], child_b: []}
    child_nodes: Dict[int, set[int]] = {child_a: {bif_node}, child_b: {bif_node}}

    for ep in distal_endpoints:
        path_bif_ep, len_bif_ep = shortest_path(adjacency, bif_node, ep)
        if len(path_bif_ep) < 2:
            continue
        first_hop = int(path_bif_ep[1])
        if first_hop not in child_entries:
            continue
        child_entries[first_hop].append((int(ep), float(len_bif_ep), path_bif_ep))
        child_nodes[first_hop].update(int(n) for n in path_bif_ep)

    if not child_entries[child_a]:
        path_seed_a, len_seed_a = shortest_path(adjacency, bif_node, seed_a)
        child_entries[child_a] = [(seed_a, float(len_seed_a), path_seed_a)]
        child_nodes[child_a].update(int(n) for n in path_seed_a)
    if not child_entries[child_b]:
        path_seed_b, len_seed_b = shortest_path(adjacency, bif_node, seed_b)
        child_entries[child_b] = [(seed_b, float(len_seed_b), path_seed_b)]
        child_nodes[child_b].update(int(n) for n in path_seed_b)

    groups: List[Dict[str, Any]] = []
    for child in (child_a, child_b):
        entries = child_entries.get(child, [])
        if not entries:
            continue
        representative_ep = max(entries, key=lambda item: item[1])[0]
        representative_len = max(float(item[1]) for item in entries)
        node_ids = sorted(child_nodes.get(child, {bif_node}))
        centroid_ids = np.array(node_ids, dtype=int)
        centroid = np.mean(pts[centroid_ids, :], axis=0) if centroid_ids.size > 0 else np.array(pts[bif_node], dtype=float)
        groups.append(
            dict(
                child=int(child),
                endpoints=sorted({int(item[0]) for item in entries}),
                node_ids=node_ids,
                centroid=np.array(centroid, dtype=float),
                representative_endpoint=int(representative_ep),
                representative_length=float(representative_len),
            )
        )

    if len(groups) != 2:
        return info

    n0 = len(groups[0]["endpoints"])
    n1 = len(groups[1]["endpoints"])
    subtree_balance = float(clamp(min(n0, n1) / max(n0, n1, 1), 0.0, 1.0))
    conf = float(
        clamp(
            0.25
            + 0.30 * best_metrics.get("divergence", 0.0)
            + 0.25 * best_metrics.get("balance", 0.0)
            + 0.20 * best_metrics.get("common_frac", 0.0)
            + 0.20 * subtree_balance,
            0.0,
            1.0,
        )
    )

    info.update(
        bif_node=int(bif_node),
        groups=groups,
        confidence=conf,
        warn=int(conf < 0.55),
        pair_endpoints=(seed_a, seed_b),
    )
    return info


def build_provisional_si_lr_frame(
    cl_pts: np.ndarray,
    inlet_node: int,
    bif_node: int,
    iliac_groups: List[Dict[str, Any]],
    warnings: List[str],
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Stage 1/2 frame: +Z superior (bifurcation -> inlet), provisional X from the
    two iliac subtrees, origin at the aortic bifurcation.
    """
    if inlet_node < 0 or bif_node < 0 or len(iliac_groups) < 2:
        add_warning(warnings, "W_FRAME_003", "Insufficient iliac topology for anatomy-informed frame; using generic frame fallback.")
        reps = [int(g.get("representative_endpoint")) for g in iliac_groups if g.get("representative_endpoint") is not None]
        return canonical_transform_from_centerlines_and_terminations(cl_pts, inlet_node, reps, warnings)

    origin = np.array(cl_pts[bif_node], dtype=float)
    si = unit(np.array(cl_pts[inlet_node], dtype=float) - origin)
    if np.linalg.norm(si) < 1e-8:
        add_warning(warnings, "W_FRAME_003", "Could not stabilize superior/inferior axis from inlet and bifurcation; using generic frame fallback.")
        reps = [int(g.get("representative_endpoint")) for g in iliac_groups if g.get("representative_endpoint") is not None]
        return canonical_transform_from_centerlines_and_terminations(cl_pts, inlet_node, reps, warnings)

    centroid_a = np.array(iliac_groups[0]["centroid"], dtype=float)
    centroid_b = np.array(iliac_groups[1]["centroid"], dtype=float)
    lr_vec = centroid_a - centroid_b
    lr_vec = lr_vec - float(np.dot(lr_vec, si)) * si

    if np.linalg.norm(lr_vec) < 1e-8:
        rep_a = iliac_groups[0].get("representative_endpoint")
        rep_b = iliac_groups[1].get("representative_endpoint")
        if rep_a is not None and rep_b is not None:
            lr_vec = np.array(cl_pts[int(rep_a)] - cl_pts[int(rep_b)], dtype=float)
            lr_vec = lr_vec - float(np.dot(lr_vec, si)) * si

    if np.linalg.norm(lr_vec) < 1e-8:
        add_warning(warnings, "W_FRAME_003", "Could not stabilize iliac left/right axis from subtree geometry; using generic frame fallback.")
        reps = [int(g.get("representative_endpoint")) for g in iliac_groups if g.get("representative_endpoint") is not None]
        return canonical_transform_from_centerlines_and_terminations(cl_pts, inlet_node, reps, warnings)

    lr = unit(lr_vec)
    ap = unit(np.cross(si, lr))
    if np.linalg.norm(ap) < 1e-8:
        add_warning(warnings, "W_FRAME_003", "Could not construct provisional AP axis from iliac scaffold; using generic frame fallback.")
        reps = [int(g.get("representative_endpoint")) for g in iliac_groups if g.get("representative_endpoint") is not None]
        return canonical_transform_from_centerlines_and_terminations(cl_pts, inlet_node, reps, warnings)

    lr = unit(np.cross(ap, si))
    R = np.vstack([lr, ap, si])
    t = -R @ origin

    group_sizes = [max(1, len(g.get("endpoints", []))) for g in iliac_groups[:2]]
    group_balance = float(clamp(min(group_sizes) / max(group_sizes), 0.0, 1.0))
    sep = float(np.linalg.norm(lr_vec))
    conf = float(
        clamp(
            0.35
            + 0.25 * clamp(sep / 40.0, 0.0, 1.0)
            + 0.20 * group_balance
            + 0.20 * clamp(min(group_sizes) / 2.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    return R, t, conf, 0


def resolve_ap_sign_from_ventral_branches(
    adjacency: Dict[int, Dict[int, float]],
    pts_provisional: np.ndarray,
    endpoints: List[int],
    inlet_node: int,
    bif_node: int,
    trunk_nodes: List[int],
    iliac_groups: List[Dict[str, Any]],
    dist_from_inlet: Dict[int, float],
    warnings: List[str],
) -> Tuple[int, float, int]:
    """
    Resolve AP sign from non-iliac side branches off the trunk in the provisional
    frame. Prefers ventral visceral-style branches and suppresses paired lateral
    renal-like branches.
    """
    trunk_set = set(int(n) for n in trunk_nodes)
    if len(trunk_nodes) < 2:
        add_warning(warnings, "W_AP_001", "Insufficient trunk path to resolve anterior/posterior sign; keeping provisional AP axis.")
        return 1, 0.0, 1

    iliac_endpoint_set: set[int] = set()
    for group in iliac_groups:
        iliac_endpoint_set.update(int(ep) for ep in group.get("endpoints", []))

    bif_dist = float(dist_from_inlet.get(bif_node, 0.0))
    raw_candidates: List[Dict[str, Any]] = []
    for ep in endpoints:
        if ep == inlet_node or ep in iliac_endpoint_set or ep not in dist_from_inlet:
            continue

        inlet_path, _ = shortest_path(adjacency, inlet_node, ep)
        if not inlet_path:
            continue
        common = [int(n) for n in inlet_path if n in trunk_set]
        if not common:
            continue
        takeoff = max(common, key=lambda n: dist_from_inlet.get(n, -1.0))
        takeoff_dist = float(dist_from_inlet.get(takeoff, NAN))
        if not math.isfinite(takeoff_dist):
            continue
        if takeoff_dist >= bif_dist - 5.0:
            continue

        branch_nodes, branch_len = shortest_path(adjacency, takeoff, ep)
        if len(branch_nodes) < 2 or not math.isfinite(branch_len) or branch_len < 2.0:
            continue

        branch_pts = pts_provisional[np.array(branch_nodes, dtype=int), :]
        sample_len = min(12.0, max(3.0, 0.60 * float(branch_len)))
        sample_point, _ = interpolate_along_polyline(branch_pts, sample_len)
        vec = np.array(sample_point - branch_pts[0], dtype=float)
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm < 1e-8:
            continue
        vec_unit = vec / vec_norm

        raw_candidates.append(
            dict(
                endpoint=int(ep),
                takeoff=int(takeoff),
                takeoff_dist=takeoff_dist,
                vec=vec_unit.astype(float),
                lateral=float(abs(vec_unit[0])),
                apness=float(abs(vec_unit[1])),
                inferior=float(max(0.0, -float(vec_unit[2]))),
                branch_len=float(branch_len),
                x=float(sample_point[0]),
            )
        )

    renal_like: set[int] = set()
    for i in range(len(raw_candidates)):
        for j in range(i + 1, len(raw_candidates)):
            ci = raw_candidates[i]
            cj = raw_candidates[j]
            if ci["x"] * cj["x"] >= 0:
                continue
            if abs(float(ci["takeoff_dist"]) - float(cj["takeoff_dist"])) > 20.0:
                continue
            if float(ci["lateral"]) < 0.45 or float(cj["lateral"]) < 0.45:
                continue
            renal_like.add(i)
            renal_like.add(j)

    weighted_sum = np.zeros(3, dtype=float)
    total_weight = 0.0
    ventral_candidates = []
    for idx, cand in enumerate(raw_candidates):
        if idx in renal_like:
            continue
        if float(cand["apness"]) < 0.30:
            continue
        if float(cand["lateral"]) > 0.75 and float(cand["apness"]) < float(cand["lateral"]):
            continue
        if float(cand["inferior"]) > 0.80 and float(cand["apness"]) < 0.50:
            continue

        weight = (
            max(0.0, float(cand["apness"]) - 0.20)
            * (1.0 - 0.60 * float(cand["lateral"]))
            * min(1.0, float(cand["branch_len"]) / 12.0)
        )
        if weight <= 1e-6:
            continue

        weighted_sum += weight * np.array(cand["vec"], dtype=float)
        total_weight += weight
        ventral_candidates.append(cand)

    if not ventral_candidates or np.linalg.norm(weighted_sum) < 1e-8:
        add_warning(
            warnings,
            "W_AP_001",
            "No reliable ventral trunk branches were found to resolve anterior/posterior sign; keeping provisional AP axis.",
        )
        return 1, 0.0, 1

    mean_vec = unit(weighted_sum)
    ap_sign = 1 if float(mean_vec[1]) >= 0.0 else -1
    conf = float(
        clamp(
            0.20
            + 0.35 * abs(float(mean_vec[1]))
            + 0.25 * clamp(len(ventral_candidates) / 3.0, 0.0, 1.0)
            + 0.20 * clamp(total_weight, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    warn = int(conf < 0.55)
    if warn:
        add_warning(
            warnings,
            "W_AP_002",
            f"Ventral branch AP orientation is low-confidence (confidence={conf:.3f}, candidates={len(ventral_candidates)}); left/right may remain ambiguous.",
        )
    return ap_sign, conf, warn


def finalize_canonical_transform_from_provisional(
    R_provisional: np.ndarray,
    origin: np.ndarray,
    ap_sign: int,
) -> Tuple[np.ndarray, np.ndarray]:
    flip = np.eye(3, dtype=float)
    if ap_sign < 0:
        flip[0, 0] = -1.0
        flip[1, 1] = -1.0

    R_tmp = flip @ np.array(R_provisional, dtype=float)
    z_axis = unit(R_tmp[2])
    x_axis = unit(R_tmp[0] - float(np.dot(R_tmp[0], z_axis)) * z_axis)
    y_axis = unit(np.cross(z_axis, x_axis))
    x_axis = unit(np.cross(y_axis, z_axis))
    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ np.array(origin, dtype=float)
    return R, t


def choose_left_right_external_endpoints_from_iliac_subtrees(
    adjacency: Dict[int, Dict[int, float]],
    pts: np.ndarray,
    bif_node: int,
    iliac_groups: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int], List[int], List[int], float]:
    """
    Choose one distal continuation endpoint per iliac subtree, then assign right/left
    from the final X coordinate of the two subtree centroids.
    """
    if bif_node < 0 or len(iliac_groups) < 2:
        return None, None, [], [], 0.0

    bif_p = np.array(pts[bif_node], dtype=float)
    summaries: List[Dict[str, Any]] = []
    for group in iliac_groups[:2]:
        endpoints_group = sorted({int(ep) for ep in group.get("endpoints", [])})
        if not endpoints_group:
            continue

        best_ep: Optional[int] = None
        best_score = -1e18
        for ep in endpoints_group:
            _, branch_len = shortest_path(adjacency, bif_node, ep)
            if not math.isfinite(branch_len):
                continue
            inferior = float(bif_p[2] - pts[ep][2])
            lateral = float(abs(pts[ep][0] - bif_p[0]))
            score = float(branch_len) + 0.80 * inferior + 0.10 * lateral
            if score > best_score:
                best_score = score
                best_ep = int(ep)

        if best_ep is None:
            best_ep = int(group.get("representative_endpoint", endpoints_group[0]))
            best_score = 0.0

        node_ids = [int(n) for n in group.get("node_ids", [])] or endpoints_group
        centroid_x = float(np.mean(pts[np.array(node_ids, dtype=int), 0])) if node_ids else float(pts[best_ep][0])
        summaries.append(
            dict(
                endpoints=endpoints_group,
                external_endpoint=best_ep,
                external_score=float(best_score),
                centroid_x=centroid_x,
            )
        )

    if len(summaries) < 2:
        return None, None, [], [], 0.0

    if summaries[0]["centroid_x"] >= summaries[1]["centroid_x"]:
        right_info, left_info = summaries[0], summaries[1]
    else:
        right_info, left_info = summaries[1], summaries[0]

    x_sep = abs(float(right_info["centroid_x"]) - float(left_info["centroid_x"]))
    group_balance = float(
        clamp(
            min(len(right_info["endpoints"]), len(left_info["endpoints"]))
            / max(len(right_info["endpoints"]), len(left_info["endpoints"]), 1),
            0.0,
            1.0,
        )
    )
    conf = float(
        clamp(
            0.25
            + 0.35 * clamp(x_sep / 30.0, 0.0, 1.0)
            + 0.20 * group_balance
            + 0.20 * clamp(min(float(right_info["external_score"]), float(left_info["external_score"])) / 40.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    return (
        int(right_info["external_endpoint"]),
        int(left_info["external_endpoint"]),
        list(right_info["endpoints"]),
        list(left_info["endpoints"]),
        conf,
    )


def canonical_transform_from_centerlines_and_terminations(
    cl_pts: np.ndarray,
    inlet_node: int,
    distal_nodes_guess: List[int],
    warnings: List[str],
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Compute canonical frame (X=LR, Y=AP, Z=SI) using centerline points.
    Returns (R, t, confidence, warn_flag) for transform x' = R x + t.
    """
    warn = 0
    if cl_pts.shape[0] < 3 or inlet_node < 0:
        add_warning(warnings, "W_FRAME_001", "Insufficient data to compute canonical frame; using identity.")
        return np.eye(3), np.zeros(3), 0.0, 1

    origin = np.mean(cl_pts, axis=0)

    inlet_p = cl_pts[inlet_node]

    # Determine distal mean using provided distal nodes (e.g., iliac endpoints)
    if distal_nodes_guess:
        distal_p = np.mean(cl_pts[distal_nodes_guess, :], axis=0)
    else:
        # fallback: farthest points from inlet
        d2 = np.sum((cl_pts - inlet_p[None, :]) ** 2, axis=1)
        far_idx = int(np.argmax(d2))
        distal_p = cl_pts[far_idx]

    # Superior direction (Z) points from distal to inlet
    si = unit(inlet_p - distal_p)
    if np.linalg.norm(si) < 1e-8:
        # PCA fallback
        X = cl_pts - origin
        cov = (X.T @ X) / max(X.shape[0], 1)
        w, V = np.linalg.eigh(cov)
        si = unit(V[:, np.argmax(w)])
        warn = 1

    # Left-right axis (X): use spread of distal nodes in plane orthogonal to si
    lr = None
    if len(distal_nodes_guess) >= 2:
        # choose two most separated distal points
        cand = cl_pts[distal_nodes_guess, :]
        best = None
        best_d = -1.0
        for i in range(cand.shape[0]):
            for j in range(i + 1, cand.shape[0]):
                d = float(np.linalg.norm(cand[i] - cand[j]))
                if d > best_d:
                    best_d = d
                    best = (cand[i], cand[j])
        if best is not None:
            v = best[0] - best[1]
            v = v - np.dot(v, si) * si
            lr = unit(v)

    if lr is None or np.linalg.norm(lr) < 1e-8:
        # PCA in plane orthogonal to si
        X = cl_pts - origin
        # remove component along si
        Xp = X - (X @ si)[:, None] * si[None, :]
        cov = (Xp.T @ Xp) / max(Xp.shape[0], 1)
        w, V = np.linalg.eigh(cov)
        lr = unit(V[:, np.argmax(w)])
        warn = 1

    ap = unit(np.cross(si, lr))
    if np.linalg.norm(ap) < 1e-8:
        # fallback
        ap = unit(np.cross(lr, si))
        warn = 1

    # Re-orthonormalize
    lr = unit(lr - np.dot(lr, si) * si)
    ap = unit(np.cross(si, lr))

    # Build rotation that maps original -> canonical: [lr; ap; si] as rows
    R = np.vstack([lr, ap, si])
    t = -R @ origin

    # Confidence heuristic: based on stability of axes norms
    conf = float(1.0 - 0.5 * warn)
    return R, t, conf, warn


def identify_distal_endpoints_for_iliacs(
    endpoints: List[int],
    inlet_node: int,
    pts: np.ndarray,
    dist_from_inlet: Dict[int, float],
) -> List[int]:
    """
    Pick up to 4 distal endpoints likely belonging to iliac system (external/internal ends),
    using farthest-from-inlet heuristic.
    """
    cand = [e for e in endpoints if e != inlet_node and e in dist_from_inlet]
    if not cand:
        return []
    cand_sorted = sorted(cand, key=lambda n: dist_from_inlet.get(n, -1.0), reverse=True)
    return cand_sorted[: min(6, len(cand_sorted))]


def choose_left_right_external_endpoints(
    distal_candidates: List[int],
    pts: np.ndarray,
    dist_from_inlet: Dict[int, float],
    terminations: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Choose two opposite-side distal endpoints as (right_external, left_external) in the current coordinate frame.
    Right is defined as +X after canonical transform; left is -X.
    Returns (right_node, left_node, confidence)
    """
    if len(distal_candidates) < 2:
        return None, None, 0.0

    # Diameter estimates from termination candidates if possible: map by nearest termination center
    term_centers = np.array([np.array(t["center"], float) for t in terminations], dtype=float) if terminations else None
    term_deq = np.array([float(t.get("diameter_eq", 0.0)) for t in terminations], dtype=float) if terminations else None

    def endpoint_deq(e: int) -> float:
        if term_centers is None or term_deq is None or term_centers.shape[0] == 0:
            return 0.0
        d2 = np.sum((term_centers - pts[e][None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        return float(term_deq[j])

    best_pair = None
    best_score = -1e18
    cand = distal_candidates[: min(8, len(distal_candidates))]
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a = cand[i]
            b = cand[j]
            xa, xb = float(pts[a][0]), float(pts[b][0])
            # Want opposite sides; if not, penalize but still allow
            opp = 1.0 if xa * xb < 0 else 0.3
            score = 0.0
            score += opp * 30.0
            score += 0.8 * (dist_from_inlet.get(a, 0.0) + dist_from_inlet.get(b, 0.0))
            score += 1.0 * abs(xa - xb)
            score += 2.0 * (endpoint_deq(a) + endpoint_deq(b))
            score -= 0.5 * abs(dist_from_inlet.get(a, 0.0) - dist_from_inlet.get(b, 0.0))
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        return None, None, 0.0

    a, b = best_pair
    # Assign right as +X, left as -X in current frame
    if pts[a][0] >= pts[b][0]:
        right, left = a, b
    else:
        right, left = b, a

    # Confidence: based on opposite-side and separation
    opp = 1.0 if float(pts[right][0]) * float(pts[left][0]) < 0 else 0.4
    sep = float(abs(float(pts[right][0]) - float(pts[left][0])))
    conf = float(clamp(0.2 + 0.6 * opp + 0.2 * clamp(sep / 40.0, 0.0, 1.0), 0.0, 1.0))
    return right, left, conf


def find_common_iliac_bifurcation(
    adjacency: Dict[int, Dict[int, float]],
    aortic_bif_node: int,
    external_node: int,
    candidate_other_endpoints_same_side: List[int],
    pts: np.ndarray,
    dist_from_inlet: Dict[int, float],
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Attempt to find (iliac_bif_node, internal_endpoint_node, confidence) on a given side.
    Uses path overlap between external endpoint and candidate endpoints on same side.
    """
    if aortic_bif_node < 0 or external_node is None:
        return None, None, 0.0

    ext_path, _ = shortest_path(adjacency, aortic_bif_node, external_node)
    if not ext_path:
        return None, None, 0.0

    best = None
    best_common_idx = -1
    best_score = -1e18
    for ep in candidate_other_endpoints_same_side:
        if ep == external_node:
            continue
        pth, _ = shortest_path(adjacency, aortic_bif_node, ep)
        if not pth:
            continue
        # Find deepest common node along ext_path
        common_nodes = [n for n in ext_path if n in set(pth)]
        if not common_nodes:
            continue
        # index along ext_path
        common_indices = [ext_path.index(n) for n in common_nodes]
        common_idx = max(common_indices)
        # require that endpoint diverges after some length (to avoid picking external again)
        if common_idx < 3:
            continue
        # score by overlap length and endpoint distance
        score = 0.0
        score += 5.0 * common_idx
        score += 0.3 * dist_from_inlet.get(ep, 0.0)
        score -= 0.1 * abs(pts[ep][2] - pts[external_node][2])  # internal often less inferior
        if score > best_score:
            best_score = score
            best_common_idx = common_idx
            best = (ep, ext_path[common_idx])

    if best is None:
        return None, None, 0.0

    internal_ep, bif_node = best[0], best[1]
    # Confidence: based on overlap depth
    conf = float(clamp(0.3 + 0.05 * best_common_idx, 0.0, 1.0))
    return bif_node, internal_ep, conf


def compute_tortuosity_and_max_angulation(points: np.ndarray, step_mm: float = 1.0) -> Tuple[float, float]:
    """
    Compute tortuosity and max angulation (degrees) of a vessel centerline polyline.
    - Tortuosity = centerline length / straight distance
    - Max angulation = max angle between successive tangent vectors
    """
    if points.shape[0] < 2:
        return NAN, NAN
    L = polyline_length(points)
    chord = float(np.linalg.norm(points[-1] - points[0]))
    tort = float(L / chord) if chord > 1e-6 else NAN

    # Resample roughly at step_mm for stable tangent estimation
    Ltot = L
    if not math.isfinite(Ltot) or Ltot <= 0:
        return tort, NAN
    n_samples = int(max(5, round(Ltot / max(step_mm, 0.5))) + 1)
    s_vals = np.linspace(0.0, Ltot, n_samples)
    res = np.array([interpolate_along_polyline(points, float(s))[0] for s in s_vals], dtype=float)
    # Tangents
    tang = np.zeros_like(res)
    tang[1:-1] = res[2:] - res[:-2]
    tang[0] = res[1] - res[0]
    tang[-1] = res[-1] - res[-2]
    tang = np.array([unit(t) for t in tang], dtype=float)

    angles = []
    for i in range(tang.shape[0] - 1):
        a = tang[i]
        b = tang[i + 1]
        dot = float(clamp(float(np.dot(a, b)), -1.0, 1.0))
        ang = math.degrees(math.acos(dot))
        angles.append(ang)
    max_ang = float(np.max(angles)) if angles else NAN
    return tort, max_ang


def compute_diameter_profile_along_path(
    surface: vtkPolyData,
    path_points: np.ndarray,
    step_mm: float,
    s_start: float,
    s_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample D_eq along a path from s_start to s_end at given step.
    Returns (s_samples, D_samples).
    """
    if path_points.shape[0] < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    L = polyline_length(path_points)
    if not math.isfinite(L) or L <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    s0 = clamp(float(s_start), 0.0, L)
    s1 = clamp(float(s_end), 0.0, L)
    if s1 < s0:
        s0, s1 = s1, s0
    if s1 - s0 < 1e-6:
        s_samples = np.array([s0], dtype=float)
    else:
        n = int(max(2, math.floor((s1 - s0) / max(step_mm, 0.5)) + 1))
        s_samples = np.linspace(s0, s1, n)
    D = np.zeros_like(s_samples, dtype=float)
    for i, s in enumerate(s_samples):
        p, t = interpolate_along_polyline(path_points, float(s))
        d, _, _ = cross_section_diameter_eq(surface, p, t)
        D[i] = d
    return s_samples, D


def compute_measurements(
    surface: vtkPolyData,
    centerlines: vtkPolyData,
    terminations: List[Dict[str, Any]],
    results: Dict[str, Any],
    warnings: List[str],
) -> None:
    """
    Main measurement routine.
    """
    # Build graph
    n_cl_points, n_cl_cells, n_cl_lines = polydata_line_stats(centerlines)
    adjacency, pts = build_graph_from_centerlines(centerlines)
    if not adjacency:
        add_warning(
            warnings,
            "W_GRAPH_001",
            "Centerline graph construction failed (empty adjacency). "
            f"Centerline stats: points={n_cl_points}, cells={n_cl_cells}, line_cells={n_cl_lines}.",
        )
        results["warn_centerlines"] = 1
        return

    nodes = list(adjacency.keys())
    degrees = {n: len(adjacency[n]) for n in nodes}
    endpoints = [n for n, deg in degrees.items() if deg == 1]
    if not endpoints:
        # fallback: consider nodes with small degree
        endpoints = sorted(nodes, key=lambda n: degrees.get(n, 99))[:8]
        add_warning(warnings, "W_GRAPH_002", "No degree-1 endpoints on centerlines; using low-degree nodes as endpoints.")

    # Inlet identification (pre-canonical)
    inlet_node, inlet_conf, inlet_warn = pick_inlet_node(endpoints, pts, terminations, warnings)
    if inlet_node < 0:
        results["warn_centerlines"] = 1
        add_warning(warnings, "W_INLET_004", "Failed to identify inlet node on centerlines.")
        return
    results["conf_inlet_identification"] = inlet_conf
    if inlet_warn:
        results["warn_inlet_identification"] = 1

    # Distances from inlet
    dist_root, _ = dijkstra(adjacency, inlet_node)

    # Identify distal candidates for iliac endpoints and infer an anatomy-driven
    # iliac scaffold before building the canonical frame.
    distal_candidates = identify_distal_endpoints_for_iliacs(endpoints, inlet_node, pts, dist_root)
    semantic_endpoints_pre = map_semantic_terminations_to_centerline_endpoints(endpoints, pts, terminations)
    use_semantic_orientation = (
        semantic_endpoints_pre.get("ext_iliac_right") is not None
        and semantic_endpoints_pre.get("ext_iliac_left") is not None
        and min(
            float(semantic_endpoints_pre.get("ext_iliac_right_conf", 0.0)),
            float(semantic_endpoints_pre.get("ext_iliac_left_conf", 0.0)),
        ) >= 0.35
    )
    iliac_scaffold = infer_provisional_aortic_bifurcation_and_iliac_subtrees(adjacency, endpoints, inlet_node, pts, dist_root)
    use_anatomic_orientation = (
        iliac_scaffold.get("bif_node") is not None
        and len(iliac_scaffold.get("groups", [])) >= 2
    )

    provisional_bif_node: Optional[int] = int(iliac_scaffold["bif_node"]) if use_anatomic_orientation else None
    right_group_endpoints: List[int] = []
    left_group_endpoints: List[int] = []

    if use_semantic_orientation:
        seed_nodes = [
            int(semantic_endpoints_pre["ext_iliac_right"]),
            int(semantic_endpoints_pre["ext_iliac_left"]),
        ]
        R0, t0, base_frame_conf, base_frame_warn = canonical_transform_from_centerlines_and_terminations(
            pts,
            inlet_node,
            seed_nodes,
            warnings,
        )
        pts_tmp = (R0 @ pts.T).T + t0[None, :]
        terminations_tmp = transform_terminations(terminations, R0, t0)
        flip_needed, lr_conf, ap_conf, lr_warn, ap_warn = resolve_frame_sign_from_semantics(
            pts_tmp,
            terminations_tmp,
            semantic_endpoints_pre,
        )
        if flip_needed:
            flip = np.diag([-1.0, -1.0, 1.0])
            R = flip @ R0
            t = flip @ t0
        else:
            R = R0
            t = t0

        results["conf_iliac_lr_orientation"] = float(lr_conf)
        results["conf_ap_orientation"] = float(ap_conf)
        if lr_warn:
            results["warn_iliac_lr_orientation"] = 1
            add_warning(
                warnings,
                "W_ILIAC_ORIENT_001",
                "External iliac outlet labels were available but could not anchor left/right orientation confidently.",
            )
        if ap_warn:
            results["warn_ap_orientation"] = 1
            add_warning(
                warnings,
                "W_AP_003",
                "Semantic ventral outlet evidence was insufficient to lock anterior/posterior sign confidently; AP axis may be approximate.",
            )
        if lr_conf < 0.55:
            results["warn_left_right_mirror_ambiguity"] = 1
            add_warning(
                warnings,
                "W_LR_001",
                "Left/right labeling remains potentially mirrored because semantic iliac outlet matching was low-confidence.",
            )
        frame_conf = float(
            clamp(
                0.35 * base_frame_conf + 0.40 * lr_conf + 0.25 * max(ap_conf, lr_conf),
                0.0,
                1.0,
            )
        )
        frame_warn = int(bool(base_frame_warn and lr_conf < 0.55 and ap_conf < 0.55))
    elif use_anatomic_orientation and provisional_bif_node is not None:
        results["conf_iliac_lr_orientation"] = float(iliac_scaffold.get("confidence", 0.0))
        if iliac_scaffold.get("warn", 0):
            results["warn_iliac_lr_orientation"] = 1
            add_warning(
                warnings,
                "W_ILIAC_ORIENT_001",
                f"Provisional iliac subtree scaffold is low-confidence (confidence={float(iliac_scaffold.get('confidence', 0.0)):.3f}).",
            )

        trunk_nodes0, _ = shortest_path(adjacency, inlet_node, provisional_bif_node)
        R0, t0, si_lr_conf, si_lr_warn = build_provisional_si_lr_frame(
            pts,
            inlet_node,
            provisional_bif_node,
            list(iliac_scaffold.get("groups", [])),
            warnings,
        )
        pts_provisional = (R0 @ pts.T).T + t0[None, :]
        ap_sign, ap_conf, ap_warn = resolve_ap_sign_from_ventral_branches(
            adjacency,
            pts_provisional,
            endpoints,
            inlet_node,
            provisional_bif_node,
            trunk_nodes0,
            list(iliac_scaffold.get("groups", [])),
            dist_root,
            warnings,
        )
        R, t = finalize_canonical_transform_from_provisional(R0, pts[provisional_bif_node], ap_sign)
        frame_conf = float(clamp(0.65 * si_lr_conf + 0.35 * ap_conf, 0.0, 1.0))
        frame_warn = int(bool(si_lr_warn or ap_warn or iliac_scaffold.get("warn", 0)))
        results["conf_ap_orientation"] = ap_conf
        if ap_warn:
            results["warn_ap_orientation"] = 1
        if ap_conf < 0.55:
            results["warn_left_right_mirror_ambiguity"] = 1
            add_warning(
                warnings,
                "W_LR_001",
                "Left/right labeling remains potentially mirrored because ventral trunk branch orientation was low-confidence.",
            )
    else:
        results["warn_iliac_lr_orientation"] = 1
        results["warn_ap_orientation"] = 1
        results["conf_iliac_lr_orientation"] = 0.0
        results["conf_ap_orientation"] = 0.0
        add_warning(
            warnings,
            "W_ILIAC_ORIENT_001",
            "Could not infer a bilateral iliac scaffold anatomically; falling back to generic distal-endpoint orientation.",
        )
        R, t, frame_conf, frame_warn = canonical_transform_from_centerlines_and_terminations(
            pts, inlet_node, distal_candidates, warnings
        )
        results["warn_left_right_mirror_ambiguity"] = 1
        add_warning(warnings, "W_LR_001", "Left/right labeling may be mirrored because the input .vtp has no absolute patient orientation metadata.")

    results["Canonical_frame_confidence"] = frame_conf
    if frame_warn:
        add_warning(warnings, "W_FRAME_002", "Canonical frame may be unreliable; anatomy left/right may be ambiguous.")

    # Apply transform to surface and centerlines in-place (create transformed copies)
    surface_can = apply_linear_transform_to_polydata(surface, R, t)
    centerlines_can = apply_linear_transform_to_polydata(centerlines, R, t)

    # Transform termination centers to canonical frame for consistent spatial heuristics
    terminations_can = transform_terminations(terminations, R, t)

    # Update graph with canonical points (adjacency weights unchanged because rigid)
    # Recompute endpoints and distances to be safe
    n_can_points, n_can_cells, n_can_lines = polydata_line_stats(centerlines_can)
    adjacency_can, pts_can2 = build_graph_from_centerlines(centerlines_can)
    if not adjacency_can:
        add_warning(
            warnings,
            "W_GRAPH_003",
            "Centerline graph rebuild after canonical transform failed. "
            f"Centerline stats: points={n_can_points}, cells={n_can_cells}, line_cells={n_can_lines}.",
        )
        results["warn_centerlines"] = 1
        return
    adjacency = adjacency_can
    pts = pts_can2
    nodes = list(adjacency.keys())
    degrees = {n: len(adjacency[n]) for n in nodes}
    endpoints = [n for n, deg in degrees.items() if deg == 1] or endpoints
    dist_root, _ = dijkstra(adjacency, inlet_node)

    # Determine right/left distal external endpoints
    semantic_endpoints_can = map_semantic_terminations_to_centerline_endpoints(endpoints, pts, terminations_can)
    semantic_right_int_ep: Optional[int] = None
    semantic_left_int_ep: Optional[int] = None
    semantic_orientation_active = bool(use_semantic_orientation)
    if semantic_orientation_active:
        right_ext = semantic_endpoints_can.get("ext_iliac_right")
        left_ext = semantic_endpoints_can.get("ext_iliac_left")
        iliac_pair_conf = float(
            min(
                float(semantic_endpoints_can.get("ext_iliac_right_conf", 0.0)),
                float(semantic_endpoints_can.get("ext_iliac_left_conf", 0.0)),
            )
        )
        semantic_right_int_ep = semantic_endpoints_can.get("int_iliac_right")
        semantic_left_int_ep = semantic_endpoints_can.get("int_iliac_left")
        right_group_endpoints = [int(ep) for ep in (right_ext, semantic_right_int_ep) if ep is not None]
        left_group_endpoints = [int(ep) for ep in (left_ext, semantic_left_int_ep) if ep is not None]
        if right_ext is None or left_ext is None or iliac_pair_conf < 0.35:
            semantic_orientation_active = False
            results["warn_iliac_lr_orientation"] = 1
            add_warning(
                warnings,
                "W_ILIAC_ORIENT_002",
                "Semantic outlet labels were present, but endpoint rematching on the centerline graph failed; falling back to graph heuristics.",
            )

    if (not semantic_orientation_active) and use_anatomic_orientation and provisional_bif_node is not None:
        right_ext, left_ext, right_group_endpoints, left_group_endpoints, iliac_pair_conf = (
            choose_left_right_external_endpoints_from_iliac_subtrees(
                adjacency,
                pts,
                provisional_bif_node,
                list(iliac_scaffold.get("groups", [])),
            )
        )
        if right_ext is None or left_ext is None:
            results["warn_iliac_lr_orientation"] = 1
            add_warning(
                warnings,
                "W_ILIAC_ORIENT_002",
                "Iliac subtree endpoint selection failed after frame alignment; falling back to generic distal endpoint pairing.",
            )
            distal_candidates = identify_distal_endpoints_for_iliacs(endpoints, inlet_node, pts, dist_root)
            right_ext, left_ext, iliac_pair_conf = choose_left_right_external_endpoints(
                distal_candidates, pts, dist_root, terminations_can
            )
            right_group_endpoints = [e for e in endpoints if e not in {inlet_node, left_ext} and pts[e][0] >= 0]
            left_group_endpoints = [e for e in endpoints if e not in {inlet_node, right_ext} and pts[e][0] < 0]
    elif not semantic_orientation_active:
        distal_candidates = identify_distal_endpoints_for_iliacs(endpoints, inlet_node, pts, dist_root)
        right_ext, left_ext, iliac_pair_conf = choose_left_right_external_endpoints(distal_candidates, pts, dist_root, terminations_can)
        right_group_endpoints = [e for e in endpoints if e not in {inlet_node, left_ext} and pts[e][0] >= 0]
        left_group_endpoints = [e for e in endpoints if e not in {inlet_node, right_ext} and pts[e][0] < 0]

    if right_ext is None or left_ext is None:
        add_warning(warnings, "W_ILIAC_001", "Failed to identify bilateral distal iliac endpoints; cannot compute iliac-based metrics robustly.")
        results["warn_aortic_bifurcation"] = 1
        return
    results["conf_iliac_lr_orientation"] = float(
        clamp(
            0.50 * safe_float(results.get("conf_iliac_lr_orientation", 0.0)) + 0.50 * iliac_pair_conf,
            0.0,
            1.0,
        )
    )

    # Aortic bifurcation: common node on paths from inlet to bilateral distal endpoints
    path_r, _ = shortest_path(adjacency, inlet_node, right_ext)
    path_l, _ = shortest_path(adjacency, inlet_node, left_ext)
    if not path_r or not path_l:
        add_warning(warnings, "W_BIF_001", "Could not get paths from inlet to distal iliac endpoints.")
        results["warn_aortic_bifurcation"] = 1
        return
    dist_root2, _ = dijkstra(adjacency, inlet_node)
    bif_node = path_common_node_with_max_distance(path_r, path_l, dist_root2)
    if bif_node is None:
        if provisional_bif_node is not None:
            bif_node = provisional_bif_node
            results["warn_aortic_bifurcation"] = 1
            add_warning(
                warnings,
                "W_BIF_002",
                "Failed to recompute aortic bifurcation from bilateral external iliac paths; using provisional iliac-topology bifurcation.",
            )
        else:
            add_warning(warnings, "W_BIF_002", "Failed to find aortic bifurcation as common node of iliac paths.")
            results["warn_aortic_bifurcation"] = 1
            return
    results["conf_aortic_bifurcation"] = float(
        clamp(
            0.20
            + 0.45 * iliac_pair_conf
            + 0.35 * safe_float(results.get("conf_iliac_lr_orientation", 0.0)),
            0.0,
            1.0,
        )
    )

    # Trunk path: inlet -> bif
    trunk_nodes, trunk_len = shortest_path(adjacency, inlet_node, bif_node)
    if not trunk_nodes:
        add_warning(warnings, "W_TRUNK_001", "Failed to extract trunk path.")
        results["warn_aortic_bifurcation"] = 1
        return
    trunk_points = pts[np.array(trunk_nodes, dtype=int), :]

    # Identify candidate renal endpoints:
    excluded = {inlet_node, right_ext, left_ext}
    # Build same-side endpoint lists for internal bif detection
    if right_group_endpoints and left_group_endpoints:
        same_side_right = [e for e in right_group_endpoints if e not in excluded]
        same_side_left = [e for e in left_group_endpoints if e not in excluded]
    else:
        same_side_right = [e for e in endpoints if e not in excluded and pts[e][0] >= 0]
        same_side_left = [e for e in endpoints if e not in excluded and pts[e][0] < 0]

    # Find common iliac bifurcations and internal endpoints
    right_iliac_bif = None
    right_int_ep = semantic_right_int_ep
    right_ci_conf = 0.0
    if semantic_right_int_ep is not None:
        right_iliac_bif, right_ci_conf = find_shared_branch_bifurcation(
            adjacency,
            bif_node,
            right_ext,
            semantic_right_int_ep,
            dist_root2,
        )
    if right_iliac_bif is None:
        right_iliac_bif, right_int_ep, right_ci_conf = find_common_iliac_bifurcation(
            adjacency, bif_node, right_ext, same_side_right, pts, dist_root2
        )

    left_iliac_bif = None
    left_int_ep = semantic_left_int_ep
    left_ci_conf = 0.0
    if semantic_left_int_ep is not None:
        left_iliac_bif, left_ci_conf = find_shared_branch_bifurcation(
            adjacency,
            bif_node,
            left_ext,
            semantic_left_int_ep,
            dist_root2,
        )
    if left_iliac_bif is None:
        left_iliac_bif, left_int_ep, left_ci_conf = find_common_iliac_bifurcation(
            adjacency, bif_node, left_ext, same_side_left, pts, dist_root2
        )

    if right_iliac_bif is None:
        results["warn_right_common_iliac_bifurcation"] = 1
        add_warning(warnings, "W_CI_R_001", "Right common iliac bifurcation (into EIA/IIA) not found; using distal endpoint as surrogate.")
        right_iliac_bif = right_ext
    if left_iliac_bif is None:
        results["warn_left_common_iliac_bifurcation"] = 1
        add_warning(warnings, "W_CI_L_001", "Left common iliac bifurcation (into EIA/IIA) not found; using distal endpoint as surrogate.")
        left_iliac_bif = left_ext

    # Identify renal endpoints from remaining endpoints: must branch off trunk, be above bif (higher Z), and lateral
    trunk_set = set(trunk_nodes)
    renal_candidates = build_semantic_renal_candidates(
        adjacency,
        pts,
        inlet_node,
        bif_node,
        trunk_nodes,
        dist_root2,
        semantic_endpoints_can,
    )
    semantic_renal_endpoints = {int(rc["endpoint"]) for rc in renal_candidates}
    for e in endpoints:
        if e in semantic_renal_endpoints or e in excluded or e == right_int_ep or e == left_int_ep:
            continue
        # must not be too distal
        if dist_root2.get(e, 0.0) > dist_root2.get(bif_node, 0.0) + 5.0:
            continue
        # path from inlet to endpoint
        pth, _ = shortest_path(adjacency, inlet_node, e)
        if not pth:
            continue
        common = [n for n in pth if n in trunk_set]
        if not common:
            continue
        takeoff = max(common, key=lambda n: dist_root2.get(n, -1.0))
        takeoff_dist = dist_root2.get(takeoff, NAN)
        # should be above bif by at least 5mm along trunk
        if not math.isfinite(takeoff_dist):
            continue
        if takeoff_dist > dist_root2.get(bif_node, 0.0) - 5.0:
            continue
        # lateral criterion relative to iliac separation
        iliac_sep = float(abs(pts[right_ext][0] - pts[left_ext][0]))
        if iliac_sep > 1e-6 and abs(float(pts[e][0])) < 0.15 * iliac_sep:
            continue
        # direction mostly lateral (perpendicular to SI which is +Z in canonical)
        v = pts[e] - pts[takeoff]
        v_perp = v.copy()
        v_perp[2] = 0.0
        r = float(np.linalg.norm(v_perp) / (np.linalg.norm(v) + 1e-12))
        if r < 0.55:
            continue
        renal_candidates.append(dict(endpoint=e, takeoff=takeoff, takeoff_dist=takeoff_dist, x=float(pts[e][0]), z=float(pts[e][2]), r=r))

    # Pick renal pair (left/right) if available
    renals_left = [rc for rc in renal_candidates if rc["x"] < 0]
    renals_right = [rc for rc in renal_candidates if rc["x"] >= 0]
    chosen_renals = []
    if renals_left and renals_right:
        # pick pair with closest takeoff distance and maximal lateral magnitude
        best = None
        best_score = -1e18
        for rl in renals_left:
            for rr in renals_right:
                dt = abs(float(rl["takeoff_dist"]) - float(rr["takeoff_dist"]))
                lat = abs(float(rl["x"])) + abs(float(rr["x"]))
                score = 5.0 * lat - 2.0 * dt + 10.0 * (float(rl["r"]) + float(rr["r"]))
                if score > best_score:
                    best_score = score
                    best = (rl, rr)
        if best is not None:
            chosen_renals = [best[0], best[1]]
    else:
        # If only one side, take the most plausible
        if renal_candidates:
            chosen_renals = [max(renal_candidates, key=lambda rc: float(rc["r"]))]

    if not chosen_renals:
        results["warn_lowest_renal"] = 1
        results["conf_lowest_renal"] = 0.0
        add_warning(warnings, "W_RENAL_001", "Renal arteries not reliably identified; renal-referenced measurements will be NaN.")
        # Still compute Maximum_aneurysm_diameter best-effort along trunk (inlet->bif)
        s_trunk, D_trunk = compute_diameter_profile_along_path(surface_can, trunk_points, step_mm=2.0, s_start=0.0, s_end=trunk_len)
        if D_trunk.size > 0 and np.any(np.isfinite(D_trunk)):
            results["Maximum_aneurysm_diameter"] = float(np.nanmax(D_trunk))
        return

    add_warning(warnings, "W_RENAL_REF_001", "Lowest renal reference uses centerline branch takeoff point (approximation of inferior ostium edge).")

    # Lowest renal is the one with maximum takeoff_dist
    lowest_renal = max(chosen_renals, key=lambda rc: float(rc["takeoff_dist"]))
    s0 = float(lowest_renal["takeoff_dist"])
    lowest_takeoff_node = int(lowest_renal["takeoff"])
    results["conf_lowest_renal"] = float(clamp(0.3 + 0.7 * (len(chosen_renals) / 2.0), 0.0, 1.0))
    if len(chosen_renals) < 2:
        results["warn_lowest_renal"] = 1
        add_warning(warnings, "W_RENAL_002", "Only one renal branch identified; lowest renal reference may be uncertain.")

    # Trunk diameter profile from s0 to bif for aneurysm detection and D0/D5/D10/D15
    step = 1.0
    s_samples, D_samples = compute_diameter_profile_along_path(surface_can, trunk_points, step_mm=step, s_start=s0, s_end=trunk_len)
    if D_samples.size < 5 or not np.any(np.isfinite(D_samples)):
        add_warning(warnings, "W_DIAM_AO_001", "Failed to compute aortic trunk diameter profile; proximal neck and aneurysm metrics may be NaN.")
        results["warn_centerlines"] = 1
        return

    s_rel = s_samples - float(s_samples[0])  # s0-based
    D_s = D_samples.copy()

    # Neck diameters: use nearest samples to 0,5,10,15 mm
    def sample_at(mm: float) -> float:
        if s_rel.size == 0:
            return NAN
        idx = int(np.argmin(np.abs(s_rel - mm)))
        return float(D_s[idx]) if math.isfinite(float(D_s[idx])) else NAN

    results["Proximal_neck_D0"] = sample_at(0.0)
    results["Proximal_neck_D5"] = sample_at(5.0)
    results["Proximal_neck_D10"] = sample_at(10.0)
    results["Proximal_neck_D15"] = sample_at(15.0)

    # Detect aneurysm start
    aneurysm_start_rel, aneurysm_conf = detect_aneurysm_start_distance(s_rel, D_s, step_mm=step)
    if not math.isfinite(aneurysm_start_rel):
        add_warning(warnings, "W_ANEUR_001", "Aneurysm start not detected via derivative criteria; proximal neck length set to NaN and sac flags set to NaN.")
        results["Proximal_neck_length"] = NAN
        results["D0_within_sac"] = "NaN"
        results["D5_within_sac"] = "NaN"
        results["D10_within_sac"] = "NaN"
        results["D15_within_sac"] = "NaN"
        # Still compute maximum diameter on available trunk profile
        results["Maximum_aneurysm_diameter"] = float(np.nanmax(D_s)) if np.any(np.isfinite(D_s)) else NAN
    else:
        results["Proximal_neck_length"] = float(max(0.0, aneurysm_start_rel))
        # within-sac flags
        def within(mm: float) -> str:
            return "true" if aneurysm_start_rel <= mm + 1e-6 else "false"
        results["D0_within_sac"] = within(0.0)
        results["D5_within_sac"] = within(5.0)
        results["D10_within_sac"] = within(10.0)
        results["D15_within_sac"] = within(15.0)

        # Max aneurysm diameter in sac region [aneurysm_start_rel, end]
        sac_mask = s_rel >= aneurysm_start_rel
        if np.any(sac_mask) and np.any(np.isfinite(D_s[sac_mask])):
            results["Maximum_aneurysm_diameter"] = float(np.nanmax(D_s[sac_mask]))
        else:
            results["Maximum_aneurysm_diameter"] = float(np.nanmax(D_s)) if np.any(np.isfinite(D_s)) else NAN

    # Length lowest renal to aortic bifurcation along trunk:
    results["Length_lowest_renal_aortic_bifurcation"] = float(dist_root2.get(bif_node, NAN) - dist_root2.get(lowest_takeoff_node, NAN))

    # Define helper to compute common iliac and external iliac metrics for each side
    def vessel_metrics_side(
        external_ep: int,
        common_iliac_bif_node: int,
        side_tag: str,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # Common iliac: aortic bif -> common iliac bif (or endpoint if not found)
        ci_end = common_iliac_bif_node
        ci_path_nodes, ci_len = shortest_path(adjacency, bif_node, ci_end)
        if not ci_path_nodes:
            out["ci_len"] = NAN
            out["ci_D0"] = NAN
            out["ci_D10"] = NAN
            out["ci_D15"] = NAN
            out["ci_D20"] = NAN
        else:
            ci_pts = pts[np.array(ci_path_nodes, dtype=int), :]
            out["ci_len"] = float(ci_len)
            # Measure diameters at offsets; avoid exact bifurcation with small offset
            offsets = [0.0, 10.0, 15.0, 20.0]
            diams = {}
            for off in offsets:
                if ci_len < off + 1e-6:
                    diams[off] = NAN
                    continue
                eps = 1.0 if off == 0.0 else 0.0
                if off == 0.0:
                    add_warning(warnings, f"W_CI_{side_tag}_D0", "Common iliac D0 measured at 1mm distal to aortic bifurcation to avoid bifurcation plane ambiguity.")
                p, tan = interpolate_along_polyline(ci_pts, float(off + eps))
                d, _, _ = cross_section_diameter_eq(surface_can, p, tan)
                diams[off] = d
            out["ci_D0"] = float(diams[0.0])
            out["ci_D10"] = float(diams[10.0])
            out["ci_D15"] = float(diams[15.0])
            out["ci_D20"] = float(diams[20.0])

        # External iliac: common iliac bif -> external endpoint
        eia_start = common_iliac_bif_node
        eia_end = external_ep
        eia_path_nodes, _ = shortest_path(adjacency, eia_start, eia_end) if eia_start is not None else ([], float("inf"))
        if not eia_path_nodes or len(eia_path_nodes) < 2:
            # fallback: use path from aortic bif to external endpoint
            eia_path_nodes, _ = shortest_path(adjacency, bif_node, eia_end)
            out["eia_warn"] = 1
        else:
            out["eia_warn"] = 0

        if not eia_path_nodes or len(eia_path_nodes) < 2:
            out["eia_len"] = NAN
            out["eia_min"] = NAN
            out["eia_distal20_avg"] = NAN
            out["eia_tort"] = NAN
            out["eia_max_ang"] = NAN
            return out

        eia_pts = pts[np.array(eia_path_nodes, dtype=int), :]
        out["eia_len"] = float(polyline_length(eia_pts))

        tort, max_ang = compute_tortuosity_and_max_angulation(eia_pts, step_mm=1.0)
        out["eia_tort"] = tort
        out["eia_max_ang"] = max_ang

        # Diameter sampling along EIA (step 2mm for speed)
        L = float(polyline_length(eia_pts))
        if not math.isfinite(L) or L <= 1.0:
            out["eia_min"] = NAN
            out["eia_distal20_avg"] = NAN
            return out
        ss, DD = compute_diameter_profile_along_path(surface_can, eia_pts, step_mm=2.0, s_start=0.0, s_end=L)
        if DD.size == 0 or not np.any(np.isfinite(DD)):
            out["eia_min"] = NAN
            out["eia_distal20_avg"] = NAN
            return out
        out["eia_min"] = float(np.nanmin(DD))

        # Distal 20mm average
        if L < 20.0:
            # average over last 40% if too short
            start = 0.6 * L
        else:
            start = L - 20.0
        ss2, DD2 = compute_diameter_profile_along_path(surface_can, eia_pts, step_mm=2.0, s_start=start, s_end=L)
        out["eia_distal20_avg"] = float(np.nanmean(DD2)) if DD2.size > 0 and np.any(np.isfinite(DD2)) else NAN

        return out

    # Right side metrics
    right_metrics = vessel_metrics_side(right_ext, right_iliac_bif, "R")
    results["Right_common_iliac_length"] = safe_float(right_metrics.get("ci_len", NAN))
    results["Right_common_iliac_D0"] = safe_float(right_metrics.get("ci_D0", NAN))
    results["Right_common_iliac_D10"] = safe_float(right_metrics.get("ci_D10", NAN))
    results["Right_common_iliac_D15"] = safe_float(right_metrics.get("ci_D15", NAN))
    results["Right_common_iliac_D20"] = safe_float(right_metrics.get("ci_D20", NAN))

    results["Right_external_iliac_min_diameter"] = safe_float(right_metrics.get("eia_min", NAN))
    results["Right_external_iliac_distal20mm_avg_diameter"] = safe_float(right_metrics.get("eia_distal20_avg", NAN))
    results["Right_external_iliac_diameter"] = safe_float(right_metrics.get("eia_distal20_avg", NAN))
    results["Right_external_iliac_tortuosity"] = safe_float(right_metrics.get("eia_tort", NAN))
    results["Right_external_iliac_max_angulation_deg"] = safe_float(right_metrics.get("eia_max_ang", NAN))
    if right_metrics.get("eia_warn", 0):
        results["warn_right_external_iliac"] = 1
        add_warning(warnings, "W_EIA_R_001", "Right EIA start uncertain (CI bif not found); EIA metrics may include common iliac segment.")
    results["conf_right_external_iliac"] = float(clamp(0.3 + 0.7 * right_ci_conf, 0.0, 1.0))

    # Left side metrics
    left_metrics = vessel_metrics_side(left_ext, left_iliac_bif, "L")
    results["Left_common_iliac_length"] = safe_float(left_metrics.get("ci_len", NAN))
    results["Left_common_iliac_D0"] = safe_float(left_metrics.get("ci_D0", NAN))
    results["Left_common_iliac_D10"] = safe_float(left_metrics.get("ci_D10", NAN))
    results["Left_common_iliac_D15"] = safe_float(left_metrics.get("ci_D15", NAN))
    results["Left_common_iliac_D20"] = safe_float(left_metrics.get("ci_D20", NAN))

    results["Left_external_iliac_min_diameter"] = safe_float(left_metrics.get("eia_min", NAN))
    results["Left_external_iliac_distal20mm_avg_diameter"] = safe_float(left_metrics.get("eia_distal20_avg", NAN))
    results["Left_external_iliac_diameter"] = safe_float(left_metrics.get("eia_distal20_avg", NAN))
    results["Left_external_iliac_tortuosity"] = safe_float(left_metrics.get("eia_tort", NAN))
    results["Left_external_iliac_max_angulation_deg"] = safe_float(left_metrics.get("eia_max_ang", NAN))
    if left_metrics.get("eia_warn", 0):
        results["warn_left_external_iliac"] = 1
        add_warning(warnings, "W_EIA_L_001", "Left EIA start uncertain (CI bif not found); EIA metrics may include common iliac segment.")
    results["conf_left_external_iliac"] = float(clamp(0.3 + 0.7 * left_ci_conf, 0.0, 1.0))

    # Renal to iliac bif distances (centerline distance from lowest renal takeoff node)
    # Use Dijkstra from lowest_takeoff_node
    dist_lr, _ = dijkstra(adjacency, lowest_takeoff_node)
    results["Length_lowest_renal_iliac_bifurcation_right"] = safe_float(dist_lr.get(right_iliac_bif, NAN))
    results["Length_lowest_renal_iliac_bifurcation_left"] = safe_float(dist_lr.get(left_iliac_bif, NAN))


# ----------------------------
# Limited fallback when centerlines unavailable
# ----------------------------

def fallback_max_diameter_by_slicing(surface: vtkPolyData, terminations: List[Dict[str, Any]], warnings: List[str]) -> float:
    """
    Best-effort estimation of maximum equivalent diameter by slicing along principal axis and taking maximum
    of largest cross-section per slice.
    This does not require centerlines.
    """
    pts = get_points_numpy(surface)
    if pts.shape[0] < 3:
        return NAN
    c = np.mean(pts, axis=0)
    X = pts - c
    cov = (X.T @ X) / max(len(X), 1)
    w, V = np.linalg.eigh(cov)
    axis = unit(V[:, np.argmax(w)])

    # Choose slicing range using projections
    proj = X @ axis
    tmin, tmax = float(np.min(proj)), float(np.max(proj))
    if not math.isfinite(tmin) or not math.isfinite(tmax) or tmax - tmin < 1e-6:
        return NAN

    # sampling step: use rough of 2mm in mm-units
    n = int(max(30, round((tmax - tmin) / 2.0)))
    ts = np.linspace(tmin, tmax, n)
    max_d = NAN
    for tt in ts:
        origin = c + tt * axis
        # Use axis as normal
        area, _ = cross_section_area_vtk(surface, origin, axis)
        if math.isfinite(area) and area > 0:
            d = math.sqrt(4.0 * area / math.pi)
            if (not math.isfinite(max_d)) or d > max_d:
                max_d = d
    if not math.isfinite(max_d):
        add_warning(warnings, "W_FALLBACK_001", "Fallback slicing could not compute any valid cross-section areas.")
    else:
        add_warning(warnings, "W_FALLBACK_002", "Maximum_aneurysm_diameter computed via slicing fallback (no centerlines). Other metrics set to NaN.")
    return max_d


# ----------------------------
# Main entrypoint
# ----------------------------

def main() -> None:
    results = make_results_template()
    warnings: List[str] = []

    # Ensure output always written
    try:
        if not vtk_available():
            results["warn_centerlines"] = 1
            results["warn_scale_inference"] = 1
            add_warning(warnings, "E_VTK_001", f"VTK import failed: {_VTK_IMPORT_ERROR}")
            write_output_txt(OUTPUT_TXT_PATH, results, warnings)
            return

        if not os.path.exists(INPUT_VTP_PATH):
            results["warn_centerlines"] = 1
            add_warning(warnings, "E_IO_001", f"Input file does not exist: {INPUT_VTP_PATH}")
            write_output_txt(OUTPUT_TXT_PATH, results, warnings)
            return

        # Load and preprocess
        surface = load_vtp(INPUT_VTP_PATH)
        if surface is None or surface.GetNumberOfPoints() == 0:
            results["warn_centerlines"] = 1
            add_warning(warnings, "E_IO_002", "Loaded surface is empty or invalid.")
            write_output_txt(OUTPUT_TXT_PATH, results, warnings)
            return

        surface_tri = clean_and_triangulate(surface)

        # Detect terminations and mode
        terminations, mode, face_array = detect_terminations_and_mode(surface_tri, warnings)
        annotate_terminations_with_semantic_labels(terminations, INPUT_VTP_PATH)
        results["Input_mode"] = mode

        if mode == "unsupported" or len(terminations) < 2:
            add_warning(
                warnings,
                "W_MODE_001",
                "Input appears incompatible for full EVAR measurements: could not identify >=2 vessel terminations either as open boundaries or cap faces. Proceeding with best-effort fallback.",
            )

        # Infer scale and apply scaling (to mm)
        scale_factor, scale_conf, scale_warn = infer_scale_to_mm(surface_tri, terminations, warnings)
        results["Scale_factor_to_mm"] = scale_factor
        results["Scale_confidence"] = scale_conf
        results["warn_scale_inference"] = int(scale_warn)

        S = np.eye(3) * float(scale_factor)
        surface_mm = apply_linear_transform_to_polydata(surface_tri, S, np.zeros(3))

        # Scale termination centers/areas consistently
        terminations_mm: List[Dict[str, Any]] = []
        for t in terminations:
            tt = dict(t)
            tt["center"] = np.array(t["center"], dtype=float) * float(scale_factor)
            tt["area"] = float(t.get("area", 0.0)) * float(scale_factor ** 2)
            if tt.get("area", 0.0) > 0:
                tt["diameter_eq"] = math.sqrt(4.0 * float(tt["area"]) / math.pi)
            terminations_mm.append(tt)

        # Centerlines with VMTK if available
        centerlines = None
        vtkvmtk_mod, _ = try_import_vmtk()
        if vtkvmtk_mod is not None:
            results["VMTK_available"] = "true"
        else:
            results["VMTK_available"] = "false"

        if vtkvmtk_mod is not None and len(terminations_mm) >= 2 and mode != "unsupported":
            centerlines, _cl_info = compute_centerlines_vmtk(surface_mm, terminations_mm, mode, warnings)
            if centerlines is None:
                results["warn_centerlines"] = 1
                add_warning(warnings, "W_CENTER_001", "Centerline extraction failed; will fallback where possible.")
        else:
            results["warn_centerlines"] = 1
            if vtkvmtk_mod is None:
                add_warning(warnings, "W_CENTER_002", "VMTK not available; centerline-based measurements not possible.")
            elif len(terminations_mm) < 2:
                add_warning(warnings, "W_CENTER_003", "Insufficient terminations for centerline seeding; centerline-based measurements not possible.")
            else:
                add_warning(warnings, "W_CENTER_004", f"Input mode '{mode}' not suitable for centerline extraction in this script.")

        if centerlines is not None:
            # Compute full measurements
            compute_measurements(surface_mm, centerlines, terminations_mm, results, warnings)
        else:
            # Fallback: compute maximum diameter and write others as NaN
            max_d = fallback_max_diameter_by_slicing(surface_mm, terminations_mm, warnings)
            if math.isfinite(max_d):
                results["Maximum_aneurysm_diameter"] = float(max_d)

        write_output_txt(OUTPUT_TXT_PATH, results, warnings)

    except Exception as e:
        add_warning(warnings, "E_FATAL_001", f"Unhandled exception: {e}")
        add_warning(warnings, "E_FATAL_002", traceback.format_exc().strip().replace("\n", " | "))
        # Ensure output still written
        try:
            write_output_txt(OUTPUT_TXT_PATH, results, warnings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
