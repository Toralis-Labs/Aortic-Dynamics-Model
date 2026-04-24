#!/usr/bin/env python3
"""
Automated EVAR-relevant geometric measurements from an unlabeled abdominal aorta lumen .vtp surface.

Single-file, end-to-end pipeline:
- Load + robustly clean mesh
- Detect open boundary loops (vessel termini)
- Compute canonical frame (best-effort) for anatomy inference
- Cap open boundaries (for stable centerline/section computations)
- Extract centerline tree (VMTK preferred)
- Infer anatomy (inlet, aortic bifurcation, common iliacs, EIA/IIA splits, renal ostia)
- Compute orthogonal cross-section equivalent diameters (area-based) when feasible
- Derivative/persistence-based aneurysm start detection (central logic)
- Write all metrics + warning flags/confidence to a plain .txt file

Notes:
- This script is designed to run as a standalone Python script.
- It attempts multiple VMTK import pathways (conda vmtk, python-wrapped vtkvmtk, Slicer VMTK modules).
- If VMTK is unavailable or centerline extraction fails, it writes NaNs with detailed warnings (script still "works").
"""

# =========================
# USER-EDITABLE PATHS
# =========================
INPUT_VTP_PATH = "input_lumen.vtp"
OUTPUT_TXT_PATH = "evar_measurements.txt"

# =========================
# TUNABLE PARAMETERS
# =========================
MIN_OUTLET_DIAMETER_MM = 1.5          # filter out tiny boundary loops as likely artifacts
MAX_OUTLETS_FOR_CENTERLINES = 64      # safety cap, most AAA models are far below this
CENTERLINE_RESAMPLE_STEP_MM = 1.0     # used in merge/resampling; sections computed at these points
SECTION_AREA_ARRAY = "CenterlineSectionArea"
SECTION_CLOSED_ARRAY = "CenterlineSectionClosed"
RADIUS_ARRAY = "MaximumInscribedSphereRadius"

# Aneurysm start detection parameters
ANEURYSM_SMOOTH_WINDOW_MM = 11.0      # smoothing kernel/window along aorta centerline
ANEURYSM_DERIV_PERSIST_MM = 5.0       # sustained positive widening distance criterion (central)
ANEURYSM_DERIV_MIN_POS = 0.02         # mm/mm (dimensionless) minimum positive slope to consider widening
ANEURYSM_EXTRA_SLOPE_MARGIN = 0.03    # slope increment above baseline neck slope (helps conical necks)

# Renal ostium inferior-edge offset approximation:
RENAL_INFERIOR_EDGE_OFFSET_FACTOR = 0.5  # distal shift along aorta centerline = factor * renal_diameter_near_origin

# Angulation computation along EIA:
ANGULATION_SEGMENT_MM = 15.0          # use two ~15mm segments around each point (best-effort)

# Decimation for centerline extraction speed/robustness:
ENABLE_DECIMATION_FOR_CENTERLINES = True
DECIMATE_TARGET_REDUCTION_IF_LARGE = 0.80  # 0.80 means reduce ~80% (keep 20%)
DECIMATE_CELL_THRESHOLD = 300000           # if surface has more than this many cells, decimate for centerline extraction

# Units heuristic (optional):
ENABLE_UNIT_HEURISTIC = True
UNIT_CM_TO_MM_THRESHOLD = 8.0         # if inferred inlet diameter < 8 (likely cm), scale by 10

# Misc
NEAREST_ENDPOINT_TOL_MM = 8.0         # for matching loop centers to centerline endpoints
NUMERIC_FMT = "{:.6f}"               # for output floats
NAN_STR = "NaN"


import os
import sys
import math
import time
import traceback
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set
import numpy as np

try:
    import vtk
    from vtkmodules.util import numpy_support
except Exception as e:
    vtk = None
    numpy_support = None


# =========================
# Data containers
# =========================
@dataclass
class BoundaryLoop:
    loop_id: int
    n_pts: int
    centroid: np.ndarray
    normal: np.ndarray
    area: float
    diam_eq: float
    is_closed: bool
    plane_fit_rms: float


# =========================
# Utilities
# =========================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _nan() -> float:
    return float("nan")


def _isfinite(x) -> bool:
    try:
        return bool(np.isfinite(x))
    except Exception:
        return False


def _as_bool_str(x) -> str:
    if x is True:
        return "true"
    if x is False:
        return "false"
    return NAN_STR


def _fmt_val(val):
    if isinstance(val, (bool, np.bool_)):
        return _as_bool_str(bool(val))
    if val is None:
        return NAN_STR
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if not np.isfinite(val):
            return NAN_STR
        return NUMERIC_FMT.format(float(val))
    return str(val)


def _polygon_area_2d(xy: np.ndarray) -> float:
    """Shoelace formula. Expects xy shape (N,2) and assumes points ordered along loop."""
    if xy.shape[0] < 3:
        return 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _unit_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=float)
    return v / n


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _robust_interp_nan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fill NaNs in y by linear interpolation over x (1D). Keeps NaNs if too few points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = y.copy()
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return out
    out[~finite] = np.interp(x[~finite], x[finite], y[finite])
    return out


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if win <= 1 or y.size < win:
        return y.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    # pad reflect to reduce boundary effects
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def _smooth_signal(y: np.ndarray, window_mm: float, step_mm: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y.copy()
    win = int(round(window_mm / max(step_mm, 1e-6)))
    if win < 5:
        win = 5
    if win % 2 == 0:
        win += 1
    if y.size < win:
        win = max(3, (y.size // 2) * 2 + 1)
    # Attempt Savitzky-Golay if available:
    try:
        from scipy.signal import savgol_filter
        yy = y.copy()
        return savgol_filter(yy, window_length=win, polyorder=2, mode="interp")
    except Exception:
        return _moving_average(y, win=win)


def _sample_profile_at(s: np.ndarray, v: np.ndarray, s_query: float, window: float = 1.0) -> float:
    """Sample value near s_query. Returns median within window if possible; else interpolate nearest valid."""
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    if s.size == 0:
        return _nan()
    m = np.isfinite(v)
    if not np.isfinite(s_query):
        return _nan()
    # Windowed median
    sel = (np.abs(s - s_query) <= window) & m
    if np.any(sel):
        return float(np.median(v[sel]))
    # Nearest neighbors for interpolation
    idx_sorted = np.argsort(s)
    s2 = s[idx_sorted]
    v2 = v[idx_sorted]
    m2 = np.isfinite(v2)
    if m2.sum() < 2:
        return _nan() if m2.sum() == 0 else float(v2[m2][0])
    # find insertion point
    k = int(np.searchsorted(s2, s_query))
    # search left
    i_left = None
    for i in range(k - 1, -1, -1):
        if m2[i]:
            i_left = i
            break
    i_right = None
    for i in range(k, s2.size):
        if m2[i]:
            i_right = i
            break
    if i_left is None and i_right is None:
        return _nan()
    if i_left is None:
        return float(v2[i_right])
    if i_right is None:
        return float(v2[i_left])
    if s2[i_right] == s2[i_left]:
        return float(v2[i_left])
    t = (s_query - s2[i_left]) / (s2[i_right] - s2[i_left])
    return float(v2[i_left] * (1.0 - t) + v2[i_right] * t)


def _avg_profile_over(s: np.ndarray, v: np.ndarray, s0: float, s1: float) -> float:
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    if s.size == 0:
        return _nan()
    if not np.isfinite(s0) or not np.isfinite(s1):
        return _nan()
    if s1 < s0:
        s0, s1 = s1, s0
    sel = (s >= s0) & (s <= s1) & np.isfinite(v)
    if not np.any(sel):
        return _nan()
    return float(np.mean(v[sel]))


def _min_profile_over(s: np.ndarray, v: np.ndarray, s0: float, s1: float) -> float:
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    if s.size == 0:
        return _nan()
    if not np.isfinite(s0) or not np.isfinite(s1):
        return _nan()
    if s1 < s0:
        s0, s1 = s1, s0
    sel = (s >= s0) & (s <= s1) & np.isfinite(v)
    if not np.any(sel):
        return _nan()
    return float(np.min(v[sel]))


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _safe_acos(x: float) -> float:
    return math.acos(_clamp(x, -1.0, 1.0))


# =========================
# VMTK import + factory
# =========================
def try_import_vmtk_modules(warnings: List[str]) -> Optional[Dict[str, object]]:
    mods = {}
    # Option 1: conda vmtk python package
    try:
        from vmtk import vtkvmtk as vtkvmtk_mod  # type: ignore
        mods["vtkvmtk"] = vtkvmtk_mod
        return mods
    except Exception:
        pass
    # Option 2: vtkvmtk module directly
    try:
        import vtkvmtk as vtkvmtk_mod  # type: ignore
        mods["vtkvmtk"] = vtkvmtk_mod
        return mods
    except Exception:
        pass
    # Option 3: Slicer VMTK modules
    try:
        import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry  # type: ignore
        import vtkvmtkMiscPython as vtkvmtkMisc  # type: ignore
        mods["vtkvmtkComputationalGeometry"] = vtkvmtkComputationalGeometry
        mods["vtkvmtkMisc"] = vtkvmtkMisc
        return mods
    except Exception:
        pass

    warnings.append("VMTK_IMPORT_FAILED: Could not import VMTK modules (vmtk.vtkvmtk / vtkvmtk / Slicer VMTK). Centerline-based metrics will be NaN.")
    return None


def vmtk_class(mods: Dict[str, object], class_name: str):
    for m in mods.values():
        if hasattr(m, class_name):
            return getattr(m, class_name)
    return None


# =========================
# VTK I/O and mesh processing
# =========================
def load_vtp(path: str):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(reader.GetOutput())
    return poly


def clean_surface(poly: Any, warnings: List[str]) -> Any:
    # Remove unused points / duplicates
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.PointMergingOn()
    cleaner.Update()

    # Ensure triangles
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cleaner.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()

    # Optional: compute consistent normals (helps some VMTK operations)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(tri.GetOutputPort())
    normals.SetConsistency(1)
    normals.SetAutoOrientNormals(1)
    normals.SetSplitting(0)
    normals.SetComputePointNormals(1)
    normals.SetComputeCellNormals(1)
    normals.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())

    # Non-manifold edges detection
    try:
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(out)
        fe.BoundaryEdgesOff()
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOn()
        fe.Update()
        n_nonman = fe.GetOutput().GetNumberOfCells()
        if n_nonman > 0:
            warnings.append(f"SURFACE_NONMANIFOLD_EDGES: detected {n_nonman} non-manifold edges. Results may be less reliable.")
    except Exception:
        pass

    # Connectivity: keep largest connected component if multiple
    try:
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(out)
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        oc = vtk.vtkPolyData()
        oc.DeepCopy(conn.GetOutput())
        if oc.GetNumberOfCells() < out.GetNumberOfCells():
            warnings.append("SURFACE_MULTIPLE_COMPONENTS: kept largest connected component.")
        out = oc
    except Exception:
        pass

    return out


def maybe_scale_surface(poly: Any, scale: float) -> Any:
    if abs(scale - 1.0) < 1e-12:
        out = vtk.vtkPolyData()
        out.DeepCopy(poly)
        return out
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(transform)
    tf.SetInputData(poly)
    tf.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(tf.GetOutput())
    return out


def maybe_decimate_for_centerlines(poly: Any, warnings: List[str]) -> Any:
    if not ENABLE_DECIMATION_FOR_CENTERLINES:
        out = vtk.vtkPolyData()
        out.DeepCopy(poly)
        return out
    n_cells = poly.GetNumberOfCells()
    if n_cells < DECIMATE_CELL_THRESHOLD:
        out = vtk.vtkPolyData()
        out.DeepCopy(poly)
        return out
    try:
        dec = vtk.vtkDecimatePro()
        dec.SetInputData(poly)
        dec.SetTargetReduction(float(DECIMATE_TARGET_REDUCTION_IF_LARGE))
        dec.PreserveTopologyOn()
        dec.BoundaryVertexDeletionOff()
        dec.SplittingOff()
        dec.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(dec.GetOutput())
        if out.GetNumberOfCells() < 1000:
            warnings.append("DECIMATION_FAILED_OR_TOO_AGGRESSIVE: decimated surface too small, using original surface for centerlines.")
            out = vtk.vtkPolyData()
            out.DeepCopy(poly)
            return out
        warnings.append(f"DECIMATION_APPLIED_FOR_CENTERLINES: cells {n_cells} -> {out.GetNumberOfCells()}.")
        return out
    except Exception:
        warnings.append("DECIMATION_FAILED: using original surface for centerlines.")
        out = vtk.vtkPolyData()
        out.DeepCopy(poly)
        return out
    
# =========================
# Boundary loop detection
# =========================
def extract_boundary_loops(poly: Any, warnings: List[str]) -> List[BoundaryLoop]:
    loops: List[BoundaryLoop] = []
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.Update()
    edges = fe.GetOutput()
    if edges.GetNumberOfCells() == 0:
        warnings.append("NO_BOUNDARY_EDGES_FOUND: surface may already be closed/capped or invalid input for lumen with open termini.")
        return loops

    # Join contiguous segments into polylines
    stripper = vtk.vtkStripper()
    stripper.SetInputData(edges)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    line_pd = stripper.GetOutput()
    n_cells = line_pd.GetNumberOfCells()
    if n_cells == 0:
        warnings.append("BOUNDARY_LOOP_STRIPPER_EMPTY: failed to build boundary polylines.")
        return loops

    pts = line_pd.GetPoints()
    if pts is None or pts.GetNumberOfPoints() == 0:
        warnings.append("BOUNDARY_LOOP_POINTS_EMPTY: failed to get boundary points.")
        return loops

    def get_cell_point_ids(cell_id: int) -> List[int]:
        cell = line_pd.GetCell(cell_id)
        ids = cell.GetPointIds()
        return [ids.GetId(i) for i in range(ids.GetNumberOfIds())]

    for cid in range(n_cells):
        pids = get_cell_point_ids(cid)
        if len(pids) < 8:
            continue
        coords = np.array([pts.GetPoint(pid) for pid in pids], dtype=float)

        # heuristic closure
        is_closed = _euclid(coords[0], coords[-1]) < 0.75
        if is_closed and len(coords) >= 3:
            coords[-1] = coords[0]

        centroid = coords.mean(axis=0)
        X = coords - centroid
        # Plane fit via SVD/PCA
        try:
            _, svals, vt = np.linalg.svd(X, full_matrices=False)
            e1 = vt[0]
            e2 = vt[1]
            n = vt[2]
            # plane-fit RMS using smallest singular value
            plane_rms = float(svals[2] / math.sqrt(max(coords.shape[0], 1)))
        except Exception:
            # fallback
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 1.0, 0.0])
            n = np.array([0.0, 0.0, 1.0])
            plane_rms = float("nan")

        # Project to 2D in plane basis
        xy = np.stack([X @ e1, X @ e2], axis=1)
        area = _polygon_area_2d(xy)
        if area <= 0.0:
            continue
        diam_eq = math.sqrt(4.0 * area / math.pi)

        loops.append(
            BoundaryLoop(
                loop_id=len(loops),
                n_pts=int(coords.shape[0]),
                centroid=centroid,
                normal=_unit_vector(n),
                area=float(area),
                diam_eq=float(diam_eq),
                is_closed=bool(is_closed),
                plane_fit_rms=float(plane_rms),
            )
        )

    if len(loops) == 0:
        warnings.append("BOUNDARY_LOOPS_NONE_VALID: boundary edges detected but no valid loops computed.")
        return loops

    # Filter tiny loops
    filtered: List[BoundaryLoop] = []
    for lp in loops:
        if lp.diam_eq >= MIN_OUTLET_DIAMETER_MM:
            filtered.append(lp)
    if len(filtered) < len(loops):
        warnings.append(f"BOUNDARY_LOOPS_FILTERED_TINY: removed {len(loops)-len(filtered)} loops with diameter < {MIN_OUTLET_DIAMETER_MM}mm.")
    loops = filtered

    if len(loops) < 2:
        warnings.append("BOUNDARY_LOOPS_INSUFFICIENT: need at least inlet + one outlet.")
    return loops

# =========================
# Canonical frame inference
# =========================
def compute_pca_axes(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean, eigvec0, eigvec1, eigvec2) as a right-handed basis:
    - eigvec0: largest variance direction
    """
    pts = np.asarray(points_xyz, dtype=float)
    mu = pts.mean(axis=0)
    X = pts - mu
    C = (X.T @ X) / max(pts.shape[0], 1)
    w, v = np.linalg.eigh(C)  # ascending eigenvalues
    idx = np.argsort(w)[::-1]  # descending
    v = v[:, idx]
    e0 = _unit_vector(v[:, 0])
    e1 = _unit_vector(v[:, 1])
    e2 = _unit_vector(np.cross(e0, e1))
    # Re-orthonormalize e1 to ensure orthogonal to e0 and e2
    e1 = _unit_vector(np.cross(e2, e0))
    return mu, e0, e1, e2


def infer_canonical_frame(surface: Any, loops: List[BoundaryLoop], inlet_loop_idx: int, warnings: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Infer canonical axes:
    - axis_z: superior(+)->inferior(-) along major length direction
    - axis_x: left/right lateral based on inferior outlets if possible
    - axis_y: anterior/posterior via right-handed completion
    Returns: (origin, axis_x, axis_y, axis_z)
    """
    pts_vtk = surface.GetPoints()
    pts_np = numpy_support.vtk_to_numpy(pts_vtk.GetData()).reshape(-1, 3)

    mu, e0, e1, e2 = compute_pca_axes(pts_np)
    axis_z = e0

    inlet_c = loops[inlet_loop_idx].centroid
    if float(np.dot(inlet_c - mu, axis_z)) < 0.0:
        axis_z = -axis_z  # make inlet at +z (superior)

    # Infer axis_x from two largest inferior loops if possible
    axis_x = e1
    if len(loops) >= 3:
        # Use loop centroids projected on axis_z to find inferior candidates
        z_coords = np.array([float(np.dot(lp.centroid - mu, axis_z)) for lp in loops], dtype=float)
        # Inferior = low z
        k = min(6, len(loops))
        idx_inf = np.argsort(z_coords)[:k]
        # Choose two largest area among inferior candidates (exclude inlet if present)
        inf_candidates = [i for i in idx_inf if i != inlet_loop_idx]
        if len(inf_candidates) >= 2:
            inf_candidates_sorted = sorted(inf_candidates, key=lambda ii: loops[ii].area, reverse=True)
            iA, iB = inf_candidates_sorted[0], inf_candidates_sorted[1]
            v_lr = loops[iB].centroid - loops[iA].centroid
            # Orthogonalize to axis_z
            v_lr = v_lr - np.dot(v_lr, axis_z) * axis_z
            if np.linalg.norm(v_lr) > 1e-6:
                axis_x = _unit_vector(v_lr)
            else:
                warnings.append("CANONICAL_AXIS_X_FALLBACK: inferior outlet vector degenerate, using PCA axis.")
        else:
            warnings.append("CANONICAL_AXIS_X_FALLBACK: insufficient inferior outlets, using PCA axis.")
    else:
        warnings.append("CANONICAL_AXIS_X_FALLBACK: insufficient loops, using PCA axis.")

    axis_y = _unit_vector(np.cross(axis_z, axis_x))
    axis_x = _unit_vector(np.cross(axis_y, axis_z))  # ensure orthonormal right-handed

    # Axis sign ambiguity left/right remains fundamentally ambiguous without external reference
    warnings.append("LEFT_RIGHT_SIGN_AMBIGUITY: right/left labels are relative to inferred canonical axis_x and may be swapped.")

    return mu, axis_x, axis_y, axis_z


def canonical_coords(p: np.ndarray, origin: np.ndarray, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    d = np.asarray(p, dtype=float) - np.asarray(origin, dtype=float)
    return np.array([float(np.dot(d, ax)), float(np.dot(d, ay)), float(np.dot(d, az))], dtype=float)


# =========================
# VMTK pipeline: cap + centerlines + merge + sections
# =========================
def cap_surface_vmtk(poly: Any, vmtk_mods: Dict[str, object], warnings: List[str]) -> Tuple[Optional[Any], Optional[Any]]:
    CapPolyData = vmtk_class(vmtk_mods, "vtkvmtkCapPolyData")
    if CapPolyData is None:
        warnings.append("VMTK_CAP_CLASS_NOT_FOUND: vtkvmtkCapPolyData missing.")
        return None, None
    try:
        capper = CapPolyData()
        capper.SetInputData(poly)
        # displacement allows moving barycenters; keep 0 for fidelity
        if hasattr(capper, "SetDisplacement"):
            capper.SetDisplacement(0.0)
        if hasattr(capper, "SetInPlaneDisplacement"):
            capper.SetInPlaneDisplacement(0.0)
        capper.Update()
        capped = vtk.vtkPolyData()
        capped.DeepCopy(capper.GetOutput())
        cap_ids = capper.GetCapCenterIds()
        if cap_ids is None or cap_ids.GetNumberOfIds() == 0:
            warnings.append("VMTK_CAP_FAILED: cap center ids empty.")
            return capped, None
        return capped, cap_ids
    except Exception:
        warnings.append("VMTK_CAP_EXCEPTION: exception during capping.")
        return None, None


def match_inlet_cap_id(capped: Any, cap_ids: Any, loops: List[BoundaryLoop], inlet_loop_idx: int) -> Optional[int]:
    """Choose cap center id closest to inlet loop centroid."""
    inlet_c = loops[inlet_loop_idx].centroid
    pts = capped.GetPoints()
    best = None
    best_d = float("inf")
    for k in range(cap_ids.GetNumberOfIds()):
        pid = int(cap_ids.GetId(k))
        p = np.array(pts.GetPoint(pid), dtype=float)
        d = float(np.linalg.norm(p - inlet_c))
        if d < best_d:
            best_d = d
            best = pid
    return best


def compute_centerlines_vmtk(
    capped_surface: Any,
    cap_center_ids: Any,
    inlet_cap_center_point_id: int,
    target_cap_center_point_ids: List[int],
    vmtk_mods: Dict[str, object],
    warnings: List[str],
) -> Optional[Any]:
    PolyDataCenterlines = vmtk_class(vmtk_mods, "vtkvmtkPolyDataCenterlines")
    if PolyDataCenterlines is None:
        warnings.append("VMTK_CENTERLINES_CLASS_NOT_FOUND: vtkvmtkPolyDataCenterlines missing.")
        return None
    try:
        source_ids = vtk.vtkIdList()
        source_ids.InsertNextId(int(inlet_cap_center_point_id))
        target_ids = vtk.vtkIdList()
        for pid in target_cap_center_point_ids:
            target_ids.InsertNextId(int(pid))

        cl = PolyDataCenterlines()
        cl.SetInputData(capped_surface)
        if hasattr(cl, "SetCapCenterIds") and cap_center_ids is not None:
            cl.SetCapCenterIds(cap_center_ids)
        cl.SetSourceSeedIds(source_ids)
        cl.SetTargetSeedIds(target_ids)
        if hasattr(cl, "SetRadiusArrayName"):
            cl.SetRadiusArrayName(RADIUS_ARRAY)
        if hasattr(cl, "SetCostFunction"):
            cl.SetCostFunction("1/R")
        if hasattr(cl, "SetFlipNormals"):
            cl.SetFlipNormals(0)
        if hasattr(cl, "SetAppendEndPointsToCenterlines"):
            cl.SetAppendEndPointsToCenterlines(0)
        if hasattr(cl, "SetSimplifyVoronoi"):
            cl.SetSimplifyVoronoi(0)  # robust with VTK>=9
        if hasattr(cl, "SetCenterlineResampling"):
            cl.SetCenterlineResampling(0)  # resample later
        if hasattr(cl, "SetResamplingStepLength"):
            cl.SetResamplingStepLength(float(CENTERLINE_RESAMPLE_STEP_MM))
        cl.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(cl.GetOutput())
        if out.GetNumberOfPoints() == 0 or out.GetNumberOfCells() == 0:
            warnings.append("CENTERLINES_EMPTY: VMTK centerlines produced empty output.")
            return None
        return out
    except Exception:
        warnings.append("CENTERLINES_EXCEPTION: exception during VMTK centerline extraction.")
        return None


def branch_extract_and_merge_vmtk(centerlines: Any, vmtk_mods: Dict[str, object], warnings: List[str]) -> Any:
    CenterlineBranchExtractor = vmtk_class(vmtk_mods, "vtkvmtkCenterlineBranchExtractor")
    MergeCenterlines = vmtk_class(vmtk_mods, "vtkvmtkMergeCenterlines")
    if CenterlineBranchExtractor is None or MergeCenterlines is None:
        warnings.append("VMTK_BRANCH_OR_MERGE_CLASS_MISSING: using raw centerlines without branch merge (may reduce topology robustness).")
        out = vtk.vtkPolyData()
        out.DeepCopy(centerlines)
        return out

    try:
        be = CenterlineBranchExtractor()
        be.SetInputData(centerlines)
        if hasattr(be, "SetBlankingArrayName"):
            be.SetBlankingArrayName("Blanking")
        if hasattr(be, "SetRadiusArrayName"):
            be.SetRadiusArrayName(RADIUS_ARRAY)
        if hasattr(be, "SetGroupIdsArrayName"):
            be.SetGroupIdsArrayName("GroupIds")
        if hasattr(be, "SetCenterlineIdsArrayName"):
            be.SetCenterlineIdsArrayName("CenterlineIds")
        if hasattr(be, "SetTractIdsArrayName"):
            be.SetTractIdsArrayName("TractIds")
        be.Update()
        branched = be.GetOutput()

        mc = MergeCenterlines()
        mc.SetInputData(branched)
        if hasattr(mc, "SetRadiusArrayName"):
            mc.SetRadiusArrayName(RADIUS_ARRAY)
        if hasattr(mc, "SetGroupIdsArrayName"):
            mc.SetGroupIdsArrayName("GroupIds")
        if hasattr(mc, "SetCenterlineIdsArrayName"):
            mc.SetCenterlineIdsArrayName("CenterlineIds")
        if hasattr(mc, "SetTractIdsArrayName"):
            mc.SetTractIdsArrayName("TractIds")
        if hasattr(mc, "SetBlankingArrayName"):
            mc.SetBlankingArrayName("Blanking")
        if hasattr(mc, "SetResamplingStepLength"):
            mc.SetResamplingStepLength(float(CENTERLINE_RESAMPLE_STEP_MM))
        if hasattr(mc, "SetMergeBlanked"):
            mc.SetMergeBlanked(1)
        mc.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(mc.GetOutput())
        if out.GetNumberOfPoints() == 0 or out.GetNumberOfCells() == 0:
            warnings.append("MERGE_CENTERLINES_EMPTY: merge produced empty output, falling back to branched.")
            out = vtk.vtkPolyData()
            out.DeepCopy(branched)
        return out
    except Exception:
        warnings.append("BRANCH_EXTRACT_OR_MERGE_EXCEPTION: falling back to raw centerlines.")
        out = vtk.vtkPolyData()
        out.DeepCopy(centerlines)
        return out


def compute_centerline_sections_vmtk(
    surface: Any, centerlines: Any, vmtk_mods: Dict[str, object], warnings: List[str]
) -> Any:
    PolyDataCenterlineSections = vmtk_class(vmtk_mods, "vtkvmtkPolyDataCenterlineSections")
    if PolyDataCenterlineSections is None:
        warnings.append("VMTK_SECTIONS_CLASS_NOT_FOUND: vtkvmtkPolyDataCenterlineSections missing. Using radius-based diameters.")
        out = vtk.vtkPolyData()
        out.DeepCopy(centerlines)
        return out
    try:
        cs = PolyDataCenterlineSections()
        cs.SetInputData(surface)
        if hasattr(cs, "SetCenterlines"):
            cs.SetCenterlines(centerlines)
        if hasattr(cs, "SetCenterlineSectionAreaArrayName"):
            cs.SetCenterlineSectionAreaArrayName(SECTION_AREA_ARRAY)
        if hasattr(cs, "SetCenterlineSectionMinSizeArrayName"):
            cs.SetCenterlineSectionMinSizeArrayName("CenterlineSectionMinSize")
        if hasattr(cs, "SetCenterlineSectionMaxSizeArrayName"):
            cs.SetCenterlineSectionMaxSizeArrayName("CenterlineSectionMaxSize")
        if hasattr(cs, "SetCenterlineSectionShapeArrayName"):
            cs.SetCenterlineSectionShapeArrayName("CenterlineSectionShape")
        if hasattr(cs, "SetCenterlineSectionClosedArrayName"):
            cs.SetCenterlineSectionClosedArrayName(SECTION_CLOSED_ARRAY)
        cs.Update()
        out = vtk.vtkPolyData()
        # The filter provides updated centerlines via GetCenterlines()
        if hasattr(cs, "GetCenterlines"):
            out.DeepCopy(cs.GetCenterlines())
        else:
            out.DeepCopy(centerlines)
            warnings.append("VMTK_SECTIONS_NO_GETCENTERLINES: using original centerlines.")
        return out
    except Exception:
        warnings.append("CENTERLINE_SECTIONS_EXCEPTION: using radius-based diameters.")
        out = vtk.vtkPolyData()
        out.DeepCopy(centerlines)
        return out


def extract_centerline_diameters(centerlines: Any, warnings: List[str]) -> np.ndarray:
    n = centerlines.GetNumberOfPoints()
    if n == 0:
        return np.zeros((0,), dtype=float)

    pd = centerlines.GetPointData()
    area_arr = pd.GetArray(SECTION_AREA_ARRAY) if pd is not None else None
    closed_arr = pd.GetArray(SECTION_CLOSED_ARRAY) if pd is not None else None
    radius_arr = pd.GetArray(RADIUS_ARRAY) if pd is not None else None

    D = np.full((n,), np.nan, dtype=float)
    if area_arr is not None:
        area = numpy_support.vtk_to_numpy(area_arr).astype(float)
        if closed_arr is not None:
            closed = numpy_support.vtk_to_numpy(closed_arr).astype(float)
        else:
            closed = np.ones_like(area)
        ok = (area > 0.0) & (closed > 0.5) & np.isfinite(area)
        D[ok] = np.sqrt(4.0 * area[ok] / math.pi)
        if np.sum(ok) < max(10, n // 10):
            warnings.append("SECTION_AREA_SPARSE_OR_OPEN: many sections not closed; diameters may be incomplete near branches/termini.")
        return D

    if radius_arr is not None:
        r = numpy_support.vtk_to_numpy(radius_arr).astype(float)
        ok = (r > 0.0) & np.isfinite(r)
        D[ok] = 2.0 * r[ok]
        warnings.append("DIAMETER_FROM_RADIUS_FALLBACK: CenterlineSectionArea missing; using 2*MaximumInscribedSphereRadius.")
        return D

    warnings.append("DIAMETER_ARRAYS_MISSING: no area or radius arrays found on centerlines.")
    return D


# =========================
# Centerline graph + tree
# =========================
def build_graph_from_centerlines(centerlines: Any, warnings: List[str]) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float]]]]:
    pts = centerlines.GetPoints()
    if pts is None or pts.GetNumberOfPoints() == 0:
        return np.zeros((0, 3), dtype=float), {}
    coords = numpy_support.vtk_to_numpy(pts.GetData()).reshape(-1, 3).astype(float)

    neighbors: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(coords.shape[0])}
    lines = centerlines.GetLines()
    if lines is None or centerlines.GetNumberOfCells() == 0:
        warnings.append("CENTERLINES_NO_LINES: cannot build centerline graph.")
        return coords, neighbors

    # Build edges
    seen_edges: Set[Tuple[int, int]] = set()
    id_list = vtk.vtkIdList()
    lines.InitTraversal()
    n_edges = 0
    while lines.GetNextCell(id_list):
        m = id_list.GetNumberOfIds()
        if m < 2:
            continue
        ids = [int(id_list.GetId(i)) for i in range(m)]
        for i in range(m - 1):
            a = ids[i]
            b = ids[i + 1]
            if a == b:
                continue
            u, v = (a, b) if a < b else (b, a)
            if (u, v) in seen_edges:
                continue
            seen_edges.add((u, v))
            w = float(np.linalg.norm(coords[a] - coords[b]))
            neighbors[a].append((b, w))
            neighbors[b].append((a, w))
            n_edges += 1

    if n_edges == 0:
        warnings.append("CENTERLINE_GRAPH_EMPTY: no edges found.")
    return coords, neighbors


def dijkstra_root_tree(neighbors: Dict[int, List[Tuple[int, float]]], root: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    n = len(neighbors)
    dist = np.full((n,), np.inf, dtype=float)
    parent = np.full((n,), -1, dtype=int)

    dist[root] = 0.0
    h = [(0.0, int(root))]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        for v, w in neighbors.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(h, (nd, int(v)))

    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            children[p].append(i)
    return dist, parent, children


def undirected_degrees(neighbors: Dict[int, List[Tuple[int, float]]]) -> np.ndarray:
    n = len(neighbors)
    deg = np.zeros((n,), dtype=int)
    for i in range(n):
        deg[i] = len(neighbors.get(i, []))
    return deg


def find_nearest_endpoint(endpoints: List[int], coords: np.ndarray, point: np.ndarray) -> Tuple[Optional[int], float]:
    if len(endpoints) == 0:
        return None, float("inf")
    p = np.asarray(point, dtype=float)
    best = None
    best_d = float("inf")
    for e in endpoints:
        d = float(np.linalg.norm(coords[e] - p))
        if d < best_d:
            best_d = d
            best = int(e)
    return best, best_d


def ancestors_set(parent: np.ndarray, node: int) -> Set[int]:
    s = set()
    u = int(node)
    while u >= 0 and u not in s:
        s.add(u)
        u = int(parent[u])
    return s


def lca(parent: np.ndarray, a: int, b: int) -> Optional[int]:
    sa = ancestors_set(parent, a)
    u = int(b)
    seen = set()
    while u >= 0 and u not in seen:
        if u in sa:
            return u
        seen.add(u)
        u = int(parent[u])
    return None


def path_ancestor_to_descendant(parent: np.ndarray, ancestor: int, descendant: int) -> Optional[List[int]]:
    """Return path nodes from ancestor -> descendant along parent pointers (assumes ancestor is on chain)."""
    chain = []
    u = int(descendant)
    seen = set()
    while u >= 0 and u not in seen:
        chain.append(u)
        if u == int(ancestor):
            break
        seen.add(u)
        u = int(parent[u])
    if len(chain) == 0 or chain[-1] != int(ancestor):
        return None
    chain.reverse()
    return chain


def collect_subtree_leaves(children: Dict[int, List[int]], start: int) -> List[int]:
    leaves = []
    stack = [int(start)]
    while stack:
        u = stack.pop()
        ch = children.get(u, [])
        if len(ch) == 0:
            leaves.append(u)
        else:
            stack.extend(ch)
    return leaves


# =========================
# Anatomy inference heuristics
# =========================
def infer_inlet_loop(loops: List[BoundaryLoop], warnings: List[str]) -> int:
    if len(loops) == 0:
        warnings.append("INLET_INFERENCE_FAILED_NO_LOOPS")
        return -1
    areas = np.array([lp.area for lp in loops], dtype=float)
    idx = int(np.argmax(areas))
    # confidence based on separation from second largest
    if len(loops) >= 2:
        a1 = float(np.sort(areas)[-1])
        a2 = float(np.sort(areas)[-2])
        if a2 > 0:
            ratio = a1 / a2
            if ratio < 1.15:
                warnings.append("INLET_LOW_SEPARATION: largest boundary loop area not much larger than 2nd; inlet inference less confident.")
    return idx


def infer_scale_factor_from_inlet_diameter(inlet_diam: float, warnings: List[str]) -> float:
    if not ENABLE_UNIT_HEURISTIC:
        return 1.0
    if not np.isfinite(inlet_diam) or inlet_diam <= 0:
        return 1.0
    # Typical suprarenal/infrarenal aorta is ~20-30mm; if inlet is ~2-3, likely cm.
    if inlet_diam < UNIT_CM_TO_MM_THRESHOLD:
        warnings.append(f"UNITS_HEURISTIC_APPLIED: inlet diameter {inlet_diam:.3f} suggests cm; scaling geometry by 10 to output mm.")
        return 10.0
    return 1.0


def select_iliac_representative_endpoints(
    endpoints: List[int],
    root_ep: int,
    dist: np.ndarray,
    coords: np.ndarray,
    origin: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    warnings: List[str],
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Pick two distal endpoints (one on +x and one on -x) among farthest endpoints by centerline distance.
    Returns (right_rep, left_rep, confidence).
    """
    eps = [e for e in endpoints if e != int(root_ep) and np.isfinite(dist[e])]
    if len(eps) < 2:
        warnings.append("ILIAC_ENDPOINTS_INSUFFICIENT: could not find two distal endpoints.")
        return None, None, 0.0
    # Take farthest K by dist
    K = min(10, len(eps))
    eps_sorted = sorted(eps, key=lambda e: dist[e], reverse=True)[:K]
    xx = np.array([canonical_coords(coords[e], origin, ax, ay, az)[0] for e in eps_sorted], dtype=float)

    # right = max x, left = min x
    i_max = int(np.argmax(xx))
    i_min = int(np.argmin(xx))
    right_rep = int(eps_sorted[i_max])
    left_rep = int(eps_sorted[i_min])

    # Confidence: separation of x and both far distances
    x_sep = float(abs(xx[i_max] - xx[i_min]))
    d_sep = float(dist[right_rep] + dist[left_rep]) / 2.0
    conf = _clamp((x_sep / (0.25 * max(d_sep, 1e-6))), 0.0, 1.0)  # heuristic
    if x_sep < 5.0:
        warnings.append("ILIAC_LR_SEPARATION_SMALL: iliac endpoints not well-separated in inferred x; left/right assignment uncertain.")
        conf *= 0.5
    return right_rep, left_rep, conf


def infer_common_iliac_split_and_external_internal(
    parent: np.ndarray,
    children: Dict[int, List[int]],
    bif_node: int,
    side_child: int,
    dist: np.ndarray,
    coords: np.ndarray,
    origin: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    warnings: List[str],
    side_name: str,
) -> Tuple[Optional[int], Optional[int], Optional[int], float]:
    """
    Within a side subtree, identify external/internal iliac endpoints and common iliac bifurcation (LCA).
    Returns (common_iliac_split_node, external_leaf, internal_leaf, confidence).
    """
    leaves = collect_subtree_leaves(children, side_child)
    # Remove any leaf that is actually root side artifacts by demanding it's distal
    leaves = [l for l in leaves if np.isfinite(dist[l]) and dist[l] > dist[bif_node] + 5.0]
    if len(leaves) < 1:
        warnings.append(f"{side_name}_SUBTREE_NO_LEAVES: cannot infer iliac branches on this side.")
        return None, None, None, 0.0
    if len(leaves) == 1:
        warnings.append(f"{side_name}_ONLY_ONE_DISTAL_LEAF: internal/external iliac bifurcation not present; EIA/IIA labeling unavailable.")
        # Best-effort: treat the single leaf as "external" for downstream EIA metrics, but flag low confidence.
        return None, int(leaves[0]), None, 0.2

    # External tends to be longer and/or larger. Use max dist as external.
    leaves_sorted = sorted(leaves, key=lambda l: dist[l], reverse=True)
    external_leaf = int(leaves_sorted[0])
    internal_leaf = int(leaves_sorted[1])

    split = lca(parent, external_leaf, internal_leaf)
    if split is None:
        warnings.append(f"{side_name}_COMMON_ILIAC_SPLIT_LCA_FAILED")
        return None, external_leaf, internal_leaf, 0.2

    # Confidence: split should be distal to aortic bif by at least some length
    length_ci = float(dist[split] - dist[bif_node]) if np.isfinite(dist[split]) else 0.0
    conf = 0.8
    if length_ci < 5.0:
        warnings.append(f"{side_name}_COMMON_ILIAC_SPLIT_TOO_PROXIMAL: inferred split very close to aortic bifurcation.")
        conf *= 0.4

    # Sanity: internal vs external x/y behavior (weak), just for confidence
    ext_c = canonical_coords(coords[external_leaf], origin, ax, ay, az)
    int_c = canonical_coords(coords[internal_leaf], origin, ax, ay, az)
    # external often more lateral/anterior than internal; but uncertain
    if abs(ext_c[0]) < abs(int_c[0]) and abs(ext_c[1]) < abs(int_c[1]):
        conf *= 0.85

    return int(split), external_leaf, internal_leaf, _clamp(conf, 0.0, 1.0)


def detect_renal_branches_on_trunk(
    parent: np.ndarray,
    children: Dict[int, List[int]],
    trunk_path: List[int],
    coords: np.ndarray,
    diam: np.ndarray,
    dist: np.ndarray,
    origin: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    warnings: List[str],
) -> Tuple[Optional[int], Optional[int], Optional[int], float]:
    """
    Identify renal branch origin nodes (on trunk path) using paired-lateral branch heuristic.
    Returns (lowest_renal_origin_node, left_renal_origin_node, right_renal_origin_node, confidence).
    """
    trunk_set = set(trunk_path)
    # For each trunk node with extra children, consider side branches
    branch_candidates = []
    for u in trunk_path:
        ch = children.get(u, [])
        if len(ch) < 1:
            continue
        # Determine trunk continuation child (the one that is next trunk node with higher dist)
        trunk_next = None
        for v in ch:
            if v in trunk_set:
                trunk_next = v
                break
        # Side branches are children not in trunk_set
        for v in ch:
            if v == trunk_next:
                continue
            # Compute features for this side branch
            # Find a representative distal leaf
            leaves = collect_subtree_leaves(children, v)
            if len(leaves) == 0:
                continue
            leaf = max(leaves, key=lambda l: dist[l] if np.isfinite(dist[l]) else -np.inf)
            # Take direction vector: from u to a point along side branch ~10mm away
            p0 = coords[u]
            # walk down the branch for ~10mm
            walk = [v]
            cur = v
            length = 0.0
            while True:
                nxts = children.get(cur, [])
                if len(nxts) == 0:
                    break
                # choose child with greatest dist
                nxt = max(nxts, key=lambda t: dist[t])
                length += float(np.linalg.norm(coords[nxt] - coords[cur]))
                walk.append(nxt)
                cur = nxt
                if length >= 10.0 or len(walk) > 50:
                    break
            p1 = coords[walk[-1]]
            dvec = _unit_vector(p1 - p0)
            d_can = np.array([float(np.dot(dvec, ax)), float(np.dot(dvec, ay)), float(np.dot(dvec, az))], dtype=float)
            lateralness = abs(d_can[0])
            inferiorness = abs(d_can[2])
            # diameter proxy near branch origin: sample diam within first few mm along side branch (if possible)
            # We'll use median of diam values among nodes in 'walk'
            dvals = [diam[w] for w in walk if w < diam.size and np.isfinite(diam[w])]
            d0 = float(np.median(dvals)) if len(dvals) > 0 else float("nan")

            # Candidate filter: renal-ish size and laterality
            if np.isfinite(d0) and 2.5 <= d0 <= 12.0 and lateralness >= 0.45 and inferiorness <= 0.75:
                branch_candidates.append((u, v, int(leaf), float(d0), float(lateralness), float(dist[u])))

    if len(branch_candidates) == 0:
        warnings.append("RENAL_NOT_FOUND: no renal-like paired lateral side branches detected.")
        return None, None, None, 0.0

    # Pairing: choose two candidates with opposite lateral direction (based on leaf x sign)
    # We use canonical x of the leaf.
    candidates_enriched = []
    for (u, v, leaf, d0, lat, su) in branch_candidates:
        x_leaf = canonical_coords(coords[leaf], origin, ax, ay, az)[0]
        candidates_enriched.append((u, v, leaf, d0, lat, su, x_leaf))

    # Build pairs
    best_pair = None
    best_score = -1e18
    for i in range(len(candidates_enriched)):
        for j in range(i + 1, len(candidates_enriched)):
            a = candidates_enriched[i]
            b = candidates_enriched[j]
            # Opposite sides in x
            if a[6] * b[6] >= 0.0:
                continue
            # Similar takeoff distance along trunk (within 50mm)
            if abs(a[5] - b[5]) > 50.0:
                continue
            # Similar diameter
            dd = abs(a[3] - b[3])
            if dd > 6.0:
                continue
            # Score: lateralness + similarity + closeness in takeoff
            score = (a[4] + b[4]) * 10.0 - dd * 2.0 - abs(a[5] - b[5]) * 0.1
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        # Best-effort: choose single best candidate as "lowest renal"
        best = max(candidates_enriched, key=lambda t: (t[4], t[5]))  # more lateral and more distal
        warnings.append("RENAL_PAIRING_FAILED: using best single renal-like branch as lowest renal (very low confidence).")
        return int(best[0]), None, None, 0.2

    a, b = best_pair
    # Determine which side is "right" vs "left" by canonical x sign (right = +x)
    right = a if a[6] > 0 else b
    left = b if a[6] > 0 else a

    # Lowest renal = more distal along trunk (larger dist[u])
    lowest = right if right[5] >= left[5] else left
    conf = 0.85
    # Reduce confidence if pairing score just marginal
    if best_score < 3.0:
        conf *= 0.6
        warnings.append("RENAL_PAIR_LOW_SCORE: renal pairing confidence reduced.")

    return int(lowest[0]), int(left[0]), int(right[0]), _clamp(conf, 0.0, 1.0)


# =========================
# Aneurysm start detection (derivative/persistence-based)
# =========================
def detect_aneurysm_start_distance(
    s: np.ndarray,
    D: np.ndarray,
    step_mm: float,
    warnings: List[str],
) -> Tuple[float, float]:
    """
    Detect aneurysm start as first location where diameter begins sustained widening.
    Central logic uses smoothed derivative + persistence; conical neck handled via baseline slope and/or dd peak gating.
    Returns (aneurysm_start_s, confidence).
    """
    s = np.asarray(s, dtype=float)
    D = np.asarray(D, dtype=float)
    if s.size < 10 or np.sum(np.isfinite(D)) < 8:
        warnings.append("ANEURYSM_START_INSUFFICIENT_PROFILE: too few samples.")
        return _nan(), 0.0

    # Fill NaNs for analysis only (keep original D for reporting elsewhere if needed)
    Df = _robust_interp_nan(s, D)
    # Smooth
    Ds = _smooth_signal(Df, window_mm=ANEURYSM_SMOOTH_WINDOW_MM, step_mm=step_mm)

    # Derivatives
    d1 = np.gradient(Ds, s, edge_order=1)
    d2 = np.gradient(d1, s, edge_order=1)

    # Baseline slope in first ~10mm (or first 15% if short)
    base_len = min(10.0, float(s[-1]) * 0.15)
    base_sel = (s >= s[0]) & (s <= s[0] + base_len)
    base_slope = float(np.median(d1[base_sel])) if np.any(base_sel) else float(np.median(d1[:max(3, s.size//10)]))
    # Thresholds
    d_thresh = max(float(ANEURYSM_DERIV_MIN_POS), base_slope + float(ANEURYSM_EXTRA_SLOPE_MARGIN))
    # For dd threshold, robust
    dd_med = float(np.median(d2))
    dd_mad = _mad(d2)
    dd_thresh = max(1e-3, dd_med + 3.0 * 1.4826 * dd_mad)

    persist = max(2, int(round(ANEURYSM_DERIV_PERSIST_MM / max(step_mm, 1e-6))))
    # Ignore first 2-3mm for noise
    start_idx = int(np.searchsorted(s, s[0] + 3.0))

    # Primary: look for dd peak indicating increase in widening rate, then sustained positive d1
    conf = 0.7
    for i in range(start_idx, s.size - persist - 1):
        if d2[i] > dd_thresh and d1[i] > 0.0:
            window = d1[i:i + persist]
            frac_pos = float(np.mean(window > d_thresh))
            if frac_pos >= 0.8:
                s0 = float(s[i])
                conf_local = 0.9 * _clamp(frac_pos, 0.0, 1.0)
                return s0, conf_local

    # Fallback: sustained positive derivative above threshold
    conf *= 0.8
    for i in range(start_idx, s.size - persist - 1):
        window = d1[i:i + persist]
        frac_pos = float(np.mean(window > d_thresh))
        if frac_pos >= 0.8:
            s0 = float(s[i])
            return s0, conf * _clamp(frac_pos, 0.0, 1.0)

    # Secondary fallback (not primary): mild threshold on diameter increase
    D0 = float(np.median(Ds[base_sel])) if np.any(base_sel) else float(np.median(Ds[:max(3, s.size//10)]))
    if np.isfinite(D0) and D0 > 0:
        target = 1.10 * D0
        for i in range(start_idx, s.size - persist - 1):
            if Ds[i] >= target:
                # check mild persistence (non-decreasing trend)
                window = Ds[i:i + persist]
                if float(np.mean(np.diff(window) >= -0.02)) >= 0.8:
                    warnings.append("ANEURYSM_START_FALLBACK_THRESHOLD_USED: derivative-based sustained widening not detected; used diameter increment fallback.")
                    return float(s[i]), 0.3

    warnings.append("ANEURYSM_START_NOT_FOUND: unable to detect sustained positive widening.")
    return _nan(), 0.0


# =========================
# EIA angulation
# =========================
def max_angulation_along_path(coords_path: np.ndarray, s_path: np.ndarray, seg_mm: float) -> float:
    coords_path = np.asarray(coords_path, dtype=float)
    s_path = np.asarray(s_path, dtype=float)
    n = coords_path.shape[0]
    if n < 5:
        return _nan()
    # Estimate spacing
    ds = np.diff(s_path)
    ds_med = float(np.median(ds)) if ds.size > 0 else 1.0
    if not np.isfinite(ds_med) or ds_med <= 0:
        ds_med = 1.0
    k = int(round((seg_mm / 2.0) / ds_med))
    k = max(1, k)
    if n < 2 * k + 2:
        k = max(1, (n - 2) // 2)
    if k < 1:
        return _nan()
    max_angle = 0.0
    for i in range(k, n - k - 1):
        v1 = coords_path[i] - coords_path[i - k]
        v2 = coords_path[i + k] - coords_path[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        v1 /= n1
        v2 /= n2
        ang = _safe_acos(float(np.dot(v1, v2))) * (180.0 / math.pi)
        if ang > max_angle:
            max_angle = ang
    return float(max_angle) if max_angle > 0 else _nan()


# =========================
# Output formatting
# =========================
def write_results_txt(output_path: str, results: Dict[str, object], ordered_keys: List[str]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for k in ordered_keys:
            v = results.get(k, _nan())
            f.write(f"{k}={_fmt_val(v)}\n")


# =========================
# Main routine
# =========================
def main():
    t0 = time.time()
    warnings: List[str] = []
    results: Dict[str, object] = {}

    # Required output variables (must appear even if NaN)
    primary_keys = [
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
    extra_keys = [
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
    meta_keys = [
        "InputFile",
        "OutputFile",
        "Units",
        "ScaleFactorApplied",
        "RunTimeSeconds",
    ]
    warning_flag_keys = [
        "warning_vmtk_missing",
        "warning_inlet",
        "warning_aortic_bifurcation",
        "warning_lowest_renal",
        "warning_right_common_iliac_bifurcation",
        "warning_left_common_iliac_bifurcation",
        "warning_right_external_iliac",
        "warning_left_external_iliac",
        "warning_left_right_ambiguity",
        "warning_aneurysm_start",
    ]
    confidence_keys = [
        "confidence_inlet",
        "confidence_aortic_bifurcation",
        "confidence_lowest_renal",
        "confidence_common_iliac_right",
        "confidence_common_iliac_left",
        "confidence_external_iliac_right",
        "confidence_external_iliac_left",
        "confidence_left_right_assignment",
        "confidence_aneurysm_start",
    ]
    summary_keys = ["WarningsSummary"]

    ordered_keys = meta_keys + primary_keys + extra_keys + warning_flag_keys + confidence_keys + summary_keys

    # Initialize defaults
    results["InputFile"] = INPUT_VTP_PATH
    results["OutputFile"] = OUTPUT_TXT_PATH
    results["Units"] = "mm"
    results["ScaleFactorApplied"] = 1.0
    for k in primary_keys + extra_keys:
        results[k] = _nan()
    for k in warning_flag_keys:
        results[k] = False
    for k in confidence_keys:
        results[k] = 0.0
    results["WarningsSummary"] = ""

    if vtk is None or numpy_support is None:
        warnings.append("VTK_IMPORT_FAILED: vtk not available; cannot process .vtp.")
        results["warning_vmtk_missing"] = True
        results["WarningsSummary"] = ";".join(warnings)
        results["RunTimeSeconds"] = _safe_float(time.time() - t0)
        write_results_txt(OUTPUT_TXT_PATH, results, ordered_keys)
        return

    # Allow overriding paths via CLI (optional)
    if len(sys.argv) >= 2:
        results["InputFile"] = sys.argv[1]
    if len(sys.argv) >= 3:
        results["OutputFile"] = sys.argv[2]
    in_path = str(results["InputFile"])
    out_path = str(results["OutputFile"])

    if not os.path.exists(in_path):
        warnings.append(f"INPUT_NOT_FOUND: {in_path}")
        results["WarningsSummary"] = ";".join(warnings)
        results["RunTimeSeconds"] = _safe_float(time.time() - t0)
        write_results_txt(out_path, results, ordered_keys)
        return

    try:
        poly0 = load_vtp(in_path)
        if poly0.GetNumberOfPoints() == 0 or poly0.GetNumberOfCells() == 0:
            warnings.append("INPUT_EMPTY_GEOMETRY: loaded polydata has no points/cells.")
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        surface = clean_surface(poly0, warnings)

        # Boundary loops on (unscaled) surface for unit heuristic + canonical axes
        loops0 = extract_boundary_loops(surface, warnings)
        inlet_idx0 = infer_inlet_loop(loops0, warnings) if len(loops0) > 0 else -1

        # Units heuristic
        scale = 1.0
        if inlet_idx0 >= 0 and len(loops0) > 0:
            inlet_d = float(loops0[inlet_idx0].diam_eq)
            scale = infer_scale_factor_from_inlet_diameter(inlet_d, warnings)
        surface = maybe_scale_surface(surface, scale)
        results["ScaleFactorApplied"] = float(scale)

        # Recompute loops after scaling (for mm-based output + better inference)
        loops = extract_boundary_loops(surface, warnings)
        inlet_loop_idx = infer_inlet_loop(loops, warnings) if len(loops) > 0 else -1
        if inlet_loop_idx < 0:
            warnings.append("INLET_INFERENCE_FAILED: cannot continue anatomy inference reliably.")
            results["warning_inlet"] = True
            results["confidence_inlet"] = 0.0
        else:
            results["confidence_inlet"] = 0.7

        # Canonical frame
        if inlet_loop_idx >= 0 and len(loops) > 1:
            origin, ax, ay, az = infer_canonical_frame(surface, loops, inlet_loop_idx, warnings)
        else:
            # fallback axes
            pts_np = numpy_support.vtk_to_numpy(surface.GetPoints().GetData()).reshape(-1, 3).astype(float)
            origin, e0, e1, e2 = compute_pca_axes(pts_np)
            az = e0
            ax = e1
            ay = e2
            warnings.append("CANONICAL_FRAME_FALLBACK_PCA: limited loops; using PCA axes.")
            results["warning_left_right_ambiguity"] = True
            results["confidence_left_right_assignment"] = 0.3

        results["warning_left_right_ambiguity"] = True
        results["confidence_left_right_assignment"] = 0.5

        # Import VMTK
        vmtk_mods = try_import_vmtk_modules(warnings)
        if vmtk_mods is None:
            results["warning_vmtk_missing"] = True
            # Write best-effort output (NaNs + warnings)
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        # Centerline extraction uses possibly decimated surface for speed
        surface_for_centerlines = maybe_decimate_for_centerlines(surface, warnings)

        # Cap surface for centerlines/sections
        capped_surface, cap_ids = cap_surface_vmtk(surface_for_centerlines, vmtk_mods, warnings)
        if capped_surface is None or cap_ids is None or cap_ids.GetNumberOfIds() < 2 or inlet_loop_idx < 0:
            warnings.append("CAPPING_FAILED_OR_INSUFFICIENT_CAPS: cannot compute centerlines.")
            results["warning_vmtk_missing"] = False
            results["warning_aortic_bifurcation"] = True
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        inlet_cap_pid = match_inlet_cap_id(capped_surface, cap_ids, loops, inlet_loop_idx)
        if inlet_cap_pid is None:
            warnings.append("INLET_CAP_MATCH_FAILED: cannot locate inlet cap center.")
            results["warning_inlet"] = True
            results["confidence_inlet"] = 0.0
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        # target caps: exclude inlet; limit maximum
        target_pids = []
        for k in range(cap_ids.GetNumberOfIds()):
            pid = int(cap_ids.GetId(k))
            if pid == int(inlet_cap_pid):
                continue
            target_pids.append(pid)
        if len(target_pids) == 0:
            warnings.append("NO_TARGET_CAPS: only one opening found after capping.")
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return
        if len(target_pids) > MAX_OUTLETS_FOR_CENTERLINES:
            warnings.append(f"TARGET_CAPS_TRUNCATED: {len(target_pids)} -> {MAX_OUTLETS_FOR_CENTERLINES} (safety cap).")
            target_pids = target_pids[:MAX_OUTLETS_FOR_CENTERLINES]

        # Compute centerlines
        centerlines_raw = compute_centerlines_vmtk(
            capped_surface, cap_ids, inlet_cap_pid, target_pids, vmtk_mods, warnings
        )
        if centerlines_raw is None:
            warnings.append("CENTERLINES_FAILED: cannot proceed with centerline-based metrics.")
            results["warning_aortic_bifurcation"] = True
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        # Branch extraction + merge/resample
        centerlines_merged = branch_extract_and_merge_vmtk(centerlines_raw, vmtk_mods, warnings)

        # For sections, use capped version of full-resolution surface if possible (for robust closed sections).
        capped_full, _ = cap_surface_vmtk(surface, vmtk_mods, warnings)
        if capped_full is None:
            capped_full = surface  # fallback
            warnings.append("CAPPED_FULL_SURFACE_UNAVAILABLE: using unclipped surface for sections.")

        centerlines_with_sections = compute_centerline_sections_vmtk(capped_full, centerlines_merged, vmtk_mods, warnings)

        # Build graph
        cl_coords, neighbors = build_graph_from_centerlines(centerlines_with_sections, warnings)
        if cl_coords.shape[0] < 10 or len(neighbors) == 0:
            warnings.append("CENTERLINE_GRAPH_BUILD_FAILED: insufficient graph size.")
            results["warning_aortic_bifurcation"] = True
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        deg = undirected_degrees(neighbors)
        endpoints = [int(i) for i in range(deg.size) if deg[i] == 1]
        if len(endpoints) < 2:
            warnings.append("CENTERLINE_ENDPOINTS_INSUFFICIENT: cannot root tree properly.")
            results["warning_aortic_bifurcation"] = True
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        # Root endpoint: nearest to inlet loop centroid (in canonical mm space)
        inlet_centroid = loops[inlet_loop_idx].centroid
        root_ep, root_ep_d = find_nearest_endpoint(endpoints, cl_coords, inlet_centroid)
        if root_ep is None or root_ep_d > 20.0:
            warnings.append(f"ROOT_ENDPOINT_MATCH_WEAK: nearest endpoint distance {root_ep_d:.3f}mm; root inference uncertain.")
        root_ep = int(root_ep) if root_ep is not None else int(endpoints[0])

        dist_root, parent, children = dijkstra_root_tree(neighbors, root_ep)

        # Diameters at centerline points (area-based preferred)
        D_cl = extract_centerline_diameters(centerlines_with_sections, warnings)

        # Iliac representative endpoints (+x, -x)
        right_rep, left_rep, conf_lr = select_iliac_representative_endpoints(
            endpoints, root_ep, dist_root, cl_coords, origin, ax, ay, az, warnings
        )
        results["confidence_left_right_assignment"] = float(conf_lr)

        if right_rep is None or left_rep is None:
            warnings.append("AORTIC_BIFURCATION_FAILED: could not select iliac representatives.")
            results["warning_aortic_bifurcation"] = True
            results["confidence_aortic_bifurcation"] = 0.0
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        bif_node = lca(parent, right_rep, left_rep)
        if bif_node is None:
            warnings.append("AORTIC_BIFURCATION_LCA_FAILED.")
            results["warning_aortic_bifurcation"] = True
            results["confidence_aortic_bifurcation"] = 0.0
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        bif_node = int(bif_node)
        results["confidence_aortic_bifurcation"] = 0.8
        results["warning_aortic_bifurcation"] = False

        # Get trunk path: root -> aortic bif
        trunk_path = path_ancestor_to_descendant(parent, root_ep, bif_node)
        if trunk_path is None or len(trunk_path) < 10:
            warnings.append("TRUNK_PATH_FAILED: cannot build aortic trunk path.")
            results["warning_aortic_bifurcation"] = True
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        # Determine side child nodes (first node after bif on path to each rep) for subtree collection
        right_path_from_bif = path_ancestor_to_descendant(parent, bif_node, right_rep)
        left_path_from_bif = path_ancestor_to_descendant(parent, bif_node, left_rep)
        if right_path_from_bif is None or len(right_path_from_bif) < 2 or left_path_from_bif is None or len(left_path_from_bif) < 2:
            warnings.append("SIDE_PATHS_FROM_BIF_FAILED: cannot isolate iliac subtrees.")
            results["warning_aortic_bifurcation"] = True
            results["WarningsSummary"] = ";".join(warnings)
            results["RunTimeSeconds"] = _safe_float(time.time() - t0)
            write_results_txt(out_path, results, ordered_keys)
            return

        right_side_child = int(right_path_from_bif[1])
        left_side_child = int(left_path_from_bif[1])

        # Common iliac split (internal/external) and labeling
        (right_ci_split, right_eia_leaf, right_iia_leaf, conf_ci_r) = infer_common_iliac_split_and_external_internal(
            parent, children, bif_node, right_side_child, dist_root, cl_coords, origin, ax, ay, az, warnings, "RIGHT"
        )
        (left_ci_split, left_eia_leaf, left_iia_leaf, conf_ci_l) = infer_common_iliac_split_and_external_internal(
            parent, children, bif_node, left_side_child, dist_root, cl_coords, origin, ax, ay, az, warnings, "LEFT"
        )
        results["confidence_common_iliac_right"] = float(conf_ci_r)
        results["confidence_common_iliac_left"] = float(conf_ci_l)

        # Common iliac lengths
        if right_ci_split is not None and np.isfinite(dist_root[right_ci_split]):
            results["Right_common_iliac_length"] = float(dist_root[right_ci_split] - dist_root[bif_node])
            results["warning_right_common_iliac_bifurcation"] = False
        else:
            results["Right_common_iliac_length"] = _nan()
            results["warning_right_common_iliac_bifurcation"] = True

        if left_ci_split is not None and np.isfinite(dist_root[left_ci_split]):
            results["Left_common_iliac_length"] = float(dist_root[left_ci_split] - dist_root[bif_node])
            results["warning_left_common_iliac_bifurcation"] = False
        else:
            results["Left_common_iliac_length"] = _nan()
            results["warning_left_common_iliac_bifurcation"] = True

        # Build right CIA path for diameters: aortic bif -> right CI split (if present), else bif -> right rep leaf
        def compute_segment_profile(ancestor: int, descendant: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            path = path_ancestor_to_descendant(parent, ancestor, descendant)
            if path is None:
                return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0, 3), dtype=float)
            s = dist_root[np.array(path, dtype=int)] - dist_root[int(ancestor)]
            d = D_cl[np.array(path, dtype=int)] if D_cl.size >= max(path)+1 else np.full((len(path),), np.nan, dtype=float)
            c = cl_coords[np.array(path, dtype=int)]
            return s.astype(float), d.astype(float), c.astype(float)

        # Right common iliac diameters at 0,10,15,20 from aortic bif
        if right_ci_split is not None:
            s_rci, d_rci, _ = compute_segment_profile(bif_node, right_ci_split)
        else:
            s_rci, d_rci, _ = compute_segment_profile(bif_node, right_rep)

        if s_rci.size > 0:
            results["Right_common_iliac_D0"] = _sample_profile_at(s_rci, d_rci, 0.0, window=1.0)
            results["Right_common_iliac_D10"] = _sample_profile_at(s_rci, d_rci, 10.0, window=1.0)
            results["Right_common_iliac_D15"] = _sample_profile_at(s_rci, d_rci, 15.0, window=1.0)
            results["Right_common_iliac_D20"] = _sample_profile_at(s_rci, d_rci, 20.0, window=1.0)
        else:
            results["warning_right_common_iliac_bifurcation"] = True

        # Left common iliac diameters at 0,10,15,20 from aortic bif
        if left_ci_split is not None:
            s_lci, d_lci, _ = compute_segment_profile(bif_node, left_ci_split)
        else:
            s_lci, d_lci, _ = compute_segment_profile(bif_node, left_rep)

        if s_lci.size > 0:
            results["Left_common_iliac_D0"] = _sample_profile_at(s_lci, d_lci, 0.0, window=1.0)
            results["Left_common_iliac_D10"] = _sample_profile_at(s_lci, d_lci, 10.0, window=1.0)
            results["Left_common_iliac_D15"] = _sample_profile_at(s_lci, d_lci, 15.0, window=1.0)
            results["Left_common_iliac_D20"] = _sample_profile_at(s_lci, d_lci, 20.0, window=1.0)
        else:
            results["warning_left_common_iliac_bifurcation"] = True

        # External iliac segments and metrics
        def compute_external_iliac_metrics(side: str, ci_split: Optional[int], eia_leaf: Optional[int]) -> Dict[str, float]:
            out = {
                "avg_distal20": _nan(),
                "min_diam": _nan(),
                "tortuosity": _nan(),
                "max_ang": _nan(),
            }
            if ci_split is None or eia_leaf is None:
                return out
            path = path_ancestor_to_descendant(parent, int(ci_split), int(eia_leaf))
            if path is None or len(path) < 5:
                return out
            s = dist_root[np.array(path, dtype=int)] - dist_root[int(ci_split)]
            d = D_cl[np.array(path, dtype=int)]
            c = cl_coords[np.array(path, dtype=int)]

            length = float(s[-1]) if s.size > 0 else _nan()
            if not np.isfinite(length) or length <= 0:
                return out
            # distal 20mm average
            a0 = max(0.0, length - 20.0)
            out["avg_distal20"] = _avg_profile_over(s, d, a0, length)
            out["min_diam"] = _min_profile_over(s, d, 0.0, length)
            # tortuosity
            chord = float(np.linalg.norm(c[-1] - c[0]))
            if chord > 1e-6:
                out["tortuosity"] = float(length / chord)
            # max angulation
            out["max_ang"] = max_angulation_along_path(c, s, ANGULATION_SEGMENT_MM)
            return out

        right_eia_metrics = compute_external_iliac_metrics("RIGHT", right_ci_split, right_eia_leaf)
        left_eia_metrics = compute_external_iliac_metrics("LEFT", left_ci_split, left_eia_leaf)

        # Interpret requested EIA diameter as distal20mm avg diameter
        results["Right_external_iliac_diameter"] = right_eia_metrics["avg_distal20"]
        results["Left_external_iliac_diameter"] = left_eia_metrics["avg_distal20"]

        results["Right_external_iliac_distal20mm_avg_diameter"] = right_eia_metrics["avg_distal20"]
        results["Left_external_iliac_distal20mm_avg_diameter"] = left_eia_metrics["avg_distal20"]

        results["Right_external_iliac_min_diameter"] = right_eia_metrics["min_diam"]
        results["Left_external_iliac_min_diameter"] = left_eia_metrics["min_diam"]

        results["Right_external_iliac_tortuosity"] = right_eia_metrics["tortuosity"]
        results["Left_external_iliac_tortuosity"] = left_eia_metrics["tortuosity"]

        results["Right_external_iliac_max_angulation_deg"] = right_eia_metrics["max_ang"]
        results["Left_external_iliac_max_angulation_deg"] = left_eia_metrics["max_ang"]

        results["warning_right_external_iliac"] = not np.isfinite(results["Right_external_iliac_diameter"])
        results["warning_left_external_iliac"] = not np.isfinite(results["Left_external_iliac_diameter"])
        results["confidence_external_iliac_right"] = float(0.8 if not results["warning_right_external_iliac"] else 0.2)
        results["confidence_external_iliac_left"] = float(0.8 if not results["warning_left_external_iliac"] else 0.2)

        # Renal detection along trunk
        (lowest_renal_origin, left_renal_origin, right_renal_origin, conf_renal) = detect_renal_branches_on_trunk(
            parent, children, trunk_path, cl_coords, D_cl, dist_root, origin, ax, ay, az, warnings
        )
        results["confidence_lowest_renal"] = float(conf_renal)
        if lowest_renal_origin is None:
            results["warning_lowest_renal"] = True
            results["confidence_lowest_renal"] = 0.0
        else:
            results["warning_lowest_renal"] = False

        # Define proximal neck origin (inferior edge of lowest renal ostium plane; approximated)
        # We approximate renal diameter near origin by sampling diam along the renal branch child if possible.
        s_origin_global = float("nan")
        if lowest_renal_origin is not None:
            # sample renal branch diameter: take median of diam at branch origin +/- small neighborhood on trunk (proxy)
            # Better: take median diam of side branch child path for first few mm.
            # Find a likely renal branch child: any child not in trunk_set
            trunk_set = set(trunk_path)
            renal_child = None
            for ch in children.get(int(lowest_renal_origin), []):
                if ch not in trunk_set:
                    renal_child = int(ch)
                    break
            renal_diam = float("nan")
            if renal_child is not None:
                # walk down renal branch for ~8mm
                walk = [renal_child]
                cur = renal_child
                length = 0.0
                while True:
                    nxts = children.get(cur, [])
                    if len(nxts) == 0:
                        break
                    nxt = max(nxts, key=lambda t: dist_root[t])
                    length += float(np.linalg.norm(cl_coords[nxt] - cl_coords[cur]))
                    walk.append(nxt)
                    cur = nxt
                    if length >= 8.0 or len(walk) > 40:
                        break
                dvals = [D_cl[w] for w in walk if w < D_cl.size and np.isfinite(D_cl[w])]
                if len(dvals) > 0:
                    renal_diam = float(np.median(dvals))

            # Inferior edge offset
            offset = 0.0
            if np.isfinite(renal_diam) and renal_diam > 0:
                offset = float(RENAL_INFERIOR_EDGE_OFFSET_FACTOR * renal_diam)
            else:
                warnings.append("RENAL_DIAMETER_UNKNOWN_FOR_OFFSET: using zero offset for inferior edge approximation.")
            s_origin_global = float(dist_root[int(lowest_renal_origin)] + offset)
        else:
            # Fallback: use inlet as origin (not per strict definition)
            warnings.append("RENAL_REFERENCE_FALLBACK_INLET: lowest renal not found; using inlet loop centroid level as reference (metrics flagged unreliable).")
            # approximate origin distance as root zero
            s_origin_global = 0.0
            results["warning_lowest_renal"] = True
            results["confidence_lowest_renal"] = 0.0

        # Aortic trunk profile for neck/aneurysm: from origin to aortic bif
        trunk_nodes = np.array(trunk_path, dtype=int)
        s_trunk_global = dist_root[trunk_nodes]
        D_trunk = D_cl[trunk_nodes]
        # Convert to relative from neck origin
        if np.isfinite(s_origin_global):
            rel_sel = s_trunk_global >= (s_origin_global - 1e-6)
            s_trunk_rel = s_trunk_global[rel_sel] - s_origin_global
            D_trunk_rel = D_trunk[rel_sel]
            coords_trunk_rel = cl_coords[trunk_nodes[rel_sel]]
        else:
            s_trunk_rel = np.zeros((0,), dtype=float)
            D_trunk_rel = np.zeros((0,), dtype=float)
            coords_trunk_rel = np.zeros((0, 3), dtype=float)

        # Proximal neck diameters at 0,5,10,15mm (best-effort)
        if s_trunk_rel.size > 0:
            results["Proximal_neck_D0"] = _sample_profile_at(s_trunk_rel, D_trunk_rel, 0.0, window=1.0)
            results["Proximal_neck_D5"] = _sample_profile_at(s_trunk_rel, D_trunk_rel, 5.0, window=1.0)
            results["Proximal_neck_D10"] = _sample_profile_at(s_trunk_rel, D_trunk_rel, 10.0, window=1.0)
            results["Proximal_neck_D15"] = _sample_profile_at(s_trunk_rel, D_trunk_rel, 15.0, window=1.0)
        else:
            warnings.append("NECK_PROFILE_EMPTY: cannot compute proximal neck diameters.")
            results["warning_lowest_renal"] = True

        # Aneurysm start detection (derivative/persistence-based)
        aneurysm_start_rel, conf_as = detect_aneurysm_start_distance(
            s_trunk_rel, D_trunk_rel, step_mm=float(CENTERLINE_RESAMPLE_STEP_MM), warnings=warnings
        )
        results["confidence_aneurysm_start"] = float(conf_as)
        if not np.isfinite(aneurysm_start_rel):
            results["warning_aneurysm_start"] = True
            results["Proximal_neck_length"] = _nan()
        else:
            results["warning_aneurysm_start"] = False
            results["Proximal_neck_length"] = float(aneurysm_start_rel)

        # Within sac flags for D0/D5/D10/D15
        def within_sac(k_mm: float) -> object:
            if not np.isfinite(aneurysm_start_rel):
                return None
            return bool(k_mm >= float(aneurysm_start_rel))

        results["D0_within_sac"] = within_sac(0.0)
        results["D5_within_sac"] = within_sac(5.0)
        results["D10_within_sac"] = within_sac(10.0)
        results["D15_within_sac"] = within_sac(15.0)

        # Lengths from lowest renal to aortic bif and iliac bifurcations
        if np.isfinite(s_origin_global):
            results["Length_lowest_renal_aortic_bifurcation"] = float(dist_root[bif_node] - s_origin_global)
            if right_ci_split is not None:
                results["Length_lowest_renal_iliac_bifurcation_right"] = float(dist_root[int(right_ci_split)] - s_origin_global)
            if left_ci_split is not None:
                results["Length_lowest_renal_iliac_bifurcation_left"] = float(dist_root[int(left_ci_split)] - s_origin_global)

            if right_ci_split is None:
                results["Length_lowest_renal_iliac_bifurcation_right"] = _nan()
            if left_ci_split is None:
                results["Length_lowest_renal_iliac_bifurcation_left"] = _nan()
        else:
            results["Length_lowest_renal_aortic_bifurcation"] = _nan()
            results["Length_lowest_renal_iliac_bifurcation_right"] = _nan()
            results["Length_lowest_renal_iliac_bifurcation_left"] = _nan()

        # Maximum aneurysm diameter along aortic trunk sac region (aneurysm_start -> aortic bif)
        if s_trunk_rel.size > 0:
            if np.isfinite(aneurysm_start_rel):
                sac_sel = (s_trunk_rel >= aneurysm_start_rel) & np.isfinite(D_trunk_rel)
                if np.any(sac_sel):
                    results["Maximum_aneurysm_diameter"] = float(np.max(D_trunk_rel[sac_sel]))
                else:
                    results["Maximum_aneurysm_diameter"] = _nan()
            else:
                # best-effort if aneurysm start unknown: max along infrarenal segment (all available)
                valid = np.isfinite(D_trunk_rel)
                results["Maximum_aneurysm_diameter"] = float(np.max(D_trunk_rel[valid])) if np.any(valid) else _nan()
                warnings.append("MAX_ANEURYSM_DIAMETER_FALLBACK: aneurysm start unknown; used max diameter along trunk segment (origin->bif).")

        # Inlet / landmark warnings and confidences
        results["warning_inlet"] = bool(inlet_loop_idx < 0)
        results["warning_left_right_ambiguity"] = True  # fundamental without reference
        results["confidence_inlet"] = float(results["confidence_inlet"]) if inlet_loop_idx >= 0 else 0.0

        # Add warnings summary
        results["WarningsSummary"] = ";".join(warnings)

        # Runtime
        results["RunTimeSeconds"] = _safe_float(time.time() - t0)

        # Write
        write_results_txt(out_path, results, ordered_keys)

    except Exception:
        warnings.append("UNHANDLED_EXCEPTION:\n" + traceback.format_exc())
        results["WarningsSummary"] = ";".join(warnings)
        results["RunTimeSeconds"] = _safe_float(time.time() - t0)
        try:
            write_results_txt(OUTPUT_TXT_PATH, results, ordered_keys)
        except Exception:
            pass


if __name__ == "__main__":
    main()
