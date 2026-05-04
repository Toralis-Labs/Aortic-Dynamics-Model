from __future__ import annotations

import argparse
from collections import Counter
import heapq
import importlib
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import vtk

from src.common.geometry import (
    as_point,
    concatenate_polylines,
    cumulative_arclength,
    distance,
    equivalent_diameter_from_area,
    point_at_arclength,
    polygon_area_normal,
    tangent_at_arclength,
    unit,
)
from src.common.json_io import read_json, write_json
from src.common.paths import build_pipeline_paths
from src.common.vtk_helpers import (
    add_int_cell_array,
    add_uchar3_cell_array,
    build_polyline_polydata,
    build_segment_point_locator,
    cell_centers,
    cell_points,
    clone_geometry_only,
    get_cell_array,
    read_vtp,
    segment_color,
    triangle_area,
    write_vtp,
)


PROXIMAL_BOUNDARY_SELECTION_ALGORITHM = "vmtk_branch_clip_group_boundary_v1"
MESH_PARTITION_ALGORITHM = "vmtk_branch_clipper_v1"
SURFACE_ASSIGNMENT_ALGORITHM = "vmtk_branch_group_segmentation_v1"
TUNNEL_ASSIGNMENT_ALGORITHM = "face_map_outlet_routes_parent_junction_v1"
_STEP2_VMTK_REEXEC_ENV = "STEP2_VMTK_REEXEC_ACTIVE"
_STEP2_VMTK_PYTHON_ENV = "STEP2_VMTK_PYTHON"
_VMTK_REQUIRED_SCRIPT_CLASSES = ("vmtkBranchExtractor", "vmtkBranchClipper", "vmtkBranchSections")
EXPECTED_ANATOMY_NAMES = {
    "abdominal_aorta_inlet",
    "celiac_artery",
    "celiac_branch",
    "superior_mesenteric_artery",
    "left_renal_artery",
    "right_renal_artery",
    "inferior_mesenteric_artery",
    "left_external_iliac",
    "right_external_iliac",
    "left_internal_iliac",
    "right_internal_iliac",
}


class Step2Failure(RuntimeError):
    pass


class Step2RequiresReview(Step2Failure):
    pass


def _normalize_path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _iter_vmtk_python_candidates(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(raw: str | Path | None) -> None:
        if not raw:
            return
        path = Path(raw).expanduser()
        try:
            path = path.resolve()
        except Exception:
            return
        if not path.is_file():
            return
        key = _normalize_path_key(path)
        if key in seen or key == _normalize_path_key(sys.executable):
            return
        seen.add(key)
        candidates.append(path)

    add(os.environ.get(_STEP2_VMTK_PYTHON_ENV))
    add(Path(os.environ.get("CONDA_PREFIX", "")) / "python.exe")
    add(project_root / ".tools" / "m" / "envs" / "vmtk-step2" / "python.exe")
    add(project_root / ".tools" / "micromamba_root" / "envs" / "vmtk-step2" / "python.exe")

    home = Path.home()
    for root in (home / "miniconda3", home / "anaconda3", home / "mambaforge", home / "miniforge3"):
        for env_name in ("vmtk-step2", "vmtk_env", "vmtk", "simvascular", "sv"):
            add(root / "envs" / env_name / "python.exe")
        envs_dir = root / "envs"
        if envs_dir.is_dir():
            for pattern in ("*vmtk*", "*vascular*"):
                for match in envs_dir.glob(pattern):
                    add(match / "python.exe")
    return candidates


def _python_supports_vmtk_branch_tools(python_exe: str | Path) -> bool:
    probe = """
import importlib
import sys
try:
    mod = importlib.import_module("vmtk.vmtkscripts")
    required = ("vmtkBranchExtractor", "vmtkBranchClipper", "vmtkBranchSections")
    sys.exit(0 if all(hasattr(mod, name) for name in required) else 1)
except Exception:
    sys.exit(1)
""".strip()
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0


def _current_python_supports_vmtk_branch_tools() -> bool:
    try:
        mod = importlib.import_module("vmtk.vmtkscripts")
    except Exception:
        return False
    return all(hasattr(mod, name) for name in _VMTK_REQUIRED_SCRIPT_CLASSES)


def _maybe_reexec_with_vmtk_python(project_root: Path) -> None:
    if os.environ.get(_STEP2_VMTK_REEXEC_ENV) == "1":
        return
    if _current_python_supports_vmtk_branch_tools():
        return
    for candidate in _iter_vmtk_python_candidates(project_root):
        if not _python_supports_vmtk_branch_tools(candidate):
            continue
        entrypoint = Path(sys.argv[0]).resolve()
        env = os.environ.copy()
        env[_STEP2_VMTK_REEXEC_ENV] = "1"
        env[_STEP2_VMTK_PYTHON_ENV] = str(candidate)
        sys.stderr.write(f"INFO: relaunching STEP2 with VMTK-capable interpreter: {candidate}\n")
        completed = subprocess.run([str(candidate), str(entrypoint), *sys.argv[1:]], env=env, check=False)
        raise SystemExit(int(completed.returncode))


def _import_vmtk_scripts() -> tuple[Any, Dict[str, Any]]:
    try:
        mod = importlib.import_module("vmtk.vmtkscripts")
    except Exception as exc:
        raise Step2RequiresReview(
            "VMTK branch tooling is required for STEP2 but could not be imported. "
            "Expected vmtk.vmtkscripts with vmtkBranchExtractor and vmtkBranchClipper."
        ) from exc
    missing = [name for name in _VMTK_REQUIRED_SCRIPT_CLASSES if not hasattr(mod, name)]
    if missing:
        raise Step2RequiresReview("VMTK branch tooling is missing required script class(es): " + ", ".join(missing))
    return mod, {
        "vmtk_import_source": "vmtk.vmtkscripts",
        "vmtk_python": str(Path(sys.executable).resolve()),
        "vmtk_reexec_active": os.environ.get(_STEP2_VMTK_REEXEC_ENV) == "1",
    }


@dataclass
class NetworkEdge:
    edge_id: int
    cell_id: int
    start_node: int
    end_node: int
    points: np.ndarray
    length: float


@dataclass
class SegmentBoundaryProfile:
    source_type: str
    centroid: np.ndarray
    normal: np.ndarray
    area: float
    equivalent_diameter: Optional[float]
    major_diameter: Optional[float]
    minor_diameter: Optional[float]
    arclength: float
    confidence: float
    method: str
    attempts: list[Dict[str, Any]] = field(default_factory=list)
    loop_points: Optional[np.ndarray] = None

    def to_contract(self) -> Dict[str, Any]:
        row = {
            "source_type": self.source_type,
            "centroid": self.centroid.tolist(),
            "normal": unit(self.normal).tolist(),
            "area": float(self.area),
            "equivalent_diameter": self.equivalent_diameter,
            "major_diameter": self.major_diameter,
            "minor_diameter": self.minor_diameter,
            "arclength": float(self.arclength),
            "confidence": float(self.confidence),
            "method": self.method,
            "boundary_source": self.source_type,
            "selection_algorithm": PROXIMAL_BOUNDARY_SELECTION_ALGORITHM,
            "attempts": self.attempts,
        }
        if self.loop_points is not None:
            row.update(
                {
                    "cut_method": "surface_plane_intersection_loop",
                    "cut_loop_point_count": int(np.asarray(self.loop_points).shape[0]),
                    "loop_area": float(self.area),
                    "mesh_partition_algorithm": MESH_PARTITION_ALGORITHM,
                }
            )
        return row


@dataclass
class GeometrySegment:
    segment_id: int
    name_hint: str
    segment_type: str
    proximal_node: int
    distal_node: int
    edge_ids: list[int]
    points: np.ndarray
    parent_segment_id: Optional[int] = None
    child_segment_ids: list[int] = field(default_factory=list)
    terminal_face_id: Optional[int] = None
    terminal_face_name: Optional[str] = None
    descendant_terminal_names: list[str] = field(default_factory=list)
    tunnel_source_outlets: list[str] = field(default_factory=list)
    parent_junction_node_id: Optional[int] = None
    distal_junction_node_ids: list[int] = field(default_factory=list)
    outlet_route_node_paths: dict[str, list[int]] = field(default_factory=dict)
    vmtk_group_ids: list[int] = field(default_factory=list)
    cell_count: int = 0
    fallback_cell_count: int = 0
    proximal_boundary: Optional[SegmentBoundaryProfile] = None
    proximal_boundary_attempts: list[Dict[str, Any]] = field(default_factory=list)
    proximal_boundary_warning: Optional[str] = None
    distal_boundaries: list[SegmentBoundaryProfile] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(cumulative_arclength(self.points)[-1]) if self.points.shape[0] else 0.0

    @property
    def effective_proximal_point(self) -> np.ndarray:
        if self.proximal_boundary is not None:
            return np.asarray(self.proximal_boundary.centroid, dtype=float)
        if self.points.shape[0]:
            return np.asarray(self.points[0], dtype=float)
        return np.zeros(3, dtype=float)


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def _face_map_by_id(raw: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            face_id = int(key)
        except Exception:
            continue
        if isinstance(value, dict):
            row = dict(value)
        else:
            row = {"name": str(value)}
        row["face_id"] = face_id
        out[face_id] = row
    return out


def _face_name(face_map: Dict[int, Dict[str, Any]], face_id: int) -> str:
    return str(face_map.get(int(face_id), {}).get("name", f"face_{int(face_id)}"))


def _find_face_id_by_name(face_map: Dict[int, Dict[str, Any]], name: str) -> Optional[int]:
    target = str(name).strip().lower()
    for face_id, row in face_map.items():
        if str(row.get("name", "")).strip().lower() == target:
            return int(face_id)
    return None


def _validate_expected_anatomy(face_map: Dict[int, Dict[str, Any]]) -> None:
    present = {str(row.get("name", "")).strip().lower() for row in face_map.values()}
    missing = sorted(EXPECTED_ANATOMY_NAMES - present)
    if missing:
        raise Step2Failure("Face map is missing required anatomy entries: " + ", ".join(missing))


def _read_network_edges(network_pd: vtk.vtkPolyData) -> tuple[dict[int, NetworkEdge], dict[int, np.ndarray]]:
    cd = network_pd.GetCellData()
    start_arr = cd.GetArray("StartNodeId")
    end_arr = cd.GetArray("EndNodeId")
    edge_arr = cd.GetArray("EdgeId")
    length_arr = cd.GetArray("EdgeLength")
    if start_arr is None or end_arr is None:
        raise Step2Failure("STEP1 centerline_network.vtp is missing StartNodeId/EndNodeId arrays.")

    edges: dict[int, NetworkEdge] = {}
    node_coords: dict[int, np.ndarray] = {}
    for cell_id in range(network_pd.GetNumberOfCells()):
        cell = network_pd.GetCell(cell_id)
        if cell.GetNumberOfPoints() < 2:
            continue
        point_ids = cell.GetPointIds()
        pts = np.asarray([network_pd.GetPoint(point_ids.GetId(i)) for i in range(point_ids.GetNumberOfIds())], dtype=float)
        start_node = int(start_arr.GetTuple1(cell_id))
        end_node = int(end_arr.GetTuple1(cell_id))
        edge_id = int(edge_arr.GetTuple1(cell_id)) if edge_arr is not None else int(cell_id)
        length = float(length_arr.GetTuple1(cell_id)) if length_arr is not None else float(cumulative_arclength(pts)[-1])
        edges[edge_id] = NetworkEdge(
            edge_id=edge_id,
            cell_id=int(cell_id),
            start_node=start_node,
            end_node=end_node,
            points=pts,
            length=length,
        )
        node_coords.setdefault(start_node, pts[0].copy())
        node_coords.setdefault(end_node, pts[-1].copy())

    if not edges:
        raise Step2Failure("STEP1 centerline_network.vtp did not contain usable polyline edges.")
    return edges, node_coords


def _build_graph(edges: dict[int, NetworkEdge]) -> dict[int, dict[int, tuple[float, int]]]:
    graph: dict[int, dict[int, tuple[float, int]]] = {}
    for edge in edges.values():
        graph.setdefault(edge.start_node, {})[edge.end_node] = (float(edge.length), int(edge.edge_id))
        graph.setdefault(edge.end_node, {})[edge.start_node] = (float(edge.length), int(edge.edge_id))
    return graph


def _dijkstra(graph: dict[int, dict[int, tuple[float, int]]], root: int) -> tuple[dict[int, float], dict[int, Optional[int]]]:
    dist: dict[int, float] = {int(root): 0.0}
    prev: dict[int, Optional[int]] = {int(root): None}
    queue: list[tuple[float, int]] = [(0.0, int(root))]
    while queue:
        d, node = heapq.heappop(queue)
        if d > dist.get(node, float("inf")) + 1.0e-9:
            continue
        for nbr, (length, _) in graph.get(node, {}).items():
            nd = d + float(length)
            if nd < dist.get(nbr, float("inf")):
                dist[nbr] = nd
                prev[nbr] = node
                heapq.heappush(queue, (nd, nbr))
    return dist, prev


def _path_to_root(prev: dict[int, Optional[int]], root: int, node: int) -> list[int]:
    if node not in prev:
        return []
    out = [int(node)]
    cur = int(node)
    while cur != int(root):
        parent = prev.get(cur)
        if parent is None:
            return []
        cur = int(parent)
        out.append(cur)
    out.reverse()
    return out


def _common_prefix(paths: list[list[int]]) -> list[int]:
    if not paths:
        return []
    prefix: list[int] = []
    for values in zip(*paths):
        if len(set(values)) == 1:
            prefix.append(int(values[0]))
        else:
            break
    return prefix


def _edge_for_nodes(graph: dict[int, dict[int, tuple[float, int]]], a: int, b: int) -> int:
    try:
        return int(graph[int(a)][int(b)][1])
    except KeyError as exc:
        raise Step2Failure(f"Missing centerline edge between node {a} and node {b}.") from exc


def _polyline_for_node_path(node_path: list[int], graph: dict[int, dict[int, tuple[float, int]]], edges: dict[int, NetworkEdge]) -> tuple[np.ndarray, list[int]]:
    parts: list[np.ndarray] = []
    edge_ids: list[int] = []
    for a, b in zip(node_path[:-1], node_path[1:]):
        edge_id = _edge_for_nodes(graph, a, b)
        edge = edges[edge_id]
        pts = edge.points
        if edge.start_node == int(a) and edge.end_node == int(b):
            oriented = pts
        elif edge.start_node == int(b) and edge.end_node == int(a):
            oriented = pts[::-1]
        else:
            raise Step2Failure(f"Edge {edge_id} does not connect expected nodes {a}->{b}.")
        parts.append(oriented)
        edge_ids.append(edge_id)
    return concatenate_polylines(parts), edge_ids


def _nearest_node(point: Iterable[float], node_coords: dict[int, np.ndarray]) -> tuple[int, float]:
    p = as_point(point)
    best_node = -1
    best_dist = float("inf")
    for node_id, coord in node_coords.items():
        d = distance(p, coord)
        if d < best_dist:
            best_node = int(node_id)
            best_dist = float(d)
    return best_node, best_dist


def _map_face_terminations_to_nodes(
    step1_metadata: Dict[str, Any],
    face_map: Dict[int, Dict[str, Any]],
    node_coords: dict[int, np.ndarray],
) -> Dict[int, Dict[str, Any]]:
    mapped: Dict[int, Dict[str, Any]] = {}
    for term_index, term in enumerate(step1_metadata.get("terminations", [])):
        face_id = term.get("face_id")
        if face_id is None:
            continue
        face_id_i = int(face_id)
        if face_id_i not in face_map:
            continue
        center = term.get("center")
        if center is None:
            continue
        node_id, node_distance = _nearest_node(center, node_coords)
        mapped[face_id_i] = {
            "face_id": face_id_i,
            "face_name": _face_name(face_map, face_id_i),
            "termination_index": int(term_index),
            "terminal_node_id": int(node_id),
            "node_distance": float(node_distance),
            "termination": dict(term),
        }
    return mapped


def _resolve_inlet(
    face_map: Dict[int, Dict[str, Any]],
    face_node_map: Dict[int, Dict[str, Any]],
) -> tuple[int, Dict[str, Any]]:
    inlet_face_id = _find_face_id_by_name(face_map, "abdominal_aorta_inlet")
    if inlet_face_id is None:
        raise Step2Failure("Aortic inlet cannot be resolved: face map has no abdominal_aorta_inlet entry.")
    inlet_map = face_node_map.get(int(inlet_face_id))
    if inlet_map is None:
        raise Step2Failure(f"Aortic inlet cannot be resolved: face {inlet_face_id} has no STEP1 termination.")
    return int(inlet_map["terminal_node_id"]), inlet_map


def _resolve_aortic_bifurcation_node(
    face_map: Dict[int, Dict[str, Any]],
    face_node_map: Dict[int, Dict[str, Any]],
    graph: dict[int, dict[int, tuple[float, int]]],
    root_node: int,
) -> tuple[int, dict[str, Any]]:
    dist, prev = _dijkstra(graph, root_node)
    iliac_names = {
        "left_external_iliac",
        "left_internal_iliac",
        "right_external_iliac",
        "right_internal_iliac",
    }
    iliac_faces: list[int] = []
    missing_names: list[str] = []
    for name in sorted(iliac_names):
        face_id = _find_face_id_by_name(face_map, name)
        if face_id is None or face_id not in face_node_map:
            missing_names.append(name)
            continue
        iliac_faces.append(int(face_id))

    if len(iliac_faces) < 2:
        raise Step2Failure("Aortic end cannot be resolved: fewer than two iliac outlet terminations were mapped.")

    paths: list[list[int]] = []
    for face_id in iliac_faces:
        node_id = int(face_node_map[face_id]["terminal_node_id"])
        path = _path_to_root(prev, root_node, node_id)
        if not path:
            raise Step2Failure(f"Aortic end cannot be resolved: no route from inlet to face {face_id}.")
        paths.append(path)

    prefix = _common_prefix(paths)
    if len(prefix) < 2:
        raise Step2Failure("Aortic end cannot be resolved: iliac routes do not share a valid trunk.")

    bif_node = int(prefix[-1])
    if bif_node == int(root_node) or len(graph.get(bif_node, {})) < 3:
        raise Step2Failure("Aortic end cannot be resolved: common iliac split node is not a valid junction.")

    detail = {
        "method": "deepest_common_node_across_mapped_iliac_routes",
        "used_iliac_faces": [
            {
                "face_id": int(fid),
                "face_name": _face_name(face_map, int(fid)),
                "terminal_node_id": int(face_node_map[int(fid)]["terminal_node_id"]),
            }
            for fid in iliac_faces
        ],
        "missing_iliac_names": missing_names,
        "path_count": len(paths),
    }
    return bif_node, detail


def _build_outlet_routes(
    face_map: Dict[int, Dict[str, Any]],
    face_node_map: Dict[int, Dict[str, Any]],
    graph: dict[int, dict[int, tuple[float, int]]],
    root_node: int,
) -> dict[str, dict[str, Any]]:
    _, prev = _dijkstra(graph, root_node)
    routes: dict[str, dict[str, Any]] = {}
    for face_id, row in sorted(face_node_map.items()):
        face_name = _face_name(face_map, int(face_id))
        if face_name == "abdominal_aorta_inlet":
            continue
        terminal_node = int(row["terminal_node_id"])
        node_path = _path_to_root(prev, root_node, terminal_node)
        if not node_path:
            raise Step2Failure(f"Outlet tunnel cannot be resolved: no route from inlet to {face_name}.")
        edge_ids = [_edge_for_nodes(graph, a, b) for a, b in zip(node_path[:-1], node_path[1:])]
        routes[face_name] = {
            "face_id": int(face_id),
            "terminal_node_id": terminal_node,
            "node_path": [int(v) for v in node_path],
            "edge_ids": [int(v) for v in edge_ids],
        }

    required_outlets = sorted(EXPECTED_ANATOMY_NAMES - {"abdominal_aorta_inlet"})
    missing_routes = [name for name in required_outlets if name not in routes]
    if missing_routes:
        raise Step2Failure("STEP1 metadata is missing required outlet tunnel routes: " + ", ".join(missing_routes))
    return routes


def _collect_descendant_terminal_names(
    node: int,
    child_map: dict[int, list[int]],
    terminal_node_to_face: dict[int, int],
    face_map: Dict[int, Dict[str, Any]],
) -> list[str]:
    out: list[str] = []
    stack = [int(node)]
    while stack:
        cur = stack.pop()
        face_id = terminal_node_to_face.get(cur)
        if face_id is not None:
            out.append(_face_name(face_map, face_id))
        stack.extend(child_map.get(cur, []))
    return sorted(set(out))


def _name_hint_for_segment(
    segment_type: str,
    terminal_face_name: Optional[str],
    descendant_names: list[str],
) -> str:
    if segment_type == "aorta_trunk":
        return "abdominal_aorta_trunk"
    names = set(descendant_names)
    if terminal_face_name:
        return str(terminal_face_name)
    if {"left_external_iliac", "left_internal_iliac"}.issubset(names):
        return "left_common_iliac_candidate"
    if {"right_external_iliac", "right_internal_iliac"}.issubset(names):
        return "right_common_iliac_candidate"
    if names and all("celiac" in n for n in names):
        return "celiac_proximal_candidate"
    if len(names) == 1:
        return f"{next(iter(names))}_proximal_candidate"
    return "topology_segment"


def _build_segments(
    root_node: int,
    bif_node: int,
    graph: dict[int, dict[int, tuple[float, int]]],
    edges: dict[int, NetworkEdge],
    face_map: Dict[int, Dict[str, Any]],
    face_node_map: Dict[int, Dict[str, Any]],
    outlet_routes: dict[str, dict[str, Any]],
) -> tuple[list[GeometrySegment], list[int]]:
    dist, prev = _dijkstra(graph, root_node)
    aorta_node_path = _path_to_root(prev, root_node, bif_node)
    if len(aorta_node_path) < 2:
        raise Step2Failure("A single trustworthy aorta trunk cannot be authored from inlet to bifurcation.")
    aorta_points, aorta_edge_ids = _polyline_for_node_path(aorta_node_path, graph, edges)
    if aorta_points.shape[0] < 2:
        raise Step2Failure("A single trustworthy aorta trunk cannot be authored: empty trunk polyline.")

    directed_edges: list[tuple[float, int, int, int]] = []
    child_map: dict[int, list[int]] = {}
    for edge in edges.values():
        a, b = int(edge.start_node), int(edge.end_node)
        if dist.get(a, float("inf")) <= dist.get(b, float("inf")):
            parent, child = a, b
        else:
            parent, child = b, a
        directed_edges.append((float(dist.get(parent, 0.0)), int(edge.edge_id), parent, child))
        child_map.setdefault(parent, []).append(child)

    terminal_node_to_face = {
        int(row["terminal_node_id"]): int(face_id)
        for face_id, row in face_node_map.items()
        if int(face_id) in face_map
    }
    edge_to_outlet_names: dict[int, list[str]] = {}
    for outlet_name, route in outlet_routes.items():
        for edge_id in route.get("edge_ids", []):
            edge_to_outlet_names.setdefault(int(edge_id), []).append(str(outlet_name))
    all_outlet_names = sorted(outlet_routes.keys())

    segments: list[GeometrySegment] = [
        GeometrySegment(
            segment_id=1,
            name_hint="abdominal_aorta_trunk",
            segment_type="aorta_trunk",
            proximal_node=int(root_node),
            distal_node=int(bif_node),
            edge_ids=list(aorta_edge_ids),
            points=aorta_points,
            parent_segment_id=None,
            descendant_terminal_names=all_outlet_names,
            tunnel_source_outlets=all_outlet_names,
            parent_junction_node_id=None,
            distal_junction_node_ids=[int(bif_node)],
            outlet_route_node_paths={name: list(route["node_path"]) for name, route in outlet_routes.items()},
        )
    ]
    edge_to_segment: dict[int, int] = {int(edge_id): 1 for edge_id in aorta_edge_ids}
    distal_node_to_segment: dict[int, int] = {int(bif_node): 1}
    for node in aorta_node_path[1:]:
        distal_node_to_segment.setdefault(int(node), 1)

    next_segment_id = 2
    for _, edge_id, parent, child in sorted(directed_edges):
        if int(edge_id) in edge_to_segment:
            continue
        edge = edges[int(edge_id)]
        pts = edge.points if edge.start_node == parent else edge.points[::-1]
        terminal_face_id = terminal_node_to_face.get(int(child))
        terminal_face_name = _face_name(face_map, terminal_face_id) if terminal_face_id is not None else None
        descendants = _collect_descendant_terminal_names(child, child_map, terminal_node_to_face, face_map)
        tunnel_outlets = sorted(edge_to_outlet_names.get(int(edge_id), descendants))
        if int(parent) in set(aorta_node_path):
            parent_segment_id = 1
        else:
            parent_segment_id = distal_node_to_segment.get(int(parent))
        seg_type = "terminal_branch" if terminal_face_id is not None else "topology_branch"
        name_hint = _name_hint_for_segment(seg_type, terminal_face_name, descendants)
        segment = GeometrySegment(
            segment_id=int(next_segment_id),
            name_hint=name_hint,
            segment_type=seg_type,
            proximal_node=int(parent),
            distal_node=int(child),
            edge_ids=[int(edge_id)],
            points=np.asarray(pts, dtype=float),
            parent_segment_id=parent_segment_id,
            terminal_face_id=terminal_face_id,
            terminal_face_name=terminal_face_name,
            descendant_terminal_names=descendants,
            tunnel_source_outlets=tunnel_outlets,
            parent_junction_node_id=int(parent),
            distal_junction_node_ids=[int(v) for v in child_map.get(int(child), [])],
            outlet_route_node_paths={
                name: list(outlet_routes[name]["node_path"])
                for name in tunnel_outlets
                if name in outlet_routes
            },
        )
        segments.append(segment)
        edge_to_segment[int(edge_id)] = int(next_segment_id)
        distal_node_to_segment[int(child)] = int(next_segment_id)
        next_segment_id += 1

    by_id = {seg.segment_id: seg for seg in segments}
    for seg in segments:
        if seg.parent_segment_id is not None and seg.parent_segment_id in by_id:
            by_id[seg.parent_segment_id].child_segment_ids.append(int(seg.segment_id))
    return segments, aorta_node_path


@dataclass
class VmtkBranchSegmentation:
    surface: vtk.vtkPolyData
    split_centerlines: vtk.vtkPolyData
    group_to_segment_id: dict[int, int]
    group_to_descendant_names: dict[int, list[str]]
    cell_group_ids: np.ndarray
    clipping_cell_values: np.ndarray
    diagnostics: Dict[str, Any]


def _face_id_from_step1_source(source: str) -> Optional[int]:
    marker = "ModelFaceID:"
    if marker not in str(source):
        return None
    tail = str(source).split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
            continue
        break
    if not digits:
        return None
    return int("".join(digits))


def _path_id_to_terminal_name(step1_metadata: Dict[str, Any], face_map: Dict[int, Dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw_path in step1_metadata.get("centerline_extraction", {}).get("raw_paths", []):
        path_id = raw_path.get("path_id")
        face_id = _face_id_from_step1_source(str(raw_path.get("termination_source", "")))
        if path_id is None or face_id is None:
            continue
        if int(face_id) not in face_map:
            continue
        name = _face_name(face_map, int(face_id))
        if name == "abdominal_aorta_inlet":
            continue
        out[int(path_id)] = str(name)
    if not out:
        raise Step2RequiresReview("STEP1 raw centerline metadata does not map VMTK PathId values to named outlet faces.")
    return out


def _vmtk_group_descendants(
    split_centerlines: vtk.vtkPolyData,
    path_to_terminal_name: dict[int, str],
) -> dict[int, list[str]]:
    cd = split_centerlines.GetCellData()
    group_arr = cd.GetArray("GroupIds")
    path_arr = cd.GetArray("PathId")
    blanking_arr = cd.GetArray("Blanking")
    if group_arr is None or path_arr is None:
        raise Step2RequiresReview("VMTK branch extractor output is missing GroupIds or PathId arrays.")

    group_desc: dict[int, set[str]] = {}
    for cell_id in range(split_centerlines.GetNumberOfCells()):
        blanking = int(round(blanking_arr.GetTuple1(cell_id))) if blanking_arr is not None else 0
        if blanking != 0:
            continue
        path_id = int(round(path_arr.GetTuple1(cell_id)))
        terminal_name = path_to_terminal_name.get(path_id)
        if terminal_name is None:
            continue
        group_id = int(round(group_arr.GetTuple1(cell_id)))
        group_desc.setdefault(group_id, set()).add(str(terminal_name))
    if not group_desc:
        raise Step2RequiresReview("VMTK branch extractor produced no nonblank outlet-descendant groups.")
    return {int(group_id): sorted(names) for group_id, names in sorted(group_desc.items())}


def _cell_values_from_point_array(surface: vtk.vtkPolyData, array_name: str) -> tuple[np.ndarray, int]:
    arr = surface.GetPointData().GetArray(array_name)
    if arr is None:
        raise Step2RequiresReview(f"VMTK branch clipper output is missing point array {array_name}.")
    values = np.zeros((surface.GetNumberOfCells(),), dtype=np.int32)
    mixed = 0
    ids = vtk.vtkIdList()
    for cell_id in range(surface.GetNumberOfCells()):
        surface.GetCellPoints(cell_id, ids)
        cell_values = [int(round(arr.GetTuple1(ids.GetId(idx)))) for idx in range(ids.GetNumberOfIds())]
        if not cell_values:
            continue
        counts = Counter(cell_values)
        if len(counts) > 1:
            mixed += 1
        values[cell_id] = int(counts.most_common(1)[0][0])
    return values, int(mixed)


def _cell_values_from_optional_point_array(surface: vtk.vtkPolyData, array_name: str) -> np.ndarray:
    arr = surface.GetPointData().GetArray(array_name)
    if arr is None:
        return np.zeros((surface.GetNumberOfCells(),), dtype=np.int32)
    values = np.zeros((surface.GetNumberOfCells(),), dtype=np.int32)
    ids = vtk.vtkIdList()
    for cell_id in range(surface.GetNumberOfCells()):
        surface.GetCellPoints(cell_id, ids)
        if ids.GetNumberOfIds() <= 0:
            continue
        values[cell_id] = int(round(arr.GetTuple1(ids.GetId(0))))
    return values


def _map_vmtk_groups_to_segments(
    group_descendants: dict[int, list[str]],
    segments: list[GeometrySegment],
) -> dict[int, int]:
    descendant_key_to_segment: dict[tuple[str, ...], int] = {}
    for segment in segments:
        if segment.segment_type == "aorta_trunk":
            continue
        key = tuple(sorted(str(name) for name in segment.descendant_terminal_names))
        if key:
            descendant_key_to_segment[key] = int(segment.segment_id)

    group_to_segment: dict[int, int] = {}
    for group_id, descendant_names in sorted(group_descendants.items()):
        key = tuple(sorted(str(name) for name in descendant_names))
        group_to_segment[int(group_id)] = int(descendant_key_to_segment.get(key, 1))
    return group_to_segment


def _terminal_face_group_confidence(
    clipped_surface: vtk.vtkPolyData,
    cell_group_ids: np.ndarray,
    group_to_segment_id: dict[int, int],
    segments: list[GeometrySegment],
) -> dict[str, Any]:
    model_face = get_cell_array(clipped_surface, "ModelFaceID")
    if model_face is None:
        raise Step2RequiresReview("VMTK clipped surface is missing ModelFaceID, so terminal groups cannot be validated.")
    model_face_int = model_face.astype(int)
    rows: list[Dict[str, Any]] = []
    worst_confidence = 1.0
    for segment in segments:
        if segment.terminal_face_id is None:
            continue
        face_cells = np.flatnonzero(model_face_int == int(segment.terminal_face_id))
        segment_groups = sorted(
            int(group_id)
            for group_id, mapped_segment_id in group_to_segment_id.items()
            if int(mapped_segment_id) == int(segment.segment_id)
        )
        if face_cells.size == 0 or not segment_groups:
            confidence = 0.0
            dominant_group_id = None
            dominant_count = 0
        else:
            counts = Counter(int(cell_group_ids[cell_id]) for cell_id in face_cells.tolist())
            dominant_group_id, dominant_count = counts.most_common(1)[0]
            matching = sum(count for group_id, count in counts.items() if int(group_id) in segment_groups)
            confidence = float(matching / max(1, int(face_cells.size)))
        worst_confidence = min(worst_confidence, confidence)
        rows.append(
            {
                "segment_id": int(segment.segment_id),
                "segment_name": str(segment.name_hint),
                "terminal_face_id": int(segment.terminal_face_id),
                "vmtk_group_ids": segment_groups,
                "dominant_terminal_face_group_id": dominant_group_id,
                "dominant_terminal_face_group_cell_count": int(dominant_count),
                "terminal_face_cell_count": int(face_cells.size),
                "confidence": float(confidence),
            }
        )
    low = [row for row in rows if float(row["confidence"]) < 0.90]
    if low:
        names = ", ".join(f"{row['segment_name']}={row['confidence']:.2f}" for row in low)
        raise Step2RequiresReview("VMTK terminal face-to-group mapping confidence is too low: " + names)
    return {
        "terminal_face_group_mappings": rows,
        "terminal_face_group_min_confidence": float(worst_confidence),
    }


def _assign_vmtk_boundary_profiles(segments: list[GeometrySegment], group_to_segment_id: dict[int, int]) -> None:
    group_ids_by_segment: dict[int, list[int]] = {}
    for group_id, segment_id in group_to_segment_id.items():
        group_ids_by_segment.setdefault(int(segment_id), []).append(int(group_id))
    for segment in segments:
        segment.vmtk_group_ids = sorted(group_ids_by_segment.get(int(segment.segment_id), []))
        if segment.segment_type == "aorta_trunk":
            continue
        origin = segment.points[0] if segment.points.shape[0] else np.zeros(3, dtype=float)
        normal = tangent_at_arclength(segment.points, 0.0, window=0.75) if segment.points.shape[0] >= 2 else np.zeros(3)
        segment.proximal_boundary = SegmentBoundaryProfile(
            source_type="vmtk_branch_clip",
            centroid=np.asarray(origin, dtype=float),
            normal=unit(normal),
            area=0.0,
            equivalent_diameter=None,
            major_diameter=None,
            minor_diameter=None,
            arclength=0.0,
            confidence=0.95,
            method=PROXIMAL_BOUNDARY_SELECTION_ALGORITHM,
            attempts=[
                {
                    "source": "vmtkBranchClipper",
                    "vmtk_group_ids": [int(v) for v in segment.vmtk_group_ids],
                    "selected_reason": "authoritative_vmtk_branch_group_boundary",
                }
            ],
            loop_points=None,
        )


def _run_vmtk_branch_group_segmentation(
    *,
    surface: vtk.vtkPolyData,
    raw_centerlines: vtk.vtkPolyData,
    step1_metadata: Dict[str, Any],
    face_map: Dict[int, Dict[str, Any]],
    segments: list[GeometrySegment],
) -> VmtkBranchSegmentation:
    vmtkscripts, runtime_info = _import_vmtk_scripts()

    extractor = vmtkscripts.vmtkBranchExtractor()
    extractor.Centerlines = raw_centerlines
    extractor.Execute()
    split_centerlines = extractor.Centerlines
    if split_centerlines is None or split_centerlines.GetNumberOfCells() <= 0:
        raise Step2RequiresReview("VMTK branch extractor returned empty centerlines.")

    clipper = vmtkscripts.vmtkBranchClipper()
    clipper.Surface = surface
    clipper.Centerlines = split_centerlines
    clipper.Interactive = 0
    clipper.Execute()
    clipped_surface = clipper.Surface
    if clipped_surface is None or clipped_surface.GetNumberOfCells() <= 0:
        raise Step2RequiresReview("VMTK branch clipper returned an empty surface.")

    cell_group_ids, mixed_group_cells = _cell_values_from_point_array(clipped_surface, "GroupIds")
    mixed_threshold = max(25, int(math.ceil(0.001 * max(1, clipped_surface.GetNumberOfCells()))))
    if mixed_group_cells > mixed_threshold:
        raise Step2RequiresReview(
            f"VMTK branch clipper produced {mixed_group_cells} mixed group cells, above threshold {mixed_threshold}."
        )

    path_to_name = _path_id_to_terminal_name(step1_metadata, face_map)
    group_descendants = _vmtk_group_descendants(split_centerlines, path_to_name)
    group_to_segment_id = _map_vmtk_groups_to_segments(group_descendants, segments)
    surface_groups = sorted(set(int(v) for v in cell_group_ids.tolist()))
    unmapped_surface_groups = [group_id for group_id in surface_groups if group_id not in group_to_segment_id]
    if unmapped_surface_groups:
        raise Step2RequiresReview("VMTK clipped surface has unmapped GroupIds: " + ", ".join(str(v) for v in unmapped_surface_groups))

    _assign_vmtk_boundary_profiles(segments, group_to_segment_id)
    terminal_mapping = _terminal_face_group_confidence(clipped_surface, cell_group_ids, group_to_segment_id, segments)
    clipping_values = _cell_values_from_optional_point_array(clipped_surface, "ClippingArray")

    return VmtkBranchSegmentation(
        surface=clipped_surface,
        split_centerlines=split_centerlines,
        group_to_segment_id=group_to_segment_id,
        group_to_descendant_names=group_descendants,
        cell_group_ids=cell_group_ids,
        clipping_cell_values=clipping_values,
        diagnostics={
            **runtime_info,
            **terminal_mapping,
            "vmtk_branch_extractor_group_count": int(len(group_descendants)),
            "vmtk_clipped_surface_group_count": int(len(surface_groups)),
            "vmtk_mixed_group_cells": int(mixed_group_cells),
            "vmtk_mixed_group_cell_threshold": int(mixed_threshold),
            "vmtk_group_to_segment_id": {str(group_id): int(segment_id) for group_id, segment_id in sorted(group_to_segment_id.items())},
            "vmtk_group_descendant_names": {str(group_id): list(names) for group_id, names in sorted(group_descendants.items())},
            "vmtk_branch_clipper_input_cells": int(surface.GetNumberOfCells()),
            "vmtk_branch_clipper_output_cells": int(clipped_surface.GetNumberOfCells()),
        },
    )


def _compute_aorta_local_radii(
    aorta_segment: "GeometrySegment",
    raw_centerlines: vtk.vtkPolyData,
    cell_arclengths: np.ndarray,
) -> np.ndarray:
    """Per-cell local aorta radius by interpolating MISR from the raw VMTK centerlines."""
    misr_arr = raw_centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if misr_arr is None or raw_centerlines.GetNumberOfPoints() == 0:
        return np.full(cell_arclengths.shape, 1.5, dtype=float)

    n_raw = raw_centerlines.GetNumberOfPoints()
    raw_pts = np.asarray([raw_centerlines.GetPoint(i) for i in range(n_raw)], dtype=float)
    raw_misr = np.asarray([misr_arr.GetTuple1(i) for i in range(n_raw)], dtype=float)

    aorta_pts = np.asarray(aorta_segment.points, dtype=float)

    diff_raw_to_aorta = raw_pts[:, np.newaxis, :] - aorta_pts[np.newaxis, :, :]
    min_dist_per_raw = np.sqrt((diff_raw_to_aorta * diff_raw_to_aorta).sum(axis=2).min(axis=1))
    aorta_path_mask = min_dist_per_raw < 0.60
    if aorta_path_mask.sum() < 10:
        aorta_path_mask = np.ones(n_raw, dtype=bool)
    filtered_raw_pts = raw_pts[aorta_path_mask]
    filtered_raw_misr = raw_misr[aorta_path_mask]

    diff = aorta_pts[:, np.newaxis, :] - filtered_raw_pts[np.newaxis, :, :]
    dist_sq = (diff * diff).sum(axis=2)
    nearest_idx = dist_sq.argmin(axis=1)
    aorta_misr = filtered_raw_misr[nearest_idx]

    aorta_s = cumulative_arclength(aorta_pts)
    local_radius = np.interp(cell_arclengths, aorta_s, aorta_misr)
    return np.maximum(local_radius, 0.45)


def _assign_surface_cells_from_vmtk(
    surface: vtk.vtkPolyData,
    segments: list[GeometrySegment],
    vmtk_result: VmtkBranchSegmentation,
    raw_centerlines: Optional[vtk.vtkPolyData] = None,
) -> tuple[np.ndarray, Dict[str, int], Dict[int, int], Dict[str, np.ndarray]]:
    n_cells = int(surface.GetNumberOfCells())
    labels = np.zeros((n_cells,), dtype=np.int32)
    unmapped = 0
    for cell_id, group_id in enumerate(vmtk_result.cell_group_ids.tolist()):
        segment_id = int(vmtk_result.group_to_segment_id.get(int(group_id), 0))
        if segment_id <= 0:
            unmapped += 1
        labels[cell_id] = segment_id
    if unmapped:
        raise Step2RequiresReview(f"VMTK group assignment left {unmapped} clipped surface cells unmapped.")

    assignment_mode = np.full((n_cells,), 10, dtype=np.int32)
    adjacency = _surface_cell_adjacency(surface)
    model_face = get_cell_array(surface, "ModelFaceID")
    branch_owner_id = np.where(labels > 1, labels, 0).astype(np.int32)
    original_labels = labels.copy()
    island_counts, island_debug = _reject_orphan_branch_components(
        labels,
        assignment_mode,
        adjacency,
        branch_owner_id,
        model_face,
        segments,
    )

    # Footprint rescue: VMTK cut planes can include aorta-wall cells just
    # proximal to an ostium in a branch group. Re-run _branch_origin_allowed_mask
    # (strict=True) on every branch-labeled cell; anything that fails the
    # footprint/parent-competition check is relabeled to aorta.
    centers = cell_centers(surface)
    distance_by_segment, arclength_by_segment, radius_by_segment, _ = _precompute_segment_projections(
        centers, segments
    )
    terminal_face_mask = np.zeros((n_cells,), dtype=bool)
    if model_face is not None:
        model_face_int = model_face.astype(int)
        for seg in segments:
            if seg.terminal_face_id is not None:
                terminal_face_mask |= model_face_int == int(seg.terminal_face_id)
    aorta_id = int(next(s.segment_id for s in segments if s.segment_type == "aorta_trunk"))
    aorta_segment = next(s for s in segments if s.segment_type == "aorta_trunk")
    aorta_radius = max(float(radius_by_segment.get(aorta_id, 1.0)), 0.45)

    if raw_centerlines is not None:
        local_aorta_radii = _compute_aorta_local_radii(
            aorta_segment, raw_centerlines, arclength_by_segment[aorta_id]
        )
    else:
        local_aorta_radii = np.full(n_cells, aorta_radius, dtype=float)

    branch_parent_wall_before_rescue = 0
    for segment in segments:
        if segment.segment_type == "aorta_trunk":
            continue
        seg_id = int(segment.segment_id)
        radius = max(float(radius_by_segment.get(seg_id, 1.0)), 0.45)
        branch_cells = np.flatnonzero((labels == seg_id) & ~terminal_face_mask)
        if branch_cells.size == 0:
            continue
        child_score = distance_by_segment[seg_id][branch_cells] / radius
        local_radius_bc = local_aorta_radii[branch_cells]
        parent_score = distance_by_segment[aorta_id][branch_cells] / local_radius_bc
        # Primary: fails the origin-allowed mask (proximal-plane + competition).
        allowed = _branch_origin_allowed_mask(
            centers,
            branch_cells,
            segment,
            radius,
            distance_by_segment[seg_id],
            arclength_by_segment[seg_id],
            distance_by_segment,
            radius_by_segment,
            strict=True,
        )
        mask_rescue = ~allowed & (child_score > 0.80)
        aorta_wall_rescue = parent_score < 1.20
        leak_cells = branch_cells[mask_rescue | aorta_wall_rescue]
        branch_parent_wall_before_rescue += int(leak_cells.size)
        if leak_cells.size:
            labels[leak_cells] = 1
            assignment_mode[leak_cells] = 11

    # Sweep up isolated spots: small disconnected patches of any label surrounded
    # by a different label get absorbed by their dominant neighbor.
    terminal_face_set: set[int] = set()
    for seg in segments:
        if seg.terminal_face_id is not None:
            terminal_face_set.add(int(seg.terminal_face_id))
    _cleanup_small_label_islands(
        surface, labels, terminal_face_set, assignment_mode, max_island_cells=100
    )

    branch_group_aorta_labels = int(np.count_nonzero((original_labels > 1) & (labels == 1)))
    aorta_group_branch_labels = int(np.count_nonzero((original_labels == 1) & (labels > 1)))
    counts = {int(seg.segment_id): int(np.count_nonzero(labels == int(seg.segment_id))) for seg in segments}
    for segment in segments:
        segment.cell_count = counts.get(int(segment.segment_id), 0)
        segment.fallback_cell_count = 0

    group_counts = Counter(int(v) for v in vmtk_result.cell_group_ids.tolist())
    assignment_counts: Dict[str, int] = {
        SURFACE_ASSIGNMENT_ALGORITHM: 1,
        "fallback": 0,
        "fallback_assigned_cells": 0,
        "vmtk_branch_group_assigned_cells": int(np.count_nonzero(labels > 0)),
        "vmtk_group_count": int(len(group_counts)),
        "vmtk_mixed_group_cells": int(vmtk_result.diagnostics.get("vmtk_mixed_group_cells", 0)),
        "vmtk_branch_extractor_group_count": int(vmtk_result.diagnostics.get("vmtk_branch_extractor_group_count", 0)),
        "vmtk_clipped_surface_group_count": int(vmtk_result.diagnostics.get("vmtk_clipped_surface_group_count", 0)),
        "branch_groups_labeled_as_aorta": int(branch_group_aorta_labels),
        "aorta_groups_labeled_as_branch": int(aorta_group_branch_labels),
        "branch_owned_aorta_labeled_cells": int(branch_group_aorta_labels),
        "branch_labeled_parent_aorta_wall_before_rescue": int(branch_parent_wall_before_rescue),
        "branch_labeled_parent_aorta_wall_cells": int(aorta_group_branch_labels),
        "vmtk_terminal_face_group_min_confidence_x1000": int(
            round(1000.0 * float(vmtk_result.diagnostics.get("terminal_face_group_min_confidence", 0.0)))
        ),
        "mesh_partition_input_cells": int(vmtk_result.diagnostics.get("vmtk_branch_clipper_input_cells", 0)),
        "mesh_partition_output_cells": int(vmtk_result.diagnostics.get("vmtk_branch_clipper_output_cells", n_cells)),
        "mesh_partition_authored_cuts": int(len(vmtk_result.group_to_segment_id)),
        "mesh_partition_skipped_cuts": 0,
        **island_counts,
    }
    for group_id, count in sorted(group_counts.items()):
        assignment_counts[f"vmtk_group_{int(group_id)}_cells"] = int(count)
    for segment in segments:
        assignment_counts[f"seed_segment_{int(segment.segment_id)}"] = int(segment.cell_count)
    assignment_counts["seed_count_total"] = int(sum(segment.cell_count for segment in segments))

    diagnostics = {
        "AssignmentMode": assignment_mode,
        "VmtkGroupId": vmtk_result.cell_group_ids.astype(np.int32),
        "VmtkClippingArray": vmtk_result.clipping_cell_values.astype(np.int32),
        "VmtkOriginalSegmentId": original_labels.astype(np.int32),
        **island_debug,
    }
    return labels, assignment_counts, counts, diagnostics


def _cell_area_and_centroid(surface: vtk.vtkPolyData, cell_id: int) -> tuple[float, np.ndarray]:
    pts = cell_points(surface, int(cell_id))
    area = triangle_area(pts)
    centroid = np.mean(pts, axis=0) if pts.shape[0] else np.zeros(3, dtype=float)
    return float(area), centroid


def _safe_projected_major_minor_diameters(points: np.ndarray, normal_hint: Optional[np.ndarray] = None) -> tuple[Optional[float], Optional[float]]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return None, None
    center = np.mean(pts, axis=0)
    normal = np.asarray(normal_hint, dtype=float) if normal_hint is not None else np.zeros(3, dtype=float)
    if float(np.linalg.norm(normal)) <= 1.0e-12:
        _, normal, _ = polygon_area_normal(pts)
    normal = unit(normal)
    if float(np.linalg.norm(normal)) <= 1.0e-12:
        return None, None
    reference = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(normal, reference))) > 0.90:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=float)
    u = unit(np.cross(normal, reference))
    v = unit(np.cross(normal, u))
    if float(np.linalg.norm(u)) <= 1.0e-12 or float(np.linalg.norm(v)) <= 1.0e-12:
        return None, None
    x_values: list[float] = []
    y_values: list[float] = []
    for point in pts.tolist():
        dx = float(point[0]) - float(center[0])
        dy = float(point[1]) - float(center[1])
        dz = float(point[2]) - float(center[2])
        x_values.append(dx * float(u[0]) + dy * float(u[1]) + dz * float(u[2]))
        y_values.append(dx * float(v[0]) + dy * float(v[1]) + dz * float(v[2]))
    x_extent = max(x_values) - min(x_values)
    y_extent = max(y_values) - min(y_values)
    major = float(max(x_extent, y_extent))
    minor = float(min(x_extent, y_extent))
    if not math.isfinite(major) or not math.isfinite(minor):
        return None, None
    return major, minor


def _face_region_profile(
    surface: vtk.vtkPolyData,
    face_id: int,
    metadata_term: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_face = get_cell_array(surface, "ModelFaceID")
    if model_face is None:
        raise Step2Failure("Aortic inlet cannot be resolved: surface is missing ModelFaceID.")
    ids = np.flatnonzero(model_face.astype(int) == int(face_id))
    if ids.size == 0:
        raise Step2Failure(f"Aortic inlet cannot be resolved: no cells found for ModelFaceID {face_id}.")

    weighted_center = np.zeros(3, dtype=float)
    total_area = 0.0
    point_ids: set[int] = set()
    for cid in ids.tolist():
        area, centroid = _cell_area_and_centroid(surface, int(cid))
        total_area += area
        weighted_center += area * centroid
        cell = surface.GetCell(int(cid))
        cids = cell.GetPointIds()
        for i in range(cids.GetNumberOfIds()):
            point_ids.add(int(cids.GetId(i)))
    if total_area > 0.0:
        center = weighted_center / total_area
    else:
        center = np.asarray(metadata_term.get("center"), dtype=float) if metadata_term else np.zeros(3, dtype=float)

    pts = np.asarray([surface.GetPoint(pid) for pid in sorted(point_ids)], dtype=float)
    normal = np.asarray(metadata_term.get("normal"), dtype=float) if metadata_term and metadata_term.get("normal") is not None else np.zeros(3)
    if np.linalg.norm(normal) <= 1.0e-12 and pts.shape[0] >= 3:
        _, normal, _ = polygon_area_normal(pts)
    major, minor = _safe_projected_major_minor_diameters(pts, normal_hint=normal)
    eq = equivalent_diameter_from_area(total_area)
    confidence = 0.95 if metadata_term else 0.85
    return {
        "source_type": "face_map_model_face",
        "face_id": int(face_id),
        "centroid": center.tolist(),
        "normal": unit(normal).tolist(),
        "area": float(total_area),
        "equivalent_diameter": eq,
        "major_diameter": major,
        "minor_diameter": minor,
        "cell_count": int(ids.size),
        "confidence": float(confidence),
        "source": metadata_term.get("source") if metadata_term else f"ModelFaceID:{face_id}",
    }


def _contour_profiles_from_plane(
    surface: vtk.vtkPolyData,
    origin: np.ndarray,
    normal: np.ndarray,
) -> list[Dict[str, Any]]:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    n = unit(normal)
    plane.SetNormal(float(n[0]), float(n[1]), float(n[2]))

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(surface)
    cutter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(cutter.GetOutput())
    cleaner.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputData(cleaner.GetOutput())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    contours = stripper.GetOutput()

    profiles: list[Dict[str, Any]] = []
    for cell_id in range(contours.GetNumberOfCells()):
        cell = contours.GetCell(cell_id)
        ids = cell.GetPointIds()
        if ids.GetNumberOfIds() < 3:
            continue
        pts = np.asarray([contours.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
        closed_gap = distance(pts[0], pts[-1])
        if closed_gap <= 1.0e-5:
            pts = pts[:-1]
        if pts.shape[0] < 3:
            continue
        area, profile_normal, rms = polygon_area_normal(pts)
        if area <= 1.0e-8:
            continue
        centroid = np.mean(pts, axis=0)
        major, minor = _safe_projected_major_minor_diameters(pts, normal_hint=n)
        profiles.append(
            {
                "cell_id": int(cell_id),
                "points": pts,
                "centroid": centroid,
                "area": float(area),
                "normal": unit(profile_normal if np.linalg.norm(profile_normal) > 0 else n),
                "rms_planarity": float(rms),
                "closed_gap": float(closed_gap),
                "equivalent_diameter": equivalent_diameter_from_area(area),
                "major_diameter": major,
                "minor_diameter": minor,
                "distance_to_origin": distance(centroid, origin),
                "point_count": int(pts.shape[0]),
            }
        )
    return profiles


def _extract_aorta_end_profile(
    surface: vtk.vtkPolyData,
    aorta_points: np.ndarray,
    start_equivalent_diameter: Optional[float],
) -> Dict[str, Any]:
    length = float(cumulative_arclength(aorta_points)[-1])
    if length <= 1.0e-6:
        raise Step2Failure("Aortic end cannot be measured: aorta trunk length is zero.")
    base_offset = 0.75
    if start_equivalent_diameter is not None and start_equivalent_diameter > 0.0:
        base_offset = min(1.25, max(0.4, 0.25 * float(start_equivalent_diameter)))
    offsets = [base_offset, 1.0, 1.5, 2.0, 0.25]
    tangent = tangent_at_arclength(aorta_points, max(0.0, length - base_offset), window=1.0)
    attempts: list[Dict[str, Any]] = []
    for offset in offsets:
        s = max(0.0, length - float(offset))
        origin = point_at_arclength(aorta_points, s)
        n = tangent_at_arclength(aorta_points, s, window=1.0)
        if np.linalg.norm(n) <= 1.0e-12:
            n = tangent
        profiles = _contour_profiles_from_plane(surface, origin, n)
        attempts.append(
            {
                "offset_from_bifurcation_mm": float(offset),
                "section_arclength": float(s),
                "candidate_count": int(len(profiles)),
            }
        )
        if not profiles:
            continue
        best = min(profiles, key=lambda row: (float(row["distance_to_origin"]), -float(row["area"])))
        if float(best["distance_to_origin"]) > 10.0:
            continue
        confidence = 0.9
        if float(best["closed_gap"]) > 0.25:
            confidence = 0.75
        return {
            "surface_derived": True,
            "extraction_method": "surface_plane_cut_pre_bifurcation",
            "section_offset_from_bifurcation_mm": float(offset),
            "section_arclength": float(s),
            "boundary_centroid": best["centroid"].tolist(),
            "boundary_normal": unit(n).tolist(),
            "area": float(best["area"]),
            "equivalent_diameter": best["equivalent_diameter"],
            "major_diameter": best["major_diameter"],
            "minor_diameter": best["minor_diameter"],
            "rms_planarity": float(best["rms_planarity"]),
            "closed_gap": float(best["closed_gap"]),
            "point_count": int(best["point_count"]),
            "confidence": float(confidence),
            "attempts": attempts,
        }
    raise Step2Failure(f"Aortic end cannot be measured: no valid pre-bifurcation surface section found. Attempts: {attempts}")


def _project_point_to_polyline(point: np.ndarray, points: np.ndarray) -> tuple[float, float, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    p = as_point(point)
    if pts.shape[0] == 0:
        return float("inf"), 0.0, np.zeros(3, dtype=float)
    if pts.shape[0] == 1:
        return distance(p, pts[0]), 0.0, pts[0].copy()

    best_distance = float("inf")
    best_s = 0.0
    best_point = pts[0].copy()
    running_s = 0.0
    for idx in range(pts.shape[0] - 1):
        a = pts[idx]
        b = pts[idx + 1]
        ab = b - a
        seg_len2 = float(np.dot(ab, ab))
        if seg_len2 <= 1.0e-12:
            continue
        t = float(np.clip(np.dot(p - a, ab) / seg_len2, 0.0, 1.0))
        q = a + t * ab
        d = distance(p, q)
        seg_len = math.sqrt(seg_len2)
        if d < best_distance:
            best_distance = float(d)
            best_s = float(running_s + t * seg_len)
            best_point = q
        running_s += seg_len
    return best_distance, best_s, best_point


def _trim_polyline_from_arclength(points: np.ndarray, start_s: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return pts.copy()
    s = cumulative_arclength(pts)
    start = float(np.clip(start_s, 0.0, float(s[-1])))
    start_point = point_at_arclength(pts, start)
    keep_idx = int(np.searchsorted(s, start, side="right"))
    tail = pts[keep_idx:] if keep_idx < pts.shape[0] else np.zeros((0, 3), dtype=float)
    if tail.shape[0] and distance(start_point, tail[0]) <= 1.0e-8:
        return tail.copy()
    return np.vstack([start_point.reshape(1, 3), tail])


def _project_points_to_polyline(points: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(points, dtype=float)
    pts = np.asarray(polyline, dtype=float)
    if query.shape[0] == 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    if pts.shape[0] < 2:
        d = np.linalg.norm(query - pts[0].reshape(1, 3), axis=1) if pts.shape[0] else np.full((query.shape[0],), np.inf)
        return d.astype(float), np.zeros((query.shape[0],), dtype=float)

    cumulative = cumulative_arclength(pts)
    best_d2 = np.full((query.shape[0],), np.inf, dtype=float)
    best_s = np.zeros((query.shape[0],), dtype=float)
    for idx in range(pts.shape[0] - 1):
        a = pts[idx]
        b = pts[idx + 1]
        ab = b - a
        seg_len2 = float(np.dot(ab, ab))
        if seg_len2 <= 1.0e-12:
            continue
        t = np.clip(np.sum((query - a.reshape(1, 3)) * ab.reshape(1, 3), axis=1) / seg_len2, 0.0, 1.0)
        proj = a.reshape(1, 3) + t.reshape(-1, 1) * ab.reshape(1, 3)
        d2 = np.sum((query - proj) ** 2, axis=1)
        improved = d2 < best_d2
        if np.any(improved):
            best_d2[improved] = d2[improved]
            best_s[improved] = cumulative[idx] + t[improved] * math.sqrt(seg_len2)
    return np.sqrt(np.maximum(best_d2, 0.0)).astype(float), best_s.astype(float)


def _extract_branch_proximal_boundary(
    surface: vtk.vtkPolyData,
    segment: GeometrySegment,
    expected_area: Optional[float] = None,
    expected_diameter: Optional[float] = None,
) -> Optional[SegmentBoundaryProfile]:
    length = float(segment.length)
    if segment.segment_type == "aorta_trunk" or length <= 1.0e-6:
        return None

    offsets = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    valid_offsets = [offset for offset in offsets if offset < max(0.0, length - 0.1)]
    if not valid_offsets:
        valid_offsets = [0.0]

    attempts: list[Dict[str, Any]] = []
    accepted_candidates: list[Dict[str, Any]] = []

    def reference_diameter_for(eq_diameter: Optional[float]) -> float:
        if expected_diameter is not None and expected_diameter > 0.0:
            return float(expected_diameter)
        if eq_diameter is not None and float(eq_diameter) > 0.0:
            return float(eq_diameter)
        return 1.0

    def add_rejection(reasons: Dict[str, int], reason: str) -> None:
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    for offset_index, offset in enumerate(valid_offsets):
        origin = point_at_arclength(segment.points, float(offset))
        tangent = tangent_at_arclength(segment.points, float(offset), window=0.75)
        profiles = _contour_profiles_from_plane(surface, origin, tangent)
        attempt: Dict[str, Any] = {
            "offset_mm": float(offset),
            "candidate_count": int(len(profiles)),
            "accepted_candidate_count": 0,
            "rejection_reasons": {},
            "normalized_origin_distance": None,
            "normalized_centroid_distance": None,
            "stability_run_id": None,
            "selected_reason": None,
        }
        if not profiles:
            attempts.append(attempt)
            continue

        rejection_reasons: Dict[str, int] = {}
        ranked: list[tuple[float, Dict[str, Any], Dict[str, float]]] = []
        for profile in profiles:
            centroid = np.asarray(profile["centroid"], dtype=float)
            centroid_distance, _, _ = _project_point_to_polyline(centroid, segment.points)
            origin_distance = distance(centroid, origin)
            closed_gap = float(profile["closed_gap"])
            area = float(profile["area"])
            eq_diameter = profile.get("equivalent_diameter")
            reference_diameter = reference_diameter_for(eq_diameter)
            origin_tolerance = max(0.50, 0.90 * reference_diameter)
            centroid_tolerance = max(0.40, 0.75 * reference_diameter)
            normalized_origin_distance = origin_distance / max(reference_diameter, 1.0e-6)
            normalized_centroid_distance = centroid_distance / max(reference_diameter, 1.0e-6)

            candidate_rejections: list[str] = []
            if origin_distance > origin_tolerance:
                candidate_rejections.append("origin_distance_gt_normalized_tolerance")
            if centroid_distance > centroid_tolerance:
                candidate_rejections.append("centroid_distance_gt_normalized_tolerance")
            if closed_gap > 1.0:
                candidate_rejections.append("closed_gap_gt_1mm")
            if area <= 1.0e-8:
                candidate_rejections.append("nonpositive_area")
            if expected_area is not None and expected_area > 0.0 and area > max(5.0 * float(expected_area), float(expected_area) + 10.0):
                candidate_rejections.append("area_gt_expected_branch_limit")
            if (
                expected_diameter is not None
                and expected_diameter > 0.0
                and eq_diameter is not None
                and float(eq_diameter) > max(2.75 * float(expected_diameter), float(expected_diameter) + 3.5)
            ):
                candidate_rejections.append("diameter_gt_expected_branch_limit")

            if candidate_rejections:
                for reason in candidate_rejections:
                    add_rejection(rejection_reasons, reason)
                continue

            origin_fraction = origin_distance / max(origin_tolerance, 1.0e-6)
            centroid_fraction = centroid_distance / max(centroid_tolerance, 1.0e-6)
            closure_fraction = closed_gap / max(0.25 * reference_diameter, 1.0e-6)
            score = float(origin_fraction + 0.5 * centroid_fraction + 0.25 * closure_fraction)
            ranked.append(
                (
                    score,
                    profile,
                    {
                        "origin_distance": float(origin_distance),
                        "centroid_distance": float(centroid_distance),
                        "normalized_origin_distance": float(normalized_origin_distance),
                        "normalized_centroid_distance": float(normalized_centroid_distance),
                        "origin_fraction": float(origin_fraction),
                        "centroid_fraction": float(centroid_fraction),
                        "reference_diameter": float(reference_diameter),
                    },
                )
            )

        attempt["rejection_reasons"] = rejection_reasons
        attempt["accepted_candidate_count"] = int(len(ranked))
        if not ranked:
            attempts.append(attempt)
            continue

        score, selected, metrics = min(ranked, key=lambda item: item[0])
        attempt["selected_candidate_score"] = float(score)
        attempt["selected_candidate_area"] = float(selected["area"])
        attempt["selected_candidate_equivalent_diameter"] = selected["equivalent_diameter"]
        attempt["normalized_origin_distance"] = float(metrics["normalized_origin_distance"])
        attempt["normalized_centroid_distance"] = float(metrics["normalized_centroid_distance"])
        attempt["origin_distance"] = float(metrics["origin_distance"])
        attempt["centroid_distance"] = float(metrics["centroid_distance"])
        attempt["reference_diameter"] = float(metrics["reference_diameter"])
        confidence = 0.85
        if float(selected["closed_gap"]) > 0.25:
            confidence = 0.7
        if float(metrics["origin_fraction"]) > 0.85 or float(metrics["centroid_fraction"]) > 0.85:
            confidence = min(confidence, 0.78)

        boundary = SegmentBoundaryProfile(
            source_type="parent_junction_surface_cut",
            centroid=np.asarray(selected["centroid"], dtype=float),
            normal=unit(tangent),
            area=float(selected["area"]),
            equivalent_diameter=selected["equivalent_diameter"],
            major_diameter=selected["major_diameter"],
            minor_diameter=selected["minor_diameter"],
            arclength=float(offset),
            confidence=float(confidence),
            method=PROXIMAL_BOUNDARY_SELECTION_ALGORITHM,
            attempts=[],
            loop_points=np.asarray(selected["points"], dtype=float),
        )
        accepted_candidates.append(
            {
                "offset_index": int(offset_index),
                "attempt_index": int(len(attempts)),
                "offset": float(offset),
                "boundary": boundary,
                "score": float(score),
                "area": float(selected["area"]),
                "equivalent_diameter": float(selected["equivalent_diameter"])
                if selected["equivalent_diameter"] is not None
                else float(metrics["reference_diameter"]),
                "centroid": np.asarray(selected["centroid"], dtype=float),
                "normal": unit(tangent),
                "confidence": float(confidence),
            }
        )
        attempts.append(attempt)

    selected_candidate: Optional[Dict[str, Any]] = None
    selected_reason = ""
    stable_run_id = 0
    for idx in range(0, max(0, len(accepted_candidates) - 2)):
        run = accepted_candidates[idx : idx + 3]
        if any(int(run[j + 1]["offset_index"]) != int(run[j]["offset_index"]) + 1 for j in range(2)):
            continue
        if float(run[-1]["offset"]) - float(run[0]["offset"]) > 0.31:
            continue
        diameters = np.asarray([max(float(item["equivalent_diameter"]), 1.0e-6) for item in run], dtype=float)
        areas = np.asarray([max(float(item["area"]), 1.0e-8) for item in run], dtype=float)
        median_diameter = float(np.median(diameters))
        diameter_ratio = float(np.max(diameters) / max(float(np.min(diameters)), 1.0e-6))
        area_ratio = float(np.max(areas) / max(float(np.min(areas)), 1.0e-8))
        centroid_steps = [distance(np.asarray(run[j]["centroid"], dtype=float), np.asarray(run[j + 1]["centroid"], dtype=float)) for j in range(2)]
        max_centroid_step = max(centroid_steps) if centroid_steps else 0.0
        normal_dots = [
            abs(float(np.dot(unit(np.asarray(run[j]["normal"], dtype=float)), unit(np.asarray(run[j + 1]["normal"], dtype=float)))))
            for j in range(2)
        ]
        if diameter_ratio > 1.75 or area_ratio > 2.25:
            continue
        if max_centroid_step > max(0.40, 0.45 * median_diameter):
            continue
        if min(normal_dots) < 0.50:
            continue
        stable_run_id += 1
        for item in run:
            attempts[int(item["attempt_index"])]["stability_run_id"] = int(stable_run_id)
        attempts[int(run[0]["attempt_index"])]["selected_reason"] = "earliest_candidate_in_first_stable_run"
        attempts[int(run[0]["attempt_index"])]["stable_run_diameter_ratio"] = diameter_ratio
        attempts[int(run[0]["attempt_index"])]["stable_run_area_ratio"] = area_ratio
        attempts[int(run[0]["attempt_index"])]["stable_run_max_centroid_step"] = float(max_centroid_step)
        selected_candidate = run[0]
        selected_reason = "earliest_candidate_in_first_stable_run"
        break

    if selected_candidate is None and accepted_candidates:
        strong_candidates = [item for item in accepted_candidates if float(item["confidence"]) >= 0.75]
        selected_candidate = strong_candidates[0] if strong_candidates else accepted_candidates[0]
        selected_reason = "earliest_strong_candidate_no_stable_run" if strong_candidates else "earliest_candidate_no_stable_run"
        attempts[int(selected_candidate["attempt_index"])]["selected_reason"] = selected_reason
        boundary = selected_candidate["boundary"]
    elif selected_candidate is not None:
        boundary = selected_candidate["boundary"]
    else:
        segment.proximal_boundary_attempts = attempts
        return None

    attempts[int(selected_candidate["attempt_index"])]["selected_reason"] = selected_reason
    boundary.attempts = attempts
    return boundary


def _extract_tunnel_boundary_at_arclength(
    surface: vtk.vtkPolyData,
    segment: GeometrySegment,
    arclength: float,
    source_type: str,
) -> Optional[SegmentBoundaryProfile]:
    if segment.points.shape[0] < 2:
        return None
    length = float(segment.length)
    s = float(np.clip(arclength, 0.0, length))
    origin = point_at_arclength(segment.points, s)
    tangent = tangent_at_arclength(segment.points, s, window=0.75)
    profiles = _contour_profiles_from_plane(surface, origin, tangent)
    attempt = {
        "offset_mm": float(s),
        "candidate_count": int(len(profiles)),
        "selected_reason": None,
    }
    if not profiles:
        return None
    ranked = []
    for profile in profiles:
        centroid = np.asarray(profile["centroid"], dtype=float)
        centroid_distance, _, _ = _project_point_to_polyline(centroid, segment.points)
        origin_distance = distance(centroid, origin)
        ranked.append((float(origin_distance + 0.5 * centroid_distance), profile, origin_distance, centroid_distance))
    _, selected, origin_distance, centroid_distance = min(ranked, key=lambda item: item[0])
    attempt.update(
        {
            "selected_reason": "closest_loop_to_tunnel_junction",
            "origin_distance": float(origin_distance),
            "centroid_distance": float(centroid_distance),
            "selected_candidate_area": float(selected["area"]),
            "selected_candidate_equivalent_diameter": selected["equivalent_diameter"],
        }
    )
    return SegmentBoundaryProfile(
        source_type=source_type,
        centroid=np.asarray(selected["centroid"], dtype=float),
        normal=unit(tangent),
        area=float(selected["area"]),
        equivalent_diameter=selected["equivalent_diameter"],
        major_diameter=selected["major_diameter"],
        minor_diameter=selected["minor_diameter"],
        arclength=float(s),
        confidence=0.82,
        method=PROXIMAL_BOUNDARY_SELECTION_ALGORITHM,
        attempts=[attempt],
        loop_points=np.asarray(selected["points"], dtype=float),
    )


def _refine_branch_boundaries(
    surface: vtk.vtkPolyData,
    segments: list[GeometrySegment],
    face_node_map: Dict[int, Dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    for segment in segments:
        if segment.segment_type == "aorta_trunk":
            continue
        terminal = dict(face_node_map.get(int(segment.terminal_face_id), {}).get("termination", {})) if segment.terminal_face_id is not None else {}
        boundary = _extract_branch_proximal_boundary(
            surface,
            segment,
            expected_area=float(terminal["area"]) if terminal.get("area") is not None else None,
            expected_diameter=float(terminal["diameter_eq"]) if terminal.get("diameter_eq") is not None else None,
        )
        if boundary is not None:
            boundary.source_type = "parent_junction_derived_surface_opening_cut"
        if boundary is None:
            warning = (
                f"W_STEP2_PROXIMAL_BOUNDARY_UNRESOLVED: segment {segment.segment_id} "
                f"({segment.name_hint}) kept graph-node proximal start."
            )
            segment.proximal_boundary_warning = warning
            warnings.append(warning)
            continue
        if float(boundary.confidence) < 0.75:
            warning = (
                f"W_STEP2_PROXIMAL_BOUNDARY_LOW_CONFIDENCE: segment {segment.segment_id} "
                f"({segment.name_hint}) used fallback boundary selection."
            )
            segment.proximal_boundary_warning = warning
            warnings.append(warning)
        segment.proximal_boundary = boundary
        segment.proximal_boundary_attempts = boundary.attempts
        if segment.segment_type == "topology_branch" and segment.distal_junction_node_ids:
            distal_boundary = _extract_tunnel_boundary_at_arclength(
                surface,
                segment,
                float(segment.length),
                "distal_child_junction_surface_cut",
            )
            if distal_boundary is not None:
                segment.distal_boundaries = [distal_boundary]
    return warnings


def _surface_cell_adjacency(surface: vtk.vtkPolyData) -> list[set[int]]:
    edge_to_cells: dict[tuple[int, int], list[int]] = {}
    for cell_id in range(surface.GetNumberOfCells()):
        cell = surface.GetCell(int(cell_id))
        ids = cell.GetPointIds()
        n_ids = ids.GetNumberOfIds()
        for idx in range(n_ids):
            a = int(ids.GetId(idx))
            b = int(ids.GetId((idx + 1) % n_ids))
            edge = (a, b) if a < b else (b, a)
            edge_to_cells.setdefault(edge, []).append(int(cell_id))

    adjacency = [set() for _ in range(surface.GetNumberOfCells())]
    for cells in edge_to_cells.values():
        if len(cells) < 2:
            continue
        for cell_id in cells:
            adjacency[cell_id].update(other for other in cells if other != cell_id)
    return adjacency


def _cleanup_small_label_islands(
    surface: vtk.vtkPolyData,
    labels: np.ndarray,
    protected_face_ids: set[int],
    assignment_mode: np.ndarray,
    *,
    protected_cells: Optional[np.ndarray] = None,
    max_island_cells: int = 20,
) -> Dict[str, int]:
    model_face = get_cell_array(surface, "ModelFaceID")
    protected = np.zeros((surface.GetNumberOfCells(),), dtype=bool)
    if model_face is not None and protected_face_ids:
        protected = np.isin(model_face.astype(int), list(protected_face_ids))
    if protected_cells is not None:
        protected |= np.asarray(protected_cells, dtype=bool)

    adjacency = _surface_cell_adjacency(surface)
    visited = np.zeros((surface.GetNumberOfCells(),), dtype=bool)
    relabeled_components = 0
    relabeled_cells = 0

    for start_cell in range(surface.GetNumberOfCells()):
        if visited[start_cell]:
            continue
        label = int(labels[start_cell])
        stack = [int(start_cell)]
        component: list[int] = []
        visited[start_cell] = True
        neighbor_labels: dict[int, int] = {}
        touches_protected = False

        while stack:
            cell_id = stack.pop()
            component.append(cell_id)
            touches_protected = touches_protected or bool(protected[cell_id])
            for nbr in adjacency[cell_id]:
                nbr_label = int(labels[nbr])
                if nbr_label == label and not visited[nbr]:
                    visited[nbr] = True
                    stack.append(int(nbr))
                elif nbr_label != label and nbr_label > 0:
                    neighbor_labels[nbr_label] = neighbor_labels.get(nbr_label, 0) + 1

        if touches_protected or len(component) > max_island_cells or not neighbor_labels:
            continue
        new_label = max(neighbor_labels.items(), key=lambda item: item[1])[0]
        if int(new_label) == label:
            continue
        for cell_id in component:
            labels[cell_id] = int(new_label)
            assignment_mode[cell_id] = 6
        relabeled_components += 1
        relabeled_cells += len(component)

    return {
        "cleanup_relabel_components": int(relabeled_components),
        "cleanup_relabel_cells": int(relabeled_cells),
        "cleanup_max_island_cells": int(max_island_cells),
    }


def _segment_assignment_radius(segment: GeometrySegment) -> float:
    if segment.segment_type == "aorta_trunk":
        return 3.0
    if segment.proximal_boundary is not None and segment.proximal_boundary.equivalent_diameter is not None:
        return max(0.5 * float(segment.proximal_boundary.equivalent_diameter), 0.45)
    return 1.0


def _branch_ostium_footprint_radius(segment: GeometrySegment, fallback_radius: float) -> float:
    boundary = segment.proximal_boundary
    if boundary is None:
        return max(float(fallback_radius), 0.45)

    eq_radius = (
        0.5 * float(boundary.equivalent_diameter)
        if boundary.equivalent_diameter is not None and float(boundary.equivalent_diameter) > 0.0
        else float(fallback_radius)
    )
    major_radius = (
        0.5 * float(boundary.major_diameter)
        if boundary.major_diameter is not None and float(boundary.major_diameter) > 0.0
        else eq_radius
    )
    # Oblique ostium cuts can have a longer major axis, but letting that axis dominate
    # makes branch labels smear onto the parent wall. Use it only as a capped allowance.
    capped_major = min(major_radius, eq_radius + max(0.18, 0.18 * eq_radius))
    return max(float(fallback_radius), eq_radius, capped_major, 0.45)


def _assignment_polyline(segment: GeometrySegment) -> np.ndarray:
    points = np.asarray(segment.points, dtype=float)
    if segment.segment_type != "aorta_trunk" and segment.proximal_boundary is not None:
        return _trim_polyline_from_arclength(points, float(segment.proximal_boundary.arclength))
    return points


def _loop_polydata(loop_points: np.ndarray) -> vtk.vtkPolyData:
    pts_np = np.asarray(loop_points, dtype=float)
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    if pts_np.shape[0] < 3:
        out = vtk.vtkPolyData()
        out.SetPoints(points)
        out.SetLines(lines)
        return out

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(int(pts_np.shape[0]) + 1)
    for idx, point in enumerate(pts_np):
        points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        polyline.GetPointIds().SetId(int(idx), int(idx))
    polyline.GetPointIds().SetId(int(pts_np.shape[0]), 0)
    lines.InsertNextCell(polyline)

    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)
    return out


def _build_boundary_loop_debug_polydata(segments: list[GeometrySegment]) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    segment_ids: list[int] = []
    for segment in segments:
        boundaries = [segment.proximal_boundary] + list(segment.distal_boundaries)
        for boundary in boundaries:
            if boundary is None or boundary.loop_points is None:
                continue
            loop = np.asarray(boundary.loop_points, dtype=float)
            if loop.shape[0] < 3:
                continue
            start_id = points.GetNumberOfPoints()
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(int(loop.shape[0]) + 1)
            for idx, point in enumerate(loop):
                points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
                polyline.GetPointIds().SetId(int(idx), int(start_id + idx))
            polyline.GetPointIds().SetId(int(loop.shape[0]), int(start_id))
            lines.InsertNextCell(polyline)
            segment_ids.append(int(segment.segment_id))

    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)
    if segment_ids:
        add_int_cell_array(out, "SegmentId", segment_ids)
    return out


def _split_surface_by_plane(surface: vtk.vtkPolyData, origin: np.ndarray, normal: np.ndarray) -> vtk.vtkPolyData:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    n = unit(normal)
    plane.SetNormal(float(n[0]), float(n[1]), float(n[2]))

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surface)
    clipper.SetClipFunction(plane)
    clipper.GenerateClippedOutputOn()
    clipper.GenerateClipScalarsOff()
    clipper.Update()

    append = vtk.vtkAppendPolyData()
    positive = clipper.GetOutput()
    negative = clipper.GetClippedOutput()
    if positive is not None and positive.GetNumberOfCells() > 0:
        append.AddInputData(positive)
    if negative is not None and negative.GetNumberOfCells() > 0:
        append.AddInputData(negative)
    append.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    return out


def _partition_surface_by_branch_boundaries(
    surface: vtk.vtkPolyData,
    segments: list[GeometrySegment],
) -> tuple[vtk.vtkPolyData, Dict[str, int]]:
    partitioned = vtk.vtkPolyData()
    partitioned.DeepCopy(surface)
    authored_cuts = 0
    skipped_cuts = 0
    input_cells = int(surface.GetNumberOfCells())

    for segment in segments:
        if segment.segment_type == "aorta_trunk":
            continue
        for boundary in [segment.proximal_boundary] + list(segment.distal_boundaries):
            if boundary is None or boundary.loop_points is None:
                continue
            loop = np.asarray(boundary.loop_points, dtype=float)
            if loop.shape[0] < 3:
                skipped_cuts += 1
                continue
            partitioned = _split_surface_by_plane(partitioned, np.asarray(boundary.centroid, dtype=float), np.asarray(boundary.normal, dtype=float))
            authored_cuts += 1

    return partitioned, {
        "mesh_partition_input_cells": input_cells,
        "mesh_partition_output_cells": int(partitioned.GetNumberOfCells()),
        "mesh_partition_authored_cuts": int(authored_cuts),
        "mesh_partition_skipped_cuts": int(skipped_cuts),
    }


def _build_ostium_barriers(
    centers: np.ndarray,
    adjacency: list[set[int]],
    segments: list[GeometrySegment],
) -> tuple[set[tuple[int, int]], np.ndarray, Dict[str, int]]:
    n_cells = int(centers.shape[0])
    blocked_edges: set[tuple[int, int]] = set()
    barrier_segment_id = np.zeros((n_cells,), dtype=np.int32)
    barrier_cells: set[int] = set()
    interfaces_used = 0
    segment_by_id = {int(seg.segment_id): seg for seg in segments}

    for child in segments:
        if child.segment_type == "aorta_trunk" or child.proximal_boundary is None or child.parent_segment_id is None:
            continue
        parent = segment_by_id.get(int(child.parent_segment_id))
        if parent is None:
            continue
        radius = _segment_assignment_radius(child)
        parent_radius = _segment_assignment_radius(parent)
        boundary = child.proximal_boundary
        normal = unit(np.asarray(boundary.normal, dtype=float))
        if float(np.linalg.norm(normal)) <= 1.0e-12:
            continue
        patch_radius = max(1.0, 1.35 * radius)
        patch_ids = np.flatnonzero(np.linalg.norm(centers - boundary.centroid.reshape(1, 3), axis=1) <= patch_radius)
        if patch_ids.size == 0:
            continue
        interfaces_used += 1
        patch_set = {int(v) for v in patch_ids.tolist()}
        signed = (centers[patch_ids] - boundary.centroid.reshape(1, 3)) @ normal
        signed_by_cell = {int(cell_id): float(value) for cell_id, value in zip(patch_ids.tolist(), signed.tolist())}
        patch_centers = centers[patch_ids]
        child_distance, _ = _project_points_to_polyline(patch_centers, _assignment_polyline(child))
        parent_distance, _ = _project_points_to_polyline(patch_centers, _assignment_polyline(parent))
        child_score = child_distance / max(radius, 0.45)
        parent_score = parent_distance / max(parent_radius, 0.45)
        signed_floor = -max(0.02 * radius, 0.03)
        child_like_values = (
            (signed >= signed_floor)
            & (child_distance <= max(1.35 * radius, radius + 0.25))
            & (child_score + 0.35 < parent_score)
        )
        child_like_by_cell = {int(cell_id): bool(value) for cell_id, value in zip(patch_ids.tolist(), child_like_values.tolist())}

        band = max(0.10 * radius, 0.08)
        for cell_id in patch_set:
            if abs(signed_by_cell[cell_id]) <= max(0.25 * radius, 0.15):
                barrier_cells.add(cell_id)
                barrier_segment_id[cell_id] = int(child.segment_id)
            for nbr in adjacency[cell_id]:
                nbr = int(nbr)
                if nbr not in patch_set or cell_id > nbr:
                    continue
                a = signed_by_cell[cell_id]
                b = signed_by_cell[nbr]
                crosses_plane = (a <= -band and b >= band) or (b <= -band and a >= band)
                changes_parent_child_side = child_like_by_cell[cell_id] != child_like_by_cell[nbr]
                if crosses_plane or changes_parent_child_side:
                    blocked_edges.add((cell_id, nbr) if cell_id < nbr else (nbr, cell_id))

    return blocked_edges, barrier_segment_id, {
        "barrier_interfaces_used": int(interfaces_used),
        "barrier_cell_count": int(len(barrier_cells)),
        "blocked_edge_count": int(len(blocked_edges)),
    }


def _branch_origin_allowed_mask(
    centers: np.ndarray,
    candidates: np.ndarray,
    segment: GeometrySegment,
    radius: float,
    distance: np.ndarray,
    arclength: np.ndarray,
    distance_by_segment: dict[int, np.ndarray],
    radius_by_segment: dict[int, float],
    *,
    strict: bool,
) -> np.ndarray:
    if candidates.size == 0:
        return np.zeros((0,), dtype=bool)
    if segment.segment_type == "aorta_trunk" or segment.proximal_boundary is None:
        return np.ones((candidates.size,), dtype=bool)

    normal = unit(np.asarray(segment.proximal_boundary.normal, dtype=float))
    origin_vectors = centers[candidates] - segment.proximal_boundary.centroid.reshape(1, 3)
    signed = (origin_vectors * normal).sum(axis=1)
    radial_vectors = origin_vectors - signed.reshape(-1, 1) * normal.reshape(1, 3)
    radial_distance = np.sqrt((radial_vectors * radial_vectors).sum(axis=1))
    signed_floor = -max(0.02 * radius, 0.03)
    allowed = signed >= signed_floor

    footprint_radius = _branch_ostium_footprint_radius(segment, radius)
    signed_forward = np.maximum(signed, 0.0)
    footprint_tolerance = max(0.10 * radius, 0.08) if strict else max(0.18 * radius, 0.12)
    footprint_growth = np.minimum(0.20 * signed_forward, max(0.45 * radius, 0.35))
    footprint_limit = footprint_radius + footprint_tolerance + footprint_growth
    inside_footprint = radial_distance <= footprint_limit
    origin_footprint_zone = (arclength[candidates] <= max(1.20 * radius, 0.80)) | (
        signed <= max(0.85 * radius, 0.55)
    )
    allowed &= (~origin_footprint_zone) | inside_footprint

    if segment.parent_segment_id is None:
        return allowed

    parent_id = int(segment.parent_segment_id)
    parent_distance = distance_by_segment.get(parent_id)
    if parent_distance is None:
        return allowed

    parent_radius = max(radius_by_segment.get(parent_id, 1.0), 0.45)
    child_score = distance[candidates] / max(radius, 0.45)
    parent_score = parent_distance[candidates] / parent_radius
    near_origin = (arclength[candidates] <= max(1.50 * radius, 0.90)) | (signed <= max(0.55 * radius, 0.30))
    if strict:
        # Inside-footprint cells still need parent competition, just a smaller margin.
        # This prevents aorta-wall cells flanking the ostium from getting branch ownership.
        parent_margin = np.where(inside_footprint, 0.10, 0.38)
        parent_competition_required = near_origin
    else:
        parent_margin = 0.28
        parent_competition_required = near_origin & ~inside_footprint
    allowed &= (~parent_competition_required) | (child_score + parent_margin < parent_score)
    return allowed


def _precompute_segment_projections(
    centers: np.ndarray,
    segments: list[GeometrySegment],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float], np.ndarray]:
    distance_by_segment: dict[int, np.ndarray] = {}
    arclength_by_segment: dict[int, np.ndarray] = {}
    radius_by_segment: dict[int, float] = {}
    nearest_segment = np.zeros((centers.shape[0],), dtype=np.int32)
    nearest_score = np.full((centers.shape[0],), np.inf, dtype=float)

    for segment in segments:
        seg_id = int(segment.segment_id)
        radius = _segment_assignment_radius(segment)
        distance, arclength = _project_points_to_polyline(centers, _assignment_polyline(segment))
        distance_by_segment[seg_id] = distance
        arclength_by_segment[seg_id] = arclength
        radius_by_segment[seg_id] = radius

    all_cells = np.arange(centers.shape[0], dtype=np.int32)
    for segment in segments:
        seg_id = int(segment.segment_id)
        radius = radius_by_segment[seg_id]
        distance = distance_by_segment[seg_id]
        score = distance / max(radius, 0.45)
        if segment.segment_type != "aorta_trunk":
            allowed = _branch_origin_allowed_mask(
                centers,
                all_cells,
                segment,
                radius,
                distance,
                arclength_by_segment[seg_id],
                distance_by_segment,
                radius_by_segment,
                strict=False,
            )
            score = np.where(allowed, score, np.inf)
        better = score < nearest_score
        nearest_score[better] = score[better]
        nearest_segment[better] = seg_id
    return distance_by_segment, arclength_by_segment, radius_by_segment, nearest_segment


def _assign_seed(
    labels: np.ndarray,
    assignment_mode: np.ndarray,
    seed_segment_id: np.ndarray,
    fixed: np.ndarray,
    seed_score: np.ndarray,
    cell_ids: np.ndarray,
    segment_id: int,
    mode: int,
    score_values: Optional[np.ndarray] = None,
    *,
    force: bool = False,
) -> int:
    if cell_ids.size == 0:
        return 0
    if score_values is None:
        score_values = np.zeros((cell_ids.size,), dtype=float)
    assigned = 0
    for cell_id_raw, score_raw in zip(cell_ids.tolist(), score_values.tolist()):
        cell_id = int(cell_id_raw)
        score = float(score_raw)
        if fixed[cell_id] and not force:
            continue
        if force or labels[cell_id] <= 0 or score < float(seed_score[cell_id]):
            labels[cell_id] = int(segment_id)
            assignment_mode[cell_id] = int(mode)
            seed_segment_id[cell_id] = int(segment_id)
            seed_score[cell_id] = float(score)
            assigned += 1
    return assigned


def _build_segment_seeds(
    labels: np.ndarray,
    assignment_mode: np.ndarray,
    seed_segment_id: np.ndarray,
    fixed: np.ndarray,
    centers: np.ndarray,
    model_face: Optional[np.ndarray],
    segments: list[GeometrySegment],
    inlet_face_id: int,
    segment_by_terminal_face: dict[int, int],
    distance_by_segment: dict[int, np.ndarray],
    arclength_by_segment: dict[int, np.ndarray],
    radius_by_segment: dict[int, float],
    barrier_segment_id: np.ndarray,
) -> Dict[str, int]:
    seed_score = np.full((labels.shape[0],), np.inf, dtype=float)
    counts: Dict[str, int] = {
        "inlet_face_seed_cells": 0,
        "terminal_face_seed_cells": 0,
        "ostium_footprint_seed_cells": 0,
        "core_seed_cells": 0,
        "weak_core_seed_cells": 0,
    }
    seed_counts_by_segment: dict[int, int] = {int(seg.segment_id): 0 for seg in segments}

    if model_face is not None:
        inlet_ids = np.flatnonzero(model_face.astype(int) == int(inlet_face_id))
        counts["inlet_face_seed_cells"] = _assign_seed(
            labels,
            assignment_mode,
            seed_segment_id,
            fixed,
            seed_score,
            inlet_ids,
            1,
            1,
            force=True,
        )
        fixed[inlet_ids] = True
        seed_counts_by_segment[1] += int(inlet_ids.size)

        for face_id, segment_id in segment_by_terminal_face.items():
            face_ids = np.flatnonzero(model_face.astype(int) == int(face_id))
            assigned = _assign_seed(
                labels,
                assignment_mode,
                seed_segment_id,
                fixed,
                seed_score,
                face_ids,
                int(segment_id),
                2,
                force=True,
            )
            fixed[face_ids] = True
            counts["terminal_face_seed_cells"] += int(assigned)
            seed_counts_by_segment[int(segment_id)] = seed_counts_by_segment.get(int(segment_id), 0) + int(face_ids.size)

    for segment in segments:
        seg_id = int(segment.segment_id)
        radius = radius_by_segment[seg_id]
        distance = distance_by_segment[seg_id]
        arclength = arclength_by_segment[seg_id]
        available = ~fixed
        if segment.segment_type == "aorta_trunk":
            seed_radius = 1.35
            candidates = np.flatnonzero(available & (distance <= seed_radius) & (barrier_segment_id == 0))
            scores = distance[candidates] / max(seed_radius, 0.45) if candidates.size else np.zeros((0,), dtype=float)
        else:
            ostium_seed_radius = max(1.20 * radius, 0.65)
            ostium_seed_offset = max(1.10 * radius, 0.75)
            ostium_candidates = np.flatnonzero(
                available
                & (distance <= ostium_seed_radius)
                & (arclength <= ostium_seed_offset)
            )
            if ostium_candidates.size:
                ostium_candidates = ostium_candidates[
                    _branch_origin_allowed_mask(
                        centers,
                        ostium_candidates,
                        segment,
                        radius,
                        distance,
                        arclength,
                        distance_by_segment,
                        radius_by_segment,
                        strict=True,
                    )
                ]
            if ostium_candidates.size:
                ostium_scores = (
                    -0.25
                    + 0.50 * distance[ostium_candidates] / max(radius, 0.45)
                    + 0.20 * arclength[ostium_candidates] / max(ostium_seed_offset, 1.0e-6)
                )
                ostium_assigned = _assign_seed(
                    labels,
                    assignment_mode,
                    seed_segment_id,
                    fixed,
                    seed_score,
                    ostium_candidates,
                    seg_id,
                    8,
                    ostium_scores,
                )
                counts["ostium_footprint_seed_cells"] += int(ostium_assigned)
                seed_counts_by_segment[seg_id] = seed_counts_by_segment.get(seg_id, 0) + int(ostium_assigned)

            core_offset = max(1.25 * radius, 0.80)
            seed_radius = max(0.85 * radius, 0.40)
            candidates = np.flatnonzero(available & (distance <= seed_radius) & (arclength >= core_offset))
            if candidates.size:
                candidates = candidates[
                    _branch_origin_allowed_mask(
                        centers,
                        candidates,
                        segment,
                        radius,
                        distance,
                        arclength,
                        distance_by_segment,
                        radius_by_segment,
                        strict=True,
                    )
                ]
            scores = distance[candidates] / max(radius, 0.45) if candidates.size else np.zeros((0,), dtype=float)

        assigned = _assign_seed(
            labels,
            assignment_mode,
            seed_segment_id,
            fixed,
            seed_score,
            candidates,
            seg_id,
            3,
            scores,
        )
        counts["core_seed_cells"] += int(assigned)
        seed_counts_by_segment[seg_id] = seed_counts_by_segment.get(seg_id, 0) + int(assigned)

        if seed_counts_by_segment.get(seg_id, 0) <= 0:
            weak_candidates = np.flatnonzero((~fixed) & np.isfinite(distance))
            if segment.segment_type != "aorta_trunk":
                weak_candidates = weak_candidates[arclength[weak_candidates] >= max(0.25 * radius, 0.2)]
                if weak_candidates.size:
                    weak_candidates = weak_candidates[
                        _branch_origin_allowed_mask(
                            centers,
                            weak_candidates,
                            segment,
                            radius,
                            distance,
                            arclength,
                            distance_by_segment,
                            radius_by_segment,
                            strict=False,
                        )
                    ]
            if weak_candidates.size:
                order = np.argsort(distance[weak_candidates])[: min(12, int(weak_candidates.size))]
                selected = weak_candidates[order]
                weak_scores = 1.0 + distance[selected] / max(radius, 0.45)
                weak_assigned = _assign_seed(
                    labels,
                    assignment_mode,
                    seed_segment_id,
                    fixed,
                    seed_score,
                    selected,
                    seg_id,
                    9,
                    weak_scores,
                )
                counts["weak_core_seed_cells"] += int(weak_assigned)
                seed_counts_by_segment[seg_id] = seed_counts_by_segment.get(seg_id, 0) + int(weak_assigned)

    for seg_id, count in sorted(seed_counts_by_segment.items()):
        counts[f"seed_segment_{int(seg_id)}"] = int(count)
    counts["seed_count_total"] = int(sum(seed_counts_by_segment.values()))
    return counts


def _propagate_surface_labels(
    centers: np.ndarray,
    adjacency: list[set[int]],
    labels: np.ndarray,
    assignment_mode: np.ndarray,
    fixed: np.ndarray,
    segments: list[GeometrySegment],
    distance_by_segment: dict[int, np.ndarray],
    arclength_by_segment: dict[int, np.ndarray],
    radius_by_segment: dict[int, float],
    blocked_edges: set[tuple[int, int]],
    rejected_segment: np.ndarray,
) -> np.ndarray:
    segment_by_id = {int(seg.segment_id): seg for seg in segments}
    best_cost = np.full((labels.shape[0],), np.inf, dtype=float)
    queue: list[tuple[float, int, int]] = []
    for cell_id in np.flatnonzero(labels > 0).tolist():
        seg_id = int(labels[int(cell_id)])
        best_cost[int(cell_id)] = 0.0
        heapq.heappush(queue, (0.0, int(cell_id), seg_id))

    while queue:
        cost, cell_id, seg_id = heapq.heappop(queue)
        if cost > float(best_cost[cell_id]) + 1.0e-12 or int(labels[cell_id]) != int(seg_id):
            continue
        segment = segment_by_id.get(int(seg_id))
        if segment is None:
            continue
        radius = max(radius_by_segment.get(int(seg_id), 1.0), 0.45)
        for nbr_raw in adjacency[cell_id]:
            nbr = int(nbr_raw)
            edge = (cell_id, nbr) if cell_id < nbr else (nbr, cell_id)
            if edge in blocked_edges:
                continue
            if fixed[nbr] and int(labels[nbr]) != int(seg_id):
                continue
            dist_to_segment = float(distance_by_segment[int(seg_id)][nbr])
            if segment.segment_type != "aorta_trunk":
                max_branch_distance = max(2.0 * radius + 0.35, 1.35)
                if dist_to_segment > max_branch_distance:
                    continue
                if segment.proximal_boundary is not None:
                    allowed = _branch_origin_allowed_mask(
                        centers,
                        np.asarray([nbr], dtype=np.int32),
                        segment,
                        radius,
                        distance_by_segment[int(seg_id)],
                        arclength_by_segment[int(seg_id)],
                        distance_by_segment,
                        radius_by_segment,
                        strict=True,
                    )
                    if not bool(allowed[0]):
                        rejected_segment[nbr] = int(seg_id)
                        continue
            edge_len = distance(centers[cell_id], centers[nbr])
            dist_norm = dist_to_segment / radius
            new_cost = float(cost + edge_len / radius + 0.18 * dist_norm)
            if new_cost + 1.0e-9 < float(best_cost[nbr]):
                best_cost[nbr] = new_cost
                labels[nbr] = int(seg_id)
                if assignment_mode[nbr] <= 0 or assignment_mode[nbr] == 5:
                    assignment_mode[nbr] = 4
                heapq.heappush(queue, (new_cost, nbr, int(seg_id)))
    return best_cost


def _recover_unassigned_cells(
    labels: np.ndarray,
    assignment_mode: np.ndarray,
    nearest_segment: np.ndarray,
) -> int:
    missing = np.flatnonzero(labels <= 0)
    if missing.size == 0:
        return 0
    labels[missing] = np.maximum(nearest_segment[missing], 1)
    assignment_mode[missing] = 5
    return int(missing.size)


def _segment_owner_priority(segment: GeometrySegment) -> int:
    if segment.segment_type == "terminal_branch":
        return 0
    if segment.segment_type == "topology_branch":
        return 1
    return 2


def _connected_component_in_mask(
    adjacency: list[set[int]],
    allowed: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    seed_ids = [int(v) for v in np.asarray(seeds, dtype=np.int32).tolist() if bool(allowed[int(v)])]
    if not seed_ids:
        return np.zeros((0,), dtype=np.int32)
    visited = np.zeros((allowed.shape[0],), dtype=bool)
    stack = list(dict.fromkeys(seed_ids))
    for cell_id in stack:
        visited[int(cell_id)] = True
    component: list[int] = []
    while stack:
        cell_id = int(stack.pop())
        component.append(cell_id)
        for nbr_raw in adjacency[cell_id]:
            nbr = int(nbr_raw)
            if not visited[nbr] and bool(allowed[nbr]):
                visited[nbr] = True
                stack.append(nbr)
    return np.asarray(component, dtype=np.int32)


def _build_branch_tunnel_ownership(
    centers: np.ndarray,
    model_face: Optional[np.ndarray],
    adjacency: list[set[int]],
    segments: list[GeometrySegment],
    distance_by_segment: dict[int, np.ndarray],
    arclength_by_segment: dict[int, np.ndarray],
    radius_by_segment: dict[int, float],
) -> tuple[np.ndarray, Dict[str, int], Dict[str, np.ndarray]]:
    n_cells = int(centers.shape[0])
    owner_id = np.zeros((n_cells,), dtype=np.int32)
    owner_score = np.full((n_cells,), np.inf, dtype=float)
    owner_priority = np.full((n_cells,), 99, dtype=np.int32)
    tunnel_seed_id = np.zeros((n_cells,), dtype=np.int32)
    tunnel_candidate_id = np.zeros((n_cells,), dtype=np.int32)
    tunnel_component_id = np.zeros((n_cells,), dtype=np.int32)
    overlap_reassignments = 0
    terminal_priority_wins = 0
    topology_seed_cells = 0
    terminal_seed_cells = 0
    component_claimed_cells = 0
    empty_candidate_segments = 0
    missing_seed_segments = 0
    component_index = 0

    def claim_cells(
        cell_ids: np.ndarray,
        segment: GeometrySegment,
        scores: np.ndarray,
    ) -> int:
        nonlocal overlap_reassignments, terminal_priority_wins
        if cell_ids.size == 0:
            return 0
        seg_id = int(segment.segment_id)
        priority = _segment_owner_priority(segment)
        current_scores = owner_score[cell_ids]
        current_priority = owner_priority[cell_ids]
        current_owner = owner_id[cell_ids]
        better = scores + 1.0e-9 < current_scores
        priority_tie = (np.abs(scores - current_scores) <= 0.10) & (priority < current_priority)
        update = better | priority_tie
        if not np.any(update):
            return 0

        updated_cells = cell_ids[update]
        replaced = current_owner[update] > 0
        overlap_reassignments += int(np.count_nonzero(replaced))
        terminal_priority_wins += int(
            np.count_nonzero(
                replaced
                & (segment.segment_type == "terminal_branch")
                & (priority < current_priority[update])
            )
        )
        owner_id[updated_cells] = seg_id
        owner_score[updated_cells] = scores[update]
        owner_priority[updated_cells] = int(priority)
        return int(updated_cells.size)

    for segment in sorted(segments, key=_segment_owner_priority):
        if segment.segment_type == "aorta_trunk":
            continue
        seg_id = int(segment.segment_id)
        radius = max(float(radius_by_segment.get(seg_id, 1.0)), 0.45)
        distance_to_segment = distance_by_segment[seg_id]
        arclength = arclength_by_segment[seg_id]
        assignment_points = _assignment_polyline(segment)
        assignment_length = float(cumulative_arclength(assignment_points)[-1]) if assignment_points.shape[0] >= 2 else 0.0
        if assignment_points.shape[0] < 2 or assignment_length <= 1.0e-6:
            continue

        max_s = assignment_length + max(0.25 * radius, 0.12)
        candidate_radius = max(1.35 * radius + 0.14, 0.64)
        priority_bias = -0.08 if segment.segment_type == "terminal_branch" else 0.04
        all_cells = np.arange(n_cells, dtype=np.int32)
        candidate = _branch_origin_allowed_mask(
            centers,
            all_cells,
            segment,
            radius,
            distance_to_segment,
            arclength,
            distance_by_segment,
            radius_by_segment,
            strict=True,
        )
        candidate &= (distance_to_segment <= candidate_radius) & (arclength <= max_s)
        candidate_ids = np.flatnonzero(candidate)
        if candidate_ids.size == 0:
            empty_candidate_segments += 1
            continue
        tunnel_candidate_id[candidate & (tunnel_candidate_id == 0)] = seg_id

        seed_mask = np.zeros((n_cells,), dtype=bool)
        if model_face is not None and segment.terminal_face_id is not None:
            face_seed = np.flatnonzero(model_face.astype(int) == int(segment.terminal_face_id))
            if face_seed.size:
                seed_mask[face_seed] = True
                terminal_seed_cells += int(face_seed.size)
        if not np.any(seed_mask):
            midpoint = 0.50 * assignment_length
            seed_band = max(0.18 * assignment_length, max(0.75 * radius, 0.35))
            corridor = candidate & (np.abs(arclength - midpoint) <= seed_band)
            corridor_ids = np.flatnonzero(corridor)
            if corridor_ids.size == 0:
                corridor_ids = candidate_ids
            reference_distance = float(np.min(distance_to_segment[corridor_ids]))
            seed_ids = corridor_ids[distance_to_segment[corridor_ids] <= reference_distance + max(0.16 * radius, 0.07)]
            if seed_ids.size == 0:
                seed_ids = corridor_ids[np.argsort(distance_to_segment[corridor_ids])[: min(20, int(corridor_ids.size))]]
            seed_mask[seed_ids] = True
            topology_seed_cells += int(seed_ids.size)

        seed_ids = np.flatnonzero(seed_mask)
        tunnel_seed_id[seed_ids] = seg_id
        component = _connected_component_in_mask(adjacency, candidate | seed_mask, seed_ids)
        if component.size == 0:
            missing_seed_segments += 1
            continue
        component_index += 1
        tunnel_component_id[component] = int(component_index)
        scores = distance_to_segment[component] / radius + priority_bias
        component_claimed_cells += claim_cells(component, segment, scores)

    return owner_id, {
        "tunnel_seed_cells": int(np.count_nonzero(tunnel_seed_id > 0)),
        "tunnel_terminal_seed_cells": int(terminal_seed_cells),
        "tunnel_topology_seed_cells": int(topology_seed_cells),
        "tunnel_candidate_cells": int(np.count_nonzero(tunnel_candidate_id > 0)),
        "tunnel_component_cells": int(np.count_nonzero(tunnel_component_id > 0)),
        "tunnel_component_count": int(component_index),
        "tunnel_component_claimed_cells": int(component_claimed_cells),
        "tunnel_empty_candidate_segments": int(empty_candidate_segments),
        "tunnel_missing_seed_segments": int(missing_seed_segments),
        "branch_tunnel_candidate_cells": int(np.count_nonzero(tunnel_candidate_id > 0)),
        "branch_tunnel_owned_cells": int(np.count_nonzero(owner_id > 0)),
        "branch_tunnel_owner_segments": int(len(set(int(v) for v in owner_id.tolist() if int(v) > 0))),
        "branch_tunnel_overlap_reassigned_cells": int(overlap_reassignments),
        "branch_tunnel_terminal_priority_wins": int(terminal_priority_wins),
    }, {
        "TunnelCandidateId": tunnel_candidate_id,
        "TunnelSeedId": tunnel_seed_id,
        "TunnelComponentId": tunnel_component_id,
    }


def _dominant_neighbor_label(
    component: np.ndarray,
    labels: np.ndarray,
    adjacency: list[set[int]],
    rejected_mask: np.ndarray,
) -> int:
    neighbor_counts: dict[int, int] = {}
    component_set = {int(v) for v in component.tolist()}
    for cell_id in component_set:
        for nbr_raw in adjacency[cell_id]:
            nbr = int(nbr_raw)
            if nbr in component_set or bool(rejected_mask[nbr]):
                continue
            label = int(labels[nbr])
            if label > 0:
                neighbor_counts[label] = neighbor_counts.get(label, 0) + 1
    if not neighbor_counts:
        return 1
    return int(max(neighbor_counts.items(), key=lambda item: item[1])[0])


def _reject_orphan_branch_components(
    labels: np.ndarray,
    assignment_mode: np.ndarray,
    adjacency: list[set[int]],
    branch_owner_id: np.ndarray,
    model_face: Optional[np.ndarray],
    segments: list[GeometrySegment],
) -> tuple[Dict[str, int], Dict[str, np.ndarray]]:
    rejected_island_id = np.zeros((labels.shape[0],), dtype=np.int32)
    total_components = 0
    kept_components = 0
    rejected_components = 0
    rejected_cells = 0
    segments_with_orphans = 0

    model_face_int = model_face.astype(int) if model_face is not None else None
    for segment in segments:
        if segment.segment_type == "aorta_trunk":
            continue
        seg_id = int(segment.segment_id)
        segment_cells = np.flatnonzero(labels == seg_id)
        if segment_cells.size == 0:
            continue

        segment_mask = labels == seg_id
        visited: set[int] = set()
        components: list[np.ndarray] = []
        for start_raw in segment_cells.tolist():
            start = int(start_raw)
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component: list[int] = []
            while stack:
                cell_id = int(stack.pop())
                component.append(cell_id)
                for nbr_raw in adjacency[cell_id]:
                    nbr = int(nbr_raw)
                    if nbr not in visited and bool(segment_mask[nbr]):
                        visited.add(nbr)
                        stack.append(nbr)
            components.append(np.asarray(component, dtype=np.int32))

        if not components:
            continue
        total_components += int(len(components))

        seed_face_id = segment.terminal_face_id
        seed_touching: list[int] = []
        owner_touching: list[int] = []
        for idx, component in enumerate(components):
            if model_face_int is not None and seed_face_id is not None and np.any(model_face_int[component] == int(seed_face_id)):
                seed_touching.append(idx)
            if np.any(branch_owner_id[component] == seg_id):
                owner_touching.append(idx)

        if seed_touching:
            keep_idx = max(seed_touching, key=lambda idx: int(components[idx].size))
        elif owner_touching:
            keep_idx = max(owner_touching, key=lambda idx: int(components[idx].size))
        else:
            keep_idx = max(range(len(components)), key=lambda idx: int(components[idx].size))
        kept_components += 1

        segment_rejected = 0
        for idx, component in enumerate(components):
            if idx == keep_idx:
                continue
            target_label = _dominant_neighbor_label(component, labels, adjacency, rejected_island_id > 0)
            labels[component] = int(target_label)
            assignment_mode[component] = 9
            rejected_island_id[component] = seg_id
            rejected_components += 1
            rejected_cells += int(component.size)
            segment_rejected += 1
        if segment_rejected:
            segments_with_orphans += 1

    return {
        "branch_connected_components": int(total_components),
        "branch_kept_components": int(kept_components),
        "orphan_branch_components": 0,
        "rejected_orphan_branch_components": int(rejected_components),
        "rejected_orphan_branch_component_cells": int(rejected_cells),
        "branch_segments_with_orphans": int(segments_with_orphans),
    }, {
        "RejectedIslandId": rejected_island_id,
    }


def _assign_surface_cells(
    surface: vtk.vtkPolyData,
    segments: list[GeometrySegment],
    inlet_face_id: int,
    *,
    cleanup: bool = True,
) -> tuple[np.ndarray, Dict[str, int], Dict[int, int], Dict[str, np.ndarray]]:
    n_cells = int(surface.GetNumberOfCells())
    labels = np.full((n_cells,), -1, dtype=np.int32)
    assignment_mode = np.zeros((n_cells,), dtype=np.int32)
    seed_segment_id = np.zeros((n_cells,), dtype=np.int32)
    propagation_cost = np.full((n_cells,), np.inf, dtype=float)
    assignment_counts: Dict[str, int] = {
        SURFACE_ASSIGNMENT_ALGORITHM: 1,
        "fixed_seed_cells": 0,
        "core_seed_cells": 0,
        "propagated_cells": 0,
        "component_nearest_assigned_cells": 0,
        "component_branch_owner_assigned_cells": 0,
        "fallback": 0,
        "aorta_excluded_branch_seed_cells": 0,
        "branch_owned_aorta_labeled_cells": 0,
        "branch_labeled_parent_aorta_wall_cells": 0,
        "orphan_branch_components": 0,
        "rejected_orphan_branch_components": 0,
        "rejected_orphan_branch_component_cells": 0,
    }

    segment_by_terminal_face = {
        int(seg.terminal_face_id): int(seg.segment_id)
        for seg in segments
        if seg.terminal_face_id is not None
    }
    centers = cell_centers(surface)
    model_face = get_cell_array(surface, "ModelFaceID")
    protected_face_ids = {int(inlet_face_id), *segment_by_terminal_face.keys()}
    adjacency = _surface_cell_adjacency(surface)
    distance_by_segment: dict[int, np.ndarray] = {}
    arclength_by_segment: dict[int, np.ndarray] = {}
    radius_by_segment: dict[int, float] = {}
    nearest_segment = np.zeros((n_cells,), dtype=np.int32)
    nearest_score = np.full((n_cells,), np.inf, dtype=float)
    for segment in segments:
        seg_id = int(segment.segment_id)
        radius = _segment_assignment_radius(segment)
        distance_to_segment, arclength = _project_points_to_polyline(centers, _assignment_polyline(segment))
        distance_by_segment[seg_id] = distance_to_segment
        arclength_by_segment[seg_id] = arclength
        radius_by_segment[seg_id] = radius
        score = distance_to_segment / max(radius, 0.45)
        better = score < nearest_score
        nearest_score[better] = score[better]
        nearest_segment[better] = seg_id

    branch_owner_id, branch_owner_counts, branch_owner_debug = _build_branch_tunnel_ownership(
        centers,
        model_face,
        adjacency,
        segments,
        distance_by_segment,
        arclength_by_segment,
        radius_by_segment,
    )
    assignment_counts.update(branch_owner_counts)

    fixed = np.zeros((n_cells,), dtype=bool)
    seed_score = np.full((n_cells,), np.inf, dtype=float)
    seed_counts_by_segment: dict[int, int] = {int(seg.segment_id): 0 for seg in segments}
    if model_face is not None:
        model_face_int = model_face.astype(int)
        inlet_ids = np.flatnonzero(model_face_int == int(inlet_face_id))
        assignment_counts["inlet_face_seed_cells"] = _assign_seed(
            labels,
            assignment_mode,
            seed_segment_id,
            fixed,
            seed_score,
            inlet_ids,
            1,
            1,
            force=True,
        )
        fixed[inlet_ids] = True
        seed_counts_by_segment[1] += int(inlet_ids.size)

        terminal_seed_total = 0
        for face_id, segment_id in segment_by_terminal_face.items():
            face_ids = np.flatnonzero(model_face_int == int(face_id))
            assigned = _assign_seed(
                labels,
                assignment_mode,
                seed_segment_id,
                fixed,
                seed_score,
                face_ids,
                int(segment_id),
                2,
                force=True,
            )
            fixed[face_ids] = True
            terminal_seed_total += int(assigned)
            seed_counts_by_segment[int(segment_id)] = seed_counts_by_segment.get(int(segment_id), 0) + int(face_ids.size)
        assignment_counts["terminal_face_seed_cells"] = int(terminal_seed_total)
    else:
        assignment_counts["inlet_face_seed_cells"] = 0
        assignment_counts["terminal_face_seed_cells"] = 0

    assignment_counts["fixed_seed_cells"] = int(assignment_counts.get("inlet_face_seed_cells", 0)) + int(
        assignment_counts.get("terminal_face_seed_cells", 0)
    )

    for segment in segments:
        seg_id = int(segment.segment_id)
        radius = radius_by_segment[seg_id]
        distance_to_segment = distance_by_segment[seg_id]
        arclength = arclength_by_segment[seg_id]
        available = ~fixed
        if segment.segment_type == "aorta_trunk":
            seed_radius = max(1.25 * radius, 3.0)
            aorta_seed_candidates = available & (distance_to_segment <= seed_radius)
            excluded = aorta_seed_candidates & (branch_owner_id > 0)
            assignment_counts["aorta_excluded_branch_seed_cells"] = int(np.count_nonzero(excluded))
            candidates = np.flatnonzero(aorta_seed_candidates & (branch_owner_id == 0))
        else:
            owner_candidates = available & (branch_owner_id == seg_id)
            candidates = np.flatnonzero(owner_candidates)
        scores = distance_to_segment[candidates] / max(radius, 0.45) if candidates.size else np.zeros((0,), dtype=float)
        assigned = _assign_seed(
            labels,
            assignment_mode,
            seed_segment_id,
            fixed,
            seed_score,
            candidates,
            seg_id,
            3,
            scores,
        )
        assignment_counts["core_seed_cells"] += int(assigned)
        seed_counts_by_segment[seg_id] = seed_counts_by_segment.get(seg_id, 0) + int(assigned)

    segment_by_id = {int(seg.segment_id): seg for seg in segments}
    queue: list[tuple[float, int, int]] = []
    for cell_id in np.flatnonzero(labels > 0).tolist():
        seg_id = int(labels[int(cell_id)])
        propagation_cost[int(cell_id)] = 0.0
        heapq.heappush(queue, (0.0, int(cell_id), seg_id))

    while queue:
        cost, cell_id, seg_id = heapq.heappop(queue)
        if cost > float(propagation_cost[cell_id]) + 1.0e-12 or int(labels[cell_id]) != int(seg_id):
            continue
        segment = segment_by_id.get(int(seg_id))
        if segment is None:
            continue
        radius = max(radius_by_segment.get(int(seg_id), 1.0), 0.45)
        for nbr_raw in adjacency[cell_id]:
            nbr = int(nbr_raw)
            if fixed[nbr] and int(labels[nbr]) != int(seg_id):
                continue
            owner = int(branch_owner_id[nbr])
            if segment.segment_type == "aorta_trunk" and owner > 0:
                continue
            if segment.segment_type != "aorta_trunk" and owner != int(seg_id):
                continue
            dist_to_segment = float(distance_by_segment[int(seg_id)][nbr])
            if segment.segment_type != "aorta_trunk" and dist_to_segment > max(3.5 * radius + 0.75, 2.0):
                continue
            edge_len = distance(centers[cell_id], centers[nbr])
            dist_norm = dist_to_segment / radius
            new_cost = float(cost + edge_len / radius + 0.10 * dist_norm)
            if new_cost + 1.0e-9 < float(propagation_cost[nbr]):
                propagation_cost[nbr] = new_cost
                labels[nbr] = int(seg_id)
                if assignment_mode[nbr] <= 0:
                    assignment_mode[nbr] = 4
                heapq.heappush(queue, (new_cost, nbr, int(seg_id)))
    assignment_counts["propagated_cells"] = int(np.count_nonzero(assignment_mode == 4))

    missing = np.flatnonzero(labels <= 0)
    if missing.size:
        visited = np.zeros((n_cells,), dtype=bool)
        assigned_by_component = 0
        assigned_by_owner_component = 0
        for start_cell in missing.tolist():
            start_cell = int(start_cell)
            if visited[start_cell] or labels[start_cell] > 0:
                continue
            stack = [start_cell]
            component: list[int] = []
            visited[start_cell] = True
            while stack:
                cell_id = stack.pop()
                if labels[cell_id] > 0:
                    continue
                component.append(cell_id)
                for nbr in adjacency[cell_id]:
                    nbr = int(nbr)
                    nbr_label = int(labels[nbr])
                    if nbr_label <= 0 and not visited[nbr]:
                        visited[nbr] = True
                        stack.append(nbr)
            if not component:
                continue
            component_ids = np.asarray(component, dtype=np.int32)
            owned_mask = branch_owner_id[component_ids] > 0
            if np.any(owned_mask):
                owned_ids = component_ids[owned_mask]
                labels[owned_ids] = branch_owner_id[owned_ids]
                assignment_mode[owned_ids] = 7
                assigned_by_owner_component += int(owned_ids.size)
                assigned_by_component += int(owned_ids.size)
                component_ids = component_ids[~owned_mask]
                if component_ids.size == 0:
                    continue
            best_segment_id = 1
            labels[component_ids] = int(best_segment_id)
            assignment_mode[component_ids] = 7
            assigned_by_component += int(component_ids.size)
        assignment_counts["component_nearest_assigned_cells"] = int(assigned_by_component)
        assignment_counts["component_branch_owner_assigned_cells"] = int(assigned_by_owner_component)
    else:
        assignment_counts["component_branch_owner_assigned_cells"] = 0

    owner_missing = np.flatnonzero((labels <= 0) & (branch_owner_id > 0))
    if owner_missing.size:
        labels[owner_missing] = branch_owner_id[owner_missing]
        assignment_mode[owner_missing] = 5
    assignment_counts["fallback_branch_owner_assigned_cells"] = int(owner_missing.size)
    assignment_counts["fallback"] = int(np.count_nonzero(labels <= 0))
    if assignment_counts["fallback"]:
        assignment_counts["fallback"] = _recover_unassigned_cells(labels, assignment_mode, nearest_segment)

    cleanup_counts = {
        "cleanup_relabel_components": 0,
        "cleanup_relabel_cells": 0,
        "cleanup_max_island_cells": 0,
    }
    if cleanup:
        cleanup_counts = _cleanup_small_label_islands(
            surface,
            labels,
            protected_face_ids,
            assignment_mode,
            protected_cells=branch_owner_id > 0,
        )
    assignment_counts.update(cleanup_counts)

    island_counts, island_debug = _reject_orphan_branch_components(
        labels,
        assignment_mode,
        adjacency,
        branch_owner_id,
        model_face,
        segments,
    )
    assignment_counts.update(island_counts)

    terminal_face_mask = np.zeros((n_cells,), dtype=bool)
    if model_face is not None and segment_by_terminal_face:
        terminal_face_mask = np.isin(model_face.astype(int), list(segment_by_terminal_face.keys()))
    branch_parent_wall = np.flatnonzero((labels > 1) & (branch_owner_id == 0) & ~terminal_face_mask)
    assignment_counts["branch_labeled_parent_aorta_wall_before_rescue"] = int(branch_parent_wall.size)
    if branch_parent_wall.size:
        labels[branch_parent_wall] = 1
        assignment_mode[branch_parent_wall] = 8
    assignment_counts["branch_labeled_parent_aorta_wall_cells"] = int(
        np.count_nonzero((labels > 1) & (branch_owner_id == 0) & ~terminal_face_mask)
    )

    branch_owned_as_aorta = np.flatnonzero((branch_owner_id > 0) & (labels == 1))
    assignment_counts["branch_owned_aorta_labels_before_rescue"] = int(branch_owned_as_aorta.size)
    assignment_counts["branch_owned_reassigned_from_aorta"] = 0
    assignment_counts["branch_owned_aorta_labeled_cells"] = int(np.count_nonzero((branch_owner_id > 0) & (labels == 1)))

    counts = {int(seg.segment_id): int(np.count_nonzero(labels == int(seg.segment_id))) for seg in segments}
    for seg in segments:
        seg.cell_count = counts.get(int(seg.segment_id), 0)
        seg.fallback_cell_count = 0
    diagnostics = {
        "AssignmentMode": assignment_mode,
        "SeedSegmentId": seed_segment_id,
        "PropagationCost": np.where(np.isfinite(propagation_cost), propagation_cost, -1.0),
        "NearestSegmentId": nearest_segment,
        "BranchTunnelOwnerId": branch_owner_id,
        **branch_owner_debug,
        **island_debug,
    }
    for seg_id, count in sorted(seed_counts_by_segment.items()):
        assignment_counts[f"seed_segment_{int(seg_id)}"] = int(count)
    assignment_counts["seed_count_total"] = int(sum(seed_counts_by_segment.values()))
    return labels, assignment_counts, counts, diagnostics


def _build_segments_surface(surface: vtk.vtkPolyData, labels: np.ndarray) -> vtk.vtkPolyData:
    out = clone_geometry_only(surface)
    add_int_cell_array(out, "SegmentId", labels.astype(int).tolist())
    colors = [segment_color(int(seg_id)) for seg_id in labels.tolist()]
    add_uchar3_cell_array(out, "SegmentColorRGB", colors)
    return out


def _build_debug_segments_surface(
    surface: vtk.vtkPolyData,
    labels: np.ndarray,
    diagnostics: Dict[str, np.ndarray],
) -> vtk.vtkPolyData:
    out = _build_segments_surface(surface, labels)
    for name, values in diagnostics.items():
        vals = np.asarray(values)
        if vals.dtype.kind in {"i", "u", "b"}:
            add_int_cell_array(out, name, vals.astype(int).tolist())
            continue
        arr = vtk.vtkDoubleArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(1)
        arr.SetNumberOfTuples(int(vals.shape[0]))
        for idx, value in enumerate(vals.tolist()):
            arr.SetValue(idx, float(value))
        out.GetCellData().AddArray(arr)
    return out


def _segment_contract_rows(segments: list[GeometrySegment], node_coords: dict[int, np.ndarray]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for seg in segments:
        proximal_point = node_coords.get(int(seg.proximal_node), seg.points[0]) if seg.segment_type == "aorta_trunk" else seg.effective_proximal_point
        row = {
            "segment_id": int(seg.segment_id),
            "name_hint": str(seg.name_hint),
            "segment_type": str(seg.segment_type),
            "parent_segment_id": seg.parent_segment_id,
            "child_segment_ids": [int(v) for v in sorted(seg.child_segment_ids)],
            "proximal_node_id": int(seg.proximal_node),
            "distal_node_id": int(seg.distal_node),
            "proximal_point": np.asarray(proximal_point, dtype=float).tolist(),
            "distal_point": node_coords.get(int(seg.distal_node), seg.points[-1]).tolist(),
            "edge_ids": [int(eid) for eid in seg.edge_ids],
            "length": float(seg.length),
            "terminal_face_id": seg.terminal_face_id,
            "terminal_face_name": seg.terminal_face_name,
            "descendant_terminal_names": [str(v) for v in seg.descendant_terminal_names],
            "tunnel_source_outlets": [str(v) for v in seg.tunnel_source_outlets],
            "parent_junction_node_id": seg.parent_junction_node_id,
            "distal_junction_node_ids": [int(v) for v in seg.distal_junction_node_ids],
            "outlet_route_node_paths": {str(name): [int(v) for v in path] for name, path in seg.outlet_route_node_paths.items()},
            "tunnel_assignment_algorithm": TUNNEL_ASSIGNMENT_ALGORITHM,
            "vmtk_group_ids": [int(v) for v in sorted(seg.vmtk_group_ids)],
            "assignment_source": "vmtk_branch_clip_group",
            "cell_count": int(seg.cell_count),
            "fallback_cell_count": int(seg.fallback_cell_count),
        }
        if seg.proximal_boundary is not None:
            row["proximal_boundary"] = seg.proximal_boundary.to_contract()
            row["proximal_boundary_source"] = seg.proximal_boundary.source_type
            row["proximal_boundary_confidence"] = float(seg.proximal_boundary.confidence)
            row["proximal_boundary_arclength"] = float(seg.proximal_boundary.arclength)
            row["proximal_boundary_selection_algorithm"] = PROXIMAL_BOUNDARY_SELECTION_ALGORITHM
            if seg.proximal_boundary.loop_points is not None:
                row["cut_loop_point_count"] = int(np.asarray(seg.proximal_boundary.loop_points).shape[0])
                row["cut_method"] = "surface_plane_intersection_loop"
                row["loop_area"] = float(seg.proximal_boundary.area)
                row["mesh_partition_algorithm"] = MESH_PARTITION_ALGORITHM
            row["proximal_cut_loop_point_count"] = row.get("cut_loop_point_count", 0)
            if seg.proximal_boundary_warning:
                row["proximal_boundary_warning"] = seg.proximal_boundary_warning
        elif seg.segment_type != "aorta_trunk":
            row["proximal_boundary_source"] = "step1_graph_node_fallback"
            row["proximal_boundary_confidence"] = 0.45
            row["proximal_boundary_arclength"] = 0.0
            if seg.proximal_boundary_warning:
                row["proximal_boundary_warning"] = seg.proximal_boundary_warning
            if seg.proximal_boundary_attempts:
                row["proximal_boundary_attempts"] = seg.proximal_boundary_attempts
        if seg.distal_boundaries:
            row["distal_boundaries"] = [boundary.to_contract() for boundary in seg.distal_boundaries]
            row["distal_cut_loop_point_count"] = int(sum(np.asarray(boundary.loop_points).shape[0] for boundary in seg.distal_boundaries if boundary.loop_points is not None))
        else:
            row["distal_cut_loop_point_count"] = 0
        rows.append(row)
    return rows


def _boundary_summary(
    start_profile: Dict[str, Any],
    end_profile: Dict[str, Any],
    face_node_map: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    terminal_profiles: list[Dict[str, Any]] = []
    for face_id, row in sorted(face_node_map.items()):
        term = dict(row.get("termination", {}))
        terminal_profiles.append(
            {
                "face_id": int(face_id),
                "face_name": str(row.get("face_name", "")),
                "terminal_node_id": int(row.get("terminal_node_id", -1)),
                "center": term.get("center"),
                "area": term.get("area"),
                "equivalent_diameter": term.get("diameter_eq"),
                "normal": term.get("normal"),
                "source": term.get("source"),
                "confidence": 0.9 if not term.get("fallback_used") else 0.5,
            }
        )
    return {
        "aorta_start": start_profile,
        "aorta_end_pre_bifurcation": end_profile,
        "mapped_terminal_boundaries": terminal_profiles,
    }


def _make_contract(
    *,
    status: str,
    warnings: list[str],
    paths: Dict[str, str],
    step1_metadata: Optional[Dict[str, Any]] = None,
    start_profile: Optional[Dict[str, Any]] = None,
    end_metadata: Optional[Dict[str, Any]] = None,
    aorta_length: Optional[float] = None,
    aorta_node_path: Optional[list[int]] = None,
    segments: Optional[list[GeometrySegment]] = None,
    node_coords: Optional[dict[int, np.ndarray]] = None,
    face_node_map: Optional[Dict[int, Dict[str, Any]]] = None,
    assignment_counts: Optional[Dict[str, int]] = None,
    vmtk_diagnostics: Optional[Dict[str, Any]] = None,
    total_cells: Optional[int] = None,
) -> Dict[str, Any]:
    total_fallback = int((assignment_counts or {}).get("fallback", 0))
    assigned_cells = int(total_cells or 0)
    fallback_pct = float(total_fallback / assigned_cells) if assigned_cells else 0.0
    priority_fallback = False

    qa_status = status
    if status == "success":
        boundary_requires_review = any("W_STEP2_PROXIMAL_BOUNDARY_" in str(w) for w in warnings)
        if fallback_pct > 0.15:
            qa_status = "failed"
            warnings.append("W_STEP2_FALLBACK_GT_15_PERCENT: fallback cell assignment exceeded 15%.")
        elif fallback_pct > 0.05 or priority_fallback:
            qa_status = "requires_review"
            warnings.append("W_STEP2_FALLBACK_REQUIRES_REVIEW: fallback assignment exceeded review policy.")
        elif boundary_requires_review:
            qa_status = "requires_review"

    assignment_count_values = {
        str(key): int(value)
        for key, value in (assignment_counts or {}).items()
        if isinstance(value, (int, np.integer))
    }
    seed_counts_by_segment = {
        key.replace("seed_segment_", ""): int(value)
        for key, value in assignment_count_values.items()
        if key.startswith("seed_segment_")
    }
    segment_rows = _segment_contract_rows(segments or [], node_coords or {}) if segments else []
    boundary = _boundary_summary(start_profile or {}, end_metadata or {}, face_node_map or {}) if start_profile and end_metadata else {}
    non_empty_segments = sum(1 for row in segment_rows if int(row.get("cell_count", 0)) > 0)
    return {
        "schema_version": 1,
        "step_name": "step2_geometry_contract",
        "step_status": qa_status,
        "warnings": sorted(set(str(w) for w in warnings)),
        "input_paths": {
            "input_vtp": paths.get("input_vtp"),
            "face_map": paths.get("face_map"),
            "surface_cleaned": paths.get("surface_cleaned"),
            "centerlines_raw_debug": paths.get("centerlines_raw_debug"),
            "centerline_network": paths.get("centerline_network"),
            "step1_metadata": paths.get("step1_metadata"),
        },
        "output_paths": {
            "segments_vtp": paths.get("segments_vtp"),
            "aorta_centerline": paths.get("aorta_centerline"),
            "contract_json": paths.get("contract_json"),
            **({"boundary_debug_vtp": paths["boundary_debug_vtp"]} if "boundary_debug_vtp" in paths else {}),
            **({"step2_debug_vtp": paths["step2_debug_vtp"]} if "step2_debug_vtp" in paths else {}),
            **({"step2_debug_json": paths["step2_debug_json"]} if "step2_debug_json" in paths else {}),
        },
        "upstream_references": {
            "surface_cleaned": paths.get("surface_cleaned"),
            "centerlines_raw_debug": paths.get("centerlines_raw_debug"),
            "centerline_network": paths.get("centerline_network"),
            "centerline_network_metadata": paths.get("step1_metadata"),
        },
        "coordinate_system": {
            "name": "source_model_coordinates",
            "units": "mm",
        },
        "units": "mm",
        "aorta_start": start_profile or {},
        "aorta_end": end_metadata or {},
        "aorta_trunk": {
            "length": aorta_length,
            "node_path": aorta_node_path or [],
            "centerline_vtp": paths.get("aorta_centerline"),
        },
        "segment_summary": segment_rows,
        "boundary_summary": boundary,
        "fallback_details": {
            "fallback_assigned_cells": total_fallback,
            "fallback_percentage": fallback_pct,
            "priority_region_fallback_authored": priority_fallback,
            "policy": {
                "requires_review_gt": 0.05,
                "failed_gt": 0.15,
                "priority_regions_force_review": [
                    "aortic_inlet",
                    "aortic_bifurcation",
                    "renal_origins",
                    "iliac_split_boundaries",
                ],
            },
        },
        "qa": {
            "assignment_algorithm": SURFACE_ASSIGNMENT_ALGORITHM,
            "tunnel_assignment_algorithm": TUNNEL_ASSIGNMENT_ALGORITHM,
            "mesh_partition_algorithm": MESH_PARTITION_ALGORITHM,
            "total_surface_cells": int(total_cells or 0),
            "assigned_cells": assigned_cells,
            "unassigned_cells": 0 if total_cells is not None else None,
            "fallback_assigned_cells": total_fallback,
            "fallback_percentage": fallback_pct,
            "assignment_mode_counts": assignment_count_values,
            "seed_counts_by_segment": seed_counts_by_segment,
            "vmtk": vmtk_diagnostics or {},
            "vmtk_import_source": (vmtk_diagnostics or {}).get("vmtk_import_source"),
            "vmtk_python": (vmtk_diagnostics or {}).get("vmtk_python"),
            "vmtk_branch_extractor_group_count": assignment_count_values.get("vmtk_branch_extractor_group_count", 0),
            "vmtk_clipped_surface_group_count": assignment_count_values.get("vmtk_clipped_surface_group_count", 0),
            "vmtk_mixed_group_cells": assignment_count_values.get("vmtk_mixed_group_cells", 0),
            "terminal_face_group_min_confidence": (vmtk_diagnostics or {}).get("terminal_face_group_min_confidence"),
            "branch_groups_labeled_as_aorta": assignment_count_values.get("branch_groups_labeled_as_aorta", 0),
            "aorta_groups_labeled_as_branch": assignment_count_values.get("aorta_groups_labeled_as_branch", 0),
            "barrier_cell_count": assignment_count_values.get("barrier_cell_count", 0),
            "blocked_edge_count": assignment_count_values.get("blocked_edge_count", 0),
            "tunnel_seed_cells": assignment_count_values.get("tunnel_seed_cells", 0),
            "tunnel_candidate_cells": assignment_count_values.get("tunnel_candidate_cells", 0),
            "tunnel_component_cells": assignment_count_values.get("tunnel_component_cells", 0),
            "tunnel_component_count": assignment_count_values.get("tunnel_component_count", 0),
            "tunnel_component_claimed_cells": assignment_count_values.get("tunnel_component_claimed_cells", 0),
            "orphan_branch_components": assignment_count_values.get("orphan_branch_components", 0),
            "rejected_orphan_branch_components": assignment_count_values.get("rejected_orphan_branch_components", 0),
            "rejected_orphan_branch_component_cells": assignment_count_values.get("rejected_orphan_branch_component_cells", 0),
            "branch_tunnel_owned_cells": assignment_count_values.get("branch_tunnel_owned_cells", 0),
            "branch_tunnel_owner_segments": assignment_count_values.get("branch_tunnel_owner_segments", 0),
            "aorta_excluded_branch_seed_cells": assignment_count_values.get("aorta_excluded_branch_seed_cells", 0),
            "branch_owned_aorta_labeled_cells": assignment_count_values.get("branch_owned_aorta_labeled_cells", 0),
            "branch_owned_reassigned_from_aorta": assignment_count_values.get("branch_owned_reassigned_from_aorta", 0),
            "branch_labeled_parent_aorta_wall_cells": assignment_count_values.get("branch_labeled_parent_aorta_wall_cells", 0),
            "non_empty_segment_count": int(non_empty_segments),
            "segment_count": int(len(segment_rows)),
            "aorta_start_confidence": (start_profile or {}).get("confidence"),
            "aorta_end_confidence": (end_metadata or {}).get("confidence"),
            "upstream_step1_warning_count": len((step1_metadata or {}).get("warnings", [])),
            "step_status": qa_status,
        },
    }


def run_step2(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    paths_obj = build_pipeline_paths(project_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths_obj.step2_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    step1_metadata_path = Path(args.step1_metadata).resolve() if args.step1_metadata else paths_obj.step1_dir / "centerline_network_metadata.json"
    if not step1_metadata_path.exists():
        raise Step2Failure(f"Missing STEP1 metadata: {step1_metadata_path}")
    step1_metadata = read_json(step1_metadata_path)

    if args.input_vtp:
        input_vtp_path = Path(args.input_vtp).resolve()
    else:
        metadata_input = Path(step1_metadata.get("input_path", "")).expanduser()
        input_vtp_path = metadata_input.resolve() if str(metadata_input) and metadata_input.exists() else paths_obj.default_input_vtp
    face_map_path = Path(args.face_map).resolve() if args.face_map else paths_obj.default_face_map
    step1_outputs = step1_metadata.get("output_paths", {})
    surface_cleaned_path = Path(args.surface_cleaned).resolve() if args.surface_cleaned else Path(step1_outputs.get("surface_cleaned", paths_obj.step1_dir / "surface_cleaned.vtp")).resolve()
    centerlines_raw_path = (
        Path(args.centerlines_raw_debug).resolve()
        if args.centerlines_raw_debug
        else Path(step1_outputs.get("centerlines_raw_debug", paths_obj.step1_dir / "centerlines_raw_debug.vtp")).resolve()
    )
    centerline_network_path = Path(args.centerline_network).resolve() if args.centerline_network else Path(step1_outputs.get("centerline_network_output", paths_obj.step1_dir / "centerline_network.vtp")).resolve()

    required = {
        "input_vtp": input_vtp_path,
        "face_map": face_map_path,
        "surface_cleaned": surface_cleaned_path,
        "centerlines_raw_debug": centerlines_raw_path,
        "centerline_network": centerline_network_path,
        "step1_metadata": step1_metadata_path,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise Step2Failure("Missing required input(s): " + "; ".join(missing))

    output_segments = output_dir / "segmentscolored.vtp"
    output_aorta = output_dir / "aorta_centerline.vtp"
    output_contract = output_dir / "step2_geometry_contract.json"
    output_boundary_debug = output_dir / "boundary_debug.vtp"
    output_debug_vtp = output_dir / "step2_debug.vtp"
    output_debug_json = output_dir / "step2_debug.json"
    if not args.write_debug:
        for stale_debug_path in (output_boundary_debug, output_debug_vtp, output_debug_json):
            try:
                stale_debug_path.unlink()
            except FileNotFoundError:
                pass
    path_strings = {
        "input_vtp": _abs(input_vtp_path),
        "face_map": _abs(face_map_path),
        "surface_cleaned": _abs(surface_cleaned_path),
        "centerlines_raw_debug": _abs(centerlines_raw_path),
        "centerline_network": _abs(centerline_network_path),
        "step1_metadata": _abs(step1_metadata_path),
        "segments_vtp": _abs(output_segments),
        "aorta_centerline": _abs(output_aorta),
        "contract_json": _abs(output_contract),
    }

    warnings: list[str] = []
    face_map = _face_map_by_id(read_json(face_map_path))
    _validate_expected_anatomy(face_map)
    surface = read_vtp(surface_cleaned_path)
    raw_centerlines = read_vtp(centerlines_raw_path)
    network = read_vtp(centerline_network_path)
    edges, node_coords = _read_network_edges(network)
    graph = _build_graph(edges)
    face_node_map = _map_face_terminations_to_nodes(step1_metadata, face_map, node_coords)
    inlet_node, inlet_row = _resolve_inlet(face_map, face_node_map)
    inlet_face_id = int(inlet_row["face_id"])
    outlet_routes = _build_outlet_routes(face_map, face_node_map, graph, inlet_node)

    bif_node, bif_detail = _resolve_aortic_bifurcation_node(face_map, face_node_map, graph, inlet_node)
    segments, aorta_node_path = _build_segments(inlet_node, bif_node, graph, edges, face_map, face_node_map, outlet_routes)
    vmtk_result = _run_vmtk_branch_group_segmentation(
        surface=surface,
        raw_centerlines=raw_centerlines,
        step1_metadata=step1_metadata,
        face_map=face_map,
        segments=segments,
    )
    partitioned_surface = vmtk_result.surface
    aorta_segment = segments[0]
    aorta_length = float(aorta_segment.length)
    if aorta_length <= 0.0:
        raise Step2Failure("A single trustworthy aorta trunk cannot be authored: non-positive length.")

    start_term = dict(inlet_row.get("termination", {}))
    start_profile = _face_region_profile(surface, inlet_face_id, start_term)
    end_profile = _extract_aorta_end_profile(surface, aorta_segment.points, start_profile.get("equivalent_diameter"))
    bif_point = node_coords[int(bif_node)]
    end_metadata = {
        "centerline_landmark": {
            "name": "aorta_end_pre_bifurcation",
            "node_id": int(bif_node),
            "point": bif_point.tolist(),
            "source_type": "step1_centerline_topology",
            "confidence": 0.9,
            "resolution_detail": bif_detail,
        },
        "bifurcation_point": bif_point.tolist(),
        "boundary_centroid": end_profile.get("boundary_centroid"),
        "boundary_normal": end_profile.get("boundary_normal"),
        "area": end_profile.get("area"),
        "equivalent_diameter": end_profile.get("equivalent_diameter"),
        "major_diameter": end_profile.get("major_diameter"),
        "minor_diameter": end_profile.get("minor_diameter"),
        "surface_boundary_profile": end_profile,
        "local_tangent": tangent_at_arclength(aorta_segment.points, aorta_length, window=1.0).tolist(),
        "confidence": float(min(0.9, float(end_profile.get("confidence", 0.0)))),
        "extraction_method": "centerline_common_iliac_routes_plus_surface_plane_cut",
        "surface_derived": True,
        "centerline_derived": True,
        "source_components": ["centerline_landmark", "surface_derived_boundary_profile"],
    }

    labels, assignment_counts, _, diagnostics = _assign_surface_cells_from_vmtk(
        partitioned_surface, segments, vmtk_result, raw_centerlines
    )
    if int(np.count_nonzero(labels <= 0)) > 0:
        raise Step2Failure("Core segment assignment is too unreliable: some cells remained unassigned.")
    if len([seg for seg in segments if seg.cell_count > 0]) < 2:
        raise Step2Failure("Core segment assignment is too unreliable: fewer than two non-empty segments.")

    segments_surface = _build_segments_surface(partitioned_surface, labels)
    write_vtp(segments_surface, output_segments)
    write_vtp(build_polyline_polydata(aorta_segment.points), output_aorta)

    if args.write_debug:
        write_vtp(_build_boundary_loop_debug_polydata(segments), output_boundary_debug)
        write_vtp(_build_debug_segments_surface(partitioned_surface, labels, diagnostics), output_debug_vtp)
        boundary_debug = {
            str(seg.segment_id): {
                "name_hint": seg.name_hint,
                "segment_type": seg.segment_type,
                "selected_boundary": seg.proximal_boundary.to_contract() if seg.proximal_boundary is not None else None,
                "cut_loop_points": np.asarray(seg.proximal_boundary.loop_points, dtype=float).tolist()
                if seg.proximal_boundary is not None and seg.proximal_boundary.loop_points is not None
                else [],
                "warning": seg.proximal_boundary_warning,
                "attempts": seg.proximal_boundary_attempts,
            }
            for seg in segments
            if seg.segment_type != "aorta_trunk"
        }
        write_json(
            {
                "aorta_node_path": aorta_node_path,
                "outlet_routes": outlet_routes,
                "face_node_map": face_node_map,
                "assignment_counts": assignment_counts,
                "vmtk_diagnostics": vmtk_result.diagnostics,
                "diagnostic_arrays": {key: np.asarray(value).tolist() for key, value in diagnostics.items()},
                "bifurcation_detail": bif_detail,
                "branch_proximal_boundaries": boundary_debug,
            },
            output_debug_json,
        )
        path_strings["boundary_debug_vtp"] = _abs(output_boundary_debug)
        path_strings["step2_debug_vtp"] = _abs(output_debug_vtp)
        path_strings["step2_debug_json"] = _abs(output_debug_json)

    contract = _make_contract(
        status="success",
        warnings=warnings,
        paths=path_strings,
        step1_metadata=step1_metadata,
        start_profile=start_profile,
        end_metadata=end_metadata,
        aorta_length=aorta_length,
        aorta_node_path=aorta_node_path,
        segments=segments,
        node_coords=node_coords,
        face_node_map=face_node_map,
        assignment_counts=assignment_counts,
        vmtk_diagnostics=vmtk_result.diagnostics,
        total_cells=int(partitioned_surface.GetNumberOfCells()),
    )
    write_json(contract, output_contract)
    return contract


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STEP2 geometry contract and aorta trunk authoring.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]), help="Project root.")
    parser.add_argument("--input-vtp", default="", help="Input lumen surface. Defaults to STEP1 metadata input_path.")
    parser.add_argument("--face-map", default="", help="face_id_to_name.json path.")
    parser.add_argument("--step1-metadata", default="", help="STEP1 centerline_network_metadata.json path.")
    parser.add_argument("--surface-cleaned", default="", help="STEP1 surface_cleaned.vtp path.")
    parser.add_argument("--centerlines-raw-debug", default="", help="STEP1 centerlines_raw_debug.vtp path.")
    parser.add_argument("--centerline-network", default="", help="STEP1 centerline_network.vtp path.")
    parser.add_argument("--output-dir", default="", help="STEP2 output directory.")
    parser.add_argument("--write-debug", action="store_true", help="Write optional STEP2 debug artifacts.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    _maybe_reexec_with_vmtk_python(project_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else build_pipeline_paths(project_root).step2_dir
    output_contract = output_dir / "step2_geometry_contract.json"
    try:
        contract = run_step2(args)
    except Step2Failure as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_status = "requires_review" if isinstance(exc, Step2RequiresReview) else "failed"
        failure_contract = _make_contract(
            status=failure_status,
            warnings=[str(exc)],
            paths={
                "contract_json": _abs(output_contract),
                "input_vtp": _abs(args.input_vtp) if args.input_vtp else "",
                "face_map": _abs(args.face_map) if args.face_map else "",
            },
        )
        write_json(failure_contract, output_contract)
        print(f"STEP2 failed: {exc}")
        return 1
    print(
        "STEP2 completed: "
        f"{contract.get('step_status')} | "
        f"segments={contract.get('qa', {}).get('segment_count')} | "
        f"fallback={contract.get('fallback_details', {}).get('fallback_assigned_cells')}"
    )
    if contract.get("step_status") == "failed":
        return 1
    return 0
