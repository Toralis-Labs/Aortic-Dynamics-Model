from __future__ import annotations

import argparse
import heapq
import math
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
    projected_major_minor_diameters,
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


PROXIMAL_BOUNDARY_SELECTION_ALGORITHM = "first_stable_surface_ostium_v2"
SURFACE_ASSIGNMENT_ALGORITHM = "surface_seeded_propagation_v3_ostium_footprint"


class Step2Failure(RuntimeError):
    pass


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

    def to_contract(self) -> Dict[str, Any]:
        return {
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
            "selection_algorithm": PROXIMAL_BOUNDARY_SELECTION_ALGORITHM,
            "attempts": self.attempts,
        }


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
    cell_count: int = 0
    fallback_cell_count: int = 0
    proximal_boundary: Optional[SegmentBoundaryProfile] = None
    proximal_boundary_attempts: list[Dict[str, Any]] = field(default_factory=list)
    proximal_boundary_warning: Optional[str] = None

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
            descendant_terminal_names=_collect_descendant_terminal_names(bif_node, child_map, terminal_node_to_face, face_map),
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


def _cell_area_and_centroid(surface: vtk.vtkPolyData, cell_id: int) -> tuple[float, np.ndarray]:
    pts = cell_points(surface, int(cell_id))
    area = triangle_area(pts)
    centroid = np.mean(pts, axis=0) if pts.shape[0] else np.zeros(3, dtype=float)
    return float(area), centroid


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
    major, minor = projected_major_minor_diameters(pts, normal_hint=normal)
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
        major, minor = projected_major_minor_diameters(pts, normal_hint=n)
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

    offsets = [0.1 * idx for idx in range(1, 51)]
    valid_offsets = [offset for offset in offsets if offset < max(0.0, length - 0.1)]
    if not valid_offsets:
        valid_offsets = [min(0.25, 0.5 * length)]

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
            source_type="surface_ostium_plane_cut",
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
        boundary.confidence = float(min(float(boundary.confidence), 0.68))
    elif selected_candidate is not None:
        boundary = selected_candidate["boundary"]
    else:
        segment.proximal_boundary_attempts = attempts
        return None

    attempts[int(selected_candidate["attempt_index"])]["selected_reason"] = selected_reason
    boundary.attempts = attempts
    return boundary


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
    max_island_cells: int = 20,
) -> Dict[str, int]:
    model_face = get_cell_array(surface, "ModelFaceID")
    protected = np.zeros((surface.GetNumberOfCells(),), dtype=bool)
    if model_face is not None and protected_face_ids:
        protected = np.isin(model_face.astype(int), list(protected_face_ids))

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
    signed = origin_vectors @ normal
    radial_vectors = origin_vectors - signed.reshape(-1, 1) * normal.reshape(1, 3)
    radial_distance = np.linalg.norm(radial_vectors, axis=1)
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
    parent_margin = 0.38 if strict else 0.28
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
    barrier_segment_id = np.zeros((n_cells,), dtype=np.int32)
    rejected_segment = np.zeros((n_cells,), dtype=np.int32)
    propagation_cost = np.full((n_cells,), np.inf, dtype=float)
    assignment_counts: Dict[str, int] = {
        SURFACE_ASSIGNMENT_ALGORITHM: 1,
        "fixed_seed_cells": 0,
        "core_seed_cells": 0,
        "weak_core_seed_cells": 0,
        "propagated_cells": 0,
        "fallback": 0,
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
    distance_by_segment, arclength_by_segment, radius_by_segment, nearest_segment = _precompute_segment_projections(centers, segments)
    blocked_edges, barrier_segment_id, barrier_counts = _build_ostium_barriers(centers, adjacency, segments)
    assignment_counts.update(barrier_counts)

    fixed = np.zeros((n_cells,), dtype=bool)
    seed_counts = _build_segment_seeds(
        labels,
        assignment_mode,
        seed_segment_id,
        fixed,
        centers,
        model_face,
        segments,
        inlet_face_id,
        segment_by_terminal_face,
        distance_by_segment,
        arclength_by_segment,
        radius_by_segment,
        barrier_segment_id,
    )
    assignment_counts.update(seed_counts)
    assignment_counts["fixed_seed_cells"] = int(assignment_counts.get("inlet_face_seed_cells", 0)) + int(
        assignment_counts.get("terminal_face_seed_cells", 0)
    )

    propagation_cost = _propagate_surface_labels(
        centers,
        adjacency,
        labels,
        assignment_mode,
        fixed,
        segments,
        distance_by_segment,
        arclength_by_segment,
        radius_by_segment,
        blocked_edges,
        rejected_segment,
    )
    assignment_counts["propagated_cells"] = int(np.count_nonzero(assignment_mode == 4))
    assignment_counts["branch_origin_gate_rejected_cells"] = int(np.count_nonzero(rejected_segment > 0))
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
        )
    assignment_counts.update(cleanup_counts)

    counts = {int(seg.segment_id): int(np.count_nonzero(labels == int(seg.segment_id))) for seg in segments}
    for seg in segments:
        seg.cell_count = counts.get(int(seg.segment_id), 0)
        seg.fallback_cell_count = 0
    diagnostics = {
        "AssignmentMode": assignment_mode,
        "SeedSegmentId": seed_segment_id,
        "PropagationCost": np.where(np.isfinite(propagation_cost), propagation_cost, -1.0),
        "BarrierSegmentId": barrier_segment_id,
        "NearestSegmentId": nearest_segment,
        "RejectedSegmentId": rejected_segment,
    }
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
            "cell_count": int(seg.cell_count),
            "fallback_cell_count": int(seg.fallback_cell_count),
        }
        if seg.proximal_boundary is not None:
            row["proximal_boundary"] = seg.proximal_boundary.to_contract()
            row["proximal_boundary_source"] = seg.proximal_boundary.source_type
            row["proximal_boundary_confidence"] = float(seg.proximal_boundary.confidence)
            row["proximal_boundary_arclength"] = float(seg.proximal_boundary.arclength)
            row["proximal_boundary_selection_algorithm"] = PROXIMAL_BOUNDARY_SELECTION_ALGORITHM
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
            "centerline_network": paths.get("centerline_network"),
            "step1_metadata": paths.get("step1_metadata"),
        },
        "output_paths": {
            "segments_vtp": paths.get("segments_vtp"),
            "aorta_centerline": paths.get("aorta_centerline"),
            "contract_json": paths.get("contract_json"),
            **({"boundary_debug_vtp": paths["boundary_debug_vtp"]} if "boundary_debug_vtp" in paths else {}),
            **({"step2_debug_json": paths["step2_debug_json"]} if "step2_debug_json" in paths else {}),
        },
        "upstream_references": {
            "surface_cleaned": paths.get("surface_cleaned"),
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
            "total_surface_cells": int(total_cells or 0),
            "assigned_cells": assigned_cells,
            "unassigned_cells": 0 if total_cells is not None else None,
            "fallback_assigned_cells": total_fallback,
            "fallback_percentage": fallback_pct,
            "assignment_mode_counts": assignment_count_values,
            "seed_counts_by_segment": seed_counts_by_segment,
            "barrier_cell_count": assignment_count_values.get("barrier_cell_count", 0),
            "blocked_edge_count": assignment_count_values.get("blocked_edge_count", 0),
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
    centerline_network_path = Path(args.centerline_network).resolve() if args.centerline_network else Path(step1_outputs.get("centerline_network_output", paths_obj.step1_dir / "centerline_network.vtp")).resolve()

    required = {
        "input_vtp": input_vtp_path,
        "face_map": face_map_path,
        "surface_cleaned": surface_cleaned_path,
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
    output_debug_json = output_dir / "step2_debug.json"
    path_strings = {
        "input_vtp": _abs(input_vtp_path),
        "face_map": _abs(face_map_path),
        "surface_cleaned": _abs(surface_cleaned_path),
        "centerline_network": _abs(centerline_network_path),
        "step1_metadata": _abs(step1_metadata_path),
        "segments_vtp": _abs(output_segments),
        "aorta_centerline": _abs(output_aorta),
        "contract_json": _abs(output_contract),
    }

    warnings: list[str] = []
    face_map = _face_map_by_id(read_json(face_map_path))
    surface = read_vtp(surface_cleaned_path)
    network = read_vtp(centerline_network_path)
    edges, node_coords = _read_network_edges(network)
    graph = _build_graph(edges)
    face_node_map = _map_face_terminations_to_nodes(step1_metadata, face_map, node_coords)
    inlet_node, inlet_row = _resolve_inlet(face_map, face_node_map)
    inlet_face_id = int(inlet_row["face_id"])

    bif_node, bif_detail = _resolve_aortic_bifurcation_node(face_map, face_node_map, graph, inlet_node)
    segments, aorta_node_path = _build_segments(inlet_node, bif_node, graph, edges, face_map, face_node_map)
    warnings.extend(_refine_branch_boundaries(surface, segments, face_node_map))
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

    labels, assignment_counts, _, diagnostics = _assign_surface_cells(surface, segments, inlet_face_id)
    if int(np.count_nonzero(labels <= 0)) > 0:
        raise Step2Failure("Core segment assignment is too unreliable: some cells remained unassigned.")
    if len([seg for seg in segments if seg.cell_count > 0]) < 2:
        raise Step2Failure("Core segment assignment is too unreliable: fewer than two non-empty segments.")

    segments_surface = _build_segments_surface(surface, labels)
    write_vtp(segments_surface, output_segments)
    write_vtp(build_polyline_polydata(aorta_segment.points), output_aorta)

    if args.write_debug:
        debug_pd = _build_debug_segments_surface(surface, labels, diagnostics)
        write_vtp(debug_pd, output_boundary_debug)
        boundary_debug = {
            str(seg.segment_id): {
                "name_hint": seg.name_hint,
                "segment_type": seg.segment_type,
                "selected_boundary": seg.proximal_boundary.to_contract() if seg.proximal_boundary is not None else None,
                "warning": seg.proximal_boundary_warning,
                "attempts": seg.proximal_boundary_attempts,
            }
            for seg in segments
            if seg.segment_type != "aorta_trunk"
        }
        write_json(
            {
                "aorta_node_path": aorta_node_path,
                "face_node_map": face_node_map,
                "assignment_counts": assignment_counts,
                "bifurcation_detail": bif_detail,
                "branch_proximal_boundaries": boundary_debug,
            },
            output_debug_json,
        )
        path_strings["boundary_debug_vtp"] = _abs(output_boundary_debug)
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
        total_cells=int(surface.GetNumberOfCells()),
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
    parser.add_argument("--centerline-network", default="", help="STEP1 centerline_network.vtp path.")
    parser.add_argument("--output-dir", default="", help="STEP2 output directory.")
    parser.add_argument("--write-debug", action="store_true", help="Write optional STEP2 debug artifacts.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else build_pipeline_paths(project_root).step2_dir
    output_contract = output_dir / "step2_geometry_contract.json"
    try:
        contract = run_step2(args)
    except Step2Failure as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_contract = _make_contract(
            status="failed",
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
