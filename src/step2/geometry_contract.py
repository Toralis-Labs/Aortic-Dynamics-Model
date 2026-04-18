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

    @property
    def length(self) -> float:
        return float(cumulative_arclength(self.points)[-1]) if self.points.shape[0] else 0.0


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


def _assign_surface_cells(
    surface: vtk.vtkPolyData,
    segments: list[GeometrySegment],
    inlet_face_id: int,
) -> tuple[np.ndarray, Dict[str, int], Dict[int, int]]:
    n_cells = int(surface.GetNumberOfCells())
    labels = np.zeros((n_cells,), dtype=np.int32)
    assignment_counts: Dict[str, int] = {
        "face_terminal_override": 0,
        "inlet_face_override": 0,
        "nearest_centerline": 0,
        "fallback": 0,
    }

    segment_by_terminal_face = {
        int(seg.terminal_face_id): int(seg.segment_id)
        for seg in segments
        if seg.terminal_face_id is not None
    }
    segment_points = [(int(seg.segment_id), np.asarray(seg.points, dtype=float)) for seg in segments]
    locator, locator_pd = build_segment_point_locator(segment_points)
    locator_segment_array = locator_pd.GetPointData().GetArray("SegmentId")
    centers = cell_centers(surface)
    model_face = get_cell_array(surface, "ModelFaceID")

    for cell_id in range(n_cells):
        face_id = int(model_face[cell_id]) if model_face is not None else -1
        if face_id == int(inlet_face_id):
            labels[cell_id] = 1
            assignment_counts["inlet_face_override"] += 1
            continue
        mapped_segment = segment_by_terminal_face.get(face_id)
        if mapped_segment is not None:
            labels[cell_id] = int(mapped_segment)
            assignment_counts["face_terminal_override"] += 1
            continue
        nearest_point_id = int(locator.FindClosestPoint(centers[cell_id]))
        if nearest_point_id >= 0:
            labels[cell_id] = int(locator_segment_array.GetTuple1(nearest_point_id))
            assignment_counts["nearest_centerline"] += 1
        else:
            labels[cell_id] = 1
            assignment_counts["fallback"] += 1

    counts = {int(seg.segment_id): int(np.count_nonzero(labels == int(seg.segment_id))) for seg in segments}
    for seg in segments:
        seg.cell_count = counts.get(int(seg.segment_id), 0)
        seg.fallback_cell_count = 0
    return labels, assignment_counts, counts


def _build_segments_surface(surface: vtk.vtkPolyData, labels: np.ndarray) -> vtk.vtkPolyData:
    out = clone_geometry_only(surface)
    add_int_cell_array(out, "SegmentId", labels.astype(int).tolist())
    colors = [segment_color(int(seg_id)) for seg_id in labels.tolist()]
    add_uchar3_cell_array(out, "SegmentColorRGB", colors)
    return out


def _segment_contract_rows(segments: list[GeometrySegment], node_coords: dict[int, np.ndarray]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for seg in segments:
        rows.append(
            {
                "segment_id": int(seg.segment_id),
                "name_hint": str(seg.name_hint),
                "segment_type": str(seg.segment_type),
                "parent_segment_id": seg.parent_segment_id,
                "child_segment_ids": [int(v) for v in sorted(seg.child_segment_ids)],
                "proximal_node_id": int(seg.proximal_node),
                "distal_node_id": int(seg.distal_node),
                "proximal_point": node_coords.get(int(seg.proximal_node), seg.points[0]).tolist(),
                "distal_point": node_coords.get(int(seg.distal_node), seg.points[-1]).tolist(),
                "edge_ids": [int(eid) for eid in seg.edge_ids],
                "length": float(seg.length),
                "terminal_face_id": seg.terminal_face_id,
                "terminal_face_name": seg.terminal_face_name,
                "descendant_terminal_names": [str(v) for v in seg.descendant_terminal_names],
                "cell_count": int(seg.cell_count),
                "fallback_cell_count": int(seg.fallback_cell_count),
            }
        )
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
        if fallback_pct > 0.15:
            qa_status = "failed"
            warnings.append("W_STEP2_FALLBACK_GT_15_PERCENT: fallback cell assignment exceeded 15%.")
        elif fallback_pct > 0.05 or priority_fallback:
            qa_status = "requires_review"
            warnings.append("W_STEP2_FALLBACK_REQUIRES_REVIEW: fallback assignment exceeded review policy.")

    segment_rows = _segment_contract_rows(segments or [], node_coords or {}) if segments else []
    boundary = _boundary_summary(start_profile or {}, end_metadata or {}, face_node_map or {}) if start_profile and end_metadata else {}
    non_empty_segments = sum(1 for row in segment_rows if int(row.get("cell_count", 0)) > 0)
    return {
        "schema_name": "step2_geometry_contract",
        "schema_version": 1,
        "status": qa_status,
        "inputs": {
            "source_lumen_surface": paths.get("input_vtp"),
            "face_id_to_name": paths.get("face_map"),
        },
        "upstream_step1_references": {
            "surface_cleaned": paths.get("surface_cleaned"),
            "centerline_network": paths.get("centerline_network"),
            "centerline_network_metadata": paths.get("step1_metadata"),
        },
        "coordinate_system": {
            "name": "source_model_coordinates",
            "units": "mm",
        },
        "aorta_start": start_profile or {},
        "aorta_end": end_metadata or {},
        "aorta_trunk": {
            "length": aorta_length,
            "node_path": aorta_node_path or [],
            "centerline_vtp": paths.get("aorta_centerline"),
        },
        "segments_vtp": paths.get("segments_vtp"),
        "segment_summary": segment_rows,
        "boundary_summary": boundary,
        "fallback_summary": {
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
        "qa_summary": {
            "total_surface_cells": int(total_cells or 0),
            "assigned_cells": assigned_cells,
            "unassigned_cells": 0 if total_cells is not None else None,
            "fallback_assigned_cells": total_fallback,
            "fallback_percentage": fallback_pct,
            "assignment_mode_counts": assignment_counts or {},
            "non_empty_segment_count": int(non_empty_segments),
            "segment_count": int(len(segment_rows)),
            "aorta_start_confidence": (start_profile or {}).get("confidence"),
            "aorta_end_confidence": (end_metadata or {}).get("confidence"),
            "upstream_step1_warning_count": len((step1_metadata or {}).get("warnings", [])),
            "step_status": qa_status,
        },
        "warnings": sorted(set(str(w) for w in warnings)),
        "final_status": qa_status,
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

    input_vtp_path = Path(args.input_vtp).resolve() if args.input_vtp else Path(step1_metadata.get("input_path", "")).resolve()
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

    labels, assignment_counts, _ = _assign_surface_cells(surface, segments, inlet_face_id)
    if int(np.count_nonzero(labels <= 0)) > 0:
        raise Step2Failure("Core segment assignment is too unreliable: some cells remained unassigned.")
    if len([seg for seg in segments if seg.cell_count > 0]) < 2:
        raise Step2Failure("Core segment assignment is too unreliable: fewer than two non-empty segments.")

    segments_surface = _build_segments_surface(surface, labels)
    write_vtp(segments_surface, output_segments)
    write_vtp(build_polyline_polydata(aorta_segment.points), output_aorta)

    if args.write_debug:
        debug_pd = build_polyline_polydata(aorta_segment.points)
        write_vtp(debug_pd, output_boundary_debug)
        write_json(
            {
                "aorta_node_path": aorta_node_path,
                "face_node_map": face_node_map,
                "assignment_counts": assignment_counts,
                "bifurcation_detail": bif_detail,
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
        f"{contract.get('final_status')} | "
        f"segments={contract.get('qa_summary', {}).get('segment_count')} | "
        f"fallback={contract.get('fallback_summary', {}).get('fallback_assigned_cells')}"
    )
    if contract.get("final_status") == "failed":
        return 1
    return 0
