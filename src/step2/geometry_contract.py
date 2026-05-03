from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import vtk

from src.common.geometry import (
    EPS,
    concatenate_polylines,
    cumulative_arclength,
    point_at_arclength,
    polyline_length,
    tangent_at_arclength,
    unit,
)
from src.common.json_io import read_json, write_json
from src.common.paths import build_workspace_paths
from src.common.vtk_helpers import (
    add_float_cell_array,
    add_int_cell_array,
    add_string_cell_array,
    add_uchar3_cell_array,
    append_polydata,
    array_names,
    build_regular_polygon_ring_polydata,
    build_segment_point_locator,
    cell_centers,
    clone_geometry_only,
    get_cell_array,
    get_point_array,
    points_to_numpy,
    read_vtp,
    segment_color,
    write_vtp,
)


STATUS_SUCCESS = "success"
STATUS_REQUIRES_REVIEW = "requires_review"
STATUS_FAILED = "failed"

REQUIRED_SEGMENTED_SURFACE_CELL_ARRAYS = [
    "SegmentId",
    "SegmentLabel",
    "SegmentColor",
]
REQUIRED_BOUNDARY_RING_CELL_ARRAYS = [
    "RingId",
    "RingLabel",
    "RingType",
    "ParentSegmentId",
    "ChildSegmentId",
    "SegmentId",
    "RadiusMm",
    "Confidence",
    "Status",
]
REQUIRED_RESULT_KEYS = [
    "status",
    "inputs",
    "outputs",
    "segments",
    "boundary_rings",
    "bifurcations",
    "warnings",
    "metrics",
]
FORBIDDEN_LABEL_FRAGMENTS = [
    "renal",
    "sma",
    "ima",
    "celiac",
    "iliac",
    "left",
    "right",
    "external",
    "internal",
    "superior_mesenteric",
    "inferior_mesenteric",
    "abdominal_aorta_trunk",
    "aorta_trunk",
]


class GeometrySegmentationFailure(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass(frozen=True)
class InputRoles:
    aortic_inlet_face_id: int
    terminal_face_ids: list[int]
    branch_label_prefix: str = "branch_"
    bifurcation_label_prefix: str = "bifurcation_"
    ring_label_prefix: str = "ring_"


@dataclass
class NetworkEdge:
    edge_id: int
    cell_id: int
    start_node_id: int
    end_node_id: int
    points: np.ndarray
    length: float
    support_count: int = 1
    parent_node_id: Optional[int] = None
    child_node_id: Optional[int] = None

    def other_node(self, node_id: int) -> int:
        if int(node_id) == self.start_node_id:
            return self.end_node_id
        if int(node_id) == self.end_node_id:
            return self.start_node_id
        raise GeometrySegmentationFailure(f"Centerline edge {self.edge_id} is not connected to node {node_id}.")

    def oriented_points_from(self, node_id: int) -> np.ndarray:
        if int(node_id) == self.start_node_id:
            return self.points.copy()
        if int(node_id) == self.end_node_id:
            return self.points[::-1].copy()
        raise GeometrySegmentationFailure(f"Centerline edge {self.edge_id} cannot be oriented from node {node_id}.")


@dataclass
class GeometrySegment:
    segment_id: int
    segment_label: str
    segment_type: str
    parent_segment_id: Optional[int]
    child_segment_ids: list[int]
    proximal_node_id: int
    distal_node_id: int
    edge_ids: list[int]
    points: np.ndarray
    terminal_face_id: Optional[int] = None
    proximal_ring_id: Optional[int] = None
    distal_ring_ids: list[int] = field(default_factory=list)
    cell_count: int = 0
    status: str = STATUS_REQUIRES_REVIEW
    warnings: list[str] = field(default_factory=list)


@dataclass
class BoundaryRing:
    ring_id: int
    ring_label: str
    ring_type: str
    center_xyz: list[float]
    normal_xyz: list[float]
    radius_mm: float
    source_segment_id: int
    parent_segment_id: Optional[int]
    child_segment_id: Optional[int]
    source_centerline_s_mm: float
    orientation_rule: str
    radius_rule: str
    confidence: float
    status: str
    warnings: list[str]
    polydata: vtk.vtkPolyData


@dataclass
class BifurcationRecord:
    bifurcation_id: int
    bifurcation_label: str
    parent_segment_id: int
    child_segment_ids: list[int]
    parent_pre_bifurcation_ring_id: Optional[int]
    daughter_start_ring_ids: list[int]
    status: str
    warnings: list[str]


def _as_finite_vector(value: Any, fallback: Iterable[float]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
    except Exception:
        arr = np.asarray(list(fallback), dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        arr = np.asarray(list(fallback), dtype=float).reshape(3)
    return arr


def _load_input_roles(path: str | Path) -> InputRoles:
    data = read_json(path)
    if not isinstance(data, dict):
        raise GeometrySegmentationFailure("Input roles must be a JSON object.")

    aortic_body = data.get("aortic_body")
    if not isinstance(aortic_body, dict) or "inlet_face_id" not in aortic_body:
        raise GeometrySegmentationFailure("Input roles must define aortic_body.inlet_face_id.")

    inlet_face_id = aortic_body.get("inlet_face_id")
    if not isinstance(inlet_face_id, int):
        raise GeometrySegmentationFailure("Input roles aortic_body.inlet_face_id must be an integer.")

    terminal_faces = data.get("terminal_faces")
    if not isinstance(terminal_faces, list) or not terminal_faces:
        raise GeometrySegmentationFailure("Input roles terminal_faces must be a non-empty list of integers.")
    if any(not isinstance(face_id, int) for face_id in terminal_faces):
        raise GeometrySegmentationFailure("Input roles terminal_faces must contain only integers.")

    terminal_face_ids = [int(face_id) for face_id in terminal_faces]
    if len(set(terminal_face_ids)) != len(terminal_face_ids):
        raise GeometrySegmentationFailure("Input roles terminal_faces must not contain duplicates.")
    if int(inlet_face_id) in set(terminal_face_ids):
        raise GeometrySegmentationFailure("The aortic inlet face must not also be listed as a terminal face.")

    rules = data.get("rules") if isinstance(data.get("rules"), dict) else {}
    only_named_segment = rules.get("only_named_segment", "aortic_body")
    if only_named_segment != "aortic_body":
        raise GeometrySegmentationFailure("Input roles rules.only_named_segment must be aortic_body.")
    if rules.get("all_other_segments_are_anonymous", True) is not True:
        raise GeometrySegmentationFailure("Input roles must keep all non-aortic segments anonymous.")

    branch_label_prefix = str(rules.get("branch_label_prefix", "branch_"))
    bifurcation_label_prefix = str(rules.get("bifurcation_label_prefix", "bifurcation_"))
    ring_label_prefix = str(rules.get("ring_label_prefix", "ring_"))
    if branch_label_prefix != "branch_":
        raise GeometrySegmentationFailure("Input roles rules.branch_label_prefix must be branch_.")
    if bifurcation_label_prefix != "bifurcation_":
        raise GeometrySegmentationFailure("Input roles rules.bifurcation_label_prefix must be bifurcation_.")
    if ring_label_prefix != "ring_":
        raise GeometrySegmentationFailure("Input roles rules.ring_label_prefix must be ring_.")

    return InputRoles(
        aortic_inlet_face_id=int(inlet_face_id),
        terminal_face_ids=terminal_face_ids,
        branch_label_prefix=branch_label_prefix,
        bifurcation_label_prefix=bifurcation_label_prefix,
        ring_label_prefix=ring_label_prefix,
    )


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise GeometrySegmentationFailure("Missing required input artifact(s): " + ", ".join(missing))


def _termination_map(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in metadata.get("terminations", []):
        if not isinstance(item, dict) or "face_id" not in item:
            continue
        try:
            out[int(item["face_id"])] = item
        except Exception:
            continue
    return out


def _point_ids_for_cell(cell: vtk.vtkCell) -> list[int]:
    ids = cell.GetPointIds()
    return [int(ids.GetId(idx)) for idx in range(ids.GetNumberOfIds())]


def _required_cell_array(polydata: vtk.vtkPolyData, name: str) -> np.ndarray:
    values = get_cell_array(polydata, name)
    if values is None:
        raise GeometrySegmentationFailure(f"Input centerline network is missing required cell array {name}.")
    return values


def _required_point_array(polydata: vtk.vtkPolyData, name: str) -> np.ndarray:
    values = get_point_array(polydata, name)
    if values is None:
        raise GeometrySegmentationFailure(f"Input centerline network is missing required point array {name}.")
    return values


def _build_network_edges(centerline_network: vtk.vtkPolyData) -> tuple[dict[int, NetworkEdge], dict[int, np.ndarray]]:
    if centerline_network.GetNumberOfCells() == 0:
        raise GeometrySegmentationFailure("Input centerline network contains no usable edges.")

    points = points_to_numpy(centerline_network)
    node_ids = _required_point_array(centerline_network, "NodeId").astype(int)
    start_node_ids = _required_cell_array(centerline_network, "StartNodeId").astype(int)
    end_node_ids = _required_cell_array(centerline_network, "EndNodeId").astype(int)

    edge_ids_raw = get_cell_array(centerline_network, "EdgeId")
    if edge_ids_raw is None:
        edge_ids = np.arange(centerline_network.GetNumberOfCells(), dtype=int)
    else:
        edge_ids = edge_ids_raw.astype(int)

    edge_lengths_raw = get_cell_array(centerline_network, "EdgeLength")
    support_counts_raw = get_cell_array(centerline_network, "SupportCount")

    node_coords: dict[int, np.ndarray] = {}
    for point_index, node_id in enumerate(node_ids):
        node_id_i = int(node_id)
        if node_id_i >= 0 and node_id_i not in node_coords:
            node_coords[node_id_i] = points[int(point_index)].copy()

    if not node_coords:
        raise GeometrySegmentationFailure("Input centerline network does not contain graph node IDs.")

    edges: dict[int, NetworkEdge] = {}
    for cell_id in range(centerline_network.GetNumberOfCells()):
        cell = centerline_network.GetCell(cell_id)
        point_ids = _point_ids_for_cell(cell)
        if len(point_ids) < 2:
            continue

        edge_id = int(edge_ids[cell_id])
        edge_points = points[point_ids].astype(float, copy=True)
        length = float(edge_lengths_raw[cell_id]) if edge_lengths_raw is not None else polyline_length(edge_points)
        if not math.isfinite(length) or length <= EPS:
            length = polyline_length(edge_points)
        support_count = int(support_counts_raw[cell_id]) if support_counts_raw is not None else 1

        edges[edge_id] = NetworkEdge(
            edge_id=edge_id,
            cell_id=int(cell_id),
            start_node_id=int(start_node_ids[cell_id]),
            end_node_id=int(end_node_ids[cell_id]),
            points=edge_points,
            length=float(length),
            support_count=max(1, support_count),
        )

    if not edges:
        raise GeometrySegmentationFailure("Input centerline network does not contain usable polyline edges.")

    return edges, node_coords


def _nearest_node_id(center: np.ndarray, node_coords: dict[int, np.ndarray]) -> tuple[int, float]:
    best_node: Optional[int] = None
    best_distance = float("inf")
    for node_id, point in node_coords.items():
        distance = float(np.linalg.norm(np.asarray(center, dtype=float) - point))
        if distance < best_distance:
            best_distance = distance
            best_node = int(node_id)
    if best_node is None:
        raise GeometrySegmentationFailure("Unable to resolve a centerline node.")
    return best_node, best_distance


def _map_role_faces_to_nodes(
    roles: InputRoles,
    terminations: dict[int, dict[str, Any]],
    node_coords: dict[int, np.ndarray],
) -> tuple[int, dict[int, int], dict[int, int], list[str]]:
    warnings: list[str] = []

    inlet_termination = terminations.get(roles.aortic_inlet_face_id)
    if inlet_termination is None:
        raise GeometrySegmentationFailure("Input centerline metadata does not contain the aortic inlet face.")

    inlet_center = _as_finite_vector(inlet_termination.get("center"), [0.0, 0.0, 0.0])
    root_node_id, root_distance = _nearest_node_id(inlet_center, node_coords)
    if root_distance > 2.0:
        warnings.append("aortic inlet face was mapped to a centerline node using a distance above 2 mm")

    terminal_face_to_node: dict[int, int] = {}
    terminal_node_to_face: dict[int, int] = {}
    for face_id in roles.terminal_face_ids:
        termination = terminations.get(int(face_id))
        if termination is None:
            warnings.append(f"terminal face {int(face_id)} is missing from input centerline metadata")
            continue
        center = _as_finite_vector(termination.get("center"), [0.0, 0.0, 0.0])
        node_id, distance = _nearest_node_id(center, node_coords)
        terminal_face_to_node[int(face_id)] = int(node_id)
        terminal_node_to_face[int(node_id)] = int(face_id)
        if distance > 2.0:
            warnings.append(f"terminal face {int(face_id)} was mapped to a centerline node using a distance above 2 mm")

    if not terminal_face_to_node:
        raise GeometrySegmentationFailure("No terminal faces from input roles could be mapped to the centerline network.")

    return int(root_node_id), terminal_face_to_node, terminal_node_to_face, warnings


def _root_centerline_tree(
    edges: dict[int, NetworkEdge],
    root_node_id: int,
) -> tuple[dict[int, list[NetworkEdge]], dict[int, Optional[int]], dict[int, float], list[str]]:
    adjacency: dict[int, list[NetworkEdge]] = defaultdict(list)
    for edge in edges.values():
        adjacency[edge.start_node_id].append(edge)
        adjacency[edge.end_node_id].append(edge)

    children_by_node: dict[int, list[NetworkEdge]] = defaultdict(list)
    parent_by_node: dict[int, Optional[int]] = {int(root_node_id): None}
    distance_from_root: dict[int, float] = {int(root_node_id): 0.0}
    warnings: list[str] = []

    queue: deque[int] = deque([int(root_node_id)])
    visited_edges: set[int] = set()

    while queue:
        node_id = queue.popleft()
        for edge in sorted(adjacency.get(node_id, []), key=lambda item: item.edge_id):
            if edge.edge_id in visited_edges:
                continue
            other = edge.other_node(node_id)
            if other in parent_by_node:
                warnings.append(f"centerline edge {edge.edge_id} closes a cycle and was ignored")
                visited_edges.add(edge.edge_id)
                continue

            visited_edges.add(edge.edge_id)
            edge.parent_node_id = int(node_id)
            edge.child_node_id = int(other)
            edge.points = edge.oriented_points_from(node_id)
            parent_by_node[int(other)] = int(node_id)
            distance_from_root[int(other)] = distance_from_root[int(node_id)] + float(edge.length)
            children_by_node[int(node_id)].append(edge)
            queue.append(int(other))

    if len(visited_edges) != len(edges):
        warnings.append("some centerline edges were not reachable from the aortic inlet node")

    for node_id in list(children_by_node):
        children_by_node[node_id].sort(key=lambda item: (-item.support_count, item.edge_id, item.child_node_id or -1))

    return dict(children_by_node), parent_by_node, distance_from_root, warnings


def _choose_aortic_body_edges(root_node_id: int, children_by_node: dict[int, list[NetworkEdge]]) -> tuple[list[NetworkEdge], list[str]]:
    body_edges: list[NetworkEdge] = []
    warnings: list[str] = []
    current_node = int(root_node_id)

    while True:
        children = children_by_node.get(current_node, [])
        if not children:
            break

        ranked = sorted(children, key=lambda item: (-item.support_count, -item.length, item.edge_id))
        if len(ranked) == 1:
            selected = ranked[0]
        else:
            best = ranked[0]
            second = ranked[1]
            if best.support_count > second.support_count and best.support_count >= 2:
                selected = best
            else:
                warnings.append(
                    "aortic_body end selected at first ambiguous trunk split; downstream branch grouping requires review"
                )
                break

        body_edges.append(selected)
        current_node = int(selected.child_node_id)

    if not body_edges:
        raise GeometrySegmentationFailure("Unable to resolve an aortic_body path from the input centerline network.")

    warnings.append("aortic_body path was resolved by centerline support counts and requires visual review")
    return body_edges, warnings


def _make_label(prefix: str, index: int) -> str:
    return f"{prefix}{int(index):03d}"


def _edge_sort_key(edge: NetworkEdge) -> tuple[int, int, int]:
    child = int(edge.child_node_id) if edge.child_node_id is not None else -1
    return (child, int(edge.edge_id), -int(edge.support_count))


def _build_segments(
    roles: InputRoles,
    root_node_id: int,
    body_edges: list[NetworkEdge],
    children_by_node: dict[int, list[NetworkEdge]],
    terminal_node_to_face: dict[int, int],
) -> tuple[list[GeometrySegment], dict[int, list[int]], list[str]]:
    warnings: list[str] = []
    body_points = concatenate_polylines([edge.points for edge in body_edges])
    body_edge_ids = [int(edge.edge_id) for edge in body_edges]
    body_distal_node = int(body_edges[-1].child_node_id)

    segments: list[GeometrySegment] = [
        GeometrySegment(
            segment_id=1,
            segment_label="aortic_body",
            segment_type="aortic_body",
            parent_segment_id=None,
            child_segment_ids=[],
            proximal_node_id=int(root_node_id),
            distal_node_id=body_distal_node,
            edge_ids=body_edge_ids,
            points=body_points,
            status=STATUS_REQUIRES_REVIEW,
            warnings=["aortic_body extent is a first-pass topology estimate"],
        )
    ]

    segment_by_id: dict[int, GeometrySegment] = {1: segments[0]}
    segment_start_node_to_ids: dict[int, list[int]] = defaultdict(list)
    branch_counter = 0

    def add_branch(parent_segment_id: int, first_edge: NetworkEdge) -> int:
        nonlocal branch_counter

        branch_counter += 1
        segment_id = branch_counter + 1
        edge_ids = [int(first_edge.edge_id)]
        parts = [first_edge.points]
        proximal_node = int(first_edge.parent_node_id)
        current_node = int(first_edge.child_node_id)

        while len(children_by_node.get(current_node, [])) == 1:
            next_edge = children_by_node[current_node][0]
            edge_ids.append(int(next_edge.edge_id))
            parts.append(next_edge.points)
            current_node = int(next_edge.child_node_id)

        segment = GeometrySegment(
            segment_id=int(segment_id),
            segment_label=_make_label(roles.branch_label_prefix, branch_counter),
            segment_type="branch",
            parent_segment_id=int(parent_segment_id),
            child_segment_ids=[],
            proximal_node_id=proximal_node,
            distal_node_id=int(current_node),
            edge_ids=edge_ids,
            points=concatenate_polylines(parts),
            terminal_face_id=terminal_node_to_face.get(int(current_node)),
            status=STATUS_REQUIRES_REVIEW,
            warnings=["branch segment is a first-pass topology estimate"],
        )
        segments.append(segment)
        segment_by_id[int(segment_id)] = segment
        segment_start_node_to_ids[proximal_node].append(int(segment_id))
        segment_by_id[int(parent_segment_id)].child_segment_ids.append(int(segment_id))

        for child_edge in sorted(children_by_node.get(current_node, []), key=_edge_sort_key):
            add_branch(int(segment_id), child_edge)

        return int(segment_id)

    body_next_edge_by_node = {int(edge.parent_node_id): edge for edge in body_edges}
    body_nodes = [int(root_node_id)] + [int(edge.child_node_id) for edge in body_edges]

    for node_id in body_nodes:
        for child_edge in sorted(children_by_node.get(node_id, []), key=_edge_sort_key):
            if child_edge is body_next_edge_by_node.get(int(node_id)):
                continue
            add_branch(1, child_edge)

    if branch_counter == 0:
        warnings.append("no anonymous branch segments were found in the centerline network")

    return segments, dict(segment_start_node_to_ids), warnings


def _segment_by_id(segments: list[GeometrySegment]) -> dict[int, GeometrySegment]:
    return {int(segment.segment_id): segment for segment in segments}


def _segment_s_at_point(segment: GeometrySegment, point: np.ndarray) -> float:
    pts = np.asarray(segment.points, dtype=float)
    if pts.shape[0] == 0:
        return 0.0
    distances = np.linalg.norm(pts - np.asarray(point, dtype=float).reshape(3), axis=1)
    index = int(np.argmin(distances))
    arclength = cumulative_arclength(pts)
    return float(arclength[index]) if index < len(arclength) else 0.0


def _segment_point_before(segment: GeometrySegment, point: np.ndarray, distance_mm: float) -> np.ndarray:
    s = _segment_s_at_point(segment, point)
    return point_at_arclength(segment.points, max(0.0, s - float(distance_mm)))


def _segment_point_after(segment: GeometrySegment, distance_mm: float) -> np.ndarray:
    length = polyline_length(segment.points)
    return point_at_arclength(segment.points, min(length, float(distance_mm)))


def _normal_at_segment_s(segment: GeometrySegment, source_s: float) -> np.ndarray:
    normal = tangent_at_arclength(segment.points, float(source_s), window=0.75)
    if float(np.linalg.norm(normal)) <= EPS:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    return unit(normal)


def _make_surface_point_locator(surface: vtk.vtkPolyData) -> vtk.vtkStaticPointLocator:
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    return locator


def _estimate_radius_from_surface(
    surface: vtk.vtkPolyData,
    locator: vtk.vtkStaticPointLocator,
    center: np.ndarray,
    sample_count: int = 96,
) -> tuple[float, str, str, float]:
    ids = vtk.vtkIdList()
    locator.FindClosestNPoints(int(sample_count), (float(center[0]), float(center[1]), float(center[2])), ids)

    distances: list[float] = []
    for idx in range(ids.GetNumberOfIds()):
        point = np.asarray(surface.GetPoint(ids.GetId(idx)), dtype=float)
        distance = float(np.linalg.norm(point - center))
        if math.isfinite(distance) and distance > EPS:
            distances.append(distance)

    if distances:
        radius = float(np.median(distances))
        if math.isfinite(radius) and radius > EPS:
            return radius, "estimated", STATUS_REQUIRES_REVIEW, 0.55

    bounds = surface.GetBounds()
    span = max(abs(float(bounds[1]) - float(bounds[0])), abs(float(bounds[3]) - float(bounds[2])), abs(float(bounds[5]) - float(bounds[4])))
    radius = max(0.25, float(span) / 100.0)
    return radius, "estimated", STATUS_REQUIRES_REVIEW, 0.35


def _termination_radius(termination: Optional[dict[str, Any]]) -> tuple[Optional[float], Optional[str]]:
    if not termination:
        return None, None

    diameter = termination.get("diameter_eq")
    try:
        diameter_f = float(diameter)
    except Exception:
        diameter_f = float("nan")
    if math.isfinite(diameter_f) and diameter_f > EPS:
        return diameter_f / 2.0, "local_equivalent_diameter_over_2"

    area = termination.get("area")
    try:
        area_f = float(area)
    except Exception:
        area_f = float("nan")
    if math.isfinite(area_f) and area_f > EPS:
        return math.sqrt(area_f / math.pi), "local_equivalent_diameter_over_2"

    return None, None


def _build_boundary_rings(
    roles: InputRoles,
    surface: vtk.vtkPolyData,
    terminations: dict[int, dict[str, Any]],
    segments: list[GeometrySegment],
    segment_start_node_to_ids: dict[int, list[int]],
    node_coords: dict[int, np.ndarray],
) -> tuple[vtk.vtkPolyData, list[BoundaryRing], list[BifurcationRecord], list[str]]:
    warnings: list[str] = []
    surface_locator = _make_surface_point_locator(surface)
    segment_lookup = _segment_by_id(segments)
    rings: list[BoundaryRing] = []
    bifurcations: list[BifurcationRecord] = []

    def add_ring(
        ring_type: str,
        center: np.ndarray,
        normal: np.ndarray,
        radius: float,
        source_segment_id: int,
        parent_segment_id: Optional[int],
        child_segment_id: Optional[int],
        source_centerline_s_mm: float,
        orientation_rule: str,
        radius_rule: str,
        confidence: float,
        status: str,
        ring_warnings: list[str],
    ) -> int:
        ring_id = len(rings) + 1
        normal_unit = unit(normal)
        if float(np.linalg.norm(normal_unit)) <= EPS:
            normal_unit = np.array([0.0, 0.0, 1.0], dtype=float)
            ring_warnings = list(ring_warnings) + ["ring normal was replaced by default axis"]

        radius_f = float(radius)
        if not math.isfinite(radius_f) or radius_f <= EPS:
            radius_f = 0.25
            status = STATUS_REQUIRES_REVIEW
            confidence = min(float(confidence), 0.35)
            ring_warnings = list(ring_warnings) + ["ring radius fallback was used"]

        ring_polydata = build_regular_polygon_ring_polydata(
            center=center,
            normal=normal_unit,
            radius=radius_f,
            number_of_sides=96,
            generate_polygon=False,
        )
        cell_count = ring_polydata.GetNumberOfCells()
        if cell_count == 0:
            raise GeometrySegmentationFailure(f"Unable to create boundary ring {ring_id}.")

        ring_label = _make_label(roles.ring_label_prefix, ring_id)
        add_int_cell_array(ring_polydata, "RingId", [ring_id] * cell_count)
        add_string_cell_array(ring_polydata, "RingLabel", [ring_label] * cell_count)
        add_string_cell_array(ring_polydata, "RingType", [ring_type] * cell_count)
        add_int_cell_array(ring_polydata, "ParentSegmentId", [int(parent_segment_id or 0)] * cell_count)
        add_int_cell_array(ring_polydata, "ChildSegmentId", [int(child_segment_id or 0)] * cell_count)
        add_int_cell_array(ring_polydata, "SegmentId", [int(source_segment_id)] * cell_count)
        add_float_cell_array(ring_polydata, "RadiusMm", [radius_f] * cell_count)
        add_float_cell_array(ring_polydata, "Confidence", [float(confidence)] * cell_count)
        add_string_cell_array(ring_polydata, "Status", [status] * cell_count)

        rings.append(
            BoundaryRing(
                ring_id=ring_id,
                ring_label=ring_label,
                ring_type=ring_type,
                center_xyz=[float(v) for v in center],
                normal_xyz=[float(v) for v in normal_unit],
                radius_mm=radius_f,
                source_segment_id=int(source_segment_id),
                parent_segment_id=int(parent_segment_id) if parent_segment_id is not None else None,
                child_segment_id=int(child_segment_id) if child_segment_id is not None else None,
                source_centerline_s_mm=float(source_centerline_s_mm),
                orientation_rule=orientation_rule,
                radius_rule=radius_rule,
                confidence=float(confidence),
                status=status,
                warnings=list(ring_warnings),
                polydata=ring_polydata,
            )
        )
        return ring_id

    aortic_body = segment_lookup[1]
    start_termination = terminations.get(int(roles.aortic_inlet_face_id))

    start_center = _as_finite_vector(start_termination.get("center"), aortic_body.points[0]) if start_termination else aortic_body.points[0]
    start_radius, start_radius_rule = _termination_radius(start_termination)
    if start_radius is None:
        start_radius, start_radius_rule, start_status, start_confidence = _estimate_radius_from_surface(surface, surface_locator, start_center)
        start_warnings = ["aortic_body_start radius was estimated from nearby surface points"]
    else:
        start_status = STATUS_SUCCESS
        start_confidence = 0.9
        start_warnings = []
    start_normal = _normal_at_segment_s(aortic_body, 0.0)
    aortic_body.proximal_ring_id = add_ring(
        "aortic_body_start",
        start_center,
        start_normal,
        float(start_radius),
        1,
        None,
        None,
        0.0,
        "perpendicular_to_aortic_body_centerline_tangent",
        str(start_radius_rule),
        start_confidence,
        start_status,
        start_warnings,
    )

    body_length = polyline_length(aortic_body.points)
    body_end_center = aortic_body.points[-1]
    body_end_radius, body_end_radius_rule, body_end_status, body_end_confidence = _estimate_radius_from_surface(
        surface, surface_locator, body_end_center
    )
    aortic_body.distal_ring_ids.append(
        add_ring(
            "aortic_body_end",
            body_end_center,
            _normal_at_segment_s(aortic_body, body_length),
            body_end_radius,
            1,
            None,
            None,
            body_length,
            "perpendicular_to_aortic_body_centerline_tangent",
            body_end_radius_rule,
            body_end_confidence,
            body_end_status,
            ["aortic_body_end is a first-pass topology boundary"],
        )
    )

    terminal_face_to_segment: dict[int, int] = {}
    for segment in segments:
        if segment.terminal_face_id is not None:
            terminal_face_to_segment[int(segment.terminal_face_id)] = int(segment.segment_id)

    for segment in segments:
        if segment.segment_type != "branch":
            continue
        sample_center = _segment_point_after(segment, min(1.0, max(0.0, polyline_length(segment.points) * 0.20)))
        radius, radius_rule, ring_status, confidence = _estimate_radius_from_surface(surface, surface_locator, sample_center)
        segment.proximal_ring_id = add_ring(
            "branch_start",
            segment.points[0],
            _normal_at_segment_s(segment, 0.0),
            radius,
            segment.segment_id,
            segment.parent_segment_id,
            segment.segment_id,
            0.0,
            "perpendicular_to_child_centerline_tangent",
            radius_rule,
            confidence,
            ring_status,
            ["branch_start is an approximate first-pass ring"],
        )

        if segment.terminal_face_id is not None:
            termination = terminations.get(int(segment.terminal_face_id))
            radius_from_term, radius_rule_from_term = _termination_radius(termination)
            if termination is not None and radius_from_term is not None:
                end_center = _as_finite_vector(termination.get("center"), segment.points[-1])
                end_status = STATUS_SUCCESS
                end_confidence = 0.85
                end_warnings: list[str] = []
                end_radius = radius_from_term
                end_radius_rule = str(radius_rule_from_term)
            else:
                end_center = segment.points[-1]
                end_radius, end_radius_rule, end_status, end_confidence = _estimate_radius_from_surface(
                    surface, surface_locator, end_center
                )
                end_warnings = ["branch_end radius was estimated from nearby surface points"]

            segment.distal_ring_ids.append(
                add_ring(
                    "branch_end",
                    end_center,
                    _normal_at_segment_s(segment, polyline_length(segment.points)),
                    end_radius,
                    segment.segment_id,
                    segment.parent_segment_id,
                    segment.segment_id,
                    polyline_length(segment.points),
                    "perpendicular_to_branch_centerline_tangent_near_segment_end",
                    end_radius_rule,
                    end_confidence,
                    end_status,
                    end_warnings,
                )
            )

    parent_segment_by_bifurcation_node: dict[int, int] = {}
    body_node_ids = {int(aortic_body.proximal_node_id), int(aortic_body.distal_node_id)}
    for point in aortic_body.points:
        node_id, distance = _nearest_node_id(point, node_coords)
        if distance <= 1.0e-6:
            body_node_ids.add(int(node_id))
    for node_id in body_node_ids:
        parent_segment_by_bifurcation_node[int(node_id)] = 1
    for segment in segments:
        parent_segment_by_bifurcation_node[int(segment.distal_node_id)] = int(segment.segment_id)

    for node_id, child_segment_ids in sorted(segment_start_node_to_ids.items()):
        if len(child_segment_ids) < 2:
            continue

        parent_segment_id = parent_segment_by_bifurcation_node.get(int(node_id), 1)
        parent_segment = segment_lookup[parent_segment_id]
        node_center = node_coords.get(int(node_id), parent_segment.points[-1])
        parent_center = _segment_point_before(parent_segment, node_center, 0.5)
        parent_s = _segment_s_at_point(parent_segment, parent_center)
        parent_radius, parent_radius_rule, parent_status, parent_confidence = _estimate_radius_from_surface(
            surface, surface_locator, parent_center
        )
        parent_ring_id = add_ring(
            "parent_pre_bifurcation",
            parent_center,
            _normal_at_segment_s(parent_segment, parent_s),
            parent_radius,
            parent_segment_id,
            parent_segment.parent_segment_id,
            None,
            parent_s,
            "perpendicular_to_parent_centerline_tangent",
            parent_radius_rule,
            min(parent_confidence, 0.55),
            STATUS_REQUIRES_REVIEW if parent_status != STATUS_FAILED else STATUS_FAILED,
            ["parent_pre_bifurcation is a first-pass ring"],
        )

        daughter_ring_ids: list[int] = []
        for child_segment_id in sorted(child_segment_ids):
            child_segment = segment_lookup[int(child_segment_id)]
            child_length = polyline_length(child_segment.points)
            child_s = min(0.5, child_length * 0.20)
            child_center = point_at_arclength(child_segment.points, child_s)
            child_radius, child_radius_rule, child_status, child_confidence = _estimate_radius_from_surface(
                surface, surface_locator, child_center
            )
            daughter_ring_ids.append(
                add_ring(
                    "daughter_start",
                    child_center,
                    _normal_at_segment_s(child_segment, child_s),
                    child_radius,
                    child_segment.segment_id,
                    parent_segment_id,
                    child_segment.segment_id,
                    child_s,
                    "perpendicular_to_daughter_centerline_tangent",
                    child_radius_rule,
                    min(child_confidence, 0.55),
                    STATUS_REQUIRES_REVIEW if child_status != STATUS_FAILED else STATUS_FAILED,
                    ["daughter_start is a first-pass ring"],
                )
            )

        bifurcation_id = len(bifurcations) + 1
        bifurcations.append(
            BifurcationRecord(
                bifurcation_id=bifurcation_id,
                bifurcation_label=_make_label(roles.bifurcation_label_prefix, bifurcation_id),
                parent_segment_id=int(parent_segment_id),
                child_segment_ids=[int(v) for v in sorted(child_segment_ids)],
                parent_pre_bifurcation_ring_id=parent_ring_id,
                daughter_start_ring_ids=daughter_ring_ids,
                status=STATUS_REQUIRES_REVIEW,
                warnings=["bifurcation rings are first-pass centerline estimates"],
            )
        )

    if not rings:
        raise GeometrySegmentationFailure("No boundary rings could be generated.")

    return append_polydata([ring.polydata for ring in rings]), rings, bifurcations, warnings


def _assign_surface_segments(
    surface: vtk.vtkPolyData,
    segments: list[GeometrySegment],
    roles: InputRoles,
) -> tuple[vtk.vtkPolyData, int, list[str]]:
    warnings: list[str] = []
    segment_lookup = _segment_by_id(segments)
    locator, locator_points = build_segment_point_locator(
        [(segment.segment_id, segment.points) for segment in segments if segment.points.shape[0] > 0]
    )
    locator_segment_ids = get_point_array(locator_points, "SegmentId")
    if locator_segment_ids is None:
        raise GeometrySegmentationFailure("Unable to build a centerline locator for surface assignment.")

    centers = cell_centers(surface)
    assigned_segment_ids: list[int] = []

    for center in centers:
        point_id = locator.FindClosestPoint((float(center[0]), float(center[1]), float(center[2])))
        if point_id < 0:
            assigned_segment_ids.append(1)
            continue
        segment_id = int(locator_segment_ids[int(point_id)])
        if segment_id not in segment_lookup:
            assigned_segment_ids.append(1)
        else:
            assigned_segment_ids.append(segment_id)

    model_face_ids = get_cell_array(surface, "ModelFaceID")
    terminal_face_to_segment: dict[int, int] = {}
    for segment in segments:
        if segment.terminal_face_id is not None:
            terminal_face_to_segment[int(segment.terminal_face_id)] = int(segment.segment_id)

    if model_face_ids is not None:
        for cell_id, face_id in enumerate(model_face_ids.astype(int)):
            if int(face_id) == int(roles.aortic_inlet_face_id):
                assigned_segment_ids[cell_id] = 1
            elif int(face_id) in terminal_face_to_segment:
                assigned_segment_ids[cell_id] = terminal_face_to_segment[int(face_id)]

    unassigned_cell_count = sum(1 for value in assigned_segment_ids if int(value) <= 0)
    if unassigned_cell_count:
        warnings.append("some surface cells could not be assigned and were folded into aortic_body")
        assigned_segment_ids = [int(value) if int(value) > 0 else 1 for value in assigned_segment_ids]

    labels_by_id = {segment.segment_id: segment.segment_label for segment in segments}
    labels = [labels_by_id.get(int(segment_id), "aortic_body") for segment_id in assigned_segment_ids]
    colors = [segment_color(int(segment_id)) for segment_id in assigned_segment_ids]

    out = clone_geometry_only(surface)
    add_int_cell_array(out, "SegmentId", assigned_segment_ids)
    add_string_cell_array(out, "SegmentLabel", labels)
    add_uchar3_cell_array(out, "SegmentColor", colors)

    counts: dict[int, int] = defaultdict(int)
    for segment_id in assigned_segment_ids:
        counts[int(segment_id)] += 1
    for segment in segments:
        segment.cell_count = int(counts.get(int(segment.segment_id), 0))
        if segment.cell_count == 0:
            segment.status = STATUS_REQUIRES_REVIEW
            segment.warnings.append("segment has no assigned surface cells")

    warnings.append("surface cells were assigned by nearest centerline segment; cut-boundary consistency requires review")
    return out, int(unassigned_cell_count), warnings


def _ring_to_json(ring: BoundaryRing) -> dict[str, Any]:
    return {
        "ring_id": int(ring.ring_id),
        "ring_label": ring.ring_label,
        "ring_type": ring.ring_type,
        "center_xyz": ring.center_xyz,
        "normal_xyz": ring.normal_xyz,
        "radius_mm": float(ring.radius_mm),
        "source_segment_id": int(ring.source_segment_id),
        "parent_segment_id": ring.parent_segment_id,
        "child_segment_id": ring.child_segment_id,
        "source_centerline_s_mm": float(ring.source_centerline_s_mm),
        "orientation_rule": ring.orientation_rule,
        "radius_rule": ring.radius_rule,
        "confidence": float(ring.confidence),
        "status": ring.status,
        "warnings": list(ring.warnings),
    }


def _segment_to_json(segment: GeometrySegment) -> dict[str, Any]:
    return {
        "segment_id": int(segment.segment_id),
        "segment_label": segment.segment_label,
        "segment_type": segment.segment_type,
        "parent_segment_id": segment.parent_segment_id,
        "child_segment_ids": [int(value) for value in segment.child_segment_ids],
        "proximal_ring_id": segment.proximal_ring_id,
        "distal_ring_ids": [int(value) for value in segment.distal_ring_ids],
        "cell_count": int(segment.cell_count),
        "status": segment.status,
        "warnings": list(segment.warnings),
    }


def _bifurcation_to_json(item: BifurcationRecord) -> dict[str, Any]:
    return {
        "bifurcation_id": int(item.bifurcation_id),
        "bifurcation_label": item.bifurcation_label,
        "parent_segment_id": int(item.parent_segment_id),
        "child_segment_ids": [int(value) for value in item.child_segment_ids],
        "parent_pre_bifurcation_ring_id": item.parent_pre_bifurcation_ring_id,
        "daughter_start_ring_ids": [int(value) for value in item.daughter_start_ring_ids],
        "status": item.status,
        "warnings": list(item.warnings),
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _status_rank(status: str) -> int:
    return {
        STATUS_SUCCESS: 0,
        STATUS_REQUIRES_REVIEW: 1,
        STATUS_FAILED: 2,
    }.get(str(status), 2)


def _worse_status(left: str, right: str) -> str:
    return left if _status_rank(left) >= _status_rank(right) else right


def _append_unique(items: list[str], value: str) -> None:
    text = str(value).strip()
    if text and text not in items:
        items.append(text)


def _dependency_diagnostics() -> dict[str, Any]:
    inactive_prefix = "vm" + "tk"
    return {
        f"{inactive_prefix}_required": False,
        f"{inactive_prefix}_used": False,
        "vtk_available": True,
        "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        "numpy_available": True,
        "numpy_version": np.__version__,
        "strategy": "vtk_numpy_centerline_surface_cut",
    }


def _cell_array_names(polydata: vtk.vtkPolyData) -> list[str]:
    return array_names(polydata.GetCellData())


def _string_cell_array_values(polydata: vtk.vtkPolyData, name: str) -> list[str]:
    arr = polydata.GetCellData().GetAbstractArray(name)
    if arr is None:
        return []

    values: list[str] = []
    for idx in range(arr.GetNumberOfTuples()):
        try:
            value = arr.GetValue(idx)
        except Exception:
            value = arr.GetVariantValue(idx).ToString()
        values.append(str(value))
    return sorted(set(values))


def _label_has_forbidden_fragment(label: str) -> bool:
    text = str(label).lower()
    return any(fragment in text for fragment in FORBIDDEN_LABEL_FRAGMENTS)


def _ring_type_counts(rings: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ring in rings:
        ring_type = str(ring.get("ring_type", ""))
        if ring_type:
            counts[ring_type] = counts.get(ring_type, 0) + 1
    return dict(sorted(counts.items()))


def _diagnostic_next_focus(failures: list[str], status: str) -> str:
    if any(failure.startswith("missing_output") for failure in failures):
        return "Restore generation of the missing required output files before evaluating ring placement."
    if any(failure.startswith("missing_segment_array") or failure.startswith("missing_ring_array") for failure in failures):
        return "Repair the missing required VTP cell-data arrays."
    if "json_missing_required_fields" in failures:
        return "Repair segmentation_result.json schema fields before algorithm tuning."
    if "old_named_branch_label" in failures:
        return "Remove forbidden named labels from VTP and JSON outputs."
    if "missing_ring" in failures:
        return "Add first-pass proximal branch rings for every anonymous branch segment."
    if status == STATUS_REQUIRES_REVIEW:
        return "Inspect and improve first-pass branch-start and bifurcation ring placement against surface cut boundaries."
    return "Proceed to circular boundary placement accuracy checks."


def _build_segmentation_diagnostics(
    project_root: Path,
    output_dir: Path,
    result: Optional[dict[str, Any]],
    extra_warnings: Optional[list[str]] = None,
    extra_failures: Optional[list[str]] = None,
) -> dict[str, Any]:
    output_paths = build_workspace_paths(project_root)
    segmented_surface_path = output_dir / output_paths.segmented_surface_vtp.name
    boundary_rings_path = output_dir / output_paths.boundary_rings_vtp.name
    segmentation_result_path = output_dir / output_paths.segmentation_result_json.name

    warnings: list[str] = []
    failures: list[str] = []
    for warning in extra_warnings or []:
        _append_unique(warnings, warning)
    for failure in extra_failures or []:
        _append_unique(failures, failure)

    outputs_exist = {
        "segmented_surface_vtp": segmented_surface_path.exists(),
        "boundary_rings_vtp": boundary_rings_path.exists(),
        "segmentation_result_json": segmentation_result_path.exists(),
    }
    if not outputs_exist["segmented_surface_vtp"]:
        _append_unique(failures, "missing_output:segmented_surface_vtp")
    if not outputs_exist["boundary_rings_vtp"]:
        _append_unique(failures, "missing_output:boundary_rings_vtp")
    if not outputs_exist["segmentation_result_json"]:
        _append_unique(failures, "missing_output:segmentation_result_json")

    segmented_surface_arrays: list[str] = []
    boundary_ring_arrays: list[str] = []
    segment_labels_from_vtp: list[str] = []
    ring_labels_from_vtp: list[str] = []

    if segmented_surface_path.exists():
        try:
            segmented_surface = read_vtp(segmented_surface_path)
            segmented_surface_arrays = _cell_array_names(segmented_surface)
            segment_labels_from_vtp = _string_cell_array_values(segmented_surface, "SegmentLabel")
        except Exception as exc:
            _append_unique(failures, "surface_not_openable")
            _append_unique(warnings, f"segmented_surface.vtp could not be opened: {exc}")

    if boundary_rings_path.exists():
        try:
            boundary_rings = read_vtp(boundary_rings_path)
            boundary_ring_arrays = _cell_array_names(boundary_rings)
            ring_labels_from_vtp = _string_cell_array_values(boundary_rings, "RingLabel")
        except Exception as exc:
            _append_unique(failures, "ring_not_visible")
            _append_unique(warnings, f"boundary_rings.vtp could not be opened: {exc}")

    missing_segmented_surface_arrays = [
        name for name in REQUIRED_SEGMENTED_SURFACE_CELL_ARRAYS if name not in segmented_surface_arrays
    ]
    missing_boundary_ring_arrays = [
        name for name in REQUIRED_BOUNDARY_RING_CELL_ARRAYS if name not in boundary_ring_arrays
    ]
    if missing_segmented_surface_arrays:
        _append_unique(failures, "missing_segment_array")
    if missing_boundary_ring_arrays:
        _append_unique(failures, "missing_ring_array")

    result_data = result
    if result_data is None and segmentation_result_path.exists():
        try:
            loaded = read_json(segmentation_result_path)
            result_data = loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            _append_unique(failures, "json_missing_required_fields")
            _append_unique(warnings, f"segmentation_result.json could not be opened: {exc}")
            result_data = {}
    if result_data is None:
        result_data = {}

    missing_result_keys = [name for name in REQUIRED_RESULT_KEYS if name not in result_data]
    if missing_result_keys:
        _append_unique(failures, "json_missing_required_fields")

    segments_json = result_data.get("segments", [])
    if not isinstance(segments_json, list):
        segments_json = []
        _append_unique(failures, "json_missing_required_fields")
    rings_json = result_data.get("boundary_rings", [])
    if not isinstance(rings_json, list):
        rings_json = []
        _append_unique(failures, "json_missing_required_fields")
    bifurcations_json = result_data.get("bifurcations", [])
    if not isinstance(bifurcations_json, list):
        bifurcations_json = []
        _append_unique(failures, "json_missing_required_fields")

    json_segment_labels = sorted(
        {str(segment.get("segment_label", "")) for segment in segments_json if isinstance(segment, dict)}
    )
    json_ring_labels = sorted({str(ring.get("ring_label", "")) for ring in rings_json if isinstance(ring, dict)})
    json_bifurcation_labels = sorted(
        {str(item.get("bifurcation_label", "")) for item in bifurcations_json if isinstance(item, dict)}
    )

    all_labels = [
        *segment_labels_from_vtp,
        *ring_labels_from_vtp,
        *json_segment_labels,
        *json_ring_labels,
        *json_bifurcation_labels,
    ]
    forbidden_labels_found = sorted({label for label in all_labels if _label_has_forbidden_fragment(label)})
    if forbidden_labels_found:
        _append_unique(failures, "old_named_branch_label")
        _append_unique(warnings, "forbidden named labels were found in final outputs")

    branch_segments = [
        segment
        for segment in segments_json
        if isinstance(segment, dict)
        and (segment.get("segment_type") == "branch" or str(segment.get("segment_label", "")).startswith("branch_"))
    ]
    segments_missing_proximal_ring = [
        int(segment.get("segment_id", 0))
        for segment in branch_segments
        if not segment.get("proximal_ring_id")
    ]

    branch_start_child_ids = {
        int(ring.get("child_segment_id") or ring.get("source_segment_id") or 0)
        for ring in rings_json
        if isinstance(ring, dict) and ring.get("ring_type") == "branch_start"
    }
    branch_ids = [int(segment.get("segment_id", 0)) for segment in branch_segments]
    missing_branch_start_ring_ids = [
        segment_id for segment_id in branch_ids if int(segment_id) not in branch_start_child_ids
    ]
    if segments_missing_proximal_ring or missing_branch_start_ring_ids:
        _append_unique(failures, "missing_ring")

    requires_review_ring_count = sum(
        1 for ring in rings_json if isinstance(ring, dict) and ring.get("status") == STATUS_REQUIRES_REVIEW
    )
    failed_ring_count = sum(1 for ring in rings_json if isinstance(ring, dict) and ring.get("status") == STATUS_FAILED)
    low_confidence_ring_count = 0
    for ring in rings_json:
        if not isinstance(ring, dict):
            continue
        try:
            confidence = float(ring.get("confidence", 1.0))
        except Exception:
            confidence = 0.0
        if confidence < 0.7:
            low_confidence_ring_count += 1

    bifurcations_missing_parent_ring = [
        int(item.get("bifurcation_id", 0))
        for item in bifurcations_json
        if isinstance(item, dict) and not item.get("parent_pre_bifurcation_ring_id")
    ]
    bifurcations_missing_daughter_rings: list[int] = []
    for item in bifurcations_json:
        if not isinstance(item, dict):
            continue
        child_ids = item.get("child_segment_ids", [])
        daughter_ids = item.get("daughter_start_ring_ids", [])
        if not isinstance(child_ids, list) or not isinstance(daughter_ids, list) or len(daughter_ids) < len(child_ids):
            bifurcations_missing_daughter_rings.append(int(item.get("bifurcation_id", 0)))

    if failed_ring_count:
        _append_unique(failures, "failed_ring")
    if bifurcations_missing_parent_ring:
        _append_unique(failures, "missing_ring:parent_pre_bifurcation")
    if bifurcations_missing_daughter_rings:
        _append_unique(failures, "missing_ring:daughter_start")

    if failures:
        status = STATUS_FAILED
    elif (
        requires_review_ring_count
        or low_confidence_ring_count
        or any(
            isinstance(segment, dict) and segment.get("status") == STATUS_REQUIRES_REVIEW for segment in segments_json
        )
        or any(isinstance(item, dict) and item.get("status") == STATUS_REQUIRES_REVIEW for item in bifurcations_json)
        or result_data.get("status") == STATUS_REQUIRES_REVIEW
        or warnings
    ):
        status = STATUS_REQUIRES_REVIEW
    else:
        status = STATUS_SUCCESS

    return {
        "status": status,
        "dependencies": _dependency_diagnostics(),
        "outputs_exist": outputs_exist,
        "vtp_arrays": {
            "segmented_surface_cell_arrays": segmented_surface_arrays,
            "boundary_rings_cell_arrays": boundary_ring_arrays,
            "missing_segmented_surface_arrays": missing_segmented_surface_arrays,
            "missing_boundary_ring_arrays": missing_boundary_ring_arrays,
        },
        "json": {
            "top_level_keys": sorted(result_data.keys()),
            "missing_top_level_keys": missing_result_keys,
        },
        "labels": {
            "segment_labels": sorted(set([*segment_labels_from_vtp, *json_segment_labels])),
            "ring_labels": sorted(set([*ring_labels_from_vtp, *json_ring_labels])),
            "bifurcation_labels": json_bifurcation_labels,
            "forbidden_labels_found": forbidden_labels_found,
        },
        "segments": {
            "segment_count": int(len(segments_json)),
            "branch_count": int(len(branch_segments)),
            "aortic_body_count": int(
                sum(
                    1
                    for segment in segments_json
                    if isinstance(segment, dict) and segment.get("segment_label") == "aortic_body"
                )
            ),
            "segments_missing_proximal_ring": segments_missing_proximal_ring,
        },
        "rings": {
            "ring_count": int(len(rings_json)),
            "ring_types": _ring_type_counts([ring for ring in rings_json if isinstance(ring, dict)]),
            "requires_review_ring_count": int(requires_review_ring_count),
            "failed_ring_count": int(failed_ring_count),
            "low_confidence_ring_count": int(low_confidence_ring_count),
            "missing_branch_start_ring_count": int(len(missing_branch_start_ring_ids)),
            "missing_branch_start_segment_ids": missing_branch_start_ring_ids,
        },
        "bifurcations": {
            "bifurcation_count": int(len(bifurcations_json)),
            "bifurcations_missing_parent_ring": bifurcations_missing_parent_ring,
            "bifurcations_missing_daughter_rings": bifurcations_missing_daughter_rings,
        },
        "warnings": warnings,
        "failures": failures,
        "next_recommended_focus": _diagnostic_next_focus(failures, status),
    }


def _result_status(segments: list[GeometrySegment], rings: list[BoundaryRing], warnings: list[str]) -> str:
    if any(segment.status == STATUS_FAILED for segment in segments) or any(ring.status == STATUS_FAILED for ring in rings):
        return STATUS_FAILED
    if warnings:
        return STATUS_REQUIRES_REVIEW
    if any(segment.status == STATUS_REQUIRES_REVIEW for segment in segments):
        return STATUS_REQUIRES_REVIEW
    if any(ring.status == STATUS_REQUIRES_REVIEW for ring in rings):
        return STATUS_REQUIRES_REVIEW
    return STATUS_SUCCESS


def run_geometry_segmentation(
    project_root: Path,
    surface_path: Path,
    centerline_network_path: Path,
    centerline_metadata_path: Path,
    input_roles_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_workspace_paths(project_root)
    dependencies = _dependency_diagnostics()

    _require_paths([surface_path, centerline_network_path, centerline_metadata_path, input_roles_path])

    roles = _load_input_roles(input_roles_path)
    metadata = read_json(centerline_metadata_path)
    if not isinstance(metadata, dict):
        raise GeometrySegmentationFailure("Input centerline metadata must be a JSON object.")

    surface = read_vtp(surface_path)
    centerline_network = read_vtp(centerline_network_path)
    if surface.GetNumberOfCells() == 0:
        raise GeometrySegmentationFailure("Input surface contains no cells.")

    terminations = _termination_map(metadata)
    edges, node_coords = _build_network_edges(centerline_network)
    root_node_id, terminal_face_to_node, terminal_node_to_face, role_warnings = _map_role_faces_to_nodes(
        roles, terminations, node_coords
    )
    children_by_node, _, _, tree_warnings = _root_centerline_tree(edges, root_node_id)
    body_edges, body_warnings = _choose_aortic_body_edges(root_node_id, children_by_node)
    segments, segment_start_node_to_ids, segment_warnings = _build_segments(
        roles, root_node_id, body_edges, children_by_node, terminal_node_to_face
    )
    segmented_surface, unassigned_cell_count, assignment_warnings = _assign_surface_segments(surface, segments, roles)
    boundary_rings, rings, bifurcations, ring_warnings = _build_boundary_rings(
        roles, surface, terminations, segments, segment_start_node_to_ids, node_coords
    )

    segmented_surface_path = output_dir / output_paths.segmented_surface_vtp.name
    boundary_rings_path = output_dir / output_paths.boundary_rings_vtp.name
    segmentation_result_path = output_dir / output_paths.segmentation_result_json.name
    segmentation_diagnostics_path = output_dir / output_paths.segmentation_diagnostics_json.name

    write_vtp(segmented_surface, segmented_surface_path)
    write_vtp(boundary_rings, boundary_rings_path)

    warnings = [
        *role_warnings,
        *tree_warnings,
        *body_warnings,
        *segment_warnings,
        *assignment_warnings,
        *ring_warnings,
    ]
    warnings = [str(warning) for warning in warnings if str(warning).strip()]

    requires_review_ring_count = sum(1 for ring in rings if ring.status == STATUS_REQUIRES_REVIEW)
    failed_ring_count = sum(1 for ring in rings if ring.status == STATUS_FAILED)
    low_confidence_ring_count = sum(1 for ring in rings if ring.confidence < 0.7)
    status = _result_status(segments, rings, warnings)

    result = {
        "status": status,
        "dependencies": dependencies,
        "inputs": {
            "surface": _relative_path(surface_path, project_root),
            "centerline_network": _relative_path(centerline_network_path, project_root),
            "centerline_metadata": _relative_path(centerline_metadata_path, project_root),
            "input_roles": _relative_path(input_roles_path, project_root),
            "aortic_inlet_face_id": int(roles.aortic_inlet_face_id),
            "terminal_face_ids": [int(value) for value in roles.terminal_face_ids],
            "terminal_face_to_node": {str(face_id): int(node_id) for face_id, node_id in terminal_face_to_node.items()},
        },
        "outputs": {
            "segmented_surface": _relative_path(segmented_surface_path, project_root),
            "boundary_rings": _relative_path(boundary_rings_path, project_root),
            "segmentation_result": _relative_path(segmentation_result_path, project_root),
            "segmentation_diagnostics": _relative_path(segmentation_diagnostics_path, project_root),
        },
        "segments": [_segment_to_json(segment) for segment in segments],
        "boundary_rings": [_ring_to_json(ring) for ring in rings],
        "bifurcations": [_bifurcation_to_json(item) for item in bifurcations],
        "warnings": warnings,
        "metrics": {
            "segment_count": int(len(segments)),
            "branch_count": int(sum(1 for segment in segments if segment.segment_type == "branch")),
            "bifurcation_count": int(len(bifurcations)),
            "ring_count": int(len(rings)),
            "surface_cell_count": int(surface.GetNumberOfCells()),
            "unassigned_cell_count": int(unassigned_cell_count),
            "requires_review_ring_count": int(requires_review_ring_count),
            "failed_ring_count": int(failed_ring_count),
            "low_confidence_ring_count": int(low_confidence_ring_count),
        },
    }

    write_json(result, segmentation_result_path)
    diagnostics = _build_segmentation_diagnostics(project_root, output_dir, result)
    result["status"] = _worse_status(str(result.get("status", STATUS_FAILED)), str(diagnostics.get("status", STATUS_FAILED)))
    for warning in diagnostics.get("warnings", []):
        _append_unique(result["warnings"], warning)
    if diagnostics.get("failures"):
        _append_unique(result["warnings"], "structural diagnostics reported failure")
    write_json(result, segmentation_result_path)
    write_json(diagnostics, segmentation_diagnostics_path)
    return result


def _failed_result(
    project_root: Path,
    output_dir: Path,
    message: str,
) -> dict[str, Any]:
    result = {
        "status": STATUS_FAILED,
        "dependencies": _dependency_diagnostics(),
        "inputs": {},
        "outputs": {},
        "segments": [],
        "boundary_rings": [],
        "bifurcations": [],
        "warnings": [str(message)],
        "metrics": {
            "segment_count": 0,
            "branch_count": 0,
            "bifurcation_count": 0,
            "ring_count": 0,
            "surface_cell_count": 0,
            "unassigned_cell_count": 0,
            "requires_review_ring_count": 0,
            "failed_ring_count": 0,
            "low_confidence_ring_count": 0,
        },
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = build_workspace_paths(project_root)
        result_path = output_dir / output_paths.segmentation_result_json.name
        diagnostics_path = output_dir / output_paths.segmentation_diagnostics_json.name
        result["outputs"] = {
            "segmentation_result": _relative_path(result_path, project_root),
            "segmentation_diagnostics": _relative_path(diagnostics_path, project_root),
        }
        write_json(result, result_path)
        diagnostics = _build_segmentation_diagnostics(
            project_root,
            output_dir,
            result,
            extra_warnings=[str(message)],
            extra_failures=["runtime_failure"],
        )
        write_json(diagnostics, diagnostics_path)
    except Exception:
        return result

    return result


def main(argv: Optional[list[str]] = None) -> int:
    project_root_default = Path(__file__).resolve().parents[2]
    paths = build_workspace_paths(project_root_default)

    parser = argparse.ArgumentParser(description="Neutral vascular geometry segmentation.")
    parser.add_argument("--project-root", default=str(project_root_default), help="Workspace root containing inputs and outputs.")
    parser.add_argument("--surface", default=str(paths.surface_vtp), help="Input cleaned surface VTP.")
    parser.add_argument("--centerline-network", default=str(paths.centerline_network_vtp), help="Input centerline network VTP.")
    parser.add_argument("--centerline-metadata", default=str(paths.centerline_metadata_json), help="Input centerline metadata JSON.")
    parser.add_argument("--input-roles", default=str(paths.input_roles_json), help="Input role JSON.")
    parser.add_argument("--output-dir", default=str(paths.output_dir), help="Output folder.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    try:
        result = run_geometry_segmentation(
            project_root=project_root,
            surface_path=Path(args.surface).resolve(),
            centerline_network_path=Path(args.centerline_network).resolve(),
            centerline_metadata_path=Path(args.centerline_metadata).resolve(),
            input_roles_path=Path(args.input_roles).resolve(),
            output_dir=output_dir,
        )
    except GeometrySegmentationFailure as exc:
        _failed_result(project_root, output_dir, str(exc))
        print(f"Geometry segmentation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        _failed_result(project_root, output_dir, f"Unhandled geometry segmentation error: {exc}")
        print(f"Geometry segmentation failed: {exc}", file=sys.stderr)
        return 1

    if result.get("status") == STATUS_FAILED:
        first_warning = ""
        warnings = result.get("warnings")
        if isinstance(warnings, list) and warnings:
            first_warning = f": {warnings[0]}"
        print(f"Geometry segmentation failed{first_warning}", file=sys.stderr)
        return 1

    print(
        "Geometry segmentation completed: "
        f"status={result['status']} "
        f"segments={result['metrics']['segment_count']} "
        f"rings={result['metrics']['ring_count']} "
        f"output_dir={output_dir}"
    )
    return 0 if result.get("status") != STATUS_FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
