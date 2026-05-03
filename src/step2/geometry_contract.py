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
    orthonormal_frame,
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

BRANCH_START_RING_ALGORITHM = "surface_validated_branch_start_ring_v1"
BRANCH_RING_SEARCH_START_MM = 0.0
BRANCH_RING_SEARCH_MAX_MM = 8.0
BRANCH_RING_SEARCH_FRACTION = 0.35
BRANCH_RING_SEARCH_STEP_MM = 0.25
BACKWARD_REFINE_STEP_MM = 0.10
MIN_CUT_COMPONENT_POINTS = 8
ZERO_OFFSET_TOLERANCE_MM = 0.05
MIN_STABLE_COMPACTNESS = 0.80
MIN_STABLE_SCORE = 0.75
MAX_STABLE_PARENT_CONTAMINATION = 0.25
MAX_SUCCESS_PARENT_CONTAMINATION = 0.30
MAX_REJECT_PARENT_CONTAMINATION = 0.45
MAX_STABLE_RADIUS_SPREAD_RATIO = 0.45
MAX_CLEAN_RADIUS_RATIO = 1.40
MAX_ZERO_OFFSET_RADIUS_RATIO_DELTA = 0.20


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
    metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class BranchStartSelection:
    center: np.ndarray
    normal: np.ndarray
    radius_mm: float
    source_s_mm: float
    radius_rule: str
    confidence: float
    status: str
    warnings: list[str]
    metadata: dict[str, Any]


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


def _project_point_to_polyline_s(points: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(points, dtype=float)
    target = np.asarray(point, dtype=float).reshape(3)
    if pts.shape[0] == 0:
        return 0.0, float("inf")
    if pts.shape[0] == 1:
        return 0.0, float(np.linalg.norm(target - pts[0]))

    arclength = cumulative_arclength(pts)
    best_s = 0.0
    best_distance = float("inf")
    for idx in range(pts.shape[0] - 1):
        a = pts[idx]
        b = pts[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= EPS:
            t = 0.0
            projected = a
        else:
            t = float(np.clip(np.dot(target - a, ab) / denom, 0.0, 1.0))
            projected = a + t * ab
        distance = float(np.linalg.norm(target - projected))
        if distance < best_distance:
            best_distance = distance
            best_s = float(arclength[idx] + t * np.linalg.norm(ab))
    return best_s, best_distance


def _plane_cut_surface(surface: vtk.vtkPolyData, center: np.ndarray, normal: np.ndarray) -> vtk.vtkPolyData:
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(center[0]), float(center[1]), float(center[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(surface)
    cutter.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(cutter.GetOutput())
    return out


def _connected_cut_components(cut: vtk.vtkPolyData) -> list[np.ndarray]:
    points = points_to_numpy(cut)
    point_count = int(points.shape[0])
    if point_count == 0:
        return []
    if cut.GetNumberOfCells() == 0:
        return [np.arange(point_count, dtype=int)]

    parent = list(range(point_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root

    referenced: set[int] = set()
    for cell_id in range(cut.GetNumberOfCells()):
        cell = cut.GetCell(cell_id)
        ids = _point_ids_for_cell(cell)
        if not ids:
            continue
        referenced.update(int(value) for value in ids)
        for idx in range(len(ids) - 1):
            union(int(ids[idx]), int(ids[idx + 1]))

    if not referenced:
        return [np.arange(point_count, dtype=int)]

    groups: dict[int, list[int]] = defaultdict(list)
    for point_id in referenced:
        groups[find(int(point_id))].append(int(point_id))

    components = [np.asarray(sorted(values), dtype=int) for values in groups.values() if values]
    components.sort(key=lambda item: int(item.shape[0]), reverse=True)
    return components


def _projected_component_area(points: np.ndarray, center: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, 0.0

    u, v = orthonormal_frame(normal)
    rel = pts - np.asarray(center, dtype=float).reshape(3)
    xy = np.column_stack([rel @ u, rel @ v])
    centroid_xy = np.mean(xy, axis=0)
    angles = np.arctan2(xy[:, 1] - centroid_xy[1], xy[:, 0] - centroid_xy[0])
    order = np.argsort(angles)
    ordered = xy[order]

    x = ordered[:, 0]
    y = ordered[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    deltas = np.diff(np.vstack([ordered, ordered[0]]), axis=0)
    perimeter = float(np.sum(np.linalg.norm(deltas, axis=1)))
    if not math.isfinite(area):
        area = 0.0
    if not math.isfinite(perimeter):
        perimeter = 0.0
    return area, perimeter


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _surface_cut_candidate_metrics(
    surface: vtk.vtkPolyData,
    segment: GeometrySegment,
    offset_mm: float,
    search_pass: str,
) -> dict[str, Any]:
    branch_length = polyline_length(segment.points)
    center = point_at_arclength(segment.points, float(np.clip(offset_mm, 0.0, branch_length)))
    normal = _normal_at_segment_s(segment, float(offset_mm))
    cut = _plane_cut_surface(surface, center, normal)
    cut_points = points_to_numpy(cut)
    components = _connected_cut_components(cut)

    candidate: dict[str, Any] = {
        "offset_mm": float(offset_mm),
        "center_xyz": [float(value) for value in center],
        "normal_xyz": [float(value) for value in normal],
        "cut_point_count": int(cut_points.shape[0]),
        "cut_component_count": int(len(components)),
        "selected_component_point_count": 0,
        "selected_component_centroid_xyz": None,
        "selected_component_area_estimate": None,
        "selected_component_equivalent_radius_mm": None,
        "selected_component_max_radius_mm": None,
        "selected_component_min_radius_mm": None,
        "equivalent_radius_mm": None,
        "centroid_distance_to_centerline_mm": None,
        "radius_spread_mm": None,
        "radius_cv_or_spread": None,
        "surface_distance_mean_mm": None,
        "surface_distance_max_mm": None,
        "compactness_score": 0.0,
        "parent_contamination_score": 1.0,
        "stability_score": 0.0,
        "classification": "invalid_no_cut",
        "accepted": False,
        "rejection_reason": "plane cut produced no usable contour component",
        "search_pass": search_pass,
        "_center_np": center,
        "_normal_np": normal,
        "_radial_cv": float("inf"),
    }

    if cut_points.shape[0] == 0 or not components:
        return candidate

    component_stats: list[dict[str, Any]] = []
    for component in components:
        if int(component.shape[0]) < 3:
            continue
        pts = cut_points[component]
        centroid = np.mean(pts, axis=0)
        radii = np.linalg.norm(pts - centroid, axis=1)
        finite_radii = radii[np.isfinite(radii)]
        if finite_radii.shape[0] < 3:
            continue
        area, perimeter = _projected_component_area(pts, centroid, normal)
        median_radius = float(np.median(finite_radii))
        if area > EPS:
            equivalent_radius = float(math.sqrt(area / math.pi))
        else:
            equivalent_radius = median_radius
            area = float(math.pi * equivalent_radius * equivalent_radius)
        max_radius = float(np.max(finite_radii))
        min_radius = float(np.min(finite_radii))
        radius_spread = max_radius - min_radius
        radial_mean = float(np.mean(finite_radii))
        radial_cv = float(np.std(finite_radii) / radial_mean) if radial_mean > EPS else float("inf")
        circularity = 0.0
        if perimeter > EPS and area > EPS:
            circularity = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0))
        compactness = float(np.clip(circularity * max(0.0, 1.0 - min(radial_cv, 1.0)), 0.0, 1.0))
        centroid_distance = float(np.linalg.norm(centroid - center))
        proximity_score = centroid_distance / max(equivalent_radius, EPS)
        score = proximity_score + (1.0 - compactness) + 0.02 * abs(int(component.shape[0]) - 96) / 96.0
        component_stats.append(
            {
                "point_ids": component,
                "point_count": int(component.shape[0]),
                "centroid": centroid,
                "area": float(area),
                "equivalent_radius": equivalent_radius,
                "max_radius": max_radius,
                "min_radius": min_radius,
                "radius_spread": float(radius_spread),
                "radial_cv": radial_cv,
                "compactness": compactness,
                "centroid_distance": centroid_distance,
                "score": float(score),
            }
        )

    if not component_stats:
        candidate["classification"] = "invalid_no_clean_component"
        candidate["rejection_reason"] = "plane cut components were too small for circular scoring"
        return candidate

    selected = min(component_stats, key=lambda item: float(item["score"]))
    equivalent_radius = float(selected["equivalent_radius"])
    radius_spread = float(selected["radius_spread"])
    centroid_distance = float(selected["centroid_distance"])
    radial_cv = float(selected["radial_cv"])
    compactness = float(selected["compactness"])

    large_nearby_components = 0
    for component in component_stats:
        if int(component["point_count"]) < MIN_CUT_COMPONENT_POINTS:
            continue
        if float(component["equivalent_radius"]) >= 0.65 * max(equivalent_radius, EPS):
            if float(component["centroid_distance"]) <= max(3.0 * equivalent_radius, 1.0):
                large_nearby_components += 1

    centroid_ratio = centroid_distance / max(equivalent_radius, EPS)
    spread_ratio = radius_spread / max(equivalent_radius, EPS)
    parent_score = 0.45 * min(1.0, centroid_ratio / 1.5)
    parent_score += 0.35 * min(1.0, spread_ratio / 2.0)
    parent_score += 0.20 * min(1.0, radial_cv / 0.6)
    if large_nearby_components > 1:
        parent_score = min(1.0, parent_score + 0.20)
    stability = compactness
    stability -= 0.20 * min(1.0, centroid_ratio / 1.25)
    stability -= 0.20 * min(1.0, spread_ratio / 1.75)
    stability -= 0.15 * min(1.0, radial_cv / 0.5)
    stability = float(np.clip(stability, 0.0, 1.0))

    candidate.update(
        {
            "selected_component_point_count": int(selected["point_count"]),
            "selected_component_centroid_xyz": [float(value) for value in selected["centroid"]],
            "selected_component_area_estimate": float(selected["area"]),
            "selected_component_equivalent_radius_mm": equivalent_radius,
            "selected_component_max_radius_mm": float(selected["max_radius"]),
            "selected_component_min_radius_mm": float(selected["min_radius"]),
            "equivalent_radius_mm": equivalent_radius,
            "centroid_distance_to_centerline_mm": centroid_distance,
            "radius_spread_mm": radius_spread,
            "radius_cv_or_spread": radial_cv,
            "surface_distance_mean_mm": 0.0,
            "surface_distance_max_mm": 0.0,
            "compactness_score": compactness,
            "parent_contamination_score": float(np.clip(parent_score, 0.0, 1.0)),
            "stability_score": stability,
            "classification": "candidate_surface_cut_evaluated",
            "rejection_reason": "",
            "_radial_cv": radial_cv,
        }
    )
    return candidate


def _candidate_offsets(branch_length: float) -> tuple[list[float], float]:
    length = max(0.0, float(branch_length))
    if length <= EPS:
        return [0.0], 0.0

    max_search = min(BRANCH_RING_SEARCH_MAX_MM, BRANCH_RING_SEARCH_FRACTION * length, length)
    if max_search <= EPS:
        return [0.0], 0.0

    offsets: list[float] = []
    current = BRANCH_RING_SEARCH_START_MM
    while current <= max_search + 1.0e-9:
        offsets.append(round(float(current), 6))
        current += BRANCH_RING_SEARCH_STEP_MM

    if offsets[-1] < max_search - 1.0e-6:
        offsets.append(round(float(max_search), 6))
    if len(offsets) == 1 and max_search > EPS:
        offsets.append(round(float(max_search), 6))

    return sorted(set(offsets)), float(max_search)


def _round_float(value: Any, ndigits: int = 6) -> Optional[float]:
    value_f = _safe_float(value, float("nan"))
    if not math.isfinite(value_f):
        return None
    return round(float(value_f), int(ndigits))


def _candidate_offset(candidate: dict[str, Any]) -> float:
    return _safe_float(candidate.get("offset_mm"), 0.0)


def _candidate_radius(candidate: dict[str, Any]) -> float:
    return _safe_float(candidate.get("equivalent_radius_mm"), float("nan"))


def _candidate_spread_ratio(candidate: dict[str, Any]) -> float:
    radius = _candidate_radius(candidate)
    if not math.isfinite(radius) or radius <= EPS:
        return float("inf")
    return _safe_float(candidate.get("radius_spread_mm"), float("inf")) / max(radius, EPS)


def _candidate_centroid_ratio(candidate: dict[str, Any]) -> float:
    radius = _candidate_radius(candidate)
    if not math.isfinite(radius) or radius <= EPS:
        return float("inf")
    return _safe_float(candidate.get("centroid_distance_to_centerline_mm"), float("inf")) / max(radius, EPS)


def _candidate_is_valid_surface_cut(candidate: dict[str, Any]) -> bool:
    radius = _candidate_radius(candidate)
    if not math.isfinite(radius) or radius <= 0.05:
        return False
    if int(candidate.get("selected_component_point_count") or 0) < MIN_CUT_COMPONENT_POINTS:
        return False
    if _safe_float(candidate.get("compactness_score"), 0.0) <= 0.0:
        return False
    return True


def _candidate_quality_score(candidate: dict[str, Any]) -> float:
    if not _candidate_is_valid_surface_cut(candidate):
        return -1.0e6
    offset = _candidate_offset(candidate)
    compactness = _safe_float(candidate.get("compactness_score"), 0.0)
    stability = _safe_float(candidate.get("stability_score"), 0.0)
    parent_score = _safe_float(candidate.get("parent_contamination_score"), 1.0)
    spread_ratio = min(2.0, _candidate_spread_ratio(candidate))
    centroid_ratio = min(2.0, _candidate_centroid_ratio(candidate))
    return (
        stability
        + 0.35 * compactness
        - 0.75 * parent_score
        - 0.25 * spread_ratio
        - 0.25 * centroid_ratio
        + min(0.10, 0.02 * max(0.0, offset))
    )


def _candidate_reference_eligible(candidate: dict[str, Any], *, relaxed: bool, allow_zero: bool) -> bool:
    if not _candidate_is_valid_surface_cut(candidate):
        return False
    if not allow_zero and _candidate_offset(candidate) <= ZERO_OFFSET_TOLERANCE_MM:
        return False

    radius = _candidate_radius(candidate)
    compactness = _safe_float(candidate.get("compactness_score"), 0.0)
    parent_score = _safe_float(candidate.get("parent_contamination_score"), 1.0)
    stability = _safe_float(candidate.get("stability_score"), 0.0)
    centroid_distance = _safe_float(candidate.get("centroid_distance_to_centerline_mm"), float("inf"))
    spread_ratio = _candidate_spread_ratio(candidate)

    min_compactness = 0.70 if relaxed else MIN_STABLE_COMPACTNESS
    max_parent = 0.40 if relaxed else MAX_STABLE_PARENT_CONTAMINATION
    min_stability = 0.58 if relaxed else MIN_STABLE_SCORE
    max_spread_ratio = 0.75 if relaxed else MAX_STABLE_RADIUS_SPREAD_RATIO
    centroid_limit = max(0.35 if relaxed else 0.25, (1.00 if relaxed else 0.75) * radius)

    return bool(
        compactness >= min_compactness
        and parent_score <= max_parent
        and stability >= min_stability
        and spread_ratio <= max_spread_ratio
        and centroid_distance <= centroid_limit
    )


def _stable_candidate_groups(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(candidates, key=lambda item: _candidate_offset(item))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    max_gap = max(BRANCH_RING_SEARCH_STEP_MM + 1.0e-6, 2.5 * BACKWARD_REFINE_STEP_MM)

    for candidate in ordered:
        if not current:
            current = [candidate]
            continue
        if _candidate_offset(candidate) - _candidate_offset(current[-1]) <= max_gap:
            current.append(candidate)
        else:
            groups.append(current)
            current = [candidate]
    if current:
        groups.append(current)

    plateau_groups: list[list[dict[str, Any]]] = []
    for group in groups:
        radii = [_candidate_radius(candidate) for candidate in group if _candidate_is_valid_surface_cut(candidate)]
        if not radii:
            continue
        if len(radii) == 1:
            plateau_groups.append(group)
            continue
        median_radius = float(np.median(radii))
        radius_cv = float(np.std(radii) / max(median_radius, EPS))
        if radius_cv <= 0.25:
            plateau_groups.append(group)
    return plateau_groups or groups


def _stable_group_score(group: list[dict[str, Any]]) -> float:
    radii = [_candidate_radius(candidate) for candidate in group if _candidate_is_valid_surface_cut(candidate)]
    radius_cv = 0.0
    if len(radii) > 1:
        radius_cv = float(np.std(radii) / max(float(np.median(radii)), EPS))
    avg_stability = float(np.mean([_safe_float(c.get("stability_score"), 0.0) for c in group]))
    avg_compactness = float(np.mean([_safe_float(c.get("compactness_score"), 0.0) for c in group]))
    avg_parent = float(np.mean([_safe_float(c.get("parent_contamination_score"), 1.0) for c in group]))
    avg_spread = float(np.mean([min(2.0, _candidate_spread_ratio(c)) for c in group]))
    median_offset = float(np.median([_candidate_offset(c) for c in group]))
    length_bonus = min(0.12, 0.04 * len(group))
    distal_bonus = min(0.12, 0.025 * max(0.0, median_offset))
    return avg_stability + 0.30 * avg_compactness - 0.65 * avg_parent - 0.25 * avg_spread - 0.25 * radius_cv + length_bonus + distal_bonus


def _choose_stable_reference_from_group(group: list[dict[str, Any]]) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    best_group = max(_stable_candidate_groups(group), key=_stable_group_score)
    reference_candidate = max(best_group, key=_candidate_quality_score)
    reference_radius = float(np.median([_candidate_radius(candidate) for candidate in best_group]))
    return reference_candidate, reference_radius, best_group


def _find_stable_daughter_reference(
    candidates: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[float], list[dict[str, Any]], bool]:
    strict = [
        candidate
        for candidate in candidates
        if _candidate_reference_eligible(candidate, relaxed=False, allow_zero=False)
    ]
    if strict:
        reference_candidate, reference_radius, reference_group = _choose_stable_reference_from_group(strict)
        return reference_candidate, reference_radius, reference_group, False

    relaxed = [
        candidate
        for candidate in candidates
        if _candidate_reference_eligible(candidate, relaxed=True, allow_zero=False)
    ]
    if relaxed:
        reference_candidate, reference_radius, reference_group = _choose_stable_reference_from_group(relaxed)
        return reference_candidate, reference_radius, reference_group, True

    zero_only = [
        candidate
        for candidate in candidates
        if _candidate_reference_eligible(candidate, relaxed=False, allow_zero=True)
        and _candidate_offset(candidate) <= ZERO_OFFSET_TOLERANCE_MM
    ]
    if zero_only:
        reference_candidate, reference_radius, reference_group = _choose_stable_reference_from_group(zero_only)
        return reference_candidate, reference_radius, reference_group, True

    return None, None, [], True


def _candidate_clean_rejection_reasons(
    candidate: dict[str, Any],
    reference_radius: float,
    reference_candidate: dict[str, Any],
    *,
    relaxed: bool,
) -> list[str]:
    if not _candidate_is_valid_surface_cut(candidate):
        return ["surface cut did not produce a valid compact component"]

    radius = _candidate_radius(candidate)
    radius_ratio = radius / max(reference_radius, EPS)
    parent_score = _safe_float(candidate.get("parent_contamination_score"), 1.0)
    reference_parent = _safe_float(reference_candidate.get("parent_contamination_score"), 1.0)
    compactness = _safe_float(candidate.get("compactness_score"), 0.0)
    reference_compactness = _safe_float(reference_candidate.get("compactness_score"), compactness)
    stability = _safe_float(candidate.get("stability_score"), 0.0)
    reference_stability = _safe_float(reference_candidate.get("stability_score"), stability)
    centroid_distance = _safe_float(candidate.get("centroid_distance_to_centerline_mm"), float("inf"))
    spread_ratio = _candidate_spread_ratio(candidate)
    reference_spread_ratio = _candidate_spread_ratio(reference_candidate)

    parent_limit = min(
        0.36 if relaxed else MAX_SUCCESS_PARENT_CONTAMINATION,
        max(MAX_STABLE_PARENT_CONTAMINATION, reference_parent + (0.12 if relaxed else 0.08)),
    )
    compactness_floor = max(0.64 if relaxed else 0.72, min(MIN_STABLE_COMPACTNESS, reference_compactness - (0.20 if relaxed else 0.12)))
    stability_floor = max(0.50 if relaxed else 0.62, min(MIN_STABLE_SCORE, reference_stability - (0.26 if relaxed else 0.18)))
    spread_limit = max(0.58 if not relaxed else 0.75, min(0.90, reference_spread_ratio + (0.35 if relaxed else 0.25)))
    centroid_limit = max(0.35 if relaxed else 0.25, 0.85 * reference_radius, 0.65 * radius)

    reasons: list[str] = []
    if parent_score >= MAX_REJECT_PARENT_CONTAMINATION:
        reasons.append("parent contamination score is high")
    elif parent_score > parent_limit:
        reasons.append("parent contamination is higher than the distal stable reference")
    if radius_ratio > MAX_CLEAN_RADIUS_RATIO:
        reasons.append("cut radius is much larger than the distal stable daughter reference")
    if radius_ratio < 0.45:
        reasons.append("cut radius is much smaller than the distal stable daughter reference")
    if compactness < compactness_floor:
        reasons.append("compactness is below the distal stable daughter reference")
    if stability < stability_floor:
        reasons.append("stability is below the distal stable daughter reference")
    if spread_ratio > spread_limit:
        reasons.append("radius spread is above the distal stable daughter reference")
    if centroid_distance > centroid_limit:
        reasons.append("cut component centroid drifts from the child centerline")
    return reasons


def _candidate_clean_against_reference(
    candidate: dict[str, Any],
    reference_radius: float,
    reference_candidate: dict[str, Any],
    *,
    relaxed: bool,
) -> bool:
    return not _candidate_clean_rejection_reasons(
        candidate,
        reference_radius,
        reference_candidate,
        relaxed=relaxed,
    )


def _best_candidate_by_offset(candidates: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    by_offset: dict[float, dict[str, Any]] = {}
    for candidate in candidates:
        offset = round(_candidate_offset(candidate), 6)
        previous = by_offset.get(offset)
        if previous is None or _candidate_quality_score(candidate) > _candidate_quality_score(previous):
            by_offset[offset] = candidate
    return by_offset


def _choose_fallback_surface_candidate(candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    valid = [candidate for candidate in candidates if _candidate_is_valid_surface_cut(candidate)]
    if not valid:
        return None
    nonzero = [candidate for candidate in valid if _candidate_offset(candidate) > ZERO_OFFSET_TOLERANCE_MM]
    if nonzero:
        best_nonzero = max(nonzero, key=_candidate_quality_score)
        best_any = max(valid, key=_candidate_quality_score)
        if _candidate_quality_score(best_nonzero) >= _candidate_quality_score(best_any) - 0.15:
            return best_nonzero
    return max(valid, key=_candidate_quality_score)


def _zero_offset_proof(
    candidates: list[dict[str, Any]],
    selected_candidate: dict[str, Any],
    reference_candidate: Optional[dict[str, Any]],
    reference_radius: Optional[float],
) -> tuple[bool, str]:
    if _candidate_offset(selected_candidate) > ZERO_OFFSET_TOLERANCE_MM:
        return False, ""
    if reference_candidate is None or reference_radius is None or reference_radius <= EPS:
        return False, "no distal stable daughter reference was available"
    if _candidate_offset(reference_candidate) <= ZERO_OFFSET_TOLERANCE_MM:
        return False, "stable daughter reference did not move beyond the topology origin"

    selected_radius = _candidate_radius(selected_candidate)
    selected_parent = _safe_float(selected_candidate.get("parent_contamination_score"), 1.0)
    selected_compactness = _safe_float(selected_candidate.get("compactness_score"), 0.0)
    reference_compactness = _safe_float(reference_candidate.get("compactness_score"), selected_compactness)
    selected_stability = _safe_float(selected_candidate.get("stability_score"), 0.0)
    reference_stability = _safe_float(reference_candidate.get("stability_score"), selected_stability)
    selected_centroid = _safe_float(selected_candidate.get("centroid_distance_to_centerline_mm"), float("inf"))

    if selected_parent > MAX_STABLE_PARENT_CONTAMINATION:
        return False, "zero-offset parent contamination exceeded the clean daughter threshold"
    radius_delta = abs(selected_radius - reference_radius) / max(reference_radius, EPS)
    if radius_delta > MAX_ZERO_OFFSET_RADIUS_RATIO_DELTA:
        return False, "zero-offset radius did not match the distal stable daughter reference"
    if selected_compactness < max(MIN_STABLE_COMPACTNESS, reference_compactness - 0.05):
        return False, "zero-offset compactness was not comparable to the distal stable reference"
    if selected_centroid > max(0.25, 0.50 * reference_radius):
        return False, "zero-offset cut centroid was not close enough to the child centerline"

    later_candidates = [
        candidate
        for candidate in candidates
        if _candidate_is_valid_surface_cut(candidate)
        and _candidate_offset(candidate) > ZERO_OFFSET_TOLERANCE_MM
    ]
    if any(
        _safe_float(candidate.get("parent_contamination_score"), 1.0) < selected_parent - 0.05
        for candidate in later_candidates
    ):
        return False, "a later candidate had substantially lower parent contamination"
    if any(
        _safe_float(candidate.get("stability_score"), 0.0) > selected_stability + 0.07
        for candidate in later_candidates
    ):
        return False, "a later candidate had substantially higher stability"
    if reference_stability > selected_stability + 0.05:
        return False, "distal stable reference was stronger than the topology-origin cut"
    return True, ""


def _classify_branch_start_candidates(
    candidates: list[dict[str, Any]],
    reference_candidate: Optional[dict[str, Any]],
    reference_radius: Optional[float],
    selected_candidate: Optional[dict[str, Any]],
    *,
    reference_relaxed: bool,
) -> None:
    selected_offset = _candidate_offset(selected_candidate) if selected_candidate is not None else float("inf")
    reference_offset = _candidate_offset(reference_candidate) if reference_candidate is not None else float("inf")
    for candidate in sorted(candidates, key=lambda item: (_candidate_offset(item), str(item.get("search_pass", "")))):
        if not _candidate_is_valid_surface_cut(candidate):
            candidate["accepted"] = False
            if not str(candidate.get("classification", "")).startswith("invalid_"):
                candidate["classification"] = "invalid_no_clean_component"
            candidate["rejection_reason"] = candidate.get("rejection_reason") or "no cut-based radius could be measured"
            continue

        if reference_candidate is None or reference_radius is None:
            candidate["accepted"] = False
            candidate["classification"] = "ambiguous_surface_cut_requires_review"
            candidate["rejection_reason"] = "no distal stable daughter reference could be established"
            continue

        reasons = _candidate_clean_rejection_reasons(
            candidate,
            reference_radius,
            reference_candidate,
            relaxed=reference_relaxed,
        )
        is_selected = candidate is selected_candidate
        if not reasons:
            offset = _candidate_offset(candidate)
            if offset < selected_offset - 1.0e-6:
                candidate["classification"] = "too_proximal_parent_contaminated"
                candidate["accepted"] = False
                candidate["rejection_reason"] = "candidate lies proximal to the selected last clean daughter boundary"
            elif offset <= selected_offset + 1.0e-6 or is_selected:
                candidate["classification"] = "stable_child_tube"
                candidate["accepted"] = True
                candidate["rejection_reason"] = ""
            else:
                candidate["classification"] = "too_distal_after_stable_section"
                candidate["accepted"] = False
                candidate["rejection_reason"] = "an earlier clean daughter-tube boundary was selected after reference-first refinement"
            continue

        candidate["accepted"] = False
        if _candidate_offset(candidate) <= max(selected_offset, reference_offset) + 1.0e-6:
            candidate["classification"] = "too_proximal_parent_contaminated"
        else:
            candidate["classification"] = "unstable_surface_cut_requires_review"
        candidate["rejection_reason"] = "; ".join(reasons)

    if (
        reference_candidate is not None
        and reference_radius is not None
        and selected_candidate is not None
        and _candidate_is_valid_surface_cut(selected_candidate)
    ):
        selected_candidate["classification"] = "stable_child_tube"
        selected_candidate["accepted"] = True
        selected_candidate["rejection_reason"] = ""


def _candidate_summary(
    candidates: list[dict[str, Any]],
    selected_candidate: Optional[dict[str, Any]],
    reference_candidate: Optional[dict[str, Any]],
    reference_radius: Optional[float],
    *,
    fallback_used: bool,
    zero_offset_proof_passed: bool,
    zero_offset_success_forbidden_reason: str,
    selected_reason: str,
) -> dict[str, Any]:
    stable_offsets = [
        _candidate_offset(candidate)
        for candidate in candidates
        if candidate.get("classification") == "stable_child_tube"
    ]
    valid_candidates = [candidate for candidate in candidates if _candidate_is_valid_surface_cut(candidate)]
    best_stability = max(valid_candidates, key=lambda item: _safe_float(item.get("stability_score"), 0.0), default=None)
    lowest_parent = min(
        valid_candidates,
        key=lambda item: _safe_float(item.get("parent_contamination_score"), 1.0),
        default=None,
    )
    selected_offset = _candidate_offset(selected_candidate) if selected_candidate is not None else 0.0
    return {
        "first_stable_offset_mm": _round_float(min(stable_offsets), 6) if stable_offsets else None,
        "selected_offset_mm": _round_float(selected_offset, 6),
        "reference_offset_mm": _round_float(_candidate_offset(reference_candidate), 6) if reference_candidate is not None else None,
        "reference_radius_mm": _round_float(reference_radius, 6),
        "best_stability_offset_mm": _round_float(_candidate_offset(best_stability), 6) if best_stability is not None else None,
        "lowest_contamination_offset_mm": _round_float(_candidate_offset(lowest_parent), 6) if lowest_parent is not None else None,
        "rejected_too_proximal_count": int(
            sum(1 for candidate in candidates if candidate.get("classification") == "too_proximal_parent_contaminated")
        ),
        "invalid_cut_count": int(
            sum(1 for candidate in candidates if str(candidate.get("classification", "")).startswith("invalid_"))
        ),
        "fallback_used": bool(fallback_used),
        "zero_offset_selected": bool(selected_offset <= ZERO_OFFSET_TOLERANCE_MM),
        "zero_offset_proof_passed": bool(zero_offset_proof_passed),
        "zero_offset_success_forbidden_reason": str(zero_offset_success_forbidden_reason),
        "selected_reason": str(selected_reason),
    }


def _surface_validated_branch_start_ring_v1(
    surface: vtk.vtkPolyData,
    surface_locator: vtk.vtkStaticPointLocator,
    segment: GeometrySegment,
) -> BranchStartSelection:
    branch_length = polyline_length(segment.points)
    search_offsets, search_max = _candidate_offsets(branch_length)
    candidates = [
        _surface_cut_candidate_metrics(surface, segment, offset, "forward_search")
        for offset in search_offsets
    ]

    reference_candidate, reference_radius, _, reference_relaxed = _find_stable_daughter_reference(candidates)
    backward_refinement_used = False
    selected_candidate: Optional[dict[str, Any]] = None
    selected_reason = ""
    fallback_used = False

    if reference_candidate is not None and reference_radius is not None:
        reference_offset = _candidate_offset(reference_candidate)
        if reference_offset > BACKWARD_REFINE_STEP_MM:
            existing_offsets = {round(_candidate_offset(candidate), 6) for candidate in candidates}
            backward_offsets: list[float] = []
            current = reference_offset - BACKWARD_REFINE_STEP_MM
            while current >= -1.0e-9:
                rounded = round(max(0.0, float(current)), 6)
                if rounded not in existing_offsets:
                    backward_offsets.append(rounded)
                    existing_offsets.add(rounded)
                current -= BACKWARD_REFINE_STEP_MM
            if backward_offsets:
                backward_refinement_used = True
                candidates.extend(
                    _surface_cut_candidate_metrics(surface, segment, offset, "backward_refinement")
                    for offset in backward_offsets
                )

        candidate_by_offset = _best_candidate_by_offset(candidates)
        selected_candidate = reference_candidate
        selected_reason = "distal stable daughter reference selected"
        for offset in sorted(
            [value for value in candidate_by_offset if value < reference_offset - 1.0e-6],
            reverse=True,
        ):
            candidate = candidate_by_offset[offset]
            if _candidate_clean_against_reference(
                candidate,
                reference_radius,
                reference_candidate,
                relaxed=reference_relaxed,
            ):
                selected_candidate = candidate
                selected_reason = "most proximal clean candidate before parent contamination"
                continue
            selected_reason = "last clean candidate before proximal parent contamination"
            break

    if selected_candidate is None:
        fallback_used = True
        selected_candidate = _choose_fallback_surface_candidate(candidates)
        reference_candidate = None
        reference_radius = None
        selected_reason = "fallback surface candidate requires review" if selected_candidate is not None else "topology fallback requires review"

    _classify_branch_start_candidates(
        candidates,
        reference_candidate,
        reference_radius,
        selected_candidate,
        reference_relaxed=reference_relaxed,
    )

    if selected_candidate is None:
        sample_center = _segment_point_after(segment, min(1.0, max(0.0, branch_length * 0.20)))
        radius, _, _, confidence = _estimate_radius_from_surface(surface, surface_locator, sample_center)
        zero_forbidden_reason = "no distal stable daughter reference was available"
        metadata = {
            "selection_algorithm": BRANCH_START_RING_ALGORITHM,
            "topology_start_xyz": [float(value) for value in segment.points[0]],
            "selected_offset_mm": 0.0,
            "search_max_mm": float(search_max),
            "candidate_count": int(len(candidates)),
            "accepted_candidate_count": 0,
            "selected_candidate_classification": "topology_fallback_requires_review",
            "selected_radius_rule": "estimated_fallback_requires_review",
            "surface_cut_used": False,
            "backward_refinement_used": bool(backward_refinement_used),
            "cells_reassigned_to_parent_count": 0,
            "surface_assignment_adjusted": False,
            "candidate_summary": _candidate_summary(
                candidates,
                None,
                reference_candidate,
                reference_radius,
                fallback_used=True,
                zero_offset_proof_passed=False,
                zero_offset_success_forbidden_reason=zero_forbidden_reason,
                selected_reason=selected_reason,
            ),
        }
        return BranchStartSelection(
            center=np.asarray(segment.points[0], dtype=float),
            normal=_normal_at_segment_s(segment, 0.0),
            radius_mm=float(radius),
            source_s_mm=0.0,
            radius_rule="estimated_fallback_requires_review",
            confidence=min(float(confidence), 0.55),
            status=STATUS_REQUIRES_REVIEW,
            warnings=[
                "branch_start ring used topology fallback because no stable surface-cut candidate was accepted"
            ],
            metadata=metadata,
        )

    candidates_sorted = sorted(candidates, key=lambda item: (float(item.get("offset_mm", 0.0)), str(item.get("search_pass", ""))))
    selected_offset = _candidate_offset(selected_candidate)
    selected_radius = _candidate_radius(selected_candidate)
    parent_contamination_detected = any(
        candidate.get("classification") == "too_proximal_parent_contaminated"
        for candidate in candidates
        if _candidate_offset(candidate) <= selected_offset + BRANCH_RING_SEARCH_STEP_MM
    )
    selected_stability = _safe_float(selected_candidate.get("stability_score"), 0.0)
    selected_compactness = _safe_float(selected_candidate.get("compactness_score"), 0.0)
    selected_parent = _safe_float(selected_candidate.get("parent_contamination_score"), 1.0)
    zero_proof_passed, zero_forbidden_reason = _zero_offset_proof(
        candidates,
        selected_candidate,
        reference_candidate,
        reference_radius,
    )
    confidence = 0.70 + 0.12 * min(1.0, selected_stability) + 0.10 * min(1.0, selected_compactness)
    confidence -= 0.20 * min(1.0, selected_parent)
    if reference_relaxed:
        confidence = min(confidence, 0.76)
    if not backward_refinement_used:
        confidence = min(confidence, 0.78)
    if parent_contamination_detected:
        confidence = min(confidence, 0.84)
    if fallback_used:
        confidence = min(confidence, 0.58 if selected_offset <= ZERO_OFFSET_TOLERANCE_MM else 0.62)
    if selected_parent >= MAX_SUCCESS_PARENT_CONTAMINATION:
        confidence = min(confidence, 0.64)
    if selected_offset <= ZERO_OFFSET_TOLERANCE_MM and not zero_proof_passed:
        confidence = min(confidence, 0.62)
    elif selected_offset <= ZERO_OFFSET_TOLERANCE_MM:
        confidence = min(confidence, 0.86)

    status = STATUS_SUCCESS
    if fallback_used or reference_relaxed or selected_parent >= MAX_SUCCESS_PARENT_CONTAMINATION:
        status = STATUS_REQUIRES_REVIEW
    if selected_offset <= ZERO_OFFSET_TOLERANCE_MM and not zero_proof_passed:
        status = STATUS_REQUIRES_REVIEW
    if selected_candidate is reference_candidate and selected_offset > max(0.5, BRANCH_RING_SEARCH_STEP_MM):
        status = STATUS_REQUIRES_REVIEW
        confidence = min(confidence, 0.72)

    warnings: list[str] = []
    if fallback_used:
        warnings.append("branch_start used the best available surface-cut candidate because no stable daughter reference was proven")
    if selected_offset <= ZERO_OFFSET_TOLERANCE_MM and not zero_proof_passed:
        warnings.append("zero-offset branch_start was not accepted as proven clean daughter boundary")
    if parent_contamination_detected:
        warnings.append("proximal parent-wall contamination was detected and rejected before selecting this branch_start ring")
    if reference_relaxed:
        warnings.append("branch_start stable daughter reference was weak and requires visual review")
    if selected_parent >= MAX_SUCCESS_PARENT_CONTAMINATION:
        warnings.append("selected branch_start candidate has parent contamination above the success threshold")
    if selected_candidate is reference_candidate and selected_offset > max(0.5, BRANCH_RING_SEARCH_STEP_MM):
        warnings.append("backward refinement stopped at the distal stable reference because proximal candidates were contaminated")

    if not zero_forbidden_reason and selected_offset <= ZERO_OFFSET_TOLERANCE_MM and not zero_proof_passed:
        zero_forbidden_reason = "zero-offset branch_start was not accepted as proven clean daughter boundary"

    selected_radius_rule = "surface_cut_fallback_requires_review" if fallback_used else "stable_child_tube_equivalent_radius"

    metadata = {
        "selection_algorithm": BRANCH_START_RING_ALGORITHM,
        "topology_start_xyz": [float(value) for value in segment.points[0]],
        "selected_offset_mm": float(selected_offset),
        "search_max_mm": float(search_max),
        "candidate_count": int(len(candidates)),
        "accepted_candidate_count": int(sum(1 for candidate in candidates if bool(candidate.get("accepted")))),
        "selected_candidate_classification": str(selected_candidate.get("classification", "")),
        "selected_radius_rule": selected_radius_rule,
        "surface_cut_used": True,
        "backward_refinement_used": bool(backward_refinement_used),
        "cells_reassigned_to_parent_count": 0,
        "surface_assignment_adjusted": False,
        "candidate_summary": _candidate_summary(
            candidates_sorted,
            selected_candidate,
            reference_candidate,
            reference_radius,
            fallback_used=fallback_used,
            zero_offset_proof_passed=zero_proof_passed,
            zero_offset_success_forbidden_reason=zero_forbidden_reason,
            selected_reason=selected_reason,
        ),
    }

    return BranchStartSelection(
        center=np.asarray(selected_candidate["_center_np"], dtype=float),
        normal=np.asarray(selected_candidate["_normal_np"], dtype=float),
        radius_mm=selected_radius,
        source_s_mm=selected_offset,
        radius_rule=selected_radius_rule,
        confidence=float(np.clip(confidence, 0.45, 0.95)),
        status=status,
        warnings=warnings,
        metadata=metadata,
    )


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
        metadata: Optional[dict[str, Any]] = None,
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
                metadata=dict(metadata or {}),
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

    branch_start_selection_by_segment_id: dict[int, BranchStartSelection] = {}

    for segment in segments:
        if segment.segment_type != "branch":
            continue
        selection = _surface_validated_branch_start_ring_v1(surface, surface_locator, segment)
        branch_start_selection_by_segment_id[int(segment.segment_id)] = selection
        segment.proximal_ring_id = add_ring(
            "branch_start",
            selection.center,
            selection.normal,
            selection.radius_mm,
            segment.segment_id,
            segment.parent_segment_id,
            segment.segment_id,
            selection.source_s_mm,
            "perpendicular_to_child_centerline_tangent",
            selection.radius_rule,
            selection.confidence,
            selection.status,
            selection.warnings,
            selection.metadata,
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
            child_selection = branch_start_selection_by_segment_id.get(int(child_segment.segment_id))
            if child_selection is None:
                child_selection = _surface_validated_branch_start_ring_v1(surface, surface_locator, child_segment)
                branch_start_selection_by_segment_id[int(child_segment.segment_id)] = child_selection
            daughter_metadata = dict(child_selection.metadata)
            daughter_metadata["selection_algorithm"] = BRANCH_START_RING_ALGORITHM
            daughter_metadata["daughter_start_reused_branch_start_refinement"] = True
            daughter_ring_ids.append(
                add_ring(
                    "daughter_start",
                    child_selection.center,
                    child_selection.normal,
                    child_selection.radius_mm,
                    child_segment.segment_id,
                    parent_segment_id,
                    child_segment.segment_id,
                    child_selection.source_s_mm,
                    "perpendicular_to_daughter_centerline_tangent",
                    child_selection.radius_rule,
                    min(child_selection.confidence, 0.70),
                    STATUS_REQUIRES_REVIEW if child_selection.status != STATUS_FAILED else STATUS_FAILED,
                    [*child_selection.warnings, "daughter_start mirrors branch_start surface-cut refinement and requires bifurcation review"],
                    daughter_metadata,
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
                warnings=["parent_pre_bifurcation is first-pass; daughter_start rings reuse branch_start surface-cut refinement"],
            )
        )

    if not rings:
        raise GeometrySegmentationFailure("No boundary rings could be generated.")

    return append_polydata([ring.polydata for ring in rings]), rings, bifurcations, warnings


def _set_uniform_numeric_cell_array(polydata: vtk.vtkPolyData, name: str, value: float | int) -> None:
    arr = polydata.GetCellData().GetArray(name)
    if arr is None:
        if isinstance(value, int):
            add_int_cell_array(polydata, name, [int(value)] * polydata.GetNumberOfCells())
        else:
            add_float_cell_array(polydata, name, [float(value)] * polydata.GetNumberOfCells())
        return
    for idx in range(polydata.GetNumberOfCells()):
        arr.SetValue(idx, value)


def _set_uniform_string_cell_array(polydata: vtk.vtkPolyData, name: str, value: str) -> None:
    arr = polydata.GetCellData().GetAbstractArray(name)
    if arr is None:
        add_string_cell_array(polydata, name, [str(value)] * polydata.GetNumberOfCells())
        return
    for idx in range(polydata.GetNumberOfCells()):
        arr.SetValue(idx, str(value))


def _sync_ring_polydata_contract_arrays(ring: BoundaryRing) -> None:
    _set_uniform_numeric_cell_array(ring.polydata, "RingId", int(ring.ring_id))
    _set_uniform_string_cell_array(ring.polydata, "RingLabel", ring.ring_label)
    _set_uniform_string_cell_array(ring.polydata, "RingType", ring.ring_type)
    _set_uniform_numeric_cell_array(ring.polydata, "ParentSegmentId", int(ring.parent_segment_id or 0))
    _set_uniform_numeric_cell_array(ring.polydata, "ChildSegmentId", int(ring.child_segment_id or 0))
    _set_uniform_numeric_cell_array(ring.polydata, "SegmentId", int(ring.source_segment_id))
    _set_uniform_numeric_cell_array(ring.polydata, "RadiusMm", float(ring.radius_mm))
    _set_uniform_numeric_cell_array(ring.polydata, "Confidence", float(ring.confidence))
    _set_uniform_string_cell_array(ring.polydata, "Status", ring.status)


def _assign_surface_segments(
    surface: vtk.vtkPolyData,
    segments: list[GeometrySegment],
    roles: InputRoles,
    rings: Optional[list[BoundaryRing]] = None,
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

    branch_start_rings_by_segment: dict[int, BoundaryRing] = {}
    for ring in rings or []:
        if ring.ring_type != "branch_start":
            continue
        child_id = int(ring.child_segment_id or ring.source_segment_id)
        if child_id in segment_lookup:
            branch_start_rings_by_segment[child_id] = ring

    assignment_stats: dict[int, dict[str, int]] = {
        int(segment_id): {"reassigned": 0, "retained": 0}
        for segment_id in branch_start_rings_by_segment
    }
    if branch_start_rings_by_segment:
        for cell_id, center in enumerate(centers):
            assigned_segment_id = int(assigned_segment_ids[cell_id])
            ring = branch_start_rings_by_segment.get(assigned_segment_id)
            if ring is None:
                continue
            segment = segment_lookup.get(assigned_segment_id)
            if segment is None or segment.segment_type != "branch":
                continue
            selected_s = float(max(0.0, ring.source_centerline_s_mm))
            if selected_s <= EPS:
                assignment_stats[assigned_segment_id]["retained"] += 1
                continue

            projected_s, _ = _project_point_to_polyline_s(segment.points, center)
            ring_center = np.asarray(ring.center_xyz, dtype=float)
            ring_normal = unit(np.asarray(ring.normal_xyz, dtype=float))
            signed_plane_distance = float(np.dot(np.asarray(center, dtype=float) - ring_center, ring_normal))
            near_ring_tolerance = max(0.25, 0.25 * float(ring.radius_mm))
            before_selected_s = projected_s < selected_s - 0.05
            near_and_proximal = projected_s < selected_s + near_ring_tolerance and signed_plane_distance < -0.05
            if before_selected_s or near_and_proximal:
                parent_segment_id = int(segment.parent_segment_id or 1)
                assigned_segment_ids[cell_id] = parent_segment_id if parent_segment_id in segment_lookup else 1
                assignment_stats[assigned_segment_id]["reassigned"] += 1
            else:
                assignment_stats[assigned_segment_id]["retained"] += 1

        for ring in branch_start_rings_by_segment.values():
            child_id = int(ring.child_segment_id or ring.source_segment_id)
            stats = assignment_stats.get(child_id, {"reassigned": 0, "retained": 0})
            ring.metadata["cells_reassigned_to_parent_count"] = int(stats["reassigned"])
            ring.metadata["surface_assignment_adjusted"] = bool(stats["reassigned"] > 0)
            summary = ring.metadata.get("candidate_summary")
            if isinstance(summary, dict):
                summary["surface_assignment_adjusted"] = bool(stats["reassigned"] > 0)
                summary["cells_reassigned_to_parent_count"] = int(stats["reassigned"])

            selected_s = float(max(0.0, ring.source_centerline_s_mm))
            zero_proof_passed = bool(summary.get("zero_offset_proof_passed")) if isinstance(summary, dict) else False
            if selected_s <= ZERO_OFFSET_TOLERANCE_MM and not zero_proof_passed:
                ring.status = STATUS_REQUIRES_REVIEW
                ring.confidence = min(float(ring.confidence), 0.62)
                _append_unique(
                    ring.warnings,
                    "zero-offset branch_start was not accepted as proven clean daughter boundary",
                )
            elif selected_s > ZERO_OFFSET_TOLERANCE_MM and stats["reassigned"] == 0:
                ring.status = STATUS_REQUIRES_REVIEW
                ring.confidence = min(float(ring.confidence), 0.68)
                _append_unique(
                    ring.warnings,
                    "selected branch_start moved distally but no proximal child cells were reassigned",
                )

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

    if branch_start_rings_by_segment:
        corrected_total = sum(int(stats["reassigned"]) for stats in assignment_stats.values())
        warnings.append(
            "surface cells were assigned by nearest centerline segment and then corrected using selected branch_start ring offsets"
        )
        if corrected_total:
            warnings.append(f"{int(corrected_total)} proximal branch cell(s) were reassigned to parent segments")
    else:
        warnings.append("surface cells were assigned by nearest centerline segment; cut-boundary consistency requires review")
    return out, int(unassigned_cell_count), warnings


def _ring_to_json(ring: BoundaryRing) -> dict[str, Any]:
    data = {
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
    for key, value in ring.metadata.items():
        if key not in data:
            data[key] = value
    return data


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


def _branch_start_refinement_diagnostics(result_data: dict[str, Any]) -> dict[str, Any]:
    rings_json = result_data.get("boundary_rings", [])
    if not isinstance(rings_json, list):
        rings_json = []
    branch_start_rings = [
        ring for ring in rings_json if isinstance(ring, dict) and ring.get("ring_type") == "branch_start"
    ]
    segments_json = result_data.get("segments", [])
    if not isinstance(segments_json, list):
        segments_json = []
    segment_labels_by_id = {
        int(_safe_float(segment.get("segment_id"), 0.0)): str(segment.get("segment_label", ""))
        for segment in segments_json
        if isinstance(segment, dict)
    }

    per_ring_summary: list[dict[str, Any]] = []
    refined_ring_count = 0
    topology_fallback_ring_count = 0
    stable_candidate_found_count = 0
    parent_contamination_detected_count = 0
    cells_reassigned_total = 0
    rings_requiring_review: list[int] = []
    zero_offset_selected_count = 0
    zero_offset_success_count = 0
    zero_offset_requires_review_count = 0
    rings_moved_distally_count = 0
    rings_with_surface_assignment_adjusted_count = 0

    for ring in branch_start_rings:
        candidate_summary = ring.get("candidate_summary", {})
        if not isinstance(candidate_summary, dict):
            candidate_summary = {}
        selected_offset = _safe_float(ring.get("selected_offset_mm", ring.get("source_centerline_s_mm")), 0.0)
        zero_offset_selected = bool(selected_offset <= ZERO_OFFSET_TOLERANCE_MM)
        zero_offset_proof_passed = bool(candidate_summary.get("zero_offset_proof_passed", False))
        fallback_used = bool(candidate_summary.get("fallback_used", False))
        stable_found = candidate_summary.get("reference_offset_mm") is not None or candidate_summary.get("first_stable_offset_mm") is not None
        parent_contamination_found = int(_safe_float(candidate_summary.get("rejected_too_proximal_count"), 0.0)) > 0
        parent_contamination_found = parent_contamination_found or any(
            "contamination" in str(warning).lower() for warning in ring.get("warnings", []) if isinstance(warning, str)
        )
        if ring.get("surface_cut_used") is True:
            refined_ring_count += 1
        if ring.get("selected_candidate_classification") == "topology_fallback_requires_review" or fallback_used:
            topology_fallback_ring_count += 1
        if stable_found:
            stable_candidate_found_count += 1
        if parent_contamination_found:
            parent_contamination_detected_count += 1
        if ring.get("status") == STATUS_REQUIRES_REVIEW:
            try:
                rings_requiring_review.append(int(ring.get("ring_id", 0)))
            except Exception:
                rings_requiring_review.append(0)

        segment_id = int(_safe_float(ring.get("child_segment_id") or ring.get("source_segment_id"), 0.0))
        cells_reassigned = int(_safe_float(ring.get("cells_reassigned_to_parent_count"), 0.0))
        cells_reassigned_total += cells_reassigned
        if zero_offset_selected:
            zero_offset_selected_count += 1
            if ring.get("status") == STATUS_SUCCESS:
                zero_offset_success_count += 1
            if ring.get("status") == STATUS_REQUIRES_REVIEW:
                zero_offset_requires_review_count += 1
        if selected_offset > ZERO_OFFSET_TOLERANCE_MM:
            rings_moved_distally_count += 1
        if bool(ring.get("surface_assignment_adjusted", False)):
            rings_with_surface_assignment_adjusted_count += 1
        warnings = ring.get("warnings", [])
        warning_count = len(warnings) if isinstance(warnings, list) else 0
        per_ring_summary.append(
            {
                "ring_id": int(_safe_float(ring.get("ring_id"), 0.0)),
                "segment_id": segment_id,
                "segment_label": segment_labels_by_id.get(segment_id, ""),
                "selected_offset_mm": ring.get("selected_offset_mm"),
                "status": ring.get("status"),
                "classification": ring.get("selected_candidate_classification"),
                "confidence": ring.get("confidence"),
                "candidate_count": int(_safe_float(ring.get("candidate_count"), 0.0)),
                "warning_count": int(warning_count),
                "cells_reassigned_to_parent_count": int(cells_reassigned),
                "first_stable_offset_mm": candidate_summary.get("first_stable_offset_mm"),
                "reference_offset_mm": candidate_summary.get("reference_offset_mm"),
                "zero_offset_selected": bool(zero_offset_selected),
                "zero_offset_proof_passed": bool(zero_offset_proof_passed),
                "fallback_used": bool(fallback_used),
            }
        )

    return {
        "algorithm": BRANCH_START_RING_ALGORITHM,
        "branch_count": int(len(branch_start_rings)),
        "refined_ring_count": int(refined_ring_count),
        "topology_fallback_ring_count": int(topology_fallback_ring_count),
        "stable_candidate_found_count": int(stable_candidate_found_count),
        "parent_contamination_detected_count": int(parent_contamination_detected_count),
        "cells_reassigned_to_parent_total": int(cells_reassigned_total),
        "zero_offset_selected_count": int(zero_offset_selected_count),
        "zero_offset_success_count": int(zero_offset_success_count),
        "zero_offset_requires_review_count": int(zero_offset_requires_review_count),
        "rings_moved_distally_count": int(rings_moved_distally_count),
        "rings_with_surface_assignment_adjusted_count": int(rings_with_surface_assignment_adjusted_count),
        "rings_requiring_review": rings_requiring_review,
        "per_ring_summary": per_ring_summary,
    }


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
        "outputs_exist": outputs_exist,
        "vtp_arrays": {
            "segmented_surface_cell_arrays": segmented_surface_arrays,
            "boundary_rings_cell_arrays": boundary_ring_arrays,
            "missing_segmented_surface_arrays": missing_segmented_surface_arrays,
            "missing_boundary_ring_arrays": missing_boundary_ring_arrays,
        },
        "labels": {
            "segment_labels": sorted(set([*segment_labels_from_vtp, *json_segment_labels])),
            "ring_labels": sorted(set([*ring_labels_from_vtp, *json_ring_labels])),
            "bifurcation_labels": json_bifurcation_labels,
            "forbidden_labels_found": forbidden_labels_found,
        },
        "branch_start_refinement": _branch_start_refinement_diagnostics(result_data),
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
    boundary_rings, rings, bifurcations, ring_warnings = _build_boundary_rings(
        roles, surface, terminations, segments, segment_start_node_to_ids, node_coords
    )
    segmented_surface, unassigned_cell_count, assignment_warnings = _assign_surface_segments(
        surface, segments, roles, rings
    )
    for ring in rings:
        _sync_ring_polydata_contract_arrays(ring)
    boundary_rings = append_polydata([ring.polydata for ring in rings])

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
