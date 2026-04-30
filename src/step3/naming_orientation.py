from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import vtk

from src.common.geometry import (
    concatenate_polylines,
    cumulative_arclength,
    distance,
    equivalent_diameter_from_area,
    polygon_area_normal,
    projected_major_minor_diameters,
    unit,
)
from src.common.json_io import read_json, write_json
from src.common.paths import build_pipeline_paths
from src.common.vtk_helpers import (
    add_int_cell_array,
    add_uchar3_cell_array,
    clone_geometry_only,
    get_cell_array,
    points_to_numpy,
    read_vtp,
    segment_color,
    write_vtp,
)


PRIORITY_VESSELS = (
    "abdominal_aorta_trunk",
    "left_renal_artery",
    "right_renal_artery",
    "superior_mesenteric_artery",
    "inferior_mesenteric_artery",
    "celiac_artery",
    "left_common_iliac",
    "right_common_iliac",
    "left_external_iliac",
    "right_external_iliac",
    "left_internal_iliac",
    "right_internal_iliac",
)

PRIORITY_REVIEW_THRESHOLD = 0.75


class Step3Failure(RuntimeError):
    pass


@dataclass
class NetworkEdge:
    edge_id: int
    start_node: int
    end_node: int
    points: np.ndarray


@dataclass
class SegmentRecord:
    segment_id: int
    name_hint: str
    segment_type: str
    parent_segment_id: Optional[int]
    child_segment_ids: list[int]
    proximal_node_id: int
    distal_node_id: int
    proximal_point: np.ndarray
    distal_point: np.ndarray
    edge_ids: list[int]
    length: float
    terminal_face_id: Optional[int]
    terminal_face_name: Optional[str]
    descendant_terminal_names: list[str]
    cell_count: int
    proximal_boundary: Optional[Dict[str, Any]] = None
    proximal_boundary_source: Optional[str] = None
    proximal_boundary_confidence: Optional[float] = None
    proximal_boundary_arclength: Optional[float] = None


@dataclass
class NamedSegment:
    record: SegmentRecord
    segment_name: str
    naming_source: str
    naming_confidence: float
    inferred: bool
    face_map_support: list[str]
    centerline_points: np.ndarray
    centerline_mode: str


@dataclass
class Step2ContractView:
    contract_mode: str
    schema_version: Any
    step_status: str
    warnings: list[str]
    segment_rows: list[Dict[str, Any]]
    aorta_start: Dict[str, Any]
    aorta_end: Dict[str, Any]
    mapped_terminal_boundaries: Dict[int, Dict[str, Any]]
    upstream_references: Dict[str, Any]
    coordinate_system: Dict[str, Any]
    units: Any


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def _dedupe_warnings(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _face_map_by_id(raw: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            face_id = int(key)
        except Exception:
            continue
        row = dict(value) if isinstance(value, dict) else {"name": str(value)}
        row["face_id"] = face_id
        out[face_id] = row
    return out


def _normalize_step2_contract(raw: Dict[str, Any]) -> Step2ContractView:
    boundary_summary = raw.get("boundary_summary") or {}
    mapped_terminal = boundary_summary.get("mapped_terminal_boundaries") or {}
    mapped_terminal_by_face: Dict[int, Dict[str, Any]] = {}
    for row in mapped_terminal if isinstance(mapped_terminal, list) else []:
        try:
            mapped_terminal_by_face[int(row.get("face_id"))] = dict(row)
        except Exception:
            continue

    if "step_name" not in raw:
        contract_mode = "legacy"
        step_status = str(raw.get("final_status") or raw.get("status") or "failed")
        warnings = [str(v) for v in raw.get("warnings", [])]
        segment_rows = list(raw.get("segment_summary", []))
        aorta_start = dict(raw.get("aorta_start") or boundary_summary.get("aorta_start") or {})
        aorta_end = dict(raw.get("aorta_end") or boundary_summary.get("aorta_end_pre_bifurcation") or {})
        upstream_references = dict(raw.get("upstream_step1_references") or {})
        coordinate_system = dict(raw.get("coordinate_system") or {})
        units = coordinate_system.get("units")
    else:
        contract_mode = "canonical"
        if raw.get("step_name") != "step2_geometry_contract":
            raise Step3Failure(f"STEP2 contract has unexpected step_name: {raw.get('step_name')}")
        step_status = str(raw.get("step_status", "failed"))
        warnings = [str(v) for v in raw.get("warnings", [])]
        segment_rows = list(raw.get("segment_summary", []))
        aorta_start = dict(raw.get("aorta_start") or {})
        aorta_end = dict(raw.get("aorta_end") or {})
        upstream_references = dict(raw.get("upstream_references") or {})
        coordinate_system = dict(raw.get("coordinate_system") or {})
        units = raw.get("units")

    if not segment_rows:
        raise Step3Failure("STEP2 contract is missing segment_summary; STEP3 cannot name segments.")
    if not aorta_start:
        raise Step3Failure("STEP2 contract is missing aorta_start metadata.")
    if not aorta_end:
        raise Step3Failure("STEP2 contract is missing aorta_end metadata.")

    return Step2ContractView(
        contract_mode=contract_mode,
        schema_version=raw.get("schema_version"),
        step_status=step_status,
        warnings=warnings,
        segment_rows=segment_rows,
        aorta_start=aorta_start,
        aorta_end=aorta_end,
        mapped_terminal_boundaries=mapped_terminal_by_face,
        upstream_references=upstream_references,
        coordinate_system=coordinate_system,
        units=units,
    )


def _segment_records(rows: list[Dict[str, Any]]) -> list[SegmentRecord]:
    records: list[SegmentRecord] = []
    for row in rows:
        segment_id = int(row["segment_id"])
        proximal_boundary_confidence = row.get("proximal_boundary_confidence")
        proximal_boundary_arclength = row.get("proximal_boundary_arclength")
        records.append(
            SegmentRecord(
                segment_id=segment_id,
                name_hint=str(row.get("name_hint", "")),
                segment_type=str(row.get("segment_type", "")),
                parent_segment_id=int(row["parent_segment_id"]) if row.get("parent_segment_id") is not None else None,
                child_segment_ids=[int(v) for v in row.get("child_segment_ids", [])],
                proximal_node_id=int(row["proximal_node_id"]),
                distal_node_id=int(row["distal_node_id"]),
                proximal_point=np.asarray(row.get("proximal_point", [0.0, 0.0, 0.0]), dtype=float),
                distal_point=np.asarray(row.get("distal_point", [0.0, 0.0, 0.0]), dtype=float),
                edge_ids=[int(v) for v in row.get("edge_ids", [])],
                length=float(row.get("length", 0.0)),
                terminal_face_id=int(row["terminal_face_id"]) if row.get("terminal_face_id") is not None else None,
                terminal_face_name=str(row["terminal_face_name"]) if row.get("terminal_face_name") is not None else None,
                descendant_terminal_names=[str(v) for v in row.get("descendant_terminal_names", [])],
                cell_count=int(row.get("cell_count", 0)),
                proximal_boundary=dict(row["proximal_boundary"]) if isinstance(row.get("proximal_boundary"), dict) else None,
                proximal_boundary_source=str(row["proximal_boundary_source"]) if row.get("proximal_boundary_source") is not None else None,
                proximal_boundary_confidence=float(proximal_boundary_confidence) if proximal_boundary_confidence is not None else None,
                proximal_boundary_arclength=float(proximal_boundary_arclength) if proximal_boundary_arclength is not None else None,
            )
        )
    return sorted(records, key=lambda row: row.segment_id)


def _read_network_edges(network_pd: vtk.vtkPolyData) -> dict[int, NetworkEdge]:
    cd = network_pd.GetCellData()
    start_arr = cd.GetArray("StartNodeId")
    end_arr = cd.GetArray("EndNodeId")
    edge_arr = cd.GetArray("EdgeId")
    if start_arr is None or end_arr is None:
        raise Step3Failure("STEP1 centerline_network.vtp is missing StartNodeId/EndNodeId arrays.")

    edges: dict[int, NetworkEdge] = {}
    for cell_id in range(network_pd.GetNumberOfCells()):
        cell = network_pd.GetCell(cell_id)
        if cell is None or cell.GetNumberOfPoints() < 2:
            continue
        ids = cell.GetPointIds()
        pts = np.asarray([network_pd.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
        edge_id = int(edge_arr.GetTuple1(cell_id)) if edge_arr is not None else int(cell_id)
        edges[edge_id] = NetworkEdge(
            edge_id=edge_id,
            start_node=int(start_arr.GetTuple1(cell_id)),
            end_node=int(end_arr.GetTuple1(cell_id)),
            points=pts,
        )
    if not edges:
        raise Step3Failure("STEP1 centerline_network.vtp did not contain usable polyline edges.")
    return edges


def _find_edge_walk(
    adjacency: dict[int, list[tuple[int, int]]],
    current_node: int,
    target_node: int,
    remaining_edges: frozenset[int],
) -> Optional[list[tuple[int, int, int]]]:
    if not remaining_edges:
        return [] if int(current_node) == int(target_node) else None
    for neighbor, edge_id in adjacency.get(int(current_node), []):
        if edge_id not in remaining_edges:
            continue
        sub = _find_edge_walk(adjacency, neighbor, target_node, remaining_edges - {edge_id})
        if sub is not None:
            return [(int(current_node), int(neighbor), int(edge_id))] + sub
    return None


def _polyline_from_edges(record: SegmentRecord, edges: dict[int, NetworkEdge]) -> np.ndarray:
    if not record.edge_ids:
        raise Step3Failure(f"STEP2 segment {record.segment_id} is missing edge_ids.")
    missing = [edge_id for edge_id in record.edge_ids if edge_id not in edges]
    if missing:
        raise Step3Failure(
            f"STEP2 segment {record.segment_id} references missing centerline edge(s): {', '.join(str(v) for v in missing)}"
        )

    adjacency: dict[int, list[tuple[int, int]]] = {}
    for edge_id in record.edge_ids:
        edge = edges[edge_id]
        adjacency.setdefault(edge.start_node, []).append((edge.end_node, edge_id))
        adjacency.setdefault(edge.end_node, []).append((edge.start_node, edge_id))

    walk = _find_edge_walk(adjacency, record.proximal_node_id, record.distal_node_id, frozenset(record.edge_ids))
    if walk is None:
        raise Step3Failure(
            f"STEP2 segment {record.segment_id} edge_ids could not be ordered from node "
            f"{record.proximal_node_id} to node {record.distal_node_id}."
        )

    parts: list[np.ndarray] = []
    for start_node, end_node, edge_id in walk:
        edge = edges[edge_id]
        if edge.start_node == int(start_node) and edge.end_node == int(end_node):
            parts.append(edge.points)
        elif edge.start_node == int(end_node) and edge.end_node == int(start_node):
            parts.append(edge.points[::-1])
        else:
            raise Step3Failure(
                f"Centerline edge {edge_id} does not connect expected nodes {start_node} and {end_node}."
            )
    polyline = concatenate_polylines(parts)
    if polyline.shape[0] < 2:
        raise Step3Failure(f"STEP2 segment {record.segment_id} reconstructed to fewer than 2 centerline points.")
    return polyline


def _straight_polyline(record: SegmentRecord) -> np.ndarray:
    pts = np.vstack([record.proximal_point.reshape(1, 3), record.distal_point.reshape(1, 3)])
    if distance(pts[0], pts[1]) <= 1.0e-9:
        raise Step3Failure(f"STEP2 segment {record.segment_id} has coincident proximal/distal points.")
    return pts


def _extract_single_polyline(polydata: vtk.vtkPolyData) -> np.ndarray:
    if polydata.GetNumberOfCells() <= 0:
        pts = points_to_numpy(polydata)
        if pts.shape[0] >= 2:
            return pts
        raise Step3Failure("STEP2 aorta_centerline.vtp did not contain a usable polyline.")
    cell = polydata.GetCell(0)
    ids = cell.GetPointIds()
    points = np.asarray([polydata.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
    if points.shape[0] < 2:
        raise Step3Failure("STEP2 aorta_centerline.vtp did not contain a usable polyline.")
    return points


def _priority_classification(segment_name: str) -> str:
    return "high_priority" if segment_name in PRIORITY_VESSELS else "supporting"


def _named_segment(
    record: SegmentRecord,
    *,
    face_map_names: dict[int, str],
    by_id: dict[int, SegmentRecord],
) -> tuple[str, str, float, bool, list[str]]:
    descendants = set(record.descendant_terminal_names)
    support_names = sorted(descendants)

    if record.segment_type == "aorta_trunk":
        return "abdominal_aorta_trunk", "step2_aorta_trunk", 0.98, True, ["abdominal_aorta_inlet"]

    if record.terminal_face_name == "celiac_branch":
        return "celiac_branch", "direct_face_map_supporting_branch", 0.9, False, ["celiac_branch"]

    if record.terminal_face_name == "celiac_artery":
        parent = by_id.get(record.parent_segment_id) if record.parent_segment_id is not None else None
        if parent is not None and ("celiac" in parent.name_hint or all("celiac" in name for name in parent.descendant_terminal_names)):
            return "celiac_artery_distal", "direct_face_map_supporting_branch", 0.9, False, ["celiac_artery"]
        return "celiac_artery", "direct_face_map", 0.97, False, ["celiac_artery"]

    if record.name_hint == "left_common_iliac_candidate" or {"left_external_iliac", "left_internal_iliac"}.issubset(descendants):
        return "left_common_iliac", "topology_descendant_inference", 0.9, True, ["left_external_iliac", "left_internal_iliac"]

    if record.name_hint == "right_common_iliac_candidate" or {"right_external_iliac", "right_internal_iliac"}.issubset(descendants):
        return "right_common_iliac", "topology_descendant_inference", 0.9, True, ["right_external_iliac", "right_internal_iliac"]

    if record.name_hint == "celiac_proximal_candidate" or (descendants and all("celiac" in name for name in descendants)):
        return "celiac_artery", "topology_descendant_inference", 0.82, True, sorted(descendants)

    if record.terminal_face_name:
        return str(record.terminal_face_name), "direct_face_map", 0.97, False, [str(record.terminal_face_name)]

    if support_names:
        return str(record.name_hint or support_names[0]), "step2_name_hint", 0.65, True, support_names

    return f"segment_{record.segment_id}", "step2_name_hint", 0.4, True, []


def _build_named_segments(
    records: list[SegmentRecord],
    face_map: Dict[int, Dict[str, Any]],
    aorta_centerline: np.ndarray,
    upstream_network_path: Optional[Path],
    warnings: list[str],
) -> list[NamedSegment]:
    face_map_names = {face_id: str(row.get("name", "")) for face_id, row in face_map.items()}
    by_id = {record.segment_id: record for record in records}
    network_edges: dict[int, NetworkEdge] = {}
    if upstream_network_path is not None and upstream_network_path.exists():
        network_edges = _read_network_edges(read_vtp(upstream_network_path))
    elif upstream_network_path is not None:
        warnings.append(f"W_STEP3_CENTERLINE_NETWORK_MISSING: {upstream_network_path}")
    else:
        warnings.append("W_STEP3_CENTERLINE_NETWORK_UNDECLARED: STEP2 contract did not declare centerline_network.")

    named_segments: list[NamedSegment] = []
    for record in records:
        segment_name, naming_source, naming_confidence, inferred, support = _named_segment(
            record,
            face_map_names=face_map_names,
            by_id=by_id,
        )
        if record.segment_id == 1:
            centerline_points = np.asarray(aorta_centerline, dtype=float)
            centerline_mode = "step2_aorta_centerline"
        else:
            if network_edges:
                try:
                    centerline_points = _polyline_from_edges(record, network_edges)
                    centerline_mode = "step2_upstream_centerline_network"
                except Step3Failure as exc:
                    centerline_points = _straight_polyline(record)
                    centerline_mode = "step2_segment_endpoints_fallback"
                    warnings.append(f"W_STEP3_CENTERLINE_FALLBACK_SEGMENT_{record.segment_id}: {exc}")
            else:
                centerline_points = _straight_polyline(record)
                centerline_mode = "step2_segment_endpoints_fallback"
                warnings.append(
                    f"W_STEP3_CENTERLINE_FALLBACK_SEGMENT_{record.segment_id}: upstream centerline network unavailable."
                )

        named_segments.append(
            NamedSegment(
                record=record,
                segment_name=segment_name,
                naming_source=naming_source,
                naming_confidence=float(naming_confidence),
                inferred=bool(inferred),
                face_map_support=list(support),
                centerline_points=np.asarray(centerline_points, dtype=float),
                centerline_mode=centerline_mode,
            )
        )
    return named_segments


def _extract_segment_surface(surface: vtk.vtkPolyData, segment_id: int) -> vtk.vtkPolyData:
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(surface)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "SegmentId")
    if hasattr(threshold, "SetLowerThreshold"):
        threshold.SetLowerThreshold(float(segment_id))
        threshold.SetUpperThreshold(float(segment_id))
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    else:  # pragma: no cover - VTK compatibility fallback
        threshold.ThresholdBetween(float(segment_id), float(segment_id))
    threshold.Update()

    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(threshold.GetOutputPort())
    geometry.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(geometry.GetOutputPort())
    cleaner.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(cleaner.GetOutput())
    return out


def _boundary_profiles(surface: vtk.vtkPolyData) -> list[Dict[str, Any]]:
    if surface.GetNumberOfCells() <= 0:
        return []

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(feature_edges.GetOutputPort())
    cleaner.Update()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cleaner.GetOutputPort())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    loops = stripper.GetOutput()

    profiles: list[Dict[str, Any]] = []
    for cell_id in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(cell_id)
        if cell is None or cell.GetNumberOfPoints() < 3:
            continue
        ids = cell.GetPointIds()
        pts = np.asarray([loops.GetPoint(ids.GetId(i)) for i in range(ids.GetNumberOfIds())], dtype=float)
        if pts.shape[0] < 3:
            continue
        closed_gap = distance(pts[0], pts[-1])
        if closed_gap <= 1.0e-4:
            pts = pts[:-1]
        if pts.shape[0] < 3:
            continue
        area, normal, rms = polygon_area_normal(pts)
        if area <= 1.0e-8:
            continue
        centroid = np.mean(pts, axis=0)
        major, minor = projected_major_minor_diameters(pts, normal_hint=normal)
        profiles.append(
            {
                "cell_id": int(cell_id),
                "centroid": centroid,
                "normal": unit(normal),
                "area": float(area),
                "equivalent_diameter": equivalent_diameter_from_area(area),
                "major_diameter": major,
                "minor_diameter": minor,
                "rms_planarity": float(rms),
                "closed_gap": float(closed_gap),
                "point_count": int(pts.shape[0]),
            }
        )
    return profiles


def _serialize_profile(profile: Dict[str, Any], anchor_point: np.ndarray) -> Dict[str, Any]:
    return {
        "boundary_centroid": np.asarray(profile["centroid"], dtype=float).tolist(),
        "boundary_normal": np.asarray(profile["normal"], dtype=float).tolist(),
        "area": float(profile["area"]),
        "equivalent_diameter": profile["equivalent_diameter"],
        "major_diameter": profile["major_diameter"],
        "minor_diameter": profile["minor_diameter"],
        "rms_planarity": float(profile["rms_planarity"]),
        "closed_gap": float(profile["closed_gap"]),
        "point_count": int(profile["point_count"]),
        "distance_to_anchor": float(distance(np.asarray(profile["centroid"], dtype=float), anchor_point)),
    }


def _serialize_step2_proximal_boundary(boundary: Dict[str, Any], anchor_point: np.ndarray) -> Dict[str, Any]:
    centroid = np.asarray(boundary.get("centroid", boundary.get("boundary_centroid", anchor_point)), dtype=float)
    normal = np.asarray(boundary.get("normal", boundary.get("boundary_normal", [0.0, 0.0, 0.0])), dtype=float)
    return {
        "boundary_centroid": centroid.tolist(),
        "boundary_normal": normal.tolist(),
        "area": boundary.get("area"),
        "equivalent_diameter": boundary.get("equivalent_diameter"),
        "major_diameter": boundary.get("major_diameter"),
        "minor_diameter": boundary.get("minor_diameter"),
        "rms_planarity": boundary.get("rms_planarity"),
        "closed_gap": boundary.get("closed_gap"),
        "point_count": boundary.get("point_count"),
        "distance_to_anchor": float(distance(centroid, anchor_point)),
        "step2_boundary": dict(boundary),
    }


def _pick_nearest_profile(profiles: list[Dict[str, Any]], anchor_point: np.ndarray) -> Optional[Dict[str, Any]]:
    if not profiles:
        return None
    return min(
        profiles,
        key=lambda row: (
            float(distance(np.asarray(row["centroid"], dtype=float), anchor_point)),
            -float(row["area"]),
        ),
    )


def _loop_confidence(profile: Optional[Dict[str, Any]], anchor_point: np.ndarray, loop_count: int) -> float:
    if profile is None:
        return 0.55
    dist_mm = float(distance(np.asarray(profile["centroid"], dtype=float), anchor_point))
    confidence = 0.92
    if dist_mm > 6.0:
        confidence -= 0.30
    elif dist_mm > 3.0:
        confidence -= 0.18
    elif dist_mm > 1.5:
        confidence -= 0.08
    if float(profile["closed_gap"]) > 0.5:
        confidence -= 0.08
    if loop_count > 2:
        confidence -= 0.05
    return max(0.35, min(0.98, confidence))


def _proximal_metadata(named: NamedSegment, profiles: list[Dict[str, Any]], step2: Step2ContractView) -> Dict[str, Any]:
    record = named.record
    if record.segment_id == 1:
        return {
            "segment_id": 1,
            "segment_name": named.segment_name,
            "source_type": "step2_geometry_contract",
            "surface_derived": True,
            "centerline_derived": True,
            "centerline_anchor_point": record.proximal_point.tolist(),
            "boundary_profile": dict(step2.aorta_start),
            "confidence": float(step2.aorta_start.get("confidence", 0.95)),
            "method": "step2_aorta_start_reference",
        }

    if record.proximal_boundary is not None:
        confidence = float(
            record.proximal_boundary_confidence
            if record.proximal_boundary_confidence is not None
            else record.proximal_boundary.get("confidence", 0.85)
        )
        return {
            "segment_id": int(record.segment_id),
            "segment_name": named.segment_name,
            "source_type": record.proximal_boundary_source or "step2_proximal_boundary",
            "surface_derived": True,
            "centerline_derived": True,
            "centerline_anchor_point": record.proximal_point.tolist(),
            "boundary_profile": _serialize_step2_proximal_boundary(record.proximal_boundary, record.proximal_point),
            "boundary_loop_count": int(len(profiles)),
            "confidence": confidence,
            "method": "step2_surface_authored_proximal_boundary",
            "step2_proximal_boundary_arclength": record.proximal_boundary_arclength,
        }

    best = _pick_nearest_profile(profiles, record.proximal_point)
    if best is None:
        return {
            "segment_id": int(record.segment_id),
            "segment_name": named.segment_name,
            "source_type": "centerline_anchor_fallback",
            "surface_derived": False,
            "centerline_derived": True,
            "centerline_anchor_point": record.proximal_point.tolist(),
            "boundary_profile": None,
            "confidence": 0.55,
            "method": "segment_surface_boundary_loop_nearest_proximal_anchor",
        }

    return {
        "segment_id": int(record.segment_id),
        "segment_name": named.segment_name,
        "source_type": "segment_surface_boundary_loop",
        "surface_derived": True,
        "centerline_derived": True,
        "centerline_anchor_point": record.proximal_point.tolist(),
        "boundary_profile": _serialize_profile(best, record.proximal_point),
        "boundary_loop_count": int(len(profiles)),
        "confidence": _loop_confidence(best, record.proximal_point, len(profiles)),
        "method": "segment_surface_boundary_loop_nearest_proximal_anchor",
    }


def _distal_metadata(
    named: NamedSegment,
    profiles: list[Dict[str, Any]],
    step2: Step2ContractView,
) -> Dict[str, Any]:
    record = named.record
    if record.segment_id == 1:
        return {
            "segment_id": 1,
            "segment_name": named.segment_name,
            "source_type": "step2_geometry_contract",
            "surface_derived": bool(step2.aorta_end.get("surface_derived", True)),
            "centerline_derived": bool(step2.aorta_end.get("centerline_derived", True)),
            "centerline_anchor_point": record.distal_point.tolist(),
            "boundary_profile": dict(step2.aorta_end.get("surface_boundary_profile") or {}),
            "confidence": float(step2.aorta_end.get("confidence", 0.9)),
            "method": "step2_aorta_end_reference",
        }

    if record.terminal_face_id is not None and record.terminal_face_id in step2.mapped_terminal_boundaries:
        terminal_row = dict(step2.mapped_terminal_boundaries[record.terminal_face_id])
        best = None
        if profiles:
            center = np.asarray(terminal_row.get("center", record.distal_point.tolist()), dtype=float)
            best = _pick_nearest_profile(profiles, center)
        return {
            "segment_id": int(record.segment_id),
            "segment_name": named.segment_name,
            "source_type": "step2_mapped_terminal_boundary",
            "surface_derived": True,
            "centerline_derived": True,
            "centerline_anchor_point": record.distal_point.tolist(),
            "face_id": int(record.terminal_face_id),
            "face_name": str(record.terminal_face_name),
            "boundary_profile": {
                "boundary_centroid": terminal_row.get("center"),
                "boundary_normal": terminal_row.get("normal"),
                "area": terminal_row.get("area"),
                "equivalent_diameter": terminal_row.get("equivalent_diameter"),
                "major_diameter": terminal_row.get("major_diameter"),
                "minor_diameter": terminal_row.get("minor_diameter"),
                "source": terminal_row.get("source"),
                "distance_to_centerline_anchor": float(
                    distance(np.asarray(terminal_row.get("center", record.distal_point.tolist()), dtype=float), record.distal_point)
                ),
                "matched_segment_boundary": _serialize_profile(best, np.asarray(terminal_row.get("center", record.distal_point.tolist()), dtype=float))
                if best is not None
                else None,
            },
            "confidence": float(terminal_row.get("confidence", 0.9)),
            "method": "step2_terminal_boundary_reference",
        }

    distal_candidates = sorted(
        (_serialize_profile(profile, record.distal_point) for profile in profiles),
        key=lambda row: (float(row["distance_to_anchor"]), -float(row["area"])),
    )
    return {
        "segment_id": int(record.segment_id),
        "segment_name": named.segment_name,
        "source_type": "centerline_junction_landmark",
        "surface_derived": bool(distal_candidates),
        "centerline_derived": True,
        "centerline_anchor_point": record.distal_point.tolist(),
        "child_segment_ids": [int(v) for v in record.child_segment_ids],
        "boundary_candidates": distal_candidates[:3],
        "confidence": 0.82 if distal_candidates else 0.6,
        "method": "segment_surface_boundary_candidates_near_distal_anchor",
    }


def _add_string_cell_array(polydata: vtk.vtkPolyData, name: str, values: Iterable[str]) -> None:
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    vals = [str(v) for v in values]
    arr.SetNumberOfValues(len(vals))
    for idx, value in enumerate(vals):
        arr.SetValue(idx, value)
    polydata.GetCellData().AddArray(arr)


def _build_named_surface(
    surface: vtk.vtkPolyData,
    labels: np.ndarray,
    segment_name_map: Dict[int, str],
) -> vtk.vtkPolyData:
    out = clone_geometry_only(surface)
    label_values = labels.astype(int).tolist()
    add_int_cell_array(out, "SegmentId", label_values)
    _add_string_cell_array(out, "SegmentName", [segment_name_map[int(value)] for value in label_values])
    add_uchar3_cell_array(out, "SegmentColorRGB", [segment_color(int(value)) for value in label_values])
    return out


def _build_named_centerlines(named_segments: list[NamedSegment]) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    segment_ids = vtk.vtkIntArray()
    segment_ids.SetName("SegmentId")
    segment_names = vtk.vtkStringArray()
    segment_names.SetName("SegmentName")

    point_offset = 0
    for named in sorted(named_segments, key=lambda row: row.record.segment_id):
        pts = np.asarray(named.centerline_points, dtype=float)
        if pts.shape[0] < 2:
            raise Step3Failure(f"Named centerline for segment {named.record.segment_id} has fewer than 2 points.")
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(int(pts.shape[0]))
        for idx, point in enumerate(pts):
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
            polyline.GetPointIds().SetId(idx, int(point_offset + idx))
        lines.InsertNextCell(polyline)
        point_offset += int(pts.shape[0])
        segment_ids.InsertNextValue(int(named.record.segment_id))
        segment_names.InsertNextValue(str(named.segment_name))

    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)
    out.GetCellData().AddArray(segment_ids)
    out.GetCellData().AddArray(segment_names)
    return out


def _required_priority_names() -> set[str]:
    return set(PRIORITY_VESSELS)


def _make_failure_contract(
    *,
    warnings: list[str],
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "step_name": "step3_naming_orientation",
        "step_status": "failed",
        "warnings": _dedupe_warnings(warnings),
        "input_paths": input_paths,
        "output_paths": output_paths,
        "step2_references": {},
        "segment_name_map": {},
        "vessel_priority_classification": {},
        "matched_face_map_entries": [],
        "unmatched_face_map_entries": [],
        "inferred_vessels": [],
        "landmark_registry": {},
        "proximal_start_metadata": {},
        "distal_end_metadata": {},
        "confidence_flags": {
            "priority_start_confidence_threshold": PRIORITY_REVIEW_THRESHOLD,
            "review_triggers": [],
        },
        "qa": {
            "step_status": "failed",
            "named_segment_count": 0,
        },
    }


def run_step3(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    paths = build_pipeline_paths(project_root)
    step2_dir = Path(args.step2_dir).resolve() if args.step2_dir else paths.step2_dir
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths.step3_dir
    face_map_path = Path(args.face_map).resolve() if args.face_map else paths.default_face_map
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_path = step2_dir / "segmentscolored.vtp"
    aorta_centerline_path = step2_dir / "aorta_centerline.vtp"
    step2_contract_path = step2_dir / "step2_geometry_contract.json"
    named_surface_path = output_dir / "named_segmentscolored.vtp"
    named_centerlines_path = output_dir / "named_centerlines.vtp"
    contract_path = output_dir / "step3_naming_orientation_contract.json"

    input_paths = {
        "face_map": _abs(face_map_path),
        "segments_vtp": _abs(segments_path),
        "aorta_centerline": _abs(aorta_centerline_path),
        "step2_contract_json": _abs(step2_contract_path),
    }
    output_paths = {
        "named_segments_vtp": _abs(named_surface_path),
        "named_centerlines_vtp": _abs(named_centerlines_path),
        "contract_json": _abs(contract_path),
    }

    required = {
        "segmentscolored.vtp": segments_path,
        "aorta_centerline.vtp": aorta_centerline_path,
        "step2_geometry_contract.json": step2_contract_path,
        "face_id_to_name.json": face_map_path,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise Step3Failure("Missing required STEP3 input(s): " + "; ".join(missing))

    try:
        face_map = _face_map_by_id(read_json(face_map_path))
    except Exception as exc:
        raise Step3Failure(f"Could not read face map JSON: {face_map_path} ({exc})") from exc
    if not face_map:
        raise Step3Failure(f"Face map JSON did not contain any usable entries: {face_map_path}")

    try:
        step2_raw = read_json(step2_contract_path)
    except Exception as exc:
        raise Step3Failure(f"Could not read STEP2 contract JSON: {step2_contract_path} ({exc})") from exc
    step2 = _normalize_step2_contract(step2_raw)
    if step2.step_status == "failed":
        raise Step3Failure("STEP2 contract reports failed status; STEP3 cannot proceed.")

    warnings = list(step2.warnings)
    if step2.step_status == "requires_review":
        warnings.append("W_STEP3_UPSTREAM_STEP2_REQUIRES_REVIEW: STEP2 contract status is requires_review.")

    step2_surface = read_vtp(segments_path)
    labels = get_cell_array(step2_surface, "SegmentId")
    if labels is None:
        raise Step3Failure("STEP2 segmentscolored.vtp is missing the SegmentId cell array.")
    if int(step2_surface.GetNumberOfCells()) <= 0:
        raise Step3Failure("STEP2 segmentscolored.vtp did not contain any surface cells.")

    aorta_centerline = _extract_single_polyline(read_vtp(aorta_centerline_path))
    records = _segment_records(step2.segment_rows)

    upstream_network_path: Optional[Path] = None
    network_ref = step2.upstream_references.get("centerline_network")
    if network_ref:
        upstream_network_path = Path(network_ref).resolve()

    named_segments = _build_named_segments(records, face_map, aorta_centerline, upstream_network_path, warnings)
    segment_name_map = {named.record.segment_id: named.segment_name for named in named_segments}

    missing_priorities = sorted(_required_priority_names() - set(segment_name_map.values()))
    if missing_priorities:
        raise Step3Failure(
            "STEP3 could not resolve all priority vessels from STEP2 geometry: " + ", ".join(missing_priorities)
        )

    loop_profiles_by_segment: Dict[int, list[Dict[str, Any]]] = {}
    for named in named_segments:
        segment_surface = _extract_segment_surface(step2_surface, named.record.segment_id)
        loop_profiles_by_segment[named.record.segment_id] = _boundary_profiles(segment_surface)
        if named.record.segment_id != 1 and not loop_profiles_by_segment[named.record.segment_id]:
            warnings.append(
                f"W_STEP3_SURFACE_BOUNDARY_MISSING_SEGMENT_{named.record.segment_id}: "
                f"no segment boundary loops were found for {named.segment_name}."
            )

    proximal_start_metadata = {
        named.segment_name: _proximal_metadata(named, loop_profiles_by_segment[named.record.segment_id], step2)
        for named in named_segments
    }
    distal_end_metadata = {
        named.segment_name: _distal_metadata(named, loop_profiles_by_segment[named.record.segment_id], step2)
        for named in named_segments
    }

    matched_face_map_entries: list[Dict[str, Any]] = []
    matched_face_ids: set[int] = set()
    inlet_face_id = None
    for face_id, row in face_map.items():
        if str(row.get("name", "")).strip().lower() == "abdominal_aorta_inlet":
            inlet_face_id = int(face_id)
            matched_face_ids.add(int(face_id))
            matched_face_map_entries.append(
                {
                    "face_id": int(face_id),
                    "face_name": str(row.get("name", "")),
                    "segment_id": 1,
                    "segment_name": "abdominal_aorta_trunk",
                    "match_type": "inlet_reference",
                }
            )
            break

    for named in named_segments:
        record = named.record
        if record.terminal_face_id is None:
            continue
        matched_face_ids.add(int(record.terminal_face_id))
        matched_face_map_entries.append(
            {
                "face_id": int(record.terminal_face_id),
                "face_name": str(record.terminal_face_name),
                "segment_id": int(record.segment_id),
                "segment_name": named.segment_name,
                "match_type": "direct_face_map",
            }
        )

    unmatched_face_map_entries = [
        {
            "face_id": int(face_id),
            "face_name": str(row.get("name", "")),
        }
        for face_id, row in sorted(face_map.items())
        if int(face_id) not in matched_face_ids
    ]

    inferred_vessels = [
        {
            "segment_id": int(named.record.segment_id),
            "segment_name": named.segment_name,
            "inference_method": named.naming_source,
            "supporting_face_map_names": list(named.face_map_support),
            "confidence": float(named.naming_confidence),
        }
        for named in named_segments
        if named.inferred
    ]

    landmark_registry = {
        "aorta_start": dict(step2.aorta_start),
        "aorta_end_pre_bifurcation": dict(step2.aorta_end),
        "segment_landmarks": {
            named.segment_name: {
                "segment_id": int(named.record.segment_id),
                "parent_segment_id": named.record.parent_segment_id,
                "child_segment_ids": [int(v) for v in named.record.child_segment_ids],
                "proximal_node_id": int(named.record.proximal_node_id),
                "distal_node_id": int(named.record.distal_node_id),
                "proximal_point": named.record.proximal_point.tolist(),
                "proximal_boundary_source": named.record.proximal_boundary_source,
                "proximal_boundary_confidence": named.record.proximal_boundary_confidence,
                "proximal_boundary_arclength": named.record.proximal_boundary_arclength,
                "proximal_boundary": dict(named.record.proximal_boundary) if named.record.proximal_boundary is not None else None,
                "distal_point": named.record.distal_point.tolist(),
                "terminal_face_id": named.record.terminal_face_id,
                "terminal_face_name": named.record.terminal_face_name,
                "descendant_terminal_names": list(named.record.descendant_terminal_names),
            }
            for named in named_segments
        },
    }

    review_triggers: list[str] = []
    confidence_flags: Dict[str, Any] = {
        "priority_start_confidence_threshold": PRIORITY_REVIEW_THRESHOLD,
        "review_triggers": review_triggers,
        "per_segment": {},
    }
    for named in named_segments:
        start_conf = float(proximal_start_metadata[named.segment_name].get("confidence", 0.0))
        end_conf = float(distal_end_metadata[named.segment_name].get("confidence", 0.0))
        priority_class = _priority_classification(named.segment_name)
        requires_review = False
        if priority_class == "high_priority" and start_conf < PRIORITY_REVIEW_THRESHOLD:
            requires_review = True
            review_triggers.append(
                f"priority vessel {named.segment_name} proximal start confidence {start_conf:.2f} is below {PRIORITY_REVIEW_THRESHOLD:.2f}"
            )
        if named.centerline_mode == "step2_segment_endpoints_fallback":
            requires_review = True
            review_triggers.append(f"{named.segment_name} centerline used step2 endpoint fallback.")
        confidence_flags["per_segment"][named.segment_name] = {
            "segment_id": int(named.record.segment_id),
            "priority_class": priority_class,
            "naming_confidence": float(named.naming_confidence),
            "proximal_start_confidence": start_conf,
            "distal_end_confidence": end_conf,
            "centerline_mode": named.centerline_mode,
            "requires_review": requires_review,
            "inferred_vessel": bool(named.inferred),
        }

    for unmatched in unmatched_face_map_entries:
        warnings.append(
            f"W_STEP3_UNMATCHED_FACE_MAP_ENTRY: face {unmatched['face_id']} ({unmatched['face_name']}) was not matched to a STEP2 segment."
        )

    step_status = "requires_review" if review_triggers else "success"
    vessel_priority_classification = {
        named.segment_name: {
            "segment_id": int(named.record.segment_id),
            "priority_class": _priority_classification(named.segment_name),
            "naming_source": named.naming_source,
        }
        for named in named_segments
    }

    named_surface = _build_named_surface(step2_surface, labels.astype(int), segment_name_map)
    write_vtp(named_surface, named_surface_path)

    named_centerlines = _build_named_centerlines(named_segments)
    write_vtp(named_centerlines, named_centerlines_path)

    contract = {
        "schema_version": 1,
        "step_name": "step3_naming_orientation",
        "step_status": step_status,
        "warnings": _dedupe_warnings(warnings),
        "input_paths": input_paths,
        "output_paths": output_paths,
        "step2_references": {
            "step2_contract_mode": step2.contract_mode,
            "step2_schema_version": step2.schema_version,
            "step2_status": step2.step_status,
            "step2_warning_count": int(len(step2.warnings)),
            "step2_segment_count": int(len(records)),
            "upstream_centerline_network": _abs(upstream_network_path) if upstream_network_path is not None else "",
            "coordinate_system": step2.coordinate_system,
            "units": step2.units,
        },
        "segment_name_map": {str(segment_id): name for segment_id, name in sorted(segment_name_map.items())},
        "vessel_priority_classification": vessel_priority_classification,
        "matched_face_map_entries": matched_face_map_entries,
        "unmatched_face_map_entries": unmatched_face_map_entries,
        "inferred_vessels": inferred_vessels,
        "landmark_registry": landmark_registry,
        "proximal_start_metadata": proximal_start_metadata,
        "distal_end_metadata": distal_end_metadata,
        "confidence_flags": confidence_flags,
        "qa": {
            "step_status": step_status,
            "named_segment_count": int(len(named_segments)),
            "named_centerline_count": int(named_centerlines.GetNumberOfCells()),
            "matched_face_map_count": int(len(matched_face_map_entries)),
            "unmatched_face_map_count": int(len(unmatched_face_map_entries)),
            "inferred_vessel_count": int(len(inferred_vessels)),
            "surface_derived_proximal_start_count": int(
                sum(1 for row in proximal_start_metadata.values() if row.get("surface_derived"))
            ),
            "surface_derived_distal_end_count": int(sum(1 for row in distal_end_metadata.values() if row.get("surface_derived"))),
            "priority_vessels_present": sorted(set(segment_name_map.values()) & _required_priority_names()),
            "priority_vessels_missing": missing_priorities,
            "centerline_fallback_segments": sorted(
                named.segment_name for named in named_segments if named.centerline_mode == "step2_segment_endpoints_fallback"
            ),
        },
    }
    write_json(contract, contract_path)
    return contract


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    paths = build_pipeline_paths(project_root)
    parser = argparse.ArgumentParser(description="STEP3 vessel naming and orientation contract.")
    parser.add_argument("--project-root", default=str(project_root), help="Project root.")
    parser.add_argument("--step2-dir", default=str(paths.step2_dir), help="STEP2 output directory.")
    parser.add_argument("--output-dir", default=str(paths.step3_dir), help="STEP3 output directory.")
    parser.add_argument("--face-map", default=str(paths.default_face_map), help="face_id_to_name.json path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    paths = build_pipeline_paths(project_root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths.step3_dir
    step2_dir = Path(args.step2_dir).resolve() if args.step2_dir else paths.step2_dir
    face_map_path = Path(args.face_map).resolve() if args.face_map else paths.default_face_map
    contract_path = output_dir / "step3_naming_orientation_contract.json"

    input_paths = {
        "face_map": _abs(face_map_path),
        "segments_vtp": _abs(step2_dir / "segmentscolored.vtp"),
        "aorta_centerline": _abs(step2_dir / "aorta_centerline.vtp"),
        "step2_contract_json": _abs(step2_dir / "step2_geometry_contract.json"),
    }
    output_paths = {
        "named_segments_vtp": _abs(output_dir / "named_segmentscolored.vtp"),
        "named_centerlines_vtp": _abs(output_dir / "named_centerlines.vtp"),
        "contract_json": _abs(contract_path),
    }

    try:
        contract = run_step3(args)
    except Step3Failure as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_contract = _make_failure_contract(
            warnings=[str(exc)],
            input_paths=input_paths,
            output_paths=output_paths,
        )
        write_json(failure_contract, contract_path)
        print(f"STEP3 failed: {exc}")
        return 1

    print(
        "STEP3 completed: "
        f"{contract.get('step_status')} | "
        f"named_segments={contract.get('qa', {}).get('named_segment_count')} | "
        f"matched_faces={contract.get('qa', {}).get('matched_face_map_count')}"
    )
    return 1 if contract.get("step_status") == "failed" else 0
