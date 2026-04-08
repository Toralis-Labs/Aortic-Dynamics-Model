#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 only: orient an abdominal aorta surface VTP into a consistent frame.

This script intentionally stops after:
- loading and cleaning the surface
- detecting input mode / terminations
- inferring scale
- extracting centerlines
- inferring a canonical anatomy frame
- applying the final transform
- writing an oriented VTP plus report/meta sidecars

It does not compute EVAR measurements.
"""

from __future__ import annotations

import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sizingnew as core


def derive_output_paths(input_path: str) -> Tuple[str, str, str]:
    stem, _ = os.path.splitext(input_path)
    return (
        stem + "_oriented.vtp",
        stem + "_orientation_report.txt",
        stem + "_orientation_meta.json",
    )


INPUT_VTP_PATH = core.INPUT_VTP_PATH
OUTPUT_ORIENTED_VTP_PATH, OUTPUT_ORIENTATION_REPORT_PATH, OUTPUT_ORIENTATION_META_PATH = derive_output_paths(
    INPUT_VTP_PATH
)


NAN = float("nan")


def format_scalar(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return f"{value:.6f}"
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=6, separator=", ")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_scalar(v) for v in value) + "]"
    return str(value)


def json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value


def write_vtp(path: str, pd: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = core.vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(pd)
    writer.Write()


def write_orientation_report(path: str, report: Dict[str, Any], warnings: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ordered_keys = [
        "success",
        "input_vtp_path",
        "output_oriented_vtp_path",
        "input_mode",
        "face_partition_array",
        "n_termination_candidates",
        "semantic_termination_matches",
        "scale_factor_to_mm",
        "scale_confidence",
        "vmtk_available",
        "centerlines_succeeded",
        "centerline_points",
        "centerline_cells",
        "centerline_line_cells",
        "orientation_method",
        "ap_source",
        "final_centering_reference",
        "inlet_node",
        "bifurcation_node",
        "right_external_endpoint_node",
        "left_external_endpoint_node",
        "inlet_point_mm_original",
        "bifurcation_point_mm_original",
        "bifurcation_point_canonical",
        "frame_confidence",
        "iliac_lr_confidence",
        "ap_confidence",
        "inlet_confidence",
        "bifurcation_confidence",
        "warn_scale_inference",
        "warn_centerlines",
        "warn_inlet_identification",
        "warn_iliac_lr_orientation",
        "warn_ap_orientation",
        "warn_aortic_bifurcation",
        "warn_left_right_mirror_ambiguity",
    ]

    lines: List[str] = []
    for key in ordered_keys:
        lines.append(f"{key}={format_scalar(report.get(key))}")

    extra_keys = sorted(k for k in report.keys() if k not in set(ordered_keys))
    for key in extra_keys:
        lines.append(f"{key}={format_scalar(report[key])}")

    lines.append("")
    lines.append(f"Warnings_count={len(warnings)}")
    for idx, warning in enumerate(warnings, start=1):
        lines.append(f"WARNING_{idx:03d}={warning}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_orientation_meta(path: str, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(json_ready(meta), fh, indent=2, sort_keys=True)


def base_report(input_path: str, output_vtp_path: str) -> Dict[str, Any]:
    return dict(
        success=False,
        input_vtp_path=input_path,
        output_oriented_vtp_path=output_vtp_path,
        input_mode="unknown",
        face_partition_array=None,
        n_termination_candidates=0,
        semantic_termination_matches=0,
        scale_factor_to_mm=NAN,
        scale_confidence=NAN,
        vmtk_available=False,
        centerlines_succeeded=False,
        centerline_points=0,
        centerline_cells=0,
        centerline_line_cells=0,
        orientation_method="unresolved",
        ap_source="fallback",
        final_centering_reference="none",
        inlet_node=None,
        bifurcation_node=None,
        right_external_endpoint_node=None,
        left_external_endpoint_node=None,
        inlet_point_mm_original=None,
        bifurcation_point_mm_original=None,
        bifurcation_point_canonical=None,
        frame_confidence=0.0,
        iliac_lr_confidence=0.0,
        ap_confidence=0.0,
        inlet_confidence=0.0,
        bifurcation_confidence=0.0,
        warn_scale_inference=0,
        warn_centerlines=0,
        warn_inlet_identification=0,
        warn_iliac_lr_orientation=0,
        warn_ap_orientation=0,
        warn_aortic_bifurcation=0,
        warn_left_right_mirror_ambiguity=0,
    )


def build_orientation_metadata(
    report: Dict[str, Any],
    input_path: str,
    oriented_vtp_path: str,
    report_path: str,
    meta_path: str,
    warnings: List[str],
    transform_info: Dict[str, Any],
) -> Dict[str, Any]:
    return dict(
        input_vtp_path=input_path,
        output_oriented_vtp_path=oriented_vtp_path,
        output_orientation_report_path=report_path,
        output_orientation_meta_path=meta_path,
        success=bool(report.get("success", False)),
        input_mode=report.get("input_mode"),
        face_partition_array=report.get("face_partition_array"),
        n_termination_candidates=report.get("n_termination_candidates"),
        semantic_termination_matches=report.get("semantic_termination_matches"),
        scale_factor_to_mm=report.get("scale_factor_to_mm"),
        scale_confidence=report.get("scale_confidence"),
        centerlines_succeeded=report.get("centerlines_succeeded"),
        vmtk_available=report.get("vmtk_available"),
        orientation_method=report.get("orientation_method"),
        ap_source=report.get("ap_source"),
        final_centering_reference=report.get("final_centering_reference"),
        inlet_node=report.get("inlet_node"),
        bifurcation_node=report.get("bifurcation_node"),
        right_external_endpoint_node=report.get("right_external_endpoint_node"),
        left_external_endpoint_node=report.get("left_external_endpoint_node"),
        inlet_point_mm_original=report.get("inlet_point_mm_original"),
        bifurcation_point_mm_original=report.get("bifurcation_point_mm_original"),
        bifurcation_point_canonical=report.get("bifurcation_point_canonical"),
        confidence=dict(
            frame=report.get("frame_confidence"),
            iliac_lr=report.get("iliac_lr_confidence"),
            ap=report.get("ap_confidence"),
            inlet=report.get("inlet_confidence"),
            bifurcation=report.get("bifurcation_confidence"),
        ),
        warnings=warnings,
        warning_flags=dict(
            warn_scale_inference=report.get("warn_scale_inference"),
            warn_centerlines=report.get("warn_centerlines"),
            warn_inlet_identification=report.get("warn_inlet_identification"),
            warn_iliac_lr_orientation=report.get("warn_iliac_lr_orientation"),
            warn_ap_orientation=report.get("warn_ap_orientation"),
            warn_aortic_bifurcation=report.get("warn_aortic_bifurcation"),
            warn_left_right_mirror_ambiguity=report.get("warn_left_right_mirror_ambiguity"),
        ),
        transform=transform_info,
    )


def orient_surface_from_centerlines(
    surface_mm: Any,
    centerlines: Any,
    terminations_mm: List[Dict[str, Any]],
    warnings: List[str],
    report: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    n_cl_points, n_cl_cells, n_cl_lines = core.polydata_line_stats(centerlines)
    report["centerline_points"] = int(n_cl_points)
    report["centerline_cells"] = int(n_cl_cells)
    report["centerline_line_cells"] = int(n_cl_lines)

    adjacency, pts = core.build_graph_from_centerlines(centerlines)
    if not adjacency:
        report["warn_centerlines"] = 1
        core.add_warning(
            warnings,
            "W_GRAPH_001",
            "Centerline graph construction failed during orientation stage.",
        )
        return None, {}

    nodes = list(adjacency.keys())
    degrees = {n: len(adjacency[n]) for n in nodes}
    endpoints = [n for n, deg in degrees.items() if deg == 1]
    if not endpoints:
        endpoints = sorted(nodes, key=lambda n: degrees.get(n, 99))[:8]
        core.add_warning(
            warnings,
            "W_GRAPH_002",
            "No degree-1 centerline endpoints found; falling back to low-degree nodes for orientation.",
        )

    inlet_node, inlet_conf, inlet_warn = core.pick_inlet_node(endpoints, pts, terminations_mm, warnings)
    report["inlet_node"] = int(inlet_node) if inlet_node >= 0 else None
    report["inlet_confidence"] = float(inlet_conf)
    if inlet_warn:
        report["warn_inlet_identification"] = 1
    if inlet_node < 0:
        report["warn_centerlines"] = 1
        core.add_warning(warnings, "W_INLET_004", "Failed to identify inlet node for orientation.")
        return None, {}

    dist_root, _ = core.dijkstra(adjacency, inlet_node)
    distal_candidates = core.identify_distal_endpoints_for_iliacs(endpoints, inlet_node, pts, dist_root)
    semantic_endpoints_pre = core.map_semantic_terminations_to_centerline_endpoints(endpoints, pts, terminations_mm)
    iliac_scaffold = core.infer_provisional_aortic_bifurcation_and_iliac_subtrees(
        adjacency,
        endpoints,
        inlet_node,
        pts,
        dist_root,
    )

    use_semantic_orientation = (
        semantic_endpoints_pre.get("ext_iliac_right") is not None
        and semantic_endpoints_pre.get("ext_iliac_left") is not None
        and min(
            float(semantic_endpoints_pre.get("ext_iliac_right_conf", 0.0)),
            float(semantic_endpoints_pre.get("ext_iliac_left_conf", 0.0)),
        )
        >= 0.35
    )
    use_anatomic_orientation = (
        iliac_scaffold.get("bif_node") is not None
        and len(iliac_scaffold.get("groups", [])) >= 2
    )
    provisional_bif_node = int(iliac_scaffold["bif_node"]) if use_anatomic_orientation else None

    right_group_endpoints: List[int] = []
    left_group_endpoints: List[int] = []
    R: np.ndarray
    t: np.ndarray

    if use_semantic_orientation:
        seed_nodes = [
            int(semantic_endpoints_pre["ext_iliac_right"]),
            int(semantic_endpoints_pre["ext_iliac_left"]),
        ]
        R0, t0, base_frame_conf, base_frame_warn = core.canonical_transform_from_centerlines_and_terminations(
            pts,
            inlet_node,
            seed_nodes,
            warnings,
        )
        pts_tmp = (R0 @ pts.T).T + t0[None, :]
        terminations_tmp = core.transform_terminations(terminations_mm, R0, t0)
        flip_needed, lr_conf, ap_conf, lr_warn, ap_warn = core.resolve_frame_sign_from_semantics(
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

        report["orientation_method"] = "semantic_outlet_orientation"
        report["ap_source"] = "semantic_terminations" if ap_conf > 0.0 else "fallback"
        report["iliac_lr_confidence"] = float(lr_conf)
        report["ap_confidence"] = float(ap_conf)
        if lr_warn:
            report["warn_iliac_lr_orientation"] = 1
            core.add_warning(
                warnings,
                "W_ILIAC_ORIENT_001",
                "Semantic external iliac labels were available but could not anchor left/right confidently.",
            )
        if ap_warn:
            report["warn_ap_orientation"] = 1
            core.add_warning(
                warnings,
                "W_AP_003",
                "Semantic ventral outlet evidence was insufficient to lock AP sign confidently.",
            )
        if lr_conf < 0.55:
            report["warn_left_right_mirror_ambiguity"] = 1
            core.add_warning(
                warnings,
                "W_LR_001",
                "Left/right may remain mirrored because semantic outlet matching is low-confidence.",
            )
        frame_conf = float(core.clamp(0.35 * base_frame_conf + 0.40 * lr_conf + 0.25 * max(ap_conf, lr_conf), 0.0, 1.0))
        frame_warn = int(bool(base_frame_warn and lr_conf < 0.55 and ap_conf < 0.55))
    elif use_anatomic_orientation and provisional_bif_node is not None:
        report["orientation_method"] = "iliac_scaffold_ventral_branches"
        report["ap_source"] = "ventral_branches"
        report["iliac_lr_confidence"] = float(iliac_scaffold.get("confidence", 0.0))
        if iliac_scaffold.get("warn", 0):
            report["warn_iliac_lr_orientation"] = 1
            core.add_warning(
                warnings,
                "W_ILIAC_ORIENT_001",
                f"Provisional iliac subtree scaffold is low-confidence (confidence={float(iliac_scaffold.get('confidence', 0.0)):.3f}).",
            )

        trunk_nodes0, _ = core.shortest_path(adjacency, inlet_node, provisional_bif_node)
        R0, t0, si_lr_conf, si_lr_warn = core.build_provisional_si_lr_frame(
            pts,
            inlet_node,
            provisional_bif_node,
            list(iliac_scaffold.get("groups", [])),
            warnings,
        )
        pts_provisional = (R0 @ pts.T).T + t0[None, :]
        ap_sign, ap_conf, ap_warn = core.resolve_ap_sign_from_ventral_branches(
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
        R, t = core.finalize_canonical_transform_from_provisional(R0, pts[provisional_bif_node], ap_sign)
        frame_conf = float(core.clamp(0.65 * si_lr_conf + 0.35 * ap_conf, 0.0, 1.0))
        frame_warn = int(bool(si_lr_warn or ap_warn or iliac_scaffold.get("warn", 0)))
        report["ap_confidence"] = float(ap_conf)
        if ap_warn:
            report["warn_ap_orientation"] = 1
        if ap_conf < 0.55:
            report["warn_left_right_mirror_ambiguity"] = 1
            core.add_warning(
                warnings,
                "W_LR_001",
                "Left/right may remain mirrored because ventral branch AP resolution is low-confidence.",
            )
    else:
        report["orientation_method"] = "generic_distal_endpoint_fallback"
        report["ap_source"] = "fallback"
        report["warn_iliac_lr_orientation"] = 1
        report["warn_ap_orientation"] = 1
        report["warn_left_right_mirror_ambiguity"] = 1
        core.add_warning(
            warnings,
            "W_ILIAC_ORIENT_001",
            "Could not infer a reliable bilateral iliac scaffold; using generic distal-endpoint orientation.",
        )
        core.add_warning(
            warnings,
            "W_LR_001",
            "Left/right may be mirrored because the input VTP has no absolute patient orientation metadata.",
        )
        report["iliac_lr_confidence"] = 0.0
        report["ap_confidence"] = 0.0
        R, t, frame_conf, frame_warn = core.canonical_transform_from_centerlines_and_terminations(
            pts,
            inlet_node,
            distal_candidates,
            warnings,
        )

    report["frame_confidence"] = float(frame_conf)
    if frame_warn:
        core.add_warning(
            warnings,
            "W_FRAME_002",
            "Canonical orientation frame may be unreliable.",
        )

    centerlines_can = core.apply_linear_transform_to_polydata(centerlines, R, t)
    terminations_can = core.transform_terminations(terminations_mm, R, t)
    adjacency_can, pts_can = core.build_graph_from_centerlines(centerlines_can)
    if not adjacency_can:
        report["warn_centerlines"] = 1
        core.add_warning(
            warnings,
            "W_GRAPH_003",
            "Centerline graph rebuild failed after provisional orientation.",
        )
        return None, {}

    nodes_can = list(adjacency_can.keys())
    degrees_can = {n: len(adjacency_can[n]) for n in nodes_can}
    endpoints_can = [n for n, deg in degrees_can.items() if deg == 1] or endpoints
    dist_root_can, _ = core.dijkstra(adjacency_can, inlet_node)

    semantic_endpoints_can = core.map_semantic_terminations_to_centerline_endpoints(endpoints_can, pts_can, terminations_can)
    semantic_orientation_active = bool(use_semantic_orientation)
    semantic_right_int_ep: Optional[int] = None
    semantic_left_int_ep: Optional[int] = None

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
            report["warn_iliac_lr_orientation"] = 1
            core.add_warning(
                warnings,
                "W_ILIAC_ORIENT_002",
                "Semantic endpoint rematching on the centerline graph failed; falling back to graph heuristics.",
            )

    if (not semantic_orientation_active) and use_anatomic_orientation and provisional_bif_node is not None:
        right_ext, left_ext, right_group_endpoints, left_group_endpoints, iliac_pair_conf = (
            core.choose_left_right_external_endpoints_from_iliac_subtrees(
                adjacency_can,
                pts_can,
                provisional_bif_node,
                list(iliac_scaffold.get("groups", [])),
            )
        )
        if right_ext is None or left_ext is None:
            report["warn_iliac_lr_orientation"] = 1
            core.add_warning(
                warnings,
                "W_ILIAC_ORIENT_002",
                "Iliac subtree endpoint selection failed after frame alignment; falling back to distal endpoint pairing.",
            )
            distal_candidates = core.identify_distal_endpoints_for_iliacs(endpoints_can, inlet_node, pts_can, dist_root_can)
            right_ext, left_ext, iliac_pair_conf = core.choose_left_right_external_endpoints(
                distal_candidates,
                pts_can,
                dist_root_can,
                terminations_can,
            )
            right_group_endpoints = [e for e in endpoints_can if e not in {inlet_node, left_ext} and pts_can[e][0] >= 0]
            left_group_endpoints = [e for e in endpoints_can if e not in {inlet_node, right_ext} and pts_can[e][0] < 0]
    elif not semantic_orientation_active:
        distal_candidates = core.identify_distal_endpoints_for_iliacs(endpoints_can, inlet_node, pts_can, dist_root_can)
        right_ext, left_ext, iliac_pair_conf = core.choose_left_right_external_endpoints(
            distal_candidates,
            pts_can,
            dist_root_can,
            terminations_can,
        )
        right_group_endpoints = [e for e in endpoints_can if e not in {inlet_node, left_ext} and pts_can[e][0] >= 0]
        left_group_endpoints = [e for e in endpoints_can if e not in {inlet_node, right_ext} and pts_can[e][0] < 0]

    if right_ext is None or left_ext is None:
        report["warn_aortic_bifurcation"] = 1
        report["warn_iliac_lr_orientation"] = 1
        core.add_warning(
            warnings,
            "W_ILIAC_001",
            "Failed to identify bilateral distal iliac endpoints after orientation.",
        )
        return None, {}

    report["right_external_endpoint_node"] = int(right_ext)
    report["left_external_endpoint_node"] = int(left_ext)
    report["iliac_lr_confidence"] = float(
        core.clamp(
            0.50 * float(report.get("iliac_lr_confidence", 0.0)) + 0.50 * float(iliac_pair_conf),
            0.0,
            1.0,
        )
    )

    path_r, _ = core.shortest_path(adjacency_can, inlet_node, int(right_ext))
    path_l, _ = core.shortest_path(adjacency_can, inlet_node, int(left_ext))
    dist_root2, _ = core.dijkstra(adjacency_can, inlet_node)
    bif_node = None
    if path_r and path_l:
        bif_node = core.path_common_node_with_max_distance(path_r, path_l, dist_root2)
    if bif_node is None and provisional_bif_node is not None:
        bif_node = provisional_bif_node
        report["warn_aortic_bifurcation"] = 1
        core.add_warning(
            warnings,
            "W_BIF_002",
            "Using provisional iliac-topology bifurcation because the final bilateral path intersection was unstable.",
        )
    if bif_node is None:
        report["warn_aortic_bifurcation"] = 1
        core.add_warning(
            warnings,
            "W_BIF_001",
            "Failed to identify the aortic bifurcation after orientation.",
        )
        return None, {}

    report["bifurcation_node"] = int(bif_node)
    report["bifurcation_confidence"] = float(
        core.clamp(
            0.20 + 0.45 * float(iliac_pair_conf) + 0.35 * float(report.get("iliac_lr_confidence", 0.0)),
            0.0,
            1.0,
        )
    )

    t_final = -np.array(R, dtype=float) @ np.array(pts[int(bif_node)], dtype=float)
    surface_oriented = core.apply_linear_transform_to_polydata(surface_mm, R, t_final)
    centerlines_oriented = core.apply_linear_transform_to_polydata(centerlines, R, t_final)
    terminations_oriented = core.transform_terminations(terminations_mm, R, t_final)
    pts_oriented = (R @ pts.T).T + t_final[None, :]

    report["success"] = True
    report["centerlines_succeeded"] = True
    report["final_centering_reference"] = "aortic_bifurcation"
    report["inlet_point_mm_original"] = np.array(pts[int(inlet_node)], dtype=float)
    report["bifurcation_point_mm_original"] = np.array(pts[int(bif_node)], dtype=float)
    report["bifurcation_point_canonical"] = np.array(pts_oriented[int(bif_node)], dtype=float)

    scale_factor = float(report.get("scale_factor_to_mm", 1.0))
    transform_info = dict(
        orientation_rotation_matrix=np.array(R, dtype=float),
        orientation_translation_mm=np.array(t_final, dtype=float),
        combined_linear_matrix_input_to_oriented_mm=np.array(R, dtype=float) * scale_factor,
        combined_translation_input_to_oriented_mm=np.array(t_final, dtype=float),
        scale_factor_to_mm=scale_factor,
        origin_point_mm_original=np.array(pts[int(bif_node)], dtype=float),
        inlet_point_mm_original=np.array(pts[int(inlet_node)], dtype=float),
        bifurcation_point_mm_original=np.array(pts[int(bif_node)], dtype=float),
        bifurcation_point_canonical=np.array(pts_oriented[int(bif_node)], dtype=float),
        right_external_endpoint_point_mm_original=np.array(pts[int(right_ext)], dtype=float),
        left_external_endpoint_point_mm_original=np.array(pts[int(left_ext)], dtype=float),
        semantic_endpoint_map=semantic_endpoints_can,
        provisional_bifurcation_node=provisional_bif_node,
        right_group_endpoints=right_group_endpoints,
        left_group_endpoints=left_group_endpoints,
        semantic_right_internal_endpoint=semantic_right_int_ep,
        semantic_left_internal_endpoint=semantic_left_int_ep,
    )

    return surface_oriented, dict(
        report=report,
        transform_info=transform_info,
        centerlines_oriented=centerlines_oriented,
        terminations_oriented=terminations_oriented,
    )


def main() -> None:
    warnings: List[str] = []
    report = base_report(INPUT_VTP_PATH, OUTPUT_ORIENTED_VTP_PATH)
    transform_info: Dict[str, Any] = {}

    try:
        if not core.vtk_available():
            report["warn_centerlines"] = 1
            report["warn_scale_inference"] = 1
            core.add_warning(warnings, "E_VTK_001", f"VTK import failed: {core._VTK_IMPORT_ERROR}")
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
            return

        if not os.path.exists(INPUT_VTP_PATH):
            report["warn_centerlines"] = 1
            core.add_warning(warnings, "E_IO_001", f"Input file does not exist: {INPUT_VTP_PATH}")
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
            return

        surface = core.load_vtp(INPUT_VTP_PATH)
        if surface is None or surface.GetNumberOfPoints() == 0:
            report["warn_centerlines"] = 1
            core.add_warning(warnings, "E_IO_002", "Loaded surface is empty or invalid.")
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
            return

        surface_tri = core.clean_and_triangulate(surface)
        terminations, mode, face_array = core.detect_terminations_and_mode(surface_tri, warnings)
        matched_semantics = core.annotate_terminations_with_semantic_labels(terminations, INPUT_VTP_PATH)

        report["input_mode"] = mode
        report["face_partition_array"] = face_array
        report["n_termination_candidates"] = int(len(terminations))
        report["semantic_termination_matches"] = int(matched_semantics)

        if mode == "unsupported" or len(terminations) < 2:
            report["warn_centerlines"] = 1
            core.add_warning(
                warnings,
                "W_MODE_001",
                "Could not identify enough vessel terminations for orientation (need at least inlet plus one outlet).",
            )
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
            return

        scale_factor, scale_conf, scale_warn = core.infer_scale_to_mm(surface_tri, terminations, warnings)
        report["scale_factor_to_mm"] = float(scale_factor)
        report["scale_confidence"] = float(scale_conf)
        report["warn_scale_inference"] = int(scale_warn)

        scale_matrix = np.eye(3, dtype=float) * float(scale_factor)
        surface_mm = core.apply_linear_transform_to_polydata(surface_tri, scale_matrix, np.zeros(3, dtype=float))
        terminations_mm: List[Dict[str, Any]] = []
        for termination in terminations:
            scaled = dict(termination)
            scaled["center"] = np.array(termination["center"], dtype=float) * float(scale_factor)
            scaled["area"] = float(termination.get("area", 0.0)) * float(scale_factor ** 2)
            if scaled.get("area", 0.0) > 0.0:
                scaled["diameter_eq"] = math.sqrt(4.0 * float(scaled["area"]) / math.pi)
            terminations_mm.append(scaled)

        vtkvmtk_mod, vtkvmtk_err = core.try_import_vmtk()
        report["vmtk_available"] = bool(vtkvmtk_mod is not None)
        if vtkvmtk_mod is None:
            report["warn_centerlines"] = 1
            core.add_warning(warnings, "W_CENTER_002", f"VMTK not available; orientation cannot proceed. {vtkvmtk_err}")
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
            return

        centerlines, centerline_info = core.compute_centerlines_vmtk(surface_mm, terminations_mm, mode, warnings)
        transform_info["centerline_info"] = centerline_info
        if centerlines is None:
            report["warn_centerlines"] = 1
            core.add_warning(
                warnings,
                "W_CENTER_001",
                "Centerline extraction failed; stage-1 orientation stopped before writing an oriented VTP.",
            )
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
            return

        surface_oriented, orientation_info = orient_surface_from_centerlines(
            surface_mm,
            centerlines,
            terminations_mm,
            warnings,
            report,
        )
        if surface_oriented is None:
            report["warn_centerlines"] = 1
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
            return

        report = orientation_info["report"]
        transform_info.update(orientation_info["transform_info"])

        write_vtp(OUTPUT_ORIENTED_VTP_PATH, surface_oriented)

        meta = build_orientation_metadata(
            report,
            INPUT_VTP_PATH,
            OUTPUT_ORIENTED_VTP_PATH,
            OUTPUT_ORIENTATION_REPORT_PATH,
            OUTPUT_ORIENTATION_META_PATH,
            warnings,
            transform_info,
        )
        write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
        write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)

    except Exception as exc:
        core.add_warning(warnings, "E_FATAL_001", f"Unhandled exception: {exc}")
        core.add_warning(warnings, "E_FATAL_002", traceback.format_exc().strip().replace("\n", " | "))
        try:
            meta = build_orientation_metadata(
                report,
                INPUT_VTP_PATH,
                OUTPUT_ORIENTED_VTP_PATH,
                OUTPUT_ORIENTATION_REPORT_PATH,
                OUTPUT_ORIENTATION_META_PATH,
                warnings,
                transform_info,
            )
            write_orientation_report(OUTPUT_ORIENTATION_REPORT_PATH, report, warnings)
            write_orientation_meta(OUTPUT_ORIENTATION_META_PATH, meta)
        except Exception:
            pass


if __name__ == "__main__":
    main()
