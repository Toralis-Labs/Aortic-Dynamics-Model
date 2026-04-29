# STEP2 Proximal Boundary Refinement Plan

## Step 2 Purpose

STEP2 consumes the STEP1 cleaned lumen surface and STEP1 centerline network. It creates the strict geometry contract used by downstream STEP3 naming/orientation and later measurement stages.

STEP2 must produce reliable:

- segment topology
- segment surfaces in `Output files/STEP2/segmentscolored.vtp`
- centerline paths in `Output files/STEP2/aorta_centerline.vtp` and the contract
- terminal boundaries
- branch proximal boundaries
- QA warnings and review flags

STEP2 is the geometry-authoring layer. STEP3 should consume STEP2 geometry instead of re-solving vessel starts, ends, or parent-child boundaries.

## Proximal Boundary Problem

Graph nodes are not sufficient to define true ostia. A graph node is a centerline/topology landmark, not a surface boundary.

Daughter-only orthogonal cuts can select an internal stable branch section rather than the true parent/daughter transition. This is useful as a safe reference section, but it is often too distal to be the anatomical proximal boundary.

Parent/daughter junctions are difficult in AAA anatomy because of:

- oblique branch ostia
- steep IMA takeoff
- close branch origins
- noisy or broad junction geometry
- aneurysmal parent dilation near visceral branches

The contract must therefore distinguish the internal stable daughter section from the final proximal boundary used as the vessel start.

## Selected STEP2 Algorithm

The active proximal-boundary selection algorithm is:

`backward_refined_from_stable_section_v1`

The stable-section reference algorithm is:

`first_stable_surface_ostium_v2`

Algorithm:

1. Use the daughter centerline and lumen surface to find a stable internal branch section.
2. Store that section only as `stable_section_reference`.
3. Refine backward from the stable section toward the parent ostium in small arclength steps.
4. At each backward candidate section:
   - cut the surface with the daughter-centerline-normal plane
   - extract contour candidates
   - rank branch-like contours closest to the daughter centerline
   - reject contours that look parent-contaminated, unstable, too large, too far from the daughter centerline, open, fragmented, or invalid
5. Stop when parent contamination or instability begins, when the branch start is reached, or when the maximum search distance is reached.
6. Use the last accepted candidate before failure as the refined `proximal_boundary`.
7. If no backward candidate is accepted, retain the stable section as fallback and mark `requires_review`.

The final core VTP and downstream cell assignment must use only the final refined/fallback `proximal_boundary`. The stable section must remain QA metadata only.

## Required Contract Metadata

Every non-aorta branch with a proximal boundary must preserve the existing STEP2 segment fields:

- `proximal_boundary`
- `proximal_boundary_source`
- `proximal_boundary_confidence`
- `proximal_boundary_arclength`

The `proximal_boundary` object must include:

- `source_type`
- `centroid`
- `normal`
- `area`
- `equivalent_diameter`
- `major_diameter`
- `minor_diameter`
- `arclength`
- `confidence`
- `method`
- `selection_algorithm`
- `attempts`
- `stable_section_reference`
- `backward_refinement`

The `stable_section_reference` object must include:

- `centroid`
- `normal`
- `area`
- `equivalent_diameter`
- `major_diameter`
- `minor_diameter`
- `arclength`
- `method`
- `confidence`

The `backward_refinement` object must include:

- `step_mm`
- `max_search_mm`
- `accepted_step_count`
- `stopped_reason`
- `search_distance_mm`
- `initial_stable_arclength`
- `final_refined_arclength`
- `attempts`

Each backward refinement attempt should include:

- `offset_from_stable_mm`
- `arclength`
- `candidate_count`
- `accepted`
- `rejection_reason`
- `candidate_equivalent_diameter`
- `candidate_area`
- `candidate_centroid_distance_to_origin`
- `candidate_centroid_jump_from_previous`
- `diameter_ratio_to_stable`
- `area_ratio_to_stable`

Fallback, unresolved, or low-confidence cases must add:

- `requires_review` when fallback or low confidence affects the boundary
- `warnings` explaining fallback, unresolved selection, or low confidence

## Required Code Hygiene Acceptance Criteria

`src/step2/geometry_contract.py` must satisfy all of the following:

- exactly one `SegmentBoundaryProfile` dataclass
- exactly one `_project_points_to_polyline` function
- exactly one `_rank_branch_section_candidates` function
- exactly one `_extract_branch_stable_section_boundary` function
- exactly one `_refine_branch_ostium_backward_from_stable_section` function
- exactly one `_refine_branch_boundaries` function
- no duplicate function or class definitions
- no duplicate contradictory algorithm constants
- no code after `return warnings` inside `_refine_branch_boundaries`
- `_refine_branch_boundaries` must call `_extract_branch_stable_section_boundary`
- `_refine_branch_boundaries` must call `_refine_branch_ostium_backward_from_stable_section`
- old `_extract_branch_proximal_boundary` logic must not remain as the active path
- `python -m py_compile src/step2/geometry_contract.py step2_geometry_contract.py` must pass
- AST duplicate-definition check must pass

The intended singleton constants are:

- `PROXIMAL_BOUNDARY_SELECTION_ALGORITHM = "backward_refined_from_stable_section_v1"`
- `STABLE_SECTION_SELECTION_ALGORITHM = "first_stable_surface_ostium_v2"`
- `BACKWARD_OSTIUM_REFINEMENT_STEP_MM = 0.1`
- `BACKWARD_OSTIUM_REFINEMENT_MAX_MM = 4.0`
- `BACKWARD_OSTIUM_DIAMETER_RATIO_STOP = 1.35`
- `BACKWARD_OSTIUM_AREA_RATIO_STOP = 1.80`
- `BACKWARD_OSTIUM_CENTROID_JUMP_RATIO_STOP = 0.50`
- `BACKWARD_OSTIUM_CENTERLINE_DISTANCE_RATIO_STOP = 0.60`
- `BACKWARD_OSTIUM_MIN_ACCEPTED_STEPS_FOR_REFINED = 1`

## Downstream STEP3 Contract

STEP3 should trust STEP2 `proximal_boundary` metadata as the authored vessel-start geometry. STEP3 must preserve the proximal-boundary `source_type`, `confidence`, `requires_review`, and `warnings` in its own contract where relevant.

STEP3 may name, group, and orient vessels, but it should not silently redefine the STEP2 proximal boundary. If STEP3 needs to flag a named priority vessel whose STEP2 boundary is fallback or low confidence, it should return `requires_review`.

## STEP2 TODO Status

- [x] Clean duplicate/corrupt attempted proximal-boundary patch.
- [x] Implement single proximal boundary control flow.
- [x] Add stable-section reference metadata.
- [x] Add backward-refinement metadata.
- [x] Preserve STEP3 compatibility by keeping existing segment summary fields and adding metadata inside `proximal_boundary`.
- [x] Add and run acceptance checks: `py_compile`, AST duplicate-definition check, control-flow text check, resources documentation check.

