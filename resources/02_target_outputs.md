# Target Outputs

## Minimal Output Contract

Required output files only:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

Allowed optional diagnostic file:

```text
outputs/segmentation_diagnostics.json
```

The diagnostic file is allowed only if compact and bounded by `resources/05_validation_and_iteration.md`.

No new required output files are allowed unless a future prompt explicitly requests them.

Forbidden extra output files include:

```text
candidate_rings.vtp
candidate_cuts.vtp
debug_surface_sections.vtp
all_candidates.vtp
raw_cut_components.vtp
branch_clouds.vtp
ostium_debug.vtp
```

## Inputs

Inputs must remain in:

```text
inputs/
```

Do not edit input files to satisfy this contract.

## outputs/segmented_surface.vtp

This file must contain the full vascular surface.

Required and allowed cell-data arrays are exactly:

```text
SegmentId
SegmentLabel
SegmentColor
```

No extra cell arrays are allowed unless a future prompt explicitly requests them.

Forbidden arrays include:

```text
anatomy-name arrays
vessel-name arrays
debug arrays
candidate arrays
surface-cut metric arrays
```

The only anatomical value allowed in `SegmentLabel` is:

```text
aortic_body
```

All other segment labels must be anonymous:

```text
branch_001
branch_002
branch_003
```

The selected circular ring must affect surface cell assignment.

The branch color must begin at the `branch_start` ring.

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring plane beyond tolerance must be reassigned to the parent segment.

If `segmented_surface.vtp` and `boundary_rings.vtp` disagree, the output status must be:

```text
requires_review
```

or:

```text
failed
```

## outputs/boundary_rings.vtp

`boundary_rings.vtp` is a minimal interface output.

It is not a full internal debug dump.

This file must contain only circular ring geometry for operational boundaries needed to interpret `outputs/segmented_surface.vtp`.

Required and allowed cell-data arrays are exactly:

```text
RingId
RingLabel
RingType
ParentSegmentId
ChildSegmentId
SegmentId
RadiusMm
Confidence
Status
```

No extra ring arrays are allowed unless a future prompt explicitly requests them.

Do not put candidate metrics into `boundary_rings.vtp`.

Do not put large debug metadata into VTP arrays.

VTP files are for visual inspection and minimum segment/ring selection.

Detailed candidate reasoning belongs only in compact JSON.

Default visible ring types are:

```text
aortic_body_start
aortic_body_end
branch_start
parent_pre_bifurcation only if distinct and useful
```

Hidden/internal-only ring concepts by default are:

```text
branch_end
duplicate daughter_start
```

`branch_start` rings are required and visible.

`aortic_body_start` and `aortic_body_end` may remain visible.

`parent_pre_bifurcation` rings may remain visible only when they represent distinct useful parent boundaries.

`branch_end` rings are not written by default unless a future prompt explicitly requests them.

Duplicate `daughter_start` rings are not written by default.

If a `daughter_start` concept is needed for JSON topology, it should reference the corresponding `branch_start` ring ID instead of creating duplicate visible geometry.

The expected visible ring count is roughly:

```text
visible branch_start rings
+ aortic_body_start/end
+ distinct parent_pre_bifurcation rings if retained
```

It must not include duplicate `daughter_start` rings or routine `branch_end` rings.

## outputs/segmentation_result.json

This file must contain compact decision metadata.

Required top-level keys are exactly:

```text
status
inputs
outputs
segments
boundary_rings
bifurcations
warnings
metrics
```

No additional required top-level keys are allowed unless a future prompt explicitly requests them.

## Segment Objects

Segment objects must contain only:

```text
segment_id
segment_label
segment_type
parent_segment_id
child_segment_ids
proximal_ring_id
distal_ring_ids
cell_count
status
warnings
```

## Boundary Ring Objects

Boundary ring objects must contain the core ring fields plus compact selection metadata.

Core fields:

```text
ring_id
ring_label
ring_type
center_xyz
normal_xyz
radius_mm
source_segment_id
parent_segment_id
child_segment_id
source_centerline_s_mm
orientation_rule
radius_rule
confidence
status
warnings
```

Allowed compact selection metadata:

```text
selection_algorithm
topology_start_xyz
selected_offset_mm
search_max_mm
candidate_count
accepted_candidate_count
selected_candidate_classification
selected_radius_rule
surface_cut_used
backward_refinement_used
cells_reassigned_to_parent_count
candidate_summary
```

Allowed compact surface assignment consistency metadata for each `branch_start` ring:

```text
surface_assignment_mode
ring_plane_assignment_tolerance_mm
cells_reassigned_to_parent_count
cells_reassigned_to_child_count
cells_ambiguous_near_ring_count
ring_plane_parent_side_violation_count
ring_surface_consistency_status
```

Allowed values for `surface_assignment_mode`:

```text
ring_plane_gated
ring_plane_gated_with_connectivity_cleanup
projection_only_requires_review
```

If the code still uses projection-only assignment, the ring status or overall status must be:

```text
requires_review
```

Required branch-start selection algorithm value:

```text
surface_validated_branch_start_ring_v1
```

Allowed compact `candidate_summary` fields:

```text
first_stable_offset_mm
selected_offset_mm
rejected_too_proximal_count
invalid_cut_count
fallback_used
```

Do not store raw candidate points.

Do not store raw cut contour points.

Do not store every plane-cut component.

Do not store giant debug arrays.

Do not store candidate metric arrays by default.

## Bifurcation Objects

Bifurcation objects must remain compact:

```text
bifurcation_id
bifurcation_label
parent_segment_id
child_segment_ids
parent_pre_bifurcation_ring_id
daughter_start_ring_ids
status
warnings
```

If `daughter_start_ring_ids` would duplicate visible `branch_start` ring geometry, they must reference the corresponding `branch_start` ring IDs instead of requiring separate visible rings.

## Metrics

`metrics` must contain compact run-level counts only.

Allowed metrics include:

```text
segment_count
branch_count
bifurcation_count
ring_count
visible_ring_count
branch_start_ring_count
hidden_or_suppressed_duplicate_ring_count
ring_plane_assignment_tolerance_mm
ring_plane_parent_side_violation_total
ambiguous_near_ring_total
segments_with_color_crossing_ring_count
surface_cell_count
unassigned_cell_count
requires_review_ring_count
failed_ring_count
```
