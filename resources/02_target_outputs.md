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

Extra VTP arrays are forbidden by default.

A future prompt must explicitly change this contract before any extra VTP arrays are added.

Forbidden arrays include:

```text
ring plane distance
tolerance
clip status
original cell id
debug flags
region id
parent-side violation
connectivity region
candidate metrics
anatomy-name arrays
vessel-name arrays
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

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring plane beyond tolerance must be reassigned to the parent segment, clipped, or counted against success.

If whole-cell recoloring cannot create a clean color boundary at the ring, clipped output still writes to the same `outputs/segmented_surface.vtp` file.

No extra output files are allowed for clipping or tolerance diagnostics.

If `segmented_surface.vtp` and `boundary_rings.vtp` disagree, the output status must be:

```text
requires_review
```

or:

```text
failed
```

## outputs/boundary_rings.vtp

`boundary_rings.vtp` is an interface file.

It is not a full internal ring dump.

By default, this file must contain only circular ring geometry for visible operational `branch_start` boundaries needed to interpret `outputs/segmented_surface.vtp`.

Default visible ring type:

```text
branch_start
```

This is the branch_start-only visible rings contract.

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

Extra ring arrays are forbidden by default.

A future prompt must explicitly change this contract before any extra ring arrays are added.

Do not put candidate metrics into `boundary_rings.vtp`.

Do not put large debug metadata into VTP arrays.

VTP files are for visual inspection and minimum segment/ring selection.

Detailed candidate reasoning belongs only in compact JSON.

Hidden/internal-only ring concepts by default are:

```text
aortic_body_start
aortic_body_end
parent_pre_bifurcation
branch_end
daughter_start
```

`parent_pre_bifurcation` is internal/hidden by default and must not appear in `boundary_rings.vtp` unless a future prompt explicitly requests review/internal ring visualization.

`aortic_body_start` and `aortic_body_end` are internal/hidden by default and must not appear in `boundary_rings.vtp` unless a future prompt explicitly requests them.

`branch_end` rings must not be written to `boundary_rings.vtp` by default.

`daughter_start` rings must not be written to `boundary_rings.vtp` by default.

If `daughter_start` topology is needed, it must reference an existing `branch_start` ring ID in JSON instead of creating visible duplicate geometry.

Every branch segment must have exactly one visible `branch_start` ring unless the `branch_start` is invalid and intentionally suppressed with `requires_review`.

`visible_ring_count` must equal `branch_start_ring_count` by default.

If `visible_ring_count` is greater than `branch_start_ring_count`, the output is not minimal and must be `requires_review` or `failed`.

The visible ring count must be checked against the number of visible branch starts. A mismatch means the visual interface is cluttered and must not be marked success.

A future prompt may explicitly request review rings or hidden/internal ring visualization, but that is not allowed by default.

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
zero_offset_proof_passed
cells_reassigned_to_parent_count
candidate_summary
```

Allowed compact surface assignment consistency metadata for each `branch_start` ring:

```text
surface_assignment_mode
ring_plane_assignment_tolerance_mm
clip_boundary_used
clip_boundary_required
clip_boundary_unresolved
cells_reassigned_to_parent_count
cells_reassigned_to_child_count
cells_ambiguous_near_ring_count
ring_plane_parent_side_violation_count
neighbor_contour_leak_count
ring_surface_consistency_status
```

Allowed values for `surface_assignment_mode`:

```text
ring_plane_gated
ring_plane_gated_with_connectivity_cleanup
clipped_ring_boundary
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
non_branch_start_visible_ring_count
hidden_or_suppressed_duplicate_ring_count
ring_plane_assignment_tolerance_mm
ring_plane_parent_side_violation_total
ambiguous_near_ring_total
high_ambiguity_ring_count
clip_boundary_used
clip_boundary_required_count
clip_boundary_unresolved_count
segments_with_color_crossing_ring_count
segments_with_neighbor_contour_leak_count
surface_cell_count
unassigned_cell_count
requires_review_ring_count
failed_ring_count
```
