# Validation And Iteration

## Purpose

Every run must validate only the minimal outputs.

The main visual inspection target is:

```text
ring-to-color consistency
branch color must begin at the branch_start ring
```

Do not add debug outputs to compensate for uncertain geometry.

When uncertain, mark:

```text
requires_review
```

## Required Output Check

Required files:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

Optional compact diagnostic file:

```text
outputs/segmentation_diagnostics.json
```

If any required output is missing, the run is:

```text
failed
```

## VTP Array Check

`outputs/segmented_surface.vtp` must contain only these required cell arrays:

```text
SegmentId
SegmentLabel
SegmentColor
```

`outputs/boundary_rings.vtp` must contain only these required cell arrays:

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

No extra VTP arrays are allowed unless a future prompt explicitly requests them.

## JSON Check

`outputs/segmentation_result.json` must contain only these required top-level keys:

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

JSON must be compact and bounded.

Do not include raw surface points, raw candidate points, raw cut contour points, every plane-cut component, large candidate lists, or debug dumps.

## Branch-Start Refinement Check

Every `branch_start` ring must use:

```text
surface_validated_branch_start_ring_v1
```

The output must contain:

```text
stable daughter section
backward refinement
```

as the required strategy in resource-controlled metadata or warnings when relevant.

If topology fallback is used, the ring must be:

```text
requires_review
```

If `selected_offset_mm` is `0.0`, the JSON must prove surface-cut validation at that location or the ring must be:

```text
requires_review
```

The topology start is only a search origin.

topology-only branch_start rings must be requires_review.

## Ring And Surface Agreement

If `boundary_rings.vtp` shows a refined ring but `segmented_surface.vtp` still starts the branch at the old topology point, the run must be:

```text
requires_review
```

or:

```text
failed
```

If surface coloring and ring placement disagree, the run must be:

```text
requires_review
```

or:

```text
failed
```

For each `branch_start` ring, validate that child `SegmentColor` begins at that ring.

Check that child color does not extend proximal to the ring plane beyond tolerance.

Surface cells on the parent side of the ring plane must be reassigned to parent.

Check:

```text
ring_plane_parent_side_violation_count
cells_ambiguous_near_ring_count
```

If any branch has color crossing the ring, status must be:

```text
requires_review
```

or:

```text
failed
```

If projection-only assignment is still used, status must be:

```text
requires_review
```

## Visible Ring Minimalism Check

`boundary_rings.vtp` ring count should be minimal.

`boundary_rings.vtp` is a minimal interface output.

Default visible rings should be only:

```text
aortic_body_start
aortic_body_end
branch_start
parent_pre_bifurcation only if distinct and useful
```

`branch_end` rings are not written by default.

Duplicate `daughter_start` rings are not written by default.

`branch_end` rings should not be visible by default.

Duplicate `daughter_start` rings should not be visible by default.

The visible ring count should roughly equal:

```text
branch_start_ring_count
+ aortic_body_start/end
+ distinct parent_pre_bifurcation rings if retained
```

## Diagnostics Contract

If `outputs/segmentation_diagnostics.json` exists, it must be compact.

Allowed top-level diagnostic sections only:

```text
status
outputs_exist
vtp_arrays
labels
visible_rings
branch_start_refinement
surface_assignment_consistency
warnings
failures
next_recommended_focus
```

Allowed `branch_start_refinement` fields:

```text
algorithm
branch_count
refined_ring_count
topology_fallback_ring_count
stable_candidate_found_count
parent_contamination_detected_count
cells_reassigned_to_parent_total
rings_requiring_review
per_ring_summary
```

Allowed `visible_rings` fields:

```text
visible_ring_count
branch_start_ring_count
hidden_or_suppressed_duplicate_ring_count
branch_end_rings_suppressed
duplicate_daughter_start_rings_suppressed
```

Allowed `surface_assignment_consistency` fields:

```text
surface_assignment_mode
ring_plane_assignment_tolerance_mm
ring_plane_parent_side_violation_total
ambiguous_near_ring_total
segments_with_color_crossing_ring_count
cells_reassigned_to_parent_total
cells_reassigned_to_child_total
rings_requiring_review
per_ring_summary
```

Each `per_ring_summary` item must contain only:

```text
ring_id
segment_id
segment_label
surface_assignment_mode
ring_plane_assignment_tolerance_mm
selected_offset_mm
status
classification
confidence
candidate_count
warning_count
cells_reassigned_to_parent_count
cells_reassigned_to_child_count
cells_ambiguous_near_ring_count
ring_plane_parent_side_violation_count
ring_surface_consistency_status
```

`segmentation_diagnostics.json` should include compact aggregate values:

```text
visible_ring_count
branch_start_ring_count
hidden_or_suppressed_duplicate_ring_count
ring_plane_assignment_tolerance_mm
ring_plane_parent_side_violation_total
ambiguous_near_ring_total
segments_with_color_crossing_ring_count
```

Do not add per-cell diagnostics.

Diagnostics must not include raw surface points, raw candidate points, raw cut contour points, raw candidate contours, large arrays, or plane-cut component dumps.

## Label Check

The only anatomical/readable label allowed is:

```text
aortic_body
```

All branch, bifurcation, and ring labels must be anonymous.

If output labels violate this rule, the run is:

```text
failed
```

## Iteration Rule

Every code iteration must target the smallest specific failure.

For the next code prompt, the only allowed target is:

```text
ring-plane-gated surface assignment and visible ring minimalism
```

Do not add output files, VTP arrays, per-cell diagnostics, broad helpers, or architecture while validating this target.
