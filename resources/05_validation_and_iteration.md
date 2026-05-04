# Validation And Iteration

## Purpose

Every run must validate only the minimal outputs.

The main visual inspection target is:

```text
branch_start-only visible rings
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

Extra output files are forbidden by default.

A future prompt must explicitly change this contract before any extra output files are added.

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

Extra VTP arrays are forbidden by default.

A future prompt must explicitly change this contract before any extra VTP arrays are added.

Forbidden VTP arrays include:

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
```

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

If `selected_offset_mm` is `0.0`, the JSON must prove surface-cut validation at that location through `zero_offset_proof_passed = true` or the ring must be:

```text
requires_review
```

Unproven zero-offset `branch_start` rings are not clean operational rings.

Unproven zero-offset rings must be suppressed from default `boundary_rings.vtp` and reported in JSON.

The topology start is only a search origin.

Topology-only `branch_start` rings must be `requires_review`.

## Visible Ring Minimalism Check

`boundary_rings.vtp` is an interface file.

By default, `boundary_rings.vtp` must contain only `RingType = branch_start`.

`visible_ring_count` must equal `branch_start_ring_count`.

`non_branch_start_visible_ring_count` must be `0`.

If `visible_ring_count` is greater than `branch_start_ring_count`, the run must be:

```text
requires_review
```

or:

```text
failed
```

No `parent_pre_bifurcation` rings are visible by default.

No `aortic_body_start` or `aortic_body_end` rings are visible by default.

No `branch_end` rings are visible by default.

No `daughter_start` rings are visible by default.

The visible ring count must be checked against the number of visible branch starts. A mismatch means the visual interface is cluttered and must not be marked success.

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

For each visible `branch_start` ring, validate that child `SegmentColor` begins at that ring.

Check that child color does not extend proximal to the ring plane by more than:

```text
RING_PLANE_ASSIGNMENT_TOLERANCE_MM = 0.10
```

Allowed tolerance range:

```text
0.05 mm to 0.15 mm
```

Any tolerance above `0.15 mm` must force `requires_review` unless explicitly justified.

No branch color may extend beyond the branch_start ring by more than `0.10 mm`.

No child-colored cells may remain on parent/aortic or neighboring vessel contour when status is `success`.

Surface cells on the parent side of the ring plane must be reassigned to parent, clipped, or counted against success.

Check:

```text
ring_plane_parent_side_violation_count
cells_ambiguous_near_ring_count
segments_with_color_crossing_ring_count
segments_with_neighbor_contour_leak_count
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

If visual boundary cannot be clean with recoloring, `vtkClipPolyData` or equivalent clipping is required.

If clipping is not implemented and visual mismatch remains, status must be `requires_review`.

## Ambiguous Near-Ring Check

Ambiguous near-ring cells are not harmless.

`ambiguous_near_ring_total` must be reported.

`ambiguous_near_ring_total` must not be high if status is `success`.

`high_ambiguity_ring_count` must be reported.

If `cells_ambiguous_near_ring_count` is greater than 5% of the child segment cell count, that branch ring must be `requires_review` unless clipping resolves the ambiguity.

The code must not report `ring_surface_consistency_status = success` while thousands of cells remain ambiguous around that ring unless those cells were clipped/resolved or explicitly justified.

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
zero_offset_unproven_ring_count
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
non_branch_start_visible_ring_count
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
high_ambiguity_ring_count
clip_boundary_used
clip_boundary_required_count
clip_boundary_unresolved_count
segments_with_color_crossing_ring_count
segments_with_neighbor_contour_leak_count
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
zero_offset_proof_passed
clip_boundary_used
clip_boundary_required
clip_boundary_unresolved
status
classification
confidence
candidate_count
warning_count
cells_reassigned_to_parent_count
cells_reassigned_to_child_count
cells_ambiguous_near_ring_count
ring_plane_parent_side_violation_count
neighbor_contour_leak_count
ring_surface_consistency_status
```

`segmentation_diagnostics.json` should include compact aggregate values:

```text
visible_ring_count
branch_start_ring_count
non_branch_start_visible_ring_count
ring_plane_assignment_tolerance_mm
ambiguous_near_ring_total
high_ambiguity_ring_count
clip_boundary_used
clip_boundary_required_count
clip_boundary_unresolved_count
segments_with_color_crossing_ring_count
segments_with_neighbor_contour_leak_count
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
branch_start-only visible rings, ring-plane-gated or clipped surface assignment, and child vessel contour consistency
```

Do not add output files, VTP arrays, per-cell diagnostics, broad helpers, or architecture while validating this target.
