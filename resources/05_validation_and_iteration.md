# Validation And Iteration

## Purpose

Every run must validate only the minimal outputs.

The main visual inspection target is:

```text
branch does not start on parent/aortic wall
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

## Diagnostics Contract

If `outputs/segmentation_diagnostics.json` exists, it must be compact.

Allowed top-level diagnostic sections only:

```text
status
outputs_exist
vtp_arrays
labels
branch_start_refinement
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

Each `per_ring_summary` item must contain only:

```text
ring_id
segment_id
segment_label
selected_offset_mm
status
classification
confidence
candidate_count
warning_count
cells_reassigned_to_parent_count
```

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
surface-validated branch_start ring selection and surface assignment correction
```

Do not add outputs, new schemas, broad helpers, or architecture while validating this target.
