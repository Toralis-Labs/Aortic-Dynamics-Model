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

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring must be reassigned to the parent segment.

If `segmented_surface.vtp` and `boundary_rings.vtp` disagree, the output status must be:

```text
requires_review
```

or:

```text
failed
```

## outputs/boundary_rings.vtp

This file must contain only circular ring geometry for actual cut-boundaries.

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

## Metrics

`metrics` must contain compact run-level counts only.

Allowed metrics include:

```text
segment_count
branch_count
bifurcation_count
ring_count
surface_cell_count
unassigned_cell_count
requires_review_ring_count
failed_ring_count
```
