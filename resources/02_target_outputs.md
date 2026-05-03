# Target Outputs

## Required Input Folder

All input artifacts must live in:

```text
inputs/
```

Required inputs:

```text
inputs/surface_cleaned.vtp
inputs/centerline_network.vtp
inputs/centerline_network_metadata.json
inputs/input_roles.json
```

## Required Input Role File

The input role file must be neutral.

It must not contain named branch labels.

Expected structure:

```json
{
  "aortic_body": {
    "inlet_face_id": 2
  },
  "terminal_faces": [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12
  ],
  "rules": {
    "only_named_segment": "aortic_body",
    "all_other_segments_are_anonymous": true,
    "branch_label_prefix": "branch_",
    "bifurcation_label_prefix": "bifurcation_",
    "ring_label_prefix": "ring_"
  }
}
```

The code may use face IDs, terminal IDs, graph topology, and centerline routes.

The code must not require named branch labels.

## Required Output Folder

All generated outputs must be written to:

```text
outputs/
```

Required outputs:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

## outputs/segmented_surface.vtp

This file must contain the full vascular surface.

It must include cell-data arrays:

```text
SegmentId
SegmentLabel
SegmentColor
```

Required meaning:

```text
SegmentId = integer segment identifier
SegmentLabel = aortic_body or anonymous branch label
SegmentColor = RGB color used for ParaView visualization
```

The only readable anatomical label allowed in `SegmentLabel` is:

```text
aortic_body
```

All non-aortic-body labels must be anonymous:

```text
branch_001
branch_002
branch_003
```

The segmented surface must be directly openable in ParaView.

The surface should be visually separable by `SegmentColor` or `SegmentId`.

## outputs/boundary_rings.vtp

This file must contain circular ring geometry.

The rings represent actual cut-boundaries.

This file must include cell-data arrays:

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

Expected ring labels:

```text
ring_001
ring_002
ring_003
```

Expected ring types:

```text
aortic_body_start
aortic_body_end
branch_start
branch_end
parent_pre_bifurcation
daughter_start
```

For every branch segment, there should be a branch-start ring.

For every bifurcation, the preferred representation is:

```text
one parent_pre_bifurcation ring
one daughter_start ring for each daughter segment
```

The geometric distance between parent pre-bifurcation rings and daughter-start rings must be preserved when measurable.

Do not collapse all rings into the same point unless the geometry genuinely has no measurable separation.

## outputs/segmentation_result.json

This JSON must describe the result in a machine-readable way.

Required top-level structure:

```json
{
  "status": "success | requires_review | failed",
  "inputs": {},
  "outputs": {},
  "segments": [],
  "boundary_rings": [],
  "bifurcations": [],
  "warnings": [],
  "metrics": {}
}
```

## Segment Object Contract

Each segment object should include:

```json
{
  "segment_id": 1,
  "segment_label": "aortic_body",
  "segment_type": "aortic_body | branch",
  "parent_segment_id": null,
  "child_segment_ids": [],
  "proximal_ring_id": null,
  "distal_ring_ids": [],
  "cell_count": 0,
  "status": "success | requires_review | failed",
  "warnings": []
}
```

For non-aortic-body branches:

```json
{
  "segment_id": 2,
  "segment_label": "branch_001",
  "segment_type": "branch",
  "parent_segment_id": 1,
  "child_segment_ids": [],
  "proximal_ring_id": 1,
  "distal_ring_ids": [],
  "cell_count": 0,
  "status": "success",
  "warnings": []
}
```

## Boundary Ring Object Contract

Each ring object should include:

```json
{
  "ring_id": 1,
  "ring_label": "ring_001",
  "ring_type": "branch_start",
  "center_xyz": [0.0, 0.0, 0.0],
  "normal_xyz": [0.0, 0.0, 1.0],
  "radius_mm": 1.0,
  "source_segment_id": 2,
  "parent_segment_id": 1,
  "child_segment_id": 2,
  "source_centerline_s_mm": 0.0,
  "orientation_rule": "perpendicular_to_child_centerline_tangent",
  "radius_rule": "local_equivalent_diameter_over_2",
  "confidence": 1.0,
  "status": "success",
  "warnings": []
}
```

## Bifurcation Object Contract

Each bifurcation object should include:

```json
{
  "bifurcation_id": 1,
  "bifurcation_label": "bifurcation_001",
  "parent_segment_id": 1,
  "child_segment_ids": [2, 3],
  "parent_pre_bifurcation_ring_id": 4,
  "daughter_start_ring_ids": [5, 6],
  "status": "success | requires_review | failed",
  "warnings": []
}
```

## Output Status Rules

Use:

```text
success
```

only when outputs are written and all required segment/ring information is available.

Use:

```text
requires_review
```

when the code writes outputs but one or more rings are uncertain, low-confidence, visually questionable, or geometrically ambiguous.

Use:

```text
failed
```

when required outputs cannot be written, required inputs are missing, the surface cannot be segmented, or the result is not usable.

## Required Metrics

The JSON should include metrics such as:

```json
{
  "segment_count": 0,
  "branch_count": 0,
  "bifurcation_count": 0,
  "ring_count": 0,
  "surface_cell_count": 0,
  "unassigned_cell_count": 0,
  "requires_review_ring_count": 0,
  "failed_ring_count": 0
}
```

## Forbidden Output Concepts

The output must not contain old named branch labels.

The output must not contain clinical labels.

The output must not contain downstream workflow fields.

The output must remain focused on:

```text
aortic_body
anonymous branch segments
bifurcations
boundary rings
surface segmentation
validation status
```