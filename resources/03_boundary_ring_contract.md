# Boundary Ring Contract

## Purpose

Boundary rings are the central object in this repository.

A boundary ring is a clean circular cut-boundary that separates a parent segment from a child segment.

The ring is not decorative.

The ring is the simplified geometric representation of the actual cut used to separate surface cells.

The ring is also the reviewable geometry object that makes the segmentation understandable in ParaView.

## Required Ring Geometry

Each ring must be represented by:

```text
center point
normal vector
radius
ring type
parent segment ID
child segment ID, if applicable
confidence
status
warnings
```

The ring must be visualized as circular polyline geometry in:

```text
outputs/boundary_rings.vtp
```

The same ring must be recorded in:

```text
outputs/segmentation_result.json
```

## Ring Labeling

Ring labels must be anonymous.

Allowed pattern:

```text
ring_001
ring_002
ring_003
```

Do not use clinical or vessel-name ring labels.

## Ring Types

Allowed ring types:

```text
aortic_body_start
aortic_body_end
branch_start
branch_end
parent_pre_bifurcation
daughter_start
```

Do not introduce named-branch ring types.

## Ring Radius

Preferred rule:

```text
radius = local equivalent diameter / 2
```

If the local equivalent diameter cannot be measured reliably, fallback to:

```text
radius = centerline radius
```

If no reliable radius is available, the ring may be estimated, but it must be marked:

```text
requires_review
```

The radius must be recorded in millimetres as:

```text
radius_mm
```

## Ring Orientation

For branch-start rings:

```text
normal = child branch centerline tangent at estimated branch start
```

The ring plane is perpendicular to this tangent.

For branch-end rings:

```text
normal = branch centerline tangent near segment end
```

For parent pre-bifurcation rings:

```text
normal = parent segment centerline tangent before the bifurcation
```

For daughter-start rings:

```text
normal = daughter segment centerline tangent after the bifurcation
```

## Ring Center

The preferred center is:

```text
the centerline point at the estimated boundary location
```

The center may be adjusted using cross-section centroid information only if the adjustment improves boundary placement.

If adjusted, the JSON must record:

```text
original_centerline_center_xyz
adjusted_center_xyz
center_adjustment_reason
```

## Aortic Body Rings

The aortic body should have:

```text
aortic_body_start ring
aortic_body_end ring
```

These rings define the beginning and end of the aortic body.

## Branch Rings

Every branch segment should have:

```text
branch_start ring
```

A branch may also have:

```text
branch_end ring
```

if its distal endpoint is a terminal cap or a bifurcation transition.

## Bifurcation Rings

At a bifurcation, do not collapse the parent ring and daughter rings into one location by default.

Preferred representation:

```text
parent segment:
  parent_pre_bifurcation ring

daughter segments:
  daughter_start ring for each daughter
```

The distance between these rings matters.

If the distance is measurable, preserve it.

If the distance is effectively zero, record that explicitly.

## Parent/Child Separation

The ring must separate parent and child surface assignment.

A child segment must not include obvious parent wall.

A parent segment must not absorb the true beginning of a child branch.

The ring should be placed as close as possible to the true branch origin while remaining stable and circular.

## Surface And Ring Acceptance

This isolated geometry segmentation branch intentionally avoids VMTK branch tooling and VMTK compiled wrappers. It uses VTK + NumPy + input centerline/surface artifacts.

A branch-start ring is acceptable only if it satisfies all of the following:

```text
it is near the branch origin
it does not include obvious parent wall inside the child segment
it does not start too far distal inside the daughter branch
its normal follows the child centerline tangent
its radius is consistent with local branch size
its position is consistent with surface-cut evidence and parent-child cell assignment
```

If surface evidence and the ring candidate disagree, the ring should be marked:

```text
requires_review
```

unless the code can refine the ring to a better position.

The final authority is whether the circular ring correctly separates parent and child surface geometry.

A ring must not be accepted only because it is located at a centerline graph node.

The ring must be checked against:

```text
actual surface geometry
local branch direction
local branch size
parent-child cell assignment
visible boundary-ring placement
```

## Circularity Requirement

The output ring should be circular, not jagged.

A regular polygon/polyline approximation is acceptable.

Recommended:

```text
number_of_sides = 96
```

Lower bound:

```text
number_of_sides >= 32
```

The ring should appear circular in ParaView.

## Boundary Placement Failure Modes

A ring is invalid or requires review if:

```text
ring is missing
ring is not circular
ring is not visible in boundary_rings.vtp
ring is placed too far into the parent
ring is placed too far into the child
ring angle does not follow segment direction
ring radius is much too large
ring radius is much too small
ring does not correspond to the surface split
ring does not appear in segmentation_result.json
ring was accepted only from a centerline node
ring was accepted without surface validation
```

## Quality Fields

Every ring must have:

```text
confidence
status
warnings
```

Recommended confidence interpretation:

```text
0.90-1.00 = high confidence
0.70-0.89 = acceptable
0.50-0.69 = requires review
0.00-0.49 = failed or unreliable
```

## Status Values

Allowed ring status values:

```text
success
requires_review
failed
```

## Cut-Boundary Meaning

The ring is not just a display object.

The ring defines the cut plane/boundary used to separate parent and child segment surface cells.

The surface segmentation must be consistent with the ring.

If the surface cell split and the ring disagree, the output requires review.
