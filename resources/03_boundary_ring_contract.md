# Boundary Ring Contract

## Ring Meaning

A boundary ring is an actual circular cut-boundary.

It separates a parent segment from a child segment.

It is not decorative.

Operational visible boundary rings are written to:

```text
outputs/boundary_rings.vtp
```

`boundary_rings.vtp` is a minimal interface output, not a full internal ring dump.

It must be recorded compactly in:

```text
outputs/segmentation_result.json
```

It must be consistent with:

```text
outputs/segmented_surface.vtp
```

## Topology Origin Rule

The topology start is only a search origin.

The branch centerline start/topology split is not the final ostium by default.

It is not acceptable to place the final `branch_start` ring at `segment.points[0]` unless surface-cut validation proves that location is a stable daughter-tube boundary.

topology-only branch_start rings must be requires_review.

A topology-only ring must never be marked:

```text
success
```

## Stable Daughter Section + Backward Refinement

For `branch_start` placement, Codex must not treat the topology start as final.

Branch-start rings must use:

```text
surface_validated_branch_start_ring_v1
```

The algorithm must:

```text
1. Generate candidate cuts along the child centerline.
2. Search distally until a stable daughter-tube section is found.
3. Define stability using compactness, radius plausibility, centroid closeness to centerline, and absence of parent-wall contamination.
4. Use the stable daughter section as the reference diameter/geometry.
5. Walk backward toward the parent.
6. Reject candidates when parent contamination appears, radius jumps, centroid drifts, or cut compactness collapses.
7. Select the most proximal clean candidate before contamination.
8. Mark fallback rings requires_review if no stable section is found.
9. Update surface cell assignment so the segmented surface respects the selected ring.
```

The selected `branch_start` ring must be the earliest stable daughter-boundary after backward refinement, not the first candidate that produces a cut.

Required summary:

```text
stable daughter section first
then backward refinement
then last clean candidate before parent contamination
```

## Rejection Rules

A `branch_start` candidate must be rejected as too proximal if any of the following are true:

```text
the cut includes obvious parent/aortic wall
the cut component is much larger than the stable daughter-tube reference
centroid is far from the branch centerline
radius spread is high
multiple large components are present
compactness is poor
the candidate occurs before the first stable daughter-tube section
using the ring would assign parent wall cells to the child segment
```

A `branch_start` candidate must be rejected as too distal if:

```text
an earlier clean stable candidate exists
the candidate is farther inside the branch without meaningful improvement
radius and compactness already plateaued earlier
```

A ring must be marked `requires_review` if the code cannot prove the selected cut is clean and stable.

## Surface Assignment Contract

The circular ring is not only a visual marker.

The selected ring must affect surface cell assignment.

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring plane beyond tolerance must be reassigned to the parent segment.

The segmented surface must be consistent with the selected boundary ring.

If the surface coloring and the ring disagree, the output must be:

```text
requires_review
```

or:

```text
failed
```

Forbidden state:

```text
boundary_rings.vtp shows a refined ring
segmented_surface.vtp still starts the branch at the old topology point
```

## Ring-To-Color Consistency Contract

Ring-to-color consistency is the primary target.

The `branch_start` ring is the start of the child segment color.

The branch color must begin at the `branch_start` ring.

The segmented surface must not show child color on the parent/proximal side of the selected `branch_start` ring.

The color boundary must not extend proximal to the ring beyond tolerance.

Surface cells on the parent side of the ring plane must be reassigned to parent.

If the child color crosses the ring beyond tolerance, the output status must be:

```text
requires_review
```

or:

```text
failed
```

Ring-plane-gated assignment is required.

Centerline projection alone is insufficient for steep branches because parent-wall cells may project onto the child centerline after the selected offset.

## Ring-Plane-Gated Surface Assignment

Ring-plane-gated surface assignment means:

```text
For each branch_start ring, the ring center and ring normal define a boundary plane.
A child segment cell is valid only if it lies on the distal/child side of that plane within the allowed tolerance.
A child segment cell that lies proximal/parent-side of the ring plane beyond tolerance must be reassigned to the parent segment.
```

This is required because the interface goal is that the visible child color starts at the `branch_start` ring.

The implementation must use ring-plane signed distance and/or clipping/connectivity logic to enforce the boundary.

## Millimetre Tolerance Contract

The code must define and report a ring-plane assignment tolerance.

Recommended first value:

```text
RING_PLANE_ASSIGNMENT_TOLERANCE_MM = 0.20
```

Allowed range for the first implementation:

```text
0.10 mm to 0.30 mm
```

This tolerance controls how close a surface cell center may be to the `branch_start` ring plane before it is treated as ambiguous rather than definitely parent-side or child-side.

The tolerance must be recorded compactly in `segmentation_result.json` metrics or `segmentation_diagnostics.json`.

If cells near the ring plane are ambiguous, the run may be:

```text
requires_review
```

Do not create a new VTP array for tolerance.

Do not create extra debug files for tolerance.

## Surface Connectivity Contract

After ring-plane gating, the child-colored region should be the connected distal child-side component.

If disconnected child-colored patches remain proximal to the ring or isolated around the parent wall, they must be reassigned to parent or the output must be:

```text
requires_review
```

The code may use VTK connectivity logic or its own cell-adjacency connected-component cleanup.

The goal is not just to classify individual cells by nearest centerline.

The goal is a visually continuous child branch segment beginning at the `branch_start` ring.

## Ring Geometry

Each ring must have:

```text
center point
normal vector
radius
ring type
parent segment ID
child segment ID when applicable
confidence
status
warnings
```

The ring must be circular.

A regular polyline approximation is allowed.

Recommended side count:

```text
96
```

Minimum side count:

```text
32
```

## Ring Types

Allowed ring types as internal concepts only:

```text
aortic_body_start
aortic_body_end
branch_start
branch_end
parent_pre_bifurcation
daughter_start
```

Do not introduce anatomy-specific ring types.

Default visible ring types in `outputs/boundary_rings.vtp` are only:

```text
aortic_body_start
aortic_body_end
branch_start
parent_pre_bifurcation only if distinct and useful
```

`branch_end` rings are not written by default unless a future prompt explicitly requests them.

Duplicate `daughter_start` rings are not written by default.

If a `daughter_start` concept is needed for JSON topology, it should reference the corresponding `branch_start` ring ID instead of creating duplicate visible geometry.

Visible duplicate rings must be removed or suppressed when they do not represent distinct boundaries.

The default `boundary_rings.vtp` should be visually uncluttered.

## Ring Radius

Preferred radius rule:

```text
local equivalent diameter / 2
```

Fallback radius rule:

```text
centerline radius
```

If the radius is weakly supported, the ring status must be:

```text
requires_review
```

## Ring Orientation

Required orientation rules when that ring concept is emitted or recorded:

```text
branch_start normal = child branch centerline tangent
branch_end normal = branch centerline tangent near segment end
parent_pre_bifurcation normal = parent segment centerline tangent before the bifurcation
daughter_start normal = daughter segment centerline tangent after the bifurcation
```

## Minimal Ring VTP Contract

`outputs/boundary_rings.vtp` must contain only the arrays listed in `resources/02_target_outputs.md`.

Do not put candidate metrics, raw surface data, or debug metadata into ring VTP arrays.
