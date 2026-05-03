# Boundary Ring Contract

## Ring Meaning

A boundary ring is an actual circular cut-boundary.

It separates a parent segment from a child segment.

It is not decorative.

It must be visible in:

```text
outputs/boundary_rings.vtp
```

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

Surface cells associated with a child branch but lying proximal to the selected `branch_start` ring must be reassigned to the parent segment.

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

Allowed ring types only:

```text
aortic_body_start
aortic_body_end
branch_start
branch_end
parent_pre_bifurcation
daughter_start
```

Do not introduce anatomy-specific ring types.

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

Required orientation rules:

```text
branch_start normal = child branch centerline tangent
branch_end normal = branch centerline tangent near segment end
parent_pre_bifurcation normal = parent segment centerline tangent before the bifurcation
daughter_start normal = daughter segment centerline tangent after the bifurcation
```

## Minimal Ring VTP Contract

`outputs/boundary_rings.vtp` must contain only the arrays listed in `resources/02_target_outputs.md`.

Do not put candidate metrics, raw surface data, or debug metadata into ring VTP arrays.
