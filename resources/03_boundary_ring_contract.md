# Boundary Ring Contract

## Ring Meaning

A boundary ring is an actual circular cut-boundary.

It separates a parent segment from a child segment.

It is not decorative.

Operational visible boundary rings are written to:

```text
outputs/boundary_rings.vtp
```

`boundary_rings.vtp` is an interface file.

It is not a full internal ring dump.

By default, `boundary_rings.vtp` must contain only visible operational `branch_start` rings.

It must be recorded compactly in:

```text
outputs/segmentation_result.json
```

It must be consistent with:

```text
outputs/segmented_surface.vtp
```

## Default Visible Ring Contract

The default visible ring type is:

```text
branch_start
```

This is the branch_start-only visible rings contract.

Every branch segment must have exactly one visible `branch_start` ring unless the `branch_start` is invalid and intentionally suppressed with `requires_review`.

`visible_ring_count` must equal `branch_start_ring_count` by default.

If `visible_ring_count` is greater than `branch_start_ring_count`, the output is not minimal and must be `requires_review` or `failed`.

The visible ring count must be checked against the number of visible branch starts. A mismatch means the visual interface is cluttered and must not be marked success.

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

A future prompt may explicitly request review rings or hidden/internal ring visualization, but that is not allowed by default.

## Topology Origin Rule

The topology start is only a search origin.

The branch centerline start/topology split is not the final ostium by default.

It is not acceptable to place the final `branch_start` ring at `segment.points[0]` unless surface-cut validation proves that location is a stable daughter-tube boundary.

Topology-only `branch_start` rings must be `requires_review`.

A topology-only ring must never be marked:

```text
success
```

Unproven zero-offset `branch_start` rings are not clean operational rings.

If a `branch_start` ring is selected at `0.0 mm` and `zero_offset_proof_passed` is false, it must not be shown as a clean operational interface ring by default.

It must be suppressed from `boundary_rings.vtp` and reported as `requires_review` in JSON unless a future prompt explicitly asks for review rings.

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

Required summary:

```text
stable daughter section first
then backward refinement
then last clean candidate before parent contamination
```

## Ring-To-Color Consistency Contract

Ring-to-color consistency is the primary target.

The `branch_start` ring is the start of the child segment color.

The branch color must begin at the `branch_start` ring.

The segmented surface must not show child color on the parent/proximal side of the selected `branch_start` ring.

The color boundary must not extend proximal to the ring beyond tolerance.

Surface cells on the parent side of the ring plane must be reassigned to parent, clipped, or counted against success.

If the child color crosses the ring beyond tolerance, the output status must be:

```text
requires_review
```

or:

```text
failed
```

Ring-plane-gated assignment is required.

Centerline projection alone is insufficient for final surface coloring.

## Ring-Plane-Gated Surface Assignment

Ring-plane-gated surface assignment means:

```text
For each branch_start ring, the ring center and ring normal define a boundary plane.
A child segment cell is valid only if it lies on the distal/child side of that plane within the allowed tolerance.
A child segment cell that lies proximal/parent-side of the ring plane beyond tolerance must be reassigned to the parent segment, clipped, or counted against success.
```

The implementation must use ring-plane signed distance and/or clipping/connectivity logic to enforce the boundary.

## Clipped Ring Boundary Contract

The clipped ring boundary contract means the `branch_start` ring is not only a drawn circle and not only a cell-center recoloring threshold.

The selected `branch_start` ring defines a geometric boundary plane.

The current mesh may contain cells that cross the selected `branch_start` ring plane.

Cell recoloring is not enough if the visual boundary is wrong.

If existing mesh cells cross that boundary plane, cell-center recoloring may not be enough to make a clean visual interface.

When ring-plane-gated recoloring cannot make the visible `SegmentColor` boundary match the `branch_start` ring within tolerance, the implementation must use `vtkClipPolyData` or equivalent polygonal clipping/splitting logic to create a clean cut boundary.

`vtkClipPolyData` is allowed and preferred for this visual-boundary problem because it can cut polygonal cells, not merely recolor whole cells.

Preferred implementation concept:

```text
1. Use the branch_start ring center and normal to define a vtkPlane.
2. Clip or split child/parent surface regions with vtkClipPolyData or equivalent logic.
3. Keep the child-side clipped surface connected to the distal child branch.
4. Reassign or discard parent-side child-colored fragments.
5. Preserve only the minimal SegmentId, SegmentLabel, and SegmentColor arrays.
6. Do not create extra debug VTP files.
```

## Branch Contour Consistency Contract

A child segment is not valid merely because its cells are distal to the `branch_start` ring plane.

The color should follow the child vessel contour, not surrounding vessels.

Cells should remain child-colored only if they satisfy all of the following:

```text
1. They are on the child/distal side of the branch_start ring plane.
2. They are connected to the distal child branch component.
3. They are spatially consistent with the child branch corridor.
4. They lie within the expected local branch radius/diameter envelope plus tolerance.
5. They are not parent/aortic wall or neighboring-branch surface.
```

If branch color extends onto parent/aortic wall or neighboring branch contours, the `ring_surface_consistency_status` must not be `success`.

## Ambiguous Near-Ring Cell Contract

Ambiguous near-ring cells are not harmless.

They represent uncertainty at the interface.

Required rules:

```text
1. ambiguous_near_ring_total must be reported.
2. ambiguous_near_ring_total must be reduced where possible by clipping/splitting or conservative reassignment.
3. A high ambiguous_near_ring_total must force requires_review.
4. Per-ring cells_ambiguous_near_ring_count must be compared to branch cell count or local ring-neighborhood cell count.
5. If a branch has a large ambiguous near-ring count, that branch ring must not be success.
6. The code must not report ring_surface_consistency_status = success while thousands of cells remain ambiguous around that ring unless those cells were clipped/resolved or explicitly justified.
```

If `cells_ambiguous_near_ring_count` is greater than 5% of the child segment cell count, that `branch_start` ring must be `requires_review` unless clipping resolves the ambiguity.

Do not add per-cell diagnostics.

Only compact counts are allowed.

## Millimetre Tolerance Contract

The code must define and report a ring-plane assignment tolerance.

Default target value:

```text
RING_PLANE_ASSIGNMENT_TOLERANCE_MM = 0.10
```

Allowed range:

```text
0.05 mm to 0.15 mm
```

Any tolerance above `0.15 mm` must force `requires_review` unless explicitly justified.

This tolerance controls how close a surface cell center may be to the `branch_start` ring plane before it is treated as ambiguous rather than definitely parent-side or child-side.

Ambiguous near-ring cells must not be allowed to silently remain child-colored. They must be clipped, reassigned, or counted against success.

The tolerance must be recorded compactly in `segmentation_result.json` metrics or `segmentation_diagnostics.json`.

Do not create a new VTP array for tolerance.

Do not create extra debug files for tolerance.

## Surface Connectivity Contract

After ring-plane gating or clipping, the child-colored region should be the connected distal child-side component.

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

Only `branch_start` is visible by default in `outputs/boundary_rings.vtp`.

Visible duplicate rings must be removed or suppressed when they do not represent operational `branch_start` boundaries.

The default `boundary_rings.vtp` must be visually uncluttered.

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
