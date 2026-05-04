# Algorithm Strategy

## Scope

Use VTK/NumPy centerline and surface-cut logic.

VMTK must not be imported, required, or reintroduced.

Do not add broad architecture.

Do not create new output types.

Do not solve naming, clinical interpretation, measurement extraction, simulation, or downstream workflow problems.

The next code prompt must focus only on:

```text
branch_start-only visible rings
surface assignment consistency with the selected branch_start ring
clipped/color boundary correctness
```

The next code change must focus on visible ring count and surface assignment, not broad ring search.

Do not primarily move rings again unless validation proves the selected ring itself is invalid.

## Required Inputs

Use existing input artifacts from:

```text
inputs/
```

Do not edit inputs.

Do not require vessel-name metadata.

## Required Outputs

Write only:

```text
outputs/segmented_surface.vtp
outputs/boundary_rings.vtp
outputs/segmentation_result.json
```

Optional compact diagnostics are governed by `resources/05_validation_and_iteration.md`.

No candidate VTPs, all-candidate dumps, branch cloud files, raw component files, or clipping debug files are allowed unless a future prompt explicitly requests them.

There must be no extra VTP arrays and no extra output files for this fix.

## Default Ring Visibility

By default, `outputs/boundary_rings.vtp` contains only operational `branch_start` rings.

`visible_ring_count` must equal `branch_start_ring_count` by default.

`parent_pre_bifurcation` is internal/hidden by default.

`aortic_body_start`, `aortic_body_end`, `branch_end`, and `daughter_start` are internal/hidden by default.

If `daughter_start` topology is needed, it must reference an existing `branch_start` ring ID in JSON instead of creating visible duplicate geometry.

## Stable Daughter Section + Backward Refinement

For branch starts, the final algorithm strategy is:

```text
stable daughter section first
then backward refinement
then last clean candidate before parent contamination
```

For `branch_start` placement, Codex must not treat the topology start as final.

The implementation must:

```text
1. Start from the branch topology origin.
2. Search distally along the child centerline.
3. Generate surface-cut candidates perpendicular to the child tangent.
4. Find a stable daughter-tube section using compactness, radius plausibility, centroid closeness to centerline, and absence of parent-wall contamination.
5. Use that stable section to establish a reliable daughter radius/contour reference.
6. Walk backward toward the parent in small steps.
7. Stop when parent contamination or unstable mixed parent-child geometry appears.
8. Select the most proximal candidate that is still clean and stable.
9. Use that candidate as the circular branch_start cut-boundary.
10. Reassign, clip, or split surface cells so the surface boundary is consistent with the selected ring.
```

Required branch-start algorithm name:

```text
surface_validated_branch_start_ring_v1
```

## Forbidden Branch-Start Strategy

Do not use:

```text
topology start -> first acceptable candidate -> final ring
```

The topology start is only a search origin.

The final ring requires surface-cut evidence and surface assignment consistency.

Unproven zero-offset `branch_start` rings are not clean operational rings.

## Surface Assignment Correction

The selected circular ring must drive parent-child surface assignment.

Surface cells associated with the child branch but proximal to the selected `branch_start` ring plane beyond tolerance must be reassigned to the parent, clipped, or counted against success.

`segmented_surface.vtp` must agree with `boundary_rings.vtp`.

If the ring and surface split disagree, the run status must be:

```text
requires_review
```

or:

```text
failed
```

The implementation must define and report:

```text
RING_PLANE_ASSIGNMENT_TOLERANCE_MM = 0.10
```

Allowed tolerance range:

```text
0.05 mm to 0.15 mm
```

Any tolerance above `0.15 mm` must force `requires_review` unless explicitly justified.

Do not create a VTP array for tolerance.

Do not create extra debug files for tolerance.

## Ring-Plane-Gated Surface Assignment

The next code change must focus on surface assignment, not ring search.

Ring-plane-gated surface assignment is required because the branch color must begin at the `branch_start` ring.

Algorithm:

```text
1. Use the already selected branch_start ring.
2. Build a signed plane from the ring center and normal.
3. For each cell assigned to the child segment, calculate signed distance to the ring plane.
4. Reassign parent-side cells to the parent if they are beyond tolerance.
5. Treat ambiguous near-ring cells conservatively.
6. Keep only the connected distal child-side component.
7. Reassign isolated proximal/parent-wall patches to the parent.
8. Mark requires_review if coloring and ring still disagree.
```

Surface assignment must use the selected `branch_start` ring plane as a hard gate, not only centerline projection.

Centerline projection alone is insufficient for final surface coloring.

The selected method must keep outputs minimal.

## Cell Recoloring Is Not Enough If The Visual Boundary Is Wrong

The current mesh may contain cells that cross the selected `branch_start` ring plane.

If the code only recolors whole cells based on cell centers, the visible boundary may not align with the circular ring.

Cell recoloring is not enough if the visual boundary is wrong.

If color-ring mismatch remains after ring-plane gating, Codex must implement clipping/splitting with `vtkClipPolyData` or equivalent logic, or mark the result `requires_review`.

Codex must not solve this by adding debug files or extra arrays.

Codex must solve it by making `segmented_surface.vtp` itself visually correct.

## Clipped Ring Boundary Contract

The clipped ring boundary contract means the `branch_start` ring defines a geometric boundary plane.

When ring-plane-gated recoloring cannot make the visible `SegmentColor` boundary match the `branch_start` ring within tolerance, the implementation must use `vtkClipPolyData` or equivalent polygonal clipping/splitting logic.

Preferred implementation concept:

```text
1. Use the branch_start ring center and normal to define a vtkPlane.
2. Clip or split child/parent surface regions with vtkClipPolyData or equivalent logic.
3. Keep the child-side clipped surface connected to the distal child branch.
4. Reassign or discard parent-side child-colored fragments.
5. Preserve only the minimal SegmentId, SegmentLabel, and SegmentColor arrays.
6. Do not create extra debug VTP files.
```

`vtkClipPolyData` is allowed and preferred for this visual-boundary problem because it can cut polygonal cells, not merely recolor whole cells.

## Branch Contour Consistency Contract

A child segment is not valid merely because its cells are distal to the `branch_start` ring plane.

The color should follow the child vessel contour, not surrounding vessels.

Cells should remain child-colored only if they are:

```text
on the child/distal side of the branch_start ring plane
connected to the distal child branch component
spatially consistent with the child branch corridor
within the expected local branch radius/diameter envelope plus tolerance
not parent/aortic wall or neighboring-branch surface
```

If branch color extends onto parent/aortic wall or neighboring branch contours, `ring_surface_consistency_status` must not be `success`.

## Ambiguous Near-Ring Cells

Ambiguous near-ring cells are not harmless.

They represent uncertainty at the interface.

The implementation must report `ambiguous_near_ring_total`.

Ambiguous near-ring cells must be reduced where possible by clipping/splitting or conservative reassignment.

A high `ambiguous_near_ring_total` must force `requires_review`.

If `cells_ambiguous_near_ring_count` is greater than 5% of the child segment cell count, that `branch_start` ring must be `requires_review` unless clipping resolves the ambiguity.

Do not add per-cell diagnostics.

Only compact counts are allowed.

## Surface Connectivity Contract

After ring-plane gating or clipping, the child-colored region should be the connected distal child-side component.

If disconnected child-colored patches remain proximal to the ring or isolated around the parent wall, they must be reassigned to parent or the output must be:

```text
requires_review
```

The goal is not just to classify individual cells by nearest centerline.

The goal is a visually continuous child branch segment beginning at the `branch_start` ring.

## Bifurcation Strategy

Bifurcation output must stay minimal.

Internal bifurcation topology may stay in compact JSON.

Do not write non-`branch_start` bifurcation rings to `boundary_rings.vtp` by default.

Do not collapse bifurcation topology in JSON unless the geometry has no measurable separation.

## Acceptance Authority

The final authority is:

```text
branch_start-only visible rings
clean circular ring placement
surface-cut evidence
ring-to-color consistency
parent-child surface assignment consistency
child vessel contour consistency
minimal output contract compliance
```

Centerline topology is an origin and routing guide only.

It is not final placement authority.
